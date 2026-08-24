"""The judge validation study (build plan v3 §13), computed per dimension.

v3 §13 is explicit that validating a local 20B judge is "a component study, not
a checkbox", and the shape of the study follows from one expectation: **the
judge will be decent on the structural dimensions and poor on naturalness.** If
that is true, a single overall agreement number is the worst possible summary —
it would licence the naturalness comparison, which is precisely the finding the
study most expects to be interesting and least expects the judge to measure.
Everything here is therefore computed per dimension, and the confirmatory /
exploratory decision is made per dimension too.

Five metrics, each answering a different question about the instrument:

1. **Self-consistency** — ask the same question five times at 0.7; how much does
   the answer move? Answers "is this measurement stable at all".
2. **Positional bias** — re-run with the anchor order reversed; does the score
   change? Answers "is the judge reading the rubric or the layout".
3. **Span grounding** — two rates, kept separate. The *automatic* rate is how
   often a cited span could be located verbatim, which is a hallucination rate
   and is free. The *support* rate is how often a located span actually
   justifies the score it was attached to, which no program can decide and
   which v3 §13 assigns to a manual spot-check of 30 spans. Reporting only the
   automatic rate would be reporting the easy half and implying the hard one.
4. **Validity** — Krippendorff's alpha (ordinal) and Spearman's rho against
   human consensus, per dimension. Answers "does this agree with the thing it
   is standing in for".
5. **The pre-specified threshold** — fixed below, before the numbers exist,
   because a threshold chosen after seeing the data is not a threshold.

Everything is a pure function of `JudgeResult` objects and human scores. No
model calls, no database, no I/O — the whole study recomputes from the judge
cache in milliseconds, so a change to the grounding rule is re-analysed rather
than re-judged.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from carelite.eval.judge.agreement import (
    AgreementResult,
    Metric,
    krippendorff_alpha,
    paired_series,
    spearman_rho,
)
from carelite.eval.judge.judge import JudgeResult
from carelite.eval.rubric.dimensions import to_quality
from carelite.types import RUBRIC_DIMENSIONS

__all__ = [
    "MIN_ALPHA_FOR_CONFIRMATORY",
    "MIN_RHO_FOR_CONFIRMATORY",
    "MIN_UNITS_FOR_CONFIRMATORY",
    "N_SPANS_TO_REVIEW",
    "VALIDATION_PLAN_VERSION",
    "DimensionValidity",
    "Discrimination",
    "EvidenceStatus",
    "PositionalBias",
    "SelfConsistency",
    "SpanGroundingAudit",
    "SpanReviewItem",
    "SpanReviewVerdict",
    "SpanSupportReport",
    "ValidationReport",
    "classify_dimension",
    "discrimination",
    "judge_among_raters_alpha",
    "judge_human_validity",
    "positional_bias",
    "sample_spans_for_review",
    "self_consistency",
    "span_grounding_audit",
    "span_support_rate",
]

# ---------------------------------------------------------------------------
# PRE-SPECIFIED. Fixed at sprint 0, before any eval data existed. Changing a
# number below after seeing results turns a confirmatory study into a story.
# ---------------------------------------------------------------------------

#: Bump only with a documented, dated amendment to the pre-registration.
VALIDATION_PLAN_VERSION = "1.0.0"

#: Krippendorff's conventional cut for drawing tentative conclusions. The
#: stricter 0.800 is the bar for treating a coding as reliable outright; at a
#: single-rater-fallback sample size, holding out for 0.800 would demote every
#: dimension and make the threshold decorative. 0.667 is the honest bar for
#: "this number may be reported as a finding".
MIN_ALPHA_FOR_CONFIRMATORY = 0.667

#: Rho is a second, weaker gate: alpha can be dragged down by a constant offset
#: (a judge that is systematically one point generous still *ranks* correctly),
#: and a study that reports rankings wants to know that separately. A dimension
#: must clear both.
MIN_RHO_FOR_CONFIRMATORY = 0.5

#: Below this many paired units, no coefficient is stable enough to license a
#: confirmatory claim whatever its value.
MIN_UNITS_FOR_CONFIRMATORY = 30

#: v3 §13: "Spot-check 30 spans manually."
N_SPANS_TO_REVIEW = 30


class EvidenceStatus(StrEnum):
    """What a judge-only result on this dimension may be reported as."""

    #: Agreement cleared the pre-specified threshold. Report as a finding.
    CONFIRMATORY = "confirmatory"
    #: Below threshold, or too few units. Report as exploratory, and say so in
    #: the sentence that reports it, not only in a limitations paragraph.
    EXPLORATORY = "exploratory"


def classify_dimension(alpha: float, rho: float, n_units: int) -> EvidenceStatus:
    """Apply the pre-specified threshold to one dimension.

    A `nan` coefficient — undefined because a series was constant or too sparse —
    is treated as failing. Undefined agreement is not evidence of agreement.
    """
    if n_units < MIN_UNITS_FOR_CONFIRMATORY:
        return EvidenceStatus.EXPLORATORY
    if math.isnan(alpha) or math.isnan(rho):
        return EvidenceStatus.EXPLORATORY
    if alpha >= MIN_ALPHA_FOR_CONFIRMATORY and rho >= MIN_RHO_FOR_CONFIRMATORY:
        return EvidenceStatus.CONFIRMATORY
    return EvidenceStatus.EXPLORATORY


# ---------------------------------------------------------------------------
# 1. Self-consistency
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SelfConsistency:
    """Stability of one dimension across the five validation samples."""

    dimension: str
    #: Generations with at least two admitted samples on this dimension.
    n_generations: int
    mean_variance: float
    mean_sd: float
    mean_range: float
    #: Share of generations where all admitted samples agreed exactly.
    pct_unanimous: float
    #: Share where the five samples spanned two or more scale points. This is
    #: the number that decides which scenarios get excluded in the v3 §14
    #: sensitivity analysis.
    pct_range_ge_2: float


def self_consistency(results: Iterable[JudgeResult]) -> dict[str, SelfConsistency]:
    """Inter-sample variance per dimension, over multi-sample judge results.

    Single-sample results contribute nothing: `DimensionResult.variance` is
    `None` for n=1 rather than 0.0 precisely so a full-run pass cannot be
    mistaken for evidence of perfect stability.
    """
    buckets: dict[str, list[tuple[float, int]]] = {k: [] for k in RUBRIC_DIMENSIONS}
    for result in results:
        for key in RUBRIC_DIMENSIONS:
            dim = result.dimensions[key]
            if dim.variance is None or dim.score_range is None:
                continue
            buckets[key].append((dim.variance, dim.score_range))

    out: dict[str, SelfConsistency] = {}
    for key, rows in buckets.items():
        if not rows:
            out[key] = SelfConsistency(key, 0, math.nan, math.nan, math.nan, math.nan, math.nan)
            continue
        variances = [v for v, _ in rows]
        ranges = [r for _, r in rows]
        n = len(rows)
        out[key] = SelfConsistency(
            dimension=key,
            n_generations=n,
            mean_variance=statistics.fmean(variances),
            mean_sd=statistics.fmean([math.sqrt(v) for v in variances]),
            mean_range=statistics.fmean([float(r) for r in ranges]),
            pct_unanimous=sum(1 for r in ranges if r == 0) / n,
            pct_range_ge_2=sum(1 for r in ranges if r >= 2) / n,
        )
    return out


# ---------------------------------------------------------------------------
# 1b. Discrimination — the metric self-consistency cannot substitute for
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Discrimination:
    """Whether a dimension's scores differ *across* responses at all.

    Self-consistency measures variance **within** a generation, across the five
    samples. It answers "is this measurement stable". It cannot answer "is this
    measurement of anything", and the two come apart in the worst possible way:
    **a judge that returns the same score for every response is perfectly
    self-consistent.** It scores variance 0.0, unanimity 100%, and tops the
    stability table while carrying no information whatsoever.

    That is not hypothetical here. On the first validation arm, `ie` and
    `ritualistic` were scored identically on all twelve responses — spanning
    five conditions including the deliberately degraded negative control — and
    came out as the two most "stable" dimensions in the §13 self-consistency
    table.

    So `between_variance` is reported beside `SelfConsistency.mean_variance`,
    and `ratio` is the one to read: a dimension is informative when it varies
    more between responses than between samples of the same response. Below
    about 1.0 the sampling noise is as large as the signal; at 0.0 there is no
    signal at all.

    **This also explains low agreement that has nothing to do with disagreement.**
    Krippendorff's alpha normalises observed disagreement by *expected*
    disagreement, so when a dimension's scores barely vary, expected
    disagreement collapses and ordinary rater noise dominates the ratio — the
    coefficient falls toward zero while the raters are in fact agreeing on
    nearly every unit. It is the same prevalence effect that produces the
    well-known kappa paradox.

    Measured on the first validation arm, against a synthetic panel built by one
    generator at one noise level, the correlation between a dimension's
    `between_variance` and its recovered alpha was **r = 0.818** across the
    eleven dimensions: mean alpha 0.884 for the five high-variance dimensions
    and 0.216 for the six low-variance ones. So a low alpha on a floored
    dimension is a restatement of `between_variance`, not independent evidence
    about raters — and **no sample size repairs it.** Read the two together or
    neither.
    """

    dimension: str
    n_generations: int
    #: Variance of the reported (median) score across generations.
    between_variance: float
    #: Mean within-generation sample variance, from `SelfConsistency`.
    within_variance: float
    #: How many distinct score values the judge actually used, of five.
    n_distinct: int
    #: Share of generations receiving the single most common score.
    modal_share: float

    @property
    def ratio(self) -> float:
        """`between / within`. `inf` when a dimension is perfectly self-consistent."""
        if self.within_variance == 0:
            return math.inf if self.between_variance > 0 else 0.0
        return self.between_variance / self.within_variance

    @property
    def degenerate(self) -> bool:
        """The judge used one value for every response. Measures nothing."""
        return self.n_distinct <= 1


def discrimination(
    results: Iterable[JudgeResult],
    consistency: Mapping[str, SelfConsistency] | None = None,
) -> dict[str, Discrimination]:
    """Between-generation spread per dimension, on the quality scale.

    Quality scale via `quality_scores()`, so `ritualistic` points the same way
    as its ten neighbours. Reversing a scale cannot change a variance, but it
    keeps every table leaving this module pointing one way.
    """
    rows = [r.quality_scores() for r in results]
    consistency = consistency if consistency is not None else self_consistency(results)

    out: dict[str, Discrimination] = {}
    for key in RUBRIC_DIMENSIONS:
        values = [float(v) for row in rows if (v := row.get(key)) is not None]
        counts = Counter(values)
        n = len(values)
        out[key] = Discrimination(
            dimension=key,
            n_generations=n,
            between_variance=statistics.variance(values) if n > 1 else 0.0,
            within_variance=consistency[key].mean_variance
            if not math.isnan(consistency[key].mean_variance)
            else 0.0,
            n_distinct=len(counts),
            modal_share=(counts.most_common(1)[0][1] / n) if n else math.nan,
        )
    return out


# ---------------------------------------------------------------------------
# 2. Positional bias
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionalBias:
    """Effect of reversing anchor order, on the quality scale.

    `mean_signed_delta` is `descending - ascending` **after** `to_quality`, so a
    positive value means reversing the anchors made the judge score the response
    as *better*, on every dimension including the reverse-coded one. Reporting
    this on the raw scale would make `ritualistic` point the other way from its
    ten neighbours in the same table.
    """

    dimension: str
    n_paired: int
    mean_signed_delta: float
    mean_abs_delta: float
    #: Share of generations where reversing the order moved the score by a full
    #: scale point or more. A mean near zero with a large share here is order
    #: noise rather than order bias, and the two want different responses.
    pct_shift_ge_1: float


def positional_bias(
    ascending: Iterable[JudgeResult],
    descending: Iterable[JudgeResult],
) -> dict[str, PositionalBias]:
    """Paired comparison of the same generations judged in both anchor orders.

    Only generations present in both arms and admitted in both arms contribute;
    a rejected score on one side removes that pair, which is correct — a
    grounding failure is not a zero delta.
    """
    asc = {r.generation_id: r.quality_scores() for r in ascending}
    desc = {r.generation_id: r.quality_scores() for r in descending}

    out: dict[str, PositionalBias] = {}
    for key in RUBRIC_DIMENSIONS:
        left = {gid: scores.get(key) for gid, scores in asc.items()}
        right = {gid: scores.get(key) for gid, scores in desc.items()}
        a_vals, d_vals, _ = paired_series(left, right)
        if not a_vals:
            out[key] = PositionalBias(key, 0, math.nan, math.nan, math.nan)
            continue
        deltas = [d - a for a, d in zip(a_vals, d_vals, strict=True)]
        out[key] = PositionalBias(
            dimension=key,
            n_paired=len(deltas),
            mean_signed_delta=statistics.fmean(deltas),
            mean_abs_delta=statistics.fmean([abs(d) for d in deltas]),
            pct_shift_ge_1=sum(1 for d in deltas if abs(d) >= 1.0) / len(deltas),
        )
    return out


# ---------------------------------------------------------------------------
# 3. Span grounding
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpanGroundingAudit:
    """Automatic half of the grounding check: could the quote be found at all.

    This is a **hallucination rate**, not a support rate. It says nothing about
    whether the quote justifies the score — see `SpanSupportReport` for that.
    """

    #: (dimension, sample) pairs the judge attempted to score.
    n_attempted: int
    n_admitted: int
    n_rejected: int
    #: Rejections by reason. `span_not_found` is fabrication; `missing_span` is
    #: non-compliance; they are different problems with different fixes.
    reasons: Mapping[str, int]
    #: Share of admitted spans that matched byte-for-byte rather than after
    #: typography canonicalisation.
    exact_rate: float
    #: Share of admitted spans found only in the sanitised text the judge saw,
    #: not in the stored response. Expected to be ~0; a rising number means
    #: sanitisation is changing responses more than it should.
    presented_only_rate: float
    per_dimension: Mapping[str, float] = field(default_factory=dict)

    @property
    def admitted_rate(self) -> float:
        return self.n_admitted / self.n_attempted if self.n_attempted else math.nan


def span_grounding_audit(results: Iterable[JudgeResult]) -> SpanGroundingAudit:
    """Count admissions and rejections across every dimension of every sample."""
    reasons: Counter[str] = Counter()
    per_dim_attempted: Counter[str] = Counter()
    per_dim_admitted: Counter[str] = Counter()
    n_attempted = 0
    n_admitted = 0
    n_exact = 0
    n_presented_only = 0

    for result in results:
        for sample in result.samples:
            for key, admitted in sample.scores.items():
                n_attempted += 1
                per_dim_attempted[key] += 1
                if admitted.admitted and admitted.span is not None:
                    n_admitted += 1
                    per_dim_admitted[key] += 1
                    if admitted.span.exact:
                        n_exact += 1
                    if admitted.span.source == "presented":
                        n_presented_only += 1
                elif admitted.rejection is not None:
                    reasons[str(admitted.rejection)] += 1

    return SpanGroundingAudit(
        n_attempted=n_attempted,
        n_admitted=n_admitted,
        n_rejected=n_attempted - n_admitted,
        reasons=dict(reasons),
        exact_rate=n_exact / n_admitted if n_admitted else math.nan,
        presented_only_rate=n_presented_only / n_admitted if n_admitted else math.nan,
        per_dimension={
            key: (per_dim_admitted[key] / per_dim_attempted[key] if per_dim_attempted[key] else 0.0)
            for key in RUBRIC_DIMENSIONS
        },
    )


@dataclass(frozen=True, slots=True)
class SpanReviewItem:
    """One span put in front of a human, with everything needed to judge it.

    The reviewer answers one question: *does this quote support this score on
    this dimension?* They are given the whole response so they can tell a
    misleading excerpt from a fair one — a span can be perfectly verbatim and
    still be evidence for a different dimension entirely, and that failure is
    invisible to any automatic check.
    """

    item_id: str
    generation_id: str
    dimension: str
    score: int
    span: str
    rationale: str
    response: str


def sample_spans_for_review(
    results: Sequence[JudgeResult],
    responses: Mapping[str, str],
    *,
    n: int = N_SPANS_TO_REVIEW,
    seed: int = 20260822,
) -> list[SpanReviewItem]:
    """Draw a reproducible, dimension-stratified sample of spans to spot-check.

    Stratified by dimension because an unstratified draw of 30 from eleven
    dimensions leaves several dimensions with zero or one span, and "the judge's
    spans support its scores 87% of the time" is not a useful sentence if all
    the failures were concentrated in a dimension the sample happened to miss.
    Seeded so the sample is fixed before anyone looks at it.
    """
    by_dimension: dict[str, list[SpanReviewItem]] = {k: [] for k in RUBRIC_DIMENSIONS}
    for result in results:
        response = responses.get(result.generation_id, "")
        for key in RUBRIC_DIMENSIONS:
            dim = result.dimensions[key]
            if dim.score is None or not dim.span:
                continue
            by_dimension[key].append(
                SpanReviewItem(
                    item_id=f"{result.generation_id}:{key}",
                    generation_id=result.generation_id,
                    dimension=key,
                    score=dim.score,
                    span=dim.span,
                    rationale=dim.rationale,
                    response=response,
                )
            )

    rng = random.Random(seed)
    for items in by_dimension.values():
        rng.shuffle(items)

    # Round-robin across dimensions until n is reached: equal representation
    # while any dimension still has spans left, rather than a fixed quota that
    # would silently under-fill when a dimension is mostly rejected.
    picked: list[SpanReviewItem] = []
    cursors = dict.fromkeys(RUBRIC_DIMENSIONS, 0)
    while len(picked) < n:
        progressed = False
        for key in RUBRIC_DIMENSIONS:
            idx = cursors[key]
            items = by_dimension[key]
            if idx < len(items):
                picked.append(items[idx])
                cursors[key] = idx + 1
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
    return picked


@dataclass(frozen=True, slots=True)
class SpanReviewVerdict:
    """A human's answer for one reviewed span."""

    item_id: str
    dimension: str
    supports: bool
    note: str = ""


@dataclass(frozen=True, slots=True)
class SpanSupportReport:
    """The v3 §13 headline: how often a cited span actually supports its score."""

    n_reviewed: int
    n_supported: int
    support_rate: float
    #: Wilson 95% interval. At n=30 the normal-approximation interval is wrong
    #: enough near the boundary to matter, and a rate reported without an
    #: interval at n=30 reads far more precise than it is.
    ci_low: float
    ci_high: float
    per_dimension: Mapping[str, float]


def span_support_rate(verdicts: Sequence[SpanReviewVerdict]) -> SpanSupportReport:
    """Aggregate the manual spot-check."""
    n = len(verdicts)
    supported = sum(1 for v in verdicts if v.supports)
    rate = supported / n if n else math.nan
    low, high = _wilson_interval(supported, n)

    per_dim_total: Counter[str] = Counter()
    per_dim_ok: Counter[str] = Counter()
    for v in verdicts:
        per_dim_total[v.dimension] += 1
        if v.supports:
            per_dim_ok[v.dimension] += 1

    return SpanSupportReport(
        n_reviewed=n,
        n_supported=supported,
        support_rate=rate,
        ci_low=low,
        ci_high=high,
        per_dimension={
            key: per_dim_ok[key] / per_dim_total[key]
            for key in sorted(per_dim_total)
            if per_dim_total[key]
        },
    )


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. `(nan, nan)` for an empty sample."""
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# 4. Validity against human consensus
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DimensionValidity:
    """Judge-vs-human agreement on one dimension, with its reporting status."""

    agreement: AgreementResult
    status: EvidenceStatus

    @property
    def dimension(self) -> str:
        return self.agreement.dimension


def _quality_view(scores: Mapping[str, int | None], dimension: str) -> float | None:
    """One dimension's score on the higher-is-better scale, or `None`.

    Goes through `to_quality` rather than a local flip. The reversal exists in
    exactly one place in this codebase and this is not it.
    """
    value = scores.get(dimension)
    return None if value is None else float(to_quality(dimension, value))


def judge_human_validity(
    judge_results: Iterable[JudgeResult],
    human_consensus: Mapping[str, Mapping[str, int | None]],
) -> dict[str, DimensionValidity]:
    """Alpha (ordinal) and rho between the judge and human consensus, per dimension.

    Args:
        judge_results: One per generation. Raw scores; canonicalised here.
        human_consensus: `generation_id -> {dimension: raw score}`, typically
            from `carelite.eval.human.reliability.human_consensus`. Raw scale,
            same as the judge, canonicalised here.

    Note on interpretation: agreement against a *consensus* is not the same
    quantity as agreement among individuals, and reads higher, because
    averaging two raters removes some of their noise before the comparison. It
    is the right comparison for "can the judge stand in for the panel", and
    `judge_among_raters_alpha` gives the stricter "is the judge just another
    rater" view.
    """
    judge_scores = {r.generation_id: r.scores() for r in judge_results}

    out: dict[str, DimensionValidity] = {}
    for key in RUBRIC_DIMENSIONS:
        left = {gid: _quality_view(s, key) for gid, s in judge_scores.items()}
        right = {gid: _quality_view(s, key) for gid, s in human_consensus.items()}
        xs, ys, kept = paired_series(left, right)

        alpha = krippendorff_alpha([xs, ys], metric=Metric.ORDINAL) if xs else math.nan
        rho, p = spearman_rho(xs, ys)
        agreement = AgreementResult(
            dimension=key,
            n_units=len(kept),
            n_observers=2,
            alpha=alpha,
            rho=rho,
            rho_p=p,
        )
        out[key] = DimensionValidity(
            agreement=agreement,
            status=classify_dimension(alpha, rho, len(kept)),
        )
    return out


def judge_among_raters_alpha(
    judge_results: Iterable[JudgeResult],
    human_scores: Mapping[str, Mapping[str, Mapping[str, int | None]]],
) -> dict[str, float]:
    """Alpha per dimension with the judge treated as one more rater.

    The stricter view: instead of asking whether the judge tracks a smoothed
    consensus, it asks whether adding the judge to the panel degrades the
    panel's reliability. If human-only alpha is 0.72 and alpha-with-judge is
    0.71, the judge is behaving like a rater. If it drops to 0.40, it is not.

    Args:
        judge_results: One per generation.
        human_scores: `rater_id -> generation_id -> {dimension: raw score}`.
    """
    judge_scores = {r.generation_id: r.scores() for r in judge_results}
    units = sorted(set(judge_scores) | {gid for s in human_scores.values() for gid in s})

    out: dict[str, float] = {}
    for key in RUBRIC_DIMENSIONS:
        rows: list[list[float | None]] = [
            [_quality_view(judge_scores.get(gid, {}), key) for gid in units]
        ]
        for rater in sorted(human_scores):
            rows.append([_quality_view(human_scores[rater].get(gid, {}), key) for gid in units])
        out[key] = krippendorff_alpha(rows, metric=Metric.ORDINAL)
    return out


# ---------------------------------------------------------------------------
# 5. The report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Every §13 metric in one object, plus the pre-specified verdict.

    `render()` produces the text that goes in the write-up. It leads with the
    exploratory dimensions rather than burying them, because the whole purpose
    of computing agreement per dimension is to know which numbers to discount.
    """

    judge_model: str
    judge_digest: str
    generator_model: str
    plan_version: str
    rubric_version: str
    prompt_version: str
    n_generations: int
    self_consistency: Mapping[str, SelfConsistency]
    discrimination: Mapping[str, Discrimination]
    positional_bias: Mapping[str, PositionalBias]
    grounding: SpanGroundingAudit
    span_support: SpanSupportReport | None
    validity: Mapping[str, DimensionValidity]

    @property
    def confirmatory_dimensions(self) -> list[str]:
        return [k for k, v in self.validity.items() if v.status is EvidenceStatus.CONFIRMATORY]

    @property
    def exploratory_dimensions(self) -> list[str]:
        return [k for k, v in self.validity.items() if v.status is EvidenceStatus.EXPLORATORY]

    @property
    def degenerate_dimensions(self) -> list[str]:
        """Dimensions where the judge used a single score for every response.

        Reported separately from the threshold because it is a different kind of
        failure and it is not visible in an agreement coefficient: a constant
        series makes alpha and rho undefined rather than low, and
        `classify_dimension` already treats `nan` as failing. The point of
        naming these is that they would otherwise appear at the *top* of the
        self-consistency table.
        """
        return [k for k, v in self.discrimination.items() if v.degenerate]

    def render(self) -> str:
        """Plain-text report. Independence first, per v3 §13's 'report it prominently'."""
        lines: list[str] = []
        lines.append("JUDGE VALIDATION STUDY (build plan v3 §13)")
        lines.append(
            f"plan version {self.plan_version} | rubric {self.rubric_version} | "
            f"prompt {self.prompt_version}"
        )
        lines.append("")
        lines.append("INDEPENDENCE")
        lines.append(
            f"  Judge {self.judge_model} (digest {self.judge_digest[:16]}) is a different model "
            f"family from the generator {self.generator_model}. No shared weights, no shared "
            "post-training."
        )
        lines.append(f"  Generations judged: {self.n_generations}")
        lines.append("")

        lines.append("PRE-SPECIFIED THRESHOLD")
        lines.append(
            f"  Confirmatory requires ordinal alpha >= {MIN_ALPHA_FOR_CONFIRMATORY} AND "
            f"Spearman rho >= {MIN_RHO_FOR_CONFIRMATORY} on >= {MIN_UNITS_FOR_CONFIRMATORY} "
            "paired units."
        )
        lines.append(f"  Confirmatory: {', '.join(self.confirmatory_dimensions) or '(none)'}")
        lines.append(
            f"  EXPLORATORY (judge-only results on these must be labelled as such in the "
            f"sentence that reports them): {', '.join(self.exploratory_dimensions) or '(none)'}"
        )
        lines.append("")

        lines.append("SPAN GROUNDING")
        lines.append(
            f"  Automatic: {self.grounding.n_admitted}/{self.grounding.n_attempted} scores "
            f"carried a locatable verbatim span ({self.grounding.admitted_rate:.1%}); "
            f"{self.grounding.n_rejected} rejected."
        )
        for reason, count in sorted(self.grounding.reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {reason}: {count}")
        if self.span_support is not None:
            s = self.span_support
            lines.append(
                f"  Manual spot-check: {s.n_supported}/{s.n_reviewed} spans actually support "
                f"the score ({s.support_rate:.1%}, 95% CI {s.ci_low:.1%}-{s.ci_high:.1%})."
            )
        else:
            lines.append("  Manual spot-check: NOT YET RUN — the support rate is unreported.")
        lines.append("")

        degenerate = self.degenerate_dimensions
        if degenerate:
            lines.append("DISCRIMINATION — READ BEFORE THE STABILITY COLUMN")
            lines.append(
                f"  The judge used a SINGLE score for every response on: {', '.join(degenerate)}. "
                "Those dimensions measure nothing here, and because a constant series is "
                "perfectly self-consistent they appear as the most stable in the table below."
            )
            lines.append("")

        lines.append(
            f"{'dimension':<14}{'alpha':>8}{'rho':>8}{'n':>6}  {'status':<13}"
            f"{'var':>7}{'unanim':>8}{'btwn':>7}{'ratio':>7}{'posbias':>9}"
        )
        for key in RUBRIC_DIMENSIONS:
            v = self.validity.get(key)
            sc = self.self_consistency.get(key)
            pb = self.positional_bias.get(key)
            dc = self.discrimination.get(key)
            alpha = v.agreement.alpha if v else math.nan
            rho = v.agreement.rho if v else math.nan
            n = v.agreement.n_units if v else 0
            status = str(v.status) if v else "-"
            lines.append(
                f"{key:<14}{alpha:>8.3f}{rho:>8.3f}{n:>6}  {status:<13}"
                f"{(sc.mean_variance if sc else math.nan):>7.2f}"
                f"{(sc.pct_unanimous if sc else math.nan):>8.0%}"
                f"{(dc.between_variance if dc else math.nan):>7.2f}"
                f"{_ratio(dc.ratio if dc else math.nan):>7}"
                f"{(pb.mean_signed_delta if pb else math.nan):>+9.2f}"
            )
        return "\n".join(lines)


def _ratio(value: float) -> str:
    """`inf` prints as a word; a degenerate ratio should not read as a big number."""
    if math.isnan(value):
        return "  n/a"
    if math.isinf(value):
        return "  inf"
    return f"{value:.2f}"


def build_validation_report(
    *,
    validation_results: Sequence[JudgeResult],
    responses: Mapping[str, str],
    human_consensus: Mapping[str, Mapping[str, int | None]] | None = None,
    reversed_results: Sequence[JudgeResult] = (),
    span_verdicts: Sequence[SpanReviewVerdict] = (),
    judge_model: str = "",
    judge_digest: str = "",
    generator_model: str = "",
    prompt_version: str = "",
    rubric_version: str = "",
) -> ValidationReport:
    """Assemble the whole §13 study from already-computed judge results.

    `responses` is `generation_id -> response text`, needed only so the span
    review worksheet can show a reviewer the full response. `human_consensus`
    may be `None` before human rating happens, in which case every dimension
    lands as exploratory — which is the correct state of the world at that
    point, not a placeholder to be filled in optimistically.
    """
    consensus = human_consensus or {}
    return ValidationReport(
        judge_model=judge_model
        or (validation_results[0].judge_model if validation_results else ""),
        judge_digest=judge_digest
        or (validation_results[0].judge_digest if validation_results else ""),
        generator_model=generator_model,
        plan_version=VALIDATION_PLAN_VERSION,
        rubric_version=rubric_version
        or (validation_results[0].rubric_version if validation_results else ""),
        prompt_version=prompt_version
        or (validation_results[0].prompt_version if validation_results else ""),
        n_generations=len(validation_results),
        self_consistency=(_sc := self_consistency(validation_results)),
        discrimination=discrimination(validation_results, _sc),
        positional_bias=positional_bias(validation_results, reversed_results),
        grounding=span_grounding_audit(validation_results),
        span_support=span_support_rate(span_verdicts) if span_verdicts else None,
        validity=judge_human_validity(validation_results, consensus),
    )
