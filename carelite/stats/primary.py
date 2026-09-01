"""The primary test sequence: Friedman omnibus, Wilcoxon post-hoc, Holm-Bonferroni.

**D10: every result below is descriptive.** The plan is kept (docs/preregistration.md)
and still governs the analysis; it was never registered, so nothing here is
confirmatory. `CONFIRMATORY_FAMILY` keeps its name because `carelite/viz` imports
it; `PLANNED_FAMILY` is the alias new code should use, and no rendered string in
this module uses the word.

Analysis plan §8.1, quoted because the exact wording is what this module
implements:

    Friedman omnibus test across conditions {A, B, C} for each of the 11 rubric
    dimensions (on to_quality()-transformed scores), followed by Wilcoxon
    signed-rank on every pairwise comparison listed in §4, followed by
    Holm-Bonferroni correction applied across the whole
    pairwise-comparison-by-dimension family -- not per dimension separately. A
    dimension is not tested in isolation from the others for correction purposes.

==========================================================================
THE CORRECTION FAMILY, STATED EXPLICITLY
==========================================================================
Getting this wrong is the most common way a result of this shape falls apart
under review, so the reading is written down rather than left in the code.

**The Holm family is the eight comparisons the analysis plan enumerates** -- the primary outcome (§3) plus the seven directional secondary
outcomes (§4) -- corrected together, in one Holm-Bonferroni step, as
`CONFIRMATORY_FAMILY`. Each of those eight names its own outcome measure: five
are on the NURSE composite, one on the Four Habits composite, one on
`naturalness`, one on `ritualistic`. So the family spans several comparisons
*and* several dimensions, and the naturalness test is corrected in the same
step as the NURSE-composite tests. That is what "not per dimension separately"
forbids, and it is what this module does.

**What is deliberately NOT in the Holm family.** §8.1 says "every
pairwise comparison listed in §4", and §4 lists eight comparisons each with a
named measure. It does not list A vs C, and it does not list, say, `de` for
A vs B. Testing all five condition pairs on all eleven dimensions would be 55
tests, of which 47 are hypotheses nobody planned; folding them into the
Holm family would multiply the correction on the planned eight by
about seven for the sake of tests that are exploratory by §1's own definition.
Those tests are still available and still worth looking at -- `dimension_expansion`
builds them, they are corrected within their own separate family, and every one
of them is labelled EXPLORATORY in the result object itself.

**The wording admits a second reading** -- "pairwise-comparison-by-dimension
family" could be read as that 55-cell cross product. If the project prefers it,
it is a one-line change (`CONFIRMATORY_FAMILY = (*PRESPECIFIED_HYPOTHESES,
*dimension_expansion())`). D9.2 settled it as the eight, before the holdout data
existed, rather than after seeing which reading is kinder to the numbers.

**The eleven Friedman omnibus tests are not in the Holm family.** §8.1 puts the
correction on the pairwise family; an omnibus test is the gate in front of it,
in the standard omnibus-then-post-hoc shape. Their raw p-values are reported,
and a Holm correction *within the eleven omnibus tests* is reported beside them
so a reader who wants that adjustment does not have to compute it -- clearly
labelled, and not the basis of any claim.

==========================================================================
TWO-SIDED TESTS FOR DIRECTIONAL HYPOTHESES
==========================================================================
§4 states a direction for every secondary outcome; §8.1 fixes the test but not
its sidedness. Every test here is therefore **two-sided**, which is the
conservative choice: a one-sided test at alpha = 0.05 would be easier to pass in
the hypothesised direction and would have no power at all against the opposite
one -- and secondary outcome 4 (naturalness, A > B) is precisely a hypothesis
the study expects to come out against the system, so an analysis that could not
detect the opposite direction would defeat the purpose of stating it in advance. The
hypothesised direction is recorded on every result and compared against the
observed one, so "significant, in the predicted direction" and "significant, in
the opposite direction" are distinguishable without a one-sided test.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

from carelite.eval.judge.validation import EvidenceStatus
from carelite.stats.effects import (
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    PairedEffects,
    paired_effects,
)
from carelite.stats.evidence import Label, RaterScope, label_for
from carelite.stats.instrument import (
    Discrimination,
    MeasureTestability,
    measure_testability,
)
from carelite.stats.measures import Measure, cell_means, measure, paired_matrix
from carelite.types import RUBRIC_DIMENSIONS, Condition

__all__ = [
    "CONFIRMATORY_FAMILY",
    "FRIEDMAN_CONDITIONS",
    "PLANNED_FAMILY",
    "PRESPECIFIED_HYPOTHESES",
    "FamilyResult",
    "FriedmanResult",
    "Hypothesis",
    "PairwiseResult",
    "WilcoxonResult",
    "dimension_expansion",
    "friedman_across_conditions",
    "friedman_omnibus",
    "holm_bonferroni",
    "run_family",
    "run_pairwise",
    "wilcoxon_paired",
]

#: Analysis plan §8.1. The omnibus runs across these three only.
FRIEDMAN_CONDITIONS: tuple[Condition, ...] = (Condition.A, Condition.B, Condition.C)


# ---------------------------------------------------------------------------
# The hypotheses planned in advance
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One planned (or exploratory) pairwise comparison.

    `left` and `right` are in the order the analysis plan states the
    comparison, and every effect size is computed as left-relative-to-right, so
    a positive effect always means "left scored higher". `expected_higher` is
    the direction predicted in the plan, or `None` for outcome 6, whose
    prediction is that there is no difference.
    """

    key: str
    measure_key: str
    left: Condition
    right: Condition
    expected_higher: Condition | None
    description: str
    role: str = "secondary"
    prespecified: bool = True
    #: Non-empty when this comparison is not runnable **by decision** rather than
    #: for want of data. The two look identical in a table of missing rows and
    #: are completely different claims: "the run did not produce this" versus
    #: "the run produced too little of this to analyse, and analysing it anyway
    #: would be worse than not". D11 was the live case until D13 re-opened it;
    #: no hypothesis carries one now, and the machinery stays because the next
    #: decision of that shape must not have to rebuild it.
    not_computable_reason: str = ""
    #: Qualifications this comparison cannot be read without — a confound the
    #: design cannot remove, a question narrower than the one its name suggests.
    #: They are **not** the same thing as `not_computable_reason`: the comparison
    #: runs, and the caveats travel with the number. Every one of them is pushed
    #: through `label_for` as a demotion reason, so it reaches the `label` column
    #: of `effect-sizes.csv` and the rendered result alike, and printed above the
    #: effect size in `PairwiseResult.render()`. A caveat that lived only in a
    #: docstring would be absent from every artefact anyone actually reads.
    caveats: tuple[str, ...] = ()

    @property
    def retired_by_decision(self) -> bool:
        return bool(self.not_computable_reason)

    @property
    def measure(self) -> Measure:
        return measure(self.measure_key)

    @property
    def pair_label(self) -> str:
        return f"{self.left} vs {self.right}"

    @property
    def expected_direction(self) -> str:
        """`>`, `<` or `=` for `left` relative to `right`, as predicted in the plan."""
        if self.expected_higher is None:
            return "="
        return ">" if self.expected_higher == self.left else "<"


#: Analysis plan §3 (primary) and §4 (secondary outcomes 1-7), in that order.
PRESPECIFIED_HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis(
        key="primary_nurse_A_vs_B",
        measure_key="nurse_composite",
        left=Condition.A,
        right=Condition.B,
        expected_higher=Condition.B,
        description=(
            "PRIMARY (§3). Composite NURSE adherence, A vs B. Framework prompting beats the "
            "bare model; the structural-adherence effect v3 §11 expects to be large."
        ),
        role="primary",
    ),
    Hypothesis(
        key="secondary1_four_habits_A_vs_B",
        measure_key="four_habits_composite",
        left=Condition.A,
        right=Condition.B,
        expected_higher=Condition.B,
        description="§4.1 Composite Four Habits adherence, A vs B. Same direction, same mechanism.",
    ),
    Hypothesis(
        key="secondary2_nurse_B_vs_C",
        measure_key="nurse_composite",
        left=Condition.B,
        right=Condition.C,
        expected_higher=Condition.C,
        description=(
            "§4.2 Composite NURSE adherence, B vs C. Retrieval adds grounded guidance beyond "
            "framework prompting. Expected to be the smallest effect in the study; this is the "
            "comparison that set n (§6)."
        ),
    ),
    Hypothesis(
        key="secondary3_nurse_C_vs_LC",
        measure_key="nurse_composite",
        left=Condition.C,
        right=Condition.LC,
        expected_higher=Condition.C,
        description=(
            "§4.3 Composite NURSE adherence, C vs LC. Query-dependent retrieval against a "
            "fixed context. Retired by D11 when LC stopped at 39 of 180 cells; restored by "
            "D13, which generated all 180 under vLLM. The arm is `served_by = 'vllm'` and "
            "nothing else — see `carelite.stats.arms`."
        ),
        caveats=(
            "CONFOUNDED BY SERVING STACK. The LC arm was served by vLLM and condition C by "
            "Ollama: a GGUF against HF safetensors, different quantisation, different sampling "
            "defaults, different hardware, and per D13 a different realised context pack. No "
            "analysis can separate the architecture from the stack that served it, so a "
            "difference here is not attributable to long context versus retrieval.",
            "THE REDUCED FORM OF THE QUESTION (D7). The corpus does not fit the window: the "
            "production pack admits 116/116 knowledge base entries but only 151/471 chunks. LC "
            "is a fixed, query-independent sample, not the whole corpus, and any selection rule "
            "is itself a form of retrieval. This asks whether query-dependent selection beats a "
            "fixed context — not whether curated retrieval beats stuffing everything in, which "
            "is the question build plan v3 §3 posed and this run cannot answer.",
        ),
    ),
    Hypothesis(
        key="secondary4_naturalness_A_vs_B",
        measure_key="naturalness",
        left=Condition.A,
        right=Condition.B,
        expected_higher=Condition.A,
        description=(
            "§4.4 naturalness, A vs B, hypothesised A > B. The against-the-system finding this "
            "analysis plan exists to protect. Under D10 it is descriptive, not registered."
        ),
    ),
    Hypothesis(
        key="secondary5_ritualistic_A_vs_B",
        measure_key="ritualistic",
        left=Condition.A,
        right=Condition.B,
        expected_higher=Condition.A,
        description=(
            "§4.5 ritualistic, A vs B. REVERSE-CODED: analysed on to_quality(), on which the "
            "prediction 'B has a higher raw ritualistic score' becomes 'A scores "
            "higher quality than B', consistent with outcome 4's direction."
        ),
    ),
    Hypothesis(
        key="secondary6_nurse_A_vs_A2",
        measure_key="nurse_composite",
        left=Condition.A,
        right=Condition.A2,
        expected_higher=None,
        description=(
            "§4.6 Composite NURSE adherence, A vs A2. Cross-model baseline; the "
            "prediction is NO significant difference. A significant result here is evidence "
            "about model family, not about the framework."
        ),
    ),
    Hypothesis(
        key="secondary7_nurse_B_vs_D",
        measure_key="nurse_composite",
        left=Condition.B,
        right=Condition.D,
        expected_higher=Condition.B,
        description=(
            "§4.7 Composite NURSE adherence, B vs D. The negative control: if the rubric cannot "
            "separate B from the deliberately degraded prompt, the rubric is not measuring the "
            "construct it claims to. See carelite.stats.negative_control."
        ),
    ),
)

#: The Holm family. See the module docstring for what is in it and what is not.
#: The name is retained because `carelite/viz` imports it; under D10 read it as
#: "the family planned in advance", never as a claim of confirmatory status.
CONFIRMATORY_FAMILY: tuple[Hypothesis, ...] = PRESPECIFIED_HYPOTHESES

#: The name new code should use for the same tuple.
PLANNED_FAMILY: tuple[Hypothesis, ...] = CONFIRMATORY_FAMILY


def dimension_expansion(
    hypotheses: Sequence[Hypothesis] = PRESPECIFIED_HYPOTHESES,
    dimensions: Sequence[str] = RUBRIC_DIMENSIONS,
) -> tuple[Hypothesis, ...]:
    """Every planned condition pair crossed with every rubric dimension.

    EXPLORATORY by construction: `prespecified=False` on every one, so the label
    machinery in `carelite.stats.evidence` demotes them whatever the judge
    validation says. Corrected within its own family, never folded into
    `CONFIRMATORY_FAMILY`.
    """
    pairs: list[tuple[Condition, Condition]] = []
    for h in hypotheses:
        if (h.left, h.right) not in pairs:
            pairs.append((h.left, h.right))

    planned = {(h.left, h.right, h.measure_key) for h in hypotheses}
    out: list[Hypothesis] = []
    for left, right in pairs:
        for dim in dimensions:
            if (left, right, dim) in planned:
                continue
            out.append(
                Hypothesis(
                    key=f"exploratory_{dim}_{left}_vs_{right}",
                    measure_key=dim,
                    left=left,
                    right=right,
                    expected_higher=None,
                    description=(
                        f"EXPLORATORY. {dim}, {left} vs {right}. Not named in the "
                        "analysis plan; no predicted direction."
                    ),
                    role="exploratory",
                    prespecified=False,
                )
            )
    return tuple(out)


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FriedmanResult:
    """Friedman omnibus for one measure across the conditions tested."""

    measure_key: str
    conditions: tuple[str, ...]
    statistic: float
    p_value: float
    df: int
    n_blocks: int
    label: Label
    #: Holm-corrected within the set of omnibus tests only. See the module
    #: docstring: this is reported for the reader's convenience and is not the
    #: planned correction, which applies to the pairwise family.
    p_holm_within_omnibus: float = math.nan
    #: The judge did not resolve this dimension. Its p-value is uninterpretable
    #: as a statement about the conditions. See `carelite.stats.instrument`.
    degenerate: bool = False


def friedman_omnibus(matrix: pd.DataFrame) -> tuple[float, float, int, int]:
    """Friedman chi-square across the columns of `matrix`. Rows are scenarios.

    Returns `(statistic, p_value, df, n_blocks)`. `(nan, nan, df, n)` when the
    test is undefined: fewer than three conditions, fewer than two complete
    blocks, or a matrix in which every block is constant so no ranking exists.
    """
    k = matrix.shape[1]
    n = matrix.shape[0]
    df = max(k - 1, 0)
    if k < 3 or n < 2:
        return math.nan, math.nan, df, n
    columns = [matrix.iloc[:, j].to_numpy(dtype=float) for j in range(k)]
    if np.allclose(matrix.to_numpy(dtype=float).std(axis=1), 0.0):
        # Every block is tied across conditions: the statistic is 0/0.
        return math.nan, math.nan, df, n
    statistic, p_value = stats.friedmanchisquare(*columns)
    return float(statistic), float(p_value), df, n


@dataclass(frozen=True, slots=True)
class WilcoxonResult:
    """One Wilcoxon signed-rank test, with the bookkeeping needed to read it."""

    statistic: float
    p_value: float
    n_pairs: int
    n_nonzero: int
    alternative: str
    method: str


def wilcoxon_paired(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    alternative: str = "two-sided",
) -> WilcoxonResult:
    """Wilcoxon signed-rank on paired scenario-level values.

    Zeros are dropped (`zero_method="wilcox"`), matching the effect size in
    `carelite.stats.effects`. scipy chooses the exact distribution for small
    samples without ties and the normal approximation with continuity correction
    otherwise; which one it used is recorded in `method`.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must be the same length")
    d = a - b
    nonzero = int(np.count_nonzero(d))
    if nonzero == 0:
        return WilcoxonResult(math.nan, math.nan, int(a.size), 0, alternative, "undefined")

    has_ties = np.unique(np.abs(d[d != 0])).size != nonzero
    mode = "exact" if (nonzero <= 25 and not has_ties) else "approx"
    result = stats.wilcoxon(
        a,
        b,
        zero_method="wilcox",
        alternative=alternative,
        method=mode,
        correction=mode == "approx",
    )
    return WilcoxonResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        n_pairs=int(a.size),
        n_nonzero=nonzero,
        alternative=alternative,
        method=mode,
    )


def holm_bonferroni(p_values: Sequence[float], *, family_size: int | None = None) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order.

    The step-down procedure: sort ascending, multiply the i-th smallest by
    (m - i), enforce monotonicity by running a cumulative maximum up the sorted
    list, and cap at 1.

    `family_size` defaults to `len(p_values)` and exists for the case where a
    planned test could not be computed. The family is fixed in advance by
    the analysis plan, so an undefined test still consumes its share of the
    correction rather than making the surviving tests easier to pass. `nan`
    inputs are returned as `nan` and are excluded from the ordering.
    """
    values = list(p_values)
    m = len(values) if family_size is None else family_size
    if m <= 0:
        return []
    finite = [(p, i) for i, p in enumerate(values) if not math.isnan(p)]
    finite.sort(key=lambda t: t[0])

    adjusted = [math.nan] * len(values)
    running = 0.0
    for rank, (p, index) in enumerate(finite):
        candidate = (m - rank) * p
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


# ---------------------------------------------------------------------------
# One comparison, and one family of them
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PairwiseResult:
    """One pairwise comparison. Effect size first, deliberately.

    The field order is the reporting order the analysis plan fixes in §8.2,
    and `render()` follows it. There is no constructor path that produces a
    p-value without the effect size and its interval beside it.
    """

    hypothesis: Hypothesis
    effects: PairedEffects
    test: WilcoxonResult
    label: Label
    n_scenarios: int
    n_dropped: int
    p_holm: float = math.nan
    family_size: int = 0
    #: Whether the judge resolved this comparison's measure at all
    #: (`carelite.stats.instrument`). `None` when the diagnostic was not run.
    #: When it says the measure is untestable, `render()` says so ABOVE the
    #: p-value, because a reader who sees the p-value first has already
    #: misunderstood the result.
    testability: MeasureTestability | None = None

    @property
    def not_testable(self) -> bool:
        return self.testability is not None and not self.testability.testable

    @property
    def observed_direction(self) -> str:
        return self.effects.direction

    @property
    def direction_as_registered(self) -> bool | None:
        """True/False when a direction was predicted, `None` when none was.

        Outcome 6 predicts "no significant difference", which has no direction
        to agree or disagree with, so it returns `None` rather than a misleading
        boolean.
        """
        expected = self.hypothesis.expected_direction
        if expected == "=":
            return None
        return self.observed_direction == expected

    def significant(self, alpha: float = 0.05) -> bool:
        """Holm-corrected significance. Never the raw p-value.

        Note that this stays `False` for an untestable comparison, which is
        correct as arithmetic and misleading as English — `not_testable` is the
        field to check before turning this into a sentence, and `render()` does.
        """
        return (not math.isnan(self.p_holm)) and self.p_holm < alpha

    def render(self, alpha: float = 0.05) -> str:
        e = self.effects
        head = (
            f"{self.hypothesis.pair_label} on {self.hypothesis.measure.label} [{self.label.tag()}]"
        )
        instrument_lines: list[str] = []
        # Above the effect size, not below the p-value. A reader who has already
        # read the number has already formed the impression the caveat exists to
        # prevent, which is the same ordering argument the instrument diagnostic
        # rests on.
        for caveat in self.hypothesis.caveats:
            instrument_lines.append(f"    !!! {caveat}")
        if self.testability is not None and self.testability.note:
            prefix = "    !!! " if self.not_testable else "    "
            instrument_lines.append(f"{prefix}{self.testability.note}")
        if self.not_testable:
            # How thin the evidence actually is, in the two numbers that say it.
            # A degenerate dimension is not necessarily *flat* -- naturalness on
            # this run left 36 of 60 pairs non-tied while ritualistic left 4 --
            # and collapsing that difference into one verdict would overstate one
            # case and understate the other.
            tied = self.n_scenarios - self.test.n_nonzero
            hl = self.effects.hodges_lehmann
            instrument_lines.append(
                f"        the test rests on {self.test.n_nonzero} of {self.n_scenarios} "
                f"scenarios ({tied} tied exactly); shift = {hl.point:+.3f} rubric points "
                f"[{hl.ci.low:+.3f}, {hl.ci.high:+.3f}]"
                + (
                    " — interval includes zero"
                    if not hl.ci.excludes_zero
                    else " — interval excludes zero"
                )
            )
        effect_line = (
            f"    effect (rank-biserial) {e.rank_biserial.render()}"
            f"  |  dz {e.cohens_dz.point:+.3f} "
            f"[{e.cohens_dz.ci.low:+.3f}, {e.cohens_dz.ci.high:+.3f}]"
            f"  |  Hodges-Lehmann {e.hodges_lehmann.point:+.3f} points "
            f"[{e.hodges_lehmann.ci.low:+.3f}, {e.hodges_lehmann.ci.high:+.3f}]"
        )
        if self.not_testable:
            verdict = "NOT TESTABLE — the p-value below describes the judge, not the conditions"
        elif self.significant(alpha):
            verdict = f"significant at alpha = {alpha}"
        else:
            verdict = f"not significant at alpha = {alpha}"
        p_line = (
            f"    then p: Wilcoxon W = {self.test.statistic:.1f}, "
            f"p = {self.test.p_value:.4g}, Holm-adjusted p = {self.p_holm:.4g} "
            f"(family of {self.family_size}), {verdict}"
        )
        direction = self.direction_as_registered
        if direction is None:
            dir_line = "    predicted before the run: no difference"
        else:
            dir_line = (
                f"    predicted direction {self.hypothesis.expected_direction}, observed "
                f"{self.observed_direction} — "
                f"{'as predicted' if direction else 'AGAINST the predicted direction'}"
            )
        counts = (
            f"    n = {self.n_scenarios} paired scenarios"
            f"{f' ({self.n_dropped} dropped for an incomplete pair)' if self.n_dropped else ''}"
            f", {self.test.n_nonzero} with a nonzero difference"
        )
        return "\n".join([head, *instrument_lines, effect_line, p_line, dir_line, counts])


def run_pairwise(
    long: pd.DataFrame,
    hypothesis: Hypothesis,
    *,
    rater_type: str | None = None,
    statuses: Mapping[str, EvidenceStatus] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    extra_reasons: Iterable[str] = (),
    discrimination: Mapping[str, Discrimination] | None = None,
) -> PairwiseResult | None:
    """Effect sizes then the test, for one hypothesis. `None` if no pairs survive.

    `long` is a long-format score frame (see `carelite.stats.data`). Cell means
    are recomputed here rather than passed in, so the `to_quality` transform is
    applied on this call's own data and cannot be inherited from somewhere it
    was skipped.

    A hypothesis retired by decision (`not_computable_reason`) returns `None`
    without touching the data. It is not tested even if rows for it happen to be
    present, because the decision to retire it was taken on the sample's
    provenance rather than on its size, and quietly analysing it because the
    rows exist would defeat the decision.

    `discrimination` is `carelite.stats.instrument`'s per-dimension verdict. When
    supplied, the result carries whether its measure was resolvable at all, so
    the untestable case can never be rendered as a plain non-significant one.

    Raises:
        carelite.stats.arms.MixedBackendError: when one of the two conditions in
            `long` carries rows from two serving stacks. See `_backend_reasons`:
            after D13 that is not an arm, and computing an effect over it would
            produce a number for a comparison nobody specified.
    """
    if hypothesis.retired_by_decision:
        return None
    backend_reasons = _backend_reasons(long, hypothesis)
    cells = cell_means(long, hypothesis.measure)
    scope = _scope_for(long, rater_type)
    matrix = paired_matrix(cells, (hypothesis.left, hypothesis.right), rater_type=rater_type)
    available = cells["scenario_id"].nunique() if not cells.empty else 0
    if matrix.empty:
        return None

    left = matrix[str(hypothesis.left)].to_numpy(dtype=float)
    right = matrix[str(hypothesis.right)].to_numpy(dtype=float)

    testability = (
        measure_testability(hypothesis.measure, discrimination)
        if discrimination is not None
        else None
    )
    reasons = [
        *extra_reasons,
        *(_caveat_headline(c) for c in hypothesis.caveats),
        *backend_reasons,
    ]
    if testability is not None and not testability.testable:
        reasons.append(
            "the judge did not resolve "
            + ", ".join(testability.degenerate_dimensions)
            + " on this run, so this comparison is untestable rather than null"
        )

    return PairwiseResult(
        hypothesis=hypothesis,
        effects=paired_effects(left, right, n_boot=n_boot, seed=seed),
        test=wilcoxon_paired(left, right),
        label=label_for(
            hypothesis.measure,
            prespecified=hypothesis.prespecified,
            rater_scope=scope,
            statuses=statuses,
            extra_reasons=reasons,
        ),
        n_scenarios=int(matrix.shape[0]),
        n_dropped=max(0, int(available) - int(matrix.shape[0])),
        testability=testability,
    )


def _caveat_headline(caveat: str) -> str:
    """The first sentence of a caveat, for the label tag.

    The tag is read inline — in a table cell, a figure caption, a CSV column — and
    a paragraph pasted into it stops being read at all. The caveats are written
    with their claim in the opening sentence for exactly this reason, so the tag
    names the objection and `render()` and `effect-sizes.csv` carry it in full.
    Both come from one string; there is no second wording to drift.
    """
    head = caveat.split(". ", 1)[0].strip()
    return head.rstrip(".") if head else caveat


def _backend_reasons(long: pd.DataFrame, hypothesis: Hypothesis) -> list[str]:
    """Check the arms this comparison rests on came from one serving stack each.

    Only comparisons touching a condition that exists under two stacks are
    checked; for the five conditions this study served one way, there is nothing
    to confuse. D13 makes `LC` the one such condition.

    Two outcomes, deliberately different in severity:

    * **A frame that pools two stacks within one of the two conditions raises.**
      The frame carries `served_by` and it says the arm is two arms. There is no
      reading of that comparison that means anything, so it does not get computed
      and quietly labelled — `carelite.stats.arms.restrict_to_analysis_arms` is
      the selection that makes it well-defined.
    * **A frame with no `served_by` column at all is demoted, not refused.** It
      may be a legacy read or a hand-built fixture, and the LC rows in it cannot
      be confirmed as the vLLM arm. That uncertainty is recorded on the label so
      it travels into the rendered result and the CSV, rather than being decided
      by a guess in either direction.
    """
    from carelite.stats.arms import (
        AMBIGUOUS_WITHOUT_BACKEND,
        assert_single_backend_per_condition,
    )

    touched = {str(hypothesis.left), str(hypothesis.right)} & AMBIGUOUS_WITHOUT_BACKEND
    if not touched or long.empty or "condition" not in long.columns:
        return []
    if "served_by" not in long.columns:
        return [
            "this frame carries no `served_by` column, so the "
            + ", ".join(sorted(touched))
            + " rows could not be confirmed as the single-stack analysis arm D13 defines"
        ]
    pair = long[long["condition"].astype(str).isin({str(hypothesis.left), str(hypothesis.right)})]
    assert_single_backend_per_condition(pair, what=f"the {hypothesis.pair_label} comparison")
    return []


def _scope_for(long: pd.DataFrame, rater_type: str | None) -> RaterScope:
    if rater_type is not None:
        return RaterScope.from_rater_types([rater_type])
    if "rater_type" in long.columns:
        return RaterScope.from_rater_types(long["rater_type"].dropna().unique())
    return RaterScope.MIXED


def friedman_across_conditions(
    long: pd.DataFrame,
    *,
    dimensions: Sequence[str] = RUBRIC_DIMENSIONS,
    conditions: Sequence[Condition] = FRIEDMAN_CONDITIONS,
    rater_type: str | None = None,
    statuses: Mapping[str, EvidenceStatus] | None = None,
    discrimination: Mapping[str, Discrimination] | None = None,
) -> tuple[FriedmanResult, ...]:
    """The eleven planned omnibus tests (§8.1), on `to_quality()` scores.

    A dimension the judge did not resolve produces a large omnibus p-value for
    the same reason it produces a large pairwise one, so `degenerate` is carried
    on the row and the rendered table marks it. An omnibus test on a constant
    dimension is not a finding of no difference across conditions.
    """
    scope = _scope_for(long, rater_type)
    results: list[FriedmanResult] = []
    for key in dimensions:
        m = measure(key)
        matrix = paired_matrix(cell_means(long, m), conditions, rater_type=rater_type)
        statistic, p_value, df, n_blocks = friedman_omnibus(matrix)
        results.append(
            FriedmanResult(
                measure_key=key,
                conditions=tuple(str(c) for c in conditions),
                statistic=statistic,
                p_value=p_value,
                df=df,
                n_blocks=n_blocks,
                label=label_for(m, prespecified=True, rater_scope=scope, statuses=statuses),
                degenerate=(
                    discrimination is not None
                    and discrimination.get(key) is Discrimination.DEGENERATE
                ),
            )
        )
    adjusted = holm_bonferroni([r.p_value for r in results])
    return tuple(
        replace(r, p_holm_within_omnibus=p) for r, p in zip(results, adjusted, strict=True)
    )


@dataclass(frozen=True, slots=True)
class FamilyResult:
    """One Holm-corrected family of pairwise tests, plus the omnibus tests.

    `name` and `correction_family` are printed on every render because the
    family is the thing a reader has to check, and a table that does not say
    what was corrected across cannot be checked.
    """

    name: str
    alpha: float
    results: tuple[PairwiseResult, ...]
    #: The size of the correction family, which is fixed by the analysis plan
    #: before the data exist. NOT `len(results)`: a planned test that could
    #: not be computed still consumes its share of the correction, so the two
    #: numbers differ whenever a condition is missing from the data and the
    #: rendered `m` has to be this one.
    family_size: int = 0
    friedman: tuple[FriedmanResult, ...] = ()
    correction: str = "Holm-Bonferroni"
    correction_family: str = (
        "the whole set of pairwise comparisons planned in advance, across measures and dimensions "
        "together, not per dimension"
    )
    notes: tuple[str, ...] = ()

    @property
    def confirmatory(self) -> tuple[PairwiseResult, ...]:
        return tuple(r for r in self.results if r.label.is_confirmatory)

    @property
    def exploratory(self) -> tuple[PairwiseResult, ...]:
        return tuple(r for r in self.results if not r.label.is_confirmatory)

    def by_key(self, key: str) -> PairwiseResult | None:
        for r in self.results:
            if r.hypothesis.key == key:
                return r
        return None

    def render(self) -> str:
        lines = [
            f"{self.name.upper()}",
            f"  correction: {self.correction} across {self.correction_family}",
            f"  family size m = {self.family_size or len(self.results)} "
            f"({len(self.results)} computed), alpha = {self.alpha}",
            "  effect sizes and 95% bootstrap CIs are reported BEFORE p-values "
            "(analysis plan §8.2)",
        ]
        for note in self.notes:
            lines.append(f"  note: {note}")
        if self.friedman:
            lines.append("")
            lines.append(
                "  FRIEDMAN OMNIBUS across "
                f"{{{', '.join(self.friedman[0].conditions)}}} — not part of the Holm family; "
                "the within-omnibus adjustment is shown for reference only"
            )
            lines.append(
                f"    {'dimension':<14}{'chi2':>9}{'df':>4}{'n':>5}{'p':>10}"
                f"{'p(omnibus Holm)':>18}  instrument"
            )
            for f in self.friedman:
                marker = "DEGENERATE — p uninterpretable" if f.degenerate else "resolved"
                lines.append(
                    f"    {f.measure_key:<14}{f.statistic:>9.3f}{f.df:>4}{f.n_blocks:>5}"
                    f"{f.p_value:>10.4g}{f.p_holm_within_omnibus:>18.4g}  {marker}"
                )
        lines.append("")
        for r in self.results:
            lines.append(r.render(self.alpha))
            lines.append("")
        return "\n".join(lines).rstrip()


def run_family(
    long: pd.DataFrame,
    hypotheses: Sequence[Hypothesis] = CONFIRMATORY_FAMILY,
    *,
    name: str = "primary analysis (analysis plan §8.1)",
    alpha: float = 0.05,
    rater_type: str | None = None,
    statuses: Mapping[str, EvidenceStatus] | None = None,
    include_friedman: bool = True,
    friedman_conditions: Sequence[Condition] = FRIEDMAN_CONDITIONS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    extra_reasons: Iterable[str] = (),
    notes: Sequence[str] = (),
    discrimination: Mapping[str, Discrimination] | None = None,
) -> FamilyResult:
    """Run a whole family and Holm-correct across all of it in one step.

    The correction is applied over `hypotheses` as given. The family size used
    for the correction is the number of hypotheses submitted, including any whose
    test came out undefined -- the family is fixed by the analysis plan before
    the data exist, so a test that could not run does not make its neighbours
    easier to pass.

    That rule did real work while D11 stood: secondary outcome 3 (C vs LC) could
    not be computed and kept its slot, so m stayed at 8. D13 restored the
    comparison and all eight are now computable, which changes nothing about the
    rule — dropping to m = 7 after seeing which test could not run would lower
    every other comparison's adjusted p-value, and a correction whose size
    depends on what the data turned out to contain is a correction chosen to suit
    the data.
    """
    computed: list[tuple[Hypothesis, PairwiseResult | None]] = []
    for h in hypotheses:
        computed.append(
            (
                h,
                run_pairwise(
                    long,
                    h,
                    rater_type=rater_type,
                    statuses=statuses,
                    n_boot=n_boot,
                    seed=seed,
                    extra_reasons=extra_reasons,
                    discrimination=discrimination,
                ),
            )
        )

    family_size = len(computed)
    p_values = [(r.test.p_value if r is not None else math.nan) for _, r in computed]
    adjusted = holm_bonferroni(p_values, family_size=family_size)

    results: list[PairwiseResult] = []
    missing: list[str] = []
    retired: list[Hypothesis] = []
    for (h, r), p_adj in zip(computed, adjusted, strict=True):
        if r is None:
            if h.retired_by_decision:
                retired.append(h)
            else:
                missing.append(h.key)
            continue
        results.append(replace(r, p_holm=p_adj, family_size=family_size))

    all_notes = list(notes)
    for h in retired:
        all_notes.append(f"{h.key} NOT COMPUTED. {h.not_computable_reason}")
    if missing:
        all_notes.append(
            "no paired data for "
            + ", ".join(missing)
            + f"; these still count toward the family size (m = {family_size})"
        )

    friedman = (
        friedman_across_conditions(
            long,
            conditions=friedman_conditions,
            rater_type=rater_type,
            statuses=statuses,
            discrimination=discrimination,
        )
        if include_friedman
        else ()
    )

    return FamilyResult(
        name=name,
        alpha=alpha,
        results=tuple(results),
        family_size=family_size,
        friedman=friedman,
        notes=tuple(all_notes),
    )
