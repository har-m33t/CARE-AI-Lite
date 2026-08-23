"""The judge itself: prompt, call, parse, ground, aggregate.

Reading order for the two things that decide whether the study's numbers mean
anything:

**Grounding is enforced here, not advised.** `_admit` is the only path by which
a score reaches a `RubricScore`, and it drops any score whose evidence span
cannot be located in the response. A dropped dimension becomes `None` -> NULL,
which is loud. It never becomes a number.

**Aggregation goes through `to_quality`, never by hand.** `ritualistic` is
reverse-coded and `JudgeResult` stores the raw value — that is what the schema's
`rubric_score.ritualistic` column holds and what the human raters produce, so
raw is the interchange format. Anything that mixes dimensions calls
`quality_scores()`, which delegates to `carelite.eval.rubric.dimensions`. There
is no `6 - x` anywhere in this file, deliberately: a second implementation of
the reversal is a second place for the sign to be wrong.

Two sampling regimes, both from `settings.experiment` (v3 §13):

- **Full run** — one sample at temperature 0. 1,080 generations, ~8h.
- **Validation subset** — five samples at 0.7, median reported, inter-sample
  variance reported as a stability metric. Five samples over the whole run would
  be ~35h, which is why the split exists.

`for_full_run()` and `for_validation()` read those numbers from config rather
than taking them as arguments, so the two regimes cannot drift apart from what
the pre-registration says they are.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from carelite.config import get_settings, seed_for
from carelite.eval.judge.cache import CachedSample, JudgeCache, cache_key
from carelite.eval.judge.client import ChatClient
from carelite.eval.judge.grounding import GroundedSpan, SpanRejection, ground_span
from carelite.eval.judge.parsing import (
    JudgeParseError,
    ParsedJudgeOutput,
    parse_judge_output,
    score_in_range,
)
from carelite.eval.judge.prompt import (
    JUDGE_PROMPT_VERSION,
    OptionOrder,
    build_judge_prompt,
    presented_response,
)
from carelite.eval.rubric.dimensions import RUBRIC_VERSION, to_quality
from carelite.types import RUBRIC_DIMENSIONS, Generation, RaterType, RubricScore

__all__ = [
    "AdmittedScore",
    "DimensionResult",
    "JudgeResult",
    "JudgeSample",
    "LLMJudge",
]


@dataclass(frozen=True, slots=True)
class AdmittedScore:
    """One dimension from one sample, after the grounding rule has been applied.

    `score is None` means rejected, and `rejection` says why. The two are never
    both set and never both absent.
    """

    dimension: str
    score: int | None
    span: GroundedSpan | None
    rationale: str
    rejection: SpanRejection | None

    @property
    def admitted(self) -> bool:
        return self.score is not None


@dataclass(frozen=True, slots=True)
class JudgeSample:
    """One call to the judge, parsed and grounded. Eleven `AdmittedScore`s."""

    sample_idx: int
    order: OptionOrder
    seed: int | None
    scores: Mapping[str, AdmittedScore]
    safety_flags: tuple[str, ...]
    raw_output: str
    from_cache: bool
    latency_ms: int | None = None
    unknown_keys: tuple[str, ...] = ()

    def admitted_scores(self) -> dict[str, int]:
        return {k: v.score for k, v in self.scores.items() if v.score is not None}


@dataclass(frozen=True, slots=True)
class DimensionResult:
    """One dimension aggregated across however many samples survived grounding.

    `score` is the reported value: the sample itself for a single-pass run, the
    median for the five-sample validation regime. Raw scale, so `ritualistic`
    is still higher-is-worse — call `JudgeResult.quality_scores()` before any
    cross-dimension arithmetic.
    """

    dimension: str
    score: int | None
    #: Every admitted raw score across samples, in sample order.
    raw_scores: tuple[int, ...]
    #: Exact median before integer rounding. `None` when nothing was admitted.
    median: float | None
    #: Sample variance across samples. `None` for fewer than two samples — this
    #: is the v3 §13 self-consistency statistic and reporting 0.0 for n=1 would
    #: fabricate perfect stability.
    variance: float | None
    #: Widest disagreement between two samples on this dimension.
    score_range: int | None
    span: str | None
    span_source: str | None
    span_exact: bool
    rationale: str
    #: One entry per sample whose score was thrown away, with the reason.
    rejections: tuple[SpanRejection, ...]

    @property
    def n_admitted(self) -> int:
        return len(self.raw_scores)


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """Everything the judge concluded about one generation, with its provenance."""

    generation_id: str
    judge_model: str
    judge_digest: str
    prompt_version: str
    rubric_version: str
    temperature: float
    n_samples_requested: int
    order: OptionOrder
    dimensions: Mapping[str, DimensionResult]
    samples: tuple[JudgeSample, ...]
    safety_flags: tuple[str, ...] = ()
    rater_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # -- reporting -------------------------------------------------------

    def scores(self) -> dict[str, int | None]:
        """Raw reported score per dimension. `ritualistic` is higher-is-worse."""
        return {k: self.dimensions[k].score for k in RUBRIC_DIMENSIONS}

    def quality_scores(self) -> dict[str, int | None]:
        """Every dimension on the common higher-is-better scale.

        The only sanctioned way to compare or combine dimensions. Delegates to
        `carelite.eval.rubric.dimensions.to_quality`; the reversal is not
        reimplemented here.
        """
        return {
            key: (None if value is None else to_quality(key, value))
            for key, value in self.scores().items()
        }

    def evidence_spans(self) -> dict[str, str]:
        """Dimension -> the verbatim quote that survived grounding."""
        return {k: d.span for k, d in self.dimensions.items() if d.span}

    @property
    def n_rejected(self) -> int:
        """Dimensions with no admissible score at all."""
        return sum(1 for d in self.dimensions.values() if d.score is None)

    @property
    def complete(self) -> bool:
        """True if all eleven dimensions survived grounding."""
        return self.n_rejected == 0

    # -- persistence -----------------------------------------------------

    def to_rubric_score(self, rater_id: str | None = None, sample_idx: int = 0) -> RubricScore:
        """The aggregate as a `rubric_score` row.

        Written out dimension by dimension rather than splatted: the eleven
        keys are a frozen contract and a typo in a splatted dict is silently
        dropped by pydantic's extra-ignore.
        """
        s = self.scores()
        return RubricScore(
            generation_id=self.generation_id,
            rater_type=RaterType.LLM_JUDGE,
            rater_id=rater_id or self.rater_id or self.judge_model,
            name=s["name"],
            understand=s["understand"],
            respect=s["respect"],
            support=s["support"],
            explore=s["explore"],
            ib=s["ib"],
            epp=s["epp"],
            de=s["de"],
            ie=s["ie"],
            naturalness=s["naturalness"],
            ritualistic=s["ritualistic"],
            safety_flags=list(self.safety_flags),
            evidence_spans=self.evidence_spans(),
        )

    def per_sample_rubric_scores(self, rater_id: str | None = None) -> list[RubricScore]:
        """One `rubric_score` row per self-consistency sample.

        `rubric_score` carries a `sample_idx` column precisely so the five
        validation samples can be stored individually and the variance
        recomputed from the table rather than trusted from a report. The
        aggregate row is stored separately under a `-median` rater id so it
        cannot collide with sample 0 on the table's unique key.
        """
        base = rater_id or self.rater_id or self.judge_model
        rows: list[RubricScore] = []
        for sample in self.samples:
            s = {k: v.score for k, v in sample.scores.items()}
            rows.append(
                RubricScore(
                    generation_id=self.generation_id,
                    rater_type=RaterType.LLM_JUDGE,
                    rater_id=base,
                    name=s.get("name"),
                    understand=s.get("understand"),
                    respect=s.get("respect"),
                    support=s.get("support"),
                    explore=s.get("explore"),
                    ib=s.get("ib"),
                    epp=s.get("epp"),
                    de=s.get("de"),
                    ie=s.get("ie"),
                    naturalness=s.get("naturalness"),
                    ritualistic=s.get("ritualistic"),
                    safety_flags=list(sample.safety_flags),
                    evidence_spans={
                        k: v.span.text for k, v in sample.scores.items() if v.span is not None
                    },
                )
            )
        return rows


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _median_int(values: Sequence[int]) -> tuple[int, float]:
    """Median of an odd or even sample, plus the integer to report.

    Five samples give an integer median, so the even case only arises when
    grounding rejected some of them. Ties round half **up**, which is stated
    rather than left to `round()`'s banker's rounding: an even/odd asymmetry in
    a tie-break rule is the kind of thing that produces a 0.02 effect out of
    nothing and is impossible to find later.
    """
    exact = float(statistics.median(values))
    reported = int(exact + 0.5) if exact >= 0 else int(exact - 0.5)
    return reported, exact


def _aggregate(dimension: str, samples: Sequence[JudgeSample]) -> DimensionResult:
    """Collapse one dimension across samples, keeping the rejections visible."""
    admitted = [s.scores[dimension] for s in samples if dimension in s.scores]
    raw = tuple(a.score for a in admitted if a.score is not None)
    rejections = tuple(a.rejection for a in admitted if a.rejection is not None)

    if not raw:
        return DimensionResult(
            dimension=dimension,
            score=None,
            raw_scores=(),
            median=None,
            variance=None,
            score_range=None,
            span=None,
            span_source=None,
            span_exact=False,
            rationale="",
            rejections=rejections,
        )

    reported, exact = _median_int(raw)
    variance = statistics.variance(raw) if len(raw) >= 2 else None

    # Report the span from the sample closest to the reported score, so the
    # quote in the database actually justifies the number next to it. Ties go to
    # the earliest sample, which is deterministic.
    with_span = [a for a in admitted if a.span is not None and a.score is not None]
    chosen = min(with_span, key=lambda a: abs((a.score or 0) - reported)) if with_span else None

    return DimensionResult(
        dimension=dimension,
        score=reported,
        raw_scores=raw,
        median=exact,
        variance=variance,
        score_range=max(raw) - min(raw),
        span=chosen.span.text if chosen and chosen.span else None,
        span_source=chosen.span.source if chosen and chosen.span else None,
        span_exact=bool(chosen and chosen.span and chosen.span.exact),
        rationale=chosen.rationale if chosen else "",
        rejections=rejections,
    )


def _admit(parsed: ParsedJudgeOutput, response: str, presented: str) -> dict[str, AdmittedScore]:
    """Apply the v3 §13 grounding rule to every dimension of one sample.

    Order of checks is deliberate: a missing score is reported as `NO_SCORE`
    even if the span is also missing, because "the judge skipped the dimension"
    and "the judge scored it but would not quote" are different failures and
    the validation report separates them.
    """
    out: dict[str, AdmittedScore] = {}
    for key in RUBRIC_DIMENSIONS:
        pd = parsed.get(key)

        if pd.score is None:
            out[key] = AdmittedScore(key, None, None, pd.rationale, SpanRejection.NO_SCORE)
            continue
        if not score_in_range(pd.score):
            out[key] = AdmittedScore(
                key, None, None, pd.rationale, SpanRejection.SCORE_OUT_OF_RANGE
            )
            continue

        span, rejection = ground_span(pd.span, response, presented)
        if span is None:
            out[key] = AdmittedScore(key, None, None, pd.rationale, rejection)
            continue

        out[key] = AdmittedScore(key, pd.score, span, pd.rationale, None)
    return out


def _all_rejected(reason: SpanRejection, rationale: str = "") -> dict[str, AdmittedScore]:
    return {k: AdmittedScore(k, None, None, rationale, reason) for k in RUBRIC_DIMENSIONS}


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------


@dataclass
class LLMJudge:
    """Scores one generation on all eleven dimensions, with enforced grounding.

    Construct through `for_full_run` or `for_validation` unless you are
    deliberately running an off-protocol configuration — those two classmethods
    are where the pre-registered sampling regimes live.
    """

    client: ChatClient
    temperature: float
    n_samples: int
    order: OptionOrder = OptionOrder.ASCENDING
    cache: JudgeCache | None = None
    rater_id: str = ""
    prompt_version: str = JUDGE_PROMPT_VERSION
    rubric_version: str = RUBRIC_VERSION

    @classmethod
    def for_full_run(
        cls,
        client: ChatClient,
        *,
        cache: JudgeCache | None = None,
        order: OptionOrder = OptionOrder.ASCENDING,
        rater_id: str = "",
    ) -> LLMJudge:
        """Single-pass at temperature 0, per `settings.experiment`."""
        exp = get_settings().experiment
        return cls(
            client=client,
            temperature=exp.judge_temperature_full_run,
            n_samples=exp.judge_samples_full_run,
            order=order,
            cache=cache,
            rater_id=rater_id,
        )

    @classmethod
    def for_validation(
        cls,
        client: ChatClient,
        *,
        cache: JudgeCache | None = None,
        order: OptionOrder = OptionOrder.ASCENDING,
        rater_id: str = "",
    ) -> LLMJudge:
        """Five samples at temperature 0.7, median reported, per `settings.experiment`."""
        exp = get_settings().experiment
        return cls(
            client=client,
            temperature=exp.judge_temperature_validation,
            n_samples=exp.judge_samples_validation,
            order=order,
            cache=cache,
            rater_id=rater_id,
        )

    # -- one sample ------------------------------------------------------

    def _key(self, generation_id: str, sample_idx: int) -> str:
        return cache_key(
            generation_id=generation_id,
            model=self.client.model,
            digest=self.client.digest,
            prompt_version=self.prompt_version,
            rubric_version=self.rubric_version,
            temperature=self.temperature,
            sample_idx=sample_idx,
            order=str(self.order),
        )

    def judge_sample(
        self,
        *,
        generation_id: str,
        scenario_text: str,
        response_text: str,
        sample_idx: int,
    ) -> JudgeSample:
        """One judged sample, served from cache when the key already exists."""
        presented = presented_response(response_text)
        key = self._key(generation_id, sample_idx)
        seed = seed_for(generation_id, f"judge-{self.order}", sample_idx)

        cached = self.cache.get(key) if self.cache is not None else None
        if cached is not None:
            return self._build_sample(
                raw=cached.raw_output,
                sample_idx=sample_idx,
                seed=cached.seed,
                response=response_text,
                presented=presented,
                from_cache=True,
                latency_ms=cached.latency_ms,
            )

        prompt = build_judge_prompt(
            scenario_text=scenario_text,
            response_text=response_text,
            order=self.order,
        )
        started = time.monotonic()
        raw = self.client.chat(prompt.as_messages(), temperature=self.temperature, seed=seed)
        latency_ms = int((time.monotonic() - started) * 1000)

        if self.cache is not None:
            self.cache.put(
                CachedSample(
                    key=key,
                    generation_id=generation_id,
                    model=self.client.model,
                    digest=self.client.digest,
                    prompt_version=self.prompt_version,
                    rubric_version=self.rubric_version,
                    temperature=self.temperature,
                    sample_idx=sample_idx,
                    order=str(self.order),
                    seed=seed,
                    raw_output=raw,
                    latency_ms=latency_ms,
                )
            )

        return self._build_sample(
            raw=raw,
            sample_idx=sample_idx,
            seed=seed,
            response=response_text,
            presented=presented,
            from_cache=False,
            latency_ms=latency_ms,
        )

    def _build_sample(
        self,
        *,
        raw: str,
        sample_idx: int,
        seed: int | None,
        response: str,
        presented: str,
        from_cache: bool,
        latency_ms: int | None,
    ) -> JudgeSample:
        """Parse + ground one raw output. Pure; no model call, no cache write."""
        try:
            parsed = parse_judge_output(raw)
        except JudgeParseError as exc:
            # An unparseable sample is a total rejection, not an exception: one
            # bad sample out of five must not lose the other four, and one bad
            # generation must not end an eight-hour run.
            return JudgeSample(
                sample_idx=sample_idx,
                order=self.order,
                seed=seed,
                scores=_all_rejected(SpanRejection.NO_SCORE, rationale=f"unparseable: {exc}"),
                safety_flags=(),
                raw_output=raw,
                from_cache=from_cache,
                latency_ms=latency_ms,
            )

        return JudgeSample(
            sample_idx=sample_idx,
            order=self.order,
            seed=seed,
            scores=_admit(parsed, response, presented),
            safety_flags=tuple(parsed.safety_flags),
            raw_output=raw,
            from_cache=from_cache,
            latency_ms=latency_ms,
            unknown_keys=tuple(parsed.unknown_keys),
        )

    # -- one generation --------------------------------------------------

    def score(self, generation: Generation, scenario_text: str) -> JudgeResult:
        """Judge one generation with the configured sampling regime."""
        return self.score_text(
            generation_id=generation.generation_id,
            scenario_text=scenario_text,
            response_text=generation.response,
        )

    def score_text(
        self,
        *,
        generation_id: str,
        scenario_text: str,
        response_text: str,
    ) -> JudgeResult:
        """Judge a bare `(scenario, response)` pair.

        Exists because the calibration set and the fixtures are responses
        without `Generation` rows, and requiring a database row to exercise the
        judge would make the whole lane untestable before the orchestrator lane
        lands.
        """
        samples = tuple(
            self.judge_sample(
                generation_id=generation_id,
                scenario_text=scenario_text,
                response_text=response_text,
                sample_idx=i,
            )
            for i in range(self.n_samples)
        )

        dimensions = {key: _aggregate(key, samples) for key in RUBRIC_DIMENSIONS}

        flags: list[str] = []
        for sample in samples:
            for flag in sample.safety_flags:
                if flag not in flags:
                    flags.append(flag)

        return JudgeResult(
            generation_id=generation_id,
            judge_model=self.client.model,
            judge_digest=self.client.digest,
            prompt_version=self.prompt_version,
            rubric_version=self.rubric_version,
            temperature=self.temperature,
            n_samples_requested=self.n_samples,
            order=self.order,
            dimensions=dimensions,
            samples=samples,
            safety_flags=tuple(flags),
            rater_id=self.rater_id or self.client.model,
        )
