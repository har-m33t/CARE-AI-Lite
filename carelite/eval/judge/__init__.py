"""LLM-as-judge for the CARELite rubric, and the study that validates it.

The judge is `gpt-oss:20b`, judging generations from `gemma4:12b`. **Different
model families.** Build plan v3 §13 asks for that independence to be reported
prominently, so it is stated here, in `client.py`, and in the rendered
validation report: a judge sharing weights or post-training with the generator
would be scoring its own dialect, and the agreement number would mean nothing.

Two invariants hold everywhere in this package.

**Every score carries a verbatim span, or it is not a score.** `grounding.py`
locates each cited quote in the response; a quote that cannot be found makes the
dimension `None`, which becomes a NULL in `rubric_score`. Nothing rescues it,
defaults it, or logs-and-continues. An absent score costs one cell; a fabricated
one contaminates every mean it enters and is indistinguishable from a real one.

**`ritualistic` is reverse-coded and the reversal lives in exactly one place.**
Raw scores stay raw all the way to the database — that is what the schema holds
and what humans produce. Anything that mixes dimensions calls
`JudgeResult.quality_scores()`, which delegates to
`carelite.eval.rubric.dimensions.to_quality`. There is no `6 - x` in this
package.

Typical use::

    from carelite.eval.judge import JudgeCache, LLMJudge, OllamaChatClient, judge_generations

    with JudgeCache(runs_dir / "judge" / "full.jsonl") as cache:
        judge = LLMJudge.for_full_run(OllamaChatClient(), cache=cache)
        run = judge_generations(generations, scenario_texts, judge)

Interrupt it and run it again: the cache is keyed per sample, so it picks up
where it stopped.
"""

from carelite.eval.judge.agreement import (
    AgreementResult,
    Metric,
    krippendorff_alpha,
    paired_series,
    spearman_rho,
)
from carelite.eval.judge.arms import (
    LC_ANALYSIS_BACKEND,
    LC_EQUIVALENCE_BACKEND,
    SERVING_BACKENDS,
    Arm,
    MixedBackendError,
    UnpairableCells,
    fetch_arm,
    is_partial_record,
    lc_analysis_arm,
    lc_equivalence_sample,
    pair_cells,
)
from carelite.eval.judge.backend_equivalence import (
    BACKEND_CONFOUNDS,
    MIN_UNITS_FOR_EQUIVALENCE_CLAIM,
    BackendEquivalence,
    DimensionEquivalence,
    compare_backends,
    run_backend_equivalence,
)
from carelite.eval.judge.cache import CachedSample, JudgeCache, cache_key
from carelite.eval.judge.client import ChatClient, JudgeCallError, OllamaChatClient, ReplayClient
from carelite.eval.judge.grounding import GroundedSpan, SpanRejection, ground_span, locate
from carelite.eval.judge.judge import (
    AdmittedScore,
    DimensionResult,
    JudgeResult,
    JudgeSample,
    LLMJudge,
)
from carelite.eval.judge.parsing import JudgeParseError, ParsedJudgeOutput, parse_judge_output
from carelite.eval.judge.prompt import (
    JUDGE_PROMPT_VERSION,
    OptionOrder,
    build_judge_prompt,
    rubric_block,
)
from carelite.eval.judge.runner import JudgeError, JudgeRun, RunProgress, judge_generations
from carelite.eval.judge.validation import (
    MIN_ALPHA_FOR_CONFIRMATORY,
    MIN_RHO_FOR_CONFIRMATORY,
    MIN_UNITS_FOR_CONFIRMATORY,
    N_SPANS_TO_REVIEW,
    VALIDATION_PLAN_VERSION,
    DimensionValidity,
    EvidenceStatus,
    PositionalBias,
    SelfConsistency,
    SpanGroundingAudit,
    SpanReviewItem,
    SpanReviewVerdict,
    SpanSupportReport,
    ValidationReport,
    build_validation_report,
    classify_dimension,
    judge_among_raters_alpha,
    judge_human_validity,
    positional_bias,
    sample_spans_for_review,
    self_consistency,
    span_grounding_audit,
    span_support_rate,
)

__all__ = [
    "BACKEND_CONFOUNDS",
    "JUDGE_PROMPT_VERSION",
    "LC_ANALYSIS_BACKEND",
    "LC_EQUIVALENCE_BACKEND",
    "MIN_ALPHA_FOR_CONFIRMATORY",
    "MIN_RHO_FOR_CONFIRMATORY",
    "MIN_UNITS_FOR_CONFIRMATORY",
    "MIN_UNITS_FOR_EQUIVALENCE_CLAIM",
    "N_SPANS_TO_REVIEW",
    "SERVING_BACKENDS",
    "VALIDATION_PLAN_VERSION",
    "AdmittedScore",
    "AgreementResult",
    "Arm",
    "BackendEquivalence",
    "CachedSample",
    "ChatClient",
    "DimensionEquivalence",
    "DimensionResult",
    "DimensionValidity",
    "EvidenceStatus",
    "GroundedSpan",
    "JudgeCache",
    "JudgeCallError",
    "JudgeError",
    "JudgeParseError",
    "JudgeResult",
    "JudgeRun",
    "JudgeSample",
    "LLMJudge",
    "Metric",
    "MixedBackendError",
    "OllamaChatClient",
    "OptionOrder",
    "ParsedJudgeOutput",
    "PositionalBias",
    "ReplayClient",
    "RunProgress",
    "SelfConsistency",
    "SpanGroundingAudit",
    "SpanRejection",
    "SpanReviewItem",
    "SpanReviewVerdict",
    "SpanSupportReport",
    "UnpairableCells",
    "ValidationReport",
    "build_judge_prompt",
    "build_validation_report",
    "cache_key",
    "classify_dimension",
    "compare_backends",
    "fetch_arm",
    "ground_span",
    "is_partial_record",
    "judge_among_raters_alpha",
    "judge_generations",
    "judge_human_validity",
    "krippendorff_alpha",
    "lc_analysis_arm",
    "lc_equivalence_sample",
    "locate",
    "pair_cells",
    "paired_series",
    "parse_judge_output",
    "positional_bias",
    "rubric_block",
    "run_backend_equivalence",
    "sample_spans_for_review",
    "self_consistency",
    "span_grounding_audit",
    "span_support_rate",
    "spearman_rho",
]
