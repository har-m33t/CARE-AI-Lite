"""What a result may be *reported* as, carried in the result rather than the prose.

Two independent things can demote a number to exploratory, and they compose:

**1. It was not pre-specified.** Pre-registration §1: "Everything not explicitly
listed as primary, secondary, or a pre-specified sensitivity analysis below is
exploratory. That includes any comparison, subgroup, or figure that occurs to a
reader of the results but was not written down here first." So every analysis
object in this package carries `prespecified`, and nothing in the package can
produce an unlabelled comparison — the label is a required field, not a
convention a writer has to remember.

**2. The judge did not clear the pre-specified agreement threshold on that
dimension.** Pre-registration §9 fixes it: ordinal Krippendorff's alpha >= 0.667,
Spearman's rho >= 0.5, on >= 30 paired units, per dimension, or judge-only
results on that dimension are exploratory. That decision belongs to
`carelite.eval.judge.validation.classify_dimension` and is consumed from there;
it is not reimplemented here and the thresholds are not restated here.

**The composite rule, which the pre-registration does not state and which is
therefore an implementation decision recorded in the open:** a composite
measure is confirmatory only if *every* dimension it averages is confirmatory.
Weakest link. A NURSE composite whose `explore` component the judge cannot
measure reliably is not a reliable NURSE composite, and averaging a demoted
dimension into a confirmatory one launders it. If the project prefers a
different rule, it is a pre-registration amendment, and it changes exactly one
function below.

**The gate applies to judge scores.** A human-only analysis is not gated by the
judge's agreement with humans — that would be circular. `RaterScope` is what
distinguishes the two, and a pooled analysis containing any judge score is
gated, because it contains judge scores.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from carelite.eval.judge.validation import (
    DimensionValidity,
    EvidenceStatus,
    classify_dimension,
)
from carelite.stats.measures import Measure
from carelite.types import RUBRIC_DIMENSIONS, RaterType

__all__ = [
    "EvidenceStatus",
    "Label",
    "RaterScope",
    "dimension_statuses",
    "judge_gate_unavailable",
    "label_for",
    "measure_status",
    "status_from_agreement",
]


class RaterScope(StrEnum):
    """Whose scores an analysis was computed from.

    `MIXED` is not "the average of a judge and a human" — it is any analysis
    whose input contains rows from more than one rater type, which is gated
    because it contains judge rows.
    """

    JUDGE = "judge_only"
    HUMAN = "human_only"
    DETERMINISTIC = "deterministic_only"
    MIXED = "mixed"

    @classmethod
    def from_rater_types(cls, rater_types: Iterable[str | RaterType]) -> RaterScope:
        present = {str(r) for r in rater_types}
        if not present:
            return cls.MIXED
        if present == {str(RaterType.LLM_JUDGE)}:
            return cls.JUDGE
        if present == {str(RaterType.HUMAN)}:
            return cls.HUMAN
        if present == {str(RaterType.DETERMINISTIC)}:
            return cls.DETERMINISTIC
        return cls.MIXED

    @property
    def judge_gated(self) -> bool:
        """True if judge scores are in the analysis, so §9's threshold applies."""
        return self in (RaterScope.JUDGE, RaterScope.MIXED)


@dataclass(frozen=True, slots=True)
class Label:
    """The reporting status of one result, and why.

    `tag()` is what goes in the sentence that states the number. It is a method
    on the result's own label rather than a formatting choice at render time,
    so a table, a figure caption and a log line cannot disagree about whether a
    result was confirmatory.
    """

    status: EvidenceStatus
    prespecified: bool
    rater_scope: RaterScope
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_confirmatory(self) -> bool:
        return self.status is EvidenceStatus.CONFIRMATORY

    def tag(self) -> str:
        if self.is_confirmatory:
            return "CONFIRMATORY"
        why = "; ".join(self.reasons) if self.reasons else "not pre-specified"
        return f"EXPLORATORY ({why})"

    def demoted(self, reason: str) -> Label:
        """A copy demoted to exploratory with one more reason recorded."""
        return Label(
            status=EvidenceStatus.EXPLORATORY,
            prespecified=self.prespecified,
            rater_scope=self.rater_scope,
            reasons=(*self.reasons, reason),
        )


def status_from_agreement(alpha: float, rho: float, n_units: int) -> EvidenceStatus:
    """Delegate to the judge lane's pre-specified classifier. No local threshold."""
    return classify_dimension(alpha, rho, n_units)


def judge_gate_unavailable() -> dict[str, EvidenceStatus]:
    """Every dimension exploratory, which is the state of the world before human rating.

    Pre-registration §9 and `docs/limitations.md` §4: the judge-validation study
    needs human ratings that do not exist yet, so no dimension has cleared the
    threshold. Returning this rather than `None` means the absence of validation
    data produces a correctly-labelled result instead of an unlabelled one.
    """
    return dict.fromkeys(RUBRIC_DIMENSIONS, EvidenceStatus.EXPLORATORY)


def dimension_statuses(
    validity: Mapping[str, DimensionValidity] | None,
) -> dict[str, EvidenceStatus]:
    """Per-dimension confirmatory/exploratory status from the judge validation study.

    `None` — no validation report — means every dimension is exploratory, not
    that every dimension passes.
    """
    if validity is None:
        return judge_gate_unavailable()
    out = judge_gate_unavailable()
    for key, value in validity.items():
        out[key] = value.status
    return out


def measure_status(
    m: Measure,
    statuses: Mapping[str, EvidenceStatus],
) -> tuple[EvidenceStatus, tuple[str, ...]]:
    """Weakest-link status for a measure, plus the dimensions that demoted it."""
    failing = tuple(
        d
        for d in m.dimensions
        if statuses.get(d, EvidenceStatus.EXPLORATORY) is not EvidenceStatus.CONFIRMATORY
    )
    if failing:
        return EvidenceStatus.EXPLORATORY, failing
    return EvidenceStatus.CONFIRMATORY, ()


def label_for(
    m: Measure,
    *,
    prespecified: bool,
    rater_scope: RaterScope,
    statuses: Mapping[str, EvidenceStatus] | None = None,
    extra_reasons: Iterable[str] = (),
) -> Label:
    """Build the label for one result. Both demotion routes applied, in order.

    Args:
        m: the outcome measure the result is on.
        prespecified: whether the pre-registration names this analysis.
        rater_scope: whose scores produced it; decides whether §9's gate applies.
        statuses: per-dimension judge-validation statuses, from
            `dimension_statuses`. `None` means no validation study has run, which
            demotes every judge-gated result.
        extra_reasons: anything else that demotes this particular result — an
            unspecified threshold, a subgroup too small to support the test.
    """
    reasons: list[str] = []
    status = EvidenceStatus.CONFIRMATORY

    if not prespecified:
        status = EvidenceStatus.EXPLORATORY
        reasons.append("not pre-specified")

    if rater_scope.judge_gated:
        resolved = judge_gate_unavailable() if statuses is None else statuses
        gate_status, failing = measure_status(m, resolved)
        if gate_status is not EvidenceStatus.CONFIRMATORY:
            status = EvidenceStatus.EXPLORATORY
            if statuses is None:
                reasons.append("judge validation study has not run")
            else:
                reasons.append(
                    "judge agreement below the pre-specified threshold on " + ", ".join(failing)
                )

    for reason in extra_reasons:
        status = EvidenceStatus.EXPLORATORY
        reasons.append(reason)

    return Label(
        status=status,
        prespecified=prespecified,
        rater_scope=rater_scope,
        reasons=tuple(reasons),
    )
