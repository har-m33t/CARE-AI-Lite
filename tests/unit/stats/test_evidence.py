"""The reporting label, which is what stops a demoted result being read as a finding.

Two independent demotion routes compose here: not pre-specified (§1), and the
judge failing §9's agreement threshold on a constituent dimension. The threshold
itself belongs to `carelite.eval.judge.validation.classify_dimension` and is
consumed, not restated — one test asserts that delegation directly, because a
second copy of 0.667 in this package is exactly how the two drift apart.
"""

from __future__ import annotations

import math

import pytest

from carelite.eval.judge.validation import (
    MIN_ALPHA_FOR_CONFIRMATORY,
    MIN_RHO_FOR_CONFIRMATORY,
    MIN_UNITS_FOR_CONFIRMATORY,
    AgreementResult,
    DimensionValidity,
    EvidenceStatus,
    classify_dimension,
)
from carelite.stats.evidence import (
    Label,
    RaterScope,
    dimension_statuses,
    judge_gate_unavailable,
    label_for,
    measure_status,
    status_from_agreement,
)
from carelite.stats.measures import NURSE_COMPOSITE, measure
from carelite.types import RUBRIC_DIMENSIONS, RaterType


def _validity(dimension: str, alpha: float, rho: float, n: int) -> DimensionValidity:
    return DimensionValidity(
        agreement=AgreementResult(
            dimension=dimension, n_units=n, n_observers=2, alpha=alpha, rho=rho, rho_p=0.01
        ),
        status=classify_dimension(alpha, rho, n),
    )


# ---------------------------------------------------------------------------
# The threshold is the judge lane's, not a copy
# ---------------------------------------------------------------------------


def test_status_delegates_to_the_judge_lanes_classifier() -> None:
    cases = [
        (0.9, 0.9, 60),
        (0.5, 0.9, 60),
        (0.9, 0.2, 60),
        (0.9, 0.9, 10),
        (math.nan, 0.9, 60),
    ]
    for alpha, rho, n in cases:
        assert status_from_agreement(alpha, rho, n) is classify_dimension(alpha, rho, n)


def test_the_thresholds_are_the_registered_ones() -> None:
    """§9's numbers, asserted where the analysis consumes them."""
    assert MIN_ALPHA_FOR_CONFIRMATORY == 0.667
    assert MIN_RHO_FOR_CONFIRMATORY == 0.5
    assert MIN_UNITS_FOR_CONFIRMATORY == 30


def test_an_undefined_coefficient_fails_rather_than_being_treated_as_missing() -> None:
    assert status_from_agreement(math.nan, 0.9, 60) is EvidenceStatus.EXPLORATORY


# ---------------------------------------------------------------------------
# Before the validation study exists
# ---------------------------------------------------------------------------


def test_no_validation_study_means_every_dimension_is_exploratory() -> None:
    """docs/limitations.md §4: the study cannot run until human ratings exist."""
    statuses = dimension_statuses(None)
    assert set(statuses) == set(RUBRIC_DIMENSIONS)
    assert all(s is EvidenceStatus.EXPLORATORY for s in statuses.values())
    assert statuses == judge_gate_unavailable()


def test_a_partial_validation_report_leaves_the_rest_exploratory() -> None:
    statuses = dimension_statuses({"name": _validity("name", 0.8, 0.7, 40)})
    assert statuses["name"] is EvidenceStatus.CONFIRMATORY
    assert statuses["explore"] is EvidenceStatus.EXPLORATORY


# ---------------------------------------------------------------------------
# The composite rule
# ---------------------------------------------------------------------------


def test_a_composite_is_confirmatory_only_if_every_constituent_is() -> None:
    all_good = dict.fromkeys(RUBRIC_DIMENSIONS, EvidenceStatus.CONFIRMATORY)
    status, failing = measure_status(NURSE_COMPOSITE, all_good)
    assert status is EvidenceStatus.CONFIRMATORY
    assert failing == ()


def test_one_weak_dimension_demotes_the_whole_composite() -> None:
    statuses = dict.fromkeys(RUBRIC_DIMENSIONS, EvidenceStatus.CONFIRMATORY)
    statuses["support"] = EvidenceStatus.EXPLORATORY
    status, failing = measure_status(NURSE_COMPOSITE, statuses)
    assert status is EvidenceStatus.EXPLORATORY
    assert failing == ("support",)


def test_a_dimension_outside_the_composite_does_not_demote_it() -> None:
    statuses = dict.fromkeys(RUBRIC_DIMENSIONS, EvidenceStatus.CONFIRMATORY)
    statuses["naturalness"] = EvidenceStatus.EXPLORATORY
    status, _ = measure_status(NURSE_COMPOSITE, statuses)
    assert status is EvidenceStatus.CONFIRMATORY


def test_an_unknown_dimension_status_is_treated_as_failing() -> None:
    status, failing = measure_status(NURSE_COMPOSITE, {})
    assert status is EvidenceStatus.EXPLORATORY
    assert set(failing) == set(NURSE_COMPOSITE.dimensions)


# ---------------------------------------------------------------------------
# Rater scope
# ---------------------------------------------------------------------------


def test_scope_is_derived_from_the_rater_types_present() -> None:
    assert RaterScope.from_rater_types(["llm_judge"]) is RaterScope.JUDGE
    assert RaterScope.from_rater_types(["human"]) is RaterScope.HUMAN
    assert RaterScope.from_rater_types(["human", "llm_judge"]) is RaterScope.MIXED
    assert RaterScope.from_rater_types([RaterType.DETERMINISTIC]) is RaterScope.DETERMINISTIC


def test_the_judge_gate_applies_to_judge_and_mixed_analyses_only() -> None:
    assert RaterScope.JUDGE.judge_gated
    assert RaterScope.MIXED.judge_gated
    assert not RaterScope.HUMAN.judge_gated
    assert not RaterScope.DETERMINISTIC.judge_gated


def test_a_human_only_analysis_is_not_gated_by_judge_agreement() -> None:
    """Gating human ratings on the judge's agreement with humans would be circular."""
    label = label_for(
        NURSE_COMPOSITE, prespecified=True, rater_scope=RaterScope.HUMAN, statuses=None
    )
    assert label.is_confirmatory


def test_a_judge_only_analysis_is_gated() -> None:
    label = label_for(
        NURSE_COMPOSITE, prespecified=True, rater_scope=RaterScope.JUDGE, statuses=None
    )
    assert not label.is_confirmatory
    assert "judge validation study has not run" in label.tag()


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_not_pre_specified_demotes_regardless_of_the_judge() -> None:
    statuses = dict.fromkeys(RUBRIC_DIMENSIONS, EvidenceStatus.CONFIRMATORY)
    label = label_for(
        NURSE_COMPOSITE,
        prespecified=False,
        rater_scope=RaterScope.JUDGE,
        statuses=statuses,
    )
    assert not label.is_confirmatory
    assert "not planned in advance" in label.tag()


def test_both_reasons_are_recorded_when_both_apply() -> None:
    statuses = dict.fromkeys(RUBRIC_DIMENSIONS, EvidenceStatus.EXPLORATORY)
    label = label_for(
        NURSE_COMPOSITE,
        prespecified=False,
        rater_scope=RaterScope.JUDGE,
        statuses=statuses,
    )
    assert len(label.reasons) == 2
    assert "not planned in advance" in label.tag()
    assert "below the fixed threshold" in label.tag()


def test_an_extra_reason_demotes_a_result_that_would_otherwise_pass() -> None:
    statuses = dict.fromkeys(RUBRIC_DIMENSIONS, EvidenceStatus.CONFIRMATORY)
    label = label_for(
        measure("naturalness"),
        prespecified=True,
        rater_scope=RaterScope.JUDGE,
        statuses=statuses,
        extra_reasons=["threshold for this analysis is not pre-specified"],
    )
    assert not label.is_confirmatory
    assert "threshold for this analysis is not pre-specified" in label.tag()


def test_the_tag_is_the_sentence_the_write_up_uses() -> None:
    confirmatory = Label(EvidenceStatus.CONFIRMATORY, True, RaterScope.HUMAN)
    # D10: the word "confirmatory" must never reach a rendered string.
    assert confirmatory.tag() == "DESCRIPTIVE (planned in advance; judge gate cleared)"
    assert "confirmatory" not in confirmatory.tag().lower()
    exploratory = Label(EvidenceStatus.EXPLORATORY, False, RaterScope.JUDGE, ("not pre-specified",))
    assert exploratory.tag() == "EXPLORATORY (not pre-specified)"


def test_demoting_a_label_keeps_its_history() -> None:
    label = Label(EvidenceStatus.CONFIRMATORY, True, RaterScope.HUMAN)
    demoted = label.demoted("subgroup too small")
    assert demoted.status is EvidenceStatus.EXPLORATORY
    assert demoted.reasons == ("subgroup too small",)
    assert label.is_confirmatory, "the original label must not be mutated"


def test_a_label_cannot_be_confirmatory_without_a_reason_free_history() -> None:
    """Every exploratory label carries at least one stated reason."""
    for prespecified in (True, False):
        for scope in RaterScope:
            label = label_for(
                NURSE_COMPOSITE,
                prespecified=prespecified,
                rater_scope=scope,
                statuses=None,
            )
            if not label.is_confirmatory:
                assert label.reasons, f"{scope} {prespecified} demoted without a reason"


@pytest.mark.parametrize("dimension", list(RUBRIC_DIMENSIONS))
def test_every_dimension_can_be_gated_individually(dimension: str) -> None:
    statuses = dict.fromkeys(RUBRIC_DIMENSIONS, EvidenceStatus.CONFIRMATORY)
    statuses[dimension] = EvidenceStatus.EXPLORATORY
    label = label_for(
        measure(dimension),
        prespecified=True,
        rater_scope=RaterScope.JUDGE,
        statuses=statuses,
    )
    assert not label.is_confirmatory
    assert dimension in label.tag()
