"""The one pre-specified subgroup, and the label every other one carries."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from carelite.eval.judge.validation import EvidenceStatus
from carelite.stats.power import detectable_effect
from carelite.stats.subgroups import (
    EQUITY_COVERAGE_GAPS,
    RACIAL_ETHNIC_DESCRIPTION,
    equity_subgroup,
    exploratory_subgroup,
)
from carelite.types import RUBRIC_DIMENSIONS
from tests.unit.stats.conftest import constant_scores, make_long


@pytest.fixture
def stratified_long(nurse_dimensions: tuple[str, ...]) -> pd.DataFrame:
    """20 scenarios, the first 8 in the equity stratum, B above A throughout."""
    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(20):
        scenario = f"SC-{i:03d}"
        for sample in range(3):
            scores[(scenario, "A", sample)] = constant_scores(nurse_dimensions, 2)
            scores[(scenario, "B", sample)] = constant_scores(nurse_dimensions, 4)
    long = make_long(scores=scores, equity_scenarios=[f"SC-{i:03d}" for i in range(8)])
    long["equity_kind"] = long["scenario_id"].map(
        lambda s: "ses" if s < "SC-004" else ("lep" if s < "SC-008" else None)
    )
    return long


# ---------------------------------------------------------------------------
# The pre-specified subgroup
# ---------------------------------------------------------------------------


def test_the_equity_subgroup_uses_only_stratum_scenarios(
    stratified_long: pd.DataFrame,
) -> None:
    result = equity_subgroup(stratified_long, n_boot=200)
    assert result.n_scenarios == 8
    primary = result.family.by_key("primary_nurse_A_vs_B")
    assert primary is not None
    assert primary.n_scenarios == 8


def test_the_equity_subgroup_counts_the_axes_separately(
    stratified_long: pd.DataFrame,
) -> None:
    result = equity_subgroup(stratified_long, n_boot=200)
    assert result.n_by_equity_kind == {"ses": 4, "lep": 4}


def test_the_equity_subgroup_is_pre_specified_and_can_be_confirmatory(
    stratified_long: pd.DataFrame,
) -> None:
    statuses = dict.fromkeys(RUBRIC_DIMENSIONS, EvidenceStatus.CONFIRMATORY)
    result = equity_subgroup(stratified_long, statuses=statuses, n_boot=200)
    assert result.prespecified
    primary = result.family.by_key("primary_nurse_A_vs_B")
    assert primary is not None
    assert primary.label.is_confirmatory


def test_the_equity_subgroup_reports_its_own_detectable_effect(
    stratified_long: pd.DataFrame,
) -> None:
    """The n it actually had, not the 60 the main comparison had."""
    result = equity_subgroup(stratified_long, n_boot=200)
    assert result.detectable_effect_dz == pytest.approx(detectable_effect(8))
    assert result.detectable_effect_dz > detectable_effect(60)


def test_the_equity_result_carries_the_d5_description_of_racial_ethnic(
    stratified_long: pd.DataFrame,
) -> None:
    result = equity_subgroup(stratified_long, n_boot=200)
    assert result.racial_ethnic_description is RACIAL_ETHNIC_DESCRIPTION
    text = result.render()
    assert "anticipated dismissal" in text
    assert "credibility-management" in text
    assert "not race-based disparity in communication generally" in text


def test_the_equity_result_carries_both_pre_specified_coverage_gaps(
    stratified_long: pd.DataFrame,
) -> None:
    result = equity_subgroup(stratified_long, n_boot=200)
    assert result.coverage_gaps == EQUITY_COVERAGE_GAPS
    text = result.render()
    assert "emotion_intensity = 1" in text
    assert "adherence_barrier" in text
    assert "PRE-SPECIFIED COVERAGE GAPS" in text


def test_the_equity_render_corrects_the_thirty_five_scenario_figure(
    stratified_long: pd.DataFrame,
) -> None:
    """§8.4 says 35, which counts the train split; the analysis is holdout-only."""
    result = equity_subgroup(stratified_long, n_boot=200)
    assert any("35-scenario figure" in note for note in result.family.notes)


def test_the_equity_subgroup_needs_the_stratum_column() -> None:
    long = make_long(scores={("SC-000", "A", 0): {"name": 3}}).drop(columns=["equity_stratum"])
    with pytest.raises(KeyError, match="equity_stratum"):
        equity_subgroup(long, n_boot=50)


def test_an_empty_stratum_produces_an_empty_result_not_an_error(
    nurse_dimensions: tuple[str, ...],
) -> None:
    scores = {("SC-000", "A", 0): constant_scores(nurse_dimensions, 3)}
    result = equity_subgroup(make_long(scores=scores), n_boot=50)
    assert result.n_scenarios == 0
    assert math.isnan(result.detectable_effect_dz)


# ---------------------------------------------------------------------------
# Everything else
# ---------------------------------------------------------------------------


def test_an_exploratory_subgroup_is_labelled_in_the_output_structure(
    stratified_long: pd.DataFrame,
) -> None:
    result = exploratory_subgroup(stratified_long, "challenge_type", "emotional_cue", n_boot=200)
    assert result.prespecified is False
    assert result.status is EvidenceStatus.EXPLORATORY
    for pairwise in result.family.results:
        assert not pairwise.label.is_confirmatory
        assert not pairwise.hypothesis.prespecified


def test_an_exploratory_subgroup_stays_exploratory_even_with_a_perfect_judge(
    stratified_long: pd.DataFrame,
) -> None:
    """The judge clearing §9 cannot promote an analysis nobody registered."""
    statuses = dict.fromkeys(RUBRIC_DIMENSIONS, EvidenceStatus.CONFIRMATORY)
    result = exploratory_subgroup(
        stratified_long, "challenge_type", "emotional_cue", statuses=statuses, n_boot=200
    )
    for pairwise in result.family.results:
        assert not pairwise.label.is_confirmatory
        assert "not pre-specified" in pairwise.label.tag()


def test_the_exploratory_render_says_so_before_the_numbers(
    stratified_long: pd.DataFrame,
) -> None:
    result = exploratory_subgroup(stratified_long, "challenge_type", "emotional_cue", n_boot=200)
    text = result.render()
    assert text.startswith("EXPLORATORY SUBGROUP")
    assert text.index("NOT pre-specified") < text.index("effect (rank-biserial)")


def test_an_exploratory_subgroup_accepts_an_arbitrary_predicate(
    stratified_long: pd.DataFrame,
) -> None:
    result = exploratory_subgroup(
        stratified_long,
        "emotion_intensity",
        "high",
        predicate=lambda df: df["emotion_intensity"] >= 3,
        n_boot=200,
    )
    assert result.n_scenarios == 20
    assert result.status is EvidenceStatus.EXPLORATORY


def test_an_unknown_column_is_an_error_not_an_empty_subgroup(
    stratified_long: pd.DataFrame,
) -> None:
    with pytest.raises(KeyError, match="no column"):
        exploratory_subgroup(stratified_long, "clinician_mood", "sunny", n_boot=50)
