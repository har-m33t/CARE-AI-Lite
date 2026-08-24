"""Variance decomposition: a hand-computed moments example, then model recovery.

The moments estimator is checked against arithmetic done on paper. The
mixed-effects fit is then checked against synthetic data whose components were
chosen before the fit ran — a model that recovers a planted effect of exactly
0.5 from data it has never seen is verified; one that merely produces a number
is not.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from carelite.stats.evidence import EvidenceStatus
from carelite.stats.measures import NURSE_COMPOSITE, measure
from carelite.stats.mixed import (
    fit_random_intercept,
    independence_check,
    variance_components_moments,
    within_cell_variance,
)
from carelite.types import RUBRIC_DIMENSIONS, Condition
from tests.unit.stats.conftest import constant_scores, make_long

# ---------------------------------------------------------------------------
# Method of moments, computed by hand
# ---------------------------------------------------------------------------


def test_variance_components_match_the_hand_computed_example() -> None:
    """Three groups of two: (1, 3), (5, 7), (9, 11).

    Group means are 2, 6, 10 and the grand mean is 6, so
      SS_between = 2 * ((2-6)^2 + 0 + (10-6)^2) = 64, MS_between = 32;
      SS_within  = 2 + 2 + 2 = 6,                     MS_within  = 6 / 3 = 2;
      n0 = (6 - 12/6) / 2 = 2;
      between = (32 - 2) / 2 = 15, within = 2, ICC = 15 / 17.
    """
    components = variance_components_moments(
        [1.0, 3.0, 5.0, 7.0, 9.0, 11.0], ["g1", "g1", "g2", "g2", "g3", "g3"]
    )
    assert components.n_groups == 3
    assert components.n_observations == 6
    assert components.within == pytest.approx(2.0)
    assert components.between == pytest.approx(15.0)
    assert components.icc == pytest.approx(15.0 / 17.0)
    assert components.balanced is True


def test_identical_groups_give_a_negative_between_variance_not_a_clamped_zero() -> None:
    """Groups more alike than chance predicts: the raw estimate is reported."""
    components = variance_components_moments(
        [1.0, 3.0, 1.0, 3.0, 1.0, 3.0], ["a", "a", "b", "b", "c", "c"]
    )
    assert components.between < 0
    assert components.icc == 0.0  # the ICC has no negative reading, so it is clamped


def test_perfectly_separated_groups_give_an_icc_of_one() -> None:
    components = variance_components_moments(
        [1.0, 1.0, 5.0, 5.0, 9.0, 9.0], ["a", "a", "b", "b", "c", "c"]
    )
    assert components.within == pytest.approx(0.0)
    assert components.icc == pytest.approx(1.0)


def test_the_unbalanced_effective_group_size_reduces_to_n_when_balanced() -> None:
    """Same numbers, one group given an extra observation: still computable."""
    components = variance_components_moments(
        [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 10.0], ["g1", "g1", "g2", "g2", "g3", "g3", "g3"]
    )
    assert components.balanced is False
    assert not math.isnan(components.between)


def test_a_single_group_is_undefined() -> None:
    components = variance_components_moments([1.0, 2.0, 3.0], ["g", "g", "g"])
    assert math.isnan(components.between)


# ---------------------------------------------------------------------------
# Within-cell variance
# ---------------------------------------------------------------------------


def test_within_cell_variance_is_zero_when_the_samples_agree(
    separated_ab: pd.DataFrame,
) -> None:
    assert within_cell_variance(separated_ab, NURSE_COMPOSITE) == pytest.approx(0.0)


def test_within_cell_variance_measures_the_samples_and_nothing_else() -> None:
    """One cell, three samples with composite values 1, 3, 5: variance 4."""
    scores = {
        ("SC-000", "A", 0): constant_scores(NURSE_COMPOSITE.dimensions, 1),
        ("SC-000", "A", 1): constant_scores(NURSE_COMPOSITE.dimensions, 3),
        ("SC-000", "A", 2): constant_scores(NURSE_COMPOSITE.dimensions, 5),
    }
    assert within_cell_variance(make_long(scores=scores), NURSE_COMPOSITE) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# The fitted model
# ---------------------------------------------------------------------------


def test_the_model_recovers_a_planted_condition_effect(
    noisy_mixed_frame: pd.DataFrame,
) -> None:
    """B was built to score exactly 0.5 above A. The fit must find it."""
    result = fit_random_intercept(noisy_mixed_frame, NURSE_COMPOSITE)
    assert result is not None
    assert result.converged
    effect = next(e for e in result.effects if e.term == "B vs A")
    assert effect.coefficient == pytest.approx(0.5, abs=0.12)
    assert effect.ci_low < 0.5 < effect.ci_high


def test_the_model_recovers_the_planted_variance_components(
    noisy_mixed_frame: pd.DataFrame,
) -> None:
    """Scenario SD 0.9 and residual SD 0.35 were planted before the fit ran."""
    result = fit_random_intercept(noisy_mixed_frame, NURSE_COMPOSITE)
    assert result is not None
    assert result.scenario_variance == pytest.approx(0.81, abs=0.35)
    assert result.residual_variance == pytest.approx(0.1225, abs=0.06)
    # Most of the variance is between scenarios, which is why the intercept exists.
    assert result.icc > 0.75


def test_the_moments_cross_check_agrees_with_the_fitted_scenario_variance(
    noisy_mixed_frame: pd.DataFrame,
) -> None:
    result = fit_random_intercept(noisy_mixed_frame, NURSE_COMPOSITE)
    assert result is not None
    assert result.moments.icc == pytest.approx(result.icc, abs=0.15)


def test_the_model_reports_the_units_it_actually_fitted(
    noisy_mixed_frame: pd.DataFrame,
) -> None:
    """40 scenarios x 2 conditions x 3 samples = 240 generations, not 240 scenarios."""
    result = fit_random_intercept(noisy_mixed_frame, NURSE_COMPOSITE)
    assert result is not None
    assert result.n_observations == 240
    assert result.n_scenarios == 40
    assert result.samples_per_cell == pytest.approx(3.0)


def test_the_fit_uses_the_quality_scale_for_ritualistic() -> None:
    """Condition D is maximally ritualistic (raw 5); it must fit *below* B."""
    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(12):
        scenario = f"SC-{i:03d}"
        for sample in range(3):
            scores[(scenario, "B", sample)] = {"ritualistic": 1}
            scores[(scenario, "D", sample)] = {"ritualistic": 5}
    result = fit_random_intercept(
        make_long(scores=scores), measure("ritualistic"), reference=Condition.B
    )
    assert result is not None
    effect = next(e for e in result.effects if e.term == "D vs B")
    # Raw would put D four points above B; on the quality scale it is four below.
    assert effect.coefficient == pytest.approx(-4.0, abs=1e-6)


def test_the_fit_returns_none_rather_than_a_model_of_nothing() -> None:
    empty = make_long(scores={})
    assert fit_random_intercept(empty, NURSE_COMPOSITE) is None


def test_one_condition_is_not_a_comparison() -> None:
    scores = {
        (f"SC-{i:03d}", "A", 0): constant_scores(NURSE_COMPOSITE.dimensions, 3) for i in range(5)
    }
    assert fit_random_intercept(make_long(scores=scores), NURSE_COMPOSITE) is None


def test_the_model_label_carries_the_judge_gate(
    noisy_mixed_frame: pd.DataFrame,
) -> None:
    result = fit_random_intercept(noisy_mixed_frame, NURSE_COMPOSITE)
    assert result is not None
    assert not result.label.is_confirmatory
    statuses = dict.fromkeys(RUBRIC_DIMENSIONS, EvidenceStatus.CONFIRMATORY)
    validated = fit_random_intercept(noisy_mixed_frame, NURSE_COMPOSITE, statuses=statuses)
    assert validated is not None
    assert validated.label.is_confirmatory


# ---------------------------------------------------------------------------
# What treating the samples as independent would cost
# ---------------------------------------------------------------------------


def test_the_independence_check_counts_both_unit_definitions(
    noisy_mixed_frame: pd.DataFrame,
) -> None:
    check = independence_check(noisy_mixed_frame, NURSE_COMPOSITE, "B", "A")
    assert check is not None
    assert check.n_scenarios == 40
    assert check.n_sample_pairs == 120  # three times as many, which is the error


def test_a_scenario_by_condition_interaction_makes_the_naive_se_understate() -> None:
    """Plant an interaction; the sample-level SE must come out too small.

    With interaction variance I and within-cell variance sigma^2, the
    sample-level paired SE is smaller than the cell-level one by
    sqrt((2 sigma^2 + I) / (2 sigma^2 + 3I)). Here I is large relative to
    sigma^2, so the understatement is unmistakable.
    """
    rng = np.random.default_rng(20260822)
    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(30):
        scenario = f"SC-{i:03d}"
        offsets = {"A": rng.normal(0, 1.0), "B": rng.normal(0, 1.0)}
        for condition in ("A", "B"):
            for sample in range(3):
                value = 3.0 + offsets[condition] + rng.normal(0, 0.15)
                grid = int(np.clip(round(value), 1, 5))
                scores[(scenario, condition, sample)] = {"name": grid}
    check = independence_check(make_long(scores=scores), measure("name"), "B", "A")
    assert check is not None
    assert check.understatement > 1.2


def test_no_interaction_leaves_the_two_standard_errors_close() -> None:
    """When cells differ only by sampling noise the two definitions agree.

    Which is exactly why the module reports the ratio as a measurement rather
    than asserting a fixed sqrt(3) penalty.
    """
    rng = np.random.default_rng(5)
    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(30):
        scenario = f"SC-{i:03d}"
        base = rng.integers(2, 5)
        for condition in ("A", "B"):
            for sample in range(3):
                scores[(scenario, condition, sample)] = {
                    "name": int(np.clip(base + rng.integers(-1, 2), 1, 5))
                }
    check = independence_check(make_long(scores=scores), measure("name"), "B", "A")
    assert check is not None
    assert check.understatement == pytest.approx(1.0, abs=0.35)


def test_the_render_shows_the_independence_cost(
    noisy_mixed_frame: pd.DataFrame,
) -> None:
    result = fit_random_intercept(noisy_mixed_frame, NURSE_COMPOSITE)
    assert result is not None
    text = result.render()
    assert "WHAT TREATING THE 3 SAMPLES AS INDEPENDENT WOULD COST" in text
    assert "n sample pairs" in text
    assert "ICC (scenario)" in text
