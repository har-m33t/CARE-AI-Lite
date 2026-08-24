"""Effect sizes and bootstrap intervals against hand-computed answers.

The worked example used throughout is the paired difference vector
``d = (1, 2, 3, 4, 5, -6)``, whose statistics are all computable on paper:

* signed ranks are 1..6 with no ties, W+ = 15, W- = 6, so the matched-pairs
  rank-biserial is (15 - 6) / 21 = 3/7;
* mean(d) = 1.5 and the sum of squared deviations is 77.5, so with ddof = 1 the
  variance is 15.5 and dz = 1.5 / sqrt(15.5);
* the 21 Walsh averages sort to a median of 2.5.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from carelite.stats.effects import (
    bootstrap_ci,
    cohens_dz,
    hodges_lehmann,
    paired_effects,
    rank_biserial,
)

D = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -6.0])
ZERO = np.zeros(6)


# ---------------------------------------------------------------------------
# Point estimators
# ---------------------------------------------------------------------------


def test_rank_biserial_matches_the_hand_computed_value() -> None:
    assert rank_biserial(D, ZERO) == pytest.approx(9.0 / 21.0)


def test_rank_biserial_is_exactly_one_when_every_pair_moves_the_same_way() -> None:
    assert rank_biserial(np.array([2.0, 3.0, 9.0]), np.array([1.0, 1.0, 1.0])) == pytest.approx(1.0)
    assert rank_biserial(np.array([1.0, 1.0, 1.0]), np.array([2.0, 3.0, 9.0])) == pytest.approx(
        -1.0
    )


def test_rank_biserial_uses_midranks_for_tied_magnitudes() -> None:
    """d = (+1, -1, +3): the two magnitude-1 pairs share rank 1.5.

    W+ = 1.5 + 3 = 4.5, W- = 1.5, so the estimate is 3/6 = 0.5.
    """
    x = np.array([1.0, -1.0, 3.0])
    assert rank_biserial(x, ZERO[:3]) == pytest.approx(0.5)


def test_rank_biserial_is_undefined_not_zero_when_every_pair_is_tied() -> None:
    assert math.isnan(rank_biserial(np.array([2.0, 2.0]), np.array([2.0, 2.0])))


def test_rank_biserial_ignores_tied_pairs_entirely() -> None:
    """Adding tied pairs must not dilute the estimate; they carry no direction."""
    padded = np.concatenate([D, np.zeros(20)])
    assert rank_biserial(padded, np.zeros(26)) == pytest.approx(rank_biserial(D, ZERO))


def test_cohens_dz_matches_the_hand_computed_value() -> None:
    assert cohens_dz(D, ZERO) == pytest.approx(1.5 / math.sqrt(15.5))


def test_cohens_dz_is_undefined_for_a_constant_difference() -> None:
    assert math.isnan(cohens_dz(np.array([2.0, 2.0, 2.0]), np.zeros(3)))


def test_hodges_lehmann_matches_the_hand_computed_median_walsh_average() -> None:
    assert hodges_lehmann(D, ZERO) == pytest.approx(2.5)


def test_hodges_lehmann_is_the_shift_it_is_meant_to_estimate() -> None:
    """A constant shift is recovered exactly."""
    x = np.array([1.0, 4.0, 9.0, 2.0]) + 1.75
    assert hodges_lehmann(x, np.array([1.0, 4.0, 9.0, 2.0])) == pytest.approx(1.75)


def test_estimators_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        rank_biserial(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_of_a_constant_gives_a_degenerate_interval() -> None:
    """Every resample of a constant sample is the same constant."""
    units = np.full((25, 1), 4.0)
    ci = bootstrap_ci(units, lambda u: float(u.mean()), n_boot=200)
    assert ci.low == pytest.approx(4.0)
    assert ci.high == pytest.approx(4.0)
    assert ci.n_units == 25
    assert not ci.excludes_zero or ci.low > 0


def test_bootstrap_is_reproducible_from_its_seed() -> None:
    rng = np.random.default_rng(1)
    units = rng.normal(size=(40, 1))
    first = bootstrap_ci(units, lambda u: float(u.mean()), n_boot=500, seed=7)
    second = bootstrap_ci(units, lambda u: float(u.mean()), n_boot=500, seed=7)
    assert (first.low, first.high) == (second.low, second.high)


def test_a_different_seed_gives_a_different_interval() -> None:
    rng = np.random.default_rng(1)
    units = rng.normal(size=(40, 1))
    first = bootstrap_ci(units, lambda u: float(u.mean()), n_boot=500, seed=7)
    second = bootstrap_ci(units, lambda u: float(u.mean()), n_boot=500, seed=8)
    assert (first.low, first.high) != (second.low, second.high)


def test_bootstrap_resamples_whole_units_never_within_a_unit() -> None:
    """The statistic sees exactly `n_units` rows, drawn from the original rows.

    This is the property that keeps the three samples in a cell from being
    resampled independently: a scenario enters a replicate whole or not at all.
    """
    units = np.arange(30, dtype=float).reshape(30, 1)
    seen: list[np.ndarray] = []

    def _record(resampled: np.ndarray) -> float:
        seen.append(resampled.copy())
        return float(resampled.mean())

    bootstrap_ci(units, _record, n_boot=50, seed=3)
    assert len(seen) == 50
    for replicate in seen:
        assert replicate.shape == (30, 1)
        assert set(replicate.flatten()).issubset(set(units.flatten()))
    # With replacement, so at least one replicate must repeat a unit.
    assert any(len(set(r.flatten())) < 30 for r in seen)


def test_bootstrap_of_a_symmetric_sample_brackets_the_mean() -> None:
    rng = np.random.default_rng(20260822)
    units = rng.normal(loc=0.5, scale=1.0, size=(200, 1))
    ci = bootstrap_ci(units, lambda u: float(u.mean()), n_boot=2000, seed=5)
    assert ci.low < 0.5 < ci.high
    assert ci.excludes_zero


def test_bootstrap_counts_degenerate_replicates_rather_than_hiding_them() -> None:
    """A statistic that is undefined on some resamples reports a reduced count."""
    units = np.zeros((10, 1))
    ci = bootstrap_ci(units, lambda u: float("nan"), n_boot=100, seed=1)
    assert ci.n_valid_resamples == 0
    assert math.isnan(ci.low)
    assert not ci.excludes_zero


def test_bootstrap_of_an_empty_sample_is_nan_not_an_error() -> None:
    ci = bootstrap_ci(np.empty((0, 1)), lambda u: float(u.mean()), n_boot=10)
    assert math.isnan(ci.low) and math.isnan(ci.high)
    assert ci.n_units == 0


def test_a_wider_confidence_level_gives_a_wider_interval() -> None:
    rng = np.random.default_rng(4)
    units = rng.normal(size=(60, 1))
    narrow = bootstrap_ci(units, lambda u: float(u.mean()), confidence=0.80, n_boot=1000, seed=2)
    wide = bootstrap_ci(units, lambda u: float(u.mean()), confidence=0.99, n_boot=1000, seed=2)
    assert wide.low <= narrow.low and wide.high >= narrow.high


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------


def test_paired_effects_reports_all_three_estimators_with_intervals() -> None:
    effects = paired_effects(D, ZERO, n_boot=500, seed=1)
    assert effects.rank_biserial.point == pytest.approx(9.0 / 21.0)
    assert effects.cohens_dz.point == pytest.approx(1.5 / math.sqrt(15.5))
    assert effects.hodges_lehmann.point == pytest.approx(2.5)
    for estimate in (effects.rank_biserial, effects.cohens_dz, effects.hodges_lehmann):
        assert estimate.ci.n_units == 6
        assert estimate.ci.confidence == pytest.approx(0.95)


def test_the_headline_estimator_is_the_one_belonging_to_the_test() -> None:
    effects = paired_effects(D, ZERO, n_boot=200, seed=1)
    assert effects.headline is effects.rank_biserial


def test_paired_effects_records_how_many_pairs_carried_direction() -> None:
    x = np.array([1.0, 2.0, 3.0, 3.0])
    y = np.array([0.0, 0.0, 0.0, 3.0])
    effects = paired_effects(x, y, n_boot=100, seed=1)
    assert effects.n_pairs == 4
    assert effects.n_nonzero == 3


def test_direction_is_read_off_the_headline_estimate() -> None:
    up = paired_effects(np.array([2.0, 3.0, 4.0]), np.ones(3), n_boot=100, seed=1)
    down = paired_effects(np.ones(3), np.array([2.0, 3.0, 4.0]), n_boot=100, seed=1)
    flat = paired_effects(np.ones(3), np.ones(3), n_boot=100, seed=1)
    assert (up.direction, down.direction, flat.direction) == (">", "<", "=")


def test_render_shows_the_point_and_interval_together() -> None:
    effects = paired_effects(D, ZERO, n_boot=200, seed=1)
    text = effects.rank_biserial.render()
    assert "95% bootstrap CI" in text
    assert "6 scenarios" in text
