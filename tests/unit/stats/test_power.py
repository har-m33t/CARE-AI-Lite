"""The power analysis must reproduce the table that gets registered.

Pre-registration §6 and build plan v3 §11 tabulate the sample sizes that justify
n = 60. If this module cannot recompute them, then either the table or the code
is wrong, and the number that went on the OSF form is not the number the study
can defend. So the pre-registered brackets are asserted directly.
"""

from __future__ import annotations

import math

import pytest

from carelite.config import get_settings
from carelite.stats.power import (
    ALPHA,
    ARE_WILCOXON_VS_T,
    POWER,
    build_power_report,
    detectable_effect,
    required_n,
)


def test_the_registered_table_is_reproduced() -> None:
    """v3 §11: large ~15-20, medium ~35-45, small ~90+."""
    assert 15 <= required_n(0.8) <= 20
    assert 35 <= required_n(0.5) <= 45
    assert required_n(0.3) >= 90


def test_the_registered_table_is_reproduced_exactly() -> None:
    """Pinned so a change to the formula is visible rather than absorbed."""
    assert required_n(0.8) == 15
    assert required_n(0.5) == 35
    assert required_n(0.3) == 94


def test_required_n_matches_the_closed_form() -> None:
    """n = (((z_a + z_b) / d)^2 + z_a^2 / 2) / ARE, rounded up."""
    from scipy import stats

    z_alpha = float(stats.norm.ppf(1 - ALPHA / 2))
    z_beta = float(stats.norm.ppf(POWER))
    for d in (0.2, 0.35, 0.5, 0.8, 1.2):
        expected = math.ceil((((z_alpha + z_beta) / d) ** 2 + z_alpha**2 / 2) / ARE_WILCOXON_VS_T)
        assert required_n(d) == expected


def test_the_are_is_three_over_pi() -> None:
    """The normal-differences ARE of the signed-rank test against the t test."""
    assert pytest.approx(3.0 / math.pi) == ARE_WILCOXON_VS_T
    assert pytest.approx(0.9549, abs=1e-4) == ARE_WILCOXON_VS_T


def test_required_n_decreases_as_the_effect_grows() -> None:
    sizes = [required_n(d) for d in (0.2, 0.4, 0.6, 0.8, 1.0)]
    assert sizes == sorted(sizes, reverse=True)


def test_more_power_costs_more_scenarios() -> None:
    assert required_n(0.5, power=0.90) > required_n(0.5, power=0.80)


def test_a_one_sided_test_needs_fewer_scenarios() -> None:
    assert required_n(0.5, two_sided=False) < required_n(0.5, two_sided=True)


def test_required_n_rejects_a_non_positive_effect() -> None:
    with pytest.raises(ValueError, match="standardised paired effect"):
        required_n(0.0)


def test_detectable_effect_inverts_required_n() -> None:
    """At n = required_n(d), the detectable effect must be no larger than d."""
    for d in (0.3, 0.5, 0.8):
        assert detectable_effect(required_n(d)) <= d + 1e-9


def test_detectable_effect_at_the_frozen_n() -> None:
    """n = 60, alpha = 0.05 two-sided, power = 0.80 resolves dz ~ 0.376."""
    assert detectable_effect(60) == pytest.approx(0.376, abs=0.005)


def test_detectable_effect_shrinks_as_n_grows() -> None:
    values = [detectable_effect(n) for n in (20, 40, 60, 120)]
    assert values == sorted(values, reverse=True)


def test_detectable_effect_is_nan_when_n_cannot_support_the_design() -> None:
    assert math.isnan(detectable_effect(1))


def test_the_equity_subgroup_n_resolves_only_a_large_effect() -> None:
    """20 held-out equity scenarios (10 ses, 4 lep, 6 racial_ethnic)."""
    assert detectable_effect(20) == pytest.approx(0.676, abs=0.01)
    assert detectable_effect(20) > detectable_effect(60)


def test_the_report_reads_n_from_the_frozen_config_not_a_local_constant() -> None:
    report = build_power_report()
    experiment = get_settings().experiment
    assert report.n_holdout == experiment.n_scenarios_holdout == 60
    assert report.samples_per_cell == experiment.samples_per_cell == 3


def test_the_report_names_the_comparison_that_set_n() -> None:
    """v3 §11: power for the comparison you care about least is what sets n."""
    report = build_power_report()
    assert "B vs Condition C" in report.sizing_comparison
    text = report.render()
    assert "SMALLEST effect" in text
    assert "not the primary outcome" in text


def test_the_family_corrected_figure_is_reported_as_context_not_as_the_plan() -> None:
    report = build_power_report(family_size=8)
    assert report.detectable_at_n_family_corrected is not None
    assert report.detectable_at_n_family_corrected > report.detectable_at_n
    text = report.render()
    assert "The registered n stands on the nominal figure" in text


def test_omitting_the_family_size_omits_the_corrected_figure() -> None:
    report = build_power_report()
    assert report.detectable_at_n_family_corrected is None
    assert "Charging the whole Holm family" not in report.render()
