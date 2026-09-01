"""The backend-equivalence comparison: what 39 paired cells can and cannot say.

Design §4 W5 asks for agreement between the two serving stacks on the LC cells
they both produced, and says: report agreement, do not pool arms that disagree.
These tests pin the arithmetic and, as importantly, pin the refusal to dress the
result up — the sample is 13 scenarios that were never randomised for partial
analysis, and the report has to say so in its own structure rather than in a
sentence someone can drop.

No database and no model. `compare_backends` is a pure function of two score
mappings.
"""

from __future__ import annotations

import math

import pytest

from carelite.eval.judge.backend_equivalence import (
    BACKEND_CONFOUNDS,
    MIN_UNITS_FOR_EQUIVALENCE_CLAIM,
    BackendEquivalence,
    compare_backends,
)
from carelite.types import RUBRIC_DIMENSIONS

DIMS = list(RUBRIC_DIMENSIONS)


def scores(value: int | dict[str, int | None]) -> dict[str, int | None]:
    if isinstance(value, int):
        return dict.fromkeys(DIMS, value)
    return {d: value.get(d, 3) for d in DIMS}


def build(
    n_scenarios: int = 13,
    n_samples: int = 3,
    *,
    left_value: int = 3,
    right_value: int = 3,
) -> tuple[dict[str, dict[str, int | None]], dict[str, dict[str, int | None]], dict[str, str]]:
    left: dict[str, dict[str, int | None]] = {}
    right: dict[str, dict[str, int | None]] = {}
    scenario_of: dict[str, str] = {}
    for s in range(n_scenarios):
        scenario = f"SC-{s:03d}"
        for k in range(n_samples):
            unit = f"{scenario}/{k}"
            left[unit] = scores(left_value)
            right[unit] = scores(right_value)
            scenario_of[unit] = scenario
    return left, right, scenario_of


def run(left, right, scenario_of, **kwargs) -> BackendEquivalence:  # type: ignore[no-untyped-def]
    kwargs.setdefault("n_boot", 200)
    return compare_backends(
        left,
        right,
        scenario_of=scenario_of,
        left_backend="ollama",
        right_backend="vllm",
        **kwargs,
    )


class TestShape:
    def test_every_rubric_dimension_is_reported(self) -> None:
        report = run(*build())
        assert [d.dimension for d in report.dimensions] == DIMS

    def test_the_unit_counts_are_carried_on_the_result(self) -> None:
        report = run(*build(n_scenarios=13, n_samples=3))
        assert report.n_cell_pairs == 39
        assert report.n_scenarios == 13

    def test_the_backends_are_named_on_the_result(self) -> None:
        report = run(*build())
        assert report.left_backend == "ollama"
        assert report.right_backend == "vllm"

    def test_pairing_a_stack_against_itself_is_refused(self) -> None:
        left, right, scenario_of = build()
        with pytest.raises(ValueError, match="same serving stack"):
            compare_backends(
                left,
                right,
                scenario_of=scenario_of,
                left_backend="vllm",
                right_backend="vllm",
            )

    def test_a_unit_with_no_scenario_is_refused(self) -> None:
        left, right, scenario_of = build(n_scenarios=2, n_samples=1)
        scenario_of.pop(next(iter(scenario_of)))
        with pytest.raises(KeyError, match="scenario"):
            run(left, right, scenario_of)


class TestAgreement:
    def test_identical_scores_agree_exactly_on_every_cell(self) -> None:
        report = run(*build(left_value=4, right_value=4))
        for dim in report.dimensions:
            assert dim.n_pairs == 39
            assert dim.exact_agreement == 1.0
            assert dim.within_one_agreement == 1.0
            assert dim.mean_difference == 0.0

    def test_a_constant_disagreement_is_not_reported_as_perfect_reliability(self) -> None:
        """Alpha is undefined when nothing varies. `nan`, never 1.0."""
        report = run(*build(left_value=3, right_value=3))
        for dim in report.dimensions:
            assert math.isnan(dim.alpha)
            assert dim.exact_agreement == 1.0

    def test_a_one_point_offset_shows_up_as_a_signed_difference(self) -> None:
        report = run(*build(left_value=3, right_value=4))
        name = next(d for d in report.dimensions if d.dimension == "name")
        assert name.exact_agreement == 0.0
        assert name.within_one_agreement == 1.0
        # right - left, on the quality scale, so the sign says which stack scored higher.
        assert name.mean_difference == pytest.approx(1.0)
        assert name.mean_left == pytest.approx(3.0)
        assert name.mean_right == pytest.approx(4.0)

    def test_ritualistic_is_compared_on_the_quality_scale(self) -> None:
        """Reverse-coded raw scores must not read as a disagreement in the wrong
        direction beside their neighbours."""
        left, right, scenario_of = build(n_scenarios=5, n_samples=2)
        for unit in left:
            left[unit] = {**left[unit], "ritualistic": 1}  # raw 1 = not ritualistic = quality 5
            right[unit] = {**right[unit], "ritualistic": 2}
        report = run(left, right, scenario_of)
        ritual = next(d for d in report.dimensions if d.dimension == "ritualistic")
        assert ritual.mean_left == pytest.approx(5.0)
        assert ritual.mean_right == pytest.approx(4.0)
        # vLLM is *worse* on ritual here, and the difference is negative.
        assert ritual.mean_difference == pytest.approx(-1.0)

    def test_a_missing_score_on_either_side_drops_the_pair_not_the_cell(self) -> None:
        """The judge nulls an ungrounded dimension. Pairwise deletion, per dimension."""
        left, right, scenario_of = build(n_scenarios=4, n_samples=1)
        first = sorted(left)[0]
        left[first] = {**left[first], "name": None}
        report = run(left, right, scenario_of)
        name = next(d for d in report.dimensions if d.dimension == "name")
        understand = next(d for d in report.dimensions if d.dimension == "understand")
        assert name.n_pairs == 3
        assert understand.n_pairs == 4

    def test_a_dimension_scored_on_no_pair_reports_nan_rather_than_zero(self) -> None:
        left, right, scenario_of = build(n_scenarios=3, n_samples=1)
        for unit in left:
            left[unit] = {**left[unit], "name": None}
        report = run(left, right, scenario_of)
        name = next(d for d in report.dimensions if d.dimension == "name")
        assert name.n_pairs == 0
        assert math.isnan(name.mean_difference)
        assert math.isnan(name.alpha)

    def test_real_disagreement_lowers_alpha_below_the_threshold(self) -> None:
        left, right, scenario_of = build(n_scenarios=13, n_samples=3)
        units = sorted(left)
        for i, unit in enumerate(units):
            left[unit] = {**left[unit], "name": 1 + (i % 5)}
            right[unit] = {**right[unit], "name": 5 - (i % 5)}
        report = run(left, right, scenario_of)
        name = next(d for d in report.dimensions if d.dimension == "name")
        assert name.alpha < 0.0
        assert name.rho < 0.0
        assert not name.clears_threshold


class TestResolutionLimit:
    def test_the_detectable_effect_is_reported_with_the_sample_size(self) -> None:
        report = run(*build(n_scenarios=13, n_samples=3))
        # 13 scenario-level pairs resolve only very large effects.
        assert report.detectable_dz_scenarios == pytest.approx(0.865, abs=0.01)
        assert report.detectable_dz_cells == pytest.approx(0.471, abs=0.01)

    def test_thirteen_scenarios_cannot_license_an_equivalence_claim(self) -> None:
        report = run(*build(n_scenarios=13, n_samples=3, left_value=4, right_value=4))
        assert report.n_scenarios < MIN_UNITS_FOR_EQUIVALENCE_CLAIM
        assert report.supports_equivalence_claim is False
        # Perfect agreement must still not read as "the stacks are interchangeable".
        assert any("never randomised" in limit for limit in report.limits)

    def test_the_confounds_travel_with_the_result(self) -> None:
        """A disagreement here does not isolate the serving stack, and the
        result has to carry that rather than leaving it to the prose."""
        report = run(*build())
        assert report.confounds == BACKEND_CONFOUNDS
        assert len(BACKEND_CONFOUNDS) >= 4
        joined = " ".join(BACKEND_CONFOUNDS).lower()
        for confound in ("gguf", "quantis", "sampling", "pack"):
            assert confound in joined

    def test_a_disagreeing_dimension_blocks_pooling(self) -> None:
        left, right, scenario_of = build(n_scenarios=13, n_samples=3)
        for i, unit in enumerate(sorted(left)):
            left[unit] = {**left[unit], "name": 1 + (i % 5)}
            right[unit] = {**right[unit], "name": 5 - (i % 5)}
        report = run(left, right, scenario_of)
        assert report.poolable is False
        assert "name" in report.dimensions_failing_threshold

    def test_pooling_is_refused_even_on_perfect_agreement(self) -> None:
        """The arms are not pooled because the sample cannot license it, which is
        a separate fact from whether the numbers happen to agree."""
        report = run(*build(n_scenarios=13, n_samples=3, left_value=4, right_value=4))
        assert report.poolable is False


class TestRendering:
    def test_the_report_renders_numbers_and_names_its_limits(self) -> None:
        text = run(*build(left_value=3, right_value=4)).render()
        assert "ollama" in text and "vllm" in text
        assert "EXPLORATORY" in text
        assert "39" in text and "13" in text
        assert "name" in text

    def test_the_payload_round_trips_to_json(self) -> None:
        import json

        payload = run(*build()).to_dict()
        assert json.loads(json.dumps(payload))["n_cell_pairs"] == 39
        assert payload["poolable"] is False
