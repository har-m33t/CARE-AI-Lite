"""The assembled report: the whole plan in the registered order, or nothing at all."""

from __future__ import annotations

import pandas as pd
import pytest

from carelite.eval.judge.validation import EvidenceStatus
from carelite.stats.report import run_analysis
from carelite.types import RUBRIC_DIMENSIONS
from tests.unit.stats.conftest import constant_scores, make_long


@pytest.fixture
def full_long(nurse_dimensions: tuple[str, ...]) -> pd.DataFrame:
    """Every registered condition on 15 scenarios, on all eleven dimensions."""
    values = {"A": 2, "A2": 2, "B": 4, "C": 4, "LC": 3, "D": 1}
    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(15):
        scenario = f"SC-{i:03d}"
        for condition, value in values.items():
            for sample in range(3):
                cell = dict.fromkeys(RUBRIC_DIMENSIONS, value)
                # Framework-prompted conditions are more ritualistic (raw, higher = worse)
                # and less natural, which is the registered secondary prediction.
                cell["ritualistic"] = 4 if condition in ("B", "C") else 2
                cell["naturalness"] = 2 if condition in ("B", "C") else 4
                scores[(scenario, condition, sample)] = cell
    long = make_long(scores=scores, equity_scenarios=[f"SC-{i:03d}" for i in range(6)])
    long["equity_kind"] = "ses"
    return long


def test_the_report_runs_every_pre_specified_analysis(full_long: pd.DataFrame) -> None:
    report = run_analysis(long=full_long, n_boot=200)
    # Seven computed, not eight: C vs LC is retired by D11 and is never run.
    # It keeps its slot in the family, which is the point of the next assertion.
    assert len(report.primary.results) == 7
    assert report.primary.by_key("secondary3_nurse_C_vs_LC") is None
    assert report.primary.family_size == 8
    assert any("D11" in note for note in report.primary.notes)
    assert len(report.primary.friedman) == len(RUBRIC_DIMENSIONS)
    assert len(report.mixed) == 2  # NURSE and Four Habits composites
    assert report.equity is not None
    assert report.negative_control_result is not None
    assert report.sensitivity is not None


def test_the_report_counts_its_own_units(full_long: pd.DataFrame) -> None:
    report = run_analysis(long=full_long, n_boot=200)
    assert report.n_scenarios == 15
    assert report.n_generations == 15 * 5 * 3  # six conditions generated, LC dropped by D11
    assert report.rater_types == ("llm_judge",)


def test_the_judge_gate_is_printed_before_the_results_it_governs(
    full_long: pd.DataFrame,
) -> None:
    report = run_analysis(long=full_long, n_boot=200)
    text = report.render()
    assert text.index("JUDGE-VALIDATION GATE") < text.index("PRIMARY ANALYSIS")
    assert "EVERY dimension is exploratory" in text


def test_every_result_is_exploratory_without_a_validation_study(
    full_long: pd.DataFrame,
) -> None:
    report = run_analysis(long=full_long, n_boot=200)
    assert len(report.exploratory_dimensions) == len(RUBRIC_DIMENSIONS)
    assert report.primary.confirmatory == ()


def test_the_registered_naturalness_prediction_is_detected(
    full_long: pd.DataFrame,
) -> None:
    """The fixture plants A > B on naturalness, which §4.4 registers."""
    report = run_analysis(long=full_long, n_boot=200)
    naturalness = report.primary.by_key("secondary4_naturalness_A_vs_B")
    assert naturalness is not None
    assert naturalness.observed_direction == ">"
    assert naturalness.direction_as_registered is True


def test_the_registered_ritualistic_prediction_is_read_on_the_quality_scale(
    full_long: pd.DataFrame,
) -> None:
    """B is more ritualistic in raw terms, so A must score HIGHER on quality."""
    report = run_analysis(long=full_long, n_boot=200)
    ritual = report.primary.by_key("secondary5_ritualistic_A_vs_B")
    assert ritual is not None
    assert ritual.observed_direction == ">"
    assert ritual.effects.hodges_lehmann.point == pytest.approx(2.0)


def test_the_cross_model_baseline_shows_no_difference(full_long: pd.DataFrame) -> None:
    """§4.6 registers A ~ A2; the fixture makes them identical."""
    report = run_analysis(long=full_long, n_boot=200)
    baseline = report.primary.by_key("secondary6_nurse_A_vs_A2")
    assert baseline is not None
    assert baseline.observed_direction == "="
    assert baseline.direction_as_registered is None
    assert not baseline.significant()


def test_the_negative_control_passes_on_a_working_rubric(
    full_long: pd.DataFrame,
) -> None:
    report = run_analysis(long=full_long, n_boot=200)
    assert report.negative_control_result is not None
    assert report.negative_control_result.rubric_separates


def test_a_validated_judge_promotes_only_fully_validated_measures(
    full_long: pd.DataFrame,
) -> None:
    from carelite.eval.judge.validation import AgreementResult, DimensionValidity

    validity = {
        key: DimensionValidity(
            agreement=AgreementResult(key, 40, 2, 0.8, 0.7, 0.001),
            status=EvidenceStatus.CONFIRMATORY,
        )
        for key in RUBRIC_DIMENSIONS
    }
    validity["naturalness"] = DimensionValidity(
        agreement=AgreementResult("naturalness", 40, 2, 0.3, 0.2, 0.4),
        status=EvidenceStatus.EXPLORATORY,
    )
    report = run_analysis(long=full_long, validity=validity, n_boot=200)
    assert report.exploratory_dimensions == ("naturalness",)
    primary = report.primary.by_key("primary_nurse_A_vs_B")
    naturalness = report.primary.by_key("secondary4_naturalness_A_vs_B")
    assert primary is not None and naturalness is not None
    assert primary.label.is_confirmatory
    assert not naturalness.label.is_confirmatory


def test_an_empty_database_renders_the_structure_and_says_it_is_empty() -> None:
    report = run_analysis(long=make_long(scores={}), n_boot=50)
    assert report.empty
    assert report.n_generations == 0
    text = report.render()
    assert "NO RESULTS DATA" in text
    assert "carelite.eval.judge.load" in text
    # The power analysis does not need data and must still be there.
    assert "POWER ANALYSIS" in text


def test_the_report_survives_a_frame_with_only_one_condition(
    nurse_dimensions: tuple[str, ...],
) -> None:
    scores = {(f"SC-{i:03d}", "A", 0): constant_scores(nurse_dimensions, 3) for i in range(5)}
    report = run_analysis(long=make_long(scores=scores), n_boot=50)
    assert report.primary.results == ()
    assert report.mixed == ()
    assert report.negative_control_result is None
    assert "not computable" in report.render()


def test_a_missing_validation_study_says_so_rather_than_blaming_the_judge(
    full_long: pd.DataFrame,
) -> None:
    """'has not run' and 'failed the threshold' are different claims about the judge."""
    report = run_analysis(long=full_long, n_boot=200)
    primary = report.primary.by_key("primary_nurse_A_vs_B")
    assert primary is not None
    assert "judge validation study has not run" in primary.label.tag()
    assert "below the pre-specified threshold" not in primary.label.tag()


def test_a_real_validation_failure_names_the_dimension_that_failed(
    full_long: pd.DataFrame,
) -> None:
    from carelite.eval.judge.validation import AgreementResult, DimensionValidity

    validity = {
        key: DimensionValidity(
            agreement=AgreementResult(key, 40, 2, 0.8, 0.7, 0.001),
            status=EvidenceStatus.CONFIRMATORY,
        )
        for key in RUBRIC_DIMENSIONS
    }
    validity["explore"] = DimensionValidity(
        agreement=AgreementResult("explore", 40, 2, 0.1, 0.1, 0.6),
        status=EvidenceStatus.EXPLORATORY,
    )
    report = run_analysis(long=full_long, validity=validity, n_boot=200)
    primary = report.primary.by_key("primary_nurse_A_vs_B")
    assert primary is not None
    assert "below the fixed threshold on explore" in primary.label.tag()
