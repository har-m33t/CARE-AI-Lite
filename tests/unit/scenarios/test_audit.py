"""The stratum audit passes on the real bank, and fails loudly when it should."""

from __future__ import annotations

import pytest

from carelite.scenarios.audit import (
    EQUITY_MIN_PER_CHALLENGE,
    INTENSITIES,
    PHASES,
    AuditReport,
    StratumCoverageError,
    assert_full_coverage,
    main,
    run_audit,
)
from carelite.scenarios.bank import CHALLENGE_TYPES, LITERACY_SIGNALS, load_bank
from carelite.types import Split

# --------------------------------------------------------------------------
# The real bank
# --------------------------------------------------------------------------


def test_the_bank_passes_the_audit() -> None:
    report = assert_full_coverage()
    assert report.ok
    assert (report.n_total, report.n_train, report.n_holdout) == (100, 40, 60)


def test_the_design_cell_is_complete() -> None:
    # 10 challenge types x 5 encounter phases, two scenarios each. This is the
    # concrete meaning of "every stratum cell populated" for this bank.
    report = run_audit()
    table = report.cells["challenge_type x encounter_phase"]
    assert len(table) == len(CHALLENGE_TYPES) * len(PHASES) == 50
    assert set(table.values()) == {2}


def test_every_factor_level_is_populated() -> None:
    report = run_audit()
    assert set(report.marginals["challenge_type"]) == set(CHALLENGE_TYPES)
    assert set(report.marginals["emotion_intensity"]) == set(INTENSITIES)
    assert set(report.marginals["encounter_phase"]) == set(PHASES)
    assert set(report.marginals["literacy_signal"]) == set(LITERACY_SIGNALS)
    assert set(report.marginals["equity_stratum"]) == {True, False}


def test_no_gated_view_has_an_empty_cell() -> None:
    report = run_audit()
    empties = {view: report.empty_cells(view) for view in report.cells}
    assert all(not cells for cells in empties.values()), empties


def test_equity_is_not_confounded_with_challenge_type() -> None:
    # If equity scenarios clustered into one topic, a subgroup effect would be
    # indistinguishable from a topic effect.
    report = run_audit()
    per_type = report.cells["equity_stratum x challenge_type"]
    assert min(per_type.values()) >= EQUITY_MIN_PER_CHALLENGE


def test_equity_is_not_confounded_with_emotion_intensity() -> None:
    report = run_audit()
    table = report.cells["equity_stratum x emotion_intensity"]
    for level in INTENSITIES:
        assert table[(True, level)] >= 1, f"no equity scenario at emotion_intensity {level}"
        assert table[(False, level)] >= 1


def test_split_is_stratified_not_arbitrary() -> None:
    report = run_audit()
    by_type = report.cells["split x challenge_type"]
    assert all(by_type[(Split.TRAIN, ct)] == 4 for ct in CHALLENGE_TYPES)
    assert all(by_type[(Split.HOLDOUT, ct)] == 6 for ct in CHALLENGE_TYPES)
    by_phase = report.cells["split x encounter_phase"]
    assert all(by_phase[(Split.TRAIN, p)] == 8 for p in PHASES)
    assert all(by_phase[(Split.HOLDOUT, p)] == 12 for p in PHASES)


def test_equity_subgroup_is_analysable_in_holdout() -> None:
    report = run_audit()
    assert report.cells["split x equity_stratum"][(Split.HOLDOUT, True)] >= 15


def test_main_exits_zero() -> None:
    assert main() == 0


# --------------------------------------------------------------------------
# The audit has to fail when coverage breaks
# --------------------------------------------------------------------------


def test_emptying_a_design_cell_is_caught() -> None:
    # Move both scenarios out of one (challenge_type, phase) cell.
    rows = list(load_bank())
    target = next(r for r in rows if r.challenge_type == "emotional_cue")
    moved = [
        r.model_copy(update={"challenge_type": "trust_rupture"})
        if r.challenge_type == target.challenge_type and r.encounter_phase == target.encounter_phase
        else r
        for r in rows
    ]
    with pytest.raises(StratumCoverageError, match="EMPTY CELL"):
        assert_full_coverage(moved)


def test_dropping_an_intensity_level_is_caught() -> None:
    rows = [
        r.model_copy(update={"emotion_intensity": 3}) if r.emotion_intensity == 1 else r
        for r in load_bank()
    ]
    with pytest.raises(StratumCoverageError, match="emotion_intensity/1"):
        assert_full_coverage(rows)


def test_dropping_a_literacy_level_is_caught() -> None:
    rows = [
        r.model_copy(update={"literacy_signal": "unmarked"})
        if r.literacy_signal == "numeracy_gap"
        else r
        for r in load_bank()
    ]
    with pytest.raises(StratumCoverageError, match="literacy_signal/numeracy_gap"):
        assert_full_coverage(rows)


def test_removing_the_equity_stratum_is_caught() -> None:
    rows = [
        r.model_copy(update={"equity_stratum": False, "equity_kind": None}) for r in load_bank()
    ]
    with pytest.raises(StratumCoverageError) as exc:
        assert_full_coverage(rows)
    assert "equity_stratum/True" in str(exc.value)


def test_a_lopsided_split_is_caught() -> None:
    rows = [r.model_copy(update={"split": Split.TRAIN}) for r in load_bank()]
    with pytest.raises(StratumCoverageError, match="split/train"):
        assert_full_coverage(rows)


def test_every_violation_is_reported_not_just_the_first() -> None:
    rows = [
        r.model_copy(update={"split": Split.TRAIN, "emotion_intensity": 3}) for r in load_bank()
    ]
    report = run_audit(rows)
    assert len(report.violations) > 5, report.violations


def test_report_is_inspectable_without_raising() -> None:
    empty = run_audit([])
    assert isinstance(empty, AuditReport)
    assert not empty.ok
    assert any("expected 100 scenarios" in line for line in empty.violations)
