"""The stratum audit passes on the real bank, and fails loudly when it should."""

from __future__ import annotations

import pytest

from carelite.scenarios.audit import (
    ACCEPTED_EMPTY_CELLS,
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


def test_no_gated_view_has_an_empty_cell_except_the_one_that_was_decided() -> None:
    report = run_audit()
    empties = {
        (view, key)
        for view in report.cells
        for key in report.empty_cells(view)
        if (view, key) not in ACCEPTED_EMPTY_CELLS
    }
    assert not empties, empties


def test_the_accepted_gap_allowlist_holds_exactly_one_entry() -> None:
    # This is the guard on the exemption mechanism itself. `ACCEPTED_EMPTY_CELLS`
    # is meant to record the one hole DECISIONS.md D2 knowingly left, not to become
    # a place where inconvenient coverage failures go to be silenced. Adding a
    # second entry has to break this test and be argued for in the diff.
    assert ACCEPTED_EMPTY_CELLS.keys() == {("equity_stratum x emotion_intensity", (True, 1))}
    reason = next(iter(ACCEPTED_EMPTY_CELLS.values()))
    assert "DECISIONS.md D2" in reason


def test_the_accepted_gap_is_still_reported_rather_than_hidden() -> None:
    # An exemption that stopped being printed would be indistinguishable from a
    # coverage claim the bank cannot make.
    report = run_audit()
    assert report.ok
    assert len(report.exemptions) == 1
    assert "emotion_intensity=1" in report.exemptions[0]
    assert "ACCEPTED:" in report.exemptions[0]


def test_equity_is_not_confounded_with_challenge_type() -> None:
    # If equity scenarios clustered into one topic, a subgroup effect would be
    # indistinguishable from a topic effect.
    report = run_audit()
    per_type = report.cells["equity_stratum x challenge_type"]
    assert min(per_type.values()) >= EQUITY_MIN_PER_CHALLENGE


def test_equity_is_not_confounded_with_emotion_intensity() -> None:
    # The original claim was that the equity stratum spans all five intensity
    # levels. DECISIONS.md D2 ended that: SC-010 was the only equity scenario at
    # intensity 1, and it left the stratum because its LEP signal was carried by
    # register rather than by a situation. The stratum now spans 2-5. That is a
    # documented gap, not a silent one -- see ACCEPTED_EMPTY_CELLS -- and the
    # remaining levels still have to be populated, which is what stops the
    # stratum drifting into the top of the range.
    report = run_audit()
    table = report.cells["equity_stratum x emotion_intensity"]
    assert table[(True, 1)] == 0
    for level in INTENSITIES:
        if level != 1:
            assert table[(True, level)] >= 1, f"no equity scenario at emotion_intensity {level}"
        assert table[(False, level)] >= 1
    # The confound the check exists for: equity must not concentrate at high intensity.
    assert table[(True, 2)] + table[(True, 3)] >= table[(True, 4)] + table[(True, 5)]


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
