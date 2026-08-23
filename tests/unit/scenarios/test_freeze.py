"""The held-out set is write-once, and this is the test that enforces it.

If you are reading this because it just failed: you changed a held-out scenario.
That is a protocol amendment, not a routine edit. Build plan v3, Part V, names
the held-out set as the only defence against GEPA overfitting, and every
held-out result produced before the change is incomparable to one produced
after it. Revert, or amend deliberately and record it in the limitations.
"""

from __future__ import annotations

import os

import pytest

from carelite.scenarios.bank import load_bank
from carelite.scenarios.freeze import (
    HOLDOUT_DIGEST,
    LOCK_PATH,
    UNFREEZE_ENV,
    UNFREEZE_TOKEN,
    HoldoutTamperError,
    canonical_record,
    holdout_digest,
    read_lock,
    record_digest,
    record_digests,
    verify_holdout,
    write_lock,
)
from carelite.types import Split


def test_holdout_matches_its_frozen_checksum() -> None:
    assert verify_holdout() == HOLDOUT_DIGEST


def test_lock_file_covers_exactly_the_holdout_set() -> None:
    locked = set(read_lock())
    holdout = {r.scenario_id for r in load_bank() if r.split is Split.HOLDOUT}
    assert locked == holdout
    assert len(locked) == 60


def test_lock_file_agrees_with_the_computed_digests() -> None:
    assert read_lock() == record_digests()


def test_lock_header_records_the_aggregate() -> None:
    header = LOCK_PATH.read_text(encoding="utf-8").splitlines()[2]
    assert header == f"# aggregate: {HOLDOUT_DIGEST}"


# --------------------------------------------------------------------------
# What the checksum actually catches
# --------------------------------------------------------------------------


def _holdout_index() -> int:
    rows = load_bank()
    return next(i for i, r in enumerate(rows) if r.split is Split.HOLDOUT)


def test_editing_holdout_text_is_caught() -> None:
    rows = list(load_bank())
    i = _holdout_index()
    rows[i] = rows[i].model_copy(update={"text": rows[i].text + " Sorry, one more thing."})
    with pytest.raises(HoldoutTamperError, match=r"modified: " + rows[i].scenario_id):
        verify_holdout(rows)


def test_editing_a_holdout_stratum_value_is_caught() -> None:
    rows = list(load_bank())
    i = _holdout_index()
    bumped = 1 if rows[i].emotion_intensity == 5 else rows[i].emotion_intensity + 1
    rows[i] = rows[i].model_copy(update={"emotion_intensity": bumped})
    with pytest.raises(HoldoutTamperError, match="modified"):
        verify_holdout(rows)


def test_moving_a_scenario_out_of_holdout_is_caught() -> None:
    rows = list(load_bank())
    i = _holdout_index()
    rows[i] = rows[i].model_copy(update={"split": Split.TRAIN})
    with pytest.raises(HoldoutTamperError, match="removed from holdout"):
        verify_holdout(rows)


def test_moving_a_scenario_into_holdout_is_caught() -> None:
    rows = list(load_bank())
    i = next(j for j, r in enumerate(rows) if r.split is Split.TRAIN)
    rows[i] = rows[i].model_copy(update={"split": Split.HOLDOUT})
    with pytest.raises(HoldoutTamperError, match="added to holdout"):
        verify_holdout(rows)


def test_dropping_a_holdout_scenario_is_caught() -> None:
    rows = [r for r in load_bank() if r.scenario_id != load_bank()[_holdout_index()].scenario_id]
    with pytest.raises(HoldoutTamperError, match="removed from holdout"):
        verify_holdout(rows)


def test_editing_the_train_split_is_not_caught_by_this_checksum() -> None:
    # Train scenarios are meant to be worked on. Scoping the freeze to holdout
    # is what keeps the guarantee meaningful rather than merely noisy.
    rows = list(load_bank())
    i = next(j for j, r in enumerate(rows) if r.split is Split.TRAIN)
    rows[i] = rows[i].model_copy(update={"text": "Completely different utterance, honestly."})
    assert verify_holdout(rows) == HOLDOUT_DIGEST


def test_curator_metadata_is_outside_the_freeze() -> None:
    # A reviewer sharpening a note at the wave-2 gate is not an edit to the
    # evaluation item.
    rows = list(load_bank())
    i = _holdout_index()
    rows[i] = rows[i].model_copy(
        update={"curator_note": "Reworded during second-person review.", "hard_case": ["x"]}
    )
    assert verify_holdout(rows) == HOLDOUT_DIGEST


# --------------------------------------------------------------------------
# Determinism and the unfreeze guard
# --------------------------------------------------------------------------


def test_digests_are_deterministic_across_calls() -> None:
    assert holdout_digest() == holdout_digest() == HOLDOUT_DIGEST


def test_canonical_record_is_order_independent() -> None:
    record = load_bank()[_holdout_index()]
    round_tripped = record.model_copy(update={"curator_note": "different"})
    assert canonical_record(record) == canonical_record(round_tripped)
    assert record_digest(record) == record_digest(round_tripped)


def test_write_lock_refuses_without_the_explicit_acknowledgement(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(UNFREEZE_ENV, raising=False)
    with pytest.raises(HoldoutTamperError, match="write-once"):
        write_lock(path=tmp_path / "holdout.lock")
    assert not (tmp_path / "holdout.lock").exists()


def test_write_lock_refuses_a_wrong_acknowledgement(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(UNFREEZE_ENV, "yes")
    with pytest.raises(HoldoutTamperError):
        write_lock(path=tmp_path / "holdout.lock")


def test_write_lock_works_when_deliberately_unfrozen(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(UNFREEZE_ENV, UNFREEZE_TOKEN)
    dest = tmp_path / "holdout.lock"
    digest = write_lock(path=dest)
    assert digest == HOLDOUT_DIGEST
    assert read_lock(dest) == record_digests()
    # The real lock file is untouched by a redirected write.
    assert LOCK_PATH.read_text(encoding="utf-8").count("SC-") == 60


def test_the_unfreeze_token_is_not_set_in_this_environment() -> None:
    # A CI job or a shell profile that exported this would silently disarm the
    # guard for everybody.
    assert os.environ.get(UNFREEZE_ENV) != UNFREEZE_TOKEN
