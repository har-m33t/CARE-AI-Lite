"""The five-response calibration set (v3 §12).

Two jobs here. First, structural integrity: the consensus scores must cover
every dimension, stay on the scale, and cite evidence that is verbatim in the
response they describe — the same grounding rule v3 §13 imposes on the judge,
applied to the material the judge is calibrated against.

Second, the teaching points. Each item exists to prevent one specific rater
error, and those errors are only prevented if the consensus scores actually
demonstrate them. If someone "tidies" CAL-03's `respect` score up to match the
rest of the response, the no-halo lesson silently disappears.
"""

from __future__ import annotations

import pytest

from carelite.eval.rubric.calibration import (
    CALIBRATED_AGAINST,
    CALIBRATION_SCENARIO,
    CALIBRATION_SET,
    NATURAL_ARCHETYPE_ID,
    SCRIPT_ARCHETYPE_ID,
    CalibrationItem,
    item,
    validate_calibration_set,
)
from carelite.eval.rubric.dimensions import RUBRIC_VERSION, SCALE_MAX, SCALE_MIN
from carelite.eval.rubric.scorers import (
    jargon_density,
    message_count,
    teach_back_present,
)
from carelite.types import RUBRIC_DIMENSIONS, RaterType

ALL_ITEMS = pytest.mark.parametrize("c", CALIBRATION_SET, ids=[c.item_id for c in CALIBRATION_SET])


# --------------------------------------------------------------- structure ---


def test_there_are_five_responses() -> None:
    assert len(CALIBRATION_SET) == 5
    assert len({c.item_id for c in CALIBRATION_SET}) == 5


def test_the_set_validates() -> None:
    validate_calibration_set()


@ALL_ITEMS
def test_every_dimension_is_scored_and_explained(c: CalibrationItem) -> None:
    assert set(c.consensus) == set(RUBRIC_DIMENSIONS)
    assert set(c.rationales) == set(RUBRIC_DIMENSIONS)
    for key, value in c.consensus.items():
        assert SCALE_MIN <= value <= SCALE_MAX, f"{c.item_id}.{key}"


@ALL_ITEMS
def test_every_evidence_span_is_verbatim(c: CalibrationItem) -> None:
    for key, span in c.evidence_spans.items():
        assert key in RUBRIC_DIMENSIONS
        assert span in c.response, f"{c.item_id}.{key} span is not in the response"


@ALL_ITEMS
def test_the_consensus_converts_to_a_rubric_score_row(c: CalibrationItem) -> None:
    row = c.rubric_score()
    assert row.generation_id == c.item_id
    assert row.rater_type is RaterType.HUMAN
    assert row.rater_id == "consensus"
    for key in RUBRIC_DIMENSIONS:
        assert getattr(row, key) == c.consensus[key]


def test_the_set_is_pinned_to_the_current_rubric_version() -> None:
    """Consensus scores are only meaningful against the rubric they were agreed
    under. Bumping the rubric requires re-scoring the set, not carrying it over."""
    assert CALIBRATED_AGAINST == RUBRIC_VERSION


def test_all_five_answer_the_same_patient_turn() -> None:
    assert "spot on my lung" in CALIBRATION_SCENARIO
    assert "I've been up all night" in CALIBRATION_SCENARIO


# --------------------------------------------------------- teaching points ---


def test_cal_01_shows_that_naturalness_is_not_quality() -> None:
    """A blunt, empathy-free information dump that is nonetheless natural speech."""
    c = item("CAL-01")
    assert c.consensus["name"] == 1
    assert c.consensus["de"] == 1
    assert c.consensus["naturalness"] >= 4
    assert c.consensus["ritualistic"] == 1


def test_cal_02_shows_adherence_and_ritual_coexisting() -> None:
    c = item(SCRIPT_ARCHETYPE_ID)
    assert c.consensus["ritualistic"] == 5  # reverse-coded: the worst score
    assert c.consensus["naturalness"] == 1
    assert c.consensus["name"] >= 4 and c.consensus["understand"] >= 4
    # and `de` must NOT be the average of the NURSE items
    nurse = [c.consensus[k] for k in ("name", "understand", "respect", "support", "explore")]
    assert c.consensus["de"] < min(nurse)


def test_cal_03_shows_that_there_is_no_halo() -> None:
    """The best response in the set still scores low where a move is absent."""
    c = item(NATURAL_ARCHETYPE_ID)
    assert c.consensus["respect"] <= 2
    assert c.consensus["naturalness"] == 5
    assert c.consensus["ritualistic"] == 1
    assert c.consensus["de"] == 5


def test_cal_04_shows_that_warmth_is_not_empathy() -> None:
    c = item("CAL-04")
    assert c.consensus["explore"] == 1
    assert c.consensus["ie"] == 1
    assert c.consensus["epp"] == 1
    # ritual without any template at all
    assert c.consensus["ritualistic"] >= 4


def test_cal_05_is_where_the_counters_and_the_raters_meet() -> None:
    c = item("CAL-05")
    assert teach_back_present(c.response), "CAL-05 must contain a genuine teach-back"
    assert message_count(c.response) > 3, "…and still exceed the three-key-message ceiling"
    assert jargon_density(c.response) > 0.02, "…while being jargon-dense"
    assert c.consensus["ie"] == 3, "which is exactly why `ie` is capped at the midpoint"


# ------------------------------------- consistency with the deterministic arm ---


def test_teach_back_detection_agrees_with_the_consensus_rationales() -> None:
    """Only CAL-03 and CAL-05 contain a genuine teach-back request."""
    detected = {c.item_id for c in CALIBRATION_SET if teach_back_present(c.response)}
    assert detected == {"CAL-03", "CAL-05"}


def test_items_that_score_ie_low_have_no_teach_back() -> None:
    for c in CALIBRATION_SET:
        if c.consensus["ie"] <= 2:
            assert not teach_back_present(c.response), c.item_id


def test_unknown_item_id_raises_a_useful_error() -> None:
    with pytest.raises(KeyError, match="CAL-99"):
        item("CAL-99")
