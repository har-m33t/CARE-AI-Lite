"""The sign test.

`ritualistic` is the only reverse-coded dimension in the rubric: a raw 5 is the
WORST response, not the best. Build plan v3 predicts Condition B loses to
Condition A on naturalness precisely because framework prompting induces
formulaic output, and `ritualistic` is the dimension that detects that
mechanism.

A missing `6 - x` anywhere downstream inverts that headline secondary finding
and still produces numbers that look entirely plausible. That is what this file
exists to prevent.

IF A TEST IN THIS FILE FAILS, DO NOT FIX IT BY FLIPPING A CONSTANT.
Something downstream is about to report the opposite of the truth.
"""

from __future__ import annotations

import pytest

from carelite.eval.rubric.calibration import (
    NATURAL_ARCHETYPE_ID,
    SCRIPT_ARCHETYPE_ID,
    item,
)
from carelite.eval.rubric.dimensions import (
    DIMENSIONS,
    REVERSE_CODED,
    SCALE_MAX,
    SCALE_MIN,
    Polarity,
    is_reverse_coded,
    to_quality,
)
from carelite.eval.rubric.scorers import deterministic_rubric_score, ritualistic_proxy
from carelite.types import RUBRIC_DIMENSIONS

NURSE_KEYS = ("name", "understand", "respect", "support", "explore")


# --------------------------------------------------------------- the set ---


def test_ritualistic_is_the_only_reverse_coded_dimension() -> None:
    assert frozenset({"ritualistic"}) == REVERSE_CODED


def test_every_other_dimension_is_higher_is_better() -> None:
    for key in RUBRIC_DIMENSIONS:
        if key == "ritualistic":
            continue
        assert not is_reverse_coded(key), f"{key} must not be reverse-coded"
        assert DIMENSIONS[key].polarity is Polarity.HIGHER_IS_BETTER


def test_ritualistic_polarity_is_lower_is_better() -> None:
    assert DIMENSIONS["ritualistic"].polarity is Polarity.LOWER_IS_BETTER
    assert is_reverse_coded("ritualistic")


# ------------------------------------------------------- the arithmetic ---


@pytest.mark.parametrize(
    ("raw", "expected_quality"),
    [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)],
)
def test_ritualistic_raw_5_is_the_worst_quality(raw: int, expected_quality: int) -> None:
    """A raw 5 on `ritualistic` must map to quality 1, and a raw 1 to quality 5."""
    assert to_quality("ritualistic", raw) == expected_quality


@pytest.mark.parametrize("key", [k for k in RUBRIC_DIMENSIONS if k != "ritualistic"])
@pytest.mark.parametrize("raw", [1, 2, 3, 4, 5])
def test_other_dimensions_pass_through_unchanged(key: str, raw: int) -> None:
    assert to_quality(key, raw) == raw


def test_to_quality_is_its_own_inverse() -> None:
    for raw in range(SCALE_MIN, SCALE_MAX + 1):
        assert to_quality("ritualistic", to_quality("ritualistic", raw)) == raw


def test_to_quality_rejects_off_scale_scores() -> None:
    for bad in (0, 6, -1):
        with pytest.raises(ValueError):
            to_quality("ritualistic", bad)


# -------------------------------------------- the direction, end to end ---


def test_the_script_archetype_scores_higher_ritualistic_than_the_natural_one() -> None:
    """CAL-02 is a framework-labelled script; CAL-03 is the natural target.

    On the RAW scale the script must score HIGHER, because higher is worse.
    """
    script = item(SCRIPT_ARCHETYPE_ID)
    natural = item(NATURAL_ARCHETYPE_ID)

    assert script.consensus["ritualistic"] > natural.consensus["ritualistic"]
    assert script.consensus["ritualistic"] == SCALE_MAX  # the worst score available
    assert natural.consensus["ritualistic"] == SCALE_MIN


def test_on_the_quality_scale_the_direction_flips() -> None:
    """The same comparison, after `to_quality`, must come out the other way."""
    script = to_quality("ritualistic", item(SCRIPT_ARCHETYPE_ID).consensus["ritualistic"])
    natural = to_quality("ritualistic", item(NATURAL_ARCHETYPE_ID).consensus["ritualistic"])
    assert script < natural, "the scripted response must be the WORSE one on the quality scale"


def test_naturalness_runs_the_normal_way_round() -> None:
    """Sanity check against a global sign flip: naturalness is higher-is-better."""
    script = item(SCRIPT_ARCHETYPE_ID)
    natural = item(NATURAL_ARCHETYPE_ID)
    assert script.consensus["naturalness"] < natural.consensus["naturalness"]
    assert to_quality("naturalness", script.consensus["naturalness"]) < to_quality(
        "naturalness", natural.consensus["naturalness"]
    )


def test_high_nurse_adherence_and_maximum_ritual_coexist() -> None:
    """The v3 prediction, encoded.

    Condition B's failure mode is a response that executes the NURSE moves
    correctly AND is a script. If a rater or a judge cannot express both at
    once, the effect the study is designed to detect disappears.
    """
    script = item(SCRIPT_ARCHETYPE_ID)
    assert script.consensus["ritualistic"] == SCALE_MAX
    assert all(script.consensus[k] >= 3 for k in NURSE_KEYS), script.consensus


# ------------------------------------------------- the deterministic arm ---


def test_ritualistic_proxy_uses_the_same_reverse_coded_direction() -> None:
    """The mechanical proxy must not be on the opposite scale from the rubric."""
    script = item(SCRIPT_ARCHETYPE_ID)
    natural = item(NATURAL_ARCHETYPE_ID)
    assert ritualistic_proxy(script.response) > ritualistic_proxy(natural.response)
    assert ritualistic_proxy(script.response) == SCALE_MAX
    assert ritualistic_proxy(natural.response) == SCALE_MIN


def test_deterministic_rater_row_carries_the_reverse_coded_value() -> None:
    script = deterministic_rubric_score("gen-script", item(SCRIPT_ARCHETYPE_ID).response)
    natural = deterministic_rubric_score("gen-natural", item(NATURAL_ARCHETYPE_ID).response)

    assert script.ritualistic is not None and natural.ritualistic is not None
    assert script.ritualistic > natural.ritualistic

    # Only `ritualistic` is filled; a fabricated value on any other dimension
    # would contaminate every aggregate that reads `rubric_score`.
    for key in RUBRIC_DIMENSIONS:
        if key != "ritualistic":
            assert getattr(script, key) is None, f"{key} must stay NULL for a deterministic rater"


def test_proxy_agrees_with_human_consensus_within_one_point() -> None:
    """Screening proxy, not a rating — but it must not be pointing the wrong way."""
    from carelite.eval.rubric.calibration import CALIBRATION_SET

    for c in CALIBRATION_SET:
        proxy = ritualistic_proxy(c.response)
        consensus = c.consensus["ritualistic"]
        assert abs(proxy - consensus) <= 1, (
            f"{c.item_id}: proxy {proxy} vs consensus {consensus} — "
            "a gap this large means the proxy is measuring something else"
        )
