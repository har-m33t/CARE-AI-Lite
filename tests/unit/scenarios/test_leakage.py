"""Held-out text must not appear in anything an optimiser writes.

The checksum in `test_freeze.py` catches held-out scenarios being *edited*.
This catches them being *copied* -- which is the failure mode Sprint 9 actually
risks, because few-shot optimisers embed their example pool verbatim in the
prompt they produce.
"""

from __future__ import annotations

import pytest

from carelite.config import REPO_ROOT
from carelite.scenarios.bank import holdout_scenarios, train_scenarios
from carelite.scenarios.leakage import (
    DEFAULT_SHINGLE_N,
    HoldoutLeakError,
    assert_no_holdout_leakage,
    find_leaks,
    holdout_shingles,
    normalise,
    scan_paths,
)

#: Directories an optimiser or a generator writes into. Missing ones are skipped,
#: so this test is meaningful before those lanes have landed and stays meaningful
#: after.
WATCHED = (
    REPO_ROOT / "carelite" / "prompts",
    REPO_ROOT / "prompts",
    REPO_ROOT / "runs" / "optimised",
)


def test_every_holdout_scenario_produces_shingles() -> None:
    grams = holdout_shingles()
    assert len(grams) == 60
    assert all(grams.values()), "a held-out scenario produced no n-grams to search for"


def test_a_copied_holdout_utterance_is_detected() -> None:
    victim = holdout_scenarios()[0]
    prompt = f"You are a clinician. Example patient turn: {victim.text}\nRespond warmly."
    leaks = find_leaks(prompt, source="fake_prompt.txt")
    assert [leak.scenario_id for leak in leaks] == [victim.scenario_id]
    with pytest.raises(HoldoutLeakError, match=victim.scenario_id):
        assert_no_holdout_leakage(prompt, source="fake_prompt.txt")


def test_detection_survives_reformatting() -> None:
    # Casing, punctuation and whitespace are normalised away, so a prompt that
    # cleaned up the quoting still trips the guard.
    victim = holdout_scenarios()[1]
    mangled = victim.text.upper().replace(",", "").replace("\n", "  ")
    assert find_leaks(mangled)


def test_detection_survives_dropped_speaker_tags() -> None:
    multi = next(r for r in holdout_scenarios() if r.text.startswith("["))
    stripped = multi.text.replace("[", "").replace("]", "")
    assert [leak.scenario_id for leak in find_leaks(stripped)] == [multi.scenario_id]


def test_train_scenarios_do_not_trip_the_guard() -> None:
    # Optimising on the training split is the sanctioned path; it must be quiet.
    pool = "\n\n".join(r.text for r in train_scenarios())
    assert find_leaks(pool) == []
    assert_no_holdout_leakage(pool, source="train pool")


def test_ordinary_clinical_prose_does_not_trip_the_guard() -> None:
    prose = (
        "Name the emotion tentatively before offering information. Ask an open question "
        "that follows the patient's own words. Confirm understanding with a teach-back "
        "request rather than a yes/no check. Do not reassure before acknowledging."
    )
    assert find_leaks(prose) == []


def test_empty_and_trivial_inputs_are_safe() -> None:
    assert find_leaks("") == []
    assert find_leaks("   \n  ") == []
    assert_no_holdout_leakage("")


def test_normalise_strips_everything_but_words() -> None:
    assert normalise("[Daughter] She won't -- tell you!") == [
        "daughter",
        "she",
        "won't",
        "tell",
        "you",
    ]


def test_shingle_length_is_distinctive_enough() -> None:
    # An 8-gram of ordinary English colliding by chance would make the guard
    # useless through false alarms; a 3-gram would.
    assert DEFAULT_SHINGLE_N >= 6
    short = find_leaks("i don't know what to do about it", n=DEFAULT_SHINGLE_N)
    assert short == []


def test_no_holdout_text_has_leaked_into_prompt_artifacts() -> None:
    paths = [
        p
        for directory in WATCHED
        if directory.is_dir()
        for p in directory.rglob("*")
        if p.is_file()
    ]
    leaks = scan_paths(paths)
    assert not leaks, "held-out scenario text found in optimiser output:\n" + "\n".join(
        f"  - {leak}" for leak in leaks
    )
