"""The bank loads, validates, and matches the frozen contract."""

from __future__ import annotations

import json
import re

import pytest
from pydantic import ValidationError

from carelite.scenarios import bank
from carelite.scenarios.bank import (
    CHALLENGE_TYPES,
    EQUITY_KINDS,
    LITERACY_SIGNALS,
    BankError,
    CuratedScenario,
    by_id,
    equity_scenarios,
    holdout_scenarios,
    load_bank,
    scenarios,
    train_scenarios,
)
from carelite.types import EncounterPhase, Scenario, Split


def test_bank_loads_one_hundred_scenarios() -> None:
    assert len(load_bank()) == 100


def test_split_sizes_are_forty_sixty() -> None:
    assert len(train_scenarios()) == 40
    assert len(holdout_scenarios()) == 60


def test_splits_are_disjoint_and_cover_the_bank() -> None:
    train = {r.scenario_id for r in train_scenarios()}
    holdout = {r.scenario_id for r in holdout_scenarios()}
    assert not train & holdout
    assert train | holdout == {r.scenario_id for r in load_bank()}


def test_aliases_point_at_the_right_splits() -> None:
    # The names exist so a call site says which set it is asking for.
    assert bank.for_optimisation() == train_scenarios()
    assert bank.for_final_evaluation() == holdout_scenarios()


def test_every_record_narrows_to_the_frozen_scenario_contract() -> None:
    frozen = scenarios()
    assert len(frozen) == 100
    assert all(isinstance(s, Scenario) for s in frozen)
    assert {f for f in Scenario.model_fields} == {
        "scenario_id",
        "text",
        "challenge_type",
        "emotion_intensity",
        "encounter_phase",
        "literacy_signal",
        "equity_stratum",
        "split",
    }


def test_ids_are_well_formed_and_sorted() -> None:
    ids = [r.scenario_id for r in load_bank()]
    assert ids == sorted(ids)
    assert all(sid.startswith("SC-") and sid[3:].isdigit() for sid in ids)
    assert len(set(ids)) == 100


def test_controlled_vocabularies_hold() -> None:
    for record in load_bank():
        assert record.challenge_type in CHALLENGE_TYPES
        assert record.literacy_signal in LITERACY_SIGNALS
        assert isinstance(record.encounter_phase, EncounterPhase)
        assert isinstance(record.split, Split)
        assert 1 <= record.emotion_intensity <= 5
        if record.equity_kind is not None:
            assert record.equity_kind in EQUITY_KINDS


def test_equity_boolean_and_equity_kind_agree() -> None:
    # The schema stores only the boolean; the kind is what a reviewer reads.
    # A record where they disagree would review as one thing and analyse as another.
    for record in equity_scenarios():
        assert record.equity_kind is not None
    for record in load_bank():
        assert record.equity_stratum == (record.equity_kind is not None)


def test_the_d2_reclassification_holds() -> None:
    # DECISIONS.md D2 (2026-08-24) moved SC-077 and SC-010 out of the equity
    # stratum and changed nothing else about them. Both directions matter and
    # both are asserted here: the stratum flags are gone, and the utterances are
    # byte-identical to the text D2 quotes. A later edit that quietly restored
    # either scenario to the stratum -- or that "tidied" the text of a held-out
    # scenario while it happened to be open -- would be caught here as well as by
    # the holdout digest.
    for sid in ("SC-010", "SC-077"):
        record = by_id(sid)
        assert record.equity_stratum is False, f"{sid} is back in the equity stratum"
        assert record.equity_kind is None, f"{sid} still carries an equity_kind"
        assert record.split is Split.HOLDOUT

    assert by_id("SC-010").text == (
        "Yes, thank you doctor. Everything is good. Maybe my son can call you with "
        "the questions. I don't want to take more of your time."
    )
    assert by_id("SC-077").text == (
        "I'll be honest with you, I never filled it. I read what it does to your "
        "kidneys, and a man at my church said the same thing happened to his brother. "
        "I know that's not — I know."
    )
    # They stay in the bank as non-equity scenarios; they were not deleted.
    assert by_id("SC-010").challenge_type == "emotional_cue"
    assert by_id("SC-077").challenge_type == "adherence_barrier"


def test_every_scenario_carries_curation_metadata() -> None:
    for record in load_bank():
        assert record.hard_case, f"{record.scenario_id} has no hard_case tags"
        assert len(record.curator_note) > 20, f"{record.scenario_id} has no curator note"


#: Four mechanically detectable markers of spoken rather than written language.
#: Not a quality metric -- a regression guard. Curation quality is the ceiling on
#: this study's validity, and the cheapest way to destroy it is to replace hand
#: written turns with clean generated prose. Generated prose scores near zero on
#: all four of these; this bank scores 75.
_SPEECH_DEVICES = {
    # trailing off, self-interruption, or a speaker tag
    "marker": lambda t: any(m in t for m in ("...", "--", "—", "[")),
    # immediate repetition: "Forty minutes. Forty."
    "repetition": lambda t: re.search(r"\b(\w+)\b[,.!?]?\s+\1\b", t, re.I) is not None,
    # a question delivered flat, without the question mark
    "flat_question": lambda t: (
        re.search(
            r"(?:^|[.!?]\s+)(?:is|are|do|does|did|can|could|will|would|how|what|which|when"
            r"|where|why|am|should|was|were|have|has)\b[^.?!]*\.",
            t,
            re.I,
        )
        is not None
    ),
    # a one- or two-word sentence fragment
    "fragment": lambda t: re.search(r"(?:^|[.!?]\s+)\w+(?:\s+\w+)?\.", t) is not None,
}

#: Deliberately a floor, not a total. Roughly a quarter of the bank is speech
#: like through structure alone -- short declarative bursts, tag questions,
#: run-ons -- which no regex will detect. Claiming 100 would be claiming a
#: precision this heuristic does not have.
_MIN_DISFLUENT = 70


def test_utterances_look_like_speech_not_prose() -> None:
    hits = [r for r in load_bank() if any(f(r.text) for f in _SPEECH_DEVICES.values())]
    assert len(hits) >= _MIN_DISFLUENT, (
        f"only {len(hits)} of 100 scenarios carry a detectable spoken-language device; "
        f"expected at least {_MIN_DISFLUENT}"
    )


def test_every_device_is_used_somewhere() -> None:
    rows = load_bank()
    for name, detect in _SPEECH_DEVICES.items():
        assert any(detect(r.text) for r in rows), f"no scenario uses the {name} device"


def test_utterances_are_single_turns_not_monologues() -> None:
    # The rubric's unit of analysis is one clinician turn answering one patient
    # utterance. A 200-word patient speech would change what is being measured.
    for record in load_bank():
        words = len(record.text.split())
        assert 8 <= words <= 60, f"{record.scenario_id}: {words} words"
        assert "\n" not in record.text, f"{record.scenario_id}: multi-line utterance"


def test_no_scenario_contains_an_identifier_shaped_string() -> None:
    # Synthetic-only guarantee, mechanically: nothing that looks like an MRN,
    # a date of birth, a phone number or an email should exist in the bank.
    patterns = {
        "long digit run": re.compile(r"\d{5,}"),
        "date": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
        "email": re.compile(r"[\w.]+@[\w.]+"),
    }
    for record in load_bank():
        for name, pattern in patterns.items():
            assert not pattern.search(record.text), f"{record.scenario_id}: {name} in text"


def test_extra_fields_are_rejected() -> None:
    payload = load_bank()[0].model_dump()
    payload["invented_field"] = "x"
    with pytest.raises(ValidationError):
        CuratedScenario.model_validate(payload)


def test_equity_mismatch_is_rejected() -> None:
    payload = load_bank()[0].model_dump()
    payload["equity_stratum"] = not payload["equity_stratum"]
    with pytest.raises(ValidationError):
        CuratedScenario.model_validate(payload)


def test_unknown_challenge_type_is_rejected() -> None:
    payload = load_bank()[0].model_dump()
    payload["challenge_type"] = "vibes"
    with pytest.raises(ValidationError):
        CuratedScenario.model_validate(payload)


def test_wrong_count_fails_loudly(tmp_path) -> None:
    short = tmp_path / "short.jsonl"
    rows = [r.model_dump(mode="json") for r in load_bank()[:5]]
    short.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(BankError, match="expected 100 scenarios"):
        load_bank(str(short))


def test_malformed_line_fails_loudly(tmp_path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"scenario_id": "SC-001"\n', encoding="utf-8")
    with pytest.raises(BankError, match="not valid JSON"):
        load_bank(str(bad))


def test_by_id_round_trips() -> None:
    record = by_id("SC-001")
    assert record.scenario_id == "SC-001"
    with pytest.raises(KeyError):
        by_id("SC-999")
