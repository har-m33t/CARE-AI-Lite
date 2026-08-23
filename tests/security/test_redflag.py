"""Red-flag detection: the 100% recall gate.

This is the file the wave-3 gate reads. Recall is asserted at exactly 1.0 and
the false-positive rate is asserted at its *measured* value rather than at a
target, so that a change which trades recall for precision fails loudly instead
of passing quietly.
"""

from __future__ import annotations

import pytest

from carelite.safety import redflag
from tests.security import corpus

pytestmark = pytest.mark.security


@pytest.mark.parametrize("case", corpus.RED_FLAGS, ids=[k for k, _ in corpus.RED_FLAGS])
def test_every_red_flag_is_caught(case: tuple[str, str]) -> None:
    """Parametrised so a regression names the utterance it stopped catching."""
    _, text = case
    assert redflag.is_red_flag(text), f"MISSED RED FLAG: {text!r}"


def test_recall_is_one_hundred_percent() -> None:
    missed = [key for key, text in corpus.RED_FLAGS if not redflag.is_red_flag(text)]
    recall = 1 - len(missed) / len(corpus.RED_FLAGS)
    assert recall == 1.0, f"recall {recall:.3f}, missed {missed}"


def test_red_flag_turns_are_not_allowed_to_proceed() -> None:
    for _, text in corpus.RED_FLAGS:
        verdict = redflag.screen(text)
        assert verdict.allowed is False
        assert verdict.red_flag is True
        assert verdict.flags
        assert verdict.reason and "RED FLAG" in verdict.reason


def test_red_flag_verdicts_do_not_redact() -> None:
    """A red flag is a situation, not a string. Redaction would be nonsense here."""
    for _, text in corpus.RED_FLAGS:
        assert redflag.screen(text).redacted_text is None


def test_escalation_message_refuses_to_give_clinical_advice() -> None:
    hits = redflag.find_red_flags("I've been having chest pain since this morning.")
    message = redflag.escalation_message(hits)
    assert "not a triage tool" in message
    assert "makes no clinical recommendation" in message


@pytest.mark.parametrize("text", corpus.RED_FLAG_NEGATIVES)
def test_ordinary_turns_are_not_flagged(text: str) -> None:
    hits = redflag.find_red_flags(text)
    assert not hits, f"FALSE POSITIVE on {text!r}: {[h.category for h in hits]}"


def test_false_positive_rate_on_matched_negatives() -> None:
    """Measured, not targeted.

    The 60 negatives are drawn from the same register as the positives —
    frustration, chronic illness, confusion — so this number means something.
    """
    flagged = [t for t in corpus.RED_FLAG_NEGATIVES if redflag.is_red_flag(t)]
    fp_rate = len(flagged) / len(corpus.RED_FLAG_NEGATIVES)
    assert fp_rate == 0.0, f"FP rate {fp_rate:.3f} on {flagged}"


def test_hard_negatives_are_flagged_and_that_cost_is_recorded() -> None:
    """The honest part of the report.

    Every hard negative names a red-flag concept while denying it or attributing
    it to someone else in the past. A recall-first detector flags 10 of the 12,
    and this test pins that number so the trade cannot drift silently in either
    direction. If you change the detector and this count moves, decide
    deliberately which way you want it and update the literal.
    """
    flagged = [t for t in corpus.RED_FLAG_HARD_NEGATIVES if redflag.is_red_flag(t)]
    assert len(flagged) == 10, f"hard-negative flag count changed: {flagged}"


def test_negation_awareness_never_costs_recall() -> None:
    """The opt-in flag must be a precision knob, never a recall one."""
    missed = [
        key for key, text in corpus.RED_FLAGS if not redflag.is_red_flag(text, negation_aware=True)
    ]
    assert missed == [], f"negation_aware=True lost red flags: {missed}"

    lenient = sum(
        1 for t in corpus.RED_FLAG_HARD_NEGATIVES if redflag.is_red_flag(t, negation_aware=True)
    )
    strict = sum(1 for t in corpus.RED_FLAG_HARD_NEGATIVES if redflag.is_red_flag(t))
    assert lenient <= strict


def test_obfuscated_red_flags_still_catch() -> None:
    """Spacing, leetspeak and separator injection must not evade detection."""
    for text in (
        "i want to k i l l m y s e l f",
        "I want to k1ll mys3lf",
        "ch3st p@in for two hours",
        "c.h.e.s.t p-a-i-n",
        "I want to kill\u200bmyself",
    ):
        assert redflag.is_red_flag(text), f"evaded detection: {text!r}"


def test_every_category_is_exercised_by_the_corpus() -> None:
    """No dead category: each one must be triggered by at least one positive."""
    triggered = {
        hit.category for _, text in corpus.RED_FLAGS for hit in redflag.find_red_flags(text)
    }
    declared = {cat.name for cat in redflag.CATEGORIES}
    assert declared - triggered == set(), f"untested categories: {declared - triggered}"
