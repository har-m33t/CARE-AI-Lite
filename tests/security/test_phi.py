"""PHI detection and the refuse-to-persist rule."""

from __future__ import annotations

import pytest

from carelite.safety import phi
from tests.security import corpus

pytestmark = pytest.mark.security


@pytest.mark.parametrize("case", corpus.PHI_SAMPLES, ids=[k for k, _ in corpus.PHI_SAMPLES])
def test_every_phi_sample_is_detected(case: tuple[str, str]) -> None:
    key, text = case
    assert phi.contains_phi(text), f"UNDETECTED PHI [{key}]: {text!r}"


def test_detection_rate_on_the_corpus() -> None:
    missed = [key for key, text in corpus.PHI_SAMPLES if not phi.contains_phi(text)]
    assert missed == [], f"missed {len(missed)}/{len(corpus.PHI_SAMPLES)}: {missed}"


@pytest.mark.parametrize("case", corpus.PHI_SAMPLES, ids=[k for k, _ in corpus.PHI_SAMPLES])
def test_phi_turns_are_blocked_and_marked_do_not_persist(case: tuple[str, str]) -> None:
    _, text = case
    verdict = phi.screen(text)
    assert verdict.allowed is False
    assert verdict.phi_detected is True
    assert phi.DO_NOT_PERSIST in verdict.flags
    assert phi.may_persist(verdict) is False


@pytest.mark.parametrize("case", corpus.PHI_SAMPLES, ids=[k for k, _ in corpus.PHI_SAMPLES])
def test_redaction_removes_the_identifier(case: tuple[str, str]) -> None:
    """A redacted turn must be safe to show, even though it is never persisted."""
    _, text = case
    redacted = phi.redact(text)
    assert "[REDACTED:" in redacted
    for hit in phi.find_phi(text):
        assert hit.span not in redacted, f"{hit.kind} span survived redaction: {hit.span!r}"


def test_redacted_text_carries_no_residual_phi() -> None:
    for _, text in corpus.PHI_SAMPLES:
        assert not phi.contains_phi(phi.redact(text)), f"redaction was incomplete for {text!r}"


@pytest.mark.parametrize("text", corpus.PHI_NEGATIVES)
def test_identifier_free_talk_is_not_flagged(text: str) -> None:
    hits = phi.find_phi(text)
    assert not hits, f"FALSE POSITIVE on {text!r}: {[(h.kind, h.span) for h in hits]}"


def test_false_positive_rate_on_matched_negatives() -> None:
    flagged = [t for t in corpus.PHI_NEGATIVES if phi.contains_phi(t)]
    fp_rate = len(flagged) / len(corpus.PHI_NEGATIVES)
    assert fp_rate == 0.0, f"FP rate {fp_rate:.3f} on {flagged}"


def test_may_persist_is_true_for_a_clean_turn() -> None:
    verdict = phi.screen("I don't understand why I need another test.")
    assert verdict.allowed is True
    assert phi.may_persist(verdict) is True


def test_every_identifier_kind_is_exercised_by_the_corpus() -> None:
    covered = {hit.kind for _, text in corpus.PHI_SAMPLES for hit in phi.find_phi(text)}
    declared = {det.kind for det in phi.DETECTORS}
    assert declared - covered == set(), f"untested identifier kinds: {declared - covered}"
