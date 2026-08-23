"""Unit tests for carelite.corpus.chunk — boundary behaviour, ordinal/chunk_id
stability, and dependence on settings.retrieval for the default sizes."""

from __future__ import annotations

import itertools

import pytest

from carelite.corpus.chunk import chunk_text, ordinal_of
from carelite.types import Chunk


def _sentences(n: int) -> str:
    return " ".join(f"Sentence number {i} has five words." for i in range(1, n + 1))


def test_chunk_text_on_empty_string_returns_no_chunks():
    assert chunk_text("paper-1", "") == []
    assert chunk_text("paper-1", "   \n\n  ") == []


def test_chunk_ids_are_stable_and_ordinal_is_monotonic():
    chunks = chunk_text("paper-1", _sentences(30), target_tokens=20, overlap_tokens=5)
    assert len(chunks) > 1
    for expected_ordinal, c in enumerate(chunks):
        assert c.chunk_id == f"paper-1::{expected_ordinal:04d}"
        assert c.paper_id == "paper-1"
        assert ordinal_of(c) == expected_ordinal


def test_chunk_text_is_deterministic_across_calls():
    text = _sentences(25)
    first = chunk_text("paper-1", text, target_tokens=15, overlap_tokens=5)
    second = chunk_text("paper-1", text, target_tokens=15, overlap_tokens=5)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_chunks_never_split_mid_sentence():
    text = _sentences(20)
    chunks = chunk_text("paper-1", text, target_tokens=15, overlap_tokens=4)

    # Every one of the 20 numbered sentences must appear intact (never
    # truncated) in at least one chunk.
    for i in range(1, 21):
        needle = f"Sentence number {i} has five words."
        assert any(needle in c.text for c in chunks), f"sentence {i} was split or dropped"

    # And every chunk is itself a clean concatenation of whole sentences: it
    # ends on a sentence terminator, never mid-word.
    for c in chunks:
        assert c.text.strip().endswith("."), c.text


def test_chunk_overlap_shares_content_between_consecutive_chunks():
    chunks = chunk_text("paper-1", _sentences(30), target_tokens=20, overlap_tokens=8)
    assert len(chunks) > 1
    for a, b in itertools.pairwise(chunks):
        a_sentences = {s.strip() for s in a.text.split(".") if s.strip()}
        b_sentences = {s.strip() for s in b.text.split(".") if s.strip()}
        assert a_sentences & b_sentences, "consecutive chunks should share overlap content"


def test_chunk_force_breaks_at_section_headings():
    text = (
        "Introduction\n\n"
        "Sentence A is here. Sentence B is here.\n\n"
        "Methods\n\n"
        "Sentence C is here. Sentence D is here."
    )
    chunks = chunk_text("paper-1", text, target_tokens=1000, overlap_tokens=0)

    assert len(chunks) == 2
    assert chunks[0].text.startswith("Introduction")
    assert "Sentence A" in chunks[0].text
    assert "Sentence B" in chunks[0].text
    assert chunks[1].text.startswith("Methods")
    assert "Sentence C" in chunks[1].text
    assert "Sentence D" in chunks[1].text


def test_degenerate_all_caps_marker_never_becomes_its_own_chunk():
    """Regression test: chunk.py's ALL-CAPS heading heuristic misreads a
    short parenthetical like "(PDF)" as a section heading (str.isupper()
    ignores punctuation), force-breaking it into an isolated chunk that
    min_chunk_tokens never catches because it's a section boundary, not an
    overflow split. The upstream fix is carelite.corpus.extract excluding
    <supplementary-material> structurally; this proves the chunker itself is
    also safe against whatever slips past that."""
    text = (
        "Introduction\n\n"
        "Real content sentence one is here. Real content sentence two is here.\n\n"
        "(PDF)\n\n"
        "Methods\n\n"
        "We interviewed patients about their care experience in this study."
    )
    chunks = chunk_text("paper-1", text, target_tokens=1000, overlap_tokens=0)

    assert not any(c.text.strip() == "(PDF)" for c in chunks)
    assert not any(len(c.text.split()) < 3 for c in chunks)
    # the real sections around it must survive intact
    assert any(c.text.startswith("Introduction") for c in chunks)
    assert any(c.text.startswith("Methods") for c in chunks)


def test_absolute_min_tokens_drops_short_chunks_not_merges_them():
    """Dropped, not merged: merging a degenerate standalone chunk would mean
    crossing the section boundary that force-broke it in the first place."""
    text = (
        "Introduction\n\n"
        "Real content sentence one is here.\n\n"
        "OK\n\n"
        "Methods\n\n"
        "We interviewed patients about their care today."
    )
    # "OK" (all-caps, 1 token) force-breaks like a heading; with a floor of 3
    # tokens it must be dropped entirely, not glued onto its neighbour.
    chunks = chunk_text(
        "paper-1", text, target_tokens=1000, overlap_tokens=0, absolute_min_tokens=3
    )
    texts = [c.text for c in chunks]
    assert not any(t.strip() == "OK" for t in texts)
    assert any(t.startswith("Introduction") for t in texts)
    assert any(t.startswith("Methods") for t in texts)


def test_absolute_min_tokens_zero_keeps_every_chunk():
    text = (
        "Introduction\n\n"
        "Real content sentence one is here.\n\n"
        "OK\n\n"
        "Methods\n\n"
        "We interviewed patients about their care today."
    )
    chunks = chunk_text(
        "paper-1", text, target_tokens=1000, overlap_tokens=0, absolute_min_tokens=0
    )
    assert any(c.text.strip() == "OK" for c in chunks)


def test_chunks_meet_minimum_size_except_when_corpus_is_tiny():
    chunks = chunk_text("paper-1", _sentences(30), target_tokens=20, overlap_tokens=5)
    min_tokens = 20 // 5
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.split()) >= min_tokens


def test_ordinal_of_raises_on_unrecoverable_chunk_id():
    bogus = Chunk(chunk_id="not-formatted-right", paper_id="paper-1", text="x")
    with pytest.raises(ValueError):
        ordinal_of(bogus)


def test_chunk_text_uses_settings_retrieval_defaults(monkeypatch):
    class FakeRetrieval:
        chunk_target_tokens = 10
        chunk_overlap_tokens = 2

    class FakeSettings:
        retrieval = FakeRetrieval()

    import carelite.corpus.chunk as chunk_mod

    monkeypatch.setattr(chunk_mod, "get_settings", lambda: FakeSettings())
    chunks = chunk_mod.chunk_text("paper-1", _sentences(20))
    assert len(chunks) > 1  # target=10 tokens forces multiple chunks out of 20*6≈120 tokens
