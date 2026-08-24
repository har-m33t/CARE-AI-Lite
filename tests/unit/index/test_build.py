"""Unit tests for carelite.index.build.

The resumability decision (`_needs_embedding`), batching, and text
construction are pure functions and tested here without a connection — part
of `make check`. Tests that touch Postgres are `@pytest.mark.db`; they use a
`FakeEmbedder` (matching `OllamaEmbedder`'s duck-typed surface: `.digest`,
`.embed_documents`) so they exercise real resumability against the live
schema without needing Ollama running.
"""

from __future__ import annotations

import pytest

from carelite.db.connection import connect, transaction
from carelite.index.build import (
    BuildStats,
    _batched,
    _needs_embedding,
    build_chunks,
    build_kb_entries,
    ensure_state_table,
    kb_entry_embedding_text,
)


def test_kb_entry_embedding_text_matches_tsv_field_order():
    """Must mirror the frozen `kb_entry.tsv` generated column
    (`finding || ' ' || practical_takeaway || ' ' || example_behavior`) so
    dense and lexical search index the same text."""
    row = {
        "finding": "Teach-back improves recall.",
        "practical_takeaway": "Ask patients to restate the plan.",
        "example_behavior": "'How would you describe that to a friend?'",
    }
    text = kb_entry_embedding_text(row)
    assert text == (
        "Teach-back improves recall. Ask patients to restate the plan. "
        "'How would you describe that to a friend?'"
    )


@pytest.mark.parametrize(
    "has_embedding,state_digest,state_hash,current_digest,current_hash,expected",
    [
        (False, None, None, "d1", "h1", True),  # never embedded
        (True, None, None, "d1", "h1", True),  # embedded but no provenance row
        (True, "d1", "h1", "d1", "h1", False),  # fully current
        (True, "d0", "h1", "d1", "h1", True),  # model digest changed
        (True, "d1", "h0", "d1", "h1", True),  # text changed (content hash)
        (True, "d0", "h0", "d1", "h1", True),  # both changed
    ],
)
def test_needs_embedding_decision_table(
    has_embedding, state_digest, state_hash, current_digest, current_hash, expected
):
    assert (
        _needs_embedding(
            has_embedding=has_embedding,
            state_digest=state_digest,
            state_hash=state_hash,
            current_digest=current_digest,
            current_hash=current_hash,
        )
        is expected
    )


def test_batched_splits_evenly_and_keeps_remainder():
    assert _batched([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert _batched([], 2) == []
    assert _batched([1], 5) == [[1]]


def test_build_stats_str_is_legible():
    stats = BuildStats(kind="chunk", total=10, embedded=7, skipped_current=2, errors=1)
    s = str(stats)
    assert "chunk" in s
    assert "10 total" in s
    assert "7 embedded" in s
    assert "2 already current" in s
    assert "1 errors" in s


# ---------------------------------------------------------------------------
# @pytest.mark.db — real Postgres, fake embedder (no Ollama needed).
#
# SAFETY: every call below passes `only_ref_ids` scoped to the throwaway
# fixture row(s). Never call `build_chunks`/`build_kb_entries` unscoped in a
# test against this shared, live database — the resumability logic is (by
# design) whole-table, and an unscoped call with a test digest will decide
# every real corpus row is stale and overwrite production embeddings with
# fake vectors. That happened once while writing this suite; recovered by
# re-running `python -m carelite.index.build` against the real model, but
# it should not be possible to repeat by accident, hence the scoping here.
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Duck-types the slice of `OllamaEmbedder` that `build.py` calls:
    `.digest`, `.model_tag`, `.embed_documents`. Deterministic vectors keyed
    off text length so distinct inputs get distinct (fake) embeddings."""

    def __init__(self, digest: str = "fake-digest-1", dim: int = 1024):
        self.digest = digest
        self.model_tag = "fake-model"
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t) % 7)] * self.dim for t in texts]


@pytest.fixture
def _temp_chunk():
    """One throwaway paper + chunk, cleaned up (state row included) after
    the test. Uses a paper_id/chunk_id unlikely to collide with the real
    corpus, and every `build_chunks` call in these tests passes
    `only_ref_ids=[chunk_id]` so nothing outside this fixture is ever read
    or written."""
    ensure_state_table()
    paper_id = "test-index-build-paper"
    chunk_id = f"{paper_id}::0000"
    with transaction() as conn:
        conn.execute(
            "INSERT INTO paper (paper_id, apa_citation, evidence_tier) "
            "VALUES (%s, %s, 'strong') ON CONFLICT (paper_id) DO NOTHING",
            [paper_id, "Test, T. (2026). A fixture paper."],
        )
        conn.execute(
            "INSERT INTO chunk (chunk_id, paper_id, ordinal, text) "
            "VALUES (%s, %s, 0, %s) "
            "ON CONFLICT (chunk_id) DO UPDATE SET text = EXCLUDED.text, embedding = NULL",
            [chunk_id, paper_id, "Fixture chunk text for carelite-index build tests."],
        )
    yield chunk_id
    with transaction() as conn:
        conn.execute(
            "DELETE FROM index_embedding_state WHERE ref_id = %s AND kind = 'chunk'", [chunk_id]
        )
        conn.execute("DELETE FROM paper WHERE paper_id = %s", [paper_id])


@pytest.mark.db
def test_build_chunks_embeds_a_new_row_and_records_state(_temp_chunk):
    embedder = FakeEmbedder(digest="digest-v1")
    stats = build_chunks(embedder, batch_size=8, only_ref_ids=[_temp_chunk])
    assert stats.total == 1
    assert stats.embedded == 1
    assert embedder.calls  # actually called

    with connect() as conn:
        row = conn.execute(
            "SELECT embedding IS NOT NULL AS has_embedding FROM chunk WHERE chunk_id = %s",
            [_temp_chunk],
        ).fetchone()
        assert row["has_embedding"]
        state = conn.execute(
            "SELECT model_digest, content_hash FROM index_embedding_state "
            "WHERE ref_id = %s AND kind = 'chunk'",
            [_temp_chunk],
        ).fetchone()
        assert state["model_digest"] == "digest-v1"


@pytest.mark.db
def test_build_chunks_is_resumable_skips_current_rows(_temp_chunk):
    embedder = FakeEmbedder(digest="digest-v1")
    build_chunks(embedder, batch_size=8, only_ref_ids=[_temp_chunk])

    # Second run, same digest, unchanged text: nothing new to embed.
    embedder2 = FakeEmbedder(digest="digest-v1")
    stats2 = build_chunks(embedder2, batch_size=8, only_ref_ids=[_temp_chunk])
    assert not embedder2.calls
    assert stats2.skipped_current == 1
    assert stats2.embedded == 0


@pytest.mark.db
def test_build_chunks_reembeds_on_digest_change(_temp_chunk):
    build_chunks(FakeEmbedder(digest="digest-v1"), batch_size=8, only_ref_ids=[_temp_chunk])
    embedder2 = FakeEmbedder(digest="digest-v2")
    stats2 = build_chunks(embedder2, batch_size=8, only_ref_ids=[_temp_chunk])
    assert embedder2.calls  # model changed -> re-embedded
    assert stats2.embedded == 1


@pytest.mark.db
def test_build_chunks_reembeds_on_text_change(_temp_chunk):
    build_chunks(FakeEmbedder(digest="digest-v1"), batch_size=8, only_ref_ids=[_temp_chunk])
    with transaction() as conn:
        conn.execute(
            "UPDATE chunk SET text = %s WHERE chunk_id = %s",
            ["Edited fixture text — content changed, digest did not.", _temp_chunk],
        )
    embedder2 = FakeEmbedder(digest="digest-v1")
    stats2 = build_chunks(embedder2, batch_size=8, only_ref_ids=[_temp_chunk])
    assert embedder2.calls  # text changed under the same digest -> re-embedded
    assert stats2.embedded == 1


@pytest.mark.db
def test_build_kb_entries_handles_empty_scope_without_calling_embedder():
    """Scoped to a ref_id that certainly doesn't exist, so this never touches
    whatever real rows `carelite-kb` may have written by the time this runs."""
    embedder = FakeEmbedder(digest="digest-v1")
    stats = build_kb_entries(
        embedder, batch_size=8, only_ref_ids=["definitely-not-a-real-entry-id"]
    )
    assert stats.total == 0
    assert not embedder.calls


@pytest.fixture
def _temp_kb_entry():
    """One throwaway paper + kb_entry, cleaned up (state row included) after
    the test. Mirrors `_temp_chunk`: every `build_kb_entries` call here passes
    `only_ref_ids=[entry_id]`, so — same rationale as `_temp_chunk`'s docstring
    — nothing outside this fixture is ever read or written. This is the real
    row exercise `test_build_kb_entries_handles_empty_scope_without_calling_
    embedder` above deliberately doesn't attempt (it only proves the empty
    path never calls Ollama); the gap this fills is proving the actual
    embed-and-store path — the one that sat unexercised against a populated
    `kb_entry` table until this incident (see `build.py`'s module docstring)
    — works end to end against real Postgres.
    """
    ensure_state_table()
    paper_id = "test-index-build-kb-paper"
    entry_id = "test-index-build-kb-entry"
    with transaction() as conn:
        conn.execute(
            "INSERT INTO paper (paper_id, apa_citation, evidence_tier) "
            "VALUES (%s, %s, 'strong') ON CONFLICT (paper_id) DO NOTHING",
            [paper_id, "Test, T. (2026). A fixture paper."],
        )
        conn.execute(
            "INSERT INTO kb_entry (entry_id, theme, finding, practical_takeaway, "
            "example_behavior, evidence_tier, action_type, verbatim_span) "
            "VALUES (%s, 'teach_back', %s, %s, %s, 'strong', 'generation', %s) "
            "ON CONFLICT (entry_id) DO UPDATE SET "
            "finding = EXCLUDED.finding, practical_takeaway = EXCLUDED.practical_takeaway, "
            "example_behavior = EXCLUDED.example_behavior, embedding = NULL",
            [
                entry_id,
                "Fixture finding for carelite-index build tests.",
                "Fixture practical takeaway.",
                "Fixture example behavior.",
                "Fixture finding for carelite-index build tests.",
            ],
        )
    yield entry_id
    with transaction() as conn:
        conn.execute(
            "DELETE FROM index_embedding_state WHERE ref_id = %s AND kind = 'kb_entry'",
            [entry_id],
        )
        conn.execute("DELETE FROM kb_entry WHERE entry_id = %s", [entry_id])
        conn.execute("DELETE FROM paper WHERE paper_id = %s", [paper_id])


@pytest.mark.db
def test_build_kb_entries_embeds_a_new_row_and_records_state(_temp_kb_entry):
    embedder = FakeEmbedder(digest="digest-v1")
    stats = build_kb_entries(embedder, batch_size=8, only_ref_ids=[_temp_kb_entry])
    assert stats.total == 1
    assert stats.embedded == 1
    assert embedder.calls  # actually called

    with connect() as conn:
        row = conn.execute(
            "SELECT embedding IS NOT NULL AS has_embedding FROM kb_entry WHERE entry_id = %s",
            [_temp_kb_entry],
        ).fetchone()
        assert row["has_embedding"]
        state = conn.execute(
            "SELECT model_digest, content_hash FROM index_embedding_state "
            "WHERE ref_id = %s AND kind = 'kb_entry'",
            [_temp_kb_entry],
        ).fetchone()
        assert state["model_digest"] == "digest-v1"


@pytest.mark.db
def test_build_kb_entries_is_resumable_skips_current_rows(_temp_kb_entry):
    embedder = FakeEmbedder(digest="digest-v1")
    build_kb_entries(embedder, batch_size=8, only_ref_ids=[_temp_kb_entry])

    # Second run, same digest, unchanged text: nothing new to embed.
    embedder2 = FakeEmbedder(digest="digest-v1")
    stats2 = build_kb_entries(embedder2, batch_size=8, only_ref_ids=[_temp_kb_entry])
    assert not embedder2.calls
    assert stats2.skipped_current == 1
    assert stats2.embedded == 0


@pytest.mark.db
def test_build_kb_entries_reembeds_on_digest_change(_temp_kb_entry):
    build_kb_entries(FakeEmbedder(digest="digest-v1"), batch_size=8, only_ref_ids=[_temp_kb_entry])
    embedder2 = FakeEmbedder(digest="digest-v2")
    stats2 = build_kb_entries(embedder2, batch_size=8, only_ref_ids=[_temp_kb_entry])
    assert embedder2.calls  # model changed -> re-embedded
    assert stats2.embedded == 1


@pytest.mark.db
def test_build_kb_entries_reembeds_on_text_change(_temp_kb_entry):
    build_kb_entries(FakeEmbedder(digest="digest-v1"), batch_size=8, only_ref_ids=[_temp_kb_entry])
    with transaction() as conn:
        conn.execute(
            "UPDATE kb_entry SET finding = %s WHERE entry_id = %s",
            ["Edited fixture finding — content changed, digest did not.", _temp_kb_entry],
        )
    embedder2 = FakeEmbedder(digest="digest-v1")
    stats2 = build_kb_entries(embedder2, batch_size=8, only_ref_ids=[_temp_kb_entry])
    assert embedder2.calls  # text changed under the same digest -> re-embedded
    assert stats2.embedded == 1
