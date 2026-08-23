"""carelite.index.build — embed `chunk` and `kb_entry` rows into Postgres.

Resumable and digest-tracked. Two things make that a real requirement here,
not a nicety:

1. Inference is hardware-serialized — one Ollama daemon, 24GB unified memory,
   shared with `carelite-kb`'s extraction lane. An interrupted build (account
   limits, a laptop sleep, another lane needing the daemon) must cost nothing
   to resume.
2. The corpus is not static while this runs: `carelite-corpus` is fixing junk
   chunks and will reload, which can *change chunk_ids* — `chunk_id` is
   `{paper_id}::{ordinal:04d}` (see `carelite.corpus.chunk`), so deleting one
   junk chunk mid-paper renumbers every later chunk in that paper. A resumable
   design keyed on "does this chunk_id currently have a current embedding"
   handles that correctly by construction: renumbered chunks look like new
   rows (get embedded), orphaned old chunk_ids are swept from the state table
   (see `_prune_orphans`), nothing needs to know a reload happened.

**Digest tracking.** `chunk.embedding` / `kb_entry.embedding` are frozen
`vector(1024)` columns in `schema.sql` with no room for a digest or content
hash alongside them, and `schema.sql` may not be edited by this lane. So
resumability state lives in a side table this module owns and creates itself,
`index_embedding_state` — outside the frozen schema, inside `carelite/index/`.
A row's embedding is considered current only if the state table's
`(model_digest, content_hash)` for that ref_id both match what would be
computed right now; the `content_hash` half is what catches an unchanged
model but changed text (a corpus reload that edits a chunk's wording without
changing its id), and the `model_digest` half is what catches a model swap.
Either mismatch means "not current" means re-embed, per `_needs_embedding`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from carelite.db.connection import connect, transaction
from carelite.index.embed import EmbedDimensionError, OllamaEmbedder, hash_text
from carelite.types import Chunk

__all__ = [
    "BuildStats",
    "build_all",
    "build_chunks",
    "build_kb_entries",
    "ensure_state_table",
    "kb_entry_embedding_text",
    "main",
]

# `ref_id` alone isn't unique across the two indexed tables (a chunk_id and an
# entry_id could theoretically collide), hence the composite (ref_id, kind) key.
_STATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS index_embedding_state (
    ref_id       TEXT NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('chunk', 'kb_entry')),
    model_tag    TEXT NOT NULL,
    model_digest TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ref_id, kind)
)
"""

_UPSERT_STATE_SQL = """
INSERT INTO index_embedding_state (ref_id, kind, model_tag, model_digest, content_hash, embedded_at)
VALUES (%(ref_id)s, %(kind)s, %(model_tag)s, %(model_digest)s, %(content_hash)s, now())
ON CONFLICT (ref_id, kind) DO UPDATE SET
    model_tag = EXCLUDED.model_tag,
    model_digest = EXCLUDED.model_digest,
    content_hash = EXCLUDED.content_hash,
    embedded_at = EXCLUDED.embedded_at
"""

_CHUNK_SELECT_SQL = """
SELECT c.chunk_id, c.paper_id, c.text, c.contextual_prefix,
       c.embedding IS NOT NULL AS has_embedding,
       s.model_digest AS state_digest, s.content_hash AS state_hash
FROM chunk c
LEFT JOIN index_embedding_state s ON s.ref_id = c.chunk_id AND s.kind = 'chunk'
WHERE %(only_ids)s::text[] IS NULL OR c.chunk_id = ANY(%(only_ids)s::text[])
ORDER BY c.paper_id, c.ordinal
"""

_KB_SELECT_SQL = """
SELECT e.entry_id, e.finding, e.practical_takeaway, e.example_behavior,
       e.embedding IS NOT NULL AS has_embedding,
       s.model_digest AS state_digest, s.content_hash AS state_hash
FROM kb_entry e
LEFT JOIN index_embedding_state s ON s.ref_id = e.entry_id AND s.kind = 'kb_entry'
WHERE %(only_ids)s::text[] IS NULL OR e.entry_id = ANY(%(only_ids)s::text[])
ORDER BY e.entry_id
"""


def ensure_state_table() -> None:
    """Idempotent, like `carelite.db.connection.apply_schema` for the frozen
    tables. Safe to call on every run."""
    with connect(autocommit=True) as conn:
        conn.execute(_STATE_TABLE_SQL)


def kb_entry_embedding_text(row: dict[str, Any]) -> str:
    """What gets embedded for a kb_entry: the same three fields, in the same
    order, that the frozen `kb_entry.tsv` generated column concatenates
    (`finding || ' ' || practical_takeaway || ' ' || example_behavior`) —
    so the dense and lexical representations of an entry agree on what text
    they're indexing, even though they're built by two different lanes'
    code."""
    return f"{row['finding']} {row['practical_takeaway']} {row['example_behavior']}"


def _needs_embedding(
    *,
    has_embedding: bool,
    state_digest: str | None,
    state_hash: str | None,
    current_digest: str,
    current_hash: str,
) -> bool:
    if not has_embedding:
        return True
    if state_digest is None or state_hash is None:
        # An embedding exists but predates this state table (or the state
        # row was pruned/never written) — no provenance to trust, re-embed.
        return True
    return state_digest != current_digest or state_hash != current_hash


def _batched(seq: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def _prune_orphans(kind: str, table: str, id_column: str) -> int:
    """Delete state rows whose ref_id no longer exists in `table`. This is
    what keeps a chunk-id renumbering (see module docstring) from leaving
    stale rows behind forever — harmless if left, but they'd never be
    cleaned up otherwise since nothing else ever reads them by surprise."""
    sql = f"""
        DELETE FROM index_embedding_state
        WHERE kind = %(kind)s
          AND ref_id NOT IN (SELECT {id_column} FROM {table})
    """
    with transaction() as conn:
        cur = conn.execute(sql, {"kind": kind})
        return cur.rowcount or 0


@dataclass
class BuildStats:
    kind: str
    total: int = 0
    skipped_current: int = 0
    embedded: int = 0
    errors: int = 0
    error_messages: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.kind}: {self.total} total, {self.embedded} embedded, "
            f"{self.skipped_current} already current, {self.errors} errors"
        )


def build_chunks(
    embedder: OllamaEmbedder,
    *,
    batch_size: int = 16,
    limit: int | None = None,
    only_ref_ids: Sequence[str] | None = None,
) -> BuildStats:
    """`only_ref_ids` restricts both the read and every write to that exact
    set of `chunk_id`s — a targeted rebuild of specific rows rather than the
    whole table. This exists for two reasons: it is a legitimate operational
    need (re-embed just the chunks a corpus reload touched, without a full
    pass), and it is what makes this function safe to exercise against a
    *live, shared* database in a test — without it, a test embedder with a
    throwaway digest would make `_needs_embedding` see every real corpus row
    as stale and overwrite production embeddings with test vectors. (That is
    exactly what happened once while writing this lane's test suite; see
    `tests/unit/index/test_build.py`'s fixtures for how it's now scoped.)
    Orphan pruning is skipped when scoped, since a targeted call has no
    business doing table-wide cleanup as a side effect.
    """
    ensure_state_table()
    pruned = 0 if only_ref_ids is not None else _prune_orphans("chunk", "chunk", "chunk_id")
    stats = BuildStats(kind="chunk")

    with connect() as conn:
        rows = conn.execute(
            _CHUNK_SELECT_SQL, {"only_ids": list(only_ref_ids) if only_ref_ids else None}
        ).fetchall()
    stats.total = len(rows)
    if pruned:
        stats.error_messages.append(f"pruned {pruned} orphaned chunk state row(s)")

    digest = embedder.digest
    pending: list[tuple[str, str, str]] = []  # (chunk_id, text_to_embed, content_hash)
    for r in rows:
        chunk = Chunk(
            chunk_id=r["chunk_id"],
            paper_id=r["paper_id"],
            text=r["text"],
            contextual_prefix=r["contextual_prefix"],
        )
        text_to_embed = chunk.embedding_text
        content_hash = hash_text(text_to_embed)
        if _needs_embedding(
            has_embedding=r["has_embedding"],
            state_digest=r["state_digest"],
            state_hash=r["state_hash"],
            current_digest=digest,
            current_hash=content_hash,
        ):
            pending.append((r["chunk_id"], text_to_embed, content_hash))
        else:
            stats.skipped_current += 1

    if limit is not None:
        pending = pending[:limit]

    for batch in _batched(pending, batch_size):
        _embed_and_store_batch(
            embedder, batch, kind="chunk", table="chunk", id_column="chunk_id", stats=stats
        )

    return stats


def build_kb_entries(
    embedder: OllamaEmbedder,
    *,
    batch_size: int = 16,
    limit: int | None = None,
    only_ref_ids: Sequence[str] | None = None,
) -> BuildStats:
    """Handles an empty `kb_entry` table gracefully: `kb_entry` is empty until
    `carelite-kb` finishes its extraction pass. No rows means no work and,
    critically, zero calls to Ollama — a good citizen on a shared daemon.
    See `build_chunks` for what `only_ref_ids` is for and why it matters."""
    ensure_state_table()
    pruned = 0 if only_ref_ids is not None else _prune_orphans("kb_entry", "kb_entry", "entry_id")
    stats = BuildStats(kind="kb_entry")

    with connect() as conn:
        rows = conn.execute(
            _KB_SELECT_SQL, {"only_ids": list(only_ref_ids) if only_ref_ids else None}
        ).fetchall()
    stats.total = len(rows)
    if pruned:
        stats.error_messages.append(f"pruned {pruned} orphaned kb_entry state row(s)")

    if not rows:
        return stats  # nothing to do; no digest resolution, no network call

    digest = embedder.digest
    pending: list[tuple[str, str, str]] = []
    for r in rows:
        text_to_embed = kb_entry_embedding_text(r)
        content_hash = hash_text(text_to_embed)
        if _needs_embedding(
            has_embedding=r["has_embedding"],
            state_digest=r["state_digest"],
            state_hash=r["state_hash"],
            current_digest=digest,
            current_hash=content_hash,
        ):
            pending.append((r["entry_id"], text_to_embed, content_hash))
        else:
            stats.skipped_current += 1

    if limit is not None:
        pending = pending[:limit]

    for batch in _batched(pending, batch_size):
        _embed_and_store_batch(
            embedder, batch, kind="kb_entry", table="kb_entry", id_column="entry_id", stats=stats
        )

    return stats


def _embed_and_store_batch(
    embedder: OllamaEmbedder,
    batch: list[tuple[str, str, str]],
    *,
    kind: str,
    table: str,
    id_column: str,
    stats: BuildStats,
) -> None:
    """One batch, one transaction. On failure the batch is skipped (not
    retried in-process beyond what `OllamaEmbedder`'s own tenacity retry
    already does) so one bad batch can't stall the whole run — a re-run of
    `build_chunks`/`build_kb_entries` will simply see those rows as still
    pending and try again, which is the resumability contract doing its job.
    """
    ids = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    hashes = [b[2] for b in batch]
    try:
        vectors = embedder.embed_documents(texts)
    except EmbedDimensionError:
        raise  # never swallow a dimension mismatch — see module/embed.py docstrings
    except Exception as exc:  # network hiccup, model unloaded, etc.
        stats.errors += len(batch)
        stats.error_messages.append(f"batch of {len(batch)} failed: {exc}")
        return

    digest = embedder.digest
    with transaction() as conn:
        for ref_id, vec, content_hash in zip(ids, vectors, hashes, strict=True):
            conn.execute(f"UPDATE {table} SET embedding = %s WHERE {id_column} = %s", [vec, ref_id])
            conn.execute(
                _UPSERT_STATE_SQL,
                {
                    "ref_id": ref_id,
                    "kind": kind,
                    "model_tag": embedder.model_tag,
                    "model_digest": digest,
                    "content_hash": content_hash,
                },
            )
    stats.embedded += len(batch)


def build_all(
    embedder: OllamaEmbedder | None = None, *, batch_size: int = 16, limit: int | None = None
) -> dict[str, BuildStats]:
    """Chunks first, then kb_entries — no ordering dependency between them,
    but chunks are almost always the larger, slower job, so running them
    first surfaces a dimension/connectivity problem before any kb_entry work
    is attempted."""
    embedder = embedder or OllamaEmbedder(batch_size=batch_size)
    embedder.verify_dimension()
    return {
        "chunk": build_chunks(embedder, batch_size=batch_size, limit=limit),
        "kb_entry": build_kb_entries(embedder, batch_size=batch_size, limit=limit),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="cap rows embedded per table")
    args = parser.parse_args()

    embedder = OllamaEmbedder(batch_size=args.batch_size)
    print(f"carelite index build — model {embedder.model_tag}")
    try:
        embedder.verify_dimension()
    except EmbedDimensionError as exc:
        print(f"FATAL: {exc}")
        return 1
    print(f"  digest: {embedder.digest}")

    results = build_all(embedder, batch_size=args.batch_size, limit=args.limit)
    for stats in results.values():
        print(f"  {stats}")
        for msg in stats.error_messages:
            print(f"    note: {msg}")
    embedder.close()

    total_errors = sum(s.errors for s in results.values())
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
