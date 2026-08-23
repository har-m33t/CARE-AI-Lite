"""carelite.index — dense (pgvector) and lexical (Postgres FTS) indexing.

Owned by the carelite-index lane. Builds and queries the two indexes that
`carelite-retrieval` fuses; it does no ranking or fusion of its own.

    embed.py   — Ollama embedding client (query/document asymmetric prefixes,
                 batching, retry, content-hash cache)
    build.py   — resumable, digest-tracked embedding of `chunk` and `kb_entry`
    fts.py     — query-side Postgres full-text search
    probes.py  — the 10-probe wave-2 retrieval-quality gate
"""

from __future__ import annotations
