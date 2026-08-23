"""carelite.corpus.load — upsert papers and chunks into Postgres.

Every test in here that touches a real connection is `@pytest.mark.db`
(excluded from `make check`; Postgres is not installed yet in this
environment). The upsert SQL itself is exercised by `tests/unit/corpus/test_load.py`
without a connection, via a fake cursor, so the query shape is still covered
by `make check`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from carelite.corpus.chunk import ordinal_of
from carelite.db.connection import transaction
from carelite.types import Chunk, Paper

_UPSERT_PAPER_SQL = """
INSERT INTO paper (paper_id, doi, apa_citation, year, design, evidence_tier, pdf_path)
VALUES (%(paper_id)s, %(doi)s, %(apa_citation)s, %(year)s, %(design)s, %(evidence_tier)s, %(pdf_path)s)
ON CONFLICT (paper_id) DO UPDATE SET
    doi = EXCLUDED.doi,
    apa_citation = EXCLUDED.apa_citation,
    year = EXCLUDED.year,
    design = EXCLUDED.design,
    evidence_tier = EXCLUDED.evidence_tier,
    pdf_path = EXCLUDED.pdf_path
"""

_UPSERT_CHUNK_SQL = """
INSERT INTO chunk (chunk_id, paper_id, ordinal, text, contextual_prefix)
VALUES (%(chunk_id)s, %(paper_id)s, %(ordinal)s, %(text)s, %(contextual_prefix)s)
ON CONFLICT (chunk_id) DO UPDATE SET
    ordinal = EXCLUDED.ordinal,
    text = EXCLUDED.text,
    contextual_prefix = EXCLUDED.contextual_prefix
"""


def paper_params(paper: Paper) -> dict[str, object]:
    return {
        "paper_id": paper.paper_id,
        "doi": paper.doi,
        "apa_citation": paper.apa_citation,
        "year": paper.year,
        "design": paper.design,
        "evidence_tier": paper.evidence_tier.value,
        "pdf_path": paper.pdf_path,
    }


def chunk_params(chunk: Chunk, ordinal: int | None = None) -> dict[str, object]:
    if ordinal is None:
        try:
            ordinal = ordinal_of(chunk)
        except ValueError:
            ordinal = 0
    return {
        "chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "ordinal": ordinal,
        "text": chunk.text,
        "contextual_prefix": chunk.contextual_prefix,
    }


def upsert_papers(papers: Iterable[Paper]) -> int:
    """Idempotent upsert into `paper`, keyed on `paper_id`. Returns rows written."""
    papers = list(papers)
    with transaction() as conn:
        for paper in papers:
            conn.execute(_UPSERT_PAPER_SQL, paper_params(paper))
    return len(papers)


def upsert_chunks(chunks: Sequence[Chunk]) -> int:
    """Idempotent upsert into `chunk`, keyed on `chunk_id`.

    Ordinal is recovered from `chunk_id` (see `carelite.corpus.chunk.ordinal_of`)
    so callers don't have to pass per-paper enumeration order explicitly, and
    reloading a partial/reordered subset still lands the correct ordinal.
    """
    with transaction() as conn:
        for chunk in chunks:
            conn.execute(_UPSERT_CHUNK_SQL, chunk_params(chunk))
    return len(chunks)


def upsert_corpus(papers: Iterable[Paper], chunks: Sequence[Chunk]) -> tuple[int, int]:
    """Papers first (chunk.paper_id has an FK to paper.paper_id with ON DELETE CASCADE)."""
    n_papers = upsert_papers(papers)
    n_chunks = upsert_chunks(chunks)
    return n_papers, n_chunks
