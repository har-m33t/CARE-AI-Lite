"""carelite.corpus.load — upsert papers and chunks into Postgres.

Every test in here that touches a real connection is `@pytest.mark.db`
(excluded from `make check`; Postgres is not installed yet in this
environment). The upsert SQL itself is exercised by `tests/unit/corpus/test_load.py`
without a connection, via a fake cursor, so the query shape is still covered
by `make check`.

`paper.doi` carries its own UNIQUE constraint in the frozen schema
(`paper_doi_key`) alongside `paper_id`'s PRIMARY KEY. Two `Paper` objects
that share a doi but disagree on `paper_id` collide on `paper_doi_key`
before `ON CONFLICT (paper_id)` ever gets a chance to see it, and psycopg
raises a raw `UniqueViolation` instead of a clean upsert. That is a real
risk here: the manifest has 7 `duplicate_of` pairs that legitimately share
a DOI. `canonical_paper_id` is the single source of truth for the rule
`carelite.corpus.fetch.manifest_papers()` already follows (`paper_id =
slug(doi)`), and `paper_params` *validates* incoming `Paper` objects
against it rather than silently rewriting `paper_id` — a silent rewrite
here could desync `paper.paper_id` from whatever `paper_id` a caller
already used to build that paper's `Chunk` objects (chunk.paper_id has an
FK to paper.paper_id), trading one confusing failure for another. Once a
`Paper` is validated as consistent, a doi collision without a matching
paper_id collision is impossible by construction, so the existing
`ON CONFLICT (paper_id)` clause is sufficient and unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from carelite.corpus.chunk import ordinal_of
from carelite.corpus.fetch import slug
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


def canonical_paper_id(paper_id: str, doi: str | None) -> str:
    """The paper_id a `Paper` must carry, given its doi.

    `doi` is the true natural key when present — a paper has exactly one
    doi, and `paper.doi` is UNIQUE in the schema. When `doi` is falsy (the
    manifest's 5 no-DOI rows), there is nothing to derive from, so the
    caller-supplied `paper_id` passes through unchanged.
    """
    return slug(doi) if doi else paper_id


def paper_params(paper: Paper) -> dict[str, object]:
    expected_id = canonical_paper_id(paper.paper_id, paper.doi)
    if paper.doi and paper.paper_id != expected_id:
        raise ValueError(
            f"Paper {paper.paper_id!r} carries doi {paper.doi!r}, whose canonical "
            f"paper_id is {expected_id!r} (see carelite.corpus.load.canonical_paper_id). "
            "paper.doi is UNIQUE in the schema, so two Paper objects that share a doi "
            "but disagree on paper_id can never both be upserted safely — construct "
            "paper_id via canonical_paper_id(paper_id, doi) (or "
            "carelite.corpus.fetch.slug(doi)) before creating the Paper."
        )
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
    """Idempotent upsert into `paper`, keyed on `paper_id`. Returns rows written.

    Safe for two `Paper` objects that share a doi (the manifest's
    `duplicate_of` pairs): both validate to the same canonical `paper_id`
    and land on the same row via `ON CONFLICT (paper_id)`, last write wins.
    """
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


def replace_paper_chunks(paper_id: str, chunks: Sequence[Chunk]) -> int:
    """Replace ALL chunks for one paper: delete existing rows for `paper_id`,
    then insert `chunks` fresh, in one transaction.

    Plain `upsert_chunks` only ever adds or updates rows by `chunk_id` — it
    cannot remove one. That leaves stale rows behind whenever a re-chunk of
    the same paper produces a *smaller* chunk set than a previous load (e.g.
    a chunker bug fix that now drops degenerate chunks it used to emit, or
    an extraction fix that removes junk source content). Delete-then-insert
    guarantees the stored set for `paper_id` exactly matches `chunks`, with
    nothing left over. Passing an empty `chunks` clears every chunk for that
    paper — used when extraction fails and there's nothing usable to load.
    """
    with transaction() as conn:
        conn.execute("DELETE FROM chunk WHERE paper_id = %(paper_id)s", {"paper_id": paper_id})
        for chunk in chunks:
            conn.execute(_UPSERT_CHUNK_SQL, chunk_params(chunk))
    return len(chunks)


def replace_corpus(
    papers: Iterable[Paper], chunks_by_paper: dict[str, Sequence[Chunk]]
) -> tuple[int, int]:
    """Upsert every paper, then *replace* (not upsert) each paper's chunk set.

    Prefer this over `upsert_corpus` whenever a paper's chunk set may have
    shrunk since the last load — the common case for a re-run after a
    chunking or extraction bug fix. Papers not present in `papers` are left
    untouched (including their existing chunks); only papers actually being
    reloaded have their chunk rows replaced.
    """
    papers = list(papers)
    n_papers = upsert_papers(papers)
    n_chunks = 0
    for paper in papers:
        n_chunks += replace_paper_chunks(paper.paper_id, chunks_by_paper.get(paper.paper_id, []))
    return n_papers, n_chunks
