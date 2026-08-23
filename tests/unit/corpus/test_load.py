"""Unit tests for carelite.corpus.load.

The SQL-parameter-shaping helpers are plain unit tests (no connection, part
of `make check`). Anything that opens an actual Postgres connection is
`@pytest.mark.db` — excluded from `make check` by design, and not runnable
in this environment yet (Postgres is not installed). Written anyway so the
round-trip is verified the moment a database exists.
"""

from __future__ import annotations

import pytest

from carelite.corpus.chunk import chunk_text
from carelite.corpus.load import (
    chunk_params,
    paper_params,
    upsert_corpus,
    upsert_papers,
)
from carelite.types import Chunk, EvidenceTier, Paper


def _paper(paper_id: str = "paper-1") -> Paper:
    return Paper(
        paper_id=paper_id,
        doi="10.1/xyz",
        apa_citation="Smith, J. (2020). A study. Journal of Testing.",
        year=2020,
        design="RCT",
        evidence_tier=EvidenceTier.STRONG,
        pdf_path="/data/pdfs/2020_10-1-xyz.pdf",
    )


def test_paper_params_maps_every_field_including_enum_value():
    params = paper_params(_paper())
    assert params["paper_id"] == "paper-1"
    assert params["doi"] == "10.1/xyz"
    assert params["evidence_tier"] == "strong"  # enum -> raw string for the CHECK constraint
    assert params["pdf_path"] == "/data/pdfs/2020_10-1-xyz.pdf"


def test_chunk_params_recovers_ordinal_from_stable_chunk_id():
    chunk = Chunk(chunk_id="paper-1::0007", paper_id="paper-1", text="body text")
    params = chunk_params(chunk)
    assert params["ordinal"] == 7
    assert params["chunk_id"] == "paper-1::0007"
    assert params["contextual_prefix"] is None


def test_chunk_params_falls_back_to_zero_for_unparseable_chunk_id():
    chunk = Chunk(chunk_id="not-a-standard-id", paper_id="paper-1", text="body text")
    params = chunk_params(chunk)
    assert params["ordinal"] == 0


def test_chunk_params_explicit_ordinal_overrides_parsed_one():
    chunk = Chunk(chunk_id="paper-1::0007", paper_id="paper-1", text="body text")
    params = chunk_params(chunk, ordinal=99)
    assert params["ordinal"] == 99


def test_chunks_from_chunk_text_upsert_with_monotonic_ordinals():
    chunks = chunk_text(
        "paper-1", "Sentence one is here. Sentence two is here. Sentence three is here."
    )
    ordinals = [chunk_params(c)["ordinal"] for c in chunks]
    assert ordinals == list(range(len(chunks)))


@pytest.mark.db
def test_upsert_papers_and_chunks_round_trip():
    from carelite.db import apply_schema, fetch_all, fetch_one

    apply_schema()
    paper = _paper("paper-roundtrip")
    chunks = chunk_text(paper.paper_id, "First sentence here. Second sentence here. Third one too.")

    n_papers, n_chunks = upsert_corpus([paper], chunks)
    assert n_papers == 1
    assert n_chunks == len(chunks)

    row = fetch_one("SELECT * FROM paper WHERE paper_id = %s", (paper.paper_id,))
    assert row is not None
    assert row["evidence_tier"] == "strong"

    stored_chunks = fetch_all(
        "SELECT * FROM chunk WHERE paper_id = %s ORDER BY ordinal", (paper.paper_id,)
    )
    assert [r["ordinal"] for r in stored_chunks] == list(range(len(chunks)))


@pytest.mark.db
def test_upsert_is_idempotent():
    from carelite.db import apply_schema, fetch_all

    apply_schema()
    paper = _paper("paper-idempotent")
    chunks = chunk_text(paper.paper_id, "Alpha sentence here. Beta sentence here.")

    upsert_corpus([paper], chunks)
    upsert_corpus([paper], chunks)  # re-run should not duplicate rows

    stored_chunks = fetch_all("SELECT chunk_id FROM chunk WHERE paper_id = %s", (paper.paper_id,))
    assert len(stored_chunks) == len(chunks)


@pytest.mark.db
def test_upsert_paper_updates_fields_on_conflict():
    from carelite.db import apply_schema, fetch_one

    apply_schema()
    paper = _paper("paper-update")
    upsert_papers([paper])

    updated = paper.model_copy(update={"apa_citation": "Updated citation text."})
    upsert_papers([updated])

    row = fetch_one("SELECT apa_citation FROM paper WHERE paper_id = %s", (paper.paper_id,))
    assert row["apa_citation"] == "Updated citation text."
