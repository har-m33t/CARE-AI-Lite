"""Unit tests for carelite.corpus.load.

The SQL-parameter-shaping helpers, `canonical_paper_id`, and the
paper_id/doi consistency validation are plain unit tests (no connection,
part of `make check`). Anything that opens an actual Postgres connection is
`@pytest.mark.db` — excluded from `make check` by design.

`_paper()` always derives `paper_id` from `doi` via `canonical_paper_id`,
matching how `carelite.corpus.fetch.manifest_papers()` builds `Paper`
objects in production — so these tests exercise the same invariant the
real pipeline relies on, not an artificial paper_id/doi pairing.
"""

from __future__ import annotations

import pytest

from carelite.corpus.chunk import chunk_text
from carelite.corpus.load import (
    canonical_paper_id,
    chunk_params,
    paper_params,
    replace_corpus,
    replace_paper_chunks,
    update_chunk_prefix,
    upsert_corpus,
    upsert_papers,
)
from carelite.types import Chunk, EvidenceTier, Paper


@pytest.fixture(autouse=True)
def _cleanup_live_db_test_fixtures(request: pytest.FixtureRequest):
    """Every @pytest.mark.db test in this file writes to the shared, live
    Postgres instance — there is no isolated test database yet. Left alone,
    a paper like "10.1/roundtrip-test" sits in `paper`/`chunk` right next to
    the real corpus, and carelite-index embeds whatever is in `chunk`. This
    deletes anything a db-marked test created (every fixture doi here
    contains "test") once that test finishes; a no-op, no connection
    attempted, for the non-db tests that make up most of this file."""
    yield
    if request.node.get_closest_marker("db") is None:
        return
    from carelite.db import connect

    with connect() as conn:
        conn.execute(
            "DELETE FROM chunk WHERE paper_id IN (SELECT paper_id FROM paper WHERE doi LIKE %s)",
            ("%test%",),
        )
        conn.execute("DELETE FROM paper WHERE doi LIKE %s", ("%test%",))
        conn.commit()


def _paper(doi: str = "10.1/xyz", **overrides: object) -> Paper:
    fields: dict[str, object] = {
        "paper_id": canonical_paper_id("ignored-when-doi-present", doi),
        "doi": doi,
        "apa_citation": "Smith, J. (2020). A study. Journal of Testing.",
        "year": 2020,
        "design": "RCT",
        "evidence_tier": EvidenceTier.STRONG,
        "pdf_path": "/data/pdfs/2020_10-1-xyz.pdf",
    }
    fields.update(overrides)
    return Paper(**fields)  # type: ignore[arg-type]


def test_canonical_paper_id_derives_from_doi_when_present():
    assert canonical_paper_id("whatever-id", "10.1370/afm.348") == "10-1370-afm-348"


def test_canonical_paper_id_falls_back_to_given_id_without_a_doi():
    assert canonical_paper_id("manual-paper-1", None) == "manual-paper-1"
    assert canonical_paper_id("manual-paper-1", "") == "manual-paper-1"


def test_paper_params_maps_every_field_including_enum_value():
    params = paper_params(_paper())
    assert params["paper_id"] == "10-1-xyz"
    assert params["doi"] == "10.1/xyz"
    assert params["evidence_tier"] == "strong"  # enum -> raw string for the CHECK constraint
    assert params["pdf_path"] == "/data/pdfs/2020_10-1-xyz.pdf"


def test_paper_params_rejects_paper_id_doi_mismatch():
    """The real-world trigger: a Paper whose paper_id was not derived from
    its doi would collide on `paper_doi_key` (UNIQUE) before ON CONFLICT
    (paper_id) ever saw it. paper_params catches this before it ever
    reaches the database, with a message that says why."""
    mismatched = Paper(
        paper_id="not-the-slug",
        doi="10.1/xyz",
        apa_citation="X",
        evidence_tier=EvidenceTier.EMERGING,
    )
    with pytest.raises(ValueError, match="canonical paper_id"):
        paper_params(mismatched)


def test_paper_params_allows_any_paper_id_when_doi_is_absent():
    no_doi = Paper(
        paper_id="manual-lookup-1",
        doi=None,
        apa_citation="X",
        evidence_tier=EvidenceTier.EMERGING,
    )
    params = paper_params(no_doi)
    assert params["paper_id"] == "manual-lookup-1"


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
def test_duplicate_doi_manifest_case_upserts_to_a_single_row():
    """Models the manifest's `duplicate_of` scenario directly: the real-world
    trigger for the paper_doi_key UniqueViolation is two Paper objects built
    independently for the same doi (e.g. "0030415.pdf" and its byte-identical
    "0030415_1.pdf"). `fetch.manifest_papers()` already filters `duplicate_of`
    rows so only one `Paper` per doi is ever emitted in production, but this
    proves the DB layer itself is safe even if it weren't: two independently
    constructed Papers sharing a doi resolve to the same canonical paper_id
    and upsert onto a single row instead of raising.
    """
    from carelite.db import apply_schema, fetch_all

    apply_schema()
    doi = "10.1370/afm.348-test-dup"
    primary = _paper(doi=doi, apa_citation="Primary citation.")
    duplicate = _paper(doi=doi, apa_citation="Re-resolved citation, supersedes the first.")
    assert primary.paper_id == duplicate.paper_id  # the guarantee under test

    upsert_papers([primary, duplicate])

    rows = fetch_all("SELECT paper_id, apa_citation FROM paper WHERE doi = %s", (doi,))
    assert len(rows) == 1
    assert rows[0]["apa_citation"] == "Re-resolved citation, supersedes the first."


@pytest.mark.db
def test_replace_paper_chunks_removes_stale_rows_not_just_upserts():
    """The bug this guards against: a re-chunk that now produces fewer
    chunks than a previous load (e.g. a chunker fix that drops degenerate
    chunks) must not leave the dropped ones behind. Plain upsert_chunks
    can't do this — it only ever adds/updates by chunk_id."""
    from carelite.db import apply_schema, fetch_all

    apply_schema()
    paper = _paper(doi="10.1/replace-shrink-test")
    big = chunk_text(
        paper.paper_id,
        "Alpha sentence one. Beta sentence two. Gamma sentence three. Delta sentence four.",
        target_tokens=6,
        overlap_tokens=0,
    )
    assert len(big) > 1
    upsert_papers([paper])
    replace_paper_chunks(paper.paper_id, big)

    small = chunk_text(paper.paper_id, "Alpha sentence one.", target_tokens=1000, overlap_tokens=0)
    assert len(small) < len(big)
    replace_paper_chunks(paper.paper_id, small)

    stored = fetch_all("SELECT chunk_id FROM chunk WHERE paper_id = %s", (paper.paper_id,))
    assert len(stored) == len(small)
    assert {r["chunk_id"] for r in stored} == {c.chunk_id for c in small}


@pytest.mark.db
def test_replace_paper_chunks_with_empty_list_clears_all_chunks():
    from carelite.db import apply_schema, fetch_all

    apply_schema()
    paper = _paper(doi="10.1/replace-clear-test")
    chunks = chunk_text(paper.paper_id, "Some sentence here. Another sentence here.")
    upsert_papers([paper])
    replace_paper_chunks(paper.paper_id, chunks)

    replace_paper_chunks(paper.paper_id, [])

    stored = fetch_all("SELECT chunk_id FROM chunk WHERE paper_id = %s", (paper.paper_id,))
    assert stored == []


@pytest.mark.db
def test_replace_corpus_upserts_papers_and_replaces_each_papers_chunks():
    from carelite.db import apply_schema, fetch_all

    apply_schema()
    paper = _paper(doi="10.1/replace-corpus-test")
    old_chunks = chunk_text(paper.paper_id, "First old sentence. Second old sentence.")
    upsert_papers([paper])
    replace_paper_chunks(paper.paper_id, old_chunks)

    new_chunks = chunk_text(paper.paper_id, "Only one new sentence here.", target_tokens=1000)
    n_papers, n_chunks = replace_corpus([paper], {paper.paper_id: new_chunks})

    assert n_papers == 1
    assert n_chunks == len(new_chunks)
    stored = fetch_all("SELECT chunk_id FROM chunk WHERE paper_id = %s", (paper.paper_id,))
    assert {r["chunk_id"] for r in stored} == {c.chunk_id for c in new_chunks}


@pytest.mark.db
def test_update_chunk_prefix_writes_only_the_prefix():
    from carelite.db import apply_schema, fetch_one

    apply_schema()
    paper = _paper(doi="10.1/update-prefix-test")
    chunks = chunk_text(paper.paper_id, "First sentence here. Second sentence here.")
    upsert_papers([paper])
    replace_paper_chunks(paper.paper_id, chunks)
    target = chunks[0]

    update_chunk_prefix(target.chunk_id, "Situates this chunk within the paper's introduction.")

    row = fetch_one(
        "SELECT ordinal, text, contextual_prefix FROM chunk WHERE chunk_id = %s", (target.chunk_id,)
    )
    assert row is not None
    assert row["contextual_prefix"] == "Situates this chunk within the paper's introduction."
    assert row["text"] == target.text  # untouched
    assert row["ordinal"] == 0  # untouched


@pytest.mark.db
def test_upsert_papers_and_chunks_round_trip():
    from carelite.db import apply_schema, fetch_all, fetch_one

    apply_schema()
    paper = _paper(doi="10.1/roundtrip-test")
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
    paper = _paper(doi="10.1/idempotent-test")
    chunks = chunk_text(paper.paper_id, "Alpha sentence here. Beta sentence here.")

    upsert_corpus([paper], chunks)
    upsert_corpus([paper], chunks)  # re-run should not duplicate rows

    stored_chunks = fetch_all("SELECT chunk_id FROM chunk WHERE paper_id = %s", (paper.paper_id,))
    assert len(stored_chunks) == len(chunks)


@pytest.mark.db
def test_upsert_paper_updates_fields_on_conflict():
    from carelite.db import apply_schema, fetch_one

    apply_schema()
    paper = _paper(doi="10.1/update-test")
    upsert_papers([paper])

    updated = paper.model_copy(update={"apa_citation": "Updated citation text."})
    upsert_papers([updated])

    row = fetch_one("SELECT apa_citation FROM paper WHERE paper_id = %s", (paper.paper_id,))
    assert row["apa_citation"] == "Updated citation text."
