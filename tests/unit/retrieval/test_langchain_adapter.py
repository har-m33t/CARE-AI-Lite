"""The LangChain adapter, and the equivalence measurement that justifies it.

Everything here that does not carry a marker runs offline: the adapter takes
its retrievers and its record map by injection, so the ensemble wiring, the
document canonicalisation and the two comparison statistics are all testable
without Postgres and without an embedding model.

The one test that produces the actual reported number is `@pytest.mark.db`
and `@pytest.mark.inference`, because it must read the real 587 embedded rows
and embed real queries with `bge-m3`. See `EQUIVALENCE_COMMAND` in the module
under test for the command that runs it.
"""

from __future__ import annotations

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from carelite.retrieval.flags import PRESETS, RetrievalFlags, preset
from carelite.retrieval.langchain_adapter import (
    EQUIVALENCE_QUERIES,
    MATCHED_NATIVE_FLAGS,
    CorpusRecord,
    EquivalenceReport,
    LangChainRetrievalAdapter,
    overlap_at_k,
    spearman,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ScriptedRetriever(BaseRetriever):
    """Returns a fixed document list regardless of query. No I/O."""

    documents: list[Document] = Field(default_factory=list)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        return list(self.documents)


def record(ref_id: str, *, kind: str = "chunk", text: str | None = None) -> CorpusRecord:
    return CorpusRecord(
        ref_id=ref_id,
        kind=kind,
        index_text=text or f"guidance passage {ref_id}",
        text=text or f"guidance passage {ref_id}",
        paper_id="p1",
        citation="Author, A. (2020). A paper.",
        theme=None,
        evidence_tier="strong",
    )


def scripted(records: list[CorpusRecord]) -> ScriptedRetriever:
    return ScriptedRetriever(documents=[r.as_document() for r in records])


# ---------------------------------------------------------------------------
# Flag selection
# ---------------------------------------------------------------------------


def test_the_adapter_is_off_by_default() -> None:
    """The native stack produced all 939 existing generations and remains what
    condition C runs. A default that silently swapped it would change the
    study."""
    assert RetrievalFlags().langchain_adapter is False
    for name in ("R0", "R4", "R7", "R8", "R9"):
        assert PRESETS[name].langchain_adapter is False


def test_the_adapter_has_its_own_preset() -> None:
    flags = preset("LCHAIN")
    assert flags.langchain_adapter is True
    assert flags.dense and flags.lexical
    assert not flags.graph and not flags.rerank and not flags.crag


def test_the_flag_honours_the_environment_idiom(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every other switch in `flags.py` is env-overridable; this one must be
    too, or it is a parallel mechanism rather than the existing one."""
    monkeypatch.setenv("CARELITE_RETRIEVAL_LANGCHAIN_ADAPTER", "1")
    assert RetrievalFlags().langchain_adapter is True


def test_importing_the_pipeline_does_not_import_langchain() -> None:
    """Mirrors the lazy-reranker rule: a run that does not select the adapter
    must not pay for importing langchain. Checked in a fresh interpreter,
    because this test session has already imported it."""
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import carelite.retrieval.pipeline; "
            "print([m for m in sys.modules if m.startswith('langchain')])",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "[]", proc.stdout


def test_matched_flags_describe_the_adapter_stage_for_stage() -> None:
    """The comparison is only interpretable if the native side is configured
    to do what the adapter does and nothing more."""
    f = MATCHED_NATIVE_FLAGS
    assert f.dense and f.lexical
    assert not f.router
    assert not f.query_expansion
    assert not f.hyde
    assert not f.graph
    assert not f.rerank
    assert not f.tier_weighting
    assert not f.crag
    assert not f.drop_boilerplate
    assert not f.metadata_filter
    assert f.langchain_adapter is False


# ---------------------------------------------------------------------------
# Records and documents
# ---------------------------------------------------------------------------


def test_a_record_round_trips_through_a_document() -> None:
    r = record("c1")
    doc = r.as_document()
    assert doc.page_content == r.index_text
    assert doc.metadata["ref_id"] == "c1"
    assert doc.metadata["kind"] == "chunk"


def test_an_item_carries_what_the_evidence_panel_needs() -> None:
    item = record("c1").as_item(score=0.5)
    assert item.ref_id == "c1"
    assert item.text == "guidance passage c1"
    assert item.citation
    assert item.paper_id
    assert item.evidence_tier is not None


def test_an_unparseable_theme_does_not_raise() -> None:
    r = CorpusRecord(
        ref_id="k1",
        kind="kb_entry",
        index_text="t",
        text="t",
        paper_id=None,
        citation=None,
        theme="not-a-theme",
        evidence_tier="not-a-tier",
    )
    item = r.as_item(score=0.1)
    assert item.theme is None
    assert item.evidence_tier is None


# ---------------------------------------------------------------------------
# Ensemble wiring
# ---------------------------------------------------------------------------


def test_the_ensemble_fuses_both_legs_by_rank() -> None:
    """A document ranked mid-list by both legs must beat one ranked first by
    a single leg — that is the whole point of RRF, and the property that says
    LangChain's ensemble is doing the fusion rather than one leg winning."""
    records = {r.ref_id: r for r in (record("a"), record("b"), record("c"), record("d"))}
    dense = scripted([records["a"], records["b"], records["c"]])
    lexical = scripted([records["d"], records["b"], records["c"]])
    adapter = LangChainRetrievalAdapter(
        records=records, retrievers=[dense, lexical], rrf_k=60, top_n=4
    )
    ids = [i.ref_id for i in adapter.retrieve("anything")]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c", "d"}


def test_retrieve_respects_top_n() -> None:
    records = {r.ref_id: r for r in (record("a"), record("b"), record("c"))}
    adapter = LangChainRetrievalAdapter(
        records=records, retrievers=[scripted(list(records.values()))], rrf_k=60, top_n=2
    )
    assert len(adapter.retrieve("anything")) == 2


def test_scores_are_monotone_in_rank() -> None:
    """The ensemble does not surface its fused score, so the adapter derives
    one from the final rank. It must at least be order-consistent."""
    records = {r.ref_id: r for r in (record("a"), record("b"), record("c"))}
    adapter = LangChainRetrievalAdapter(
        records=records, retrievers=[scripted(list(records.values()))], rrf_k=60, top_n=3
    )
    scores = [i.score for i in adapter.retrieve("anything")]
    assert scores == sorted(scores, reverse=True)


def test_a_document_with_no_record_is_dropped_not_faked() -> None:
    """Provenance comes from the database row. A hit the adapter cannot
    resolve is discarded rather than emitted with empty citation fields."""
    records = {r.ref_id: r for r in (record("a"),)}
    orphan = Document(page_content="x", metadata={"ref_id": "ghost", "kind": "chunk"})
    retriever = ScriptedRetriever(documents=[orphan, records["a"].as_document()])
    adapter = LangChainRetrievalAdapter(records=records, retrievers=[retriever], rrf_k=60, top_n=4)
    assert [i.ref_id for i in adapter.retrieve("anything")] == ["a"]


def test_the_adapter_produces_a_pipeline_shaped_result() -> None:
    records = {r.ref_id: r for r in (record("a"), record("b"))}
    adapter = LangChainRetrievalAdapter(
        records=records, retrievers=[scripted(list(records.values()))], rrf_k=60, top_n=2
    )
    result = adapter.result("I'm scared this is cancer.")
    assert len(result.trace.retrieved) == 2
    assert result.trace.queries == ["I'm scared this is cancer."]
    assert result.trace.fell_back_to_b is False
    assert result.flags.langchain_adapter is True
    assert any("no CRAG gate" in note for note in result.leg_notes)


# ---------------------------------------------------------------------------
# The two statistics
# ---------------------------------------------------------------------------


def test_overlap_at_k_is_a_fraction_of_k() -> None:
    assert overlap_at_k(["a", "b", "c", "d"], ["a", "b", "x", "y"], k=4) == 0.5
    assert overlap_at_k(["a", "b"], ["a", "b"], k=4) == 1.0
    assert overlap_at_k(["a"], ["z"], k=4) == 0.0


def test_overlap_at_k_ignores_order_within_the_cut() -> None:
    assert overlap_at_k(["a", "b", "c", "d"], ["d", "c", "b", "a"], k=4) == 1.0


def test_overlap_at_k_of_two_empty_lists_is_undefined() -> None:
    """Neither side retrieved anything. That is not agreement."""
    assert overlap_at_k([], [], k=4) is None


def test_spearman_of_identical_orderings_is_one() -> None:
    assert spearman(["a", "b", "c", "d"], ["a", "b", "c", "d"], depth=4) == pytest.approx(1.0)


def test_spearman_of_reversed_orderings_is_minus_one() -> None:
    assert spearman(["a", "b", "c", "d"], ["d", "c", "b", "a"], depth=4) == pytest.approx(-1.0)


def test_spearman_penalises_a_missing_document() -> None:
    """Absent from the other list means rank `depth + 1`, so disjoint lists
    score below identical ones rather than being silently dropped."""
    both = spearman(["a", "b", "c"], ["a", "b", "c"], depth=3)
    partial = spearman(["a", "b", "c"], ["a", "b", "z"], depth=3)
    assert both is not None and partial is not None
    assert partial < both


def test_spearman_is_undefined_without_variance() -> None:
    assert spearman(["a"], ["a"], depth=4) is None


# ---------------------------------------------------------------------------
# The query set and the report
# ---------------------------------------------------------------------------


def test_the_query_set_is_fixed_and_committed() -> None:
    """Reported over a set that could drift, the number means nothing."""
    assert isinstance(EQUIVALENCE_QUERIES, tuple)
    assert len(EQUIVALENCE_QUERIES) >= 10
    assert len(set(EQUIVALENCE_QUERIES)) == len(EQUIVALENCE_QUERIES)


def test_the_query_set_includes_turns_the_corpus_cannot_answer() -> None:
    """Agreement on on-domain turns alone would hide the case where the two
    implementations disagree most: when neither has anything good to return."""
    from carelite.retrieval.ablation import OFF_DOMAIN_TURNS

    assert any(q in EQUIVALENCE_QUERIES for q in OFF_DOMAIN_TURNS)


def test_the_report_aggregates_per_query_rows() -> None:
    report = EquivalenceReport(
        label="test",
        k=4,
        depth=20,
        rows=[
            {"query": "q1", "overlap_at_k": 0.5, "spearman": 0.8, "n_native": 4, "n_adapter": 4},
            {"query": "q2", "overlap_at_k": 1.0, "spearman": 0.6, "n_native": 4, "n_adapter": 4},
        ],
    )
    assert report.mean_overlap == pytest.approx(0.75)
    assert report.mean_spearman == pytest.approx(0.7)
    assert report.n_queries == 2
    assert "0.75" in report.format_markdown()


def test_the_report_survives_undefined_rows() -> None:
    """A turn where one side returned nothing contributes no statistic; it
    must not be scored as zero agreement, which would be a different claim."""
    report = EquivalenceReport(
        label="test",
        k=4,
        depth=20,
        rows=[
            {"query": "q1", "overlap_at_k": 1.0, "spearman": None, "n_native": 4, "n_adapter": 4},
            {"query": "q2", "overlap_at_k": None, "spearman": None, "n_native": 0, "n_adapter": 0},
        ],
    )
    assert report.mean_overlap == pytest.approx(1.0)
    assert report.mean_spearman is None
    assert report.n_scored_overlap == 1


# ---------------------------------------------------------------------------
# Live: the number that actually gets reported
# ---------------------------------------------------------------------------


@pytest.mark.db
@pytest.mark.inference
def test_equivalence_over_the_committed_query_set() -> None:
    """Produces the reported comparison. Asserts that it was *computed* over
    every query, not that it reached any particular value: a threshold here
    would be an invitation to tune the adapter until the number looked good,
    which is the failure mode this measurement exists to avoid."""
    from carelite.retrieval.langchain_adapter import run_equivalence

    report = run_equivalence(matched=True, production=False)["matched"]
    assert report.n_queries == len(EQUIVALENCE_QUERIES)
    assert report.n_scored_overlap > 0
    assert report.mean_overlap is not None


@pytest.mark.db
def test_the_adapter_reads_the_existing_tables_and_builds_no_second_copy() -> None:
    """`chunk` and `kb_entry` are the corpus. A `langchain_pg_*` table would
    mean the adapter had copied and re-embedded it."""
    from carelite.db.connection import fetch_all
    from carelite.retrieval.langchain_adapter import load_records

    records = load_records()
    counts = fetch_all(
        "SELECT (SELECT count(*) FROM chunk) AS c, (SELECT count(*) FROM kb_entry) AS k"
    )[0]
    assert len(records) == counts["c"] + counts["k"]

    rows = fetch_all(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name LIKE 'langchain%'"
    )
    assert rows == []
