"""Live checks against the real database and models.

All `@pytest.mark.db` / `@pytest.mark.inference`, so excluded from `make check`.
Run explicitly:

    pytest -m db tests/unit/retrieval/test_live.py
    pytest -m "db and inference" tests/unit/retrieval/test_live.py

**Every test here is read-only.** A prior lane silently overwrote all 475 live
embeddings from an unscoped db-marked test and it took a probe sanity check to
notice, so nothing in this file writes to Postgres.
"""

from __future__ import annotations

import pytest

from carelite.retrieval.crag import DENSE_NULL_ANCHOR, DENSE_SIGNAL_ANCHOR
from carelite.retrieval.flags import preset
from carelite.retrieval.fusion import dense_search, graph_search, lexical_search, rrf_fuse
from carelite.retrieval.query import build_queries

pytestmark = pytest.mark.db


def test_lexical_leg_returns_one_list_per_target() -> None:
    lists = lexical_search("teach-back", top_k=10)
    targets = {rl.target for rl in lists}
    assert targets == {"chunk", "kb_entry"}


def test_kb_entries_are_not_starved_by_chunk_scores() -> None:
    """The bug this split fixes: `ts_rank_cd` is incomparable between a
    512-token chunk and a one-sentence curated entry, so a merged-and-sorted
    lexical leg returned zero KB entries for "teach-back" even though the KB
    holds several. The curated KB is the most valuable thing in the corpus."""
    lists = lexical_search("teach-back", top_k=10)
    kb = next(rl for rl in lists if rl.target == "kb_entry")
    if kb.hits:  # kb_entry is populated by the carelite-kb lane
        assert kb.hits[0].rank == 1
        assert all(h.kind == "kb_entry" for h in kb.hits)


def test_lexical_backoff_reports_itself() -> None:
    """A sentence-length lexical query collapses to nothing under
    AND-of-terms tsquery semantics; the backoff must say so in the note."""
    lists = lexical_search("scared cancer nobody", top_k=10)
    chunk_list = next(rl for rl in lists if rl.target == "chunk")
    if not chunk_list.hits:
        assert "matched no chunk rows" in chunk_list.note


def test_graph_leg_is_inert_but_not_broken() -> None:
    """`graph_edge` is empty until the carelite-graph lane lands. That is a
    no-op in the fusion, not an error."""
    ranked = graph_search(["kb-0001"], top_k=10)
    assert ranked.hits == []
    assert ranked.note


@pytest.mark.inference
def test_dense_leg_retrieves_and_carries_provenance() -> None:
    from carelite.index.embed import OllamaEmbedder

    with OllamaEmbedder() as embedder:
        qs = build_queries("I'm scared this is cancer and nobody explains anything to me.")
        vec = embedder.embed_query(qs.dense_queries[0])
        lists = dense_search(vec, qs.dense_queries[0], top_k=10, metadata=qs.metadata)

    chunks = next(rl for rl in lists if rl.target == "chunk")
    assert chunks.hits
    top = chunks.hits[0]
    assert top.rank == 1
    # The CLI evidence panel needs a citation and a tier for every row.
    assert top.paper_id and top.citation and top.evidence_tier


@pytest.mark.inference
def test_on_domain_beats_the_measured_off_domain_ceiling() -> None:
    """The separation the CRAG cosine anchors rest on: over 12 on-domain and
    15 off-domain probes, on-domain top-1 cosine bottomed out at 0.587 while
    off-domain topped out at 0.513. This re-checks one probe from each side so
    a corpus reload that destroyed the separation would be caught."""
    from carelite.index.embed import OllamaEmbedder

    with OllamaEmbedder() as embedder:
        on = embedder.embed_query(
            "empathic response to patient distress: naming and acknowledging the emotion"
        )
        off = embedder.embed_query("How do I replace the oil filter on a 2003 Honda Civic?")
        on_hits = dense_search(on, "on", top_k=1, include_kb=False)[0].hits
        off_hits = dense_search(off, "off", top_k=1, include_kb=False)[0].hits

    assert on_hits[0].raw_score > DENSE_SIGNAL_ANCHOR - 0.1
    assert off_hits[0].raw_score < DENSE_NULL_ANCHOR + 0.15
    assert on_hits[0].raw_score > off_hits[0].raw_score


@pytest.mark.inference
def test_hyde_closes_the_register_gap() -> None:
    """Measured over five patient turns: raw utterance mean top-4 cosine
    0.516, framework query 0.655, HyDE 0.708 — and the raw utterance figure
    sits at the *off-domain* ceiling (0.513), which is how severe the
    patient-language-versus-guidance-document gap is."""
    from carelite.index.embed import OllamaEmbedder
    from carelite.retrieval.hyde import generate_hyde_passage
    from carelite.retrieval.llm import LLMClient

    utterance = "I'm scared this is cancer and nobody explains anything to me."
    with OllamaEmbedder() as embedder, LLMClient() as llm:
        raw = dense_search(embedder.embed_query(utterance), "raw", top_k=4, include_kb=False)[
            0
        ].hits
        result = generate_hyde_passage(utterance, client=llm)
        if not result:
            pytest.skip("generator unavailable")
        hyde = dense_search(
            embedder.embed_document(result.passage), "hyde", top_k=4, include_kb=False
        )[0].hits

    raw_mean = sum(h.raw_score for h in raw) / len(raw)
    hyde_mean = sum(h.raw_score for h in hyde) / len(hyde)
    assert hyde_mean > raw_mean


@pytest.mark.inference
def test_full_stack_produces_a_populated_trace() -> None:
    from carelite.retrieval import retrieve

    trace = retrieve(
        "Do I have to choose between surgery and radiation? What would you do?",
        flags=preset("R9").with_(use_llm_crag=False),
    )
    assert trace.route is not None
    assert trace.queries
    assert trace.latency_ms is not None
    if trace.retrieved:
        item = trace.retrieved[0]
        assert item.ref_id and item.text
        assert item.rerank_score is not None


@pytest.mark.inference
def test_off_domain_turn_falls_back_to_condition_b() -> None:
    """The end-to-end version of the gate's reason for existing, against the
    real corpus and the real judge model."""
    from carelite.config import get_settings
    from carelite.retrieval import retrieve
    from carelite.retrieval.llm import LLMClient

    with LLMClient(model_tag=get_settings().models.judge.tag) as judge:
        trace = retrieve(
            "How do I replace the oil filter on a 2003 Honda Civic?",
            flags=preset("R9"),
            grader_client=judge,
        )
    assert trace.fell_back_to_b is True
    assert trace.retrieved == []


def test_rrf_over_live_legs_is_stable() -> None:
    """Two identical fusions of the same live lexical result must agree —
    retrieval feeding a controlled comparison cannot be order-dependent."""
    lists = lexical_search("shared decision", top_k=10)
    assert [i.ref_id for i in rrf_fuse(lists)] == [i.ref_id for i in rrf_fuse(lists)]


def test_graph_leg_is_live_now_that_graph_edge_is_populated() -> None:
    """The inert leg activated as designed.

    `graph_edge` was empty when this leg was written and `graph_search`
    returned `[]` as a documented no-op. carelite-graph has since landed 623
    edges, and the leg started doing real work without a change here — which
    is the property the pure/IO split in `fusion.py` was for.
    """
    from carelite.db.connection import fetch_all

    n_edges = fetch_all("SELECT count(*) c FROM graph_edge")[0]["c"]
    if not n_edges:
        pytest.skip("graph_edge is empty; the leg is a documented no-op")

    seeds = [
        r["entry_id"] for r in fetch_all("SELECT entry_id FROM kb_entry ORDER BY entry_id LIMIT 3")
    ]
    if not seeds:
        pytest.skip("kb_entry is empty")

    ranked = graph_search(seeds, top_k=10)
    assert ranked.hits, "populated graph_edge should reach neighbours from KB seeds"
    assert all(h.hops is not None and h.hops >= 1 for h in ranked.hits)
    assert all(h.ref_id not in seeds for h in ranked.hits), "seeds are not their own hits"


def test_lc_sample_fits_the_budget_and_covers_every_paper() -> None:
    """D7's sampler against the real corpus: the whole corpus does not fit, so
    LC-sample takes a fixed round-robin slice. Every paper must survive —
    losing one would make LC's content an accident of the seed."""
    from carelite.db.connection import fetch_all
    from carelite.retrieval.ablation import CHARS_PER_TOKEN, lc_sample_stats

    stats = lc_sample_stats()
    assert stats["fits"] is False, "if the corpus now fits, D7 needs revisiting"
    assert stats["sample_chunks"] < stats["n_chunks"]

    ids = stats["sample_chunk_ids"]
    assert len(set(ids)) == len(ids), "no chunk may be selected twice"

    all_papers = {r["paper_id"] for r in fetch_all("SELECT DISTINCT paper_id FROM chunk")}
    sampled_papers = {
        r["paper_id"]
        for r in fetch_all(
            "SELECT DISTINCT paper_id FROM chunk WHERE chunk_id = ANY(%(ids)s)",
            {"ids": ids},
        )
    }
    assert sampled_papers == all_papers, "every paper must be represented"

    chars = fetch_all(
        "SELECT coalesce(sum(length(text)), 0) AS c FROM chunk WHERE chunk_id = ANY(%(ids)s)",
        {"ids": ids},
    )[0]["c"]
    assert int(chars) // CHARS_PER_TOKEN <= stats["budget_tokens"]


def test_lc_sample_is_stable_across_calls() -> None:
    """A fixed context that moved between rows would make C-vs-LC-sample
    partly a comparison of two different LC conditions."""
    from carelite.retrieval.ablation import lc_sample

    assert lc_sample() == lc_sample()
