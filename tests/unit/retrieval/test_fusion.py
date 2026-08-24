"""RRF fusion, the backoff ladder, and the pure graph traversal.

Nothing here touches Postgres: `rrf_fuse`, `bfs_hops` and `_backoff_terms` are
pure, which is why they are the parts worth unit-testing at all.
"""

from __future__ import annotations

from carelite.retrieval.fusion import LegHit, RankedList, _backoff_terms, bfs_hops, rrf_fuse


def hit(
    ref_id: str,
    rank: int,
    *,
    kind: str = "chunk",
    score: float = 0.5,
    tier: str | None = None,
    theme: str | None = None,
    hops: int | None = None,
) -> LegHit:
    return LegHit(
        ref_id=ref_id,
        kind=kind,
        text=f"text for {ref_id}",
        raw_score=score,
        rank=rank,
        evidence_tier=tier,
        theme=theme,
        hops=hops,
    )


def test_rrf_prefers_documents_found_by_several_legs() -> None:
    """The whole point of fusion: corroboration across independent legs beats
    a single confident hit."""
    dense = RankedList(leg="dense", query="q", hits=[hit("a", 1), hit("b", 2)])
    lexical = RankedList(leg="lexical", query="q", hits=[hit("b", 1), hit("c", 2)])
    fused = rrf_fuse([dense, lexical], rrf_k=60)
    assert fused[0].ref_id == "b"


def test_rrf_records_per_leg_ranks_for_the_explain_view() -> None:
    dense = RankedList(leg="dense", query="q", hits=[hit("a", 2)])
    lexical = RankedList(leg="lexical", query="q", hits=[hit("a", 14)])
    graph = RankedList(leg="graph", query="q", hits=[hit("a", 1, hops=2)])
    item = rrf_fuse([dense, lexical, graph])[0]
    assert (item.dense_rank, item.lexical_rank, item.graph_hops) == (2, 14, 2)


def test_rrf_keeps_the_best_rank_but_accumulates_every_occurrence() -> None:
    """Repeated retrieval across independent queries is corroborating evidence
    and should accumulate, while the rank shown is the best one achieved."""
    q1 = RankedList(leg="dense", query="q1", hits=[hit("a", 5)])
    q2 = RankedList(leg="dense", query="q2", hits=[hit("a", 1)])
    only_once = rrf_fuse([q2])
    twice = rrf_fuse([q1, q2])
    assert twice[0].dense_rank == 1
    assert twice[0].score > only_once[0].score


def test_rrf_k_damps_the_top_ranks() -> None:
    lists = [RankedList(leg="dense", query="q", hits=[hit("a", 1), hit("b", 2)])]
    small_k = rrf_fuse(lists, rrf_k=1)
    large_k = rrf_fuse(lists, rrf_k=1000)
    gap_small = small_k[0].score - small_k[1].score
    gap_large = large_k[0].score - large_k[1].score
    assert gap_small > gap_large


def test_rrf_is_deterministic_on_ties() -> None:
    lists = [RankedList(leg="dense", query="q", hits=[hit("b", 1), hit("a", 1)])]
    assert [i.ref_id for i in rrf_fuse(lists)] == ["a", "b"]


def test_rrf_coerces_theme_and_tier_and_ignores_unknown_values() -> None:
    good = RankedList(leg="dense", query="q", hits=[hit("a", 1, tier="strong", theme="empathy")])
    bad = RankedList(
        leg="dense", query="q", hits=[hit("b", 2, tier="nonsense", theme="not_a_theme")]
    )
    items = {i.ref_id: i for i in rrf_fuse([good, bad])}
    assert items["a"].evidence_tier is not None and items["a"].theme is not None
    assert items["b"].evidence_tier is None and items["b"].theme is None


def test_rrf_respects_the_limit() -> None:
    lists = [RankedList(leg="dense", query="q", hits=[hit(f"r{i}", i) for i in range(1, 30)])]
    assert len(rrf_fuse(lists, limit=5)) == 5


def test_empty_fusion_is_empty_not_an_error() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([RankedList(leg="dense", query="q", hits=[])]) == []


# ------------------------------------------------------------------ backoff


def test_backoff_drops_the_shortest_term_first() -> None:
    """Length proxies distinctiveness: "teach-back" and "socioeconomic" are
    rarer than "care", so short words are shed first and the distinctive term
    survives longest."""
    variants = _backoff_terms("the socioeconomic disparities")
    assert variants
    assert "socioeconomic" in variants[-1]


def test_single_term_has_no_backoff() -> None:
    assert _backoff_terms("teach-back") == []


# -------------------------------------------------------------------- graph


def test_bfs_records_hop_distance() -> None:
    adjacency = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}
    assert bfs_hops(adjacency, ["a"], max_hops=2) == {"b": 1, "c": 2}


def test_bfs_respects_max_hops() -> None:
    adjacency = {"a": ["b"], "b": ["c"], "c": ["d"]}
    assert bfs_hops(adjacency, ["a"], max_hops=1) == {"b": 1}


def test_bfs_respects_the_limit() -> None:
    adjacency = {"a": [f"n{i}" for i in range(20)]}
    assert len(bfs_hops(adjacency, ["a"], max_hops=1, limit=3)) == 3


def test_bfs_excludes_the_seeds_themselves() -> None:
    adjacency = {"a": ["b"], "b": ["a"]}
    assert "a" not in bfs_hops(adjacency, ["a"], max_hops=2)


def test_bfs_handles_cycles() -> None:
    adjacency = {"a": ["b"], "b": ["c"], "c": ["a"]}
    assert bfs_hops(adjacency, ["a"], max_hops=5) == {"b": 1, "c": 2}


def test_bfs_on_an_empty_graph_is_empty() -> None:
    """`graph_edge` is empty until the carelite-graph lane lands; the leg is a
    no-op in the fusion rather than an error."""
    assert bfs_hops({}, ["a"], max_hops=2) == {}
