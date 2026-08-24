"""`derive_edges` is pure: no I/O, so this is where the edge derivation logic
actually gets checked, against the small KB slice in `conftest.py`.
"""

from __future__ import annotations

from carelite.graph.build import (
    RELATION_APPROPRIATE_IN,
    RELATION_BELONGS_TO,
    RELATION_HAS,
    RELATION_INSTANTIATES,
    RELATION_RESTATES,
    RELATION_SUPPORTS,
    GraphEdgeRow,
    derive_edges,
    habit_node,
    nurse_node,
    phase_node,
    theme_node,
    tier_node,
)
from tests.unit.graph.conftest import CLUSTER_PAIRS, ENTRIES, PAPERS


def by_relation(edges: list[GraphEdgeRow], relation: str) -> list[GraphEdgeRow]:
    return [e for e in edges if e.relation == relation]


def test_every_source_paper_gets_a_supports_edge(fixture_edges: list[GraphEdgeRow]) -> None:
    supports = by_relation(fixture_edges, RELATION_SUPPORTS)
    pairs = {(e.source_id, e.target_id) for e in supports}
    assert pairs == {
        ("paper-strong", "kb-e1"),
        ("paper-strong", "kb-e2"),
        ("paper-moderate", "kb-e3"),
        ("paper-moderate", "kb-e4"),
    }


def test_belongs_to_uses_theme_prefixed_node(fixture_edges: list[GraphEdgeRow]) -> None:
    belongs = by_relation(fixture_edges, RELATION_BELONGS_TO)
    targets = {e.target_id for e in belongs}
    assert targets == {theme_node("teach_back"), theme_node("empathy")}


def test_appropriate_in_one_edge_per_phase(fixture_edges: list[GraphEdgeRow]) -> None:
    phases = by_relation(fixture_edges, RELATION_APPROPRIATE_IN)
    pairs = {(e.source_id, e.target_id) for e in phases}
    assert pairs == {
        ("kb-e1", phase_node("explanation")),
        ("kb-e2", phase_node("explanation")),
        ("kb-e3", phase_node("closing")),
        ("kb-e4", phase_node("opening")),
    }


def test_instantiates_only_fires_for_populated_components(
    fixture_edges: list[GraphEdgeRow],
) -> None:
    """The corpus has zero of these today (see build.py's module docstring),
    so the fixture is the only place this path is exercised at all."""
    instantiates = by_relation(fixture_edges, RELATION_INSTANTIATES)
    pairs = {(e.source_id, e.target_id) for e in instantiates}
    assert pairs == {
        ("kb-e3", nurse_node("understand")),
        ("kb-e4", habit_node("ib")),
    }


def test_has_edges_use_the_entrys_own_tier_not_the_papers(
    fixture_edges: list[GraphEdgeRow],
) -> None:
    """kb-e2 relays paper-strong's finding and is capped to `moderate`; its
    `has` edge must point at tier:moderate, not tier:strong, even though the
    paper `has` edge for paper-strong does point at tier:strong."""
    has_edges = {(e.source_id, e.target_id) for e in by_relation(fixture_edges, RELATION_HAS)}
    assert ("kb-e2", tier_node("moderate")) in has_edges
    assert ("kb-e2", tier_node("strong")) not in has_edges
    assert ("paper-strong", tier_node("strong")) in has_edges
    # entries and papers both get a `has` edge
    assert {"kb-e1", "kb-e2", "kb-e3", "kb-e4", "paper-strong", "paper-moderate"} == {
        e.source_id for e in by_relation(fixture_edges, RELATION_HAS)
    }


def test_restates_connects_the_cluster_pair_deterministically(
    fixture_edges: list[GraphEdgeRow],
) -> None:
    restates = by_relation(fixture_edges, RELATION_RESTATES)
    assert len(restates) == 1
    assert (restates[0].source_id, restates[0].target_id) == ("kb-e1", "kb-e2")


def test_restates_orders_the_pair_so_direction_is_deterministic() -> None:
    """Regardless of the order a cluster hands back the pair, the stored edge
    is canonically ordered — otherwise the same pair could be inserted as
    both (a, b) and (b, a) across reruns and violate nothing, but mean
    something different to a directed reader."""
    forward = derive_edges(ENTRIES, PAPERS, [("kb-e1", "kb-e2")])
    backward = derive_edges(ENTRIES, PAPERS, [("kb-e2", "kb-e1")])
    forward_restates = by_relation(forward, RELATION_RESTATES)[0]
    backward_restates = by_relation(backward, RELATION_RESTATES)[0]
    assert (forward_restates.source_id, forward_restates.target_id) == (
        backward_restates.source_id,
        backward_restates.target_id,
    )


def test_derive_edges_dedupes_on_source_relation_target() -> None:
    """`graph_edge` has a UNIQUE(source_id, relation, target_id) constraint;
    derive_edges must never hand it a duplicate triple, even if the same
    logical fact would otherwise be emitted twice."""
    duplicated_pairs = (*CLUSTER_PAIRS, ("kb-e1", "kb-e2"))  # same pair twice
    edges = derive_edges(ENTRIES, PAPERS, duplicated_pairs)
    triples = [(e.source_id, e.relation, e.target_id) for e in edges]
    assert len(triples) == len(set(triples))


def test_edge_count_matches_expectation(fixture_edges: list[GraphEdgeRow]) -> None:
    # 4 supports + 4 belongs_to + 2 instantiates + 4 appropriate_in
    # + 6 has (4 entries + 2 papers) + 1 restates
    assert len(fixture_edges) == 21
