"""`graph_from_edges` is pure: build a graph from rows already in memory and
check node classification and multi-edge handling, with no database.
"""

from __future__ import annotations

import networkx as nx

from carelite.graph.build import GraphEdgeRow
from carelite.graph.materialize import graph_from_edges, node_kind


def test_node_kind_classifies_every_prefix() -> None:
    assert node_kind("kb-teach_back-abc123") == "entry"
    assert node_kind("theme:teach_back") == "theme"
    assert node_kind("phase:explanation") == "phase"
    assert node_kind("tier:strong") == "tier"
    assert node_kind("nurse:understand") == "nurse_component"
    assert node_kind("habit:ib") == "four_habits"
    assert node_kind("10-1371-journal-pone-0231350") == "paper"


def test_fixture_graph_labels_every_node_kind(fixture_graph: nx.MultiDiGraph) -> None:
    kinds = {data["kind"] for _, data in fixture_graph.nodes(data=True)}
    assert kinds == {"entry", "paper", "theme", "phase", "tier", "nurse_component", "four_habits"}


def test_fixture_graph_is_undirected_reachable_both_ways(fixture_graph: nx.MultiDiGraph) -> None:
    """A curated property graph's relations are worth walking in both
    directions (fusion.py makes the same call for its own BFS) — confirm the
    directed edges this module stores still let a caller find the reverse."""
    assert fixture_graph.has_edge("kb-e1", "theme:teach_back")
    assert not fixture_graph.has_edge("theme:teach_back", "kb-e1")  # stored one-directional
    undirected = fixture_graph.to_undirected(as_view=True)
    assert undirected.has_edge("theme:teach_back", "kb-e1")


def test_graph_from_edges_keys_multi_edges_by_relation() -> None:
    """Two distinct relations between the same pair of nodes must not
    collide into a single stored edge."""
    edges = [
        GraphEdgeRow("kb-e1", "belongs_to", "theme:teach_back"),
        GraphEdgeRow("kb-e1", "has", "theme:teach_back"),  # contrived, but must not collide
    ]
    g = graph_from_edges(edges)
    assert g.number_of_edges("kb-e1", "theme:teach_back") == 2
    relations = {data["relation"] for data in g.get_edge_data("kb-e1", "theme:teach_back").values()}
    assert relations == {"belongs_to", "has"}


def test_graph_from_edges_carries_evidence_tier_and_paper_onto_edge_data() -> None:
    edges = [
        GraphEdgeRow("paper-1", "supports", "kb-e1", evidence_tier="strong", paper_id="paper-1")
    ]
    g = graph_from_edges(edges)
    data = next(iter(g.get_edge_data("paper-1", "kb-e1").values()))
    assert data["evidence_tier"] == "strong"
    assert data["paper_id"] == "paper-1"
