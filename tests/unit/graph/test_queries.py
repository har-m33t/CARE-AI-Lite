"""Pure traversal tests against the fixture graph in conftest.py.

`test_live.py` is where `outcome_level_entries_graph` gets checked against
`outcome_level_entries_sql` on the real database — that agreement is the
wave-3 gate and needs live data to mean anything. Here, the traversal logic
itself is pinned against a graph small enough to verify by hand.
"""

from __future__ import annotations

import networkx as nx

from carelite.graph.build import derive_edges
from carelite.graph.materialize import graph_from_edges
from carelite.graph.queries import (
    FOUR_HABITS_COMPONENTS,
    NURSE_COMPONENTS,
    bfs_entries,
    entries_reachable_from_theme,
    framework_component_coverage,
    outcome_level_entries_graph,
)
from tests.unit.graph.conftest import ENTRIES, PAPERS


def test_outcome_level_entries_graph_reads_entry_tier_not_paper_tier(
    fixture_graph: nx.MultiDiGraph,
) -> None:
    """kb-e2 relays paper-strong (a `strong` paper) but is itself capped to
    `moderate`; it must not appear as outcome-level. kb-e1 and kb-e4 are
    genuinely `strong` at the entry level and must appear."""
    assert outcome_level_entries_graph(fixture_graph) == ["kb-e1", "kb-e4"]


def test_outcome_level_entries_graph_ignores_paper_predecessors() -> None:
    """A paper can itself point at tier:strong (its own design tier). That
    must never be mistaken for an entry and returned as a "behavior"."""
    edges = derive_edges(ENTRIES, PAPERS, [])
    g = graph_from_edges(edges)
    result = outcome_level_entries_graph(g)
    assert "paper-strong" not in result
    assert all(g.nodes[n]["kind"] == "entry" for n in result)


def test_outcome_level_entries_graph_empty_when_no_strong_tier_node() -> None:
    g = graph_from_edges([])
    assert outcome_level_entries_graph(g) == []


def test_framework_component_coverage_counts_only_populated_components(
    fixture_graph: nx.MultiDiGraph,
) -> None:
    coverage = framework_component_coverage(fixture_graph)
    assert coverage["understand"] == 1  # kb-e3
    assert coverage["ib"] == 1  # kb-e4
    # every other canonical component is present in the report at zero,
    # which is the shape the real corpus is in today (see build.py).
    untouched = set(NURSE_COMPONENTS) | set(FOUR_HABITS_COMPONENTS)
    untouched -= {"understand", "ib"}
    assert all(coverage[c] == 0 for c in untouched)


def test_bfs_entries_finds_the_restated_sibling_at_one_hop(
    fixture_graph: nx.MultiDiGraph,
) -> None:
    reached = bfs_entries(fixture_graph, ["kb-e1"], k=2)
    assert reached["kb-e2"] == 1  # direct restates edge


def test_bfs_entries_does_not_walk_through_tier_hubs(fixture_graph: nx.MultiDiGraph) -> None:
    """kb-e1 and kb-e4 are both `strong`, sharing only tier:strong. A topical
    2-hop expansion from kb-e1 must not reach kb-e4 through that hub — it
    shares no theme, phase, or restatement with kb-e1."""
    reached = bfs_entries(fixture_graph, ["kb-e1"], k=2)
    assert "kb-e4" not in reached


def test_bfs_entries_respects_hop_budget(fixture_graph: nx.MultiDiGraph) -> None:
    # kb-e3 is two topical hops from kb-e1 (via theme:teach_back)
    assert bfs_entries(fixture_graph, ["kb-e1"], k=1) == {"kb-e2": 1}
    assert bfs_entries(fixture_graph, ["kb-e1"], k=2) == {"kb-e2": 1, "kb-e3": 2}


def test_bfs_entries_ignores_seeds_absent_from_the_graph(
    fixture_graph: nx.MultiDiGraph,
) -> None:
    assert bfs_entries(fixture_graph, ["not-a-real-node"], k=2) == {}


def test_bfs_entries_respects_limit(fixture_graph: nx.MultiDiGraph) -> None:
    reached = bfs_entries(fixture_graph, ["kb-e1"], k=2, limit=1)
    assert len(reached) == 1
    assert next(iter(reached)) == "kb-e2"  # closest hop wins under a limit


def test_entries_reachable_from_theme_matches_bfs_from_the_theme_node(
    fixture_graph: nx.MultiDiGraph,
) -> None:
    reached = entries_reachable_from_theme(fixture_graph, "teach_back", k=2)
    assert reached == {"kb-e1": 1, "kb-e2": 1, "kb-e3": 1}


def test_entries_reachable_from_unknown_theme_is_empty(fixture_graph: nx.MultiDiGraph) -> None:
    assert entries_reachable_from_theme(fixture_graph, "nonexistent_theme", k=2) == {}
