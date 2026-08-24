"""carelite.graph.materialize — load `graph_edge` into NetworkX at startup.

v3 §8: do not install Neo4j. At the scale this graph actually reaches (~600
edges over ~150 distinct nodes, see `carelite.graph.build`) NetworkX in memory
is smaller than most CSVs, and Postgres stays the single source of truth —
this module is a pure derived view of it, never written back to.

`graph_from_edges` is the pure half (no I/O, unit-testable against a fixture
list); `load_graph` is the one function here that touches Postgres.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx

from carelite.db.connection import fetch_all
from carelite.graph.build import GraphEdgeRow

__all__ = ["fetch_graph_edges", "graph_from_edges", "load_graph", "node_kind"]

_SELECT_EDGES_SQL = "SELECT source_id, relation, target_id, evidence_tier, paper_id FROM graph_edge"


def fetch_graph_edges() -> list[GraphEdgeRow]:
    """Every row of `graph_edge`, as it stands in Postgres right now."""
    return [
        GraphEdgeRow(
            source_id=str(r["source_id"]),
            relation=str(r["relation"]),
            target_id=str(r["target_id"]),
            evidence_tier=(str(r["evidence_tier"]) if r["evidence_tier"] else None),
            paper_id=(str(r["paper_id"]) if r["paper_id"] else None),
        )
        for r in fetch_all(_SELECT_EDGES_SQL)
    ]


def node_kind(node_id: str) -> str:
    """Classify a node id by the prefix scheme `carelite.graph.build` writes.

    `kb_entry.entry_id` values are `kb-<theme>-<hash>`; `paper.paper_id`
    values are DOI slugs that never start with `kb-` (confirmed against the
    live table), so the fallback is safe rather than a guess.
    """
    if node_id.startswith("kb-"):
        return "entry"
    if node_id.startswith("theme:"):
        return "theme"
    if node_id.startswith("phase:"):
        return "phase"
    if node_id.startswith("tier:"):
        return "tier"
    if node_id.startswith("nurse:"):
        return "nurse_component"
    if node_id.startswith("habit:"):
        return "four_habits"
    return "paper"


def graph_from_edges(edges: Sequence[GraphEdgeRow]) -> nx.MultiDiGraph:
    """Build the traversal graph from edge rows already in memory. Pure.

    A `MultiDiGraph` keyed by relation: two node ids are never connected by
    more than one edge of the same relation (`derive_edges` dedupes on
    exactly `(source, relation, target)`), so keying on the relation name
    both prevents an accidental duplicate edge and makes `g[u][v]["belongs_to"]`
    a legible lookup.
    """
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    for e in edges:
        for node in (e.source_id, e.target_id):
            if node not in g:
                g.add_node(node, kind=node_kind(node))
        g.add_edge(
            e.source_id,
            e.target_id,
            key=e.relation,
            relation=e.relation,
            evidence_tier=e.evidence_tier,
            paper_id=e.paper_id,
        )
    return g


def load_graph() -> nx.MultiDiGraph:
    """Materialise the live `graph_edge` table. The one DB-touching call here."""
    return graph_from_edges(fetch_graph_edges())
