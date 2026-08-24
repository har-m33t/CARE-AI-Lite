"""carelite.graph — the curated property graph over the knowledge base.

`build.py` derives `graph_edge` rows from `kb_entry` / `kb_entry_source` /
`paper`; `materialize.py` loads them into NetworkX (Postgres stays the source
of truth, v3 §8); `queries.py` holds the traversals that justify the layer,
including the wave-3 gate ("which behaviors have outcome-level evidence
rather than expert opinion") and the coverage-gap report; `retrieval_hook.py`
exposes `graph_expand` for the retrieval lane's third fusion arm.

No LLM entity extraction, no community detection, no external corpus — every
node and edge here comes from the knowledge base that already exists.
"""

from carelite.graph.build import GraphEdgeRow, build_graph_edges
from carelite.graph.materialize import graph_from_edges, load_graph, node_kind
from carelite.graph.queries import (
    CoverageReport,
    build_coverage_report,
    entries_reachable_from_theme,
    framework_component_coverage,
    outcome_level_entries_graph,
    outcome_level_entries_sql,
    render_coverage_report,
)
from carelite.graph.retrieval_hook import graph_expand

__all__ = [
    "CoverageReport",
    "GraphEdgeRow",
    "build_coverage_report",
    "build_graph_edges",
    "entries_reachable_from_theme",
    "framework_component_coverage",
    "graph_expand",
    "graph_from_edges",
    "load_graph",
    "node_kind",
    "outcome_level_entries_graph",
    "outcome_level_entries_sql",
    "render_coverage_report",
]
