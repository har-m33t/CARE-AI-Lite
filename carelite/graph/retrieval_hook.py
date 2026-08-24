"""carelite.graph.retrieval_hook — `graph_expand(seed_ids, k)`, the third
fusion arm the retrieval lane's `carelite/retrieval/fusion.py` was written to
accept once `graph_edge` held data.

**Relationship to `fusion.graph_search`.** The retrieval lane already ships
its own SQL-driven BFS (`fusion.dense_search`/`lexical_search`/`graph_search`
+ `rrf_fuse`) that queries `graph_edge` directly and needs nothing from this
module to light up — it was written to activate the moment the table is
populated, and `carelite.graph.build` populating it is what does that. This
module is a second, independent path to the same table: the NetworkX-backed
traversal this lane owns, exposed as `graph_expand` per the lane brief, for a
caller that wants the materialised graph's expansion (in particular, the
`has`-edge exclusion in `carelite.graph.queries._topical_view`, which
`fusion.bfs_hops` does not apply) rather than a fresh SQL BFS per call.
Nothing here mutates `graph_edge`; both paths read the same source of truth.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx

from carelite.db.connection import fetch_all
from carelite.graph.materialize import load_graph
from carelite.graph.queries import bfs_entries
from carelite.types import EvidenceTier, RetrievedItem, Theme

__all__ = ["graph_expand"]

_HYDRATE_SQL = """
SELECT k.entry_id AS ref_id, k.theme, k.evidence_tier,
       k.finding || ' ' || k.practical_takeaway || ' ' || k.example_behavior AS text,
       (SELECT s.paper_id FROM kb_entry_source s
         WHERE s.entry_id = k.entry_id ORDER BY s.paper_id LIMIT 1) AS paper_id,
       (SELECT p.apa_citation FROM kb_entry_source s
          JOIN paper p USING (paper_id)
         WHERE s.entry_id = k.entry_id ORDER BY s.paper_id LIMIT 1) AS citation
FROM kb_entry k
WHERE k.entry_id = ANY(%(ids)s)
"""


def _coerce_theme(value: object) -> Theme | None:
    try:
        return Theme(str(value)) if value else None
    except ValueError:
        return None


def _coerce_tier(value: object) -> EvidenceTier | None:
    try:
        return EvidenceTier(str(value)) if value else None
    except ValueError:
        return None


def graph_expand(
    seed_ids: Sequence[str],
    k: int = 2,
    *,
    limit: int = 10,
    graph: nx.MultiDiGraph | None = None,
) -> list[RetrievedItem]:
    """Expand from ids the other retrieval legs already found, out to `k` hops.

    `graph` is normally left `None`, which materialises the live table via
    `load_graph()`; a caller that already has a materialised graph in hand
    (e.g. a long-lived process that loads it once at startup, per v3 §8) may
    pass it to skip re-querying Postgres on every call.

    Returns `[]` for seeds that reach nothing, exactly like
    `fusion.graph_search` does while `graph_edge` is empty — this is a
    quiet no-op, not an error, so a caller does not need to special-case it.
    """
    g = graph if graph is not None else load_graph()
    reached = bfs_entries(g, seed_ids, k=k, limit=limit)
    if not reached:
        return []

    rows = {str(r["ref_id"]): r for r in fetch_all(_HYDRATE_SQL, {"ids": list(reached)})}
    items: list[RetrievedItem] = []
    for entry_id, hops in sorted(reached.items(), key=lambda kv: (kv[1], kv[0])):
        row = rows.get(entry_id)
        if row is None:  # graph node with no live kb_entry row behind it
            continue
        items.append(
            RetrievedItem(
                ref_id=entry_id,
                kind="kb_entry",
                text=str(row["text"]),
                score=1.0 / (1 + hops),
                graph_hops=hops,
                theme=_coerce_theme(row["theme"]),
                evidence_tier=_coerce_tier(row["evidence_tier"]),
                paper_id=(str(row["paper_id"]) if row["paper_id"] else None),
                citation=(str(row["citation"]) if row["citation"] else None),
            )
        )
    return items
