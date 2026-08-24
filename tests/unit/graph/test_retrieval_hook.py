"""`graph_expand` against a fixture graph, with `fetch_all` monkeypatched so
the hydration step (kb_entry -> RetrievedItem) is checked without a database.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pytest

import carelite.graph.retrieval_hook as retrieval_hook
from carelite.graph.retrieval_hook import graph_expand
from carelite.types import EvidenceTier, Theme

_ROWS: dict[str, dict[str, Any]] = {
    "kb-e2": {
        "ref_id": "kb-e2",
        "theme": "teach_back",
        "evidence_tier": "moderate",
        "text": "some finding some takeaway some behavior",
        "paper_id": "paper-strong",
        "citation": "Some Citation (2024)",
    },
    "kb-e3": {
        "ref_id": "kb-e3",
        "theme": "teach_back",
        "evidence_tier": "emerging",
        "text": "another finding",
        "paper_id": "paper-moderate",
        "citation": None,
    },
}


def _fake_fetch_all(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    del sql
    return [_ROWS[i] for i in params["ids"] if i in _ROWS]


@pytest.fixture(autouse=True)
def _patch_fetch_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retrieval_hook, "fetch_all", _fake_fetch_all)


def test_graph_expand_returns_reachable_entries_with_hops_and_provenance(
    fixture_graph: nx.MultiDiGraph,
) -> None:
    items = graph_expand(["kb-e1"], k=2, graph=fixture_graph)
    by_id = {i.ref_id: i for i in items}
    assert by_id["kb-e2"].graph_hops == 1
    assert by_id["kb-e3"].graph_hops == 2
    assert by_id["kb-e2"].kind == "kb_entry"
    assert by_id["kb-e2"].theme == Theme.TEACH_BACK
    assert by_id["kb-e2"].evidence_tier == EvidenceTier.MODERATE
    assert by_id["kb-e2"].citation == "Some Citation (2024)"


def test_graph_expand_orders_by_hop_then_id(fixture_graph: nx.MultiDiGraph) -> None:
    items = graph_expand(["kb-e1"], k=2, graph=fixture_graph)
    assert [i.ref_id for i in items] == ["kb-e2", "kb-e3"]


def test_graph_expand_is_a_quiet_no_op_for_unreachable_seeds(
    fixture_graph: nx.MultiDiGraph,
) -> None:
    assert graph_expand(["not-a-real-node"], k=2, graph=fixture_graph) == []


def test_graph_expand_drops_graph_nodes_with_no_live_kb_row(
    fixture_graph: nx.MultiDiGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A graph node reached by BFS but with no live `kb_entry` row behind it
    (e.g. deleted since the graph was last materialised) is dropped rather
    than raising or fabricating a `RetrievedItem`."""
    monkeypatch.setattr(retrieval_hook, "fetch_all", lambda sql, params: [])
    items = graph_expand(["kb-e1"], k=2, graph=fixture_graph)
    assert items == []
