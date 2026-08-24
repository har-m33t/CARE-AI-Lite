"""Fixtures for the graph lane's own tests.

A small, hand-built KB slice: two entries in `teach_back` that restate each
other (mirroring the corpus's actual `teach_back` cluster shape at a scale a
test can reason about), a third `teach_back` entry that doesn't, and one
`empathy` entry that instantiates a framework component (the corpus has zero
of these today — see `carelite.graph.build`'s module docstring — so this
fixture is what exercises that code path at all).
"""

from __future__ import annotations

import networkx as nx
import pytest

from carelite.graph.build import GraphEdgeRow, KBEntryRow, PaperRow, derive_edges
from carelite.graph.materialize import graph_from_edges

PAPER_STRONG = PaperRow(paper_id="paper-strong", evidence_tier="strong")
PAPER_MODERATE = PaperRow(paper_id="paper-moderate", evidence_tier="moderate")

ENTRY_1 = KBEntryRow(
    entry_id="kb-e1",
    theme="teach_back",
    evidence_tier="strong",
    encounter_phase=("explanation",),
    nurse_component=(),
    four_habits=(),
    source_paper_ids=("paper-strong",),
)
# Restates ENTRY_1: same paper, same theme, near-identical point — this pair
# is the fixture's stand-in for `redundancy_clusters`' calibrated output.
ENTRY_2 = KBEntryRow(
    entry_id="kb-e2",
    theme="teach_back",
    evidence_tier="moderate",  # relayed span from the same paper: capped tier
    encounter_phase=("explanation",),
    nurse_component=(),
    four_habits=(),
    source_paper_ids=("paper-strong",),
)
ENTRY_3 = KBEntryRow(
    entry_id="kb-e3",
    theme="teach_back",
    evidence_tier="emerging",
    encounter_phase=("closing",),
    nurse_component=("understand",),
    four_habits=(),
    source_paper_ids=("paper-moderate",),
)
ENTRY_4 = KBEntryRow(
    entry_id="kb-e4",
    theme="empathy",
    evidence_tier="strong",
    encounter_phase=("opening",),
    nurse_component=(),
    four_habits=("ib",),
    source_paper_ids=("paper-moderate",),
)

ENTRIES = (ENTRY_1, ENTRY_2, ENTRY_3, ENTRY_4)
PAPERS = (PAPER_STRONG, PAPER_MODERATE)
CLUSTER_PAIRS = (("kb-e1", "kb-e2"),)


@pytest.fixture
def fixture_edges() -> list[GraphEdgeRow]:
    return derive_edges(ENTRIES, PAPERS, CLUSTER_PAIRS)


@pytest.fixture
def fixture_graph(fixture_edges: list[GraphEdgeRow]) -> nx.MultiDiGraph:
    return graph_from_edges(fixture_edges)
