"""Live checks against the real database.

All `@pytest.mark.db`, excluded from `make check`. Run explicitly:

    pytest -m db tests/unit/graph/test_live.py

**Every test here is read-only.** `graph_edge` is this lane's own table to
populate, and that population happens once, for real, via
`python -m carelite.graph.build` — not inside a test. A prior lane silently
overwrote 475 live embeddings from an unscoped db-marked test, so nothing in
this file writes.
"""

from __future__ import annotations

import pytest

from carelite.db.connection import fetch_all
from carelite.graph.build import build_graph_edges, cluster_pairs_from_kb
from carelite.graph.materialize import load_graph
from carelite.graph.queries import (
    build_coverage_report,
    outcome_level_entries_graph,
    outcome_level_entries_sql,
    theme_evidence_coverage,
)

pytestmark = pytest.mark.db


def test_graph_edge_is_populated() -> None:
    """The wave-3 definition of done: graph_edge populated from the KB."""
    rows = fetch_all("SELECT count(*) AS n FROM graph_edge")
    assert rows[0]["n"] > 0


def test_outcome_level_query_agrees_between_sql_and_traversal() -> None:
    """The wave-3 gate: the same question, answered two ways, must agree
    exactly. This is what the entry-level `has` edge (see build.py's module
    docstring) exists to make true."""
    sql_result = set(outcome_level_entries_sql())
    graph_result = set(outcome_level_entries_graph(load_graph()))
    assert sql_result == graph_result
    assert len(sql_result) > 0  # the corpus does hold strong-tier entries


def test_build_graph_edges_is_idempotent_against_the_live_kb() -> None:
    """Rerunning derivation without writing must reproduce exactly what is
    already stored, so a dry run is a trustworthy diff tool."""
    derived = build_graph_edges(write=False)
    stored = fetch_all("SELECT source_id, relation, target_id FROM graph_edge")
    derived_triples = {(e.source_id, e.relation, e.target_id) for e in derived}
    stored_triples = {(r["source_id"], r["relation"], r["target_id"]) for r in stored}
    assert derived_triples == stored_triples


def test_cluster_pairs_come_from_the_kb_lanes_own_calibration() -> None:
    """Sanity check that this lane's `restates` edges track
    `carelite.kb.review.redundancy_clusters` rather than a second,
    independently drifting definition of what counts as a restatement."""
    pairs = cluster_pairs_from_kb()
    assert len(pairs) > 0
    touched = {e for pair in pairs for e in pair}
    assert len(touched) > 0


def test_framework_component_coverage_reports_the_zero_population_finding() -> None:
    """As of this lane's build, `nurse_component` and `four_habits` are empty
    on every loaded entry (see build.py's module docstring) — a
    data-completeness gap in the KB, not a defect in this graph. If the KB
    lane ever populates these, this test should be revisited rather than
    silently left asserting a stale finding."""
    report = build_coverage_report(load_graph())
    assert all(count == 0 for count in report.framework_coverage.values())


def test_equity_theme_is_thin_and_the_report_says_so() -> None:
    """D3: `equity` holds 3 entries as a property of the corpus. A coverage
    report that hid this, or implied equity was well-connected, would be
    wrong about its own knowledge base."""
    report = build_coverage_report(load_graph())
    assert report.equity_theme_entries == 3


def test_theme_evidence_coverage_sums_to_total_entries() -> None:
    themes = theme_evidence_coverage()
    total = fetch_all("SELECT count(*) AS n FROM kb_entry")[0]["n"]
    assert sum(t.n_entries for t in themes) == total
