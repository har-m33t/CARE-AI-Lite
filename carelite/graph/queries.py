"""carelite.graph.queries — the traversals that justify the graph layer.

Three questions, each answered two ways (SQL against Postgres, traversal
against the materialised NetworkX graph), plus the coverage-gap report:

1. **Which behaviors have outcome-level evidence rather than expert opinion**
   — the wave-3 gate. "Outcome-level" is `EvidenceTier.STRONG`:
   `carelite.kb.papers.DESIGN_TIER` maps that tier exactly to systematic
   reviews, meta-analyses, and randomised controlled trials — designs that
   measure an outcome under some form of control. `EvidenceTier.EMERGING`
   is where "expert commentary" and narrative review live in that same
   mapping, i.e. literally expert opinion. `outcome_level_entries_sql` and
   `outcome_level_entries_graph` must agree exactly; `test_queries.py`
   pins that on live data.
2. **Which framework components are under-supported** — answered from the
   `instantiates` edges `carelite.graph.build` derives from
   `kb_entry.nurse_component` / `.four_habits`.
3. **Entries reachable from a theme within k hops** — BFS over a *topical*
   view of the graph that deliberately excludes `has` edges; see
   `_topical_view` for why.

None of this queries external data or clusters anything beyond what
`carelite.kb.review` already calibrated. Scope stays at "traversals over a
curated graph," per `.claude/agents/carelite-graph.md`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from carelite.config import REPO_ROOT
from carelite.db.connection import fetch_all
from carelite.graph.build import (
    RELATION_HAS,
    theme_node,
    tier_node,
)
from carelite.types import RUBRIC_DIMENSIONS

#: Where `main()` writes the coverage-gap report by default. Lives under this
#: lane's own owned path, alongside the code that generates it, the same way
#: `carelite.kb.review`'s digest lives alongside the pipeline that produces
#: it — regenerate with `python -m carelite.graph.queries`.
REPORT_PATH = REPO_ROOT / "carelite" / "graph" / "COVERAGE.md"

__all__ = [
    "FOUR_HABITS_COMPONENTS",
    "NURSE_COMPONENTS",
    "REPORT_PATH",
    "CoverageReport",
    "ThemeCoverage",
    "bfs_entries",
    "build_coverage_report",
    "entries_reachable_from_theme",
    "framework_component_coverage",
    "outcome_level_entries_graph",
    "outcome_level_entries_sql",
    "render_coverage_report",
    "theme_evidence_coverage",
    "write_coverage_report",
]

# ---------------------------------------------------------------------------
# 1. Outcome-level evidence
# ---------------------------------------------------------------------------

_OUTCOME_SQL = """
SELECT entry_id FROM kb_entry WHERE evidence_tier = 'strong' ORDER BY entry_id
"""


def outcome_level_entries_sql() -> list[str]:
    """Entry ids with outcome-level (`strong`) evidence, read directly from
    `kb_entry`. The ground truth `outcome_level_entries_graph` must match."""
    return [str(r["entry_id"]) for r in fetch_all(_OUTCOME_SQL)]


def outcome_level_entries_graph(g: nx.MultiDiGraph) -> list[str]:
    """Same question, answered by walking `entry --has--> tier:strong`.

    Reads only `has` edges, and only from `entry`-kind predecessors — a paper
    itself may point at `tier:strong` too (its own design tier), and that
    must not be mistaken for an entry's tier. This is exactly the check that
    the entry-level `has` edge exists to make possible: without it, an
    "outcome-level" traversal could only see the *paper's* tier and would
    wrongly credit every entry under a strong paper, including its relayed,
    capped-down spans.
    """
    target = tier_node("strong")
    if target not in g:
        return []
    out = []
    for pred in g.predecessors(target):
        if g.nodes[pred].get("kind") != "entry":
            continue
        edge_data = g.get_edge_data(pred, target) or {}
        if any(d.get("relation") == RELATION_HAS for d in edge_data.values()):
            out.append(pred)
    return sorted(out)


# ---------------------------------------------------------------------------
# 2. Framework component coverage
# ---------------------------------------------------------------------------

#: The canonical NURSE and Four Habits component identifiers, taken from the
#: frozen `RUBRIC_DIMENSIONS` order in `carelite.types` (the rubric lane's
#: scored dimensions). `kb_entry.nurse_component` / `.four_habits` are
#: free-text `TEXT[]` columns with no enum constraint in `schema.sql`, so this
#: is a reference vocabulary for the coverage report, not something the graph
#: enforces — a value outside it still gets its own node and still counts.
NURSE_COMPONENTS: tuple[str, ...] = RUBRIC_DIMENSIONS[
    :5
]  # name, understand, respect, support, explore
FOUR_HABITS_COMPONENTS: tuple[str, ...] = RUBRIC_DIMENSIONS[5:9]  # ib, epp, de, ie


def framework_component_coverage(g: nx.MultiDiGraph) -> dict[str, int]:
    """Entry count instantiating each NURSE / Four Habits component.

    Walks the materialised graph rather than assuming the reference
    vocabulary above is exhaustive: any `nurse:`/`habit:` node actually
    present shows up under its own label, so a value outside the canonical
    five-plus-four is reported rather than silently absorbed into it.
    """
    counts: dict[str, int] = dict.fromkeys(NURSE_COMPONENTS, 0)
    counts.update(dict.fromkeys(FOUR_HABITS_COMPONENTS, 0))
    for node, data in g.nodes(data=True):
        if data.get("kind") in ("nurse_component", "four_habits"):
            label = node.split(":", 1)[1]
            counts[label] = g.in_degree(node)
    return counts


# ---------------------------------------------------------------------------
# 3. k-hop reachability
# ---------------------------------------------------------------------------

#: Relations that represent evidence bookkeeping rather than topical
#: relatedness. A `tier:*` node fans out to every entry (or paper) at that
#: tier — dozens of entries spanning every theme — so letting a k-hop
#: *topical* traversal walk through it would flood a search seeded on one
#: theme or entry with unrelated cross-topic results just because they
#: happen to share an evidence grade. These edges are exactly what
#: `outcome_level_entries_graph` reads directly; they are excluded only from
#: the general-purpose expansion view built below.
_NON_EXPANSION_RELATIONS = frozenset({RELATION_HAS})


def _topical_view(g: nx.MultiDiGraph) -> nx.Graph:
    """Undirected projection used for k-hop expansion, `has` edges removed."""
    h: nx.Graph = nx.Graph()
    h.add_nodes_from(g.nodes(data=True))
    for u, v, data in g.edges(data=True):
        if data.get("relation") in _NON_EXPANSION_RELATIONS:
            continue
        h.add_edge(u, v)
    return h


def bfs_entries(
    g: nx.MultiDiGraph,
    seeds: Sequence[str],
    *,
    k: int = 2,
    limit: int | None = None,
) -> dict[str, int]:
    """Hop distance from any of `seeds` to every reachable `entry`-kind node.

    Pure once `g` is in memory. Seeds not present in the graph are ignored
    rather than raising, since a caller's seed id (from another retrieval
    leg) is not guaranteed to be a graph node. When several seeds reach the
    same entry, the shortest distance wins.
    """
    live_seeds = [s for s in dict.fromkeys(seeds) if s in g]
    if not live_seeds:
        return {}
    view = _topical_view(g)
    best: dict[str, int] = {}
    for seed in live_seeds:
        for node, dist in nx.single_source_shortest_path_length(view, seed, cutoff=k).items():
            if dist == 0 or g.nodes[node].get("kind") != "entry":
                continue
            if node not in best or dist < best[node]:
                best[node] = dist
    if limit is not None and len(best) > limit:
        best = dict(sorted(best.items(), key=lambda kv: (kv[1], kv[0]))[:limit])
    return best


def entries_reachable_from_theme(g: nx.MultiDiGraph, theme: str, *, k: int = 2) -> dict[str, int]:
    """Entries reachable from a theme within `k` topical hops, with hop count."""
    return bfs_entries(g, [theme_node(theme)], k=k)


# ---------------------------------------------------------------------------
# Coverage-gap report
# ---------------------------------------------------------------------------

_THEME_COVERAGE_SQL = """
SELECT k.theme, k.evidence_tier, count(*) AS n
FROM kb_entry k
GROUP BY k.theme, k.evidence_tier
"""

_EQUITY_SQL = """
SELECT
    (SELECT count(*) FROM kb_entry WHERE theme = 'equity') AS equity_theme_entries,
    (SELECT count(*) FROM kb_entry WHERE equity_relevant) AS equity_relevant_entries,
    (SELECT count(*) FROM kb_entry) AS total_entries
"""


@dataclass(frozen=True)
class ThemeCoverage:
    theme: str
    n_entries: int
    n_strong: int
    n_moderate: int
    n_emerging: int

    @property
    def outcome_level_share(self) -> float:
        return self.n_strong / self.n_entries if self.n_entries else 0.0


@dataclass(frozen=True)
class CoverageReport:
    theme_coverage: tuple[ThemeCoverage, ...]
    framework_coverage: dict[str, int]
    equity_theme_entries: int
    equity_relevant_entries: int
    total_entries: int
    n_restatement_clusters: int
    n_entries_in_clusters: int


def theme_evidence_coverage() -> list[ThemeCoverage]:
    """Per-theme entry counts by tier, read straight from `kb_entry`.

    This is the SQL half a traversal could reconstruct via `belongs_to` +
    `has` edges, but a report generator has no reason to walk the graph for
    numbers `GROUP BY` answers directly — the graph earns its keep on the
    *relational* questions, not on aggregates SQL already does well.
    """
    by_theme: dict[str, dict[str, int]] = {}
    for r in fetch_all(_THEME_COVERAGE_SQL):
        theme = str(r["theme"])
        by_theme.setdefault(theme, {"strong": 0, "moderate": 0, "emerging": 0})
        by_theme[theme][str(r["evidence_tier"])] = int(r["n"])
    out = []
    for theme, tiers in sorted(by_theme.items()):
        n = tiers["strong"] + tiers["moderate"] + tiers["emerging"]
        out.append(ThemeCoverage(theme, n, tiers["strong"], tiers["moderate"], tiers["emerging"]))
    return out


def build_coverage_report(g: nx.MultiDiGraph) -> CoverageReport:
    """Assemble the coverage-gap findings: thin evidence, thin framework
    mapping, thin equity coverage, and how much of the KB is restatement.

    Nothing here is a bug to paper over (per the lane brief) — a theme or
    component with zero support is reported as exactly that.
    """
    themes = theme_evidence_coverage()
    framework = framework_component_coverage(g)
    equity_row = fetch_all(_EQUITY_SQL)[0]

    # `restates` edges form a clique within each cluster (see `build.py`), so
    # cluster membership is recovered as the connected components of the
    # subgraph they induce, rather than recomputing `redundancy_clusters`.
    restate_graph: nx.Graph = nx.Graph()
    for u, v, data in g.edges(data=True):
        if data.get("relation") == "restates":
            restate_graph.add_edge(u, v)
    n_clusters = nx.number_connected_components(restate_graph)

    return CoverageReport(
        theme_coverage=tuple(themes),
        framework_coverage=framework,
        equity_theme_entries=int(equity_row["equity_theme_entries"]),
        equity_relevant_entries=int(equity_row["equity_relevant_entries"]),
        total_entries=int(equity_row["total_entries"]),
        n_restatement_clusters=n_clusters,
        n_entries_in_clusters=restate_graph.number_of_nodes(),
    )


def render_coverage_report(report: CoverageReport) -> str:
    lines = ["# Graph coverage report", ""]
    lines.append(
        f"{report.total_entries} entries total; "
        f"{report.n_entries_in_clusters} fall into {report.n_restatement_clusters} "
        "restatement cluster(s) — not independent evidence, see `restates` edges."
    )
    lines.append("")
    lines.append("## Evidence tier by theme")
    lines.append("")
    lines.append("| theme | entries | strong | moderate | emerging | outcome-level share |")
    lines.append("|---|---|---|---|---|---|")
    for tc in report.theme_coverage:
        lines.append(
            f"| {tc.theme} | {tc.n_entries} | {tc.n_strong} | {tc.n_moderate} | "
            f"{tc.n_emerging} | {tc.outcome_level_share:.0%} |"
        )
    lines.append("")
    lines.append("## Framework component coverage")
    lines.append("")
    lines.append(
        "Entry count `instantiates`-ing each NURSE / Four Habits component. "
        "0 for every component below means the mapping was never populated in "
        "this load of the knowledge base — a data-completeness gap in "
        "`kb_entry.nurse_component` / `.four_habits`, not a defect in this graph."
    )
    lines.append("")
    lines.append("| component | supporting entries |")
    lines.append("|---|---|")
    for component, count in sorted(report.framework_coverage.items()):
        flag = " **(unsupported)**" if count == 0 else ""
        lines.append(f"| {component} | {count}{flag} |")
    lines.append("")
    lines.append("## Equity coverage")
    lines.append("")
    lines.append(
        f"`equity` theme: {report.equity_theme_entries} of {report.total_entries} entries. "
        f"`equity_relevant` flag (any theme): {report.equity_relevant_entries}. "
        "Per D3, this is a property of the corpus — the literature that measures a "
        "disparity is not the literature that prescribes a remedy — not of extraction "
        "or of this graph."
    )
    lines.append("")
    return "\n".join(lines)


def write_coverage_report(path: Path | str = REPORT_PATH) -> Path:
    """Materialise the live graph and write the rendered coverage report.

    The one function in this module that touches both Postgres (via
    `build_coverage_report`'s SQL half) and the filesystem; everything it
    calls is otherwise separately testable.
    """
    from carelite.graph.materialize import load_graph  # local: avoid a cycle at import time

    path = Path(path)
    report = build_coverage_report(load_graph())
    path.write_text(render_coverage_report(report), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate the graph coverage-gap report.")
    ap.add_argument("--path", default=str(REPORT_PATH))
    args = ap.parse_args(list(argv) if argv is not None else None)
    path = write_coverage_report(args.path)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
