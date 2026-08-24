"""carelite.graph.build — derive `graph_edge` rows from the knowledge base.

Postgres is the source of truth (v3 §8): this module reads `kb_entry`,
`kb_entry_source`, and `paper`, and writes the edges those tables imply into
`graph_edge`. Nothing here invents a node or a relation that is not already
present in the curated knowledge base — no entity extraction, no clustering
beyond what `carelite.kb.review` already calibrated, no external corpus.

**Node id scheme.** An `entry_id` (`kb-<theme>-<hash>`) and a `paper_id`
(a DOI slug) are used verbatim, because `carelite/retrieval/fusion.py`'s
`graph_search` resolves reached graph ids by looking them up directly in
`kb_entry` — an id that isn't a real entry_id or paper_id is silently
dropped there "rather than guessed at". Everything else is a node this graph
introduces and therefore gets a prefix, so a traversal can never confuse, say,
the theme "empathy" with an entry about empathy:

    theme:<Theme>            entry --belongs_to--> theme
    phase:<EncounterPhase>   entry --appropriate_in--> phase
    tier:<EvidenceTier>      paper --has--> tier ; entry --has--> tier
    nurse:<component>        entry --instantiates--> nurse component
    habit:<component>        entry --instantiates--> four-habits component

**Six relations from the build plan, plus two this lane adds and justifies:**

`supports`, `belongs_to`, `instantiates` (x2 targets), `appropriate_in`, and
`has` (paper -> tier) are exactly the spec in `.claude/agents/carelite-graph.md`.
Two more are added because the orchestrator's brief identified specific ways
the spec's six, alone, would let a traversal mislead:

- **`entry --has--> tier` (not just `paper --has--> tier`).** Evidence
  provenance is not uniform within a paper: a span relaying another study's
  result is capped below the paper's own tier (moderate from a systematic
  review, emerging otherwise — `carelite.kb.papers` / D3). `kb_entry.evidence_tier`
  already carries that per-entry cap; `paper.evidence_tier` does not. An
  "outcome-level evidence" traversal that only reaches `paper --has--> tier`
  would read every entry under a `strong` paper as `strong`, which overstates
  what a relayed span actually supports. This edge is what makes the
  traversal in `queries.outcome_level_entries_graph` agree with the SQL
  `WHERE evidence_tier = 'strong'` query exactly.
- **`entry --restates--> entry`.** Roughly a third of the KB restates itself:
  the `carelite-kb` lane's `redundancy_clusters` (recalibrated to threshold
  0.47) groups 40 of 116 entries into 12 clusters where one paper makes one
  point through several quoted sentences — `teach_back` alone has 10 of its
  15 entries in a single cluster. Left unmarked, a traversal that reaches
  several members of one cluster presents restatement as independent,
  convergent evidence. This edge makes the relationship explicit so a
  consumer can collapse a cluster to one citation instead of counting it as
  several. Pairs come from `carelite.kb.review.redundancy_clusters` directly
  — that threshold was calibrated against this exact knowledge base and this
  lane does not re-derive it.

**A finding this file exposes rather than works around: `nurse_component` and
`four_habits` are empty on all 116 loaded entries.** `kb_entry` carries both
columns and `carelite.kb.load` passes them through unchanged, but nothing in
the extraction pipeline populates them today. So `instantiates` edges are
derived correctly here — the code activates the moment those columns hold
values, exactly like `graph_search`'s "inert leg" — but zero exist right now.
`carelite.graph.queries.framework_component_coverage` reports this as what it
is: every NURSE and Four Habits component is currently unsupported by the
graph, not because the corpus lacks the evidence, but because the mapping was
never written down.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from carelite.db.connection import fetch_all, transaction
from carelite.kb.review import fetch_review_rows, redundancy_clusters

__all__ = [
    "RELATION_APPROPRIATE_IN",
    "RELATION_BELONGS_TO",
    "RELATION_HAS",
    "RELATION_INSTANTIATES",
    "RELATION_RESTATES",
    "RELATION_SUPPORTS",
    "GraphEdgeRow",
    "KBEntryRow",
    "PaperRow",
    "build_graph_edges",
    "cluster_pairs_from_kb",
    "derive_edges",
    "fetch_entry_rows",
    "fetch_paper_rows",
    "habit_node",
    "nurse_node",
    "phase_node",
    "theme_node",
    "tier_node",
]

# ---------------------------------------------------------------------------
# Node id helpers — the one place each prefix is spelled.
# ---------------------------------------------------------------------------


def theme_node(theme: str) -> str:
    return f"theme:{theme}"


def phase_node(phase: str) -> str:
    return f"phase:{phase}"


def tier_node(tier: str) -> str:
    return f"tier:{tier}"


def nurse_node(component: str) -> str:
    return f"nurse:{component}"


def habit_node(component: str) -> str:
    return f"habit:{component}"


RELATION_SUPPORTS = "supports"
RELATION_BELONGS_TO = "belongs_to"
RELATION_INSTANTIATES = "instantiates"
RELATION_APPROPRIATE_IN = "appropriate_in"
RELATION_HAS = "has"
RELATION_RESTATES = "restates"


@dataclass(frozen=True, slots=True)
class GraphEdgeRow:
    """One row of `graph_edge`, matching the frozen schema column-for-column."""

    source_id: str
    relation: str
    target_id: str
    evidence_tier: str | None = None
    paper_id: str | None = None


@dataclass(frozen=True, slots=True)
class KBEntryRow:
    """What `derive_edges` needs from `kb_entry` + `kb_entry_source`.

    Deliberately not `carelite.types.KBEntry`: that model requires a fully
    validated entry (e.g. a 20-char verbatim span) to construct, which is the
    right contract for the extraction pipeline and the wrong one for reading
    whatever is actually live in Postgres.
    """

    entry_id: str
    theme: str
    evidence_tier: str
    encounter_phase: tuple[str, ...]
    nurse_component: tuple[str, ...]
    four_habits: tuple[str, ...]
    source_paper_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PaperRow:
    paper_id: str
    evidence_tier: str


# ---------------------------------------------------------------------------
# Pure derivation
# ---------------------------------------------------------------------------


def derive_edges(
    entries: Sequence[KBEntryRow],
    papers: Sequence[PaperRow],
    cluster_pairs: Sequence[tuple[str, str]] = (),
) -> list[GraphEdgeRow]:
    """Turn KB rows into graph edges. Pure — no I/O, fully unit-testable.

    `cluster_pairs` is `(entry_id, entry_id)` pairs already computed by
    `carelite.kb.review.redundancy_clusters`; kept as a parameter rather than
    computed here so this function needs no database to test.
    """
    edges: list[GraphEdgeRow] = []
    seen: set[tuple[str, str, str]] = set()

    def emit(
        source: str,
        relation: str,
        target: str,
        *,
        evidence_tier: str | None = None,
        paper_id: str | None = None,
    ) -> None:
        key = (source, relation, target)
        if key in seen:
            return
        seen.add(key)
        edges.append(GraphEdgeRow(source, relation, target, evidence_tier, paper_id))

    for e in entries:
        primary_paper = e.source_paper_ids[0] if e.source_paper_ids else None
        for paper_id in e.source_paper_ids:
            # paper --supports--> entry, tagged with the *entry's* tier (see
            # module docstring: not the paper's, because relay capping can
            # make the two differ).
            emit(
                paper_id,
                RELATION_SUPPORTS,
                e.entry_id,
                evidence_tier=e.evidence_tier,
                paper_id=paper_id,
            )

        emit(
            e.entry_id,
            RELATION_BELONGS_TO,
            theme_node(e.theme),
            evidence_tier=e.evidence_tier,
            paper_id=primary_paper,
        )
        for component in e.nurse_component:
            emit(
                e.entry_id,
                RELATION_INSTANTIATES,
                nurse_node(component),
                evidence_tier=e.evidence_tier,
                paper_id=primary_paper,
            )
        for component in e.four_habits:
            emit(
                e.entry_id,
                RELATION_INSTANTIATES,
                habit_node(component),
                evidence_tier=e.evidence_tier,
                paper_id=primary_paper,
            )
        for phase in e.encounter_phase:
            emit(
                e.entry_id,
                RELATION_APPROPRIATE_IN,
                phase_node(phase),
                evidence_tier=e.evidence_tier,
                paper_id=primary_paper,
            )
        emit(
            e.entry_id,
            RELATION_HAS,
            tier_node(e.evidence_tier),
            evidence_tier=e.evidence_tier,
            paper_id=primary_paper,
        )

    for p in papers:
        emit(
            p.paper_id,
            RELATION_HAS,
            tier_node(p.evidence_tier),
            evidence_tier=p.evidence_tier,
            paper_id=p.paper_id,
        )

    for a, b in cluster_pairs:
        lo, hi = sorted((a, b))
        emit(lo, RELATION_RESTATES, hi)

    return edges


# ---------------------------------------------------------------------------
# I/O: read the live KB, write the derived edges
# ---------------------------------------------------------------------------

_ENTRY_SQL = """
SELECT e.entry_id, e.theme, e.evidence_tier, e.encounter_phase,
       e.nurse_component, e.four_habits,
       array_agg(s.paper_id ORDER BY s.paper_id) AS source_paper_ids
FROM kb_entry e
JOIN kb_entry_source s USING (entry_id)
GROUP BY e.entry_id
ORDER BY e.entry_id
"""

_PAPER_SQL = "SELECT paper_id, evidence_tier FROM paper ORDER BY paper_id"

_DELETE_SQL = "DELETE FROM graph_edge"

_INSERT_SQL = """
INSERT INTO graph_edge (source_id, relation, target_id, evidence_tier, paper_id)
VALUES (%(source_id)s, %(relation)s, %(target_id)s, %(evidence_tier)s, %(paper_id)s)
ON CONFLICT (source_id, relation, target_id) DO UPDATE
    SET evidence_tier = EXCLUDED.evidence_tier, paper_id = EXCLUDED.paper_id
"""


def fetch_entry_rows() -> list[KBEntryRow]:
    """Every `kb_entry` joined to its source papers. Requires a live database."""
    return [
        KBEntryRow(
            entry_id=str(r["entry_id"]),
            theme=str(r["theme"]),
            evidence_tier=str(r["evidence_tier"]),
            encounter_phase=tuple(r["encounter_phase"] or ()),
            nurse_component=tuple(r["nurse_component"] or ()),
            four_habits=tuple(r["four_habits"] or ()),
            source_paper_ids=tuple(r["source_paper_ids"] or ()),
        )
        for r in fetch_all(_ENTRY_SQL)
    ]


def fetch_paper_rows() -> list[PaperRow]:
    return [
        PaperRow(paper_id=str(r["paper_id"]), evidence_tier=str(r["evidence_tier"]))
        for r in fetch_all(_PAPER_SQL)
    ]


def cluster_pairs_from_kb() -> list[tuple[str, str]]:
    """Every unordered pair of entry ids that restate each other.

    Reuses `carelite.kb.review`'s calibrated clustering (threshold 0.47,
    single-linkage within one (theme, paper) group) rather than re-deriving
    it — that calibration is already tuned against this exact knowledge base
    and re-implementing it here would be a second definition to keep in sync.
    """
    rows = fetch_review_rows()
    clusters = redundancy_clusters(rows)
    pairs: list[tuple[str, str]] = []
    for cluster in clusters:
        ids = [row.entry_id for row in cluster.rows]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ids[i], ids[j]))
    return pairs


def _write_edges(edges: Sequence[GraphEdgeRow]) -> None:
    """Replace `graph_edge`'s contents with `edges`, in one transaction.

    A full delete-and-reinsert rather than a diff: `graph_edge` is a pure
    derived view of the KB (module docstring), so it is always safe and cheap
    to regenerate wholesale — at ~600 rows this is milliseconds, not a
    migration concern.
    """
    with transaction() as conn:
        conn.execute(_DELETE_SQL)
        for e in edges:
            conn.execute(
                _INSERT_SQL,
                {
                    "source_id": e.source_id,
                    "relation": e.relation,
                    "target_id": e.target_id,
                    "evidence_tier": e.evidence_tier,
                    "paper_id": e.paper_id,
                },
            )


def build_graph_edges(*, write: bool = True) -> list[GraphEdgeRow]:
    """Derive every edge from the live KB and, by default, replace `graph_edge`.

    Idempotent: rerunning against an unchanged KB reproduces the same edge
    set (`ON CONFLICT ... DO UPDATE` plus the preceding `DELETE` means a
    rerun after the KB changes drops edges for anything removed, too).
    """
    entries = fetch_entry_rows()
    papers = fetch_paper_rows()
    pairs = cluster_pairs_from_kb()
    edges = derive_edges(entries, papers, pairs)
    if write:
        _write_edges(edges)
    return edges


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Derive graph_edge from the live knowledge base.")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="derive and report edge counts without writing to graph_edge",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    edges = build_graph_edges(write=not args.dry_run)
    by_relation: dict[str, int] = {}
    for e in edges:
        by_relation[e.relation] = by_relation.get(e.relation, 0) + 1
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {len(edges)} edge(s):")
    for relation, count in sorted(by_relation.items()):
        print(f"  {relation}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
