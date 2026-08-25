"""carelite.retrieval.fusion — the three retrieval legs and Reciprocal Rank Fusion.

Dense (pgvector cosine), lexical (Postgres FTS), and graph (`graph_edge`
traversal) each produce a ranked list; RRF combines them into one.

**Why RRF rather than weighted score blending.** The three legs produce scores
on incomparable scales: cosine similarity lives in roughly [0.25, 0.75] on
this corpus, `ts_rank_cd` is an unbounded cover-density figure, and graph
proximity is an integer hop count. Normalising three such distributions onto
a common scale requires per-leg calibration constants that would need
re-fitting whenever the corpus or the embedder changed. RRF sidesteps the
problem entirely by discarding the scores and fusing the *ranks*:

    score(d) = sum over legs of 1 / (rrf_k + rank_leg(d))

`rrf_k` (60, from the frozen `settings.retrieval.rrf_k`) damps the influence
of the very top ranks so that one leg cannot dominate on the strength of a
single confident hit.

**The measured reason the lexical leg exists.** Framework vocabulary is
exact-match vocabulary. "NURSE", "teach-back", "SPIKES", "Four Habits" are
tokens a paraphrase-tolerant embedder has no obligation to rank highly, and
the carelite-index lane confirmed that lexical search surfaces them when
dense search does not. The converse is equally measured: embedding the raw
patient utterance and searching returns a mean top-4 cosine of 0.516 against
this corpus, which is indistinguishable from what an *off-domain* query
scores (0.513 measured ceiling over 15 off-domain probes). Neither leg is
sufficient alone.

**The lexical leg's zero-hit backoff is not optional.** `websearch_to_tsquery`
ANDs every content word, so a query of three terms is a three-way conjunction
that a 512-token chunk may simply not satisfy. `_lexical_with_backoff`
therefore retries a zero-hit query with progressively fewer terms, longest
term last, rather than reporting a lexical miss that is an artefact of query
length rather than of the corpus.

**The graph leg is inert today, by design and not by omission.** `graph_edge`
is empty until the `carelite-graph` lane lands; `graph_search` returns `[]`
and the fusion is unaffected. The traversal itself is written and unit-tested
against fixtures (`bfs_hops` is pure), so the leg lights up the moment the
table is populated without a change here.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from carelite.config import get_settings
from carelite.db.connection import fetch_all
from carelite.index.fts import search_chunks, search_kb_entries
from carelite.retrieval.query import MetadataFilter
from carelite.types import EvidenceTier, RetrievedItem, Theme

__all__ = [
    "LegHit",
    "RankedList",
    "bfs_hops",
    "dense_search",
    "graph_search",
    "lexical_search",
    "rrf_fuse",
]


@dataclass(frozen=True, slots=True)
class LegHit:
    """One hit from one leg, before fusion."""

    ref_id: str
    kind: str  # "chunk" | "kb_entry"
    text: str
    raw_score: float
    rank: int
    paper_id: str | None = None
    citation: str | None = None
    theme: str | None = None
    evidence_tier: str | None = None
    hops: int | None = None


@dataclass
class RankedList:
    """One leg's output for one query against one target table.

    **One list per (leg, target), never a merged one.** `chunk` and `kb_entry`
    produce scores on incomparable scales — a 512-token chunk repeats a query
    term several times and earns a far higher `ts_rank_cd` than a one-sentence
    curated KB entry can, and cosine behaves differently over short text too.
    Merging the two and truncating to `top_k` therefore starves the KB
    completely: measured on the live database, `search_kb_entries("teach-back")`
    returns 5 entries while a merged-and-sorted lexical leg returned **zero**
    of them, because ten chunks outscored every one. The curated, human-verified
    KB is the most valuable thing in the corpus, so losing it to a scale
    artefact is the worst possible failure here.

    Keeping the targets in separate lists is precisely what RRF is for: each
    list is ranked from 1 independently, and fusion compares ranks rather than
    scores, so neither table can crowd the other out.
    """

    leg: str  # "dense" | "lexical" | "graph"
    query: str
    hits: list[LegHit] = field(default_factory=list)
    note: str = ""
    target: str = "chunk"  # "chunk" | "kb_entry"


# ---------------------------------------------------------------------------
# Dense leg
# ---------------------------------------------------------------------------

_DENSE_CHUNK_SQL = """
SELECT c.chunk_id AS ref_id, c.text, c.paper_id,
       p.apa_citation, p.evidence_tier,
       1 - (c.embedding <=> %(vec)s::vector) AS score
FROM chunk c
JOIN paper p USING (paper_id)
WHERE c.embedding IS NOT NULL
ORDER BY c.embedding <=> %(vec)s::vector
LIMIT %(top_k)s
"""

#: kb_entry carries its own theme / phase / equity columns, so the metadata
#: filter is applied here and only here. `chunk` has no such columns (see
#: schema.sql) and is deliberately left unfiltered rather than given an
#: invented theme label.
_DENSE_KB_SQL = """
SELECT k.entry_id AS ref_id,
       k.finding || ' ' || k.practical_takeaway || ' ' || k.example_behavior AS text,
       k.theme, k.evidence_tier,
       (SELECT s.paper_id FROM kb_entry_source s
         WHERE s.entry_id = k.entry_id ORDER BY s.paper_id LIMIT 1) AS paper_id,
       (SELECT p.apa_citation FROM kb_entry_source s
          JOIN paper p USING (paper_id)
         WHERE s.entry_id = k.entry_id ORDER BY s.paper_id LIMIT 1) AS apa_citation,
       1 - (k.embedding <=> %(vec)s::vector) AS score
FROM kb_entry k
WHERE k.embedding IS NOT NULL
  {filter_sql}
ORDER BY k.embedding <=> %(vec)s::vector
LIMIT %(top_k)s
"""


def _kb_filter_sql(mf: MetadataFilter | None) -> tuple[str, dict[str, Any]]:
    """Build the kb_entry metadata predicate. Parameterised throughout: no
    filter value is ever interpolated into SQL text."""
    if mf is None or mf.is_empty:
        return "", {}
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if mf.themes:
        clauses.append("AND k.theme = ANY(%(themes)s)")
        params["themes"] = [t.value for t in mf.themes]
    if mf.encounter_phase is not None:
        # `encounter_phase` is TEXT[]; `&&` is array overlap. An entry with no
        # phase recorded is treated as applying to every phase rather than to
        # none, so a sparsely annotated KB does not silently filter itself out.
        clauses.append(
            "AND (k.encounter_phase = '{}' OR k.encounter_phase && ARRAY[%(phase)s]::text[])"
        )
        params["phase"] = mf.encounter_phase.value
    if mf.equity_relevant is not None:
        clauses.append("AND k.equity_relevant = %(equity)s")
        params["equity"] = mf.equity_relevant
    return " ".join(clauses), params


def _rows_to_hits(rows: Iterable[Any], kind: str) -> list[LegHit]:
    hits: list[LegHit] = []
    for rank, row in enumerate(rows, start=1):
        hits.append(
            LegHit(
                ref_id=str(row["ref_id"]),
                kind=kind,
                text=str(row["text"]),
                raw_score=float(row["score"]),
                rank=rank,
                paper_id=(str(row["paper_id"]) if row.get("paper_id") else None),
                citation=(str(row["apa_citation"]) if row.get("apa_citation") else None),
                theme=(str(row["theme"]) if row.get("theme") else None),
                evidence_tier=(str(row["evidence_tier"]) if row.get("evidence_tier") else None),
            )
        )
    return hits


def dense_search(
    vector: Sequence[float],
    query_label: str,
    *,
    top_k: int,
    metadata: MetadataFilter | None = None,
    include_kb: bool = True,
) -> list[RankedList]:
    """Cosine search over `chunk` and `kb_entry` via the HNSW indexes.

    Returns **one `RankedList` per target table** — see `RankedList`'s
    docstring for why they are never merged.

    `vector` is supplied by the caller rather than embedded here so that
    `pipeline.py` can batch every framework query and the HyDE passage into a
    single `embed_queries` / `embed_document` round trip.
    """
    vec = list(vector)
    out: list[RankedList] = []

    chunk_hits = _rows_to_hits(fetch_all(_DENSE_CHUNK_SQL, {"vec": vec, "top_k": top_k}), "chunk")
    chunk_hits.sort(key=lambda h: -h.raw_score)
    out.append(
        RankedList(
            leg="dense",
            query=query_label,
            target="chunk",
            hits=[_with_rank(h, i) for i, h in enumerate(chunk_hits[:top_k], start=1)],
        )
    )

    if include_kb:
        note = ""
        filter_sql, params = _kb_filter_sql(metadata)
        kb_rows = fetch_all(
            _DENSE_KB_SQL.format(filter_sql=filter_sql),
            {"vec": vec, "top_k": top_k, **params},
        )
        if not kb_rows and filter_sql:
            # A filter that eliminates everything is worse than no filter: the
            # KB is small and a three-way predicate can legitimately match
            # nothing. Retry open and say so in the trace.
            kb_rows = fetch_all(_DENSE_KB_SQL.format(filter_sql=""), {"vec": vec, "top_k": top_k})
            if kb_rows:
                note = "metadata filter matched no kb_entry rows; retried unfiltered"
        kb_hits = _rows_to_hits(kb_rows, "kb_entry")
        kb_hits.sort(key=lambda h: -h.raw_score)
        out.append(
            RankedList(
                leg="dense",
                query=query_label,
                target="kb_entry",
                note=note,
                hits=[_with_rank(h, i) for i, h in enumerate(kb_hits[:top_k], start=1)],
            )
        )
    return out


def _with_rank(hit: LegHit, rank: int) -> LegHit:
    return LegHit(
        ref_id=hit.ref_id,
        kind=hit.kind,
        text=hit.text,
        raw_score=hit.raw_score,
        rank=rank,
        paper_id=hit.paper_id,
        citation=hit.citation,
        theme=hit.theme,
        evidence_tier=hit.evidence_tier,
        hops=hit.hops,
    )


# ---------------------------------------------------------------------------
# Lexical leg
# ---------------------------------------------------------------------------


def _backoff_terms(query: str) -> list[str]:
    """Progressively shorter conjunctions, most distinctive term last.

    Length is the proxy for distinctiveness: in this corpus's vocabulary
    "teach-back" and "socioeconomic" are longer and rarer than "care" or
    "patient", so dropping short words first preserves the term most likely
    to be the one worth matching.
    """
    words = query.split()
    if len(words) <= 1:
        return []
    ordered = sorted(words, key=len)  # shortest first = dropped first
    variants: list[str] = []
    remaining = list(words)
    for drop in ordered[:-1]:
        remaining = [w for w in remaining if w != drop]
        if remaining:
            variants.append(" ".join(remaining))
    return variants


def lexical_search(
    query: str,
    *,
    top_k: int,
    metadata: MetadataFilter | None = None,
    include_kb: bool = True,
) -> list[RankedList]:
    """Postgres FTS over `chunk.tsv` and `kb_entry.tsv`, with zero-hit backoff.

    Returns one `RankedList` per target table. Each table gets its own backoff
    ladder, because a query can legitimately match the KB and not the papers:
    "teach-back" hits 5 curated entries and 15 chunks, while a rarer framework
    term may hit only the KB.
    """
    del metadata  # kb_entry FTS filtering is not applied: see module docstring
    out: list[RankedList] = []
    targets: list[tuple[str, Any]] = [("chunk", search_chunks)]
    if include_kb:
        targets.append(("kb_entry", search_kb_entries))

    for target, search_fn in targets:
        attempted = [query, *_backoff_terms(query)]
        note = ""
        hits: list[LegHit] = []
        used = query
        for attempt_idx, candidate in enumerate(attempted):
            found = search_fn(candidate, top_k=top_k)
            if found:
                used = candidate
                if attempt_idx:
                    note = (
                        f"query {query!r} matched no {target} rows "
                        f"(websearch_to_tsquery ANDs every content word); "
                        f"backed off to {candidate!r}"
                    )
                merged = [
                    LegHit(
                        ref_id=h.ref_id,
                        kind=h.kind,
                        text=h.text,
                        raw_score=h.score,
                        rank=0,
                        paper_id=h.paper_id,
                        theme=h.theme,
                    )
                    for h in found
                ]
                merged.sort(key=lambda h: -h.raw_score)
                hits = [_with_rank(h, i) for i, h in enumerate(merged[:top_k], start=1)]
                break
        else:
            note = f"query {query!r} and every backoff matched no {target} rows"
        out.append(
            RankedList(
                leg="lexical",
                query=used,
                target=target,
                hits=_hydrate_provenance(hits),
                note=note,
            )
        )
    return out


_PROVENANCE_SQL = """
SELECT c.chunk_id AS ref_id, c.paper_id, p.apa_citation, p.evidence_tier
FROM chunk c JOIN paper p USING (paper_id)
WHERE c.chunk_id = ANY(%(ids)s)
"""

_KB_PROVENANCE_SQL = """
SELECT k.entry_id AS ref_id, k.theme, k.evidence_tier,
       (SELECT s.paper_id FROM kb_entry_source s
         WHERE s.entry_id = k.entry_id ORDER BY s.paper_id LIMIT 1) AS paper_id,
       (SELECT p.apa_citation FROM kb_entry_source s
          JOIN paper p USING (paper_id)
         WHERE s.entry_id = k.entry_id ORDER BY s.paper_id LIMIT 1) AS apa_citation
FROM kb_entry k
WHERE k.entry_id = ANY(%(ids)s)
"""


def _hydrate_provenance(hits: list[LegHit]) -> list[LegHit]:
    """`fts.py`'s `FTSHit` carries no citation or evidence tier — it is
    deliberately limited to what the FTS query itself selects. The CLI
    evidence panel needs both, so fill them in with one extra query per kind
    rather than one per hit."""
    if not hits:
        return hits
    chunk_ids = [h.ref_id for h in hits if h.kind == "chunk"]
    kb_ids = [h.ref_id for h in hits if h.kind == "kb_entry"]
    meta: dict[str, dict[str, Any]] = {}
    if chunk_ids:
        for row in fetch_all(_PROVENANCE_SQL, {"ids": chunk_ids}):
            meta[str(row["ref_id"])] = dict(row)
    if kb_ids:
        for row in fetch_all(_KB_PROVENANCE_SQL, {"ids": kb_ids}):
            meta[str(row["ref_id"])] = dict(row)

    out: list[LegHit] = []
    for h in hits:
        m = meta.get(h.ref_id, {})
        out.append(
            LegHit(
                ref_id=h.ref_id,
                kind=h.kind,
                text=h.text,
                raw_score=h.raw_score,
                rank=h.rank,
                paper_id=h.paper_id or (str(m["paper_id"]) if m.get("paper_id") else None),
                citation=(str(m["apa_citation"]) if m.get("apa_citation") else None),
                theme=h.theme or (str(m["theme"]) if m.get("theme") else None),
                evidence_tier=(str(m["evidence_tier"]) if m.get("evidence_tier") else None),
                hops=h.hops,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Graph leg
# ---------------------------------------------------------------------------


def bfs_hops(
    adjacency: dict[str, list[str]],
    seeds: Sequence[str],
    *,
    max_hops: int = 2,
    limit: int = 10,
) -> dict[str, int]:
    """Breadth-first hop distance from `seeds`. Pure — no I/O, no database.

    Split out from `graph_search` precisely so the traversal is unit-testable
    against a fixture while `graph_edge` is still empty.
    """
    seen: dict[str, int] = {s: 0 for s in seeds}
    queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)
    ordered: list[str] = []
    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbour in adjacency.get(node, ()):
            if neighbour in seen:
                continue
            seen[neighbour] = depth + 1
            ordered.append(neighbour)
            queue.append((neighbour, depth + 1))
            if len(ordered) >= limit:
                return {n: seen[n] for n in ordered}
    return {n: seen[n] for n in ordered}


#: Ceiling on nodes visited during traversal, before resolution. Generous
#: rather than tight: see `graph_search` for why a tight cap silently emptied
#: this leg. The graph is ~715 edges, so this is never the binding constraint.
_TRAVERSAL_CAP = 500

_ADJACENCY_SQL = """
SELECT source_id, target_id FROM graph_edge
WHERE source_id = ANY(%(ids)s) OR target_id = ANY(%(ids)s)
"""


def _load_adjacency(ids: Sequence[str]) -> dict[str, list[str]]:
    """One frontier's worth of edges. Undirected: a curated property graph's
    relations ("supports", "is_evidence_for") are worth traversing in both
    directions when the question is "what else bears on this"."""
    if not ids:
        return {}
    adjacency: dict[str, list[str]] = {}
    for row in fetch_all(_ADJACENCY_SQL, {"ids": list(ids)}):
        src, tgt = str(row["source_id"]), str(row["target_id"])
        adjacency.setdefault(src, []).append(tgt)
        adjacency.setdefault(tgt, []).append(src)
    for key in adjacency:
        adjacency[key] = sorted(set(adjacency[key]))  # deterministic ordering
    return adjacency


def graph_search(
    seed_ids: Sequence[str],
    *,
    top_k: int,
    max_hops: int = 2,
) -> RankedList:
    """Expand from ids the other legs already found, out to `max_hops`.

    Returns `[]` while `graph_edge` is empty, which is the state today. That
    is a no-op in the fusion, not an error.
    """
    if not seed_ids:
        return RankedList(
            leg="graph", query="(no seeds)", target="kb_entry", hits=[], note="no seed ids"
        )

    frontier = list(dict.fromkeys(seed_ids))
    adjacency = _load_adjacency(frontier)
    if not adjacency:
        return RankedList(
            leg="graph",
            query=",".join(frontier[:3]),
            target="kb_entry",
            hits=[],
            note="graph_edge has no edges touching the seed ids",
        )

    # A second hop needs the neighbours' own edges loaded too.
    first_ring = {n for ns in adjacency.values() for n in ns}
    if max_hops > 1 and first_ring:
        for key, value in _load_adjacency(sorted(first_ring)).items():
            adjacency.setdefault(key, [])
            adjacency[key] = sorted(set(adjacency[key]) | set(value))

    # Traverse without a tight cap, then truncate *after* resolution.
    #
    # The limit must not be applied to raw nodes. This is a curated property
    # graph: `theme:*`, `phase:*`, `tier:*`, `nurse:*`, `habit:*` and paper ids
    # are hub nodes that connect entries but carry no retrievable text, and
    # `_fetch_graph_nodes` drops them. Capping the traversal at `top_k` raw
    # nodes therefore spends the budget on nodes that are about to be
    # discarded — measured against the live graph, a 20-seed traversal with
    # limit=10 returned 5 usable entries, and inside the full pipeline (whose
    # seeds are mostly chunk ids, and *no* edge has a chunk as its source) it
    # returned **zero**, making the graph leg silently inert even though
    # `graph_edge` holds 715 rows.
    #
    # Resolving first and truncating second costs nothing here — a 2-hop
    # traversal over ~715 edges is trivial — and avoids hardcoding the graph
    # lane's id conventions into this module, which would rot the moment they
    # add a node kind.
    reached = bfs_hops(adjacency, frontier, max_hops=max_hops, limit=_TRAVERSAL_CAP)
    if not reached:
        return RankedList(
            leg="graph",
            query=",".join(frontier[:3]),
            target="kb_entry",
            hits=[],
            note="no nodes reached",
        )

    hits = _fetch_graph_nodes(reached)[:top_k]
    note = ""
    if not hits and reached:
        note = (
            f"{len(reached)} nodes reached but none resolve to a kb_entry "
            f"(hub nodes carry no retrievable text)"
        )
    return RankedList(
        leg="graph", query=",".join(frontier[:3]), target="kb_entry", hits=hits, note=note
    )


def _fetch_graph_nodes(reached: dict[str, int]) -> list[LegHit]:
    """Materialise reached node ids as kb_entry or chunk rows.

    A graph node id is whatever the `carelite-graph` lane writes into
    `graph_edge`; ids that resolve to neither table (theme nodes, framework
    component nodes) are dropped rather than guessed at.
    """
    ids = sorted(reached, key=lambda n: (reached[n], n))
    rows = list(fetch_all(_KB_PROVENANCE_SQL, {"ids": ids}))
    found = {str(r["ref_id"]): dict(r) for r in rows}
    texts = {
        str(r["ref_id"]): str(r["text"])
        for r in fetch_all(
            "SELECT entry_id AS ref_id, "
            "finding || ' ' || practical_takeaway || ' ' || example_behavior AS text "
            "FROM kb_entry WHERE entry_id = ANY(%(ids)s)",
            {"ids": ids},
        )
    }
    hits: list[LegHit] = []
    for rank, node in enumerate((n for n in ids if n in found), start=1):
        meta = found[node]
        hits.append(
            LegHit(
                ref_id=node,
                kind="kb_entry",
                text=texts.get(node, ""),
                raw_score=1.0 / (1 + reached[node]),
                rank=rank,
                paper_id=(str(meta["paper_id"]) if meta.get("paper_id") else None),
                citation=(str(meta["apa_citation"]) if meta.get("apa_citation") else None),
                theme=(str(meta["theme"]) if meta.get("theme") else None),
                evidence_tier=(str(meta["evidence_tier"]) if meta.get("evidence_tier") else None),
                hops=reached[node],
            )
        )
    return hits


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _coerce_theme(value: str | None) -> Theme | None:
    try:
        return Theme(value) if value else None
    except ValueError:
        return None


def _coerce_tier(value: str | None) -> EvidenceTier | None:
    try:
        return EvidenceTier(value) if value else None
    except ValueError:
        return None


def rrf_fuse(
    ranked_lists: Sequence[RankedList],
    *,
    rrf_k: int | None = None,
    limit: int = 20,
) -> list[RetrievedItem]:
    """Fuse ranked lists into `RetrievedItem`s ordered by RRF score.

    Per-leg ranks are preserved onto each item (`dense_rank`, `lexical_rank`,
    `graph_hops`) because the CLI's `--explain` view shows exactly that
    breakdown — a reader should be able to see that a given chunk placed 2nd
    on dense and 14th on lexical, not just that it "scored 0.031".

    When a document is found by several queries on the same leg, the *best*
    rank is the one recorded, while every occurrence contributes to the score.
    That is standard RRF: repeated retrieval across independent queries is
    corroborating evidence and should accumulate.
    """
    k = rrf_k if rrf_k is not None else get_settings().retrieval.rrf_k

    scores: dict[str, float] = {}
    best: dict[str, LegHit] = {}
    dense_rank: dict[str, int] = {}
    lexical_rank: dict[str, int] = {}
    graph_hops: dict[str, int] = {}

    for ranked in ranked_lists:
        for hit in ranked.hits:
            scores[hit.ref_id] = scores.get(hit.ref_id, 0.0) + 1.0 / (k + hit.rank)
            prior = best.get(hit.ref_id)
            if prior is None or hit.raw_score > prior.raw_score or not prior.text:
                best[hit.ref_id] = hit
            if ranked.leg == "dense":
                dense_rank[hit.ref_id] = min(dense_rank.get(hit.ref_id, hit.rank), hit.rank)
            elif ranked.leg == "lexical":
                lexical_rank[hit.ref_id] = min(lexical_rank.get(hit.ref_id, hit.rank), hit.rank)
            elif ranked.leg == "graph" and hit.hops is not None:
                graph_hops[hit.ref_id] = min(graph_hops.get(hit.ref_id, hit.hops), hit.hops)

    ordered = sorted(scores, key=lambda r: (-scores[r], r))[:limit]
    items: list[RetrievedItem] = []
    for ref_id in ordered:
        hit = best[ref_id]
        items.append(
            RetrievedItem(
                ref_id=ref_id,
                kind=hit.kind,
                text=hit.text,
                score=scores[ref_id],
                dense_rank=dense_rank.get(ref_id),
                lexical_rank=lexical_rank.get(ref_id),
                graph_hops=graph_hops.get(ref_id),
                theme=_coerce_theme(hit.theme),
                evidence_tier=_coerce_tier(hit.evidence_tier),
                paper_id=hit.paper_id,
                citation=hit.citation,
            )
        )
    return items
