"""carelite.index.fts — query-side Postgres full-text search.

`chunk.tsv` and `kb_entry.tsv` are `GENERATED ALWAYS AS to_tsvector('english', ...)`
columns, frozen in `carelite/db/schema.sql`. This module owns everything on
the *query* side: turning a raw string into a `tsquery`, running it against
those columns, and ranking the results.

Why this matters for the project specifically: "NURSE", "teach-back", and
"Four Habits" are exact-match framework tokens that dense (embedding)
retrieval can miss — a paraphrase-tolerant embedding model has no special
incentive to place "the NURSE mnemonic" near a chunk that literally says
"NURSE statements" rather than "empathic response techniques". Lexical search
is what guarantees these terms are findable verbatim, so query construction
here is deliberately conservative rather than clever.

**What English stemming does to these terms** (verified against the live
corpus, see `tests/unit/index/test_fts.py`):

- `"teach-back"` tokenizes to three lexemes: the hyphenated compound itself
  *and* its two parts — `'teach-back' <-> 'teach' <-> 'back'`. `to_tsquery`,
  `plainto_tsquery`, and `websearch_to_tsquery` all reproduce this, so the
  hyphenated term is retrievable both as a whole and via its parts. Not lost.
- `"Four Habits Model"` survives as `'four' & 'habit' & 'model'` under a
  bag-of-words query, or as an ordered phrase (`'four' <-> 'habit' <-> 'model'`)
  under `phraseto_tsquery` / a quoted `websearch_to_tsquery` phrase. Not lost.
- `"NURSE"` stems to `'nurs'` — the same stem as "nurse"/"nursing" the
  healthcare profession. This is real and worth naming explicitly: it is a
  **precision** cost (a search for the mnemonic can also surface chunks about
  nursing staff), not a **recall** failure (the mnemonic itself is never
  dropped; every chunk containing the literal word "NURSE" is still
  reachable by `to_tsquery('english', 'NURSE')`). Disambiguating the two
  senses is a ranking/fusion problem for `carelite-retrieval`, not something
  the 'english' text search config can solve by itself — switching to the
  'simple' config would fix this collision but would also stop stemming
  everything else in the corpus (plurals, verb forms), a worse trade for a
  491-chunk corpus of biomedical prose. Documented here rather than "fixed"
  because there is no config-level fix that doesn't cost more than it saves.

Ranking uses `ts_rank_cd` (cover density): it rewards multiple query terms
appearing close together, which is exactly what distinguishes an incidental
one-word match from an actual discussion of a framework term.

**`websearch_to_tsquery` ANDs every content word by default** — there is no
implicit OR. `"teach-back method for confirming patient comprehension"`
(5 content words after stopword removal) returns *zero* hits against this
491-chunk corpus, not because "teach-back" is unfindable but because no
single chunk happens to also contain "method", "confirm", and "comprehens"
in the same 512-token window. This was caught empirically while writing
`probes.py` (three of the five lexical probes returned zero hits at first
with verbose natural-sentence queries) and is the reason every lexical
query in this module — probes included — is kept short and keyword-focused
(2-3 content words) rather than phrased as a full sentence. A caller that
wants sentence-length input handled forgivingly should either pull out the
key terms first or accept that recall drops sharply as query length grows;
this is inherent to AND-of-terms lexical search, not specific to this corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from carelite.db.connection import fetch_all

__all__ = [
    "FTSHit",
    "TSQueryMode",
    "framework_term_query",
    "search_chunks",
    "search_kb_entries",
    "to_tsquery_sql",
]

TSQueryMode = Literal["websearch", "plain", "phrase", "raw"]

#: Maps a validated mode name to the Postgres tsquery-construction function.
#: Validated against this allow-list (never interpolated from unchecked
#: input) because the mode selects a SQL *function name*, which cannot be
#: parametrized the way a value can.
_MODE_TO_FN: dict[str, str] = {
    "websearch": "websearch_to_tsquery",
    "plain": "plainto_tsquery",
    "phrase": "phraseto_tsquery",
    "raw": "to_tsquery",
}


def to_tsquery_sql(mode: TSQueryMode = "websearch") -> str:
    """The Postgres function name for `mode`. Raises on anything not in the
    allow-list rather than silently falling back, since this string is
    spliced directly into a query (see `search_chunks` / `search_kb_entries`)."""
    try:
        return _MODE_TO_FN[mode]
    except KeyError as exc:
        raise ValueError(
            f"unknown tsquery mode {mode!r}; expected one of {sorted(_MODE_TO_FN)}"
        ) from exc


def framework_term_query(term: str) -> str:
    """Best-effort query string for a specific framework term ("NURSE",
    "teach-back", "Four Habits Model", "SPIKES", "ask-tell-ask", ...).

    `websearch` mode is used by default throughout this module because it is
    the most forgiving of natural phrasing; framework terms are short enough
    that `mode="phrase"` (exact adjacency, via `phraseto_tsquery`) is also a
    reasonable choice for a caller that wants to eliminate false positives
    from word-order-scrambled matches. This helper exists so callers (this
    lane's own probes, and `carelite-retrieval`'s framework-query
    construction) do not have to remember the term's exact spelling/hyphenation
    conventions each time — it is a passthrough today, but is the one seam
    where term-specific handling (e.g. a future synonym table) would go.
    """
    return term


@dataclass(frozen=True, slots=True)
class FTSHit:
    """One lexical search result. Mirrors the fields of
    `carelite.types.RetrievedItem` that lexical search can actually fill in;
    `carelite-retrieval` is responsible for assembling the full item."""

    ref_id: str
    kind: str  # "chunk" | "kb_entry"
    text: str
    score: float
    paper_id: str | None = None
    theme: str | None = None


_CHUNK_SQL_TMPL = """
SELECT chunk_id, paper_id, text,
       ts_rank_cd(tsv, {fn}('english', %(query)s)) AS score
FROM chunk
WHERE tsv @@ {fn}('english', %(query)s)
ORDER BY score DESC
LIMIT %(top_k)s
"""

_KB_SQL_TMPL = """
SELECT entry_id, theme,
       finding || ' ' || practical_takeaway || ' ' || example_behavior AS text,
       ts_rank_cd(tsv, {fn}('english', %(query)s)) AS score
FROM kb_entry
WHERE tsv @@ {fn}('english', %(query)s)
ORDER BY score DESC
LIMIT %(top_k)s
"""


def search_chunks(query: str, *, top_k: int = 20, mode: TSQueryMode = "websearch") -> list[FTSHit]:
    """Lexical search over `chunk.tsv`. Returns at most `top_k` hits, ranked
    by cover-density rank, highest first. Returns `[]` (not an error) for a
    query that matches nothing or that reduces to an empty tsquery (e.g. all
    stopwords) — callers treat lexical-miss as a normal outcome, same as
    `carelite-retrieval`'s CRAG grading does for dense misses.
    """
    fn = to_tsquery_sql(mode)
    sql = _CHUNK_SQL_TMPL.format(fn=fn)
    rows = fetch_all(sql, {"query": query, "top_k": top_k})
    return [
        FTSHit(
            ref_id=r["chunk_id"],
            kind="chunk",
            text=r["text"],
            score=float(r["score"]),
            paper_id=r["paper_id"],
        )
        for r in rows
    ]


def search_kb_entries(
    query: str, *, top_k: int = 20, mode: TSQueryMode = "websearch"
) -> list[FTSHit]:
    """Lexical search over `kb_entry.tsv`. Same contract as `search_chunks`.
    Returns `[]` gracefully when `kb_entry` is empty (nothing to match) —
    important right now, since `carelite-kb` has not populated it yet."""
    fn = to_tsquery_sql(mode)
    sql = _KB_SQL_TMPL.format(fn=fn)
    rows = fetch_all(sql, {"query": query, "top_k": top_k})
    return [
        FTSHit(
            ref_id=r["entry_id"],
            kind="kb_entry",
            text=r["text"],
            score=float(r["score"]),
            theme=r["theme"],
        )
        for r in rows
    ]
