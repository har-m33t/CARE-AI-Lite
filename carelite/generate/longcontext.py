"""Condition LC: the corpus in the window instead of retrieval.

LC is the baseline a reviewer will ask for. If stuffing everything into a
128K window matches or beats hybrid retrieval, the retrieval pipeline is not
earning its complexity and that is a result worth reporting.

**The corpus does not fit, and that has to be said plainly rather than
engineered around.** Build plan v3 says "your whole corpus fits in a 256K
window". Measured against the loaded database it does not fit in the 128K
window this project's models are configured for: 475 chunks are roughly 328,000
estimated tokens, about 2.6x the window. So LC cannot be "the whole corpus" and
any implementation claiming to be is truncating silently.

What LC is instead, stated exactly:

1. **Every knowledge base entry, always.** The 127 entries are the curated
   evidence layer, they cost roughly 13,000 tokens in total, and they are the
   material condition C most often retrieves. Dropping any of them would make
   LC a sampling decision rather than a baseline.
2. **Paper chunks round-robin by paper until the budget is spent.** Ordering is
   `ordinal` within `paper_id`, taken one paper at a time, so every paper in
   the corpus contributes its opening chunks before any paper contributes its
   second. Reading straight through in `paper_id` order would have filled the
   window with whichever papers sort first and excluded most of the corpus
   entirely; round-robin gives a subset that at least covers every source.
3. **Coverage is recorded, not assumed.** `CorpusPack.coverage` reports how
   many chunks of how many were included and what share of the budget was used,
   and the runner writes it alongside the generation. "LC saw 38% of the
   corpus" is a caveat the result has to carry; it is not a detail to leave in
   a docstring.

Ordering is deterministic, so the same database produces the same pack, so the
generation cache key means what it says.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from carelite.config import get_settings
from carelite.generate.model import estimate_tokens
from carelite.types import EvidenceTier, RetrievedItem, Theme

__all__ = ["CorpusPack", "CorpusUnits", "build_pack", "fetch_corpus_units", "pack_units"]

#: Tokens held back from the window for the system prompt, the patient turn,
#: the task line and the model's own output.
RESERVE_TOKENS = 4096


@dataclass(frozen=True, slots=True)
class CorpusPack:
    """The stuffed context for one long-context generation, plus what it omits."""

    items: tuple[RetrievedItem, ...]
    n_kb_included: int
    n_kb_total: int
    n_chunks_included: int
    n_chunks_total: int
    est_tokens: int
    budget_tokens: int

    @property
    def truncated(self) -> bool:
        return self.n_chunks_included < self.n_chunks_total or self.n_kb_included < self.n_kb_total

    @property
    def coverage(self) -> dict[str, Any]:
        """The caveat this condition has to carry, as data the runner can store."""
        return {
            "kb_included": self.n_kb_included,
            "kb_total": self.n_kb_total,
            "chunks_included": self.n_chunks_included,
            "chunks_total": self.n_chunks_total,
            "chunk_fraction": (
                round(self.n_chunks_included / self.n_chunks_total, 4)
                if self.n_chunks_total
                else 0.0
            ),
            "est_tokens": self.est_tokens,
            "budget_tokens": self.budget_tokens,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class CorpusUnits:
    kb: tuple[RetrievedItem, ...] = ()
    chunks_by_paper: tuple[tuple[RetrievedItem, ...], ...] = ()
    n_chunks_total: int = 0
    kb_total: int = 0


def fetch_corpus_units() -> CorpusUnits:
    """Read every knowledge base entry and every chunk, in a fixed order.

    A separate function from `pack_units` so the packing rule — the part with a
    decision in it — is unit-testable with no database.
    """
    from carelite.db.connection import fetch_all

    kb_rows = fetch_all(
        """
        SELECT k.entry_id, k.theme, k.evidence_tier, k.finding, k.practical_takeaway,
               k.example_behavior,
               (SELECT s.paper_id FROM kb_entry_source s
                 WHERE s.entry_id = k.entry_id ORDER BY s.paper_id LIMIT 1) AS paper_id,
               (SELECT p.apa_citation FROM kb_entry_source s
                  JOIN paper p USING (paper_id)
                 WHERE s.entry_id = k.entry_id ORDER BY p.paper_id LIMIT 1) AS citation
          FROM kb_entry k
         ORDER BY k.entry_id
        """
    )
    kb = tuple(
        RetrievedItem(
            ref_id=str(row["entry_id"]),
            kind="kb_entry",
            text=(
                f"{row['finding']}\n"
                f"Practical takeaway: {row['practical_takeaway']}\n"
                f"Example: {row['example_behavior']}"
            ),
            score=0.0,
            theme=_as_theme(row["theme"]),
            evidence_tier=_as_tier(row["evidence_tier"]),
            paper_id=_opt_str(row["paper_id"]),
            citation=_opt_str(row["citation"]),
        )
        for row in kb_rows
    )

    chunk_rows = fetch_all(
        """
        SELECT c.chunk_id, c.paper_id, c.ordinal, c.text, c.contextual_prefix,
               p.apa_citation
          FROM chunk c JOIN paper p USING (paper_id)
         ORDER BY c.paper_id, c.ordinal
        """
    )
    grouped: dict[str, list[RetrievedItem]] = {}
    for row in chunk_rows:
        prefix = row["contextual_prefix"]
        text = f"{prefix}\n\n{row['text']}" if prefix else str(row["text"])
        grouped.setdefault(str(row["paper_id"]), []).append(
            RetrievedItem(
                ref_id=str(row["chunk_id"]),
                kind="chunk",
                text=text,
                score=0.0,
                paper_id=str(row["paper_id"]),
                citation=_opt_str(row["apa_citation"]),
            )
        )
    by_paper = tuple(tuple(grouped[pid]) for pid in sorted(grouped))
    return CorpusUnits(
        kb=kb,
        chunks_by_paper=by_paper,
        n_chunks_total=len(chunk_rows),
        kb_total=len(kb_rows),
    )


def pack_units(units: CorpusUnits, *, budget_tokens: int) -> CorpusPack:
    """Fill `budget_tokens` with the knowledge base, then round-robin chunks.

    Knowledge base entries go in first and unconditionally; if they alone
    overflow the budget the pack is truncated there and `coverage` says so,
    which is a configuration error worth seeing rather than one to paper over.
    """
    items: list[RetrievedItem] = []
    used = 0
    kb_included = 0
    for entry in units.kb:
        cost = estimate_tokens(entry.text) + 16
        if used + cost > budget_tokens:
            break
        items.append(entry)
        used += cost
        kb_included += 1

    chunks_included = 0
    cursors = [0] * len(units.chunks_by_paper)
    exhausted = False
    while not exhausted and used < budget_tokens:
        exhausted = True
        for i, paper_chunks in enumerate(units.chunks_by_paper):
            if cursors[i] >= len(paper_chunks):
                continue
            exhausted = False
            chunk = paper_chunks[cursors[i]]
            cost = estimate_tokens(chunk.text) + 16
            if used + cost > budget_tokens:
                cursors[i] = len(paper_chunks)  # this paper's turn is over
                continue
            items.append(chunk)
            used += cost
            chunks_included += 1
            cursors[i] += 1

    return CorpusPack(
        items=tuple(items),
        n_kb_included=kb_included,
        n_kb_total=units.kb_total,
        n_chunks_included=chunks_included,
        n_chunks_total=units.n_chunks_total,
        est_tokens=used,
        budget_tokens=budget_tokens,
    )


def build_pack(*, budget_tokens: int | None = None) -> CorpusPack:
    """The long-context pack for this database and this configured window."""
    if budget_tokens is None:
        window = get_settings().models.long_context.context_window
        budget_tokens = max(window - RESERVE_TOKENS, 1024)
    return pack_units(fetch_corpus_units(), budget_tokens=budget_tokens)


def _opt_str(value: object) -> str | None:
    return None if value is None else str(value)


def _as_theme(value: object) -> Theme | None:
    try:
        return Theme(str(value))
    except ValueError:  # pragma: no cover - a theme outside the frozen vocabulary
        return None


def _as_tier(value: object) -> EvidenceTier | None:
    try:
        return EvidenceTier(str(value))
    except ValueError:  # pragma: no cover
        return None
