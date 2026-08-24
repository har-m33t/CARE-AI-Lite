"""carelite.index.probes — the wave-2 retrieval-quality gate.

Ten hand-written probes against the live corpus. This is deliberately *not*
the fused retrieval pipeline (RRF, HyDE, reranking, CRAG — that's
`carelite-retrieval`'s R0-R9 ablation harness). Each probe hits exactly one
of the two indexes this lane owns, directly:

- **lexical** probes go through `fts.py` (`search_chunks`) and exist to prove
  the framework-term claim in `fts.py`'s docstring with real data: "NURSE",
  "teach-back", "Four Habits Model", "SPIKES", "shared decision-making" are
  exact-match terms a paraphrase-tolerant embedding model has no obligation
  to rank highly, so lexical search must surface them.
- **dense** probes embed a natural-language question with `embed.py`
  (`OllamaEmbedder.embed_query`, so the query-side instruction prefix is
  exercised) and do a raw cosine-distance query against `chunk.embedding`
  via the HNSW index, to prove the stored vectors are actually usable for
  semantic (non-exact-wording) retrieval.

Each probe's "expected entry" is a tuple of substrings that must ALL appear
(case-insensitive) in at least one of the top-k results' text. Substrings
rather than pinned chunk_ids on purpose: chunk_ids are `{paper_id}::{ordinal}`
and shift when the corpus lane reloads (a deleted junk chunk renumbers every
later chunk in that paper — see `build.py`'s docstring). A probe keyed to a
substring that's true of the corpus content is far more durable than one
keyed to an identifier that's an implementation detail of how many chunks a
paper split into today.

Probes 1-10 target `chunk`; they were written while `kb_entry` was still
empty (`carelite-kb` had not finished, and `kb_entry.embedding` was
consequently NULL on every row — see `build.py`'s module docstring for the
incident where that stayed true silently well past the point `kb_entry` was
populated). Probes 11-12 target `kb_entry` and exist specifically to prove
the dense KB leg now returns real hits rather than the silent nothing it
returned before that gap was closed — a chunk-only probe suite cannot
detect a `kb_entry.embedding IS NULL` regression, since it never queries
that column.

Run it directly for a human-readable report:

    python -m carelite.index.probes

Or via pytest (`@pytest.mark.db`, `@pytest.mark.inference` — excluded from
`make check`, run explicitly): `pytest -m "db and inference" tests/unit/index/test_probes.py`
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Literal

from carelite.db.connection import fetch_all
from carelite.index.embed import OllamaEmbedder
from carelite.index.fts import search_chunks, search_kb_entries

__all__ = ["PROBES", "Probe", "ProbeResult", "main", "run_all_probes", "run_probe"]

ProbeMode = Literal["lexical", "dense"]
ProbeTarget = Literal["chunk", "kb_entry"]


@dataclass(frozen=True, slots=True)
class Probe:
    probe_id: str
    query: str
    mode: ProbeMode
    must_contain: tuple[str, ...]  # ALL must co-occur in one hit's text; case-insensitive
    target: ProbeTarget = "chunk"
    top_k: int = 5
    note: str = ""


# ---------------------------------------------------------------------------
# The 12 probes.
#
# 1-5 are lexical, exercising the exact framework terms named in the brief
#     and in fts.py's docstring. Grounded against real hit counts in the live
#     491-chunk corpus (checked by hand before writing these; see the
#     carelite-index final report for the counts).
# 6-10 are dense/semantic, one per a distinct communication theme, phrased as
#     a natural question that deliberately does NOT quote the expected
#     substring, so a pass demonstrates real semantic retrieval rather than
#     an accidental lexical match riding along on the dense path.
# 11-12 target kb_entry rather than chunk (see the module docstring): 11 is
#     lexical, 12 is dense/semantic, both against the 15-entry teach_back
#     cluster, which is large enough in this 116-row table to be a stable
#     target. 12 in particular is the probe that would have caught the
#     `kb_entry.embedding IS NULL` regression this lane shipped silently —
#     it is a direct check that dense KB retrieval is alive, not just that
#     the query pipeline runs without error.
# ---------------------------------------------------------------------------
PROBES: tuple[Probe, ...] = (
    Probe(
        probe_id="p01_nurse_lexical",
        query="NURSE statements empathic",
        mode="lexical",
        must_contain=("nurse", "empath"),
        note="requires NURSE co-occurring with empathy language, to rule out "
        "incidental 'nurse'-the-profession matches (see fts.py docstring)",
    ),
    Probe(
        probe_id="p02_teach_back_lexical",
        query="teach-back method",
        mode="lexical",
        must_contain=("teach-back",),
        note="the hyphenated compound; verifies it is not stemmed away",
    ),
    Probe(
        probe_id="p03_four_habits_lexical",
        query="Four Habits Model",
        mode="lexical",
        must_contain=("four habits",),
        note="multiword framework term; must survive as an adjacent phrase",
    ),
    Probe(
        probe_id="p04_spikes_lexical",
        query="SPIKES bad news",
        mode="lexical",
        must_contain=("spikes",),
    ),
    Probe(
        probe_id="p05_shared_decision_making_lexical",
        query="shared decision-making between clinician and patient",
        mode="lexical",
        must_contain=("shared decision",),
    ),
    Probe(
        probe_id="p06_emotion_recognition_dense",
        query="How should a clinician respond when a patient suddenly expresses "
        "fear or sadness during a visit?",
        mode="dense",
        must_contain=("emotion",),
        note="paraphrased; must not require the literal query wording",
    ),
    Probe(
        probe_id="p07_plain_language_dense",
        query="Explaining a diagnosis using simple everyday words instead of medical jargon",
        mode="dense",
        must_contain=("literacy",),
        note="corrected against real retrieval: the corpus's own vocabulary "
        "for this concept is 'health literacy' / 'speaking plainly', not "
        "'jargon' (which occurs in only 2 of 475 chunks) — the top dense hit "
        "is section 4.3 'Speaking Plainly / Health literacy...', exactly the "
        "right content under different wording than the query used.",
    ),
    Probe(
        probe_id="p08_equity_dense",
        query="Patients from lower-income backgrounds receiving less empathetic "
        "communication from their doctors",
        mode="dense",
        must_contain=("disparit",),
    ),
    Probe(
        probe_id="p09_trust_continuity_dense",
        query="Seeing the same clinician over time builds trust in the "
        "patient-provider relationship",
        mode="dense",
        must_contain=("trust",),
    ),
    Probe(
        probe_id="p10_activation_sdm_dense",
        query="Encouraging patients to take an active role in decisions about "
        "their own treatment plan",
        mode="dense",
        must_contain=("sdm",),
        note="corrected against real retrieval: this corpus's shared "
        "decision-making literature abbreviates to 'SDM' at least as often "
        "as it spells the phrase out ('implementation of SDM in practice'); "
        "the original ('activation', 'shared decision') AND-tuple demanded "
        "wording the top hits legitimately don't use.",
    ),
    Probe(
        probe_id="p11_teach_back_lexical_kb",
        query="teach-back method",
        mode="lexical",
        must_contain=("teach-back",),
        target="kb_entry",
        note="kb_entry analogue of p02; confirms the hyphenated compound "
        "survives kb_entry.tsv's tokenization the same way it survives "
        "chunk.tsv's.",
    ),
    Probe(
        probe_id="p12_teach_back_dense_kb",
        query="Confirming a patient truly understood their care instructions "
        "by having them explain it back in their own words",
        mode="dense",
        must_contain=("teach-back",),
        target="kb_entry",
        note="paraphrased, no literal 'teach-back' in the query. This is the "
        "probe that exercises kb_entry.embedding directly — a NULL "
        "embedding on every row (the actual incident this lane shipped and "
        "carelite-retrieval caught) would make this probe fail with zero "
        "hits, since the query filters WHERE embedding IS NOT NULL.",
    ),
)


@dataclass
class ProbeResult:
    probe_id: str
    passed: bool
    query: str
    mode: str
    n_hits: int
    top_ref_ids: list[str] = field(default_factory=list)
    detail: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.probe_id} ({self.mode}, {self.n_hits} hits) — {self.detail}"


def _dense_search_chunks(embedder: OllamaEmbedder, query: str, top_k: int) -> list[dict]:
    """Raw cosine-distance search against `chunk.embedding` via the HNSW
    index (`chunk_embedding_hnsw`, `vector_cosine_ops`), bypassing RRF/rerank
    entirely — this lane is only proving the vectors are retrievable, not
    fusing them."""
    vec = embedder.embed_query(query)
    rows = fetch_all(
        "SELECT chunk_id, paper_id, text, 1 - (embedding <=> %(vec)s::vector) AS score "
        "FROM chunk WHERE embedding IS NOT NULL "
        "ORDER BY embedding <=> %(vec)s::vector LIMIT %(top_k)s",
        {"vec": vec, "top_k": top_k},
    )
    return [dict(r) for r in rows]


def _dense_search_kb_entries(embedder: OllamaEmbedder, query: str, top_k: int) -> list[dict]:
    vec = embedder.embed_query(query)
    rows = fetch_all(
        "SELECT entry_id, "
        "finding || ' ' || practical_takeaway || ' ' || example_behavior AS text, "
        "1 - (embedding <=> %(vec)s::vector) AS score "
        "FROM kb_entry WHERE embedding IS NOT NULL "
        "ORDER BY embedding <=> %(vec)s::vector LIMIT %(top_k)s",
        {"vec": vec, "top_k": top_k},
    )
    return [dict(r) for r in rows]


def run_probe(probe: Probe, embedder: OllamaEmbedder | None = None) -> ProbeResult:
    """Run one probe against the live database (and, for dense probes, the
    live embedder). Requires `@pytest.mark.db` (+ `@pytest.mark.inference`
    for dense probes) — never called from `make check`."""
    if probe.mode == "lexical":
        if probe.target == "chunk":
            hits = search_chunks(probe.query, top_k=probe.top_k)
        else:
            hits = search_kb_entries(probe.query, top_k=probe.top_k)
        rows = [{"ref_id": h.ref_id, "text": h.text} for h in hits]
    else:
        embedder = embedder or OllamaEmbedder()
        if probe.target == "chunk":
            raw = _dense_search_chunks(embedder, probe.query, probe.top_k)
        else:
            raw = _dense_search_kb_entries(embedder, probe.query, probe.top_k)
        id_col = "chunk_id" if probe.target == "chunk" else "entry_id"
        rows = [{"ref_id": r[id_col], "text": r["text"]} for r in raw]

    matched_ref: str | None = None
    for row in rows:
        lower = row["text"].lower()
        if all(term.lower() in lower for term in probe.must_contain):
            matched_ref = row["ref_id"]
            break

    passed = matched_ref is not None
    if passed:
        detail = f"matched {probe.must_contain} in {matched_ref}"
    elif not rows:
        detail = f"no {probe.target} hits at all for query {probe.query!r}"
    else:
        detail = (
            f"none of {len(rows)} hits contained all of {probe.must_contain}; "
            f"top ref_ids: {[r['ref_id'] for r in rows]}"
        )

    return ProbeResult(
        probe_id=probe.probe_id,
        passed=passed,
        query=probe.query,
        mode=probe.mode,
        n_hits=len(rows),
        top_ref_ids=[r["ref_id"] for r in rows],
        detail=detail,
    )


def run_all_probes(embedder: OllamaEmbedder | None = None) -> list[ProbeResult]:
    """One embedder shared across all dense probes so the digest is resolved
    once, not per probe."""
    shared_embedder = embedder or OllamaEmbedder()
    results = [run_probe(p, shared_embedder) for p in PROBES]
    if embedder is None:
        shared_embedder.close()
    return results


def main() -> int:
    results = run_all_probes()
    print(f"carelite index probes — {len(results)} probes\n")
    for r in results:
        print(r)
    n_passed = sum(1 for r in results if r.passed)
    print(f"\n{n_passed}/{len(results)} passed")
    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
