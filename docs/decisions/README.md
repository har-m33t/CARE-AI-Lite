# Decision log

A dated record of the decisions that shaped this build, in the order they were made. This log is
owned by `carelite-repro`; it is a record, not a debate — a lane that disagrees with an entry here
raises it rather than editing history.

This is a different document from `DECISIONS.md` at the repo root, which is owned by the
orchestrating session and holds the decisions v3 explicitly routed to the project owner (D1–D5:
theme taxonomy, equity-stratum membership, KB re-extraction approval, dropping the human-verification
claim, and the `racial_ethnic` axis description). Entries below cite `DECISIONS.md` rather than
duplicate it. This log's job is the decisions `DECISIONS.md` does not cover: the foundational
architecture calls build plan v3 argued for and wave 0 encoded directly into the frozen contracts,
before there was an orchestrating session to delegate to.

---

## 2026-08-22 — Postgres over SQLite (and over Chroma) as the system of record

**Decision: PostgreSQL + pgvector holds everything — corpus, knowledge base, graph edges, every
generation, every rubric score — in one instance. No separate vector store, no SQLite.**

Encoded directly in `carelite/db/schema.sql` and `carelite/config.py.Settings.database_url` at wave
0 (commit `f76447a`). Build plan v3 §5–7 makes the case: the project's hardest analysis queries are
three- and four-way joins across generations, scores, scenarios, and retrieval traces (*"mean
NURSE-Respect score by condition, restricted to held-out scenarios in the equity stratum, where
retrieval grade was 'relevant', grouped by prompt version"*), and a result set split across CSVs,
JSON files, and a separate vector database cannot answer that reliably. The vector count here
(~2,000 embeddings) is trivial for pgvector's HNSW index regardless of choice, which removes
performance as a deciding axis and leaves the decision to be made on transactional consistency, a
single backup artifact, and SQL as the analysis language `pandas.read_sql` reads directly. Chroma +
SQLite remains the documented fallback (v3 §7) if Postgres becomes a genuine local blocker; it has
not.

## 2026-08-22 — Knowledge base re-derived from the actual corpus, not the planning-time count

**Decision: the knowledge base is built from what the 33 retrieved papers actually support, via
LLM-assisted extraction with automated provenance validation — not hand-authored to hit build plan
v3's "45 entries, 10 themes" figure or README's "15 sample entries."**

`carelite/kb/extract.py`, `carelite/kb/validate.py`, and `carelite/kb/spans.py` implement this at
wave 0–2. The planning-time entry counts were never derived from a specific accounting of the
corpus — `knowledge_base/TAXONOMY.md`'s finding that build plan v3's "10 themes" appears exactly
once, in passing, with no supporting list anywhere in the repository, is the same pattern: a round
number asserted rather than counted. The project owner's later ruling on the adjacent 15-vs-45
question (see the user's memory record, "CARELite KB counts are floors") settles the same way —
both figures describe different things (an early sample vs. the full base) and the actual count is
whatever the corpus and the provenance gate produce — and it has kept moving even after being
reported here as settled twice, so treat any total in this document as a snapshot rather than a
target and re-query `knowledge_base/review/kb_review_digest.md` before citing one. As of this
writing that is 116 loaded entries across 33 papers. `DECISIONS.md` D3's outcome (2026-08-24)
settled the one figure this project can say is not still moving for extraction reasons: the equity
theme's count of 3 is established as a property of the corpus rather than an unfinished extraction,
even though the overall total moved twice more afterward for unrelated reasons (a later variant
window's incidental non-equity yield, then a redundancy-check recalibration — see
`docs/limitations.md` §2 for both). See `docs/limitations.md` §1–2 for the coverage this leaves
skewed, and `DECISIONS.md` D3–D4 for the two decisions this made necessary downstream (equity
re-extraction and its negative result, dropping the human-verification claim).

## 2026-08-22 — The model roster: local, cross-family judge, cross-model baseline

**Decision: `gemma4:12b` generator, `qwen3.5:9b` cross-model baseline (Condition A2), `gpt-oss:20b`
judge, `bge-m3` embedder, `BAAI/bge-reranker-v2-m3` reranker — all local via Ollama or
`sentence-transformers`, no hosted API in the inference path.**

Fixed in `carelite/config.py.Models` at wave 0. Two properties of this roster are load-bearing
rather than incidental. First, **the judge is a different model family from the generator**
(`gpt-oss:20b` judging `gemma4:12b`/`qwen3.5:9b` output), which is what lets v3 §13's independence
requirement hold — a same-family judge would be scoring outputs shaped by training it shares,
which is exactly the blind spot judge validation is supposed to catch. Second, **everything runs
local and offline** (`Settings.ollama_host` is pinned to `localhost` with the comment "no egress at
inference"), which is what makes 1,080 generations plus a 5-sample self-consistency judge pass on
the validation subset affordable at all — the free-inference argument in v3 §11 that lets the
project budget 100 scenarios instead of trimming to fit an API budget only holds because nothing
here is metered per token. The cost of this choice is stated plainly in `docs/limitations.md` §6:
every result is bounded by a local-model capability ceiling, not the ceiling of the largest hosted
frontier models. Digests, not tags, are what get recorded per generation (`ModelSpec.digest`, v3
§16) — Ollama tags are mutable, so `make pin-models` records the digest actually pulled at run time
rather than trusting a tag string to still mean the same weights later.

## 2026-08-22 — Judge-primary evaluation, human rating deferred and validated as its own study

**Decision: the full 1,080-generation run is scored by the local LLM judge; human rating is a
smaller, separately-validated study layered on top, not the primary measurement instrument, and it
has not yet been conducted.**

This follows directly from build plan v3 §12's framing of a single-rater ceiling as "the hardest
solo constraint" and its own ranked list of imperfect options. Encoded across
`carelite/eval/judge/` (the judge pipeline and its five-part validation study —
self-consistency, positional bias, span grounding, per-dimension Krippendorff's α / Spearman's ρ,
and the pre-specified confirmatory threshold in `carelite/eval/judge/validation.py`) and
`carelite/eval/human/` (the blinding, packet, and reliability machinery, deliberately exercised
against a synthetic-rater generator in `carelite/eval/human/synthetic.py` before any real rater
sees a response). The consequence of this decision is stated in `docs/preregistration.md` §9 and
repeated in `docs/limitations.md` §4 rather than left implicit: every number this project reports
is judge-only, with that caveat carried in the sentence that states it, until a real human-rating
pass exists and the validation study clears its pre-registered threshold on a given dimension.

---

## Later decisions

`DECISIONS.md` D1–D5 (2026-08-24, orchestrating session) — theme taxonomy (seven, not ten), equity
stratum membership (SC-077 and SC-010 reclassified out), the equity knowledge-base re-extraction
(approved, sequenced after corpus fixes, not yet run as of `docs/limitations.md`'s last update),
dropping the human-verification claim, and describing the `racial_ethnic` axis as narrower than its
name — are recorded there, not duplicated here. `docs/limitations.md` and `docs/preregistration.md`
both cite the specific decisions that affect their content directly.

Future entries in this log should be dated, should state the decision in one sentence before the
reasoning, and should link the commit or file that encodes it — the pattern followed above.
