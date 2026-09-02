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
frontier models. Digests, not tags, are what get recorded per generation (v3 §16), since
a tag is mutable and can mean different weights tomorrow. Each backend's `resolve_digest` asks the
serving stack what it is actually serving as the run starts, `runner.assert_digests_resolved`
refuses to write a single cell if the stack will not say, and `generation.model_digest` carries the
answer per row. **(Corrected 2026-09-01.)** This entry previously named `make pin-models` as that
mechanism. The target invoked `carelite.models.pin`, a module that never existed, so it had always
failed; it was removed rather than implemented, because a digest written into the running config is
not a record the run ever reads back and the persisted column always did the work.

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

`DECISIONS.md` D1–D13 (2026-08-24 through 2026-09-01, orchestrating session and project owner) are
recorded there, not duplicated here; `docs/limitations.md` and `docs/preregistration.md` cite the
specific decisions that affect their content directly, at the point they affect it. In brief, for
orientation: theme taxonomy (D1, seven not ten); equity stratum membership (D2); the equity
knowledge-base re-extraction, approved and — per its own outcome entry — run with a negative result:
zero net new equity entries, because the mechanism is a property of the corpus (D3); dropping the
human-verification claim (D4); the `racial_ethnic` axis described as narrower than its name (D5);
`README.md` assigned to this lane (D6); Condition LC redefined as `LC-sample` because the corpus
does not fit the context window (D7); NURSE `respect` and `support` having zero knowledge-base
grounding (D8); six analysis specifications the pre-registration left open, settled (D9); **OSF
pre-registration dropped — this project is a local proof of concept and every result is descriptive
(D10),** which is the decision that changes how every other document in this project may describe
its own findings, this one included; Condition LC stopped at 39 of its planned 180 holdout cells
after costing ~33× the other conditions per cell on rented GPU hardware, the second lane to reach
that conclusion independently (D11); and `generation.gate_blocked` added so a response the output
safety gate refused is visible and excludable rather than silently scored as though it had passed —
17 of 939 holdout generations, 13 of them on one scenario (D12); and **D11's cost premise tested
against a second serving stack and not surviving it (D13)** — vLLM with prefix caching generates an
LC cell in 3.61 s warm against Ollama's 198 s, a 54.9× difference, so Condition LC was completed in
full at 180 cells and `generation` now holds 1,119 rows.

**D13 bears directly on the 2026-08-22 model-roster entry above and on the reasoning behind it, so
the qualification is recorded here rather than left for a reader to notice.** That entry argued the
roster from "everything runs local and offline," and the argument's substance — open weights, no
hosted vendor model, nothing metered per token — is intact: the vLLM route serves the same
open-weight family from a pod the project starts, pins by commit sha, and deletes. What is no
longer literally true is the word *local*. One condition's 180 cells were served from a rented GPU
over an authenticated endpoint, with the generation loop and the database still on the operator's
machine. `generation.served_by` exists so that distinction is a column rather than a recollection,
and `docs/limitations.md` §6 states the ceiling in terms of open weights rather than locality for
the same reason.

**D13 also corrects something this log helped propagate: a runtime measurement was twice mistaken
for a method result.** D11 and the judge lane independently concluded that a long-context baseline
was unaffordable at this scale, and their agreement was read as confirmation. Both had measured
Ollama. The limitation that survives is about a serving stack that re-prefills a shared prefix on
every request, not about long-context evaluation; `docs/limitations.md` §4 now says so in those
terms. Recorded here because the general form is worth carrying forward — when a cost measurement
closes a question, the runtime is a variable in it and has to be named as one.

Future entries in this log should be dated, should state the decision in one sentence before the
reasoning, and should link the commit or file that encodes it — the pattern followed above.
