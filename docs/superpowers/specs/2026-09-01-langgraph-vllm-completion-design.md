# Design — LangGraph, LangChain, and the vLLM completion of condition LC

**Date:** 2026-09-01
**Status:** approved, ready for implementation
**Source:** `.claude/Additional_builds.md`, reconciled against the working tree and the database

> **Ownership note.** `docs/` belongs to `carelite-repro`. This subtree,
> `docs/superpowers/specs/`, holds orchestrating-session design records rather than
> project documentation, and `carelite-repro` neither maintains nor is bound by it.
> Nothing here is a deliverable; the deliverables it produces land in the lanes named
> below.

---

## 1. What this is, and what it is not

`.claude/Additional_builds.md` proposes building a LangGraph six-condition pipeline,
hybrid RAG, an LLM-as-judge, and a mixed-effects analysis. Almost all of that already
exists. The document was written against an assumed state of the repository that the
repository passed some time ago, which is the failure mode `CLAUDE.md` warns about
directly: the planning documents undercount and go stale; the tree and the database are
the authority.

Measured state at the time of writing, from the database rather than from memory:

| Claimed missing | Actual |
|---|---|
| six conditions to build | `A, A2, B, C, LC, D` exist as configuration over one code path |
| hybrid RAG to build | HyDE, RRF fusion, cross-encoder rerank, CRAG, adaptive router, R0–R9 ablation |
| property graph | 715 rows in `graph_edge`, NetworkX traversal live |
| judge and rubric | cross-family `gpt-oss:20b` judge, 939 rows in `rubric_score` |
| "reconcile the 939 figure" | `SELECT COUNT(*) FROM generation` = 939 |
| "prove +0.67, p < 0.001" | MixedLM reports `B vs A +0.6724 [+0.5408, +0.8040], p = 1.33e-23` |

The headline numbers therefore need no reconstruction. They are already in Postgres,
and this work is built forward from the code rather than backward from a claim.

**Four gaps are real**, and they are what this design addresses.

1. **LangGraph is not installed.** `carelite/generate/graph.py` declares its topology
   once in `NODES`/`EDGES` and compiles it two ways: `to_langgraph()` for the real
   library, `_Executor` for its absence. Only the second path has ever run, because
   `langgraph` is absent from `pyproject.toml` and the orchestrator lane does not own
   that file. The statement "LangGraph orchestrates a stateful six-condition pipeline"
   is not true today.
2. **LangChain is not used at all.** Retrieval is hand-rolled and more capable than the
   `EnsembleRetriever` sketch in the source document.
3. **There is no vLLM backend and no `served_by` column.** The generation client speaks
   only to Ollama, so no row in `generation` records which serving stack produced it.
4. **Condition LC stands at 39 of 180 cells.** `runs/holdout/lc.log` shows it stopped
   mid-run at roughly 2.2 minutes per cell.

## 2. The finding that motivates the expensive half of this work

D11 dropped condition LC for cost: 3.3 minutes per cell on an L40S against 6 seconds
for the A/A2/D group, with about 8.5 hours and $8.50 left to run. That decision is
settled and was correct on its evidence.

D11 also states the cause precisely. Every LC prompt shares an identical ~119,500-token
prefix, because `lc_sample()` is query-independent and deterministic by design under D7.
That prefix should be nearly free after the first prefill through KV cache reuse. The
measured per-cell time says Ollama re-prefills it on every request instead. In D11's own
words: *the design anticipated the saving; the runtime does not deliver it.*

**That is a statement about Ollama, not about long-context evaluation.** vLLM's
automatic prefix caching is the mechanism whose absence D11 measured. If it delivers
what Ollama did not, D11's cost premise does not survive, and secondary outcome 3 —
C vs LC, "does query-dependent selection beat a fixed context" — becomes testable again.

**D11 is not re-opened on this reasoning alone.** The project owner's decision is to
measure first: benchmark ten LC cells under vLLM with prefix caching enabled, report the
measured cost per cell, and amend D11 only against that measurement. If prefix caching
does not help, D11 stands and its premise has been upgraded from an inference about
Ollama's behavior to a verified cross-runtime finding. Either outcome is a result.

## 3. Architecture: local driver, remote inference

The generation loop runs on the Mac against local Postgres. Only model serving moves to
the pod.

```
  Mac                                        RunPod pod
  ---                                        ----------
  carelite.generate.runner   --HTTPS-->      vLLM  :8000
    |                                        google/gemma-4-12B-it
    |                                        --enable-prefix-caching
    v                                        --api-key $VLLM_API_KEY
  Postgres 18.6 (localhost)
```

The alternative designs are worse for reasons that are not primarily about convenience.
Exposing Postgres to the pod means putting the system of record on the public internet.
A reverse SSH tunnel means the run dies with the tunnel. Moving Postgres to the pod means
a `pg_dump`/`pg_restore` merge, which is an opportunity to lose or duplicate rows in the
table the entire study is read from. Driving from the Mac costs one HTTPS round trip per
cell and keeps every write local and transactional.

Sending the ~119.5k-token LC prompt over the wire on each of 180 requests is roughly
500 KB per request and under 100 MB in total. That is not the bottleneck; prefill is.
Prefix caching is server-side and unaffected by where the client sits.

**The vLLM server is authenticated.** RunPod proxy URLs are public and unauthenticated
by default, and an open inference endpoint is abuse-prone. The server runs with
`--api-key` set to a locally generated token that lives in `.env` and is never committed,
logged, or passed on a command line that gets recorded.

## 4. Work packages

W1, W2 and W3 are independent and run in parallel. W4 needs W3's client. W5 needs W4's
generations. W6 needs W5's scores.

### W1 — LangGraph made real (`carelite-orchestrator`)

`to_langgraph()` already compiles the same `NODES`/`EDGES`/`CONDITIONAL_TARGETS`
declaration that `_Executor` walks. It has never executed.

- `langgraph` and `langgraph-checkpoint-postgres` added to `pyproject.toml`
  (by the orchestrating session — the lane does not own that file).
- **A differential test is the deliverable, not the dependency.** Over a fixed set of
  seeded states spanning every branch — safety-blocked, retrieval on and off,
  self-check on and off, generation failure — `_Executor` and `to_langgraph()` must
  produce identical final states. This is what makes "one topology, two compilers"
  a property of the code rather than a comment in its docstring.
- The Postgres checkpointer is wired into the runner so an interrupted run resumes.
  Resumability is already a requirement of the runner; this makes it survive a crash
  mid-graph rather than only between cells.
- `build_graph()` keeps its fallback. A machine without `langgraph` must still run the
  system, because the 939 existing rows were produced that way.

### W2 — LangChain retrieval adapter (`carelite-retrieval`)

- `carelite/retrieval/langchain_adapter.py`: a `PGVector`-backed dense retriever, a
  `BM25Retriever` over the same text, and an `EnsembleRetriever` fusing them, reading the
  existing `chunk` and `kb_entry` tables — not a second copy of the corpus.
- **An equivalence test is the deliverable.** Over a fixed query set, report overlap@k
  and rank correlation between the adapter and the native pipeline. A number, not an
  assertion.
- **The adapter is not the default and does not touch the study.** It is selectable
  behind a flag. The native stack produced all 939 rows and continues to. The claim this
  makes true is "hybrid RAG via LangChain is implemented and measured against the native
  pipeline", which is what the code will then support.

### W3 — vLLM backend and `served_by` (`carelite-orchestrator`, plus schema)

- `GenerationClient` gains an OpenAI-compatible path selected by
  `CARELITE_BACKEND=ollama|vllm`, reading `VLLM_BASE_URL` and `VLLM_API_KEY` from the
  environment. The graph, the nodes, the conditions and the runner do not change.
- `generation.served_by text` added to `db/schema.sql`; the existing 939 rows backfilled
  to `'ollama'`, which is what they are.
- `model_digest` semantics: Ollama's digest identifies a GGUF; vLLM serves HF
  safetensors. These are different artifacts of the same model family and the schema must
  not imply otherwise. `served_by` is what distinguishes them.

### W4 — Provision, benchmark, then decide (orchestrating session)

- One RTX PRO 6000 Blackwell 96 GB, community cloud, $1.69/hr, ~$25 authorized ceiling.
  96 GB holds `gemma-4-12B-it` in bf16 (~24 GB) plus a single cached ~119.5k-token prefix
  with room for concurrent decode.
- vLLM serves with `--enable-prefix-caching`, `--max-model-len 131072`, `--api-key`.
- **Gate: benchmark ten LC cells and report measured seconds and dollars per cell before
  generating the remaining cells.** This is the D11 measurement, and it is a stop point,
  not a formality.
- If the measurement justifies it, all 180 LC cells are generated under vLLM.
- The pod is terminated when the run completes. It is not left idling.

### W5 — Judge the new cells and test the backends against each other (`carelite-judge`)

- Score the new LC generations with the existing cross-family `gpt-oss:20b` judge,
  through the existing cache and grounding checks. The judge is unchanged; only its input
  set grows.
- **Backend equivalence.** The 39 pre-existing LC cells were served by Ollama. Re-running
  them under vLLM at identical scenario and seed produces 39 paired observations across
  two serving stacks — the check §8.6 of the source document asks for, as data rather
  than as an assurance. Report agreement; do not pool arms that disagree.
- The LC analysis arm is single-backend: `served_by = 'vllm'`, 180 cells. The 39 Ollama
  cells are retained as the equivalence sample and excluded from the arm.

### W6 — Analysis, figures, and the D11 amendment (`carelite-stats`, `carelite-viz`, `carelite-repro`)

- Secondary outcome 3, C vs LC, computed for the first time.
- Backend equivalence added as a sensitivity analysis.
- Figures regenerated; `make reproduce` runs clean from the database.
- **`make reproduce` emits the run's headline numbers as text derived from the database**
  — generation count, the B-vs-A coefficient and its interval — so no downstream document
  can carry a stale figure forward. This is the direct fix for the failure mode that
  produced `Additional_builds.md`.
- D11 amended by the orchestrating session with the measurement, whichever way it falls.

## 5. What changes in the reported numbers, and what does not

**`generation` grows from 939 rows to about 1,119** — 939 existing, plus 180 vLLM LC
cells, retaining the 39 Ollama LC cells as the equivalence sample. Any prose carrying
"939 generations" becomes wrong at that moment and must be regenerated from the database,
which is what the `make reproduce` change in W6 enforces.

**The +0.6724 does not move.** It is the B-vs-A contrast on the composite NURSE outcome.
Condition LC does not enter it, and no LC cell is scored into it.

**The instrument does not improve.** `ie`, `naturalness` and `ritualistic` are degenerate
on this run and remain so; the judge validation study has not run and every result stays
labelled EXPLORATORY. Completing LC adds a comparison. It does not upgrade the evidential
status of any existing one, and nothing in this work should be described as if it did.

## 6. Conditions are not renumbered

`Additional_builds.md` proposes an A–F ladder in which D is graph traversal, E is a
self-check loop, and F is the long-context baseline. The repository already places the
graph inside condition C through `retrieval/retrieval_hook.py`, already runs the
self-check for B, C and LC, and uses D as the deliberately degraded negative control
whose job is to prove the rubric can tell a bad response from a good one.

Renumbering would invalidate 939 rows keyed on the existing condition labels and would
destroy the negative control that D8's and D11's analysis depends on. The existing
labels stand.

## 7. Security constraints

- No credential is written to a tracked file. `.env` is gitignored; `detect-secrets`
  runs in `.githooks/pre-commit`.
- The RunPod API key stays in the MCP client. It is not read from, written to, or copied
  into the repository.
- The vLLM endpoint requires `--api-key`; the token is generated locally, stored in
  `.env`, and never echoed into a log or a command recorded in run output.
- No inbound network exposure. Postgres stays on localhost and the pod initiates no
  connection to the Mac.
- No patient data is involved at any point: scenarios are authored, not derived from
  records. The pod receives scenario text and knowledge base passages only.
- The pod is terminated at the end of the run.

## 8. Definition of done

- `make check` passes.
- The differential test proves `_Executor` and the compiled LangGraph agree on every
  branch.
- The LangChain adapter's overlap against the native pipeline is a reported number.
- Every row in `generation` carries a `served_by` value.
- The D11 benchmark is measured and reported, and D11 is amended to match — whether that
  amendment completes LC or confirms the original decision on stronger evidence.
- `make reproduce` regenerates every figure and the headline numbers from the database.
- No pod is left running.
