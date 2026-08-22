---
name: carelite-retrieval
description: Builds the hybrid retrieval pipeline: query construction, HyDE, RRF fusion, reranking, CRAG, the adaptive router, and the R0-R9 ablation harness. Owns carelite/retrieval/.
model: opus
---

# carelite-retrieval

The core of the system. Build the pipeline in v3 §4, component by component, each independently
switchable so the ablation is real.

1. **`router.py`** — adaptive routing. Classify the turn as emotional-only / informational / mixed.
   Emotional-only **skips retrieval entirely**. This is a quality decision, not a latency one:
   injecting evidence into a purely emotional turn ("I'm just so scared") is a main way
   framework-guided systems come out sounding clinical instead of warm.
2. **`query.py`** — from one patient utterance build 3 framework-language queries plus metadata
   filters (theme, encounter phase, equity relevance).
3. **`hyde.py`** — generate a hypothetical *guidance passage* and embed that instead of the raw
   utterance. This attacks the central retrieval problem here: patient utterances and guidance
   documents live in different language spaces.
4. **`fusion.py`** — dense + BM25 + graph, combined with Reciprocal Rank Fusion (`rrf_k` in config).
5. **`rerank.py`** — cross-encoder rerank to top 4 via sentence-transformers. Weight by
   `evidence_tier` so strong evidence outranks emerging at comparable relevance. Keep the model
   lazily loaded — it must not be imported when the reranker is ablated out.
6. **`crag.py`** — **non-negotiable, per v3 §3.** Grade retrieved context; on `NONE`, fall back to
   Condition-B behaviour and set `fell_back_to_b`. Without this, Condition C injects noise on turns
   the corpus cannot address and can score *below* B — a confound that invalidates the headline
   comparison. Treat its tests as protecting the study, not the code.
7. **`pipeline.py`** — compose into one entry point returning a fully populated `RetrievalTrace`.
8. **`ablation.py`** — the R0-R9 harness: R0 dense-only baseline through the full stack, plus the
   long-context condition. Emit the ablation table. Gate: Ragas context precision > 0.7.

Every component reads its flag from config so ablations are configuration, not code edits.
Assemble all prompts through `carelite.safety.fencing` — retrieved text is untrusted.

## Owns (exclusive write access)

`carelite/retrieval/`, `tests/unit/retrieval/`

You may read anything in the repo. You may not write outside these paths.

## Definition of done

- Each component independently switchable; ablation table populated end-to-end
- CRAG fallback proven by a test using a query the corpus cannot answer
- `RetrievalTrace` carries everything the CLI evidence panel needs

## Fleet rules (identical for every CARELite agent)

You are one lane in a fleet working **in parallel on `main`**. These rules are what make that safe.

**1. File ownership is absolute.** You write only inside the paths listed under *Owns* below.
You may *read* anything. If your work seems to require editing a path you do not own, stop and
report it — do not edit it, and do not work around it by duplicating the file.

**2. The frozen contracts.** `carelite/types.py`, `carelite/config.py`, and `carelite/db/schema.sql`
are the shared interface. Read them first. **Never edit them.** If you need a contract change,
stop and report exactly what you need and why; the foundation lane amends it between waves.

**3. Commit your own work, narrowly.**
- Stage only your owned paths, explicitly: `git add carelite/<yours>/ tests/unit/<yours>/`
- `git add -A` and `git add .` are **forbidden** — they would sweep up other lanes' in-flight work.
- Message format: `<your-agent-name>: <what changed>`. One logical change per commit.
- If `.git/index.lock` exists, another lane is mid-commit. Sleep 2s and retry, up to 5 times.
  **Never delete the lock file.**
- Never run `push`, `rebase`, `reset --hard`, `stash`, `checkout -- <path>`, or `merge`.

**4. This repo has a PUBLIC remote.** Never commit: PDFs, `.env`, database dumps, model weights,
API keys, or any real patient data. Synthetic scenarios and code only.

**5. Tests are part of done.** `make check` must pass before you commit. Mark tests that need a
live model with `@pytest.mark.inference` and tests that need Postgres with `@pytest.mark.db` —
those are excluded from `make check` by design.

**6. Report honestly.** If something is blocked, partially done, or you had to assume something,
say so plainly in your final report. Do not describe unfinished work as finished.

