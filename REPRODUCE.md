# Reproducing CARELite AI

Cold-start instructions for a machine that is not the one this project was built on. Follow the
sections in order — each one is a gate for the next. Runtimes are stated honestly, including the
multi-hour inference lanes; do not expect this to finish in an afternoon.

**Before you start anything here, read `docs/preregistration.md`.** It is an analysis plan, not an
active registration — per `DECISIONS.md` D10 this project is a local proof of concept and OSF
registration was dropped by decision, so there is no registration gate to satisfy before §6/§7's
holdout generation runs. What still applies: every result this reproduction produces is
**descriptive**, not confirmatory or pre-specified in a registered sense, however precisely the
analysis plan fixed its thresholds in advance — say so in anything you write up from this run.

**Hardware requirement, stated plainly because it is not optional: this experiment does not
reproduce on a consumer laptop in reasonable time.** The judge lane measured ~21 minutes per
Condition-LC generation on local hardware and projected ~59 hours for that condition alone; even on
a rented 48 GB GPU, LC still cost ~33× the other conditions per cell (`DECISIONS.md` D11) and was
stopped after 39 of its planned 180 cells for exactly that reason. **This project's own holdout run
was executed on a rented Runpod L40S (48 GB VRAM), not a local machine.** If you are reproducing
this on a laptop-class GPU (24 GB or less), plan on Condition LC being effectively unaffordable at
this scale even though the other five conditions (A, A2, B, C, D — 180 cells each, the study's
primary comparisons) are tractable; §7 has the measured per-cell numbers.

---

## 0. What you're reproducing

Corpus retrieval → knowledge-base extraction → index build → scenario-conditioned generation across
six experimental conditions → LLM-as-judge scoring → statistics → figures and tables. Every stage
writes to one PostgreSQL database, and `make reproduce` at the end regenerates every figure and
table from what's in it — it does not re-run inference. Re-running inference is a separate, much
longer step described in §6–7. **This project's own run completed at 939 generations, not the
1,080 originally specified** — Condition LC was stopped by decision (`DECISIONS.md` D11), not by
failure; see §7 and `docs/limitations.md` §4 for the full account.

---

## 1. Toolchain

- **Python 3.13** exactly (`pyproject.toml`: `requires-python = ">=3.13"`, and `mypy`/`ruff` are
  configured against 3.13 specifically — an older interpreter will pass `uv sync` and then fail
  mypy in ways that look unrelated).
- **[uv](https://docs.astral.sh/uv/)**, used for both the virtualenv and the lockfile. This
  project was built against `uv 0.12.5`; any uv new enough to support `uv sync --extra dev` against
  a `uv.lock` file works.
- **Ollama**, for local model serving. Built against `ollama 0.32.15`. Install from
  [ollama.com](https://ollama.com) or your platform's package manager.
- **PostgreSQL with the `pgvector` and `pg_trgm` extensions**, version details in §3.
- macOS (Apple Silicon) and Linux are both fine; nothing here depends on macOS specifically except
  that the EDB Postgres installer path in §3 is macOS-specific. On Linux, install Postgres and
  pgvector via your distribution's packages or the pgvector project's own instructions.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv isn't already installed
uv --version                                       # confirm
```

---

## 2. Clone and sync the environment

```sh
git clone <repo-url> carelite-ai && cd carelite-ai
uv sync --extra dev
```

This creates `.venv/` from `uv.lock` exactly — **not** from `pyproject.toml`'s loose version
ranges. `uv.lock` is what makes this reproducible; do not delete it and re-resolve unless you are
deliberately updating dependencies, and if you do, that is a change worth its own commit, not a
side effect of "getting reproduce to run."

```sh
make install     # equivalent to the uv sync above, via the Makefile
make check       # lint + typecheck + unit tests, no model, no DB — should pass clean on a fresh clone
```

If `make check` does not pass on a clean clone before you've touched anything, stop here and file
that as a bug rather than working around it — everything downstream assumes it does.

---

## 3. PostgreSQL + pgvector

**Pin exact versions, not "recent."** This project was developed against **PostgreSQL 18.6** and
**pgvector 0.8.6**. `docs/limitations.md` §5 records that this is a deliberate deviation from build
plan v3's original assumption of the Homebrew `postgresql@17` package — the EDB installer was used
instead because it was available and because pgvector 0.8.6 was compiled from source against it.
Both paths are documented below; pick one and pin the versions.

### Option A — Docker (the build plan's original recommendation, §6: "one `docker compose up`")

No `docker-compose.yml` ships in this repository yet — this project's own development environment
used Option B below, so Option A is **documented but not verified end-to-end in this repo**. If you
set it up, use the official `pgvector/pgvector:pg18` image (or `pg17` to match the build plan
exactly) so the extension version matches what was tested here:

```sh
docker run -d --name carelite-pg -p 5432:5432 \
  -e POSTGRES_PASSWORD=carelite -e POSTGRES_USER=carelite -e POSTGRES_DB=carelite \
  pgvector/pgvector:pg18
```

### Option B — Native install (what this project's own environment used)

macOS, via the [EDB PostgreSQL installer](https://www.postgresql.org/download/macosx/) for
PostgreSQL 18, then pgvector built from source against it:

```sh
git clone --branch v0.8.6 https://github.com/pgvector/pgvector.git
cd pgvector
export PG_CONFIG=/Library/PostgreSQL/18/bin/pg_config   # adjust to your install path
make && sudo make install
```

On Linux, your distribution's `postgresql-18` and a matching `pgvector` package (or the same
build-from-source steps against your `pg_config`) are equivalent.

### Either way: create the role, database, and schema

```sh
createuser carelite -P                       # set a password when prompted
createdb carelite -O carelite
cp .env.example .env
# edit .env: CARELITE_DATABASE_URL=postgresql://carelite:<password>@localhost:5432/carelite
make db-up      # applies carelite/db/schema.sql (CREATE EXTENSION vector, pg_trgm; all tables)
make db-check   # wave-0 gate: extension present, all expected tables present, a 3-way join returns
```

`make db-check` must show every line as `PASS` before continuing. A missing `pgvector` extension is
the most common failure here and shows up as a `FAIL` on the extension line specifically, not as a
connection error.

---

## 4. Ollama models, pinned by digest

**Tags are mutable — pin the digest, not the tag.** Pull every model in the roster
(`carelite/config.py.Models`):

```sh
ollama pull gemma4:12b       # primary generator (conditions A, B, C, D, LC)
ollama pull qwen3.5:9b       # cross-model baseline (condition A2)
ollama pull gpt-oss:20b      # judge — deliberately a different model family from the generators
ollama pull bge-m3           # embedder
```

The reranker (`BAAI/bge-reranker-v2-m3`) is a `sentence-transformers` cross-encoder, not an Ollama
model — it's pulled automatically on first use once you `uv sync --extra rerank` (see §5).

Record the digest actually pulled — this is what gets stored on every generation per build plan v3
§16, because a tag can point to different weights tomorrow than it does today:

```sh
curl -s http://localhost:11434/api/tags | python3 -c "
import json, sys
for m in json.load(sys.stdin)['models']:
    print(f\"{m['name']:20s} {m['digest']}\")"
```

`make pin-models` is intended to do this automatically and record it against `ModelSpec.digest` in
the running config, but **`carelite.models.pin` does not exist in this repository as of this
writing** — the Makefile target references a module that has not been built yet. Until it lands,
run the `curl` above manually and record the digests in your run notes; a generation's persisted
`model_digest` column is what actually matters for reproducibility, not this convenience wrapper.

Digests observed during this project's own development run (yours will very likely differ — these
are recorded as an example of the format, not as values to expect to match):

| model | digest (truncated) |
|---|---|
| `gemma4:12b` | `sha256:4eb23ef187e2c546256…` |
| `qwen3.5:9b` | `sha256:6488c96fa5faab64bb6…` |
| `gpt-oss:20b` | `sha256:17052f91a42e97930aa…` |
| `bge-m3` | `sha256:7907646426070047a77…` |

---

## 5. Install the reranker extra and rebuild the corpus

```sh
uv sync --extra dev --extra rerank    # pulls sentence-transformers + torch for the cross-encoder
```

Rebuild the paper corpus. This step is network-bound and polite by design (`time.sleep(1)` between
manifest rows in `carelite/corpus/fetch.py`) — expect roughly **one to two minutes** for 43 unique
DOI lookups, most of it idle wait, not compute:

```sh
uv run python -m carelite.corpus.fetch --email you@example.com
```

Expect **33 of 43 unique manifest DOIs to resolve**; the other 10 are genuinely paywalled (Unpaywall
and the NCBI ID-converter both report `idIsNotOpenAccess`) and land in
`data/pdfs/_manual_needed.csv` rather than silently failing. This is the corpus's actual, documented
shape — see `docs/limitations.md` §1 — not a sign that the rebuild went wrong. Re-running is safe;
already-downloaded PDFs are skipped.

Then run extraction, chunking, contextualization, and load into Postgres — the concrete entry
points for this live in `carelite/corpus/` and `carelite/kb/`; consult those modules' own docstrings
for the current command names, since this pipeline is still being actively corrected
(`docs/limitations.md`'s "Status" note) and a command documented here today could drift from what
those lanes ship. As a floor: after this step, `make db-check` should report the `paper` and
`kb_entry` tables populated, and

```sh
uv run python -c "from carelite.db.connection import fetch_all; print(fetch_all('select count(*) from paper')); print(fetch_all('select count(*) from kb_entry'))"
```

should report figures in the neighborhood of 33 papers and roughly 120–130 knowledge-base entries —
not the README's early "15 sample entries" or build plan v3's planning-time "45," both of which
were superseded once the corpus was actually processed (`docs/decisions/README.md`, "Knowledge base
re-derived from the actual corpus").

---

## 6. Build the index

Dense (pgvector HNSW) and lexical (Postgres full-text, `tsvector`/`tsquery`) indexes are built over
`chunk` and `kb_entry` by `carelite/index/`. This is a local, GPU-optional, CPU-bound embedding pass
over roughly 1,500–2,000 text units (chunks plus KB entries) using `bge-m3` via Ollama — expect on
the order of **10–20 minutes** on a modern laptop CPU, faster with a GPU-backed Ollama install.

**One thing worth knowing before you run this, because it will look like a bug otherwise:**
`bge-m3` is deliberately embedded with **no query or document prefix** (`""` for both) in this
project's retrieval config, against that model's documented usage pattern. Adding even the
7-character prefix `"query: "` was measured during this project's own index build to raise cosine
similarity between unrelated one-word queries from 0.52 to 0.72–0.84 and flatten rankings entirely.
If you patch a prefix back in because a tutorial told you to, retrieval quality will degrade
silently rather than error. See `docs/limitations.md` §5.

**Verified state of this project's own index build, as a reference point for your own run:**
471/471 chunks embedded; the 342 embeddings that predated the corpus lane's extraction fixes were
confirmed byte-identical against a fresh embed (0 mismatches) rather than silently left stale;
mean pairwise cosine across the corpus is 0.5788 with no discontinuity between old and new
embeddings; and 10/10 hand-picked retrieval probes pass, including two that only pass because the
extraction fixes landed first (`teach-back` now matches inside the Talevski systematic review,
`disparit` inside the PLOS empathy-disparities paper). If your own run's numbers look wildly
different, check that the corpus and KB steps in §5 actually completed before this step ran, not
that this step is broken.

**One reconciliation trap, not a bug:** `index_embedding_state` legitimately carries a few more
rows for `kind='chunk'` than `chunk` has rows — 475 against 471 as of this writing — left over from
chunk-ID renumbering. Scoped (`only_ref_ids`) embedding calls skip orphan pruning by design, since
a targeted call has no way to know a stale ID was ever valid; only a whole-table pass sweeps them.
It's harmless to retrieval and only worth knowing so you don't spend an hour chasing a row-count
mismatch that isn't a bug.

**Condition LC is `LC-sample`, not the whole corpus stuffed into context, per `DECISIONS.md` D7 —
know this before you run §7's generation step.** 471 chunks is ~326,526 tokens against a
128,000-token context window (255% utilisation), so build plan v3 §3's original LC specification is
not implementable against this corpus. What actually runs is a fixed, **query-independent**
round-robin selection across all 33 papers at a pinned seed
(`carelite.retrieval.ablation.lc_sample`): 169 chunks, 35.9% of the corpus. Round-robin rather than
random sampling guarantees every paper is represented, so LC's content is not an accident of the
seed. **Any selection rule is itself a form of retrieval** — LC-sample therefore does not test
"retrieval vs. no retrieval" against Condition C. It tests whether Condition C's *query-dependent*
selection beats LC-sample's *fixed* context, which is a real question and a narrower one. See
`docs/preregistration.md` §2 and §4 for the full statement this document only summarizes.

---

## 7. Generate and score

**This is the multi-hour part. Plan for it — do not start it expecting to watch it finish in one
sitting.**

### Smoke test first

```sh
make eval-smoke    # 5 scenarios x all 6 conditions, end to end — minutes, not hours
```

`carelite/eval/smoke.py` (landed in commit `b9eacd3`) drives the real runner, the real graph, the
real prompts, and the real safety gate over 5 scenarios × 6 conditions, then audits the rows it got
back rather than just checking that something came back. **It defaults to the train split**, writes
to a JSONL journal under `runs/smoke/`, and truncates that journal on every invocation — never the
`generation` table — because the generation cache would otherwise turn a second run into a no-op
that reports success while testing nothing.

**Named hard failures (non-zero exit):** a condition producing no rows; an empty response; a row
carrying `DIGEST_UNAVAILABLE` as its model digest; a `prompt_id` that disagrees with what its
`ConditionSpec` actually specifies; any pipeline node degrading instead of raising; **condition C
retrieving nothing on every scenario**; retrieval leaking into a condition that is supposed to have
none; an empty long-context pack; a self-check that ran when configured off, or the reverse; and two
conditions emitting byte-identical text on every shared scenario. **Warnings that do not fail the
run:** an uncommitted prompt, an input the safety screen escalated, a truncated long-context pack,
partial (rather than total) retrieval failure on condition C.

It also prints the router's route-per-scenario, with a marker beside every `emotional_only` route,
and fails hard if *every* scenario routes that way — an `emotional_only` route retrieves nothing, so
a run in that state finishes with the right row count and no errors while condition C has silently
collapsed into condition B, and the route printout is the only place that shows.

**As of this writing, `make eval-smoke` has never been run against a live model.** It is verified
against fakes and `--dry-run` only — inference was deliberately held off while the retrieval lane's
R0–R9 ablation held the Ollama daemon. Treat it as implemented and not yet exercised, not as an
executed step, until that changes.

```sh
uv run python -m carelite.generate.runner --limit 5 --dry-run   # plan and count only, no model calls
```

### Full evaluation run — what this project actually did, and how to repeat it

**No registration gate applies — `DECISIONS.md` D10 dropped OSF registration by decision.** Every
response this step generates is analysis data from a single rented-GPU run, and any conclusion
drawn from it is reported as descriptive, not as a pre-specified finding.

**Generate with `--store jsonl`, not straight to Postgres — this is not a style preference.** An
earlier attempt on this project wrote directly to Postgres running on the rental pod's container
disk; the container was restarted mid-run and **~863 already-produced generations were lost**
(`carelite/generate/load.py`'s own docstring records this). The journal files under `runs/holdout/`
are the durable artifact of the experiment; Postgres is loaded from them afterward, not written to
live:

```sh
uv run python -m carelite.generate.runner --store jsonl \
    --journal runs/holdout/generations-b.jsonl --conditions B --split holdout --register-prompts
```

**This project ran four such invocations in parallel, split by condition** (one process each for
the fast group A/A2/D, for B, for C, and for LC — matching "four parallel workers by condition" in
the record), each writing its own journal file. `--dry-run` first to see the planned cell count
without generating anything; `--limit N` to scope a partial run while testing. Every cell is keyed
by `(scenario_id, condition, prompt_version, model_digest, seed, sample_idx)` and already-generated
cells are skipped on rerun, so an interrupted invocation resumes rather than restarts. Then bridge
every journal into Postgres:

```sh
uv run python -m carelite.generate.load runs/holdout/generations-*.jsonl
```

**What this project's own run actually produced: 939 generations, zero failures at generation
time**, across A/A2/B/C/D at 180 cells each (all 60 holdout scenarios × 3 samples) plus 39 partial
LC cells. **Condition LC does not complete in this pipeline as originally specified — expect to stop
it.** Measured on a rented L40S: ~3.3 minutes per LC cell against ~6 seconds for the fast group,
roughly 33×, which is why `DECISIONS.md` D11 stopped it at 39/180 cells rather than let it run to
completion at a projected extra 8+ hours and several dollars for one condition. `docs/limitations.md`
§4 has the full account, including why the cost does not shrink with a smaller token budget (the
serving stack re-prefills LC's shared prefix on every request rather than reusing it via KV cache,
which the design assumed and the runtime did not deliver) and exactly which comparisons are and are
not affected by LC's incompleteness. Budget accordingly if you intend to complete LC yourself: the
other five conditions are tractable on a single rented mid-tier GPU in a few hours; LC at full scale
is not, on this serving stack, regardless of hardware tier.

**Judging** (`python -m carelite.eval.judge.holdout`) now has a real CLI — it did not when this
document was last drafted, and `carelite/eval/judge/runner.py`'s bare `judge_generations` function
is the library call it wraps, not the entry point to use directly:

```sh
uv run python -m carelite.eval.judge.holdout \
    --glob 'runs/holdout/generations-*.jsonl' --out runs/holdout/judge --workers 8
```

Single-pass at temperature 0 deliberately, not the 5-sample self-consistency regime, which stays
scoped to the smaller `carelite.eval.judge.study` validation subset (`docs/preregistration.md` §9
explains why that split is safe). **This project's own judging run: 939/939 judged, zero errors,
206 minutes**, with results written to `runs/judge-holdout/rubric_scores.jsonl` and a manifest at
`runs/judge-holdout/manifest.json` carrying the CRAG-fallback rate, the LC-partial flag, and the
output-gate-blocked breakdown — read the manifest before reading the scores; it has the caveats the
raw rows don't carry on their own. `--workers` runs a thread pool over generations (the model is
resident and requests are independent, so this is a real speedup, not oversubscription); resumption
is automatic via `JudgeCache`, keyed by generation, model digest, prompt version, rubric version,
temperature, sample index, and anchor order. `--skip-lc` drops the partial LC rows entirely if you
would rather not carry them downstream at all.

### Human rating (separate track, not blocking `make reproduce`)

Per `docs/preregistration.md` §12: 60 responses, blinded, calibration set first. As of this writing
no human rating has occurred; `carelite/eval/human/` is exercised against synthetic rater data
(`carelite/eval/human/synthetic.py`) so the harness is proven before a real rater's time is spent.
This step is manual and does not have a `make` target — see `carelite/eval/human/packet.py` for how
a rating packet is generated.

---

## 8. Analysis: `make reproduce`

```sh
make reproduce
```

Runs `python -m carelite.repro`, which regenerates every figure and table **from the database** —
it performs no inference and makes no model calls, so it is safe to re-run repeatedly and is fast
(seconds to low minutes) regardless of how long §7 took. It is the only step in this document that
is idempotent in the ordinary sense: run it as many times as you like against the same database
state and get the same output.

What it does, concretely: connects to `CARELITE_DATABASE_URL`, checks that the schema and expected
tables are present (reusing `carelite.db.connection.check_database`), reports what pipeline stages
have and have not produced data yet, and hands off to `carelite.stats` and `carelite.viz` for the
statistical tables and figures — this is a genuine gap-check, not a silent no-op, because as of this
writing only one of those two lanes has landed under the contract this entry point looks for.
**`carelite.viz.reproduce.run(output_dir)` exists and works** — it regenerates every figure it can
from whatever data is actually in the database, skipping (with a reason, not silently) any figure
that needs a table still empty, so `make reproduce` on this project's own database currently
produces at least the retrieval-quality figure even before `rubric_score` is loaded.
**`carelite.stats` exists (`carelite/stats/`, with `report.py` and its own
`python -m carelite.stats` CLI) but does not yet expose `carelite.stats.reproduce.run(output_dir)`**,
so `make reproduce` correctly reports its tables as `[pending]` rather than silently producing
nothing — run `python -m carelite.stats` directly for the statistical write-up in the meantime; see
`carelite/repro.py`'s own module docstring for the current state of that wiring, since it is kept
current there rather than restated here where it would drift.

---

## 9. Known-good order, summarized

```
uv sync --extra dev --extra rerank
make db-up && make db-check
ollama pull gemma4:12b qwen3.5:9b gpt-oss:20b bge-m3
uv run python -m carelite.corpus.fetch --email you@example.com
# ... KB extraction/load (see §5) ...
# ... index build (see §6) ...
make eval-smoke
# --- no registration gate (DECISIONS.md D10); results below are descriptive ---
# --- expect to stop Condition LC partway through (D11); see §7 ---
uv run python -m carelite.generate.runner --store jsonl \
    --journal runs/holdout/generations-X.jsonl --conditions X --split holdout  # once per condition group, in parallel
uv run python -m carelite.generate.load runs/holdout/generations-*.jsonl
uv run python -m carelite.eval.judge.holdout --glob 'runs/holdout/generations-*.jsonl'     --out runs/holdout/judge --workers 8
make reproduce
```

If any step in this document doesn't match what actually exists in the repository by the time you
read it, that is a documentation-drift bug in a fast-moving repo, not an instruction to guess —
check the module's own docstring, which several of the referenced modules deliberately keep more
current than this file can promise to.
