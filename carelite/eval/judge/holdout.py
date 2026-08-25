"""Judge a completed holdout generation run. **Temperature 0, single pass.**

    python -m carelite.eval.judge.holdout --glob 'runs/holdout/generations-*.jsonl' \
        --out runs/holdout/judge --workers 8

This is the other half of the lane from `study.py`. `study.py` runs the v3 §13
validation study on the train split at five samples and temperature 0.7;
this judges the real experiment once, at temperature 0, because that split is
what keeps the run tractable — five samples over 939 generations is not a longer
job, it is a different project.

**It will be interrupted, so resumption is the design.** Every judged sample
lands in `JudgeCache` keyed by generation, model digest, prompt version, rubric
version, temperature, sample index and anchor order. Re-running judges only what
is genuinely missing, and `JudgeRun.n_from_cache` is how you check that actually
happened rather than inferring it from a progress bar moving too fast.

**Concurrency.** 939 sequential judgements is hours of rented GPU time for no
reason: the model is resident, the requests are independent, and Ollama serves
them in parallel. `--workers` runs a thread pool over generations. `JudgeCache`
takes a lock per append, so the cache stays one file and one resumable unit.
Ordering is not preserved and does not matter — every result carries its own
generation id.

**What travels with each score, and why.**

`trace.fell_back_to_b` is copied onto every emitted row. On this run, condition
C fell back to B on **69 of 180 cells (38%)** because CRAG graded the retrieved
evidence irrelevant, and on those cells C is materially identical to B. Any
C-vs-B comparison that pools them is comparing a condition against itself for a
third of its mass. The flag is not derivable from the score, so if it does not
survive into what this module emits, the distinction is lost downstream and the
attenuation is silently absorbed into the effect size.

**Condition LC is a partial record, not a condition.** D11 stopped LC generation
at 39 of 180 cells, covering 13 of 60 scenarios, and those cells were never
randomised for partial analysis — they are the scenarios LC happened to reach
before it was stopped. They are judged (39 calls is nothing) and every LC row is
stamped `partial_condition: true`, with the coverage recorded in the manifest,
so nothing downstream can read LC as a complete arm. `--skip-lc` drops them
entirely if that is preferred.

**All results are descriptive.** D10 dropped the pre-registration; this is a
local proof of concept. Nothing this module produces is confirmatory or
pre-specified, and the judge's own validation study constrains what its scores
can support — see `manifest.json["judge_caveats"]`.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carelite.config import get_settings
from carelite.eval.judge.cache import JudgeCache
from carelite.eval.judge.client import ChatClient, OllamaChatClient
from carelite.eval.judge.judge import JudgeResult, LLMJudge
from carelite.eval.judge.prompt import OptionOrder
from carelite.types import RUBRIC_DIMENSIONS, Condition, Generation, RaterType, Split

__all__ = [
    "GenerationMeta",
    "HoldoutJudgeRun",
    "judge_holdout",
    "load_holdout",
    "main",
    "rows_for",
]

#: Cells LC actually reached before D11 stopped it, out of the 180 a full arm has.
LC_PLANNED_CELLS = 180


@dataclass(frozen=True, slots=True)
class GenerationMeta:
    """The experimental identity of one generation, carried alongside its score.

    Everything here is read off the stored row rather than recomputed, and all of
    it is needed downstream: `condition` and `sample_idx` to group, `split` to
    prove which half of the bank this is, and `fell_back_to_b` / `crag_grade` to
    separate the condition-C cells that actually retrieved from the ones that did
    not.
    """

    generation_id: str
    scenario_id: str
    condition: str
    sample_idx: int
    split: str
    fell_back_to_b: bool
    crag_grade: str | None
    route: str | None
    n_retrieved: int
    model: str
    model_digest: str
    output_gate_blocked: bool
    output_gate_flags: tuple[str, ...]


def load_holdout(
    paths: Sequence[Path],
    *,
    require_split: str | None = Split.HOLDOUT.value,
) -> tuple[list[Generation], dict[str, str], dict[str, GenerationMeta]]:
    """Read generation journals into `(generations, scenario_texts, metadata)`.

    Args:
        paths: JSONL journals written by `carelite.generate.store.JsonlStore`.
        require_split: refuse any row whose sidecar disagrees. Checked on the
            *record*, never on a filename or a flag, for the same reason
            `generate.runner` records the split from the scenario: a filename
            says what someone meant, a record says what the row is. Pass `None`
            to accept a mixed journal deliberately.

    Raises:
        RuntimeError: on a split mismatch or a duplicate generation id.
    """
    from carelite.generate.store import JsonlStore
    from carelite.scenarios.bank import load_bank

    texts = {s.scenario_id: s.text for s in load_bank()}
    generations: list[Generation] = []
    scenario_texts: dict[str, str] = {}
    meta: dict[str, GenerationMeta] = {}

    for path in paths:
        for record in JsonlStore(path=Path(path)).read_all():
            extra = record.extra or {}
            split = str(extra.get("split", ""))
            if require_split is not None and split != require_split:
                raise RuntimeError(
                    f"{path}: row {record.key.scenario_id}/{record.key.condition} has "
                    f"split={split!r}, expected {require_split!r}"
                )
            gid = record.generation_id
            if gid in meta:
                raise RuntimeError(f"duplicate generation_id {gid} in {path}")

            trace = record.trace or {}
            generations.append(
                Generation(
                    generation_id=gid,
                    scenario_id=record.key.scenario_id,
                    condition=Condition(record.key.condition),
                    prompt_id=record.key.prompt_id,
                    model=record.model,
                    model_digest=record.key.model_digest,
                    seed=record.key.seed,
                    temperature=record.temperature,
                    sample_idx=record.key.sample_idx,
                    response=record.response,
                    latency_ms=record.latency_ms,
                )
            )
            scenario_texts[record.key.scenario_id] = texts.get(record.key.scenario_id, "")
            meta[gid] = GenerationMeta(
                generation_id=gid,
                scenario_id=record.key.scenario_id,
                condition=record.key.condition,
                sample_idx=record.key.sample_idx,
                split=split,
                fell_back_to_b=bool(trace.get("fell_back_to_b", False)),
                crag_grade=trace.get("crag_grade"),
                route=(extra.get("context_note") or {}).get("route"),
                n_retrieved=len(trace.get("retrieved_ids") or []),
                model=record.model,
                model_digest=record.key.model_digest,
                output_gate_blocked=bool(extra.get("output_gate_blocked", False)),
                output_gate_flags=tuple(extra.get("output_gate_flags") or ()),
            )
    return generations, scenario_texts, meta


@dataclass
class HoldoutJudgeRun:
    """Outcome of one batch. Errors are first-class, not a log line."""

    results: list[JudgeResult] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    n_from_cache: int = 0
    n_called: int = 0
    elapsed_s: float = 0.0

    @property
    def n_judged(self) -> int:
        return len(self.results)

    @property
    def complete_rate(self) -> float:
        if not self.results:
            return float("nan")
        return sum(1 for r in self.results if r.complete) / len(self.results)


def judge_holdout(
    generations: Sequence[Generation],
    scenario_texts: Mapping[str, str],
    *,
    cache_path: Path,
    client: ChatClient | None = None,
    workers: int = 1,
    num_ctx: int | None = None,
    on_progress: Any = None,
) -> HoldoutJudgeRun:
    """Judge every generation once at temperature 0, resumably and in parallel.

    One `LLMJudge` is shared across workers. It holds no mutable per-call state —
    the only shared object is the cache, which locks — so sharing it is safe and
    keeps every sample under one cache file and one resumable unit.
    """
    from carelite.eval.judge.study import JUDGE_NUM_CTX

    run = HoldoutJudgeRun()
    started = time.monotonic()
    lock = threading.Lock()
    done = 0

    with JudgeCache(cache_path) as cache:
        judge = LLMJudge.for_full_run(
            client or OllamaChatClient(num_ctx=num_ctx or JUDGE_NUM_CTX),
            cache=cache,
            order=OptionOrder.ASCENDING,
            rater_id="holdout-judge",
        )

        def one(generation: Generation) -> None:
            nonlocal done
            try:
                text = scenario_texts.get(generation.scenario_id)
                if text is None:
                    raise KeyError(
                        f"no scenario text for {generation.scenario_id!r}; refusing to "
                        "judge a response without the turn it answers"
                    )
                result = judge.score(generation, text)
                with lock:
                    run.results.append(result)
                    run.n_from_cache += sum(1 for s in result.samples if s.from_cache)
                    run.n_called += sum(1 for s in result.samples if not s.from_cache)
            except Exception as exc:  # one bad cell must not end a 939-cell run
                with lock:
                    run.errors.append(
                        {
                            "generation_id": generation.generation_id,
                            "error_type": type(exc).__name__,
                            "message": str(exc)[:500],
                        }
                    )
            finally:
                with lock:
                    done += 1
                    if on_progress is not None:
                        on_progress(done, len(generations), time.monotonic() - started)

        if workers <= 1:
            for generation in generations:
                one(generation)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(one, g) for g in generations]
                for future in as_completed(futures):
                    future.result()  # `one` swallows its own errors; this re-raises bugs

    run.elapsed_s = time.monotonic() - started
    return run


def rows_for(
    results: Iterable[JudgeResult],
    meta: Mapping[str, GenerationMeta],
) -> list[dict[str, Any]]:
    """One JSON row per judged generation: a `rubric_score` plus its identity.

    The eleven dimensions are written on the **raw** scale, which is what
    `rubric_score` stores and what a human rater produces. `ritualistic` is still
    higher-is-worse here; anything mixing dimensions must call `to_quality`
    first. Emitting a quality-scaled column beside a raw one is how a reversal
    gets applied twice.
    """
    out: list[dict[str, Any]] = []
    for result in results:
        score = result.to_rubric_score(sample_idx=0)
        m = meta.get(result.generation_id)
        row: dict[str, Any] = {
            "generation_id": result.generation_id,
            "rater_type": RaterType.LLM_JUDGE.value,
            "rater_id": score.rater_id,
            "sample_idx": 0,
            "safety_flags": list(result.safety_flags),
            "evidence_spans": result.evidence_spans(),
            "judge_model": result.judge_model,
            "judge_digest": result.judge_digest,
            "prompt_version": result.prompt_version,
            "rubric_version": result.rubric_version,
            "temperature": result.temperature,
            "n_dimensions_scored": 11 - result.n_rejected,
            "complete": result.complete,
        }
        dumped = score.model_dump()
        for key in RUBRIC_DIMENSIONS:
            row[key] = dumped[key]
        if m is not None:
            row.update(
                {
                    "scenario_id": m.scenario_id,
                    "condition": m.condition,
                    "generation_sample_idx": m.sample_idx,
                    "split": m.split,
                    "fell_back_to_b": m.fell_back_to_b,
                    "crag_grade": m.crag_grade,
                    "route": m.route,
                    "n_retrieved": m.n_retrieved,
                    "generator_model": m.model,
                    "generator_digest": m.model_digest,
                    "partial_condition": m.condition == Condition.LC.value,
                    "output_gate_blocked": m.output_gate_blocked,
                    "output_gate_flags": list(m.output_gate_flags),
                }
            )
        out.append(row)
    return out


def build_manifest(
    run: HoldoutJudgeRun,
    rows: Sequence[Mapping[str, Any]],
    meta: Mapping[str, GenerationMeta],
) -> dict[str, Any]:
    """Counts, provenance, and the caveats a reader needs before using the scores."""
    from collections import Counter

    settings = get_settings()
    by_condition = Counter(str(r.get("condition")) for r in rows)
    lc_scenarios = {m.scenario_id for m in meta.values() if m.condition == Condition.LC.value}
    fell_back = [r for r in rows if r.get("fell_back_to_b")]
    incomplete = [r for r in rows if not r.get("complete")]

    return {
        "judged": run.n_judged,
        "errors": len(run.errors),
        "error_detail": run.errors[:50],
        "from_cache": run.n_from_cache,
        "called": run.n_called,
        "elapsed_s": round(run.elapsed_s, 1),
        "complete_rate": None if run.n_judged == 0 else round(run.complete_rate, 4),
        "n_incomplete_rows": len(incomplete),
        "by_condition": dict(sorted(by_condition.items())),
        "regime": {
            "temperature": settings.experiment.judge_temperature_full_run,
            "samples": settings.experiment.judge_samples_full_run,
            "note": (
                "Single pass at temperature 0. The five-sample self-consistency "
                "regime belongs to the v3 §13 validation subset only; running it "
                "over the whole holdout is a different project, not a longer job."
            ),
        },
        "provenance": {
            "judge_tag": settings.models.judge.tag,
            "generator_tag": settings.models.generator.tag,
            "cross_family": True,
            "judge_model": rows[0]["judge_model"] if rows else None,
            "judge_digest": rows[0]["judge_digest"] if rows else None,
            "prompt_version": rows[0]["prompt_version"] if rows else None,
            "rubric_version": rows[0]["rubric_version"] if rows else None,
        },
        "condition_c_fallback": {
            "n_fell_back_to_b": len(fell_back),
            "n_condition_c": by_condition.get(Condition.C.value, 0),
            "share": (
                round(len(fell_back) / by_condition[Condition.C.value], 4)
                if by_condition.get(Condition.C.value)
                else None
            ),
            "note": (
                "CRAG graded the retrieved evidence irrelevant on these cells and the "
                "graph fell back to condition-B behaviour, so C is materially identical "
                "to B there. Split on `fell_back_to_b` before comparing C with B; "
                "pooling compares a condition against itself for that share of its mass."
            ),
        },
        "condition_lc": {
            "n_cells": by_condition.get(Condition.LC.value, 0),
            "planned_cells": LC_PLANNED_CELLS,
            "n_scenarios": len(lc_scenarios),
            "partial": True,
            "note": (
                "D11 stopped LC generation partway. These are the scenarios LC happened "
                "to reach before it was stopped, not a randomised subsample, so they "
                "support no comparison against a complete arm. Every LC row carries "
                "`partial_condition: true`."
            ),
        },
        "output_gate_blocked": _gate_block_summary(rows),
        "score_summary": summarise_scores(rows),
        "reporting": {
            "descriptive_only": True,
            "note": (
                "D10: local proof of concept, no pre-registration. No result from these "
                "scores may be described as confirmatory or pre-specified."
            ),
        },
        "judge_caveats": {
            "note": (
                "From the v3 §13 validation study on the train split (n=30, five samples "
                "at 0.7). These bound what the holdout scores can support and should be "
                "read before any dimension is interpreted."
            ),
            "degenerate_on_validation": ["ritualistic"],
            "low_discrimination_on_validation": ["naturalness", "ie"],
            "detail": (
                "`ritualistic` was scored 1 on all 30 validation responses - perfectly "
                "self-consistent and measuring nothing. `naturalness` had a "
                "discrimination ratio of 0.68, meaning sample noise exceeded between-"
                "response signal. Build plan v3 predicts B loses to A on naturalness "
                "because framework prompting induces ritual; if these hold here, that "
                "comparison is not detectable by this judge and a null on it is evidence "
                "about the instrument, not about the conditions."
            ),
            "span_support_rate_validation": 0.80,
            "automatic_span_grounding_validation": 0.96,
        },
    }


def _gate_block_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The cells whose text the output safety gate refused, and why they are hard.

    These rows are in the run like any other — the gate refused the text, it did
    not stop the cell being generated — so judging them scores output the system
    would never have delivered. They are judged and flagged rather than dropped,
    because **neither choice is neutral and the choice is not this module's to
    make**:

    * Leaving them in scores refused content as ordinary model output.
    * Taking them out is not a symmetric trim. On this run the blocked cells are
      not spread evenly: condition A2 carries 7 of 17, four of them on scenarios
      no other condition tripped, so excluding them removes A2's worst-behaved
      outputs and flatters its mean.

    Both facts are recorded here so whoever runs the analysis chooses knowingly
    and can run it both ways. `--skip-gate-blocked` drops them at load time.
    """
    from collections import Counter

    blocked = [r for r in rows if r.get("output_gate_blocked")]
    by_condition = Counter(str(r.get("condition")) for r in blocked)
    by_scenario = Counter(str(r.get("scenario_id")) for r in blocked)
    flags = Counter(f for r in blocked for f in (r.get("output_gate_flags") or []))
    # Scenarios where every blocked cell came from a single condition: the
    # asymmetric ones, which are what makes exclusion a judgement call.
    conditions_per_scenario: dict[str, set[str]] = {}
    for r in blocked:
        conditions_per_scenario.setdefault(str(r.get("scenario_id")), set()).add(
            str(r.get("condition"))
        )
    single = {
        scenario: next(iter(conds))
        for scenario, conds in sorted(conditions_per_scenario.items())
        if len(conds) == 1
    }

    return {
        "n_blocked": len(blocked),
        "by_condition": dict(sorted(by_condition.items())),
        "by_scenario": dict(sorted(by_scenario.items())),
        "by_flag": dict(sorted(flags.items())),
        "scenarios_only_one_condition_tripped": single,
        "default_analysis": (
            "EXCLUDE these rows from any headline condition comparison: the gate "
            "refused this text, so scoring it measures something the system would "
            "not have said."
        ),
        "but_note": (
            "Exclusion is not symmetric on this run. The blocked cells concentrate "
            "in A2, which carries 7 of 17 across four scenarios no other condition "
            "tripped, so dropping them removes that arm's worst-behaved outputs. "
            "Report the comparison both ways rather than picking one silently."
        ),
    }


def summarise_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-dimension distribution over the judged holdout, and the C-split means.

    Two questions this answers that nothing else does.

    **Does the validation study's degeneracy hold at scale?** On the train-split
    validation subset `ritualistic` was scored 1 on all 30 responses and
    `naturalness` had a discrimination ratio of 0.68. Those were 30 responses;
    this is 939. `n_distinct` and `modal_share` per dimension say whether the
    judge is discriminating here, and a dimension that used one value across 939
    responses spanning five conditions is measuring nothing, however
    self-consistent it looks.

    **Does condition C differ from B where it actually retrieved?** C fell back
    to B on 38% of cells, and on those it *is* B. Means are reported for C split
    on `fell_back_to_b` alongside B, so the comparison that matters is visible
    without re-deriving it. Quality scale via `to_quality`, so every dimension
    points the same way and `ritualistic` does not read backwards next to its
    neighbours.
    """
    import statistics
    from collections import Counter

    from carelite.eval.rubric.dimensions import to_quality

    scored = [r for r in rows if not r.get("output_gate_blocked")]

    dimensions: dict[str, Any] = {}
    for key in RUBRIC_DIMENSIONS:
        values = [r[key] for r in scored if r.get(key) is not None]
        counts = Counter(values)
        dimensions[key] = {
            "n": len(values),
            "distribution": dict(sorted(counts.items())),
            "n_distinct": len(counts),
            "modal_share": round(counts.most_common(1)[0][1] / len(values), 4) if values else None,
            "between_variance": round(statistics.variance(values), 4) if len(values) > 1 else 0.0,
            "degenerate": len(counts) <= 1,
        }

    def _mean(subset: Sequence[Mapping[str, Any]], key: str) -> float | None:
        vals = [to_quality(key, r[key]) for r in subset if r.get(key) is not None]
        return round(statistics.fmean(vals), 3) if vals else None

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in scored:
        groups.setdefault(str(r.get("condition")), []).append(r)
    c_rows = groups.get("C", [])
    groups["C (retrieved)"] = [r for r in c_rows if not r.get("fell_back_to_b")]
    groups["C (fell back to B)"] = [r for r in c_rows if r.get("fell_back_to_b")]

    return {
        "n_rows": len(rows),
        "n_scored_excluding_gate_blocked": len(scored),
        "dimensions": dimensions,
        "degenerate_dimensions": [k for k, v in dimensions.items() if v["degenerate"]],
        "quality_means_by_group": {
            group: {"n": len(subset), **{k: _mean(subset, k) for k in RUBRIC_DIMENSIONS}}
            for group, subset in sorted(groups.items())
            if subset
        },
        "note": (
            "Quality scale (`to_quality`), so higher is better on every dimension "
            "including `ritualistic`. Gate-blocked rows are excluded here; they are "
            "present in rubric_scores.jsonl and flagged."
        ),
    }


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(
        prog="carelite.eval.judge.holdout",
        description="Judge a completed holdout run. Temperature 0, single pass, resumable.",
    )
    parser.add_argument("--glob", default="runs/holdout/generations-*.jsonl")
    parser.add_argument("--out", type=Path, default=Path("runs/holdout/judge"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--skip-lc", action="store_true", help="drop the partial LC arm")
    parser.add_argument(
        "--skip-gate-blocked",
        action="store_true",
        help="drop cells the output safety gate refused (see the manifest first)",
    )
    parser.add_argument("--allow-any-split", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    # `Path().glob` refuses an absolute pattern; the stdlib `glob` does not,
    # and an absolute path is the normal way to name a journal on a pod whose
    # working directory is not the repo.
    paths = [Path(p) for p in sorted(glob.glob(args.glob))]
    if not paths:
        print(f"no journals matched {args.glob!r}", file=sys.stderr)
        return 2

    generations, texts, meta = load_holdout(
        paths, require_split=None if args.allow_any_split else Split.HOLDOUT.value
    )
    if args.skip_lc:
        generations = [g for g in generations if g.condition is not Condition.LC]
    if args.skip_gate_blocked:
        generations = [g for g in generations if not meta[g.generation_id].output_gate_blocked]
    # Deterministic order so a resumed run works through the same sequence.
    generations.sort(key=lambda g: (g.scenario_id, g.condition.value, g.sample_idx))
    if args.limit:
        generations = generations[: args.limit]

    from collections import Counter

    print(f"journals: {[p.name for p in paths]}")
    print(
        f"generations: {len(generations)}  conditions: "
        f"{dict(sorted(Counter(g.condition.value for g in generations).items()))}"
    )
    print(f"scenarios: {len({g.scenario_id for g in generations})}")
    if args.dry_run:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / "judge-cache.jsonl"

    def progress(done: int, total: int, elapsed: float) -> None:
        if done % 10 and done != total:
            return
        rate = elapsed / done if done else 0.0
        print(
            f"[{done}/{total}] {elapsed / 60:.1f}m elapsed, eta {rate * (total - done) / 60:.0f}m",
            flush=True,
        )

    run = judge_holdout(
        generations,
        texts,
        cache_path=cache_path,
        workers=args.workers,
        num_ctx=args.num_ctx,
        on_progress=progress,
    )

    rows = rows_for(run.results, meta)
    rows.sort(key=lambda r: (r.get("scenario_id", ""), r.get("condition", ""), r["sample_idx"]))
    _write(args.out / "rubric_scores.jsonl", rows)
    manifest = build_manifest(run, rows, meta)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"judged={run.n_judged} errors={len(run.errors)} cached={run.n_from_cache} "
        f"called={run.n_called} complete_rate={run.complete_rate:.1%} "
        f"elapsed={run.elapsed_s / 60:.1f}m"
    )
    print(f"wrote {args.out / 'rubric_scores.jsonl'} and manifest.json")
    for err in run.errors[:10]:
        print(
            f"  error {err['generation_id']}: {err['error_type']}: {err['message']}",
            file=sys.stderr,
        )
    return 1 if run.errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
