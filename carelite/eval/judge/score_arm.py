"""Judge one arm straight out of Postgres, with the unchanged judge.

    python -m carelite.eval.judge.score_arm --condition LC --served-by vllm \
        --out runs/judge-lc-vllm --workers 4

**Why this exists beside `holdout.py`.** `holdout.py` judges a run from its JSONL
journals. The vLLM LC cells (D13) were written by a driver that persisted to
Postgres, so there is no journal to read and the table is the record. This module
is the same judging pass with the table as its source: the same `LLMJudge`, the
same `gpt-oss:20b` cross-family judge, the same rubric, the same prompt version,
the same cache file, the same grounding rule, the same single pass at temperature
0. It reuses `judge_holdout`, `rows_for` and `build_manifest` rather than
restating them, because **the 939 existing scores came from that stack and a
second judge configuration would make the new arm incomparable to every other
arm.** The input set grows; the instrument does not change.

**One arm at a time, and the backend is not optional.** After D13, `condition =
'LC'` matches 219 rows across two serving stacks. `--served-by` is required and
`carelite.eval.judge.arms` re-checks what came back, so a pooled selection is a
refusal rather than a quietly larger n.

**Resumption is the existing mechanism.** `--cache` defaults to the run's own
cache file; point it at the holdout run's cache to share one resumable unit
across both. A generation already carrying the judge's aggregate row is skipped
before the batch starts (`--rejudge` turns that off), and inside the batch the
sample cache resumes at sample granularity.

Scores are written to `rubric_scores.jsonl`, not to `rubric_score`. Loading is
`python -m carelite.eval.judge.load`, which validates every row against the
table before writing one — including, now, that the row's `served_by` matches
the generation it points at.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from carelite.eval.judge.arms import Arm, fetch_arm
from carelite.eval.judge.holdout import (
    GenerationMeta,
    HoldoutJudgeRun,
    build_manifest,
    judge_holdout,
    rows_for,
)
from carelite.types import Generation

__all__ = ["ARM_META_SQL", "fetch_arm_meta", "main", "meta_from_rows", "score_arm"]

#: Everything `GenerationMeta` needs that `generation` alone does not carry.
#: `retrieval_trace` is a LEFT JOIN because a condition that does not retrieve —
#: LC is exactly that, by D7's design — has no trace row, and an inner join would
#: silently return nothing for the arm this module was written to score.
ARM_META_SQL = """
SELECT g.generation_id, g.scenario_id, g.condition, g.sample_idx, g.model,
       g.model_digest, g.gate_blocked, g.served_by, sc.split,
       COALESCE(t.fell_back_to_b, FALSE) AS fell_back_to_b,
       t.crag_grade,
       t.route_taken,
       COALESCE(array_length(t.retrieved_ids, 1), 0) AS n_retrieved
FROM generation g
JOIN scenario sc USING (scenario_id)
LEFT JOIN retrieval_trace t ON t.generation_id = g.generation_id
WHERE g.generation_id = ANY(%(ids)s)
"""


def meta_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, GenerationMeta]:
    """`generation_id -> GenerationMeta` from database rows.

    `output_gate_flags` stays empty: the flags live in the run's sidecar and have
    no column, and inventing an empty tuple is honest where inventing a value
    would not be. `gate_blocked` itself is a column (D12) and is read from it.
    """
    return {
        str(row["generation_id"]): GenerationMeta(
            generation_id=str(row["generation_id"]),
            scenario_id=str(row["scenario_id"]),
            condition=str(row["condition"]),
            sample_idx=int(row["sample_idx"]),
            split=str(row["split"]),
            fell_back_to_b=bool(row["fell_back_to_b"]),
            crag_grade=row["crag_grade"],
            route=row["route_taken"],
            n_retrieved=int(row["n_retrieved"] or 0),
            model=str(row["model"]),
            model_digest=str(row["model_digest"]),
            output_gate_blocked=bool(row["gate_blocked"]),
            output_gate_flags=(),
            served_by=str(row["served_by"] or "ollama"),
        )
        for row in rows
    }


def fetch_arm_meta(generation_ids: Sequence[str]) -> dict[str, GenerationMeta]:
    """The experimental identity of each generation, read from the table."""
    from carelite.db import connect

    if not generation_ids:
        return {}
    with connect() as conn:
        rows = conn.execute(ARM_META_SQL, {"ids": list(generation_ids)}).fetchall()
    return meta_from_rows(rows)


def score_arm(
    arm: Arm,
    *,
    cache_path: Path,
    workers: int = 1,
    num_ctx: int | None = None,
    on_progress: Any = None,
    client: Any = None,
) -> tuple[HoldoutJudgeRun, list[dict[str, Any]], dict[str, GenerationMeta]]:
    """Judge every generation in one arm. Returns `(run, rows, meta)`."""
    from carelite.eval.judge.store import fetch_scenario_texts

    generations: list[Generation] = sorted(
        arm.generations, key=lambda g: (g.scenario_id, g.sample_idx)
    )
    texts = fetch_scenario_texts({g.scenario_id for g in generations})
    meta = fetch_arm_meta([g.generation_id for g in generations])

    run = judge_holdout(
        generations,
        texts,
        cache_path=cache_path,
        client=client,
        workers=workers,
        num_ctx=num_ctx,
        on_progress=on_progress,
    )
    rows = rows_for(run.results, meta)
    rows.sort(key=lambda r: (r.get("scenario_id", ""), r["sample_idx"], r["generation_id"]))
    return run, rows, meta


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(
        prog="carelite.eval.judge.score_arm",
        description="Judge one condition under one serving stack, from the database.",
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument(
        "--served-by",
        required=True,
        help="'ollama' or 'vllm'. Required: after D13 the condition alone is not an arm.",
    )
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="judge cache file; defaults to <out>/judge-cache.jsonl",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--rater-id", default="holdout-judge")
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="judge cells that already carry this rater's aggregate row",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    arm = fetch_arm(
        condition=args.condition,
        served_by=args.served_by,
        split=args.split,
        unjudged_by=None if args.rejudge else args.rater_id,
    )
    full = fetch_arm(condition=args.condition, served_by=args.served_by, split=args.split)
    print(f"arm: {full.summary()}")
    print(f"to judge: {arm.n_cells} cells over {arm.n_scenarios} scenarios")
    if arm.n_cells == 0:
        print("nothing to judge; every cell already carries this rater's aggregate row")
        return 0
    if args.limit:
        arm = Arm(
            condition=arm.condition,
            served_by=arm.served_by,
            split=arm.split,
            generations=arm.generations[: args.limit],
            gate_blocked_ids=arm.gate_blocked_ids,
        )
        print(f"limited to {arm.n_cells} cells")
    if args.dry_run:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache or (args.out / "judge-cache.jsonl")
    started = time.monotonic()

    def progress(done: int, total: int, elapsed: float) -> None:
        if done % 5 and done != total:
            return
        rate = elapsed / done if done else 0.0
        print(
            f"[{done}/{total}] {elapsed / 60:.1f}m elapsed, eta {rate * (total - done) / 60:.0f}m",
            flush=True,
        )

    run, rows, meta = score_arm(
        arm,
        cache_path=cache_path,
        workers=args.workers,
        num_ctx=args.num_ctx,
        on_progress=progress,
    )

    _write(args.out / "rubric_scores.jsonl", rows)
    manifest = build_manifest(run, rows, meta)
    manifest["arm"] = {
        "condition": arm.condition,
        "served_by": arm.served_by,
        "split": arm.split,
        "n_cells_judged": arm.n_cells,
        "n_cells_in_arm": full.n_cells,
        "n_scenarios_in_arm": full.n_scenarios,
        "is_partial_record": full.is_partial,
        "source": "postgres `generation`, not a JSONL journal",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"judged={run.n_judged} errors={len(run.errors)} cached={run.n_from_cache} "
        f"called={run.n_called} complete_rate={run.complete_rate:.1%} "
        f"elapsed={(time.monotonic() - started) / 60:.1f}m"
    )
    print(f"wrote {args.out / 'rubric_scores.jsonl'} and manifest.json")
    print(f"load with: python -m carelite.eval.judge.load {args.out / 'rubric_scores.jsonl'}")
    for err in run.errors[:10]:
        print(
            f"  error {err['generation_id']}: {err['error_type']}: {err['message']}",
            file=sys.stderr,
        )
    return 1 if run.errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
