"""Inference lane III: 60 held-out scenarios x 6 conditions x 3 samples = 1,080 cells.

    python -m carelite.generate.runner --store postgres
    python -m carelite.generate.runner --store jsonl --conditions A,B --limit 5
    python -m carelite.generate.runner --split train --store jsonl   # 40 x 6 x 3

**It will be interrupted.** A run of this size on local inference takes hours,
the machine will sleep, the daemon will be restarted, and someone will press
ctrl-C. So resumption is the design rather than a feature bolted on afterwards:

* Every cell's identity is `(scenario_id, condition, prompt_id, model_digest,
  seed, sample_idx)` — build plan v3 section 16 — and that tuple is also the
  `UNIQUE` constraint on `generation`, so the skip rule and the database's
  refusal to duplicate are the same rule.
* Every seed comes from `config.seed_for`, which is blake2b over the cell's
  identity. Nothing in the plan depends on iteration order or on process state,
  so the plan a resumed run computes is byte-identical to the plan the killed
  run was working through.
* Every record is durably stored before the next cell starts. A `SIGKILL` costs
  the cell in flight and nothing else.

There is deliberately no checkpoint file and no run-level state. A checkpoint is
one more thing that can disagree with the data; `completed_keys()` asks the
store what is actually there.

**Which split, and why the two cannot contaminate each other.**

`--split holdout` is the default and generates the 60 held-out scenarios;
`--split train` generates the 40 training scenarios, which is what the v3 §13
judge-validation study is scored against. That is the only difference between
the two runs: same plan builder, same cache key, same graph, same seeds.

*They cannot collide in the cache key.* The key's first field is `scenario_id`;
`scenarios.bank.load_bank` raises on a duplicate id anywhere in the 100 records;
and the splits are a partition of exactly those records. So no train cell and no
holdout cell can ever produce the same key. A resumed run of either split reads
the whole store — including the other split's rows — and simply finds keys that
are not in its plan, which is a no-op. Nothing skips a cell it owns and nothing
adopts a cell it does not.

*For the same reason the split is never passed to `config.seed_for`.* A seed is
a property of the cell — `(scenario_id, condition, sample_idx)` — not of the run
that asked for it. A scenario would keep its seeds even if it were ever moved
between splits, and a train run and a holdout run agree on every seed whose
inputs they share. `tests/unit/generate/test_runner_split.py` asserts both
halves of this rather than leaving them to the reader.

*The split is recorded on every row,* in `extra["split"]`, taken from the
scenario record itself rather than from the run's flag. The frozen `generation`
table has no split column and this lane does not own the schema, so the sidecar
is where it goes; the point is that a table holding both can be read back
without a join to the bank, because the pre-registration turns on exactly that
distinction.

**Do not start a holdout run before the OSF pre-registration is submitted.**
That is a project gate recorded in `DECISIONS.md`, not something this module
enforces — `--split holdout` stays the default because every existing invocation
means it. `main()` prints the split it is about to generate, and warns on
holdout, so the gate is visible at the top of the log rather than discovered
afterwards.

**Digests are resolved before the plan is built,** because `model_digest` is
part of the key. If the daemon is restarted mid-run with re-pulled weights, the
keys change and the affected cells are regenerated under the new digest rather
than silently mixing two models in one column. That is the behaviour a mutable
tag makes necessary.

**Two things get recorded that a reader might expect to be dropped.**

*Gate-blocked generations are stored, flagged.* When the output gate withholds a
response, the draft is written with `output_gate_blocked` in the sidecar. The
alternative — no row — makes a run look complete while a systematic gate
interaction sits in the gap, and means the cell is regenerated on every resume
forever.

*The PHI screen's opinion on a bank scenario is recorded but not acted on.* The
runner drives the graph under `InputPolicy.CURATED_BANK`: see `graph.py` for why
redacting a frozen held-out scenario would be worse than not redacting it. The
flags land in the sidecar either way.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carelite.config import get_settings, seed_for
from carelite.generate import prompts
from carelite.generate.conditions import SPEC, ConditionSpec, spec_for
from carelite.generate.graph import (
    GraphDeps,
    InputPolicy,
    build_graph,
    initial_state,
)
from carelite.generate.model import DIGEST_UNAVAILABLE, GenerationClient
from carelite.generate.store import CacheKey, GenerationRecord, GenerationStore, JsonlStore
from carelite.types import Condition, GuidanceRequest, Scenario, Split

__all__ = ["Cell", "RunReport", "build_plan", "main", "run", "scenarios_for_split"]


def scenarios_for_split(split: Split | str) -> list[Scenario]:
    """The bank's scenarios for one split, narrowed to the frozen contract.

    The two loaders live in `carelite.scenarios.bank` and this is the only place
    the runner names either of them, so "which split am I about to generate" has
    exactly one answer per run instead of one per call site.
    """
    from carelite.scenarios.bank import holdout_scenarios, train_scenarios

    chosen = Split(split)
    loader = train_scenarios if chosen is Split.TRAIN else holdout_scenarios
    return [s.to_scenario() for s in loader()]


@dataclass(frozen=True, slots=True)
class Cell:
    """One generation to produce. Everything about it is decided before it runs."""

    scenario: Scenario
    spec: ConditionSpec
    sample_idx: int
    seed: int
    model_digest: str

    @property
    def key(self) -> CacheKey:
        return CacheKey(
            scenario_id=self.scenario.scenario_id,
            condition=self.spec.condition.value,
            prompt_id=self.spec.prompt_id,
            model_digest=self.model_digest,
            seed=self.seed,
            sample_idx=self.sample_idx,
        )


@dataclass
class RunReport:
    """What happened. Counts, not a log — a log is not something you can assert on."""

    planned: int = 0
    skipped: int = 0
    generated: int = 0
    gate_blocked: int = 0
    input_blocked: int = 0
    failed: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    blocked_scenarios: list[str] = field(default_factory=list)

    split_counts: dict[str, int] = field(default_factory=dict)
    """Planned cells per split, read off the scenarios rather than off the flag.
    A run handed a hand-built list is described correctly, and a list that
    accidentally mixed splits is visible instead of averaged away."""

    routes: dict[str, int] = field(default_factory=dict)
    """Observed router outcome per generated cell. `route` runs for every
    condition, so this is the distribution over the turns this run actually
    saw. It is here because a router that quietly sends everything down
    `emotional_only` turns condition C into condition B, and the resulting run
    looks entirely healthy: same row count, same latencies, no errors."""

    @property
    def split_label(self) -> str:
        return "+".join(sorted(self.split_counts)) or "none"

    def summary(self) -> str:
        return (
            f"split={self.split_label} planned={self.planned} skipped={self.skipped} "
            f"generated={self.generated} gate_blocked={self.gate_blocked} "
            f"input_blocked={self.input_blocked} failed={self.failed} "
            f"elapsed={self.elapsed_s:.0f}s"
        )

    def route_summary(self) -> str:
        if not self.routes:
            return "routes: none observed"
        total = sum(self.routes.values())
        parts = ", ".join(
            f"{name}={count} ({count / total:.0%})"
            for name, count in sorted(self.routes.items(), key=lambda kv: -kv[1])
        )
        return f"routes over {total} generated cells: {parts}"


def build_plan(
    scenarios: Sequence[Scenario],
    conditions: Sequence[Condition],
    *,
    samples: int,
    digests: dict[Condition, str],
) -> list[Cell]:
    """Every cell, in a fixed order. Pure: same inputs, same plan, every run."""
    cells: list[Cell] = []
    for scenario in scenarios:
        for condition in conditions:
            spec = spec_for(condition)
            for sample_idx in range(samples):
                cells.append(
                    Cell(
                        scenario=scenario,
                        spec=spec,
                        sample_idx=sample_idx,
                        seed=seed_for(scenario.scenario_id, condition.value, sample_idx),
                        model_digest=digests.get(condition, DIGEST_UNAVAILABLE),
                    )
                )
    return cells


def _trace_payload(state: dict[str, Any]) -> dict[str, Any] | None:
    trace = state.get("trace")
    if trace is None:
        return None
    return {
        "retrieved_ids": [item.ref_id for item in trace.retrieved],
        "scores": [float(item.score) for item in trace.retrieved],
        "crag_grade": trace.crag_grade.value,
        "route_taken": trace.route.value,
        "fell_back_to_b": trace.fell_back_to_b,
        "hyde_passage": trace.hyde_passage,
        "latency_ms": trace.latency_ms,
    }


def _extra_payload(cell: Cell, state: dict[str, Any]) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "scenario_id": cell.scenario.scenario_id,
        "condition": cell.spec.condition.value,
        # From the scenario record, never from the run's --split flag: the flag
        # says what was asked for, the record says what this row actually is.
        "split": cell.scenario.split.value,
        "sample_idx": cell.sample_idx,
        "num_ctx": state.get("num_ctx"),
        "prompt_chars": state.get("prompt_chars"),
    }
    verdict = state.get("input_safety")
    if verdict is not None and verdict.flags:
        extra["input_safety_flags"] = list(verdict.flags)
    check = state.get("self_check")
    if check is not None:
        extra.update(check.as_record())
    note = state.get("context_note") or {}
    if note:
        extra["context_note"] = note
    errors = state.get("errors") or []
    if errors:
        extra["errors"] = list(errors)
    gate = state.get("output_safety")
    if gate is not None and not gate.allowed:
        extra["output_gate_blocked"] = True
        extra["output_gate_flags"] = list(gate.flags)
    return extra


def run(
    *,
    store: GenerationStore,
    split: Split | str | None = None,
    scenarios: Sequence[Scenario] | None = None,
    conditions: Sequence[Condition] | None = None,
    samples: int | None = None,
    deps: GraphDeps | None = None,
    graph: Any | None = None,
    digests: dict[Condition, str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    require_committed: bool = False,
    on_cell: Callable[[Cell, int, int], None] | None = None,
) -> RunReport:
    """Generate every planned cell that is not already stored.

    Args:
        store: where finished cells go and where completed keys come from.
        split: which half of the frozen bank to generate — `Split.HOLDOUT`
            (the default, 60 scenarios) or `Split.TRAIN` (40). Mutually
            exclusive with `scenarios`: passing both would let the flag and the
            data disagree about what the run is, which is the one thing the
            pre-registration cannot tolerate.
        scenarios: an explicit list, for tests and ablations. Defaults to the
            held-out scenarios from the frozen bank.
        conditions: defaults to all six.
        samples: defaults to `settings.experiment.samples_per_cell`.
        deps: graph collaborators. The default drives real Ollama; the
            resumability test passes fakes and no service is touched.
        digests: condition -> model digest. Resolved from the daemon when not
            given, because the digest is part of the cache key.
        require_committed: refuse to start if any prompt's assembled text is not
            a blob in the git object database. Use it for the real run: a
            result produced by a prompt that exists only in a working tree is
            not reproducible.
    """
    settings = get_settings()
    started = time.monotonic()
    report = RunReport()

    if scenarios is not None and split is not None:
        raise ValueError(
            "pass `split` or `scenarios`, not both: the split of an explicit list is "
            "read off its scenarios, so a `split` argument here could only contradict it."
        )
    if scenarios is None:
        scenarios = scenarios_for_split(Split.HOLDOUT if split is None else split)
    if conditions is None:
        conditions = list(SPEC)
    if samples is None:
        samples = settings.experiment.samples_per_cell
    if deps is None:
        deps = GraphDeps(input_policy=InputPolicy.CURATED_BANK)
    deps.input_policy = InputPolicy.CURATED_BANK

    prompt_ids = sorted({spec_for(c).prompt_id for c in conditions} | {"selfcheck.v1"})
    if require_committed:
        status = prompts.verify_committed(prompt_ids)
        missing = sorted(p for p, ok in status.items() if not ok)
        if missing:
            raise RuntimeError(
                "these prompts are not committed, so a result generated now could not be "
                f"recovered from history: {missing}. Commit them, or drop "
                "--require-committed and accept that the run is not reproducible."
            )

    if digests is None:
        client = deps.client if isinstance(deps.client, GenerationClient) else GenerationClient()
        digests = {c: client.resolve_digest(spec_for(c).model_tag) for c in conditions}

    plan = build_plan(scenarios, conditions, samples=samples, digests=digests)
    report.planned = len(plan)
    for cell in plan:
        name = cell.scenario.split.value
        report.split_counts[name] = report.split_counts.get(name, 0) + 1

    done = store.completed_keys()
    todo = [c for c in plan if c.key not in done]
    report.skipped = len(plan) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    if dry_run:
        report.elapsed_s = time.monotonic() - started
        return report

    compiled = graph if graph is not None else build_graph()
    temperature = settings.experiment.generation_temperature

    for index, cell in enumerate(todo, start=1):
        if on_cell is not None:
            on_cell(cell, index, len(todo))
        request = GuidanceRequest(
            utterance=cell.scenario.text,
            condition=cell.spec.condition,
            encounter_phase=cell.scenario.encounter_phase,
            seed=cell.seed,
            temperature=temperature,
        )
        state = initial_state(request, deps=deps)
        try:
            final = dict(compiled.invoke(state))
        except Exception as exc:  # a store or daemon fault: the cell retries next run
            report.failed += 1
            report.errors.append(f"{cell.key.as_tuple()}: {type(exc).__name__}: {exc}")
            continue

        verdict = final.get("input_safety")
        if verdict is not None and not verdict.allowed and not verdict.phi_detected:
            # Red flag or injection. Escalation is the correct behaviour and
            # there is no generation to store; the scenario is named in the
            # report so it is visible rather than quietly absent from the table.
            report.input_blocked += 1
            report.blocked_scenarios.append(cell.scenario.scenario_id)
            continue

        text = final.get("text") or final.get("draft") or ""
        if not text:
            report.failed += 1
            report.errors.append(
                f"{cell.key.as_tuple()}: {final.get('halt_reason', 'no text produced')}"
            )
            continue

        observed = (final.get("context_note") or {}).get("route")
        if observed:
            report.routes[str(observed)] = report.routes.get(str(observed), 0) + 1

        gate = final.get("output_safety")
        if gate is not None and not gate.allowed:
            report.gate_blocked += 1

        store.record(
            GenerationRecord(
                key=cell.key,
                model=final.get("model") or cell.spec.model_tag,
                temperature=temperature,
                response=text,
                latency_ms=final.get("latency_ms"),
                trace=_trace_payload(final),
                extra=_extra_payload(cell, final),
            )
        )
        report.generated += 1

    report.elapsed_s = time.monotonic() - started
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_conditions(raw: str | None) -> list[Condition] | None:
    if not raw:
        return None
    return [Condition[name.strip().upper()] for name in raw.split(",") if name.strip()]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(
        prog="carelite.generate.runner",
        description=(
            "Generate an evaluation set. Safe to interrupt and rerun. "
            "Defaults to the held-out split; --split train generates the 40 "
            "training scenarios instead, which is what the judge-validation "
            "study is scored against."
        ),
    )
    parser.add_argument(
        "--split",
        choices=tuple(s.value for s in Split),
        default=Split.HOLDOUT.value,
        help="which half of the frozen bank to generate (default: holdout)",
    )
    parser.add_argument("--store", choices=("postgres", "jsonl"), default="postgres")
    parser.add_argument("--journal", type=Path, default=None, help="path for --store jsonl")
    parser.add_argument("--conditions", default=None, help="comma-separated, e.g. A,B,C")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="stop after N new cells")
    parser.add_argument("--dry-run", action="store_true", help="plan and count, generate nothing")
    parser.add_argument("--require-committed", action="store_true")
    parser.add_argument("--register-prompts", action="store_true", help="upsert prompt_version")
    args = parser.parse_args(argv)

    settings = get_settings()
    split = Split(args.split)

    # The two splits get different default filenames. Correctness does not need
    # it — the cache key already keeps them apart — but a 1,080-row holdout
    # journal and a judge-validation train journal are read by different people
    # for different reasons, and holdout keeps the historical name so every
    # existing invocation lands exactly where it always did.
    suffix = "" if split is Split.HOLDOUT else f"-{split.value}"
    if args.store == "jsonl":
        path = args.journal or (settings.runs_dir / "generate" / f"generations{suffix}.jsonl")
        store: GenerationStore = JsonlStore(path=path)
    else:
        from carelite.generate.store import PostgresStore

        sidecar = settings.runs_dir / "generate" / f"metadata{suffix}.jsonl"
        store = PostgresStore(sidecar=sidecar)
        if args.register_prompts:
            inserted = prompts.register()
            print(f"prompt_version rows inserted: {inserted}")

    print(f"split={split.value}", flush=True)
    if split is Split.HOLDOUT and not args.dry_run:
        print(
            "  NOTE: this is the held-out split. DECISIONS.md gates it on the OSF "
            "pre-registration being submitted; this runner does not enforce that.",
            file=sys.stderr,
            flush=True,
        )

    def progress(cell: Cell, index: int, total: int) -> None:
        print(
            f"[{index}/{total}] {cell.scenario.scenario_id} "
            f"{cell.spec.condition.value} sample={cell.sample_idx}",
            flush=True,
        )

    try:
        report = run(
            store=store,
            split=split,
            conditions=_parse_conditions(args.conditions),
            samples=args.samples,
            limit=args.limit,
            dry_run=args.dry_run,
            require_committed=args.require_committed,
            on_cell=progress,
        )
    finally:
        store.close()

    print(report.summary())
    print(f"  {report.route_summary()}")
    for err in report.errors[:20]:
        print(f"  error: {err}", file=sys.stderr)
    if report.blocked_scenarios:
        print(f"  input-blocked scenarios: {sorted(set(report.blocked_scenarios))}")
    return 1 if report.failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
