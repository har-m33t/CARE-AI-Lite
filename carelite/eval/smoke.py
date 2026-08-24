"""End-to-end smoke test: 5 scenarios x all six conditions, before the real run.

    make eval-smoke
    python -m carelite.eval.smoke --scenarios 5 --split train
    python -m carelite.eval.smoke --dry-run          # plan and audit wiring only

The full run is 1,080 local generations and takes hours. This is the cheap
version of the same code path, and its entire value is failing *here* instead of
four hours into that. So it is deliberately not a "did anything come back"
check. It drives `carelite.generate.runner.run` — the real runner, the real
graph, the real prompts, the real safety gate — and then audits the rows it got
back against what each condition is supposed to have done.

**It defaults to the train split.** Every response it generates is real
generated data; producing it on held-out scenarios would spend part of the
holdout before the OSF pre-registration is submitted, which `DECISIONS.md`
gates. Five held-out cells is a small leak and a leak nonetheless, and a smoke
test is the last place worth taking that risk.

**It never writes to `generation`.** The store is a JSONL journal under
`runs/smoke/`, truncated at the start of every invocation. That is what makes it
safe to run repeatedly: the cache would otherwise turn the second run into a
no-op, and a smoke test that skips everything reports success without having
tested anything. Truncating is the point, not a shortcut.

**It fails loudly.** Anything that would make the full run worthless is a
`failure` and exits non-zero: a condition that produced no rows, a row with no
model digest, retrieval that never fired for condition C, retrieval that fired
for a condition that is supposed to have none, a long-context cell with no
corpus pack, a self-check that ran where it should not have. Things that are
merely worth seeing — an uncommitted prompt, a scenario the input screen
escalated — are `warnings` and do not fail the run.

**It reports the route distribution it observed,** per scenario and in
aggregate. This is here because of a measured finding: across the 100-scenario
bank the adaptive router sends 99 turns to `informational` and 1 to
`emotional_only`, so condition C really does differ from condition B — but an
ad-hoc turn like *"I'm scared this is cancer. Nobody will give me a straight
answer"* routes `emotional_only`, retrieves nothing, and makes C behave exactly
like B. A run in that state finishes with the right row count, the right
latencies and no errors; the route distribution is the only place it shows.
Surfacing it on every smoke run is how that class of regression gets caught in
routine use rather than by accident.

The optional `--with-judge` step extends the check through
`carelite.eval.judge`, which is what `REPRODUCE.md` describes the target as
covering. It is opt-in because it loads a second model and belongs to another
lane; without it this module covers generation, routing, retrieval, the
self-check and the output gate.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carelite.config import get_settings
from carelite.generate import prompts
from carelite.generate.conditions import SPEC, spec_for
from carelite.generate.graph import GraphDeps, InputPolicy
from carelite.generate.model import DIGEST_UNAVAILABLE
from carelite.generate.runner import RunReport, run, scenarios_for_split
from carelite.generate.store import GenerationRecord, JsonlStore
from carelite.types import Condition, Scenario, Split

__all__ = ["DEFAULT_SCENARIOS", "SmokeResult", "main", "smoke"]

#: Five scenarios is what the `Makefile` target and `REPRODUCE.md` both promise.
DEFAULT_SCENARIOS = 5

#: Conditions whose `ConditionSpec` says they see no retrieved evidence. If a
#: trace with hits shows up on one of these, the comparison the study rests on
#: is already broken and every number downstream is meaningless.
_NO_RETRIEVAL = tuple(c for c, spec in SPEC.items() if not spec.use_retrieval)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class SmokeResult:
    """What the smoke run found. `ok` is the exit status; the rest is the log."""

    report: RunReport
    records: list[GenerationRecord] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    journal: Path | None = None

    @property
    def ok(self) -> bool:
        return not self.failures

    def by_condition(self) -> dict[str, list[GenerationRecord]]:
        out: dict[str, list[GenerationRecord]] = {}
        for record in self.records:
            out.setdefault(record.key.condition, []).append(record)
        return out

    def routes_by_scenario(self) -> dict[str, str]:
        """`scenario_id -> route`. The router is deterministic and condition-
        independent, so one entry per scenario is the whole story."""
        out: dict[str, str] = {}
        for record in self.records:
            note = record.extra.get("context_note") or {}
            route = note.get("route")
            if route:
                out[record.key.scenario_id] = str(route)
        return dict(sorted(out.items()))

    def render(self) -> str:
        lines = [
            "carelite eval smoke",
            f"  {self.report.summary()}",
            f"  {self.report.route_summary()}",
        ]
        if self.journal is not None:
            lines.append(f"  journal: {self.journal}")

        lines.append("")
        lines.append("  per condition:")
        buckets = self.by_condition()
        for condition in SPEC:
            got = buckets.get(condition.value, [])
            retrieved = sum(1 for r in got if (r.trace or {}).get("retrieved_ids"))
            checked = sum(1 for r in got if r.extra.get("self_check_available"))
            lines.append(
                f"    {condition.value:<3} rows={len(got):<3} "
                f"retrieval_hits={retrieved:<3} self_check={checked:<3} "
                f"prompt={spec_for(condition).prompt_id}"
            )

        lines.append("")
        lines.append("  route per scenario:")
        for scenario_id, route in self.routes_by_scenario().items():
            marker = "  <- retrieves nothing" if route == "emotional_only" else ""
            lines.append(f"    {scenario_id:<10} {route}{marker}")

        if self.warnings:
            lines.append("")
            lines.append("  warnings:")
            lines.extend(f"    - {w}" for w in self.warnings)
        if self.failures:
            lines.append("")
            lines.append("  FAILURES:")
            lines.extend(f"    - {f}" for f in self.failures)
        lines.append("")
        lines.append("  OK" if self.ok else f"  FAILED ({len(self.failures)})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


def _audit(result: SmokeResult, *, scenarios: Sequence[Scenario], samples: int) -> None:
    """Turn a finished run into named failures. Every rule here is a thing that
    would make the 1,080-cell run worthless, stated once."""
    report = result.report
    records = result.records
    buckets = result.by_condition()

    if report.failed:
        for err in report.errors[:10]:
            result.failures.append(f"cell failed: {err}")

    expected = len(scenarios) * len(SPEC) * samples
    if report.planned != expected:
        result.failures.append(
            f"planned {report.planned} cells, expected {expected} "
            f"({len(scenarios)} scenarios x {len(SPEC)} conditions x {samples} samples)"
        )
    if report.skipped:
        result.failures.append(
            f"{report.skipped} cells were skipped as already done. The smoke journal is "
            "truncated per run, so nothing should ever be skipped; a skip here means the "
            "run tested less than it reported."
        )

    if report.input_blocked:
        result.warnings.append(
            f"the input screen escalated {report.input_blocked} cell(s) and they were not "
            f"generated: {sorted(set(report.blocked_scenarios))}"
        )

    # Every condition has to have actually produced something. This is the check
    # that catches a condition whose model tag is not pulled.
    for condition in SPEC:
        if not buckets.get(condition.value):
            result.failures.append(
                f"condition {condition.value} produced no rows at all — it is in the plan "
                "and nothing came back, so the full run would report five-sixths of a study"
            )

    for record in records:
        key = record.key
        where = f"{key.scenario_id}/{key.condition}/sample={key.sample_idx}"
        if not record.response.strip():
            result.failures.append(f"{where}: stored an empty response")
        if key.model_digest in ("", DIGEST_UNAVAILABLE):
            result.failures.append(
                f"{where}: no model digest. Tags are mutable; a row without a digest "
                "cannot be attributed to a model afterwards."
            )
        if key.prompt_id != spec_for(Condition(key.condition)).prompt_id:
            result.failures.append(
                f"{where}: prompt_id {key.prompt_id!r} is not the one its condition "
                f"specifies ({spec_for(Condition(key.condition)).prompt_id!r})"
            )
        for message in record.extra.get("errors") or []:
            # A node that degraded instead of raising — retrieval falling back,
            # the self-check declining to parse. The cell still produced a row,
            # so nothing upstream counts it as a failure and the full run would
            # absorb it silently. Here it is the whole point.
            result.failures.append(f"{where}: a node reported an error: {message}")

    _audit_retrieval(result, buckets)
    _audit_long_context(result, buckets)
    _audit_self_check(result, buckets)
    _audit_conditions_differ(result, buckets)
    _audit_routes(result)


def _audit_retrieval(result: SmokeResult, buckets: dict[str, list[GenerationRecord]]) -> None:
    c_rows = buckets.get(Condition.C.value, [])
    with_hits = [r for r in c_rows if (r.trace or {}).get("retrieved_ids")]
    if c_rows and not with_hits:
        result.failures.append(
            "condition C retrieved no evidence on any scenario. C is B plus retrieval, so "
            "with no retrieval the two conditions are the same program and the headline "
            "comparison measures nothing. Check the router's decision (below), the index, "
            "and the CRAG threshold."
        )
    elif c_rows and len(with_hits) < len(c_rows):
        silent = sorted({r.key.scenario_id for r in c_rows if r not in with_hits})
        result.warnings.append(
            f"condition C retrieved nothing on {len(c_rows) - len(with_hits)}/{len(c_rows)} "
            f"cells ({silent}); on those it is behaving as condition B"
        )

    for condition in _NO_RETRIEVAL:
        leaked = [
            r for r in buckets.get(condition.value, []) if (r.trace or {}).get("retrieved_ids")
        ]
        if leaked:
            result.failures.append(
                f"condition {condition.value} is configured with no retrieval but "
                f"{len(leaked)} of its rows carry retrieved evidence — the conditions are "
                "not differing only by configuration any more"
            )


def _audit_long_context(result: SmokeResult, buckets: dict[str, list[GenerationRecord]]) -> None:
    for record in buckets.get(Condition.LC.value, []):
        coverage = (record.extra.get("context_note") or {}).get("long_context")
        where = f"{record.key.scenario_id}/LC"
        if not coverage:
            result.failures.append(
                f"{where}: no corpus pack was built, so the long-context baseline saw the "
                "same empty context as condition A and is not a long-context baseline"
            )
            continue
        if not coverage.get("chunks_included"):
            result.failures.append(f"{where}: the corpus pack contains no chunks")
        elif coverage.get("truncated"):
            result.warnings.append(
                f"{where}: pack covers {coverage.get('chunk_fraction')} of the corpus chunks — "
                "a caveat the LC result has to carry"
            )


def _audit_self_check(result: SmokeResult, buckets: dict[str, list[GenerationRecord]]) -> None:
    for condition, spec in SPEC.items():
        for record in buckets.get(condition.value, []):
            ran = bool(record.extra.get("self_check_available"))
            where = f"{record.key.scenario_id}/{condition.value}"
            if spec.self_check and not ran:
                reason = record.extra.get("self_check_reason") or "no reason recorded"
                result.failures.append(
                    f"{where}: the self-check is configured on for this condition and did "
                    f"not run ({reason})"
                )
            if not spec.self_check and ran:
                result.failures.append(
                    f"{where}: the self-check ran on a condition configured without one. "
                    "A negative control that gets a repair pass is a repaired control."
                )


def _audit_conditions_differ(
    result: SmokeResult, buckets: dict[str, list[GenerationRecord]]
) -> None:
    """Two conditions producing byte-identical text on every scenario means the
    configuration is not reaching the model at all."""
    texts: dict[str, dict[tuple[str, int], str]] = {
        name: {(r.key.scenario_id, r.key.sample_idx): r.response for r in rows}
        for name, rows in buckets.items()
    }
    names = sorted(texts)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = set(texts[left]) & set(texts[right])
            if shared and all(texts[left][k] == texts[right][k] for k in shared):
                result.failures.append(
                    f"conditions {left} and {right} produced identical text on all "
                    f"{len(shared)} shared cells — their configuration is not reaching "
                    "the model"
                )


def _audit_routes(result: SmokeResult) -> None:
    routes = result.routes_by_scenario()
    emotional = sorted(s for s, r in routes.items() if r == "emotional_only")
    if emotional:
        result.warnings.append(
            f"{len(emotional)}/{len(routes)} scenarios routed emotional_only and therefore "
            f"retrieved nothing ({emotional}). On those turns condition C is condition B. "
            "This is the experiment-corrupting direction the router's own docstring names."
        )
    if routes and not any(r != "emotional_only" for r in routes.values()):
        result.failures.append(
            "every scenario routed emotional_only, so retrieval never fired anywhere"
        )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def smoke(
    *,
    split: Split | str = Split.TRAIN,
    n_scenarios: int = DEFAULT_SCENARIOS,
    samples: int = 1,
    conditions: Sequence[Condition] | None = None,
    journal: Path | None = None,
    deps: GraphDeps | None = None,
    graph: Any | None = None,
    digests: dict[Condition, str] | None = None,
    dry_run: bool = False,
    on_cell: Any | None = None,
) -> SmokeResult:
    """Generate a handful of cells through the real pipeline and audit them.

    Args:
        split: which half of the bank to draw from. Train by default; see the
            module docstring for why a smoke test must not touch the holdout.
        n_scenarios: how many scenarios, taken from the front of the split's
            id-sorted order so two invocations exercise the same turns.
        samples: 1 by default. The smoke test is a wiring check, not a variance
            estimate, and three samples would triple its cost for nothing.
        journal: where the generations go. Truncated first. Never `generation`.
    """
    settings = get_settings()
    chosen = list(scenarios_for_split(split))[:n_scenarios]
    if not chosen:
        raise RuntimeError(f"no scenarios in split {split!r}")

    path = journal or (settings.runs_dir / "smoke" / "generations.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Truncate: a cached smoke run is a smoke run that tested nothing.
    path.write_text("", encoding="utf-8")

    store = JsonlStore(path=path)
    try:
        report = run(
            store=store,
            scenarios=chosen,
            conditions=list(conditions) if conditions is not None else list(SPEC),
            samples=samples,
            deps=deps if deps is not None else GraphDeps(input_policy=InputPolicy.CURATED_BANK),
            graph=graph,
            digests=digests,
            dry_run=dry_run,
            on_cell=on_cell,
        )
    finally:
        store.close()

    result = SmokeResult(
        report=report,
        records=list(JsonlStore(path=path).read_all()),
        journal=path,
    )

    uncommitted = sorted(
        p for p, ok in prompts.verify_committed(_prompt_ids(conditions)).items() if not ok
    )
    if uncommitted:
        result.warnings.append(
            f"these prompts are not committed: {uncommitted}. Fine for a smoke run; the full "
            "run should use --require-committed."
        )

    if dry_run:
        if report.planned != len(chosen) * len(conditions or SPEC) * samples:
            result.failures.append(f"dry run planned {report.planned} cells, which is not the grid")
        return result

    _audit(result, scenarios=chosen, samples=samples)
    return result


def _prompt_ids(conditions: Sequence[Condition] | None) -> list[str]:
    chosen = list(conditions) if conditions is not None else list(SPEC)
    return sorted({spec_for(c).prompt_id for c in chosen} | {"selfcheck.v1"})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _judge(result: SmokeResult) -> None:  # pragma: no cover - needs a second model
    """Extend the check through the judge, as `REPRODUCE.md` describes it.

    Opt-in: it loads a second model and `carelite/eval/judge/` belongs to
    another lane, so a break in it must not be able to fail this lane's wiring
    check by default.
    """
    from carelite.eval.judge import JudgeCache, LLMJudge, OllamaChatClient, judge_generations
    from carelite.scenarios.bank import by_id
    from carelite.types import Generation

    generations = [
        Generation(
            generation_id=r.generation_id,
            scenario_id=r.key.scenario_id,
            condition=Condition(r.key.condition),
            prompt_id=r.key.prompt_id,
            model=r.model,
            model_digest=r.key.model_digest,
            seed=r.key.seed,
            temperature=r.temperature,
            sample_idx=r.key.sample_idx,
            response=r.response,
        )
        for r in result.records
        if r.response.strip()
    ]
    texts = {g.scenario_id: by_id(g.scenario_id).text for g in generations}
    assert result.journal is not None
    with JudgeCache(result.journal.parent / "judge.jsonl") as cache:
        run_ = judge_generations(
            generations, texts, LLMJudge.for_full_run(OllamaChatClient(), cache=cache)
        )
    if run_.errors:
        for err in run_.errors[:5]:
            result.failures.append(f"judge: {err.generation_id}: {err.error_type}: {err.message}")
    if generations and not run_.results:
        result.failures.append("the judge scored nothing")
    else:
        result.warnings.append(
            f"judge scored {len(run_.results)} generations, complete_rate={run_.complete_rate:.0%}"
        )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(
        prog="carelite.eval.smoke",
        description=(
            "5 scenarios x all six conditions, end to end, on the train split. "
            "Run it before committing to the 1,080-cell holdout run."
        ),
    )
    parser.add_argument(
        "--split",
        choices=tuple(s.value for s in Split),
        default=Split.TRAIN.value,
        help="default train: a smoke test must not spend the holdout (see DECISIONS.md)",
    )
    parser.add_argument("--scenarios", type=int, default=DEFAULT_SCENARIOS)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--journal", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="plan only, touch no model")
    parser.add_argument("--with-judge", action="store_true", help="also run the LLM judge")
    args = parser.parse_args(argv)

    split = Split(args.split)
    if split is Split.HOLDOUT:
        print(
            "WARNING: --split holdout generates held-out responses. DECISIONS.md gates "
            "held-out generation on the OSF pre-registration being submitted.",
            file=sys.stderr,
            flush=True,
        )

    total = args.scenarios * len(SPEC) * args.samples

    def progress(cell: Any, index: int, _total: int) -> None:
        print(
            f"[{index}/{total}] {cell.scenario.scenario_id} {cell.spec.condition.value}",
            flush=True,
        )

    try:
        result = smoke(
            split=split,
            n_scenarios=args.scenarios,
            samples=args.samples,
            journal=args.journal,
            dry_run=args.dry_run,
            on_cell=None if args.dry_run else progress,
        )
    except Exception as exc:
        # A break upstream of the first generation — no index, no model, no bank
        # — is exactly what this target exists to catch, so it is reported as a
        # failure rather than as a traceback the reader has to interpret.
        print(f"carelite eval smoke\n  FAILED before any generation: {type(exc).__name__}: {exc}")
        return 1

    if args.with_judge and not args.dry_run:
        _judge(result)

    print(result.render())
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
