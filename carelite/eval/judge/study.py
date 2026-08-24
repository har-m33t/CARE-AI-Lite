"""Driver for the v3 §13 judge-validation study. **Train split only.**

    python -m carelite.eval.judge.study --stage subset      # what will be judged
    python -m carelite.eval.judge.study --stage generate    # 60 responses, train split
    python -m carelite.eval.judge.study --stage judge       # 5 samples @0.7, ascending
    python -m carelite.eval.judge.study --stage reversed    # positional-bias arm
    python -m carelite.eval.judge.study --stage report      # analyse; no model calls

`validation.py` is the study; this module is the thing that *runs* it. The
separation is not cosmetic. Everything in `validation.py` is a pure function of
cached judge output, so the analysis re-runs in milliseconds after a change to
the grounding rule, and only the stages here ever touch a model.

**The holdout is not touched, at any stage.** The OSF pre-registration is not
submitted, `DECISIONS.md` gates the holdout on it, and build plan §10's argument
— that pre-registration is what makes an against-you naturalness result
credible — is worth more than the convenience of judging a slightly better
sample. `_require_train` refuses any scenario whose own record says `holdout`,
checking the record rather than the flag, which is the same rule
`generate.runner` applies at line 237.

**Why 10 scenarios x 6 conditions x 1 sample, and not `--limit 60`.**

§13 wants ~60 responses. The obvious way to get them — `--split train --limit 60`
— is wrong here. `build_plan` is scenario-major and the bank is ordered by
scenario id, and ids are allocated in blocks of ten per challenge type, so the
first 60 cells are three challenge types and nothing else. That does not matter
for a study of the *conditions*, which is not what this is; it matters a great
deal for a study of the *judge*, whose failure modes are expected to be
content-shaped — a judge that cannot score `naturalness` on a trust rupture is
not discoverable in a sample of jargon questions.

So the subset is stratified: one scenario per challenge type (all ten), chosen
to spread encounter phase and to carry the bank's equity share, crossed with all
six conditions at `sample_idx=0`. Sixty responses, and every one of them is a
cell the full train run would produce anyway — same seed, same cache key — so
this is a subset of that run rather than a parallel artefact.

Sixty is also the smallest number that lets the pre-specified threshold mean
anything: `MIN_UNITS_FOR_CONFIRMATORY` is 30 paired units, and a subset that
lands near it demotes every dimension for a reason that has nothing to do with
the judge.

**On the human half.** There are no human ratings yet — they are sprint 10 — so
the validity half of §13 cannot be answered, and the `--stage report` verdict is
that all eleven dimensions are exploratory *for want of a comparator*. That is
the honest state of the world, not a placeholder. `--stage harness` separately
drives the whole human pipeline against synthetic raters to prove the
instrument works and would return `confirmatory` when agreement is genuinely
high; its numbers are about the harness, never about the judge, and the report
keeps them in a different section for that reason.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from carelite.config import get_settings
from carelite.eval.human.reliability import scores_by_rater
from carelite.eval.judge.cache import JudgeCache
from carelite.eval.judge.client import ChatClient, OllamaChatClient
from carelite.eval.judge.judge import JudgeResult, LLMJudge
from carelite.eval.judge.prompt import OptionOrder
from carelite.eval.judge.runner import JudgeRun, RunProgress, judge_generations
from carelite.eval.judge.validation import (
    MIN_ALPHA_FOR_CONFIRMATORY,
    MIN_RHO_FOR_CONFIRMATORY,
    MIN_UNITS_FOR_CONFIRMATORY,
    SpanReviewItem,
    SpanReviewVerdict,
    build_validation_report,
    classify_dimension,
    judge_among_raters_alpha,
    judge_human_validity,
    sample_spans_for_review,
)
from carelite.types import RUBRIC_DIMENSIONS, Condition, Generation, Scenario, Split

__all__ = [
    "N_REVERSED",
    "N_SUBSET_SCENARIOS",
    "SubsetCell",
    "balanced_order",
    "judge_subset",
    "load_generations",
    "main",
    "select_subset",
    "study_dir",
]

#: One per challenge type. Crossed with six conditions, this is the ~60 of §13.
N_SUBSET_SCENARIOS = 10

#: Generations re-judged with the anchor order reversed. Fewer than the full
#: subset because the arm costs a second five-sample pass over every cell it
#: covers; a paired delta over 30 cells detects a bias worth acting on, and
#: spending the same hours on the other 30 would buy a third decimal place.
N_REVERSED = 30


def study_dir() -> Path:
    return get_settings().runs_dir / "judge"


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SubsetCell:
    """One planned (scenario, condition) pair of the validation subset."""

    scenario_id: str
    condition: str
    challenge_type: str
    encounter_phase: str
    equity_stratum: bool


def _require_train(scenarios: Sequence[Scenario]) -> None:
    """Refuse anything the *scenario record* calls holdout.

    Checked on the record and not on the flag that fetched it, for the same
    reason `generate.runner._extra_payload` records the record's split: a flag
    says what was asked for and a record says what the thing is. A careless
    invocation must not be able to spend the holdout.
    """
    leaked = sorted(s.scenario_id for s in scenarios if s.split is not Split.TRAIN)
    if leaked:
        raise RuntimeError(
            "the judge-validation study runs on the train split only — the OSF "
            "pre-registration is not submitted and DECISIONS.md gates the holdout on "
            f"it. These scenarios say holdout: {leaked}"
        )


def select_subset(
    n_scenarios: int = N_SUBSET_SCENARIOS,
    *,
    conditions: Sequence[Condition] | None = None,
) -> tuple[list[Scenario], list[SubsetCell]]:
    """Pick a stratified slice of the train split, deterministically.

    One scenario per challenge type, taking the types in bank order. Within a
    type the pick is the scenario that keeps the running encounter-phase counts
    flattest, breaking ties toward the equity stratum and then by scenario id.
    Deterministic and seedless: a shuffle would give a defensible sample too,
    but this one is stable across machines and re-derivable by hand from the
    bank, which matters for a subset that gets quoted in a write-up.
    """
    from carelite.scenarios.bank import train_scenarios

    picks_by_type: dict[str, list[Scenario]] = defaultdict(list)
    order: list[str] = []
    for curated in train_scenarios():
        scenario = curated.to_scenario()
        if scenario.challenge_type not in picks_by_type:
            order.append(scenario.challenge_type)
        picks_by_type[scenario.challenge_type].append(scenario)

    phase_counts: dict[str, int] = defaultdict(int)
    chosen: list[Scenario] = []
    for challenge in order[:n_scenarios]:
        candidates = picks_by_type[challenge]
        best = min(
            candidates,
            key=lambda s: (
                phase_counts[str(s.encounter_phase)],
                0 if s.equity_stratum else 1,
                s.scenario_id,
            ),
        )
        phase_counts[str(best.encounter_phase)] += 1
        chosen.append(best)

    _require_train(chosen)

    wanted = list(conditions) if conditions is not None else list(Condition)
    cells = [
        SubsetCell(
            scenario_id=s.scenario_id,
            condition=c.value,
            challenge_type=s.challenge_type,
            encounter_phase=str(s.encounter_phase),
            equity_stratum=s.equity_stratum,
        )
        for s in chosen
        for c in wanted
    ]
    return chosen, cells


# ---------------------------------------------------------------------------
# Generation (drives the orchestrator lane's runner; owns none of it)
# ---------------------------------------------------------------------------


def generate_subset(
    scenarios: Sequence[Scenario],
    *,
    journal: Path,
    conditions: Sequence[Condition] | None = None,
) -> Any:
    """Generate the subset through the real runner, into a JSONL journal.

    `scenarios` is passed explicitly rather than `split=train`, so the runner
    generates exactly these cells and no others. It is the same code path, the
    same graph, the same seeds; `sample_idx=0` of each cell is the same row the
    full train run would write.
    """
    from carelite.generate.runner import run
    from carelite.generate.store import JsonlStore

    _require_train(scenarios)
    store = JsonlStore(path=journal)
    try:
        return run(
            store=store,
            scenarios=list(scenarios),
            conditions=list(conditions) if conditions is not None else None,
            samples=1,
            on_cell=lambda cell, i, n: print(
                f"[{i}/{n}] {cell.scenario.scenario_id} {cell.spec.condition.value}", flush=True
            ),
        )
    finally:
        store.close()


def load_generations(journal: Path) -> tuple[list[Generation], dict[str, str], dict[str, str]]:
    """Read a journal back as `(generations, scenario_texts, responses)`.

    Refuses any row whose sidecar says `holdout`. The journal is written by
    another lane and read here; a check on the boundary costs nothing and is the
    only place this module can catch a mislabelled row.
    """
    from carelite.generate.store import JsonlStore
    from carelite.scenarios.bank import load_bank

    texts = {s.scenario_id: s.text for s in load_bank()}
    generations: list[Generation] = []
    scenario_texts: dict[str, str] = {}
    responses: dict[str, str] = {}

    for record in JsonlStore(path=journal).read_all():
        split = str(record.extra.get("split", ""))
        if split and split != Split.TRAIN.value:
            raise RuntimeError(
                f"{journal} contains a {split!r} row ({record.key.scenario_id}); the "
                "judge-validation study must not read held-out data"
            )
        gid = record.generation_id
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
        responses[gid] = record.response
    return generations, scenario_texts, responses


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


def balanced_order(generations: Sequence[Generation]) -> list[Generation]:
    """Reorder so that *any prefix* is spread across scenarios and conditions.

    The journal is written scenario-major — SC-002 in all six conditions, then
    SC-017, and so on — because that is the order `build_plan` produces. Judging
    in that order means an interrupted run leaves a prefix that is a handful of
    whole scenarios: stop halfway through sixty and five of the ten challenge
    types are missing entirely, which is the one property the subset was chosen
    to avoid.

    Round-robin over conditions, taking scenarios in rotation within each, so
    the first six items are six conditions on six different scenarios and the
    first thirty cover every scenario and every condition. A partial run is then
    a smaller version of the study rather than a different one, and the number
    of judged generations is the only thing that has to be discounted.
    """
    by_condition: dict[str, list[Generation]] = {}
    for generation in generations:
        by_condition.setdefault(generation.condition.value, []).append(generation)
    for bucket in by_condition.values():
        bucket.sort(key=lambda g: g.scenario_id)

    order = sorted(by_condition)
    out: list[Generation] = []
    index = 0
    while len(out) < len(generations):
        progressed = False
        for offset, condition in enumerate(order):
            bucket = by_condition[condition]
            if not bucket:
                continue
            # Rotate the scenario cursor per condition so consecutive items do
            # not land on the same scenario.
            pick = (index + offset) % len(bucket)
            out.append(bucket.pop(pick))
            progressed = True
        if not progressed:
            break
        index += 1
    return out


def judge_subset(
    generations: Sequence[Generation],
    scenario_texts: Mapping[str, str],
    *,
    order: OptionOrder,
    cache_path: Path,
    client: ChatClient | None = None,
) -> JudgeRun:
    """Run the five-sample validation regime over `generations`, resumably.

    The cache is per anchor order by construction — `order` is in the key — so
    the ascending and descending arms share a file without either serving the
    other's answers.
    """
    started = time.monotonic()

    def progress(p: RunProgress) -> None:
        done = p.index
        rate = (time.monotonic() - started) / done if done else 0.0
        eta = rate * (p.total - done)
        print(
            f"[{done}/{p.total}] {p.generation_id} {'ok' if p.ok else 'ERR'} "
            f"{p.elapsed_s:.1f}s  eta {eta / 60:.0f}m",
            flush=True,
        )

    with JudgeCache(cache_path) as cache:
        judge = LLMJudge.for_validation(
            client or OllamaChatClient(),
            cache=cache,
            order=order,
            rater_id=f"judge-{order.value}",
        )
        return judge_generations(generations, scenario_texts, judge, on_progress=progress)


def replay_from_cache(
    generations: Sequence[Generation],
    scenario_texts: Mapping[str, str],
    *,
    order: OptionOrder,
    cache_path: Path,
    min_samples: int = 1,
) -> list[JudgeResult]:
    """Re-derive results from cached raw output, calling no model.

    Grounding, medians and every §13 statistic are pure functions of bytes
    already on disk, so the whole study re-analyses in milliseconds after a
    change to the grounding rule. Nothing here may reach a model: a
    `--stage report` that quietly starts a six-hour inference run is the wrong
    kind of surprise, and `_NoCallClient` turns a cache miss into a skip rather
    than a call.

    **Samples are collected individually, not all-or-nothing.** A six-hour run
    gets interrupted, and it stops in the middle of a generation as often as
    between two. Rebuilding only the generations whose five samples are all
    present would throw away the work done on the generation in flight and —
    worse — bias what survives: the generations that happen to be complete are
    not a random subset when the run stopped partway through a sweep. A
    generation with three cached samples is reported from three, and
    `DimensionResult.variance` already refuses to compute a self-consistency
    number from fewer than two.

    Args:
        min_samples: Generations with fewer cached samples than this are
            dropped. One is right for grounding and agreement, which are
            per-sample; the self-consistency table needs two and enforces that
            itself.
    """
    from carelite.eval.judge.judge import _aggregate

    results: list[JudgeResult] = []
    client = _NoCallClient(get_settings().models.judge.tag)

    with JudgeCache(cache_path) as cache:
        judge = LLMJudge.for_validation(
            client, cache=cache, order=order, rater_id=f"judge-{order.value}"
        )
        for generation in generations:
            samples = []
            for index in range(judge.n_samples):
                try:
                    samples.append(
                        judge.judge_sample(
                            generation_id=generation.generation_id,
                            scenario_text=scenario_texts.get(generation.scenario_id, ""),
                            response_text=generation.response,
                            sample_idx=index,
                        )
                    )
                except _WouldCallModel:
                    continue
            if len(samples) < min_samples:
                continue

            flags: list[str] = []
            for sample in samples:
                for flag in sample.safety_flags:
                    if flag not in flags:
                        flags.append(flag)

            results.append(
                JudgeResult(
                    generation_id=generation.generation_id,
                    judge_model=client.model,
                    judge_digest=client.digest,
                    prompt_version=judge.prompt_version,
                    rubric_version=judge.rubric_version,
                    temperature=judge.temperature,
                    n_samples_requested=judge.n_samples,
                    order=order,
                    dimensions={key: _aggregate(key, samples) for key in RUBRIC_DIMENSIONS},
                    samples=tuple(samples),
                    safety_flags=tuple(flags),
                    rater_id=judge.rater_id,
                )
            )
    return results


class _WouldCallModel(RuntimeError):
    """Raised instead of reaching a model during a cache-only replay."""


@dataclass
class _NoCallClient:
    """A `ChatClient` that refuses to call. Turns a cache miss into a skip."""

    model_tag: str

    @property
    def model(self) -> str:
        return self.model_tag

    @property
    def digest(self) -> str:
        return get_settings().models.judge.digest or self.model_tag

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        seed: int | None = None,
    ) -> str:
        raise _WouldCallModel("cache miss during a cache-only replay")


# ---------------------------------------------------------------------------
# Span review worksheet
# ---------------------------------------------------------------------------


def write_span_worksheet(items: Sequence[SpanReviewItem], path: Path) -> None:
    """Write the 30 sampled spans out for a reviewer to adjudicate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "item_id": i.item_id,
            "generation_id": i.generation_id,
            "dimension": i.dimension,
            "score": i.score,
            "span": i.span,
            "rationale": i.rationale,
            "response": i.response,
            "supports": None,
            "note": "",
        }
        for i in items
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_span_verdicts(path: Path) -> list[SpanReviewVerdict]:
    """Read an adjudicated worksheet. Unanswered items are dropped, not assumed."""
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        SpanReviewVerdict(
            item_id=str(r["item_id"]),
            dimension=str(r["dimension"]),
            supports=bool(r["supports"]),
            note=str(r.get("note", "")),
        )
        for r in rows
        if r.get("supports") is not None
    ]


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _finite(value: float) -> float | None:
    """JSON has no NaN. An undefined coefficient is `null`, not a number."""
    return None if value is None or math.isnan(value) else round(float(value), 6)


def report_to_json(report: Any, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The §13 report as a JSON-safe dict, per dimension."""
    out: dict[str, Any] = {
        "plan_version": report.plan_version,
        "rubric_version": report.rubric_version,
        "prompt_version": report.prompt_version,
        "judge_model": report.judge_model,
        "judge_digest": report.judge_digest,
        "generator_model": report.generator_model,
        "n_generations": report.n_generations,
        "confirmatory_dimensions": report.confirmatory_dimensions,
        "exploratory_dimensions": report.exploratory_dimensions,
        "grounding": {
            "n_attempted": report.grounding.n_attempted,
            "n_admitted": report.grounding.n_admitted,
            "n_rejected": report.grounding.n_rejected,
            "admitted_rate": _finite(report.grounding.admitted_rate),
            "exact_rate": _finite(report.grounding.exact_rate),
            "presented_only_rate": _finite(report.grounding.presented_only_rate),
            "reasons": dict(report.grounding.reasons),
            "per_dimension": {k: _finite(v) for k, v in report.grounding.per_dimension.items()},
        },
        "span_support": None,
        "dimensions": {},
    }
    if report.span_support is not None:
        s = report.span_support
        out["span_support"] = {
            "n_reviewed": s.n_reviewed,
            "n_supported": s.n_supported,
            "support_rate": _finite(s.support_rate),
            "ci_low": _finite(s.ci_low),
            "ci_high": _finite(s.ci_high),
            "per_dimension": {k: _finite(v) for k, v in s.per_dimension.items()},
        }

    for key in RUBRIC_DIMENSIONS:
        validity = report.validity.get(key)
        consistency = report.self_consistency.get(key)
        bias = report.positional_bias.get(key)
        out["dimensions"][key] = {
            "alpha": _finite(validity.agreement.alpha) if validity else None,
            "rho": _finite(validity.agreement.rho) if validity else None,
            "rho_p": _finite(validity.agreement.rho_p) if validity else None,
            "n_units": validity.agreement.n_units if validity else 0,
            "status": str(validity.status) if validity else None,
            "self_consistency": None
            if consistency is None
            else {
                "n_generations": consistency.n_generations,
                "mean_variance": _finite(consistency.mean_variance),
                "mean_sd": _finite(consistency.mean_sd),
                "mean_range": _finite(consistency.mean_range),
                "pct_unanimous": _finite(consistency.pct_unanimous),
                "pct_range_ge_2": _finite(consistency.pct_range_ge_2),
            },
            "positional_bias": None
            if bias is None
            else {
                "n_paired": bias.n_paired,
                "mean_signed_delta": _finite(bias.mean_signed_delta),
                "mean_abs_delta": _finite(bias.mean_abs_delta),
                "pct_shift_ge_1": _finite(bias.pct_shift_ge_1),
            },
        }
    if extra:
        out.update(dict(extra))
    return out


# ---------------------------------------------------------------------------
# Agreement against synthetic raters — an instrument check, never a result
# ---------------------------------------------------------------------------


def agreement_against_synthetic(
    results: Sequence[JudgeResult],
    responses: Mapping[str, str],
) -> dict[str, Any]:
    """Drive `judge_human_validity`, `judge_among_raters_alpha` and the threshold.

    Run twice, against two synthetic panels, because one run cannot tell a
    working threshold from a broken one:

    * the **null control** panel shares a latent truth unrelated to the judge,
      so judge-human agreement must land near zero and every dimension must be
      demoted to `exploratory`;
    * the **positive control** panel observes the judge's own scores through
      noise, so agreement must be high and dimensions must clear the threshold.

    Neither number says anything about the judge. What they establish is that
    the machinery `carelite-stats` is about to consume — the per-dimension
    `EvidenceStatus` — moves with the data instead of being pinned to one
    answer. The real validity question stays unanswered until human rating in
    sprint 10, and `--stage report` records exactly that.

    `ritualistic` needs no special handling anywhere in here: every comparison
    runs through `judge_human_validity` / `judge_among_raters_alpha`, both of
    which canonicalise with `to_quality` internally, and the synthetic raters
    produce raw scores exactly as a human sheet does. The reversal happens once,
    on both sides, in one place.
    """
    from carelite.eval.human.blinding import RateableItem
    from carelite.eval.human.dry_run import TruthModel, dry_run, to_json

    judge_scores = {r.generation_id: r.scores() for r in results}
    items = [
        RateableItem(
            generation_id=r.generation_id,
            scenario_text="",
            response_text=responses.get(r.generation_id, ""),
            condition="blinded",
        )
        for r in results
    ]

    arms: dict[str, Any] = {}
    for label, model in (
        ("null_control", TruthModel.INDEPENDENT),
        ("positive_control", TruthModel.JUDGE_ANCHORED),
    ):
        run = dry_run(items, truth_model=model, judge_scores=judge_scores)
        validity = judge_human_validity(results, run.consensus)
        among = judge_among_raters_alpha(results, scores_by_rater(run.panel_scores))
        for key, v in validity.items():
            recomputed = classify_dimension(v.agreement.alpha, v.agreement.rho, v.agreement.n_units)
            if recomputed is not v.status:  # pragma: no cover - a contract breach
                raise AssertionError(
                    f"{key}: threshold applied inside judge_human_validity "
                    f"({v.status}) disagrees with classify_dimension ({recomputed})"
                )

        arms[label] = {
            "harness": to_json(run),
            "dimensions": {
                key: {
                    "alpha": _finite(v.agreement.alpha),
                    "rho": _finite(v.agreement.rho),
                    "n_units": v.agreement.n_units,
                    # Re-derived from the coefficients rather than read off
                    # `v.status`, and asserted equal below. `judge_human_validity`
                    # applies the threshold internally; recomputing it here from
                    # the three published numbers is what makes the artifact
                    # auditable — a reader can check the verdict against the rule
                    # without rerunning anything.
                    "status": str(
                        classify_dimension(v.agreement.alpha, v.agreement.rho, v.agreement.n_units)
                    ),
                    "alpha_with_judge_as_rater": _finite(among[key]),
                    "alpha_humans_only": _finite(run.inter_rater[key]["alpha"]),
                }
                for key, v in validity.items()
            },
            "n_confirmatory": sum(1 for v in validity.values() if str(v.status) == "confirmatory"),
        }

    # The threshold is applied to the numbers above by `classify_dimension`; it
    # is restated here so the artifact records what was applied, not just what
    # came out of it.
    arms["threshold"] = {
        "min_alpha": MIN_ALPHA_FOR_CONFIRMATORY,
        "min_rho": MIN_RHO_FOR_CONFIRMATORY,
        "min_units": MIN_UNITS_FOR_CONFIRMATORY,
        "rule": "alpha >= min_alpha AND rho >= min_rho AND n_units >= min_units",
    }
    return arms


def render_agreement(arms: Mapping[str, Any]) -> str:
    """The two control arms as one table each."""
    lines: list[str] = []
    lines.append("AGREEMENT AGAINST SYNTHETIC RATERS — INSTRUMENT CHECK, NOT A RESULT")
    lines.append(
        "  There are no human ratings yet. These panels are generated, so the "
        "coefficients below measure whether the threshold machinery responds to "
        "signal — not whether the judge agrees with anyone."
    )
    t = arms["threshold"]
    lines.append(
        f"  Threshold: alpha >= {t['min_alpha']} AND rho >= {t['min_rho']} "
        f"on >= {t['min_units']} units."
    )
    for label in ("null_control", "positive_control"):
        arm = arms[label]
        expectation = (
            "expect ~0 agreement and 0 confirmatory"
            if label == "null_control"
            else "expect high agreement and most dimensions confirmatory"
        )
        lines.append("")
        lines.append(f"  {label.upper()} ({expectation})")
        lines.append(
            f"    {'dimension':<14}{'alpha':>8}{'rho':>8}{'n':>5}  {'status':<13}"
            f"{'a_humans':>9}{'a_w/judge':>11}"
        )
        for key in RUBRIC_DIMENSIONS:
            d = arm["dimensions"][key]
            lines.append(
                f"    {key:<14}{_fmt(d['alpha']):>8}{_fmt(d['rho']):>8}{d['n_units']:>5}  "
                f"{d['status']:<13}{_fmt(d['alpha_humans_only']):>9}"
                f"{_fmt(d['alpha_with_judge_as_rater']):>11}"
            )
        lines.append(f"    confirmatory: {arm['n_confirmatory']}/11")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


def _provenance() -> dict[str, Any]:
    """What was actually run, read from the daemon rather than from config.

    `settings.models.judge.digest` is unset until `make pin-models`, and
    `OllamaChatClient.digest` then falls back to the tag — which is the right
    fallback for a cache key but is not a provenance record. The digest the
    daemon reports is the real identity of the weights that produced these
    numbers, so it is recorded here even though nothing else consumes it.
    """
    settings = get_settings()
    out: dict[str, Any] = {
        "judge_tag": settings.models.judge.tag,
        "generator_tag": settings.models.generator.tag,
        "judge_digest_config": settings.models.judge.digest,
        "judge_digest_daemon": None,
        "cross_family": True,
        "judge_temperature": settings.experiment.judge_temperature_validation,
        "judge_samples": settings.experiment.judge_samples_validation,
    }
    try:
        import ollama

        for model in ollama.Client(host=settings.ollama_host).list().get("models", []):
            if str(model.get("model", "")).startswith(settings.models.judge.tag.split(":")[0]):
                out["judge_digest_daemon"] = model.get("digest")
    except Exception:  # provenance is best-effort; a dead daemon must not fail a replay
        pass
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(
        prog="carelite.eval.judge.study",
        description="Run the v3 §13 judge-validation study on the train split.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("subset", "generate", "judge", "reversed", "spans", "harness", "report"),
    )
    parser.add_argument("--scenarios", type=int, default=N_SUBSET_SCENARIOS)
    parser.add_argument("--n-reversed", type=int, default=N_REVERSED)
    parser.add_argument("--journal", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap generations judged")
    args = parser.parse_args(argv)

    out = study_dir()
    out.mkdir(parents=True, exist_ok=True)
    journal = args.journal or (get_settings().runs_dir / "generate" / "validation-subset.jsonl")

    scenarios, cells = select_subset(args.scenarios)

    if args.stage == "subset":
        print(
            f"{len(scenarios)} scenarios x {len(cells) // len(scenarios)} conditions "
            f"= {len(cells)} responses (train split)"
        )
        for s in scenarios:
            print(
                f"  {s.scenario_id:<8} {s.challenge_type:<22} {s.encounter_phase!s:<22} "
                f"equity={s.equity_stratum}"
            )
        (out / "subset.json").write_text(
            json.dumps([asdict(c) for c in cells], indent=2), encoding="utf-8"
        )
        return 0

    if args.stage == "generate":
        report = generate_subset(scenarios, journal=journal)
        print(report.summary())
        print(f"  {report.route_summary()}")
        for err in report.errors[:20]:
            print(f"  error: {err}", file=sys.stderr)
        return 1 if report.failed else 0

    generations, scenario_texts, responses = load_generations(journal)
    if args.limit:
        generations = generations[: args.limit]

    if args.stage in {"judge", "reversed"}:
        order = OptionOrder.ASCENDING if args.stage == "judge" else OptionOrder.DESCENDING
        ordered = balanced_order(generations)
        items = ordered if order is OptionOrder.ASCENDING else ordered[: args.n_reversed]
        run = judge_subset(
            items,
            scenario_texts,
            order=order,
            cache_path=out / f"validation-{order.value}.jsonl",
        )
        print(
            f"judged={run.n_judged} errors={len(run.errors)} "
            f"cached={run.n_from_cache} called={run.n_called} "
            f"complete_rate={run.complete_rate:.1%} elapsed={run.elapsed_s / 60:.1f}m"
        )
        for err in run.errors[:20]:
            print(f"  error {err.generation_id}: {err.error_type}: {err.message}", file=sys.stderr)
        return 0

    ascending = replay_from_cache(
        generations,
        scenario_texts,
        order=OptionOrder.ASCENDING,
        cache_path=out / "validation-ascending.jsonl",
    )

    if args.stage == "spans":
        spans = sample_spans_for_review(ascending, responses)
        write_span_worksheet(spans, out / "span_review.json")
        print(f"wrote {len(spans)} spans to {out / 'span_review.json'}")
        return 0

    if args.stage == "harness":
        arms = agreement_against_synthetic(ascending, responses)
        text = render_agreement(arms)
        print(text)
        (out / "agreement_synthetic.txt").write_text(text, encoding="utf-8")
        (out / "agreement_synthetic.json").write_text(json.dumps(arms, indent=2), encoding="utf-8")
        return 0

    descending = replay_from_cache(
        balanced_order(generations)[: args.n_reversed],
        scenario_texts,
        order=OptionOrder.DESCENDING,
        cache_path=out / "validation-descending.jsonl",
    )
    report = build_validation_report(
        validation_results=ascending,
        responses=responses,
        human_consensus=None,
        reversed_results=descending,
        span_verdicts=read_span_verdicts(out / "span_review.json"),
        generator_model=get_settings().models.generator.tag,
    )

    # The validity half of §13 has no comparator yet, and that is the verdict —
    # not a gap to be filled in with the synthetic panel sitting one function
    # away. `human_consensus=None` above is what puts every dimension in
    # `exploratory`; the arms below are kept in a separate key, with a separate
    # name, so nothing downstream can read them as the study's agreement result.
    arms = agreement_against_synthetic(ascending, responses) if ascending else {}

    payload = report_to_json(
        report,
        extra={
            "verdict": {
                "human_ratings_exist": False,
                "agreement_computable": False,
                "reason": (
                    "Human rating is sprint 10. With no human consensus there is no "
                    "comparator, so every dimension is exploratory for want of one — "
                    "which is a statement about the study's stage, not about the judge. "
                    "Nothing here may be reported as a judge-validity finding."
                ),
                "all_dimensions_exploratory": len(report.exploratory_dimensions) == 11,
            },
            "instrument_check_not_a_result": arms,
            "provenance": _provenance(),
            "subset": [asdict(c) for c in cells],
        },
    )

    text = report.render()
    if arms:
        text = f"{text}\n\n{render_agreement(arms)}"
    print(text)
    (out / "validation_report.txt").write_text(text, encoding="utf-8")
    (out / "validation_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'validation_report.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
