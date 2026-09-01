"""Do the two serving stacks score the same? The paired LC sample, and its limits.

    python -m carelite.eval.judge.backend_equivalence --out runs/judge-lc-vllm

**What this compares.** D11 stopped condition LC at 39 cells served by Ollama.
D13 re-opened it and generated all 180 cells under vLLM. The 39 Ollama cells and
39 of the vLLM cells share scenario, condition, sample index, seed and prompt id,
so they are paired observations of the same experimental cell under two serving
stacks. This module scores that pairing with the judge rows already in
`rubric_score` — no new judging, no new generation.

**What it can support.** That the judge's readings of the two stacks' output on
these cells do or do not agree, per dimension, at this sample size. That is the
§4 W5 check: *report agreement; do not pool arms that disagree.*

**What it cannot support, stated first because it is the part that gets dropped.**

* The stacks differ in more than the server. A GGUF against HF safetensors,
  different quantisation, different sampling defaults, and — per D13 — a
  different realised context pack, since the production packing rule admits
  116/116 knowledge base entries and 151/471 chunks at 117,849 real tokens and
  that is not what the Ollama run's window admitted. **A disagreement therefore
  does not isolate the serving stack**, and an agreement does not show the two
  are interchangeable in general.
* Generation ran at temperature 0.7. Identical seeds do not produce identical
  text across two sampling implementations, so even a perfectly faithful pair of
  stacks would differ here by sampling alone.
* 39 cells over 13 scenarios, never randomised for partial analysis — they are
  the scenarios LC happened to reach before D11 stopped it. At 13 scenario-level
  pairs the smallest paired effect detectable at 80% power is dz ~ 0.86, which is
  enormous. A null here is close to uninformative and the report says the
  resolution rather than printing a bare p-value.

**Units.** Agreement is computed over the 39 cell pairs, because agreement is a
question about the instrument's reading of matched cells. The distribution tests
are computed over 13 scenario-level cell means, because the three samples in a
cell are not independent and counting them as 39 units would inflate every test.
Both n's are on the result; neither is reported without the other.

Everything stays EXPLORATORY. D10 dropped the pre-registration and the judge
validation study still has not run.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carelite.eval.judge.agreement import Metric, krippendorff_alpha, spearman_rho
from carelite.eval.judge.arms import (
    LC_ANALYSIS_BACKEND,
    LC_EQUIVALENCE_BACKEND,
    UnpairableCells,
    pair_cells,
)
from carelite.eval.judge.validation import (
    MIN_ALPHA_FOR_CONFIRMATORY,
    MIN_RHO_FOR_CONFIRMATORY,
)
from carelite.types import RUBRIC_DIMENSIONS, Condition

__all__ = [
    "BACKEND_CONFOUNDS",
    "MIN_UNITS_FOR_EQUIVALENCE_CLAIM",
    "BackendEquivalence",
    "DimensionEquivalence",
    "compare_backends",
    "fetch_paired_scores",
    "main",
    "run_backend_equivalence",
]

#: Below this many *scenario-level* pairs, no agreement coefficient is stable
#: enough to license "these arms may be pooled", whatever its value. It is the
#: same cut `validation.MIN_UNITS_FOR_CONFIRMATORY` applies to judge-human
#: agreement, and for the same reason: a coefficient on a dozen units is a
#: description of a dozen units.
MIN_UNITS_FOR_EQUIVALENCE_CLAIM = 30

#: Everything that differs between the two stacks besides the server. These are
#: carried on the result, not left to the prose, because a reader who sees a
#: disagreement will otherwise attribute it to the one variable in the title.
BACKEND_CONFOUNDS: tuple[str, ...] = (
    "Model artifact: Ollama serves a GGUF, vLLM serves HF safetensors. Same model "
    "family, different files.",
    "Quantisation differs between the two artifacts and was not held constant.",
    "Sampling defaults differ between the two runtimes, and generation ran at "
    "temperature 0.7, so identical seeds do not produce identical text.",
    "Realised context pack differs (D13): the production packing rule admits "
    "116/116 knowledge base entries and 151/471 chunks at 117,849 real tokens, "
    "which is not what the Ollama run's window admitted.",
    "Different hardware: an L40S for the Ollama cells, an A100 SXM 80GB for vLLM.",
)


def _quality(dimension: str, value: float | None) -> float | None:
    """Raw -> quality, so `ritualistic` points the same way as its neighbours."""
    from carelite.eval.rubric.dimensions import to_quality

    return None if value is None else float(to_quality(dimension, int(value)))


@dataclass(frozen=True, slots=True)
class DimensionEquivalence:
    """Backend agreement on one rubric dimension, with the n that produced it."""

    dimension: str
    n_pairs: int
    n_exact: int
    n_within_one: int
    mean_left: float
    mean_right: float
    #: `right - left` on the quality scale. Positive means the right-hand stack
    #: scored higher, on every dimension including the reverse-coded one.
    mean_difference: float
    alpha: float
    rho: float
    rho_p: float
    #: Scenario-level: the paired test and its three point estimators.
    n_scenario_pairs: int
    wilcoxon_p: float
    rank_biserial: float
    rank_biserial_ci: tuple[float, float]
    cohens_dz: float
    hodges_lehmann: float
    #: Raw-scale value counts per stack, so a degenerate dimension is visible.
    distribution_left: Mapping[int, int] = field(default_factory=dict)
    distribution_right: Mapping[int, int] = field(default_factory=dict)

    @property
    def exact_agreement(self) -> float:
        return self.n_exact / self.n_pairs if self.n_pairs else math.nan

    @property
    def within_one_agreement(self) -> float:
        return self.n_within_one / self.n_pairs if self.n_pairs else math.nan

    @property
    def degenerate(self) -> bool:
        """One value used on both sides: perfectly agreed and measuring nothing."""
        return len(set(self.distribution_left) | set(self.distribution_right)) <= 1

    @property
    def clears_threshold(self) -> bool:
        """Ordinal alpha and rho both clear the study's pre-set agreement cuts.

        A `nan` alpha does not clear it. Alpha is `nan` when the observed values
        carry no variance at all, and "both stacks said 3 every time" is not
        evidence that they agree about anything — it is evidence the dimension
        did not discriminate.
        """
        if math.isnan(self.alpha) or math.isnan(self.rho):
            return False
        return self.alpha >= MIN_ALPHA_FOR_CONFIRMATORY and self.rho >= MIN_RHO_FOR_CONFIRMATORY

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "n_pairs": self.n_pairs,
            "exact_agreement": self.exact_agreement,
            "within_one_agreement": self.within_one_agreement,
            "mean_left": self.mean_left,
            "mean_right": self.mean_right,
            "mean_difference": self.mean_difference,
            "alpha_ordinal": self.alpha,
            "spearman_rho": self.rho,
            "spearman_p": self.rho_p,
            "n_scenario_pairs": self.n_scenario_pairs,
            "wilcoxon_p": self.wilcoxon_p,
            "rank_biserial": self.rank_biserial,
            "rank_biserial_ci": list(self.rank_biserial_ci),
            "cohens_dz": self.cohens_dz,
            "hodges_lehmann": self.hodges_lehmann,
            "distribution_left": {str(k): v for k, v in sorted(self.distribution_left.items())},
            "distribution_right": {str(k): v for k, v in sorted(self.distribution_right.items())},
            "degenerate": self.degenerate,
            "clears_threshold": self.clears_threshold,
        }


@dataclass(frozen=True, slots=True)
class BackendEquivalence:
    """The whole comparison: agreement, distribution, and what the sample resolves."""

    condition: str
    left_backend: str
    right_backend: str
    n_cell_pairs: int
    n_scenarios: int
    scenario_ids: tuple[str, ...]
    dimensions: tuple[DimensionEquivalence, ...]
    detectable_dz_cells: float
    detectable_dz_scenarios: float
    confounds: tuple[str, ...] = BACKEND_CONFOUNDS
    #: Cells excluded because the output gate refused them (D12), if any.
    n_gate_blocked_excluded: int = 0

    @property
    def supports_equivalence_claim(self) -> bool:
        """Is the sample large enough for "these arms may be pooled" to mean anything?"""
        return self.n_scenarios >= MIN_UNITS_FOR_EQUIVALENCE_CLAIM

    @property
    def dimensions_failing_threshold(self) -> tuple[str, ...]:
        return tuple(d.dimension for d in self.dimensions if not d.clears_threshold)

    @property
    def poolable(self) -> bool:
        """Both halves must hold: enough units, and agreement on every dimension.

        This is expected to be `False` on this study's data and that is not a
        failure of the check. D13 already decided the two stacks are not pooled;
        this reports the evidence for that decision rather than reversing it.
        """
        return self.supports_equivalence_claim and not self.dimensions_failing_threshold

    @property
    def limits(self) -> tuple[str, ...]:
        return (
            f"{self.n_cell_pairs} paired cells over {self.n_scenarios} scenarios, "
            f"never randomised for partial analysis: they are the scenarios condition "
            f"{self.condition} happened to reach before D11 stopped it.",
            f"At {self.n_scenarios} scenario-level pairs the smallest paired effect "
            f"detectable at 80% power is dz = {self.detectable_dz_scenarios:.2f}. "
            f"A non-significant result at this n is not evidence of agreement.",
            "The two stacks differ in artifact, quantisation, sampling defaults, "
            "realised context pack and hardware, so a disagreement does not isolate "
            "the serving stack and an agreement does not make them interchangeable.",
            "EXPLORATORY. D10 dropped the pre-registration and the judge validation "
            "study has not run, so no result here is confirmatory.",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "left_backend": self.left_backend,
            "right_backend": self.right_backend,
            "n_cell_pairs": self.n_cell_pairs,
            "n_scenarios": self.n_scenarios,
            "scenario_ids": list(self.scenario_ids),
            "n_gate_blocked_excluded": self.n_gate_blocked_excluded,
            "detectable_dz_cells": self.detectable_dz_cells,
            "detectable_dz_scenarios": self.detectable_dz_scenarios,
            "min_units_for_equivalence_claim": MIN_UNITS_FOR_EQUIVALENCE_CLAIM,
            "alpha_threshold": MIN_ALPHA_FOR_CONFIRMATORY,
            "rho_threshold": MIN_RHO_FOR_CONFIRMATORY,
            "supports_equivalence_claim": self.supports_equivalence_claim,
            "poolable": self.poolable,
            "dimensions_failing_threshold": list(self.dimensions_failing_threshold),
            "dimensions": [d.to_dict() for d in self.dimensions],
            "confounds": list(self.confounds),
            "limits": list(self.limits),
            "reporting": "EXPLORATORY",
        }

    def render(self) -> str:
        head = (
            f"Backend equivalence, condition {self.condition}: "
            f"{self.left_backend} vs {self.right_backend}\n"
            f"  {self.n_cell_pairs} paired cells over {self.n_scenarios} scenarios "
            f"({self.n_gate_blocked_excluded} gate-blocked cells excluded)\n"
            f"  resolution: dz >= {self.detectable_dz_scenarios:.2f} at the scenario level, "
            f"dz >= {self.detectable_dz_cells:.2f} at the cell level\n"
        )
        header = (
            f"  {'dimension':<12} {'n':>3} {'exact':>7} {'+-1':>7} "
            f"{f'mean {self.left_backend}':>13} {f'mean {self.right_backend}':>13} "
            f"{'diff':>7} {'alpha':>7} {'rho':>7} {'wilcox p':>9}\n"
        )
        rows = "".join(
            f"  {d.dimension:<12} {d.n_pairs:>3} "
            f"{_pct(d.exact_agreement):>7} {_pct(d.within_one_agreement):>7} "
            f"{_num(d.mean_left):>13} {_num(d.mean_right):>13} "
            f"{_num(d.mean_difference, sign=True):>7} {_num(d.alpha):>7} "
            f"{_num(d.rho):>7} {_num(d.wilcoxon_p):>9}"
            + ("  <- degenerate" if d.degenerate else "")
            + "\n"
            for d in self.dimensions
        )
        verdict = (
            f"  poolable: {self.poolable}  "
            f"(needs n >= {MIN_UNITS_FOR_EQUIVALENCE_CLAIM} scenarios and every dimension "
            f"at alpha >= {MIN_ALPHA_FOR_CONFIRMATORY} and rho >= {MIN_RHO_FOR_CONFIRMATORY})\n"
        )
        limits = "".join(f"  - {line}\n" for line in self.limits)
        return head + header + rows + verdict + "  limits:\n" + limits


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.1%}"


def _num(value: float, *, sign: bool = False) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:+.3f}" if sign else f"{value:.3f}"


def compare_backends(
    left: Mapping[str, Mapping[str, int | None]],
    right: Mapping[str, Mapping[str, int | None]],
    *,
    scenario_of: Mapping[str, str],
    left_backend: str,
    right_backend: str,
    condition: str = Condition.LC.value,
    n_gate_blocked_excluded: int = 0,
    n_boot: int = 2000,
) -> BackendEquivalence:
    """Agreement and distribution difference between two stacks, per dimension.

    Args:
        left: `unit id -> dimension -> raw score`, the reference stack.
        right: the same, for the stack being compared against it.
        scenario_of: `unit id -> scenario id`. Every paired unit must be in it;
            a missing entry means the scenario-level test would silently drop a
            cell, so it raises.
        left_backend, right_backend: the two `served_by` values. They must
            differ — comparing a stack against itself measures sampling noise.
        n_boot: bootstrap resamples for the effect-size intervals.

    Raises:
        ValueError: if the two backends are the same.
        KeyError: if a paired unit has no scenario.
    """
    from carelite.stats.effects import paired_effects
    from carelite.stats.power import detectable_effect
    from carelite.stats.primary import wilcoxon_paired

    if left_backend == right_backend:
        raise ValueError(
            f"both sides are the same serving stack ({left_backend!r}); pairing a "
            f"backend against itself measures sampling noise, not the backend"
        )

    units = sorted(set(left) & set(right))
    missing = [u for u in units if u not in scenario_of]
    if missing:
        raise KeyError(
            f"no scenario recorded for {len(missing)} paired unit(s): {missing[:5]}; "
            f"the scenario-level test cannot run on a cell it cannot group"
        )
    scenarios = sorted({scenario_of[u] for u in units})

    dimensions: list[DimensionEquivalence] = []
    for dim in RUBRIC_DIMENSIONS:
        pairs: list[tuple[str, float, float]] = []
        raw_left: dict[int, int] = {}
        raw_right: dict[int, int] = {}
        for unit in units:
            a_raw, b_raw = left[unit].get(dim), right[unit].get(dim)
            if a_raw is not None:
                raw_left[int(a_raw)] = raw_left.get(int(a_raw), 0) + 1
            if b_raw is not None:
                raw_right[int(b_raw)] = raw_right.get(int(b_raw), 0) + 1
            a, b = _quality(dim, a_raw), _quality(dim, b_raw)
            if a is None or b is None:
                continue
            pairs.append((unit, a, b))

        n = len(pairs)
        xs = [p[1] for p in pairs]
        ys = [p[2] for p in pairs]
        n_exact = sum(1 for _, a, b in pairs if a == b)
        n_within_one = sum(1 for _, a, b in pairs if abs(a - b) <= 1)

        alpha = krippendorff_alpha([xs, ys], metric=Metric.ORDINAL) if n else math.nan
        rho, rho_p = spearman_rho(xs, ys) if n else (math.nan, math.nan)

        # Scenario level: the three samples in a cell are not independent.
        by_scenario_left: dict[str, list[float]] = {}
        by_scenario_right: dict[str, list[float]] = {}
        for unit, a, b in pairs:
            by_scenario_left.setdefault(scenario_of[unit], []).append(a)
            by_scenario_right.setdefault(scenario_of[unit], []).append(b)
        shared = sorted(set(by_scenario_left) & set(by_scenario_right))
        sl = [statistics.fmean(by_scenario_left[s]) for s in shared]
        sr = [statistics.fmean(by_scenario_right[s]) for s in shared]

        if len(shared) >= 3:
            wil = wilcoxon_paired(sr, sl)
            eff = paired_effects(sr, sl, n_boot=n_boot)
            wilcoxon_p = wil.p_value
            rb, rb_ci = eff.rank_biserial.point, eff.rank_biserial.interval
            dz, hl = eff.cohens_dz.point, eff.hodges_lehmann.point
        else:
            wilcoxon_p = math.nan
            rb, rb_ci = math.nan, (math.nan, math.nan)
            dz, hl = math.nan, math.nan

        dimensions.append(
            DimensionEquivalence(
                dimension=dim,
                n_pairs=n,
                n_exact=n_exact,
                n_within_one=n_within_one,
                mean_left=statistics.fmean(xs) if xs else math.nan,
                mean_right=statistics.fmean(ys) if ys else math.nan,
                mean_difference=(
                    statistics.fmean([b - a for _, a, b in pairs]) if pairs else math.nan
                ),
                alpha=alpha,
                rho=rho,
                rho_p=rho_p,
                n_scenario_pairs=len(shared),
                wilcoxon_p=wilcoxon_p,
                rank_biserial=rb,
                rank_biserial_ci=rb_ci,
                cohens_dz=dz,
                hodges_lehmann=hl,
                distribution_left=raw_left,
                distribution_right=raw_right,
            )
        )

    return BackendEquivalence(
        condition=condition,
        left_backend=left_backend,
        right_backend=right_backend,
        n_cell_pairs=len(units),
        n_scenarios=len(scenarios),
        scenario_ids=tuple(scenarios),
        dimensions=tuple(dimensions),
        detectable_dz_cells=detectable_effect(len(units)) if len(units) > 1 else math.nan,
        detectable_dz_scenarios=detectable_effect(len(scenarios))
        if len(scenarios) > 1
        else math.nan,
        n_gate_blocked_excluded=n_gate_blocked_excluded,
    )


# ---------------------------------------------------------------------------
# The database path
# ---------------------------------------------------------------------------

_PAIRED_SQL = """
SELECT g.generation_id, g.scenario_id, g.condition, g.sample_idx, g.seed,
       g.prompt_id, g.served_by, g.gate_blocked,
       {dims}
FROM generation g
JOIN rubric_score rs ON rs.generation_id = g.generation_id
WHERE g.condition = %(condition)s
  AND g.served_by = %(served_by)s
  AND rs.rater_type = 'llm_judge'
  AND rs.rater_id = %(rater_id)s
ORDER BY g.scenario_id, g.sample_idx
"""


def fetch_paired_scores(
    *,
    condition: str = Condition.LC.value,
    left_backend: str = LC_EQUIVALENCE_BACKEND,
    right_backend: str = LC_ANALYSIS_BACKEND,
    rater_id: str = "holdout-judge",
    include_gate_blocked: bool = False,
) -> tuple[
    dict[str, dict[str, int | None]],
    dict[str, dict[str, int | None]],
    dict[str, str],
    int,
]:
    """Judge rows for the cells both stacks produced, keyed by cell.

    The unit key is `scenario/condition/sample_idx` — the cell, not the
    generation id, because the two stacks give the same cell different ids.
    """
    from carelite.db import connect
    from carelite.eval.judge.store import median_rater_id

    dims = ", ".join(f"rs.{d}" for d in RUBRIC_DIMENSIONS)
    sql = _PAIRED_SQL.format(dims=dims)

    def _side(backend: str) -> list[dict[str, Any]]:
        with connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    sql,
                    {
                        "condition": condition,
                        "served_by": backend,
                        "rater_id": median_rater_id(rater_id),
                    },
                ).fetchall()
            ]

    left_rows, right_rows = _side(left_backend), _side(right_backend)
    paired = pair_cells(left_rows, right_rows)

    left: dict[str, dict[str, int | None]] = {}
    right: dict[str, dict[str, int | None]] = {}
    scenario_of: dict[str, str] = {}
    n_blocked = 0
    for pair in paired.pairs:
        if pair.left["gate_blocked"] or pair.right["gate_blocked"]:
            n_blocked += 1
            if not include_gate_blocked:
                continue
        unit = f"{pair.key.scenario_id}/{pair.key.condition}/{pair.key.sample_idx}"
        left[unit] = {d: pair.left[d] for d in RUBRIC_DIMENSIONS}  # type: ignore[misc]
        right[unit] = {d: pair.right[d] for d in RUBRIC_DIMENSIONS}  # type: ignore[misc]
        scenario_of[unit] = pair.key.scenario_id
    return left, right, scenario_of, n_blocked


def run_backend_equivalence(
    *,
    condition: str = Condition.LC.value,
    left_backend: str = LC_EQUIVALENCE_BACKEND,
    right_backend: str = LC_ANALYSIS_BACKEND,
    rater_id: str = "holdout-judge",
    include_gate_blocked: bool = False,
    n_boot: int = 2000,
) -> BackendEquivalence:
    """Read the paired judge rows out of Postgres and compare the two stacks."""
    left, right, scenario_of, n_blocked = fetch_paired_scores(
        condition=condition,
        left_backend=left_backend,
        right_backend=right_backend,
        rater_id=rater_id,
        include_gate_blocked=include_gate_blocked,
    )
    if not left:
        raise UnpairableCells(
            f"no scored cell pairs for condition {condition} across "
            f"{left_backend!r} and {right_backend!r}; both sides must be judged first"
        )
    return compare_backends(
        left,
        right,
        scenario_of=scenario_of,
        left_backend=left_backend,
        right_backend=right_backend,
        condition=condition,
        n_gate_blocked_excluded=0 if include_gate_blocked else n_blocked,
        n_boot=n_boot,
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(
        prog="carelite.eval.judge.backend_equivalence",
        description="Paired backend agreement on the cells two serving stacks both produced.",
    )
    parser.add_argument("--condition", default=Condition.LC.value)
    parser.add_argument("--left", default=LC_EQUIVALENCE_BACKEND)
    parser.add_argument("--right", default=LC_ANALYSIS_BACKEND)
    parser.add_argument("--rater-id", default="holdout-judge")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    reports: dict[str, BackendEquivalence] = {}
    for label, include in (("excluding_gate_blocked", False), ("including_gate_blocked", True)):
        try:
            reports[label] = run_backend_equivalence(
                condition=args.condition,
                left_backend=args.left,
                right_backend=args.right,
                rater_id=args.rater_id,
                include_gate_blocked=include,
                n_boot=args.n_boot,
            )
        except UnpairableCells as exc:
            print(f"{label}: {exc}", file=sys.stderr)
            return 2

    # D12: neither including nor excluding refused cells is obviously right, so
    # both are reported and the reader chooses knowingly.
    for label, report in reports.items():
        print(f"[{label}]")
        print(report.render())

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        payload = {label: report.to_dict() for label, report in reports.items()}
        target = args.out / "backend_equivalence.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
