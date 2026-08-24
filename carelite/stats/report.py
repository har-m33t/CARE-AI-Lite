"""The whole pre-specified analysis, assembled and rendered in the registered order.

One entry point, `run_analysis`, so that "what the study found" is the output of
a single function against a single database read, and no part of the plan can be
run while another is quietly skipped.

**The order is the pre-registration's order, and it is enforced here rather than
recommended.** §8.2 fixes that effect sizes and their intervals are reported
before the corresponding p-values "in every table and figure". Every renderer in
this package puts the effect columns first; `AnalysisReport.render()` puts the
judge-validation gate before the results it governs, and the sensitivity flips
before the tables they came from. A writer following this output down the page
reports the study correctly without having to remember any of it.

**What the report refuses to do.** It does not decide whether a dimension is
confirmatory -- that comes from `carelite.eval.judge.validation.classify_dimension`
via `carelite.stats.evidence`, and when no validation study has run every
dimension is exploratory and every result says so. It does not pick a subgroup;
§8.4's equity stratum is run and anything else the caller asks for is stamped
exploratory. It does not soften a negative-control failure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from carelite.eval.judge.validation import DimensionValidity, EvidenceStatus
from carelite.stats.data import attach_equity_kind, load_judge_samples, load_scores
from carelite.stats.effects import DEFAULT_N_BOOT, DEFAULT_SEED
from carelite.stats.evidence import dimension_statuses
from carelite.stats.measures import FOUR_HABITS_COMPOSITE, NURSE_COMPOSITE
from carelite.stats.mixed import MixedModelResult, fit_random_intercept
from carelite.stats.negative_control import NegativeControlResult, negative_control
from carelite.stats.power import PowerReport, build_power_report
from carelite.stats.primary import CONFIRMATORY_FAMILY, FamilyResult, Hypothesis, run_family
from carelite.stats.sensitivity import SensitivityReport, run_all_sensitivity
from carelite.stats.subgroups import EquitySubgroupResult, equity_subgroup
from carelite.types import RUBRIC_DIMENSIONS, Split

__all__ = ["AnalysisReport", "run_analysis"]


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    """Every pre-specified analysis, in the order the pre-registration states them."""

    split: str
    n_scenarios: int
    n_generations: int
    rater_types: tuple[str, ...]
    dimension_statuses: Mapping[str, EvidenceStatus]
    power: PowerReport
    primary: FamilyResult
    mixed: tuple[MixedModelResult, ...]
    equity: EquitySubgroupResult | None
    sensitivity: SensitivityReport
    negative_control_result: NegativeControlResult | None
    empty: bool = False

    @property
    def exploratory_dimensions(self) -> tuple[str, ...]:
        return tuple(
            k
            for k in RUBRIC_DIMENSIONS
            if self.dimension_statuses.get(k, EvidenceStatus.EXPLORATORY)
            is not EvidenceStatus.CONFIRMATORY
        )

    def render(self) -> str:
        sections: list[str] = [
            "=" * 78,
            "CARELITE AI — PRE-SPECIFIED STATISTICAL ANALYSIS",
            "docs/preregistration.md §8; build plan v3 §14",
            "=" * 78,
            "",
            f"  split: {self.split}   scenarios: {self.n_scenarios}   "
            f"generations: {self.n_generations}   raters: "
            f"{', '.join(self.rater_types) or '(none)'}",
        ]
        if self.empty:
            sections.append(
                "\n  NO RESULTS DATA. `generation` and `rubric_score` are empty for this split.\n"
                "  Held-out generation is gated behind OSF registration (DECISIONS.md, 'Gates "
                "that remain\n  with a person'), so this is the expected state before "
                "registration, not a fault. Every\n  analysis below is structurally in place and "
                "exercised against fixtures; none has data."
            )

        sections.append("")
        sections.append("-" * 78)
        sections.append("JUDGE-VALIDATION GATE (pre-registration §9) — read before any result")
        sections.append("-" * 78)
        exploratory = self.exploratory_dimensions
        if len(exploratory) == len(RUBRIC_DIMENSIONS):
            sections.append(
                "  EVERY dimension is exploratory. Either the judge validation study has not "
                "run,\n  or no dimension cleared alpha >= 0.667 and rho >= 0.5 on >= 30 paired "
                "units.\n  No judge-only result below may be reported as confirmatory."
            )
        else:
            confirmatory = [k for k in RUBRIC_DIMENSIONS if k not in exploratory]
            sections.append(f"  confirmatory: {', '.join(confirmatory)}")
            sections.append(
                f"  EXPLORATORY (say so in the sentence that reports them): "
                f"{', '.join(exploratory) or '(none)'}"
            )

        for body in (self.power.render(), self.primary.render()):
            sections.extend(["", "-" * 78, body])

        if self.mixed:
            sections.extend(["", "-" * 78, "VARIANCE DECOMPOSITION (§8.3)", ""])
            for model in self.mixed:
                sections.append(model.render())
                sections.append("")

        if self.equity is not None:
            sections.extend(["", "-" * 78, self.equity.render()])

        sections.extend(["", "-" * 78, self.sensitivity.render()])

        sections.extend(["", "-" * 78])
        if self.negative_control_result is None:
            sections.append(
                "NEGATIVE CONTROL (§8.6): not computable — no paired B/D data on this split."
            )
        else:
            sections.append(self.negative_control_result.render())
        return "\n".join(sections)


def run_analysis(
    *,
    split: Split | str = Split.HOLDOUT,
    long: pd.DataFrame | None = None,
    judge_samples: pd.DataFrame | None = None,
    conn: Any | None = None,
    validity: Mapping[str, DimensionValidity] | None = None,
    hypotheses: Sequence[Hypothesis] = CONFIRMATORY_FAMILY,
    rater_type: str | None = None,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> AnalysisReport:
    """Run every pre-specified analysis against one split.

    `long` and `judge_samples` default to a database read; passing them makes the
    whole report a pure function of a frame, which is how the tests exercise it
    without Postgres.

    `validity` is the judge-validation study's per-dimension output. `None` means
    it has not run, which demotes every judge-gated result to exploratory --
    the correct state of the world before human rating (docs/limitations.md §4),
    not a placeholder.
    """
    if long is None:
        long = load_scores(split=split, conn=conn)
    if "equity_kind" not in long.columns and not long.empty:
        try:
            long = attach_equity_kind(long)
        except Exception:  # a missing bank file must not take the whole report down
            long = long.copy()
            long["equity_kind"] = None
    if judge_samples is None and conn is not None:
        judge_samples = load_judge_samples(split=split, conn=conn)

    # `None` when the validation study has not run, so the label machinery says
    # "has not run" rather than "failed the threshold" — the two are different
    # claims about the judge and only one of them is true right now.
    statuses = dimension_statuses(validity) if validity is not None else None
    empty = long.empty

    primary = run_family(
        long,
        hypotheses,
        alpha=alpha,
        rater_type=rater_type,
        statuses=statuses,
        n_boot=n_boot,
        seed=seed,
    )

    mixed: list[MixedModelResult] = []
    for m in (NURSE_COMPOSITE, FOUR_HABITS_COMPOSITE):
        fitted = fit_random_intercept(long, m, rater_type=rater_type, statuses=statuses)
        if fitted is not None:
            mixed.append(fitted)

    equity: EquitySubgroupResult | None = None
    if "equity_stratum" in long.columns:
        equity = equity_subgroup(
            long,
            hypotheses=hypotheses,
            rater_type=rater_type,
            statuses=statuses,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
        )

    sensitivity = run_all_sensitivity(
        long,
        primary,
        judge_samples=judge_samples,
        hypotheses=hypotheses,
        rater_type=rater_type,
        statuses=statuses,
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
    )

    return AnalysisReport(
        split=str(split),
        n_scenarios=int(long["scenario_id"].nunique()) if not empty else 0,
        n_generations=int(long["generation_id"].nunique()) if not empty else 0,
        rater_types=tuple(sorted({str(r) for r in long["rater_type"].dropna()}))
        if not empty
        else (),
        dimension_statuses=dimension_statuses(validity),
        power=build_power_report(family_size=len(hypotheses)),
        primary=primary,
        mixed=tuple(mixed),
        equity=equity,
        sensitivity=sensitivity,
        negative_control_result=negative_control(long, family=primary, alpha=alpha),
        empty=empty,
    )
