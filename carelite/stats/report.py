"""The whole analysis, assembled and rendered in an order that cannot be misread.

One entry point, `run_analysis`, so that "what the study found" is the output of
a single function against a single database read, and no part of the plan can be
run while another is quietly skipped.

**D10: every result is descriptive.** The banner says so first, before any
number. Nothing this module emits is confirmatory or pre-registered.

**The order is load-bearing, not stylistic.** Four things must be read before the
comparisons they govern, so all four are rendered above them:

1. the **data inventory** — what was read and what each exclusion costs, so a
   reader who wants different exclusions can price them;
2. the **instrument resolution table** — which dimensions the judge actually
   resolved. A reader who reaches a `naturalness` p-value without this has
   already misread it, and ordering is the only mechanism that reliably
   prevents that;
3. the **judge-validation gate** — §9's per-dimension agreement status;
4. the **power analysis** — what effect this n could resolve at all.

Then the comparisons, then retrieval asked both ways, then variance, then the
equity stratum, then sensitivity with its flips at the top, then the negative
control last because it is the verdict on everything above it.

Within every comparison, §8.2's rule holds: effect sizes and their intervals
before p-values, enforced by `PairwiseResult` having no constructor path that
produces a p-value without them.

**What the report refuses to do.** It does not decide whether a dimension
cleared the judge gate -- that comes from
`carelite.eval.judge.validation.classify_dimension` via
`carelite.stats.evidence`, and when no validation study has run every dimension
is exploratory and every result says so. It does not pick a subgroup; §8.4's
equity stratum is run and anything else the caller asks for is stamped
exploratory. It does not soften a negative-control failure, and it does not let
an untestable comparison render as a non-significant one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from carelite.eval.judge.validation import DimensionValidity, EvidenceStatus
from carelite.stats.data import (
    DataInventory,
    attach_equity_kind,
    drop_dropped_conditions,
    inventory,
    load_judge_samples,
    load_scores,
)
from carelite.stats.effects import DEFAULT_N_BOOT, DEFAULT_SEED
from carelite.stats.evidence import D10_BANNER, dimension_statuses
from carelite.stats.instrument import InstrumentReport, instrument_report
from carelite.stats.measures import FOUR_HABITS_COMPOSITE, NURSE_COMPOSITE
from carelite.stats.mixed import MixedModelResult, fit_random_intercept
from carelite.stats.negative_control import NegativeControlResult, negative_control
from carelite.stats.power import PowerReport, build_power_report
from carelite.stats.primary import CONFIRMATORY_FAMILY, FamilyResult, Hypothesis, run_family
from carelite.stats.sensitivity import (
    RetrievalContrast,
    SensitivityReport,
    retrieval_contrast,
    run_all_sensitivity,
)
from carelite.stats.subgroups import EquitySubgroupResult, equity_subgroup
from carelite.types import RUBRIC_DIMENSIONS, Split

__all__ = ["AnalysisReport", "run_analysis"]


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    """Every analysis, in the order that makes it hard to misread. See the module docstring."""

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
    inventory: DataInventory | None = None
    instrument: InstrumentReport | None = None
    retrieval: RetrievalContrast | None = None
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
            "CARELITE AI — STATISTICAL ANALYSIS OF THE HOLDOUT RUN",
            "docs/preregistration.md §8 (the plan; NOT registered — see D10); build plan v3 §14",
            "=" * 78,
            "",
            f"  {D10_BANNER}",
            "",
            f"  split: {self.split}   scenarios: {self.n_scenarios}   "
            f"generations: {self.n_generations}   raters: "
            f"{', '.join(self.rater_types) or '(none)'}",
        ]
        if self.empty:
            sections.append(
                "\n  NO RESULTS DATA for this split. Every analysis below is structurally in "
                "place and\n  exercised against fixtures; none has data. If the holdout run has "
                "completed, check that\n  `carelite.eval.judge.load` has been run to bring "
                "rubric_scores.jsonl into `rubric_score`."
            )

        if self.inventory is not None:
            sections.extend(["", "-" * 78, self.inventory.render()])

        if self.instrument is not None:
            sections.extend(["", "-" * 78, self.instrument.render()])

        sections.append("")
        sections.append("-" * 78)
        sections.append("JUDGE-VALIDATION GATE (analysis plan §9) — read before any result")
        sections.append("-" * 78)
        exploratory = self.exploratory_dimensions
        if len(exploratory) == len(RUBRIC_DIMENSIONS):
            sections.append(
                "  EVERY dimension is exploratory. Either the judge validation study has not "
                "run,\n  or no dimension cleared alpha >= 0.667 and rho >= 0.5 on >= 30 paired "
                "units.\n  No judge-only result below carries the DESCRIPTIVE label."
            )
        else:
            cleared = [k for k in RUBRIC_DIMENSIONS if k not in exploratory]
            sections.append(f"  cleared the agreement threshold: {', '.join(cleared)}")
            sections.append(
                f"  EXPLORATORY (say so in the sentence that reports them): "
                f"{', '.join(exploratory) or '(none)'}"
            )

        for body in (self.power.render(), self.primary.render()):
            sections.extend(["", "-" * 78, body])

        if self.retrieval is not None:
            sections.extend(["", "-" * 78, self.retrieval.render()])

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
    drop_conditions: bool = True,
) -> AnalysisReport:
    """Run every analysis against one split.

    `long` and `judge_samples` default to a database read; passing them makes the
    whole report a pure function of a frame, which is how the tests exercise it
    without Postgres.

    `validity` is the judge-validation study's per-dimension output. `None` means
    it has not run, which demotes every judge-gated result to exploratory --
    the correct state of the world without human rating (docs/limitations.md §4),
    not a placeholder.

    `drop_conditions` applies D11: condition LC is removed before anything is
    computed. It is a parameter rather than a hard-coded filter so a caller can
    inspect the dropped rows deliberately, and it defaults to the decision.

    **The instrument diagnostic runs before every comparison and is threaded into
    all of them.** That ordering is the point: a degenerate dimension has to be
    known before its p-value is produced, or the p-value gets rendered as a null
    result on the way past.
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

    dropped: dict[str, int] = {}
    if drop_conditions:
        long, dropped = drop_dropped_conditions(long)

    # `None` when the validation study has not run, so the label machinery says
    # "has not run" rather than "failed the threshold" — the two are different
    # claims about the judge and only one of them is true right now.
    statuses = dimension_statuses(validity) if validity is not None else None
    empty = long.empty

    counts = inventory(long, dropped=dropped) if not empty else None
    instrument = instrument_report(long) if not empty else None
    discrimination = instrument.statuses if instrument is not None else None

    primary = run_family(
        long,
        hypotheses,
        alpha=alpha,
        rater_type=rater_type,
        statuses=statuses,
        n_boot=n_boot,
        seed=seed,
        discrimination=discrimination,
    )

    retrieval = (
        retrieval_contrast(
            long,
            rater_type=rater_type,
            statuses=statuses,
            n_boot=n_boot,
            seed=seed,
            discrimination=discrimination,
        )
        if not empty and "condition" in long.columns
        else None
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
            discrimination=discrimination,
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
        discrimination=discrimination,
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
        inventory=counts,
        instrument=instrument,
        retrieval=retrieval,
        empty=empty,
    )
