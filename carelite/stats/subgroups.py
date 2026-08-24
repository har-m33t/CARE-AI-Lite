"""Subgroup analyses. One is pre-specified; every other one says so in its output.

Pre-registration §8.4 names exactly one: **the equity stratum**
(`equity_stratum = true`), run as a secondary analysis with the same test family
as §8.1, restricted to held-out scenarios in the stratum. "All other subgroup
analyses (by `challenge_type`, `literacy_signal`, `encounter_phase`, or any
other stratification not listed here) are exploratory."

`equity_subgroup` is the one; `exploratory_subgroup` is everything else, and it
cannot produce a result that is not stamped EXPLORATORY -- the flag is set
inside the function, not passed to it.

**Three things §8.4 requires be reported with the equity result, carried on the
result object so they cannot be dropped in transcription.**

1. `racial_ethnic` is described as what it measures, per `DECISIONS.md` D5:
   response to anticipated dismissal and patient credibility-management. Not the
   disparity label the axis is named for. `RACIAL_ETHNIC_DESCRIPTION` is emitted
   with any result broken down by `equity_kind`.
2. The stratum has no `emotion_intensity = 1` scenario.
3. `racial_ethnic` has no `adherence_barrier`, `decision_conflict` or
   `false_comprehension` scenario, and every one of its scenarios presents an
   already-guarded patient.

Both gaps are pre-specified limitations rather than defects to repair (D5: "A
limitation named in advance is a limitation; the same sentence written after
seeing the results is an excuse"), so they are constants here and are printed
whether or not anyone asks.

**The n this subgroup actually has.** §8.4 introduces the stratum as "35
scenarios" and then restricts the analysis to held-out scenarios in it. Those
are different numbers: 35 is the whole bank, and the held-out portion is 20
(10 `ses`, 4 `lep`, 6 `racial_ethnic`). Confirmatory analyses run on the holdout
only (§6), so the equity subgroup is an n = 20 paired analysis, which at 80%
power resolves only a large effect (dz ~ 0.68 by
`carelite.stats.power.detectable_effect`). `EquitySubgroupResult` computes and
prints its own detectable effect from the n it actually had, so the figure in
the write-up is the one the analysis ran at rather than one carried over from
the main comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from carelite.eval.judge.validation import EvidenceStatus
from carelite.stats.effects import DEFAULT_N_BOOT, DEFAULT_SEED
from carelite.stats.power import detectable_effect
from carelite.stats.primary import CONFIRMATORY_FAMILY, FamilyResult, Hypothesis, run_family

__all__ = [
    "EQUITY_COVERAGE_GAPS",
    "RACIAL_ETHNIC_DESCRIPTION",
    "EquitySubgroupResult",
    "ExploratorySubgroupResult",
    "equity_subgroup",
    "exploratory_subgroup",
]

#: `DECISIONS.md` D5. The sentence the write-up uses for this axis.
RACIAL_ETHNIC_DESCRIPTION = (
    "The `racial_ethnic` axis measures response to anticipated dismissal and patient "
    "credibility-management, not race-based disparity in communication generally: eight of the "
    "nine scenarios turn on a patient who has already been disbelieved, or expects to be, and "
    "manages the clinician accordingly (DECISIONS.md D5). The label is kept for continuity with "
    "the frozen split; the description is not."
)

#: Pre-registration §8.4, declared before any evaluation data existed.
EQUITY_COVERAGE_GAPS: tuple[str, ...] = (
    "The equity stratum contains no `emotion_intensity = 1` scenario, so it cannot say whether "
    "the disparity behaves differently on an emotionally flat turn — the turn where a system "
    "that over-reads emotion does its worst work. Flat turns are still tested outside the "
    "subgroup.",
    "`racial_ethnic` contains no `adherence_barrier`, `decision_conflict` or "
    "`false_comprehension` scenario, and every one of its scenarios presents an already-guarded "
    "patient. A system that scores well on this axis may be scoring on `handles a guarded "
    "patient` rather than on the disparity the axis claims to measure.",
)


@dataclass(frozen=True, slots=True)
class EquitySubgroupResult:
    """The one pre-specified subgroup (§8.4), with its own n and its own limits."""

    family: FamilyResult
    n_scenarios: int
    n_by_equity_kind: Mapping[str, int]
    detectable_effect_dz: float
    prespecified: bool = True
    racial_ethnic_description: str = RACIAL_ETHNIC_DESCRIPTION
    coverage_gaps: tuple[str, ...] = EQUITY_COVERAGE_GAPS

    def render(self) -> str:
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(self.n_by_equity_kind.items())) or "-"
        lines = [
            "PRE-SPECIFIED SUBGROUP: THE EQUITY STRATUM (pre-registration §8.4)",
            f"  {self.n_scenarios} held-out scenarios in the stratum ({kinds}).",
            f"  At n = {self.n_scenarios}, 80% power, alpha = 0.05 two-sided, the smallest "
            f"detectable paired effect is dz = {self.detectable_effect_dz:.3f}. A null result "
            "below that is this subgroup's resolution, not an absence of effect.",
            "",
            f"  {self.racial_ethnic_description}",
            "",
            "  PRE-SPECIFIED COVERAGE GAPS (declared before data existed, §8.4):",
        ]
        for gap in self.coverage_gaps:
            lines.append(f"    - {gap}")
        lines.append("")
        lines.append(self.family.render())
        return "\n".join(lines)


def equity_subgroup(
    long: pd.DataFrame,
    *,
    hypotheses: Sequence[Hypothesis] = CONFIRMATORY_FAMILY,
    rater_type: str | None = None,
    statuses: Mapping[str, EvidenceStatus] | None = None,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> EquitySubgroupResult:
    """Run the §8.1 family restricted to `equity_stratum = true` scenarios.

    Pre-specified, so the results are labelled confirmatory-eligible; the judge
    gate still applies per dimension, and the subgroup's own n is carried on the
    result rather than assumed to be 60.
    """
    if "equity_stratum" not in long.columns:
        raise KeyError(
            "long frame has no `equity_stratum` column; load it with "
            "carelite.stats.data.load_scores, which joins the scenario table"
        )
    stratum = long[long["equity_stratum"].astype("boolean").fillna(False)]
    n_scenarios = int(stratum["scenario_id"].nunique()) if not stratum.empty else 0

    by_kind: dict[str, int] = {}
    if "equity_kind" in stratum.columns and not stratum.empty:
        counts = (
            stratum.dropna(subset=["equity_kind"])
            .groupby("equity_kind", observed=True)["scenario_id"]
            .nunique()
        )
        by_kind = {str(k): int(v) for k, v in counts.items()}

    family = run_family(
        stratum,
        hypotheses,
        name="pre-specified secondary analysis: equity stratum (pre-registration §8.4)",
        alpha=alpha,
        rater_type=rater_type,
        statuses=statuses,
        n_boot=n_boot,
        seed=seed,
        notes=(
            f"restricted to the {n_scenarios} held-out scenarios with equity_stratum = true; "
            "the 35-scenario figure in §8.4 counts the whole bank, train split included",
        ),
    )
    return EquitySubgroupResult(
        family=family,
        n_scenarios=n_scenarios,
        n_by_equity_kind=by_kind,
        detectable_effect_dz=detectable_effect(n_scenarios) if n_scenarios > 0 else float("nan"),
    )


@dataclass(frozen=True, slots=True)
class ExploratorySubgroupResult:
    """A subgroup the pre-registration does not name. Labelled, in the object."""

    column: str
    value: str
    family: FamilyResult
    n_scenarios: int
    detectable_effect_dz: float
    prespecified: bool = False
    status: EvidenceStatus = EvidenceStatus.EXPLORATORY

    def render(self) -> str:
        return "\n".join(
            [
                f"EXPLORATORY SUBGROUP: {self.column} = {self.value}",
                "  NOT pre-specified (pre-registration §8.4 names the equity stratum and no "
                "other). Every result below is exploratory whatever its p-value.",
                f"  {self.n_scenarios} held-out scenarios; smallest detectable paired effect "
                f"dz = {self.detectable_effect_dz:.3f}.",
                "",
                self.family.render(),
            ]
        )


def exploratory_subgroup(
    long: pd.DataFrame,
    column: str,
    value: object,
    *,
    hypotheses: Sequence[Hypothesis] = CONFIRMATORY_FAMILY,
    rater_type: str | None = None,
    statuses: Mapping[str, EvidenceStatus] | None = None,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    predicate: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> ExploratorySubgroupResult:
    """Any other stratification. Exploratory by construction, not by choice.

    Every hypothesis is rewritten with `prespecified=False` before it is run, so
    the label machinery demotes each result regardless of what the judge
    validation says and regardless of what the caller passed.
    """
    if predicate is not None:
        mask = predicate(long)
    else:
        if column not in long.columns:
            raise KeyError(f"no column {column!r} on the score frame")
        mask = long[column].astype(str) == str(value)
    subset = long[mask]
    n_scenarios = int(subset["scenario_id"].nunique()) if not subset.empty else 0

    demoted = tuple(
        Hypothesis(
            key=f"exploratory_{column}_{value}_{h.key}",
            measure_key=h.measure_key,
            left=h.left,
            right=h.right,
            expected_higher=h.expected_higher,
            description=f"EXPLORATORY subgroup {column} = {value}. {h.description}",
            role="exploratory",
            prespecified=False,
        )
        for h in hypotheses
    )
    family = run_family(
        subset,
        demoted,
        name=f"EXPLORATORY subgroup: {column} = {value}",
        alpha=alpha,
        rater_type=rater_type,
        statuses=statuses,
        n_boot=n_boot,
        seed=seed,
        notes=(
            "not a pre-specified subgroup; Holm correction here is within this exploratory "
            "family only and does not account for the other subgroups anyone might run",
        ),
    )
    return ExploratorySubgroupResult(
        column=column,
        value=str(value),
        family=family,
        n_scenarios=n_scenarios,
        detectable_effect_dz=detectable_effect(n_scenarios) if n_scenarios > 0 else float("nan"),
    )
