"""Which rows are an analysis arm, and the guard that stops two stacks becoming one.

**The hazard.** After `DECISIONS.md` D13 the label `LC` no longer identifies an
arm. `generation` holds 219 rows carrying `condition = 'LC'`: 180 served by vLLM,
which are the analysis arm, and D11's 39 served by Ollama, which are a partial
record over 13 of 60 scenarios kept only as a paired equivalence sample. A frame
filtered on the condition alone contains both. That is wrong twice over — it
pools two serving stacks D13 explicitly declined to pool, and it double-counts 13
scenarios — and nothing about the rows makes either visible once they are in a
`groupby`.

The analysis frame is therefore built by `restrict_to_analysis_arms`, which drops
the excluded `(condition, served_by)` selections by name, and then by
`assert_single_backend_per_condition`, which **re-checks what came back** rather
than trusting the filter. A `WHERE` is a claim about the rows; a check is the
evidence. `tests/unit/stats/test_arms.py` builds the pooled frame on purpose and
requires the raise.

**A frame with no `served_by` column is not automatically safe.** Fixtures and
frames read before the column existed carry no backend at all. For the five
conditions this study ran on one stack that is harmless — every such row is
Ollama, which is what the schema backfills to. For condition LC it is not: the
column is the only thing separating the arm from the equivalence sample, so a
frame that contains LC rows and no `served_by` cannot be resolved and the guard
refuses it rather than guessing.

**This module does not decide whether two stacks agree.** That is
`carelite.eval.judge.backend_equivalence`, which owns the measurement, and
`carelite.stats.sensitivity.backend_equivalence_check`, which frames its result
as a sensitivity analysis. What is here is only the selection rule and its guard:
`MixedBackendError` is imported from the judge lane rather than redefined, so one
exception type covers a pooled selection wherever it is caught.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from carelite.eval.judge.arms import (
    LC_ANALYSIS_BACKEND,
    LC_EQUIVALENCE_BACKEND,
    MixedBackendError,
)
from carelite.types import Condition

__all__ = [
    "AMBIGUOUS_WITHOUT_BACKEND",
    "EXCLUDED_ARMS",
    "LC_ANALYSIS_BACKEND",
    "LC_EQUIVALENCE_BACKEND",
    "ArmExclusion",
    "ArmSelection",
    "ExcludedArm",
    "MixedBackendError",
    "assert_single_backend_per_condition",
    "backends_by_condition",
    "restrict_to_analysis_arms",
]


@dataclass(frozen=True, slots=True)
class ExcludedArm:
    """One `(condition, served_by)` selection that is not part of the analysis."""

    condition: str
    served_by: str
    reason: str

    @property
    def label(self) -> str:
        return f"{self.condition}/{self.served_by}"


#: The selections excluded from the analysis frame, each named to its decision.
#: One entry, so a second hole cannot be added without a reader seeing why the
#: first one is there.
#:
#: D13: "The LC analysis arm is `served_by = 'vllm'` and nothing else." The 39
#: Ollama cells are retained in the database and used only as the paired
#: backend-equivalence sample — see
#: `carelite.stats.sensitivity.backend_equivalence_check`.
EXCLUDED_ARMS: tuple[ExcludedArm, ...] = (
    ExcludedArm(
        condition=str(Condition.LC),
        served_by=LC_EQUIVALENCE_BACKEND,
        reason=(
            "DECISIONS.md D13. These are D11's 39 partial LC cells over 13 of 60 scenarios, "
            "never randomised for partial analysis, and served by a different stack from the "
            "180-cell arm. They are retained as the paired backend-equivalence sample and are "
            "never pooled into an arm."
        ),
    ),
)

#: Conditions whose rows exist under more than one serving stack in this study,
#: so the label alone does not identify an arm and a frame without `served_by`
#: cannot be resolved. Derived from `EXCLUDED_ARMS` rather than restated.
AMBIGUOUS_WITHOUT_BACKEND: frozenset[str] = frozenset(a.condition for a in EXCLUDED_ARMS)


@dataclass(frozen=True, slots=True)
class ArmExclusion:
    """What one exclusion actually cost on this frame."""

    condition: str
    served_by: str
    reason: str
    n_generations: int
    n_scenarios: int

    @property
    def label(self) -> str:
        return f"{self.condition}/{self.served_by}"


@dataclass(frozen=True, slots=True)
class ArmSelection:
    """The record of how the analysis frame was narrowed to single-stack arms.

    Carried on the report rather than logged: "the LC arm is the 180 vLLM cells
    and the 39 Ollama cells were excluded" is a sentence the results have to
    contain, and a number nobody should re-derive to write it.
    """

    exclusions: tuple[ArmExclusion, ...] = ()
    backends: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    served_by_present: bool = True

    @property
    def excluded_counts(self) -> dict[str, int]:
        """`"LC/ollama" -> 39`, in the shape `DataInventory.dropped_conditions` takes."""
        return {e.label: e.n_generations for e in self.exclusions}

    @property
    def n_excluded(self) -> int:
        return sum(e.n_generations for e in self.exclusions)

    def render(self) -> str:
        lines = ["ANALYSIS ARMS — one serving stack per condition (D13)"]
        if not self.served_by_present:
            lines.append(
                "  `served_by` is not in this frame. Every row is treated as the schema's "
                "default stack, which is correct for a frame that predates the column."
            )
        for condition, stacks in self.backends.items():
            detail = ", ".join(f"{k} {v}" for k, v in sorted(stacks.items())) or "-"
            lines.append(f"  {condition}: {detail}")
        for exclusion in self.exclusions:
            lines.append(
                f"  EXCLUDED {exclusion.label}: {exclusion.n_generations} generations over "
                f"{exclusion.n_scenarios} scenarios. {exclusion.reason}"
            )
        return "\n".join(lines)


def backends_by_condition(long: pd.DataFrame) -> dict[str, dict[str, int]]:
    """`condition -> served_by -> distinct generations`, on the frame as given.

    An empty frame, or one with no `condition` column, is an empty mapping: the
    question "which stacks produced this arm" has no answer rather than a
    misleading zero.
    """
    if long.empty or "condition" not in long.columns:
        return {}
    frame = long
    if "served_by" not in frame.columns:
        frame = frame.assign(served_by=LC_EQUIVALENCE_BACKEND)
    counted = (
        frame.assign(
            _condition=frame["condition"].astype(str),
            _backend=frame["served_by"].fillna(LC_EQUIVALENCE_BACKEND).astype(str),
        )
        .groupby(["_condition", "_backend"], observed=True)["generation_id"]
        .nunique()
    )
    out: dict[str, dict[str, int]] = {}
    for (condition, backend), n in counted.items():
        out.setdefault(str(condition), {})[str(backend)] = int(n)
    return out


def assert_single_backend_per_condition(
    long: pd.DataFrame,
    *,
    what: str = "the analysis frame",
) -> dict[str, str]:
    """Every condition in `long` came from exactly one serving stack, or raise.

    Returns `condition -> served_by` for the frame it accepted, so a caller can
    record which stack each arm actually ran on instead of assuming it.

    Raises:
        MixedBackendError: when a condition carries rows from two stacks, or when
            the frame has no `served_by` column and contains a condition that
            exists under more than one stack in this study. The second case is
            not pedantry: `condition = 'LC'` matches both the 180-cell vLLM arm
            and D11's 39 Ollama cells, and without the column there is nothing in
            the frame that can tell them apart.
    """
    if long.empty or "condition" not in long.columns:
        return {}

    if "served_by" not in long.columns:
        present = {str(c) for c in long["condition"].dropna().unique()}
        ambiguous = sorted(present & AMBIGUOUS_WITHOUT_BACKEND)
        if ambiguous:
            raise MixedBackendError(
                f"{what} contains condition(s) {', '.join(ambiguous)} but no `served_by` "
                f"column. After D13 the condition label does not identify an arm — "
                f"`LC` matches the {LC_ANALYSIS_BACKEND} arm and D11's "
                f"{LC_EQUIVALENCE_BACKEND} equivalence sample alike — so this frame "
                f"cannot be resolved into arms. Read it with "
                f"`carelite.stats.data.load_scores`, which selects the column."
            )
        return {}

    stacks = backends_by_condition(long)
    pooled = {c: s for c, s in stacks.items() if len([n for n in s.values() if n]) > 1}
    if pooled:
        breakdown = "; ".join(
            f"{c} ({', '.join(f'{k}={v}' for k, v in sorted(s.items()))})"
            for c, s in sorted(pooled.items())
        )
        raise MixedBackendError(
            f"{what} pools serving stacks within {len(pooled)} condition(s): {breakdown}. "
            f"D13: the two stacks serve different artifacts of the same model family, at "
            f"different quantisation and sampling defaults, and realised different context "
            f"packs, so their rows are not one arm. Select one `served_by` value per "
            f"condition — `carelite.stats.arms.restrict_to_analysis_arms` is that selection "
            f"— or compare them as the paired equivalence sample."
        )
    return {c: next(iter(s)) for c, s in stacks.items() if s}


def restrict_to_analysis_arms(
    long: pd.DataFrame,
    *,
    excluded: Sequence[ExcludedArm] = EXCLUDED_ARMS,
    strict: bool = True,
) -> tuple[pd.DataFrame, ArmSelection]:
    """Drop the non-arm selections, then prove no arm pools two stacks.

    This replaces the whole-condition drop D11 required. D13 re-opened condition
    LC, so nothing is excluded by condition any more; what is excluded is one
    `(condition, served_by)` pair, and the difference matters — a rule that drops
    `LC` would now discard the 180-cell arm the study is here to analyse.

    Args:
        long: a long-format score frame (`carelite.stats.data.load_scores`).
        excluded: the `(condition, served_by)` selections that are not arms.
        strict: run `assert_single_backend_per_condition` on the result. Turn it
            off only to inspect a pooled frame deliberately; an analysis built on
            one is wrong in a way that does not show up in its output.

    Returns:
        The narrowed frame and the `ArmSelection` recording what it cost.
    """
    if long.empty or "condition" not in long.columns:
        return long, ArmSelection(served_by_present="served_by" in long.columns)

    has_backend = "served_by" in long.columns
    kept = long
    exclusions: list[ArmExclusion] = []

    if has_backend:
        conditions = kept["condition"].astype(str)
        backends = kept["served_by"].fillna(LC_EQUIVALENCE_BACKEND).astype(str)
        for arm in excluded:
            mask = (conditions == arm.condition) & (backends == arm.served_by)
            if not mask.any():
                continue
            removed = kept.loc[mask]
            exclusions.append(
                ArmExclusion(
                    condition=arm.condition,
                    served_by=arm.served_by,
                    reason=arm.reason,
                    n_generations=int(removed["generation_id"].nunique()),
                    n_scenarios=int(removed["scenario_id"].nunique()),
                )
            )
            kept = kept.loc[~mask]
            conditions = conditions.loc[kept.index]
            backends = backends.loc[kept.index]

    selection = ArmSelection(
        exclusions=tuple(exclusions),
        backends=backends_by_condition(kept),
        served_by_present=has_backend,
    )
    if strict:
        assert_single_backend_per_condition(kept, what="the analysis frame")
    return kept, selection
