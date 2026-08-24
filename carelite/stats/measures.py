"""Outcome measures, and the single path from raw rubric scores onto them.

Two things live here and nowhere else in `carelite.stats`.

**1. The reverse-coding boundary.** `ritualistic` is scored so that a raw 5 is
the *worst* response. `carelite.eval.rubric.dimensions.to_quality` is the only
sanctioned way to put it on the same polarity as the other ten, and every
aggregation in this package — composite means, cell means, bootstrap resamples,
the mixed-effects design matrix — reads the ``quality`` column that
`attach_quality` produces and never the ``raw`` column beside it. That is
deliberate and structural: a function that averages ``raw`` across dimensions
inverts one of eleven and produces numbers that look entirely plausible. The
aggregation functions below all call `attach_quality` themselves rather than
trusting a caller to have done it, so there is no ordering in which the
transform can be skipped.

**2. What "the outcome" means.** The pre-registration (§3, §4) states outcomes
in three shapes: the composite over the five NURSE dimensions, the composite
over the four Four Habits dimensions, and two single dimensions
(`naturalness`, `ritualistic`). `Measure` is all three shapes in one type, so a
comparison on a composite and a comparison on a single dimension travel through
the same code and land in the same correction family.

**The aggregation §3 fixes**, and which `cell_means` implements verbatim: the
per-generation mean of `to_quality()` over the measure's constituent
dimensions, then one value per scenario x condition by averaging across the
three samples in that cell, per rater type.

A missing dimension is dropped from its generation's composite rather than
dropping the generation, following pre-registration §10: an ungrounded judge
score "is treated as missing for that dimension only; it is not imputed and it
is not treated as a 1". `n_dimensions` on the returned frame records how many
constituents actually contributed, so a composite computed from three of five
NURSE dimensions is visible rather than silently equal to one computed from
five.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from carelite.eval.rubric.dimensions import DIMENSIONS, to_quality
from carelite.types import RUBRIC_DIMENSIONS, Condition

__all__ = [
    "FOUR_HABITS_COMPOSITE",
    "FOUR_HABITS_DIMENSIONS",
    "LONG_COLUMNS",
    "MEASURES",
    "NURSE_COMPOSITE",
    "NURSE_DIMENSIONS",
    "Measure",
    "attach_quality",
    "cell_means",
    "measure",
    "measure_by_generation",
    "paired_matrix",
    "quality_lookup",
]

#: The five NURSE dimensions, in `RUBRIC_DIMENSIONS` order (pre-registration §3).
NURSE_DIMENSIONS: tuple[str, ...] = ("name", "understand", "respect", "support", "explore")

#: The four Four Habits dimensions (pre-registration §4.1).
FOUR_HABITS_DIMENSIONS: tuple[str, ...] = ("ib", "epp", "de", "ie")


@dataclass(frozen=True, slots=True)
class Measure:
    """One outcome measure: a composite, or a single rubric dimension.

    `dimensions` is what gets averaged after `to_quality`. For a single-dimension
    measure it is a one-tuple, which is what lets a `naturalness` comparison and a
    NURSE-composite comparison share every downstream function.
    """

    key: str
    label: str
    dimensions: tuple[str, ...]

    def __post_init__(self) -> None:
        unknown = [d for d in self.dimensions if d not in DIMENSIONS]
        if unknown:
            raise ValueError(f"{self.key}: not rubric dimensions: {unknown}")
        if not self.dimensions:
            raise ValueError(f"{self.key}: a measure needs at least one dimension")

    @property
    def is_composite(self) -> bool:
        return len(self.dimensions) > 1


NURSE_COMPOSITE = Measure(
    key="nurse_composite",
    label="Composite NURSE adherence",
    dimensions=NURSE_DIMENSIONS,
)

FOUR_HABITS_COMPOSITE = Measure(
    key="four_habits_composite",
    label="Composite Four Habits adherence",
    dimensions=FOUR_HABITS_DIMENSIONS,
)

#: Every measure the analysis plan can name: the two composites plus each of the
#: eleven dimensions on its own.
MEASURES: Mapping[str, Measure] = {
    NURSE_COMPOSITE.key: NURSE_COMPOSITE,
    FOUR_HABITS_COMPOSITE.key: FOUR_HABITS_COMPOSITE,
    **{
        key: Measure(key=key, label=DIMENSIONS[key].label, dimensions=(key,))
        for key in RUBRIC_DIMENSIONS
    },
}


def measure(key: str) -> Measure:
    """Look up a measure by key, with a useful error for a typo."""
    try:
        return MEASURES[key]
    except KeyError:
        raise KeyError(f"{key!r} is not a measure; expected one of {sorted(MEASURES)}") from None


#: Columns a long-format score frame must carry. `raw` is the database value;
#: `quality` is what every aggregation reads.
LONG_COLUMNS: tuple[str, ...] = (
    "generation_id",
    "scenario_id",
    "condition",
    "sample_idx",
    "rater_type",
    "rater_id",
    "dimension",
    "raw",
)


@lru_cache(maxsize=1)
def quality_lookup() -> Mapping[tuple[str, int], int]:
    """`(dimension, raw) -> quality`, built by calling `to_quality` on every cell.

    A 55-entry table rather than a vectorised `where(dimension in REVERSE_CODED,
    6 - raw, raw)`. The point is that the reversal is never re-expressed here: the
    numbers come out of the rubric lane's function, so if the polarity of a
    dimension ever changes there, this table changes with it and no constant in
    this package needs finding.
    """
    return {(key, raw): to_quality(key, raw) for key in RUBRIC_DIMENSIONS for raw in range(1, 6)}


def attach_quality(long: pd.DataFrame) -> pd.DataFrame:
    """Return `long` with a ``quality`` column derived from ``raw``.

    Idempotent and always recomputed: calling it twice is the same as calling it
    once, and an existing ``quality`` column is overwritten rather than trusted.
    Missing scores stay missing. Out-of-range scores raise, via `to_quality`.
    """
    missing = [c for c in ("dimension", "raw") if c not in long.columns]
    if missing:
        raise KeyError(f"long score frame is missing {missing}; expected {LONG_COLUMNS}")

    out = long.copy()
    table = quality_lookup()
    raw = pd.to_numeric(out["raw"], errors="coerce")

    def _one(dim: object, value: float) -> float:
        if pd.isna(value):
            return float("nan")
        as_int = int(value)
        if as_int != value:
            raise ValueError(f"{dim} score {value!r} is not an integer rubric score")
        try:
            return float(table[(str(dim), as_int)])
        except KeyError:
            # Delegated so the error text is the rubric lane's, not a copy of it.
            return float(to_quality(str(dim), as_int))

    out["quality"] = [_one(dim, value) for dim, value in zip(out["dimension"], raw, strict=True)]
    return out


def measure_by_generation(long: pd.DataFrame, m: Measure) -> pd.DataFrame:
    """One row per (generation, rater), carrying the measure on the quality scale.

    Columns: the identifying columns from `long`, plus ``value`` and
    ``n_dimensions``. Generations with none of the measure's dimensions scored
    are dropped rather than returned as NaN.
    """
    scored = attach_quality(long)
    subset = scored[scored["dimension"].isin(m.dimensions)]
    subset = subset[subset["quality"].notna()]
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "generation_id",
                "scenario_id",
                "condition",
                "sample_idx",
                "rater_type",
                "rater_id",
                "value",
                "n_dimensions",
            ]
        )

    keys = ["generation_id", "scenario_id", "condition", "sample_idx", "rater_type", "rater_id"]
    keys = [k for k in keys if k in subset.columns]
    grouped = subset.groupby(keys, dropna=False, observed=True)["quality"]
    out = grouped.agg(value="mean", n_dimensions="count").reset_index()
    return out


def cell_means(long: pd.DataFrame, m: Measure) -> pd.DataFrame:
    """Scenario x condition cell means, per rater type (pre-registration §3).

    The three samples in a cell are averaged here — which is the aggregation the
    primary analysis is defined on, and is *not* a claim that the three samples
    are independent. That claim is never made: `carelite.stats.mixed` goes back
    to the per-generation rows and absorbs the within-cell variance with a random
    intercept for scenario instead.

    Returns columns ``scenario_id, condition, rater_type, value, n_samples``.
    """
    per_generation = measure_by_generation(long, m)
    if per_generation.empty:
        return pd.DataFrame(
            columns=["scenario_id", "condition", "rater_type", "value", "n_samples"]
        )
    grouped = per_generation.groupby(
        ["scenario_id", "condition", "rater_type"], dropna=False, observed=True
    )["value"]
    return grouped.agg(value="mean", n_samples="count").reset_index()


def paired_matrix(
    cells: pd.DataFrame,
    conditions: Sequence[Condition | str],
    *,
    rater_type: str | None = None,
) -> pd.DataFrame:
    """Complete-case scenario x condition matrix, ready for a paired test.

    Rows are scenarios, columns are the requested conditions in the order given.
    A scenario missing any requested condition is dropped, because a paired test
    has no meaning for it — the count of dropped scenarios is what the caller
    reports, so drop it here rather than letting a NaN propagate into a rank.
    """
    frame = cells
    if rater_type is not None:
        frame = frame[frame["rater_type"] == rater_type]
    wanted = [str(c) for c in conditions]
    frame = frame[frame["condition"].astype(str).isin(wanted)]
    if frame.empty:
        return pd.DataFrame(columns=wanted, dtype=float)

    wide = frame.pivot_table(
        index="scenario_id", columns="condition", values="value", aggfunc="mean"
    )
    wide.columns = [str(c) for c in wide.columns]
    for col in wanted:
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide[wanted].dropna(axis=0, how="any")
    return wide.sort_index()
