"""Krippendorff's alpha (ordinal) and Spearman's rho. Shared by judge and human lanes.

Alpha rather than Cohen's kappa because the design demands it: raters are not
guaranteed to be the same two people on every item, some cells are missing
(a rejected judge score, a rater who skipped an item), and the 1-5 rubric is
ordinal rather than nominal. Alpha handles all three; kappa handles none of
them without a fudge.

The **ordinal** difference function is the one that matters. Under a nominal
metric a 1-vs-5 disagreement and a 4-vs-5 disagreement count the same, which on
an anchored Likert rubric is nonsense — it would make the judge look far worse
than it is on dimensions where it is merely imprecise, and equally bad on
dimensions where it is genuinely wrong. The ordinal metric weights a
disagreement by how much of the observed distribution lies between the two
values, so near-misses cost little and opposite ends cost a lot.

**Direction.** Ordinal alpha is invariant under reversing the value order (the
metric sums marginals over the interval between two values, and an interval is
the same set of categories read either way), and Spearman's rho between two
series is unchanged when *both* are reversed. So neither metric can be broken by
`ritualistic` alone — but a series where one side is raw and the other is
quality-coded is silently wrong, and would show up as a strong *negative* rho
that someone might report as "the judge disagrees on ritual". Callers therefore
canonicalise both sides with `to_quality` before calling in here; this module
takes numbers and does not know which dimension they came from.

Missing data is `None`, everywhere. Units with fewer than two observations
contribute nothing to alpha, which is the definition, not a shortcut.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "AgreementResult",
    "Metric",
    "krippendorff_alpha",
    "paired_series",
    "spearman_rho",
]


class Metric(StrEnum):
    """Difference function. `ORDINAL` is the one this study pre-specifies."""

    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    INTERVAL = "interval"


@dataclass(frozen=True, slots=True)
class AgreementResult:
    """Agreement on one dimension, with the sample size that produced it.

    `n_units` is reported alongside every coefficient because an alpha computed
    on nine units is not the same evidence as an alpha computed on sixty, and a
    coefficient printed without its n invites exactly that conflation.
    """

    dimension: str
    n_units: int
    n_observers: int
    alpha: float
    rho: float
    rho_p: float

    @property
    def alpha_defined(self) -> bool:
        return not math.isnan(self.alpha)


# ---------------------------------------------------------------------------
# Krippendorff
# ---------------------------------------------------------------------------


def _delta_squared(
    metric: Metric,
    c: float,
    k: float,
    values: Sequence[float],
    marginals: Mapping[float, float],
) -> float:
    """The difference function between two values, under `metric`."""
    if c == k:
        return 0.0
    if metric is Metric.NOMINAL:
        return 1.0
    if metric is Metric.INTERVAL:
        return float((c - k) ** 2)

    # Ordinal: sum the marginal counts across the closed interval between the
    # two values, less half of each endpoint's own count. This is what makes a
    # 4-vs-5 disagreement cheap on a rubric where most responses score 4 or 5,
    # and expensive on one where they are far apart in the observed order.
    lo, hi = (c, k) if c < k else (k, c)
    between = sum(marginals[v] for v in values if lo <= v <= hi)
    return float((between - (marginals[c] + marginals[k]) / 2.0) ** 2)


def krippendorff_alpha(
    reliability_data: Sequence[Sequence[float | None]],
    metric: Metric = Metric.ORDINAL,
) -> float:
    """Krippendorff's alpha over a raters-by-units matrix.

    Args:
        reliability_data: One row per observer, one column per unit. `None`
            marks a missing observation. Rows must be the same length.
        metric: Difference function. Ordinal for this study's Likert rubric.

    Returns:
        Alpha, or `nan` when it is undefined — fewer than two units carrying
        two or more observations, or no variance at all in the observed values
        (every rater gave every unit the same score, in which case expected
        disagreement is zero and the ratio has no meaning). `nan` is returned
        rather than `1.0` because "everyone agreed on a constant" is not
        evidence that the instrument discriminates, and reporting it as perfect
        reliability would be a lie about a degenerate sample.

    Raises:
        ValueError: if the rows are ragged.
    """
    if not reliability_data:
        return math.nan
    width = len(reliability_data[0])
    if any(len(row) != width for row in reliability_data):
        raise ValueError("reliability_data rows must all have the same length")

    # Coincidence matrix: every ordered pair of values within a unit, weighted
    # by 1/(m_u - 1) so units rated by many observers do not dominate.
    coincidence: dict[tuple[float, float], float] = {}
    n_usable_units = 0
    for unit in range(width):
        present = [float(v) for v in (row[unit] for row in reliability_data) if v is not None]
        m = len(present)
        if m < 2:
            continue
        n_usable_units += 1
        weight = 1.0 / (m - 1)
        for i, ci in enumerate(present):
            for j, kj in enumerate(present):
                if i == j:
                    continue
                coincidence[(ci, kj)] = coincidence.get((ci, kj), 0.0) + weight

    if n_usable_units < 2 or not coincidence:
        return math.nan

    marginals: dict[float, float] = {}
    for (c, k), count in coincidence.items():
        marginals[c] = marginals.get(c, 0.0) + count
        marginals.setdefault(k, 0.0)
    values = sorted(marginals)
    n = sum(marginals.values())
    if n <= 1 or len(values) < 2:
        return math.nan

    observed = 0.0
    expected = 0.0
    for ci, c in enumerate(values):
        for k in values[ci + 1 :]:
            d2 = _delta_squared(metric, c, k, values, marginals)
            if d2 == 0.0:
                continue
            observed += (coincidence.get((c, k), 0.0) + coincidence.get((k, c), 0.0)) * d2
            expected += marginals[c] * marginals[k] * 2.0 * d2

    if expected == 0.0:
        return math.nan
    return 1.0 - (n - 1.0) * observed / expected


# ---------------------------------------------------------------------------
# Spearman
# ---------------------------------------------------------------------------


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Spearman's rho and its two-sided p-value. `(nan, nan)` when undefined.

    Undefined means fewer than three complete pairs, or either series constant —
    both of which happen on real dimensions (a judge that scores every response
    3 on `respect` produces a constant series) and neither of which should raise.
    """
    if len(x) != len(y):
        raise ValueError("spearman_rho needs two series of the same length")
    if len(x) < 3:
        return math.nan, math.nan
    if len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan, math.nan

    from scipy import stats  # heavy import, deferred so importing this module is cheap

    result = stats.spearmanr(list(x), list(y))
    return float(result.statistic), float(result.pvalue)


def paired_series(
    left: Mapping[str, float | None],
    right: Mapping[str, float | None],
) -> tuple[list[float], list[float], list[str]]:
    """Complete pairs from two unit-keyed mappings, in a stable order.

    Returns `(left_values, right_values, unit_ids)`. Units missing from either
    side, or `None` on either side, are dropped — pairwise deletion, which is
    the right choice when the missingness is a rejected judge score rather than
    a property of the unit.
    """
    units = sorted(set(left) & set(right))
    xs: list[float] = []
    ys: list[float] = []
    kept: list[str] = []
    for unit in units:
        a, b = left[unit], right[unit]
        if a is None or b is None:
            continue
        xs.append(float(a))
        ys.append(float(b))
        kept.append(unit)
    return xs, ys, kept
