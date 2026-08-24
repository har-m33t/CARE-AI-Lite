"""Effect sizes with 95% bootstrap confidence intervals. Reported before p-values.

Pre-registration §8.2: "Effect sizes with 95% bootstrap confidence intervals are
computed and reported for every comparison, and are reported before the
corresponding p-value in every table and figure. At n = 60 the effect size and
its interval carry more information than the p-value; the ordering in the
write-up reflects that."

That ordering is enforced structurally rather than remembered. `PairwiseResult`
in `carelite.stats.primary` cannot be constructed without its effect estimates,
`carelite.stats.report` emits the effect columns first, and there is no function
in this package that returns a p-value on its own.

**Three estimators, because they answer three different questions and the
pre-registration names none of them specifically.** It fixes the *test*
(Wilcoxon signed-rank) and the *interval method* (bootstrap, 95%) but not the
point estimate, so all three are computed and reported together rather than one
being chosen after the fact:

- **Matched-pairs rank-biserial correlation** — the effect size that belongs to
  the Wilcoxon signed-rank test, computed from the same signed ranks the test
  uses. (W+ - W-) / (W+ + W-), so it is bounded [-1, 1], is 1 when every
  scenario moves the same way, and 0 when the signed ranks balance. This is the
  headline estimator: it is the one whose null is the tested null.
- **Cohen's dz** — the paired standardised mean difference. Reported because it
  is the scale the power analysis (§6) is stated in, so it is the only estimator
  that can be read against `detectable_effect(60)`. It assumes more than the
  rank statistic does and is included for that comparison, not as the primary
  estimate.
- **Hodges-Lehmann median difference** — the median of the Walsh averages, in
  raw rubric points. The estimator the signed-rank test inverts, so its
  interval and the test agree; it answers "how much better, in points" in a way
  neither correlation nor dz does.

**The resampling unit is the scenario.** Bootstrap resamples draw scenarios with
replacement and take each drawn scenario's whole paired difference with it. That
is the only unit the design supports: the three samples in a cell share a
scenario and a condition, and resampling them independently would manufacture
precision out of within-cell generation variance. `bootstrap_ci` takes a matrix
whose rows are units and never sees the samples at all — they are already
averaged into cell means by `carelite.stats.measures.cell_means`, and the
variance they carry is modelled properly in `carelite.stats.mixed`.

**Ties and zeros.** A scenario whose paired difference is exactly zero carries no
information about direction. It is dropped from the signed-rank effect size, as
it is from the test itself (scipy's `zero_method="wilcox"`), and `n_nonzero`
records how many were dropped. dz and Hodges-Lehmann keep them, because a zero
difference is real information about magnitude even when it is none about rank.

**Degenerate resamples.** A bootstrap resample can contain only zero
differences, or a single distinct value; the estimator is then undefined and the
replicate is `nan`. Those replicates are dropped from the percentile calculation
and counted in `n_valid_resamples`. A CI computed from noticeably fewer
replicates than were requested is a signal about the data, so it is reported
rather than hidden.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_N_BOOT",
    "DEFAULT_SEED",
    "BootstrapCI",
    "EffectEstimate",
    "PairedEffects",
    "bootstrap_ci",
    "cohens_dz",
    "hodges_lehmann",
    "paired_effects",
    "rank_biserial",
]

#: Enough replicates that the 2.5th and 97.5th percentiles are stable to about
#: the third decimal; cheap at n = 60.
DEFAULT_N_BOOT = 10_000

#: `carelite.config.Experiment.base_seed`. Fixed so a reported interval is
#: reproducible from the same database without re-running anything upstream.
DEFAULT_SEED = 20260822


# ---------------------------------------------------------------------------
# Point estimators
# ---------------------------------------------------------------------------


def _differences(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must be the same length, got {a.shape} and {b.shape}")
    if a.ndim != 1:
        raise ValueError("paired arrays must be one-dimensional")
    return a - b


def _signed_ranks(d: np.ndarray) -> tuple[float, float, int]:
    """Sum of positive ranks, sum of negative ranks, and the count of nonzero pairs.

    Ranks are over |d| with midranks for ties, matching the Wilcoxon signed-rank
    test's own treatment.
    """
    nonzero = d[d != 0]
    n = nonzero.size
    if n == 0:
        return 0.0, 0.0, 0
    magnitudes = np.abs(nonzero)
    order = np.argsort(magnitudes, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    sorted_mag = magnitudes[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_mag[j + 1] == sorted_mag[i]:
            j += 1
        midrank = (i + j + 2) / 2.0  # ranks are 1-based
        ranks[order[i : j + 1]] = midrank
        i = j + 1
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    return w_plus, w_minus, n


def rank_biserial(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, (W+ - W-) / (W+ + W-).

    Positive means `x` scores higher than `y`. `nan` when every pair is tied,
    which is undefined rather than zero: no evidence of direction is not
    evidence of no direction.
    """
    d = _differences(x, y)
    w_plus, w_minus, n = _signed_ranks(d)
    total = w_plus + w_minus
    if n == 0 or total == 0:
        return math.nan
    return (w_plus - w_minus) / total


def cohens_dz(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Paired standardised mean difference: mean(x - y) / sd(x - y), ddof = 1.

    The scale the power analysis in `carelite.stats.power` is stated in. `nan`
    for fewer than two pairs or a zero-variance difference vector.
    """
    d = _differences(x, y)
    if d.size < 2:
        return math.nan
    sd = float(np.std(d, ddof=1))
    if sd == 0:
        return math.nan
    return float(np.mean(d)) / sd


def hodges_lehmann(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Median of the Walsh averages of the paired differences, in rubric points.

    The location estimator the signed-rank test inverts: the value the test would
    fail to reject as the shift between the two conditions.
    """
    d = _differences(x, y)
    if d.size == 0:
        return math.nan
    i, j = np.triu_indices(d.size, k=0)
    walsh = (d[i] + d[j]) / 2.0
    return float(np.median(walsh))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """A percentile bootstrap interval and the bookkeeping needed to trust it."""

    low: float
    high: float
    confidence: float
    n_boot: int
    n_valid_resamples: int
    n_units: int
    seed: int
    method: str = "percentile"

    @property
    def excludes_zero(self) -> bool:
        """True when the whole interval lies strictly on one side of zero."""
        if math.isnan(self.low) or math.isnan(self.high):
            return False
        return (self.low > 0) or (self.high < 0)


def bootstrap_ci(
    units: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    confidence: float = 0.95,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> BootstrapCI:
    """Percentile bootstrap over rows of `units`, which are the resampling unit.

    `units` is `(n_units, ...)`: one row per scenario. Whole rows are drawn with
    replacement, so a scenario enters or leaves a resample as a unit and its
    paired structure is never broken.

    `statistic` maps a resampled `units` array to a scalar and may return `nan`
    for a degenerate resample; such replicates are dropped and counted.
    """
    arr = np.asarray(units, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_units = arr.shape[0]
    if n_units == 0:
        return BootstrapCI(math.nan, math.nan, confidence, n_boot, 0, 0, seed)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_boot, dtype=float)
    index = rng.integers(0, n_units, size=(n_boot, n_units))
    for b in range(n_boot):
        replicates[b] = statistic(arr[index[b]])

    valid = replicates[np.isfinite(replicates)]
    if valid.size == 0:
        return BootstrapCI(math.nan, math.nan, confidence, n_boot, 0, n_units, seed)
    tail = (1 - confidence) / 2 * 100
    low = float(np.percentile(valid, tail))
    high = float(np.percentile(valid, 100 - tail))
    return BootstrapCI(
        low=low,
        high=high,
        confidence=confidence,
        n_boot=n_boot,
        n_valid_resamples=int(valid.size),
        n_units=n_units,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class EffectEstimate:
    """One effect size with its interval. Point and interval travel together."""

    estimator: str
    point: float
    ci: BootstrapCI

    @property
    def interval(self) -> tuple[float, float]:
        return (self.ci.low, self.ci.high)

    def render(self) -> str:
        return (
            f"{self.point:+.3f} "
            f"[{self.ci.low:+.3f}, {self.ci.high:+.3f}] "
            f"({self.ci.confidence:.0%} bootstrap CI, {self.ci.n_units} scenarios)"
        )


@dataclass(frozen=True, slots=True)
class PairedEffects:
    """All three estimators for one paired comparison, on one measure.

    The headline is `rank_biserial`: it is the effect size belonging to the test
    that is actually run. The other two are reported alongside it, never instead
    of it.
    """

    n_pairs: int
    n_nonzero: int
    mean_left: float
    mean_right: float
    rank_biserial: EffectEstimate
    cohens_dz: EffectEstimate
    hodges_lehmann: EffectEstimate

    @property
    def headline(self) -> EffectEstimate:
        return self.rank_biserial

    @property
    def direction(self) -> str:
        """`>`, `<` or `=` for the observed direction of the headline estimate."""
        point = self.rank_biserial.point
        if math.isnan(point) or point == 0:
            return "="
        return ">" if point > 0 else "<"


def paired_effects(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> PairedEffects:
    """Every effect size for one paired comparison, each with a bootstrap CI.

    `x` and `y` are aligned scenario-level cell means — the left and right
    conditions of the comparison, in the order the hypothesis is stated. One
    resampling draw is shared across all three estimators (same seed, same unit
    index), so their intervals describe the same resamples and can be read
    against each other.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must be the same length")
    pairs = np.column_stack([a, b])
    d = a - b
    _, _, n_nonzero = _signed_ranks(d)

    def _rb(units: np.ndarray) -> float:
        return rank_biserial(units[:, 0], units[:, 1])

    def _dz(units: np.ndarray) -> float:
        return cohens_dz(units[:, 0], units[:, 1])

    def _hl(units: np.ndarray) -> float:
        return hodges_lehmann(units[:, 0], units[:, 1])

    return PairedEffects(
        n_pairs=int(a.size),
        n_nonzero=n_nonzero,
        mean_left=float(np.mean(a)) if a.size else math.nan,
        mean_right=float(np.mean(b)) if b.size else math.nan,
        rank_biserial=EffectEstimate(
            "rank_biserial",
            rank_biserial(a, b),
            bootstrap_ci(pairs, _rb, confidence=confidence, n_boot=n_boot, seed=seed),
        ),
        cohens_dz=EffectEstimate(
            "cohens_dz",
            cohens_dz(a, b),
            bootstrap_ci(pairs, _dz, confidence=confidence, n_boot=n_boot, seed=seed),
        ),
        hodges_lehmann=EffectEstimate(
            "hodges_lehmann",
            hodges_lehmann(a, b),
            bootstrap_ci(pairs, _hl, confidence=confidence, n_boot=n_boot, seed=seed),
        ),
    )
