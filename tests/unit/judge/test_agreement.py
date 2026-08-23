"""Krippendorff's alpha, checked against a from-definition oracle.

An alpha implementation validated only against its own output is worthless, and
alpha is the coefficient the whole judge-validation verdict turns on. So
`_alpha_from_definition` below recomputes it the long way — enumerate every
ordered pair inside every unit, build the observed and expected disagreements
straight from Krippendorff's definitions, divide — and the tests assert the two
agree. The optimised version in `agreement.py` collapses that into a coincidence
matrix; the oracle does not, so a mistake in the collapse shows up here.

The nominal case is additionally pinned to a value derived by hand in the
docstring of `test_matches_hand_computation`, so the oracle itself is anchored
to arithmetic that can be checked on paper.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import permutations

import pytest

from carelite.eval.judge.agreement import Metric, krippendorff_alpha, paired_series, spearman_rho

#: Three coders, twelve units, missing cells in both patterns that matter:
#: a unit only one coder saw, and a coder who stopped partway.
CODERS: list[list[float | None]] = [
    [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
    [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
]


def _alpha_from_definition(data: Sequence[Sequence[float | None]], metric: Metric) -> float:
    """Krippendorff's alpha computed the long way. Independent of `agreement.py`."""
    units = []
    for column in range(len(data[0])):
        values = [row[column] for row in data if row[column] is not None]
        if len(values) >= 2:
            units.append(values)

    flat = [v for unit in units for v in unit]
    n = len(flat)
    marginals: dict[float, int] = {}
    for value in flat:
        marginals[value] = marginals.get(value, 0) + 1
    ordered = sorted(marginals)

    def delta2(c: float, k: float) -> float:
        if c == k:
            return 0.0
        if metric is Metric.NOMINAL:
            return 1.0
        if metric is Metric.INTERVAL:
            return float((c - k) ** 2)
        lo, hi = (c, k) if c < k else (k, c)
        between = sum(marginals[v] for v in ordered if lo <= v <= hi)
        return float((between - (marginals[c] + marginals[k]) / 2.0) ** 2)

    observed = 0.0
    for unit in units:
        pairs = sum(delta2(a, b) for a, b in permutations(unit, 2))
        observed += pairs / (len(unit) - 1)
    observed /= n

    expected = sum(marginals[c] * marginals[k] * delta2(c, k) for c in ordered for k in ordered) / (
        n * (n - 1)
    )
    return 1.0 - observed / expected


class TestAgainstTheOracle:
    @pytest.mark.parametrize("metric", list(Metric))
    def test_matches_a_from_definition_implementation(self, metric: Metric) -> None:
        assert krippendorff_alpha(CODERS, metric=metric) == pytest.approx(
            _alpha_from_definition(CODERS, metric), abs=1e-12
        )

    def test_matches_hand_computation(self) -> None:
        """Nominal alpha for CODERS, derived on paper.

        Ten units carry two or more values, so n = 28. Marginal counts are
        n1=5, n2=10, n3=8, n4=3, n5=2. Only three units disagree: [2,2,3]
        contributes 4 ordered disagreeing pairs / 2 = 2, [1,2,3] contributes
        6 / 2 = 3, and [1,1,2] contributes 4 / 2 = 2, so the observed
        disagreement sum is 7. Expected is n^2 - sum(n_c^2) = 784 - 202 = 582.
        alpha = 1 - 27 * 7 / 582 = 0.675258...
        """
        assert krippendorff_alpha(CODERS, metric=Metric.NOMINAL) == pytest.approx(
            1 - 27 * 7 / 582, abs=1e-9
        )

    def test_ordinal_is_more_forgiving_than_nominal_here(self) -> None:
        """Why the study pre-specifies ordinal: near-misses should cost less.

        Under nominal coding a 2-vs-3 disagreement is as bad as a 1-vs-5 one,
        which on an anchored Likert rubric would understate the judge.
        """
        nominal = krippendorff_alpha(CODERS, metric=Metric.NOMINAL)
        ordinal = krippendorff_alpha(CODERS, metric=Metric.ORDINAL)
        assert ordinal > nominal


class TestBehaviour:
    def test_perfect_agreement_is_one(self) -> None:
        data = [[1, 2, 3, 4, 5, 1, 2, 3], [1, 2, 3, 4, 5, 1, 2, 3]]
        assert krippendorff_alpha(data, metric=Metric.ORDINAL) == pytest.approx(1.0)

    def test_independent_raters_land_near_zero(self) -> None:
        import random

        rng = random.Random(11)
        n = 400
        data = [
            [float(rng.randint(1, 5)) for _ in range(n)],
            [float(rng.randint(1, 5)) for _ in range(n)],
        ]
        assert abs(krippendorff_alpha(data, metric=Metric.ORDINAL)) < 0.12

    def test_systematic_reversal_is_strongly_negative(self) -> None:
        """A rater with the scale backwards is worse than a coin flip, and shows it."""
        truth = [1, 2, 3, 4, 5] * 8
        data = [[float(v) for v in truth], [float(6 - v) for v in truth]]
        assert krippendorff_alpha(data, metric=Metric.ORDINAL) < -0.5

    def test_reversing_both_series_leaves_ordinal_alpha_unchanged(self) -> None:
        """Ordinal alpha is direction-invariant, so `ritualistic` cannot break it.

        This is why the reverse-coding risk lives in Spearman's rho and in
        cross-dimension means, not here — but only when BOTH sides use the same
        coding, which is what `to_quality` at the boundary guarantees.
        """
        data = [[1, 2, 3, 4, 5, 2, 4], [1, 3, 3, 4, 5, 1, 4]]
        flipped = [[6 - v for v in row] for row in data]
        assert krippendorff_alpha(data, metric=Metric.ORDINAL) == pytest.approx(
            krippendorff_alpha(flipped, metric=Metric.ORDINAL)
        )

    def test_missing_data_is_handled_pairwise(self) -> None:
        full = [[1, 2, 3, 4, 5, 1], [1, 2, 3, 4, 5, 1]]
        holed = [[1, 2, 3, 4, 5, None], [1, 2, 3, 4, 5, 1]]
        # The unit with one observation drops out; the rest still agree perfectly.
        assert krippendorff_alpha(holed, metric=Metric.ORDINAL) == pytest.approx(
            krippendorff_alpha(full, metric=Metric.ORDINAL)
        )

    def test_no_variance_is_undefined_not_perfect(self) -> None:
        """Everyone scoring everything 3 is a degenerate sample, not reliability."""
        data = [[3, 3, 3, 3], [3, 3, 3, 3]]
        assert math.isnan(krippendorff_alpha(data, metric=Metric.ORDINAL))

    def test_too_few_paired_units_is_undefined(self) -> None:
        assert math.isnan(krippendorff_alpha([[1, None], [None, 2]], metric=Metric.ORDINAL))

    def test_empty_input_is_undefined(self) -> None:
        assert math.isnan(krippendorff_alpha([], metric=Metric.ORDINAL))

    def test_ragged_rows_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            krippendorff_alpha([[1, 2], [1]], metric=Metric.ORDINAL)


class TestSpearman:
    def test_monotone_agreement_is_one(self) -> None:
        rho, _ = spearman_rho([1, 2, 3, 4, 5], [2, 3, 3, 4, 5])
        assert rho > 0.9

    def test_constant_offset_keeps_rho_high(self) -> None:
        """The lenient-rater signature: alpha falls, rho does not.

        Reported separately because the two say different things about whether
        a ranking-based analysis is safe.
        """
        truth = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        lenient = [min(5, v + 1) for v in truth]
        rho, _ = spearman_rho(truth, lenient)
        alpha = krippendorff_alpha(
            [[float(v) for v in truth], [float(v) for v in lenient]], metric=Metric.ORDINAL
        )
        assert rho > 0.95
        assert alpha < rho

    def test_constant_series_is_undefined(self) -> None:
        rho, p = spearman_rho([3, 3, 3, 3], [1, 2, 3, 4])
        assert math.isnan(rho) and math.isnan(p)

    def test_too_short_is_undefined(self) -> None:
        assert math.isnan(spearman_rho([1, 2], [1, 2])[0])

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            spearman_rho([1, 2, 3], [1, 2])


class TestPairedSeries:
    def test_drops_units_missing_on_either_side(self) -> None:
        left = {"a": 1, "b": None, "c": 3, "d": 4}
        right = {"a": 2, "b": 2, "c": None, "e": 5}
        xs, ys, kept = paired_series(left, right)
        assert kept == ["a"]
        assert xs == [1.0] and ys == [2.0]

    def test_order_is_stable(self) -> None:
        left = {"z": 1, "a": 2, "m": 3}
        right = {"z": 1, "a": 2, "m": 3}
        assert paired_series(left, right)[2] == ["a", "m", "z"]
