"""Human agreement: inter-rater alpha, consensus, and the single-rater fallback.

v3 §12 ranks the options and this module supports all three, in order:

1. **Two or more raters.** `inter_rater_alpha` gives Krippendorff's alpha
   (ordinal) per dimension. Two raters is the floor for a defensible alpha.
2. **One external rater plus the study lead.** Same function; nothing special.
3. **One rater, scored twice, at least two weeks apart, blinded and reshuffled
   both times.** `intra_rater_reliability` treats the two occasions as two
   observers and reports intra-rater reliability. This is weaker evidence and
   the docstring says so where someone will read it — it measures how stable
   one person's judgement is, not whether two people would agree, and those are
   different claims about the rubric.

**Report alpha whatever it is.** A low alpha on `naturalness` is a finding about
the construct — it says the dimension is hard to score consistently, which is
worth knowing and is exactly the dimension this study cares about. Nothing here
drops a dimension, reweights a rater, or offers a "corrected" coefficient.

**Consensus is the median, not the mean.** With two or three raters on an
ordinal five-point scale, a mean produces 3.5s that exist on no rater's scale
and that Krippendorff's ordinal metric then has to treat as a distinct
category. The median stays on the scale. With two raters it takes the midpoint,
so `_median_int` rounds half up — the same documented tie-break the judge uses,
for the same reason.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence

from carelite.eval.judge.agreement import AgreementResult, Metric, krippendorff_alpha, spearman_rho
from carelite.eval.rubric.dimensions import to_quality
from carelite.types import RUBRIC_DIMENSIONS, RubricScore

__all__ = [
    "human_consensus",
    "inter_rater_alpha",
    "intra_rater_reliability",
    "reliability_matrix",
    "scores_by_rater",
]


def scores_by_rater(
    scores: Iterable[RubricScore],
) -> dict[str, dict[str, dict[str, int | None]]]:
    """`rater_id -> generation_id -> {dimension: raw score}`.

    Raw scale throughout: `ritualistic` stays higher-is-worse, matching the
    database column and what the rater wrote on the sheet. Canonicalisation to
    the quality scale happens at the point of comparison, once.
    """
    out: dict[str, dict[str, dict[str, int | None]]] = {}
    for score in scores:
        dumped = score.model_dump()
        out.setdefault(score.rater_id, {})[score.generation_id] = {
            key: dumped[key] for key in RUBRIC_DIMENSIONS
        }
    return out


def reliability_matrix(
    by_rater: Mapping[str, Mapping[str, Mapping[str, int | None]]],
    dimension: str,
    *,
    units: Sequence[str] | None = None,
    quality_scale: bool = True,
) -> tuple[list[list[float | None]], list[str], list[str]]:
    """Build the raters-by-units matrix Krippendorff's alpha consumes.

    Returns `(matrix, rater_ids, unit_ids)`. Missing cells are `None`.

    `quality_scale=True` runs every value through
    `carelite.eval.rubric.dimensions.to_quality` first. Ordinal alpha happens to
    be invariant to reversing a scale, so this cannot change the coefficient —
    it is done anyway so that every matrix leaving this module points the same
    way, and so a caller that mixes this output with judge output cannot end up
    comparing a raw series against a quality series.
    """
    raters = sorted(by_rater)
    if units is None:
        units = sorted({gid for scores in by_rater.values() for gid in scores})

    matrix: list[list[float | None]] = []
    for rater in raters:
        row: list[float | None] = []
        for unit in units:
            value = by_rater[rater].get(unit, {}).get(dimension)
            if value is None:
                row.append(None)
            else:
                row.append(float(to_quality(dimension, value)) if quality_scale else float(value))
        matrix.append(row)
    return matrix, raters, list(units)


def inter_rater_alpha(
    scores: Iterable[RubricScore],
    *,
    metric: Metric = Metric.ORDINAL,
) -> dict[str, AgreementResult]:
    """Krippendorff's alpha per dimension across every rater in `scores`.

    `rho` is filled only in the two-rater case, where it is well defined and
    informative — a pair of raters who rank identically but differ by a constant
    offset show low alpha and high rho, and that distinction changes what you do
    about it. With three or more raters `rho` is `nan` rather than an average of
    pairwise correlations, which would be a number with no sampling theory
    behind it.
    """
    by_rater = scores_by_rater(scores)
    out: dict[str, AgreementResult] = {}

    for key in RUBRIC_DIMENSIONS:
        matrix, raters, units = reliability_matrix(by_rater, key)
        alpha = krippendorff_alpha(matrix, metric=metric)

        rho, p = math.nan, math.nan
        if len(matrix) == 2:
            pairs = [
                (a, b)
                for a, b in zip(matrix[0], matrix[1], strict=True)
                if a is not None and b is not None
            ]
            if pairs:
                rho, p = spearman_rho([a for a, _ in pairs], [b for _, b in pairs])

        n_units = sum(
            1 for idx in range(len(units)) if sum(1 for row in matrix if row[idx] is not None) >= 2
        )
        out[key] = AgreementResult(
            dimension=key,
            n_units=n_units,
            n_observers=len(raters),
            alpha=alpha,
            rho=rho,
            rho_p=p,
        )
    return out


def _median_int(values: Sequence[int]) -> int:
    """Median rounded half up. Same tie-break as the judge, stated in both places."""
    exact = float(statistics.median(values))
    return int(exact + 0.5) if exact >= 0 else int(exact - 0.5)


def human_consensus(
    scores: Iterable[RubricScore],
    *,
    min_raters: int = 1,
) -> dict[str, dict[str, int | None]]:
    """`generation_id -> {dimension: consensus raw score}`, median across raters.

    Raw scale, so it can be handed straight to
    `carelite.eval.judge.validation.judge_human_validity`, which canonicalises
    both sides itself.

    Args:
        min_raters: Dimensions rated by fewer than this many raters become
            `None`. Leave at 1 for the single-rater fallback; set to 2 when a
            consensus should mean more than one person's opinion.
    """
    gathered: dict[str, dict[str, list[int]]] = {}
    for score in scores:
        dumped = score.model_dump()
        bucket = gathered.setdefault(score.generation_id, {k: [] for k in RUBRIC_DIMENSIONS})
        for key in RUBRIC_DIMENSIONS:
            value = dumped[key]
            if value is not None:
                bucket[key].append(int(value))

    return {
        gid: {
            key: (_median_int(values) if len(values) >= min_raters and values else None)
            for key, values in dims.items()
        }
        for gid, dims in gathered.items()
    }


def intra_rater_reliability(
    occasion_1: Iterable[RubricScore],
    occasion_2: Iterable[RubricScore],
    *,
    metric: Metric = Metric.ORDINAL,
) -> dict[str, AgreementResult]:
    """Intra-rater reliability: the same rater, twice, at least two weeks apart.

    Named `intra_rater_reliability` rather than `test_retest_*` deliberately:
    pytest collects any imported callable whose name starts with `test_`, so the
    obvious name turns this function into a broken test case in every module
    that imports it.

    The v3 §12 fallback when only one rater is available. What it measures is
    the stability of one person's judgement against itself, and that is a
    strictly weaker claim than inter-rater agreement: a rater who is
    consistently wrong in the same direction scores a high alpha here. It is
    reported as **intra-rater reliability**, never as "agreement", and the
    write-up should not let the two numbers sit in one column without a label.

    Both occasions must be blinded and reshuffled independently — a second pass
    in the same order is a memory test, and `build_packet` reshuffles per rater
    but not per occasion, so pass a distinct `rater_id` (for example
    `"R01-t2"`) for the retest. That id difference is also what lets this
    function treat the two occasions as two observers.
    """
    first = list(occasion_1)
    second = list(occasion_2)
    raters_1 = {s.rater_id for s in first}
    raters_2 = {s.rater_id for s in second}
    if raters_1 & raters_2:
        raise ValueError(
            "test-retest needs the two occasions to carry different rater ids "
            f"(shared: {sorted(raters_1 & raters_2)}); otherwise they collapse into one observer"
        )

    return inter_rater_alpha([*first, *second], metric=metric)
