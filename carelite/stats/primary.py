"""The primary test sequence: Friedman omnibus, Wilcoxon post-hoc, Holm-Bonferroni.

Pre-registration §8.1, quoted because the exact wording is what this module
implements:

    Friedman omnibus test across conditions {A, B, C} for each of the 11 rubric
    dimensions (on to_quality()-transformed scores), followed by Wilcoxon
    signed-rank on every pairwise comparison listed in §4, followed by
    Holm-Bonferroni correction applied across the whole
    pairwise-comparison-by-dimension family -- not per dimension separately. A
    dimension is not tested in isolation from the others for correction purposes.

==========================================================================
THE CORRECTION FAMILY, STATED EXPLICITLY
==========================================================================
Getting this wrong is the most common way a result of this shape falls apart
under review, so the reading is written down rather than left in the code.

**The confirmatory family is the eight hypothesis tests the pre-registration
enumerates** -- the primary outcome (§3) plus the seven directional secondary
outcomes (§4) -- corrected together, in one Holm-Bonferroni step, as
`CONFIRMATORY_FAMILY`. Each of those eight names its own outcome measure: five
are on the NURSE composite, one on the Four Habits composite, one on
`naturalness`, one on `ritualistic`. So the family spans several comparisons
*and* several dimensions, and the naturalness test is corrected in the same
step as the NURSE-composite tests. That is what "not per dimension separately"
forbids, and it is what this module does.

**What is deliberately NOT in the confirmatory family.** §8.1 says "every
pairwise comparison listed in §4", and §4 lists eight comparisons each with a
named measure. It does not list A vs C, and it does not list, say, `de` for
A vs B. Testing all five condition pairs on all eleven dimensions would be 55
tests, of which 47 are hypotheses nobody registered; folding them into the
confirmatory family would multiply the correction on the registered eight by
about seven for the sake of tests that are exploratory by §1's own definition.
Those tests are still available and still worth looking at -- `dimension_expansion`
builds them, they are corrected within their own separate family, and every one
of them is labelled EXPLORATORY in the result object itself.

**The wording admits a second reading** -- "pairwise-comparison-by-dimension
family" could be read as that 55-cell cross product. If the project prefers it,
it is a one-line change (`CONFIRMATORY_FAMILY = (*PRESPECIFIED_HYPOTHESES,
*dimension_expansion())`) and it is a pre-registration amendment, to be made
before registration rather than after seeing which reading is kinder to the
numbers.

**The eleven Friedman omnibus tests are not in the Holm family.** §8.1 puts the
correction on the pairwise family; an omnibus test is the gate in front of it,
in the standard omnibus-then-post-hoc shape. Their raw p-values are reported,
and a Holm correction *within the eleven omnibus tests* is reported beside them
so a reader who wants that adjustment does not have to compute it -- clearly
labelled, and not the basis of any confirmatory claim.

==========================================================================
TWO-SIDED TESTS FOR DIRECTIONAL HYPOTHESES
==========================================================================
§4 states a direction for every secondary outcome; §8.1 fixes the test but not
its sidedness. Every test here is therefore **two-sided**, which is the
conservative choice: a one-sided test at alpha = 0.05 would be easier to pass in
the hypothesised direction and would have no power at all against the opposite
one -- and secondary outcome 4 (naturalness, A > B) is precisely a hypothesis
the study expects to come out against the system, so an analysis that could not
detect the opposite direction would defeat the purpose of registering it. The
hypothesised direction is recorded on every result and compared against the
observed one, so "significant, in the predicted direction" and "significant, in
the opposite direction" are distinguishable without a one-sided test.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

from carelite.eval.judge.validation import EvidenceStatus
from carelite.stats.effects import (
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    PairedEffects,
    paired_effects,
)
from carelite.stats.evidence import Label, RaterScope, label_for
from carelite.stats.measures import Measure, cell_means, measure, paired_matrix
from carelite.types import RUBRIC_DIMENSIONS, Condition

__all__ = [
    "CONFIRMATORY_FAMILY",
    "FRIEDMAN_CONDITIONS",
    "PRESPECIFIED_HYPOTHESES",
    "FamilyResult",
    "FriedmanResult",
    "Hypothesis",
    "PairwiseResult",
    "WilcoxonResult",
    "dimension_expansion",
    "friedman_across_conditions",
    "friedman_omnibus",
    "holm_bonferroni",
    "run_family",
    "run_pairwise",
    "wilcoxon_paired",
]

#: Pre-registration §8.1. The omnibus runs across these three only.
FRIEDMAN_CONDITIONS: tuple[Condition, ...] = (Condition.A, Condition.B, Condition.C)


# ---------------------------------------------------------------------------
# The pre-specified hypotheses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One pre-specified (or exploratory) pairwise comparison.

    `left` and `right` are in the order the pre-registration states the
    comparison, and every effect size is computed as left-relative-to-right, so
    a positive effect always means "left scored higher". `expected_higher` is
    the registered direction, or `None` for outcome 6, whose registered
    prediction is that there is no difference.
    """

    key: str
    measure_key: str
    left: Condition
    right: Condition
    expected_higher: Condition | None
    description: str
    role: str = "secondary"
    prespecified: bool = True

    @property
    def measure(self) -> Measure:
        return measure(self.measure_key)

    @property
    def pair_label(self) -> str:
        return f"{self.left} vs {self.right}"

    @property
    def expected_direction(self) -> str:
        """`>`, `<` or `=` for `left` relative to `right`, as registered."""
        if self.expected_higher is None:
            return "="
        return ">" if self.expected_higher == self.left else "<"


#: Pre-registration §3 (primary) and §4 (secondary outcomes 1-7), in that order.
PRESPECIFIED_HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis(
        key="primary_nurse_A_vs_B",
        measure_key="nurse_composite",
        left=Condition.A,
        right=Condition.B,
        expected_higher=Condition.B,
        description=(
            "PRIMARY (§3). Composite NURSE adherence, A vs B. Framework prompting beats the "
            "bare model; the structural-adherence effect v3 §11 expects to be large."
        ),
        role="primary",
    ),
    Hypothesis(
        key="secondary1_four_habits_A_vs_B",
        measure_key="four_habits_composite",
        left=Condition.A,
        right=Condition.B,
        expected_higher=Condition.B,
        description="§4.1 Composite Four Habits adherence, A vs B. Same direction, same mechanism.",
    ),
    Hypothesis(
        key="secondary2_nurse_B_vs_C",
        measure_key="nurse_composite",
        left=Condition.B,
        right=Condition.C,
        expected_higher=Condition.C,
        description=(
            "§4.2 Composite NURSE adherence, B vs C. Retrieval adds grounded guidance beyond "
            "framework prompting. Expected to be the smallest effect in the study; this is the "
            "comparison that set n (§6)."
        ),
    ),
    Hypothesis(
        key="secondary3_nurse_C_vs_LC",
        measure_key="nurse_composite",
        left=Condition.C,
        right=Condition.LC,
        expected_higher=Condition.C,
        description=(
            "§4.3 Composite NURSE adherence, C vs LC. Curated retrieval outperforms or matches "
            "naive long-context stuffing. Registered as C >= LC; tested two-sided, so a null "
            "result is consistent with the registered hypothesis rather than a failure of it."
        ),
    ),
    Hypothesis(
        key="secondary4_naturalness_A_vs_B",
        measure_key="naturalness",
        left=Condition.A,
        right=Condition.B,
        expected_higher=Condition.A,
        description=(
            "§4.4 naturalness, A vs B, hypothesised A > B. The against-the-system finding this "
            "pre-registration exists to protect."
        ),
    ),
    Hypothesis(
        key="secondary5_ritualistic_A_vs_B",
        measure_key="ritualistic",
        left=Condition.A,
        right=Condition.B,
        expected_higher=Condition.A,
        description=(
            "§4.5 ritualistic, A vs B. REVERSE-CODED: analysed on to_quality(), on which the "
            "registered prediction 'B has a higher raw ritualistic score' becomes 'A scores "
            "higher quality than B', consistent with outcome 4's direction."
        ),
    ),
    Hypothesis(
        key="secondary6_nurse_A_vs_A2",
        measure_key="nurse_composite",
        left=Condition.A,
        right=Condition.A2,
        expected_higher=None,
        description=(
            "§4.6 Composite NURSE adherence, A vs A2. Cross-model baseline; registered "
            "prediction is NO significant difference. A significant result here is evidence "
            "about model family, not about the framework."
        ),
    ),
    Hypothesis(
        key="secondary7_nurse_B_vs_D",
        measure_key="nurse_composite",
        left=Condition.B,
        right=Condition.D,
        expected_higher=Condition.B,
        description=(
            "§4.7 Composite NURSE adherence, B vs D. The negative control: if the rubric cannot "
            "separate B from the deliberately degraded prompt, the rubric is not measuring the "
            "construct it claims to. See carelite.stats.negative_control."
        ),
    ),
)

#: The Holm family. See the module docstring for what is in it and what is not.
CONFIRMATORY_FAMILY: tuple[Hypothesis, ...] = PRESPECIFIED_HYPOTHESES


def dimension_expansion(
    hypotheses: Sequence[Hypothesis] = PRESPECIFIED_HYPOTHESES,
    dimensions: Sequence[str] = RUBRIC_DIMENSIONS,
) -> tuple[Hypothesis, ...]:
    """Every registered condition pair crossed with every rubric dimension.

    EXPLORATORY by construction: `prespecified=False` on every one, so the label
    machinery in `carelite.stats.evidence` demotes them whatever the judge
    validation says. Corrected within its own family, never folded into
    `CONFIRMATORY_FAMILY`.
    """
    pairs: list[tuple[Condition, Condition]] = []
    for h in hypotheses:
        if (h.left, h.right) not in pairs:
            pairs.append((h.left, h.right))

    registered = {(h.left, h.right, h.measure_key) for h in hypotheses}
    out: list[Hypothesis] = []
    for left, right in pairs:
        for dim in dimensions:
            if (left, right, dim) in registered:
                continue
            out.append(
                Hypothesis(
                    key=f"exploratory_{dim}_{left}_vs_{right}",
                    measure_key=dim,
                    left=left,
                    right=right,
                    expected_higher=None,
                    description=(
                        f"EXPLORATORY. {dim}, {left} vs {right}. Not named in the "
                        "pre-registration; no registered direction."
                    ),
                    role="exploratory",
                    prespecified=False,
                )
            )
    return tuple(out)


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FriedmanResult:
    """Friedman omnibus for one measure across the conditions tested."""

    measure_key: str
    conditions: tuple[str, ...]
    statistic: float
    p_value: float
    df: int
    n_blocks: int
    label: Label
    #: Holm-corrected within the set of omnibus tests only. See the module
    #: docstring: this is reported for the reader's convenience and is not the
    #: pre-specified correction, which applies to the pairwise family.
    p_holm_within_omnibus: float = math.nan


def friedman_omnibus(matrix: pd.DataFrame) -> tuple[float, float, int, int]:
    """Friedman chi-square across the columns of `matrix`. Rows are scenarios.

    Returns `(statistic, p_value, df, n_blocks)`. `(nan, nan, df, n)` when the
    test is undefined: fewer than three conditions, fewer than two complete
    blocks, or a matrix in which every block is constant so no ranking exists.
    """
    k = matrix.shape[1]
    n = matrix.shape[0]
    df = max(k - 1, 0)
    if k < 3 or n < 2:
        return math.nan, math.nan, df, n
    columns = [matrix.iloc[:, j].to_numpy(dtype=float) for j in range(k)]
    if np.allclose(matrix.to_numpy(dtype=float).std(axis=1), 0.0):
        # Every block is tied across conditions: the statistic is 0/0.
        return math.nan, math.nan, df, n
    statistic, p_value = stats.friedmanchisquare(*columns)
    return float(statistic), float(p_value), df, n


@dataclass(frozen=True, slots=True)
class WilcoxonResult:
    """One Wilcoxon signed-rank test, with the bookkeeping needed to read it."""

    statistic: float
    p_value: float
    n_pairs: int
    n_nonzero: int
    alternative: str
    method: str


def wilcoxon_paired(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    alternative: str = "two-sided",
) -> WilcoxonResult:
    """Wilcoxon signed-rank on paired scenario-level values.

    Zeros are dropped (`zero_method="wilcox"`), matching the effect size in
    `carelite.stats.effects`. scipy chooses the exact distribution for small
    samples without ties and the normal approximation with continuity correction
    otherwise; which one it used is recorded in `method`.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must be the same length")
    d = a - b
    nonzero = int(np.count_nonzero(d))
    if nonzero == 0:
        return WilcoxonResult(math.nan, math.nan, int(a.size), 0, alternative, "undefined")

    has_ties = np.unique(np.abs(d[d != 0])).size != nonzero
    mode = "exact" if (nonzero <= 25 and not has_ties) else "approx"
    result = stats.wilcoxon(
        a,
        b,
        zero_method="wilcox",
        alternative=alternative,
        method=mode,
        correction=mode == "approx",
    )
    return WilcoxonResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        n_pairs=int(a.size),
        n_nonzero=nonzero,
        alternative=alternative,
        method=mode,
    )


def holm_bonferroni(p_values: Sequence[float], *, family_size: int | None = None) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order.

    The step-down procedure: sort ascending, multiply the i-th smallest by
    (m - i), enforce monotonicity by running a cumulative maximum up the sorted
    list, and cap at 1.

    `family_size` defaults to `len(p_values)` and exists for the case where a
    pre-specified test could not be computed. The family is fixed in advance by
    the analysis plan, so an undefined test still consumes its share of the
    correction rather than making the surviving tests easier to pass. `nan`
    inputs are returned as `nan` and are excluded from the ordering.
    """
    values = list(p_values)
    m = len(values) if family_size is None else family_size
    if m <= 0:
        return []
    finite = [(p, i) for i, p in enumerate(values) if not math.isnan(p)]
    finite.sort(key=lambda t: t[0])

    adjusted = [math.nan] * len(values)
    running = 0.0
    for rank, (p, index) in enumerate(finite):
        candidate = (m - rank) * p
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


# ---------------------------------------------------------------------------
# One comparison, and one family of them
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PairwiseResult:
    """One pairwise comparison. Effect size first, deliberately.

    The field order is the reporting order the pre-registration fixes in §8.2,
    and `render()` follows it. There is no constructor path that produces a
    p-value without the effect size and its interval beside it.
    """

    hypothesis: Hypothesis
    effects: PairedEffects
    test: WilcoxonResult
    label: Label
    n_scenarios: int
    n_dropped: int
    p_holm: float = math.nan
    family_size: int = 0

    @property
    def observed_direction(self) -> str:
        return self.effects.direction

    @property
    def direction_as_registered(self) -> bool | None:
        """True/False when a direction was registered, `None` when none was.

        Outcome 6 registers "no significant difference", which has no direction
        to agree or disagree with, so it returns `None` rather than a misleading
        boolean.
        """
        expected = self.hypothesis.expected_direction
        if expected == "=":
            return None
        return self.observed_direction == expected

    def significant(self, alpha: float = 0.05) -> bool:
        """Holm-corrected significance. Never the raw p-value."""
        return (not math.isnan(self.p_holm)) and self.p_holm < alpha

    def render(self, alpha: float = 0.05) -> str:
        e = self.effects
        head = (
            f"{self.hypothesis.pair_label} on {self.hypothesis.measure.label} [{self.label.tag()}]"
        )
        effect_line = (
            f"    effect (rank-biserial) {e.rank_biserial.render()}"
            f"  |  dz {e.cohens_dz.point:+.3f} "
            f"[{e.cohens_dz.ci.low:+.3f}, {e.cohens_dz.ci.high:+.3f}]"
            f"  |  Hodges-Lehmann {e.hodges_lehmann.point:+.3f} points "
            f"[{e.hodges_lehmann.ci.low:+.3f}, {e.hodges_lehmann.ci.high:+.3f}]"
        )
        p_line = (
            f"    then p: Wilcoxon W = {self.test.statistic:.1f}, "
            f"p = {self.test.p_value:.4g}, Holm-adjusted p = {self.p_holm:.4g} "
            f"(family of {self.family_size}), {'' if self.significant(alpha) else 'not '}"
            f"significant at alpha = {alpha}"
        )
        direction = self.direction_as_registered
        if direction is None:
            dir_line = "    registered prediction: no difference"
        else:
            dir_line = (
                f"    registered direction {self.hypothesis.expected_direction}, observed "
                f"{self.observed_direction} — "
                f"{'as registered' if direction else 'AGAINST the registered direction'}"
            )
        counts = (
            f"    n = {self.n_scenarios} paired scenarios"
            f"{f' ({self.n_dropped} dropped for an incomplete pair)' if self.n_dropped else ''}"
            f", {self.test.n_nonzero} with a nonzero difference"
        )
        return "\n".join([head, effect_line, p_line, dir_line, counts])


def run_pairwise(
    long: pd.DataFrame,
    hypothesis: Hypothesis,
    *,
    rater_type: str | None = None,
    statuses: Mapping[str, EvidenceStatus] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    extra_reasons: Iterable[str] = (),
) -> PairwiseResult | None:
    """Effect sizes then the test, for one hypothesis. `None` if no pairs survive.

    `long` is a long-format score frame (see `carelite.stats.data`). Cell means
    are recomputed here rather than passed in, so the `to_quality` transform is
    applied on this call's own data and cannot be inherited from somewhere it
    was skipped.
    """
    cells = cell_means(long, hypothesis.measure)
    scope = _scope_for(long, rater_type)
    matrix = paired_matrix(cells, (hypothesis.left, hypothesis.right), rater_type=rater_type)
    available = cells["scenario_id"].nunique() if not cells.empty else 0
    if matrix.empty:
        return None

    left = matrix[str(hypothesis.left)].to_numpy(dtype=float)
    right = matrix[str(hypothesis.right)].to_numpy(dtype=float)

    return PairwiseResult(
        hypothesis=hypothesis,
        effects=paired_effects(left, right, n_boot=n_boot, seed=seed),
        test=wilcoxon_paired(left, right),
        label=label_for(
            hypothesis.measure,
            prespecified=hypothesis.prespecified,
            rater_scope=scope,
            statuses=statuses,
            extra_reasons=extra_reasons,
        ),
        n_scenarios=int(matrix.shape[0]),
        n_dropped=max(0, int(available) - int(matrix.shape[0])),
    )


def _scope_for(long: pd.DataFrame, rater_type: str | None) -> RaterScope:
    if rater_type is not None:
        return RaterScope.from_rater_types([rater_type])
    if "rater_type" in long.columns:
        return RaterScope.from_rater_types(long["rater_type"].dropna().unique())
    return RaterScope.MIXED


def friedman_across_conditions(
    long: pd.DataFrame,
    *,
    dimensions: Sequence[str] = RUBRIC_DIMENSIONS,
    conditions: Sequence[Condition] = FRIEDMAN_CONDITIONS,
    rater_type: str | None = None,
    statuses: Mapping[str, EvidenceStatus] | None = None,
) -> tuple[FriedmanResult, ...]:
    """The eleven pre-specified omnibus tests (§8.1), on `to_quality()` scores."""
    scope = _scope_for(long, rater_type)
    results: list[FriedmanResult] = []
    for key in dimensions:
        m = measure(key)
        matrix = paired_matrix(cell_means(long, m), conditions, rater_type=rater_type)
        statistic, p_value, df, n_blocks = friedman_omnibus(matrix)
        results.append(
            FriedmanResult(
                measure_key=key,
                conditions=tuple(str(c) for c in conditions),
                statistic=statistic,
                p_value=p_value,
                df=df,
                n_blocks=n_blocks,
                label=label_for(m, prespecified=True, rater_scope=scope, statuses=statuses),
            )
        )
    adjusted = holm_bonferroni([r.p_value for r in results])
    return tuple(
        replace(r, p_holm_within_omnibus=p) for r, p in zip(results, adjusted, strict=True)
    )


@dataclass(frozen=True, slots=True)
class FamilyResult:
    """One Holm-corrected family of pairwise tests, plus the omnibus tests.

    `name` and `correction_family` are printed on every render because the
    family is the thing a reader has to check, and a table that does not say
    what was corrected across cannot be checked.
    """

    name: str
    alpha: float
    results: tuple[PairwiseResult, ...]
    #: The size of the correction family, which is fixed by the analysis plan
    #: before the data exist. NOT `len(results)`: a pre-specified test that could
    #: not be computed still consumes its share of the correction, so the two
    #: numbers differ whenever a condition is missing from the data and the
    #: rendered `m` has to be this one.
    family_size: int = 0
    friedman: tuple[FriedmanResult, ...] = ()
    correction: str = "Holm-Bonferroni"
    correction_family: str = (
        "the whole set of pre-specified pairwise comparisons, across measures and dimensions "
        "together, not per dimension"
    )
    notes: tuple[str, ...] = ()

    @property
    def confirmatory(self) -> tuple[PairwiseResult, ...]:
        return tuple(r for r in self.results if r.label.is_confirmatory)

    @property
    def exploratory(self) -> tuple[PairwiseResult, ...]:
        return tuple(r for r in self.results if not r.label.is_confirmatory)

    def by_key(self, key: str) -> PairwiseResult | None:
        for r in self.results:
            if r.hypothesis.key == key:
                return r
        return None

    def render(self) -> str:
        lines = [
            f"{self.name.upper()}",
            f"  correction: {self.correction} across {self.correction_family}",
            f"  family size m = {self.family_size or len(self.results)} "
            f"({len(self.results)} computed), alpha = {self.alpha}",
            "  effect sizes and 95% bootstrap CIs are reported BEFORE p-values "
            "(pre-registration §8.2)",
        ]
        for note in self.notes:
            lines.append(f"  note: {note}")
        if self.friedman:
            lines.append("")
            lines.append(
                "  FRIEDMAN OMNIBUS across "
                f"{{{', '.join(self.friedman[0].conditions)}}} — not part of the Holm family; "
                "the within-omnibus adjustment is shown for reference only"
            )
            lines.append(
                f"    {'dimension':<14}{'chi2':>9}{'df':>4}{'n':>5}{'p':>10}"
                f"{'p(omnibus Holm)':>18}  status"
            )
            for f in self.friedman:
                lines.append(
                    f"    {f.measure_key:<14}{f.statistic:>9.3f}{f.df:>4}{f.n_blocks:>5}"
                    f"{f.p_value:>10.4g}{f.p_holm_within_omnibus:>18.4g}  {f.label.tag()}"
                )
        lines.append("")
        for r in self.results:
            lines.append(r.render(self.alpha))
            lines.append("")
        return "\n".join(lines).rstrip()


def run_family(
    long: pd.DataFrame,
    hypotheses: Sequence[Hypothesis] = CONFIRMATORY_FAMILY,
    *,
    name: str = "primary analysis (pre-registration §8.1)",
    alpha: float = 0.05,
    rater_type: str | None = None,
    statuses: Mapping[str, EvidenceStatus] | None = None,
    include_friedman: bool = True,
    friedman_conditions: Sequence[Condition] = FRIEDMAN_CONDITIONS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    extra_reasons: Iterable[str] = (),
    notes: Sequence[str] = (),
) -> FamilyResult:
    """Run a whole family and Holm-correct across all of it in one step.

    The correction is applied over `hypotheses` as given. The family size used
    for the correction is the number of hypotheses submitted, including any whose
    test came out undefined -- the family is fixed by the analysis plan before
    the data exist, so a test that could not run does not make its neighbours
    easier to pass.
    """
    computed: list[tuple[Hypothesis, PairwiseResult | None]] = []
    for h in hypotheses:
        computed.append(
            (
                h,
                run_pairwise(
                    long,
                    h,
                    rater_type=rater_type,
                    statuses=statuses,
                    n_boot=n_boot,
                    seed=seed,
                    extra_reasons=extra_reasons,
                ),
            )
        )

    family_size = len(computed)
    p_values = [(r.test.p_value if r is not None else math.nan) for _, r in computed]
    adjusted = holm_bonferroni(p_values, family_size=family_size)

    results: list[PairwiseResult] = []
    missing: list[str] = []
    for (h, r), p_adj in zip(computed, adjusted, strict=True):
        if r is None:
            missing.append(h.key)
            continue
        results.append(replace(r, p_holm=p_adj, family_size=family_size))

    all_notes = list(notes)
    if missing:
        all_notes.append(
            "no paired data for "
            + ", ".join(missing)
            + f"; these still count toward the family size (m = {family_size})"
        )

    friedman = (
        friedman_across_conditions(
            long,
            conditions=friedman_conditions,
            rater_type=rater_type,
            statuses=statuses,
        )
        if include_friedman
        else ()
    )

    return FamilyResult(
        name=name,
        alpha=alpha,
        results=tuple(results),
        family_size=family_size,
        friedman=friedman,
        notes=tuple(all_notes),
    )
