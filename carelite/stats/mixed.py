"""Variance decomposition: what belongs to the scenario, what to the condition.

Pre-registration §8.3: "With 3 samples per scenario-condition cell, a
mixed-effects model with a random intercept for scenario separates
within-scenario generation variance from between-condition effect, rather than
treating the 3 samples as independent observations."

**Why the alternative is wrong, and exactly how wrong.** The full run is 6
conditions x 60 scenarios x 3 samples = 1,080 generations. Those are not 1,080
independent observations: the three samples in a cell answer the same scenario
under the same condition, differing only by what the sampler did at temperature
0.7. `IndependenceCheck` measures the cost of pretending otherwise, per
contrast, by computing the paired standard error twice -- once over the n
scenario-level cell means the analysis plan specifies, and once over the 3n
sample-level pairs a naive analyst would use.

The ratio between them is **not** a fixed sqrt(3) penalty, and this module does
not claim it is. The arithmetic: with within-cell sampling variance
`sigma^2` and scenario-by-condition interaction variance `I`, the sample-level
paired SE is smaller than the honest one by `sqrt((2 sigma^2 + I) /
(2 sigma^2 + 3I))`. When `I` is zero -- when a cell's three samples really do
differ only by sampling noise -- the two agree, and the naive analysis happens
to be right for the wrong reason. When cells genuinely differ beyond sampling
noise, which is what a scenario-by-condition interaction *is*, the naive SE
understates and every p-value comes out too small. Simulated at the study's
design, the understatement runs about 1.3x at an interaction SD of 0.3 rubric
points and about 1.6x at 0.6. So the ratio is a measurement of how much
scenario-by-condition structure the data actually carry, reported as such,
rather than a rhetorical multiplier.

A pooled OLS fit that simply ignores the scenario grouping is a *different*
mistake and moves the other way: condition varies within scenario here, so
dropping the random intercept dumps between-scenario variance into the residual
and makes the contrast's SE too large. That fit is not reported, because a
conservative wrong answer is still a wrong answer and putting it in a table
invites someone to quote it.

**Two decompositions, deliberately both.**

`fit_random_intercept` is the model the plan names: `value ~ condition` with a
random intercept for scenario, fitted by REML. It gives the between-condition
fixed effects with standard errors that account for scenarios being repeated
across conditions, plus the scenario variance component.

`variance_components_moments` is a closed-form one-way random-effects
decomposition by the method of moments. It is not a substitute for the model --
it ignores the condition structure -- but it is analytically computable by hand,
which means it can be tested against a known answer rather than against
itself. It is the check that the variance numbers coming out of the fitted model
are the right order of magnitude, and it runs when the optimiser does not
converge.

**Reverse coding.** The design matrix is built from
`carelite.stats.measures.measure_by_generation`, which reads the `quality`
column, so `ritualistic` enters the fit as `6 - raw` like everywhere else. There
is no path in this module that touches a raw score.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from carelite.eval.judge.validation import EvidenceStatus
from carelite.stats.evidence import Label, RaterScope, label_for
from carelite.stats.measures import Measure, measure_by_generation
from carelite.types import Condition

__all__ = [
    "FixedEffect",
    "IndependenceCheck",
    "MixedModelResult",
    "VarianceComponents",
    "fit_random_intercept",
    "independence_check",
    "variance_components_moments",
    "within_cell_variance",
]


@dataclass(frozen=True, slots=True)
class VarianceComponents:
    """One-way random-effects decomposition by the method of moments.

    `between` is the variance of the group means beyond what sampling noise
    explains; it can come out negative when the groups are more alike than
    chance would predict, and that negative value is reported rather than
    clamped to zero -- a negative variance estimate is information about the
    design, and silently flooring it turns "the grouping explains nothing" into
    "the grouping explains exactly nothing", which are different statements.
    `icc` is clamped to [0, 1] because a negative ICC has no reading at all.
    """

    n_groups: int
    n_observations: int
    between: float
    within: float
    icc: float
    balanced: bool

    @property
    def total(self) -> float:
        return self.between + self.within


def variance_components_moments(
    values: Sequence[float] | np.ndarray,
    groups: Sequence[object],
) -> VarianceComponents:
    """Between-group and within-group variance for a one-way random-effects layout.

    Balanced case (every group the same size n):

        MSB = n * sum_i (mean_i - grand_mean)^2 / (k - 1)
        MSW = sum_ij (y_ij - mean_i)^2 / (N - k)
        within  = MSW
        between = (MSB - MSW) / n

    The unbalanced case replaces `n` with the usual effective group size
    `n0 = (N - sum_i n_i^2 / N) / (k - 1)`, which reduces to `n` when the design
    is balanced -- so one code path serves both and the balanced hand-computed
    test exercises the same arithmetic the real, slightly-ragged data will use.
    """
    y = np.asarray(values, dtype=float)
    g = pd.Series(list(groups))
    if y.size != g.size:
        raise ValueError("values and groups must be the same length")
    frame = pd.DataFrame({"y": y, "g": g.to_numpy()}).dropna(subset=["y"])
    if frame.empty:
        return VarianceComponents(0, 0, math.nan, math.nan, math.nan, True)

    sizes = frame.groupby("g", observed=True)["y"].size()
    k = int(sizes.size)
    n_total = int(frame.shape[0])
    if k < 2 or n_total <= k:
        return VarianceComponents(k, n_total, math.nan, math.nan, math.nan, True)

    means = frame.groupby("g", observed=True)["y"].mean()
    grand = float(frame["y"].mean())
    ss_between = float((sizes * (means - grand) ** 2).sum())
    ss_within = float(
        frame.groupby("g", observed=True)["y"].transform(lambda s: s - s.mean()).pow(2).sum()
    )
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n_total - k)

    n0 = (n_total - float((sizes**2).sum()) / n_total) / (k - 1)
    between = (ms_between - ms_within) / n0 if n0 > 0 else math.nan
    total = between + ms_within
    icc = 0.0 if total <= 0 else min(1.0, max(0.0, between / total))
    return VarianceComponents(
        n_groups=k,
        n_observations=n_total,
        between=between,
        within=ms_within,
        icc=icc,
        balanced=bool(sizes.nunique() == 1),
    )


def within_cell_variance(long: pd.DataFrame, m: Measure) -> float:
    """Mean variance across the samples inside a scenario x condition x rater cell.

    This is generation variance in its purest available form: everything about
    the prompt, the scenario and the condition is held fixed, so what is left is
    what the sampler contributed. Cells with fewer than two samples contribute
    nothing.
    """
    per_generation = measure_by_generation(long, m)
    if per_generation.empty:
        return math.nan
    grouped = per_generation.groupby(
        ["scenario_id", "condition", "rater_type"], dropna=False, observed=True
    )["value"]
    variances = grouped.var(ddof=1).dropna()
    return float(variances.mean()) if not variances.empty else math.nan


@dataclass(frozen=True, slots=True)
class FixedEffect:
    """One condition contrast against the reference condition."""

    term: str
    coefficient: float
    std_error: float
    z_value: float
    p_value: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class IndependenceCheck:
    """What treating the 3 samples in a cell as independent would cost, per contrast.

    `se_cell_level` is the paired standard error over the n scenario-level cell
    means the analysis plan specifies (§3). `se_sample_level` is the same
    quantity computed over the 3n sample-level pairs, which is what an analysis
    that treated every generation as its own observation would use.

    `understatement` is `se_cell_level / se_sample_level`. Above 1 means the
    naive analysis understates the standard error -- p-values too small,
    intervals too narrow -- and how far above 1 measures the scenario-by-condition
    structure in the data. At or below 1 means a cell's samples differ by
    sampling noise alone, where the naive analysis is accidentally right; it is
    still not the analysis that was planned, because whether that holds is
    only knowable after the fact.
    """

    pair: str
    n_scenarios: int
    n_sample_pairs: int
    se_cell_level: float
    se_sample_level: float

    @property
    def understatement(self) -> float:
        if math.isnan(self.se_sample_level) or self.se_sample_level == 0:
            return math.nan
        return self.se_cell_level / self.se_sample_level


def independence_check(
    long: pd.DataFrame,
    m: Measure,
    left: Condition | str,
    right: Condition | str,
    *,
    rater_type: str | None = None,
) -> IndependenceCheck | None:
    """Paired SE at the scenario level against the same SE at the sample level.

    Sample-level pairs are matched on `sample_idx`, which is what a naive
    analysis would do -- the samples inside a cell are exchangeable, so there is
    no better pairing available, and the arbitrariness of the choice is part of
    why the sample-level analysis is not the planned one.
    """
    per_generation = measure_by_generation(long, m)
    if per_generation.empty or "sample_idx" not in per_generation.columns:
        return None
    if rater_type is not None:
        per_generation = per_generation[per_generation["rater_type"].astype(str) == str(rater_type)]
    frame = per_generation.copy()
    frame["condition"] = frame["condition"].astype(str)
    a, b = str(left), str(right)
    frame = frame[frame["condition"].isin([a, b])]
    if frame.empty:
        return None

    cells = frame.groupby(["scenario_id", "condition"], observed=True)["value"].mean().unstack()
    if a not in cells.columns or b not in cells.columns:
        return None
    cell_diff = (cells[a] - cells[b]).dropna().to_numpy(dtype=float)

    samples = (
        frame.groupby(["scenario_id", "sample_idx", "condition"], observed=True)["value"]
        .mean()
        .unstack()
    )
    if a not in samples.columns or b not in samples.columns:
        return None
    sample_diff = (samples[a] - samples[b]).dropna().to_numpy(dtype=float)

    def _se(values: np.ndarray) -> float:
        if values.size < 2:
            return math.nan
        return float(np.std(values, ddof=1) / math.sqrt(values.size))

    return IndependenceCheck(
        pair=f"{a} vs {b}",
        n_scenarios=int(cell_diff.size),
        n_sample_pairs=int(sample_diff.size),
        se_cell_level=_se(cell_diff),
        se_sample_level=_se(sample_diff),
    )


@dataclass(frozen=True, slots=True)
class MixedModelResult:
    """The §8.3 model: fixed condition effects, random intercept for scenario."""

    measure_key: str
    reference_condition: str
    conditions: tuple[str, ...]
    n_observations: int
    n_scenarios: int
    samples_per_cell: float
    converged: bool
    scenario_variance: float
    residual_variance: float
    icc: float
    within_cell_variance: float
    moments: VarianceComponents
    effects: tuple[FixedEffect, ...]
    label: Label
    independence: tuple[IndependenceCheck, ...] = ()
    method: str = "REML, random intercept for scenario (statsmodels MixedLM)"

    def render(self) -> str:
        lines = [
            f"MIXED-EFFECTS MODEL — {self.measure_key} [{self.label.tag()}]",
            f"  {self.method}",
            f"  value ~ condition + (1 | scenario), reference = {self.reference_condition}",
            f"  {self.n_observations} generations, {self.n_scenarios} scenarios, "
            f"{self.samples_per_cell:.1f} samples per cell"
            f"{'' if self.converged else '  [OPTIMISER DID NOT CONVERGE]'}",
            "",
            "  VARIANCE",
            f"    between scenarios      {self.scenario_variance:9.4f}",
            f"    residual               {self.residual_variance:9.4f}",
            f"    ICC (scenario)         {self.icc:9.4f}",
            f"    mean within-cell var   {self.within_cell_variance:9.4f}"
            "   (the 3 samples, everything else held fixed)",
            f"    moments cross-check    between {self.moments.between:.4f}, "
            f"within {self.moments.within:.4f}, ICC {self.moments.icc:.4f}",
            "",
            "  FIXED EFFECTS (contrast vs reference; effect and CI before p)",
            f"    {'term':<22}{'coef':>9}{'95% CI':>22}{'SE':>9}{'p':>10}",
        ]
        for e in self.effects:
            lines.append(
                f"    {e.term:<22}{e.coefficient:>+9.4f}"
                f"{f'[{e.ci_low:+.4f}, {e.ci_high:+.4f}]':>22}"
                f"{e.std_error:>9.4f}{e.p_value:>10.4g}"
            )
        if self.independence:
            lines.append("")
            lines.append("  WHAT TREATING THE 3 SAMPLES AS INDEPENDENT WOULD COST (§8.3)")
            lines.append(
                f"    {'contrast':<14}{'n scenarios':>13}{'n sample pairs':>16}"
                f"{'SE (cells)':>12}{'SE (samples)':>14}{'understated by':>16}"
            )
            for check in self.independence:
                lines.append(
                    f"    {check.pair:<14}{check.n_scenarios:>13}{check.n_sample_pairs:>16}"
                    f"{check.se_cell_level:>12.4f}{check.se_sample_level:>14.4f}"
                    f"{check.understatement:>15.2f}x"
                )
            lines.append(
                "    Above 1.00x, the sample-level analysis understates the standard error and "
                "every\n    p-value it produces is too small. At 1.00x a cell's samples differ "
                "by sampling noise\n    alone — which is only knowable after the fact, and is "
                "not the planned analysis either."
            )
        return "\n".join(lines)


def fit_random_intercept(
    long: pd.DataFrame,
    m: Measure,
    *,
    conditions: Sequence[Condition | str] | None = None,
    reference: Condition | str = Condition.A,
    rater_type: str | None = None,
    statuses: Mapping[str, EvidenceStatus] | None = None,
) -> MixedModelResult | None:
    """Fit `value ~ condition + (1 | scenario)` on per-generation quality scores.

    Returns `None` when there is not enough data to fit -- fewer than two
    conditions, fewer than two scenarios, or no rows at all. That is a real
    state (it is the current state of the database) and not an error.

    The rows are per *generation*, so the three samples in a cell are three rows
    sharing a scenario, and the random intercept is what stops them being
    counted as three independent scenarios.
    """
    import statsmodels.formula.api as smf

    per_generation = measure_by_generation(long, m)
    if per_generation.empty:
        return None
    if rater_type is not None:
        per_generation = per_generation[per_generation["rater_type"].astype(str) == str(rater_type)]
    per_generation = per_generation.copy()
    per_generation["condition"] = per_generation["condition"].astype(str)
    if conditions is not None:
        wanted = [str(c) for c in conditions]
        per_generation = per_generation[per_generation["condition"].isin(wanted)]
    if per_generation.empty:
        return None

    present = sorted(per_generation["condition"].unique())
    ref = str(reference)
    if ref not in present:
        ref = present[0]
    if len(present) < 2 or per_generation["scenario_id"].nunique() < 2:
        return None

    scope = RaterScope.from_rater_types(per_generation["rater_type"].dropna().unique())
    formula = f'value ~ C(condition, Treatment(reference="{ref}"))'

    converged = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, per_generation, groups=per_generation["scenario_id"])
        try:
            fit = model.fit(reml=True, method="lbfgs")
            converged = bool(getattr(fit, "converged", True))
        except Exception:
            return None

    ci = fit.conf_int()
    scenario_variance = float(fit.cov_re.iloc[0, 0]) if fit.cov_re.size else math.nan
    residual_variance = float(fit.scale)
    denominator = scenario_variance + residual_variance
    icc = scenario_variance / denominator if denominator > 0 else math.nan

    effects: list[FixedEffect] = []
    for term in fit.params.index:
        if term in ("Intercept", "Group Var") or term.startswith("Group"):
            continue
        pretty = _pretty_term(term, ref)
        effects.append(
            FixedEffect(
                term=pretty,
                coefficient=float(fit.params[term]),
                std_error=float(fit.bse[term]),
                z_value=float(fit.tvalues[term]),
                p_value=float(fit.pvalues[term]),
                ci_low=float(ci.loc[term, 0]),
                ci_high=float(ci.loc[term, 1]),
            )
        )

    cells = per_generation.groupby(["scenario_id", "condition"], dropna=False, observed=True)[
        "value"
    ].size()

    return MixedModelResult(
        measure_key=m.key,
        reference_condition=ref,
        conditions=tuple(present),
        n_observations=int(per_generation.shape[0]),
        n_scenarios=int(per_generation["scenario_id"].nunique()),
        samples_per_cell=float(cells.mean()) if not cells.empty else math.nan,
        converged=converged,
        scenario_variance=scenario_variance,
        residual_variance=residual_variance,
        icc=icc,
        within_cell_variance=within_cell_variance(long, m),
        moments=variance_components_moments(
            per_generation["value"].to_numpy(), per_generation["scenario_id"].tolist()
        ),
        effects=tuple(effects),
        label=label_for(m, prespecified=True, rater_scope=scope, statuses=statuses),
        independence=tuple(
            check
            for other in present
            if other != ref
            for check in [independence_check(long, m, other, ref, rater_type=rater_type)]
            if check is not None
        ),
    )


def _pretty_term(term: str, reference: str) -> str:
    """`C(condition, Treatment(reference="A"))[T.B]` -> `B vs A`."""
    if "[T." in term:
        level = term.split("[T.", 1)[1].rstrip("]")
        return f"{level} vs {reference}"
    return term
