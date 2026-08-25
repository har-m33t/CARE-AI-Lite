"""The power analysis that justifies n = 60, and what n = 60 can actually detect.

Build plan v3 §11 and analysis plan §6 fix the design: paired, Wilcoxon
signed-rank, alpha = 0.05 two-sided, power = 0.80. The planned table is

    large  (d ~ 0.8)  ->  ~15-20 scenarios
    medium (d ~ 0.5)  ->  ~35-45
    small  (d ~ 0.3)  ->  ~90+

and `required_n` reproduces it, which is the point of testing this module: the
n the plan rests on has to come out of a function someone can re-run, not out
of a remembered rule of thumb.

**Method.** There is no closed form for Wilcoxon signed-rank power without
assuming a distribution for the differences, so this uses the standard
route: the normal-theory paired *t* sample size, then divide by the asymptotic
relative efficiency of the signed-rank test against the *t* test. Under
normally-distributed differences that ARE is 3/pi ~ 0.955, which is the
conservative choice — the signed-rank test is *more* efficient than the t test
for every heavier-tailed distribution, with no upper bound, so a plan powered
under the normal assumption is not under-powered if the differences turn out
non-normal. That is also the assumption under which the planned table
was produced, and it reproduces it.

    n_t  = ((z_{1-alpha/2} + z_{1-beta}) / d)^2 + z_{1-alpha/2}^2 / 2
    n_w  = n_t / ARE

The second term of `n_t` is the usual correction for estimating the standard
deviation rather than knowing it; without it the table comes out one bracket
low at every effect size.

**The honest part.** n was not set by the comparison the study cares about. It
was set by secondary outcome 2 — Condition B vs Condition C — which build plan
v3 §11 expects to show the *smallest* effect of any comparison in the study.
The primary outcome (A vs B) is expected to be large and is heavily
over-powered at this n; B vs C is the one the sample size is actually sized
for, and if the true B-vs-C effect is smaller than `detectable_effect(60)`, a
null result there is a statement about this study's resolution and not about
retrieval. `PowerReport.render()` says so every time it prints.

**One caveat this module reports and does not silently apply.** These numbers
are per-test at alpha = 0.05. The analysis plan (§8.1) corrects a family of
planned tests with Holm-Bonferroni, so the effective alpha for the
smallest p-value in the family is alpha/m. `detectable_effect` takes an alpha,
so the family-corrected figure is available and is printed alongside the
nominal one — the planned n stands on the nominal figure, and this is
reported as context rather than as a revision to it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from scipy import stats

__all__ = [
    "ALPHA",
    "ARE_WILCOXON_VS_T",
    "POWER",
    "PRESPECIFIED_N_HOLDOUT",
    "PowerReport",
    "PowerRow",
    "build_power_report",
    "detectable_effect",
    "required_n",
]

#: Pre-registration §6 / build plan v3 §11.
ALPHA = 0.05
POWER = 0.80

#: Asymptotic relative efficiency of the Wilcoxon signed-rank test against the
#: paired t test, for normally distributed differences: 3/pi.
ARE_WILCOXON_VS_T = 3.0 / math.pi

#: `carelite.config.Experiment.n_scenarios_holdout`. Restated as a default
#: argument only; the config value is the authority and is what
#: `build_power_report` reads.
PRESPECIFIED_N_HOLDOUT = 60


def _z(p: float) -> float:
    return float(stats.norm.ppf(p))


def required_n(
    effect_size: float,
    *,
    alpha: float = ALPHA,
    power: float = POWER,
    two_sided: bool = True,
    are: float = ARE_WILCOXON_VS_T,
) -> int:
    """Scenarios needed to detect a paired effect of `effect_size` (Cohen's dz).

    Returns the smallest integer n satisfying the normal approximation above.
    Raises on a non-positive effect size, which is the shape of a caller that
    has passed a difference in raw score points rather than a standardised one.
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be positive; it is a standardised paired effect (dz)")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be strictly between 0 and 1")
    z_alpha = _z(1 - alpha / 2) if two_sided else _z(1 - alpha)
    z_beta = _z(power)
    n_t = ((z_alpha + z_beta) / effect_size) ** 2 + z_alpha**2 / 2
    return math.ceil(n_t / are)


def detectable_effect(
    n: int,
    *,
    alpha: float = ALPHA,
    power: float = POWER,
    two_sided: bool = True,
    are: float = ARE_WILCOXON_VS_T,
) -> float:
    """Smallest paired effect (Cohen's dz) detectable at `n` scenarios.

    The inverse of `required_n`, solved for d. `nan` when n is too small to
    support the design at all (the corrected denominator goes non-positive),
    which is a real answer and not a failure: at that n no effect is detectable
    at the requested alpha and power.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    z_alpha = _z(1 - alpha / 2) if two_sided else _z(1 - alpha)
    z_beta = _z(power)
    denominator = n * are - z_alpha**2 / 2
    if denominator <= 0:
        return math.nan
    return float((z_alpha + z_beta) / math.sqrt(denominator))


@dataclass(frozen=True, slots=True)
class PowerRow:
    """One row of the planned table, recomputed."""

    label: str
    effect_size: float
    n_required: int


@dataclass(frozen=True, slots=True)
class PowerReport:
    """The §6 justification, recomputed from the constants the plan fixed."""

    alpha: float
    power: float
    two_sided: bool
    are: float
    n_holdout: int
    samples_per_cell: int
    rows: tuple[PowerRow, ...]
    detectable_at_n: float
    #: Detectable effect if the whole pre-specified family's Holm correction is
    #: charged against a single test (alpha / m). Context, not the planned n.
    family_size: int | None
    detectable_at_n_family_corrected: float | None
    #: The comparison that set n, per v3 §11 and analysis plan §6.
    sizing_comparison: str = "secondary outcome 2, Condition B vs Condition C"

    def render(self) -> str:
        lines = [
            "POWER ANALYSIS (build plan v3 §11, analysis plan §6)",
            f"  paired design, Wilcoxon signed-rank, alpha = {self.alpha} "
            f"({'two' if self.two_sided else 'one'}-sided), power = {self.power}",
            f"  ARE(Wilcoxon vs paired t, normal differences) = {self.are:.4f}",
            "",
            f"  {'effect':<24}{'dz':>6}{'scenarios needed':>20}",
        ]
        for row in self.rows:
            lines.append(f"  {row.label:<24}{row.effect_size:>6.2f}{row.n_required:>20}")
        lines.append("")
        lines.append(
            f"  n = {self.n_holdout} held-out scenarios x {self.samples_per_cell} samples per cell."
        )
        lines.append(
            f"  Smallest detectable paired effect at n = {self.n_holdout}: "
            f"dz = {self.detectable_at_n:.3f}."
        )
        if self.family_size and self.detectable_at_n_family_corrected is not None:
            lines.append(
                f"  Charging the whole Holm family (m = {self.family_size}) against one test "
                f"(alpha = {self.alpha}/{self.family_size}) it is "
                f"dz = {self.detectable_at_n_family_corrected:.3f}. The planned n stands on "
                "the nominal figure above; this is context for reading a null result, not a "
                "revision of the plan."
            )
        lines.append("")
        lines.append(
            f"  n was set by {self.sizing_comparison} — the comparison expected to show the "
            "SMALLEST effect, not the primary outcome. The primary outcome (A vs B, expected "
            "large) is over-powered at this n. A null B-vs-C result at an effect below "
            f"dz = {self.detectable_at_n:.3f} is a statement about this study's resolution, "
            "not about retrieval."
        )
        return "\n".join(lines)


#: The three brackets the analysis plan tabulates, as (label, dz).
_PLANNED_BRACKETS: tuple[tuple[str, float], ...] = (
    ("large", 0.8),
    ("medium", 0.5),
    ("small", 0.3),
)


def build_power_report(
    *,
    alpha: float = ALPHA,
    power: float = POWER,
    two_sided: bool = True,
    are: float = ARE_WILCOXON_VS_T,
    n_holdout: int | None = None,
    samples_per_cell: int | None = None,
    family_size: int | None = None,
    brackets: Sequence[tuple[str, float]] = _PLANNED_BRACKETS,
) -> PowerReport:
    """Recompute the planned table and the detectable effect at the frozen n.

    `n_holdout` and `samples_per_cell` default to `carelite.config`, so the
    report tracks the frozen contract rather than a number copied out of it.
    """
    from carelite.config import get_settings

    experiment = get_settings().experiment
    n = experiment.n_scenarios_holdout if n_holdout is None else n_holdout
    per_cell = experiment.samples_per_cell if samples_per_cell is None else samples_per_cell

    rows = tuple(
        PowerRow(
            label=label,
            effect_size=d,
            n_required=required_n(d, alpha=alpha, power=power, two_sided=two_sided, are=are),
        )
        for label, d in brackets
    )
    corrected = (
        detectable_effect(n, alpha=alpha / family_size, power=power, two_sided=two_sided, are=are)
        if family_size
        else None
    )
    return PowerReport(
        alpha=alpha,
        power=power,
        two_sided=two_sided,
        are=are,
        n_holdout=n,
        samples_per_cell=per_cell,
        rows=rows,
        detectable_at_n=detectable_effect(
            n, alpha=alpha, power=power, two_sided=two_sided, are=are
        ),
        family_size=family_size,
        detectable_at_n_family_corrected=corrected,
    )
