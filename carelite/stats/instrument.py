"""Whether a dimension carries enough variance for a comparison on it to mean anything.

**This module exists because of what the holdout judging produced, not because
the analysis plan asked for it. It is a post-hoc instrument diagnostic and every
result it emits is stamped `prespecified = False`.** Saying that up front is the
point: a check invented after seeing the data is weaker evidence than one fixed
before, and the honest response is to label it, not to pretend the plan
anticipated it.

==========================================================================
THE PROBLEM, WHICH IS NOT A NULL RESULT
==========================================================================
A paired test compares ranks of within-scenario differences. If a judge gives
almost every response the same score on a dimension, almost every difference is
zero, and the test has nothing to rank. It will return a large p-value. That
p-value looks exactly like "the conditions do not differ on this dimension" and
means something completely different: **the instrument did not resolve this
dimension, so the comparison was never run in any meaningful sense.**

Reporting the first sentence when the second is true is the specific error this
module exists to prevent. So a comparison whose measure rests on a degenerate
dimension is not reported as non-significant. It is reported as **NOT TESTABLE
WITH THIS INSTRUMENT**, in the result object, above the p-value.

**Why this cannot be fixed with more scenarios.** The `carelite-judge` lane
established the mechanism independently: ordinal Krippendorff's alpha is bounded
above by the variance the judge itself produces (r = 0.878 between a dimension's
score variance and its achievable alpha). A judge that emits one value cannot
disagree with itself, cannot agree with anything either, and cannot separate two
conditions. That is a property of the measurement, and n does not enter it. A
larger holdout would produce the same floor with narrower confidence intervals
around it.

==========================================================================
THE CLASSIFICATION RULE, AND THE GUARD ON IT
==========================================================================
A dimension is **degenerate** when, on the analysed (`to_quality`) scale, **the
standard deviation of its scores is below `MIN_SD` (0.75) rubric points.**

One criterion, not a combination, and specifically *not* the share of scores
sitting on the modal value.

**Modal share was the first rule tried and it was wrong, which is worth
recording because it is the intuitive rule.** On this run `name` puts 77% of its
mass on a single value — and the other 23% is almost all at the opposite end of
the scale (683 scores of 1, 150 scores of 5). That is a dimension discriminating
about as hard as a five-point scale can: it says most responses do not name the
emotion and a substantial minority do it well. A modal-share cut at 0.75 called
it degenerate, which is plainly false. Concentration is not the failure;
*absence of spread* is, and the standard deviation is what measures that. Modal
share is still computed and printed, because it describes the shape of a
distribution usefully — it is just not the criterion.

**Where the cut sits, and why it is not doing much work.** Ordered by standard
deviation, this run's eleven dimensions fall into two groups with a wide empty
band between them: 0.16, 0.49, 0.59, then nothing until 1.00, 1.15, 1.41, 1.53,
1.57, 1.61, 1.75, 1.85. Any cut between 0.6 and 1.0 produces the same three
degenerate dimensions. 0.75 is the middle of that gap rather than a tuned value.

**A threshold chosen after seeing the data can be tuned until it produces the
answer expected of it.** That hazard is real here and it is the same one D3
recorded about the aspiration filter, so it gets the same guard rather than an
assurance. `threshold_sensitivity()` recomputes the classification across a
range of cuts and reports whether the set of degenerate dimensions changes. If
the answer is stable across the range, the exact number is not doing the work
and the reader can see that. If it is not stable, the classification is a
judgement call and the output says so. **That guard is not decorative: it is
what caught the modal-share rule above.** The rule is applied to all eleven
dimensions with their numbers printed, so anyone who prefers a different line
can draw it without re-running anything.

==========================================================================
WHAT A DEGENERATE DIMENSION DOES TO A COMPOSITE
==========================================================================
Weakest link demotes; it does not veto. A composite whose constituents are *all*
degenerate is untestable. A composite with *some* degenerate constituents is
still testable -- the surviving dimensions carry the signal -- but its effect
size is attenuated toward zero by the constant ones, which is a real distortion
of magnitude and is reported as `attenuated_by`. That is the opposite of the
weakest-link rule in `carelite.stats.evidence`, and deliberately so: that rule
governs what a result may be *called*, this one governs whether it *exists*, and
a composite that still has four discriminating dimensions in it plainly exists.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from carelite.stats.measures import Measure, attach_quality, measure
from carelite.types import RUBRIC_DIMENSIONS

__all__ = [
    "FLOOR_MECHANISM_NOTE",
    "HIGH_MODAL_SHARE",
    "MIN_SCORED_FOR_CLASSIFICATION",
    "MIN_SD",
    "RITUAL_MECHANISM_DIMENSIONS",
    "DimensionDistribution",
    "Discrimination",
    "InstrumentReport",
    "MeasureTestability",
    "classify",
    "describe_dimensions",
    "instrument_report",
    "measure_testability",
    "threshold_sensitivity",
]

#: Below this many rubric points of spread there is not enough variation for a
#: signed-rank test to rank. Sits in the middle of the empty band between this
#: run's two groups of dimensions (0.59 and 1.00) -- see the module docstring.
MIN_SD = 0.75

#: Reported, never used as a criterion. Kept as a named constant only so the
#: rendered table can say what "high" means when describing a distribution's
#: shape. The module docstring records why concentration is not degeneracy.
HIGH_MODAL_SHARE = 0.75

#: Fewer scored items than this and the distribution is not being measured, it
#: is being guessed at. Classified `UNKNOWN` rather than either verdict.
MIN_SCORED_FOR_CLASSIFICATION = 30

#: The judge lane's finding, carried here because it is the reason a bigger
#: sample does not repair a degenerate dimension.
FLOOR_MECHANISM_NOTE = (
    "A degenerate dimension is not fixable with a larger sample. The judge lane measured "
    "r = 0.878 between a dimension's score variance and the ordinal Krippendorff's alpha "
    "achievable on it: alpha is bounded above by the variance the judge produces. A judge that "
    "emits one value on a dimension cannot disagree with itself, cannot agree with a human, and "
    "cannot separate two conditions. More scenarios would narrow the intervals around the same "
    "floor."
)

#: Build plan v3 predicts B loses to A on `naturalness` *because* framework
#: prompting induces ritual -- so the prediction is carried jointly by the
#: outcome (`naturalness`) and its stated mechanism (`ritualistic`). If both are
#: degenerate the prediction is untestable rather than unsupported.
RITUAL_MECHANISM_DIMENSIONS: tuple[str, ...] = ("naturalness", "ritualistic")


class Discrimination(StrEnum):
    """Whether a dimension resolved anything on this run."""

    DISCRIMINATING = "discriminating"
    DEGENERATE = "degenerate"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DimensionDistribution:
    """What one dimension's scores actually looked like, on the analysed scale.

    `mean_quality` is on the `to_quality()` scale the tests run on;
    `mean_raw` is the database scale, so a reader checking against a hand-run
    SQL query sees the number they expect. For `ritualistic` the two differ by
    the reversal and that is exactly the confusion this pair of fields exists to
    prevent.
    """

    dimension: str
    n_scored: int
    n_missing: int
    distinct: int
    mean_quality: float
    mean_raw: float
    sd: float
    modal_value_quality: int | None
    modal_share: float
    counts_quality: Mapping[int, int] = field(default_factory=dict)

    @property
    def discrimination(self) -> Discrimination:
        if self.n_scored < MIN_SCORED_FOR_CLASSIFICATION:
            return Discrimination.UNKNOWN
        if self.sd < MIN_SD:
            return Discrimination.DEGENERATE
        return Discrimination.DISCRIMINATING

    @property
    def is_degenerate(self) -> bool:
        return self.discrimination is Discrimination.DEGENERATE

    @property
    def why(self) -> str:
        """The clause that goes in the sentence reporting this dimension."""
        if self.discrimination is Discrimination.UNKNOWN:
            return f"only {self.n_scored} scored items; not classified"
        if self.is_degenerate:
            return (
                f"sd {self.sd:.2f} < {MIN_SD}; {self.modal_share:.1%} of scores sit on the "
                f"single value {self.modal_value_quality} and only {self.distinct} of 5 scale "
                "points were ever used"
            )
        shape = ""
        if self.modal_share > HIGH_MODAL_SHARE:
            shape = (
                f" — concentrated ({self.modal_share:.1%} on {self.modal_value_quality}) but "
                "spread across the scale, which is discrimination, not a floor"
            )
        return f"sd {self.sd:.2f}, {self.distinct} values used{shape}"

    def render(self) -> str:
        counts = " ".join(f"{k}:{v}" for k, v in sorted(self.counts_quality.items()))
        return (
            f"    {self.dimension:<12} n={self.n_scored:<4} miss={self.n_missing:<3} "
            f"distinct={self.distinct}  mean(q)={self.mean_quality:.2f}  sd={self.sd:.2f}  "
            f"modal={self.modal_share:>6.1%}  {self.discrimination.value.upper():<15} [{counts}]"
        )


def describe_dimensions(
    long: pd.DataFrame,
    *,
    dimensions: Sequence[str] = RUBRIC_DIMENSIONS,
) -> tuple[DimensionDistribution, ...]:
    """Per-dimension distribution of the scores the analysis actually ran on.

    Computed from the same long frame every test reads, after `attach_quality`,
    so a dimension described as degenerate here is degenerate in the numbers the
    Wilcoxon saw -- not in a differently-filtered copy of the table.
    """
    if long.empty:
        return tuple(
            DimensionDistribution(d, 0, 0, 0, math.nan, math.nan, math.nan, None, math.nan)
            for d in dimensions
        )
    scored = attach_quality(long)
    out: list[DimensionDistribution] = []
    for dim in dimensions:
        subset = scored[scored["dimension"] == dim]
        quality = subset["quality"].dropna()
        raw = pd.to_numeric(subset["raw"], errors="coerce").dropna()
        n_scored = int(quality.size)
        n_missing = int(subset.shape[0] - n_scored)
        if n_scored == 0:
            out.append(
                DimensionDistribution(
                    dim, 0, n_missing, 0, math.nan, math.nan, math.nan, None, math.nan
                )
            )
            continue
        counts = quality.astype(int).value_counts()
        modal_value = int(counts.idxmax())
        out.append(
            DimensionDistribution(
                dimension=dim,
                n_scored=n_scored,
                n_missing=n_missing,
                distinct=int(counts.size),
                mean_quality=float(quality.mean()),
                mean_raw=float(raw.mean()) if raw.size else math.nan,
                sd=float(quality.std(ddof=1)) if n_scored > 1 else 0.0,
                modal_value_quality=modal_value,
                modal_share=float(counts.max() / n_scored),
                counts_quality={int(k): int(v) for k, v in counts.items()},
            )
        )
    return tuple(out)


def classify(
    distributions: Iterable[DimensionDistribution],
) -> dict[str, Discrimination]:
    """`dimension -> Discrimination`, from the rule in the module docstring."""
    return {d.dimension: d.discrimination for d in distributions}


def threshold_sensitivity(
    distributions: Sequence[DimensionDistribution],
    *,
    sd_cuts: Sequence[float] = (0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.95),
) -> tuple[bool, list[str]]:
    """Is the degenerate set stable across a range of cuts, or a judgement call?

    Returns `(stable, lines)`. Stability means every cut in `sd_cuts` produces
    the same set of degenerate dimensions as `MIN_SD` does. The guard against a
    threshold tuned to produce a wanted answer is that the answer be visibly
    insensitive to the threshold -- and where it is not, that the output say so
    rather than quoting the default as if it were a fact.

    The range spans the empty band this run's dimensions leave between the two
    groups. A cut outside that band would of course move the answer; the claim
    being checked is that no cut *inside* it does.
    """
    usable = [d for d in distributions if d.n_scored >= MIN_SCORED_FOR_CLASSIFICATION]
    baseline = {d.dimension for d in usable if d.is_degenerate}
    lines: list[str] = []
    stable = True
    for sd_cut in sd_cuts:
        at_cut = {d.dimension for d in usable if d.sd < sd_cut}
        if at_cut != baseline:
            stable = False
            added = sorted(at_cut - baseline)
            removed = sorted(baseline - at_cut)
            change = []
            if added:
                change.append(f"+{', '.join(added)}")
            if removed:
                change.append(f"-{', '.join(removed)}")
            lines.append(f"    sd < {sd_cut:.2f}: {' '.join(change)}")
    return stable, lines


@dataclass(frozen=True, slots=True)
class MeasureTestability:
    """Whether one outcome measure can be tested at all on this run."""

    measure_key: str
    testable: bool
    degenerate_dimensions: tuple[str, ...]
    total_dimensions: int

    @property
    def attenuated_by(self) -> tuple[str, ...]:
        """Degenerate constituents of a measure that is still testable.

        These do not stop the test; they drag its effect size toward zero by
        contributing a near-constant to the mean. Reported so a small composite
        effect is not read as a small underlying difference.
        """
        return () if not self.testable else self.degenerate_dimensions

    @property
    def note(self) -> str:
        if not self.degenerate_dimensions:
            return ""
        listed = ", ".join(self.degenerate_dimensions)
        if not self.testable:
            return (
                f"INSTRUMENT-LIMITED: every dimension of {self.measure_key} ({listed}) is "
                "degenerate on this run — the judge concentrated its scores on one value, so "
                "most within-scenario differences are exactly zero and the test ranks only the "
                "few that are not. Read the pair count and the Hodges-Lehmann shift in points "
                "below, never the p-value alone. This is a limit of the instrument, and the "
                "result must not be reported either as a clean directional finding or as "
                "'no significant difference'."
            )
        return (
            f"ATTENUATED: {len(self.degenerate_dimensions)} of {self.total_dimensions} "
            f"dimensions in {self.measure_key} are degenerate ({listed}). The comparison is "
            "testable on the dimensions that discriminate, but the composite's effect size is "
            "pulled toward zero by the constant ones and understates the difference on the "
            "dimensions that actually moved."
        )


def measure_testability(
    m: Measure | str,
    statuses: Mapping[str, Discrimination],
) -> MeasureTestability:
    """Whether a measure survives its degenerate constituents."""
    resolved = measure(m) if isinstance(m, str) else m
    degenerate = tuple(
        d for d in resolved.dimensions if statuses.get(d) is Discrimination.DEGENERATE
    )
    return MeasureTestability(
        measure_key=resolved.key,
        testable=len(degenerate) < len(resolved.dimensions),
        degenerate_dimensions=degenerate,
        total_dimensions=len(resolved.dimensions),
    )


@dataclass(frozen=True, slots=True)
class InstrumentReport:
    """The judge's resolving power, dimension by dimension. Read before any result.

    Placed before the comparisons in the rendered report on purpose. A reader
    who reaches a naturalness p-value without having read this section will
    misread it, and ordering is the only mechanism that reliably prevents that.
    """

    distributions: tuple[DimensionDistribution, ...]
    threshold_stable: bool
    threshold_changes: tuple[str, ...] = ()
    prespecified: bool = False

    @property
    def statuses(self) -> dict[str, Discrimination]:
        return classify(self.distributions)

    @property
    def degenerate(self) -> tuple[str, ...]:
        return tuple(d.dimension for d in self.distributions if d.is_degenerate)

    @property
    def discriminating(self) -> tuple[str, ...]:
        return tuple(
            d.dimension
            for d in self.distributions
            if d.discrimination is Discrimination.DISCRIMINATING
        )

    @property
    def ritual_mechanism_testable(self) -> bool:
        """Can the v3 naturalness-via-ritual prediction be tested at all?

        False when every dimension carrying that prediction is degenerate. The
        prediction is then neither supported nor refuted by this run — the run
        had no way to address it.
        """
        degenerate = set(self.degenerate)
        return not all(d in degenerate for d in RITUAL_MECHANISM_DIMENSIONS)

    def testability(self, m: Measure | str) -> MeasureTestability:
        return measure_testability(m, self.statuses)

    def render(self) -> str:
        lines = [
            "INSTRUMENT RESOLUTION — does each dimension carry any variance? [NOT PLANNED IN "
            "ADVANCE: post-hoc diagnostic]",
            "",
            "  A dimension the judge scored at a floor or a ceiling cannot separate two "
            "conditions. A",
            "  large p-value on such a dimension means the instrument did not resolve it, NOT "
            "that the",
            "  conditions are alike. Read this table before any comparison below.",
            "",
            f"  Rule: degenerate when sd < {MIN_SD} rubric points. Modal share is REPORTED, not "
            "used as a criterion —",
            "  a bimodal dimension can be 77% concentrated and still discriminate hard (see "
            "`name` below).",
            "  Scored on the to_quality() scale the tests run on. Chosen after seeing this run's "
            "scores,",
            "  so the numbers for every dimension are printed and the cut can be moved by the "
            "reader.",
            "",
        ]
        for d in self.distributions:
            lines.append(d.render())
        lines.append("")
        degenerate = self.degenerate
        if degenerate:
            lines.append(f"  DEGENERATE ({len(degenerate)}): {', '.join(degenerate)}")
            for d in self.distributions:
                if d.is_degenerate:
                    lines.append(f"    - {d.dimension}: {d.why}")
        else:
            lines.append("  No dimension is degenerate on this run.")
        lines.append(
            f"  DISCRIMINATING ({len(self.discriminating)}): "
            f"{', '.join(self.discriminating) or '(none)'}"
        )
        lines.append("")
        if self.threshold_stable:
            lines.append(
                "  Threshold check: the degenerate set is identical at every sd cut from 0.60 to "
                "0.95 — the whole empty band between this run's two groups. The exact "
                "threshold is not doing the work."
            )
        else:
            lines.append(
                "  Threshold check: the degenerate set MOVES within the range tried, so this "
                "classification is a judgement call at the margin. What changes, and where:"
            )
            lines.extend(self.threshold_changes)
        lines.append("")
        if not self.ritual_mechanism_testable:
            lines.append(
                "  *** THE RITUAL MECHANISM CANNOT BE TESTED ON THIS RUN. ***\n"
                "  Build plan v3 predicts Condition B loses to A on `naturalness` BECAUSE "
                "framework prompting\n"
                "  induces ritual. Both dimensions carrying that prediction — "
                f"{', '.join(RITUAL_MECHANISM_DIMENSIONS)} — are\n"
                "  degenerate here, so this run can neither support nor refute it. That is a "
                "null result about\n"
                "  the instrument, not about the system, and it is the honest form of the "
                "finding. Reporting it\n"
                "  as 'no significant difference in naturalness' would be a claim the data "
                "cannot carry."
            )
            lines.append("")
        lines.append(f"  {FLOOR_MECHANISM_NOTE}")
        return "\n".join(lines)


def instrument_report(
    long: pd.DataFrame,
    *,
    dimensions: Sequence[str] = RUBRIC_DIMENSIONS,
) -> InstrumentReport:
    """Describe and classify every dimension, and check the cut is not load-bearing."""
    distributions = describe_dimensions(long, dimensions=dimensions)
    stable, changes = threshold_sensitivity(distributions)
    return InstrumentReport(
        distributions=distributions,
        threshold_stable=stable,
        threshold_changes=tuple(changes),
    )
