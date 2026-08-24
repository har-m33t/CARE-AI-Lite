"""The negative control: can the rubric tell Condition D from Condition B at all?

Pre-registration §4.7 and §8.6, and build plan v3 §14: "if the rubric cannot
separate Condition B from the deliberately degraded Condition D, that is
reported as a rubric validity failure, not explained away."

This is the one analysis in the package whose interesting outcome is the
failure. Everything else asks how large an effect is; this asks whether the
instrument works. If it does not, every other number in the study is measuring
something, but not the thing it says it is measuring, and no amount of Holm
correction repairs that. `NegativeControlResult.render()` therefore leads with
the verdict in the first line, and the failure text is not a footnote.

**What Condition D is, and what it is not.** Pre-registration §2: D is
instructed to be brief, avoid dwelling on feeling, avoid open questions and
close topics quickly -- degraded on the communication dimensions the rubric
scores, and **not** on safety. The same output-safety gate applies to D as to
every other condition, so a D response the gate blocks is a real failure rather
than the control working as designed, and §10 excludes it from rubric analysis
and logs it as a safety event. Nothing in this module treats a safety block as
evidence that the control worked; those rows never reach it.

**The verdict rule, stated before the data exist.** The rubric separates B from
D when all three hold:

1. the observed direction is B > D -- a rubric that ranks the degraded prompt
   *higher* is a worse failure than one that cannot tell them apart, and is
   reported as its own outcome;
2. the 95% bootstrap CI on the effect excludes zero, which is the §8.2
   effect-size-first criterion;
3. the Holm-corrected p-value from the §8.1 family is below alpha.

All three, not any of them. Requiring the interval and the corrected p to agree
is stricter than the pre-registration's own wording, which asks only whether the
rubric "can separate" them; a control that passes on a p-value while its
interval spans zero has not demonstrated separation, and this is the one place
where a generous reading buys nothing.

**Reading a pass.** §4.7 registers B > D "by a large margin". A separation that
is statistically detectable but small is a pass on the letter of the rule and is
worth stating plainly, so the effect size is reported next to the verdict and
`margin_is_large` records whether it cleared the conventional large-effect
threshold. That flag is descriptive: the pre-registration fixes no numeric
margin, and one is not invented here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

from carelite.eval.judge.validation import EvidenceStatus
from carelite.stats.effects import DEFAULT_N_BOOT, DEFAULT_SEED
from carelite.stats.primary import (
    CONFIRMATORY_FAMILY,
    FamilyResult,
    Hypothesis,
    PairwiseResult,
    run_pairwise,
)

__all__ = [
    "LARGE_RANK_BISERIAL",
    "NEGATIVE_CONTROL_HYPOTHESIS",
    "NegativeControlResult",
    "negative_control",
]

#: Conventional "large" for a rank-biserial correlation, used only to describe
#: whether a passing separation was also a wide one. Not a pre-specified cut.
LARGE_RANK_BISERIAL = 0.5

#: The §4.7 hypothesis, taken from the pre-specified family rather than
#: redefined, so the two can never drift apart.
NEGATIVE_CONTROL_HYPOTHESIS: Hypothesis = next(
    h for h in CONFIRMATORY_FAMILY if h.key == "secondary7_nurse_B_vs_D"
)


class NegativeControlResult:
    """The §8.6 verdict on the measurement instrument.

    Not a dataclass: the verdict is derived from the comparison and must not be
    settable independently of it. There is no way to construct this object
    asserting that the rubric works while holding a comparison that says it does
    not.
    """

    __slots__ = ("_alpha", "_comparison")

    def __init__(self, comparison: PairwiseResult, *, alpha: float = 0.05) -> None:
        self._comparison = comparison
        self._alpha = alpha

    @property
    def comparison(self) -> PairwiseResult:
        return self._comparison

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def effect(self) -> float:
        return self._comparison.effects.rank_biserial.point

    @property
    def direction_correct(self) -> bool:
        """True when B scored higher than D, the registered direction."""
        return self._comparison.observed_direction == ">"

    @property
    def interval_excludes_zero(self) -> bool:
        return self._comparison.effects.rank_biserial.ci.excludes_zero

    @property
    def significant(self) -> bool:
        return self._comparison.significant(self._alpha)

    @property
    def rubric_separates(self) -> bool:
        """The verdict. All three criteria, not any of them."""
        return self.direction_correct and self.interval_excludes_zero and self.significant

    @property
    def inverted(self) -> bool:
        """The worse failure: the rubric ranked the degraded prompt higher."""
        return (
            self._comparison.observed_direction == "<"
            and self.interval_excludes_zero
            and self.significant
        )

    @property
    def margin_is_large(self) -> bool:
        effect = self.effect
        return (not math.isnan(effect)) and abs(effect) >= LARGE_RANK_BISERIAL

    def render(self) -> str:
        if self.inverted:
            verdict = (
                "*** NEGATIVE CONTROL INVERTED: the rubric scored the deliberately degraded "
                "Condition D HIGHER than Condition B. ***"
            )
        elif self.rubric_separates:
            margin = "a large margin" if self.margin_is_large else "a detectable but modest margin"
            verdict = f"NEGATIVE CONTROL PASSES: the rubric separates B from D, by {margin}."
        else:
            verdict = (
                "*** NEGATIVE CONTROL FAILS: the rubric does not separate Condition B from the "
                "deliberately degraded Condition D. ***"
            )

        lines = [
            "NEGATIVE CONTROL (pre-registration §4.7, §8.6; build plan v3 §14)",
            f"  {verdict}",
        ]
        if not self.rubric_separates:
            lines.append(
                "  This is a rubric validity failure and is reported as one. If the instrument "
                "cannot tell a prompt built to be worse on the scored dimensions from the "
                "framework-prompted condition, then it is not measuring the construct it claims "
                "to, and every other comparison in this study inherits that. It is not explained "
                "away, and it is not repaired by re-scoring."
            )
        lines.extend(
            [
                "",
                f"  direction B > D:              {self.direction_correct}",
                f"  95% bootstrap CI excludes 0:  {self.interval_excludes_zero}",
                f"  Holm-corrected p < {self.alpha}:      {self.significant}",
                "",
                self._comparison.render(self._alpha),
                "",
                "  Condition D is degraded on the communication dimensions the rubric scores, "
                "not on safety (pre-registration §2). A D response blocked by the output-safety "
                "gate is a real failure, is excluded from rubric analysis under §10, and does "
                "not appear above.",
            ]
        )
        return "\n".join(lines)


def negative_control(
    long: pd.DataFrame,
    *,
    family: FamilyResult | None = None,
    rater_type: str | None = None,
    statuses: Mapping[str, EvidenceStatus] | None = None,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> NegativeControlResult | None:
    """Verify the rubric distinguishes Condition D from Condition B.

    Pass `family` -- the already-run §8.1 family -- so the Holm-corrected p-value
    is the one from the pre-specified family rather than a fresh uncorrected
    test. Without it the comparison is recomputed and its `p_holm` is the raw
    p-value corrected in a family of one, which is stated on the result and is
    weaker evidence than the family version.
    """
    if family is not None:
        found = family.by_key(NEGATIVE_CONTROL_HYPOTHESIS.key)
        if found is not None:
            return NegativeControlResult(found, alpha=family.alpha)

    computed = run_pairwise(
        long,
        NEGATIVE_CONTROL_HYPOTHESIS,
        rater_type=rater_type,
        statuses=statuses,
        n_boot=n_boot,
        seed=seed,
    )
    if computed is None:
        return None
    from dataclasses import replace

    return NegativeControlResult(
        replace(computed, p_holm=computed.test.p_value, family_size=1), alpha=alpha
    )
