"""The sensitivity re-runs, and what happens if a conclusion moves.

Analysis plan §8.5 / build plan v3 §14: the primary analysis is re-run three
ways and whether the conclusions hold under each is reported.

    (a) judge-only ratings vs human-only ratings, once human ratings exist;
    (b) with and without turns where Condition C's CRAG gate fell back to
        Condition-B behaviour (`retrieval_trace.fell_back_to_b`), since a
        fallback turn is not really testing retrieval;
    (c) excluding scenarios where judge self-consistency (§9) was poor, to check
        whether the headline conclusions depend on the judge's least stable items.

A fourth was added after the run, and is labelled as not planned in advance
wherever it appears:

    (d) with and without the generations the output safety gate refused
        (`generation.gate_blocked`, D12). D12 postdates the plan. This rerun is
        the **preferred reading** of the primary comparison, for the reason
        given on `sensitivity_gate_blocked`.

`retrieval_contrast` is not a rerun and is not in this list. It reports B vs C
twice because the pooled and retrieval-only forms answer two different
questions, and it is a headline result rather than a robustness check.

**A conclusion that flips under a sensitivity analysis is the finding.** So the
comparison is not decoration around the reruns -- `ConclusionFlip` is a first-class
object, `SensitivityReport.flips` collects every one across all three, and
`render()` puts them at the top, before the tables they came from. A rerun whose
conclusions all hold prints one line saying so; a rerun that moves one prints
what moved, in which direction, and on which measure.

"Conclusion" is defined narrowly and mechanically, so the comparison cannot
drift: for each hypothesis, the pair (Holm-significant at alpha, observed
direction of the headline effect). A flip is any change in either. A change in
the *size* of an effect is reported alongside but is not called a flip -- an
effect that halves while staying significant and in the same direction is worth
seeing and is not the same event as a conclusion reversing.

==========================================================================
ONE THING THE PLAN DOES NOT FIX, FLAGGED RATHER THAN DECIDED
==========================================================================
§8.5(c) says "excluding scenarios where judge self-consistency was poor". It
does not define *poor* numerically. D9.4 settled it at `pct_range_ge_2 > 0.25`,
which is the default below — but that decision was recorded in `DECISIONS.md`
rather than written into the plan document, and under D10 there is no registry
to have written it into. So the cut is an explicit parameter, its value is
recorded on the result, and `JudgeConsistencyExclusion.threshold_prespecified`
stays `False` and prints in the output.

The default implements the one hook the codebase already offers --
`carelite.eval.judge.validation.SelfConsistency.pct_range_ge_2`, whose own
docstring says it is "the number that decides which scenarios get excluded in
the v3 §14 sensitivity analysis" -- as: exclude a scenario when more than
`max_pct_range_ge_2` of its judged (generation, dimension) items had five
samples spanning two or more scale points. The default cut of 0.25 is a
judgement carried on the result, and results computed under it say so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import pandas as pd

from carelite.eval.judge.validation import EvidenceStatus
from carelite.stats.effects import DEFAULT_N_BOOT, DEFAULT_SEED
from carelite.stats.instrument import Discrimination
from carelite.stats.measures import attach_quality
from carelite.stats.primary import (
    CONFIRMATORY_FAMILY,
    FamilyResult,
    Hypothesis,
    PairwiseResult,
    run_family,
    run_pairwise,
)
from carelite.types import Condition, RaterType

__all__ = [
    "DEFAULT_MAX_PCT_RANGE_GE_2",
    "Conclusion",
    "ConclusionFlip",
    "JudgeConsistencyExclusion",
    "RetrievalContrast",
    "SensitivityReport",
    "SensitivityRun",
    "compare_conclusions",
    "conclusions",
    "retrieval_contrast",
    "run_all_sensitivity",
    "scenario_judge_consistency",
    "sensitivity_crag_fallback",
    "sensitivity_gate_blocked",
    "sensitivity_judge_consistency",
    "sensitivity_rater_type",
]

#: NOT pre-specified. See the module docstring.
DEFAULT_MAX_PCT_RANGE_GE_2 = 0.25


# ---------------------------------------------------------------------------
# What counts as a conclusion, and what counts as a flip
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Conclusion:
    """One hypothesis's verdict, reduced to the two things that can flip."""

    key: str
    measure_key: str
    pair: str
    significant: bool
    direction: str
    effect: float
    p_holm: float
    n_scenarios: int


def conclusions(family: FamilyResult) -> dict[str, Conclusion]:
    """Reduce a family to `hypothesis key -> Conclusion`."""
    return {
        r.hypothesis.key: Conclusion(
            key=r.hypothesis.key,
            measure_key=r.hypothesis.measure_key,
            pair=r.hypothesis.pair_label,
            significant=r.significant(family.alpha),
            direction=r.observed_direction,
            effect=r.effects.rank_biserial.point,
            p_holm=r.p_holm,
            n_scenarios=r.n_scenarios,
        )
        for r in family.results
    }


@dataclass(frozen=True, slots=True)
class ConclusionFlip:
    """A conclusion that moved between the base analysis and a rerun."""

    key: str
    measure_key: str
    pair: str
    what: str
    base: Conclusion
    variant: Conclusion

    def render(self) -> str:
        return (
            f"    {self.pair} on {self.measure_key}: {self.what}\n"
            f"      base    effect {self.base.effect:+.3f}, direction {self.base.direction}, "
            f"Holm p {self.base.p_holm:.4g}, "
            f"{'significant' if self.base.significant else 'not significant'} "
            f"(n = {self.base.n_scenarios})\n"
            f"      rerun   effect {self.variant.effect:+.3f}, direction "
            f"{self.variant.direction}, Holm p {self.variant.p_holm:.4g}, "
            f"{'significant' if self.variant.significant else 'not significant'} "
            f"(n = {self.variant.n_scenarios})"
        )


def compare_conclusions(
    base: FamilyResult,
    variant: FamilyResult,
) -> tuple[ConclusionFlip, ...]:
    """Every conclusion that changed between two families.

    A hypothesis present in one family and absent from the other is itself a
    flip -- "could not be computed in the rerun" is a change in what the study
    can conclude, not a row to leave out of the table.
    """
    left = conclusions(base)
    right = conclusions(variant)
    flips: list[ConclusionFlip] = []
    for key in sorted(set(left) | set(right)):
        a = left.get(key)
        b = right.get(key)
        if a is None and b is not None:
            present, missing = b, "base"
        elif b is None and a is not None:
            present, missing = a, "rerun"
        else:
            present, missing = None, ""
        if present is not None:
            flips.append(
                ConclusionFlip(
                    key=key,
                    measure_key=present.measure_key,
                    pair=present.pair,
                    what=f"not computable in the {missing} analysis",
                    base=a if a is not None else present,
                    variant=b if b is not None else present,
                )
            )
            continue
        if a is None or b is None:  # pragma: no cover - both absent is impossible
            continue
        reasons = []
        if a.significant != b.significant:
            reasons.append("became significant" if b.significant else "lost significance")
        if a.direction != b.direction:
            reasons.append(f"direction moved {a.direction} -> {b.direction}")
        if reasons:
            flips.append(
                ConclusionFlip(
                    key=key,
                    measure_key=a.measure_key,
                    pair=a.pair,
                    what="; ".join(reasons),
                    base=a,
                    variant=b,
                )
            )
    return tuple(flips)


@dataclass(frozen=True, slots=True)
class SensitivityRun:
    """One rerun: what was varied, the family it produced, and what moved."""

    name: str
    specification: str
    family: FamilyResult
    flips: tuple[ConclusionFlip, ...]
    prespecified: bool = True
    #: Anything about this rerun that is a judgement rather than a fixed
    #: constant. Printed with the result, never only in the prose.
    caveats: tuple[str, ...] = ()

    @property
    def conclusions_hold(self) -> bool:
        return not self.flips

    @property
    def nothing_to_compare(self) -> bool:
        """True when the rerun computed no comparisons at all.

        Distinguished from `conclusions_hold` deliberately: a rerun with nothing
        in it has no flips, and reporting that as "conclusions hold" would turn
        an absence of data into a robustness claim.
        """
        return not self.family.results

    def render(self) -> str:
        lines = [
            f"  {self.name}{'' if self.prespecified else '   [NOT PLANNED IN ADVANCE]'}",
            f"    {self.specification}",
        ]
        for caveat in self.caveats:
            lines.append(f"    CAVEAT: {caveat}")
        if self.nothing_to_compare:
            lines.append(
                "    No comparison could be computed, so there is nothing to hold or move."
            )
        elif self.conclusions_hold:
            lines.append("    Conclusions hold: no comparison changed significance or direction.")
        else:
            lines.append(f"    {len(self.flips)} CONCLUSION(S) MOVED:")
            for flip in self.flips:
                lines.append(flip.render())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# (a) judge-only vs human-only
# ---------------------------------------------------------------------------


def sensitivity_rater_type(
    long: pd.DataFrame,
    base: FamilyResult,
    *,
    hypotheses: Sequence[Hypothesis] = CONFIRMATORY_FAMILY,
    statuses: Mapping[str, EvidenceStatus] | None = None,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    discrimination: Mapping[str, Discrimination] | None = None,
) -> tuple[SensitivityRun, ...]:
    """§8.5(a). One rerun per rater type present, each compared against `base`.

    Returns an empty tuple when only one rater type is in the data, which is the
    current state of the project: `docs/limitations.md` §4 records that no human
    rating has occurred, so there is nothing to compare the judge against. An
    empty result here is the honest answer and is reported as "cannot yet run",
    not as "conclusions hold".
    """
    present = sorted({str(r) for r in long.get("rater_type", pd.Series(dtype=str)).dropna()})
    if len(present) < 2:
        return ()
    runs: list[SensitivityRun] = []
    for rater in present:
        family = run_family(
            long,
            hypotheses,
            name=f"sensitivity (a): {rater} ratings only",
            alpha=alpha,
            rater_type=rater,
            statuses=statuses,
            n_boot=n_boot,
            seed=seed,
            discrimination=discrimination,
        )
        caveats: tuple[str, ...] = ()
        if rater == str(RaterType.HUMAN):
            caveats = (
                "human ratings cover the stratified §12 sample (20 scenarios x 3 conditions), "
                "not the whole holdout, so this rerun is a smaller n rather than the same "
                "analysis with different raters",
            )
        runs.append(
            SensitivityRun(
                name=f"(a) {rater}-only ratings",
                specification=(f"the §8.1 family recomputed from rater_type = {rater!r} rows only"),
                family=family,
                flips=compare_conclusions(base, family),
                caveats=caveats,
            )
        )
    return tuple(runs)


# ---------------------------------------------------------------------------
# (b) CRAG fallback turns
# ---------------------------------------------------------------------------


def sensitivity_crag_fallback(
    long: pd.DataFrame,
    base: FamilyResult,
    *,
    hypotheses: Sequence[Hypothesis] = CONFIRMATORY_FAMILY,
    rater_type: str | None = None,
    statuses: Mapping[str, EvidenceStatus] | None = None,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    discrimination: Mapping[str, Discrimination] | None = None,
) -> SensitivityRun:
    """§8.5(b). Drop generations where Condition C's CRAG gate fell back to B.

    Excluded at the *generation* level, so a scenario keeps the samples in its
    cell that did use retrieval and its cell mean is recomputed from those. A
    scenario whose Condition-C cell falls away entirely leaves the paired
    comparison, and the count of scenarios that did is reported by the family's
    own `n_dropped`. That is the honest treatment: a fallback turn is not
    testing retrieval, but neither is it a missing observation to be imputed.
    """
    if "fell_back_to_b" not in long.columns:
        kept = long
        n_excluded = 0
    else:
        fell_back = long["fell_back_to_b"].astype("boolean").fillna(False)
        n_excluded = int(long.loc[fell_back, "generation_id"].nunique())
        kept = long[~fell_back]

    family = run_family(
        kept,
        hypotheses,
        name="sensitivity (b): CRAG-fallback turns excluded",
        alpha=alpha,
        rater_type=rater_type,
        statuses=statuses,
        n_boot=n_boot,
        seed=seed,
        discrimination=discrimination,
        notes=(f"{n_excluded} generations excluded for retrieval_trace.fell_back_to_b",),
    )
    return SensitivityRun(
        name="(b) CRAG-fallback turns excluded",
        specification=(
            f"the §8.1 family recomputed with the {n_excluded} generations where "
            "retrieval_trace.fell_back_to_b is true removed; a fallback turn ran "
            "Condition-B behaviour and is not testing retrieval"
        ),
        family=family,
        flips=compare_conclusions(base, family),
    )


# ---------------------------------------------------------------------------
# (d) the output safety gate — D12, added after the analysis plan was written
# ---------------------------------------------------------------------------


def sensitivity_gate_blocked(
    long: pd.DataFrame,
    base: FamilyResult,
    *,
    hypotheses: Sequence[Hypothesis] = CONFIRMATORY_FAMILY,
    rater_type: str | None = None,
    statuses: Mapping[str, EvidenceStatus] | None = None,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    discrimination: Mapping[str, Discrimination] | None = None,
) -> SensitivityRun:
    """Rerun without the generations the output safety gate refused (D12).

    **Neither inclusion nor exclusion is obviously right, which is why both are
    reported.** Scoring refused text as though it were an ordinary response
    flatters whichever condition produced it, and the refusal is invisible in
    the scores. But the refusals are not spread evenly: D12 measured 13 of 17 on
    SC-029 alone, so excluding them removes most of one scenario across several
    conditions rather than trimming symmetrically — and a paired test then loses
    that scenario from every comparison it touches.

    **The preferred reading is the one that excludes them**, and the reason is
    asymmetric. Including refused text puts a number on the rubric scale that
    the rubric was never meant to score — the gate's output is not a
    communication attempt, so its NURSE score is a category error rather than a
    low score. Losing one scenario from a 60-scenario paired analysis costs
    precision, which is visible in the n and in the interval. A category error
    costs correctness and is not visible anywhere. Prefer the loss you can see.

    This rerun is **not** in the analysis plan: `gate_blocked` did not exist when
    the plan was written and D12 is dated after it. It is labelled accordingly.
    """
    if "gate_blocked" not in long.columns:
        kept = long
        n_excluded = 0
        scenarios: list[str] = []
    else:
        blocked = long["gate_blocked"].astype("boolean").fillna(False)
        n_excluded = int(long.loc[blocked, "generation_id"].nunique())
        scenarios = sorted({str(s) for s in long.loc[blocked, "scenario_id"]})
        kept = long[~blocked]

    family = run_family(
        kept,
        hypotheses,
        name="sensitivity (d): output-gate-blocked generations excluded",
        alpha=alpha,
        rater_type=rater_type,
        statuses=statuses,
        n_boot=n_boot,
        seed=seed,
        discrimination=discrimination,
        notes=(f"{n_excluded} generations excluded for generation.gate_blocked",),
    )
    return SensitivityRun(
        name="(d) output-gate-blocked generations excluded  [PREFERRED READING]",
        specification=(
            f"the family recomputed with the {n_excluded} generations the carelite.safety "
            f"output gate refused removed (D12); affected scenarios: "
            f"{', '.join(scenarios) or '(none)'}"
        ),
        family=family,
        flips=compare_conclusions(base, family),
        prespecified=False,
        caveats=(
            "D12 postdates the analysis plan, so this rerun was not planned in advance.",
            "The refusals are concentrated — 13 of 17 on SC-029 — so this exclusion removes one "
            "scenario unevenly across conditions rather than trimming symmetrically. The base "
            "analysis above INCLUDES the refused text; this rerun excludes it; the preferred "
            "reading is this one, because a refused response scored on a communication rubric "
            "is a category error rather than a low score.",
        ),
    )


# ---------------------------------------------------------------------------
# Retrieval, asked as two different questions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetrievalContrast:
    """B vs C computed two ways, because they answer two different questions.

    On this run CRAG graded 69 of 180 Condition-C cells `none` and C fell back to
    Condition-B behaviour on them: **38% of the arm ran the comparison condition's
    code path.** Pooling them compares C against itself on more than a third of
    its mass, which attenuates any true retrieval effect toward zero.

    So the contrast is reported twice and the two are never collapsed:

    * `offered` — all 180 cells. *Does offering retrieval help?* This is the
      honest evaluation of the deployed system, in which CRAG sometimes declines
      to retrieve and that is part of the design.
    * `retrieved` — the 111 cells where retrieval actually happened. *Does
      retrieval help?* **This is the architecture's actual claim** and the only
      one of the two that bears on whether the retrieval layer earns its place.

    Neither is the "real" answer and the difference between them is itself
    informative: if `retrieved` is much larger than `offered`, the retrieval
    works when it fires and the gate is what limits it. If they agree, the
    fallback is not what is holding the effect down.

    `retrieved` is a **conditioned subset, not a randomised one.** CRAG chose
    which cells retrieved, on the basis of whether it judged the evidence
    relevant — so the 111 are plausibly the scenarios this corpus can serve, and
    comparing them against all 180 of B is a comparison on a self-selected
    subgroup. That is a real confound, it cannot be removed after the fact, and
    it is stated on the result rather than in a footnote.
    """

    offered: PairwiseResult | None
    retrieved: PairwiseResult | None
    n_offered_cells: int
    n_retrieved_cells: int
    n_fallback_cells: int
    selection_caveat: str = (
        "The `retrieved` subset is chosen by CRAG on relevance, not randomised. Those 111 cells "
        "are plausibly the scenarios this corpus can answer, so their comparison against the "
        "full Condition-B arm is a self-selected subgroup contrast. The direction of that bias "
        "favours retrieval, so a positive result here is weaker evidence than the same number "
        "from a randomised design would be."
    )

    def render(self, alpha: float = 0.05) -> str:
        lines = [
            "RETRIEVAL, ASKED TWO WAYS (B vs C on composite NURSE adherence)",
            f"  CRAG fell back to Condition-B behaviour on {self.n_fallback_cells} of "
            f"{self.n_offered_cells} Condition-C cells "
            f"({self.n_fallback_cells / self.n_offered_cells:.0%}), leaving "
            f"{self.n_retrieved_cells} that actually retrieved."
            if self.n_offered_cells
            else "  No Condition-C cells in this frame.",
            "",
            "  (1) DOES OFFERING RETRIEVAL HELP? — all Condition-C cells, fallbacks included.",
            "      The deployed system as it behaves, gate and all.",
        ]
        lines.append(
            self.offered.render(alpha) if self.offered is not None else "      not computable"
        )
        lines.extend(
            [
                "",
                "  (2) DOES RETRIEVAL HELP? — only the cells where retrieval actually fired.",
                "      THIS is the architecture's claim. See the selection caveat below.",
            ]
        )
        lines.append(
            self.retrieved.render(alpha) if self.retrieved is not None else "      not computable"
        )
        lines.extend(
            [
                "",
                "  The p-values in this section are UNCORRECTED (family of 1). The Holm-corrected "
                "p for",
                "  this comparison is the one in the §8.1 family above; reading the arm two ways "
                "is a",
                "  second look at one planned comparison, not two new members of the family.",
                "",
                f"  CAVEAT: {self.selection_caveat}",
            ]
        )
        return "\n".join(lines)


def retrieval_contrast(
    long: pd.DataFrame,
    *,
    hypothesis: Hypothesis | None = None,
    rater_type: str | None = None,
    statuses: Mapping[str, EvidenceStatus] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    discrimination: Mapping[str, Discrimination] | None = None,
) -> RetrievalContrast:
    """B vs C, pooled and retrieval-only. See `RetrievalContrast`."""
    chosen = hypothesis or next(
        h for h in CONFIRMATORY_FAMILY if h.key == "secondary2_nurse_B_vs_C"
    )
    is_c = long["condition"].astype(str) == str(Condition.C)
    if "fell_back_to_b" in long.columns:
        fell_back = long["fell_back_to_b"].astype("boolean").fillna(False)
    else:
        fell_back = pd.Series(False, index=long.index)

    n_offered = int(long.loc[is_c, "generation_id"].nunique())
    n_fallback = int(long.loc[is_c & fell_back, "generation_id"].nunique())

    def _run(frame: pd.DataFrame, suffix: str, note: str) -> PairwiseResult | None:
        result = run_pairwise(
            frame,
            Hypothesis(
                key=f"{chosen.key}_{suffix}",
                measure_key=chosen.measure_key,
                left=chosen.left,
                right=chosen.right,
                expected_higher=chosen.expected_higher,
                description=f"{chosen.description} {note}",
                role=chosen.role,
                prespecified=chosen.prespecified and suffix == "offered",
            ),
            rater_type=rater_type,
            statuses=statuses,
            n_boot=n_boot,
            seed=seed,
            discrimination=discrimination,
            extra_reasons=()
            if suffix == "offered"
            else ("restricted to CRAG-selected cells, which is not a randomised subgroup",),
        )
        if result is None:
            return None
        # Uncorrected, and labelled as such rather than rendered as `nan` in a
        # family of zero. The pooled form's Holm-corrected p is the one in the
        # §8.1 family above; splitting the arm two ways is a second look at one
        # registered comparison, not two new members of the family.
        return replace(result, p_holm=result.test.p_value, family_size=1)

    return RetrievalContrast(
        offered=_run(long, "offered", "All Condition-C cells: does offering retrieval help?"),
        retrieved=_run(
            long[~(is_c & fell_back)],
            "retrieved",
            "Only cells where retrieval fired: does retrieval help?",
        ),
        n_offered_cells=n_offered,
        n_retrieved_cells=n_offered - n_fallback,
        n_fallback_cells=n_fallback,
    )


# ---------------------------------------------------------------------------
# (c) poor judge self-consistency
# ---------------------------------------------------------------------------


def scenario_judge_consistency(judge_samples: pd.DataFrame) -> pd.DataFrame:
    """Per-scenario judge stability, from the judge's per-sample rows.

    For each (generation, dimension) with at least two admitted samples, the
    range of the `to_quality()` scores across those samples; then per scenario,
    the share of those items whose range was >= 2 scale points. That share is
    `SelfConsistency.pct_range_ge_2` aggregated to the scenario, which is the
    quantity `carelite.eval.judge.validation` names as the input to this
    analysis.

    Returns columns `scenario_id, n_items, pct_range_ge_2, mean_range`.
    """
    if judge_samples.empty:
        return pd.DataFrame(columns=["scenario_id", "n_items", "pct_range_ge_2", "mean_range"])
    scored = attach_quality(judge_samples)
    scored = scored[scored["quality"].notna()]
    if scored.empty:
        return pd.DataFrame(columns=["scenario_id", "n_items", "pct_range_ge_2", "mean_range"])

    per_item = (
        scored.groupby(["scenario_id", "generation_id", "dimension"], observed=True)["quality"]
        .agg(["min", "max", "count"])
        .reset_index()
    )
    per_item = per_item[per_item["count"] >= 2]
    if per_item.empty:
        return pd.DataFrame(columns=["scenario_id", "n_items", "pct_range_ge_2", "mean_range"])
    per_item["score_range"] = per_item["max"] - per_item["min"]

    grouped = per_item.groupby("scenario_id", observed=True)["score_range"]
    out = pd.DataFrame(
        {
            "n_items": grouped.size(),
            "mean_range": grouped.mean(),
            "pct_range_ge_2": grouped.apply(lambda s: float((s >= 2).mean())),
        }
    ).reset_index()
    return out


@dataclass(frozen=True, slots=True)
class JudgeConsistencyExclusion:
    """Which scenarios were excluded as unstable, and on what rule.

    `threshold_prespecified` is `False` and stays `False`: §8.5(c) says "poor"
    without a number. See the module docstring.
    """

    threshold: float
    metric: str
    excluded: tuple[str, ...]
    n_scenarios_scored: int
    threshold_prespecified: bool = False
    reason: str = field(
        default=(
            "pre-registration §8.5(c) specifies the analysis but not the numeric cut for "
            "`poor` self-consistency; this threshold is an implementation choice and results "
            "under it are reported as such"
        )
    )


def sensitivity_judge_consistency(
    long: pd.DataFrame,
    base: FamilyResult,
    judge_samples: pd.DataFrame,
    *,
    max_pct_range_ge_2: float = DEFAULT_MAX_PCT_RANGE_GE_2,
    hypotheses: Sequence[Hypothesis] = CONFIRMATORY_FAMILY,
    rater_type: str | None = None,
    statuses: Mapping[str, EvidenceStatus] | None = None,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    discrimination: Mapping[str, Discrimination] | None = None,
) -> tuple[SensitivityRun, JudgeConsistencyExclusion]:
    """§8.5(c). Drop scenarios the judge scored least stably, then rerun."""
    consistency = scenario_judge_consistency(judge_samples)
    if consistency.empty:
        excluded: tuple[str, ...] = ()
    else:
        excluded = tuple(
            sorted(
                consistency.loc[
                    consistency["pct_range_ge_2"] > max_pct_range_ge_2, "scenario_id"
                ].astype(str)
            )
        )
    exclusion = JudgeConsistencyExclusion(
        threshold=max_pct_range_ge_2,
        metric="share of judged (generation, dimension) items with a 5-sample range >= 2",
        excluded=excluded,
        n_scenarios_scored=int(consistency.shape[0]),
    )

    kept = long[~long["scenario_id"].astype(str).isin(excluded)] if excluded else long
    family = run_family(
        kept,
        hypotheses,
        name="sensitivity (c): unstable-judge scenarios excluded",
        alpha=alpha,
        rater_type=rater_type,
        statuses=statuses,
        n_boot=n_boot,
        seed=seed,
        discrimination=discrimination,
        notes=(
            f"{len(excluded)} of {exclusion.n_scenarios_scored} scenarios excluded at "
            f"{exclusion.metric} > {max_pct_range_ge_2}",
        ),
    )
    run = SensitivityRun(
        name="(c) scenarios with poor judge self-consistency excluded",
        specification=(
            f"the §8.1 family recomputed without the {len(excluded)} scenarios whose "
            f"{exclusion.metric} exceeded {max_pct_range_ge_2}"
        ),
        family=family,
        flips=compare_conclusions(base, family),
        caveats=(exclusion.reason,),
    )
    return run, exclusion


# ---------------------------------------------------------------------------
# All three
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SensitivityReport:
    """All three §8.5 reruns, with every flip collected at the top."""

    runs: tuple[SensitivityRun, ...]
    judge_consistency_exclusion: JudgeConsistencyExclusion | None
    not_runnable: tuple[str, ...] = ()

    @property
    def flips(self) -> tuple[ConclusionFlip, ...]:
        return tuple(flip for run in self.runs for flip in run.flips)

    @property
    def conclusions_hold(self) -> bool:
        return not self.flips

    @property
    def nothing_to_compare(self) -> bool:
        return all(run.nothing_to_compare for run in self.runs)

    def render(self) -> str:
        lines = ["SENSITIVITY ANALYSES (analysis plan §8.5, build plan v3 §14)", ""]
        if self.nothing_to_compare:
            lines.append(
                "  No comparison could be computed in any rerun, so nothing was tested for "
                "robustness. This is not a finding that the conclusions hold."
            )
        elif self.flips:
            lines.append(
                f"  *** {len(self.flips)} CONCLUSION(S) MOVE UNDER SENSITIVITY ANALYSIS. ***"
            )
            lines.append(
                "  A conclusion that flips under a sensitivity analysis is the finding, not a "
                "footnote to one."
            )
            for flip in self.flips:
                lines.append(flip.render())
        else:
            lines.append(
                "  Every conclusion holds under every rerun that could be run: no comparison "
                "changed significance or direction."
            )
        for missing in self.not_runnable:
            lines.append(f"  NOT RUNNABLE: {missing}")
        lines.append("")
        for run in self.runs:
            lines.append(run.render())
            lines.append("")
        return "\n".join(lines).rstrip()


def run_all_sensitivity(
    long: pd.DataFrame,
    base: FamilyResult,
    *,
    judge_samples: pd.DataFrame | None = None,
    hypotheses: Sequence[Hypothesis] = CONFIRMATORY_FAMILY,
    rater_type: str | None = None,
    statuses: Mapping[str, EvidenceStatus] | None = None,
    alpha: float = 0.05,
    max_pct_range_ge_2: float = DEFAULT_MAX_PCT_RANGE_GE_2,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    discrimination: Mapping[str, Discrimination] | None = None,
) -> SensitivityReport:
    """Run (a), (b), (c) and (d), and collect every conclusion that moved."""
    runs: list[SensitivityRun] = []
    not_runnable: list[str] = []

    rater_runs = sensitivity_rater_type(
        long,
        base,
        hypotheses=hypotheses,
        statuses=statuses,
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
        discrimination=discrimination,
    )
    if rater_runs:
        runs.extend(rater_runs)
    else:
        not_runnable.append(
            "§8.5(a) judge-only vs human-only — fewer than two rater types in the data. "
            "docs/limitations.md §4: no human rating has occurred, so every number is "
            "judge-only and there is nothing yet to compare it against."
        )

    runs.append(
        sensitivity_crag_fallback(
            long,
            base,
            hypotheses=hypotheses,
            rater_type=rater_type,
            statuses=statuses,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
            discrimination=discrimination,
        )
    )

    runs.append(
        sensitivity_gate_blocked(
            long,
            base,
            hypotheses=hypotheses,
            rater_type=rater_type,
            statuses=statuses,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
            discrimination=discrimination,
        )
    )

    exclusion: JudgeConsistencyExclusion | None = None
    if judge_samples is not None and not judge_samples.empty:
        run, exclusion = sensitivity_judge_consistency(
            long,
            base,
            judge_samples,
            max_pct_range_ge_2=max_pct_range_ge_2,
            hypotheses=hypotheses,
            rater_type=rater_type,
            statuses=statuses,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
            discrimination=discrimination,
        )
        runs.append(run)
    else:
        not_runnable.append(
            "§8.5(c) excluding poorly-self-consistent scenarios — no multi-sample judge rows. "
            "The 5-sample self-consistency pass runs on the validation subset only "
            "(analysis plan §9), and the holdout was judged single-pass at temperature 0, so "
            "no per-sample rows exist for it. This rerun cannot be run on this data."
        )

    return SensitivityReport(
        runs=tuple(runs),
        judge_consistency_exclusion=exclusion,
        not_runnable=tuple(not_runnable),
    )
