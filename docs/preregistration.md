# OSF Pre-Registration — Draft

**Status: DRAFT, NOT YET REGISTERED.** This document is written to be pasted into an OSF
registration template, not to be treated as a substitute for one. Registration is an
account-holder action that only the project owner can take (`DECISIONS.md`, "Gates that remain
with a person"). This file exists so that action costs an afternoon of copy-editing, not a
from-scratch drafting effort, and so the analysis is fixed in writing before anyone can be tempted
to peek.

---

## ⚠ THE ORDERING DEPENDENCY — READ THIS FIRST

**This must be registered on OSF before any of the 1,080 holdout generations exist.**

Build plan v3 §10 states the argument this project rests on: *"Pre-registration is what makes the
naturalness result credible if it goes against you. Without it, 'Condition A beat B on
naturalness' reads as a post-hoc excuse. With it, it reads as a pre-specified secondary outcome
that came out the interesting way."* The generator, judge, and rubric all exist as code right now
and none of the 60 held-out scenarios have been run through them for the full evaluation. That is
the window this document is written in, and it is the only window in which it is free — the moment
inference lane III runs, everything below stops being a pre-specification and starts being a
description of what the numbers happened to look like.

**Everything not explicitly listed as primary, secondary, or a pre-specified sensitivity analysis
below is exploratory.** That includes any comparison, subgroup, or figure that occurs to a reader
of the results but was not written down here first.

---

## 1. Study type and design

A within-scenario, repeated-measures comparison of an AI communication-support system across six
experimental conditions, scored on an 11-dimension rubric operationalizing NURSE and the Four
Habits Model, using a local LLM-as-judge validated as a component study in its own right, over a
frozen 60-scenario held-out evaluation set.

## 2. The six conditions

Fixed in `carelite.types.Condition` (frozen contract; not editable by any build lane without a
foundation-lane amendment):

| Condition | Definition |
|---|---|
| **A** | Bare model, no framework prompting, no retrieval. Generator: `gemma4:12b`. |
| **A2** | Bare model, second model family, no framework, no retrieval. Cross-model baseline. Generator: `qwen3.5:9b`. |
| **B** | Framework-prompted (NURSE / Four Habits guidance in the system prompt), no retrieval. Generator: `gemma4:12b`. |
| **C** | Framework-prompted plus hybrid retrieval (dense + lexical + graph, RRF fusion, cross-encoder rerank, HyDE, CRAG fallback to B on irrelevant retrieval). Generator: `gemma4:12b`. |
| **LC** | Long-context: the whole corpus stuffed into context, no retrieval pipeline. Generator: `gemma4:12b`, 128k context window. |
| **D** | Deliberately degraded negative control (`carelite/prompts/condition_d.v1.md`) — instructed to be brief, avoid dwelling on feeling, avoid open questions, and close topics quickly. Degraded on the communication dimensions the rubric scores, **not** on safety: the same output-safety gate and constraints apply as every other condition, so a D response the safety gate blocks is a real failure, not the control working as intended. |

## 3. Primary outcome

**Composite NURSE adherence, Condition A vs. Condition B.** Computed as the per-generation mean of
`to_quality()` applied to the five NURSE dimensions (`name`, `understand`, `respect`, `support`,
`explore`; `carelite/eval/rubric/dimensions.py`), aggregated to one value per scenario × condition
by averaging across the 3 samples in that cell, per rater type.

**Directional hypothesis:** Condition B (framework-prompted) scores higher composite NURSE
adherence than Condition A (bare model). This is the structural-adherence effect build plan v3
expects to be large (§11).

## 4. Secondary outcomes, each with a directional hypothesis

1. **Composite Four Habits adherence** (`ib`, `epp`, `de`, `ie`, mean of `to_quality()`), Condition
   A vs. B — same direction as the primary outcome, same mechanism.
2. **Composite NURSE adherence, Condition B vs. Condition C** — hypothesized C > B: retrieval adds
   grounded, evidence-specific guidance beyond framework prompting alone. Expected effect size is
   smaller than A vs. B (v3 §11); this is the comparison that sets statistical power (§7 below).
3. **Composite NURSE adherence, Condition C vs. Condition LC** — hypothesized C ≥ LC: curated
   retrieval outperforms or matches naive long-context stuffing, per build plan Part I's argument
   against full-corpus stuffing as a retrieval substitute.
4. **`naturalness`**, Condition A vs. Condition B — hypothesized **A > B**. This is the finding the
   ordering dependency in this document exists to protect: build plan v3 predicts framework
   prompting induces formulaic, script-like output, and an against-the-system naturalness result
   is exactly the kind of finding a post-hoc write-up cannot be trusted to report honestly without
   this document existing first.
5. **`ritualistic`** (reverse-coded; higher raw score is worse), Condition A vs. Condition B —
   hypothesized B has a higher raw `ritualistic` score than A, the mechanism behind hypothesis 4.
   Analyzed on `to_quality()`, which inverts this to "B scores lower quality than A," consistent
   with hypothesis 4's direction. `docs/rubric.md`'s reverse-coding warning applies to every
   aggregation step here without exception.
6. **Composite NURSE adherence, Condition A vs. Condition A2** — hypothesized no significant
   difference; a cross-model baseline check, not expected to show the study's effects.
7. **Composite NURSE adherence, Condition B vs. Condition D** — hypothesized **B > D** by a large
   margin. This is the negative-control check specified in v3 §14: if the rubric cannot separate B
   from a deliberately degraded prompt, the rubric is not measuring the construct it claims to.

## 5. The 11 rubric dimensions and the one sanctioned aggregation rule

Fixed by `RUBRIC_DIMENSIONS` in `carelite/types.py` and defined with anchored examples in
`docs/rubric.md` (rubric version 1.0.0): `name`, `understand`, `respect`, `support`, `explore`
(NURSE); `ib`, `epp`, `de`, `ie` (Four Habits); `naturalness`; `ritualistic`. Every dimension is a
1–5 integer, blinded, rater-assigned.

**`ritualistic` is reverse-coded: 5 is the worst score, not the best.** The only sanctioned way to
put a mixed set of dimensions on a common polarity before averaging, summing, correlating, or
ranking them is `carelite.eval.rubric.dimensions.to_quality(key, raw)`, which returns `6 - raw` for
`ritualistic` and passes every other dimension through unchanged. No analysis in this study
computes a composite from raw values without this transform first. `tests/unit/rubric/test_reverse_coding.py`
pins the direction against the calibration set; a failure there is never resolved by flipping a
constant.

## 6. n and its justification

Power analysis (v3 §11), paired design, Wilcoxon signed-rank, α = 0.05, power = 0.80:

| Effect size | Scenarios needed |
|---|---|
| Large (d ≈ 0.8) | ~15–20 |
| Medium (d ≈ 0.5) | ~35–45 |
| Small (d ≈ 0.3) | ~90+ |

The comparison expected to show the *smallest* effect (secondary outcome 2, B vs. C) is what sets
n, not the comparison the study cares most about. **Frozen at 100 scenarios: 40 train / 60
held-out** (`carelite/config.py`, `Experiment.n_scenarios_train = 40`,
`n_scenarios_holdout = 60`). All confirmatory analyses below run on the 60-scenario held-out split
only; the 40 train scenarios are for prompt and retrieval development and are never scored as
evaluation data.

**Samples per cell: 3** (`Experiment.samples_per_cell = 3`), enabling the variance-decomposition
model in §8. **6 conditions × 60 holdout scenarios × 3 samples = 1,080 generations** for the full
evaluation run.

## 7. The frozen holdout set

`carelite/scenarios/freeze.py` makes the 60-scenario holdout write-once: a sha256 checksum over
the canonical serialization of every held-out record's frozen fields (`scenario_id`, `text`,
`challenge_type`, `emotion_intensity`, `encounter_phase`, `literacy_signal`, `equity_stratum`,
`equity_kind`), checked on every `make check` run, with no ordinary code path or CLI command able
to rewrite the lock file.

```
HOLDOUT_DIGEST = 5a3cb128effc78f6ec41a5a8c616e2fe0fe4105abe42cc593d5dff01cd653395
```

This is the digest **after** `DECISIONS.md` D2's amendment (SC-077 and SC-010 reclassified out of
the equity stratum) and **before** registration. It is the digest that gets registered. Any change
to this constant after OSF registration is a protocol deviation and must be declared as such,
regardless of how minor the stated reason.

Holdout composition, queried directly from the loaded database at time of writing: 60 scenarios,
spanning `challenge_type` ∈ {adherence_barrier, decision_conflict, emotional_cue,
false_comprehension, family_override, information_overload, jargon_question, misplaced_blame,
prognosis_request, trust_rupture}, `encounter_phase` ∈ {opening, information_gathering,
explanation, planning, closing}, `literacy_signal` ∈ {high_health_fluency, low_health_literacy,
numeracy_gap, unmarked}, and `emotion_intensity` ∈ {1..5}. 35 of the 100 total scenarios (train +
holdout) carry `equity_stratum = true`: 16 `ses`, 10 `lep`, 9 `racial_ethnic`
(`scenarios/EQUITY_REVIEW.md`).

## 8. Statistical analysis plan

1. **Primary test family:** Friedman omnibus test across conditions {A, B, C} for each of the 11
   rubric dimensions (on `to_quality()`-transformed scores), followed by Wilcoxon signed-rank on
   every pairwise comparison listed in §4, followed by **Holm–Bonferroni correction applied across
   the whole pairwise-comparison-by-dimension family** — not per dimension separately. A dimension
   is not tested in isolation from the others for correction purposes.
2. **Effect sizes with 95% bootstrap confidence intervals are computed and reported for every
   comparison, and are reported before the corresponding p-value in every table and figure.** At
   n = 60 the effect size and its interval carry more information than the p-value; the ordering in
   the write-up reflects that.
3. **Variance decomposition.** With 3 samples per scenario–condition cell, a mixed-effects model
   with a random intercept for scenario separates within-scenario generation variance from
   between-condition effect, rather than treating the 3 samples as independent observations.
4. **Pre-specified subgroup: the equity stratum** (35 scenarios, `equity_stratum = true`), analyzed
   as a secondary analysis using the same test family as §8.1, restricted to holdout scenarios in
   the stratum. Reported and interpreted using the corrected description of what `racial_ethnic`
   actually measures — response to anticipated dismissal and patient credibility-management, per
   `DECISIONS.md` D5 — not the disparity label the axis is named for. Two gaps are pre-specified as
   limitations of this subgroup analysis rather than repaired: the stratum has no
   `emotion_intensity = 1` scenario, and `racial_ethnic` has no `adherence_barrier`,
   `decision_conflict`, or `false_comprehension` scenario (`scenarios/EQUITY_REVIEW.md`). All other
   subgroup analyses (by `challenge_type`, `literacy_signal`, `encounter_phase`, or any other
   stratification not listed here) are exploratory.
5. **Sensitivity analyses.** The primary analysis (§8.1) is re-run three ways, and whether the
   conclusions hold under each is reported:
   - (a) judge-only ratings vs. human-only ratings, once human ratings exist;
   - (b) with and without turns where Condition C's CRAG gate fell back to Condition-B behavior
     (`retrieval_trace.fell_back_to_b`), since a fallback turn is not really testing retrieval;
   - (c) excluding scenarios where judge self-consistency (§9) was poor, to check whether the
     headline conclusions depend on the judge's least stable items.
6. **Negative control.** Secondary outcome 7 (B vs. D) is the check: if the rubric cannot separate
   Condition B from the deliberately degraded Condition D, that is reported as a rubric validity
   failure, not explained away.

## 9. LLM-as-judge validation, and its consequence

The judge (`gpt-oss:20b`, cross-family from the `gemma4:12b`/`qwen3.5:9b` generators, satisfying
v3 §13's independence requirement) is validated as a component study before its scores are treated
as confirmatory, per `carelite/eval/judge/validation.py`:

- **Self-consistency:** 5 samples at temperature 0.7 on the validation subset, median score,
  inter-sample variance reported as a stability metric.
- **Positional bias:** re-run with response-order anchors reversed; a dimension whose score moves
  under reversal is flagged.
- **Span grounding, two separate rates:** the automatic rate (how often a cited evidence span can
  be located verbatim in the response — free, mechanical) and the support rate (how often a located
  span actually justifies the score attached to it — a manual spot-check of **30 spans**, per v3
  §13, `N_SPANS_TO_REVIEW = 30`).
- **Validity, per dimension:** Krippendorff's α (ordinal) and Spearman's ρ between judge and human
  consensus, computed separately for each of the 11 dimensions rather than as one overall number —
  because the judge is expected to be reliable on structural dimensions and weaker on
  `naturalness`, and a single pooled number would hide exactly the distinction this study needs.

**Pre-specified threshold, fixed in `carelite/eval/judge/validation.py` before any validation data
exists:**

> A dimension's judge scores may be reported as **confirmatory** only if Krippendorff's α ≥
> **0.667**, Spearman's ρ ≥ **0.5**, and the paired sample has ≥ **30 units**
> (`MIN_ALPHA_FOR_CONFIRMATORY`, `MIN_RHO_FOR_CONFIRMATORY`, `MIN_UNITS_FOR_CONFIRMATORY`). A
> dimension that fails any part of this — including an undefined (`NaN`) coefficient, which is
> treated as failing rather than as missing evidence of disagreement — has its judge-only results
> demoted to exploratory and reported as exploratory in the sentence that states them, not only in
> a limitations paragraph. This is recorded whichever way it comes out, for every one of the 11
> dimensions, not selectively for the ones that pass.

**The deliberate split that keeps judging tractable:** the full 1,080-generation run is judged
single-pass at **temperature 0** (`judge_temperature_full_run = 0.0`,
`judge_samples_full_run = 1`); the 5-sample self-consistency check at **temperature 0.7**
(`judge_temperature_validation = 0.7`, `judge_samples_validation = 5`) runs only on the smaller
validation subset used for the human-agreement study. This keeps the full run affordable (v3
targets ~8h, not the ~35h that 5× sampling on every generation would cost) while still measuring
judge stability on the subset that needs it.

## 10. Exclusion criteria

- A generation that errors out (empty response, model/runtime failure) is regenerated once with the
  same deterministic seed (`carelite.config.seed_for`); a second failure excludes that cell and
  logs it by `(scenario_id, condition, sample_idx)` rather than silently dropping it from the
  denominator.
- A generation blocked outright by the output-safety gate (`carelite/safety/output_gate.py`) —
  as opposed to a Condition-D response that is merely terse and low-warmth by design — is excluded
  from rubric analysis and logged as a safety event. The safety gate is checking something the
  rubric does not score, and a real safety block is a different kind of event from a low
  `naturalness` score.
- A judge score for a specific dimension with no locatable evidence span (v3 §13 grounding
  requirement) is treated as missing for that dimension only; it is not imputed and it is not
  treated as a 1.
- No scenario, condition, or generation is excluded on the basis of its score. There is no
  post-hoc trimming.

## 11. Stopping rule

n is fixed by the power analysis in §6 before any run starts; there is no interim analysis and no
optional stopping. The full run is complete when all 1,080 holdout generations exist and have been
judged. The human-rating validation subset (§9) is a fixed random sample of ≥ 30 units, drawn
before any human rating begins and not enlarged based on how the agreement numbers look partway
through.

## 12. Human evaluation

Per v3 §12: a stratified sample of 60 responses (20 scenarios × 3 conditions), condition labels
stripped, presentation order randomized per rater, a written rubric with anchored examples
(`docs/rubric.md`) distributed before rating, and a calibration set of 5 responses scored and
discussed first (`carelite/eval/rubric/calibration.py`). Krippendorff's α (ordinal) is computed for
human–human agreement and reported whatever value it takes. **As of this writing, no human rating
has occurred**; the harness is built and exercised against synthetic rater data
(`carelite/eval/human/synthetic.py`) specifically so that a blinding bug or a reversed
`ritualistic` column is caught before a real rater's time is spent, not after.

## 13. Registration checklist

- [ ] `HOLDOUT_DIGEST` above matches `carelite.scenarios.freeze.HOLDOUT_DIGEST` at the moment of
      registration.
- [ ] Rubric version (`docs/rubric.md`, currently 1.0.0) and judge validation-plan version
      (`carelite.eval.judge.validation.VALIDATION_PLAN_VERSION`, currently `1.0.0`) match what is
      pasted into the OSF template.
- [ ] Registered on OSF, timestamped, before `carelite.repro` or any inference-lane script
      generates a single holdout response.
- [ ] Registration URL recorded in `docs/decisions/` as a dated entry once complete.

---

*This document is owned by `carelite-repro` per its lane definition. It draws its constants
directly from `carelite/config.py`, `carelite/types.py`, `carelite/scenarios/freeze.py`,
`carelite/eval/judge/validation.py`, `docs/rubric.md`, and `DECISIONS.md`, rather than restating
planning-time figures, so a future amendment to any of those files should prompt a re-check of the
corresponding section here.*
