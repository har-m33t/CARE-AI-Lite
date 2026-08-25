# Analysis Plan — Kept as a Historical Record, Not a Registration

**Status: SUPERSEDED. Per `DECISIONS.md` D10 (2026-08-24, project owner), OSF registration is
dropped and every result this project produces is descriptive.** This project is a personal proof
of concept that needs to work locally — it is not being published, submitted, or handed to anyone
else, and pre-registration existed to serve one argument (build plan v3 §10) that only pays off with
an audience. There is none, so the registration gate was removed rather than kept as a formality.

**This document is kept, not deleted, because D10 says explicitly why it is still worth something:**
it remains an accurate, git-timestamped record of the analysis plan as it stood before any holdout
data existed, and the analysis below still runs exactly as specified — Holm correction, two-sided
tests, bootstrap CIs reported before p-values, the weakest-link composite rule, judge-agreement
demotion, all of it, because that discipline is what keeps this project from fooling itself,
independent of whether anyone else is watching. What changed is only what this document may be
*claimed as*: **nothing below — the primary outcome, the eight comparisons, the equity subgroup, the
naturalness prediction — may be described as confirmatory, pre-specified, or hypothesis-testing, in
this document, in `docs/limitations.md`, in `README.md`, or in any write-up this project produces.**
Every one of them is an observation from a single local run. Where the text below still says
"primary," "confirmatory," or "registered," read it as *the plan as originally specified* — history,
not a claim about the status of a result. D1–D9 all stand; nothing about the corpus, the knowledge
base, the equity findings, or the instrument defects this document and `docs/limitations.md`
describe becomes less true because the audience changed. The one thing genuinely lost: if this
project is ever written up for a real audience later, the naturalness result cannot be reclaimed as
pre-specified after the fact — registering post hoc would be worse than never registering.

---

## What this document was written to prevent, kept for the record

**This document was drafted to be registered on OSF before any of the 1,080 holdout generations
existed — that registration never happened, by decision, not by delay.** Build plan v3 §10 states
the argument the plan below still follows even without a registry behind it: *"Pre-registration is
what makes the naturalness result credible if it goes against you. Without it, 'Condition A beat B
on naturalness' reads as a post-hoc excuse. With it, it reads as a pre-specified secondary outcome
that came out the interesting way."* Without the registry, a naturalness result against the system
reads as an honestly-run local observation rather than a registered finding — a weaker claim, stated
as what it now is rather than dressed as what it cannot be.

**Everything not explicitly listed as primary, secondary, or a specified sensitivity analysis below
was exploratory under the original plan and remains so** — but per D10, "primary" and "secondary"
themselves no longer carry registered, confirmatory weight; they describe what the plan prioritized,
not a status this run's results actually hold.

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
| **LC** | **`LC-sample`, per `DECISIONS.md` D7.** Not the whole corpus: 471 chunks is ~326,526 tokens against a 128,000-token window (255% utilisation), so the corpus does not fit. LC-sample is a fixed, **query-independent** round-robin selection across all 33 papers at a pinned seed (`carelite.retrieval.ablation.lc_sample`) — 169 chunks, 35.9% of the corpus, guaranteeing every paper is represented rather than risking random sampling dropping one by chance. Generator: `gemma4:12b`, 128k context window. **Any selection rule is a form of retrieval**, so LC no longer asks whether retrieval beats stuffing everything in — it asks whether *query-dependent* selection (Condition C) beats a *fixed* context (LC-sample). That is a real and interesting question, and a different one from what build plan v3 §3 posed; this document states the difference rather than letting a reader infer it. |
| **D** | Deliberately degraded negative control (`carelite/prompts/condition_d.v1.md`) — instructed to be brief, avoid dwelling on feeling, avoid open questions, and close topics quickly. Degraded on the communication dimensions the rubric scores, **not** on safety: the same output-safety gate and constraints apply as every other condition, so a D response the safety gate blocks is a real failure, not the control working as intended. |

## 3. Primary outcome

**Composite NURSE adherence, Condition A vs. Condition B.** Computed as the per-generation mean of
`to_quality()` applied to the five NURSE dimensions (`name`, `understand`, `respect`, `support`,
`explore`; `carelite/eval/rubric/dimensions.py`), aggregated to one value per scenario × condition
by averaging across the 3 samples in that cell, per rater type.

**Directional hypothesis:** Condition B (framework-prompted) scores higher composite NURSE
adherence than Condition A (bare model). This is the structural-adherence effect build plan v3
expects to be large (§11).

**Constraint on this outcome, stated in advance per `DECISIONS.md` D8: two of the five constituent
dimensions have zero knowledge-base grounding.** `respect` and `support` have no entries anywhere
in the 116-entry knowledge base — nothing in the 33 retrieved papers turns NURSE Respecting
(crediting the patient for something specific) or Supporting (partnership made concrete: who does
what, how to reach someone) into a finding with a quotable span and an actionable takeaway. Full
behavior-to-framework coverage across the 116 entries: `ie` 40, `epp` 17, `name` 15, `ib` 6,
`explore` 5, `understand` 5, `de` 4, **`respect` 0, `support` 0**, with 40 entries instantiating
none of the nine components. The zeros are not an artifact of an under-tuned matcher: an earlier
mapping filled both dimensions and looked plausible until every assignment was read against its
source entry, which found seven false positives, two of them exactly here — *"verbal affirmations
to show you're listening"* is a backchannel cue, not crediting the patient for anything specific,
and *"collaborative partnership"* is a stance with none of Supporting's concrete half. Removing
those false positives is what took both dimensions to zero, and a regression test pins each zero
while a companion test confirms the matcher still fires on a genuine crediting move, so a future
change that fills them has to argue with a failing test, not drift into one.

**Consequence for this outcome, stated precisely so it is a constraint rather than an excuse:**
Condition C's retrieval pipeline cannot ground two of the five dimensions the primary composite
averages. Any C-over-B advantage on `respect` or `support` therefore has some cause other than
retrieval — prompt framing, generator behavior, or noise — and the same limitation propagates to
secondary outcome 2 below (NURSE, B vs. C), which is the comparison this study's power analysis is
built around. **The composite is not being redefined to drop these two dimensions.** Choosing which
outcome dimensions to keep after seeing which ones the evidence base happens to support is exactly
what pre-registration exists to prevent, and the rubric measures what a clinician does regardless
of whether this corpus can teach it. Note also that this is a statement about the *evidence base*,
independent of the judge-agreement mechanism in §9 — see §9's closing note for how the two
constraints combine on this same composite.

## 4. Secondary outcomes, each with a directional hypothesis

1. **Composite Four Habits adherence** (`ib`, `epp`, `de`, `ie`, mean of `to_quality()`), Condition
   A vs. B — same direction as the primary outcome, same mechanism.
2. **Composite NURSE adherence, Condition B vs. Condition C** — hypothesized C > B: retrieval adds
   grounded, evidence-specific guidance beyond framework prompting alone. Expected effect size is
   smaller than A vs. B (v3 §11); this is the comparison that sets statistical power (§7 below).
   **Inherits §3's D8 constraint**: two of this composite's five dimensions (`respect`, `support`)
   have no knowledge-base grounding, so any C advantage here is bounded to at most three-fifths of
   the composite being plausibly attributable to retrieval.
3. **Composite NURSE adherence, Condition C vs. Condition LC-sample** — hypothesized C ≥ LC-sample:
   query-dependent retrieval outperforms or matches a fixed, query-independent context sample.
   **This is not the comparison build plan Part I's argument against full-corpus stuffing
   describes**, per `DECISIONS.md` D7: the corpus (471 chunks, ~326,526 tokens) does not fit the
   128k context window at 255% utilisation, so LC-sample is a round-robin 169-chunk (35.9%) sample
   across all 33 papers at a pinned seed, not the whole corpus. The comparison this hypothesis
   actually tests is curated, query-dependent selection against a fixed context — a real question,
   and a narrower one than "retrieval vs. no retrieval." See §2's LC row for the full statement.
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

1. **Primary test family: exactly the eight comparisons registered in §3 and §4, each on its own
   named measure, per `DECISIONS.md` D9(2).** (1) composite NURSE, A vs. B [§3, primary]; (2)
   composite Four Habits, A vs. B; (3) composite NURSE, B vs. C; (4) composite NURSE, C vs.
   LC-sample; (5) `naturalness`, A vs. B; (6) `ritualistic`, A vs. B; (7) composite NURSE, A vs. A2;
   (8) composite NURSE, B vs. D. Friedman omnibus across {A, B, C} where applicable, then Wilcoxon
   signed-rank on each of the eight, then **Holm–Bonferroni correction applied across this
   eight-comparison family as a whole** — not per dimension separately, and not diluted by folding
   in comparisons this document never registered (an unregistered A vs. C, or a per-dimension `de`
   test for A vs. B, are exploratory, not part of this family). **Every test is two-sided**: a
   one-sided test aimed at the registered direction would have no power against outcome 4's
   against-the-system prediction (A > B on `naturalness`), which is the finding the ordering-
   dependency warning at the top of this document exists to protect. The registered direction is recorded and compared against the
   observed one, so a significant result in the *opposite* direction stays detectable and
   reportable rather than becoming statistically invisible.
2. **A separate 5-conditions × 11-dimensions grid (55 cells) is also computed, corrected within its
   own family, and stamped EXPLORATORY throughout — never confirmatory, regardless of significance.**
   Both the eight-comparison family and the 55-cell grid are reported; only the eight carry
   confirmatory weight.
3. **Effect sizes with 95% bootstrap confidence intervals are computed and reported for every
   comparison, and are reported before the corresponding p-value in every table and figure.** At
   n = 60 the effect size and its interval carry more information than the p-value; the ordering in
   the write-up reflects that. **All three point estimators are reported for every comparison,
   unconditionally, per `DECISIONS.md` D9(5):** rank-biserial correlation (Wilcoxon's own effect
   size), Cohen's dz (the scale §6's power analysis is expressed in), and the Hodges–Lehmann
   location shift. Reporting all three removes the opportunity to choose whichever reads largest
   after seeing the numbers.
4. **Variance decomposition.** With 3 samples per scenario–condition cell, a mixed-effects model
   with a random intercept for scenario separates within-scenario generation variance from
   between-condition effect, rather than treating the 3 samples as independent observations.
5. **Pre-specified subgroup: the equity stratum, held-out n = 20 — corrected per `DECISIONS.md`
   D9(1), and analyzed as descriptive rather than confirmatory.** §7 above reports 35 scenarios
   carry `equity_stratum = true` across the *full 100-scenario bank*; that count is correct for the
   bank but was previously, incorrectly, implied to be the confirmatory subgroup's n. §6 restricts
   all confirmatory analyses to the 60-scenario holdout, and only 20 of the 35 equity scenarios are
   in it (`ses` 10, `lep` 4, `racial_ethnic` 6) — the other 15 sit in the train split and are never
   scored as evaluation data. **At n = 20 this subgroup resolves only large effects (dz ≈ 0.68);
   `racial_ethnic` at n = 6 supports no statistical claim at all.** This is stated as a descriptive
   analysis, not a powered confirmatory test, and the write-up must say so in those words rather
   than imply a powered comparison the sample cannot support. **This is the third independent
   measurement of the same underlying limitation of this evidence base**, alongside `DECISIONS.md`
   D3 (the equity knowledge-base theme holds 3 entries as a property of the corpus, not the
   extraction) and D5 (the `racial_ethnic` scenario axis narrows to a single mechanism — anticipated
   dismissal and patient credibility-management — rather than the disparity its label names). All
   three belong stated together as one finding about this evidence base, not as three separate
   caveats scattered across the document: **this corpus and this scenario bank measure disparity in
   clinician communication considerably more thoroughly than they measure how to close it, and the
   equity analysis throughout this study is descriptive for that reason.** Reported and interpreted
   using D5's corrected description of what `racial_ethnic` actually measures, not the disparity
   label the axis is named for. Two further coverage gaps remain pre-specified as limitations of
   this subgroup rather than repaired: the holdout equity stratum has no `emotion_intensity = 1`
   scenario, and `racial_ethnic` has no `adherence_barrier`, `decision_conflict`, or
   `false_comprehension` scenario (`scenarios/EQUITY_REVIEW.md`). All other subgroup analyses (by
   `challenge_type`, `literacy_signal`, `encounter_phase`, or any other stratification not listed
   here) are exploratory.
6. **Sensitivity analyses.** The primary analysis (§8.1) is re-run three ways, and whether the
   conclusions hold under each is reported:
   - (a) judge-only ratings vs. human-only ratings, once human ratings exist;
   - (b) with and without turns where Condition C's CRAG gate fell back to Condition-B behavior
     (`retrieval_trace.fell_back_to_b`), since a fallback turn is not really testing retrieval;
   - (c) excluding scenarios where judge self-consistency (§9) was poor, defined precisely per
     `DECISIONS.md` D9(4) as **`pct_range_ge_2 > 0.25`** — a generation is excluded from this
     sensitivity analysis when more than a quarter of its judge samples span two or more rubric
     points, which is disagreement about *which anchor applies* rather than rounding between
     adjacent ones. `threshold_prespecified = True` once this number is in this document; before
     that it correctly printed `False` with its reason.
7. **Negative control.** Secondary outcome 7 (B vs. D) is the check: if the rubric cannot separate
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

**A composite is confirmatory only if every constituent dimension is, per `DECISIONS.md` D9(6) —
weakest-link, not majority-rule.** If any one of a composite's constituent dimensions is demoted to
exploratory by this section's threshold, the composite built from it is reported as exploratory
too, not as "confirmatory with one soft edge." A composite's provenance is mixed the moment one
input's provenance is, and the honest label is the weaker one.

**How this combines with §3's D8 constraint on the primary composite, stated explicitly because the
two are easy to conflate and are not the same thing.** D8 is a statement about the *evidence base*:
`respect` and `support` have zero knowledge-base grounding, so retrieval cannot be the mechanism
behind any Condition C advantage on them. That is independent of this section's *judge-agreement*
mechanism — the judge can score `respect` or `support` reliably (clearing this section's
Krippendorff's-α / Spearman's-ρ threshold) whether or not the corpus can teach the system to
perform them well. Neither constraint implies the other, and both apply to the same primary
composite simultaneously: **two of its five dimensions cannot be helped by retrieval regardless of
judge agreement (D8), and any of the five — including those two — may additionally be demoted to
exploratory on judge-agreement grounds under this section's threshold and D9(6)'s weakest-link
rule.** A primary-composite result that survives both constraints is a materially stronger claim
than one that survives only one of them, and the write-up must show which is which rather than
reporting a single composite number.

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

## 13. Incorporation checklist (registration items retired by D10)

This section was a registration checklist; per `DECISIONS.md` D10 there is no registration to
complete, so it now tracks whether the plan below stayed internally consistent as decisions landed,
which is the part of this checklist's original job that still matters.

- [x] `HOLDOUT_DIGEST` above matches `carelite.scenarios.freeze.HOLDOUT_DIGEST`
      (`5a3cb128…653395`, confirmed against the live constant).
- [x] Rubric version (`docs/rubric.md`, 1.0.0) and judge validation-plan version
      (`carelite.eval.judge.validation.VALIDATION_PLAN_VERSION`, `1.0.0`) match what this document
      cites.
- [x] `DECISIONS.md` D7 (LC-sample) incorporated: §2's LC row and §4 outcome 3 state that LC is a
      fixed 169-chunk (35.9%) round-robin sample, not the whole corpus, and what the C-vs-LC
      comparison actually tests as a result.
- [x] `DECISIONS.md` D8 (zero KB grounding for `respect`/`support`) incorporated: §3 states the
      constraint on the primary composite before any evaluation data exists, and §9 states how it
      combines with judge-agreement demotion.
- [x] `DECISIONS.md` D9's six analysis specifications incorporated: the eight-comparison Holm family
      named explicitly (§8.1) with the 55-cell grid stamped exploratory (§8.2); two-sided testing
      stated (§8.1); the equity subgroup corrected to holdout n = 20 and labeled descriptive (§8.5);
      the `pct_range_ge_2 > 0.25` self-consistency threshold given a number (§8.6c); all three point
      estimators specified as always-reported (§8.3); the weakest-link composite rule stated (§9).
- [x] `DECISIONS.md` D10 incorporated: this document's status header states the plan is kept as a
      historical record rather than an active registration, and every result described anywhere in
      this project's documentation is labeled descriptive, not confirmatory.
- [x] The §13 judge-validation study's completed n = 30 findings (`ritualistic` degenerate,
      variance bounding agreement at r = 0.878, all eleven dimensions exploratory for want of a
      human comparator rather than as a failure — `runs/judge/validation_report.json`) are recorded
      in `docs/limitations.md` §4, where the results actually live, rather than duplicated here.
- [x] **OSF registration retired by decision, not left pending.** `DECISIONS.md` D10: this project
      will not be registered. No URL exists to record because none will be sought.

---

*This document is owned by `carelite-repro` per its lane definition. It draws its constants
directly from `carelite/config.py`, `carelite/types.py`, `carelite/scenarios/freeze.py`,
`carelite/eval/judge/validation.py`, `docs/rubric.md`, and `DECISIONS.md`, rather than restating
planning-time figures, so a future amendment to any of those files should prompt a re-check of the
corresponding section here.*
