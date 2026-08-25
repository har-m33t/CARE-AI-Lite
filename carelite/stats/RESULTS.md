# CARELite AI — results of the holdout evaluation

**Every result in this document is descriptive.** `DECISIONS.md` D10 dropped the pre-registration:
this is a local proof of concept, it was never registered, and nothing here is confirmatory,
pre-specified in the registered sense, or hypothesis-testing. `docs/preregistration.md` is kept as a
timestamped-in-git record of the analysis plan as it stood before any holdout data existed, and the
analysis follows it — but the plan's standing is "written down first", not "registered".

Regenerate everything below with `python -m carelite.stats`. Every number traces to a query in
`carelite/stats/data.py` and a test in `tests/unit/stats/`.

---

## 1. What was run

The holdout run produced **939 generations**: 60 held-out scenarios × 3 samples × five conditions
(A, A2, B, C, D), plus 39 partial LC cells. All 939 were judged by `gpt-oss:20b` at temperature 0,
single pass, and loaded into `rubric_score` as one aggregate row per generation.

Condition LC is **dropped** and the analysis runs on **900 generations**. The five analysed
conditions are complete: 180 cells each, no gaps.

| Exclusion | Count | Treatment |
|---|---|---|
| LC cells (D11) | 39, over 13 of 60 scenarios | Dropped entirely. See §6. |
| Output-gate refusals (D12) | 17 generations | Reported both ways. See §5. |
| CRAG fell back to B | 69 of 180 C cells (38%) | Reported both ways. See §4. |
| Incomplete on ≥1 dimension | 46 of 900 generations | Kept; the missing dimension only is dropped. |

**Missing data.** 204 of 9,900 (generation, dimension) cells are unscored — between 10 and 22 per
dimension, scattered rather than concentrated. Following analysis plan §10, an unscored dimension is
missing *for that dimension only*: it is not imputed and not treated as a 1, the generation keeps its
composite computed from the dimensions that were scored, and `n_dimensions` records how many
contributed. Dropping the whole generation instead would have discarded 46 otherwise-good rows to
recover 204 cells; imputing would have invented data. (The judge lane counts 50 incomplete rows
across all 939; the four-row difference is LC.)

---

## 2. The finding that governs everything else: three dimensions were not measured

The judge did not resolve three of the eleven rubric dimensions. Measured on the analysed
(`to_quality`) scale:

| Dimension | n | sd | modal share | distinct values |
|---|---|---|---|---|
| `ritualistic` | 883 | **0.16** | 99.0% on one value | 3 of 5 |
| `ie` | 882 | **0.49** | 93.3% on one value | 4 of 5 |
| `naturalness` | 884 | **0.59** | 85.9% on one value | 5 of 5 |
| *the other eight* | 878–890 | 1.00–1.85 | 47–80% | 5 of 5 |

**This is not a null result about the system. It is a null result about the instrument.** A paired
test ranks within-scenario differences; when the judge gives nearly every response the same score,
nearly every difference is exactly zero and there is nothing to rank. The test returns a large
p-value that looks exactly like "the conditions do not differ" and means something entirely
different.

**No sample size repairs it.** The `carelite-judge` lane established the mechanism independently:
ordinal Krippendorff's alpha is bounded above by the variance the judge itself produces (r = 0.878).
A judge that emits one value cannot disagree with itself, cannot agree with a human, and cannot
separate two conditions. A larger holdout would produce the same floor with narrower intervals
around it.

**The cost is the study's most interesting prediction.** Build plan v3 predicts Condition B loses to
A on `naturalness` *because framework prompting induces ritual*. Both dimensions carrying that
prediction — the outcome and its stated mechanism — are degenerate here. **This run can neither
support nor refute it.** §3 gives the numbers and says why they cannot be read as the v3 result.

**On the classification rule, and a mistake worth recording.** A dimension is called degenerate when
its standard deviation is below 0.75 rubric points. The rule was chosen after seeing the data and is
labelled a post-hoc diagnostic in the output itself. The first version used modal share instead, and
it was wrong: `name` puts 77% of its mass on one value and the other 23% at the *opposite* end of the
scale (683 scores of 1, 150 of 5), which is a dimension discriminating about as hard as a five-point
scale can. Concentration is not degeneracy; absence of spread is. The guard that caught this —
`threshold_sensitivity()`, which re-runs the classification across a range of cuts — is the same
guard D3 required of the aspiration filter, and it earned its place. The eleven dimensions fall into
two groups with an empty band between them (0.16, 0.49, 0.59, then nothing until 1.00, 1.15, 1.41,
1.53, 1.57, 1.61, 1.75, 1.85); every cut from 0.60 to 0.95 gives the same three. The threshold is not
doing the work.

---

## 3. The eight comparisons

Holm-Bonferroni is applied across the **whole family of eight comparisons at once — across measures
and dimensions together, not per dimension**. Seven were computed; C vs LC keeps its slot (§6). All
tests are two-sided. Effect sizes and their 95% bootstrap CIs come before p-values throughout, and
all three point estimators are reported for every comparison.

Effects are **left minus right**, so a negative effect means the right-hand condition scored higher.

| # | Comparison | rank-biserial [95% CI] | Hodges–Lehmann (points) | Holm *p* | Predicted | Observed |
|---|---|---|---|---|---|---|
| P | A vs B, NURSE | −0.866 [−0.964, −0.719] | −0.667 [−0.833, −0.467] | 1.2×10⁻⁷ | B higher | **as predicted** |
| 1 | A vs B, Four Habits | −0.744 [−0.916, −0.525] | −0.708 [−0.917, −0.500] | 4.1×10⁻⁶ | B higher | **as predicted** |
| 2 | B vs C, NURSE | +0.214 [−0.091, +0.488] | +0.100 [−0.067, +0.267] | 0.46 | C higher | **against** |
| 3 | C vs LC | *not computed (D11)* | — | — | — | — |
| 4 | A vs B, naturalness | −0.626 [−0.872, −0.315] | −0.167 [−0.333, **0.000**] | 0.0037 | A higher | **against**, and instrument-limited |
| 5 | A vs B, ritualistic | −0.200 [−1.000, +1.000] | 0.000 [0.000, 0.000] | 1.0 | A higher | instrument-limited |
| 6 | A vs A2, NURSE | −0.731 [−0.887, −0.533] | −0.400 [−0.567, −0.267] | 5.4×10⁻⁶ | *no difference* | **prediction fails** |
| 7 | B vs D, NURSE | +0.992 [+0.967, +1.000] | +1.114 [+0.933, +1.300] | 2.8×10⁻¹⁰ | B higher | **as predicted** |

Condition means on the NURSE composite (quality scale, 1–5): A 1.83, A2 2.27, B 2.50, C 2.37,
D 1.36.

### What holds

**Framework prompting works, and the effect is large.** B beats A on composite NURSE adherence by
two-thirds of a rubric point, with 56 of 60 scenarios moving in that direction. The same holds on
the Four Habits composite. This is the study's clearest result and it survives every rerun.

**The negative control passes decisively.** The rubric separates B from the deliberately degraded D
at rank-biserial 0.992 — 59 of 60 scenarios, a shift of 1.11 points. The measurement instrument
discriminates on the dimensions it resolves. Given §2, that verdict is worth stating precisely: the
rubric works on the eight dimensions that carry variance, and three dimensions is where it failed,
not the whole instrument.

### What does not hold

**Retrieval did not help. Condition C is not better than Condition B, and the point estimate is
slightly worse.** rank-biserial +0.214 favouring B, CI spanning zero, Holm p = 0.46 — and the
direction is the opposite of the one predicted. §4 reports this both ways and the conclusion does not
change. This is the architecture's central claim and this run does not support it.

**The cross-model baseline failed its prediction, and the size of the failure matters.** A vs A2 was
predicted to show no difference; it shows a large one (rank-biserial −0.731, 0.400 points). From the
mixed model, switching generator model buys +0.445 points over bare A, while the framework prompt
buys +0.672. **Roughly two-thirds of the framework effect is available by changing the model
instead.** That is not a reason to discount the framework result — B still beats A2 — but any claim
that the framework is what produces good communication has to be read against it.

### Why the naturalness result is not the v3 finding

Comparison 4 is, on its face, exactly what build plan v3 §10 predicted and hoped to be able to report
credibly: A scores higher than B on naturalness, against the system, at Holm p = 0.0037. **It should
not be reported that way**, for reasons visible in the same row:

- `naturalness` is degenerate — 86% of all scores are the single value 3.
- The test rests on 36 of 60 scenarios; the other 24 are tied exactly.
- The Hodges–Lehmann shift is 0.167 rubric points and **its confidence interval reaches zero**.
- The mechanism the prediction names — `ritualistic` — has 4 non-tied pairs out of 60 and is
  entirely uninformative.

So the honest statement is: *on the compressed range this judge produced, framework-prompted
responses were rated marginally less natural than bare-model responses, by about a sixth of a scale
point, on a dimension where the judge used essentially one value; and the ritual mechanism v3 offers
as the explanation could not be tested at all.* That is a real observation and it is not the
pre-registered against-the-system finding, which D10 has in any case retired. `PairwiseResult.render`
prints NOT TESTABLE above the p-value so the ordering cannot be lost in transcription.

---

## 4. Retrieval, asked two ways

CRAG graded 111 of 180 Condition-C cells `relevant` and 69 `none`. On those 69 (**38% of the arm**) C
fell back to Condition-B behaviour, so pooling compares C against itself on more than a third of its
mass.

| Reading | Question | rank-biserial [95% CI] | HL (points) | *p* (uncorrected) | n |
|---|---|---|---|---|---|
| All 180 cells | Does *offering* retrieval help? | +0.214 [−0.091, +0.488] | +0.100 | 0.155 | 60 |
| The 111 that retrieved | Does *retrieval* help? | +0.220 [−0.154, +0.553] | +0.133 | 0.245 | 37 |

Only the second is the architecture's claim, and **it gives the same answer**: no measurable benefit,
with the point estimate slightly favouring B in both. The fallback rate is not what is holding the
effect down. Restricting to the cells that retrieved costs 23 scenarios, which lose their entire C
cell — so the second row is both a weaker test and a *selected* one.

**The selection confound is real and cannot be removed after the fact.** CRAG chose which cells
retrieved, on its own judgement of relevance, so the 111 are plausibly the scenarios this corpus can
serve. Comparing them against the full Condition-B arm is a self-selected subgroup contrast, and the
bias runs *in favour of* retrieval. The result is null anyway, which makes it the more informative
direction to be wrong in.

This is consistent with what the rest of the project already established about the evidence base: 18
of 33 papers dominate the corpus, on-domain context precision runs 0.334–0.380 against a 0.7 target,
and D8 records that two of the five NURSE dimensions (`respect`, `support`) have **zero** knowledge
base entries and so cannot be helped by retrieval at all. A retrieval layer cannot add grounding the
corpus does not contain.

---

## 5. The output safety gate: reported both ways, and which is preferred

The `carelite.safety` output gate refused 17 generations. They are **not** spread evenly: 13 are on
SC-029, the remaining four one each on SC-055, SC-057, SC-072, SC-092; by condition A 3, A2 7, B 2,
C 2, D 3. Excluding them therefore removes most of one scenario across several conditions rather than
trimming symmetrically.

| Comparison | rank-biserial, refusals **included** | rank-biserial, refusals **excluded** | n |
|---|---|---|---|
| A vs B, NURSE | −0.866 (Holm 1.2×10⁻⁷) | −0.862 (Holm 1.9×10⁻⁷) | 60 → 59 |
| A vs B, Four Habits | −0.744 (Holm 4.1×10⁻⁶) | −0.741 (Holm 5.6×10⁻⁶) | 60 → 59 |
| B vs C, NURSE | +0.214 (Holm 0.46) | +0.203 (Holm 0.53) | 60 |
| A vs A2, NURSE | −0.731 (Holm 5.4×10⁻⁶) | −0.684 (Holm 3.0×10⁻⁵) | 60 → 59 |
| B vs D, NURSE | +0.992 (Holm 2.8×10⁻¹⁰) | +0.992 (Holm 4.1×10⁻¹⁰) | 60 → 59 |

**Nothing moves. No conclusion changes significance or direction.**

**The preferred reading excludes them**, and the reason is asymmetric rather than a preference for
the tidier number. Scoring refused text on a communication rubric is a *category error*, not a low
score — the gate's output is not a communication attempt, so its NURSE score measures nothing.
Excluding it costs one scenario from a 60-scenario paired analysis, which shows up honestly in the
`n` and in the interval. Prefer the loss you can see. Scoring refused text silently, which is what
would have happened before D12 added the column, is the one option that is definitely wrong.

This rerun was **not planned in advance** — D12 postdates the analysis plan — and is labelled as such
in the output.

---

## 6. What could not be found, and why

**Secondary outcome 3 (C vs LC) cannot be answered by this run.** D11 stopped LC generation at 39 of
180 cells covering 13 of 60 scenarios, on measured cost: 3.3 minutes per cell against 6 seconds for
the A/A2/D group, because Ollama re-prefills the ~119,500-token fixed prefix on every request instead
of reusing the KV cache. Those 39 cells are the scenarios LC happened to reach before it was stopped
and were **never randomised for partial analysis**. They are not a sample of anything.

So the comparison is **not computed**. Not computed on 13 scenarios with a caveat, and not computed
and reported as non-significant. `run_pairwise` refuses it before touching the data, so the presence
of LC rows in the table cannot quietly defeat the decision, and a test pins that.

**It keeps its slot in the Holm family — m = 8, not 7.** Dropping to 7 after seeing which test could
not run would lower every surviving comparison's adjusted p-value. The family is fixed before the
data exist or it is not a correction.

Note what was already lost before this: D7 established that the corpus is 255% of the context window,
so LC was never "no retrieval" but "no *query-dependent* retrieval" over a fixed round-robin sample.
The claim given up is second-order. That two independent lanes dropped LC for the same measured
reason on different hardware is itself the result worth recording: **under this architecture a
long-context baseline is not affordable to evaluate at the scale the rest of the design assumes.**

**Sensitivity analyses (a) and (c) could not be run.** (a) needs human ratings, and none exist —
every number in this document is judge-only. (c) needs the judge's per-sample rows, and the holdout
was judged single-pass at temperature 0; the five-sample self-consistency pass runs on the validation
subset only. Both are reported as "not runnable", never as "conclusions hold". An absence of a test
is not evidence of robustness.

Of the reruns that *could* be run — (b) CRAG fallback and (d) gate-blocked — **zero conclusions
moved.** A conclusion that flips under sensitivity analysis would be the finding; none flipped.

---

## 7. The three samples per cell are not three observations

Analysis plan §8.3. A random intercept for scenario separates within-scenario generation variance
from the between-condition effect.

Composite NURSE, REML, `value ~ condition + (1 | scenario)`, reference A, 900 generations over 60
scenarios:

| Term | Coefficient | 95% CI | *p* |
|---|---|---|---|
| A2 vs A | +0.445 | [+0.314, +0.577] | 3.4×10⁻¹¹ |
| B vs A | +0.672 | [+0.541, +0.804] | 1.3×10⁻²³ |
| C vs A | +0.545 | [+0.414, +0.677] | 4.7×10⁻¹⁶ |
| D vs A | −0.464 | [−0.596, −0.332] | 4.9×10⁻¹² |

Between-scenario variance 0.219, residual 0.406, **ICC 0.350** — a third of the variance in a
response's score is the scenario it answers, not the condition that produced it.

**What treating the 1,080 samples as independent would have cost:** standard errors understated by
**1.16× to 1.30×** depending on the contrast, and every p-value correspondingly too small. That ratio
is measured per contrast, not asserted as a √3 penalty — it is a measurement of how much
scenario-by-condition structure the data actually carry, and it is above 1.0 everywhere, which means
the structure is real.

---

## 8. The equity stratum

The one subgroup planned in advance (§8.4). **Descriptive only — this is not a powered test of an
equity effect and must not be written up as one.**

n = **20** held-out scenarios (`ses` 10, `racial_ethnic` 6, `lep` 4), not the 35 the plan's
parenthetical names: 35 counts the whole bank and 15 of those are in the train split. At n = 20 the
smallest resolvable paired effect is dz = 0.676 — **large effects only**. `racial_ethnic` at n = 6
supports nothing in either direction and its breakdown is printed as a description of the data, not
as a comparison.

The pattern matches the full analysis: A vs B −0.942 (Holm 0.0030), A vs A2 −0.919 (Holm 0.0024),
B vs D +1.000 (Holm 0.0011), B vs C −0.074 (Holm 1.0, noise at this n). Nothing here is evidence of
an equity-specific effect; it is the main result, re-observed on a fifth of the data.

Two limitations were declared before any data existed and stand:

- The stratum contains no `emotion_intensity = 1` scenario, so it cannot say whether the disparity
  behaves differently on an emotionally flat turn — the turn where a system that over-reads emotion
  does its worst work.
- `racial_ethnic` contains no `adherence_barrier`, `decision_conflict` or `false_comprehension`
  scenario, and every one of its scenarios presents an already-guarded patient.

Per D5, the `racial_ethnic` axis is described as what it measures: **response to anticipated
dismissal and patient credibility-management**, not race-based disparity in communication generally.
The label is kept for continuity with the frozen split; the description is not.

---

## 9. What this study can and cannot claim

**Can claim, on this run, descriptively:**

1. Framework prompting (B) substantially outperforms the bare model (A) on composite NURSE and Four
   Habits adherence, by roughly two-thirds of a rubric point, consistently across scenarios.
2. The rubric separates the framework condition from a deliberately degraded one at near-ceiling
   effect size, so it is measuring something real on the eight dimensions that carry variance.
3. Adding retrieval (C) produced no measurable improvement over framework prompting alone, whether
   or not the fallback cells are included.
4. Changing the generator model produces an effect about two-thirds the size of the framework
   prompt's, so model family is a substantial confound in any claim about prompt architecture.
5. A third of score variance is scenario, not condition — the three samples per cell are not three
   observations, and analysing them as such would understate every standard error by 1.16–1.30×.

**Cannot claim:**

1. Anything about naturalness or ritual. The instrument did not resolve either dimension, and the
   central v3 prediction that connects them is untestable on this data.
2. Anything about `ie` (invests in the end), for the same reason.
3. Anything about curated retrieval versus long-context stuffing. Not computed; LC is not a sample.
4. That the judge's scores agree with human judgement, on any dimension. No human rating exists, so
   every dimension is exploratory under the §9 gate and every result in this document says so in its
   own label.
5. Anything confirmatory or pre-registered, about anything. D10.

**The two failures that matter most are both about measurement, not about the system.** Three
dimensions were not resolved by the judge, and the one comparison that would have tested the
retrieval architecture's reason for existing came back null on a corpus already known to be too thin
and too skewed to ground two of the five primary dimensions. Neither is fixed by more scenarios.

---

## 10. Provenance

- `python -m carelite.stats` regenerates every number here; `python -m carelite.eval.judge.load`
  loads `runs/judge-holdout/rubric_scores.jsonl` into `rubric_score` first.
- Bootstrap: 10,000 percentile replicates, resampling **scenarios** as whole units, seed 20260822.
- `ritualistic` is reverse-coded. `to_quality()` is the only path onto a common polarity, and every
  aggregation in the package reads the derived `quality` column, never `raw`.
- One bug found and fixed while running this analysis, recorded because it is the exact failure mode
  the lane's own brief warns about. `carelite/stats/data.py::_read` used `pandas.read_sql` over a
  psycopg connection carrying the `dict_row` factory; `read_sql` iterates each row to build a column,
  and iterating a dict yields its **keys**. Every cell came back as the name of its own column — a
  frame of the right shape, full of strings, which every downstream `to_numeric(errors="coerce")`
  turns silently into NaN. Nothing raised. The analysis would have found no data and reported that as
  though it were a fact about the study. It was invisible for as long as `rubric_score` was empty,
  which is how it survived to the run. `tests/unit/stats/test_data.py` now pins it with a fake
  dict-row cursor, so it stays fixed without needing Postgres.
