# CARELite AI — results of the holdout evaluation

**Every result in this document is descriptive, and every one is EXPLORATORY.** `DECISIONS.md` D10
dropped the pre-registration: this is a local proof of concept, it was never registered, and nothing
here is confirmatory, pre-specified in the registered sense, or hypothesis-testing. On top of that,
**0 of 11 rubric dimensions have cleared the judge-agreement gate** (analysis plan §9), because
`rating_assignment` holds 0 rows and the judge validation study has not run. That label is a measured
state of the database, not a hedge. `docs/preregistration.md` is kept as a timestamped-in-git record
of the analysis plan as it stood before any holdout data existed, and the analysis follows it — but
the plan's standing is "written down first", not "registered".

Regenerate everything below with `make reproduce`, which writes `runs/repro/`. **Where this document
and `runs/repro/headline-numbers.txt` disagree, that file is right and this one is stale** — it is
queried from Postgres at write time and this is prose. Every number traces to a query in
`carelite/stats/data.py` and a test in `tests/unit/stats/`.

---

## 1. What was run

`generation` holds **1,119** rows and every one of them has a rubric score: `rubric_score` is 1,119
as well, so nothing was generated and left unjudged. All judging was `gpt-oss:20b` at temperature 0,
single pass, one aggregate row per generation.

**The analysis frame is 1,080 generations over 60 held-out scenarios** — six conditions × 60
scenarios × 3 samples, with every condition complete at 180 cells and no gaps. Under D13 the arms are
narrowed to one serving stack per condition: A, A2, B, C and D are `served_by = 'ollama'`, and the LC
arm is `served_by = 'vllm'` and nothing else. The census counts 939 generations from Ollama and 180
from vLLM.

| Exclusion | Count | Treatment |
|---|---|---|
| LC cells served by Ollama (D11, retained under D13) | 39, over 13 of 60 scenarios | In **no arm**. Kept as the paired backend-equivalence sample. See §4. |
| Output-gate refusals (D12) | 24 generations | Included in the base reading, flagged; sensitivity (d) excludes them. See §6. |
| CRAG fell back to B | 69 of 180 C cells (38%) | Reported both ways. See §5. |
| Incomplete on ≥1 dimension | 58 of 1,080 generations | Kept; the missing dimension only is dropped. |

The 39 Ollama LC cells are the difference between the 1,119-row census and the 1,080-row analysis
frame. They are not part of the LC arm and never were: they are the scenarios LC happened to reach
before D11 stopped it, never randomised for partial analysis, and served by a different stack from
the 180-cell arm. **Completing LC did not rescue them; it gave them a counterpart to be compared
against.** That comparison is §4.

The refusals are concentrated rather than spread: 15 of the 24 are on SC-029 and 3 on SC-031, with
SC-012, SC-013, SC-055, SC-057, SC-072 and SC-092 contributing one each; by condition A 3, A2 7, B 2,
C 2, D 3, LC 7.

**Missing data.** 249 of 11,880 (generation, dimension) cells are unscored — between 16 and 26 per
dimension, scattered rather than concentrated. Following analysis plan §10, an unscored dimension is
missing *for that dimension only*: it is not imputed and not treated as a 1, the generation keeps its
composite computed from the dimensions that were scored, and `n_dimensions` records how many
contributed. Dropping the whole generation instead would have discarded 58 otherwise-good rows to
recover 249 cells; imputing would have invented data.

---

## 2. The finding that governs everything else: three dimensions were not measured

The judge did not resolve three of the eleven rubric dimensions. Measured on the analysed
(`to_quality`) scale:

| Dimension | n | sd | modal share | distinct values |
|---|---|---|---|---|
| `ritualistic` | 1,056 | **0.20** | 99.0% on one value | 4 of 5 |
| `ie` | 1,058 | **0.49** | 93.4% on one value | 4 of 5 |
| `naturalness` | 1,054 | **0.65** | 83.1% on one value | 5 of 5 |
| *the other eight* | 1,054–1,064 | 0.99–1.89 | 44.8–79.2% | 5 of 5 |

**This is not a null result about the system. It is a null result about the instrument.** A paired
test ranks within-scenario differences; when the judge gives nearly every response the same score,
nearly every difference is exactly zero and there is nothing to rank. The test returns a large
p-value that looks exactly like "the conditions do not differ" and means something entirely
different.

**No sample size repairs it.** The `carelite-judge` lane established the mechanism independently:
ordinal Krippendorff's alpha is bounded above by the variance the judge itself produces (r = 0.878).
A judge that emits one value cannot disagree with itself, cannot agree with a human, and cannot
separate two conditions. A larger holdout would produce the same floor with narrower intervals
around it. Completing LC added 180 generations and moved none of this.

**The cost is the study's most interesting prediction.** Build plan v3 predicts Condition B loses to
A on `naturalness` *because framework prompting induces ritual*. Both dimensions carrying that
prediction — the outcome and its stated mechanism — are degenerate here. **This run can neither
support nor refute it.** §3 gives the numbers and says why they cannot be read as the v3 result.

**On the classification rule, and two things worth recording.** A dimension is called degenerate when
its standard deviation is below 0.75 rubric points. The rule was chosen after seeing the data and is
labelled a post-hoc diagnostic in the output itself.

The first version used modal share instead, and it was wrong: `respect` puts 79.2% of its mass on one
value and spreads the remaining 20.8% across the whole scale (843 scores of 1, 142 of 5), which is a
dimension discriminating about as hard as a five-point scale can. Concentration is not degeneracy;
absence of spread is.

The second is a correction to what this document previously claimed. On the earlier partial run the
eleven dimensions fell into two groups with an empty band between them and every cut from 0.60 to
0.95 gave the same three, so the threshold was doing no work. **That is no longer true.** On the full
run `naturalness` sits at sd 0.65, and `threshold_sensitivity()` reports that at a cut of 0.60
`naturalness` leaves the degenerate set. The classification is now a judgement call at the margin for
one of the three dimensions, and the analysis output says so in those words. `ie` (0.49) and
`ritualistic` (0.20) are unaffected by any cut in the range tried. Nothing downstream changes —
`naturalness` is instrument-limited on the pair count regardless of which side of 0.60 it falls, as
§3 shows — but the guard that would have hidden this is the same guard D3 required of the aspiration
filter, and it earned its place a second time.

---

## 3. The eight comparisons

Holm-Bonferroni is applied across the **whole family of eight comparisons at once — across measures
and dimensions together, not per dimension**. All eight were computed this time; C vs LC is no longer
a reserved slot. All tests are two-sided. Effect sizes and their 95% bootstrap CIs come before
p-values throughout, and all three point estimators are reported for every comparison.

Effects are **left minus right**, so a negative effect means the right-hand condition scored higher.

| # | Comparison | rank-biserial [95% CI] | Hodges–Lehmann (points) | Holm *p* | Predicted | Observed |
|---|---|---|---|---|---|---|
| P | A vs B, NURSE | −0.866 [−0.964, −0.719] | −0.667 [−0.833, −0.467] | 1.24×10⁻⁷ | B higher | **as predicted** |
| 1 | A vs B, Four Habits | −0.744 [−0.916, −0.525] | −0.708 [−0.917, −0.500] | 4.07×10⁻⁶ | B higher | **as predicted**, attenuated |
| 2 | B vs C, NURSE | +0.214 [−0.091, +0.488] | +0.100 [−0.067, +0.267] | 0.31 | C higher | **against** |
| 3 | C vs LC, NURSE | −0.411 [−0.654, −0.133] | −0.233 [−0.422, −0.067] | 0.0197 | C higher | **against**, and see §4 |
| 4 | A vs B, naturalness | −0.626 [−0.872, −0.315] | −0.167 [−0.333, **0.000**] | 0.0037 | A higher | **against**, and instrument-limited |
| 5 | A vs B, ritualistic | −0.200 [−1.000, +1.000] | 0.000 [0.000, 0.000] | 0.85 | A higher | instrument-limited |
| 6 | A vs A2, NURSE | −0.731 [−0.887, −0.533] | −0.400 [−0.567, −0.267] | 5.39×10⁻⁶ | *no difference* | **prediction fails** |
| 7 | B vs D, NURSE | +0.992 [+0.967, +1.000] | +1.114 [+0.933, +1.300] | 2.80×10⁻¹⁰ | B higher | **as predicted** |

Comparison 1 is flagged ATTENUATED in the output: 1 of the 4 dimensions in the Four Habits composite
(`ie`) is degenerate, so the composite's effect size is pulled toward zero by a constant dimension
and understates the difference on the ones that moved.

Comparison 2's Holm-adjusted p moved from 0.46 to 0.31 with no change to its data. That is the
correction family working as intended: on the partial run only seven of the eight members could be
computed, and the eighth now takes a rank in the Holm ordering. **The uncorrected p is 0.155 in both
runs.** A comparison's adjusted p-value is a property of its family, not of itself.

The Friedman omnibus across {A, B, C} is reported for each dimension but is **not** a member of the
Holm family; its within-omnibus adjustment is shown for reference only. It is significant on `name`
(χ² = 44.9, p = 1.8×10⁻¹⁰), `de` (47.3, 5.3×10⁻¹¹), `epp` (33.5, 5.3×10⁻⁸) and `explore` (31.8,
1.3×10⁻⁷), and not on `respect` or `support`. On the three degenerate dimensions its p-value is
printed as uninterpretable rather than as a result.

**Power.** At n = 60 the smallest detectable paired effect is dz = 0.376 at the nominal α; charging
the whole Holm family against one test (α = 0.05/8) it is dz = 0.489. n was set by secondary outcome
2, B vs C — the comparison expected to show the *smallest* effect — so the primary outcome is
over-powered at this n and a null B-vs-C result below dz = 0.376 is a statement about this study's
resolution rather than about retrieval.

### What holds

**Framework prompting works, and the effect is large.** B beats A on composite NURSE adherence by
two-thirds of a rubric point, with 56 of 60 scenarios moving in that direction. The same holds on
the Four Habits composite, where 59 of 60 move. This is the study's clearest result and it survives
every rerun that could be run.

**The negative control passes decisively.** The rubric separates B from the deliberately degraded D
at rank-biserial 0.992 — 59 of 60 scenarios, a shift of 1.114 points. All three of the pass
conditions hold: direction B > D, bootstrap CI excluding zero, Holm-corrected p below 0.05. The
measurement instrument discriminates on the dimensions it resolves. Given §2, that verdict is worth
stating precisely: the rubric works on the eight dimensions that carry variance, and three dimensions
is where it failed, not the whole instrument. Condition D is degraded on the communication dimensions
the rubric scores, not on safety.

### What does not hold

**Retrieval did not help. Condition C is not better than Condition B, and the point estimate is
slightly worse.** rank-biserial +0.214 favouring B, CI spanning zero, Holm p = 0.31 — and the
direction is the opposite of the one predicted. §5 reports this both ways and the conclusion does not
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

- `naturalness` is degenerate — 83.1% of all scores are the single value 3.
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

## 4. Secondary outcome 3: C vs LC, and why it cannot be read as an architecture result

**One sentence first, because it is the sentence that will be misquoted.** Condition LC scored higher
than Condition C on composite NURSE adherence, and **this is not evidence that long context beats
retrieval.** The two arms were served by different inference stacks, those stacks were measured
disagreeing with each other on the same cells by more than this comparison's whole effect, the
comparison loses significance the moment CRAG-fallback turns are removed, and it reverses sign on the
other composite. Each of those is a number below.

The comparison itself, computed for the first time on this run:

| | value |
|---|---|
| Hodges–Lehmann | **−0.233 rubric points** [−0.422, −0.067] |
| rank-biserial | −0.411 [−0.654, −0.133] |
| Cohen's dz | −0.361 [−0.626, −0.112] |
| Holm-adjusted *p* | 0.0197 (family of 8), significant at α = 0.05 |
| n | 60 paired scenarios, 58 with a nonzero difference |
| direction | predicted >, observed < — **AGAINST the predicted direction** |

The analysis output carries two `!!!` banners on this comparison wherever it prints, and its
`caveats` field in `runs/repro/effect-sizes.csv` carries both in full. They are not decoration.

**(i) It is confounded by serving stack, and the confound is measured rather than asserted.** Under
D13 the LC arm is vLLM and Condition C is Ollama. The 39 Ollama LC cells D11 left behind share
scenario, sample index and seed with their vLLM counterparts, which makes them paired observations of
the same cells under two stacks — the only direct measurement of the confound this study has.
`carelite.eval.judge.backend_equivalence` reports, on 37 paired cells over 13 scenarios (2 gate-blocked
cells excluded):

| dimension | mean Ollama | mean vLLM | diff | Wilcoxon *p* |
|---|---|---|---|---|
| `understand` | 2.394 | 3.758 | **+1.364** | 0.009 |
| `epp` | 1.800 | 2.771 | **+0.971** | 0.032 |
| `de` | 2.583 | 3.500 | **+0.917** | 0.030 |
| `explore` | 2.257 | 3.114 | +0.857 | 0.090 |
| `name` | 1.333 | 2.167 | +0.833 | 0.057 |
| `respect` | 2.086 | 2.000 | −0.086 | 0.688 |
| `support` | 1.743 | 1.743 | +0.000 | 1.000 |

**Verdict: `poolable: False`** — the check requires n ≥ 30 scenarios and every dimension at
α ≥ 0.667 and ρ ≥ 0.5, and no dimension but `support` comes close. Every disagreement runs in the
same direction, vLLM scoring higher, and on three NURSE dimensions the gap between the *stacks* is
larger than the 0.233-point gap between the *conditions*. A difference of this shape is exactly what
a serving-stack artefact looks like, and no analysis on this data can separate it from an
architectural effect.

The equivalence check has its own limits and they cut both ways. 13 scenario-level pairs resolve only
dz ≥ 0.86, so a non-significant dimension there is not evidence of agreement. Generation ran at
temperature 0.7, so two sampling implementations differ from the same seed by sampling alone. And the
stacks differ in more than the server — GGUF against HF safetensors, quantisation, sampling defaults,
realised context pack, hardware — so the disagreement does not isolate the serving stack any more
than agreement would have made them interchangeable. **Neither direction identifies a cause.** What
the check establishes is narrower and sufficient: the two arms are not measurements on a common
scale, so their difference is uninterpretable as architecture.

**(ii) It is the reduced form of the question (D7).** The corpus does not fit the window. The
production pack admits **116/116 knowledge base entries but only 151/471 chunks** at 117,849 real
tokens. LC is therefore a fixed, query-independent sample of the corpus, not the corpus — and any
selection rule is itself a form of retrieval. What this compares is *query-dependent selection
against a fixed context*. It is not *curated retrieval against stuffing everything in*, which is the
question build plan v3 §3 posed and which this run cannot answer at all.

**(iii) It does not survive sensitivity analysis (b).** `runs/repro/analysis.txt` prints
**`*** 1 CONCLUSION(S) MOVE UNDER SENSITIVITY ANALYSIS. ***`** and this is the conclusion that moves.
Removing the 69 generations where CRAG fell back to Condition-B behaviour — turns that ran B's code
path and are not testing retrieval — takes C vs LC from Holm p 0.0197 (significant, n = 60) to Holm
p 0.137 (not significant, n = 37), with the effect barely changed at −0.389. **A conclusion that
flips under a sensitivity analysis is the finding, not a footnote to one.** The effect size is stable
and the significance is not; what moved is n.

**(iv) It reverses on the other composite.** In the mixed-effects models of §7, LC sits above C on
`nurse_composite` (+0.7973 against +0.5453 relative to A) and **below** C on
`four_habits_composite` (+0.5290 against +0.6949). The ordering of the two conditions is not stable
across the two measures the study reports. A robust architectural difference would not behave this
way; a serving-stack difference interacting with what each rubric happens to reward would.

**(v) In the equity stratum it is nothing at all.** rank-biserial −0.171 [−0.643, +0.338],
Hodges–Lehmann −0.100 points, Holm p = 1.0 on 20 scenarios.

**What can be said.** On this run, under this judge, the vLLM-served long-context arm scored about a
quarter of a rubric point higher than the Ollama-served retrieval arm on composite NURSE adherence,
in a comparison that is confounded by serving stack, restricted to the reduced form of the question,
not robust to excluding fallback turns, and reversed on the other composite. **That is a fact about
this run. It is not a result about retrieval.**

**And completing LC did not make the study stronger.** It added one comparison. It changed no item's
evidential status: `ie`, `naturalness` and `ritualistic` are degenerate on the full run and stay
degenerate, the judge validation study still has not run, `rating_assignment` still holds 0 rows, and
every result in this document — the ones that existed before LC and the one LC produced — is
EXPLORATORY. A new comparison is not a stronger comparison.

---

## 5. Retrieval, asked two ways

CRAG graded 111 of 180 Condition-C cells `relevant` and 69 `none`. On those 69 (**38% of the arm**) C
fell back to Condition-B behaviour, so pooling compares C against itself on more than a third of its
mass.

| Reading | Question | rank-biserial [95% CI] | HL (points) | *p* (uncorrected) | n |
|---|---|---|---|---|---|
| All 180 cells | Does *offering* retrieval help? | +0.214 [−0.091, +0.488] | +0.100 | 0.155 | 60 |
| The 111 that retrieved | Does *retrieval* help? | +0.220 [−0.154, +0.553] | +0.133 | 0.245 | 37 |

The p-values in this section are uncorrected, family of 1. Reading one planned comparison two ways is
a second look, not two new members of the Holm family; the corrected p for B vs C is the 0.31 in §3.

Only the second row is the architecture's claim, and **it gives the same answer**: no measurable
benefit, with the point estimate slightly favouring B in both. The fallback rate is not what is
holding the effect down. Restricting to the cells that retrieved costs 23 scenarios, which lose their
entire C cell — so the second row is both a weaker test and a *selected* one.

**The selection confound is real and cannot be removed after the fact.** CRAG chose which cells
retrieved, on its own judgement of relevance, so the 111 are plausibly the scenarios this corpus can
serve. Comparing them against the full Condition-B arm is a self-selected subgroup contrast, and the
bias runs *in favour of* retrieval. The result is null anyway, which makes it the more informative
direction to be wrong in.

This is consistent with what the rest of the project already established about the evidence base: 18
of 33 papers dominate the corpus, on-domain context precision runs 0.334–0.380 against a 0.7 target,
and D8 records that two of the five NURSE dimensions (`respect`, `support`) have **zero** knowledge
base entries and so cannot be helped by retrieval at all. A retrieval layer cannot add grounding the
corpus does not contain. Those two dimensions also carry the two largest Friedman p-values across
{A, B, C} (`support` 0.245, `respect` 0.062), and they are the two where the backend-equivalence
check of §4 finds the two serving stacks agreeing.

---

## 6. The output safety gate: reported both ways, and which is preferred

The `carelite.safety` output gate refused 24 generations, listed by scenario and condition in §1.
Because 15 of them are on SC-029, excluding them removes most of one scenario across several
conditions rather than trimming symmetrically, and a paired test loses that scenario from every
comparison it touches.

Sensitivity analysis (d) is the rerun that excludes them. **Conclusions hold: no comparison changed
significance or direction.** The per-comparison figures for both readings are in
`runs/repro/analysis.txt` under sensitivity (d), which is where they should be read from rather than
transcribed here — a rerun will change them and this file will not notice.

**The preferred reading excludes them**, and the reason is asymmetric rather than a preference for
the tidier number. Scoring refused text on a communication rubric is a *category error*, not a low
score — the gate's output is not a communication attempt, so its NURSE score measures nothing.
Excluding it costs one scenario from a 60-scenario paired analysis, which shows up honestly in the
`n` and in the interval. Prefer the loss you can see. Scoring refused text silently, which is what
would have happened before D12 added the column, is the one option that is definitely wrong. Note
that the base analysis in §3 and the negative-control verdict both **include** the refused text, as
D12 specifies for the base reading; compare the two before quoting either margin.

This rerun was **not planned in advance** — D12 postdates the analysis plan — and is labelled as such
in the output. The concentration figure quoted above is computed from the frame at run time; it was
previously a hardcoded `13 of 17 on SC-029` that described the partial run and silently went stale
when the holdout completed. `tests/unit/stats/test_sensitivity.py` now pins it against a fixture
whose refusals sit on a different scenario, so a literal cannot pass.

---

## 7. The three samples per cell are not three observations

Analysis plan §8.3. A random intercept for scenario separates within-scenario generation variance
from the between-condition effect.

Composite NURSE, REML, `value ~ condition + (1 | scenario)`, reference A, 1,079 generations over 60
scenarios:

| Term | Coefficient | 95% CI | *p* |
|---|---|---|---|
| A2 vs A | +0.4452 | [+0.3071, +0.5832] | 2.6×10⁻¹⁰ |
| B vs A | +0.6724 | [+0.5344, +0.8105] | 1.3×10⁻²¹ |
| C vs A | +0.5453 | [+0.4072, +0.6833] | 9.8×10⁻¹⁵ |
| D vs A | −0.4639 | [−0.6019, −0.3258] | 4.5×10⁻¹¹ |
| LC vs A | +0.7973 | [+0.6590, +0.9355] | 1.3×10⁻²⁹ |

**The LC row is the largest coefficient in the table and must not be read as the strongest
condition.** It carries the whole of §4's serving-stack confound: it is the only row whose condition
was served by a different stack from the reference. On `four_habits_composite` the same model puts
C at +0.6949 and LC at +0.5290, reversing their order. The four Ollama-served contrasts are
within-stack and do not have this problem.

Between-scenario variance 0.2389, residual 0.4465, **ICC 0.3485** — a third of the variance in a
response's score is the scenario it answers, not the condition that produced it. Mean within-cell
variance is 0.3301. A method-of-moments cross-check gives between 0.2276, within 0.6445, ICC 0.2610;
the two estimators bracket the same conclusion rather than agreeing exactly, and both are printed.

**What treating the 1,080 samples as independent would have cost:** standard errors understated by
**1.16× to 1.31×** depending on the contrast, and every p-value correspondingly too small. That ratio
is measured per contrast, not asserted as a √3 penalty — it is a measurement of how much
scenario-by-condition structure the data actually carry, and it is above 1.0 everywhere, which means
the structure is real. The largest penalty, 1.31×, falls on LC vs A.

---

## 8. The equity stratum

The one subgroup planned in advance (§8.4). **Descriptive only — this is not a powered test of an
equity effect and must not be written up as one** (D9.1). Every other subgroup is exploratory and
carries that label in the output structure, not only in prose.

n = **20** held-out scenarios (`ses` 10, `racial_ethnic` 6, `lep` 4), not the 35 the plan's
parenthetical names: 35 counts the whole bank and 15 of those are in the train split. At n = 20 the
smallest resolvable paired effect is dz = 0.676 — **large effects only**. `racial_ethnic` at n = 6
supports nothing in either direction and its breakdown is printed as a description of the data, not
as a comparison.

The pattern matches the full analysis: A vs B −0.942 (Holm 0.0030), A vs B on Four Habits −0.762
(Holm 0.015), A vs A2 −0.919 (Holm 0.0024), B vs D +1.000 (Holm 0.0011), B vs C −0.074 (Holm 1.0),
C vs LC −0.171 (Holm 1.0). `ritualistic` has **0 of 20** non-tied pairs here, so every statistic on it
is `nan` and is printed as `nan` rather than as a zero. Nothing in this stratum is evidence of an
equity-specific effect; it is the main result, re-observed on a third of the data.

Two limitations were declared before any data existed and stand:

- The stratum contains no `emotion_intensity = 1` scenario, so it cannot say whether the disparity
  behaves differently on an emotionally flat turn — the turn where a system that over-reads emotion
  does its worst work. Flat turns are still tested outside the subgroup.
- `racial_ethnic` contains no `adherence_barrier`, `decision_conflict` or `false_comprehension`
  scenario, and every one of its scenarios presents an already-guarded patient. A system that scores
  well on this axis may be scoring on *handles a guarded patient* rather than on the disparity the
  axis claims to measure.

Per D5, the `racial_ethnic` axis is described as what it measures: **response to anticipated
dismissal and patient credibility-management**, not race-based disparity in communication generally —
eight of the nine such scenarios in the bank turn on a patient who has already been disbelieved, or
expects to be, and manages the clinician accordingly. The label is kept for continuity with the
frozen split; the description is not.

---

## 9. What the sensitivity analyses could and could not do

Build plan v3 §14 asks for three reruns. Two of them cannot be run on this data and say so in those
words:

- **(a) judge-only vs human-only — NOT RUNNABLE.** Fewer than two rater types exist. No human rating
  has occurred, so every number in this document is judge-only and there is nothing to compare it
  against.
- **(c) excluding scenarios with poor judge self-consistency — NOT RUNNABLE.** No multi-sample judge
  rows exist. The five-sample self-consistency pass runs on the validation subset only, and the
  holdout was judged single-pass at temperature 0.

Both are reported as "not runnable", never as "conclusions hold". **An absence of a test is not
evidence of robustness.**

Of the reruns that could be run: **(b) CRAG-fallback excluded moved one conclusion** — C vs LC, §4.
**(d) gate-blocked excluded moved none.** **(e) backend equivalence** is new under D13, was not
planned in advance, and is reported in §4; it is not a rerun of the §8.1 family, cannot flip a
conclusion, and a clean result there would not have made the C-vs-LC comparison unconfounded, because
that confound is structural and is stated on the comparison itself.

---

## 10. What this study can and cannot claim

**Can claim, on this run, descriptively and exploratorily:**

1. Framework prompting (B) substantially outperforms the bare model (A) on composite NURSE and Four
   Habits adherence, by roughly two-thirds of a rubric point, consistently across scenarios.
2. The rubric separates the framework condition from a deliberately degraded one at near-ceiling
   effect size, so it is measuring something real on the eight dimensions that carry variance.
3. Adding retrieval (C) produced no measurable improvement over framework prompting alone, whether
   or not the fallback cells are included.
4. Changing the generator model produces an effect about two-thirds the size of the framework
   prompt's, so model family is a substantial confound in any claim about prompt architecture.
5. A third of score variance is scenario, not condition — the three samples per cell are not three
   observations, and analysing them as such would understate every standard error by 1.16–1.31×.
6. Two inference stacks serving the same model family disagree on identical scenario/seed cells by up
   to 1.36 rubric points on a single dimension, all in one direction, and are **not poolable**. That
   is a finding about evaluation methodology and it stands on its own.

**Cannot claim:**

1. Anything about naturalness or ritual. The instrument did not resolve either dimension, and the
   central v3 prediction that connects them is untestable on this data.
2. Anything about `ie` (invests in the end), for the same reason.
3. **That long context beat retrieval, or that retrieval beat long context.** The C-vs-LC comparison
   exists now and is significant before correction for its own confounds, but it is confounded by
   serving stack, is the reduced form of the question under D7, loses significance under sensitivity
   (b), and reverses direction on the other composite. See §4.
4. That curated retrieval beats stuffing the whole corpus into the window. That comparison was never
   available: the corpus does not fit, so LC is a fixed sample and not the corpus.
5. That the judge's scores agree with human judgement, on any dimension. 0 of 11 dimensions have
   cleared the §9 gate because `rating_assignment` holds 0 rows, so every dimension is exploratory
   and every result in this document says so in its own label.
6. Anything confirmatory or pre-registered, about anything. D10.

**The two failures that matter most are both about measurement, not about the system.** Three
dimensions were not resolved by the judge, and the one comparison that would have tested the
retrieval architecture's reason for existing came back null on a corpus already known to be too thin
and too skewed to ground two of the five primary dimensions. Neither is fixed by more scenarios, and
neither was fixed by completing LC.

---

## 11. Provenance

- `make reproduce` regenerates every number here into `runs/repro/`, which is tracked in git:
  `analysis.txt` is the full analysis, `headline-numbers.txt` the figures with the qualifications
  they cannot be quoted without, and `effect-sizes.csv`, `instrument-resolution.csv` and
  `data-inventory.csv` the machine-readable forms. `python -m carelite.stats` runs the analysis
  alone.
- Where this document disagrees with `runs/repro/headline-numbers.txt`, that file is right. It is
  queried from Postgres at write time; this is prose, and prose goes stale. `headline.py` exists
  because that has already happened here more than once.
- Exploratory status is carried in the output structure, not only in prose: every `PairwiseResult`
  has a `label` field, and every caveat that qualifies a comparison is a row in the `caveats` column
  of `effect-sizes.csv`.
- Bootstrap: 10,000 percentile replicates, resampling **scenarios** as whole units, seed 20260822.
- `ritualistic` is reverse-coded. `to_quality()` is the only path onto a common polarity, and every
  aggregation in the package reads the derived `quality` column, never `raw`.
- Two bugs found and fixed while running this analysis, both recorded because they are the exact
  failure mode the lane's brief warns about — a silent stats bug is indistinguishable from a result.
  - `carelite/stats/data.py::_read` used `pandas.read_sql` over a psycopg connection carrying the
    `dict_row` factory; `read_sql` iterates each row to build a column, and iterating a dict yields
    its **keys**. Every cell came back as the name of its own column — a frame of the right shape,
    full of strings, which every downstream `to_numeric(errors="coerce")` turns silently into NaN.
    Nothing raised. The analysis would have found no data and reported that as though it were a fact
    about the study. It was invisible for as long as `rubric_score` was empty, which is how it
    survived to the run. `tests/unit/stats/test_data.py` pins it with a fake dict-row cursor, so it
    stays fixed without needing Postgres.
  - `carelite/stats/sensitivity.py` printed the gate-blocked concentration as a hardcoded string from
    an earlier run. It is now computed from the frame and pinned by a test. See §6.
