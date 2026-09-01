# Limitations

Build plan v3 §17 named a limitations list before any evaluation data existed, and most of this
document was written on that principle: a limitation named before the numbers exist is a
limitation, and the same sentence written after is an excuse. **The holdout run and its judging
completed on 2026-08-25** (§4), so this document now also carries what that run actually showed,
including a limitation — the `naturalness`/`ritualistic` instrument failure — discovered only once
real generations existed to measure. It is included here in full rather than held back, because a
project that only publishes the limitations it predicted correctly is not being honest about the
ones it did not. **Per `DECISIONS.md` D10, nothing in this document is confirmatory or
pre-specified in a registered sense — every finding below, including the ones from the completed
run, is descriptive.**

Status of the underlying work as of 2026-09-01: the corpus and knowledge base are loaded, the
100-scenario bank is frozen, the five conditions A, A2, B, C and D completed in full under Ollama
on 2026-08-25, and Condition LC — stopped at 39 of 180 cells by D11 — was completed on 2026-09-01
under a second serving stack (vLLM with prefix caching, `DECISIONS.md` D13). §4 carries what that
changes and, more importantly, what it does not. `carelite-stats` owns the statistical write-up;
this document owns the end-to-end record of what was built, what was run, what was found, what
could not be found and why, and what a reader must not conclude from any of it.

**Row counts in this document are a snapshot; `runs/repro/headline-numbers.txt` is the authority.**
`make reproduce` writes that file by querying Postgres, with each figure printed next to the
qualification it cannot honestly be quoted without (`carelite/stats/headline.py`). It exists
because a planning document in this repository was written from figures carried forward in memory,
and prose in this repository has already gone stale once against the database. Where a count below
disagrees with that file, that file is right and this one needs an edit.

---

## 1. Corpus

**33 of 43 manifest DOIs were retrieved.** The DOI manifest in `carelite/corpus/fetch.py` lists 55
rows; 8 are marked duplicates of an already-fetched paper, 5 carry no DOI at all, leaving 43 unique
DOIs actually attempted. 33 resolved to a real PDF. The 10 that did not are recorded in
`data/pdfs/_manual_needed.csv`: nine are `nihms*` manuscripts that Unpaywall and the NCBI
ID-converter both report as not open access (`idIsNotOpenAccess`) despite being readable on PMC in
a browser — free to read, not licensed for programmatic retrieval — and one more failed for the
same reason under a different HTTP path. The remaining 5 manual-needed rows never had a DOI to
resolve in the first place.

**The loss is not random.** `knowledge_base/TAXONOMY.md` traces which literature sits in the lost
set: the `nihms*` rows are Patient Education and Counseling and JAMA-family papers, and the
emotional-blocking and health-literacy literature is concentrated in exactly that set. Coverage
across the seven themes, counted as papers whose full text carries substantive coverage (at least
eight matches against that theme's vocabulary):

| Theme | Papers with substantive coverage |
|---|---|
| Empathy | 13 |
| Emotion recognition and response | 13 |
| Trust and relational continuity | 8 |
| Patient activation and shared decision-making | 7 |
| Equity-aware communication | 4 |
| Plain language and information clarity | **2** |
| Comprehension confirmation (teach-back) | **1** |

Teach-back and plain language are single- and near-single-source themes as a direct consequence of
which papers survived the fetch, not because the corpus is small in general. Any KB entry or
downstream claim in these two themes traces back to one or two papers; treat their evidence base as
narrow rather than as convergent across the literature.

**18 of the 33 retrieved papers are communication-skills-training studies** — interventions,
curricula, or burnout studies about how communication skill is *acquired* — rather than studies of
what to say in a given bedside moment. They support knowledge-base entries only where they report a
finding about the communication behavior itself, not about the training that taught it, which is
the largest single constraint on how many entries the corpus can support per theme.

---

## 2. Knowledge base provenance

**The knowledge base is LLM-extracted from primary sources with no human verification, not
hand-curated as build plan v3 assumed.** `DECISIONS.md` D4 records the decision to drop the planned
human sign-off gate rather than claim it falsely: `human_verified` is `FALSE` on all 116 loaded
entries (as of this writing — `knowledge_base/review/kb_review_digest.md` carries the live count
if it moves again), and that is the honest record, not a pending checkbox. Any result that
depends on knowledge-base quality inherits this limitation.

Theme distribution: activation_sdm 40, plain_language 21, teach_back 15, trust_continuity 14,
empathy 14, emotion_response 9, equity 3.

What the pipeline in `carelite/kb/validate.py` and `carelite/kb/spans.py` *can* substantiate, precisely:

- **Every entry's `verbatim_span` was located in the extracted text of the paper it cites**, and
  what is stored is a literal slice of that source text rather than the model's paraphrase of it.
  Matching folds only rendering differences — ligatures, quotation glyphs, dashes, hyphenated
  line-break splits, whitespace, case — never content. This was verified against the database and
  separately spot-checked by re-extracting sampled papers from their original files.
- **The genuine fabrication rate was measured, not estimated: 8 of 180 candidates (4.4%).** A first
  pass read this as 26 of 130 (20%); auditing every rejected span against its source text showed
  most of those were not fabrications at all. Of the 19 candidates whose span the validator could
  not locate at all: 8 substituted, dropped, or invented words the source does not contain, and are
  the genuine fabrication rate; 10 quoted a real sentence exactly but with inlined citation
  superscripts or altered punctuation the validator deliberately does not fold, because folding
  digits would let it confuse two different statistical readings; and 1 is a corpus-extraction
  defect (a PDF running footer injected mid-sentence), escalated to `carelite-corpus` rather than
  patched here. A further 12 rejections were the validator's own fault — words split or joined
  across a PDF column or line break — and are now fixed by a second normalization pass in
  `spans.py`, with the recovered span still shown with its artefact intact so a reviewer sees how
  much cleanup was applied.
- **Entries are rejected for a fabricated span, a span too short to carry evidence, an
  unrecognized vocabulary value, or a non-actionable takeaway** — the actionability check rejects
  awareness statements ("clinicians should be mindful that...") because awareness is not something
  the system can detect, generate, or reframe. Two further checks landed after the figures above
  were first measured: candidates are now also rejected for citing a subject `knowledge_base/TAXONOMY.md`
  places out of scope, and for a finding the quoted span does not actually report. **An overclaimed
  evidence tier is corrected rather than rejected, and is now derived from the source paper's study
  design outright rather than merely capped at it**: 60 of the 116 entries carry a tier that differs
  from what the extraction model originally claimed — 54 lowered, and **6 raised**, the six being
  ones a ceiling-only check could never have caught (`carelite/kb/papers.py` maps design to tier;
  `validate.py` enforces it). This is a stronger rule than the project's first version, which only
  lowered an overclaim and left an underclaim alone — under that version, two entries citing the
  same paper could carry different tiers, which `README.md`'s own definition of evidence strength
  as a property of the source does not allow. A second, related check now also caps tier for
  **second-hand findings** — an entry whose quoted span itself cites another study by number (e.g.
  a bare `(18)` marker) is capped at what *this* corpus's paper can vouch for, not the tier of the
  study it is relaying. Three papers still legitimately carry two tiers under this rule, and
  Talevski (the teach-back systematic review) is the clearest case: 8 entries at `strong` from its
  own synthesis, plus 5 at `moderate` relayed from studies it summarizes but does not itself
  produce. In every case the span, theme, finding, and takeaway are untouched by a tier correction,
  and there is a derivable right answer to substitute — unlike a fabricated quote, which has none.
- **No human read any entry for whether the finding follows from the span.** That is the
  specific judgment an automated check cannot make and it is not claimed here. The review
  machinery in `carelite/kb/review.py` exists and is exercised (`knowledge_base/review/kb_review_digest.md`
  is generated from the live database), but it is an available tool, not a completed gate: 0 of 116
  entries are signed off as of this writing.
- **Paper metadata is now complete: zero placeholder citations, zero papers missing a study
  design.** `carelite/kb/papers.py` writes `design`, `evidence_tier`, `apa_citation`, and `year`
  onto every `paper` row, with citations pulled from Crossref and frozen so a cold rebuild needs no
  network call to reproduce them. Crossref also corrected several hand-written short citations in
  the process and several manifest years that were wrong — a paper cited by its manifest filename
  rather than its Crossref-derived citation in an older document or note may now resolve to a
  different-looking but more accurate reference.

**Roughly a third of the knowledge base restates itself, and this constrains how every entry
count in this document — and in any results table — may be read.** The redundancy check
(`carelite/kb/review.py`) clusters entries within one `(theme, paper)` group that quote the same
underlying point in different words. Its first calibration was itself an instance of the defect it
exists to catch: at a pairwise-similarity threshold of 0.58 it reported 17 of 114 entries clustered
and said nothing about the pair *"Brief the interpreter on the goals and specific content of the
conversation before the patient enters the room"* and *"provide the interpreter with advanced
preparation and specific context before the encounter"* — one instruction stated twice, scoring
0.478, called independent. Recalibrated to 0.47 and checked by reading the groups that appear as
the threshold drops (not by the number alone): **the digest now reports 40 of 116 entries falling
into 12 clusters**, and the `teach_back` cluster alone grows from 6 entries to 10, which matches
what reading them shows — nearly every Talevski teach-back entry says some version of "use
teach-back to confirm the patient understood their discharge instructions." Every clustered entry
is individually evidenced — real span, real source — the defect is only in counting them as
separate support. **The calibration is deliberately set toward over-grouping**, because nothing in
this check rejects an entry: a cluster a reader disagrees with costs a moment to unmerge, while a
restatement the check misses enters the write-up as convergent evidence. A regression test
(`tests/unit/kb/test_review.py::TestClusterThresholdCalibration`) pins the 0.47 threshold against
the interpreter-briefing pair specifically, so a future tightening has to be argued for rather than
drifted into. **The consequence for this document and for any results write-up: entry counts must
never be presented as independent evidence, and a retrieval hit on several entries from one theme
is frequently one finding retrieved several times, not several findings.**

**Two of the eleven scored rubric dimensions have zero knowledge-base grounding, and this bears
directly on the primary evaluation outcome — `DECISIONS.md` D8.** The behavior-to-framework mapping
(which entries instantiate which NURSE or Four Habits component) is complete across all 116
entries: `ie` 40, `epp` 17, `name` 15, `ib` 6, `explore` 5, `understand` 5, `de` 4, **`respect` 0,
`support` 0**, with 40 entries instantiating none of the nine. Nothing in the 33 retrieved papers
turns NURSE Respecting (crediting the patient for something specific) or Supporting (partnership
made concrete — who does what, how to reach someone) into a finding with a quotable span and an
actionable takeaway. The zeros survived a correction that would have hidden them: an earlier
mapping filled both and looked plausible until every assignment was read against its source entry,
which found seven false positives, two of them exactly here (*"verbal affirmations to show you're
listening"* is a backchannel cue, not crediting; *"collaborative partnership"* is a stance with none
of Supporting's concrete half). A regression test pins each zero and a companion test confirms the
matcher still fires on a genuine crediting move. **Consequence: the primary outcome (composite
NURSE, Condition A vs. B — `docs/preregistration.md` §3) averages five dimensions, two of which
retrieval structurally cannot ground**, so any Condition C advantage on `respect` or `support`
specifically has some cause other than retrieval. This is declared in the pre-registration itself,
before any evaluation data exists, rather than surfacing as a post-hoc explanation. Separately: 13
entries are flagged `equity_relevant` while the `equity` theme (below) holds 3 — 10 of the 13 sit in
`plain_language` (5), `teach_back` (3), and `trust_continuity` (2). Reporting only the theme count
understates what the base holds about equity; reporting only the flag would overstate how much of
it is centrally *about* a disparity. Both numbers belong together, not either alone.

**The knowledge base's graph layer is populated: 715 edges** across `belongs_to` (116),
`supports` (116), `has` (149), `appropriate_in` (160), `instantiates` (92, following directly from
the coverage above), and `restates` (82, the redundancy clusters above materialized as edges rather
than only as digest prose). **All 116 entries are embedded**, mean pairwise cosine similarity 0.623
— consistent with, not independent evidence against, the redundancy finding above: a corpus where
a third of the entries restate each other should show elevated average similarity, and it does.
Chunk embeddings are unchanged at 471/471 (§5 below has the full index-verification detail).

**The equity knowledge base holds 3 entries, and — this is a finding, not a gap awaiting a
fix — `DECISIONS.md` D3's outcome establishes that this is a property of the corpus, not of the
extraction.** D3 approved re-extracting equity entries with a prompt asking for the *compensating
move* a disparity finding implies, rather than the awareness statement a faithful extraction of a
disparity naturally produces. Read individually against its guard, the variant run's candidates
broke down as: four aspiration statements ("proactively work to bridge the empathy gap") the
actionability guard correctly rejected, one exact duplicate of an existing span, and two genuine,
well-grounded entries — but in `plain_language` and `trust_continuity`, not `equity`. Both were
loaded (`kb-plain_language-2f0e0bced0`, `kb-trust_continuity-3ad5b8efed` — the latter is the
near-duplicate flagged by the redundancy check above), on the reasoning that discarding a
well-grounded entry because the prompt that surfaced it was aimed at something else would not be
principled. **`equity` itself did not move: it is still 3, and the total entry count rising from
114 to 116 must not be read as the equity problem having improved** — the variant's stated
purpose, raising equity coverage, still failed, for the structural reason below. The load was
quarantined behind `--prompt-version` throughout, so it reached the knowledge base only once its
guard had been applied, never merely because inference finished; without that guard the four
aspiration statements would also have loaded, and `equity` would have read 7 — looking like
progress while being a regression.

**Which paper failed is the whole result.** Holdsworth *describes interactions*, so a compensating
move was there to quote, and the model quoted one. Roberts is a meta-analysis: it quantifies a gap
and correctly never states what closes it. Asked for a move anyway, the model invented one — not a
prompt defect, and no prompt fixes it, because the literature that measures a disparity is not the
literature that prescribes a remedy, and this corpus holds the former. **This is one of three
independent measurements of the same gap — §3 below states all three together** rather than
scattering them as separate caveats: this corpus and this scenario bank measure disparity in
clinician communication considerably more thoroughly than they measure how to close it, or how to
power a confirmatory test of closing it.

Concentration is worse than the theme totals suggest for teach-back too, and the redundancy finding
above sharpens it rather than merely restating it: 12 of its 15 entries come from the single
Talevski (2020) systematic review — **effectively single-source** — and 10 of the theme's 15
entries now sit in one redundancy cluster. Those are the same fact seen from two directions: what
reads as convergent evidence within `teach_back` is largely one source cited many times, in
overlapping words.

---

## 3. Scenario bank and equity subgroup

Per `DECISIONS.md` D2 and D5, and detailed in `scenarios/EQUITY_REVIEW.md`:

- **The `racial_ethnic` axis measures something narrower than its name.** Nine scenarios carry
  `equity_kind = racial_ethnic`, drawn from documented emotional blocking of minority patients
  (Park et al. 2020) and the SES/race empathy gap (Roberts et al. 2021). Eight of the nine turn on
  one mechanism — the patient has already been disbelieved, or expects to be, and manages the
  clinician accordingly — so what the axis actually measures is response to *anticipated dismissal
  and patient credibility-management*, not race-based disparity in communication generally. The
  label is kept for continuity with the frozen split; the description is not.
- **The confirmatory equity subgroup is n = 20, not the 35 that carry `equity_stratum = true`
  across the full bank.** `DECISIONS.md` D9(1): confirmatory analysis is restricted to the holdout
  split, and 15 of the 35 equity scenarios sit in train. Holdout equity composition is `ses` 10,
  `lep` 4, `racial_ethnic` 6. **At n = 20 the subgroup resolves only large effects (dz ≈ 0.68), and
  `racial_ethnic` at n = 6 supports no statistical claim at all** — `docs/preregistration.md` §8.5
  states this and calls the equity analysis descriptive rather than confirmatory for exactly this
  reason.
- **Three independent measurements now converge on one finding about this evidence base, and it is
  stated here once rather than as three separate caveats:** `DECISIONS.md` D3 found the equity
  knowledge-base theme holds 3 entries as a property of the corpus, not an unfinished extraction
  (§2 above); the `racial_ethnic` axis bullet immediately above narrows to a single mechanism
  rather than the disparity its label names; and this bullet's n = 20 (n = 6 for `racial_ethnic`)
  means the confirmatory subgroup test cannot be powered even where the scenarios exist. **This
  corpus and this scenario bank measure disparity in clinician communication considerably more
  thoroughly than they measure how to close it, or how to power a confirmatory test of closing
  it.** Any equity subgroup result this project reports should be read with that ceiling in mind,
  not only the small sample size.
- **The `emotion_intensity = 1` cell is empty in the equity stratum.** SC-010, the only equity
  scenario at that intensity, was reclassified out of the `lep` axis for grammatically rather than
  situationally signaling LEP (D2). Emotionally flat turns are still tested — 12 of the 100
  scenarios sit at `emotion_intensity = 1` — just not inside the equity subgroup.
- **`racial_ethnic` contains no `adherence_barrier`, `decision_conflict`, or `false_comprehension`
  scenario**, and every one of its nine scenarios presents an already-guarded patient. A system
  that scores well on this axis may be scoring on *handles a guarded patient* rather than on the
  disparity the axis claims to measure, and the coverage audit cannot distinguish the two because
  it measures challenge type, phase, intensity, and literacy signal, none of which separate a
  guarded patient from an unguarded one.
- **The equity-stratum review was performed by the orchestrating session, not by an independent
  second person.** `scenarios/EQUITY_REVIEW.md` states this plainly and explains why it is a weaker
  check than the build plan's gate specifies: a second-person review is a genuinely different act
  from the same person checking their own reclassification against their own stated criteria. It
  remains outstanding.

The scenario bank overall: 100 scenarios, 40 train / 60 held-out (`carelite/scenarios/freeze.py`,
`HOLDOUT_DIGEST = 5a3cb128effc78f6ec41a5a8c616e2fe0fe4105abe42cc593d5dff01cd653395`), all
synthetic — no real patient utterance appears anywhere in the bank. Synthetic scenarios can be
written to be legible to a rubric in a way real clinical speech is not; nothing here validates
against an actual encounter transcript.

---

## 4. The completed holdout run, and its judge-primary evaluation

**Per `DECISIONS.md` D10, this project is a local proof of concept: OSF pre-registration was
dropped, and every result in this section — the run itself, the judge validation, the completed
holdout figures — is descriptive, not confirmatory or pre-specified in a registered sense, however
precisely a threshold or a rubric was fixed in advance of the data.** `carelite-stats` owns the
statistical analysis of this data (effect sizes, corrected tests, the write-up of what the numbers
mean); this section owns what actually happened when the run executed, what could and could not be
measured, and what a reader must not conclude from either.

### What ran, on what hardware, and what it cost

The 1,080-cell holdout run (60 scenarios × 6 conditions × 3 samples) does not run on a local laptop
in reasonable time — see `REPRODUCE.md`'s environment note for the measured reason — and was
executed on a **rented Runpod L40S (48 GB)**, all four models resident in VRAM, four parallel
workers split by condition. **One further operational fact belongs in this record, and is
independently confirmed in the codebase rather than merely relayed: partway through the project, a
Postgres instance sitting on the pod's container disk was restarted and lost its data, and roughly
863 generations already produced had to be regenerated.** `carelite/generate/load.py`'s own module
docstring states the consequence and the fix directly: *"The holdout run was written with
`--store jsonl` on purpose — an earlier attempt lost ~863 generations when Postgres sat on a
container disk that was restarted — so the durable artifact of the experiment is a set of journal
files."* That is why generation, judging, and loading became three separate steps (`carelite.generate.runner
--store jsonl` → `carelite.eval.judge.holdout` reading the journals directly → `carelite.generate.load`
bridging the journal into Postgres) rather than writing straight to the database, and it is included
here because an honest process record does not omit the parts that make the project look less
tidy.

**939 generations, zero failures at generation time, on the Ollama run of 2026-08-25.** Conditions
A, A2, B, C, and D completed in full: 180 cells each (all 60 holdout scenarios × 3 samples).
Condition LC did not, and was stopped at 39 cells: see "LC was dropped twice on a measured runtime
cost, and completed once the runtime changed" below. All 939 route decisions are `informational`;
zero route `emotional_only` anywhere in the holdout set, at any condition, including the 39
completed LC cells.

**`generation` now holds 1,119 rows, not 939.** On 2026-09-01 the remaining LC cells were generated
under vLLM (D13), so the table holds 180 cells each for A, A2, B, C and D at `served_by = 'ollama'`,
180 LC cells at `served_by = 'vllm'`, and the 39 D11 Ollama LC cells retained as a paired
backend-equivalence sample rather than pooled into the LC arm. **The primary comparison did not
move**: it runs on the 900 holdout generations with LC removed, which is what it ran on before. Do
not transcribe either figure into a write-up — read `runs/repro/headline-numbers.txt`, which prints
the census, the analysed frame, and the per-backend split as three separate numbers precisely
because collapsing them is the mistake that produced this paragraph's earlier version.

**17 of the 939 generations were refused by the output safety gate and are flagged
`generation.gate_blocked` rather than silently dropped or silently scored — `DECISIONS.md` D12.**
That decision is worth restating precisely because the direction of the risk is counter-intuitive:
a *missing* cell is visible (a condition with 178 rows where the others have 180 invites a
question), but text the safety gate refused, scored as though it had passed, is invisible and
flatters whichever condition produced it. Measured breakdown, by condition — A 3, A2 7, B 2, C 2,
D 3 — and by scenario — **SC-029 13**, SC-092 1, SC-055 1, SC-072 1, SC-057 1. SC-029 is
overwhelmingly the driver and fires across multiple conditions, so this is largely a *scenario*
property rather than a condition difference, but it is not a single-scenario story: 4 of the 17 sit
elsewhere. **Because 13 of 17 sit on one scenario, excluding blocked cells removes SC-029 unevenly
across conditions rather than symmetrically** — neither including nor excluding gate-blocked cells
from the primary analysis is obviously correct, and a reader should expect (per D12) the primary
comparison reported both ways rather than one silently chosen. What is not acceptable, and does not
happen here: refused text scored as though it were a real response. This is also the first time in
this project that the safety layer has been observed doing its job against real generated text at
scale, which is itself worth recording as a positive finding about `carelite/safety/`, not only a
data-quality caveat.

**The gate-blocked census is now 24 across all 1,119 rows: the 17 above, plus 7 on the 180 vLLM LC
cells.** The 17 is still the right figure for the five-condition Ollama frame the primary analysis
runs on, and the 24 is the right figure for the table as a whole; quoting either without saying
which frame it belongs to is how the two get confused. The 39 Ollama LC cells contributed none. **A
7-in-180 refusal rate on LC is not yet interpretable as a property of the long-context condition**
— it is a single arm, served by a different stack, and the LC cells have not been scored — so it is
recorded here as a count and nothing more.

**Condition C fell back to Condition-B behavior on 69 of 180 cells (38%).** CRAG graded the
retrieved evidence `relevant` on 111 cells and `none` on 69, and a `none` grade means C generates
exactly as B would on that cell — framework-prompted, ungrounded. **38% of the treatment arm
therefore received no treatment, and any C-vs-B comparison has to be read two ways: pooled (which
compares C's full, realistic operating behavior including its fallback rate, against B) and split
on `fell_back_to_b` (which compares C's retrieval-grounded cells specifically against B, and the
CRAG-fallback cells against nothing, since they are B in substance).** Pooling without disclosing
the split risks either understating C's best-case effect (diluted by 38% of cells that are really B)
or, read carelessly, being mistaken for a retrieval failure rate rather than what it is: the CRAG
gate correctly declining to inject irrelevant evidence, which is the gate working as designed
(`docs/preregistration.md` §8.6b already pre-specifies this as a sensitivity analysis).

### LC was dropped twice on a measured runtime cost, and completed once the runtime changed

**`DECISIONS.md` D11: Condition LC was stopped after 39 of its planned 180 cells.** Measured on the
rented L40S, LC cost **3.3 minutes per cell against 6 seconds for the A/A2/D group — roughly 33×** —
with about 8.5 hours and $8.50 left to run when the other five conditions were within the hour of
finishing. The cause is understood and is not a pipeline defect: `lc_sample()` is deterministic and
query-independent by design (D7), so every LC prompt shares an identical ~119,500-token prefix that
should be nearly free after the first prefill via KV-cache reuse. The measured per-cell time said
Ollama re-prefills that whole prefix on every request instead of reusing it — the design
anticipated a saving that runtime did not deliver. **This was the second time LC was dropped for
cost, by an independent lane, for the same measured reason**: `carelite-judge` had already excluded
LC from the §13 validation subset after measuring ~21 minutes per LC generation locally and
projecting ~59 hours.

**Both lanes were measuring a serving stack, and neither said so.** The conclusion recorded here
until 2026-09-01 was the stronger one — that under this architecture a long-context baseline is not
affordable at the scale the rest of the design assumes, offered as a result about the method. That
claim was not supported by the evidence behind it. Two independent measurements of the same runtime
are one measurement of that runtime, not two measurements of long-context evaluation, and the
agreement between them was reassuring in a way it had not earned. `DECISIONS.md` D13 tested the
premise against a second stack and it did not survive.

**The measurement (D13).** `google/gemma-4-12B-it` served by vLLM 0.28.0 with
`--enable-prefix-caching`, pinned at revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, on one
A100 SXM 80GB, driven by the production prompt assembly: **3.61 s per cell** as the mean of 9 warm
calls (min 2.03, max 6.51), against D11's 198 s — a **54.9×** difference — after a single cold
prefill of 110,653 tokens at 64.31 s. All 180 LC cells then generated in about 21 minutes of wall
clock for roughly **$1.38** of GPU time, against D11's projected ~8.5 hours and ~$8.50. The saving
D7's design anticipated is real; one runtime delivers it and the other does not.

**So the limitation is a narrower one, and it is stated in the terms the evidence supports: under
Ollama, a long-context baseline whose prompts share a large identical prefix is not affordable to
evaluate at this scale, because that stack re-prefills the shared prefix on every request rather
than reusing it from cache.** That is a property of the serving stack. It is not a property of the
design, of `lc_sample()`, or of long-context evaluation in general, and the earlier phrasing of it
here should not be quoted. **The transferable lesson is the reason it was wrong**: when a cost
measurement is used to close a scientific question, the runtime is a variable in that measurement
and has to be named as one. Twice in this project it was not, and the second agreement made the
first look like a confirmation.

**What this restores, and what it does not.** Secondary outcome 3 (C vs. LC —
`docs/preregistration.md` §4) is testable again, in the reduced form D7 already fixed: whether
query-dependent selection beats a fixed context, not whether retrieval beats stuffing the corpus
in, because the corpus still does not fit the window and any selection rule is itself retrieval.
Nothing here re-opens D7. **Nothing here improves the instrument either**: `ie`, `naturalness` and
`ritualistic` are degenerate on this run and stay degenerate, the judge validation study has still
not run, and every result in this project remains EXPLORATORY under D10. A new comparison is not a
stronger comparison, and completing a condition is not evidence about any other condition. **As of
this writing the 180 vLLM LC cells have been generated but not yet scored** — `rubric_score` holds
939 rows against 1,119 generations — so no C-vs-LC result exists, in this document or anywhere
else, and any number presented as one is fabricated. The primary outcome (composite NURSE, A vs.
B), the retrieval comparison (B vs. C), the cross-model baseline (A vs. A2) and the negative
control (B vs. D) are unchanged by all of this; LC never entered them.

**The 39 Ollama LC cells are not pooled into the LC arm, and there are now two reasons rather than
one.** They were always a non-randomised 13 of 60 scenarios. They were also served by a different
stack — a GGUF against HF safetensors, different quantisation, different sampling defaults — and
they realised a different pack: the production packing rule admits 116/116 knowledge base entries
and 151/471 chunks at 117,849 real tokens, which is not what the Ollama run's window admitted. They
are retained, marked `served_by = 'ollama'`, and used only as a paired backend-equivalence sample
against their vLLM counterparts. **The LC analysis arm is `served_by = 'vllm'` and nothing else**;
`carelite/stats/headline.py` prints the per-backend split for exactly this reason, so a pooled
count cannot pass unnoticed.

### The §13 judge-validation study, first measured at n = 30, now confirmed at n = 939

`runs/judge/validation_report.json` and `.txt` record the earlier, dedicated n = 30 validation
run; the figures below are recomputed directly from `runs/judge-holdout/rubric_scores.jsonl`, the
full holdout judging output, so what follows is independent confirmation on real production data
rather than a restatement of the smaller study. The original n = 30 run measured self-consistency
and positional bias directly and ran an instrument check against a generated (not human) synthetic
panel, because no human comparator exists yet; those mechanics are not repeated below except where
the full run adds to them. **None of the following is, or is being reported as, a confirmatory
result.** It is what the instrument does, measured, before any human rating exists to validate it
against.

- **1. The judge's `ritualistic` dimension is degenerate, and this breaks the mechanism behind the
naturalness hypothesis specifically — confirmed on all 939 holdout generations, not only the n = 30
validation set.** At n = 30 it scored 1 on every response. **Across the full 939-generation holdout
run it scored 1 on 912 of 921 scored rows (99%)**, mean 1.02, sd 0.16, only 3 distinct values used
in nearly a thousand generations spanning six conditions including the deliberately degraded
negative control. Build plan v3 predicts Condition B loses to Condition A on `naturalness`
*because* framework prompting induces ritualistic, script-like phrasing — `ritualistic` is the
dimension built to detect that mechanism (`docs/rubric.md`). A judge that emits `ritualistic ≈ 1`
for nearly every response, framework-prompted or not, negative-control or not, cannot register that
effect. `naturalness` is confirmed compromised at scale too: **922 scored rows, only 5 distinct
values, mean 3.08, sd 0.58, with a single value (3) accounting for 795 of 922 (86%)**. `ie` shows
the same shape: 920 scored rows, only 4 distinct values, mean 1.13, sd 0.49. **This is not "no
significant difference" — it is a measurement failure, and build plan v3's most interesting
predicted result (A beats B on naturalness, because B is ritualistic) cannot be tested with this
judge, on this data, regardless of what any test of these three dimensions reports.** State it
exactly that way wherever the naturalness prediction is discussed in any results write-up, not only
in this limitations list — the distinction between "tested and found not significant" and "could
not be tested" is the whole point, and collapsing it into a null result would be the single most
misleading thing this document could let stand.
- **2. Chance-corrected agreement is bounded above by how much variance the judge produces, and no
sample size repairs that.** Measured against the synthetic instrument-check panel at n = 30: the
correlation between a dimension's between-generation variance and its recovered Krippendorff's α is
**r = 0.878** (up from r = 0.818 at an earlier n = 12 check — the relationship strengthens, not
weakens, with more data). Low-variance dimensions (`ie`, `naturalness`, `ritualistic`) average α
**0.140**; high-variance ones (`name`, `understand`, `explore`, `epp`, `de`) average **0.910** —
same generator, same injected noise, same threshold, radically different apparent agreement. This
is Krippendorff's α's own prevalence effect (the kappa paradox): when a dimension's scores barely
vary, *expected* disagreement collapses, so ordinary rater noise dominates the ratio and the
coefficient falls toward zero even where raters are in fact agreeing on nearly every unit. **State
this as a methods finding about chance-corrected agreement as an evaluation instrument, not as an
apology for this judge** — it is the most transferable result this validation study has produced,
and it would recur with any judge or any rater pool scoring a low-variance construct. Its practical
consequence: a dimension failing the confirmatory threshold has **two distinguishable causes** that
deserve different sentences — the judge disagreeing with raters, or the judge not discriminating
between responses at all — and `discrimination.ratio` (not the stability/self-consistency column
alone) is what separates them. Read variance and agreement together, never one without the other.
**The full holdout run confirms which dimensions actually discriminate, at real scale.** Six of the
eleven carry real signal — `name` (sd 1.52), `understand` (sd 1.61), `respect` (sd 1.42), `explore`
(sd 1.85), `epp` (sd 1.75), `de` (sd 1.56), all using the full 1–5 range with no single value at
even half the mass — and they include four of the five NURSE dimensions. `support` (sd 1.14) and
`ib` (sd 1.01) sit in between: neither degenerate nor as sharply discriminating as the top six.
**`respect` and `support` vary meaningfully in judge scores even though `DECISIONS.md` D8
established the knowledge base can ground neither of them with a single entry.** Those are two
independent facts about two different parts of the pipeline — what the judge can tell apart, and
what the retrieved evidence can support — and a results write-up should carry both rather than
letting one imply the other: a dimension the judge discriminates well is not thereby a dimension
Condition C's retrieval can move, and a dimension retrieval cannot ground is not thereby one the
judge fails to measure.

**Judging itself completed cleanly: 939/939 judged, zero errors, 206 minutes, temperature-0
single-pass** with `gpt-oss:20b` (`runs/judge-holdout/manifest.json`) — the deliberate full-run
regime `docs/preregistration.md` §9 specifies, not the 5-sample self-consistency regime, which
stays scoped to the smaller validation subset. **889 of 939 (94.7%) are complete on all eleven
dimensions**; the 50 partial rows are scattered across dimensions rather than concentrated on one,
so no single dimension's numbers should be read as resting on a smaller effective n than the others
without checking.

- **3. All eleven dimensions are exploratory, and for this run that is a correct verdict, not a
failure to report as one.** α and ρ are undefined (`n_units = 0`) for every dimension because no
human ratings exist yet — human rating is sprint 10, deliberately deferred while the harness is exercised against synthetic raters (`carelite/eval/human/__init__.py`). `verdict.reason` in
the report states this precisely: *"With no human consensus there is no comparator, so every
dimension is exploratory for want of one — which is a statement about the study's stage, not about
the judge."* **The threshold machinery itself is independently verified working**, via a synthetic
instrument check that is explicitly labeled *not a result* in the report: a null-control panel
(independent synthetic raters) produces 0/11 confirmatory, as it should; a positive-control panel
(raters converging on a shared truth) produces **5/11 confirmatory at n = 30**, up from 0/11 at
n = 12 purely on paired-unit count clearing the ≥ 30 threshold. The gate moves with the data in the
direction it should, on data with no human in it at all — which is exactly what an instrument check
is supposed to demonstrate before a real comparator exists.

**Supporting numbers, verified directly against `runs/judge/validation_report.json`:** span
grounding admitted **1,584/1,650 = 96.0%** of attempted scores (verbatim span located), all 66
rejections `span_not_found`; **zero `no_score` results across all 1,650 attempted scores** — the
same empty-output, reasoning-token-budget-starvation failure mode documented in
`carelite/retrieval/ablation.py` (`JUDGE_NUM_PREDICT`), which once produced a false 0.000
context-precision reading on R0/R8/R9 of the retrieval ablation before being sized with headroom,
does not appear here. A manual adjudication of 30 spans
found 24 actually support their score (**80.0%, 95% CI 62.7–90.5%**) — the judge lane labeled this
explicitly as **not** the v3 §13 human spot-check, because an adjudicator is not a rater; that
distinction is preserved here rather than blurred, and **the human spot-check remains outstanding.**
Positional bias across 12 reversed-order pairs averages **|Δ| 0.26** scale points across the eleven
dimensions, with `explore` a clear outlier at **−1.08 mean signed delta, 42% of pairs shifting by at
least one full point** — worth a specific look before `explore` is trusted in any comparison that
depends on presentation order.

**A defect this project already fixed is now confirmed against real data, not only unit tests, and
belongs in the reproducibility record for that reason.** `carelite/eval/human/dry_run.py`'s
contamination check (regression-tested by
`tests/unit/judge/test_study.py::test_leaking_calibration_inflates_alpha_on_every_dimension`,
fixing commit `da38cd1`) found that leaking the 5 calibration items into a rating panel's unit list
inflates Krippendorff's α on **every one of the 11 dimensions** — raters are shown the calibration
consensus and discuss it, so calibration ratings converge on a published answer key, and
near-unanimous units mechanically raise α. A synthetic sweep at varying panel sizes
(`runs/judge/contamination_by_size.json`) confirms the direction is universal (`all_positive: true`
at every size tested) and the magnitude shrinks as the clean unit count grows — at the smallest
tested size the mean inflation is +0.048 and the largest single-dimension inflation is +0.113 (on
`support`). The judge lane's own report of the effect measured against the real n = 30 instrument
panels puts it larger still — mean +0.088, maximum **+0.783 on `ie`** — worst precisely on the
floored, low-discrimination dimensions this section already flags as least trustworthy, which is
the load-bearing part: **without this fix, the headline agreement numbers would have been flattered
most exactly where the judge is weakest.** (That specific n = 30 magnitude is reported here as
relayed from the judge lane rather than independently reproduced from a persisted artifact in this
repository as of this writing; the direction, the universality across all 11 dimensions, and the
concentration on low-discrimination dimensions are independently verified above.)

### What this section does not license a reader to conclude

- **Not** that Condition B "ties" Condition A on naturalness. The instrument cannot measure the
  effect on `ritualistic` or `naturalness`, and `ie` is nearly as degenerate — three dimensions
  where "no significant difference" and "could not be tested" are genuinely different claims, and
  only the second is true here.
- **Not** that Condition C's 180 cells may be pooled and compared to B without disclosure. 38%
  (69/180) fell back to B behavior on CRAG's own relevance grade; a pooled comparison is a
  comparison against an already-diluted-by-38%-B arm unless the fallback split is reported
  alongside it.
- **Not** that the 39 Ollama LC cells support any comparison against a complete condition. They
  cover 13 of 60 scenarios, were never randomized for partial analysis, and are served by a
  different stack than the LC arm. D11 records why they exist and D13 why they stay out of it.
- **Not** that a C-vs-LC result exists. The 180 vLLM LC cells were generated on 2026-09-01 and
  scoring is outstanding; `rubric_score` holds 939 rows against 1,119 generations. There is a seam
  here for a comparison and no number in it.
- **Not** that completing Condition LC strengthens any other result in this section. It adds one
  comparison to the study and changes nothing about the instrument, the judge validation, or the
  descriptive status D10 fixed for every finding here.
- **Not** that excluding `gate_blocked` cells is a neutral, symmetric operation. 13 of the 17 sit on
  one scenario (SC-029); excluding them removes that scenario's evidence unevenly across the five
  complete conditions rather than evenly.
- **Not** that any of the above — including the six dimensions that do discriminate — is a
  confirmatory or pre-specified finding in the registered sense `docs/preregistration.md` describes.
  Per `DECISIONS.md` D10, every number in this section is an observation from one local run.

---

## 5. Environment deviations from build plan v3

- **PostgreSQL 18.6** via the EDB installer, not the Homebrew `postgresql@17` the plan assumed.
  **pgvector 0.8.6** compiled from source.
- **Model roster differs from v3's illustrative names**: `gemma4:12b` generator, `qwen3.5:9b`
  cross-model baseline (Condition A2), `gpt-oss:20b` judge (cross-family, satisfying §13
  independence), `bge-m3` embedder at 1024 dimensions — not the `Qwen3-Embedding-0.6B` build plan
  v3 §6 names. All are recorded with digests, not tags alone, per `carelite/config.py` and v3 §16.
- **`bge-m3` is actively harmed by a query-side instruction prefix.** Adding even the 7-character
  prefix `"query: "` before embedding raised cosine similarity between unrelated one-word queries
  from 0.52 to a range of 0.72–0.84 and flattened rankings across unrelated items. Both the query
  and document prefixes in the retrieval pipeline are therefore the empty string `""`, deliberately
  and against the model card's stated usage pattern. This was found empirically during index
  build, not assumed from documentation, and is worth stating because it would not be discovered by
  reading the model card alone.
- **`websearch_to_tsquery` ANDs every content word it is given.** A natural-language lexical query
  built from a full patient utterance returns zero rows where a short keyword query against the
  same corpus returns on the order of 15 hits, because Postgres's `websearch` parser conjoins
  content terms by default rather than treating the phrase as a bag of alternatives. Query
  construction for the lexical leg of retrieval accounts for this rather than passing the raw
  utterance through.
- **Condition LC cannot be "the whole corpus stuffed into context" as build plan v3 §3 specifies,
  and is redefined rather than silently approximated — `DECISIONS.md` D7.** 471 chunks is an
  estimated 326,526 tokens against a 128,000-token context window: **255% utilisation, which is an
  estimate and not a measurement.** D7's figures come from `carelite.generate.model.estimate_tokens`,
  a heuristic, and D13 measured it against the model's own tokenizer for the first time: the
  production long-context pack estimates 123,758 tokens and really is 117,849, so the heuristic
  **overcounts this corpus by about 4.5%**. The direction is the safe one — it errs toward not
  overflowing the window, which is why nothing downstream broke — but every token figure D7 states,
  the 255% included, should be read as approximately that and not as a measured quantity. The
  conclusion it supports is unaffected: 4.5% does not close a 2.5× gap, and the corpus does not fit.
  Reserving 16K for the system prompt, patient turn, and response leaves room for roughly a third of
  the corpus. **LC is now `LC-sample`**: a fixed, query-independent round-robin selection across all 33
  papers at a pinned seed (`carelite.retrieval.ablation.lc_sample`), 169 chunks, 35.9% of the
  corpus. Round-robin rather than random sampling guarantees every paper is represented, so LC's
  content is not an accident of the seed. **The point that must not be quietly absorbed: any
  selection rule is itself a form of retrieval.** LC was meant to ask whether curated retrieval
  beats stuffing everything in; it can now only ask whether *query-dependent* selection (Condition
  C) beats a *fixed* context (LC-sample) — a real and interesting question, arguably closer to what
  a practitioner would actually build, but a different one from what build plan v3 posed. This
  distinction is carried in `docs/preregistration.md` §2 and §4 rather than left to a results-section
  footnote.
- **D7's window-filling fix made LC implementable, and under Ollama it was not affordable — which
  is why this project rents GPU hardware at all, and why LC did not complete on the first
  attempt.** `carelite/eval/judge/study.py` measured LC's prefill directly on local hardware first —
  roughly 119,500 tokens at ~95 tok/s, about 21 minutes per generation, projecting ~59 hours for LC
  alone — which is what moved the holdout run to a rented L40S. **Even rented, LC still cost roughly
  33× the other conditions per cell under Ollama** and was stopped 39 cells into its planned 180 by
  `DECISIONS.md` D11. **Both of those measurements were of Ollama.** Under vLLM 0.28.0 with
  `--enable-prefix-caching` the same work costs 3.61 s per warm cell after one 64.31 s cold prefill,
  and all 180 cells completed in about 21 minutes for roughly $1.38 (D13). The affordability
  limitation is real and belongs to the serving stack, not to the condition; §4 above ("LC was
  dropped twice on a measured runtime cost, and completed once the runtime changed") has the full
  account and the numbers on both stacks.
- **Index build is complete and independently verified: 471/471 chunks embedded.** The 342
  pre-existing embeddings were confirmed byte-identical against a fresh embed, 0 mismatches, and
  mean pairwise cosine across the corpus is 0.5788 (decomposing to old-old 0.5755, new-new 0.6100,
  cross 0.5773 — no discontinuity at the seam, which is the shape a degenerate partial rebuild
  would have shown). 10/10 retrieval probes pass, including two that depend directly on
  `carelite-corpus`'s extraction fixes landing: the `teach-back` probe now matches inside the
  Talevski systematic review and the `disparit` probe inside the PLOS empathy-disparities paper.
  **One bookkeeping artefact, harmless but worth knowing before reconciling row counts:**
  `index_embedding_state` carries 475 rows for `kind='chunk'` against 471 actual chunks — 4
  orphaned rows left over from chunk-ID renumbering. Scoped `only_ref_ids` embedding calls
  deliberately skip orphan pruning (a targeted call has no way to know a stale ID was ever valid),
  so these persist until a whole-table pass sweeps them; they do not affect retrieval, only the
  state table's row count.
- **A candidate lexical-matching gap was investigated and found not to exist, within the scope
  searched.** It was reasonable to worry that words split or joined across a PDF column or line
  break (`show ing`, `healthrelated`) would survive into the indexed chunk text and silently fail
  to match their correctly-spaced or hyphenated forms in lexical search. A general repair rule for
  this was considered for the indexing pipeline and rejected — not because the artefact doesn't
  matter, but because the rule's measured false-positive rate broke real words more often than it
  fixed anything (`healthcare` → `health`/`care`, `Pearson` → `Pears`/`on`, `asking` → `as`/`king`).
  A targeted search then checked whether the artefact class that rule would have addressed is
  actually present: the two named examples plus a ten-suffix sweep (`-based`, `-making`,
  `-related`, `-centered`, `-focused`, `-informed`, `-oriented`, `-driven`, `-friendly`,
  `-sensitive`) across all 471 indexed chunks, matching zero glued occurrences of that pattern —
  the only hits were real words such as `interrelated`. **This rules out that specific artefact
  class in the current corpus; it is not a claim that no glued word of any shape exists anywhere in
  it**, since the sweep is suffix-pattern-based rather than exhaustive. No retrieval penalty from
  this artefact class is implied or has been observed, and none should be inferred from earlier,
  less precise statements of this finding.

---

## 6. Scope, per build plan v3 §17

- **Small corpus** — 33 retrieved papers, concentrated further within themes as described above.
- **Synthetic scenarios**, not real patient utterances, throughout the bank.
- **Single- or short-turn interactions.** The system responds to one patient turn at a time with
  bounded history; it does not model a full multi-phase encounter end to end.
- **Open-weight model capability ceiling.** Every generator, judge, embedder, and reranker in this
  project is an open-weight model the project serves itself — via Ollama or `sentence-transformers`
  locally, and, for Condition LC's 180 cells, via vLLM on a rented GPU (D13). No hosted frontier
  model appears anywhere in the inference path. Results describe what this system does with these
  models, not the ceiling of the largest hosted models. **"Local" is now a statement about where
  the serving process ran, not about which models were used**, and the two conditions of that
  sentence came apart when LC moved to a rented pod; `generation.served_by` is what records which.
- **One particular operationalization of NURSE and the Four Habits Model.** `docs/rubric.md` fixes
  eleven scored dimensions with anchored examples; NURSE and 4HM each admit other defensible
  operationalizations, and this rubric's judgments are specific to the one built here.
- **No patient-reported outcomes.** Every measured quantity is a rubric score of the clinician-facing
  text the system produces — adherence to frameworks that are themselves proxies for patient
  experience, not a measurement of that experience. The frameworks' own validation literature is
  mixed: the internal notes behind this project already flagged a Four Habits Model course with no
  significant post-course empathy change.
- **No clinical deployment claim.** This is a prototype and evaluation framework, not a system
  intended, tested, or approved for use in an actual clinical encounter. See the "What this project
  is not" section of `README.md`.

---

*Last updated 2026-09-01 for `DECISIONS.md` D13, alongside `REPRODUCE.md`, `docs/reporting/`, and
`docs/decisions/`. Figures sourced by direct query against the live `carelite` database and against
`carelite/scenarios/freeze.py`, `knowledge_base/TAXONOMY.md`, `carelite/kb/validate.py`, and
`DECISIONS.md` — not carried over from planning-time estimates. Row counts stated here are a
snapshot of that query; `runs/repro/headline-numbers.txt`, regenerated by `make reproduce`, is the
authority and is what a write-up should quote. Where a figure is expected to change (the equity
re-extraction under D3, human rating, judge validation, and the scoring of the 180 vLLM LC cells),
that is stated above rather than left for the reader to infer.*
