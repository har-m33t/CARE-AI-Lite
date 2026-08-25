# Project decisions

Decisions that the build plan routed to the project owner, recorded here with the
reasoning that produced them. A decision listed here is settled; a lane that
disagrees should raise it rather than re-open it in passing.

Decisions were delegated to the orchestrating session on 2026-08-24. One gate in
the build plan is **not** recorded here because it cannot be delegated: OSF
pre-registration. See "Gates that remain with a person" at the end. The knowledge
base sign-off gate was removed rather than delegated — see D4.

---

## D1 — Theme taxonomy: seven, not ten

**Decision: adopt the seven themes already encoded in `carelite.types.Theme` and
defined in `README.md`. The "10 themes" figure in build plan v3 is retired.**

The `carelite-kb` lane's proposal in `knowledge_base/TAXONOMY.md` is accepted in
full, including its recommendation not to add serious-illness conversation as an
eighth theme. Its central observation is the decisive one: the ten-theme number has
no referent. It appears once, in a sentence arguing that the knowledge base is
already a curated knowledge graph, and there is no list, definition, or derivation
of ten themes anywhere in the repository. The choice was never between two
taxonomies — it was between a taxonomy and a number, and adopting ten would have
meant inventing three themes to satisfy a count.

The loaded knowledge base strengthens the case. At 127 entries over seven themes,
three themes already hold ten entries or fewer (emotion_response 10,
trust_continuity 13, equity 3). `settings.retrieval.rerank_top_n` is 4, so a
theme-filtered retrieval in those themes already returns barely more candidates
than the reranker consumes. Splitting finer would produce facets that cannot
function as filters.

Serious-illness conversation remains the largest coherent cluster in the corpus
without a theme of its own. It stays modelled as `EncounterPhase`, per the lane's
argument that what distinguishes it is where in the encounter and at what stakes,
not which communicative function is being performed.

`carelite/types.py` is unchanged by this decision, which is the point: it confirms
a frozen contract rather than amending one.

## D2 — Equity stratum: two scenarios reclassified, five kept

**Decision: `SC-077` and `SC-010` leave the equity stratum. Their text is
unchanged and they stay in the bank. `SC-029`, `SC-045`, `SC-065`, `SC-088`, and
`SC-090` are accepted into the equity stratum as written.**

This follows the two the `carelite-scenarios` lane nominated itself in
`scenarios/EQUITY_REVIEW.md`, on the lane's own stated criteria.

**`SC-077`** — *"…a man at my church said the same thing happened to his
brother."* Community-sourced health information weighed against a prescription is
real, well documented, and **not specific to any group**. The church detail is what
makes the scenario read as coded for one, and no finding in the corpus ties
congregational information-sharing to the disparity the `racial_ethnic` stratum is
measuring. The communication challenge is genuine and worth keeping; counting it as
evidence of a documented disparity would put a scenario in the stratum that is not
measuring what the stratum claims to measure.

**`SC-010`** — *"Yes, thank you doctor. Everything is good. Maybe my son can call
you with the questions."* This fails the review packet's own rule 2: LEP is to be
signalled situationally, never grammatically. Every other `lep` scenario marks
itself with an event — an interpreter who has left (SC-033), a nephew summarising
(SC-050), a phone interpreter laughed at (SC-086). SC-010 marks itself with
*register*: clipped sentences and deferential phrasing. That is closer to the
grammatical marker the rule forbids than to a situational one, and the rule is
right — politeness is not a nationality. It is also partly redundant with SC-050,
which covers the family-member-handles-the-questions pattern situationally.

Both remain valuable non-equity scenarios: SC-077 as an adherence and trust case,
SC-010 as a false-comprehension and deference case with a low-affect closing cue,
which is a difficulty the bank should test.

On the five kept, briefly. **SC-088** is legible as an equity scenario only to a
reader who knows the under-treatment-of-pain literature — that is acceptable,
because the stratum is an analysis label and does not need to be legible to the
model. **SC-065**'s "people like me" is doing real work: it requests a social norm
as a substitute for a preference, which is the documented lower-participation
pattern rather than a caricature of it. **SC-045**'s SES signal is carried by the
six o'clock shift alone and is the weakest of the five, but neither the mother nor
the adolescent reads as a problem rather than a person under pressure. **SC-029**
is the strongest of the seven: the difficulty is unambiguously in the discharge
material. **SC-090**'s warmth reads as a person.

**Timing matters and this is the free moment.** The bank's holdout split is frozen,
so a change to stratum metadata is a protocol amendment — but no evaluation data
exists yet and OSF pre-registration has not happened. Amending now costs nothing;
the same change after registration would have to be declared. The amended bank is
what gets registered.

Resulting counts: equity stratum 37 → 35, `lep` 11 → 10, `racial_ethnic` 10 → 9.
The `carelite-scenarios` lane re-runs its coverage audit and re-freezes.

## D3 — Equity knowledge base entries: re-extract with a revised prompt

**Decision: approved, and sequenced to run once `carelite-corpus` has landed its
extraction fixes, so it runs once against clean text rather than twice.**

The `carelite-kb` lane recommended this rather than doing it, which was the right
call to escalate. Three equity entries out of 127 understates what the corpus
holds, and the lane established that this is not under-sampling — the extractor read
the right pages, and the Roberts meta-analysis scores 162 on equity vocabulary.

The cause is structural and worth stating precisely, because it is a finding in its
own right. The equity literature *describes* a disparity, so a faithful extraction
of it produces an awareness statement — *"clinicians should be mindful of empathy
gaps in patients from lower socioeconomic backgrounds"* — and the actionability
gate rejects awareness statements correctly, because awareness is not something the
system can detect, generate, or reframe. Six of the nine equity rejections share
that one sentence shape.

The revision instructs the extractor that where a passage reports a disparity, the
entry must name the **compensating move**, not the awareness. The surviving Roberts
entry is the model: *check your assumptions about this patient's adherence and pain
needs*, not *be aware that assumptions exist*.

**The risk is real and must be guarded, not assumed away.** A prompt told to find
compensating moves will find them whether or not the passage supports one. The span
requirement does not catch this, because the span can be perfectly genuine while the
takeaway drifts beyond it. The guard is that the takeaway must be supported by the
quoted span rather than merely adjacent to it, and equity entries from the re-run
get read individually rather than sampled.

This bumps `PROMPT_VERSION` and invalidates the cached extraction windows. That cost
is accepted.

---

## D4 — The knowledge base is not human-verified, and will not claim to be

**Decision (2026-08-24, project owner): drop the human-verification gate. The
provenance claim is amended to what is actually true — "LLM-assisted extraction
with automated verbatim-span validation, no human verification" — and the build
proceeds. `human_verified` stays `FALSE` on all 127 entries as the honest record
of that.**

The gate was blocking, and the alternative to blocking on it was ticking it
falsely, which would have put an untrue sentence in the methods section of a
write-up whose whole argument is that its provenance is checkable. Removing the
claim is the honest resolution; the entries are exactly as good or as bad as they
were, and now the documentation says so.

What the knowledge base *can* still claim is not nothing, and the write-up should
state it precisely rather than retreating to a vague disclaimer:

- Every entry's `verbatim_span` was located in the extracted text of the paper it
  cites, and what is stored is a literal slice of that source rather than the
  model's rendering of it. This was verified against the database, and separately
  spot-checked by re-extracting sampled papers from their original files.
- Entries were rejected for a fabricated span, an unsupported evidence tier, a
  non-actionable takeaway, or a span too short to carry evidence. The genuine
  fabrication rate over all candidates was measured, not estimated.
- No human read the entries for whether the *finding* follows from the *span*.
  That is the specific thing an automated check cannot do and that is now not
  claimed.

The review machinery in `carelite/kb/review.py` is kept rather than deleted. It
cost little, it is tested, and it is what a later reviewer would need. It is now
an available tool rather than a required gate, and the digest should stop
describing itself as something that must be completed.

**This becomes a limitations entry, not a footnote.** `docs/limitations.md` must
record it in the same register as the corpus shortfall: the knowledge base is
machine-extracted and machine-validated, a documented share of candidates were
rejected as fabrications, and the surviving entries carry provenance that is
mechanically checkable but semantically unreviewed. Any result that depends on
knowledge base quality inherits that limitation.

---

## Gates that remain with a person

**OSF pre-registration.** Agents draft it; registration is an account-holder
action and an irreversible public act. It must happen before inference lane III
generates any evaluation data — the argument that an against-you naturalness
result is credible rests entirely on the analysis having been fixed in advance.

---

## D5 — The `racial_ethnic` axis is narrower than its name, and will be described as what it is

**Decision: keep D2's reclassification. Do not write replacement scenarios into
the holdout. Describe the axis accurately in the analysis plan and the OSF
pre-registration, and pre-specify both coverage gaps as known limitations of the
equity subgroup analysis.**

Acting on D2 cost the equity stratum two things that D2 did not anticipate, both
surfaced by the `carelite-scenarios` lane's own audit rather than by the gate that
was supposed to catch them.

**The mechanism confound.** Eight of the nine remaining `racial_ethnic` scenarios
turn on one mechanism: the patient has already been disbelieved, or expects to be,
and manages the clinician accordingly. SC-077 was the only one whose difficulty
originated outside the clinic. So a system that scores well across this axis may be
scoring on *handles a guarded patient* rather than on the disparity the axis claims
to measure — and the coverage audit cannot see this, because it measures challenge
type, phase, intensity and literacy, none of which separate a guarded patient from
an unguarded one.

**The reason not to fix it by restoring SC-077, or by amending its text.** The
obvious repair is to strip the church reference and keep the scenario in the
stratum — the review packet offered that as the alternative to reclassifying. It
does not work. The external-origin mechanism *is* the community-source detail; take
it out and what remains is a patient pre-emptively dismissing himself, which is the
same guarded-patient mechanism as the other eight. The mechanism cannot be
preserved without preserving the coding problem that D2 removed.

**And the mechanism is not anchored in this corpus anyway.** The `racial_ethnic`
axis draws on documented emotional blocking of minority patients (Park et al. 2020)
and the SES and race empathy gap (Roberts et al. 2021). Nothing in the 33 papers
documents community-sourced health information as a disparity mechanism. Writing a
replacement holdout scenario for it would be inventing coverage the literature does
not support, which is the failure mode this project rejected when it declined to
invent three themes to reach ten.

**So the axis is named wrong, not populated wrong.** What these nine scenarios
actually measure is the clinician's response to *anticipated dismissal and patient
credibility-management*. That is a real, documented, and important thing, and it is
narrower than "race-based disparity in communication". The analysis will say so.
`equity_kind` keeps its current values for continuity with the frozen split; the
description changes, in the analysis plan, the pre-registration, and the results
write-up.

**Two gaps pre-specified as limitations rather than repaired:**

- The equity stratum no longer contains an `emotion_intensity = 1` scenario, so it
  cannot say whether the disparity behaves differently on an emotionally flat turn
  — which is the turn where a system that over-reads emotion does its worst work.
  Flat turns are still tested outside the equity subgroup.
- `racial_ethnic` contains no `adherence_barrier`, `decision_conflict`, or
  `false_comprehension` scenario, and every scenario presents an already-guarded
  patient.

Both must be declared in the pre-registration **before** any evaluation data
exists. A limitation named in advance is a limitation; the same sentence written
after seeing the results is an excuse.

**On the audit allowlist.** The lane's `ACCEPTED_EMPTY_CELLS` in
`carelite/scenarios/audit.py` is accepted. An empty cell that exists *by decision*
should be loud and attributed rather than either silently tolerated or left failing
a check that three concurrent lanes depend on. It holds exactly one entry, names
D2, prints on every run, and is pinned by a test so a second hole cannot be added
quietly — which is the right shape for this. A cell not on the list still fails.

**On the second-person review.** It remains outstanding and unticked, and
`EQUITY_REVIEW.md` records plainly that the orchestrating session's review is not
an independent second-person one. Per D4's principle, an unticked box that is
honest is worth more than a ticked one that is not. The lane is right that before
pre-registration is the last point at which a finding from such a review would be
free to act on.

---

## D6 — `README.md` belongs to `carelite-repro`

**Decision: extend the `carelite-repro` lane's owned paths to include `README.md`.
No other lane may write it.**

The `carelite-repro` lane found a real hole in the fleet specification and stopped
rather than working around it, which is the behaviour the ownership contract is
supposed to produce. `README.md` appears in no lane's **Owns** list. It was authored
by `carelite-foundation` at wave 0, but foundation's ownership covers the frozen
contracts and the toolchain, not the project's front page. So the one file most
likely to drift out of date had nobody responsible for it.

`carelite-repro` is the right owner. It already owns `docs/`, `REPRODUCE.md`, the
reporting checklists, and the limitations record — it is the lane whose whole job is
keeping the written record true to what the code actually does. The README's Status
table and Project Structure section are exactly that job, and splitting them from the
rest of the documentation would reproduce this gap in a smaller form.

**What needs correcting, and why it matters more than housekeeping.** `.claude/CLAUDE.md`
still tells every agent that the README documents an *intended* layout, that
`literature/`, `framework/`, `knowledge_base/`, `behaviors/` and `docs/` do not exist,
and that the only executable file is `data/fetch_corpus.py`. That was true when it was
written and is now badly wrong: there are thirteen packages under `carelite/`, a test
suite in the four figures, a populated Postgres database, and most of those directories
exist. An agent reading that guidance today is being actively misled about the state of
the repository — which is a correctness problem for the fleet, not a presentation
problem for a reader.

The README must also stop describing the knowledge base in terms D4 retired. Anything
implying the entries are human-verified or clinician-reviewed is now false, and the
front page is the most likely place for that claim to survive after being corrected
everywhere else.

`.claude/CLAUDE.md` itself stays with the orchestrating session, since it is fleet
instruction rather than project documentation, and is corrected in the same pass.

---

## D3 — outcome (2026-08-24): the re-extraction did not work, and that is the finding

**Recorded outcome: zero net new equity entries. The variant was not loaded. `equity`
stands at 3, and it is now established as a property of the corpus rather than of
the prompt.**

D3 approved re-extracting equity entries with a prompt that asks for the
*compensating move* rather than the awareness statement, on the reasoning that three
entries out of 114 understated what the corpus held. It did not. Six candidates came
back from the two anchor papers: four were aspirations ("proactively work to bridge
the empathy gap", "engage in more consistent and attentive communication"), one
duplicated an existing span exactly, and one was a genuine compensating move that
restated advice already in the base.

**Which paper failed is the whole result.** Holdsworth *describes interactions*, so a
compensating move was there to quote and the model quoted one. Roberts is a
meta-analysis: it quantifies a gap and correctly never says what closes it. Asked for
a move, the model invented one. That is not a prompt defect and no prompt fixes it —
the literature that measures a disparity is not the literature that prescribes a
remedy, and this corpus holds the former.

**The guard in D3 was the part that mattered.** It said the takeaway must be supported
by the quoted span rather than merely adjacent to it, and that every equity entry from
the re-run be read individually rather than sampled. Both were applied, and the run was
quarantined behind `--prompt-version` so an experiment reaches the knowledge base only
once its guard has been applied — never merely because inference finished. Without
that, four aspirations would have loaded and `equity` would read 7 instead of 3, which
would have looked like progress and been a regression.

**One judgement recorded because it could have gone wrong.** The lane added three
clauses to an aspiration filter to reject that sentence shape, and correctly identified
that such a filter can become self-fulfilling — tuned until the answer it produces is
the answer expected. Its guard was that each clause be justifiable without reference to
equity and measured for false positives across the whole base; the three added match 1,
0 and 0 of 114 entries. It **declined** a fourth clause that matched three good entries,
and recorded in the code that the actionability gate cannot be the whole of the guard.
That refusal is the reason the filter is trustworthy.

**Consequence for the write-up.** `equity` at 3 entries, and the `racial_ethnic`
scenario axis narrowed under D5, are now two independent measurements of the same
underlying gap: this corpus documents disparities far better than it documents what to
do about them. That belongs in `docs/limitations.md` as a finding about the evidence
base, stated once and plainly, not as an apology distributed across three sections.

---

## D7 — The long-context baseline cannot be the whole corpus, and will not claim to be

**Decision: LC becomes a fixed, query-independent round-robin sample across all 33
papers at a pinned seed, reported as `LC-sample`. The pre-registration must state
that this is "no *query-dependent* retrieval" rather than "no retrieval", because
that is a different claim from the one build plan v3 §3 makes.**

The `carelite-retrieval` lane measured what v3 assumed: 471 chunks is roughly
**326,526 tokens against a 128,000-token window — 255% utilisation.** Reserving
16K for the system prompt, the patient turn, and the response, about **161 chunks
(34%) fit.** The long-context condition as specified is not implementable, and no
amount of care in the analysis recovers that.

**Round-robin across papers rather than random sampling.** Round-robin guarantees
every one of the 33 papers is represented and is trivially deterministic; random
selection can drop whole papers by chance, which would make LC's content an
accident of the seed and its comparison to C partly a comparison of which papers
happened to survive.

**The part that must not be quietly absorbed.** Any selection rule is a form of
retrieval. LC was supposed to be the baseline that asks whether curated retrieval
beats stuffing everything in; it can now only ask whether *query-dependent*
selection beats a *fixed* context. That remains a real and interesting question —
arguably closer to what a practitioner would actually build — but it is not the
question v3 posed, and a reader must not be able to mistake one for the other. The
row is named `LC-sample` for that reason, and the pre-registration carries the
distinction rather than a footnote in the results.

This is settled now because the pre-registration has not been submitted. After
registration it would be a protocol amendment; today it is free, and it is exactly
the class of thing §10 exists to force into the open beforehand.

## The ablation gate was mis-specified, and the fix is recorded here because it changes a reported number

The first completed R0–R9 run reported context precision of 0.167–0.250 against a
`> 0.7` gate, with every non-CRAG row marked FAIL and the two CRAG rows marked PASS
at 1.000.

Both halves were artefacts. Half the turns are `OFF_DOMAIN_TURNS` — deliberately
unanswerable — and the relevance judge correctly rules that no passage from this
corpus helps them, so each contributes a structural zero. **No non-CRAG row could
exceed roughly 0.5, against a gate of 0.7: the table was testing something no
configuration could pass.** And the two PASSes scored 1.000 on `n_scored = 1`,
which reads as CRAG improving precision when what it had done was reject five of
six turns.

Fixed: precision is computed on-domain only and that is what the gate tests;
rejection is reported as two separate columns, `off-dom rej` (high means CRAG is
working) and `on-dom fb` (what it costs); and no verdict prints below
`MIN_SCORED_FOR_GATE = 5`, so a sample of one can never read PASS again. On-domain
precision on the same run is **0.334–0.380** — still short of 0.7, still worth
understanding, and a materially different claim from "0.167 FAIL".

**The 83% fallback was investigated with evidence rather than settled by judgement.**
The stale-anchor hypothesis — that the CRAG cosine thresholds were calibrated before
the corpus was re-extracted and 129 chunks re-embedded — does not hold, for two
independent reasons: the anchors have exactly one call site, reached only when the
LLM grader is unavailable, and that run shows every turn graded by the LLM; and they
were re-measured after the re-extraction anyway, moving by less than 0.003 with
separation intact. The lane made this verifiable rather than arguable by recording
the deciding grader per turn in the table. What remains is **corpus skew**, which is
consistent with everything else known about this evidence base: 18 of 33 papers are
communication-skills *training* studies, teach-back rests on one paper, equity on
three entries. Those on-domain rejections are the gate reporting a real property of
the corpus, and they belong in `docs/limitations.md` rather than in a retuned
threshold.

---

## D8 — Two of the eleven scored dimensions cannot be grounded in this corpus

**Finding, recorded before registration because it bears on the primary outcome:
NURSE `respect` and `support` have zero knowledge base entries. The judge scores
them; retrieval cannot help with them. This must be stated in
`docs/preregistration.md`, not discovered in the results.**

`carelite-kb` derived the behavior-to-framework mapping from each entry's
`practical_takeaway` and `example_behavior` — what the clinician is told to *do* —
deliberately never from `finding` or `verbatim_span`, which say what a study
*measured*. That distinction does real work: an entry drawn from an empathy trial
can prescribe a comprehension check, and matching on the finding would file it
under `de` when the act is `ie`.

Final coverage across 116 entries: `ie` 40, `epp` 17, `name` 15, `ib` 6, `explore`
5, `understand` 5, `de` 4, **`respect` 0, `support` 0**. Forty entries instantiate
none of the nine and keep empty fields — "request an in-person interpreter" is good
advice and is not one of the nine moves.

**The zeros survived a correction that would have hidden them.** The lane's first
mapping filled `respect` and `support` and looked plausible. Reading every
assignment against its entry found seven false positives, two of which were exactly
those: *"verbal affirmations to show you're listening"* matched `respect` but is a
backchannel cue, not crediting the patient for anything specific; *"collaborative
partnership"* matched `support` but is a stance with none of the concrete half —
who does what, how to reach someone. Removing them took both dimensions to zero,
and they stayed there.

Nothing in the 33 papers turns NURSE Respecting or Supporting into a finding with a
quotable span and an actionable takeaway. A regression test pins each zero, and a
companion test proves the pattern still fires on a genuine crediting move — so the
zero is a property of the corpus, not a broken matcher, and a future change that
fills them has to argue with a failing test.

**Why this reaches the pre-registration rather than only the limitations record.**
The primary outcome is composite NURSE adherence, which averages `name`,
`understand`, `respect`, `support` and `explore`. Two of those five have no
evidential support in the retrievable base. Condition C's advantage on the primary
composite therefore cannot come from retrieval on two-fifths of its constituent
dimensions, and any C-over-B effect there has some other cause — prompt framing,
generator behaviour, or noise. Stated in advance that is a sharp, interpretable
constraint on the primary result. Discovered afterwards it would look like an
excuse.

The composite is **not** being redefined to drop them. Redefining a primary outcome
after seeing which dimensions are convenient is precisely what pre-registration
exists to prevent, and the rubric measures what a clinician does regardless of
whether this corpus can teach it. The dimensions stay; the constraint is declared.

**A related asymmetry the digest now carries:** 13 entries are flagged
`equity_relevant` while the `equity` theme holds 3, with 10 of those spread across
`plain_language`, `teach_back` and `trust_continuity`. Reporting only the theme
count understates what the base holds; reporting only the flag would overstate how
much of it is *about* a disparity. Both numbers, or neither.

---

## D9 — Six analysis specifications the pre-registration left open

`carelite-stats` implemented `docs/preregistration.md` and found six places where the
document does not determine the analysis. Each is settled here and must be written
into the document **before** it is registered; an analysis choice that is only in the
code is not pre-specified.

**1. The equity subgroup is n = 20, not 35. Factual correction.** §8.4's parenthetical
says 35 equity scenarios, but §6 restricts confirmatory analyses to the holdout and 15
of the 35 are in the train split. The holdout stratum is 20 — `ses` 10, `lep` 4,
`racial_ethnic` 6. §8.4's own restricting clause already produces this behaviour, so
the parenthetical is simply wrong rather than the design being changed. **At n = 20
the subgroup resolves only dz ≈ 0.68 — large effects only — and `racial_ethnic` at
n = 6 supports nothing.** Both numbers print with the result. This is the third
independent measurement of the same underlying problem, alongside D3's equity-at-3 and
D5's single-mechanism narrowing: the equity analysis in this study is descriptive, and
the document should say so in those words rather than implying a powered test.

**2. The Holm family is the eight registered comparisons, not fifty-five.** §8.1 admits
two readings: the eight comparisons §4 actually lists, each on the measure §4 names, or
the full 5 conditions × 11 dimensions grid. **Eight.** §4 lists eight comparisons with
their own measures and does not list A vs C, or `de` for A vs B; folding in 47
unregistered tests would multiply the correction on the registered eight roughly
sevenfold, which penalises the pre-specified analysis for exploratory breadth. The
55-cell grid is built as `dimension_expansion()`, corrected within its own family and
stamped **EXPLORATORY**. Both remain reported — the point is which one carries
confirmatory weight.

**3. Every test is two-sided.** §8.1 does not say, and the lane's reasoning decides it:
a one-sided test would have **no power against the naturalness result the document
exists to protect.** §4.4 registers A > B on `naturalness` — an against-the-system
prediction — and the whole §10 argument is that this study can report a finding that
does not flatter it. A one-sided test aimed at the registered direction would make the
opposite outcome unobservable. Two-sided throughout; the registered direction is
recorded and compared against the observed one, so "significant against the registered
direction" stays detectable and reportable.

**4. "Poor self-consistency" in §8.5(c) is `pct_range_ge_2 > 0.25`.** A number has to be
in the document or the sensitivity analysis is not pre-specified. A generation is
excluded when more than a quarter of its judge samples span two or more rubric points —
that is disagreement about *which anchor applies*, not rounding between adjacent ones.
Until this lands in the document the result correctly carries
`threshold_prespecified = False` with its reason printed.

**5. All three point estimators are reported, always.** §8.2 names none. Rank-biserial
is the estimator belonging to the Wilcoxon test, Cohen's dz is the scale the power
analysis is expressed in, and Hodges–Lehmann is the location shift. Reporting all three
for every comparison, unconditionally, removes the opportunity to choose one after
seeing which is largest.

**6. A composite is confirmatory only if every constituent dimension is.** §9 can demote
individual dimensions to exploratory, and the document does not say what that makes a
composite built from them. Weakest-link. A composite NURSE score whose `explore`
dimension failed the judge-agreement threshold is not a confirmatory result with one
soft edge; it is a number whose provenance is mixed, and the honest label is the weaker
one.

**Note how 6 interacts with D8.** D8 records that `respect` and `support` have zero
knowledge base grounding. That is a statement about the *evidence base*, not about judge
agreement, and it does not itself demote anything under §9 — the judge can score a
dimension reliably whether or not the corpus can teach it. The two constraints are
independent and both belong in the document, because together they say something
precise about the primary composite: two of its five dimensions cannot be helped by
retrieval, and any of the five may additionally be demoted on agreement grounds.

**Also confirmed and worth recording**, because it is the kind of thing that silently
doubles a sample: `carelite/eval/judge/store.py` writes both per-sample judge rows and a
`-median` aggregate row, so a naive `rater_type = 'llm_judge'` select double-counts every
generation invisibly. The stats lane's queries partition the two and a test asserts it.

---

## D10 — Scope: local proof of concept. Pre-registration dropped, results are descriptive

**Decision (2026-08-24, project owner): this is a personal proof of concept that needs
to work locally. It is not being published, submitted, or handed to anyone else. OSF
pre-registration is dropped and the holdout gate is removed.**

The pre-registration existed to serve one argument, from build plan v3 §10: that a
naturalness result going *against* the system reads as a finding rather than an
excuse, because the prediction was fixed in public beforehand. That argument only
buys anything if there is an audience. There is not, so the gate was costing a hard
block on the remaining work in exchange for credibility nobody will be asked to
extend.

**What this changes, stated plainly so no later reader has to infer it:**

- Every result this project produces is **descriptive**. The eight comparisons, the
  equity subgroup, the naturalness prediction — none may be described as
  confirmatory, pre-specified, or hypothesis-testing, in the README, in
  `docs/limitations.md`, or in any write-up. They are observations from a single
  local run.
- `docs/preregistration.md` is **kept**, not deleted. It remains an accurate,
  timestamped-in-git record of the analysis plan as it stood before any holdout data
  existed, and the analysis still follows it. That is worth something as evidence of
  intent even without a registry — it just cannot be *claimed* as pre-registration.
  Its status header must say so.
- The statistical machinery is unchanged. Holm correction, two-sided tests, bootstrap
  CIs before p-values, the weakest-link composite rule, judge-agreement demotion — all
  of it still runs, because it is how you avoid fooling yourself, which matters
  whether or not anyone else is watching.
- D1–D9 all stand. Nothing about the corpus, the knowledge base, the equity findings,
  or the instrument defects becomes less true because the audience changed.

**The `--preregistration-is-submitted` flag must not be used to get past the gate.**
Its name is an assertion, and asserting something false to unblock a run is exactly
the habit this project has spent two days building guards against. The gate is
changed to reflect the actual decision rather than defeated by a flag that lies about
it.

The one thing genuinely lost: if this project is ever written up for real, the
naturalness result cannot be reclaimed as pre-specified. Registering later, after the
data exists, would be worse than not registering at all.

---

## D11 — Condition LC is dropped from the evaluation run

**Decision (2026-08-25, project owner): stop generating LC. The 39 LC cells already
produced are kept as a partial record; the remaining 141 are not generated. The five
other conditions complete in full.**

LC was costing more than the rest of the experiment combined. Measured on the L40S at
**3.3 minutes per cell against 6 seconds for the A/A2/D group** — roughly 33× — with
~8.5 hours and about $8.50 left to run when the other five conditions were within the
hour of finishing. For a local proof of concept that is not a defensible ratio.

**The cause is understood and is not a defect in the pipeline.** `lc_sample()` is
query-independent and deterministic by design (D7), so all 1,080 LC prompts share an
identical ~119,500-token prefix. That should be nearly free after the first prefill
through KV cache reuse. The measured per-cell time says Ollama is re-prefilling the
whole prefix on every request instead. The design anticipated the saving; the runtime
does not deliver it.

**What is lost.** Secondary outcome 3 — C vs LC, "does query-dependent selection beat a
fixed context" — can no longer be tested. That was already the *reduced* form of the
question under D7, which established the corpus does not fit the window (255%
utilisation) and that any selection rule is itself a form of retrieval. So the claim
being given up is a second-order one, and it was already weaker than build plan v3 §3
intended.

**What survives intact.** The primary outcome (composite NURSE, A vs B), the retrieval
comparison that motivates the whole architecture (B vs C), the cross-model baseline
(A vs A2), and the negative control (B vs D). Every condition the study's main
questions depend on is complete at 3 samples across all 60 held-out scenarios.

**This is the second time LC was dropped for cost, by an independent lane, for the same
measured reason.** `carelite-judge` excluded it from the §13 validation subset after
measuring ~21 minutes per LC generation locally and projecting ~59 hours. Two lanes
reaching the same conclusion from different hardware is itself the finding: **under
this architecture, a long-context baseline is not affordable to evaluate at the scale
the rest of the design assumes.** That belongs in `docs/limitations.md` as a result
about the method, not as an apology for a missing column.

The 39 completed LC cells are retained rather than discarded. They are not a usable
sample for the C-vs-LC comparison — they cover 13 of 60 scenarios and were never
randomised for partial analysis — and any use of them must say so.

---

## D12 — `generation.gate_blocked`: a refused response is evidence, not a gap

**Contract amendment (2026-08-25): `generation` gains
`gate_blocked BOOLEAN NOT NULL DEFAULT FALSE`, with an idempotent migration and a
partial index. Requested by `carelite-orchestrator`, approved and applied through the
orchestrating session, which is the route the build plan specifies for a wave-0
contract change.**

The holdout run produced 939 generations, **17 of which the `carelite.safety` output
gate refused.** I had assumed those cells were absent from the run; the orchestrator
lane checked and they are present, flagged only in `extra.output_gate_blocked`.

**The hazard runs the opposite way from a missing cell, and that is why this needed a
column rather than a note.** A gap in the data is visible — a condition with 178 rows
where the others have 180 invites a question. Text the safety gate refused, scored as
though it had passed, is invisible and flatters whichever condition produced it. The
analysis must be able to exclude it with a plain `WHERE`, and that cannot depend on a
sidecar file in gitignored `runs/` that nothing in the repo currently reads.

**The rows are kept rather than deleted.** A refusal is evidence about the system —
it is the safety layer working on real generated text at scale, which is the first
time that has been observed in this project — and a silently missing row is
indistinguishable from a cell that never ran.

**What the flags actually are**, measured after backfill rather than taken on report:

| | |
|---|---|
| by condition | A 3 · A2 7 · B 2 · C 2 · D 3 |
| by scenario | SC-029 **13** · SC-092 1 · SC-055 1 · SC-072 1 · SC-057 1 |

The lane reported 16 of 17 on SC-029; the true figure is 13, with four other scenarios
contributing one each. The correction matters for how this is described: SC-029 is
still overwhelmingly the driver and still fires across multiple conditions, so it
remains largely a *scenario* property rather than a condition difference — but it is
not the clean single-scenario story the first count suggested, and the write-up should
use the measured numbers.

**Consequence for the analysis.** With 13 of the 17 on one scenario, excluding blocked
cells removes SC-029 unevenly across conditions rather than symmetrically. Neither
including nor excluding them is obviously right, so the analysis should report the
primary comparison both ways and say which it prefers. What is not acceptable is
scoring refused text silently.
