# Limitations

Build plan v3 §17 named a limitations list before any evaluation data existed. This document is
that list, kept current as the build proceeds. Everything here is written in advance of results,
which is the point: a limitation named before the numbers exist is a limitation, and the same
sentence written after is an excuse. Where a lane is still in flight, that is stated plainly rather
than described as finished.

Status of the underlying work as of 2026-08-24: the corpus and knowledge base are loaded and
queried directly for the figures below; `carelite-corpus` is fixing extraction artefacts,
`carelite-kb` is correcting evidence-tier derivation and removing out-of-scope entries, and
`carelite-retrieval` and `carelite-orchestrator` are mid-build. No evaluation data exists yet — see
`docs/preregistration.md` for why that ordering matters and what has and has not happened.

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

## 4. Judge-primary evaluation

Human ratings are the credibility ceiling on this project (build plan v3 §12) and they are
deferred: the human-rating harness (`carelite/eval/human/`) is built and exercised end-to-end
against synthetic rater data (`carelite/eval/human/synthetic.py`) rather than real raters, so that
a blinding bug or a reversed `ritualistic` column is caught before, not after, a paid rater spends
a weekend on 60 responses. As of this writing no real human rating has occurred. Every number this
project reports is judge-only until that changes, and carries that caveat in the sentence that
reports it — not only in this document.

The judge itself (`gpt-oss:20b`, a different model family from the `gemma4:12b` generator, per v3
§13's independence requirement) is validated as its own component study rather than assumed
trustworthy: self-consistency, positional-bias, span-grounding, and per-dimension
Krippendorff's-α/Spearman's-ρ checks are implemented in `carelite/eval/judge/validation.py` against
a pre-specified threshold (α ≥ 0.667, ρ ≥ 0.5, ≥ 30 paired units — see
`docs/preregistration.md`). Below that threshold on a given dimension, results on that dimension are
reported as exploratory. The validation study itself needs the human-rating data it is meant to
validate against, so it cannot run until human rating happens, which has not yet occurred.

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
  and is redefined rather than silently approximated — `DECISIONS.md` D7.** 471 chunks is
  approximately 326,526 tokens against a 128,000-token context window: 255% utilisation. Reserving
  16K for the system prompt, patient turn, and response leaves room for roughly a third of the
  corpus. **LC is now `LC-sample`**: a fixed, query-independent round-robin selection across all 33
  papers at a pinned seed (`carelite.retrieval.ablation.lc_sample`), 169 chunks, 35.9% of the
  corpus. Round-robin rather than random sampling guarantees every paper is represented, so LC's
  content is not an accident of the seed. **The point that must not be quietly absorbed: any
  selection rule is itself a form of retrieval.** LC was meant to ask whether curated retrieval
  beats stuffing everything in; it can now only ask whether *query-dependent* selection (Condition
  C) beats a *fixed* context (LC-sample) — a real and interesting question, arguably closer to what
  a practitioner would actually build, but a different one from what build plan v3 posed. This
  distinction is carried in `docs/preregistration.md` §2 and §4 rather than left to a results-section
  footnote.
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
- **Local-model capability ceiling.** Every generator, judge, embedder, and reranker in this
  project runs locally via Ollama or `sentence-transformers`. Results describe what this system
  does with these models, not the ceiling of the largest hosted frontier models.
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

*Last updated 2026-08-24, alongside `docs/preregistration.md` and `docs/decisions/`. Figures
sourced by direct query against the live `carelite` database and against
`carelite/scenarios/freeze.py`, `knowledge_base/TAXONOMY.md`, `carelite/kb/validate.py`, and
`DECISIONS.md` — not carried over from planning-time estimates. Where a figure is expected to
change (the equity re-extraction under D3, human rating, judge validation), that is stated above
rather than left for the reader to infer.*
