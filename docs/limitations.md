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
human sign-off gate rather than claim it falsely: `human_verified` is `FALSE` on all 115 loaded
entries (as of this writing — this figure moves as `carelite-kb` corrects the pipeline; see
`knowledge_base/review/kb_review_digest.md` for the live count), and that is the honest record,
not a pending checkbox. Any result that depends on knowledge-base quality inherits this
limitation.

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
  evidence tier is corrected rather than rejected, and as of this writing is derived from the source
  paper's study design outright rather than merely capped at it**: 58 of the current 115 entries
  carry a tier that differs from what the extraction model originally claimed
  (`carelite/kb/papers.py` maps design to tier; `validate.py` enforces it). This is a stronger rule
  than the project's first version, which only lowered an overclaim and left an underclaim alone —
  under that version, two entries citing the same paper could carry different tiers, which
  `README.md`'s own definition of evidence strength as a property of the source does not allow. A
  second, related check now also caps tier for **second-hand findings** — an entry whose quoted span
  itself cites another study by number (e.g. a bare `(18)` marker) is capped at what *this* corpus's
  paper can vouch for, not the tier of the study it is relaying. In every case the span, theme,
  finding, and takeaway are untouched by a tier correction, and there is a derivable right answer to
  substitute — unlike a fabricated quote, which has none.
- **No human read any entry for whether the finding follows from the span.** That is the
  specific judgment an automated check cannot make and it is not claimed here. The review
  machinery in `carelite/kb/review.py` exists and is exercised (`knowledge_base/review/kb_review_digest.md`
  is generated from the live database), but it is an available tool, not a completed gate: 0 of 115
  entries are signed off as of this writing.

**Concentration is worse than the theme totals suggest.** The equity theme holds 4 entries as of
this writing (up from 3 at an earlier pass, moved by the scope and second-hand-evidence checks
above, not by new extraction), drawn from a corpus where the literature *describes* a disparity
rather than a compensating move — a faithful extraction of "clinicians should be aware of empathy
gaps in low-SES patients" is an awareness statement, and the actionability gate correctly rejects
it. `DECISIONS.md` D3 approved a re-extraction with a revised prompt that instructs the extractor to
name the compensating move rather than the disparity, sequenced to run once `carelite-corpus`'s
extraction fixes land. **That re-extraction has not run as of this writing** — `PROMPT_VERSION` in
`carelite/kb/extract.py` is still `kb-extract-v1`, and the equity count above is still the pre-D3
figure. This document will be updated with the post-re-extraction count and, per D3's own stated
risk, with whether the individually-read equity entries hold up under the stricter guard against a
takeaway that drifts beyond its span. Separately, teach-back's 15 entries are concentrated: 12 of
them come from a single systematic review (Talevski 2020), so what reads as convergent evidence
within the theme is largely one source cited many times.

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
