# Theme Taxonomy

**Status: signed off.** `DECISIONS.md` D1 accepts this proposal in full, including the
recommendation not to add serious-illness conversation as an eighth theme. The seven themes
encoded in `carelite.types.Theme` are canonical; the "10 themes" figure in build plan v3 is
retired. `types.py` is unchanged by the decision, which is the point — it confirms a frozen
contract rather than amending one.
**Author:** `carelite-kb` lane.

The sections below are kept as the argument that produced the decision. Where later work
changed a number they cite, the section says so rather than being quietly rewritten — the
counts moved because the validator got stricter, and that is itself part of the record.

---

## The question

`README.md` defines seven communication themes and gives each a paragraph of derivation.
`.claude/CARELite-AI_build_plan.md` refers once, in passing, to "10 themes". `carelite/types.py`
encodes the seven. The two numbers have never been reconciled, and `types.py` carries a note
saying this lane would propose a canonical set.

This is a genuine open question in a way the entry counts are not. The 15-vs-45 entry numbers
describe different things (a sample set and the full base); 7-vs-10 describes the same thing twice.

## The finding that settles most of it

The ten-theme number has no referent. "10 themes" appears exactly once in the build plan, at
line 24, inside a sentence arguing that the knowledge base is already a curated knowledge graph:

> Your 45-entry knowledge base, 10 themes, and behavior-to-framework mappings *are* a curated
> knowledge graph.

There is no list of ten themes anywhere — not in the build plan, not in the README, not in any
file in the repository. There is no definition, no derivation, and no mapping from the seven to
the ten. By contrast the seven are each defined in prose, each tied to a described mechanism, and
each already encoded in a frozen contract that three other lanes import.

So the choice is not between two taxonomies. It is between one taxonomy and a number. Adopting
ten would mean inventing three themes to satisfy a count, which is the same failure mode the
project owner has already corrected on the entry counts, run in the opposite direction.

## What the 33 papers actually support

Counting papers whose full text carries substantive coverage of a theme (at least eight matches
against that theme's vocabulary — a threshold that separates a paper *about* a topic from a paper
that mentions it once in its discussion):

| Theme | Papers with substantive coverage | Anchor papers |
|---|---|---|
| Empathy | 13 | Roberts 2021 meta-analysis; Alhassan 2019 RCT; Moudatsou 2020 review |
| Emotion recognition and response | 13 | Seifart 2023 breaking-bad-news survey; Yuen 2023 |
| Patient activation and shared decision-making | 7 | Wilson BOAT asthma RCT; van Boven 2023 COPD/asthma SDM; Xu 2024 MI meta-analysis |
| Comprehension confirmation (teach-back) | **1** | Talevski 2020 systematic review |
| Plain language and information clarity | **2** | Bonanno 2024 readability study; Talevski 2020 |
| Trust and relational continuity | 8 | Epstein 2005; Altunisik 2025; Kim 2024 |
| Equity-aware communication | 4 | Roberts 2021 (SES/race empathy gap); Ho 2025 (LEP serious illness) |

Two observations matter more than the totals.

**Teach-back and plain language are single-source themes.** Teach-back has exactly one paper
giving it substantive treatment, and plain language has two. This is not an artefact of the
corpus being small — it is an artefact of *which* papers survived the fetch. Ten of the manifest
DOIs are paywalled and five have no recoverable DOI, and the emotional-blocking and
health-literacy literature is concentrated in exactly that lost set (the `nihms*` rows in
`data/pdfs/_manual_needed.csv` are Patient Education and Counseling and JAMA-family papers).
Talevski 2020 is a systematic review and will legitimately yield several distinct entries with
distinct verbatim spans, but they will share one source. That is a limitation to record, not to
paper over. Splitting themes finer makes it worse.

**The corpus is skewed toward training studies, not bedside behavior.** Eighteen of the 33 papers
are communication-skills-training interventions, curricula, or burnout studies. Those papers are
about how communication skill is *acquired*, not about what to say in a given moment. They
support KB entries only where they report a finding about the communication behavior itself
rather than about the training that taught it. This is the single biggest constraint on how many
defensible entries the corpus yields, and it is discussed further under "Consequences for the
entry count".

## Why seven and not ten, stated as a retrieval argument

Theme is not decoration. In the v3 design it is a metadata facet on `kb_entry` with its own
index (`kb_entry_theme_idx`), used for pre-filtering and rerank weighting, and
`settings.retrieval.rerank_top_n` is 4.

At 45 entries over 7 themes the average theme holds 6-7 entries — already thin, but still larger
than the rerank window, so a theme-filtered retrieval has something to choose between. At 45 over
10 the average is 4-5, which is at or below `rerank_top_n`. A filter that returns fewer candidates
than the reranker consumes is not a filter; it is a rename. The finer split would cost real
retrieval quality to buy a taxonomy nobody has written down.

## The one place the seven genuinely strain

The largest coherent cluster in this corpus has no theme of its own: serious illness
conversations, goals-of-care discussions, and breaking bad news. Nine papers give it substantive
coverage — the SICP competency review, Seifart 2023, Bonanno 2024, Ho 2025, Yuen 2023, the
VR-TALKS protocol, and the mnemonic critique in Childers 2020. Under the seven it splits across
emotion recognition, activation/SDM, and equity.

**Recommendation: do not make it an eighth theme.** The communication acts inside a serious
illness conversation are the seven themes performed under pressure — naming an emotion, checking
comprehension, eliciting a goal before recommending. What distinguishes it is *where in the
encounter and at what stakes*, which `KBEntry` already models: `encounter_phase` is a list field
on the frozen type, and `EncounterPhase.EXPLANATION` / `PLANNING` carry exactly this. Encoding
stakes as a phase rather than a theme keeps the theme axis about communicative function, which is
what retrieval needs it to be, and avoids a theme that would otherwise absorb entries from three
others and skew the facet.

If the sign-off disagrees and wants serious illness as a theme, that requires a `types.py`
amendment from the foundation lane. This lane cannot and will not make it.

## The seven, with the boundaries this lane will extract against

The README defines these; what follows adds the inclusion/exclusion line each needs to be
machine-checkable, which is what `carelite/kb/validate.py` enforces.

1. **`empathy`** — empathy as a trainable behavior: acknowledgment, perspective-taking, the
   cognitive rather than affective account. *Excludes* studies measuring only clinician empathy
   scores after a course with no link to a patient-facing act; those are training-transfer
   findings, not behaviors.
2. **`emotion_response`** — recognising an emotional cue and what is done next. Blocking,
   premature reassurance, pivoting to information. *Excludes* clinician emotion regulation with no
   patient-facing consequence.
3. **`activation_sdm`** — eliciting goals, negotiating the plan, decision-making that changes what
   the patient does. Motivational interviewing lives here. *Excludes* information-giving that is
   merely thorough.
4. **`teach_back`** — asking the patient to state back understanding, and re-explaining when the
   response is incomplete. Narrow and behavioral by design.
5. **`plain_language`** — jargon, message count, readability, analogy. The target is understood
   consent, not disclosed information.
6. **`trust_continuity`** — trust as mediator, consistency across visits, transparency about
   uncertainty, record-sharing. *Excludes* generic satisfaction scores with no communicative
   mechanism attached.
7. **`equity`** — differential delivery of any of the above by SES, race, ethnicity, or language.
   Also carried as the `equity_relevant` boolean on entries whose primary theme is another one, so
   an equity finding about empathy is reachable from both directions.

## Consequences for the entry count

45 remains the target and the floor, and this lane will extract toward it. Two honest caveats,
recorded now rather than discovered at the review gate:

- Teach-back and plain language will not reach an even share of 45 from independent sources.
  Expect them to be source-concentrated and to say so in the review digest.
- The training-study skew means some papers will yield zero bedside-actionable entries. A paper
  that reports a curriculum improved empathy scores supports a claim about training, not a
  practical takeaway a clinician can act on mid-encounter, and `validate.py` rejects entries whose
  takeaway is not actionable.

If the corpus falls short of 45 defensible entries, this lane will report the shortfall with the
per-theme breakdown rather than pad it. An unsupported entry propagates into retrieval, into
generation, and into the results.

### Outcome: 114 entries, and why `equity` is still three

*(This section was written at 127 entries. The count is now **114**: a content review found four
defects, and the fixes for two of them reject entries — 13 whose subject matter `TAXONOMY.md` §1
and §2 exclude or whose finding their span does not report, and 4 whose takeaway names an outcome
to bring about rather than an act to perform. The per-theme shape is unchanged in kind:
`activation_sdm` 40, `plain_language` 20, `teach_back` 15, `empathy` 14, `trust_continuity` 13,
`emotion_response` 9, `equity` 3. The argument below still holds; the numbers in it are the ones
that were current when it was made.)*

The corpus supports **127 validated entries** — well past the 45 floor — but they are not evenly
distributed, and one gap is structural rather than a sampling accident.

A first extraction pass ranked each paper's windows by general communication vocabulary, which
under-read the two thinnest themes: the equity anchor (Roberts et al. 2021, a meta-analysis of
socioeconomic and racial differences in clinician empathy) yielded one entry because its three
densest windows by that vocabulary were its search strategy and methods. A targeted second pass
re-ranked window selection by theme-specific vocabulary over the trust and equity papers, leaving
`SYSTEM_TEMPLATE` untouched — a prompt told to find equity findings will find them whether or not
the passage contains any. That correction worked for trust: `trust_continuity` went from 6 entries
across 5 papers to 13 across 6.

**It did not work for equity, and the reason is worth stating rather than fixing away.** The pass
read the right pages; the model proposed 12 equity-themed candidates across the whole corpus; 3
survived. Of the 9 rejected, 3 quoted a sentence that was not there and **6 were rejected as an
attitude rather than an action** — and all six are the same sentence shape:

> Clinicians should be mindful of potential empathy gaps when treating patients from lower
> socioeconomic backgrounds.

That is what happens when a faithful extractor meets this literature. The equity evidence base
*describes a disparity* — low-SES patients receive less empathy, minority patients' emotional cues
are blocked more often, LEP conversations are shorter and less checked — and the takeaway a model
naturally writes from a descriptive finding is an awareness statement. The actionability gate
rejects those correctly: CARELite recognises a kind of moment and supports a response, and "be
mindful of the empathy gap" is not a move the system can detect, generate, or reframe. The one
Roberts entry that did survive is the actionable form of the same finding — *check your assumptions
about this patient's adherence and pain needs* — which is a move.

So `equity` at 3 entries is an honest floor, not a measurement error, and it should be reported as a
limitation. The fix is a prompt change, not a threshold change: instruct the model that when a
passage reports a disparity, the takeaway must name the compensating move rather than the awareness.
That bumps `PROMPT_VERSION`, invalidates all 141 cached windows, and costs a full re-extraction, so
it is recommended here rather than done unilaterally.

### The re-extraction was approved, and run. It did not work.

`DECISIONS.md` D3 approved that prompt change with a guard: the takeaway must be supported by the
quoted span rather than merely adjacent to it, and every equity entry from the re-run gets read
individually rather than sampled. `EQUITY_PROMPT` in `carelite/kb/extract.py` carries the revision,
on its own cache version rather than a global `PROMPT_VERSION` bump — a global bump re-runs all 33
papers to change how one theme is extracted, and the cache key already separates the two.

It was run against the two papers that carry substantive equity content: **Roberts et al. 2021**
(equity vocabulary 184, the meta-analysis of SES and racial differences in clinician empathy) and
**Holdsworth et al. 2025** (90, serious illness communication with LEP patients). Six candidates,
each read against its span:

- **Four were aspirations** — the original awareness statement with an active verb bolted on the
  front. *"Be mindful of the empathy gap"* became *"proactively work to bridge the empathy gap"*,
  *"identify and address potential barriers to ensure equitable empathy"*, *"provide consistent
  high-quality empathetic communication to all patients regardless of background"*, and *"engage in
  more consistent and attentive communication so these patients feel heard"*. None is a move; all
  four cleared the verb whitelist, because `address`, `provide` and `engage` are real verbs.
  `_ASPIRATION_NOT_ACTION` in `validate.py` now rejects them.
- **One duplicated an existing entry's span exactly** and was dropped as a duplicate. Its takeaway
  was better than the loaded one's — *"explicitly ask the interpreter to verify the patient's
  understanding"* against *"proactively use interpreters ... and capture cultural nuances"* — but
  the deterministic `entry_id` is a hash of paper and span, so the two collide and the earlier one
  wins. That is a real, if minor, ordering defect worth recording: a better rendering of the same
  evidence cannot currently displace a worse one.
- **One was a genuine compensating move**, and it is a `trust_continuity` entry rather than an
  equity one: *"provide the interpreter with advanced preparation and specific context before the
  encounter"*, off a span that says being an effective voice for the clinician "required advanced
  preparation of the interpreter". The knowledge base already holds *"Brief the interpreter on the
  goals and specific content of the conversation before the patient enters the room"* from the same
  paper, so it adds no independent evidence.

**Net new equity entries: zero.** The variant's output was therefore not loaded — `load.py
--prompt-version` exists so that an experimental variant reaches the knowledge base when its guard
has been applied rather than when its inference finishes.

The structural reason is visible in which paper failed. Holdsworth is a qualitative and
mixed-methods study of LEP encounters, so its text *describes interactions* and a compensating move
is there to be quoted. Roberts is a meta-analysis: it quantifies a gap and, correctly for its
design, never says what closes it. Asked for a compensating move from a paper that contains none,
the model supplies one. That is exactly the risk D3 named, and it is why the guard was not optional.

The honest conclusion is that **`equity` at 3 entries is a property of this corpus, not of the
prompt.** Closing the gap needs equity papers that report an intervention, and the fetch already
records that the health-literacy and emotional-blocking literature is concentrated in the paywalled
and DOI-less rows of `data/pdfs/_manual_needed.csv`. The re-run is resumable — the six thinner
papers in the sweep (equity vocabulary 27 down to 12) had not completed when this was written, and
`python -m carelite.kb.extract --prompt equity --focus equity` picks up from the cache — but nothing
in those densities suggests they hold what Roberts does not.

`equity` also under-counts its own reach: 8 of the 114 entries are flagged `equity_relevant` while
sitting under another theme, which is the design working as intended — an interpreter finding is a
`plain_language` entry that is also an equity one.

## Paper evidence tiers — settled, and now in the database

`carelite.corpus.fetch.manifest_papers()` deliberately stamps every paper `emerging` with a
placeholder citation, documenting that a real tier is "a KB/human review call, not something the
fetch pipeline should assert". It was correct to defer it. This lane makes that call in
`carelite/kb/papers.py`, mapping study design to tier: systematic review / meta-analysis / RCT →
`strong`; controlled or longitudinal observational → `moderate`; protocol, survey, qualitative,
single-arm or pre-post → `emerging`. Study protocols with no results yet cap at `emerging`
regardless of the design they propose.

**Two corrections have been made to how this reaches an entry, and both came out of the content
review.**

The first is that the tier is now **derived from the design, not capped at it**. The original check
lowered an overclaim and left an underclaim alone, which is half a check — wherever the model
happened to call a randomised trial `emerging`, `emerging` survived. Four papers ended up carrying
entries at more than one tier and Talevski 2020 carried entries at all three across its fourteen.
`README.md` defines evidence strength as a property of the source, so that was incoherent: two
entries citing one paper cannot honestly carry different strengths. The derivation now runs in both
directions. Correcting rather than discarding is still right for the reason it always was — a
fabricated quote has no right answer to substitute, an ill-judged tier does — and both values
survive on the entry, with the digest printing the model's claim beside the stored one.

The second is that `papers.py` now **writes** `design`, `evidence_tier`, `apa_citation` and `year`
onto the `paper` rows. Deferring that write left the design known in Python and unknown in
Postgres: all 33 rows sat at `design IS NULL`, the fetch placeholder tier, and "[citation pending]",
so every consumer that reads the table rather than importing the module — the graph lane, the stats
lane, the CLI's evidence panel — saw placeholders while the review digest printed the real design.
Citations are derived from Crossref and frozen into `PAPER_META`, so a cold rebuild needs no
network; `python -m carelite.kb.papers --refresh-citations` re-derives them.

**One principled exception to "one paper, one tier".** A span that *relays another study* — a
systematic review reporting somebody else's trial, an introduction citing prior work — carries
evidence that belongs to a paper outside this corpus, and the citing paper's design cannot vouch
for it. Those entries are detected by `carelite/kb/scope.py`, labelled `second-hand` in the digest,
and capped at `moderate` when a synthesis is doing the relaying and `emerging` otherwise. Talevski
now reads as 8 entries at `strong` from its own synthesis and 5 at `moderate` relayed from included
studies, which is a distinction a reader can check rather than a spread nobody could explain.

## Extraction reliability: what the provenance check actually caught

Across 180 candidates from 141 windows over 33 papers, 19 carried a `verbatim_span` the validator
could not locate. Auditing every one of them against the source text showed they were four different
failures wearing one label, and reporting them as a single "fabrication rate" would have been wrong
in both directions.

*(The count is 18 after the `carelite-corpus` lane repaired the running-footer injection described
in the third bullet below. That paper's sentence is now extractable as a reader sees it, and the
entry quoting it validates. Nothing else in this audit moved: the 8 genuine misquotations are
unchanged, and 4.4% remains the honest fabrication rate over 180 candidates.)*

- **8 (4.4% of candidates) are genuine misquotation.** Substituted content words (`negative` →
  `even`, `languages` → `tensions`, `clinical decisions` → `plans`), dropped content words
  (`empathic`, `when`), a statistic with `, respectively` appended that the sentence does not
  contain, and two cases of non-adjacent sentences welded together with invented connective text.
  These are the number that belongs in the write-up as the model's fabrication rate.
- **10 (5.6%) misquote a real sentence by characters that really are in the extracted text.** Mostly
  inlined superscript citation markers — the extractor renders `silence.18,20,23,36,53` where the
  page shows a full stop and five small numerals — and punctuation altered inside a statistics
  string (`B = 0.374, β` for `B = 0.374; β`), usually because the model had welded the next result
  onto the one it was quoting. These stay rejected. Folding digits and punctuation would recover
  them and would also let the validator confuse two different readings of a result, which is where
  a provenance check stops being one.
- **1 is a defect in the corpus extractor, not this lane.** In `10-1371-journal-pone-0247259` the
  PDF's running footer is extracted into the middle of a sentence: *"lower ratings of clinician
  empathy (mean `plos one empathy disparities plos one | https://doi.org/... 5 / 16` care difference
  = -0.87 ...)"*. The model quoted the sentence as a reader sees it. No needle-side normalisation
  can reach this, because the injected text is on the haystack side. **Escalated to the corpus
  lane** — it degrades chunking and retrieval too, not only span validation.
- **12 more were the validator's own fault and are now fixed.** The extractor splits words across
  column breaks (`show ing`, `collabora tive`, `sta tistically`) and joins them across line breaks
  (`healthrelated`, `decisionmaking`, `inthe`); the model quoted the word as the printed page shows
  it. `spans.py` now folds that in a second pass, guarded to start on a word boundary, and the
  stored span is still recovered from the original text — artefact and all, which the review digest
  flags so a human sees exactly how much cleanup was applied.

The headline figure from the first run — 26 of 130 candidates, read as a 20% fabrication rate — was
therefore less than a third fabrication. The honest number is **8 in 180, 4.4%**.

A second defect surfaced in the same audit. The actionability filter's verb whitelist was written as
lemmas followed by `\w*`, so `use`, `provide`, `explore`, `acknowledge` and `validate` matched
"used" and "provides" but never "using", "providing", "exploring" — the final `e` is dropped before
`-ing`, which is exactly how an imperative takeaway is usually phrased. That silently rejected a
large share of perfectly actionable entries. Fixed, with the attitude rejection ("be mindful of",
"focus on building") made explicit so widening the whitelist could not let attitudes back in.
