# Theme Taxonomy — Proposal for Sign-Off

**Status:** proposed, awaiting sign-off. Nothing downstream should be treated as final until
this is signed off.
**Author:** `carelite-kb` lane.
**Decision requested:** adopt **seven themes**, unchanged from `README.md` and from the frozen
`carelite.types.Theme`. Do not expand to ten.

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

## Also blocked on this sign-off: paper evidence tiers

`carelite.corpus.fetch.manifest_papers()` deliberately stamps every paper `emerging` with a
placeholder citation, documenting that a real tier is "a KB/human review call, not something the
fetch pipeline should assert". It is correct to have deferred it. This lane will make that call in
`carelite/kb/papers.py`, mapping study design to tier: systematic review / meta-analysis / RCT →
`strong`; controlled or longitudinal observational → `moderate`; protocol, survey, qualitative,
single-arm or pre-post → `emerging`. Study protocols with no results yet cap at `emerging`
regardless of the design they propose. `validate.py` then checks each entry's claimed tier against
its source paper's design, so an entry cannot claim `strong` off a protocol.
