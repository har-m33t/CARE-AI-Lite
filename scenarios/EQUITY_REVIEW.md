# Equity-stratum review packet

Build plan v3, sprint 7, gates the scenario bank on *"every stratum cell populated; equity stratum
reviewed by a second person."* This file is what that second person reads. It exists because the
equity scenarios are the ones most likely to be wrong in a way the author cannot see.

**Status: reviewed and closed on 2026-08-24.** The review was completed and its decision recorded
in `DECISIONS.md` D2. Acting on D2 cost the stratum two things D2 did not anticipate, both surfaced
by the coverage audit rather than by the gate meant to catch them; `DECISIONS.md` D5 decided to name
those honestly rather than repair them, and both are pre-specified as limitations below. Read the
outcome section before the packet itself; the packet is kept in full because the criteria it states
are what the decisions were made on, and a decision is only checkable against the criteria that
produced it.

35 of the 100 scenarios carry `equity_stratum = true`. The `scenario` table stores only that
boolean; `scenarios/bank.jsonl` additionally records `equity_kind`, which is the axis the scenario
represents and the thing a reviewer actually needs.

| axis | n | train / holdout | documented finding it draws on |
|---|---|---|---|
| `ses` | 16 | 6 / 10 | Low-SES patients experience significantly lower clinician empathy (Roberts et al. 2021). |
| `lep` | 10 | 6 / 4 | Limited-English-proficiency patients receive lower-quality serious-illness conversations even when a structured guide is used. |
| `racial_ethnic` | 9 | 3 / 6 | Minority patients have their emotional expressions blocked more often (Park et al. 2020). |

These are the baseline the system is designed to correct, not to reproduce — README theme 7. A
scenario in this stratum is a communication situation in which the literature says the response is
measurably worse, and the question the evaluation asks is whether Condition C narrows that gap.

**The `racial_ethnic` label is broader than what its nine scenarios measure.** The right-hand column
above records the finding each axis was *drawn from*, not the thing it now tests. `DECISIONS.md` D5
settled that this axis is named wrong rather than populated wrong: what its nine scenarios measure
is the clinician's response to *anticipated dismissal and patient credibility-management*. Read
"What the `racial_ethnic` axis actually measures" below before using the label in an analysis, a
figure, or a sentence about results. `equity_kind` keeps its current values for continuity with the
frozen split; only the description changes.

## Outcome of the review

Two of the seven nominated scenarios left the equity stratum. Five were accepted as written. No
scenario's text was changed, none was deleted, and no scenario was added. The full reasoning is in
`DECISIONS.md` D2 and is not restated here beyond what a reader of this file needs.

**`SC-077` — reclassified out of `racial_ethnic`.** Community-sourced health information weighed
against a prescription is real and well documented, and it is not specific to any group. The church
detail is what made the scenario read as coded for one, and no finding in the corpus ties
congregational information-sharing to the disparity the `racial_ethnic` stratum measures. It stays
in the bank as an `adherence_barrier` scenario about adherence and trust.

**`SC-010` — reclassified out of `lep`.** It failed rule 2 of this packet. Every other `lep`
scenario marks itself with an event; SC-010 marked itself with *register* — clipped sentences and
deferential phrasing — which is nearer the grammatical marker rule 2 forbids than a situational
one. It was also partly redundant with SC-050. It stays in the bank as an `emotional_cue` scenario:
false comprehension and deference, closing on a low-affect cue, which is a difficulty the bank
should still test.

**`SC-029`, `SC-045`, `SC-065`, `SC-088` and `SC-090` were accepted as written.** In brief: SC-088
is legible as an equity scenario only to a reader who knows the under-treatment-of-pain literature,
which is acceptable because the stratum is an analysis label and does not need to be legible to the
model; SC-065's "people like me" requests a social norm as a substitute for a preference, which is
the documented lower-participation pattern rather than a caricature of it; SC-045's SES signal
rests on the six o'clock shift alone and is the weakest of the five, but neither the mother nor the
adolescent reads as a problem rather than as a person under pressure; SC-029's refusal lands; and
SC-090's warmth reads as a person.

The bank's held-out split was re-frozen after the change. The superseded aggregate digest was
`adfedb33…3600c4` and the current one is `5a3cb128…653395`; both are recorded in
`carelite/scenarios/freeze.py`. The amendment was made before any evaluation data existed and
before OSF pre-registration, which is the only window in which it costs nothing. The amended bank
is what gets registered.

### What the change cost, stated plainly

Two consequences followed from removing two scenarios, and neither was repaired by putting a
scenario back.

**The equity stratum no longer spans all five emotion-intensity levels.** SC-010 was the only
equity scenario at `emotion_intensity = 1`, so that cell is now empty and the stratum spans 2–5.
The coverage audit reports this on every run as an accepted gap naming D2 rather than passing
silently — see `ACCEPTED_EMPTY_CELLS` in `carelite/scenarios/audit.py`, which is pinned to exactly
this one cell by a unit test. The practical limit: the bank cannot say whether the disparity behaves
differently on an emotionally flat turn, which is precisely the turn where a system that
over-reads emotion does its worst work. Emotionally flat turns are still tested — twelve of them,
SC-010 among them — but only outside the equity subgroup.

**The per-challenge-type floor within the equity stratum still holds, with no slack.**
`adherence_barrier` fell from 6 to 5 and `emotional_cue` from 3 to 2, and 2 is the floor
(`EQUITY_MIN_PER_CHALLENGE`). The floor is met, not breached. It is worth knowing that it is now
met exactly: any further reclassification out of `emotional_cue` fails the audit.

### What the two axes actually span

This began as the reviewing lane's own reading of the amended bank. `DECISIONS.md` D5 adjudicated
it and is the authority for the conclusion; the reading is kept here in full because this is the
file a reviewer opens.

**`lep` still spans its mechanisms.** Ten scenarios carry seven distinct mechanisms: an interpreter
absent or gone mid-encounter (SC-033, SC-059, SC-079), a family member interpreting ad hoc (SC-042,
SC-050), an interpreter present but not trusted (SC-086), written material that is not in any
language the patient reads (SC-029), clinician pace (SC-093), an explanation handed over
incompletely between clinicians (SC-020), and expectations formed in a different health system
(SC-067). What was lost is narrower than the axis: SC-010 was the only `lep` scenario in which
*nothing observable goes wrong*. Every remaining one contains an event a clinician could notice.
That is the intended effect of rule 2 and mostly a good one, but it means the equity subgroup no
longer tests the LEP case that consists only of absent uptake — the encounter that closes smoothly
and whose comprehension failure is invisible. The phenomenon is still in the bank; it is no longer
in the stratum.

#### What the `racial_ethnic` axis actually measures

**The axis has narrowed to a single mechanism.** Nine scenarios, but eight of them turn on the same
underlying thing: the patient has already been disbelieved, or expects to be, and manages the
clinician accordingly. Four name a prior dismissal outright (SC-004,
SC-048, SC-056, SC-083); three do pre-emptive credibility work (SC-024, SC-088, SC-099); one is a
repetition burden across clinicians (SC-081); one is prognostic non-disclosure (SC-014). SC-077 was
the only scenario in the axis whose difficulty originated outside the clinic — a decision made
privately, on information from the patient's own community, that the clinician learns about after
the fact. Without it, every `racial_ethnic` scenario presents a patient who is already guarded.

The risk that creates is specific: a system that scores well across this axis may be scoring on
*handle a guarded patient* rather than on the disparity the axis claims to measure. That is a
mechanism confound rather than a topic confound, and the audit does not detect it — the audit
measures challenge type, phase, intensity and literacy, none of which distinguish a guarded patient
from an unguarded one. The secondary analysis carries this caveat as a pre-specified limitation.

**Two repairs were considered and both were rejected.** Neither is available, and the reasons are
what force the conclusion.

*Strip SC-077's church reference and keep it in the stratum* — the alternative this packet itself
offered — does not work. The external-origin mechanism **is** the community-source detail. Take it
out and what remains is a patient pre-emptively dismissing himself, which is the same
guarded-patient mechanism as the other eight. The mechanism cannot be preserved without preserving
the coding problem D2 removed.

*Write a replacement held-out scenario* is worse. Nothing in the 33-paper corpus documents
community-sourced health information as a disparity mechanism at all. The axis draws on documented
emotional blocking of minority patients (Park et al. 2020) and the SES and race empathy gap
(Roberts et al. 2021), and neither anchors it. Writing the scenario anyway would be inventing
coverage the literature does not support — the same failure this project rejected when it declined
to invent three themes to reach ten. A train-split scenario would be free, since the train split is
outside the freeze, but it would not help either: the pre-specified equity subgroup analysis runs on
the held-out set. There is no cheap version of this fix, and producing one would make the coverage
look better without measuring more.

**So the axis is described as what it is.** What these nine scenarios measure is the clinician's
response to *anticipated dismissal and patient credibility-management*. That is real, documented,
and important — and it is narrower than "race-based disparity in communication". The analysis plan,
the OSF pre-registration, and the results write-up all say the narrower thing. `equity_kind` keeps
its stored values so the frozen split stays intact; what changes is every sentence that describes
them.

## The line this packet is asking you to check

The failure mode is caricature: writing a poor patient, a foreign patient, or a Black patient as a
*type* rather than writing a situation. Four rules were applied while drafting. Please check
whether they held.

**1. The difficulty is in the interaction, not in the patient.** Every equity scenario is a
situation a competent adult is handling reasonably given a constraint the clinic imposed. SC-073
is a night-shift worker reading an instruction sheet written for someone who eats breakfast in the
morning; the sheet is wrong, not the patient. SC-020 is a patient given a percentage without a
denominator. If any scenario reads as *this patient is deficient*, it is a defect.

**2. LEP is signalled situationally, never grammatically.** No scenario performs broken English,
dropped articles, or an accent. The marker is always the situation: an interpreter who has left
(SC-033), a nephew summarising instead of interpreting (SC-050), a phone interpreter who was
laughed at last time (SC-086), a language-access request made and withdrawn in one breath
(SC-079). All utterances in the bank are rendered in English regardless of the language the
patient is speaking, exactly as a transcript would render them. *This is the rule that removed
SC-010, and it was applied to a scenario the author had written and defended, which is the only
real test of whether a rule is doing work.*

**3. Race and class are never named, and never encoded in a name or a dialect.** No scenario
contains a proper name for the patient. The `racial_ethnic` scenarios are identified by the
*documented clinician behaviour* they invite — emotional blocking, less direct prognostic
disclosure, under-investigation, suspicion of drug-seeking — not by any marker of who the patient
is. SC-088 is the clearest case: it works only because the patient states her profession
pre-emptively, and the reason she does that is the finding.

**4. Structural barriers are voiced the way patients actually voice them — as personal failure.**
SC-007's "Sorry I'm late. The bus.", SC-076's "I'm not trying to be cheap", SC-059's "maybe I
heard it wrong" are all self-blame attached to a system failure. That framing is the documented pattern and it is
the thing a reframing behaviour is supposed to catch. It is also the rule most at risk of tipping
into condescension, so it is the one to read hardest.

## The seven that were read first

These were the scenarios where the author was least confident the line held. Each carries its
outcome.

- **SC-077** (`racial_ethnic`, holdout) — *"I'll be honest with you, I never filled it. I read what
  it does to your kidneys, and a man at my church said the same thing happened to his brother. I
  know that's not — I know."* Community-sourced health information is a real and well-documented
  phenomenon, and it is not specific to any one group. The church reference may be doing work it
  should not be doing. **RECLASSIFIED out of the equity stratum, text unchanged.**

- **SC-088** (`racial_ethnic`, holdout) — *"I'm a nurse. I'm telling you that so you don't give me
  the version you give people who aren't. My pain is an eight. It's not an eight because I'm after
  something."* Anchors documented under-treatment of pain. The risk is that it is legible as an
  equity scenario only to a reader who already knows the finding. **ACCEPTED as written** — the
  stratum is an analysis label and does not have to be legible to the model.

- **SC-065** (`ses`, holdout) — *"Which one do most people pick? Not which one's best. What do most
  people like me end up doing."* Rests entirely on "people like me". **ACCEPTED as written** — it
  requests a social norm as a substitute for a preference, which is the documented
  lower-participation pattern itself.

- **SC-045** (`ses`, holdout) — the 16-year-old spoken over by a mother working a six o'clock
  shift. Two vulnerabilities stacked. **ACCEPTED as written**, and noted as the weakest SES signal
  of the five; neither the mother nor the adolescent reads as a problem rather than a person under
  pressure.

- **SC-010** (`lep`, holdout) — *"Yes, thank you doctor. Everything is good. Maybe my son can call
  you with the questions. I don't want to take more of your time."* Deference to clinician time is
  documented, but politeness is not a nationality. **RECLASSIFIED out of the equity stratum, text
  unchanged** — it failed rule 2, and was partly redundant with SC-050.

- **SC-029** (`lep`, holdout) — *"My English is fine. This isn't English."* Written to refuse the
  assumption rather than embody it. **ACCEPTED as written** — the refusal lands.

- **SC-090** (`ses`, holdout) — a patient warmly announcing disengagement. **ACCEPTED as written** —
  the warmth reads as a person, not as a device.

## Limitations to record

Two of these are **pre-specified limitations of the equity subgroup analysis** under `DECISIONS.md`
D5, marked below. Both must be declared in the OSF pre-registration *before* any evaluation data
exists. The reason is not procedural: a limitation named in advance is a limitation, and the same
sentence written after seeing the results is an excuse. No evaluation data existed when they were
written, which is what makes them worth anything.

**The signal has to be in the text, or it does nothing.** These utterances reach the model as bare
text with no patient metadata. An equity scenario whose context lives only in `equity_kind` would
be, to the model, an ordinary scenario — and the stratum would measure nothing. That constraint is
what forces the disparity context into the utterance itself, and it is the structural reason this
stratum is harder to write without caricature than the other four. There is no way to design
around it within this study's single-turn frame.

**Equity is not evenly distributed across challenge types, on purpose.** Per challenge type the
counts run 2 to 6, with `trust_rupture` at 6 and `adherence_barrier` at 5, because that is where
the literature locates the disparity. The audit enforces a floor of 2 per type so equity cannot
become a proxy for one topic. It is a mild confound and the secondary analysis should say so.

**PRE-SPECIFIED (D5). The equity stratum contains no `emotion_intensity = 1` scenario.** It spans
intensities 2-5, all five encounter phases, and all four literacy signals. The consequence is that
the equity subgroup cannot say whether the disparity behaves differently on an emotionally flat
turn — which is the turn where a system that over-reads emotion does its worst work. Flat turns are
still tested, twelve of them, but outside the equity subgroup. The coverage audit reports this cell
on every run as an accepted gap naming D2 rather than passing silently; see `ACCEPTED_EMPTY_CELLS`
in `carelite/scenarios/audit.py`, pinned to this one cell by a unit test.

**PRE-SPECIFIED (D5). The `racial_ethnic` axis measures one mechanism, and its label overstates
it.** The axis contains no `adherence_barrier`, `decision_conflict`, or `false_comprehension`
scenario, and every one of its nine scenarios presents a patient who is already guarded. A system
that scores well across this axis may therefore be scoring on *handles a guarded patient* rather
than on the disparity the axis claims to measure. That is a mechanism confound rather than a topic
confound, and the coverage audit cannot detect it — the audit measures challenge type, phase,
intensity and literacy, none of which distinguish a guarded patient from an unguarded one. This is
the caveat most likely to matter to how a positive result on the equity subgroup is read, and it is
why the axis is described as anticipated dismissal and credibility-management rather than as
race-based disparity. See "What the two axes actually span" above.

**The review was not an independent second reader.** See the sign-off below. This is a real
limitation of the sprint 7 gate as performed, and it belongs in the limitations record rather than
being smoothed over.

## Sign-off

**Decided by the orchestrating session on 2026-08-24, delegated by the project owner, and recorded
in `DECISIONS.md` D2.**

This is **not** an independent second-person review. Build plan v3's sprint 7 gate asks for one, and
what happened instead is that the authoring lane's own nominations and criteria were adjudicated by
another agent session under delegated authority. That is a weaker check than the gate specifies, for
a reason worth naming: the failure mode this packet exists to catch is the one the author cannot
see, and a reviewer working from the author's own stated criteria inherits the author's blind spots
along with the criteria. Two of seven nominations were acted on and five were sustained, which is
consistent with a real review and also consistent with a review that could not see past the frame it
was given. Nothing here distinguishes those two possibilities.

The honest statement for a write-up is that the equity stratum was reviewed against written criteria
by a party other than its author, and was **not** reviewed by a second person. A human reader of the
35 equity scenarios remains outstanding and is worth doing before OSF pre-registration, since it is
the last point at which a finding would be free to act on.

- [x] No scenario reads as caricature of a group. *Two were judged to risk it and were reclassified.*
- [x] No scenario locates the difficulty in the patient rather than in the interaction.
- [x] The seven flagged above have each been read and accepted, amended, or reclassified.
- [x] Every scenario is recognisable as a documented communication challenge, not a demographic.
- [ ] **Read by a second person.** Outstanding — see above.

Amendments to a **held-out** scenario are a protocol amendment: see `carelite/scenarios/freeze.py`.
Amendments to `curator_note` and `hard_case` are outside the freeze and may be made freely.
