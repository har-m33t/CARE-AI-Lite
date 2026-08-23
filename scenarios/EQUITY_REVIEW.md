# Equity-stratum review packet

Build plan v3, sprint 7, gates the scenario bank on *"every stratum cell populated; equity stratum
reviewed by a second person."* This file is what that second person reads. It exists because the
equity scenarios are the ones most likely to be wrong in a way the author cannot see.

37 of the 100 scenarios carry `equity_stratum = true`. The `scenario` table stores only that
boolean; `scenarios/bank.jsonl` additionally records `equity_kind`, which is the axis the scenario
represents and the thing a reviewer actually needs.

| axis | n | train / holdout | documented finding it draws on |
|---|---|---|---|
| `ses` | 16 | 6 / 10 | Low-SES patients experience significantly lower clinician empathy (Roberts et al. 2021). |
| `lep` | 11 | 6 / 5 | Limited-English-proficiency patients receive lower-quality serious-illness conversations even when a structured guide is used. |
| `racial_ethnic` | 10 | 3 / 7 | Minority patients have their emotional expressions blocked more often (Park et al. 2020). |

These are the baseline the system is designed to correct, not to reproduce — README theme 7. A
scenario in this stratum is a communication situation in which the literature says the response is
measurably worse, and the question the evaluation asks is whether Condition C narrows that gap.

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
patient is speaking, exactly as a transcript would render them.

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

## The seven to read first

These are the scenarios where the author is least confident the line held.

- **SC-077** (`racial_ethnic`, holdout) — *"I'll be honest with you, I never filled it. I read what
  it does to your kidneys, and a man at my church said the same thing happened to his brother. I
  know that's not — I know."* Community-sourced health information is a real and well-documented
  phenomenon, and it is not specific to any one group. The church reference may be doing work it
  should not be doing. **Consider whether this should be reclassified as non-equity, or the
  reference removed.**

- **SC-088** (`racial_ethnic`, holdout) — *"I'm a nurse. I'm telling you that so you don't give me
  the version you give people who aren't. My pain is an eight. It's not an eight because I'm after
  something."* Anchors documented under-treatment of pain. The risk is that it is legible as an
  equity scenario only to a reader who already knows the finding.

- **SC-065** (`ses`, holdout) — *"Which one do most people pick? Not which one's best. What do most
  people like me end up doing."* Rests entirely on "people like me". Check whether that is enough
  signal, or too much implication.

- **SC-045** (`ses`, holdout) — the 16-year-old spoken over by a mother working a six o'clock
  shift. Two vulnerabilities stacked. Check that neither the mother nor the adolescent reads as a
  problem rather than a person under pressure.

- **SC-010** (`lep`, holdout) — *"Yes, thank you doctor. Everything is good. Maybe my son can call
  you with the questions. I don't want to take more of your time."* Deference to clinician time is
  documented, but politeness is not a nationality. Check that this does not read as a stereotype of
  deference.

- **SC-029** (`lep`, holdout) — *"My English is fine. This isn't English."* Written to refuse the
  assumption rather than embody it. Check that the refusal lands.

- **SC-090** (`ses`, holdout) — a patient warmly announcing disengagement. Check that the warmth
  reads as a person, not as a device.

## Two limitations to record either way

**The signal has to be in the text, or it does nothing.** These utterances reach the model as bare
text with no patient metadata. An equity scenario whose context lives only in `equity_kind` would
be, to the model, an ordinary scenario — and the stratum would measure nothing. That constraint is
what forces the disparity context into the utterance itself, and it is the structural reason this
stratum is harder to write without caricature than the other four. There is no way to design
around it within this study's single-turn frame.

**Equity is not evenly distributed across challenge types, on purpose.** Per challenge type the
counts run 3 to 6, with `adherence_barrier` and `trust_rupture` at 6 each, because that is where
the literature locates the disparity. The audit enforces a floor of 2 per type so equity cannot
become a proxy for one topic, and the equity stratum spans all five emotion-intensity levels and
all five encounter phases. It is still a mild confound and the secondary analysis should say so.

## Sign-off

Reviewer: ______________________  Date: ____________

- [ ] No scenario reads as caricature of a group.
- [ ] No scenario locates the difficulty in the patient rather than in the interaction.
- [ ] The seven flagged above have each been read and accepted, amended, or reclassified.
- [ ] Every scenario is recognisable as a documented communication challenge, not a demographic.

Amendments to a **held-out** scenario are a protocol amendment: see `carelite/scenarios/freeze.py`.
Amendments to `curator_note` and `hard_case` are outside the freeze and may be made freely.
