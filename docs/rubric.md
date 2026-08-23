# CARELite AI — Rating Rubric

**Rubric version 1.0.0.** Machine-readable twin: `carelite/eval/rubric/dimensions.py`.
Calibration set: `carelite/eval/rubric/calibration.py`. Deterministic counters:
`carelite/eval/rubric/scorers.py`.

This document is what human raters read before they score anything, and what the LLM judge is
prompted against. It has to stand alone: if a rater cannot score a response using only this file,
the rubric is not finished.

---

## What is being scored

**Unit of analysis.** One *response* — a single clinician turn, addressed to the patient,
answering one patient utterance from the scenario bank. You are scoring the words the clinician
would say, not advice about what a clinician should do.

**Scale.** Every dimension is a 1–5 integer. There are eleven dimensions, fixed by
`RUBRIC_DIMENSIONS` in `carelite/types.py`: five from NURSE, four from the Four Habits Model, and
two secondary dimensions.

**Blinding.** Raters never see which condition produced a response. Presentation order is
randomised per rater and recorded in `rating_assignment` so unblinding is a join, not a guess.

---

## ⚠ REVERSE CODING — READ THIS BEFORE YOU SCORE ANYTHING

**Ten of the eleven dimensions are scored so that 5 is good.**

**`ritualistic` is the exception. On `ritualistic`, 5 is the WORST score.**

| | 1 | 5 |
|---|---|---|
| all ten other dimensions | worst | best |
| **`ritualistic`** | **best** — no ritual at all | **worst** — a script with the framework showing |

The raw value stored in `rubric_score.ritualistic` is always on that higher-is-worse scale,
because that is the direction the construct is named in. Anything that averages, sums, correlates
or ranks across dimensions must first put every dimension on the same polarity with
`dimensions.to_quality()`, which returns `6 - raw` for reverse-coded dimensions and passes
everything else through unchanged.

**Why this matters more than it looks like it does.** Build plan v3 predicts that Condition B
(framework-prompted) loses to Condition A (bare model) on naturalness *precisely because*
framework prompting induces formulaic, script-like output. `ritualistic` is the dimension that
detects that mechanism. A missing `6 - x` inverts the study's headline secondary finding and
produces numbers that still look entirely plausible. `tests/unit/rubric/test_reverse_coding.py`
asserts the direction against the calibration archetypes; do not "fix" a failure there by flipping
a constant.

---

## How to score

1. **Read the patient utterance first**, then the response, then score. Several dimensions —
   `understand`, `explore`, `epp`, `naturalness` — depend on whether the response engages what
   *this* patient actually said.
2. **Score each dimension independently.** Do not let a strong overall impression lift the weak
   dimensions. Calibration item CAL-03 is the best response in the set and scores 2 on `respect`,
   because the respecting move is not in it. Halo effects compress the variance the analysis
   depends on.
3. **Quote your evidence.** Every score that turns on a specific piece of text gets a verbatim
   quote from the response. This is the v3 §13 grounding rule and it applies to humans as well as
   to the judge. A score of 1 for an absent move has no positive evidence to quote — leave the
   span empty rather than inventing one.
4. **Absence scores 1, badly-done scores 2–3.** A move that is not attempted is a 1. A move that
   is attempted generically is a 3. The anchors below fix what "generically" means.
5. **Do not score length.** Longer responses have more surface on which to find a NURSE move.
   Word count is measured separately and checked as a confounder; it is not a quality signal.

---

## The shared anchor scenario

Every anchor example below answers the same patient turn, so the five levels of every dimension
are comparable with each other and across dimensions. This is also the scenario used by all five
calibration responses.

> **Patient:** "They said the scan showed a spot on my lung. I've been up all night. My mother
> went through this and it was awful. I don't even know what to ask you."

Synthetic, like everything in this repository.

---

# Part 1 — NURSE

The NURSE mnemonic for responding to emotion — Naming, Understanding, Respecting, Supporting,
Exploring — is described in **Back AL, Arnold RM, Baile WF, Tulsky JA, Fryer-Edwards K.
Approaching difficult communication tasks in oncology. *CA Cancer J Clin.* 2005;55(3):164–177**,
and developed at length in **Back AL, Arnold RM, Tulsky JA. *Mastering Communication with
Seriously Ill Patients.* Cambridge University Press; 2009**. Each of the five is a distinct
verbal move, and they are scored as five separate behaviours rather than as one empathy score.

---

## 1. `name` — Naming

**Definition.** The response explicitly identifies the emotion the patient is expressing, and does
so tentatively rather than as a claim to know the patient's inner state better than they do.
Naming an emotion accurately is the move that lets the patient correct you; a generic "this is a
lot" is a gesture at emotion, not a naming of one.

**Source.** Back et al. 2005 — the originating description of the mnemonic. The tentativeness
requirement ("it sounds like…", "I wonder if…") is theirs: a flat assertion of what the patient
feels is a different and worse move.

**Rating question.** *Does the response name the patient's emotion, specifically and tentatively?*

| | Anchor |
|---|---|
| **1** | No emotion is named or gestured at. <br>*"Let's talk about what the scan showed."* |
| **3** | Emotion is gestured at generically but not named. <br>*"I can tell this is a lot for you."* |
| **5** | The specific emotion is named tentatively and tied to what the patient said. <br>*"It sounds like you're frightened — and that watching your mother go through this is sitting right on top of it."* |

**Boundary.** Asking about an emotion is *Exploring*, not Naming. "What are you most afraid of?"
scores under `explore`; it does not score here, because the patient is never given a name to
accept or correct.

---

## 2. `understand` — Understanding

**Definition.** The response legitimises the emotion by showing *why* it makes sense given what
this patient has said. Back et al. are explicit that a bare "I understand" fails: it asserts
access the clinician has not demonstrated. The scoring question is whether the response shows its
working.

**Source.** Back et al. 2005; Back, Arnold & Tulsky 2009 — the "do not say *I understand* on its
own" guidance.

**Rating question.** *Does the response show why the patient's reaction makes sense?*

| | Anchor |
|---|---|
| **1** | The emotion is corrected or reassured away. <br>*"There's no reason to panic yet — a spot doesn't mean cancer."* |
| **3** | Bare formulaic legitimation with no reasoning. <br>*"I understand this is hard."* |
| **5** | Legitimation grounded in the patient's own stated reason. <br>*"Given what you watched your mother go through, of course a spot on your lung sounds like the start of that same story. Anyone would have been up all night."* |

**Note on the 1 anchor.** Premature reassurance scores *below* silence on this dimension, not
above it. "Most of these are benign", offered before the emotion has been acknowledged, corrects
the patient's reaction rather than legitimising it.

---

## 3. `respect` — Respecting

**Definition.** The response explicitly acknowledges something the patient has done, endured or
managed. Respect in NURSE is a praise or credit move, and it scores on specificity: generic
courtesy is not the same act as naming what this person actually did.

**Source.** Back et al. 2005; VitalTalk NURSE teaching materials (the "R" statement).

**Rating question.** *Does the response credit the patient for something specific?*

| | Anchor |
|---|---|
| **1** | No acknowledgement of the patient's effort or coping appears anywhere. |
| **3** | Generic courtesy or generic praise. <br>*"Thank you for coming in today."* |
| **5** | Specific, earned, not flattering. <br>*"You came in and you're asking about it even after being awake all night — that takes something."* |

**Boundary.** "You're stronger than you know" is a 3, not a 5. It is praise, but it is unearned by
anything in the conversation and is portable to any patient.

---

## 4. `support` — Supporting

**Definition.** The response states partnership and makes it concrete: who will do what, by when,
and how the patient reaches someone before then. A partnership statement with no concrete
follow-through is an assertion of support rather than support.

**Source.** Back et al. 2005; the "I will be with you through this" partnership statement in Back,
Arnold & Tulsky 2009.

**Rating question.** *Does the response commit to partnership in a way the patient could act on?*

| | Anchor |
|---|---|
| **1** | Information only, no partnership. <br>*"The radiologist's report will be in your chart."* |
| **3** | Partnership asserted but empty. <br>*"We'll take care of you."* |
| **5** | Partnership plus concrete availability. <br>*"I'm staying with you through this. I'll call you with the pulmonary appointment by Thursday, and if you can't sleep before then, my nurse can reach me."* |

**Note.** Repetition does not raise the score. Calibration item CAL-04 asserts partnership four
separate times and scores 3, because none of the four is concrete.

---

## 5. `explore` — Exploring

**Definition.** The response invites the patient to say more about the emotion or concern, with an
open question that follows the patient's own words. "Any questions?" is a closed pro-forma
invitation and is scored as such; a clinical history question is not an exploring move at all.

**Source.** Back et al. 2005 — the "E" statement, explicitly contrasted with closed clinical
questioning.

**Rating question.** *Does the response open a door for the patient to say more about the emotion?*

| | Anchor |
|---|---|
| **1** | No invitation. The response moves straight to the plan. |
| **3** | Pro-forma or closed invitation. <br>*"Any questions?"* |
| **5** | Open invitation built out of the patient's own words. <br>*"When you say it was awful for your mother — what part of it are you seeing when you think about yourself?"* |

**Boundary.** An open question aimed at *comprehension* ("What's your understanding of where that
leaves you?") is Eliciting the Patient's Perspective, and is credited under `epp`. Exploring is
about the emotion.

---

# Part 2 — The Four Habits Model

**Frankel RM, Stein T. Getting the most out of the clinical encounter: the Four Habits Model.
*Permanente Journal.* 1999;3(3):79–88** (reprinted *J Med Pract Manage.* 2001;16(4):184–191).

The anchors below follow the **Four Habits Coding Scheme — Krupat E, Frankel R, Stein T, Irish J.
The Four Habits Coding Scheme: validation of an instrument to assess clinicians' communication
behaviour. *Patient Educ Couns.* 2006;62(1):38–45** — which is itself a 1–5 anchored scale, and is
the direct methodological precedent for anchoring this rubric rather than using bare Likert items.
Inter-rater performance for that scheme is reported in **Fossli Jensen B, Gulbrandsen P, Benth JŠ,
Dahl FA, Krupat E, Finset A. Interrater reliability for the Four Habits Coding Scheme as part of a
randomized controlled trial. *Patient Educ Couns.* 2010;80(3):405–409**.

---

## 6. `ib` — Invest in the Beginning

**Definition.** The response creates rapport quickly, elicits the patient's full set of concerns
before working on any one of them, and plans the conversation with the patient. The habit's
signature failure is solving the first problem raised and never asking what else was on the list.

**Source.** Frankel & Stein 1999, Habit 1; Krupat et al. 2006 (items: rapport, elicits full
agenda, plans the visit).

**Rating question.** *Does the response orient the patient and negotiate what the conversation
covers?*

| | Anchor |
|---|---|
| **1** | Opens mid-explanation with no orientation and no agenda. <br>*"A pulmonary nodule under eight millimetres is usually followed with serial imaging."* |
| **3** | A rapport beat, then straight into the clinician's own agenda. <br>*"Before we get into the scan — how are you doing? Okay. So the next step is a CT."* |
| **5** | Rapport, full agenda, negotiated plan. <br>*"Before I say anything about the scan, I want to know what's on your list — you said you don't know what to ask, so let's start with what you're most afraid of, and I'll add the two things I think matter. Does that work?"* |

**Scenario-specific note.** This patient says *"I don't even know what to ask you."* That is an
explicit request for help setting the agenda. A response that does not answer it cannot score
above 2 here.

---

## 7. `epp` — Elicit the Patient's Perspective

**Definition.** The response asks for the patient's own model: what they think is going on, what
they specifically want from this conversation, and how the problem is affecting their life.
Checking whether the patient understood the *clinician's* model is a different act and belongs to
Invest in the End, not here.

**Source.** Frankel & Stein 1999, Habit 2; Krupat et al. 2006 (items: ideas, specific request,
impact on life). The ancestor is the explanatory-model interview — **Kleinman A, Eisenberg L,
Good B. Culture, illness and care. *Ann Intern Med.* 1978;88(2):251–258**.

**Rating question.** *Does the response ask what the patient thinks and what this is doing to
their life?*

| | Anchor |
|---|---|
| **1** | The clinician's model only. <br>*"A spot on a scan means we need another image."* |
| **3** | Comprehension of the clinician's model is checked, or one perspective element is asked for. <br>*"Does that make sense to you?"* |
| **5** | Ideas, request and impact are genuinely asked for, before the clinician's account. <br>*"What do you think is going on? And when you picture what happened to your mother, what part do you think is coming for you? I'd rather know that before I tell you what I think."* |

---

## 8. `de` — Demonstrate Empathy

**Definition.** Whether the response, **taken as a whole**, lands as empathic: emotion is welcomed
rather than managed, the register matches the patient's, and the response stays with the emotion
long enough for it to have been heard.

**This is deliberately a holistic judgement and it is NOT the average of the NURSE items.** A
response can execute Naming and Understanding correctly and still read as cold — calibration item
CAL-02 scores 4 on `name` and `understand` and 2 here. That divergence is exactly what this study
is trying to measure, and collapsing `de` into the NURSE average destroys it.

**Source.** Frankel & Stein 1999, Habit 3; Krupat et al. 2006. The blocking pattern scored at the
low anchor — pivoting to information immediately after an emotional cue — is theme 2 of
`README.md`, and the corpus documents it as falling disproportionately on minority patients.

**Rating question.** *Taken as a whole, does this response land as empathic?*

| | Anchor |
|---|---|
| **1** | Emotional cue visible and walked past — the blocking pattern. <br>*"Right, so, a spot. We'll need a CT. Any questions?"* |
| **3** | One stock empathic beat, then the emotion is abandoned. <br>*"I'm sorry, that sounds stressful. So, the next step is a CT."* |
| **5** | The response stays with the emotion across more than one beat and matches register. <br>*"You've been up all night with this. I'm not going to walk past that. Tell me about your mother."* |

---

## 9. `ie` — Invest in the End

**Definition.** The response delivers information in a form the patient can carry: a small number
of key messages in plain language, an explicit check that they landed, the patient involved in the
decision, and a clear next step. **Both halves matter** — a comprehension check attached to a
jargon-dense wall of information does not reach the top of the scale.

**Source.** Frankel & Stein 1999, Habit 4; Krupat et al. 2006. Teach-back evidence: **Schillinger
D, Piette J, Grumbach K, et al. Closing the loop: physician communication with diabetic patients
who have low health literacy. *Arch Intern Med.* 2003;163(1):83–90**; **Ha Dinh TT, Bonner A,
Clark R, Ramsbotham J, Hines S. The effectiveness of the teach-back method on adherence and
self-management in health education. *JBI Database System Rev Implement Rep.*
2016;14(1):210–247**. The three-key-message ceiling follows the plain-language and Ask Me 3
guidance in **Weiss BD. *Health Literacy and Patient Safety: Help Patients Understand.* 2nd ed.
AMA Foundation; 2007**, which is also the source of the sixth-grade reading-level target used by
the deterministic scorers.

**Rating question.** *Could the patient repeat back what matters and say what happens next?*

| | Anchor |
|---|---|
| **1** | A wall of information, no check, no next step. <br>*"…serial imaging at three, six and twelve months, and we'll reassess."* |
| **3** | Next step given, no comprehension check, no involvement. <br>*"So, CT next week, and let me know if you have questions."* |
| **5** | At most three plain-language key messages, a genuine teach-back, a choice, and a next step. <br>*"Two things to hold onto: it's small, and the next step is one more picture, not treatment. When you tell your husband tonight, what will you say? And do you want the scan before or after your daughter's visit?"* |

**Teach-back vs. pseudo-teach-back.** *"Does that make sense?"*, *"Do you understand?"* and *"Any
questions?"* are **not** teach-back. The literature contrasts them with it directly: a patient who
did not understand will usually still answer yes. Teach-back asks the patient to restate, in their
own words, what they understood or what they will do. `scorers.py` detects the two separately so
that a rater and the counters can be checked against each other.

---

# Part 3 — Secondary dimensions

These two are the reason the study is interesting. They are scored **independently of clinical
quality**: a blunt, unempathic reply can be entirely natural, and a technically perfect NURSE
response can be entirely ritual.

---

## 10. `naturalness` — Naturalness

**Definition.** Whether this reads like something a skilled clinician would actually say to this
patient at this moment: register fits, the wording is specific to what the patient said, and
nothing in it is machine-shaped.

**Naturalness is not quality.** Calibration item CAL-01 is a blunt, empathy-free information dump
that scores 1s across NURSE and **4** on naturalness, because that is what a rushed clinician
genuinely sounds like. Raters who let quality bleed into naturalness make the study's primary
secondary outcome uninterpretable.

**Source.** **Salmon P, Young B. Creativity in clinical communication: from communication skills
to skilled communication. *Med Educ.* 2011;45(3):217–226** — the argument that codified skill
lists produce performance rather than responsive speech. See also **de Haes H, Bensing J. Endpoints
in medical communication research. *Patient Educ Couns.* 2009;74(3):287–294**.

**Rating question.** *Would a skilled clinician actually say this, in these words, to this
patient?*

| | Anchor |
|---|---|
| **1** | Nobody talks like this. <br>*"I acknowledge your emotional state. Naming: you are experiencing fear. Understanding: this is understandable."* |
| **3** | Fluent but generic; could be pasted into any conversation. <br>*"That sounds really hard. I can only imagine what you're going through. Let's talk about next steps."* |
| **5** | Specific to this patient and unrepeatable elsewhere. <br>*"Up all night. And your mother's history sitting right on top of it — no wonder. Tell me which part is loudest right now and we'll start there."* |

---

## 11. `ritualistic` — Ritualistic  ⚠ **REVERSE-CODED: 5 IS THE WORST SCORE**

> **1 = no ritual at all (best). 5 = a script with the framework showing (worst).**
> This is the only dimension in the rubric where a higher number is a worse response.
> Never average, correlate or rank it without passing it through `dimensions.to_quality()` first.

**Definition.** How far the response is a *performance of a communication framework* rather than a
reply to a person. The markers are:

- **framework vocabulary or labels visible in the output** — "Naming:", "NURSE", "Invest in the
  beginning", "Demonstrate empathy";
- **templated scaffolding** — headed sections, one bullet per empathic move, numbered empathy
  steps;
- **stock empathy stems stacked in sequence** — "I hear you", "that must be so hard", "please know
  that", "every step of the way";
- **portability** — wording that would be equally applicable to any patient with any problem.

**A response can legitimately score 4–5 on the NURSE dimensions and 5 here at the same time.**
That is not a contradiction. It is the predicted failure mode of Condition B and the entire reason
this dimension exists. Do not mark the NURSE items down because the response feels like a script,
and do not mark `ritualistic` down because the NURSE moves are technically correct.

**Ritual does not require a template.** Calibration item CAL-04 has no labels and no scaffolding at
all, and scores 4, because it is eight stock empathy stems in a row with no content between them.

**Source.** Salmon & Young 2011 (ritualised performance of communication-skills lists); **Salmon P,
Young B. A new paradigm for clinical communication: critical review of literature in cancer care.
*Med Educ.* 2017;51(3):258–268**. It is also a stated project constraint: CARELite is explicitly
not a script generator, on the grounds that frameworks which become scripts stop working.

**Rating question.** *How much does this read as a framework being performed rather than a person
being answered? (1 = not at all, 5 = entirely.)*

| | Anchor |
|---|---|
| **1** *(best)* | No template, no stock stems, no framework vocabulary; the wording could only have been produced for this patient. <br>*"Up all night. And your mother's history sitting right on top of it — no wonder. Tell me which part is loudest right now."* |
| **3** | Stock empathy stems in sequence, no visible scaffolding, interchangeable across patients. <br>*"I hear you. That must be really difficult. I want you to know we're here for you. Let's talk about next steps."* |
| **5** *(worst)* | The framework is on the page. <br>*"**Naming:** It sounds like you're feeling anxious. **Understanding:** It is completely understandable that you would feel this way. **Respecting:** I want to acknowledge your strength in coming in today. **Supporting:** Please know that we are here for you. **Exploring:** Can you tell me more about how you're feeling?"* |

**Relationship to `naturalness`.** The two are expected to correlate negatively but they are not
one construct: a response can be unnatural without being ritual (odd register, wrong idiom,
non-sequitur) and — as CAL-04 shows — ritual without being disfluent. If the empirical correlation
between `naturalness` and `to_quality("ritualistic", …)` turns out to be near-unity, that is a
reportable finding about the rubric, not a licence to drop a dimension mid-study.

---

# Part 4 — Common rating errors

These are the four failures the calibration set exists to prevent. Each is named with the item
that demonstrates it.

| Error | Looks like | Item |
|---|---|---|
| **Quality bleeding into naturalness** | Scoring a cold response low on naturalness because it is cold | CAL-01 |
| **Collapsing ritual into adherence** | Marking NURSE down because the response is a script, or marking `ritualistic` down because the moves are correct | CAL-02 |
| **Halo** | Lifting a weak dimension because the response is good overall | CAL-03 |
| **Warmth mistaken for empathy** | High `de`, `explore` or `ie` for a response that is only warm | CAL-04 |

And the fifth, which is not a rater error but a coding error: **a sign error on `ritualistic`**.
See the banner at the top of this file.

---

# Part 5 — Deterministic counters

`carelite/eval/rubric/scorers.py` measures what can be counted without an opinion. Pinned at
`SCORER_VERSION` and a pure function of the response text, so re-running the study reproduces
them exactly, with no model in the loop.

**These are descriptors, not scores.** They do not replace a rating. They exist to anchor the
judge: a judge that rates `ie` = 5 on a response with a grade-16 reading level and no
comprehension check is disagreeing with a fact, not with a rater, and that disagreement is
findable.

| Measure | What it counts | Bears on | Interpretation |
|---|---|---|---|
| `jargon_density` | Clinical-jargon hits per word, against a curated lexicon | `ie`, theme 5 | Higher is worse. Conservative undercount by design — see below |
| `flesch_kincaid_grade` | 0.39·(w/s) + 11.8·(syl/w) − 15.59 | `ie` | Patient material is conventionally targeted at ~grade 6 (Weiss 2007) |
| `question_count` | Sentences ending in `?` | `explore`, `epp` | **Two-sided.** Zero questions cannot support an exploring move; all questions is an interrogation |
| `teach_back_present` | Genuine restate-it-back requests | `ie` | Presence is meaningful; absence caps `ie` |
| `pseudo_teach_back_phrases` | "Does that make sense?", "Any questions?" | `ie` | Detected *separately* — these are not teach-back |
| `message_count` | Declarative content units; an enumerated line counts as one | `ie`, theme 5 | Upper bound on message load; >3 is the flag |
| `response_length` | Word count | all | **Not a quality signal.** A confounder to check, because longer responses offer more surface on which to find a move |
| `hedge_density` | Epistemic hedges per word | trust/uncertainty | **Two-sided.** Low reads as false certainty, high as evasion |
| `ritual_markers` | Framework labels, scaffold lines, stock stems | `ritualistic` | Verbatim spans, usable directly as evidence |
| `ritualistic_proxy` | 1–5 screening estimate | `ritualistic` | **On the same reverse-coded scale: 5 is worst** |

**Jargon lexicon, and what it deliberately misses.** Terms are drawn from the comprehension
literature — **Gotlieb R, Praska C, Hendrickson MA, et al. Accuracy in patient understanding of
common medical phrases. *JAMA Netw Open.* 2022;5(11):e2242972**; **Pitt MB, Hendrickson MA.
Eradicating jargon-oblivion. *J Gen Intern Med.* 2020;35(2):598–603** — plus terms recurring in
this project's scenario domain. Words that are jargon in clinical use but ordinary English
elsewhere ("negative", "positive", "progressive", "gross", "occult") are **excluded**: they are
among the most misunderstood phrases in the literature, but a bag-of-words matcher cannot tell
"the test was negative" from "a negative experience", and a false positive is worse here than a
miss. `jargon_density` is therefore a conservative undercount and must be reported as one.

**`ritualistic_proxy` banding.** Start at 1; **+2** if any framework vocabulary or labelled section
appears at all; **+1** if two or more scaffold lines; **+1** if two or more stock stems; **+1** if
four or more stock stems or three or more framework labels; clamp to [1, 5]. It is a screening
estimate, not a rating: it cannot see a response that is formulaic in rhythm without using these
surface forms (it under-reads CAL-04 at 3 against a consensus of 4), and it over-fires on a
response that legitimately uses a numbered list for a medication schedule. Its agreement with
human consensus is reported as a validation result, not assumed.

**The deterministic rater row.** `deterministic_rubric_score()` writes a
`rater_type='deterministic'` row filling **only** `ritualistic`. The other ten dimensions are left
`NULL` rather than given a fabricated proxy — a `rubric_score` row with an invented `de` value
would silently contaminate every aggregate that reads the table.

---

# Part 6 — Calibration protocol (v3 §12)

Before rating anything, every rater independently scores the **five calibration responses** in
`carelite/eval/rubric/calibration.py`. All five answer the shared anchor scenario above. Scores
are then compared against the consensus values and the recorded rationales and disagreements, and
discussed.

| Item | Archetype | The thing it teaches | `naturalness` | `ritualistic` |
|---|---|---|---|---|
| **CAL-01** | Blunt information dump | Naturalness is not quality: 1s across NURSE, 4 on naturalness | 4 | 1 |
| **CAL-02** | Framework-labelled script | High NURSE adherence and maximum ritual coexist | 1 | **5** |
| **CAL-03** | Integrated, natural (the target) | No halo — scores 2 on `respect` because the move is absent | 5 | 1 |
| **CAL-04** | Warm and empty | Warmth is not empathy; ritual without any template | 2 | 4 |
| **CAL-05** | Genuine but jargon-heavy | Where the counters and the raters visibly interact | 4 | 1 |

Rate them **in that order**. CAL-01 and CAL-02 set the two axes raters most often conflate, and
CAL-05 is only discussable once the first four are settled.

Each item carries per-dimension rationales, verbatim evidence spans, and — where the consensus was
genuinely argued — the specific disagreement and the rule that settled it. Every evidence span is
verified verbatim against its response by `validate_calibration_set()`, the same grounding rule
v3 §13 imposes on the judge.

---

# Part 7 — Reliability and analysis notes

- **Agreement.** Krippendorff's α (ordinal) for human–human agreement, computed **per dimension**.
  Report it whatever it is; a low α on a dimension is a finding about that construct, not a
  failure to conceal.
- **Judge validity.** α and Spearman ρ between judge and human consensus, again per dimension.
  Judges are typically decent on structural items and poor on naturalness — v3 §13 and the v3
  Part V risk register both say so in advance. The pre-registered threshold below which
  judge-only results are demoted to exploratory applies per dimension.
- **`naturalness` is the dimension most at risk.** If judge–human agreement on it is poor, it
  becomes human-rated only, at n = 60. That contingency is pre-specified, not a post-hoc rescue.
- **Aggregation.** Any composite must call `dimensions.to_quality()` first. The NURSE composite
  (the study's primary outcome) is the five NURSE dimensions only, all of which are
  higher-is-better; `ritualistic` never enters a composite without being flipped.
- **Negative control.** v3 §14 includes a deliberately degraded prompt condition. If this rubric
  cannot distinguish Condition D from Condition B, the rubric is not measuring what it claims to.

---

# Version control

| Version | Change |
|---|---|
| 1.0.0 | Initial rubric: eleven dimensions, anchors, deterministic counters, five-item calibration set. |

`RUBRIC_VERSION` in `dimensions.py` is the authority and is persisted with every rating. If any
definition or anchor changes, the version is bumped and the calibration set is **re-scored** rather
than carried forward — consensus scores are only meaningful against the rubric they were agreed
under.
