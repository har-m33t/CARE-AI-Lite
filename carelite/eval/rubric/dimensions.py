"""The rubric itself, in machine-readable form.

`docs/rubric.md` is the document raters read. This module is the same rubric as
data, so the judge lane, the human-harness lane and the stats lane all agree on
what each dimension means and — critically — which way each one points.

======================================================================
REVERSE CODING. READ THIS BEFORE USING ANY `ritualistic` VALUE.
======================================================================
Ten of the eleven dimensions are scored so that **5 is good**.

`ritualistic` is the exception. It is scored so that **5 is BAD**: 5 means the
response is maximally formulaic and script-like, 1 means it shows no ritual at
all. The raw value stored in `rubric_score.ritualistic` is always on that
higher-is-worse scale, because that is the direction the construct is named in.

Anything that averages, sums, correlates or ranks across dimensions must first
put every dimension on the same polarity with `to_quality()`. Build plan v3
predicts Condition B loses to Condition A on naturalness *because* framework
prompting induces ritual; a missing `6 - x` inverts that headline finding
silently and the numbers will still look plausible. `tests/unit/rubric/`
asserts the direction. Do not "fix" those tests by flipping the constant.
======================================================================

Sources for the dimension definitions are named per dimension in `.source` and
discussed at length in `docs/rubric.md`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from carelite.types import RUBRIC_DIMENSIONS

__all__ = [
    "DIMENSIONS",
    "REVERSE_CODED",
    "RUBRIC_VERSION",
    "SCALE_MAX",
    "SCALE_MIN",
    "Dimension",
    "Framework",
    "Polarity",
    "dimension",
    "is_reverse_coded",
    "to_quality",
    "to_quality_scores",
]

#: Bumped whenever a definition or an anchor changes. Persisted alongside
#: ratings so a re-analysis can tell which rubric a score was made under.
RUBRIC_VERSION = "1.0.0"

SCALE_MIN = 1
SCALE_MAX = 5


class Polarity(StrEnum):
    """Which end of the 1-5 scale is the good end."""

    HIGHER_IS_BETTER = "higher_is_better"
    #: Reverse-coded. A raw 5 is the worst possible response on this dimension.
    LOWER_IS_BETTER = "lower_is_better"


class Framework(StrEnum):
    NURSE = "NURSE"
    FOUR_HABITS = "Four Habits"
    SECONDARY = "Secondary"


@dataclass(frozen=True, slots=True)
class Dimension:
    """One scored dimension: what it means, where it comes from, how it points."""

    key: str
    label: str
    framework: Framework
    polarity: Polarity
    definition: str
    source: str
    #: What the rater is looking at, phrased as the question they answer.
    question: str
    anchor_1: str
    anchor_3: str
    anchor_5: str

    @property
    def reverse_coded(self) -> bool:
        return self.polarity is Polarity.LOWER_IS_BETTER


_DIMENSIONS: tuple[Dimension, ...] = (
    # ------------------------------------------------------------- NURSE ---
    Dimension(
        key="name",
        label="Naming",
        framework=Framework.NURSE,
        polarity=Polarity.HIGHER_IS_BETTER,
        definition=(
            "The response explicitly identifies the emotion the patient is expressing, and does "
            "so tentatively rather than as a claim to know the patient's inner state better than "
            "they do. Naming an emotion accurately is the move that lets the patient correct "
            "you; a generic 'this is a lot' is a gesture at emotion, not a naming of one."
        ),
        source=(
            "Back AL, Arnold RM, Baile WF, Tulsky JA, Fryer-Edwards K. Approaching difficult "
            "communication tasks in oncology. CA Cancer J Clin. 2005;55(3):164-177 — the "
            "originating description of the NURSE mnemonic."
        ),
        question="Does the response name the patient's emotion, specifically and tentatively?",
        anchor_1="No emotion is named or gestured at. 'Let's talk about what the scan showed.'",
        anchor_3=(
            "Emotion is gestured at generically but not named. 'I can tell this is a lot for you.'"
        ),
        anchor_5=(
            "The specific emotion is named tentatively and tied to what the patient said. "
            "'It sounds like you're frightened — and that watching your mother go through this "
            "is sitting right on top of it.'"
        ),
    ),
    Dimension(
        key="understand",
        label="Understanding",
        framework=Framework.NURSE,
        polarity=Polarity.HIGHER_IS_BETTER,
        definition=(
            "The response legitimises the emotion by showing why it makes sense given what this "
            "patient has said. Back et al. are explicit that a bare 'I understand' fails: it "
            "asserts access the clinician has not demonstrated. The scoring question is whether "
            "the response shows its working."
        ),
        source=(
            "Back et al. 2005 (as above); Back AL, Arnold RM, Tulsky JA. Mastering Communication "
            "with Seriously Ill Patients. Cambridge University Press; 2009 — the 'do not say "
            "I understand on its own' guidance."
        ),
        question="Does the response show why the patient's reaction makes sense?",
        anchor_1=(
            "The emotion is corrected or reassured away. 'There's no reason to panic yet — a "
            "spot doesn't mean cancer.'"
        ),
        anchor_3="Bare formulaic legitimation with no reasoning. 'I understand this is hard.'",
        anchor_5=(
            "Legitimation grounded in the patient's own stated reason. 'Given what you watched "
            "your mother go through, of course a spot on your lung sounds like the start of that "
            "same story. Anyone would have been up all night.'"
        ),
    ),
    Dimension(
        key="respect",
        label="Respecting",
        framework=Framework.NURSE,
        polarity=Polarity.HIGHER_IS_BETTER,
        definition=(
            "The response explicitly acknowledges something the patient has done, endured or "
            "managed. Respect in NURSE is a praise or credit move, and it scores on specificity: "
            "generic courtesy is not the same act as naming what this person actually did."
        ),
        source="Back et al. 2005; VitalTalk NURSE teaching materials (the 'R' statement).",
        question="Does the response credit the patient for something specific?",
        anchor_1="No acknowledgement of the patient's effort or coping appears anywhere.",
        anchor_3="Generic courtesy or generic praise. 'Thank you for coming in today.'",
        anchor_5=(
            "Specific, earned, not flattering. 'You came in and you're asking about it even "
            "after being awake all night — that takes something.'"
        ),
    ),
    Dimension(
        key="support",
        label="Supporting",
        framework=Framework.NURSE,
        polarity=Polarity.HIGHER_IS_BETTER,
        definition=(
            "The response states partnership and makes it concrete: who will do what, by when, "
            "and how the patient reaches someone before then. A partnership statement with no "
            "concrete follow-through is an assertion of support rather than support."
        ),
        source=(
            "Back et al. 2005; the 'I will be with you through this' partnership statement in "
            "Back, Arnold & Tulsky 2009."
        ),
        question="Does the response commit to partnership in a way the patient could act on?",
        anchor_1="Information only, no partnership. 'The radiologist's report will be in your chart.'",
        anchor_3="Partnership asserted but empty. 'We'll take care of you.'",
        anchor_5=(
            "Partnership plus concrete availability. 'I'm staying with you through this. I'll "
            "call you with the pulmonary appointment by Thursday, and if you can't sleep before "
            "then, my nurse can reach me.'"
        ),
    ),
    Dimension(
        key="explore",
        label="Exploring",
        framework=Framework.NURSE,
        polarity=Polarity.HIGHER_IS_BETTER,
        definition=(
            "The response invites the patient to say more about the emotion or concern, with an "
            "open question that follows the patient's own words. 'Any questions?' is a closed "
            "pro-forma invitation and is scored as such; a clinical history question is not an "
            "exploring move at all."
        ),
        source="Back et al. 2005 (the 'E' statement, contrasted with closed clinical questioning).",
        question="Does the response open a door for the patient to say more about the emotion?",
        anchor_1="No invitation. The response moves straight to the plan.",
        anchor_3="Pro-forma or closed invitation. 'Any questions?'",
        anchor_5=(
            "Open invitation built out of the patient's own words. 'When you say it was awful "
            "for your mother — what part of it are you seeing when you think about yourself?'"
        ),
    ),
    # ------------------------------------------------------- Four Habits ---
    Dimension(
        key="ib",
        label="Invest in the Beginning",
        framework=Framework.FOUR_HABITS,
        polarity=Polarity.HIGHER_IS_BETTER,
        definition=(
            "The response creates rapport quickly, elicits the patient's full set of concerns "
            "before working on any one of them, and plans the conversation with the patient. "
            "The habit's signature failure is solving the first problem raised and never asking "
            "what else was on the list."
        ),
        source=(
            "Frankel RM, Stein T. Getting the most out of the clinical encounter: the Four "
            "Habits Model. Permanente Journal. 1999;3(3):79-88 (reprinted J Med Pract Manage. "
            "2001;16(4):184-191). Scoring anchors follow the Four Habits Coding Scheme: Krupat "
            "E, Frankel R, Stein T, Irish J. The Four Habits Coding Scheme: validation of an "
            "instrument to assess clinicians' communication behaviour. Patient Educ Couns. "
            "2006;62(1):38-45, which is also a 1-5 anchored scale."
        ),
        question="Does the response orient the patient and negotiate what the conversation covers?",
        anchor_1=(
            "Opens mid-explanation with no orientation and no agenda. 'A pulmonary nodule under "
            "eight millimetres is usually followed with serial imaging.'"
        ),
        anchor_3=(
            "A rapport beat, then straight into the clinician's own agenda. 'Before we get into "
            "the scan — how are you doing? Okay. So the next step is a CT.'"
        ),
        anchor_5=(
            "Rapport, full agenda, negotiated plan. 'Before I say anything about the scan, I "
            "want to know what's on your list — you said you don't know what to ask, so let's "
            "start with what you're most afraid of, and I'll add the two things I think matter. "
            "Does that work?'"
        ),
    ),
    Dimension(
        key="epp",
        label="Elicit the Patient's Perspective",
        framework=Framework.FOUR_HABITS,
        polarity=Polarity.HIGHER_IS_BETTER,
        definition=(
            "The response asks for the patient's own model: what they think is going on, what "
            "they specifically want from this conversation, and how the problem is affecting "
            "their life. Checking whether the patient understood the clinician's model is a "
            "different act and belongs to Invest in the End, not here."
        ),
        source=(
            "Frankel & Stein 1999; Krupat et al. 2006 (Habit 2 items: ideas, specific request, "
            "impact on life). The ancestor is Kleinman A, Eisenberg L, Good B. Culture, illness "
            "and care. Ann Intern Med. 1978;88(2):251-258 — the explanatory-model interview."
        ),
        question="Does the response ask what the patient thinks and what this is doing to their life?",
        anchor_1="The clinician's model only. 'A spot on a scan means we need another image.'",
        anchor_3=(
            "Comprehension of the clinician's model is checked, or one perspective element is "
            "asked for. 'Does that make sense to you?'"
        ),
        anchor_5=(
            "Ideas, request and impact are genuinely asked for, before the clinician's account. "
            "'What do you think is going on? And when you picture what happened to your mother, "
            "what part do you think is coming for you? I'd rather know that before I tell you "
            "what I think.'"
        ),
    ),
    Dimension(
        key="de",
        label="Demonstrate Empathy",
        framework=Framework.FOUR_HABITS,
        polarity=Polarity.HIGHER_IS_BETTER,
        definition=(
            "Whether the response, taken as a whole, lands as empathic: emotion is welcomed "
            "rather than managed, the register matches the patient's, and the response stays "
            "with the emotion long enough for it to have been heard. This is deliberately a "
            "holistic judgement and it is NOT the average of the NURSE items. A response can "
            "execute Naming and Understanding correctly and still read as cold, and that "
            "divergence is exactly what this study is trying to measure."
        ),
        source=(
            "Frankel & Stein 1999 (Habit 3); Krupat et al. 2006. The blocking pattern scored at "
            "the low anchor — pivoting to information immediately after an emotional cue — is "
            "theme 2 of README.md and is documented in the project corpus as falling "
            "disproportionately on minority patients."
        ),
        question="Taken as a whole, does this response land as empathic?",
        anchor_1=(
            "Emotional cue visible and walked past — the blocking pattern. 'Right, so, a spot. "
            "We'll need a CT. Any questions?'"
        ),
        anchor_3=(
            "One stock empathic beat, then the emotion is abandoned. 'I'm sorry, that sounds "
            "stressful. So, the next step is a CT.'"
        ),
        anchor_5=(
            "The response stays with the emotion across more than one beat and matches register. "
            "'You've been up all night with this. I'm not going to walk past that. Tell me about "
            "your mother.'"
        ),
    ),
    Dimension(
        key="ie",
        label="Invest in the End",
        framework=Framework.FOUR_HABITS,
        polarity=Polarity.HIGHER_IS_BETTER,
        definition=(
            "The response delivers information in a form the patient can carry: a small number "
            "of key messages in plain language, an explicit check that they landed, the patient "
            "involved in the decision, and a clear next step. Both halves matter — a "
            "comprehension check attached to a jargon-dense wall of information does not reach "
            "the top of the scale."
        ),
        source=(
            "Frankel & Stein 1999 (Habit 4); Krupat et al. 2006. Teach-back evidence: Schillinger "
            "D, Piette J, Grumbach K, et al. Closing the loop: physician communication with "
            "diabetic patients who have low health literacy. Arch Intern Med. "
            "2003;163(1):83-90; Ha Dinh TT, Bonner A, Clark R, Ramsbotham J, Hines S. The "
            "effectiveness of the teach-back method. JBI Database System Rev Implement Rep. "
            "2016;14(1):210-247. The three-key-message ceiling follows the Ask Me 3 / plain-"
            "language guidance in Weiss BD. Health Literacy and Patient Safety: Help Patients "
            "Understand. 2nd ed. AMA Foundation; 2007."
        ),
        question="Could the patient repeat back what matters and say what happens next?",
        anchor_1=(
            "A wall of information, no check, no next step. '...serial imaging at three, six and "
            "twelve months, and we'll reassess.'"
        ),
        anchor_3=(
            "Next step given, no comprehension check, no involvement. 'So, CT next week, and let "
            "me know if you have questions.'"
        ),
        anchor_5=(
            "At most three plain-language key messages, a genuine teach-back, a choice, and a "
            "next step. 'Two things to hold onto: it's small, and the next step is one more "
            "picture, not treatment. When you tell your husband tonight, what will you say? And "
            "do you want the scan before or after your daughter's visit?'"
        ),
    ),
    # --------------------------------------------------------- secondary ---
    Dimension(
        key="naturalness",
        label="Naturalness",
        framework=Framework.SECONDARY,
        polarity=Polarity.HIGHER_IS_BETTER,
        definition=(
            "Whether this reads like something a skilled clinician would actually say to this "
            "patient at this moment: register fits, the wording is specific to what the patient "
            "said, and nothing in it is machine-shaped. Naturalness is scored independently of "
            "whether the response is any good clinically — a blunt, unempathic reply can be "
            "entirely natural, and calibration item CAL-01 is exactly that case."
        ),
        source=(
            "Salmon P, Young B. Creativity in clinical communication: from communication skills "
            "to skilled communication. Med Educ. 2011;45(3):217-226 — the argument that "
            "codified skill lists produce performance rather than responsive speech. See also "
            "de Haes H, Bensing J. Endpoints in medical communication research. Patient Educ "
            "Couns. 2009;74(3):287-294."
        ),
        question="Would a skilled clinician actually say this, in these words, to this patient?",
        anchor_1=(
            "Nobody talks like this. 'I acknowledge your emotional state. Naming: you are "
            "experiencing fear. Understanding: this is understandable.'"
        ),
        anchor_3=(
            "Fluent but generic; could be pasted into any conversation. 'That sounds really "
            "hard. I can only imagine what you're going through. Let's talk about next steps.'"
        ),
        anchor_5=(
            "Specific to this patient and unrepeatable elsewhere. 'Up all night. And your "
            "mother's history sitting right on top of it — no wonder. Tell me which part is "
            "loudest right now and we'll start there.'"
        ),
    ),
    Dimension(
        key="ritualistic",
        label="Ritualistic (REVERSE-CODED — 5 is the worst score)",
        framework=Framework.SECONDARY,
        polarity=Polarity.LOWER_IS_BETTER,
        definition=(
            "How far the response is a performance of a communication framework rather than a "
            "reply to a person. The markers are: framework vocabulary or labels visible in the "
            "output ('Naming:', 'NURSE', 'Invest in the beginning'); templated scaffolding "
            "such as headed sections or one bullet per empathic move; stock empathy stems "
            "stacked in sequence; and wording that would be equally applicable to any patient "
            "with any problem.\n\n"
            "REVERSE-CODED. 1 = no ritual at all. 5 = a script with the framework showing. "
            "Higher is worse. A response can legitimately score 4-5 on the NURSE dimensions and "
            "5 here at the same time; that is not a contradiction, it is the predicted failure "
            "mode of Condition B and the reason this dimension exists."
        ),
        source=(
            "Salmon & Young 2011 (ritualised performance of communication-skills lists); Salmon "
            "P, Young B. A new paradigm for clinical communication: critical review of "
            "literature in cancer care. Med Educ. 2017;51(3):258-268. It is also a stated "
            "project constraint: CARELite is explicitly not a script generator, on the grounds "
            "that frameworks which become scripts stop working."
        ),
        question=(
            "How much does this read as a framework being performed rather than a person being "
            "answered, where 1 = not at all and 5 = entirely?"
        ),
        anchor_1=(
            "No template, no stock stems, no framework vocabulary; the wording could only have "
            "been produced for this patient. 'Up all night. And your mother's history sitting "
            "right on top of it — no wonder. Tell me which part is loudest right now.'"
        ),
        anchor_3=(
            "Stock empathy stems in sequence, no visible scaffolding, interchangeable across "
            "patients. 'I hear you. That must be really difficult. I want you to know we're "
            "here for you. Let's talk about next steps.'"
        ),
        anchor_5=(
            "The framework is on the page. '**Naming:** It sounds like you're feeling anxious. "
            "**Understanding:** It is completely understandable that you would feel this way. "
            "**Respecting:** I want to acknowledge your strength in coming in today. "
            "**Supporting:** Please know that we are here for you. **Exploring:** Can you tell "
            "me more about how you're feeling?'"
        ),
    ),
)

#: Every dimension, keyed and ordered exactly as `RUBRIC_DIMENSIONS`.
DIMENSIONS: Mapping[str, Dimension] = MappingProxyType({d.key: d for d in _DIMENSIONS})

#: The dimensions where a HIGHER raw score is a WORSE response. Currently
#: `{"ritualistic"}`. Every aggregation must consult this set.
REVERSE_CODED: frozenset[str] = frozenset(d.key for d in _DIMENSIONS if d.reverse_coded)


def dimension(key: str) -> Dimension:
    """Look up one dimension, with a useful error for a typo'd key."""
    try:
        return DIMENSIONS[key]
    except KeyError:
        raise KeyError(
            f"{key!r} is not a rubric dimension; expected one of {RUBRIC_DIMENSIONS}"
        ) from None


def is_reverse_coded(key: str) -> bool:
    """True if a higher raw score on `key` means a worse response."""
    return dimension(key).reverse_coded


def to_quality(key: str, raw: int) -> int:
    """Put one raw score onto the common higher-is-better quality scale.

    Reverse-coded dimensions are flipped with ``SCALE_MIN + SCALE_MAX - raw``
    (i.e. ``6 - raw``); every other dimension passes through unchanged.

    Call this before any mean, sum, correlation, ranking or effect size that
    mixes dimensions. Skipping it inverts the study's naturalness finding
    without producing anything that looks wrong.

        >>> to_quality("ritualistic", 5)   # a scripted response
        1
        >>> to_quality("ritualistic", 1)   # a response with no ritual
        5
        >>> to_quality("de", 5)
        5
    """
    if not SCALE_MIN <= raw <= SCALE_MAX:
        raise ValueError(f"{key} score {raw} is outside the {SCALE_MIN}-{SCALE_MAX} rubric scale")
    if is_reverse_coded(key):
        return SCALE_MIN + SCALE_MAX - raw
    return raw


def to_quality_scores(scores: Mapping[str, int | None]) -> dict[str, int | None]:
    """`to_quality` applied across a whole rating, preserving `None` for unscored.

    Accepts a dict of dimension key -> raw score, such as
    ``RubricScore.model_dump()`` filtered to `RUBRIC_DIMENSIONS`.
    """
    return {
        key: (None if value is None else to_quality(key, value))
        for key, value in scores.items()
        if key in DIMENSIONS
    }
