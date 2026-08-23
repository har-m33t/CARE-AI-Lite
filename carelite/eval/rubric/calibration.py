"""The five-response calibration set (build plan v3 §12).

Raters score these five before they touch a single study response, then the
consensus scores and rationales below are read out and discussed. The point is
not to test the raters; it is to surface the four disagreements that otherwise
show up as noise across sixty ratings:

1. **Naturalness is not quality.** CAL-01 is a blunt, empathy-free information
   dump that is nonetheless entirely natural. It scores 1s on NURSE and 4 on
   naturalness. Raters who let quality bleed into naturalness destroy the one
   secondary outcome this study most expects to be interesting.
2. **Ritual is not the absence of framework adherence.** CAL-02 performs the
   NURSE moves correctly *and* is a script. It scores 4s on the NURSE items
   and 5 — the worst possible score — on `ritualistic`. Raters must not mark
   the NURSE items down because the response feels bad, and must not mark
   `ritualistic` down because the moves are technically correct.
3. **No halo.** CAL-03 is the target response and still scores 2 on `respect`,
   because the respecting move simply is not in it. Dimensions are scored
   independently; a good response is allowed to be missing one.
4. **Warmth is not empathy.** CAL-04 is nothing but warmth and scores 1 on
   `explore`, `ib`, `epp` and `ie`. Emotional temperature is not the construct.

All five answer the same patient turn, so raters compare like with like.

REVERSE CODING: `ritualistic` runs the other way — 5 is the worst score. See
`carelite.eval.rubric.dimensions` and `docs/rubric.md`. The consensus values
below are on that raw, higher-is-worse scale.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from carelite.eval.rubric.dimensions import DIMENSIONS, RUBRIC_VERSION, SCALE_MAX, SCALE_MIN
from carelite.types import RaterType, RubricScore

__all__ = [
    "CALIBRATION_SCENARIO",
    "CALIBRATION_SET",
    "NATURAL_ARCHETYPE_ID",
    "SCRIPT_ARCHETYPE_ID",
    "CalibrationItem",
    "item",
    "validate_calibration_set",
]

#: One scenario for all five responses. Synthetic, as everything in this repo is.
CALIBRATION_SCENARIO = (
    "They said the scan showed a spot on my lung. I've been up all night. My mother went "
    "through this and it was awful. I don't even know what to ask you."
)

#: The two items the reverse-coding direction test pivots on. Referenced by
#: semantic name, never by position, so a renumbering cannot silently invert
#: the assertion.
SCRIPT_ARCHETYPE_ID = "CAL-02"
NATURAL_ARCHETYPE_ID = "CAL-03"


@dataclass(frozen=True, slots=True)
class CalibrationItem:
    """One calibration response with its agreed scores and the reasoning behind them."""

    item_id: str
    archetype: str
    response: str
    #: Raw scores, all eleven dimensions. `ritualistic` is reverse-coded.
    consensus: Mapping[str, int]
    #: Why each score is what it is. One entry per dimension.
    rationales: Mapping[str, str]
    #: Verbatim quotes from `response` supporting the scores that turn on a
    #: specific piece of text. Not every dimension has one — a score of 1 for
    #: an absent move has no positive evidence to quote.
    evidence_spans: Mapping[str, str]
    #: The single thing this item exists to teach.
    teaching_point: str
    #: Where the consensus was genuinely argued, and the rule that settled it.
    disagreements: tuple[str, ...] = ()

    def rubric_score(self, rater_id: str = "consensus") -> RubricScore:
        """The consensus as a `RubricScore` row, for seeding the rating harness."""
        c = self.consensus
        # Written out rather than splatted: the eleven dimensions are a frozen
        # contract, and a typo in a splatted dict would be silently dropped.
        return RubricScore(
            generation_id=self.item_id,
            rater_type=RaterType.HUMAN,
            rater_id=rater_id,
            evidence_spans=dict(self.evidence_spans),
            name=c["name"],
            understand=c["understand"],
            respect=c["respect"],
            support=c["support"],
            explore=c["explore"],
            ib=c["ib"],
            epp=c["epp"],
            de=c["de"],
            ie=c["ie"],
            naturalness=c["naturalness"],
            ritualistic=c["ritualistic"],
        )


# ---------------------------------------------------------------------------
# CAL-01  --  blunt information dump. Condition A's characteristic failure.
# ---------------------------------------------------------------------------

_CAL_01 = CalibrationItem(
    item_id="CAL-01",
    archetype="information_dump",
    response=(
        "A spot on the lung — a pulmonary nodule — is a common finding, and most of them are "
        "benign. The next step is a follow-up CT with contrast in three months to look for "
        "interval growth. If it is stable at twelve months we stop looking. If it grows we would "
        "talk about a biopsy. Do you have any questions?"
    ),
    consensus=MappingProxyType(
        {
            "name": 1,
            "understand": 1,
            "respect": 1,
            "support": 2,
            "explore": 2,
            "ib": 1,
            "epp": 1,
            "de": 1,
            "ie": 2,
            "naturalness": 4,
            "ritualistic": 1,
        }
    ),
    rationales=MappingProxyType(
        {
            "name": "No emotion is named or gestured at. The patient said she was up all night; the response does not register that she said anything at all.",
            "understand": "Worse than absent. 'Most of them are benign' is premature reassurance, which corrects the emotion rather than legitimising it.",
            "respect": "No acknowledgement of anything the patient did.",
            "support": "A plan exists, which is a weak form of holding — but there is no partnership statement, no availability, and no named person. 2, not 1.",
            "explore": "'Do you have any questions?' is a pro-forma closed invitation and nothing more. Anchor 3 exactly, marked down to 2 because it arrives after four dense clinical sentences that make asking harder rather than easier.",
            "ib": "No orientation, no agenda, no acknowledgement that the patient said she does not know what to ask — which was an explicit request for help setting the agenda.",
            "epp": "The clinician's model only. The patient's mother is never mentioned.",
            "de": "The blocking pattern: an emotional cue is visible and the response pivots straight to information. This is the low anchor.",
            "ie": "Next steps are clear and correct, which is real credit. But there is no comprehension check, no involvement in the decision, four key messages rather than three, and 'pulmonary nodule', 'interval growth' and 'biopsy' go unexplained.",
            "naturalness": "This is what a rushed clinician sounds like. It is bad communication and completely natural speech, and those are different questions. 4 rather than 5 only because the em-dash gloss reads slightly written.",
            "ritualistic": "REVERSE-CODED, and the lowest score on the set. No template, no stock stems, no framework vocabulary. There is no ritual here because there is no attempt at the moves at all.",
        }
    ),
    evidence_spans=MappingProxyType(
        {
            "understand": "most of them are benign",
            "explore": "Do you have any questions?",
            "ie": "The next step is a follow-up CT with contrast in three months",
        }
    ),
    teaching_point=(
        "Naturalness and ritual are not measures of quality. A response can be natural, "
        "unritualistic, and still fail every framework dimension. Raters who compress all "
        "eleven dimensions onto one good/bad axis produce data in which the naturalness "
        "comparison cannot be interpreted."
    ),
    disagreements=(
        "explore: 2 vs 3. 'Do you have any questions?' matches anchor 3 verbatim. Consensus rule: "
        "a pro-forma invitation placed where it cannot be taken up scores below the anchor.",
    ),
)


# ---------------------------------------------------------------------------
# CAL-02  --  the framework-labelled script. Condition B's predicted failure.
# ---------------------------------------------------------------------------

_CAL_02 = CalibrationItem(
    item_id="CAL-02",
    archetype="framework_script",
    response=(
        "I want to make sure I address this properly.\n"
        "\n"
        "**Naming:** It sounds like you are feeling anxious and frightened about the scan "
        "results.\n"
        "**Understanding:** It is completely understandable that you would feel this way, "
        "especially given your family history.\n"
        "**Respecting:** I want to acknowledge your strength in coming in to discuss this "
        "today.\n"
        "**Supporting:** Please know that we are here for you and we will get through this "
        "together.\n"
        "**Exploring:** Can you tell me more about what you are feeling right now?\n"
        "\n"
        "Regarding next steps, the standard approach is a follow-up CT scan. Do you have any "
        "questions?"
    ),
    consensus=MappingProxyType(
        {
            "name": 4,
            "understand": 4,
            "respect": 3,
            "support": 3,
            "explore": 4,
            "ib": 2,
            "epp": 3,
            "de": 2,
            "ie": 2,
            "naturalness": 1,
            "ritualistic": 5,
        }
    ),
    rationales=MappingProxyType(
        {
            "name": "The emotion is named, specifically ('anxious and frightened'), tentatively ('it sounds like'), and correctly. That is the move. 4 rather than 5 because it is not tied to anything the patient actually said until the next line.",
            "understand": "Legitimation is present and does reference her stated reason. 4 rather than 5 because 'especially given your family history' summarises her mother into a chart phrase instead of using what she said.",
            "respect": "Generic praise, unearned and non-specific. Anchor 3.",
            "support": "Partnership asserted twice with nothing concrete behind it — no name, no timeframe, no way to reach anyone. Anchor 3.",
            "explore": "A genuinely open invitation is made. 4, held off 5 because it invites her to elaborate on feelings in the abstract rather than on anything she said.",
            "ib": "No agenda is elicited and the opening line is a meta-announcement about the response itself. Her explicit 'I don't know what to ask' goes unanswered.",
            "epp": "She is asked to say more about her feelings, which is one element. Her ideas about what is happening and the impact on her life are never asked for.",
            "de": "This is the score raters argue about. Every empathic move is present and correct, and the response still does not land as empathic: the labels announce that empathy is being administered. Holistic judgement, deliberately not the average of the NURSE items.",
            "ie": "A next step is named and nothing else — no check, no involvement, no plain-language translation.",
            "naturalness": "No clinician has ever said this out loud. Anchor 1.",
            "ritualistic": "REVERSE-CODED, and the worst possible score. The framework is literally on the page: five labelled sections, one per NURSE letter, in mnemonic order. Anchor 5.",
        }
    ),
    evidence_spans=MappingProxyType(
        {
            "name": "It sounds like you are feeling anxious and frightened",
            "understand": "It is completely understandable that you would feel this way",
            "respect": "I want to acknowledge your strength in coming in",
            "support": "Please know that we are here for you",
            "explore": "Can you tell me more about what you are feeling right now?",
            "naturalness": "I want to make sure I address this properly.",
            "ritualistic": "**Naming:**",
        }
    ),
    teaching_point=(
        "High NURSE adherence and maximum ritual coexist in the same response, and this is the "
        "single most important pattern in the study. Build plan v3 predicts Condition B loses to "
        "Condition A on naturalness precisely because framework prompting produces this. If "
        "raters average the two together — marking NURSE down because the response is a script, "
        "or marking `ritualistic` down because the moves are correct — the effect the study is "
        "designed to detect disappears into the rating noise."
    ),
    disagreements=(
        "de: 2 vs 4. One argument was that all four Habit-3 behaviours are present. Consensus "
        "rule: `de` is a holistic judgement of whether the response lands as empathic, and a "
        "labelled performance of empathy does not. The NURSE items carry the behaviour count; "
        "`de` carries the effect.",
        "ritualistic: 5, unanimous, but worth stating aloud during calibration — 5 is the WORST "
        "score on this dimension and the only place in the rubric where that is true.",
    ),
)


# ---------------------------------------------------------------------------
# CAL-03  --  the target. Integrated, specific, no visible framework.
# ---------------------------------------------------------------------------

_CAL_03 = CalibrationItem(
    item_id="CAL-03",
    archetype="integrated_natural",
    response=(
        "Up all night. And your mother's whole story sitting right on top of this one — of "
        "course you didn't sleep.\n"
        "\n"
        "Before I tell you anything about the scan, I want to know what you are most afraid of, "
        "because I would rather answer that than the questions I am guessing at. You said you "
        "don't know what to ask — that is fine, we can start from what scares you and I will add "
        "the two things I think matter.\n"
        "\n"
        "Here is what I can tell you now: it is small, and the next thing is one more picture, "
        "not treatment. Those are the two things I would hold onto. When you tell your husband "
        "tonight, what are you going to say?\n"
        "\n"
        "I will call you myself with the appointment by Thursday. If it gets loud again before "
        "then, call the nurse line and say it is me you need."
    ),
    consensus=MappingProxyType(
        {
            "name": 4,
            "understand": 5,
            "respect": 2,
            "support": 5,
            "explore": 5,
            "ib": 5,
            "epp": 4,
            "de": 5,
            "ie": 5,
            "naturalness": 5,
            "ritualistic": 1,
        }
    ),
    rationales=MappingProxyType(
        {
            "name": "'What you are most afraid of' works with fear without naming it as this patient's emotion. A question about fear is not a naming of it. 4, and the argument for 5 is recorded below.",
            "understand": "'Of course you didn't sleep', tied to the reason she gave. Legitimation that shows its working. Anchor 5.",
            "respect": "The respecting move is simply absent. No credit is given for anything she did. Scored on its own merits at 2 — nothing in this response acknowledges her effort, and the quality of the rest of it is irrelevant to this dimension.",
            "support": "Partnership plus a named person, a named day, and a route to reach someone before then. Anchor 5.",
            "explore": "An open invitation built out of her own words and placed before any information. Anchor 5.",
            "ib": "Rapport, an explicit refusal to start with content, her stated difficulty answered directly, and a conversation plan she can accept or refuse. Anchor 5.",
            "epp": "Her fear is asked for before the clinician's account, which is the hardest part of the habit. 4 rather than 5: her ideas about what is happening and the impact on her life are still not asked for.",
            "de": "Emotion is met first, held for a beat, and the register matches hers — short sentences, her own words back to her. Anchor 5.",
            "ie": "Two key messages, in plain language, explicitly flagged as the two to keep; a genuine teach-back framed as telling her husband rather than as a quiz; and a concrete next step. Anchor 5.",
            "naturalness": "Specific to this patient and unrepeatable for anyone else. Anchor 5.",
            "ritualistic": "REVERSE-CODED, best possible score. No labels, no scaffolding, no stock stems. Every empathic move in it is made out of her own words.",
        }
    ),
    evidence_spans=MappingProxyType(
        {
            "understand": "of course you didn't sleep",
            "support": "I will call you myself with the appointment by Thursday",
            "explore": "I want to know what you are most afraid of",
            "ib": "Before I tell you anything about the scan",
            "de": "Up all night. And your mother's whole story sitting right on top of this one",
            "ie": "When you tell your husband tonight, what are you going to say?",
            "naturalness": "Up all night.",
        }
    ),
    teaching_point=(
        "No halo. This is the best response in the set and it scores 2 on `respect`, because "
        "the respecting move is not in it. Every dimension is scored on its own evidence. A "
        "rater who lets a strong overall impression lift the weak dimensions compresses the "
        "variance the analysis depends on."
    ),
    disagreements=(
        "name: 4 vs 5. The argument for 5 was that 'what you are most afraid of' plainly "
        "engages fear. Consensus rule: Naming requires the response to say what the emotion is, "
        "so the patient can correct it. Asking about an emotion is Exploring, and it is already "
        "scored there.",
        "respect: 2 vs 1. Consensus rule: 1 is reserved for a response that also has no room "
        "for the move; this one takes her seriously throughout without ever crediting her.",
    ),
)


# ---------------------------------------------------------------------------
# CAL-04  --  warm and empty. Ritual without any framework labels.
# ---------------------------------------------------------------------------

_CAL_04 = CalibrationItem(
    item_id="CAL-04",
    archetype="warm_and_empty",
    response=(
        "Oh, I am so sorry. That sounds incredibly hard. I can only imagine what you must be "
        "going through, especially with your mother. Please know that you are not alone in this "
        "and that we are here for you every step of the way. I want you to know that whatever "
        "happens, we will face this together. You are stronger than you know."
    ),
    consensus=MappingProxyType(
        {
            "name": 3,
            "understand": 3,
            "respect": 3,
            "support": 3,
            "explore": 1,
            "ib": 1,
            "epp": 1,
            "de": 3,
            "ie": 1,
            "naturalness": 2,
            "ritualistic": 4,
        }
    ),
    rationales=MappingProxyType(
        {
            "name": "'That sounds incredibly hard' gestures at distress without naming an emotion. Anchor 3 exactly.",
            "understand": "'Especially with your mother' reaches for her reason but does not say what it is or why it follows. Stock legitimation. Anchor 3.",
            "respect": "'You are stronger than you know' is praise, but generic and unearned by anything in the conversation. Anchor 3.",
            "support": "Partnership asserted four separate times and never once made concrete. Anchor 3 — and a useful demonstration that repetition does not raise the score.",
            "explore": "No invitation of any kind. The patient is given nothing to answer.",
            "ib": "No agenda, no orientation, no response to 'I don't know what to ask'.",
            "epp": "Her perspective is never requested.",
            "de": "Warm, and not responsive: the register is overheated relative to hers, and nothing in it could only have been said to her. Fluent sympathy is not the construct.",
            "ie": "No information, no check, no next step. She leaves knowing exactly what she knew before, which is the worst score available here.",
            "naturalness": "Fluent but pitched wrong for a clinic room, and every sentence is portable to any patient with any problem.",
            "ritualistic": "REVERSE-CODED, and high at 4. There are no framework labels and no scaffolding, but eight stock empathy stems are stacked in sequence. Ritual does not require a template — a response can be a script without announcing that it is one. 4 rather than 5 because the framework itself is not visible.",
        }
    ),
    evidence_spans=MappingProxyType(
        {
            "name": "That sounds incredibly hard",
            "understand": "especially with your mother",
            "respect": "You are stronger than you know",
            "support": "we are here for you every step of the way",
            "ritualistic": "I can only imagine what you must be going through",
        }
    ),
    teaching_point=(
        "Empathy dimensions are not a warmth thermometer. This is the warmest response in the "
        "set and it scores 1 on four dimensions. It is also the item that shows `ritualistic` "
        "catching a script with no framework labels in it at all — which is why the deterministic "
        "proxy, which keys on surface markers, under-reads this one."
    ),
    disagreements=(
        "ritualistic: 4 vs 3. Consensus rule: anchor 3 describes stock stems in sequence; this "
        "response is nothing but stock stems, with no content between them, which is worse.",
    ),
)


# ---------------------------------------------------------------------------
# CAL-05  --  genuine, jargon-heavy, mid-range. The hardest item to score.
# ---------------------------------------------------------------------------

_CAL_05 = CalibrationItem(
    item_id="CAL-05",
    archetype="mixed_jargon_heavy",
    response=(
        "That is a rough night, I am sorry. Let me tell you what we actually know, because I "
        "think not knowing is the worst part of this. The nodule is subcentimetre, and on the CT "
        "it has smooth margins with no spiculation, which is the pattern we associate with benign "
        "findings. That puts it in the low-risk category, so the plan is surveillance rather than "
        "biopsy — serial imaging, not intervention. I know that is a lot of words. What is your "
        "understanding of where that leaves you?"
    ),
    consensus=MappingProxyType(
        {
            "name": 2,
            "understand": 3,
            "respect": 1,
            "support": 2,
            "explore": 3,
            "ib": 2,
            "epp": 3,
            "de": 3,
            "ie": 3,
            "naturalness": 4,
            "ritualistic": 1,
        }
    ),
    rationales=MappingProxyType(
        {
            "name": "'That is a rough night' acknowledges the night, not the emotion. Below anchor 3 because it does not gesture at how she feels at all, only at what happened.",
            "understand": "'I think not knowing is the worst part' is a genuine, unformulaic attempt to say why this is hard. It is the clinician's guess rather than her stated reason, so anchor 3.",
            "respect": "Nothing.",
            "support": "A plan of continued surveillance implies ongoing contact, but no partnership is stated and no availability is offered.",
            "explore": "'What is your understanding of where that leaves you?' is open and genuinely invites her to speak. It is aimed at comprehension rather than at the emotion, so it does not reach the top of an emotion-exploring dimension. Anchor 3.",
            "ib": "Opens with a brief acknowledgement and then goes straight to the clinician's agenda. Anchor 3 behaviour, marked down to 2 because her 'I don't know what to ask' is again unanswered.",
            "epp": "Her understanding is asked for, which is one element and the right instinct, but only at the very end and only about the clinician's account.",
            "de": "Real warmth and a real attempt to reduce fear by giving information — the emotion is engaged for one beat and then left. Anchor 3.",
            "ie": "A genuine teach-back is present, which is the strongest thing in this response. It is capped at 3 by what precedes it: five key messages, and 'subcentimetre', 'spiculation', 'margins', 'surveillance' and 'serial imaging' unexplained. 'I know that is a lot of words' shows the clinician noticed and said it anyway.",
            "naturalness": "Reads exactly like a real specialist who is trying. 4.",
            "ritualistic": "REVERSE-CODED, lowest score. No labels, no scaffolding, no stock stems. Jargon-dense is not the same as formulaic.",
        }
    ),
    evidence_spans=MappingProxyType(
        {
            "name": "That is a rough night",
            "understand": "I think not knowing is the worst part of this",
            "explore": "What is your understanding of where that leaves you?",
            "ie": "The nodule is subcentimetre, and on the CT it has smooth margins with no spiculation",
        }
    ),
    teaching_point=(
        "The hardest item, and the one where the deterministic scorers are most visible. "
        "Teach-back is detected mechanically, jargon density is high, message count exceeds "
        "three, and `ie` is capped at 3 for exactly those reasons. When a rater and the "
        "counters disagree here, the disagreement is informative and should be recorded rather "
        "than resolved by deferring to either one."
    ),
    disagreements=(
        "ie: 3 vs 4. The argument for 4 was that a genuine teach-back is rare and should be "
        "rewarded. Consensus rule: Invest in the End covers delivery *and* checking; a check "
        "attached to a jargon wall cannot exceed the midpoint.",
        "explore: 3 vs 4. Consensus rule: an open question aimed at comprehension is Eliciting "
        "the Patient's Perspective, not NURSE Exploring, and it is credited under `epp`.",
    ),
)


#: The five, in presentation order. Rate them in this order: CAL-01 and CAL-02
#: set the two axes that raters most often conflate, and CAL-05 is only
#: discussable once the first four are settled.
CALIBRATION_SET: tuple[CalibrationItem, ...] = (_CAL_01, _CAL_02, _CAL_03, _CAL_04, _CAL_05)

_BY_ID: Mapping[str, CalibrationItem] = MappingProxyType({c.item_id: c for c in CALIBRATION_SET})


def item(item_id: str) -> CalibrationItem:
    """Look up one calibration item by id."""
    try:
        return _BY_ID[item_id]
    except KeyError:
        raise KeyError(f"{item_id!r} is not a calibration item; have {sorted(_BY_ID)}") from None


def validate_calibration_set() -> None:
    """Structural checks on the constants above. Called from the unit tests.

    Verifies that every item scores and explains all eleven dimensions on the
    1-5 scale, and that every evidence span is verbatim in its own response —
    the same grounding rule v3 §13 imposes on the judge, applied to the
    material the judge is calibrated against.

    Raises `ValueError` on the first problem found.
    """
    for c in CALIBRATION_SET:
        missing = set(DIMENSIONS) - set(c.consensus)
        if missing:
            raise ValueError(f"{c.item_id} is missing consensus scores for {sorted(missing)}")
        extra = set(c.consensus) - set(DIMENSIONS)
        if extra:
            raise ValueError(f"{c.item_id} scores unknown dimensions {sorted(extra)}")
        if set(c.rationales) != set(DIMENSIONS):
            raise ValueError(f"{c.item_id} rationales do not cover exactly the eleven dimensions")
        for key, value in c.consensus.items():
            if not SCALE_MIN <= value <= SCALE_MAX:
                raise ValueError(f"{c.item_id}.{key} = {value} is off the rubric scale")
        for key, span in c.evidence_spans.items():
            if key not in DIMENSIONS:
                raise ValueError(f"{c.item_id} has an evidence span for unknown dimension {key!r}")
            if span not in c.response:
                raise ValueError(f"{c.item_id}.{key} evidence span is not verbatim: {span!r}")


#: The rubric version these consensus scores were agreed under. If the rubric
#: changes, the set is re-scored rather than carried forward.
CALIBRATED_AGAINST = RUBRIC_VERSION
