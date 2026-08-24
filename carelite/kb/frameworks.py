"""Map a knowledge base entry to the NURSE and Four Habits components it instantiates.

This is the behavior-to-framework mapping that build plan v3 names as a
deliverable and that `README.md` describes as one of the two structures the
project hangs off. It fills `kb_entry.nurse_component` and
`kb_entry.four_habits`, which the graph lane turns into
`entry --instantiates--> component` edges and which connect an entry to the
eleven dimensions the judge scores and the statistics analyse.

**The mapping is derived from the entry, not asserted over it.** Two rules make
that real rather than aspirational.

*It reads the act, not the topic.* The matched surface is the entry's
`practical_takeaway` and `example_behavior` — what a clinician is told to do —
and never the `finding` or the `verbatim_span`. Those two say what a study
measured, and a study about empathy can perfectly well support an entry whose
prescribed act is a comprehension check. Matching the finding would assign
components by subject matter, which is exactly the "plausible-looking" mapping
this module has to avoid.

*It follows the rubric's definitions, not the component's name.* Every pattern
below is anchored to the distinguishing act in `carelite.eval.rubric.dimensions`,
which is the same text the judge scores against, and the docstring for each
component quotes the part of the definition the pattern is built on. Several
components are neighbours that a name-level reading would merge, and the rubric
is explicit about the boundaries:

- `epp` asks for the **patient's** model; checking whether the patient
  understood the **clinician's** model is `ie`. The rubric says so in as many
  words.
- `explore` is an open invitation to say more **about the emotion**; a closing
  "any other questions?" is a pro-forma check and belongs to `ie`; a clinical
  history question is not an exploring move at all.
- `de` is a holistic judgement and is explicitly *not* the average of the NURSE
  items, so it is matched only where the prescribed act is itself an empathic
  response — not wherever the word "empathy" appears.

**Coverage gaps are results and must survive.** A knowledge base where all nine
components have entries looks better than one where some have none, and that is
precisely the pressure to resist: the counts say which parts of the rubric this
corpus can and cannot ground. If an entry instantiates none of the nine, it gets
empty lists and the coverage report says so. No pattern in this module was added
in order to give a component its first entry, and `unmapped_entries` exists so
the size of the unmapped set stays visible instead of being tuned away.

Nothing here calls a model. The mapping is a deterministic function of text the
pipeline already holds, so it re-derives identically on every load and a reader
can check any single assignment by reading the entry beside the pattern.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from carelite.types import RUBRIC_DIMENSIONS

#: The five NURSE components, in the frozen `RUBRIC_DIMENSIONS` order.
NURSE_COMPONENTS: tuple[str, ...] = RUBRIC_DIMENSIONS[:5]

#: The four Four Habits components, likewise.
FOUR_HABITS_COMPONENTS: tuple[str, ...] = RUBRIC_DIMENSIONS[5:9]


def _pattern(*alternatives: str) -> re.Pattern[str]:
    return re.compile("|".join(alternatives), re.IGNORECASE)


# ---------------------------------------------------------------------------
# NURSE
# ---------------------------------------------------------------------------

_EMOTION = (
    r"emotion\w*|feeling\w*|fear\w*|anxiet\w*|anxious|distress(ed)?\b|anger|angry|"
    r"sad(ness)?|grief|griev\w*|frustrat\w*|worr(y|ies|ied)|upset|afraid|scared"
)

#: "The response explicitly identifies the emotion the patient is expressing."
#: The act is naming or labelling, and its object has to be an emotion — an
#: entry that says "acknowledge the patient's concerns about cost" is not a
#: naming move, so the object list is emotion words rather than "concern".
_NAME = _pattern(
    rf"\b(nam(e|ing)|label(l?ing|s)?|identif\w+|acknowledg\w+|reflect\w*|articulat\w+)\b"
    rf"[^.]{{0,40}}\b({_EMOTION})\b",
    rf"\b({_EMOTION})\b[^.]{{0,30}}\b(out loud|explicitly|by name)\b",
)

#: "Legitimises the emotion by showing why it makes sense … a bare 'I
#: understand' fails." The distinguishing act is validation or normalisation,
#: so those verbs carry the pattern rather than the word "understand", which in
#: this corpus almost always refers to the *patient's* comprehension and is a
#: different component entirely.
_UNDERSTAND = _pattern(
    r"\b(validat\w+|legitimis\w+|legitimiz\w+|normalis\w+|normaliz\w+)\b",
    r"\bmakes?\s+sense\b",
    r"\b(show|convey|demonstrat)\w*\b[^.]{0,30}\bwhy\b[^.]{0,30}\b(feel|react|respon)\w*",
)

#: "Explicitly acknowledges something the patient has done, endured or managed
#: … it scores on specificity." A praise or credit move, not general courtesy.
#: `affirm\w*` was in this pattern and is now not. It matched "verbal
#: affirmations to show the patient you are listening" — a backchannel cue,
#: which is a different act from crediting the patient for something they did.
#: It was the only thing giving `respect` any entries at all, and removing it
#: takes the component to zero. That is the correct answer: nothing in this
#: corpus prescribes the NURSE Respecting move.
_RESPECT = _pattern(
    r"\b(praise|commend|credit|appreciat\w+|honou?r)\b",
    r"\b(acknowledg\w+|recogni[sz]\w+|nam(e|ing))\b[^.]{0,45}"
    r"\b(effort|strength|courage|resilien\w*|coping|persever\w*|"
    r"what (the patient|they) (has|have) (done|managed|endured)|how hard)\b",
)

#: "States partnership and makes it concrete: who will do what, by when, and how
#: the patient reaches someone before then." A partnership claim with no
#: follow-through is excluded by requiring the concrete half.
#: The first version matched a bare "partnership" or "work together", against
#: this module's own stated rule, and pulled in four motivational-interviewing
#: and shared-decision entries that state a collaborative *stance* with no
#: follow-through at all. Every alternative now carries the concrete half the
#: rubric requires.
_SUPPORT = _pattern(
    r"\b(partnership|work(ing)? together|we (will|can))\b[^.]{0,60}"
    r"\b(follow[- ]?up|next (visit|appointment|step)|check (back|in)|call|contact|reach)\b",
    r"\b(arrang\w+|schedul\w+|offer|book)\w*\b[^.]{0,30}\b(follow[- ]?up|next (visit|appointment))\b",
    r"\b(tell|give|show|explain to)\w*\b[^.]{0,40}\bhow to (reach|contact|get hold of|call)\b",
    r"\b(state|commit to|make)\w*\b[^.]{0,25}\b(availability|a commitment|a plan to check)\b",
    r"\byou are not alone\b",
)

#: "Invites the patient to say more about the emotion or concern, with an open
#: question that follows the patient's own words." Guarded twice: the invitation
#: has to be open, and a closing pro-forma check is `ie`, not this.
#: A bare "use open-ended questions" is not enough. The rubric's question is
#: "does the response open a door for the patient to say more **about the
#: emotion**", and an open question about goals or preferences is Elicit the
#: Patient's Perspective instead. The unqualified `open-ended question`
#: alternative was assigning four entries here that belong to `epp` or `ib`.
_EXPLORE = _pattern(
    rf"\bopen[- ]ended question\w*[^.]{{0,60}}\b(concern\w*|{_EMOTION})\b",
    rf"\b(concern\w*|{_EMOTION})\b[^.]{{0,40}}\bopen[- ]ended question\w*",
    r"\b(invit\w+|encourag\w+|prompt\w*|allow\w*)\b[^.]{0,45}"
    r"\b(say more|elaborate|share more|tell you more|expand)\b",
    rf"\b(ask|explor\w+|elicit\w*|probe)\w*\b[^.]{{0,45}}\b(more about|what else is|"
    rf"concern\w*|{_EMOTION})\b",
)

#: A closing comprehension or "anything else?" check. Not a component on its
#: own — it exists to keep `_EXPLORE` off the pro-forma invitation the rubric
#: explicitly scores as a low `explore`, and to route it to `ie` instead.
_CLOSING_CHECK = _pattern(
    r"\b(any (other |remaining |further )?(questions|concerns)|anything else)\b",
    r"\bbefore (ending|concluding|leaving|you (leave|finish|end))\b",
)


# ---------------------------------------------------------------------------
# Four Habits
# ---------------------------------------------------------------------------

#: "Creates rapport quickly, elicits the patient's full set of concerns before
#: working on any one of them, and plans the conversation." The signature
#: failure is solving the first problem and never asking what else was on the
#: list, so the agenda-setting and opening moves carry the pattern.
_IB = _pattern(
    r"\bagenda[- ]?(setting|set)?\b",
    r"\b(introduce (yourself|themselves)|greet\w*)\b",
    r"\b(at|from) the (start|beginning|outset)\b",
    r"\bbefore (you |the clinician )?(begin|start|move|prioriti[sz])\w*",
    # A bare "all their concerns" is not an Invest-in-the-Beginning marker: it
    # also appears in "ask if there is anything else on their mind **before
    # ending** the interaction to ensure all concerns are addressed", which is
    # the closing check and belongs to `ie`. A negative lookahead did not fix
    # it, because the closing marker sits *before* the match. The ordering
    # requirement below carries the habit instead, and drops nothing real:
    # "list all their concerns before moving into prioritisation" still matches.
    r"\b(elicit|list|gather|solicit)\w*\b[^.]{0,35}\bconcerns\b[^.]{0,30}"
    r"\b(first|before|up front|at the outset)\b",
    r"\blet the patient (speak|talk|finish|list)\b",
    r"\bwithout interrupt\w*",
)

#: "Asks for the patient's own model: what they think is going on, what they
#: specifically want … and how the problem is affecting their life." The rubric
#: is explicit that checking the patient's comprehension of the *clinician's*
#: model is Invest in the End instead, so comprehension language is excluded by
#: matching on the patient's own beliefs, goals and circumstances.
#: The bare noun-phrase alternative that used to head this pattern — any mention
#: of "the patient's perspective / values / goals" — assigned five entries whose
#: prescribed act is not an eliciting one at all: showing interest while
#: listening, stating your own understanding of the patient's feelings,
#: perspective-taking done silently in the clinician's head. The rubric's
#: question is "does the response **ask** what the patient thinks", so an
#: elicitation verb is now required alongside the patient's-model noun.
_ELICIT = (
    r"elicit|ask|enquir\w*|inquir\w*|explor\w+|invit\w+|solicit|seek|find out|learn|identif\w+"
)

_EPP = _pattern(
    r"\bwhat matters (most )?to (you|them|the patient)\b",
    rf"\b({_ELICIT})\w*\b[^.]{{0,50}}\b(the )?(patient'?s|their|his|her) (own )?"
    r"(perspective|point of view|view of|belief\w*|idea\w*|theor\w+|explanation of|"
    r"goal\w*|preference\w*|priorit\w+|value\w*|narrative|story|circumstance\w*|"
    r"motivation\w*|input|opinion\w*|wish\w*)\b",
    rf"\b(the )?(patient'?s|their|his|her) (own )?"
    r"(perspective|goal\w*|preference\w*|value\w*|motivation\w*|input|opinion\w*|wish\w*)"
    rf"\b[^.]{{0,40}}\b({_ELICIT})\w*\b",
    rf"\b({_ELICIT})\w*\b[^.]{{0,45}}"
    r"\b(what (the patient|they) (think|believe|want|hope|expect))\b",
    r"\b(impact|effect|affect\w*|bearing)\b[^.]{0,35}\b(daily life|their life|day[- ]to[- ]day|"
    r"work|family|routine)\b",
    r"\bwhat (actually )?gets in the way\b",
    r"\b(barrier\w*)\b[^.]{0,40}\b(ask|elicit|explor\w+|identif\w+)\w*",
    r"\b(ask|elicit|explor\w+|identif\w+)\w*\b[^.]{0,40}\bbarrier\w*",
)

#: "Whether the response, taken as a whole, lands as empathic … This is
#: deliberately a holistic judgement and it is NOT the average of the NURSE
#: items." Matched only where the prescribed act *is* an empathic response, so
#: the word "empathy" describing a measured outcome cannot pull an entry in.
_DE = _pattern(
    r"\b(respond|reply|answer|react)\w*\b[^.]{0,30}\b(empath\w+|compassion\w*)\b",
    r"\b(empathic|empathetic|compassionate)\b[^.]{0,25}"
    r"\b(response|reply|statement|phrase|phrasing|language|communication|manner|tone)\b",
    r"\b(use|offer|give|convey|express|show|demonstrat)\w*\b[^.]{0,25}\b(empath\w+|compassion\w*)\b",
    r"\bstay (with|present with)\b[^.]{0,25}\b(the )?emotion\b",
    r"\b(acknowledg|address|attend to)\w*\b[^.]{0,40}\bbefore (moving|proceeding|turning|going)\b",
)

#: "Delivers information in a form the patient can carry: a small number of key
#: messages in plain language, an explicit check that they landed, the patient
#: involved in the decision, and a clear next step."
#: Two corrections after reading the assignments. The comprehension check has to
#: be a check on the **patient's** understanding: the unqualified version matched
#: "explicitly state your understanding of the patient's feelings", which is the
#: clinician's understanding and a NURSE move. And a bare "next step" matched an
#: entry about an emotional response "that leads to a specific therapeutic
#: action", so the closure alternatives now name what is actually being closed.
#:
#: The decision-involvement and closing-check alternatives are added on the
#: rubric's own wording, which lists "the patient involved in the decision, and
#: a clear next step" as part of this habit alongside the plain-language and
#: comprehension halves. That is the definition talking, not a coverage gap
#: being filled — `ie` was already the largest component before they were added.
_IE = _pattern(
    r"\bteach[- ]?back\b",
    r"\b(check|confirm|verif\w+|assess|ensure|test)\w*\b[^.]{0,45}"
    r"\b(the )?(patient'?s?|they|their|parent'?s?|famil\w+)\b[^.]{0,25}"
    r"\b(understand\w*|comprehen\w+|recall|retain\w*|grasp)\b",
    r"\b(patient|they|parent|famil\w+)[^.]{0,20}\b(understand\w*|comprehen\w+|recall\w*)\b"
    r"[^.]{0,40}\b(check|confirm|verif\w+|ensure|assess)\w*",
    r"\b(explain|state|repeat|say|describe|summari[sz]e)\w*\b[^.]{0,30}\bback\b",
    r"\bin (their|his|her) own words\b",
    r"\b(plain|simple|simplified|direct|non-?technical|everyday|clear) language\b",
    r"\b(avoid|replace|reduce|limit)\w*\b[^.]{0,30}\b(jargon|medical term\w*|technical term\w*|"
    r"abstract term\w*|euphemism\w*|metaphor\w*)\b",
    r"\bkey (message|point)s? (of|for)? ?(the )?(medical )?(advice|plan|information)\b",
    r"\bre-?explain\w*",
    r"\b(written (instruction|information|summary)|write (it |them )?down)\b",
    # "the patient involved in the decision" — the rubric's own words.
    r"\b(involv\w+|includ\w+|engag\w+|bring)\w*\b[^.]{0,45}"
    r"\b(in|into) (their|the) (treatment |care |clinical )?(decision|choice|plan)\w*",
    r"\b(shared? decision[- ]making|share decision[- ]making power)\b",
    r"\b(present|offer|lay out|explain)\w*\b[^.]{0,35}\b(options|alternatives)\b",
    # the closing check the rubric routes here rather than to Exploring
    r"\b(any (other |remaining |further )?(questions|concerns)|anything else)\b",
    r"\bbefore (ending|concluding|leaving|you (leave|finish|end))\b",
)


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentMapping:
    """Which framework components an entry instantiates, and on what evidence."""

    nurse: tuple[str, ...]
    four_habits: tuple[str, ...]
    #: component -> the substring that matched, so any assignment can be checked.
    evidence: dict[str, str]

    @property
    def is_empty(self) -> bool:
        return not self.nurse and not self.four_habits


_NURSE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("name", _NAME),
    ("understand", _UNDERSTAND),
    ("respect", _RESPECT),
    ("support", _SUPPORT),
    ("explore", _EXPLORE),
)

_HABIT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ib", _IB),
    ("epp", _EPP),
    ("de", _DE),
    ("ie", _IE),
)


def prescribed_act(practical_takeaway: str, example_behavior: str) -> str:
    """The text the mapping reads: what the clinician is told to do, and nothing else.

    `finding` and `verbatim_span` are deliberately excluded. They record what a
    study measured, and the component an entry instantiates is a property of the
    act it prescribes — an entry drawn from an empathy trial can perfectly well
    prescribe a comprehension check.
    """
    return re.sub(r"\s+", " ", f"{practical_takeaway} {example_behavior}").strip()


def map_components(practical_takeaway: str, example_behavior: str) -> ComponentMapping:
    """Derive the NURSE and Four Habits components an entry instantiates."""
    text = prescribed_act(practical_takeaway, example_behavior)
    evidence: dict[str, str] = {}

    nurse: list[str] = []
    closing_only = _CLOSING_CHECK.search(text) is not None
    for key, pattern in _NURSE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        # The rubric routes a pro-forma "any other questions?" to Invest in the
        # End rather than Exploring. Only suppress when that closing check is
        # the *only* thing the explore pattern found.
        if key == "explore" and closing_only and _EXPLORE.search(text) is match:
            span = match.group(0)
            if _CLOSING_CHECK.search(span):
                continue
        nurse.append(key)
        evidence[key] = match.group(0).strip()

    habits: list[str] = []
    for key, pattern in _HABIT_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        habits.append(key)
        evidence[key] = match.group(0).strip()

    return ComponentMapping(nurse=tuple(nurse), four_habits=tuple(habits), evidence=evidence)


@dataclass(frozen=True)
class MappedEntry:
    """One entry's identity plus its derived components, for reporting."""

    entry_id: str
    theme: str
    practical_takeaway: str
    mapping: ComponentMapping


def map_entries(
    entries: Iterable[tuple[str, str, str, str]],
) -> list[MappedEntry]:
    """Map `(entry_id, theme, practical_takeaway, example_behavior)` tuples."""
    return [
        MappedEntry(
            entry_id=entry_id,
            theme=theme,
            practical_takeaway=takeaway,
            mapping=map_components(takeaway, behavior),
        )
        for entry_id, theme, takeaway, behavior in entries
    ]


def component_counts(mapped: Sequence[MappedEntry]) -> dict[str, int]:
    """Entries instantiating each of the nine components, zeros included.

    Zeros are the point. A component with no entries is a statement about what
    this corpus grounds, and it has to be reportable rather than absent.
    """
    counts = dict.fromkeys(NURSE_COMPONENTS + FOUR_HABITS_COMPONENTS, 0)
    for entry in mapped:
        for key in entry.mapping.nurse + entry.mapping.four_habits:
            counts[key] += 1
    return counts


def unmapped_entries(mapped: Sequence[MappedEntry]) -> list[MappedEntry]:
    """Entries instantiating none of the nine components.

    Kept as a first-class output so the unmapped set stays visible. The
    temptation with a mapping like this is to widen patterns until the number
    reaches zero, which would convert honest gaps into invented coverage.
    """
    return [e for e in mapped if e.mapping.is_empty]


__all__ = [
    "FOUR_HABITS_COMPONENTS",
    "NURSE_COMPONENTS",
    "ComponentMapping",
    "MappedEntry",
    "component_counts",
    "map_components",
    "map_entries",
    "prescribed_act",
    "unmapped_entries",
]
