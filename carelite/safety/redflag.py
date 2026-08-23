"""Clinical red-flag detection. **Tuned for recall, deliberately.**

CARELite coaches communication. It has no business coaching a clinician through
a patient who has just said "I want to kill myself" or "my chest feels like an
elephant is sitting on it" — those turns need escalation, not a suggested
reframe. This module decides which turns leave the coaching path.

The asymmetry is total. A false negative here means the system smoothly coached
someone past an emergency. A false positive means the operator sees an
escalation banner on a turn that did not need one and dismisses it. So every
design choice below is made in favour of recall:

* **No negation suppression by default.** "I don't have chest pain" flags.
  Negation detection is exactly the kind of thing that works on the examples you
  wrote it for and fails on "no chest pain but my arm is numb". `screen()`
  accepts `negation_aware=True` for anyone who wants to measure the trade, and
  the security corpus reports the false-positive rate both ways — but the
  default, and the value the wave-3 gate is measured at, is off.
* **Obfuscation-tolerant matching.** Phrases are matched against the
  punctuation-tolerant, letter-spacing-tolerant, leetspeak-tolerant forms from
  `normalize`, so "k i l l  m y s e l f" and "ch3st p@in" both flag.
* **Category breadth over precision.** Explicit requests for emergency care
  ("should I call 911?") flag, because a turn that is asking about escalation is
  a turn that should escalate.

`screen()` returns `allowed=False`: the red-flagged turn must not proceed as
normal coaching. It is not an injection and not PHI, so nothing is redacted —
`redacted_text` stays `None` and the caller shows `escalation_message()`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from carelite.safety.normalize import compile_phrases, detection_forms, squeezed_forms
from carelite.types import SafetyVerdict

#: Squeezed matching drops word boundaries, so short phrases would match inside
#: unrelated words ("die" inside "diet"). Only phrases at least this long, with
#: separators removed, are checked in the squeezed form.
_MIN_SQUEEZE_LEN = 8


@dataclass(frozen=True)
class Category:
    name: str
    urgency: str
    phrases: tuple[str, ...] = ()
    regexes: tuple[str, ...] = ()
    guidance: str = ""
    _compiled: list[re.Pattern[str]] = field(default_factory=list, compare=False, repr=False)


CATEGORIES: tuple[Category, ...] = (
    Category(
        name="suicidal_ideation",
        urgency="immediate",
        guidance="Stop coaching. Follow local suicide-risk protocol; 988 in the US.",
        phrases=(
            "kill myself",
            "killing myself",
            "kill my self",
            "end my life",
            "ending my life",
            "take my own life",
            "took my own life",
            "end it all",
            "want to die",
            "wanna die",
            "want to be dead",
            "wish i was dead",
            "wish i were dead",
            "wish i had died",
            "better off dead",
            "better off without me",
            "no reason to live",
            "nothing to live for",
            "not worth living",
            "life is not worth",
            "dont want to live",
            "do not want to live",
            "dont want to be here anymore",
            "cant go on",
            "cannot go on",
            "cant do this anymore",
            "give up on life",
            "hurt myself",
            "harm myself",
            "harming myself",
            "self harm",
            "cut myself",
            "cutting myself",
            "take all my pills",
            "take all of my pills",
            "swallow all my pills",
            "took all my pills",
            "overdose on purpose",
            "stockpiling my pills",
            "saving up my pills",
            "wrote a note to my family",
            "made a plan to end",
            "put my affairs in order",
            "everyone would be better off",
            "just want it to stop forever",
        ),
        regexes=(
            r"\bsu+ic+id(e|al)?\b",
            r"\bsuicid\w*\b",
            r"\bsucid\w*\b",
            r"\bkill\w*\s+(my\s?self|me)\b",
            r"\bwant\w*\s+to\s+(die|be\s+dead|disappear\s+forever)\b",
            r"\b(thoughts?|thinking)\s+(of|about)\s+(dying|death|ending\s+it|hurting\s+myself)\b",
            r"\bself[-\s]?(harm|injur\w+)\b",
        ),
    ),
    Category(
        name="homicidal_or_violence_risk",
        urgency="immediate",
        guidance="Stop coaching. Risk to a third party; follow duty-to-warn and safety protocol.",
        phrases=(
            "hurt someone else",
            "hurt somebody",
            "kill him",
            "kill her",
            "kill them",
            "make them pay",
            "bring my gun",
            "shoot someone",
            "shoot them",
            "hurt my kids",
            "hurt my children",
        ),
        regexes=(
            r"\b(kill|shoot|stab|hurt)\s+(him|her|them|someone|somebody|everyone|my\s+\w+)\b",
        ),
    ),
    Category(
        name="interpersonal_violence",
        urgency="urgent",
        guidance="Stop coaching. Screen for safety privately; do not proceed with a companion present.",
        phrases=(
            "he hits me",
            "she hits me",
            "they hit me",
            "my husband hits me",
            "my wife hits me",
            "my partner hits me",
            "my boyfriend hits me",
            "my girlfriend hits me",
            "he threatens me",
            "not safe at home",
            "afraid to go home",
            "scared of my husband",
            "scared of my partner",
            "he took my medication away",
            "wont let me leave the house",
            "hurts me when he drinks",
        ),
        regexes=(r"\b(hits|beats|chokes|threatens)\s+me\b",),
    ),
    Category(
        name="cardiac_chest_pain",
        urgency="immediate",
        guidance="Stop coaching. Possible acute coronary syndrome; activate emergency care.",
        phrases=(
            "chest pain",
            "chest pains",
            "pain in my chest",
            "chest pressure",
            "pressure in my chest",
            "tightness in my chest",
            "chest tightness",
            "tight in my chest",
            "chest feels tight",
            "chest is heavy",
            "heaviness in my chest",
            "crushing pain",
            "elephant on my chest",
            "elephant sitting on my chest",
            "band around my chest",
            "heart attack",
            "having a heart attack",
            "pain radiating",
            "radiating to my arm",
            "radiating to my jaw",
            "down my left arm",
            "into my left arm",
            "my jaw and my arm",
            "pain in my jaw and",
            "sweating and my chest",
            "clammy and short of breath",
            "heart is pounding and i feel",
        ),
        regexes=(
            r"\bchest\b[^.\n]{0,25}\b(pain|pressure|tight\w*|heav\w+|squeez\w+|crush\w+|"
            r"burn\w+|ach\w+|hurt\w*)\b",
            r"\b(pain|pressure|tight\w*|ach\w+)\b[^.\n]{0,20}\b(in|on|across)\s+my\s+chest\b",
            r"\bheart\s+attack\b",
            r"\bangina\b",
            r"\b(elephant|weight|truck|brick|band|vice)\b[^.\n]{0,25}\b(chest|ribs)\b",
        ),
    ),
    Category(
        name="stroke",
        urgency="immediate",
        guidance="Stop coaching. Possible stroke; time-critical, activate emergency care.",
        phrases=(
            "face is drooping",
            "face drooping",
            "droopy face",
            "one side of my face",
            "half my face",
            "slurred speech",
            "slurring my words",
            "words come out wrong",
            "cant find my words",
            "trouble speaking",
            "cant speak properly",
            "weakness on one side",
            "weak on one side",
            "one side of my body",
            "numb on one side",
            "cant move my arm",
            "cant move my leg",
            "cant lift my arm",
            "arm went dead",
            "sudden vision loss",
            "lost my vision",
            "went blind in one eye",
            "sudden double vision",
            "worst headache of my life",
            "worst headache ive ever",
            "thunderclap headache",
            "sudden confusion",
            "having a stroke",
            "mini stroke",
        ),
        regexes=(
            r"\bstroke\b",
            r"\b(sudden\w*)\s+\w{0,10}\s*(weakness|numbness|confusion|blindness)\b",
            r"\bnumb\w*\b[^.\n]{0,20}\b(one\s+side|left\s+side|right\s+side|face|arm)\b",
            r"\b(one|left|right|whole|other)\s+side\b[^.\n]{0,30}"
            r"\b(numb\w*|weak\w*|paralys\w*|dead|limp|drop\w*|gave\s+out)\b",
            r"\bslur\w*\b[^.\n]{0,20}\b(speech|words|speaking|talk\w*)\b",
            r"\b(speech|words|talking)\b[^.\n]{0,20}\bslur\w*\b",
        ),
    ),
    Category(
        name="anaphylaxis_or_airway",
        urgency="immediate",
        guidance="Stop coaching. Possible airway compromise or anaphylaxis; activate emergency care.",
        phrases=(
            "throat is closing",
            "throat closing",
            "throat is tightening",
            "throat feels tight",
            "cant breathe",
            "cannot breathe",
            "can hardly breathe",
            "hard to breathe",
            "trouble breathing",
            "struggling to breathe",
            "gasping for air",
            "cant catch my breath",
            "short of breath",
            "out of breath just",
            "lips are swelling",
            "tongue is swelling",
            "tongue feels swollen",
            "face is swelling",
            "hives all over",
            "covered in hives",
            "allergic reaction",
            "anaphylaxis",
            "used my epipen",
            "need my epipen",
            "wheezing badly",
            "lips are blue",
            "turning blue",
        ),
        regexes=(
            r"\bepi\s?pen\b",
            r"\banaphyla\w+\b",
            r"\b(swell\w+|swollen)\b[^.\n]{0,20}\b(throat|tongue|lips?|face|airway)\b",
            r"\bcan\s?no?t\b[^.\n]{0,12}\bbreath\w*\b",
        ),
    ),
    Category(
        name="sepsis_or_severe_infection",
        urgency="immediate",
        guidance="Stop coaching. Possible sepsis; activate emergency care.",
        phrases=(
            "high fever",
            "fever of 103",
            "fever of 104",
            "burning up",
            "shaking chills",
            "uncontrollable shivering",
            "cant stop shivering",
            "rigors",
            "confused and feverish",
            "fever and confused",
            "red streaks",
            "wound is hot and red",
            "wound smells",
            "fever after surgery",
            "fever and my heart is racing",
            "sepsis",
            "septic",
            "not making urine",
            "havent peed all day",
        ),
        regexes=(
            r"\bsep(sis|tic)\b",
            r"\bfever\b[^.\n]{0,40}\b(confus\w+|disorient\w+|drows\w+|shak\w+|chills|"
            r"racing|faint\w*|breath\w*)\b",
            r"\b(confus\w+|disorient\w+|shak\w+|chills)\b[^.\n]{0,40}\bfever\b",
            r"\b10[3-9](\.\d)?\s*(degrees|°|f\b)",
        ),
    ),
    Category(
        name="respiratory_distress",
        urgency="immediate",
        guidance="Stop coaching. Respiratory distress; activate emergency care.",
        phrases=(
            "cant get enough air",
            "gasping",
            "breathing so fast",
            "oxygen is low",
            "sats are low",
            "using my inhaler every",
            "inhaler isnt working",
            "inhaler is not working",
            "nebulizer isnt helping",
        ),
        regexes=(r"\bo2\s+(sat\w*|level)\b[^.\n]{0,15}\b(low|8\d|7\d)\b",),
    ),
    Category(
        name="hemorrhage",
        urgency="immediate",
        guidance="Stop coaching. Active or occult bleeding; activate emergency care.",
        phrases=(
            "bleeding wont stop",
            "wont stop bleeding",
            "bleeding that wont stop",
            "vomiting blood",
            "throwing up blood",
            "coughing up blood",
            "spitting up blood",
            "blood in my stool",
            "blood in my urine",
            "blood in my vomit",
            "black tarry stool",
            "tarry stools",
            "black stools",
            "soaking through a pad",
            "bleeding heavily",
            "lost a lot of blood",
            "hemorrhage",
            "haemorrhage",
        ),
        regexes=(
            r"\bbleed\w*\b[^.\n]{0,25}\b(wont|will\s+not|cant|can\s?not|doesnt|does\s+not)\s+stop\b",
            r"\b(vomit\w*|cough\w*|throw\w*\s+up|spit\w*)\b[^.\n]{0,15}\bblood\b",
            r"\bblood\b[^.\n]{0,15}\b(stool|urine|vomit|phlegm|sputum)\b",
            r"\btarry\b",
            r"\bstools?\b[^.\n]{0,30}\bblack\b",
            r"\bblack\b[^.\n]{0,20}\bstools?\b",
        ),
    ),
    Category(
        name="altered_consciousness",
        urgency="immediate",
        guidance="Stop coaching. Loss of consciousness or seizure; activate emergency care.",
        phrases=(
            "passed out",
            "blacked out",
            "fainted",
            "lost consciousness",
            "unresponsive",
            "wouldnt wake up",
            "cant wake him up",
            "cant wake her up",
            "had a seizure",
            "having seizures",
            "convulsions",
            "not making sense",
            "keeps falling asleep and i cant",
        ),
        regexes=(
            r"\bseizure?s?\b",
            r"\bconvuls\w+\b",
            r"\bsyncope\b",
            r"\b(cant|cannot|couldnt|could\s+not|unable\s+to|wouldnt|would\s+not)\b"
            r"[^.\n]{0,15}\bwake\b",
        ),
    ),
    Category(
        name="overdose_or_poisoning",
        urgency="immediate",
        guidance="Stop coaching. Possible overdose or poisoning; activate emergency care and poison control.",
        phrases=(
            "took too many pills",
            "took too much",
            "double dosed",
            "doubled up on my",
            "swallowed bleach",
            "drank bleach",
            "poisoned",
            "took the whole bottle",
            "took a handful of",
        ),
        regexes=(r"\bover\s?dos\w+\b",),
    ),
    Category(
        name="obstetric_emergency",
        urgency="immediate",
        guidance="Stop coaching. Obstetric emergency; activate emergency care.",
        phrases=(
            "baby hasnt moved",
            "baby has not moved",
            "not feeling the baby move",
            "havent felt the baby",
            "my water broke",
            "bleeding and im pregnant",
            "pregnant and bleeding",
            "severe headache and pregnant",
            "seeing spots and pregnant",
            "preeclampsia",
            "contractions and im only",
        ),
        regexes=(
            r"\bpre\s?eclamps\w+\b",
            r"\bpregnan\w*\b[^.\n]{0,50}\b(severe\s+headache|bleed\w*|seeing\s+spots|"
            r"blurry\s+vision|sudden\s+swelling|no\s+movement|hasnt\s+moved)\b",
            r"\b(severe\s+headache|bleed\w*|seeing\s+spots|blurry\s+vision|hasnt\s+moved)\b"
            r"[^.\n]{0,50}\bpregnan\w*\b",
        ),
    ),
    Category(
        name="severe_pain",
        urgency="urgent",
        guidance="Stop coaching. Undifferentiated severe or sudden pain; needs clinical assessment now.",
        phrases=(
            "worst pain of my life",
            "worst pain ive ever",
            "10 out of 10 pain",
            "ten out of ten pain",
            "sudden severe pain",
            "ripping pain",
            "tearing pain",
            "rigid abdomen",
            "stomach is hard",
            "cant stand up from the pain",
            "pain is unbearable",
            "unbearable pain",
        ),
    ),
    Category(
        name="explicit_emergency_request",
        urgency="immediate",
        guidance="Stop coaching. The patient is asking whether to escalate; treat that as escalation.",
        phrases=(
            "call 911",
            "should i call 911",
            "should i go to the er",
            "should i go to the emergency room",
            "go to the emergency room",
            "need an ambulance",
            "call an ambulance",
            "is this an emergency",
            "i think im dying",
            "i feel like im dying",
            "something is really wrong",
        ),
        regexes=(r"\b911\b", r"\bemergency\s+room\b", r"\bambulance\b"),
    ),
)


def _compile(cat: Category) -> list[re.Pattern[str]]:
    if not cat._compiled:
        pats: list[re.Pattern[str]] = []
        if cat.phrases:
            pats.append(compile_phrases(cat.phrases))
        pats.extend(re.compile(r, re.IGNORECASE) for r in cat.regexes)
        cat._compiled.extend(pats)
    return cat._compiled


_SQUEEZED: dict[str, tuple[str, ...]] = {
    cat.name: tuple(
        sq
        for p in cat.phrases
        if len(sq := re.sub(r"[^a-z0-9]", "", p.lower())) >= _MIN_SQUEEZE_LEN
    )
    for cat in CATEGORIES
}

# Only explicit first-person denial of a symptom, and only for the reported-speech
# constructions that actually show up in a coaching transcript. Opt-in.
_NEGATION_RE = re.compile(
    r"\b(no|not|never|denies|denied|without|dont\s+have|do\s+not\s+have|didnt\s+have|"
    r"havent\s+had|isnt|wasnt|hasnt)\b[^.!?\n]{0,40}$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RedFlagHit:
    category: str
    urgency: str
    span: str
    guidance: str


def find_red_flags(text: str, *, negation_aware: bool = False) -> list[RedFlagHit]:
    """Every red-flag category triggered by `text`.

    Args:
        text: the utterance. Untrusted; normalised internally.
        negation_aware: if True, drop a hit whose match is immediately preceded
            by an explicit denial ("I don't have chest pain"). **Off by default**
            — see the module docstring for why.
    """
    forms = detection_forms(text)
    squeezed = squeezed_forms(text)
    hits: list[RedFlagHit] = []

    for cat in CATEGORIES:
        span: str | None = None
        for pattern in _compile(cat):
            for form in forms:
                m = pattern.search(form)
                if not m:
                    continue
                if negation_aware and _NEGATION_RE.search(form[: m.start()]):
                    continue
                span = m.group(0)
                break
            if span:
                break
        if span is None:
            for needle in _SQUEEZED[cat.name]:
                if any(needle in s for s in squeezed):
                    span = needle
                    break
        if span is not None:
            hits.append(
                RedFlagHit(category=cat.name, urgency=cat.urgency, span=span, guidance=cat.guidance)
            )
    return hits


def is_red_flag(text: str, *, negation_aware: bool = False) -> bool:
    """Boolean probe. This is what the recall gate is measured on."""
    return bool(find_red_flags(text, negation_aware=negation_aware))


def escalation_message(hits: list[RedFlagHit]) -> str:
    """Operator-facing text for a red-flagged turn."""
    if not hits:
        return ""
    lines = [
        "CLINICAL RED FLAG — communication coaching is suspended for this turn.",
        "",
        "Detected: " + ", ".join(sorted(h.category.replace("_", " ") for h in hits)),
        "",
    ]
    lines.extend(sorted({h.guidance for h in hits if h.guidance}))
    lines.append("")
    lines.append(
        "CARELite is a communication-support prototype. It is not a triage tool and makes no "
        "clinical recommendation. This turn requires a clinician's judgement, not a suggested "
        "phrasing."
    )
    return "\n".join(lines)


def screen(text: str, *, negation_aware: bool = False) -> SafetyVerdict:
    """Screen an utterance for clinical red flags.

    `allowed=False` on any hit. Nothing is redacted — the text is not the
    problem, the situation is — so `redacted_text` stays `None` and the caller
    surfaces `escalation_message()` instead of a coached response.
    """
    hits = find_red_flags(text, negation_aware=negation_aware)
    if not hits:
        return SafetyVerdict(allowed=True)

    immediate = any(h.urgency == "immediate" for h in hits)
    return SafetyVerdict(
        allowed=False,
        red_flag=True,
        flags=[f"redflag.{h.category}" for h in hits]
        + ([f"redflag.urgency.{'immediate' if immediate else 'urgent'}"]),
        reason=escalation_message(hits),
    )


__all__ = [
    "CATEGORIES",
    "Category",
    "RedFlagHit",
    "escalation_message",
    "find_red_flags",
    "is_red_flag",
    "screen",
]
