"""carelite.retrieval.router — adaptive routing (v3 §3, "Adaptive RAG").

Classifies a patient turn as `EMOTIONAL_ONLY`, `INFORMATIONAL`, or `MIXED`.
An `EMOTIONAL_ONLY` turn **skips retrieval entirely**.

This is a *quality* decision, not a latency one, and it is worth being
precise about why. The literature position this whole project is built on is
that communication frameworks stop working the moment they become scripts.
When a patient says "I'm just so scared", the correct clinician response is
to stay with the feeling. A system that answers that turn by injecting four
paragraphs of retrieved guidance about teach-back and shared decision-making
produces exactly the failure mode the README warns about: it comes out
clinical instead of warm. Retrieval is not free even when it is accurate.

**Why the default classifier is a lexicon and not a model.** An LLM router
would be more flexible, and `use_llm_router` turns one on. It is off by
default because the router sits *inside* the independent variable of a
controlled comparison: if the route for a given scenario can differ between
two runs of Condition C, then C is not one condition, and the Friedman test
in v3 §14 is being run over a moving target. A deterministic classifier makes
the route a fixed, inspectable property of the scenario text. `EMOTION_CUES` / `INFORMATION_CUES`
is therefore written to be *auditable* — every cue is a literal string a
reviewer can check against the scenario bank.

**The asymmetry in the tie-breaks is deliberate.** An emotional turn wrongly
routed to retrieval yields a clinical-sounding answer (bad, and the thing
this router exists to prevent). An informational turn wrongly routed away
from retrieval yields an unsupported answer (also bad, and it silently
removes the very evidence Condition C is supposed to be testing). The second
failure corrupts the experiment, so anything carrying a genuine information
request routes to `MIXED` rather than `EMOTIONAL_ONLY`, even when the
emotional signal is much stronger. `EMOTIONAL_ONLY` is reserved for turns
with an emotional cue and *no* information request at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from carelite.types import Route

__all__ = [
    "EMOTION_CUES",
    "INFORMATION_CUES",
    "QUESTION_OPENERS",
    "RouteDecision",
    "classify",
    "route_turn",
]

# ---------------------------------------------------------------------------
# Cue lexicons
#
# Phrases, not stems: "scared" and "scary" both appear, spelled out, because a
# reviewer auditing the router should be able to read the list and predict the
# routing of any scenario in the bank without running code.
# ---------------------------------------------------------------------------

#: Explicit feeling words and the idioms patients actually use for them.
EMOTION_CUES: tuple[str, ...] = (
    "afraid",
    "scared",
    "scary",
    "terrified",
    "frightened",
    "fear",
    "petrified",
    "anxious",
    "anxiety",
    "worried",
    "worry",
    "nervous",
    "panicking",
    "panic",
    "upset",
    "sad",
    "sadness",
    "depressed",
    "devastated",
    "heartbroken",
    "crying",
    "cry",
    "tears",
    "sobbing",
    "breaking down",
    "angry",
    "furious",
    "frustrated",
    "frustrating",
    "fed up",
    "sick of",
    "hopeless",
    "helpless",
    "lost",
    "alone",
    "lonely",
    "overwhelmed",
    "exhausted",
    "drained",
    "numb",
    "can't cope",
    "cannot cope",
    "can't take",
    # "tired OF" (weariness), deliberately not bare "tired": "I'm tired all
    # the time" is a fatigue symptom report that should retrieve, while "I'm
    # tired of all of it" is distress. The preposition is what separates them,
    # so the cue carries it.
    "tired of",
    "worn out",
    "had enough",
    "can't do this",
    "cannot do this",
    "can't handle",
    "falling apart",
    "giving up",
    "give up",
    "no point",
    "ashamed",
    "embarrassed",
    "guilty",
    "guilt",
    "blame myself",
    "stressed",
    "dread",
    "dreading",
    "shaken",
    "shocked",
    "stunned",
    "don't know what to do",
    "do not know what to do",
    "not sure i can",
    "keeps me up at night",
    "losing sleep",
)

#: Intensity markers. Presence roughly doubles a cue's weight — "so scared"
#: and "a bit worried" are not the same turn.
INTENSIFIERS: tuple[str, ...] = (
    "so ",
    "very ",
    "really ",
    "just so ",
    "extremely ",
    "incredibly ",
    "terribly ",
    "absolutely ",
    "completely ",
    "totally ",
    "beyond ",
)

#: Requests for information, explanation, or a decision. These are what make
#: retrieval worth doing.
INFORMATION_CUES: tuple[str, ...] = (
    "what does",
    "what do",
    "what is",
    "what are",
    "what happens",
    "what if",
    "what should",
    "what kind",
    "what sort",
    "how do",
    "how does",
    "how long",
    "how much",
    "how many",
    "how often",
    "how bad",
    "how will",
    "how can",
    "why do",
    "why does",
    "why did",
    "why is",
    "why are",
    "why can't",
    "when will",
    "when do",
    "when should",
    "where do",
    "where can",
    "which one",
    "which option",
    "who do",
    "who should",
    "explain",
    "tell me about",
    "help me understand",
    "don't understand",
    "do not understand",
    "confused",
    "unclear",
    "makes no sense",
    "mean",
    "means",
    "meaning",
    "options",
    "alternatives",
    "choices",
    "side effect",
    "side effects",
    "risk",
    "risks",
    "benefit",
    "benefits",
    "chances",
    "odds",
    "survival",
    "prognosis",
    "treatment",
    "surgery",
    "medication",
    "medicine",
    "dose",
    "dosage",
    "test",
    "tests",
    "results",
    "scan",
    "biopsy",
    "diagnosis",
    "procedure",
    "recovery",
    "next step",
    "next steps",
    "plan",
    "should i",
    "do i need",
    "can i",
    "could i",
    "is it safe",
    "is there",
    "tell me",
    "walk me through",
    "go over",
    "repeat that",
    "say that again",
)

#: Sentence openers that make a turn a question even without a question mark.
QUESTION_OPENERS: tuple[str, ...] = (
    "what",
    "how",
    "why",
    "when",
    "where",
    "which",
    "who",
    "can ",
    "could ",
    "should ",
    "will ",
    "would ",
    "is ",
    "are ",
    "do ",
    "does ",
    "did ",
)

_WORD_RE = re.compile(r"[a-z']+")

#: Cues are written in one canonical form and matched with this closed set of
#: English inflections appended. Without it, `explain` fails to match "Nobody
#: explains anything to me" and that turn routes EMOTIONAL_ONLY — losing its
#: retrieval, which is the experiment-corrupting direction this module's
#: docstring says to avoid. A closed suffix list is preferred over a stemmer
#: because it keeps `EMOTION_CUES` / `INFORMATION_CUES` auditable: a reviewer can still read a cue
#: and know exactly which surface forms it covers.
_INFLECTIONS: tuple[str, ...] = ("", "s", "es", "ed", "d", "ing")


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """The router's verdict plus the evidence for it.

    `matched_emotion` / `matched_information` are kept so the CLI's `/why`
    view and any later audit can show *which literal cue* produced a route,
    rather than asking a reviewer to trust a bare label.
    """

    route: Route
    emotion_score: float
    information_score: float
    matched_emotion: tuple[str, ...] = ()
    matched_information: tuple[str, ...] = ()
    rationale: str = ""
    used_llm: bool = False

    @property
    def should_retrieve(self) -> bool:
        """The single question the pipeline asks this object."""
        return self.route is not Route.EMOTIONAL_ONLY


def _normalise(text: str) -> str:
    return " " + " ".join(_WORD_RE.findall(text.casefold())) + " "


def _scan(text: str, cues: tuple[str, ...]) -> tuple[float, tuple[str, ...]]:
    """Weighted cue count over a normalised utterance.

    Matching is on a whitespace-padded, punctuation-stripped form so that a
    cue matches whole words only: without the padding, "mean" would fire on
    "meantime" and "cry" on "cryotherapy".
    """
    hay = _normalise(text)
    total = 0.0
    matched: list[str] = []
    for cue in cues:
        stem = cue.strip()
        hit = next(
            (form for form in (stem + suffix for suffix in _INFLECTIONS) if f" {form} " in hay),
            None,
        )
        if hit is None:
            continue
        weight = 1.0
        for intensifier in INTENSIFIERS:
            if f"{intensifier.strip()} {hit}" in hay:
                weight = 2.0
                break
        total += weight
        matched.append(hit)
    return total, tuple(matched)


def _looks_like_question(text: str) -> bool:
    if "?" in text:
        return True
    stripped = text.strip().casefold()
    return any(stripped.startswith(opener.strip() + " ") for opener in QUESTION_OPENERS)


def classify(utterance: str) -> RouteDecision:
    """Deterministic lexicon classifier. Pure: no I/O, no model, no clock."""
    emotion, emotion_hits = _scan(utterance, EMOTION_CUES)
    information, info_hits = _scan(utterance, INFORMATION_CUES)

    is_question = _looks_like_question(utterance)
    if is_question:
        # A question mark is itself an information request, and a strong one.
        information += 1.5

    if emotion <= 0 and information <= 0:
        # No cue fired at all. Retrieve: an unclassifiable turn that silently
        # skips retrieval would remove evidence from Condition C without
        # anyone noticing, which is the failure mode this module's docstring
        # calls out as corrupting the experiment.
        return RouteDecision(
            route=Route.INFORMATIONAL,
            emotion_score=emotion,
            information_score=information,
            rationale="no emotional or informational cue matched; defaulting to "
            "retrieval so an unclassified turn never silently loses its evidence",
        )

    if information > 0 or is_question:
        if emotion > 0:
            return RouteDecision(
                route=Route.MIXED,
                emotion_score=emotion,
                information_score=information,
                matched_emotion=emotion_hits,
                matched_information=info_hits,
                rationale="carries both an emotional cue and an information request",
            )
        return RouteDecision(
            route=Route.INFORMATIONAL,
            emotion_score=emotion,
            information_score=information,
            matched_information=info_hits,
            rationale="information request with no emotional cue",
        )

    return RouteDecision(
        route=Route.EMOTIONAL_ONLY,
        emotion_score=emotion,
        information_score=information,
        matched_emotion=emotion_hits,
        rationale="emotional cue with no information request; retrieval is skipped "
        "so the response can stay with the feeling rather than answering a "
        "question the patient did not ask",
    )


# ---------------------------------------------------------------------------
# Optional LLM router
# ---------------------------------------------------------------------------

#: Trusted, git-tracked system template. Never contains the utterance — that
#: arrives fenced, in the user turn, via `LLMClient.chat`.
ROUTER_SYSTEM = """You classify a single patient turn from a clinical encounter.

Choose exactly one label:
- "emotional_only": the patient is expressing feeling and is not asking for
  information, explanation, or a decision.
- "informational": the patient is asking for information, explanation, or help
  with a decision, without expressing distress.
- "mixed": both at once.

Answer with a JSON object: {"route": "<label>", "rationale": "<one short sentence>"}.
Classify only. Do not answer the patient, and do not offer clinical advice."""

_ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["emotional_only", "informational", "mixed"]},
        "rationale": {"type": "string"},
    },
    "required": ["route"],
}


def classify_with_llm(utterance: str, client: object) -> RouteDecision | None:
    """Ask a model instead of the lexicon. `None` if the model is unavailable
    or returns something unparseable, so the caller can fall back."""
    import json

    result = client.chat(  # type: ignore[attr-defined]
        system=ROUTER_SYSTEM,
        task="Classify the patient turn above. Reply with the JSON object only.",
        utterance=utterance,
        json_schema=_ROUTER_SCHEMA,
        num_predict=120,
    )
    if result is None or not result.text:
        return None
    try:
        obj = json.loads(result.text)
        route = Route(str(obj["route"]).strip().casefold())
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None
    return RouteDecision(
        route=route,
        emotion_score=float("nan"),
        information_score=float("nan"),
        rationale=str(obj.get("rationale", ""))[:300],
        used_llm=True,
    )


def route_turn(
    utterance: str,
    *,
    enabled: bool = True,
    use_llm: bool = False,
    client: object | None = None,
) -> RouteDecision:
    """The pipeline's entry point.

    `enabled=False` is the R0-R7 ablation configuration: the router is off,
    so every turn is treated as informational and retrieval always runs. That
    is what makes R8-vs-R9 in the ablation table a measurement of the router
    rather than of the router plus whatever else moved.
    """
    if not enabled:
        return RouteDecision(
            route=Route.INFORMATIONAL,
            emotion_score=0.0,
            information_score=0.0,
            rationale="router ablated off; all turns retrieve",
        )
    if use_llm and client is not None:
        decision = classify_with_llm(utterance, client)
        if decision is not None:
            return decision
        # Model unavailable or unparseable: fall through to the lexicon
        # rather than failing the turn.
    return classify(utterance)
