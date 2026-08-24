"""carelite.retrieval.query — one patient utterance in, three framework
queries plus a metadata filter out.

**The problem this solves.** Patients and guidance documents do not share a
vocabulary. A patient says "nobody ever tells me anything"; the corpus says
"information provision", "perceived clinician responsiveness", "patient
activation". Embedding the raw utterance and hoping for the best is the
naive baseline (ablation row R0), and the whole point of R1 is to measure how
much of the gap is closed by simply asking the *right question* in the
corpus's own register before retrieval runs.

**Two query languages, not one.** This module emits two different things and
the distinction is load-bearing:

- `dense_queries` are full framework-language sentences. Dense retrieval is
  paraphrase-tolerant, so a well-formed sentence in the target register is
  the best possible probe.
- `lexical_queries` are 2-3 content words. This is not a stylistic choice.
  `websearch_to_tsquery` **ANDs every content word**, with no implicit OR, so
  a sentence-length lexical query is a conjunction of six terms that no single
  512-token chunk satisfies. The carelite-index lane measured this directly:
  "teach-back method for confirming patient comprehension" returns **zero**
  rows against the live corpus while "teach-back" alone returns 15. Passing a
  natural sentence to the BM25 leg does not degrade recall gracefully — it
  collapses it to nothing. `lexical_queries` therefore never exceeds
  `MAX_LEXICAL_TERMS` content words, and `fusion.py` applies a further
  single-term backoff on a zero-hit query.

**Themes are detected, not guessed.** `THEME_CUES` maps each of the seven
`Theme` values to literal patient-language cues, and `THEME_QUERIES` maps it
to the framework-language sentences the corpus would use. The pairing is what
performs the translation. Coverage of the underlying corpus is uneven by a
wide margin — 13 papers substantively cover empathy and emotion recognition,
2 cover plain language, and exactly **1** covers teach-back — so a
well-formed teach-back query legitimately returns a weak, generic hit. That
is the corpus telling the truth about itself, and it is `crag.py`'s job to
act on it, not this module's job to paper over with a better-phrased query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from carelite.config import get_settings
from carelite.types import EncounterPhase, Theme

__all__ = [
    "MAX_LEXICAL_TERMS",
    "THEME_CUES",
    "THEME_QUERIES",
    "MetadataFilter",
    "QuerySet",
    "build_queries",
    "detect_themes",
]

#: Hard ceiling on content words in a lexical query. See the module docstring:
#: this is an empirical limit imposed by AND-of-terms tsquery semantics, not a
#: tunable preference.
MAX_LEXICAL_TERMS = 3

# ---------------------------------------------------------------------------
# Theme detection: patient language -> Theme
# ---------------------------------------------------------------------------

THEME_CUES: dict[Theme, tuple[str, ...]] = {
    Theme.EMPATHY: (
        "scared",
        "afraid",
        "frightened",
        "terrified",
        "worried",
        "anxious",
        "upset",
        "alone",
        "lonely",
        "overwhelmed",
        "hopeless",
        "crying",
        "nobody cares",
        "no one cares",
        "doesn't care",
        "does not care",
        "brushed off",
        "dismissed",
        "not listened to",
        "not heard",
        "rushed",
        "in a hurry",
        "too busy",
    ),
    Theme.EMOTION_RESPONSE: (
        "angry",
        "furious",
        "frustrated",
        "fed up",
        "sick of",
        "devastated",
        "shocked",
        "stunned",
        "numb",
        "falling apart",
        "breaking down",
        "can't cope",
        "cannot cope",
        "giving up",
        "no point",
        "panic",
        "dread",
        "guilty",
        "ashamed",
        "blame myself",
    ),
    Theme.ACTIVATION_SDM: (
        "options",
        "alternatives",
        "choices",
        "choose",
        "decide",
        "decision",
        "should i",
        "what would you do",
        "my choice",
        "up to me",
        "say in",
        "involved",
        "second opinion",
        "prefer",
        "rather not",
        "instead of",
        "weigh",
        "pros and cons",
        "trade off",
        "tradeoff",
    ),
    Theme.TEACH_BACK: (
        "understand",
        "understood",
        "confused",
        "unclear",
        "makes no sense",
        "lost me",
        "repeat",
        "say that again",
        "go over",
        "one more time",
        "did i get that right",
        "let me make sure",
        "in my own words",
        "forgot",
        "forget",
        "remember",
        # Comprehension failure is very often reported as a *pacing* complaint
        # rather than as "I don't understand". Found by probing the live
        # pipeline: "You said all that so fast, I didn't follow any of it."
        # detected no theme at all and fell through to the generic fallback
        # queries, so the strong-tier teach-back KB entries never had a chance
        # to surface. These are general patient phrasings, not probe-specific.
        "too fast",
        "so fast",
        "slow down",
        "follow",
        "keep up",
        "lost track",
        "too much at once",
        "all at once",
        "a lot of words",
    ),
    Theme.PLAIN_LANGUAGE: (
        "jargon",
        "big words",
        "medical terms",
        "technical",
        "complicated",
        "too much information",
        "over my head",
        "plain english",
        "simple terms",
        "layman",
        "what does that word mean",
        "long word",
        "fancy word",
        "in english",
        "what's that mean",
    ),
    Theme.TRUST_CONTINUITY: (
        "trust",
        "believe",
        "different doctor",
        "new doctor",
        "never the same",
        "keep seeing",
        "my regular",
        "started over",
        "tell my story again",
        "honest with me",
        "hiding something",
        "not telling me",
    ),
    Theme.EQUITY: (
        "interpreter",
        "translate",
        "translator",
        "english",
        "my language",
        "insurance",
        "afford",
        "can't pay",
        "cannot pay",
        "cost",
        "expensive",
        "transportation",
        "get a ride",
        "time off work",
        "miss work",
        "treated differently",
        "because of my",
        "looked down on",
        "judged",
    ),
}

#: Framework-language query sentences per theme. Written in the register of
#: the guidance literature rather than the clinic, because these are what get
#: embedded and matched against paper chunks.
THEME_QUERIES: dict[Theme, tuple[str, ...]] = {
    Theme.EMPATHY: (
        "empathic response to patient distress: naming and acknowledging the emotion "
        "before providing information",
        "how clinicians express empathy verbally during a difficult consultation",
        "missed empathic opportunities and their effect on patient experience",
    ),
    Theme.EMOTION_RESPONSE: (
        "recognising and responding to a patient's emotional cue rather than "
        "continuing with the clinical agenda",
        "NURSE statements: naming, understanding, respecting, supporting and "
        "exploring a patient's emotion",
        "clinician emotional blocking and premature reassurance in response to distress",
    ),
    Theme.ACTIVATION_SDM: (
        "shared decision making: presenting treatment options and eliciting the "
        "patient's preferences",
        "patient activation and involving patients in decisions about their own care",
        "eliciting patient values and goals before recommending a treatment plan",
    ),
    Theme.TEACH_BACK: (
        "teach-back: asking the patient to restate the plan in their own words to "
        "confirm comprehension",
        "confirming patient understanding of instructions without asking a yes or no question",
        "closing the communication loop to check recall of discharge instructions",
    ),
    Theme.PLAIN_LANGUAGE: (
        "using plain language and avoiding medical jargon when explaining a diagnosis",
        "health literacy and information clarity in clinician explanations",
        "chunking information and limiting the number of key points per explanation",
    ),
    Theme.TRUST_CONTINUITY: (
        "building trust in the clinician-patient relationship over repeated encounters",
        "relational continuity of care and its effect on patient trust and disclosure",
        "transparency and honesty when communicating uncertainty to a patient",
    ),
    Theme.EQUITY: (
        "communication disparities: patients of lower socioeconomic status receive "
        "less empathic communication",
        "limited English proficiency and interpreter use in clinical consultations",
        "equity-aware communication with patients from marginalised groups",
    ),
}

#: Lexical (BM25) probes per theme: framework terms that dense retrieval has
#: no obligation to rank highly, kept to <= MAX_LEXICAL_TERMS content words.
THEME_LEXICAL: dict[Theme, tuple[str, ...]] = {
    Theme.EMPATHY: ("empathy", "empathic response", "compassion"),
    Theme.EMOTION_RESPONSE: ("NURSE statements", "emotional cues", "distress"),
    Theme.ACTIVATION_SDM: ("shared decision", "patient activation", "SDM"),
    Theme.TEACH_BACK: ("teach-back", "comprehension", "recall"),
    Theme.PLAIN_LANGUAGE: ("health literacy", "plain language", "jargon"),
    Theme.TRUST_CONTINUITY: ("trust", "continuity", "therapeutic relationship"),
    Theme.EQUITY: ("disparities", "interpreter", "socioeconomic"),
}

#: Used when nothing else fires, so the pipeline never issues an empty query.
#: Phrased in the corpus's register and deliberately generic.
FALLBACK_QUERIES: tuple[str, ...] = (
    "clinician-patient communication behaviours that improve the patient's experience "
    "of the consultation",
    "how clinicians should respond to what a patient has just said",
    "patient-centred communication during a medical encounter",
)

FALLBACK_LEXICAL: tuple[str, ...] = ("communication", "patient centred", "consultation")

#: Encounter-phase-specific framework language, appended when the caller knows
#: the phase (the CLI and the scenario bank both do).
PHASE_QUERIES: dict[EncounterPhase, str] = {
    EncounterPhase.OPENING: "opening the encounter and eliciting the patient's full agenda",
    EncounterPhase.INFORMATION_GATHERING: "eliciting the patient's concerns, ideas and "
    "expectations during history taking",
    EncounterPhase.EXPLANATION: "explaining a diagnosis or test result in terms the "
    "patient can understand",
    EncounterPhase.PLANNING: "negotiating a treatment plan collaboratively with the patient",
    EncounterPhase.CLOSING: "closing the encounter, summarising and confirming next steps",
}

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "done",
        "for",
        "from",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "me",
        "my",
        "no",
        "nor",
        "not",
        "of",
        "on",
        "or",
        "our",
        "out",
        "over",
        "own",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "up",
        "us",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "am",
        "about",
        "again",
        "all",
        "any",
        "because",
        "before",
        "below",
        "between",
        "both",
        "did",
        "down",
        "during",
        "each",
        "few",
        "more",
        "most",
        "other",
        "same",
        "only",
        "own",
        "here",
        "now",
        "get",
        "got",
        "go",
        "going",
        "know",
        "like",
        "really",
        "thing",
        "things",
        "want",
        "went",
        "really",
    ]
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    """Structured pre-filter for the `kb_entry` legs.

    **Applies to `kb_entry` only, and that limitation is real.** `kb_entry`
    carries `theme`, `encounter_phase` and `equity_relevant` columns; the
    `chunk` table carries none of them (see `carelite/db/schema.sql`). There
    is therefore no honest way to theme-filter paper chunks, and this module
    does not pretend otherwise — `fusion.py` applies the filter to the
    kb_entry legs and leaves the chunk legs unfiltered rather than inventing
    a theme label for a chunk that has none.
    """

    themes: tuple[Theme, ...] = ()
    encounter_phase: EncounterPhase | None = None
    equity_relevant: bool | None = None

    @property
    def is_empty(self) -> bool:
        return not self.themes and self.encounter_phase is None and self.equity_relevant is None

    def describe(self) -> str:
        parts: list[str] = []
        if self.themes:
            parts.append("theme in {" + ", ".join(t.value for t in self.themes) + "}")
        if self.encounter_phase is not None:
            parts.append(f"phase = {self.encounter_phase.value}")
        if self.equity_relevant is not None:
            parts.append(f"equity_relevant = {self.equity_relevant}")
        return "; ".join(parts) if parts else "(none)"


@dataclass(frozen=True, slots=True)
class QuerySet:
    """Everything the retrieval legs need, derived from one utterance."""

    utterance: str
    dense_queries: tuple[str, ...]
    lexical_queries: tuple[str, ...]
    metadata: MetadataFilter
    themes: tuple[Theme, ...] = ()
    expanded: bool = True

    @property
    def all_queries(self) -> tuple[str, ...]:
        """What lands in `RetrievalTrace.queries` for the CLI evidence panel."""
        seen: dict[str, None] = {}
        for q in (*self.dense_queries, *self.lexical_queries):
            seen.setdefault(q, None)
        return tuple(seen)


def detect_themes(utterance: str, limit: int = 3) -> tuple[Theme, ...]:
    """Rank the seven themes by literal cue matches. Ties break in `Theme`
    declaration order, which is deterministic and therefore reproducible."""
    hay = " " + " ".join(_WORD_RE.findall(utterance.casefold())) + " "
    scored: list[tuple[int, int, Theme]] = []
    for order, (theme, cues) in enumerate(THEME_CUES.items()):
        hits = sum(1 for cue in cues if f" {cue.casefold()} " in hay)
        if hits:
            scored.append((-hits, order, theme))
    scored.sort()
    return tuple(theme for _, _, theme in scored[:limit])


def _content_terms(utterance: str, limit: int) -> list[str]:
    """Salient non-stopword terms from the patient's own wording.

    Kept because framework language alone can miss a concrete noun the
    patient used ("biopsy", "chemo") that is also a literal corpus term.
    """
    seen: dict[str, None] = {}
    for word in _WORD_RE.findall(utterance.casefold()):
        if len(word) < 4 or word in _STOPWORDS:
            continue
        seen.setdefault(word, None)
    return list(seen)[:limit]


def build_queries(
    utterance: str,
    *,
    encounter_phase: EncounterPhase | None = None,
    expand: bool = True,
    n_queries: int | None = None,
    equity_relevant: bool | None = None,
) -> QuerySet:
    """Build the framework queries and metadata filter for one turn.

    `expand=False` is ablation row R0: the raw utterance becomes the only
    dense query and its own content words the only lexical query. That is the
    naive baseline everything else is measured against.
    """
    settings = get_settings().retrieval
    n = n_queries or settings.n_framework_queries
    themes = detect_themes(utterance, limit=n)

    if not expand:
        terms = _content_terms(utterance, MAX_LEXICAL_TERMS)
        return QuerySet(
            utterance=utterance,
            dense_queries=(utterance,),
            lexical_queries=((" ".join(terms),) if terms else ()),
            metadata=MetadataFilter(),
            themes=themes,
            expanded=False,
        )

    dense: list[str] = []
    lexical: list[str] = []

    # One query per detected theme, taking that theme's strongest phrasing
    # first, so three distinct themes yield three genuinely different probes
    # rather than three rewordings of one.
    for theme in themes:
        dense.append(THEME_QUERIES[theme][0])
        lexical.append(THEME_LEXICAL[theme][0])

    # A detected theme with room left over contributes its secondary phrasings.
    for theme in themes:
        for extra in THEME_QUERIES[theme][1:]:
            if len(dense) >= n:
                break
            dense.append(extra)

    if encounter_phase is not None and len(dense) < n:
        dense.append(PHASE_QUERIES[encounter_phase])

    for fallback in FALLBACK_QUERIES:
        if len(dense) >= n:
            break
        dense.append(fallback)

    # The patient's own salient nouns, as one short lexical probe.
    own_terms = _content_terms(utterance, MAX_LEXICAL_TERMS)
    if own_terms:
        lexical.append(" ".join(own_terms))
    for fallback in FALLBACK_LEXICAL:
        if len(lexical) >= n:
            break
        lexical.append(fallback)

    dense = _dedupe(dense)[:n]
    lexical = [_truncate_terms(q) for q in _dedupe(lexical)][:n]

    return QuerySet(
        utterance=utterance,
        dense_queries=tuple(dense),
        lexical_queries=tuple(q for q in lexical if q),
        metadata=MetadataFilter(
            themes=themes,
            encounter_phase=encounter_phase,
            equity_relevant=(
                True if equity_relevant is None and Theme.EQUITY in themes else equity_relevant
            ),
        ),
        themes=themes,
        expanded=True,
    )


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        cleaned = item.strip()
        if cleaned:
            seen.setdefault(cleaned, None)
    return list(seen)


def _truncate_terms(query: str, limit: int = MAX_LEXICAL_TERMS) -> str:
    """Enforce the AND-of-terms ceiling from the module docstring."""
    words = query.split()
    return " ".join(words[:limit])
