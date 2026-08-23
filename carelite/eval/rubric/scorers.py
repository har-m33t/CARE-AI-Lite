"""Deterministic, non-LLM measurements over a generated response.

These are not a substitute for the judge or for human raters. They measure the
things that can be counted without an opinion — jargon load, reading level,
question count, teach-back phrasing, message count, length, hedge density, and
the surface markers of ritual — and they earn their place three ways:

1. **They anchor the judge.** A judge that rates `ie` = 5 on a response with a
   grade-16 reading level and no comprehension check is disagreeing with a
   fact, not with a rater, and that disagreement is findable.
2. **They are free and exactly reproducible.** Every one is a pure function of
   the response text at a pinned `SCORER_VERSION`. Re-running the study
   reproduces them bit for bit, with no model in the loop.
3. **They give the `ritualistic` dimension an independent estimate.** Ritual has
   real surface markers — framework labels, templated scaffolding, stacked
   stock stems — so it can be screened mechanically. `ritualistic_proxy` is a
   screening estimate on the same reverse-coded 1-5 scale, not a replacement
   for a rating, and its agreement with human consensus is itself reportable.

Every measurement below is a descriptor, not a score. Several are explicitly
two-sided: too few hedges reads as false certainty and too many reads as
evasion, and no threshold in this module encodes which is which. Turning a
measurement into a judgement is the rater's job and the rubric's job, not this
module's — with the single, documented exception of `ritualistic_proxy`.

Sources for the individual measures are named on each function.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from carelite.eval.rubric.dimensions import SCALE_MAX, SCALE_MIN
from carelite.types import RaterType, RubricScore

__all__ = [
    "HEDGE_TERMS",
    "JARGON_TERMS",
    "SCORER_VERSION",
    "DeterministicScores",
    "RitualMarkers",
    "deterministic_rubric_score",
    "flesch_kincaid_grade",
    "hedge_density",
    "jargon_density",
    "jargon_terms_found",
    "message_count",
    "pseudo_teach_back_phrases",
    "question_count",
    "response_length",
    "ritual_markers",
    "ritualistic_proxy",
    "score_text",
    "sentence_count",
    "sentences",
    "syllable_count",
    "teach_back_phrases",
    "teach_back_present",
    "words",
]

#: Bumped on any change that could move a number. Persisted with the scores so
#: a re-analysis can tell which scorer version produced a row.
SCORER_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*")  # noqa: RUF001 — curly apostrophe is real input

# Abbreviations whose full stop is not a sentence boundary. Small and explicit:
# an exhaustive list is not achievable and a long one hides its own errors.
_ABBREVIATIONS = (
    "Dr",
    "Mr",
    "Mrs",
    "Ms",
    "Prof",
    "St",
    "Jr",
    "Sr",
    "vs",
    "etc",
    "approx",
    "e.g",
    "i.e",
    "a.m",
    "p.m",
    "No",
)
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.",
    re.IGNORECASE,
)
_DOT = "\x00"
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"'’”)]*\s+")  # noqa: RUF001 — smart quotes are real input

# Structural line prefixes: markdown headings, bullets, numbered items, and
# bolded labels of the "**Naming:**" kind.
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_BULLET_RE = re.compile(r"^\s*[-*•–]\s+")  # noqa: RUF001 — en dash is a real bullet character
_NUMBERED_RE = re.compile(r"^\s*\(?\d+[.)]\s+")
# "**Naming:**" and "**Naming**:" are both templated labels; require the colon
# either inside or outside the emphasis markers.
_BOLD_LABEL_RE = re.compile(
    r"^[ \t]*(?:[-*•]\s*)?(?:\*\*|__)[^*_\n]{1,40}(?::[ \t]*(?:\*\*|__)|(?:\*\*|__)[ \t]*:)"
)
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•–]\s+|\(?\d+[.)]\s+|#{1,6}\s+)")  # noqa: RUF001 — as above


def words(text: str) -> list[str]:
    """Alphabetic word tokens. Numbers and bare punctuation are not words."""
    return _WORD_RE.findall(text)


def sentences(text: str) -> list[str]:
    """Split into sentences, treating each line of a list as its own unit.

    Abbreviations in `_ABBREVIATIONS` do not end a sentence. Markdown list and
    heading prefixes are stripped from the returned text but the line break is
    honoured as a boundary, so a bulleted response segments the way a reader
    would segment it.
    """
    protected = _ABBREV_RE.sub(lambda m: m.group(1) + _DOT, text)
    out: list[str] = []
    for raw_line in protected.splitlines():
        line = _LIST_PREFIX_RE.sub("", raw_line).strip()
        if not line:
            continue
        for piece in _SENTENCE_END_RE.split(line):
            cleaned = piece.replace(_DOT, ".").strip()
            if cleaned:
                out.append(cleaned)
    return out


def sentence_count(text: str) -> int:
    return len(sentences(text))


def syllable_count(word: str) -> int:
    """Vowel-group syllable heuristic with a silent-final-`e` correction.

    Approximate by construction. It is exact on short common words ("the" -> 1,
    "hospitalization" -> 6) and drifts on words with adjacent vowels in
    separate syllables ("immediate" -> 3, truly 4). That drift is acceptable
    here because Flesch-Kincaid is used comparatively, across conditions
    scored by the identical function, not as an absolute grade claim.
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if n > 1 and w.endswith("e") and not w.endswith(("le", "ee", "ye", "oe", "ie")):
        n -= 1
    return max(n, 1)


# ---------------------------------------------------------------------------
# Readability
# ---------------------------------------------------------------------------


def flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid grade level: 0.39*(w/s) + 11.8*(syl/w) - 15.59.

    Patient-facing material is conventionally targeted at roughly a sixth-grade
    level (Weiss BD. *Health Literacy and Patient Safety: Help Patients
    Understand.* 2nd ed. AMA Foundation; 2007). This is a descriptor, not a
    threshold: a response can be grade 4 and still be useless.

    Returns 0.0 for text with no words or no sentences.
    """
    ws = words(text)
    n_words = len(ws)
    n_sentences = sentence_count(text)
    if n_words == 0 or n_sentences == 0:
        return 0.0
    n_syllables = sum(syllable_count(w) for w in ws)
    return 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59


# ---------------------------------------------------------------------------
# Jargon
# ---------------------------------------------------------------------------

#: Clinical terms that patients demonstrably misread or do not know.
#:
#: Drawn from the comprehension literature on medical jargon — Gotlieb R,
#: Praska C, Hendrickson MA, et al. Accuracy in patient understanding of common
#: medical phrases. JAMA Netw Open. 2022;5(11):e2242972; Pitt MB, Hendrickson
#: MA. Eradicating jargon-oblivion. J Gen Intern Med. 2020;35(2):598-603 —
#: plus the terms that recur in this project's scenario domain.
#:
#: Deliberately EXCLUDED: words that are jargon in clinical use but ordinary
#: English elsewhere ("negative", "positive", "progressive", "gross",
#: "impressive", "occult"). These are among the most misunderstood phrases in
#: the literature, but a bag-of-words matcher cannot tell "the test was
#: negative" from "a negative experience", and a false positive here is worse
#: than a miss. `jargon_density` is therefore a conservative undercount, and
#: should be reported as one.
JARGON_TERMS: frozenset[str] = frozenset(
    {
        "adjuvant",
        "adherence",
        "afebrile",
        "anastomosis",
        "anterior",
        "arrhythmia",
        "aspiration",
        "asymptomatic",
        "benign",
        "bilateral",
        "biopsy",
        "bradycardia",
        "carcinoma",
        "catheter",
        "comorbid",
        "comorbidities",
        "comorbidity",
        "contraindicated",
        "contraindication",
        "contrast-enhanced",
        "cytology",
        "differential diagnosis",
        "distal",
        "dyspnea",
        "dyspnoea",
        "edema",
        "effusion",
        "embolism",
        "endoscopic",
        "etiology",
        "excision",
        "exacerbation",
        "febrile",
        "haematoma",
        "haemorrhage",
        "hematoma",
        "hemorrhage",
        "hyperglycaemia",
        "hyperglycemia",
        "hypertension",
        "hypoglycaemia",
        "hypoglycemia",
        "iatrogenic",
        "idiopathic",
        "infarction",
        "infiltrate",
        "interval growth",
        "intravenous",
        "ischaemia",
        "ischemia",
        "lateral",
        "lesion",
        "lumen",
        "malignancy",
        "malignant",
        "metastases",
        "metastasis",
        "metastatic",
        "morbidity",
        "myocardial",
        "neoadjuvant",
        "neoplasm",
        "nodule",
        "noninvasive",
        "oedema",
        "opacity",
        "palliative",
        "parenteral",
        "pathology",
        "percutaneous",
        "peripheral",
        "posterior",
        "prognosis",
        "prophylactic",
        "prophylaxis",
        "proximal",
        "pulmonary",
        "radiolucent",
        "refractory",
        "remission",
        "resection",
        "risk stratification",
        "sepsis",
        "septic",
        "serial imaging",
        "spiculation",
        "spiculated",
        "stenosis",
        "subcentimeter",
        "subcentimetre",
        "subcutaneous",
        "surveillance",
        "systemic",
        "tachycardia",
        "thrombosis",
        "titrate",
        "titration",
        "unilateral",
        "unremarkable",
        "workup",
    }
)


def _lexicon_pattern(terms: Iterable[str]) -> re.Pattern[str]:
    """One alternation over a lexicon, longest term first so phrases win."""
    ordered = sorted(terms, key=len, reverse=True)
    # re.escape stopped escaping spaces in 3.7; handle both spellings.
    body = "|".join(re.escape(t).replace("\\ ", " ").replace(" ", r"\s+") for t in ordered)
    return re.compile(rf"\b(?:{body})\b", re.IGNORECASE)


_JARGON_RE = _lexicon_pattern(JARGON_TERMS)


def jargon_terms_found(text: str) -> list[str]:
    """Every jargon hit, in order, lowercased. Verbatim substrings of `text`."""
    return [m.group(0).lower() for m in _JARGON_RE.finditer(text)]


def jargon_density(text: str) -> float:
    """Jargon hits per word, in [0, 1]. Multi-word terms count as one hit.

    Bears on the `ie` dimension and on README theme 5 (plain language). A
    response cannot reach `ie` = 5 while dense in unexplained jargon, however
    good its comprehension check.
    """
    n_words = len(words(text))
    if n_words == 0:
        return 0.0
    return len(jargon_terms_found(text)) / n_words


# ---------------------------------------------------------------------------
# Hedging
# ---------------------------------------------------------------------------

#: Epistemic hedges. TWO-SIDED by design: transparency about uncertainty is
#: theme 6 of README.md and a low hedge density can mean false certainty, while
#: a high one reads as evasion. This module reports the number and takes no
#: view on it.
HEDGE_TERMS: frozenset[str] = frozenset(
    {
        "a bit",
        "apparently",
        "arguably",
        "appears",
        "appear",
        "could be",
        "fairly",
        "generally",
        "i believe",
        "i suppose",
        "i think",
        "in some cases",
        "kind of",
        "likely",
        "may",
        "maybe",
        "might",
        "more or less",
        "often",
        "perhaps",
        "possibly",
        "presumably",
        "probably",
        "rather",
        "seem",
        "seems",
        "somewhat",
        "sometimes",
        "sort of",
        "tend to",
        "tends to",
        "to some extent",
        "typically",
        "unlikely",
        "usually",
    }
)

_HEDGE_RE = _lexicon_pattern(HEDGE_TERMS)


def hedge_density(text: str) -> float:
    """Hedge hits per word, in [0, 1]. See `HEDGE_TERMS` on interpretation."""
    n_words = len(words(text))
    if n_words == 0:
        return 0.0
    return len(_HEDGE_RE.findall(text)) / n_words


# ---------------------------------------------------------------------------
# Questions, teach-back, message count, length
# ---------------------------------------------------------------------------


def question_count(text: str) -> int:
    """Sentences ending in a question mark.

    Bears on `explore` and `epp`. Two-sided: zero questions cannot support an
    exploring move, and a response that is nothing but questions is an
    interrogation. Counting says nothing about whether the questions are open.
    """
    return sum(1 for s in sentences(text) if s.rstrip().endswith("?"))


#: Teach-back proper: the patient is asked to restate, in their own words, what
#: they will do or what they understood. Schillinger D, Piette J, Grumbach K,
#: et al. Closing the loop. Arch Intern Med. 2003;163(1):83-90; Ha Dinh TT,
#: Bonner A, Clark R, Ramsbotham J, Hines S. The effectiveness of the
#: teach-back method. JBI Database System Rev Implement Rep. 2016;14(1):210-247.
_TEACH_BACK_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"in your own words",
        r"what(?:'s| is| are)? your understanding",
        r"what (?:will|are) you (?:going to )?(?:say|tell)",
        r"what (?:would|will) you tell (?:your |them|him|her)",
        r"how would you (?:explain|describe|put) (?:it|this|that)",
        r"(?:tell|explain|describe|walk) (?:it |this |that )?(?:back )?(?:me )?(?:through )?to me",
        r"walk me through",
        r"can you (?:repeat|summari[sz]e|say) (?:it |this |that )?back",
        r"i want to (?:make sure|be sure|check) (?:that )?i(?:'ve| have)? explained",
        r"(?:just )?(?:so|to make sure) (?:i|we) (?:know|explained)",
        r"to make sure (?:i|we) (?:did|explained|got)",
        r"when you (?:get|go) home,? what",
    )
)

#: NOT teach-back. Closed comprehension checks that the literature specifically
#: contrasts with teach-back, because a patient who did not understand will
#: usually still answer "yes". Detected separately so `ie` scoring can tell the
#: two apart rather than crediting the wrong one.
_PSEUDO_TEACH_BACK_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"do(?:es)? (?:that|this|it) make sense",
        r"do you understand",
        r"is that clear",
        r"are you (?:with me|following)",
        r"any (?:other )?questions",
        r"do you have any questions",
        r"(?:that )?(?:all )?make sense\?",
        r"okay\?",
    )
)


def teach_back_phrases(text: str) -> list[str]:
    """Verbatim teach-back spans found in `text`, in order of appearance."""
    hits: list[tuple[int, str]] = []
    for pattern in _TEACH_BACK_PATTERNS:
        hits.extend((m.start(), m.group(0)) for m in pattern.finditer(text))
    return [span for _, span in sorted(hits, key=lambda h: h[0])]


def teach_back_present(text: str) -> bool:
    """True if a genuine teach-back request appears. See `teach_back_phrases`."""
    return bool(teach_back_phrases(text))


def pseudo_teach_back_phrases(text: str) -> list[str]:
    """Closed comprehension checks ('does that make sense?'), which are not teach-back."""
    hits: list[tuple[int, str]] = []
    for pattern in _PSEUDO_TEACH_BACK_PATTERNS:
        hits.extend((m.start(), m.group(0)) for m in pattern.finditer(text))
    return [span for _, span in sorted(hits, key=lambda h: h[0])]


def message_count(text: str) -> int:
    """Count of declarative content units — the 'key messages' ceiling check.

    An enumerated line (bullet, number, heading, bolded label) counts as one
    unit regardless of how many sentences it holds; elsewhere each declarative
    sentence is one unit. Questions are excluded — they are counted by
    `question_count`.

    README theme 5 flags responses exceeding three key messages, following the
    Ask Me 3 / plain-language guidance in Weiss BD. *Health Literacy and
    Patient Safety.* 2nd ed. AMA Foundation; 2007. This is a proxy and it
    over-counts: a two-sentence empathic opening reads as two units even though
    it carries no clinical message. Read it as an upper bound on message load.
    """
    total = 0
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        is_enumerated = bool(
            _HEADING_RE.match(raw_line)
            or _BULLET_RE.match(raw_line)
            or _NUMBERED_RE.match(raw_line)
            or _BOLD_LABEL_RE.match(raw_line)
        )
        if is_enumerated:
            total += 1
            continue
        total += sum(1 for s in sentences(stripped) if not s.rstrip().endswith("?"))
    return total


def response_length(text: str) -> int:
    """Word count. Length is not quality; it is a covariate.

    Longer responses have more surface on which a rater can find a NURSE move,
    so length must be reported alongside rubric scores and checked as a
    confounder before any condition difference is believed.
    """
    return len(words(text))


# ---------------------------------------------------------------------------
# Ritual markers  --  the reverse-coded dimension's mechanical signal
# ---------------------------------------------------------------------------

#: Framework vocabulary that has leaked into patient-facing output. This is the
#: single strongest ritual signal: the model is showing its scaffolding.
_FRAMEWORK_LABEL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bNURSE\b"),  # case-sensitive: the mnemonic, not the clinician
    re.compile(r"\bfour habits\b", re.IGNORECASE),
    re.compile(r"\binvest(?:ing)? in the (?:beginning|end)\b", re.IGNORECASE),
    re.compile(r"\belicit(?:ing)? the patient'?s? perspective\b", re.IGNORECASE),
    re.compile(r"\bdemonstrat(?:e|ing) empathy\b", re.IGNORECASE),
    re.compile(r"\bteach[- ]?back (?:method|technique|question)\b", re.IGNORECASE),
    # A NURSE/Four-Habits step used as a section label: "Naming:",
    # "**Understanding:**", "- Respecting**:". "Next steps:" is deliberately
    # NOT here — a next-steps heading is ordinary clinical practice, not
    # framework leakage.
    re.compile(
        r"(?mi)^[ \t>*_•-]{0,8}"
        r"(?:naming|understanding|understand|respecting|respect|supporting|support|"
        r"exploring|explore|empathy|acknowledgements?|acknowledgments?)"
        r"[ \t*_]{0,4}:"
    ),
)

#: Stock empathy stems: fluent, warm, and interchangeable across every patient
#: and every problem. Salmon P, Young B. Creativity in clinical communication.
#: Med Educ. 2011;45(3):217-226 — the ritualised-performance argument.
_STOCK_PHRASE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"i hear you",
        r"that sounds (?:really |very |incredibly |so |quite )?(?:hard|tough|difficult|scary|frightening|stressful|overwhelming|awful)",
        r"that must be (?:so |really |very |incredibly |quite )?(?:hard|tough|difficult|scary|frightening|overwhelming)",
        r"i can(?:'t| ?not)? (?:only )?imagine",
        r"it(?:'s| is) (?:completely |totally |perfectly |entirely |very )?understandable",
        r"it makes (?:complete |perfect |total )?sense that",
        r"thank you for sharing",
        r"i (?:really )?appreciate you (?:sharing|telling|opening)",
        r"i want you to know",
        r"please know that",
        r"we(?:'re| are) here for you",
        r"you(?:'re| are) not alone",
        r"every step of the way",
        r"we(?:'ll| will) (?:get through|face) (?:this|it) together",
        r"i(?:'m| am) (?:so |truly |really )?sorry (?:to hear|you'?re|that you)",
        r"you(?:'re| are) stronger than you know",
        r"i (?:want to )?acknowledge your (?:strength|courage|bravery|feelings)",
        r"that takes (?:a lot of )?courage",
        r"whatever happens,? we",
    )
)


@dataclass(frozen=True, slots=True)
class RitualMarkers:
    """Surface evidence of formulaic, script-like output.

    All three fields hold verbatim substrings of the response, so they can be
    used directly as the evidence spans that v3 §13 requires of every score.
    """

    framework_labels: tuple[str, ...]
    stock_phrases: tuple[str, ...]
    scaffold_lines: tuple[str, ...]

    @property
    def total(self) -> int:
        return len(self.framework_labels) + len(self.stock_phrases) + len(self.scaffold_lines)

    def longest_span(self) -> str | None:
        """The longest matched marker, for use as a grounding span."""
        spans: Sequence[str] = (*self.framework_labels, *self.stock_phrases, *self.scaffold_lines)
        return max(spans, key=len) if spans else None


def ritual_markers(text: str) -> RitualMarkers:
    """Extract the three mechanical signatures of ritual from a response."""
    labels = tuple(m.group(0).strip() for p in _FRAMEWORK_LABEL_PATTERNS for m in p.finditer(text))
    stock = tuple(m.group(0).strip() for p in _STOCK_PHRASE_PATTERNS for m in p.finditer(text))
    scaffold = tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and (
            _HEADING_RE.match(line)
            or _BULLET_RE.match(line)
            or _NUMBERED_RE.match(line)
            or _BOLD_LABEL_RE.match(line)
        )
    )
    return RitualMarkers(framework_labels=labels, stock_phrases=stock, scaffold_lines=scaffold)


def ritualistic_proxy(text: str) -> int:
    """Screening estimate of `ritualistic`, on the SAME REVERSE-CODED 1-5 scale.

    ==================================================================
    1 = no ritual (GOOD).   5 = a script with the framework showing (BAD).
    This is the one dimension in the rubric where a higher number is a
    worse response. Never average this into a composite without first
    passing it through `dimensions.to_quality`.
    ==================================================================

    Banding, from `ritual_markers`:

    * start at 1
    * +2 if any framework vocabulary or labelled section appears at all — the
      model showing its scaffolding is the strongest single signal
    * +1 if two or more scaffold lines (headings, bullets, numbered items,
      bolded labels) appear
    * +1 if two or more stock empathy stems appear
    * +1 if four or more stock stems appear, or three or more framework labels
    * clamped to [1, 5]

    A screening proxy, not a rating. It cannot see a response that is
    formulaic in rhythm without using any of these surface forms, and it will
    over-fire on a response that legitimately uses a numbered list for a
    medication schedule. Its agreement with human consensus is reported as a
    validation result, not assumed.
    """
    m = ritual_markers(text)
    points = 0
    if m.framework_labels:
        points += 2
    if len(m.scaffold_lines) >= 2:
        points += 1
    if len(m.stock_phrases) >= 2:
        points += 1
    if len(m.stock_phrases) >= 4 or len(m.framework_labels) >= 3:
        points += 1
    return max(SCALE_MIN, min(SCALE_MAX, SCALE_MIN + points))


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeterministicScores:
    """Every deterministic measurement for one response.

    Descriptors, not scores — with the documented exception of
    `ritualistic_proxy_score`, which is on the reverse-coded 1-5 rubric scale
    where 5 is worst.
    """

    scorer_version: str
    word_count: int
    sentence_count: int
    message_count: int
    question_count: int
    flesch_kincaid_grade: float
    jargon_density: float
    jargon_terms: tuple[str, ...]
    hedge_density: float
    teach_back_present: bool
    teach_back_spans: tuple[str, ...]
    pseudo_teach_back_spans: tuple[str, ...]
    ritual: RitualMarkers
    #: REVERSE-CODED: 1 = no ritual (good), 5 = fully scripted (bad).
    ritualistic_proxy_score: int


def score_text(text: str) -> DeterministicScores:
    """Run every deterministic scorer over one response. Pure function of `text`."""
    tb = teach_back_phrases(text)
    return DeterministicScores(
        scorer_version=SCORER_VERSION,
        word_count=response_length(text),
        sentence_count=sentence_count(text),
        message_count=message_count(text),
        question_count=question_count(text),
        flesch_kincaid_grade=flesch_kincaid_grade(text),
        jargon_density=jargon_density(text),
        jargon_terms=tuple(jargon_terms_found(text)),
        hedge_density=hedge_density(text),
        teach_back_present=bool(tb),
        teach_back_spans=tuple(tb),
        pseudo_teach_back_spans=tuple(pseudo_teach_back_phrases(text)),
        ritual=ritual_markers(text),
        ritualistic_proxy_score=ritualistic_proxy(text),
    )


def deterministic_rubric_score(
    generation_id: str,
    text: str,
    rater_id: str = f"deterministic-{SCORER_VERSION}",
) -> RubricScore:
    """A `rater_type='deterministic'` row for one generation.

    Only `ritualistic` is filled. The other ten dimensions require a judgement
    no counter can make, and are left `None` rather than given a fabricated
    proxy — a `rubric_score` row with an invented `de` value would silently
    contaminate every aggregate that reads the table.

    `evidence_spans` carries the longest matched ritual marker, satisfying the
    v3 §13 grounding rule with a span that is verbatim in the response. When
    nothing matched, the score is 1 and no span is recorded: an absence has no
    positive evidence, and claiming one would be a lie in a JSONB column.
    """
    markers = ritual_markers(text)
    span = markers.longest_span()
    return RubricScore(
        generation_id=generation_id,
        rater_type=RaterType.DETERMINISTIC,
        rater_id=rater_id,
        ritualistic=ritualistic_proxy(text),
        evidence_spans={"ritualistic": span} if span else {},
    )
