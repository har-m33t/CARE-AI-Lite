"""Scope and evidence-provenance checks on a candidate's span, finding and takeaway.

`validate.py` answers "did the paper say this?". This module answers two
questions that survive a yes to that one:

**Is the paper saying it about the right thing?** `TAXONOMY.md` draws inclusion
lines per theme, and two of them exclude a whole genre the corpus is full of:
§1 excludes "studies measuring only clinician empathy scores after a course with
no link to a patient-facing act" and §2 excludes "clinician emotion regulation
with no patient-facing consequence". Eighteen of the 33 papers are
communication-skills-training studies, so a faithful extractor reads a great many
sentences of the form *trainees' scores improved after the course*. Those
sentences are true, quotable, and about curriculum design rather than about what
to do in front of a patient. The span check cannot catch them — the span is
perfectly real — and the actionability check cannot either, because the model
writes a bedside-shaped takeaway on top of a training-shaped finding. They have
to be caught here, on the finding's own subject matter.

**Is the evidence the citing paper's own?** A systematic review's summary of
someone else's trial is a legitimate quotation, but the study it reports is not
in our corpus. Stamping that entry `strong` because *the review* is strong
conflates two different claims: it asserts that our corpus contains strong
evidence for the finding when what our corpus contains is a strong paper
mentioning it. `.claude/CLAUDE.md` asks for an anchoring paper over a general
finding; a second-hand quote is that failure inverted, and it is detectable —
these sentences carry citation markers or attributive openers, because that is
what academic prose does when it is reporting someone else.

Second-hand detection **caps the tier and flags the entry; it does not reject**.
Rejecting would discard real findings for a reason that is about attribution
rather than about truth. The cap is:

- in a systematic review or meta-analysis, no stronger than `moderate` — the
  systematic search and appraisal are real work and worth something, but the
  underlying study's design is not one we can check;
- in any other design, no stronger than `emerging` — a sentence citing other
  literature in an RCT's introduction is narrative review, and the RCT's own
  design lends it nothing.

This is the one place where two entries citing one paper may honestly carry
different tiers, and it is not the arbitrariness the tier derivation exists to
remove: for a second-hand entry the source of record is a paper outside the
corpus, so the tier is what our corpus can vouch for rather than what this paper
happens to be.

Every pattern here was written against the 127 loaded entries and checked for
what it catches *and* what it does not — a filter tuned only on its true
positives is a filter with an unmeasured false-positive rate. Where a rule could
not be made precise enough to reject on, it reports instead: see
`takeaway_span_overlap`, which is a reviewer's pointer and deliberately not a
gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from carelite.types import EvidenceTier

# ---------------------------------------------------------------------------
# Shared normalisation
# ---------------------------------------------------------------------------

#: Soft hyphens and non-breaking hyphens are how a PDF encodes a line break
#: inside a word. Removing them before matching keeps a pattern from failing on
#: `re-­admission` when it would match `re-admission`.
_SOFT = str.maketrans({"­": "", "‐": "-", "‑": "-"})  # noqa: RUF001 - real PDF glyphs


def flatten(text: str) -> str:
    """Collapse whitespace and PDF line-break hyphenation for pattern matching.

    This is *not* `spans.normalize`. That function exists to prove a quote is in
    a paper and is deliberately strict about what it folds. This one only has to
    make regexes work across a line break, so it is allowed to be blunt.
    """
    return re.sub(r"\s+", " ", text.translate(_SOFT)).strip()


@dataclass(frozen=True)
class ScopeFinding:
    """A reason a candidate falls outside what the knowledge base is for."""

    code: str
    detail: str

    def __str__(self) -> str:
        return self.detail


# ---------------------------------------------------------------------------
# Training-transfer findings (TAXONOMY.md §1)
# ---------------------------------------------------------------------------

#: Cohorts that exist *because* a training study is being described. Deliberately
#: excludes bare "physician", "nurse", "clinician" and "provider": those words
#: appear in almost every span in the corpus and using them as a training signal
#: flagged fourteen perfectly good bedside entries when this rule was first
#: written. A learner is a trainee, or it is a group defined by an intervention.
_LEARNER = re.compile(
    r"\b(trainees?|students?|residents?|interns?|learners?|attendees?|participants?"
    r"|the intervention group|the experimental group"
    r"|intervention (?:physician|clinician|doctor|nurse|provider)s?)\b",
    re.IGNORECASE,
)

_TRAINING_CONTEXT = re.compile(
    r"\b(training|courses?|curricul\w*|workshops?|programmes?|programs?|modules?|seminars?"
    r"|educational activit\w*|teaching|taught|role-?play\w*|simulat\w*|debrief\w*|didactic"
    r"|pre-?test|post-?test|pre-?post"
    r"|intervention (?:group|arm|physicians?|clinicians?))\b",
    re.IGNORECASE,
)

#: Outcomes measured on the clinician rather than on the patient or the
#: conversation: an instrument score, a rated skill, a state of the clinician.
_CLINICIAN_SIDE_OUTCOME = re.compile(
    r"\b(scores?|scales?|subscales?|ratings?"
    r"|self-?efficacy|self-?confidence|burnout|emotional exhaustion|depersonali[sz]ation"
    r"|empath(?:y|ic)|skills?|competenc\w*|coded behaviou?rs?|domains? of"
    r"|statistical significance|JSE|BEES|CARE-Measure)\b",
    re.IGNORECASE,
)

#: Anything the patient experienced, said, understood, or did. Its presence
#: vetoes the training-transfer rule: a training study that reports a *patient*
#: outcome is reporting exactly the link §1 asks for, and belongs in the base.
_PATIENT_FACING_OUTCOME = re.compile(
    r"\b(patient satisfaction|satisfaction of patients"
    r"|patients? (?:were|reported|rated|recalled|understood|felt|perceiv\w*)"
    r"|patient understanding of|comprehension|recall of|adherence|re-?admission"
    r"|quality of life|blood pressure|mortality|patient-reported|patient experience"
    r"|patient engagement|patient activation)\b",
    re.IGNORECASE,
)

#: A clinician telling a survey how good they are at something. Strong enough to
#: stand without the outcome/context conjunction: there is no reading of
#: "the lowest-rated communication skills were …" that is a finding about what
#: happens to a patient.
_SELF_RATED_SKILL = re.compile(
    r"\b((?:lowest|highest|best|worst|top)-?rated|self-?rated|self-?assessed"
    r"|self-?reported (?:skill|competenc|confidence|use)|rated (?:themselves|their own))\b",
    re.IGNORECASE,
)


def training_transfer(span: str, finding: str = "") -> ScopeFinding | None:
    """Is this a finding about how communication skill is *acquired*?

    Fires when a span reports a clinician-side measure in a training or
    learner context and reports no patient-facing outcome alongside it. The
    entry's own `finding` is searched for the context as well as the span,
    because an entry whose finding reads "intervention physicians scored
    significantly higher" is a training-transfer entry however neutrally its
    span happens to be worded.
    """
    flat_span = flatten(span)
    flat_finding = flatten(finding)

    self_rated = _SELF_RATED_SKILL.search(flat_span) or _SELF_RATED_SKILL.search(flat_finding)
    if self_rated:
        return ScopeFinding(
            "self_rated_skill",
            f"reports how clinicians rate their own skills ({self_rated.group(0)!r}) rather than "
            "an effect on a patient; TAXONOMY.md §1 excludes training-transfer findings",
        )

    outcome = _CLINICIAN_SIDE_OUTCOME.search(flat_span)
    if outcome is None:
        return None
    if _PATIENT_FACING_OUTCOME.search(flat_span):
        return None
    context = (
        _TRAINING_CONTEXT.search(flat_span)
        or _LEARNER.search(flat_span)
        or _TRAINING_CONTEXT.search(flat_finding)
        or _LEARNER.search(flat_finding)
    )
    if context is None:
        return None
    return ScopeFinding(
        "training_transfer",
        f"training-transfer finding: {context.group(0)!r} paired with the clinician-side "
        f"outcome {outcome.group(0)!r} and no patient-facing outcome in the span; "
        "TAXONOMY.md §1 excludes these",
    )


# ---------------------------------------------------------------------------
# Clinician-inward practices (TAXONOMY.md §2)
# ---------------------------------------------------------------------------

_INWARD_PRACTICE = re.compile(
    r"\b(mindfulness|meditation|resilience training|stress management|self-?care"
    r"|clinician (?:well-?being|wellness)|emotion(?:al)? regulation"
    r"|burnout (?:intervention|prevention))\b",
    re.IGNORECASE,
)

#: An act the patient could observe. Its presence means the span connects the
#: inward practice to something done in the room, which §2 does allow.
_PATIENT_FACING_ACT = re.compile(
    r"\b(said|say|says|tell|told|ask\w*|explain\w*|question\w*|respond\w*|reply|reassur\w*"
    r"|conversation|consultation|encounter|communicat\w*|disclos\w*|inform\w*|teach-?back"
    r"|acknowledg\w*|validat\w*|listen\w*|elicit\w*)\b",
    re.IGNORECASE,
)


def clinician_inward(span: str, takeaway: str = "") -> ScopeFinding | None:
    """Is this about regulating the clinician's inner state and nothing else?

    TAXONOMY.md §2 admits clinician emotion regulation only where it has a
    patient-facing consequence. A span reporting that mindfulness training moved
    a clinician-side outcome, with no communicative act anywhere in it, has no
    such consequence — and the takeaway written on top of it ("practise
    mindfulness so you notice emotional changes") is the model supplying the
    missing link rather than the paper.
    """
    flat_span = flatten(span)
    match = _INWARD_PRACTICE.search(flat_span) or _INWARD_PRACTICE.search(flatten(takeaway))
    if match is None:
        return None
    if _PATIENT_FACING_ACT.search(flat_span):
        return None
    return ScopeFinding(
        "clinician_inward",
        f"span is about a clinician-inward practice ({match.group(0)!r}) with no patient-facing "
        "act in it; TAXONOMY.md §2 excludes clinician emotion regulation with no patient-facing "
        "consequence",
    )


# ---------------------------------------------------------------------------
# Findings the span does not actually report
# ---------------------------------------------------------------------------

#: Language that asserts one thing did better than another. Narrow on purpose.
#: A first version fired on any evaluative verb — "improves", "is associated
#: with" — and flagged sixteen entries whose findings were ordinary paraphrases
#: of a descriptive span. A comparative or causal claim is different in kind:
#: it names a contrast that a Methods sentence cannot support.
_COMPARATIVE_CLAIM = re.compile(
    r"\b(resulted in|results? in|led to|leads? to|caused|significantly \w+"
    r"|compared (?:to|with)|versus|vs\.?)\b",
    re.IGNORECASE,
)

#: Anything a span could carry that counts as reporting an outcome or naming a
#: consequence. Broad on purpose: it *vetoes* the rejection, so a term missing
#: from this list costs a real entry while a term too many costs only a missed
#: catch. The causal verbs at the end are what keep the rule off ordinary
#: qualitative findings — a finding that says jargon "can lead to confusion"
#: sitting over a span that says translated words are "often leading to patient
#: confusion" is a paraphrase, not a mismatch, and three of those were flagged
#: before the veto covered causation as well as measurement.
_RESULT_EVIDENCE = re.compile(
    r"(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*%|\bp\s*[<=>]|\bOR\s*[=:]|\bMD\s*[=:]|\bCI\b"
    r"|\b(significant\w*|improv\w+|increas\w+|reduc\w+|decreas\w+|higher|lower|greater|better"
    r"|worse|differ\w*|impair\w*|modified|disparit\w*|gap|more \w+|less \w+|fewer \w+"
    r"|more likely|less likely|associated with|effect\w*|benefit\w*|gain\w*|enhanc\w+"
    r"|strengthen\w*|predict\w*|correlat\w*|superior|no difference|outcomes?"
    r"|lead\w* to|caus\w+|results? in|resulting in|produc\w+|contribut\w+ to|due to|owing to)\b",
    re.IGNORECASE,
)


def unevidenced_comparison(finding: str, span: str) -> ScopeFinding | None:
    """The finding claims A beat B; the span reports no outcome at all.

    The signature case is a Methods sentence — *"In shared decision making,
    clinicians and patients negotiated a treatment regimen"* — carrying a
    finding that says the same trial's SDM arm "resulted in significantly better
    adherence". Both halves are true of the paper; the entry is false about
    which sentence supports which claim, and a reader following the citation
    lands on prose that proves nothing.
    """
    if not _COMPARATIVE_CLAIM.search(flatten(finding)):
        return None
    if _RESULT_EVIDENCE.search(flatten(span)):
        return None
    return ScopeFinding(
        "unevidenced_comparison",
        "finding asserts a comparative or causal result that the quoted span does not report; "
        "the span describes what was done, not what happened",
    )


#: Characters that are not plausible before a number in running academic prose.
#: A `#` sitting in front of "12% in re-admission rates" is an arrow the PDF
#: extractor could not map, and reading a decrease out of it is reading a
#: direction out of a rendering failure. Genuine prefixes — parentheses,
#: brackets, comparison operators, currency, plus/minus, and the arrows
#: themselves where they survive — are excluded from the class.
_MANGLED_GLYPH = re.compile(r"(?<![\w%)\]])[#?�¡¿§¶†‡*]\s*\d")

_DIRECTION = re.compile(
    r"\b(reduc\w+|decreas\w+|lower\w*|fewer|declin\w+|drop\w*|fell|fall\w*"
    r"|increas\w+|rais\w+|ris\w+|higher|greater|improv\w+|more|less)\b|[↑↓]",
    re.IGNORECASE,
)


def unevidenced_direction(finding: str, takeaway: str, span: str) -> ScopeFinding | None:
    """The entry says a number went down; the span says it went `#`.

    Requires all three of: a mangled glyph before a number in the span, a
    directional claim in the entry, and no direction word anywhere in the span.
    Any two of those are ordinary. All three together mean the direction came
    out of a character the extractor could not read.
    """
    flat_span = flatten(span)
    glyph = _MANGLED_GLYPH.search(flat_span)
    if glyph is None:
        return None
    claim = _DIRECTION.search(flatten(finding)) or _DIRECTION.search(flatten(takeaway))
    if claim is None:
        return None
    if _DIRECTION.search(flat_span):
        return None
    return ScopeFinding(
        "unevidenced_direction",
        f"entry claims a direction ({claim.group(0)!r}) that the span carries only through the "
        f"mangled glyph {glyph.group(0)!r}; the direction is read out of an extraction artefact",
    )


def out_of_scope(span: str, finding: str, takeaway: str) -> ScopeFinding | None:
    """The first scope failure among the four, or `None` if the entry is in scope."""
    return (
        training_transfer(span, finding)
        or clinician_inward(span, takeaway)
        or unevidenced_comparison(finding, span)
        or unevidenced_direction(finding, takeaway, span)
    )


# ---------------------------------------------------------------------------
# Second-hand evidence
# ---------------------------------------------------------------------------

#: `[15]`, `[15, 16]`, `[12-14]`. Numbered references, and nothing else that
#: looks like them: the digit run is capped at three so a bracketed year cannot
#: match.
_BRACKET_CITATION = re.compile(
    r"\[\s*\d{1,3}\s*(?:[,;]\s*\d{1,3}\s*|[-–—]\s*\d{1,3}\s*)*\]"  # noqa: RUF001 - real glyphs
)

#: `(18)`, `(12, 13)` — the same thing in journals that use parentheses. Two
#: things defeat a naive version of this pattern, and both were live in the
#: corpus: an enumeration (`(1) assess the risk behavior, (2) advise change`)
#: and a count after a percentage (`66% (289) of patients`). The first is
#: handled by `_ENUMERATION` below, the second by the lookbehind here.
_PAREN_CITATION = re.compile(
    r"(?<![\d%])\s\(\s*\d{1,3}\s*(?:[,;]\s*\d{1,3}\s*|[-–—]\s*\d{1,3}\s*)*\)"  # noqa: RUF001
)

#: A span containing both `(1)` and `(2)` is listing steps, not citing sources.
_ENUMERATION = re.compile(r"\(\s*1\s*\).{0,400}?\(\s*2\s*\)", re.DOTALL)

#: "A second RCT reported that…", "One study in children with asthma found…".
#: The paper is relaying one specific study that is not itself in the corpus.
_ATTRIBUTES_ONE_STUDY = re.compile(
    r"\b(?:an?|one|another|a second|a third|the other)\s+(?:\w+[\s-]+){0,3}"
    r"(?:rcts?|randomi[sz]ed(?:\s+controlled)?\s+trials?|trials?|stud(?:y|ies)|reviews?"
    r"|meta-?analys[ei]s|surveys?|cohorts?)\b"
    r"[^.]{0,80}?\b(?:report|found|show|demonstrat|conclud|observ|note|describ|examin|assess)",
    re.IGNORECASE,
)

#: "Studies have shown…", "Research highlights that…" — a summary of a body of
#: work whose members are all outside the corpus.
_ATTRIBUTES_OTHER_STUDIES = re.compile(
    r"\b(?:studies|trials|reviews|research|literature|evidence|authors"
    r"|(?:previous|prior|earlier|other|several|many|some|two|three|four)\s+"
    r"(?:studies|trials|reviews|research|authors|works?))\b"
    r"[^.]{0,60}?\b(?:have\s+)?(?:shown|show|suggests?|reports?|reported|found|indicates?"
    r"|highlights?|demonstrates?|concludes?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SecondHand:
    """Evidence the span is relaying, not reporting."""

    kind: str
    marker: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.marker}"


def second_hand_evidence(span: str) -> SecondHand | None:
    """Is the span reporting a study other than the one it is quoted from?"""
    flat = flatten(span)

    match = _ATTRIBUTES_ONE_STUDY.search(flat)
    if match:
        return SecondHand("relays one other study", match.group(0).strip()[:80])

    match = _ATTRIBUTES_OTHER_STUDIES.search(flat)
    if match:
        return SecondHand("summarises other studies", match.group(0).strip()[:80])

    match = _BRACKET_CITATION.search(flat)
    if match:
        return SecondHand("carries a citation marker", match.group(0).strip())

    if not _ENUMERATION.search(flat):
        match = _PAREN_CITATION.search(flat)
        if match:
            return SecondHand("carries a citation marker", match.group(0).strip())

    return None


#: Designs whose whole method is to search for, appraise, and synthesise other
#: people's studies. A second-hand quotation from one of these is the paper
#: doing its job; from anything else it is a background sentence.
SYNTHESIS_DESIGNS: frozenset[str] = frozenset(
    {
        "systematic review",
        "systematic review and meta-analysis",
        "integrative systematic review",
        "meta-analysis",
    }
)


def second_hand_ceiling(design: str | None) -> EvidenceTier:
    """The strongest tier a second-hand span may carry, given who is relaying it."""
    if design in SYNTHESIS_DESIGNS:
        return EvidenceTier.MODERATE
    return EvidenceTier.EMERGING


# ---------------------------------------------------------------------------
# Takeaway support — reported, never enforced
# ---------------------------------------------------------------------------

_OVERLAP_STOPWORD_TEXT = """a an the and or but if then than that this these those of in on at to for with without
    from by as is are was were be been being it its their there here they them he she his her
    you your we our not no do does did can could should would may might will shall must have
    has had how what which who whom when where why into over under about across after before
    during between within more most less least other another same such own very too also only
    just both each any all some many few one two three per via using use used uses upon among
    while because so however thus therefore rather instead including include includes
    patient patients clinician clinicians provider providers physician physicians doctor
    doctors nurse nurses health healthcare care"""

_OVERLAP_STOPWORDS = frozenset(_OVERLAP_STOPWORD_TEXT.split())

_OVERLAP_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ied", "ers", "es", "ed", "ly", "s")


def _content_words(text: str) -> set[str]:
    out: set[str] = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z-]+", text.lower()):
        word = raw.strip("-")
        if len(word) < 3 or word in _OVERLAP_STOPWORDS:
            continue
        for suffix in _OVERLAP_SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: -len(suffix)]
                break
        out.add(word)
    return out


def takeaway_span_overlap(takeaway: str, span: str) -> float:
    """Share of the takeaway's content words that appear in the span.

    D3's guard on the equity re-extraction is that a takeaway must be supported
    by its span rather than merely adjacent to it, and this is the cheapest
    signal of adjacency: a takeaway that shares almost no vocabulary with the
    sentence it cites is usually the model writing advice *near* the evidence.

    **It is a pointer for a reader, not a gate, and it must stay one.** Measured
    over the 127 loaded entries the mean is 0.27, and the bottom of the
    distribution is a mix of genuine drift and perfectly good entries whose
    takeaway paraphrases rather than echoes — *"acknowledge the emotion before
    moving to the logical answer"* over *"these responses need to be addressed
    with empathetic responses instead of rational answers"* scores zero and is
    exactly right. Rejecting on this number would trade real entries for
    fabricated precision, which is the trade this lane exists to refuse.
    """
    words = _content_words(takeaway)
    if not words:
        return 0.0
    return len(words & _content_words(span)) / len(words)


#: Overlap below which the digest asks a reader to check the takeaway against
#: the span. Set at the 10th percentile of the loaded base rather than at a
#: level implying a defect.
LOW_OVERLAP_THRESHOLD = 0.10


__all__ = [
    "LOW_OVERLAP_THRESHOLD",
    "SYNTHESIS_DESIGNS",
    "ScopeFinding",
    "SecondHand",
    "clinician_inward",
    "flatten",
    "out_of_scope",
    "second_hand_ceiling",
    "second_hand_evidence",
    "takeaway_span_overlap",
    "training_transfer",
    "unevidenced_comparison",
    "unevidenced_direction",
]
