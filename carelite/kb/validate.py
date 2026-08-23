"""The provenance enforcer. Nothing reaches the knowledge base except through here.

Extraction is LLM-assisted, so the knowledge base's claim to be evidence-based
rests entirely on this module. The central rule:

    An entry whose `verbatim_span` cannot be located in its source paper's
    text is a fabrication and is rejected. Not downgraded, not flagged,
    not repaired — rejected.

That rule is worth being precise about, because there is a tempting failure
mode where "validation" degrades into similarity scoring: normalise a bit more,
allow a bit more edit distance, accept at 0.85, and the check now passes
everything while still being called provenance. `carelite.kb.spans` draws the
line — it folds ligatures, quotation glyphs, dashes, hyphenated line breaks,
whitespace and case, all of which are rendering differences, and it does
nothing else. A missing clause, a smoothed-out grammatical error, two sentences
welded together: all fail, because all of them mean the paper did not say the
thing being quoted.

The validator also *strengthens* what it accepts. `spans.locate_span` returns
offsets into the original text, so a surviving entry's `verbatim_span` is
replaced with the exact source substring. What lands in the database is
therefore a literal slice of the paper rather than the model's rendering of
one, and the review digest can be trusted to show a human what the paper says.

Four further checks, in the order they run:

1. **Vocabulary** — theme, tier, and action type must be real enum members.
2. **Span** — located, and long enough to actually be evidence.
3. **Tier against design** — an entry cannot claim `strong` off a study
   protocol. The ceiling is read off the `PaperText` objects being validated
   against, whose `PaperMeta` records the design and derives the tier from it.
4. **Actionability** — the takeaway has to be something a clinician can do
   during an encounter. This is where the corpus's skew toward
   training-intervention studies gets filtered: "clinicians should receive
   communication training" is a true claim about curricula and a useless
   knowledge base entry.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from carelite.kb.extract import CandidateEntry
from carelite.kb.papers import PaperText, load_paper_texts, strongest_tier, tier_at_most
from carelite.kb.spans import NormalizedText, SpanMatch, locate_span, normalize, normalized_text
from carelite.types import ActionType, EncounterPhase, EvidenceTier, KBEntry, Theme

#: A span shorter than this is not evidence. Twelve words of ordinary academic
#: English are common enough to appear in many papers by coincidence, so a
#: match on less than that proves little; `KBEntry` itself only requires 20
#: characters, which is roughly three words.
MIN_SPAN_WORDS = 12
MIN_SPAN_CHARS = 60

#: Takeaways below this are slogans, not instructions.
MIN_TAKEAWAY_CHARS = 40


# ---------------------------------------------------------------------------
# Controlled vocabulary, tolerantly parsed
# ---------------------------------------------------------------------------

_THEME_ALIASES: dict[str, Theme] = {
    "empathy": Theme.EMPATHY,
    "emotion response": Theme.EMOTION_RESPONSE,
    "emotion recognition": Theme.EMOTION_RESPONSE,
    "emotion recognition and response": Theme.EMOTION_RESPONSE,
    "emotional response": Theme.EMOTION_RESPONSE,
    "activation sdm": Theme.ACTIVATION_SDM,
    "patient activation": Theme.ACTIVATION_SDM,
    "shared decision making": Theme.ACTIVATION_SDM,
    "patient activation and shared decision making": Theme.ACTIVATION_SDM,
    "teach back": Theme.TEACH_BACK,
    "teachback": Theme.TEACH_BACK,
    "comprehension confirmation": Theme.TEACH_BACK,
    "plain language": Theme.PLAIN_LANGUAGE,
    "plain language and information clarity": Theme.PLAIN_LANGUAGE,
    "information clarity": Theme.PLAIN_LANGUAGE,
    "trust continuity": Theme.TRUST_CONTINUITY,
    "trust": Theme.TRUST_CONTINUITY,
    "trust and relational continuity": Theme.TRUST_CONTINUITY,
    "equity": Theme.EQUITY,
    "equity aware communication": Theme.EQUITY,
}

_TIER_ALIASES: dict[str, EvidenceTier] = {
    "strong": EvidenceTier.STRONG,
    "moderate": EvidenceTier.MODERATE,
    "emerging": EvidenceTier.EMERGING,
    "weak": EvidenceTier.EMERGING,
    "low": EvidenceTier.EMERGING,
}

_ACTION_ALIASES: dict[str, ActionType] = {
    "detection": ActionType.DETECTION,
    "detect": ActionType.DETECTION,
    "generation": ActionType.GENERATION,
    "generate": ActionType.GENERATION,
    "reframing": ActionType.REFRAMING,
    "reframe": ActionType.REFRAMING,
}

_PHASE_ALIASES: dict[str, EncounterPhase] = {
    "opening": EncounterPhase.OPENING,
    "information gathering": EncounterPhase.INFORMATION_GATHERING,
    "explanation": EncounterPhase.EXPLANATION,
    "planning": EncounterPhase.PLANNING,
    "closing": EncounterPhase.CLOSING,
}


def _key(raw: str) -> str:
    return re.sub(r"[^a-z ]+", " ", raw.strip().lower().replace("_", " ")).strip()


def _coerce[T](raw: str, aliases: dict[str, T]) -> T | None:
    return aliases.get(_key(raw))


# ---------------------------------------------------------------------------
# Actionability
# ---------------------------------------------------------------------------

#: Takeaways whose grammatical subject is a training programme, an institution,
#: or a curriculum. True claims, and not knowledge-base entries: the system
#: guides a clinician mid-encounter and cannot act on "run a workshop".
_NON_ACTIONABLE_SUBJECTS = re.compile(
    r"\b("
    r"communication (skills )?training|"
    r"training (program|programme|course|curricul)|"
    r"curricul(um|a)|"
    r"medical (school|education)|"
    r"(should|must) (be )?(receive|undergo|attend|complete|implement|offer|provide|invest)"
    r"\w*\s+(training|education|a course|workshops?)|"
    r"(institution|organi[sz]ation|health system|employer|policy|policymaker)s?\b|"
    r"further research|future (studies|research)|more research is needed"
    r")",
    re.IGNORECASE,
)

#: At least one of these has to appear, so the takeaway names a communicative
#: move rather than an attitude. This is a coarse filter and is meant to be:
#: it rejects "be more empathetic" while accepting anything that says what to
#: do, and the survivors still go to a human at the review gate.
_ACTION_VERBS = re.compile(
    r"\b("
    r"ask|invite|elicit|explore|check|confirm|verify|asses|assess|"
    r"name|acknowledge|reflect|validate|normali[sz]e|"
    r"pause|wait|allow|hold|respond|"
    r"explain|re-?explain|rephras|restate|summari[sz]e|repeat|"
    r"avoid|refrain|stop|replace|reduce|limit|simplif|"
    r"offer|give|provide|share|disclose|signal|state|"
    r"involve|negotiat|agree|align|tailor|adapt|adjust|match|"
    r"screen|flag|note|record|revisit|follow up|schedul|"
    r"use|apply|prompt|encourag|support"
    r")\w*\b",
    re.IGNORECASE,
)

#: An example behaviour that is nothing but a quoted sentence is a script. The
#: project's design constraint is explicit that frameworks which become scripts
#: stop working, so an entry that ships one is rejected rather than stored.
_PURE_SCRIPT = re.compile(r'^\s*["“‘\'].{10,}["”’\']\s*\.?\s*$')  # noqa: RUF001 - matches the curly quotes a model actually emits


def takeaway_is_actionable(takeaway: str) -> tuple[bool, str | None]:
    """Can a clinician do this, in a conversation, today?"""
    text = takeaway.strip()
    if len(text) < MIN_TAKEAWAY_CHARS:
        return False, f"takeaway is {len(text)} characters; too short to instruct anything"
    if _NON_ACTIONABLE_SUBJECTS.search(text):
        return (
            False,
            "takeaway is about training, curricula, or institutions, not about what to do in an encounter",
        )
    if not _ACTION_VERBS.search(text):
        return False, "takeaway names no communicative action a clinician can take"
    return True, None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedEntry:
    """A `KBEntry` that survived, plus the proof it survived on."""

    entry: KBEntry
    paper_id: str
    span_start: int
    span_end: int
    span_was_exact: bool
    paper_sha256: str

    @property
    def entry_id(self) -> str:
        return self.entry.entry_id


@dataclass(frozen=True)
class RejectedCandidate:
    candidate: CandidateEntry
    reasons: tuple[str, ...]

    @property
    def primary_reason(self) -> str:
        return self.reasons[0]


@dataclass
class ValidationReport:
    accepted: list[ValidatedEntry] = field(default_factory=list)
    rejected: list[RejectedCandidate] = field(default_factory=list)

    @property
    def n_seen(self) -> int:
        return len(self.accepted) + len(self.rejected)

    @property
    def rejection_rate(self) -> float:
        return len(self.rejected) / self.n_seen if self.n_seen else 0.0

    def reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.rejected:
            key = _reason_bucket(r.primary_reason)
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def theme_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.accepted:
            counts[entry.entry.theme.value] = counts.get(entry.entry.theme.value, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


_REASON_BUCKETS: tuple[tuple[str, str], ...] = (
    ("not found in", "fabricated span (not in source paper)"),
    ("span is", "span too short to be evidence"),
    ("unknown theme", "unparseable theme"),
    ("unknown evidence tier", "unparseable evidence tier"),
    ("unknown action type", "unparseable action type"),
    ("claims tier", "tier overclaimed against study design"),
    ("takeaway", "takeaway not actionable"),
    ("example behaviour is a script", "example behaviour is a script"),
    ("no source paper", "no usable source paper"),
    ("duplicate", "duplicate of an accepted entry"),
)


def _reason_bucket(reason: str) -> str:
    low = reason.lower()
    for needle, bucket in _REASON_BUCKETS:
        if needle in low:
            return bucket
    return reason


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def entry_id_for(theme: Theme, paper_id: str, span: str) -> str:
    """Deterministic id: same span from the same paper is always the same entry.

    Stability matters for the load step's `ON CONFLICT` upsert and for the
    review gate — a human's sign-off on an entry must still apply after a
    re-run that produced the same entry again.
    """
    digest = hashlib.blake2b(f"{paper_id}|{normalized_text(span)}".encode(), digest_size=5)
    return f"kb-{theme.value}-{digest.hexdigest()}"


def validate_candidate(
    candidate: CandidateEntry,
    *,
    papers: dict[str, PaperText],
    normalized_cache: dict[str, NormalizedText] | None = None,
) -> ValidatedEntry | RejectedCandidate:
    """Validate one candidate. Collects every failure, not just the first.

    Reporting all reasons rather than short-circuiting is what makes the
    rejection statistics readable: an entry rejected for a fabricated span
    *and* an overclaimed tier says something different about the extractor
    than one rejected for a span alone.
    """
    reasons: list[str] = []

    theme = _coerce(candidate.theme, _THEME_ALIASES)
    if theme is None:
        reasons.append(f"unknown theme {candidate.theme!r}")

    tier = _coerce(candidate.evidence_tier, _TIER_ALIASES)
    if tier is None:
        reasons.append(f"unknown evidence tier {candidate.evidence_tier!r}")

    action = _coerce(candidate.action_type, _ACTION_ALIASES)
    if action is None:
        reasons.append(f"unknown action type {candidate.action_type!r}")

    paper_ids = [p for p in candidate.source_paper_ids if p in papers]
    if not paper_ids:
        reasons.append(f"no source paper on disk among {candidate.source_paper_ids!r}")

    # ---- the provenance check ------------------------------------------------
    match: SpanMatch | None = None
    paper: PaperText | None = None
    if paper_ids:
        paper = papers[paper_ids[0]]
        cache = normalized_cache if normalized_cache is not None else {}
        normalised = cache.get(paper.paper_id)
        if normalised is None:
            normalised = normalize(paper.text)
            cache[paper.paper_id] = normalised
        match = locate_span(candidate.verbatim_span, paper.text, normalized_document=normalised)
        if match is None:
            reasons.append(
                f"verbatim_span not found in {paper.paper_id}: {candidate.verbatim_span[:120]!r}"
            )
        else:
            words = len(match.source_text.split())
            if words < MIN_SPAN_WORDS or len(match.source_text) < MIN_SPAN_CHARS:
                reasons.append(
                    f"span is {words} words / {len(match.source_text)} characters; "
                    f"below the {MIN_SPAN_WORDS}-word evidence floor"
                )

    # ---- tier against design -------------------------------------------------
    if tier is not None and paper_ids:
        # Read the ceiling off the PaperText objects actually being validated
        # against, not off the module-level table. A paper missing from the
        # table would otherwise yield no ceiling and pass this check by
        # default, which is the wrong direction to fail in.
        metas = [papers[p].meta for p in paper_ids]
        ceiling = strongest_tier(m.evidence_tier for m in metas if m is not None)
        if ceiling is not None and not tier_at_most(tier, ceiling):
            designs = ", ".join(papers[p].design or "unknown design" for p in paper_ids)
            reasons.append(
                f"claims tier {tier.value!r} but the strongest source design "
                f"({designs}) supports at most {ceiling.value!r}"
            )

    # ---- actionability -------------------------------------------------------
    ok, why = takeaway_is_actionable(candidate.practical_takeaway)
    if not ok and why:
        reasons.append(why)

    if _PURE_SCRIPT.match(candidate.example_behavior.strip()):
        reasons.append("example behaviour is a script to recite rather than a described move")

    if reasons:
        return RejectedCandidate(candidate=candidate, reasons=tuple(reasons))

    assert theme is not None and tier is not None and action is not None
    assert match is not None and paper is not None

    phases = [p for p in (_coerce(x, _PHASE_ALIASES) for x in candidate.encounter_phase) if p]

    entry = KBEntry(
        entry_id=entry_id_for(theme, paper.paper_id, match.source_text),
        theme=theme,
        finding=candidate.finding.strip(),
        practical_takeaway=candidate.practical_takeaway.strip(),
        example_behavior=candidate.example_behavior.strip(),
        evidence_tier=tier,
        action_type=action,
        # The authoritative form: an exact substring of the paper, not the
        # model's rendering of one.
        verbatim_span=match.source_text,
        source_paper_ids=paper_ids,
        encounter_phase=phases,
        equity_relevant=bool(candidate.equity_relevant) or theme is Theme.EQUITY,
    )
    return ValidatedEntry(
        entry=entry,
        paper_id=paper.paper_id,
        span_start=match.start,
        span_end=match.end,
        span_was_exact=match.exact,
        paper_sha256=paper.text_sha256,
    )


def validate_candidates(
    candidates: Iterable[CandidateEntry],
    *,
    papers: dict[str, PaperText] | None = None,
) -> ValidationReport:
    """Validate a batch, dropping duplicates of already-accepted entries.

    Overlapping extraction windows mean the same finding is often proposed
    twice with the same span. The second copy is a duplicate, not a second
    piece of evidence, and counting it would inflate the entry total with no
    new support behind it.
    """
    papers = papers if papers is not None else load_paper_texts()
    normalized_cache: dict[str, NormalizedText] = {}
    report = ValidationReport()
    seen_ids: set[str] = set()
    seen_spans: set[tuple[str, str]] = set()

    for candidate in candidates:
        outcome = validate_candidate(candidate, papers=papers, normalized_cache=normalized_cache)
        if isinstance(outcome, RejectedCandidate):
            report.rejected.append(outcome)
            continue

        span_key = (outcome.paper_id, normalized_text(outcome.entry.verbatim_span))
        if outcome.entry_id in seen_ids or span_key in seen_spans:
            report.rejected.append(
                RejectedCandidate(
                    candidate=candidate,
                    reasons=(f"duplicate of accepted entry {outcome.entry_id}",),
                )
            )
            continue

        seen_ids.add(outcome.entry_id)
        seen_spans.add(span_key)
        report.accepted.append(outcome)

    return report


def format_report(report: ValidationReport) -> str:
    """A short, honest summary. Used by the CLI entry point and the digest header."""
    lines = [
        f"{len(report.accepted)} accepted, {len(report.rejected)} rejected "
        f"of {report.n_seen} candidates ({report.rejection_rate:.0%} rejected).",
        "",
        "Rejections by reason:",
    ]
    for reason, count in report.reason_counts().items():
        lines.append(f"  {count:4d}  {reason}")
    lines += ["", "Accepted entries by theme:"]
    for theme, count in report.theme_counts().items():
        lines.append(f"  {count:4d}  {theme}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from carelite.kb.extract import CACHE_PATH, read_cache

    ap = argparse.ArgumentParser(description="Validate cached KB candidates.")
    ap.add_argument("--cache", default=str(CACHE_PATH))
    args = ap.parse_args(list(argv) if argv is not None else None)

    candidates = [c for r in read_cache(args.cache) for c in r.candidates]
    report = validate_candidates(candidates)
    print(format_report(report))
    return 0


__all__ = [
    "MIN_SPAN_CHARS",
    "MIN_SPAN_WORDS",
    "RejectedCandidate",
    "ValidatedEntry",
    "ValidationReport",
    "entry_id_for",
    "format_report",
    "takeaway_is_actionable",
    "validate_candidate",
    "validate_candidates",
]


if __name__ == "__main__":
    raise SystemExit(main())
