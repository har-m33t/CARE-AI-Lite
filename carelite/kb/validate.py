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

Five further checks, in the order they run:

1. **Vocabulary** — theme, tier, and action type must be real enum members.
2. **Span** — located, and long enough to actually be evidence.
3. **Tier from design** — the evidence tier is **derived from the recorded
   study design, not read from the model**, and this is a correction rather
   than the original behaviour. The first version treated the design tier as a
   *ceiling*: an overclaim was lowered, an underclaim was left alone. That is
   half a check, and the half it omitted let the defect back in through the
   other door. Four papers ended up carrying entries at more than one tier and
   one carried entries at all three, because wherever the model happened to say
   `emerging` about a randomised trial, `emerging` survived. Tier then recorded
   the model's mood rather than the study behind it — precisely the property
   this check exists to remove. `README.md` defines evidence strength as a
   property of the source, so two entries citing one paper cannot honestly
   carry different strengths.

   So the derivation runs in both directions, and **both values are kept**:
   `ValidatedEntry.claimed_tier` holds what the model asserted, the stored
   entry holds the design-derived tier, and the digest prints them side by
   side. Correcting rather than rejecting is still right for the same reason it
   always was — a missing span has no right answer to substitute, an
   ill-judged tier does — but "correct" now means "make it match the design",
   not "cap it".

   One exception, and it is principled rather than a loophole: a span that is
   **relaying someone else's study** is capped by
   `carelite.kb.scope.second_hand_ceiling`, because the source of record for
   that claim is a paper outside the corpus and the citing paper's design says
   nothing about it. Those entries are flagged as second-hand wherever the tier
   is shown, so the within-paper variation is explained rather than arbitrary.
4. **Scope** — `carelite.kb.scope` rejects candidates that `TAXONOMY.md`'s
   inclusion criteria exclude (training-transfer findings, clinician-inward
   practices) and candidates whose finding claims something the span does not
   report. These are entries the span check cannot catch, because their spans
   are perfectly real.
5. **Actionability** — the takeaway has to be something a clinician can do
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
from carelite.kb.frameworks import map_components
from carelite.kb.papers import PaperText, load_paper_texts, strongest_tier, tier_at_most
from carelite.kb.scope import (
    ScopeFinding,
    SecondHand,
    out_of_scope,
    second_hand_ceiling,
    second_hand_evidence,
)
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

#: Phrasings whose verb is a state of mind rather than a move. These are the
#: takeaways the filter most needs to catch, because they read like advice and
#: cannot be acted on: "be mindful of the empathy gap" tells a clinician to
#: hold an idea, not to do anything, and an entry built on one would reach
#: generation as an instruction no output could satisfy. Checked before the
#: verb whitelist, and independently of it, so that widening the whitelist
#: cannot quietly let an attitude back in.
_ATTITUDE_NOT_ACTION = re.compile(
    r"\b("
    r"be mindful|be aware|be conscious|be cognizant|bear in mind|keep in mind|"
    r"should consider|consider the|remember that|realis|realiz|"
    r"focus on (building|developing|maintaining|being|having|understanding|engaging|"
    r"establishing|creating|improving)|"
    r"strive to|aim to|try to|make an effort|"
    r"(understand|recogni[sz]e|appreciate) (the importance|that .{0,40} (is|are) important)"
    r")\b",
    re.IGNORECASE,
)

#: The second shape, and the one the D3 equity re-extraction produced. An
#: instruction can name an **outcome to bring about** rather than an act to
#: perform, and that reads far more like advice than "be mindful" does:
#: *"proactively work to bridge the empathy gap"*, *"identify and address
#: barriers to ensure equitable empathy"*, *"provide consistent empathetic
#: communication to all patients regardless of background"*. Every one of those
#: clears the verb whitelist — `address`, `provide`, `identify` are all real
#: verbs — while telling a clinician nothing they could do in the next sentence
#: of a conversation, and no observer could say whether it happened.
#:
#: This is worth naming precisely because of where it came from. D3 revised the
#: extraction prompt to ask, where a passage reports a disparity, for the
#: **compensating move** rather than the awareness, and warned that a model told
#: to find compensating moves will find them whether or not the passage supports
#: one. What the revised prompt actually produced was the same awareness
#: statement with an active verb bolted on the front. The rule is not
#: equity-specific and is not applied to equity alone: an aspiration is
#: unactionable wherever it appears. Measured against the 115 loaded entries at
#: the time it was added, it matched exactly one — itself an equity aspiration
#: that had survived the original gate.
#:
#: **Deliberately not extended further, and the reason is the limit of this
#: whole check.** The re-run also produced *"engage in more consistent and
#: attentive communication to ensure these patients feel heard and valued"*,
#: which is the same aspiration and is not matched here. The obvious extension —
#: rejecting `ensure <someone> feels <something>` — was tested against the
#: loaded base and matched three entries that are perfectly good: *"Ask for
#: permission before moving to difficult topics **to ensure the patient feels in
#: control**"* leads with a real move and appends a purpose clause. The
#: difference between an aspiration and a move with a purpose clause is which
#: one is the main verb, and no pattern this module can hold separates them
#: reliably. Three real entries to catch one aspiration is the trade this lane
#: exists to refuse, so the extension was not made. What follows is that the
#: actionability gate cannot be the whole of D3's guard: the rest is the
#: individual reading of every equity entry that D3 requires.
_ASPIRATION_NOT_ACTION = re.compile(
    r"\b("
    r"works? towards?|work to|seek to|endeavou?r|commit to|"
    r"ensure (equitable|equal|consistent|fair|that all)|"
    r"to all patients|"
    r"regardless of (their )?(background|race|ethnicit|socioeconomic|status|SES)|"
    # A comparative with no referent, attached to a disposition rather than to
    # anything said: "engage in **more consistent** and attentive
    # communication". More consistent than what? The clinician cannot check it
    # mid-encounter and an observer cannot score it, which puts it in the same
    # category as "be mindful". Deliberately restricted to dispositions — "give
    # **more detailed** information" modifies a thing that actually gets said
    # and stays acceptable. Zero matches on the loaded base when added.
    r"(more|greater|increased|better) "
    r"(consistent|attentive|effective|empathetic|empathic|thorough|careful|sensitive|responsive)"
    r")\w*\b",
    re.IGNORECASE,
)

#: At least one of these has to appear, so the takeaway names a communicative
#: move rather than an attitude. This is a coarse filter and is meant to be:
#: it rejects "be more empathetic" while accepting anything that says what to
#: do, and the survivors still go to a human at the review gate.
#:
#: Every alternative here is a **stem**, not a lemma, because the pattern
#: appends `\w*`. That distinction was a live bug: written as lemmas, `use`,
#: `provide`, `explore`, `acknowledge` and `validate` matched "used" and
#: "provides" but never "using", "providing", "exploring", "acknowledging" or
#: "validating" — the final `e` is dropped before `-ing`, so `\w*` had nothing
#: to match against. The `-ing` form is exactly how an imperative takeaway is
#: usually phrased, so the whitelist was rejecting a large share of perfectly
#: actionable entries for a reason that had nothing to do with them.
_ACTION_VERBS = re.compile(
    r"\b("
    # eliciting and checking
    r"ask|invit|elicit|explor|check|confirm|verif|assess|screen|probe|"
    # naming and responding to what was said
    r"nam(e|ing)|acknowledg|reflect|validat|normali[sz]|empathi[sz]|listen|"
    r"identif|notic|observ|monitor|detect|flag|"
    # holding the floor
    r"paus|wait|allow|hold|respond|react|"
    # making the message land
    r"explain|re-?explain|rephras|restat|summari[sz]|repeat|clarif|teach|"
    r"describ|illustrat|demonstrat|translat|frame|fram(e|ing)|present|"
    # taking things out
    r"avoid|refrain|stop|replac|reduc|limit|simplif|shorten|"
    # putting things in
    r"offer|giv(e|ing)|provid|shar(e|ing)|disclos|signal|state|brief|introduc|greet|"
    # deciding together
    r"involv|negotiat|agree|align|tailor|adapt|adjust|match|incorporat|"
    r"establish|collaborat|invit|elicit|"
    # carrying it forward
    r"note|record|revisit|follow up|schedul|arrang|request|document|"
    # general
    r"us(e|ing)|appl(y|ying)|prompt|encourag|support|address|discuss|pivot"
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
    if _ATTITUDE_NOT_ACTION.search(text):
        return (
            False,
            "takeaway asks the clinician to hold an attitude or bear something in mind, "
            "not to make a move a listener could observe",
        )
    if _ASPIRATION_NOT_ACTION.search(text):
        return (
            False,
            "takeaway names an outcome to bring about rather than an act to perform; "
            "no observer could say whether the clinician did it",
        )
    if not _ACTION_VERBS.search(text):
        return False, "takeaway names no communicative action a clinician can take"
    return True, None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedEntry:
    """A `KBEntry` that survived, plus the proof it survived on.

    `claimed_tier` is what the extraction model asserted; `entry.evidence_tier`
    is what the recorded study design supports. They differ whenever the model
    misjudged the design in either direction, and keeping both is what makes the
    correction auditable instead of silent.
    """

    entry: KBEntry
    paper_id: str
    span_start: int
    span_end: int
    span_was_exact: bool
    paper_sha256: str
    claimed_tier: EvidenceTier
    design_tier: EvidenceTier | None = None
    span_match_via: str = "normalized"
    second_hand: SecondHand | None = None

    @property
    def entry_id(self) -> str:
        return self.entry.entry_id

    @property
    def tier_corrected(self) -> bool:
        """Did the stored tier end up different from what the model claimed?"""
        return self.claimed_tier is not self.entry.evidence_tier

    @property
    def tier_direction(self) -> str:
        """`"up"`, `"down"`, or `"unchanged"` — which way the correction went."""
        if not self.tier_corrected:
            return "unchanged"
        return "up" if tier_at_most(self.claimed_tier, self.entry.evidence_tier) else "down"


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

    @property
    def tier_corrected(self) -> list[ValidatedEntry]:
        """Accepted entries whose stored tier differs from the model's claim."""
        return [e for e in self.accepted if e.tier_corrected]

    def tier_correction_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.tier_corrected:
            key = f"{e.claimed_tier.value} -> {e.entry.evidence_tier.value} ({e.tier_direction})"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    @property
    def second_hand(self) -> list[ValidatedEntry]:
        """Accepted entries whose span relays a study outside the corpus."""
        return [e for e in self.accepted if e.second_hand is not None]

    def tier_consistency(self) -> dict[str, set[str]]:
        """Tiers held per source paper, ignoring second-hand entries.

        Any paper appearing here with more than one tier is a bug: tier is
        derived from design, and a paper has one design. Second-hand entries are
        excluded because their tier is deliberately not this paper's.
        """
        out: dict[str, set[str]] = {}
        for e in self.accepted:
            if e.second_hand is not None:
                continue
            out.setdefault(e.paper_id, set()).add(e.entry.evidence_tier.value)
        return out

    def span_match_counts(self) -> dict[str, int]:
        """How much normalisation each accepted span needed to be located."""
        counts: dict[str, int] = {}
        for e in self.accepted:
            counts[e.span_match_via] = counts.get(e.span_match_via, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


#: Needles must be **distinctive**, not merely present. These are matched in
#: order against the rejection reason, so a short generic needle earlier in the
#: list silently absorbs every later reason that happens to contain it: `"span
#: is"` was matching "span is about a clinician-inward practice" and reporting
#: it as a too-short span, which made one bucket over-count and hid another
#: entirely. Prefer the phrase that appears in exactly one reason.
_REASON_BUCKETS: tuple[tuple[str, str], ...] = (
    ("not found in", "fabricated span (not in source paper)"),
    ("evidence floor", "span too short to be evidence"),
    ("unknown theme", "unparseable theme"),
    ("unknown evidence tier", "unparseable evidence tier"),
    ("unknown action type", "unparseable action type"),
    # Ordered before the training-transfer needle: a self-rated-skill reason
    # cites the same TAXONOMY.md §1 clause, so the broader needle would swallow
    # it and the two would report as one bucket.
    ("rate their own skills", "out of scope: self-rated clinician skill"),
    ("training-transfer finding", "out of scope: training-transfer finding"),
    ("clinician-inward practice", "out of scope: clinician-inward practice"),
    ("comparative or causal result", "finding not reported by its span"),
    ("mangled glyph", "direction read out of an extraction artefact"),
    ("takeaway asks the clinician to hold an attitude", "takeaway is an attitude, not an action"),
    ("outcome to bring about", "takeaway is an aspiration, not an action"),
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


def candidates_from_cache(
    cache_path: str | None = None, *, prompt_versions: Sequence[str] | None = None
) -> list[CandidateEntry]:
    """Read cached candidates, optionally restricted to particular prompt versions.

    The filter exists so an **experimental prompt variant cannot silently join
    the knowledge base**. The cache is append-only and the validator reads all
    of it, so the moment a variant's first window lands, its candidates are in
    the next load — before anyone has looked at whether the variant works. That
    is the wrong default for a change whose whole risk is that it produces
    plausible-looking entries: `DECISIONS.md` D3 approved the equity variant
    *with a guard*, and a variant's output should reach the base when the guard
    has been applied, not when the inference finishes.
    """
    from carelite.kb.extract import CACHE_PATH, read_cache

    wanted = set(prompt_versions) if prompt_versions else None
    return [
        c
        for r in read_cache(cache_path or CACHE_PATH)
        for c in r.candidates
        if wanted is None or r.prompt_version in wanted
    ]


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

    # ---- tier from design ----------------------------------------------------
    # Derived, not capped, and corrected rather than rejected. See the module
    # docstring: the design is recorded, the tier it supports is derivable, and
    # the entry's span, theme, finding and takeaway are untouched by the model's
    # misjudgment of it.
    design_tier: EvidenceTier | None = None
    design: str | None = None
    if paper_ids:
        # Read the design off the PaperText objects actually being validated
        # against, not off the module-level table. A paper missing from the
        # table would otherwise yield no tier at all and leave the model's claim
        # standing unchecked, which is the wrong direction to fail in.
        metas = [papers[p].meta for p in paper_ids]
        design_tier = strongest_tier(m.evidence_tier for m in metas if m is not None)
        primary_meta = papers[paper_ids[0]].meta
        design = primary_meta.design if primary_meta else None

    claimed_tier = tier
    if design_tier is not None:
        tier = design_tier

    # A relayed result is evidence from a paper we do not hold. The citing
    # paper's design cannot vouch for a study it merely mentions, so the tier
    # drops to what this corpus can actually stand behind.
    second_hand: SecondHand | None = None
    if match is not None:
        second_hand = second_hand_evidence(match.source_text)
        if second_hand is not None and tier is not None:
            relayed_ceiling = second_hand_ceiling(design)
            if not tier_at_most(tier, relayed_ceiling):
                tier = relayed_ceiling

    # ---- scope ---------------------------------------------------------------
    # Real span, real paper, wrong subject matter. These are rejected rather
    # than corrected because there is nothing to correct to: an entry about
    # what a course did to trainees' scores is not an entry about what to do in
    # front of a patient, and no field can be rewritten to make it one.
    scope: ScopeFinding | None = None
    if match is not None:
        scope = out_of_scope(match.source_text, candidate.finding, candidate.practical_takeaway)
        if scope is not None:
            reasons.append(str(scope))

    # ---- actionability -------------------------------------------------------
    ok, why = takeaway_is_actionable(candidate.practical_takeaway)
    if not ok and why:
        reasons.append(why)

    if _PURE_SCRIPT.match(candidate.example_behavior.strip()):
        reasons.append("example behaviour is a script to recite rather than a described move")

    if reasons:
        return RejectedCandidate(candidate=candidate, reasons=tuple(reasons))

    assert theme is not None and tier is not None and action is not None
    assert claimed_tier is not None
    assert match is not None and paper is not None

    phases = [p for p in (_coerce(x, _PHASE_ALIASES) for x in candidate.encounter_phase) if p]
    components = map_components(candidate.practical_takeaway, candidate.example_behavior)

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
        # The behavior-to-framework mapping, derived here rather than asked of
        # the model. `carelite.kb.frameworks` reads the act the entry
        # prescribes and matches it against the rubric lane's own definitions,
        # so the component an entry instantiates is checkable against the
        # entry's own words instead of being a second thing to trust the
        # extractor about. An entry that instantiates none of the nine gets
        # empty lists, which is what the coverage report is for.
        nurse_component=list(components.nurse),
        four_habits=list(components.four_habits),
        equity_relevant=bool(candidate.equity_relevant) or theme is Theme.EQUITY,
    )
    return ValidatedEntry(
        entry=entry,
        paper_id=paper.paper_id,
        span_start=match.start,
        span_end=match.end,
        span_was_exact=match.exact,
        paper_sha256=paper.text_sha256,
        claimed_tier=claimed_tier,
        design_tier=design_tier,
        span_match_via=match.via,
        second_hand=second_hand,
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
    corrections = report.tier_correction_counts()
    if corrections:
        lines += [
            "",
            f"Evidence tier derived from study design ({len(report.tier_corrected)} "
            f"of {len(report.accepted)} accepted entries differ from the model's claim):",
        ]
        for change, count in corrections.items():
            lines.append(f"  {count:4d}  {change}")
        lines.append("  (the model's claim is kept on each entry and printed in the review digest)")

    inconsistent = {p: t for p, t in report.tier_consistency().items() if len(t) > 1}
    if inconsistent:
        lines += ["", "WARNING  papers carrying entries at more than one tier:"]
        for paper_id, tiers in sorted(inconsistent.items()):
            lines.append(f"  {paper_id}: {', '.join(sorted(tiers))}")

    if report.second_hand:
        lines += [
            "",
            f"{len(report.second_hand)} accepted entr(ies) quote a span that relays another study;",
            "  their tier is capped at what this corpus can vouch for, not at the citing "
            "paper's design.",
        ]

    lines += ["", "Span located after:"]
    for via, count in report.span_match_counts().items():
        lines.append(f"  {count:4d}  {via}")

    lines += ["", "Accepted entries by theme:"]
    for theme, count in report.theme_counts().items():
        lines.append(f"  {count:4d}  {theme}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from carelite.kb.extract import CACHE_PATH

    ap = argparse.ArgumentParser(description="Validate cached KB candidates.")
    ap.add_argument("--cache", default=str(CACHE_PATH))
    ap.add_argument(
        "--prompt-version",
        action="append",
        dest="prompt_versions",
        help="only validate candidates extracted with this prompt version; repeatable",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    candidates = candidates_from_cache(args.cache, prompt_versions=args.prompt_versions)
    report = validate_candidates(candidates)
    print(format_report(report))
    return 0


__all__ = [
    "MIN_SPAN_CHARS",
    "MIN_SPAN_WORDS",
    "RejectedCandidate",
    "ValidatedEntry",
    "ValidationReport",
    "candidates_from_cache",
    "entry_id_for",
    "format_report",
    "takeaway_is_actionable",
    "validate_candidate",
    "validate_candidates",
]


if __name__ == "__main__":
    raise SystemExit(main())
