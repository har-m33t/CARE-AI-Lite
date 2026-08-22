"""`StubEngine` — a deterministic fixture implementation of `GuidanceEngine`.

Ships in wave 1 so the whole CLI (`chat`, `ask`, `retrieve`) is testable
before `carelite-orchestrator` lands the real pipeline in wave 3. Every
response is derived deterministically from the request (utterance + seed),
so the same input always renders the same evidence panel — useful both for
demos and for CLI tests that assert on rendered output.

Nothing here is real evidence. Every fixture citation is tagged `[STUB]` so
it can never be mistaken for a corpus finding once the real engine is wired
in.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from carelite.types import (
    Condition,
    CRAGGrade,
    EvidenceTier,
    GuidanceRequest,
    GuidanceResponse,
    RetrievalTrace,
    RetrievedItem,
    Route,
    SafetyVerdict,
    Theme,
)

# ---------------------------------------------------------------------------
# Fixture knowledge base — shape mirrors `carelite.types.KBEntry`, tagged as
# stub data throughout so it is never confused with the real corpus.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FixtureEntry:
    entry_id: str
    theme: Theme
    evidence_tier: EvidenceTier
    finding: str
    practical_takeaway: str
    paper_id: str
    citation: str


_FIXTURE_KB: tuple[_FixtureEntry, ...] = (
    _FixtureEntry(
        "kb-stub-001",
        Theme.EMPATHY,
        EvidenceTier.STRONG,
        "Explicit empathic statements preceding information delivery reduced perceived "
        "clinician coldness in recorded encounters.",
        "Name the emotion in one sentence before moving to information.",
        "paper-stub-01",
        "Okafor & Reyes (2021). Empathic sequencing in oncology consults. [STUB]",
    ),
    _FixtureEntry(
        "kb-stub-002",
        Theme.EMOTION_RESPONSE,
        EvidenceTier.STRONG,
        "Unaddressed emotional cues were associated with lower reported trust at 3-month "
        "follow-up.",
        "Respond to the emotional cue before answering the informational question.",
        "paper-stub-02",
        "Vance, T. (2019). Missed empathic opportunities in primary care. [STUB]",
    ),
    _FixtureEntry(
        "kb-stub-003",
        Theme.TEACH_BACK,
        EvidenceTier.STRONG,
        "Teach-back closing significantly improved recall of the care plan at discharge.",
        "Close explanations with a teach-back check: ask the patient to restate the plan.",
        "paper-stub-03",
        "Chen & Ibarra (2020). Teach-back and 30-day recall. [STUB]",
    ),
    _FixtureEntry(
        "kb-stub-004",
        Theme.PLAIN_LANGUAGE,
        EvidenceTier.MODERATE,
        "Replacing medical jargon with plain-language equivalents improved comprehension "
        "scores across literacy strata.",
        "Swap jargon for plain-language equivalents; define any term you cannot avoid.",
        "paper-stub-04",
        "Marsh, D. (2018). Health literacy and word choice. [STUB]",
    ),
    _FixtureEntry(
        "kb-stub-005",
        Theme.ACTIVATION_SDM,
        EvidenceTier.MODERATE,
        "Offering an explicit choice point increased patient-reported involvement in the decision.",
        "Offer a concrete choice ('we could do A or B') rather than a single recommendation.",
        "paper-stub-05",
        "Ibrahim & Novak (2022). Shared decision-making prompts in ambulatory visits. [STUB]",
    ),
    _FixtureEntry(
        "kb-stub-006",
        Theme.TRUST_CONTINUITY,
        EvidenceTier.MODERATE,
        "References to prior visits and continuity of the relationship correlated with "
        "higher reported trust.",
        "Reference the ongoing relationship ('since we last talked...') where it is true.",
        "paper-stub-06",
        "Sato, K. (2017). Relational continuity cues in follow-up encounters. [STUB]",
    ),
    _FixtureEntry(
        "kb-stub-007",
        Theme.EQUITY,
        EvidenceTier.EMERGING,
        "Preliminary evidence that empathic response rates are lower for patients with "
        "limited English proficiency, independent of interpreter use.",
        "Do not let interpreter-mediated exchanges shorten the empathic portion of the response.",
        "paper-stub-07",
        "Delgado, R. (2023). Equity gaps in empathic response, interpreted encounters. "
        "[STUB, preprint]",
    ),
    _FixtureEntry(
        "kb-stub-008",
        Theme.EMOTION_RESPONSE,
        EvidenceTier.EMERGING,
        "Single-arm pilot suggesting brief silence after a distress cue is read as "
        "attentiveness rather than avoidance.",
        "A brief pause after a distress cue can read as attentiveness; do not rush to fill it.",
        "paper-stub-08",
        "Huang, W. (2024). Pause duration and perceived attentiveness, pilot n=12. "
        "[STUB, preprint]",
    ),
)

_EMOTION_WORDS = (
    "scared",
    "worried",
    "afraid",
    "upset",
    "angry",
    "frustrat",
    "anxious",
    "overwhelm",
    "hopeless",
    "cry",
    "terrified",
    "nervous",
)
_INFO_WORDS = (
    "why",
    "what",
    "how",
    "when",
    "explain",
    "understand",
    "mean",
    "test",
    "procedure",
    "surgery",
    "medication",
    "diagnosis",
    "results",
)

_FRAMEWORK_LINE = {
    Route.EMOTIONAL_ONLY: (
        "Lead with an empathic reflection before any information; the patient's emotional "
        "state is the primary signal here."
    ),
    Route.INFORMATIONAL: (
        "Lead with plain-language information and close with a teach-back check before moving on."
    ),
    Route.MIXED: (
        "Acknowledge the emotion in one sentence, then offer the requested information in "
        "plain language, closing with a teach-back check."
    ),
}

_INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard your instructions",
    "disregard previous",
    "you are now",
    "system prompt",
    "reveal your prompt",
)
_RED_FLAG_PATTERNS = (
    "kill myself",
    "want to die",
    "end my life",
    "suicide",
    "hurt myself",
    "hurt someone",
)
_PHI_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")  # SSN-shaped
_FALLBACK_TRIGGERS = ("no evidence", "unsupported", "obscure condition", "made up disease")


def _stable_unit(*parts: str) -> float:
    """Deterministic pseudo-random float in [0, 1) derived from `parts`."""
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _digest6(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:6]


def _route_for(text: str) -> Route:
    lowered = text.lower()
    has_emotion = any(w in lowered for w in _EMOTION_WORDS)
    has_info = any(w in lowered for w in _INFO_WORDS)
    if has_emotion and has_info:
        return Route.MIXED
    if has_emotion:
        return Route.EMOTIONAL_ONLY
    if has_info:
        return Route.INFORMATIONAL
    return Route.MIXED


def _screen_input(utterance: str) -> SafetyVerdict:
    lowered = utterance.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lowered:
            return SafetyVerdict(
                allowed=False,
                injection_detected=True,
                flags=["injection_detected"],
                reason=(
                    "Prompt injection pattern detected in the patient utterance "
                    f"('{pattern}'); turn blocked before reaching the model."
                ),
            )
    for pattern in _RED_FLAG_PATTERNS:
        if pattern in lowered:
            return SafetyVerdict(
                allowed=False,
                red_flag=True,
                flags=["red_flag"],
                reason=(
                    "Red-flag content detected (possible self-harm or crisis disclosure). "
                    "This tool does not provide crisis guidance — escalate to a supervising "
                    "clinician or crisis line immediately."
                ),
            )
    phi_match = _PHI_PATTERN.search(utterance)
    if phi_match:
        redacted = _PHI_PATTERN.sub("[REDACTED-PHI]", utterance)
        return SafetyVerdict(
            allowed=True,
            phi_detected=True,
            flags=["phi_detected"],
            redacted_text=redacted,
            reason="Apparent identifier (SSN-shaped) redacted before processing.",
        )
    return SafetyVerdict(allowed=True)


def _screen_output(text: str, condition: Condition) -> SafetyVerdict:
    if condition == Condition.D:
        return SafetyVerdict(
            allowed=True,
            flags=["degraded_prompt_control"],
            reason=(
                "Condition D is a deliberately degraded negative control (v3 §14); output is "
                "expected to score poorly so the rubric's ability to distinguish it from B can "
                "be checked."
            ),
        )
    return SafetyVerdict(allowed=True)


def _select_fixture_items(utterance: str, n: int = 4) -> list[_FixtureEntry]:
    ranked = sorted(_FIXTURE_KB, key=lambda e: _stable_unit(utterance, e.entry_id), reverse=True)
    return list(ranked[:n])


def _build_trace(request: GuidanceRequest) -> RetrievalTrace:
    utterance = request.utterance
    route = _route_for(utterance)
    lowered = utterance.lower()
    forced_fallback = any(t in lowered for t in _FALLBACK_TRIGGERS)

    entries = _select_fixture_items(utterance, n=4)
    retrieved: list[RetrievedItem] = []
    for rank, entry in enumerate(entries, start=1):
        base = _stable_unit(utterance, entry.entry_id, "score")
        score = round(base * 0.35, 4) if forced_fallback else round(0.45 + base * 0.5, 4)
        dense_rank = rank
        lexical_rank = rank + (1 if _stable_unit(utterance, entry.entry_id, "lex") > 0.5 else -1)
        lexical_rank = max(1, lexical_rank)
        graph_hops = 1 if _stable_unit(utterance, entry.entry_id, "graph") > 0.6 else None
        rerank_score = None if forced_fallback else round(score + base * 0.1, 4)
        retrieved.append(
            RetrievedItem(
                ref_id=entry.entry_id,
                kind="kb_entry",
                text=entry.practical_takeaway,
                score=score,
                dense_rank=dense_rank,
                lexical_rank=lexical_rank,
                graph_hops=graph_hops,
                rerank_score=rerank_score,
                theme=entry.theme,
                evidence_tier=entry.evidence_tier,
                paper_id=entry.paper_id,
                citation=entry.citation,
            )
        )

    top_score = max((item.score for item in retrieved), default=0.0)
    if forced_fallback or top_score < 0.5:
        crag_grade = CRAGGrade.NONE
        fell_back = True
    elif top_score < 0.65:
        crag_grade = CRAGGrade.AMBIGUOUS
        fell_back = False
    else:
        crag_grade = CRAGGrade.RELEVANT
        fell_back = False

    queries = [
        utterance,
        f"{route.value} framework guidance for: {utterance}",
        f"evidence for responding to a patient who says: {utterance}",
    ]
    hyde_passage = (
        "A clinician might respond by first naming the patient's emotion, then offering a "
        f'plain-language answer relevant to: "{utterance}".'
    )

    return RetrievalTrace(
        route=route,
        queries=queries,
        hyde_passage=hyde_passage,
        retrieved=retrieved,
        crag_grade=crag_grade,
        fell_back_to_b=fell_back,
        latency_ms=int(40 + _stable_unit(utterance, "latency") * 160),
    )


def _response_text(request: GuidanceRequest, trace: RetrievalTrace | None) -> str:
    condition = request.condition
    utterance = request.utterance

    if condition in (Condition.A, Condition.A2):
        return "I hear you. Let's talk through what's on your mind and take it from there."
    if condition == Condition.D:
        return "It's fine, don't worry about it. Let's just move on."
    if condition == Condition.LC:
        route = _route_for(utterance)
        return f"[long-context, no retrieval] {_FRAMEWORK_LINE[route]}"
    if condition == Condition.B or trace is None:
        route = _route_for(utterance)
        return _FRAMEWORK_LINE[route]

    # Condition C: framework + retrieval.
    framework_line = _FRAMEWORK_LINE[trace.route]
    if trace.fell_back_to_b:
        return (
            "No sufficiently relevant evidence was retrieved for this utterance "
            f"(CRAG grade: {trace.crag_grade.value}), so this falls back to the framework "
            f"alone. {framework_line}"
        )
    top = max(trace.retrieved, key=lambda i: i.score, default=None)
    if top is None:
        return framework_line
    return (
        f"{framework_line} This aligns with {top.theme.value if top.theme else 'the'} "
        f"guidance ({top.evidence_tier.value if top.evidence_tier else 'unrated'} evidence): "
        f"{top.text}"
    )


class StubEngine:
    """Deterministic fixture `GuidanceEngine`. See module docstring."""

    def guide(self, request: GuidanceRequest) -> GuidanceResponse:
        input_safety = _screen_input(request.utterance)
        digest = _digest6(request.utterance, request.condition.value, str(request.seed))

        if not input_safety.allowed:
            return GuidanceResponse(
                text="",
                condition=request.condition,
                trace=None,
                input_safety=input_safety,
                output_safety=None,
                self_check_passed=False,
                model="stub-engine",
                model_digest=f"stub-{digest}",
                prompt_version="stub-fixture-v1",
                latency_ms=5,
            )

        trace = _build_trace(request) if request.condition == Condition.C else None
        text = _response_text(request, trace)
        output_safety = _screen_output(text, request.condition)

        return GuidanceResponse(
            text=text,
            condition=request.condition,
            trace=trace,
            input_safety=input_safety,
            output_safety=output_safety,
            self_check_passed=request.condition != Condition.D,
            model="stub-engine",
            model_digest=f"stub-{digest}",
            prompt_version="stub-fixture-v1",
            latency_ms=(trace.latency_ms if trace else 20),
        )
