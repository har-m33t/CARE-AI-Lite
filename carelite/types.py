"""Shared contracts for CARELite AI.

FROZEN INTERFACE. Every agent in the fleet reads this module; only the
foundation lane may change it. If a lane needs a contract change, it stops and
requests a wave-0 amendment rather than editing this file in flight.

Nothing here imports from any other `carelite` submodule, so this module stays
importable by every lane regardless of build order.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------


class Theme(StrEnum):
    """The seven evidence-based communication categories from README.md.

    NOTE: build_plan v3 refers to ten themes. The two are unreconciled; the
    `carelite-kb` lane proposes a canonical taxonomy for sign-off in wave 2.
    Until that lands, these seven are authoritative.
    """

    EMPATHY = "empathy"
    EMOTION_RESPONSE = "emotion_response"
    ACTIVATION_SDM = "activation_sdm"
    TEACH_BACK = "teach_back"
    PLAIN_LANGUAGE = "plain_language"
    TRUST_CONTINUITY = "trust_continuity"
    EQUITY = "equity"


class EvidenceTier(StrEnum):
    """Strength of the underlying finding. Drives rerank weighting."""

    STRONG = "strong"
    MODERATE = "moderate"
    EMERGING = "emerging"


class ActionType(StrEnum):
    """What the system does with a finding."""

    DETECTION = "detection"
    GENERATION = "generation"
    REFRAMING = "reframing"


class EncounterPhase(StrEnum):
    OPENING = "opening"
    INFORMATION_GATHERING = "information_gathering"
    EXPLANATION = "explanation"
    PLANNING = "planning"
    CLOSING = "closing"


class Condition(StrEnum):
    """Experimental conditions. `A2` and `D` exist for baseline and control."""

    A = "A"  # bare model, no framework, no retrieval
    A2 = "A2"  # bare model, second family (cross-model baseline)
    B = "B"  # framework-prompted, no retrieval
    C = "C"  # framework + retrieval
    LC = "LC"  # long-context: whole corpus stuffed, no retrieval
    D = "D"  # deliberately degraded prompt (negative control, v3 §14)


class Route(StrEnum):
    """Adaptive router outcome (v3 §3)."""

    EMOTIONAL_ONLY = "emotional_only"
    INFORMATIONAL = "informational"
    MIXED = "mixed"


class CRAGGrade(StrEnum):
    """Corrective-RAG relevance verdict. `NONE` triggers Condition-B fallback."""

    RELEVANT = "relevant"
    AMBIGUOUS = "ambiguous"
    NONE = "none"


class RaterType(StrEnum):
    DETERMINISTIC = "deterministic"
    LLM_JUDGE = "llm_judge"
    HUMAN = "human"


class Split(StrEnum):
    TRAIN = "train"
    HOLDOUT = "holdout"


# Scored rubric dimensions, in schema order: NURSE (5) + Four Habits (4) + 2.
RUBRIC_DIMENSIONS: tuple[str, ...] = (
    "name",
    "understand",
    "respect",
    "support",
    "explore",
    "ib",
    "epp",
    "de",
    "ie",
    "naturalness",
    "ritualistic",
)


# ---------------------------------------------------------------------------
# Corpus and knowledge base
# ---------------------------------------------------------------------------


class Paper(BaseModel):
    paper_id: str
    doi: str | None = None
    apa_citation: str
    year: int | None = None
    design: str | None = None
    evidence_tier: EvidenceTier
    pdf_path: str | None = None


class KBEntry(BaseModel):
    """A single seven-field knowledge base record.

    `verbatim_span` is the provenance guarantee: extraction is LLM-assisted, so
    every entry must quote text that actually appears in its source paper. The
    `carelite-kb` validator rejects any entry whose span is not found verbatim.
    """

    entry_id: str
    theme: Theme
    finding: str
    practical_takeaway: str
    example_behavior: str
    evidence_tier: EvidenceTier
    action_type: ActionType
    verbatim_span: str = Field(min_length=20)
    source_paper_ids: list[str] = Field(min_length=1)
    encounter_phase: list[EncounterPhase] = Field(default_factory=list)
    nurse_component: list[str] = Field(default_factory=list)
    four_habits: list[str] = Field(default_factory=list)
    equity_relevant: bool = False

    @field_validator("source_paper_ids")
    @classmethod
    def _need_a_source(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("every KB entry must cite at least one source paper")
        return v


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    text: str
    contextual_prefix: str | None = None

    @property
    def embedding_text(self) -> str:
        """What actually gets embedded: prefix + text (Anthropic contextual retrieval)."""
        return f"{self.contextual_prefix}\n\n{self.text}" if self.contextual_prefix else self.text


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class RetrievedItem(BaseModel):
    """One retrieved unit, carrying enough provenance for the CLI evidence panel."""

    ref_id: str
    kind: str  # "kb_entry" | "chunk"
    text: str
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    graph_hops: int | None = None
    rerank_score: float | None = None
    theme: Theme | None = None
    evidence_tier: EvidenceTier | None = None
    paper_id: str | None = None
    citation: str | None = None


class RetrievalTrace(BaseModel):
    """Everything needed to explain or replay one retrieval."""

    route: Route
    queries: list[str] = Field(default_factory=list)
    hyde_passage: str | None = None
    retrieved: list[RetrievedItem] = Field(default_factory=list)
    crag_grade: CRAGGrade = CRAGGrade.NONE
    fell_back_to_b: bool = False
    latency_ms: int | None = None


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


class SafetyVerdict(BaseModel):
    """Result of an input or output safety screen.

    `allowed=False` means the turn must not proceed as-is. `redacted_text` is
    populated when the screen could sanitise rather than block.
    """

    allowed: bool
    injection_detected: bool = False
    phi_detected: bool = False
    red_flag: bool = False
    flags: list[str] = Field(default_factory=list)
    redacted_text: str | None = None
    reason: str | None = None


# ---------------------------------------------------------------------------
# Scenarios, generation, scoring
# ---------------------------------------------------------------------------


class Scenario(BaseModel):
    scenario_id: str
    text: str
    challenge_type: str
    emotion_intensity: int = Field(ge=1, le=5)
    encounter_phase: EncounterPhase
    literacy_signal: str
    equity_stratum: bool
    split: Split


class GuidanceRequest(BaseModel):
    """A patient turn arriving from the terminal. UNTRUSTED INPUT."""

    utterance: str
    condition: Condition = Condition.C
    encounter_phase: EncounterPhase | None = None
    history: list[str] = Field(default_factory=list)
    seed: int | None = None
    temperature: float = 0.7


class GuidanceResponse(BaseModel):
    """What the assistant returns. `trace` powers the CLI provenance panel."""

    text: str
    condition: Condition
    trace: RetrievalTrace | None = None
    input_safety: SafetyVerdict | None = None
    output_safety: SafetyVerdict | None = None
    self_check_passed: bool = True
    model: str | None = None
    model_digest: str | None = None
    prompt_version: str | None = None
    latency_ms: int | None = None


class Generation(BaseModel):
    """A persisted generation. The cache key is the last six fields (v3 §16)."""

    generation_id: str
    scenario_id: str
    condition: Condition
    prompt_id: str
    model: str
    model_digest: str
    seed: int
    temperature: float
    sample_idx: int
    response: str
    latency_ms: int | None = None
    created_at: datetime | None = None


class RubricScore(BaseModel):
    """One rater's scores for one generation.

    Dimensions are 1-5 Likert. `evidence_spans` maps dimension name -> verbatim
    quote from the response justifying the score (v3 §13 grounding requirement).
    """

    generation_id: str
    rater_type: RaterType
    rater_id: str
    name: int | None = None
    understand: int | None = None
    respect: int | None = None
    support: int | None = None
    explore: int | None = None
    ib: int | None = None
    epp: int | None = None
    de: int | None = None
    ie: int | None = None
    naturalness: int | None = None
    ritualistic: int | None = None
    safety_flags: list[str] = Field(default_factory=list)
    evidence_spans: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# The seam the CLI builds against
# ---------------------------------------------------------------------------


@runtime_checkable
class GuidanceEngine(Protocol):
    """Implemented by the stub in wave 1 and by `carelite.generate` in wave 3.

    The CLI depends on this Protocol and never on a concrete engine, so the two
    lanes build in parallel without touching each other's files.
    """

    def guide(self, request: GuidanceRequest) -> GuidanceResponse: ...
