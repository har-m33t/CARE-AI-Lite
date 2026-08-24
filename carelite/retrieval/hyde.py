"""carelite.retrieval.hyde — Hypothetical Document Embeddings (v3 §3).

**The asymmetry HyDE attacks.** Dense retrieval compares a query vector to
document vectors, and it works best when the two live in the same language
space. Here they emphatically do not. The query side is a frightened person
speaking in the first person — "I'm scared this is cancer" — and the document
side is a methods section reporting that "clinician responses to patient
expressions of negative affect were coded using the Verona Coding
Definitions". These are different registers, different persons, different
sentence shapes. `query.py` closes part of the gap by rewriting the turn as a
framework-language *question*. HyDE closes the rest by going one step
further: generate the hypothetical guidance *passage* that would answer the
turn, and embed **that**, so the vector being matched is document-shaped from
the start.

**The generated passage is never shown to anyone and never enters a prompt.**
It is model-invented text about clinical communication with no provenance,
which makes it exactly the kind of unsourced claim this project exists not to
make. Its only use is as an embedding input. It is recorded in
`RetrievalTrace.hyde_passage` for auditability — a reviewer should be able to
see what the retriever was actually searching with — and `pipeline.py` never
passes it to `fencing.assemble` as retrieved context.

**It is embedded with `embed_document`, not `embed_query`.** A HyDE passage
stands in for a document, so it goes down the document code path. With
`bge-m3` both prefixes are empty (the carelite-index lane measured that any
query instruction prefix, even a 7-character one, collapses discrimination
between unrelated queries), so today the two paths produce identical vectors.
The distinction is kept anyway because it is the semantically correct one and
because a future embedder swap would make it numerically real again.

**Failure is silent and recorded.** If Ollama is down or the model is not
pulled, `generate_hyde_passage` returns `None`, the dense leg falls back to
the framework queries alone, and `HydeResult.available` is `False`. There is
no templated stand-in: a hand-written passage assembled from theme templates
would not be a hypothetical document, and reporting it as one in an ablation
table measuring HyDE's contribution would be a fabricated result.
"""

from __future__ import annotations

from dataclasses import dataclass

from carelite.types import EncounterPhase

__all__ = ["HYDE_SYSTEM", "HydeResult", "generate_hyde_passage"]

#: Trusted, git-tracked system template. The patient utterance never appears
#: here — it arrives fenced in the user turn via `LLMClient.chat`, which
#: raises `FencingViolation` if that rule is ever broken.
HYDE_SYSTEM = """You write short passages in the register of the peer-reviewed
literature on clinician-patient communication.

Given a patient's turn from a clinical encounter, write the paragraph that a
communication-skills guidance document or a review paper would contain about
how a clinician should handle a turn like that one.

Requirements:
- Write in the impersonal, descriptive register of a published paper or a
  practice guideline. Do not address the patient and do not address a reader.
- Use the field's own vocabulary where it applies: empathic response, emotional
  cue, patient activation, shared decision making, teach-back, comprehension
  confirmation, health literacy, plain language, relational continuity,
  communication disparities.
- Describe clinician communication behaviour only. Do not give clinical,
  diagnostic, or treatment advice, and do not name drugs or doses.
- One paragraph, 80-130 words. No headings, no lists, no citations, no preamble.

Output the paragraph and nothing else."""


@dataclass(frozen=True, slots=True)
class HydeResult:
    """The passage, or a recorded explanation of why there isn't one."""

    passage: str | None
    available: bool
    latency_ms: int = 0
    cached: bool = False
    reason: str = ""

    def __bool__(self) -> bool:
        return self.available and bool(self.passage)


#: Below this length the model has clearly not produced a paragraph (a refusal,
#: an empty string, a one-line apology), and embedding it would put noise on
#: the dense leg while reporting HyDE as having run.
MIN_PASSAGE_CHARS = 120


def generate_hyde_passage(
    utterance: str,
    *,
    client: object,
    enabled: bool = True,
    encounter_phase: EncounterPhase | None = None,
) -> HydeResult:
    """Generate the hypothetical guidance passage for one turn.

    `enabled=False` is the ablation configuration (rows R0-R3): no model call
    is made at all, so an ablation run with HyDE off costs nothing and cannot
    accidentally consult a model.
    """
    if not enabled:
        return HydeResult(passage=None, available=False, reason="hyde ablated off")

    task = "Write the guidance paragraph for the patient turn above."
    if encounter_phase is not None:
        task = (
            f"Write the guidance paragraph for the patient turn above. The turn occurs "
            f"during the {encounter_phase.value.replace('_', ' ')} phase of the encounter."
        )

    result = client.chat(  # type: ignore[attr-defined]
        system=HYDE_SYSTEM,
        task=task,
        utterance=utterance,
        num_predict=260,
    )
    if result is None:
        return HydeResult(
            passage=None,
            available=False,
            reason="generator unavailable; dense leg falls back to framework queries",
        )

    passage = _clean(result.text)
    if len(passage) < MIN_PASSAGE_CHARS:
        return HydeResult(
            passage=None,
            available=False,
            latency_ms=result.latency_ms,
            reason=f"generated passage too short ({len(passage)} chars); discarded",
        )
    return HydeResult(
        passage=passage,
        available=True,
        latency_ms=result.latency_ms,
        cached=result.cached,
    )


def _clean(text: str) -> str:
    """Strip the conversational scaffolding small models add despite the
    instruction not to ("Here is the paragraph:", markdown fences, headings)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    for lead in (
        "here is the paragraph:",
        "here's the paragraph:",
        "here is a paragraph:",
        "paragraph:",
        "guidance paragraph:",
        "passage:",
    ):
        if cleaned.casefold().startswith(lead):
            cleaned = cleaned[len(lead) :].strip()
    # Collapse to a single paragraph: the embedder gets one coherent block.
    return " ".join(cleaned.split())
