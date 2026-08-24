"""carelite.retrieval.crag — the corrective-RAG relevance gate. **Non-negotiable** (v3 §3).

Grades the retrieved context for one turn. On `CRAGGrade.NONE` the pipeline
discards the context, sets `fell_back_to_b`, and Condition C behaves exactly
like Condition B for that turn.

**Why this module protects the study rather than the code.** Condition C is
"framework + retrieval". On a turn the corpus cannot address, retrieval still
returns its four best-scoring passages — retrieval always returns something.
Injecting those into the prompt adds noise to an answer that Condition B
would have given cleanly, so C can score *below* B, and the headline
comparison the whole experiment exists to make is confounded. The gate is
what makes "C ≥ B" a claim about evidence rather than about luck.

--------------------------------------------------------------------------
The measurements that determined this module's design
--------------------------------------------------------------------------

The obvious implementation is a score threshold, and
`settings.retrieval.crag_relevance_threshold` (0.5) invites one. Measured
against the live 475-chunk corpus, **every score this pipeline produces fails
to separate on-domain turns from off-domain ones**, and the reason is
structural rather than a matter of picking a better number:

1. *Dense cosine of the framework queries.* Separates cleanly when the query
   faithfully describes the turn — 15 off-domain probes topped out at 0.513
   while 12 on-domain probes bottomed out at 0.587. But `query.py` falls back
   to generic framework queries when no theme is detected, which is exactly
   what happens on an off-domain turn. The generic query is on-topic by
   construction, so it retrieves good communication material and scores like
   an on-domain turn. Measured: all seven off-domain turns produced an
   identical top-1 rerank score of 0.8219.
2. *Cross-encoder on the raw utterance.* Collapses to near-zero for
   everything — a first-person emotional turn is not a "query" in the sense
   the reranker was trained on. On-domain minimum 0.0004 sat *below* the
   off-domain maximum of 0.0017. No separation.
3. *Dense cosine of the HyDE passage.* Fails for the most instructive reason
   of the three. HyDE's job is to translate any turn into corpus register,
   and it does that job faithfully even for "how do I replace an oil filter"
   — it produced "When a patient introduces a topic outside the clinical
   scope, the clinician must...", which is genuine, on-topic communication
   guidance. On-domain [0.674, 0.750] against off-domain [0.659, 0.712]:
   fully overlapping.

The pattern: **every component upstream of this gate is designed to find the
closest communication guidance for any input whatsoever.** That is correct
behaviour for retrieval and useless for grading, because by the time the gate
runs, the patient's actual turn has been rewritten into the corpus's own
register. A gate built from those scores is circular — it asks whether the
corpus can answer a question the corpus itself wrote.

So the grader is a **semantic judgment**, which is what CRAG is in the
literature (Yan et al. 2024 use a prompted retrieval evaluator, not a
threshold). `LLMGrader` asks the judge-family model whether each passage
actually addresses *the patient's turn as the patient said it*, and it is the
default.

**Determinism, given an LLM sits inside the independent variable.** The
concern in v3 §14 is real: a route or grade that differs between runs makes
Condition C a moving target. Three things pin it down — temperature 0, a
fixed seed from `settings.experiment.base_seed`, and a persistent cache keyed
by the full prompt hash, so a re-run of a scored cell replays the identical
grade rather than re-asking. Grading costs one short call per turn against a
generation step that costs far more.

**`ScoreGrader` remains as the offline fallback** for when no model is
reachable. It is honestly labelled: it can tell a *retrieval failure* (the
legs returned nothing, or everything scored near the floor) from a success,
but per the measurements above it **cannot** detect an off-domain turn whose
generic retrieval looks healthy. `GradeReport.reason` says so explicitly
whenever it is the grader that ran, so the limitation shows up in the trace
rather than living only in this docstring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from carelite.config import get_settings
from carelite.types import CRAGGrade, RetrievedItem

__all__ = [
    "CRAG_SYSTEM",
    "GradeReport",
    "LLMGrader",
    "ScoreGrader",
    "grade_context",
]


@dataclass(frozen=True, slots=True)
class GradeReport:
    """The verdict, the per-item detail, and how it was reached."""

    grade: CRAGGrade
    relevant_ids: tuple[str, ...] = ()
    per_item: tuple[tuple[str, float], ...] = ()
    grader: str = ""
    reason: str = ""
    latency_ms: int = 0

    @property
    def should_fall_back(self) -> bool:
        """The single question `pipeline.py` asks. Only `NONE` falls back —
        `AMBIGUOUS` keeps its context, matching the contract documented on
        `carelite.types.CRAGGrade`."""
        return self.grade is CRAGGrade.NONE


# ---------------------------------------------------------------------------
# The LLM grader (default)
# ---------------------------------------------------------------------------

#: Trusted, git-tracked. The patient turn and every retrieved passage arrive
#: fenced in the user turn via `LLMClient.chat`; neither is ever formatted
#: into this string.
CRAG_SYSTEM = """You are grading whether retrieved reference passages are
useful for responding to one patient turn from a clinical encounter.

The passages come from a corpus of peer-reviewed papers about clinician-patient
communication: empathy, responding to emotion, shared decision making,
confirming comprehension (teach-back), plain language, trust and continuity,
and equity in communication.

For each numbered passage decide whether it would genuinely help a clinician
respond well to THIS patient turn.

Judge usefulness for this specific turn, not general quality. Two failure modes
to watch for, because passages are selected by similarity search and something
is always returned:

- The passage is sound communication advice but has nothing to do with what
  this patient actually said. Mark it not useful.
- The patient turn is not about clinical communication at all (a car repair
  question, a drug-dosing question, a coding question). Then NO passage in this
  corpus is useful, however well written. Mark them all not useful.

Reply with JSON only:
{"passages": [{"id": <number>, "useful": true|false}], "overall": "relevant"|"ambiguous"|"none"}

Use "relevant" when at least one passage genuinely addresses the turn,
"ambiguous" when the passages are only tangentially related, and "none" when
no passage addresses the turn or the turn is outside the corpus's subject
matter entirely."""

_CRAG_SCHEMA = {
    "type": "object",
    "properties": {
        "passages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "useful": {"type": "boolean"},
                },
                "required": ["id", "useful"],
            },
        },
        "overall": {"type": "string", "enum": ["relevant", "ambiguous", "none"]},
    },
    "required": ["passages", "overall"],
}


@dataclass
class LLMGrader:
    """Prompted retrieval evaluator. Returns `None` if the model is unreachable
    or answers unparseably, so `grade_context` can fall back to scores."""

    client: object

    name: str = "llm"

    def grade(self, utterance: str, items: list[RetrievedItem]) -> GradeReport | None:
        if not items:
            return GradeReport(
                grade=CRAGGrade.NONE,
                grader=self.name,
                reason="no items retrieved",
            )

        numbered = [(f"PASSAGE_{n}", f"[{n}] {item.text}") for n, item in enumerate(items, start=1)]
        result = self.client.chat(  # type: ignore[attr-defined]
            system=CRAG_SYSTEM,
            task=(
                "Grade each numbered passage above for usefulness in responding to the "
                "patient turn. Reply with the JSON object only."
            ),
            utterance=utterance,
            extra_untrusted=numbered,
            json_schema=_CRAG_SCHEMA,
            num_predict=400,
        )
        if result is None or not result.text:
            return None
        try:
            obj = json.loads(result.text)
            overall = CRAGGrade(str(obj["overall"]).strip().casefold())
            flags = {int(p["id"]): bool(p["useful"]) for p in obj.get("passages", [])}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

        relevant = tuple(
            item.ref_id for n, item in enumerate(items, start=1) if flags.get(n, False)
        )
        # Reconcile a model that says "relevant" while flagging nothing useful,
        # or "none" while flagging something: the per-passage answers are the
        # more concrete judgment, so they win.
        if relevant and overall is CRAGGrade.NONE:
            overall = CRAGGrade.AMBIGUOUS
        if not relevant and overall is CRAGGrade.RELEVANT:
            overall = CRAGGrade.AMBIGUOUS

        return GradeReport(
            grade=overall,
            relevant_ids=relevant,
            per_item=tuple(
                (item.ref_id, 1.0 if flags.get(n, False) else 0.0)
                for n, item in enumerate(items, start=1)
            ),
            grader=self.name,
            reason=f"{len(relevant)}/{len(items)} passages judged useful for this turn",
            latency_ms=result.latency_ms,
        )


# ---------------------------------------------------------------------------
# The score grader (offline fallback)
# ---------------------------------------------------------------------------

#: Linear calibration anchors for raw `bge-m3` cosine, from the measurement in
#: the module docstring: 0.44 is the mean top-1 cosine of 15 off-domain probe
#: queries and 0.66 the mean top-1 of 12 on-domain ones. Their midpoint, 0.55,
#: falls in the centre of the empirically observed gap [0.513, 0.587], so the
#: default threshold of 0.5 lands on a measured boundary rather than a chosen
#: one. Anchors, not a fit: no per-query tuning happens anywhere.
DENSE_NULL_ANCHOR = 0.44
DENSE_SIGNAL_ANCHOR = 0.66


def calibrate_cosine(cosine: float) -> float:
    """Map a raw cosine onto [0, 1] using the measured anchors."""
    span = DENSE_SIGNAL_ANCHOR - DENSE_NULL_ANCHOR
    return max(0.0, min(1.0, (cosine - DENSE_NULL_ANCHOR) / span))


@dataclass
class ScoreGrader:
    """Threshold grader over whatever calibrated score the items carry.

    Prefers `rerank_score` (sigmoid-activated, genuinely in [0, 1]) and falls
    back to a calibrated fusion score. **Read the module docstring before
    trusting it**: it detects an empty or floor-scoring retrieval, not an
    off-domain turn.
    """

    threshold: float = 0.0
    ambiguous_ratio: float = 0.6
    name: str = "score"

    _limitation: str = field(
        default=(
            "score-threshold grader: detects retrieval failure but cannot detect an "
            "off-domain turn whose generic retrieval scores normally (see crag.py)"
        ),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.threshold:
            self.threshold = get_settings().retrieval.crag_relevance_threshold

    def grade(self, utterance: str, items: list[RetrievedItem]) -> GradeReport:
        del utterance  # this grader is, definitionally, blind to the turn
        if not items:
            return GradeReport(grade=CRAGGrade.NONE, grader=self.name, reason="no items retrieved")

        scored: list[tuple[str, float]] = []
        for item in items:
            if item.rerank_score is not None:
                scored.append((item.ref_id, float(item.rerank_score)))
            else:
                scored.append((item.ref_id, calibrate_cosine(float(item.score))))

        best = max(s for _, s in scored)
        relevant = tuple(ref for ref, s in scored if s >= self.threshold)
        floor = self.threshold * self.ambiguous_ratio

        if relevant:
            grade = CRAGGrade.RELEVANT
            reason = f"{len(relevant)}/{len(items)} items at or above {self.threshold:.2f}"
        elif best >= floor:
            grade = CRAGGrade.AMBIGUOUS
            reason = (
                f"best score {best:.3f} is below the {self.threshold:.2f} bar but above "
                f"the {floor:.3f} floor"
            )
        else:
            grade = CRAGGrade.NONE
            reason = f"best score {best:.3f} is below the {floor:.3f} floor"

        return GradeReport(
            grade=grade,
            relevant_ids=relevant,
            per_item=tuple(scored),
            grader=self.name,
            reason=f"{reason}; {self._limitation}",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def grade_context(
    utterance: str,
    items: list[RetrievedItem],
    *,
    enabled: bool = True,
    use_llm: bool = True,
    client: object | None = None,
    threshold: float | None = None,
    ambiguous_ratio: float = 0.6,
) -> GradeReport:
    """Grade `items` for `utterance`.

    `enabled=False` is the R0-R6 and R8 ablation configuration: the gate does
    not run and everything is reported `RELEVANT`, which is what lets the
    table attribute a C-vs-B difference to the gate itself. Shipping with the
    gate off is a study-invalidating configuration — see the module docstring.
    """
    if not enabled:
        return GradeReport(
            grade=CRAGGrade.RELEVANT,
            relevant_ids=tuple(i.ref_id for i in items),
            grader="disabled",
            reason="CRAG ablated off; all context passed through ungraded",
        )

    if use_llm and client is not None:
        report = LLMGrader(client=client).grade(utterance, items)
        if report is not None:
            return report
        # Model unreachable or unparseable: fall through to scores rather than
        # failing the turn, and the fallback labels its own limitation.

    grader = ScoreGrader(threshold=threshold or 0.0, ambiguous_ratio=ambiguous_ratio)
    return grader.grade(utterance, items)
