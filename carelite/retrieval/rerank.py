"""carelite.retrieval.rerank — cross-encoder reranking to the top `rerank_top_n`.

RRF fuses ranks without ever looking at the query and a document *together*.
A cross-encoder does exactly that: it encodes the pair jointly and scores the
match, which is why it consistently corrects the ordering that first-stage
retrieval produces. `BAAI/bge-reranker-v2-m3` (from `settings.models.reranker`)
is the model; sentence-transformers is the runtime.

**Lazy loading is a hard requirement, not an optimisation.** The lane brief
says the reranker "must not be imported when the reranker is ablated out",
and this matters for real reasons: importing `sentence_transformers` pulls in
torch, which costs seconds of startup and hundreds of megabytes of RSS in a
process that also runs an interactive terminal app. Ablation rows R0-R4 do not
rerank, and running them must not pay that cost. The import therefore lives
inside `CrossEncoderReranker._load`, nothing at module scope touches torch,
and `pipeline.py` does not even import this module unless `flags.rerank` is
set. `tests/unit/retrieval/test_rerank.py` asserts the module-not-imported
property directly, because a stray top-level import would be silently
"correct" in every other respect.

**Score calibration.** `bge-reranker-v2-m3` has a single output label, and
sentence-transformers applies sigmoid to it, so `predict` returns values in
(0, 1) that behave like calibrated relevance. Measured on a live pair from
this corpus: a genuinely relevant chunk scores ~0.29 and an off-domain one
~1.6e-05 — five orders of magnitude apart. This is what makes
`settings.retrieval.crag_relevance_threshold` meaningful as an absolute
number in `crag.py`, and it is why the reranker, when present, is the
preferred CRAG signal over calibrated cosine.

**Evidence-tier weighting.** The brief requires that strong evidence outrank
emerging evidence "at comparable relevance". The operative words are *at
comparable relevance*: tier must break near-ties, not override the model. So
the weight is multiplicative and gentle (`strong` 1.00, `moderate` 0.92,
`emerging` 0.85), which reorders a 0.61-vs-0.60 pair but leaves a
0.90-vs-0.20 pair alone. An unknown or missing tier is treated as `moderate`
rather than penalised to the floor, since "we have not tiered this paper yet"
is not evidence of weakness.

*Current data note:* all 33 papers in the corpus are presently tiered
`emerging`, so tier weighting is a uniform scaling — a no-op on ordering —
against today's database. The logic is exercised by fixtures in the unit
tests rather than by live data, and will start to bite when the corpus lane
assigns real tiers.

**Unavailability is degradation, not failure.** sentence-transformers and
torch are an optional extra (`pip install carelite[rerank]`). If they are
absent, or the model weights cannot be fetched, `rerank` returns the input
order unchanged with `available=False` recorded, rather than taking down a
bedside interface over a ranking refinement.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from carelite.config import get_settings
from carelite.types import EvidenceTier, RetrievedItem

__all__ = [
    "TIER_WEIGHTS",
    "CrossEncoderReranker",
    "RerankResult",
    "apply_tier_weight",
    "is_available",
]

#: Multiplicative, deliberately gentle. See the module docstring.
TIER_WEIGHTS: dict[EvidenceTier, float] = {
    EvidenceTier.STRONG: 1.00,
    EvidenceTier.MODERATE: 0.92,
    EvidenceTier.EMERGING: 0.85,
}

#: Applied when `evidence_tier` is None. Not the floor: an untiered paper is
#: unknown, not weak.
UNKNOWN_TIER_WEIGHT = TIER_WEIGHTS[EvidenceTier.MODERATE]

#: Cross-encoders truncate; this is per query+document pair.
MAX_PAIR_LENGTH = 512


@dataclass(frozen=True, slots=True)
class RerankResult:
    items: list[RetrievedItem]
    available: bool
    model: str = ""
    latency_ms: int = 0
    tier_weighted: bool = False
    reason: str = ""


def apply_tier_weight(score: float, tier: EvidenceTier | None) -> float:
    return score * (TIER_WEIGHTS.get(tier, UNKNOWN_TIER_WEIGHT) if tier else UNKNOWN_TIER_WEIGHT)


def is_available() -> bool:
    """True if sentence-transformers can be imported. Does **not** load the
    model or import torch eagerly — `importlib.util.find_spec` only resolves
    the module location."""
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None


@dataclass
class CrossEncoderReranker:
    """Lazily-loaded `BAAI/bge-reranker-v2-m3` cross-encoder.

    One instance holds one loaded model; `pipeline.py` keeps a process-wide
    singleton (`get_reranker`) so a CLI session pays the load cost once.
    """

    model_name: str = ""
    max_length: int = MAX_PAIR_LENGTH
    _model: Any = field(default=None, init=False, repr=False)
    _failed: str = field(default="", init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_name = self.model_name or get_settings().models.reranker.tag

    def _load(self) -> Any:
        """Import and construct on first real use. Everything torch-shaped is
        confined to this method — see the module docstring."""
        if self._model is not None or self._failed:
            return self._model
        with self._lock:
            if self._model is not None or self._failed:
                return self._model
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                self._failed = (
                    f"sentence-transformers not installed ({exc}); "
                    f"install the 'rerank' extra to enable cross-encoder reranking"
                )
                return None
            try:
                self._model = CrossEncoder(self.model_name, max_length=self.max_length)
            except Exception as exc:  # weights unavailable, no network, OOM
                self._failed = f"could not load {self.model_name}: {type(exc).__name__}: {exc}"
                return None
            return self._model

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float] | None:
        model = self._load()
        if model is None:
            return None
        raw = model.predict(pairs)
        return [float(x) for x in raw]

    def rerank(
        self,
        query: str,
        items: list[RetrievedItem],
        *,
        top_n: int | None = None,
        tier_weighting: bool = True,
    ) -> RerankResult:
        """Score every item against `query` and return the best `top_n`.

        `query` should be the *framework-language* query rather than the raw
        patient utterance: the cross-encoder is matching against
        document-register text, so the same register asymmetry that motivates
        HyDE applies here too.
        """
        import time

        n = top_n if top_n is not None else get_settings().retrieval.rerank_top_n
        if not items:
            return RerankResult(items=[], available=True, model=self.model_name)

        started = time.monotonic()
        scores = self.score_pairs([(query, item.text) for item in items])
        if scores is None:
            return RerankResult(
                items=items[:n],
                available=False,
                reason=self._failed or "reranker unavailable",
            )
        latency_ms = int((time.monotonic() - started) * 1000)

        scored: list[RetrievedItem] = []
        for item, raw in zip(items, scores, strict=True):
            final = apply_tier_weight(raw, item.evidence_tier) if tier_weighting else raw
            scored.append(
                item.model_copy(update={"rerank_score": float(raw), "score": float(final)})
            )
        # Ties break on ref_id so a rerun cannot silently reorder equal scores.
        scored.sort(key=lambda i: (-i.score, i.ref_id))
        return RerankResult(
            items=scored[:n],
            available=True,
            model=self.model_name,
            latency_ms=latency_ms,
            tier_weighted=tier_weighting,
        )


_SINGLETON: CrossEncoderReranker | None = None
_SINGLETON_LOCK = threading.Lock()


def get_reranker() -> CrossEncoderReranker:
    """Process-wide reranker, so an interactive session loads the model once."""
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                _SINGLETON = CrossEncoderReranker()
    return _SINGLETON
