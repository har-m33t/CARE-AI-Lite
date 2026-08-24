"""carelite.retrieval — the hybrid retrieval pipeline (v3 §4).

    from carelite.retrieval import retrieve, preset

    trace = retrieve("I'm scared this is cancer.")          # full stack
    trace = retrieve(utterance, flags=preset("R0"))          # dense-only baseline

`carelite.retrieval.rerank` is deliberately **not** re-exported here.
Importing it pulls sentence-transformers and torch into the process, and the
lane contract is that an ablation row with reranking switched off never pays
that cost. `pipeline.py` imports it lazily, inside the branch that needs it.
"""

from __future__ import annotations

from carelite.retrieval.crag import CRAG_SYSTEM, GradeReport, grade_context
from carelite.retrieval.flags import PRESETS, RetrievalFlags, preset
from carelite.retrieval.hyde import HydeResult, generate_hyde_passage
from carelite.retrieval.pipeline import RetrievalResult, retrieve, retrieve_detailed
from carelite.retrieval.query import MetadataFilter, QuerySet, build_queries, detect_themes
from carelite.retrieval.router import RouteDecision, classify, route_turn

__all__ = [
    "CRAG_SYSTEM",
    "PRESETS",
    "GradeReport",
    "HydeResult",
    "MetadataFilter",
    "QuerySet",
    "RetrievalFlags",
    "RetrievalResult",
    "RouteDecision",
    "build_queries",
    "classify",
    "detect_themes",
    "generate_hyde_passage",
    "grade_context",
    "preset",
    "retrieve",
    "retrieve_detailed",
    "route_turn",
]
