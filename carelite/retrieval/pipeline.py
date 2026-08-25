"""carelite.retrieval.pipeline — the one entry point, returning a populated `RetrievalTrace`.

    from carelite.retrieval import retrieve

    trace = retrieve("I'm scared this is cancer.", encounter_phase=EncounterPhase.EXPLANATION)

Composition order follows the v3 §4 diagram exactly:

    router → query construction (+ HyDE) → dense | lexical | graph → RRF
           → cross-encoder rerank → CRAG gate → trace

Every stage reads its switch from a `RetrievalFlags`, so an ablation row is a
different `flags` argument and never a different code path. `flags=preset("R0")`
runs the dense-only baseline through this same function that `preset("R9")`
runs the full stack through — which is what makes the ablation table a
measurement rather than a comparison of nine separately-written pipelines.

**Two invariants the rest of the system relies on.**

*First: `retrieved` is empty exactly when no evidence may be used.* Both the
emotional-only route and a CRAG `NONE` verdict clear it. A downstream lane
that does `fencing.assemble(retrieved=trace.retrieved, ...)` therefore cannot
inject evidence into a turn that was not supposed to have any, even if it
never inspects `route` or `crag_grade`. Discarded candidates remain on
`RetrievalResult.rejected` for the ablation harness and for debugging, which
is deliberately *not* on the trace: `RetrievalTrace` is what flows toward the
prompt, so putting rejected text on it would reintroduce exactly the leak this
invariant closes.

*Second: `route` and `fell_back_to_b` distinguish the two reasons for having
no evidence.* `route == EMOTIONAL_ONLY` with `fell_back_to_b == False` means
retrieval was deliberately skipped because the turn asked for presence, not
information. `fell_back_to_b == True` means retrieval ran and the corpus could
not address the turn. The CLI renders these differently and they mean opposite
things about the corpus, so they are never collapsed into one flag.

**Lazy reranker import.** `carelite.retrieval.rerank` is imported inside the
function, only when `flags.rerank` is set, so sentence-transformers and torch
are never loaded for a run that does not rerank. See `rerank.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from carelite.config import get_settings
from carelite.retrieval import crag as crag_mod
from carelite.retrieval import hyde as hyde_mod
from carelite.retrieval import query as query_mod
from carelite.retrieval import router as router_mod
from carelite.retrieval.flags import RetrievalFlags
from carelite.retrieval.fusion import dense_search, graph_search, lexical_search, rrf_fuse
from carelite.types import CRAGGrade, EncounterPhase, RetrievalTrace, RetrievedItem, Route

__all__ = ["RetrievalResult", "retrieve", "retrieve_detailed"]


@dataclass
class RetrievalResult:
    """The trace plus the diagnostics that must not travel with it.

    `RetrievalTrace` is the frozen contract and is what heads toward the
    prompt. Everything here that is *not* the trace — rejected candidates,
    per-leg notes, component availability — exists for the ablation harness
    and for `--explain`, and is kept off the trace on purpose (see the module
    docstring's first invariant).
    """

    trace: RetrievalTrace
    flags: RetrievalFlags
    rejected: list[RetrievedItem] = field(default_factory=list)
    route_decision: router_mod.RouteDecision | None = None
    grade_report: crag_mod.GradeReport | None = None
    queryset: query_mod.QuerySet | None = None
    hyde_result: hyde_mod.HydeResult | None = None
    leg_notes: list[str] = field(default_factory=list)
    rerank_available: bool | None = None
    stage_ms: dict[str, int] = field(default_factory=dict)
    leg_hits: dict[str, int] = field(default_factory=dict)
    """Hits contributed per `(leg, target)`, before fusion.

    Kept because "CRAG rejected this turn" and "the legs never found anything
    to reject" look identical from the trace alone, and they are opposite
    diagnoses. `n_candidates` below distinguishes them at the fusion stage;
    this attributes a thin candidate set to the leg responsible."""

    @property
    def n_candidates(self) -> int:
        """Candidates that reached the CRAG gate.

        The number that separates "the gate made a semantic call on a healthy
        set" from "the gate was handed almost nothing". `retrieved` alone
        cannot show this: a fallback empties it by design."""
        return len(self.trace.retrieved) + len(self.rejected)

    @property
    def retrieved(self) -> list[RetrievedItem]:
        return self.trace.retrieved


def _empty_trace(route: Route, queries: tuple[str, ...] = ()) -> RetrievalTrace:
    return RetrievalTrace(
        route=route,
        queries=list(queries),
        retrieved=[],
        crag_grade=CRAGGrade.NONE,
        fell_back_to_b=False,
    )


def retrieve(
    utterance: str,
    *,
    encounter_phase: EncounterPhase | None = None,
    flags: RetrievalFlags | None = None,
    embedder: Any | None = None,
    generator: Any | None = None,
    grader_client: Any | None = None,
    reranker: Any | None = None,
) -> RetrievalTrace:
    """Run retrieval for one turn and return its trace. See `retrieve_detailed`."""
    return retrieve_detailed(
        utterance,
        encounter_phase=encounter_phase,
        flags=flags,
        embedder=embedder,
        generator=generator,
        grader_client=grader_client,
        reranker=reranker,
    ).trace


def retrieve_detailed(
    utterance: str,
    *,
    encounter_phase: EncounterPhase | None = None,
    flags: RetrievalFlags | None = None,
    embedder: Any | None = None,
    generator: Any | None = None,
    grader_client: Any | None = None,
    reranker: Any | None = None,
) -> RetrievalResult:
    """Full pipeline with diagnostics.

    Every collaborator is injectable so that a unit test can drive the whole
    composition with fakes and no live service, and so the ablation harness
    can share one embedder and one loaded cross-encoder across a hundred
    turns instead of reconstructing them per call.
    """
    flags = flags or RetrievalFlags()
    settings = get_settings()
    started = time.monotonic()
    stage_ms: dict[str, int] = {}
    leg_notes: list[str] = []
    leg_hits: dict[str, int] = {}

    # -- 1. adaptive router ------------------------------------------------
    t0 = time.monotonic()
    decision = router_mod.route_turn(
        utterance,
        enabled=flags.router,
        use_llm=flags.use_llm_router,
        client=generator,
    )
    stage_ms["router"] = int((time.monotonic() - t0) * 1000)

    if not decision.should_retrieve:
        # Emotional-only: retrieval is skipped entirely. Not a fallback —
        # `fell_back_to_b` stays False. See the module docstring.
        trace = _empty_trace(decision.route)
        trace.latency_ms = int((time.monotonic() - started) * 1000)
        return RetrievalResult(
            trace=trace,
            flags=flags,
            route_decision=decision,
            stage_ms=stage_ms,
            leg_notes=["retrieval skipped: emotional-only turn"],
        )

    # -- 2. query construction --------------------------------------------
    t0 = time.monotonic()
    queryset = query_mod.build_queries(
        utterance,
        encounter_phase=encounter_phase,
        expand=flags.query_expansion,
        n_queries=flags.n_framework_queries,
    )
    metadata = queryset.metadata if flags.metadata_filter else query_mod.MetadataFilter()
    stage_ms["query"] = int((time.monotonic() - t0) * 1000)

    # -- 3. HyDE -----------------------------------------------------------
    t0 = time.monotonic()
    hyde_result = hyde_mod.HydeResult(passage=None, available=False, reason="hyde ablated off")
    if flags.hyde:
        if generator is None:
            from carelite.retrieval.llm import LLMClient

            generator = LLMClient()
        hyde_result = hyde_mod.generate_hyde_passage(
            utterance, client=generator, enabled=True, encounter_phase=encounter_phase
        )
        if not hyde_result and hyde_result.reason:
            leg_notes.append(f"hyde: {hyde_result.reason}")
    stage_ms["hyde"] = int((time.monotonic() - t0) * 1000)

    # -- 4. the three legs -------------------------------------------------
    t0 = time.monotonic()
    ranked_lists = []

    if flags.dense:
        if embedder is None:
            from carelite.index.embed import OllamaEmbedder

            embedder = OllamaEmbedder()
        dense_queries = list(queryset.dense_queries)
        try:
            vectors = embedder.embed_queries(dense_queries) if dense_queries else []
            # The HyDE passage stands in for a *document*, so it goes down the
            # document code path (see hyde.py).
            if hyde_result and hyde_result.passage:
                vectors.append(embedder.embed_document(hyde_result.passage))
                dense_queries.append("[hyde] " + hyde_result.passage[:60])
        except Exception as exc:
            vectors, dense_queries = [], []
            leg_notes.append(f"dense leg unavailable: {type(exc).__name__}: {exc}")
        for label, vec in zip(dense_queries, vectors, strict=True):
            for ranked in dense_search(vec, label, top_k=flags.dense_top_k, metadata=metadata):
                if ranked.note:
                    leg_notes.append(f"dense[{ranked.target}]: {ranked.note}")
                key = f"dense:{ranked.target}"
                leg_hits[key] = leg_hits.get(key, 0) + len(ranked.hits)
                ranked_lists.append(ranked)

    if flags.lexical:
        for lq in queryset.lexical_queries:
            for ranked in lexical_search(lq, top_k=flags.lexical_top_k, metadata=metadata):
                if ranked.note:
                    leg_notes.append(f"lexical[{ranked.target}]: {ranked.note}")
                key = f"lexical:{ranked.target}"
                leg_hits[key] = leg_hits.get(key, 0) + len(ranked.hits)
                ranked_lists.append(ranked)

    if flags.graph:
        seeds = [h.ref_id for rl in ranked_lists for h in rl.hits[:5]]
        ranked = graph_search(seeds, top_k=flags.graph_top_k)
        if ranked.note:
            leg_notes.append(f"graph: {ranked.note}")
        leg_hits["graph"] = leg_hits.get("graph", 0) + len(ranked.hits)
        ranked_lists.append(ranked)

    fused = rrf_fuse(ranked_lists, rrf_k=flags.rrf_k, limit=max(flags.dense_top_k, 20))
    if flags.drop_boilerplate:
        from carelite.retrieval.filters import drop_boilerplate

        before = len(fused)
        fused = drop_boilerplate(fused)
        if len(fused) < before:
            leg_notes.append(f"dropped {before - len(fused)} publication-boilerplate candidates")
    stage_ms["retrieve"] = int((time.monotonic() - t0) * 1000)

    # -- 5. cross-encoder rerank -------------------------------------------
    t0 = time.monotonic()
    rerank_available: bool | None = None
    candidates = fused
    if flags.rerank:
        # Imported here, not at module scope: an ablation row with rerank off
        # must never load sentence-transformers or torch. See rerank.py.
        from carelite.retrieval.rerank import get_reranker

        engine = reranker if reranker is not None else get_reranker()
        rerank_query = queryset.dense_queries[0] if queryset.dense_queries else utterance
        outcome = engine.rerank(
            rerank_query,
            candidates,
            top_n=flags.rerank_top_n,
            tier_weighting=flags.tier_weighting,
        )
        rerank_available = outcome.available
        if not outcome.available and outcome.reason:
            leg_notes.append(f"rerank: {outcome.reason}")
        candidates = outcome.items
    else:
        candidates = candidates[: flags.rerank_top_n]
    stage_ms["rerank"] = int((time.monotonic() - t0) * 1000)

    # -- 6. CRAG gate ------------------------------------------------------
    t0 = time.monotonic()
    if flags.crag and flags.use_llm_crag and grader_client is None:
        from carelite.retrieval.llm import LLMClient

        # Judge family, deliberately not the generator: grading retrieved
        # context with the same model that will generate from it is the
        # circularity v3 §13 rules out one stage later.
        grader_client = LLMClient(model_tag=settings.models.judge.tag)

    report = crag_mod.grade_context(
        utterance,
        candidates,
        enabled=flags.crag,
        use_llm=flags.use_llm_crag,
        client=grader_client,
        threshold=flags.crag_relevance_threshold,
        ambiguous_ratio=flags.crag_ambiguous_ratio,
    )
    stage_ms["crag"] = int((time.monotonic() - t0) * 1000)

    rejected: list[RetrievedItem] = []
    kept = candidates
    if report.should_fall_back:
        # Condition-B fallback: the context is discarded, not merely flagged,
        # so no downstream lane can pass it to a prompt by accident.
        rejected = list(candidates)
        kept = []

    trace = RetrievalTrace(
        route=decision.route,
        queries=list(queryset.all_queries),
        hyde_passage=hyde_result.passage if hyde_result else None,
        retrieved=kept,
        crag_grade=report.grade,
        fell_back_to_b=report.should_fall_back,
        latency_ms=int((time.monotonic() - started) * 1000),
    )
    return RetrievalResult(
        trace=trace,
        flags=flags,
        rejected=rejected,
        route_decision=decision,
        grade_report=report,
        queryset=queryset,
        hyde_result=hyde_result,
        leg_notes=leg_notes,
        rerank_available=rerank_available,
        stage_ms=stage_ms,
        leg_hits=leg_hits,
    )
