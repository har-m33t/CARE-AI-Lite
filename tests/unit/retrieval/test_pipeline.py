"""Composition tests: the invariants the rest of the system relies on.

Every collaborator is faked, so these run with no database, no Ollama and no
torch.
"""

from __future__ import annotations

import sys
import typing

import pytest

from carelite.retrieval.flags import RetrievalFlags, preset
from carelite.retrieval.pipeline import retrieve, retrieve_detailed
from carelite.types import CRAGGrade, EncounterPhase, Route

from .conftest import make_item


@pytest.fixture
def offline_legs(monkeypatch):
    """Silence the three retrieval legs without *disabling* them.

    `RetrievalFlags.validate()` rejects a configuration with every leg off —
    correctly, since that retrieves nothing and is not an ablation. So these
    tests keep the legs switched on and stub the functions instead, which also
    keeps the composition under test closer to the real one.
    """
    import carelite.retrieval.pipeline as pipeline

    monkeypatch.setattr(pipeline, "dense_search", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "lexical_search", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "graph_search", lambda *a, **k: _EMPTY_RANKED)
    return pipeline


class _EmptyRanked:
    leg = "graph"
    target = "kb_entry"
    hits: typing.ClassVar[list] = []
    note = ""
    query = ""


_EMPTY_RANKED = _EmptyRanked()


@pytest.fixture
def offline_flags() -> RetrievalFlags:
    """Full stack minus everything that needs a live service."""
    return preset("R9").with_(hyde=False, rerank=False)


def test_emotional_only_skips_retrieval_entirely(fake_llm, offline_flags, offline_legs) -> None:
    result = retrieve_detailed("I'm just so scared.", flags=offline_flags, generator=fake_llm)
    trace = result.trace

    assert trace.route is Route.EMOTIONAL_ONLY
    assert trace.retrieved == []
    # Not a fallback: nothing was rejected, retrieval was never attempted.
    assert trace.fell_back_to_b is False
    assert trace.hyde_passage is None
    assert fake_llm.calls == []  # no model was consulted at all


def test_skip_and_fallback_are_distinguishable(fake_llm, offline_legs) -> None:
    """`route` and `fell_back_to_b` mean opposite things about the corpus and
    are never collapsed into one flag: a skip says "this turn wanted presence,
    not evidence", a fallback says "the corpus could not address this turn"."""
    skipped = retrieve_detailed(
        "I'm just so scared.",
        flags=preset("R9").with_(hyde=False, rerank=False),
        generator=fake_llm,
    ).trace
    assert (skipped.route, skipped.fell_back_to_b) == (Route.EMOTIONAL_ONLY, False)


def test_crag_none_clears_retrieved_so_evidence_cannot_leak(
    monkeypatch, fake_llm, offline_legs
) -> None:
    """On NONE the context is *discarded*, not merely flagged.

    A downstream lane that does `fencing.assemble(retrieved=trace.retrieved)`
    must be unable to inject evidence into a turn the gate rejected, even if
    it never inspects `crag_grade`.
    """
    import carelite.retrieval.pipeline as pipeline

    monkeypatch.setattr(pipeline, "rrf_fuse", lambda *a, **k: [make_item("a"), make_item("b")])
    fake_llm.default = (
        '{"passages": [{"id": 1, "useful": false}, {"id": 2, "useful": false}], "overall": "none"}'
    )
    flags = preset("R9").with_(hyde=False, rerank=False, router=False)

    result = retrieve_detailed("anything at all", flags=flags, grader_client=fake_llm)

    assert result.trace.crag_grade is CRAGGrade.NONE
    assert result.trace.fell_back_to_b is True
    assert result.trace.retrieved == []
    # The rejected candidates stay off the trace and on the result.
    assert [i.ref_id for i in result.rejected] == ["a", "b"]


def test_rejected_candidates_are_not_on_the_trace(monkeypatch, fake_llm, offline_legs) -> None:
    import carelite.retrieval.pipeline as pipeline

    monkeypatch.setattr(pipeline, "rrf_fuse", lambda *a, **k: [make_item("a")])
    fake_llm.default = '{"passages": [{"id": 1, "useful": false}], "overall": "none"}'
    flags = preset("R9").with_(hyde=False, rerank=False, router=False)
    result = retrieve_detailed("x", flags=flags, grader_client=fake_llm)
    assert not hasattr(result.trace, "rejected")
    assert result.trace.retrieved == []


def test_trace_carries_what_the_cli_evidence_panel_needs(
    monkeypatch, fake_llm, offline_legs
) -> None:
    """`carelite/cli/render.py` reads exactly these fields."""
    import carelite.retrieval.pipeline as pipeline

    item = make_item("a", rerank_score=0.9, score=0.9)
    item = item.model_copy(update={"dense_rank": 1, "lexical_rank": 3, "citation": "Smith 2020"})
    monkeypatch.setattr(pipeline, "rrf_fuse", lambda *a, **k: [item])
    fake_llm.default = '{"passages": [{"id": 1, "useful": true}], "overall": "relevant"}'

    flags = preset("R9").with_(hyde=False, rerank=False, router=False)
    trace = retrieve("What does that mean?", flags=flags, grader_client=fake_llm)

    assert trace.route is not None
    assert trace.crag_grade is CRAGGrade.RELEVANT
    assert trace.latency_ms is not None and trace.latency_ms >= 0
    assert trace.queries  # the panel lists them under --explain
    got = trace.retrieved[0]
    assert got.ref_id and got.citation == "Smith 2020"
    assert got.dense_rank == 1 and got.lexical_rank == 3
    assert got.rerank_score == pytest.approx(0.9)


def test_reranker_is_not_imported_when_ablated_out(offline_flags, offline_legs) -> None:
    """The lane brief requires the cross-encoder not be imported when
    reranking is off. Importing sentence-transformers pulls in torch, which
    costs seconds and hundreds of MB in a process running an interactive
    terminal app — rows R0-R4 must not pay that."""
    for module in ("torch", "sentence_transformers"):
        sys.modules.pop(module, None)

    flags = offline_flags.with_(rerank=False, router=False, crag=False)
    retrieve("What does that mean?", flags=flags)

    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules


def test_rerank_flag_routes_through_the_injected_reranker(
    monkeypatch, fake_reranker, fake_llm, offline_legs
) -> None:
    import carelite.retrieval.pipeline as pipeline

    monkeypatch.setattr(pipeline, "rrf_fuse", lambda *a, **k: [make_item("a"), make_item("b")])
    fake_reranker.scores = {"a": 0.2, "b": 0.9}
    fake_llm.default = (
        '{"passages": [{"id": 1, "useful": true}, {"id": 2, "useful": true}], '
        '"overall": "relevant"}'
    )
    flags = preset("R9").with_(hyde=False, router=False, tier_weighting=False)

    trace = retrieve(
        "What does that mean?", flags=flags, reranker=fake_reranker, grader_client=fake_llm
    )

    assert [i.ref_id for i in trace.retrieved] == ["b", "a"]


def test_hyde_passage_is_recorded_but_never_used_as_context(
    monkeypatch, fake_llm, fake_embedder, offline_legs
) -> None:
    """The HyDE passage is model-invented text with no provenance. It is an
    embedding input and an audit record — never evidence."""
    import carelite.retrieval.pipeline as pipeline

    passage = (
        "When a patient expresses fear about a diagnosis, the clinician should name the "
        "emotion before providing further information, drawing on empathic response "
        "techniques described in the communication literature at some length here."
    )
    fake_llm.default = passage
    monkeypatch.setattr(pipeline, "rrf_fuse", lambda *a, **k: [make_item("a")])

    flags = preset("R9").with_(rerank=False, router=False, crag=False)
    trace = retrieve(
        "What does that mean?", flags=flags, generator=fake_llm, embedder=fake_embedder
    )

    assert trace.hyde_passage == passage
    assert all(passage not in item.text for item in trace.retrieved)


def test_every_preset_runs_end_to_end(
    monkeypatch, fake_llm, fake_embedder, fake_reranker, offline_legs
) -> None:
    """An ablation row is a different `flags` argument, never a different code
    path — so every preset must survive the same function."""
    import carelite.retrieval.pipeline as pipeline

    monkeypatch.setattr(pipeline, "rrf_fuse", lambda *a, **k: [make_item("a")])
    fake_llm.default = '{"passages": [{"id": 1, "useful": true}], "overall": "relevant"}'

    for name in ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"):
        flags = preset(name)
        trace = retrieve_detailed(
            "What does that mean?",
            flags=flags,
            generator=fake_llm,
            embedder=fake_embedder,
            reranker=fake_reranker,
            grader_client=fake_llm,
        ).trace
        assert trace.route is not None, name
        assert trace.latency_ms is not None, name


def test_encounter_phase_reaches_query_construction(fake_llm, offline_flags, offline_legs) -> None:
    result = retrieve_detailed(
        "What happens next?",
        encounter_phase=EncounterPhase.PLANNING,
        flags=offline_flags.with_(router=False),
        generator=fake_llm,
    )
    assert result.queryset is not None
    assert result.queryset.metadata.encounter_phase is EncounterPhase.PLANNING


def test_crag_per_passage_filtering_is_off_by_default(monkeypatch, fake_llm, offline_legs) -> None:
    """Default preserves the behaviour cells were already generated against.
    Enabling it is a study decision, not a lane decision."""
    import carelite.retrieval.pipeline as pipeline

    monkeypatch.setattr(pipeline, "rrf_fuse", lambda *a, **k: [make_item("a"), make_item("b")])
    fake_llm.default = (
        '{"passages": [{"id": 1, "useful": true}, {"id": 2, "useful": false}], '
        '"overall": "relevant"}'
    )
    flags = preset("R9").with_(hyde=False, rerank=False, router=False)
    trace = retrieve("What does that mean?", flags=flags, grader_client=fake_llm)
    assert [i.ref_id for i in trace.retrieved] == ["a", "b"]


def test_crag_per_passage_filtering_drops_what_the_grader_called_useless(
    monkeypatch, fake_llm, offline_legs
) -> None:
    """CRAG already records *which* passages help; the pipeline used only the
    aggregate verdict. Measured over 23 turns, 15 of 60 passages placed in
    condition C prompts (25%) had been judged useless by the same grader."""
    import carelite.retrieval.pipeline as pipeline

    monkeypatch.setattr(pipeline, "rrf_fuse", lambda *a, **k: [make_item("a"), make_item("b")])
    fake_llm.default = (
        '{"passages": [{"id": 1, "useful": true}, {"id": 2, "useful": false}], '
        '"overall": "relevant"}'
    )
    flags = preset("R9").with_(hyde=False, rerank=False, router=False, crag_filter_items=True)
    result = retrieve_detailed("What does that mean?", flags=flags, grader_client=fake_llm)
    assert [i.ref_id for i in result.trace.retrieved] == ["a"]
    assert [i.ref_id for i in result.rejected] == ["b"]
    assert result.trace.fell_back_to_b is False


def test_filtering_defers_rather_than_emitting_a_silent_empty_context(
    monkeypatch, fake_llm, offline_legs
) -> None:
    """A non-NONE grade with nothing kept would be indistinguishable
    downstream from a Condition-B fallback, which means something different."""
    import carelite.retrieval.pipeline as pipeline

    monkeypatch.setattr(pipeline, "rrf_fuse", lambda *a, **k: [make_item("a")])
    # Grader contradicts itself: says ambiguous, marks nothing useful.
    fake_llm.default = '{"passages": [{"id": 1, "useful": false}], "overall": "ambiguous"}'
    flags = preset("R9").with_(hyde=False, rerank=False, router=False, crag_filter_items=True)
    result = retrieve_detailed("x", flags=flags, grader_client=fake_llm)
    assert [i.ref_id for i in result.trace.retrieved] == ["a"]
    assert result.trace.fell_back_to_b is False
