"""Reranking, evidence-tier weighting, and the lazy-import contract."""

from __future__ import annotations

import subprocess
import sys

from carelite.retrieval.rerank import (
    TIER_WEIGHTS,
    UNKNOWN_TIER_WEIGHT,
    CrossEncoderReranker,
    apply_tier_weight,
)
from carelite.types import EvidenceTier

from .conftest import make_item


def test_tier_weights_are_ordered_and_gentle() -> None:
    """Tier must break near-ties, not override the model — the brief says
    strong should outrank emerging *at comparable relevance*."""
    assert (
        TIER_WEIGHTS[EvidenceTier.STRONG]
        > TIER_WEIGHTS[EvidenceTier.MODERATE]
        > TIER_WEIGHTS[EvidenceTier.EMERGING]
    )
    assert TIER_WEIGHTS[EvidenceTier.EMERGING] > 0.8


def test_tier_breaks_a_near_tie() -> None:
    strong = apply_tier_weight(0.60, EvidenceTier.STRONG)
    emerging = apply_tier_weight(0.61, EvidenceTier.EMERGING)
    assert strong > emerging


def test_tier_does_not_override_a_large_relevance_gap() -> None:
    strong = apply_tier_weight(0.20, EvidenceTier.STRONG)
    emerging = apply_tier_weight(0.90, EvidenceTier.EMERGING)
    assert emerging > strong


def test_unknown_tier_is_treated_as_moderate_not_penalised() -> None:
    """ "We have not tiered this paper yet" is not evidence of weakness. All 33
    papers currently carry `emerging`, so an untiered row must not be pushed
    below them by default."""
    assert TIER_WEIGHTS[EvidenceTier.MODERATE] == UNKNOWN_TIER_WEIGHT
    assert apply_tier_weight(0.5, None) > apply_tier_weight(0.5, EvidenceTier.EMERGING)


def test_reranker_reorders_by_score(fake_reranker) -> None:
    fake_reranker.scores = {"a": 0.1, "b": 0.9, "c": 0.5}
    items = [make_item("a"), make_item("b"), make_item("c")]
    result = fake_reranker.rerank("q", items, top_n=3, tier_weighting=False)
    assert [i.ref_id for i in result.items] == ["b", "c", "a"]


def test_reranker_truncates_to_top_n(fake_reranker) -> None:
    items = [make_item(f"r{i}") for i in range(10)]
    assert len(fake_reranker.rerank("q", items, top_n=4).items) == 4


def test_unavailable_reranker_degrades_to_input_order(fake_reranker) -> None:
    """sentence-transformers is an optional extra. A missing ranking
    refinement must not take down a bedside interface."""
    fake_reranker.available = False
    items = [make_item("a"), make_item("b")]
    result = fake_reranker.rerank("q", items, top_n=4)
    assert result.available is False
    assert [i.ref_id for i in result.items] == ["a", "b"]


def test_empty_input_is_handled() -> None:
    result = CrossEncoderReranker().rerank("q", [], top_n=4)
    assert result.items == []
    assert result.available is True


def test_missing_sentence_transformers_is_reported_not_raised(monkeypatch) -> None:
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("no sentence_transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    engine = CrossEncoderReranker()
    result = engine.rerank("q", [make_item("a")], top_n=4)
    assert result.available is False
    assert "sentence-transformers" in result.reason


def test_importing_the_package_does_not_pull_in_torch() -> None:
    """The lane contract: `carelite.retrieval` must be importable without
    paying torch's import cost. Run in a clean interpreter because torch may
    already be resident in this test session."""
    code = (
        "import sys; import carelite.retrieval; "
        "assert 'torch' not in sys.modules, 'torch imported'; "
        "assert 'sentence_transformers' not in sys.modules, 'ST imported'; "
        "print('clean')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout


def test_constructing_the_reranker_does_not_load_the_model() -> None:
    code = (
        "import sys; from carelite.retrieval.rerank import CrossEncoderReranker; "
        "CrossEncoderReranker(); "
        "assert 'torch' not in sys.modules, 'torch imported on construct'; "
        "print('clean')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout
