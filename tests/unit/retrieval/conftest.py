"""Fixtures for the retrieval lane.

Everything here is offline. The pipeline takes every collaborator by
injection precisely so the composition can be driven with fakes, and these
are those fakes. Tests that need Postgres are marked `@pytest.mark.db` and
tests that need a live model `@pytest.mark.inference`; both are excluded from
`make check` by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from carelite.types import EvidenceTier, RetrievedItem, Theme


@dataclass
class FakeChatResult:
    text: str
    model: str = "fake"
    latency_ms: int = 0
    cached: bool = False


@dataclass
class FakeLLM:
    """Scripted `LLMClient`. Records calls so a test can assert what was sent."""

    responses: list[str | None] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    default: str | None = None

    def chat(self, **kwargs: Any) -> FakeChatResult | None:
        self.calls.append(kwargs)
        text = self.responses.pop(0) if self.responses else self.default
        return None if text is None else FakeChatResult(text=text)

    def close(self) -> None:  # pragma: no cover - parity with LLMClient
        pass


@dataclass
class FakeEmbedder:
    """Deterministic embeddings without Ollama. Never touches the network."""

    dim: int = 1024
    queries: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)

    def _vec(self, text: str) -> list[float]:
        seed = sum(ord(c) for c in text) or 1
        return [((seed * (i + 1)) % 97) / 97.0 for i in range(self.dim)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def embed_document(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.queries.extend(texts)
        return [self._vec(t) for t in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents.extend(texts)
        return [self._vec(t) for t in texts]

    def close(self) -> None:  # pragma: no cover
        pass


@dataclass
class FakeReranker:
    """Cross-encoder stand-in. Scores by a caller-supplied table."""

    scores: dict[str, float] = field(default_factory=dict)
    available: bool = True
    default: float = 0.5
    seen: list[str] = field(default_factory=list)

    def rerank(
        self,
        query: str,
        items: list[RetrievedItem],
        *,
        top_n: int = 4,
        tier_weighting: bool = True,
    ) -> Any:
        from carelite.retrieval.rerank import RerankResult, apply_tier_weight

        self.seen.append(query)
        if not self.available:
            return RerankResult(items=items[:top_n], available=False, reason="fake: unavailable")
        out = []
        for item in items:
            raw = self.scores.get(item.ref_id, self.default)
            final = apply_tier_weight(raw, item.evidence_tier) if tier_weighting else raw
            out.append(item.model_copy(update={"rerank_score": raw, "score": final}))
        out.sort(key=lambda i: (-i.score, i.ref_id))
        return RerankResult(
            items=out[:top_n], available=True, model="fake", tier_weighted=tier_weighting
        )


def make_item(
    ref_id: str,
    text: str = "some retrieved guidance about empathic communication with patients",
    *,
    kind: str = "chunk",
    score: float = 0.5,
    tier: EvidenceTier | None = None,
    theme: Theme | None = None,
    rerank_score: float | None = None,
) -> RetrievedItem:
    return RetrievedItem(
        ref_id=ref_id,
        kind=kind,
        text=text,
        score=score,
        evidence_tier=tier,
        theme=theme,
        rerank_score=rerank_score,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()
