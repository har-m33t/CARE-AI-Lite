"""Unit tests for carelite.corpus.contextualize.

No test in this file calls a real model or opens a real DB connection.
`generate_prefix`, `document_text_by_paper_id`, and (for the
`run_contextualize_pass` tests) `chunks`/`persist` are always fakes injected
by the test — `make_ollama_generate_prefix` is exercised against a fake
Ollama client with a `.chat()` method, the same pattern
`tests/unit/generate/test_model.py` uses for `GenerationClient`.
"""

from __future__ import annotations

from typing import Any

import pytest
from tenacity import stop_after_attempt, wait_none

from carelite.corpus import contextualize
from carelite.types import Chunk


def _chunk(chunk_id: str = "p::0000", paper_id: str = "p", prefix: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, paper_id=paper_id, text="the chunk body", contextual_prefix=prefix
    )


def test_build_prompt_includes_document_and_chunk_text():
    chunk = _chunk()
    prompt = contextualize.build_prompt("FULL DOCUMENT TEXT", chunk)
    assert "FULL DOCUMENT TEXT" in prompt
    assert "the chunk body" in prompt
    assert "<document>" in prompt and "<chunk>" in prompt


def test_default_generate_prefix_is_not_wired_up_this_wave():
    chunk = _chunk()
    out, report = contextualize.contextualize_chunks([chunk], {"p": "doc"})
    assert out[0].contextual_prefix is None
    assert report.processed == 0
    assert report.total_failed == 1
    assert "NotImplementedError" in report.failed[0][1]


def test_resumable_skips_chunks_that_already_have_a_prefix():
    chunk = _chunk(prefix="already situated")
    calls = []

    def spy(document_text: str, c: Chunk) -> str:
        calls.append(c.chunk_id)
        return "should not be called"

    out, report = contextualize.contextualize_chunks([chunk], {"p": "doc"}, generate_prefix=spy)

    assert calls == []
    assert out[0].contextual_prefix == "already situated"
    assert report.skipped_already_done == 1
    assert report.processed == 0


def test_successful_generation_sets_prefix_without_mutating_input_chunk():
    chunk = _chunk()

    def gen(document_text: str, c: Chunk) -> str:
        return f"Situates {c.chunk_id} within {document_text[:4]}"

    out, report = contextualize.contextualize_chunks(
        [chunk], {"p": "doc text"}, generate_prefix=gen
    )

    assert (
        chunk.contextual_prefix is None
    )  # input untouched (Chunk is immutable-by-convention here)
    assert out[0].contextual_prefix == "Situates p::0000 within doc "
    assert report.processed == 1
    assert report.total_failed == 0


def test_missing_document_text_is_recorded_as_failure_not_raised():
    chunk = _chunk(paper_id="missing-paper")

    out, report = contextualize.contextualize_chunks(
        [chunk], {}, generate_prefix=lambda d, c: "unused"
    )

    assert out[0].contextual_prefix is None
    assert report.total_failed == 1
    assert "no document text" in report.failed[0][1]


def test_one_bad_chunk_does_not_abort_the_whole_run():
    good = _chunk(chunk_id="p::0000")
    bad = _chunk(chunk_id="p::0001")

    def gen(document_text: str, c: Chunk) -> str:
        if c.chunk_id == "p::0001":
            raise ValueError("boom")
        return "ok prefix"

    out, report = contextualize.contextualize_chunks([good, bad], {"p": "doc"}, generate_prefix=gen)

    assert out[0].contextual_prefix == "ok prefix"
    assert out[1].contextual_prefix is None
    assert report.processed == 1
    assert report.total_failed == 1
    assert "boom" in report.failed[0][1]


def test_non_rate_limited_exceptions_are_not_retried():
    calls = {"n": 0}

    def always_value_error(document_text: str, c: Chunk) -> str:
        calls["n"] += 1
        raise ValueError("not a rate limit")

    _out, report = contextualize.contextualize_chunks(
        [_chunk()], {"p": "doc"}, generate_prefix=always_value_error
    )

    assert calls["n"] == 1  # no tenacity retry for a non-RateLimited exception
    assert report.total_failed == 1


def test_rate_limited_is_retried_then_succeeds(monkeypatch):
    # Swap in a zero-wait retry policy so this test doesn't sleep for real.
    fast_retry = contextualize._generate_with_retry.retry_with(
        wait=wait_none(), stop=stop_after_attempt(3)
    )
    monkeypatch.setattr(contextualize, "_generate_with_retry", fast_retry)

    calls = {"n": 0}

    def flaky(document_text: str, c: Chunk) -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise contextualize.RateLimited("slow down")
        return "a situating prefix"

    out, report = contextualize.contextualize_chunks(
        [_chunk()], {"p": "doc"}, generate_prefix=flaky
    )

    assert calls["n"] == 2
    assert out[0].contextual_prefix == "a situating prefix"
    assert report.processed == 1
    assert report.total_failed == 0


# ---------------------------------------------------------------------------
# make_ollama_generate_prefix — real generator, fake ollama.Client
# ---------------------------------------------------------------------------


class _FakeOllamaClient:
    """Stands in for `ollama.Client`. Records what it was asked for and
    replies with a scripted sequence of `{"message": {"content": ...}}`
    dicts, one per call, so a test can make the first attempt (think=False)
    come back empty and check the think=True fallback fires."""

    def __init__(self, replies: list[str] | None = None, raises: Exception | None = None) -> None:
        self.replies = list(replies) if replies is not None else ["a situating prefix"]
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        reply = self.replies.pop(0) if self.replies else ""
        return {"message": {"content": reply}}


def _chunk_for_prefix(chunk_id: str = "p::0000", paper_id: str = "p") -> Chunk:
    return Chunk(chunk_id=chunk_id, paper_id=paper_id, text="the chunk body")


def test_ollama_generate_prefix_sends_think_false_first():
    fake = _FakeOllamaClient(replies=["Situates the chunk within the paper."])
    generate = contextualize.make_ollama_generate_prefix(client=fake, model="gemma4:12b")

    result = generate("FULL DOCUMENT TEXT", _chunk_for_prefix())

    assert result == "Situates the chunk within the paper."
    assert len(fake.calls) == 1
    assert fake.calls[0]["think"] is False
    assert fake.calls[0]["model"] == "gemma4:12b"
    assert "FULL DOCUMENT TEXT" in fake.calls[0]["messages"][0]["content"]
    assert fake.calls[0]["options"]["num_ctx"] >= contextualize._MIN_CONTEXT


def test_ollama_generate_prefix_falls_back_when_think_false_is_empty():
    """An empty first answer (think=False came back with nothing) is retried
    once with thinking left on, rather than accepted as the model's answer."""
    fake = _FakeOllamaClient(replies=["", "a real answer on retry"])
    generate = contextualize.make_ollama_generate_prefix(client=fake, model="gemma4:12b")

    result = generate("doc", _chunk_for_prefix())

    assert result == "a real answer on retry"
    assert len(fake.calls) == 2
    assert fake.calls[0]["think"] is False
    assert "think" not in fake.calls[1]


def test_ollama_generate_prefix_empty_both_attempts_raises_value_error():
    fake = _FakeOllamaClient(replies=["", ""])
    generate = contextualize.make_ollama_generate_prefix(client=fake, model="gemma4:12b")

    with pytest.raises(ValueError, match="empty prefix"):
        generate("doc", _chunk_for_prefix())


def test_ollama_generate_prefix_transient_error_raises_rate_limited():
    fake = _FakeOllamaClient(raises=ConnectionError("connection refused"))
    generate = contextualize.make_ollama_generate_prefix(client=fake, model="gemma4:12b")

    with pytest.raises(contextualize.RateLimited):
        generate("doc", _chunk_for_prefix())


def test_ollama_generate_prefix_non_transient_error_propagates_as_itself():
    fake = _FakeOllamaClient(raises=ValueError("some other model error"))
    generate = contextualize.make_ollama_generate_prefix(client=fake, model="gemma4:12b")

    with pytest.raises(ValueError, match="some other model error"):
        generate("doc", _chunk_for_prefix())


def test_num_ctx_is_floored_and_capped():
    assert contextualize._num_ctx_for("short", num_predict=100, window=131072) == (
        contextualize._MIN_CONTEXT
    )
    huge_prompt = "x" * 2_000_000
    assert contextualize._num_ctx_for(huge_prompt, num_predict=512, window=16384) == 16384


# ---------------------------------------------------------------------------
# run_contextualize_pass — chunk-granular persistence, fully injected
# ---------------------------------------------------------------------------


def test_run_pass_persists_each_chunk_immediately():
    chunks = [_chunk_for_prefix("p::0000", "p"), _chunk_for_prefix("p::0001", "p")]
    persisted: list[tuple[str, str]] = []

    def gen(document_text: str, c: Chunk) -> str:
        return f"prefix for {c.chunk_id}"

    report = contextualize.run_contextualize_pass(
        chunks=chunks,
        document_text_by_paper_id={"p": "doc text"},
        generate_prefix=gen,
        persist=lambda chunk_id, prefix: persisted.append((chunk_id, prefix)),
    )

    assert report.processed == 2
    assert report.total_failed == 0
    assert persisted == [
        ("p::0000", "prefix for p::0000"),
        ("p::0001", "prefix for p::0001"),
    ]


def test_run_pass_does_not_persist_on_failure():
    chunks = [_chunk_for_prefix("p::0000", "p")]
    persisted: list[tuple[str, str]] = []

    def gen(document_text: str, c: Chunk) -> str:
        raise ValueError("boom")

    report = contextualize.run_contextualize_pass(
        chunks=chunks,
        document_text_by_paper_id={"p": "doc text"},
        generate_prefix=gen,
        persist=lambda chunk_id, prefix: persisted.append((chunk_id, prefix)),
    )

    assert report.total_failed == 1
    assert persisted == []


def test_run_pass_does_not_persist_an_already_done_chunk():
    already = Chunk(chunk_id="p::0000", paper_id="p", text="body", contextual_prefix="existing")
    persisted: list[tuple[str, str]] = []

    def gen(document_text: str, c: Chunk) -> str:
        raise AssertionError("should not be called for an already-done chunk")

    report = contextualize.run_contextualize_pass(
        chunks=[already],
        document_text_by_paper_id={"p": "doc text"},
        generate_prefix=gen,
        persist=lambda chunk_id, prefix: persisted.append((chunk_id, prefix)),
    )

    assert report.skipped_already_done == 1
    assert persisted == []


def test_run_pass_calls_on_progress_for_every_chunk():
    chunks = [_chunk_for_prefix("p::0000", "p"), _chunk_for_prefix("p::0001", "p")]
    seen: list[tuple[int, int, str]] = []

    contextualize.run_contextualize_pass(
        chunks=chunks,
        document_text_by_paper_id={"p": "doc text"},
        generate_prefix=lambda d, c: "ok",
        persist=lambda chunk_id, prefix: None,
        on_progress=lambda idx, total, chunk: seen.append((idx, total, chunk.chunk_id)),
    )

    assert seen == [(1, 2, "p::0000"), (2, 2, "p::0001")]
