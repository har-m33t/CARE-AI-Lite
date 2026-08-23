"""Unit tests for carelite.corpus.contextualize.

STRUCTURE ONLY this wave: no test here calls a real model. `generate_prefix`
is always a fake injected by the test, matching how the inference lane will
wire in the real one later.
"""

from __future__ import annotations

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
