"""The generation client: digests, context sizing, and failing loudly."""

from __future__ import annotations

from typing import Any

import pytest

from carelite.generate.model import (
    ALLOW_TEST_INFERENCE_ENV,
    DIGEST_UNAVAILABLE,
    MIN_CONTEXT,
    GenerationClient,
    GenerationError,
    context_size,
    estimate_tokens,
)
from carelite.safety.fencing import assemble


class _FakeOllama:
    """Stands in for `ollama.Client`. Records what it was asked for."""

    def __init__(self, *, models: list[dict[str, str]] | None = None, reply: str = "hello") -> None:
        self.models = (
            models
            if models is not None
            else [
                {"model": "gemma4:12b", "digest": "sha256:aaa"},
                {"model": "bge-m3:latest", "digest": "sha256:bbb"},
            ]
        )
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def list(self) -> dict[str, Any]:
        return {"models": self.models}

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"message": {"content": self.reply}}


def _prompt() -> Any:
    return assemble(system="You are a test.", task="Reply.", utterance="Hello there.")


# ---------------------------------------------------------------------------
# Context sizing
# ---------------------------------------------------------------------------


def test_context_is_floored_at_the_minimum_and_capped_at_the_window() -> None:
    assert context_size(10, num_predict=100, window=131072) == MIN_CONTEXT
    assert context_size(1_000_000, num_predict=512, window=8192) == 8192
    assert context_size(500_000, num_predict=512, window=131072) <= 131072


def test_a_small_prompt_change_does_not_change_the_allocated_context() -> None:
    """Rounding to a power of two keeps `num_ctx` stable across near-identical
    prompts, so two samples of one cell are not run under different allocations."""
    assert context_size(40_000, num_predict=512, window=131072) == context_size(
        40_050, num_predict=512, window=131072
    )


def test_num_ctx_is_always_sent() -> None:
    """Ollama defaults to a few thousand tokens regardless of what the model
    supports, so a long-context condition that does not set it is silently
    truncated and the baseline measures nothing."""
    fake = _FakeOllama()
    out = GenerationClient(client=fake).generate(
        _prompt(), model_tag="gemma4:12b", seed=3, temperature=0.7, window=131072
    )
    assert fake.calls[0]["options"]["num_ctx"] == out.num_ctx
    assert out.num_ctx >= MIN_CONTEXT


def test_estimate_tokens_errs_high() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("x" * 400) == 101


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def test_the_digest_is_read_from_the_daemon_and_cached() -> None:
    client = GenerationClient(client=_FakeOllama())
    assert client.resolve_digest("gemma4:12b") == "sha256:aaa"
    assert client.resolve_digest("gemma4:12b") == "sha256:aaa"


def test_a_bare_tag_matches_the_latest_suffix() -> None:
    """`bge-m3` in the frozen roster is `bge-m3:latest` in the daemon."""
    assert GenerationClient(client=_FakeOllama()).resolve_digest("bge-m3") == "sha256:bbb"


def test_an_unlistable_daemon_yields_an_honest_placeholder() -> None:
    """`generation.model_digest` is NOT NULL, and a plausible-looking fake value
    would be worse than one that announces itself."""

    class _Broken(_FakeOllama):
        def list(self) -> dict[str, Any]:
            raise ConnectionError("daemon is down")

    assert GenerationClient(client=_Broken()).resolve_digest("gemma4:12b") == DIGEST_UNAVAILABLE


def test_an_unknown_tag_is_not_given_another_models_digest() -> None:
    assert GenerationClient(client=_FakeOllama()).resolve_digest("nope:1b") == DIGEST_UNAVAILABLE


def test_metadata_and_generation_get_separate_clocks() -> None:
    """Listing the daemon happens before the plan is built, so a saturated
    daemon must not stall a 1,080-cell run at the pre-flight with nothing
    written and nothing to resume from."""
    client = GenerationClient()
    assert client.metadata_timeout_s < client.timeout_s


# ---------------------------------------------------------------------------
# Failing loudly
# ---------------------------------------------------------------------------


def test_a_transport_failure_raises_rather_than_returning_none() -> None:
    """A missing cell is a hole in the results table. `None` would be written as
    an empty response and analysed as if the model had said nothing."""

    class _Broken(_FakeOllama):
        def chat(self, **kwargs: Any) -> dict[str, Any]:
            raise ConnectionError("connection reset")

    with pytest.raises(GenerationError, match="connection reset"):
        GenerationClient(client=_Broken()).generate(
            _prompt(), model_tag="gemma4:12b", seed=1, temperature=0.7
        )


def test_an_empty_response_is_an_error_not_a_result() -> None:
    with pytest.raises(GenerationError, match="empty response"):
        GenerationClient(client=_FakeOllama(reply="   ")).generate(
            _prompt(), model_tag="gemma4:12b", seed=1, temperature=0.7
        )


def test_seed_and_temperature_reach_the_daemon() -> None:
    fake = _FakeOllama()
    GenerationClient(client=fake).generate(
        _prompt(), model_tag="gemma4:12b", seed=987, temperature=0.7
    )
    assert fake.calls[0]["options"]["seed"] == 987
    assert fake.calls[0]["options"]["temperature"] == 0.7


def test_an_older_client_without_think_still_works() -> None:
    """`think=False` suppresses a reasoning model's visible scratchpad, but not
    every installed `ollama` client accepts the argument."""

    class _NoThink(_FakeOllama):
        def chat(self, **kwargs: Any) -> dict[str, Any]:
            if "think" in kwargs:
                raise TypeError("unexpected keyword argument 'think'")
            return super().chat(**kwargs)

    out = GenerationClient(client=_NoThink()).generate(
        _prompt(), model_tag="gemma4:12b", seed=1, temperature=0.7
    )
    assert out.text == "hello"


# ---------------------------------------------------------------------------
# The test-process guardrail
# ---------------------------------------------------------------------------


def test_a_test_process_cannot_open_a_live_model_connection() -> None:
    """`make check` excludes the `inference` marker; this makes that a property
    of the client rather than a convention a test can forget."""
    with pytest.raises(GenerationError, match="refusing to open a live model"):
        GenerationClient().generate(_prompt(), model_tag="gemma4:12b", seed=1, temperature=0.7)


def test_the_guardrail_never_touches_an_injected_client() -> None:
    out = GenerationClient(client=_FakeOllama()).generate(
        _prompt(), model_tag="gemma4:12b", seed=1, temperature=0.7
    )
    assert out.text == "hello"


def test_a_test_that_means_to_reach_a_model_can_say_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-out has to exist, or an `inference`-marked test could not run."""
    monkeypatch.setenv(ALLOW_TEST_INFERENCE_ENV, "1")
    with pytest.raises(GenerationError) as exc:
        GenerationClient(host="http://127.0.0.1:9").generate(
            _prompt(), model_tag="gemma4:12b", seed=1, temperature=0.7
        )
    assert "refusing to open a live model" not in str(exc.value)


def test_digest_resolution_degrades_rather_than_raising_under_the_guardrail() -> None:
    """`resolve_digest` runs before the plan is built. It must report the loss,
    not take the run down."""
    assert GenerationClient().resolve_digest("gemma4:12b") == DIGEST_UNAVAILABLE
