"""The second serving stack: an OpenAI-compatible client, and its credential.

`CARELITE_BACKEND` chooses between Ollama and a remote vLLM server. Nothing
above the client changes — not the graph, not the nodes, not the conditions, not
the runner — so what has to be tested here is the client's own contract: that it
reports `served_by`, that it refuses to key a row on a model identity it cannot
establish, and that the bearer token it holds does not escape into a log, a
repr, or an exception message. Run logs get committed.

There is no vLLM server to test against, so every test here drives a fake
OpenAI-compatible endpoint. The seams are the ones a real server has to satisfy:
`/v1/models` for identity and `/v1/chat/completions` for generation.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from carelite.generate.backend import (
    API_KEY_ENV,
    BACKEND_ENV,
    BASE_URL_ENV,
    MODEL_ENV,
    REVISION_ENV,
    Backend,
    VLLMClient,
    default_client,
    selected_backend,
)
from carelite.generate.model import DIGEST_UNAVAILABLE, GenerationClient, GenerationError
from carelite.safety.fencing import assemble

SECRET = "sk-carelite-not-a-real-token-0123456789"  # pragma: allowlist secret
SERVED = "google/gemma-4-12b-it"
REVISION = "b6c0f4d5e9a1c37f2d8b4e6a0c9f1b3d5e7a9c11"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test here may depend on the developer's own `.env`.

    The repository's `.env` is where the real pod URL and token live, so a test
    that read it would pass or fail according to whose machine it ran on — and
    `test_the_default_backend_is_ollama` would start asserting a local
    configuration rather than the code's default.
    """
    import carelite.generate.backend as backend_mod

    for name in (BACKEND_ENV, BASE_URL_ENV, API_KEY_ENV, MODEL_ENV, REVISION_ENV):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(backend_mod.ALIASES.get(name, name), raising=False)
    monkeypatch.setattr(backend_mod, "_dotenv", dict)


def _prompt() -> Any:
    return assemble(system="You are a test.", task="Reply.", utterance="Hello there.")


class _FakeOpenAI:
    """Stands in for `openai.OpenAI`. Records what it was asked for."""

    def __init__(
        self,
        *,
        models: list[Any] | None = None,
        reply: str = "A steady, ordinary reply.",
        raises: Exception | None = None,
    ) -> None:
        self._models = (
            models
            if models is not None
            else [SimpleNamespace(id=SERVED, revision=REVISION, max_model_len=131072)]
        )
        self.reply = reply
        self.raises = raises
        self.calls: list[dict[str, Any]] = []
        self.models = SimpleNamespace(list=self._list)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _list(self) -> Any:
        return SimpleNamespace(data=self._models)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))]
        )


def _client(**kwargs: Any) -> VLLMClient:
    fake = kwargs.pop("fake", None) or _FakeOpenAI()
    return VLLMClient(
        base_url="https://pod.example/v1",
        api_key=SECRET,
        client=fake,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_the_default_backend_is_ollama() -> None:
    assert selected_backend() is Backend.OLLAMA
    assert isinstance(default_client(), GenerationClient)


def test_the_switch_selects_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BACKEND_ENV, "vllm")
    monkeypatch.setenv(BASE_URL_ENV, "https://pod.example/v1")
    monkeypatch.setenv(API_KEY_ENV, SECRET)
    assert selected_backend() is Backend.VLLM
    assert isinstance(default_client(), VLLMClient)


def test_an_unknown_backend_is_refused_rather_than_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(BACKEND_ENV, "openai")
    with pytest.raises(ValueError, match="ollama"):
        selected_backend()


def test_vllm_without_a_base_url_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loudly, and before a plan is built: a run that silently fell back to
    Ollama would produce rows labelled with the wrong serving stack."""
    monkeypatch.setenv(BACKEND_ENV, "vllm")
    with pytest.raises(GenerationError, match=BASE_URL_ENV):
        default_client()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_served_by_is_recorded_on_the_output() -> None:
    out = _client().generate(_prompt(), model_tag="gemma4:12b", seed=3, temperature=0.7)
    assert out.served_by == "vllm"
    assert VLLMClient.served_by == "vllm"
    assert GenerationClient.served_by == "ollama"


def test_the_digest_is_the_served_repo_id_and_its_revision() -> None:
    digest = _client().resolve_digest("gemma4:12b")
    assert digest == f"vllm:{SERVED}@{REVISION}"


def test_a_pinned_revision_from_the_environment_is_used_when_the_server_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """vLLM does not always report the commit it loaded. The operator launched
    the server with `--revision`, so that value is a real identifier and not a
    placeholder."""
    monkeypatch.setenv(REVISION_ENV, REVISION)
    fake = _FakeOpenAI(models=[SimpleNamespace(id=SERVED, max_model_len=131072)])
    assert _client(fake=fake).resolve_digest("gemma4:12b") == f"vllm:{SERVED}@{REVISION}"


def test_an_unknowable_revision_is_unavailable_rather_than_invented() -> None:
    """`DIGEST_UNAVAILABLE` is what `runner.assert_digests_resolved` refuses to
    start on. A synthesised identifier would key 180 rows on a fiction."""
    fake = _FakeOpenAI(models=[SimpleNamespace(id=SERVED, max_model_len=131072)])
    assert _client(fake=fake).resolve_digest("gemma4:12b") == DIGEST_UNAVAILABLE


def test_an_unreachable_server_yields_an_unavailable_digest_not_a_crash() -> None:
    fake = _FakeOpenAI()
    fake.models = SimpleNamespace(list=lambda: (_ for _ in ()).throw(RuntimeError("no route")))
    assert _client(fake=fake).resolve_digest("gemma4:12b") == DIGEST_UNAVAILABLE


def test_the_ollama_digest_and_the_vllm_digest_cannot_collide() -> None:
    """The `generation` uniqueness key includes `model_digest`, so a vLLM re-run
    of a cell Ollama already produced is a new row rather than a conflict."""
    ollama_digest = "sha256:1f2e3d4c5b6a7988990a0b1c2d3e4f5061728394a5b6c7d8e9f0a1b2c3d4e5f6"
    assert _client().resolve_digest("gemma4:12b") != ollama_digest


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_the_fenced_prompt_is_sent_as_messages_with_the_seed_and_temperature() -> None:
    fake = _FakeOpenAI()
    prompt = _prompt()
    _client(fake=fake).generate(
        prompt, model_tag="gemma4:12b", seed=17, temperature=0.7, num_predict=256
    )
    (call,) = fake.calls
    assert call["messages"] == prompt.as_messages()
    assert call["model"] == SERVED
    assert call["seed"] == 17
    assert call["temperature"] == 0.7
    assert call["max_tokens"] == 256


def test_num_ctx_is_the_window_the_server_reports() -> None:
    """On Ollama `num_ctx` is what the client asked for. On vLLM the allocation
    is the server's `--max-model-len`, so that is what the row records."""
    out = _client().generate(_prompt(), model_tag="gemma4:12b", seed=3, temperature=0.7)
    assert out.num_ctx == 131072


def test_an_empty_response_raises_because_an_empty_cell_is_a_hole() -> None:
    with pytest.raises(GenerationError, match="empty"):
        _client(fake=_FakeOpenAI(reply="   ")).generate(
            _prompt(), model_tag="gemma4:12b", seed=3, temperature=0.7
        )


def test_a_failed_request_raises_rather_than_returning_nothing() -> None:
    fake = _FakeOpenAI(raises=RuntimeError("502 Bad Gateway"))
    with pytest.raises(GenerationError, match="502"):
        _client(fake=fake).generate(_prompt(), model_tag="gemma4:12b", seed=3, temperature=0.7)


def test_json_format_is_requested_through_the_openai_response_format() -> None:
    """The self-check asks for a constrained decode. It has to survive the
    backend swap or condition B, C and LC lose their verification pass."""
    fake = _FakeOpenAI(reply='{"verdict": "pass"}')
    _client(fake=fake).generate(
        _prompt(), model_tag="gemma4:12b", seed=3, temperature=0.0, json_format=True
    )
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# The credential
# ---------------------------------------------------------------------------


def test_the_api_key_is_not_in_the_client_repr() -> None:
    assert SECRET not in repr(_client())


def test_the_api_key_is_not_in_the_repr_of_the_graph_deps_holding_it() -> None:
    from carelite.generate.graph import GraphDeps

    assert SECRET not in repr(GraphDeps(client=_client()))


def test_the_api_key_is_not_in_a_failed_requests_error_text() -> None:
    """Assume run logs get committed. A transport error that echoed the
    Authorization header would put the token in the repository."""
    leaky = RuntimeError(f"connection refused: Authorization: Bearer {SECRET}")
    fake = _FakeOpenAI(raises=leaky)
    with pytest.raises(GenerationError) as caught:
        _client(fake=fake).generate(_prompt(), model_tag="gemma4:12b", seed=3, temperature=0.7)
    text = f"{caught.value}\n{caught.value!r}"
    assert SECRET not in text
    assert "connection refused" in text


def test_the_api_key_is_not_in_a_digest_lookup_error(capsys: pytest.CaptureFixture[str]) -> None:
    fake = _FakeOpenAI()
    fake.models = SimpleNamespace(
        list=lambda: (_ for _ in ()).throw(RuntimeError(f"401 for key {SECRET}"))
    )
    assert _client(fake=fake).resolve_digest("gemma4:12b") == DIGEST_UNAVAILABLE
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err


def test_the_api_key_is_not_in_an_unresolvable_model_error() -> None:
    fake = _FakeOpenAI(models=[SimpleNamespace(id="a"), SimpleNamespace(id="b")])
    with pytest.raises(GenerationError) as caught:
        _client(fake=fake).generate(_prompt(), model_tag="gemma4:12b", seed=3, temperature=0.7)
    assert SECRET not in str(caught.value)


# ---------------------------------------------------------------------------
# The test-process guard
# ---------------------------------------------------------------------------


def test_a_test_process_cannot_open_a_live_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same rule the Ollama client enforces: a unit test that would have
    reached a served model fails in milliseconds instead of hanging on it."""
    monkeypatch.delenv("CARELITE_ALLOW_TEST_INFERENCE", raising=False)
    client = VLLMClient(base_url="https://pod.example/v1", api_key=SECRET)
    with pytest.raises(GenerationError, match="refusing to open a live model connection"):
        client.generate(_prompt(), model_tag="gemma4:12b", seed=3, temperature=0.7)
