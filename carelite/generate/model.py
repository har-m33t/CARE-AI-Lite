"""The generation client: one Ollama call, pinned to a digest, failing loudly.

Deliberately not `carelite.retrieval.llm.LLMClient`, and the difference is the
error policy. That client returns `None` when Ollama is unreachable, because
every one of its callers is an optional component — a HyDE passage that did not
get generated degrades retrieval and is recorded as such. A *generation* has no
degraded mode. A missing cell in a 1,080-cell results table is a hole, and a
hole that arrives as `None` gets written as an empty response and analysed as
if the model had said nothing. So this client raises.

Three properties the study depends on:

**The digest, not the tag.** Ollama tags are mutable — `ollama pull gemma4:12b`
tomorrow can give different weights under the same name. `resolve_digest()`
reads the digest from the running daemon and every generation row records it,
so a results table can be partitioned by what actually produced it. Digests are
cached per process; they do not change under a running daemon.

**`num_ctx` is always set explicitly.** Ollama's default context is a few
thousand tokens regardless of what the model supports, so a long-context
condition that does not set it is silently truncated and the baseline measures
nothing. `context_size()` derives it from the assembled prompt by one rule
applied identically in every condition, rather than per-condition constants
that would put a second difference into the comparison.

**Determinism as far as the runtime allows.** Temperature and seed come from
the caller — `config.seed_for` upstream — and are sent on every call. This
makes a run repeatable against the same daemon and the same weights; it is not
a claim of bit-exactness across GPUs or Ollama versions, which no local stack
offers, and the digest is what lets a later reader tell those apart.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from carelite.config import get_settings
from carelite.safety.fencing import FencedPrompt

__all__ = [
    "ALLOW_TEST_INFERENCE_ENV",
    "DIGEST_UNAVAILABLE",
    "GenerationClient",
    "GenerationError",
    "GenerationOutput",
    "ModelClient",
    "context_size",
    "estimate_tokens",
]

#: Set to `1` to let a test process open a model connection. See `_refuse_in_tests`.
ALLOW_TEST_INFERENCE_ENV = "CARELITE_ALLOW_TEST_INFERENCE"

#: Recorded in place of a digest when the daemon cannot be asked for one. The
#: `generation.model_digest` column is NOT NULL, and writing a plausible-looking
#: placeholder that a reader might mistake for a real digest would be worse than
#: a value that announces itself.
DIGEST_UNAVAILABLE = "unavailable"

#: Smallest context we ever ask for. Below this, a framework prompt plus a few
#: retrieved passages would start to crowd the output budget.
MIN_CONTEXT = 8192


class GenerationError(RuntimeError):
    """The model did not produce a response. Never swallowed; see the docstring."""


def _refuse_in_tests() -> None:
    """Stop a unit-test process from opening a connection to a live model.

    A unit test that reaches a model is slow, non-deterministic, and dependent
    on a daemon that may not be running — which is why `pyproject.toml` has an
    `inference` marker and `make check` excludes it. This makes that a property
    of the client rather than a convention: a test that would have called a
    model gets a `GenerationError` in milliseconds instead of blocking for
    minutes behind whatever else is queued on the daemon.

    It only fires when no client was injected. Every test in
    `tests/unit/generate/` passes a fake and is unaffected, and nothing here
    runs outside pytest, so production behaviour is untouched.

    A test that genuinely means to hit a model marks itself `inference` and sets
    `CARELITE_ALLOW_TEST_INFERENCE=1`.
    """
    if "pytest" not in sys.modules:
        return
    if os.environ.get(ALLOW_TEST_INFERENCE_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return
    raise GenerationError(
        "refusing to open a live model connection from a test process. A unit test "
        "must not depend on a running Ollama daemon: inject a fake client into "
        "GraphDeps, or patch carelite.cli.engine.resolve_engine to return a stub. "
        f"A test that really means to reach a model marks itself `inference` and sets "
        f"{ALLOW_TEST_INFERENCE_ENV}=1."
    )


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    text: str
    model: str
    model_digest: str
    latency_ms: int
    num_ctx: int
    prompt_chars: int
    served_by: str = "ollama"
    """Which serving stack produced this text, carried through to the
    `generation.served_by` column. Defaults to Ollama because that is what
    produced every row in the database before a second backend existed, and
    because a default that has to be remembered is a default that gets it
    wrong."""


def estimate_tokens(text: str) -> int:
    """~4 characters per token, the usual English approximation.

    A budget check, not a count. It is used to size `num_ctx` and to decide how
    much corpus fits in the long-context condition, and in both places an
    approximation that errs high is the safe direction.
    """
    return len(text) // 4 + 1


def context_size(prompt_chars: int, *, num_predict: int, window: int) -> int:
    """`num_ctx` for a prompt of this size: one rule, every condition.

    Rounded up to a power of two so a small change in prompt length does not
    change the allocated context and, with it, potentially the output. Floored
    at `MIN_CONTEXT` and capped at the model's configured window.
    """
    needed = prompt_chars // 4 + 1 + num_predict + 256
    size = MIN_CONTEXT
    while size < needed:
        size *= 2
    return min(max(size, MIN_CONTEXT), window)


@runtime_checkable
class ModelClient(Protocol):
    """What the graph needs from a serving stack, and nothing more.

    Two implementations satisfy it: `GenerationClient` against Ollama and
    `carelite.generate.backend.VLLMClient` against an OpenAI-compatible vLLM
    server. They are selected by `CARELITE_BACKEND` and swapped at the client,
    which is the whole of the change — the graph, the nodes, the six conditions
    and the runner are identical under both, so a backend comparison is a
    comparison of serving stacks rather than of two pipelines.

    `served_by` is the value that lands in `generation.served_by`. It is a
    property of the client rather than an argument the caller passes, because a
    caller that has to remember to label a row correctly eventually does not.
    """

    @property
    def served_by(self) -> str:
        """The `generation.served_by` value: `'ollama'` or `'vllm'`.

        Declared read-only so an implementation can satisfy it with a
        `ClassVar` — which is what both do, because the serving stack is a
        property of the client class and not of an instance's configuration.
        """

    def resolve_digest(self, model_tag: str) -> str: ...

    def generate(
        self,
        prompt: FencedPrompt,
        *,
        model_tag: str,
        seed: int,
        temperature: float,
        num_predict: int = 512,
        window: int = 8192,
        json_format: bool = False,
    ) -> GenerationOutput: ...


@dataclass
class GenerationClient:
    """A pinned, fenced, loud Ollama chat client for the generator models.

    `client` is injectable so the whole graph can be driven in a unit test with
    no daemon running.
    """

    served_by: ClassVar[str] = "ollama"

    host: str = ""
    client: Any | None = None
    timeout_s: float = 600.0
    """Wall clock for one generation. Ten minutes is generous for a few hundred
    tokens and is sized for the long-context condition, whose prompt is over
    120,000 tokens, and for a daemon shared with other work."""

    metadata_timeout_s: float = 30.0
    """Separate, much shorter clock for `list()`. Asking the daemon what it has
    is a cheap metadata call, and it happens *before* the plan is built — so a
    daemon that is wedged or saturated would otherwise stall a 1,080-cell run
    at the pre-flight, with nothing written and nothing to resume from. Failing
    fast here costs the digest, which `DIGEST_UNAVAILABLE` records honestly."""

    _digests: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _pool: dict[float, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.host = self.host or get_settings().ollama_host

    # -- daemon -------------------------------------------------------------

    def _handle(self, *, timeout_s: float | None = None) -> Any:
        """One cached `ollama.Client` per distinct timeout.

        Two, in practice: the long one for generation and the short one for
        metadata. Cached rather than rebuilt per call so a 1,080-cell run reuses
        its connection pool instead of opening one per generation.
        """
        if self.client is not None:
            return self.client
        _refuse_in_tests()
        timeout = timeout_s if timeout_s is not None else self.timeout_s
        handle = self._pool.get(timeout)
        if handle is None:
            import ollama

            handle = ollama.Client(host=self.host, timeout=timeout)
            self._pool[timeout] = handle
        return handle

    def resolve_digest(self, model_tag: str) -> str:
        """The digest the daemon currently serves for this tag.

        Returns `DIGEST_UNAVAILABLE` rather than raising when the daemon cannot
        be listed: an unrecorded digest degrades provenance, while a failure
        here would take down a run whose generations are otherwise fine. A
        failure to *generate* still raises.
        """
        if model_tag in self._digests:
            return self._digests[model_tag]
        digest = DIGEST_UNAVAILABLE
        try:
            listing = self._handle(timeout_s=self.metadata_timeout_s).list()
            models = getattr(listing, "models", None)
            if models is None and isinstance(listing, dict):
                models = listing.get("models", [])
            for entry in models or []:
                name = getattr(entry, "model", None) or (
                    entry.get("model") or entry.get("name") if isinstance(entry, dict) else None
                )
                if name is None:
                    continue
                if str(name) == model_tag or str(name) == f"{model_tag}:latest":
                    raw = getattr(entry, "digest", None) or (
                        entry.get("digest") if isinstance(entry, dict) else None
                    )
                    if raw:
                        digest = str(raw)
                    break
        except Exception:
            digest = DIGEST_UNAVAILABLE
        self._digests[model_tag] = digest
        return digest

    # -- generation ---------------------------------------------------------

    def generate(
        self,
        prompt: FencedPrompt,
        *,
        model_tag: str,
        seed: int,
        temperature: float,
        num_predict: int = 512,
        window: int = 8192,
        json_format: bool = False,
    ) -> GenerationOutput:
        """Run one generation. Raises `GenerationError` on any failure.

        `prompt` must come from `fencing.assemble`; this method takes a
        `FencedPrompt` rather than strings precisely so there is no way to hand
        it an unfenced system/user pair.
        """
        digest = self.resolve_digest(model_tag)
        prompt_chars = len(prompt)
        num_ctx = context_size(prompt_chars, num_predict=num_predict, window=window)

        options: dict[str, Any] = {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        }
        kwargs: dict[str, Any] = {
            "model": model_tag,
            "messages": prompt.as_messages(),
            "options": options,
        }
        if json_format:
            kwargs["format"] = "json"

        started = time.monotonic()
        try:
            text = _invoke(self._handle(), kwargs)
        except Exception as exc:
            raise GenerationError(
                f"generation failed on {model_tag!r} (num_ctx={num_ctx}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        latency_ms = int((time.monotonic() - started) * 1000)

        if not text.strip():
            raise GenerationError(
                f"{model_tag!r} returned an empty response (num_ctx={num_ctx}, "
                f"num_predict={num_predict}). An empty cell is a hole in the results "
                "table, not a result."
            )
        return GenerationOutput(
            text=text.strip(),
            model=model_tag,
            model_digest=digest,
            latency_ms=latency_ms,
            num_ctx=num_ctx,
            prompt_chars=prompt_chars,
            served_by=self.served_by,
        )


def _invoke(client: Any, kwargs: dict[str, Any]) -> str:
    """Call `chat`, suppressing visible chain-of-thought where supported.

    `think=False` stops reasoning models emitting their scratchpad into the
    response, which would otherwise have to be stripped and would eat the
    `num_predict` budget. Older `ollama` clients reject the argument, and
    `carelite.retrieval.llm` documents a measured case where `think=False`
    combined with a constrained decode returns empty content — so an empty
    first answer is retried without it rather than treated as the model's
    answer.
    """
    for extra in ({"think": False}, {}):
        try:
            response = client.chat(**kwargs, **extra)
        except TypeError:
            response = client.chat(**kwargs)
        text = str(response["message"]["content"])
        if text.strip():
            return text
    return ""
