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

import time
from dataclasses import dataclass, field
from typing import Any

from carelite.config import get_settings
from carelite.safety.fencing import FencedPrompt

__all__ = [
    "DIGEST_UNAVAILABLE",
    "GenerationClient",
    "GenerationError",
    "GenerationOutput",
    "context_size",
    "estimate_tokens",
]

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


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    text: str
    model: str
    model_digest: str
    latency_ms: int
    num_ctx: int
    prompt_chars: int


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


@dataclass
class GenerationClient:
    """A pinned, fenced, loud Ollama chat client for the generator models.

    `client` is injectable so the whole graph can be driven in a unit test with
    no daemon running.
    """

    host: str = ""
    client: Any | None = None
    timeout_s: float = 600.0

    _digests: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.host = self.host or get_settings().ollama_host

    # -- daemon -------------------------------------------------------------

    def _handle(self) -> Any:
        if self.client is not None:
            return self.client
        import ollama

        self.client = ollama.Client(host=self.host, timeout=self.timeout_s)
        return self.client

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
            listing = self._handle().list()
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
