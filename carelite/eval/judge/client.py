"""The model call, behind a one-method Protocol.

`ChatClient` is the seam. Everything above it — prompting, grounding,
aggregation, the whole validation study — is a pure function of strings and can
be tested without Ollama running, which is what keeps `make check` model-free
while still exercising the logic that actually decides the study's numbers. The
only code that needs a live model is `OllamaChatClient` and the handful of tests
marked `@pytest.mark.inference`.

Independence, since v3 §13 asks for it to be reported prominently: the judge is
`gpt-oss:20b` and the generator is `gemma4:12b`. Different families, different
training data, different post-training. Both tags come from `settings.models`;
neither is written down here, because a hardcoded tag in the judge is exactly
how a study ends up reporting a model it did not run.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from carelite.config import get_settings

__all__ = [
    "ChatClient",
    "JudgeCallError",
    "OllamaChatClient",
    "ReplayClient",
]


class JudgeCallError(RuntimeError):
    """The model could not be reached, or returned nothing usable."""


@runtime_checkable
class ChatClient(Protocol):
    """One call to a chat model. Deterministic given `(messages, temperature, seed)`.

    Implementations must return the assistant's message content as a string.
    Reasoning channels (gpt-oss emits one) are the implementation's problem to
    strip or ignore; the parser tolerates them either way.
    """

    @property
    def model(self) -> str:
        """The model tag, for provenance."""
        ...

    @property
    def digest(self) -> str:
        """The model digest, or the tag if no digest has been pinned.

        Tags are mutable in Ollama, so the digest is the real identity of what
        was run. It is part of the judge cache key for the same reason it is
        part of the generation cache key in v3 §16.
        """
        ...

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        seed: int | None = None,
    ) -> str: ...


@dataclass
class OllamaChatClient:
    """`ChatClient` over a local Ollama server. The only networked class in the lane.

    `format="json"` is requested because the judge prompt asks for a JSON object
    and constrained decoding removes most of the parser's work. The parser stays
    tolerant anyway — `format="json"` is a request, not a guarantee, and a model
    that appends a sentence after the object should not cost us a cell.
    """

    model_tag: str = ""
    host: str = ""
    num_ctx: int | None = None
    max_attempts: int = 3
    retry_backoff_s: float = 2.0
    request_json: bool = True
    _digest: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        settings = get_settings()
        spec = settings.models.judge
        self.model_tag = self.model_tag or spec.tag
        self.host = self.host or settings.ollama_host
        if self.num_ctx is None:
            self.num_ctx = spec.context_window
        self._digest = spec.digest or ""

    @property
    def model(self) -> str:
        return self.model_tag

    @property
    def digest(self) -> str:
        # Falls back to the tag rather than to an empty string: an empty digest
        # in a cache key would collide across models, which is the one failure
        # mode a cache must not have.
        return self._digest or self.model_tag

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        seed: int | None = None,
    ) -> str:
        import ollama  # imported here so the module is importable without it

        options: dict[str, Any] = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx

        client = ollama.Client(host=self.host)
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = client.chat(
                    model=self.model_tag,
                    messages=list(messages),
                    options=options,
                    format="json" if self.request_json else None,
                )
            except Exception as exc:  # network, model-not-pulled, timeout
                last = exc
                if attempt < self.max_attempts:
                    time.sleep(self.retry_backoff_s * attempt)
                continue

            content = _content_of(response)
            if content.strip():
                return content
            last = JudgeCallError("judge returned an empty message")
            if attempt < self.max_attempts:
                time.sleep(self.retry_backoff_s * attempt)

        raise JudgeCallError(
            f"judge call to {self.model_tag} at {self.host} failed after "
            f"{self.max_attempts} attempts: {last}"
        ) from last


def _content_of(response: Any) -> str:
    """Pull the assistant text out of whatever shape the client returned.

    The `ollama` package has returned dicts and pydantic objects across
    versions; both are handled rather than pinning to one and breaking on the
    next release.
    """
    message = getattr(response, "message", None)
    if message is None and isinstance(response, dict):
        message = response.get("message")
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return str(content or "")


@dataclass
class ReplayClient:
    """Serves canned outputs in order. For tests, and for re-parsing a cached run.

    Not a mock in the testing-library sense — it is the same object the runner
    uses when every sample is already in the cache, which is why it lives in the
    package rather than in `conftest.py`.
    """

    outputs: list[str] = field(default_factory=list)
    model_tag: str = "replay"
    digest_value: str = "replay-digest"
    calls: list[dict[str, Any]] = field(default_factory=list, init=False)
    _cursor: int = field(default=0, init=False)

    @property
    def model(self) -> str:
        return self.model_tag

    @property
    def digest(self) -> str:
        return self.digest_value

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        seed: int | None = None,
    ) -> str:
        self.calls.append({"messages": list(messages), "temperature": temperature, "seed": seed})
        if self._cursor >= len(self.outputs):
            raise JudgeCallError(
                f"ReplayClient exhausted after {len(self.outputs)} outputs; "
                f"call {self._cursor + 1} has nothing to return"
            )
        out = self.outputs[self._cursor]
        self._cursor += 1
        return out
