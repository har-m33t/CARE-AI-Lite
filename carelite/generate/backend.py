"""Which serving stack runs the models, and how a vLLM-served model is identified.

    CARELITE_BACKEND=vllm \\
    CARELITE_VLLM_BASE_URL=https://<pod>-8000.proxy.runpod.net/v1 \\
    CARELITE_VLLM_API_KEY=...  python -m carelite.generate.runner --conditions LC

One environment variable chooses between a local Ollama daemon and a remote
vLLM server speaking the OpenAI-compatible API. **Nothing above the client
changes**: the graph, the six conditions, the prompts, the seeds and the runner
are byte-identical under both, because a backend that also changed the pipeline
would make a cross-backend comparison uninterpretable. `served_by` is recorded
on every row so the two can be told apart afterwards, and
`carelite/db/schema.sql` says why that column has to exist rather than being
inferred from `model`.

**Why there is a second backend at all.** D11 dropped condition LC on a measured
3.3 minutes per cell, and named the cause: every LC prompt shares an identical
~119,500-token prefix that Ollama re-prefills on every request instead of
reusing from its KV cache. vLLM's automatic prefix caching is the mechanism
whose absence that measured. Whether it delivers the saving is a question for a
benchmark, not for this module; what this module does is make the measurement
possible without touching the code under test.

**Identifying a vLLM-served model.** Ollama's digest names a GGUF blob. vLLM
serves HF safetensors, and there is no equivalent blob hash to ask for, so the
identity recorded here is `vllm:<served repo id>@<revision>` — the repo the
server says it is serving, and the commit it was launched at. The revision comes
from the server's `/v1/models` entry when it reports one, and otherwise from
`CARELITE_VLLM_MODEL_REVISION`, which is the operator's record of the
`--revision` the pod was started with. **When neither is available the digest is
`DIGEST_UNAVAILABLE` and `runner.assert_digests_resolved` refuses the run.**
Synthesising something digest-shaped from the tag would defeat the entire point
of the column, which exists because tags are mutable.

**The credential.** `CARELITE_VLLM_API_KEY` is a bearer token for a public RunPod proxy
URL. It is kept out of `repr()` by the dataclass field, and every message this
module raises or prints is passed through `_scrub`, which removes the token
before the text can reach a log — and run logs get committed. There is no code
path here that writes it to a file.

**Configuration is read from the environment, not from `carelite/config.py`.**
That module is a frozen contract this lane does not own: adding a
`vllm_base_url` field to `Settings` would be a contract change, and its
`env_prefix` means an unprefixed `VLLM_BASE_URL` could not be read through it
either way. `.env` is consulted as a fallback because that is where the project
keeps local secrets and where the settings object reads its own; the parse is
deliberately minimal and never echoes a value or a key name it found. Both the
`CARELITE_`-prefixed and the bare spellings are accepted — see `ALIASES`.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

from carelite.config import REPO_ROOT
from carelite.generate.model import (
    DIGEST_UNAVAILABLE,
    GenerationClient,
    GenerationError,
    GenerationOutput,
    ModelClient,
    _refuse_in_tests,
    context_size,
)
from carelite.safety.fencing import FencedPrompt

__all__ = [
    "ALIASES",
    "API_KEY_ENV",
    "BACKEND_ENV",
    "BASE_URL_ENV",
    "MODEL_ENV",
    "REVISION_ENV",
    "Backend",
    "VLLMClient",
    "default_client",
    "selected_backend",
]

#: `ollama` (the default) or `vllm`.
BACKEND_ENV = "CARELITE_BACKEND"
#: The OpenAI-compatible root, including `/v1`.
BASE_URL_ENV = "CARELITE_VLLM_BASE_URL"
#: Bearer token for the server's `--api-key`. Never logged, never committed.
API_KEY_ENV = "CARELITE_VLLM_API_KEY"  # pragma: allowlist secret - a variable name
#: The served repo id, when the server hosts more than one or names it
#: differently from the roster tag.
MODEL_ENV = "CARELITE_VLLM_MODEL"
#: The HF commit the server was launched at, for when it does not report one.
REVISION_ENV = "CARELITE_VLLM_MODEL_REVISION"

#: The unprefixed spellings the design record uses, accepted as aliases.
#:
#: `carelite/config.py` reads `.env` with an `env_prefix` of `CARELITE_`, and the
#: repository's `.env` already carries these under that prefix, so the prefixed
#: name is the project's own idiom and is what is checked first. The bare names
#: are what a vLLM operator would export by hand, and refusing them would be a
#: gratuitous way to make a run silently fall back to Ollama.
ALIASES: dict[str, str] = {
    BASE_URL_ENV: "VLLM_BASE_URL",
    API_KEY_ENV: "VLLM_API_KEY",  # pragma: allowlist secret - a variable name
    MODEL_ENV: "VLLM_MODEL",
    REVISION_ENV: "VLLM_MODEL_REVISION",
}


class Backend(StrEnum):
    OLLAMA = "ollama"
    VLLM = "vllm"


def _dotenv() -> dict[str, str]:
    """`KEY=value` pairs from the repository's `.env`, if there is one.

    The same file `carelite.config.Settings` reads. It is gitignored, and
    nothing here logs a value or a key name it found.
    """
    path = REPO_ROOT / ".env"
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, raw = line.partition("=")
        values[name.strip()] = raw.strip().strip("'\"")
    return values


def _env(name: str, default: str = "") -> str:
    """One setting, from the process environment first and `.env` second.

    Each name is tried under its `CARELITE_` spelling and then under the bare
    one from `ALIASES`, in both places.
    """
    names = [name] + ([ALIASES[name]] if name in ALIASES else [])
    for source in (os.environ, _dotenv()):
        for candidate in names:
            raw = source.get(candidate)
            if raw is not None and raw.strip():
                return raw.strip()
    return default


def selected_backend() -> Backend:
    """The configured backend. Raises on anything that is not one of the two."""
    raw = _env(BACKEND_ENV, Backend.OLLAMA.value).lower()
    try:
        return Backend(raw)
    except ValueError:
        raise ValueError(
            f"{BACKEND_ENV}={raw!r} is not a serving stack this project has. "
            f"Use {Backend.OLLAMA.value!r} or {Backend.VLLM.value!r}."
        ) from None


def default_client() -> ModelClient:
    """The client the configured backend calls for. What `GraphDeps` defaults to."""
    if selected_backend() is Backend.VLLM:
        return VLLMClient()
    return GenerationClient()


def _attr(entry: Any, name: str) -> Any:
    """One field off an OpenAI `Model`, however that SDK version carries it.

    vLLM returns fields the OpenAI schema does not declare — `max_model_len`,
    sometimes `revision` — so depending on the SDK version they land on the
    object, in `model_extra`, or only in a raw dict.
    """
    value = getattr(entry, name, None)
    if value is not None:
        return value
    extra = getattr(entry, "model_extra", None)
    if isinstance(extra, dict) and extra.get(name) is not None:
        return extra[name]
    if isinstance(entry, dict):
        return entry.get(name)
    return None


@dataclass
class VLLMClient:
    """An OpenAI-compatible chat client for a remote vLLM server.

    Same contract as `GenerationClient` — `ModelClient` — and the same error
    policy: a generation has no degraded mode, so a failure raises rather than
    returning empty text that would be analysed as if the model had said
    nothing.

    Three differences from the Ollama client, all forced by the runtime:

    **`num_ctx` records the server's window, not a requested one.** Ollama takes
    `num_ctx` per request and truncates silently without it. vLLM allocates once
    at launch from `--max-model-len`, so the honest value for the row is what the
    server reports, and `context_size()` is used only as a fallback when it
    reports nothing.

    **The digest is a repo id and a revision.** See the module docstring.

    **`api_key` is out of `repr` and out of every message.** See `_scrub`.
    """

    served_by: ClassVar[str] = "vllm"

    base_url: str = ""
    api_key: str = field(default="", repr=False)
    model: str = ""
    """The served repo id. Empty means "ask the server", which is unambiguous
    exactly when it serves one model — which is the deployment this project
    runs."""

    revision: str = ""
    timeout_s: float = 600.0
    metadata_timeout_s: float = 30.0
    client: Any | None = None
    """Injectable, so the whole path is testable against a fake endpoint. There
    is no vLLM server to test against and there is not going to be one in
    `make check`."""

    _digests: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _served: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url or _env(BASE_URL_ENV)
        self.api_key = self.api_key or _env(API_KEY_ENV)
        self.model = self.model or _env(MODEL_ENV)
        self.revision = self.revision or _env(REVISION_ENV)
        if not self.base_url:
            raise GenerationError(
                f"{BACKEND_ENV}={Backend.VLLM.value} but {BASE_URL_ENV} is not set. "
                "Refusing to fall back to Ollama silently: rows generated that way "
                "would be labelled with the wrong serving stack."
            )

    # -- the credential -----------------------------------------------------

    def _scrub(self, text: str) -> str:
        """Remove the bearer token from any text that might be printed.

        A transport error can echo the request headers, and run logs get
        committed. Everything this class raises or prints goes through here.
        """
        if not self.api_key:
            return text
        return text.replace(self.api_key, "***")

    # -- server -------------------------------------------------------------

    def _handle(self, *, timeout_s: float | None = None) -> Any:
        if self.client is not None:
            return self.client
        _refuse_in_tests()
        import openai

        # Constructed per distinct timeout for the same reason the Ollama client
        # is: metadata gets a short clock so a wedged server fails at pre-flight
        # instead of stalling a run that has written nothing.
        return openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "EMPTY",
            timeout=timeout_s if timeout_s is not None else self.timeout_s,
            max_retries=2,
        )

    def _catalogue(self) -> list[Any]:
        """`/v1/models`, or an empty list. Never raises, never prints the key."""
        if "entries" in self._served:
            entries: list[Any] = self._served["entries"]
            return entries
        try:
            listing = self._handle(timeout_s=self.metadata_timeout_s).models.list()
            entries = list(getattr(listing, "data", None) or [])
        except GenerationError:
            # The test-process guard, not a server fault. Swallowing it here
            # would turn "a unit test tried to reach a live model" into
            # "the server reports no models", which is a much harder thing to
            # read in a failure.
            raise
        except Exception as exc:
            print(
                "carelite.generate.backend: could not list models on the vLLM server "
                f"({self._scrub(f'{type(exc).__name__}: {exc}')})",
                file=sys.stderr,
            )
            entries = []
        self._served["entries"] = entries
        return entries

    def served_model(self, model_tag: str) -> str:
        """The repo id to send for a roster tag.

        `VLLM_MODEL` when set, otherwise the server's single served model. A
        server hosting several is ambiguous and says so rather than picking one:
        generating condition LC against the wrong weights is not a failure that
        announces itself in the output.
        """
        if self.model:
            return self.model
        entries = self._catalogue()
        ids = [str(_attr(e, "id")) for e in entries if _attr(e, "id")]
        if len(ids) == 1:
            return ids[0]
        raise GenerationError(
            self._scrub(
                f"cannot tell which served model {model_tag!r} means: the server at "
                f"{self.base_url} reports {len(ids)} models. Set {MODEL_ENV} to the "
                "repo id this run should generate against."
            )
        )

    def _entry(self, served: str) -> Any | None:
        for entry in self._catalogue():
            if str(_attr(entry, "id")) == served:
                return entry
        return None

    def resolve_digest(self, model_tag: str) -> str:
        """`vllm:<repo id>@<revision>`, or `DIGEST_UNAVAILABLE`.

        Returns rather than raises when the server cannot be asked, matching the
        Ollama client: an unrecorded digest degrades provenance, and the runner
        turns it into a pre-flight refusal before any row is written.
        """
        if model_tag in self._digests:
            return self._digests[model_tag]
        digest = DIGEST_UNAVAILABLE
        try:
            served = self.served_model(model_tag)
            entry = self._entry(served)
            revision = str(_attr(entry, "revision") or "") if entry is not None else ""
            revision = revision or self.revision
            if revision:
                digest = f"vllm:{served}@{revision}"
        except GenerationError:
            digest = DIGEST_UNAVAILABLE
        self._digests[model_tag] = digest
        return digest

    def served_window(self, served: str, *, fallback: int) -> int:
        entry = self._entry(served)
        raw = _attr(entry, "max_model_len") if entry is not None else None
        try:
            return int(raw) if raw else fallback
        except (TypeError, ValueError):
            return fallback

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

        Takes a `FencedPrompt` for the same reason the Ollama client does: there
        is no way to hand this an unfenced system/user pair.
        """
        served = self.served_model(model_tag)
        digest = self.resolve_digest(model_tag)
        prompt_chars = len(prompt)
        asked = context_size(prompt_chars, num_predict=num_predict, window=window)
        num_ctx = self.served_window(served, fallback=asked)

        kwargs: dict[str, Any] = {
            "model": served,
            "messages": prompt.as_messages(),
            "temperature": temperature,
            "seed": seed,
            "max_tokens": num_predict,
        }
        if json_format:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.monotonic()
        try:
            response = self._handle().chat.completions.create(**kwargs)
            text = str(response.choices[0].message.content or "")
        except Exception as exc:
            raise GenerationError(
                self._scrub(
                    f"generation failed on {served!r} via vLLM (num_ctx={num_ctx}): "
                    f"{type(exc).__name__}: {exc}"
                )
            ) from None
        latency_ms = int((time.monotonic() - started) * 1000)

        if not text.strip():
            raise GenerationError(
                f"{served!r} returned an empty response via vLLM (num_ctx={num_ctx}, "
                f"max_tokens={num_predict}). An empty cell is a hole in the results "
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
