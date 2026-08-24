"""carelite.retrieval.llm — the one small chat helper this lane calls.

Three components in this package can optionally consult a model: `hyde.py`
(always, when HyDE is enabled), `router.py` (only under `use_llm_router`), and
`crag.py` (only under `use_llm_crag`). They share this wrapper so that the
properties the study depends on are guaranteed in one place rather than three:

**Every call goes through `carelite.safety.fencing`.** The patient utterance
is untrusted by assumption and retrieved corpus text is untrusted because its
contextual prefixes are LLM-generated. There is no code path in this package
that concatenates either into a system prompt; `chat()` takes a *trusted*
system template and untrusted material as separate keyword arguments and
hands them to `fencing.assemble`, which raises `FencingViolation` if they are
ever wired up backwards.

**Determinism.** Temperature defaults to 0 and the seed defaults to
`settings.experiment.base_seed`, because a retrieval component that varies
run-to-run would put variance inside the independent variable of a controlled
comparison (v3 §14). HyDE passages are additionally cached by content hash.

**Failure is a degraded mode, not an exception.** `chat()` returns `None` if
Ollama is unreachable, the model is not pulled, or the call times out. Every
caller in this package treats `None` as "this optional component did not
run" and records that in the trace, because a retrieval pipeline that raises
when a *hypothetical passage generator* is down would take the whole bedside
interface with it.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carelite.config import get_settings
from carelite.safety import fencing

__all__ = ["ChatResult", "LLMClient", "NullLLMClient"]


@dataclass(frozen=True, slots=True)
class ChatResult:
    text: str
    model: str
    latency_ms: int
    cached: bool = False


@dataclass
class LLMClient:
    """A deterministic, fenced, failure-tolerant Ollama chat client.

    `model_tag` defaults to the generator; `crag.py` passes the *judge* tag
    instead so that grading retrieved context is done by a different model
    family from the one that will later be judged on its use (v3 §13's
    independence requirement, applied one stage earlier).
    """

    model_tag: str = ""
    host: str = ""
    temperature: float = 0.0
    seed: int | None = None
    num_predict: int = 400
    cache_path: Path | None = None
    use_cache: bool = True
    client: Any | None = None

    _cache: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _handle: Any = field(default=None, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        settings = get_settings()
        self.model_tag = self.model_tag or settings.models.generator.tag
        self.host = self.host or settings.ollama_host
        if self.seed is None:
            self.seed = settings.experiment.base_seed
        if self.cache_path is None:
            self.cache_path = settings.runs_dir / "retrieval" / "llm_cache.jsonl"

    # -- cache --------------------------------------------------------------

    def _load_cache(self) -> None:
        if self._loaded or not self.use_cache:
            self._loaded = True
            return
        self._loaded = True
        path = Path(self.cache_path) if self.cache_path else None
        if path is None or not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self._cache[str(obj["key"])] = str(obj["text"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue  # a torn final line from an interrupted run

    def _cache_put(self, key: str, text: str) -> None:
        if not self.use_cache or self.cache_path is None:
            return
        self._cache[key] = text
        path = Path(self.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._handle is None:
                self._handle = path.open("a", encoding="utf-8")
            self._handle.write(json.dumps({"key": key, "text": text}) + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        *,
        system: str,
        task: str | None = None,
        utterance: str | None = None,
        retrieved: Sequence[Any] = (),
        history: Sequence[str] = (),
        extra_untrusted: Sequence[tuple[str, str]] = (),
        json_schema: dict[str, Any] | None = None,
        num_predict: int | None = None,
    ) -> ChatResult | None:
        """Assemble a fenced prompt and run it. `None` on any failure.

        `system` must be trusted, git-tracked template text — this method
        does not, and cannot, sanitise it. Everything untrusted goes in the
        other arguments and lands inside a fence in the user turn.
        """
        import time

        prompt = fencing.assemble(
            system=system,
            task=task,
            utterance=utterance,
            retrieved=retrieved,
            history=list(history),
            extra_untrusted=list(extra_untrusted),
        )

        key = _cache_key(self.model_tag, prompt.render(), self.temperature, self.seed)
        self._load_cache()
        hit = self._cache.get(key)
        if hit is not None:
            return ChatResult(text=hit, model=self.model_tag, latency_ms=0, cached=True)

        started = time.monotonic()
        try:
            client = self.client
            if client is None:
                import ollama

                client = ollama.Client(host=self.host)
            kwargs: dict[str, Any] = {
                "model": self.model_tag,
                "messages": prompt.as_messages(),
                "options": {
                    "temperature": self.temperature,
                    "seed": self.seed,
                    "num_predict": num_predict or self.num_predict,
                },
            }
            if json_schema is not None:
                kwargs["format"] = json_schema
            text = _invoke(client, kwargs)
        except Exception:
            # Unreachable host, model not pulled, OOM, timeout, malformed
            # response. Every caller degrades; none of them raise.
            return None

        latency_ms = int((time.monotonic() - started) * 1000)
        if text:
            self._cache_put(key, text)
        return ChatResult(text=text, model=self.model_tag, latency_ms=latency_ms)


class NullLLMClient(LLMClient):
    """An `LLMClient` that never calls a model. Used by unit tests and by any
    ablation row that must be reproducible without a live Ollama."""

    def chat(self, **kwargs: Any) -> ChatResult | None:
        return None


def _invoke(client: Any, kwargs: dict[str, Any]) -> str:
    """Call `chat`, working around a measured reasoning-model interaction.

    `think=False` is worth sending: it suppresses the visible chain-of-thought
    that reasoning models emit, which otherwise has to be stripped and which
    burns the `num_predict` budget. But **`think=False` combined with a
    `format` JSON schema makes `gpt-oss:20b` return empty content** — measured
    directly against the live model, at three seeds and two `num_predict`
    budgets, every combination returned `content=''` with a thinking length of
    zero, while the identical call without `think=False` returned correct JSON
    every time. The model appears to need its reasoning budget in order to
    satisfy a constrained decode.

    This cost the CRAG gate its entire LLM grader once already: every grade
    silently fell back to the score grader, which by design cannot detect an
    off-domain turn, and the fallback looked like a working gate. So the ladder
    below retries rather than trusting the first empty answer, and the retry is
    unconditional on *any* empty response, not special-cased to one model tag —
    a tag-specific workaround would rot the moment the roster changes.
    """
    attempts: list[dict[str, Any]] = [{"think": False}, {}]
    for extra in attempts:
        try:
            response = client.chat(**kwargs, **extra)
        except TypeError:
            # An older ollama client that does not accept `think` at all.
            response = client.chat(**kwargs)
        text = str(response["message"]["content"]).strip()
        if text:
            return text
    return ""


def _cache_key(model: str, prompt: str, temperature: float, seed: int | None) -> str:
    payload = f"{model}|{temperature}|{seed}|{prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
