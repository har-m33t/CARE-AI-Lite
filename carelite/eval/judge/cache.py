"""Append-only judge cache. **This is what makes a 1,080-generation run finishable.**

The full run is 1,080 generations judged single-pass, plus a validation subset
judged five times. At local-inference speeds that is hours of wall clock, and it
*will* be interrupted — a laptop sleeps, a terminal closes, Ollama restarts to
reload a model. A run that cannot resume is a run that never completes, so
resumability is a correctness requirement here, not an optimisation.

What is cached is the **raw model output**, keyed at sample granularity. That
choice matters more than it looks:

- Parsing, span grounding and median aggregation become pure functions of cached
  bytes, so the entire validation study can be recomputed offline in
  milliseconds, and a bug in the grounding rule is fixed by re-running the
  analysis rather than by re-judging for eight hours.
- The judge's actual words are preserved, so "why did it score this a 2" is
  answerable six months later.

The key covers everything that could change the answer: the generation, the
model digest (tags are mutable — v3 §16), the prompt version, the rubric
version, temperature, sample index, and anchor order. A rubric edit or a prompt
edit therefore *misses* the cache rather than silently blending two rubrics'
scores in one results table.

The format is JSONL, appended and flushed per record. A process killed
mid-write leaves one truncated final line, which `load()` skips with a count
rather than refusing to open the file — losing one cached sample is cheap,
losing the file is not.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["CachedSample", "JudgeCache", "cache_key"]


def cache_key(
    *,
    generation_id: str,
    model: str,
    digest: str,
    prompt_version: str,
    rubric_version: str,
    temperature: float,
    sample_idx: int,
    order: str,
) -> str:
    """Stable hash of everything that could change the judge's answer.

    Deliberately includes both `model` and `digest`: the digest is the real
    identity, the tag is what a human recognises in a cache file. Temperature is
    formatted to three decimals so `0.7` and `0.70000000000000004` are the same
    key — float repr drift silently doubling an eight-hour run is exactly the
    kind of bug that only shows up on the day of the run.
    """
    payload = "|".join(
        (
            generation_id,
            model,
            digest,
            prompt_version,
            rubric_version,
            f"{temperature:.3f}",
            str(sample_idx),
            order,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CachedSample:
    """One judge call and its raw answer, as stored on disk."""

    key: str
    generation_id: str
    model: str
    digest: str
    prompt_version: str
    rubric_version: str
    temperature: float
    sample_idx: int
    order: str
    seed: int | None
    raw_output: str
    latency_ms: int | None = None
    created_at: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "generation_id": self.generation_id,
            "model": self.model,
            "digest": self.digest,
            "prompt_version": self.prompt_version,
            "rubric_version": self.rubric_version,
            "temperature": self.temperature,
            "sample_idx": self.sample_idx,
            "order": self.order,
            "seed": self.seed,
            "raw_output": self.raw_output,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at or datetime.now(UTC).isoformat(),
        }

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> CachedSample:
        return cls(
            key=str(obj["key"]),
            generation_id=str(obj["generation_id"]),
            model=str(obj.get("model", "")),
            digest=str(obj.get("digest", "")),
            prompt_version=str(obj.get("prompt_version", "")),
            rubric_version=str(obj.get("rubric_version", "")),
            temperature=float(obj.get("temperature", 0.0)),
            sample_idx=int(obj.get("sample_idx", 0)),
            order=str(obj.get("order", "ascending")),
            seed=obj.get("seed"),
            raw_output=str(obj.get("raw_output", "")),
            latency_ms=obj.get("latency_ms"),
            created_at=str(obj.get("created_at", "")),
        )


@dataclass
class JudgeCache:
    """A JSONL file of `CachedSample`, read once and appended to thereafter.

    **Thread-safe within one process; still not multi-process safe.** A lock
    guards the in-memory dict and the append, and each record is written as one
    complete line under that lock, so concurrent workers cannot interleave a
    partial line. Two *processes* on one file still can — the lock does not
    reach across them — so one judging process at a time remains the rule.

    The original design said no locking, on the grounds that recovery cost beat
    parallelism "at this scale". The scale changed: the holdout run is 939
    single-pass judgements, and on rented GPU time billed by the hour the
    sequential version costs hours of wall clock and real money for nothing.
    A lock around an append is cheap; the fsync it already does dominates it.
    """

    path: Path
    #: Number of unparseable lines skipped on load. Non-zero after an interrupted
    #: write, and surfaced in the run report rather than swallowed.
    corrupt_lines: int = field(default=0, init=False)
    _records: dict[str, CachedSample] = field(default_factory=dict, init=False, repr=False)
    _handle: Any = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.load()

    def load(self) -> None:
        """Read the file into memory. Idempotent; safe on a missing file."""
        self._records.clear()
        self.corrupt_lines = 0
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    record = CachedSample.from_json(obj)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    # A truncated final line from an interrupted run. Skip it and
                    # count it; the sample will simply be re-judged.
                    self.corrupt_lines += 1
                    continue
                self._records[record.key] = record

    def get(self, key: str) -> CachedSample | None:
        with self._lock:
            return self._records.get(key)

    def put(self, record: CachedSample) -> None:
        """Append one record and flush it to the OS immediately.

        `flush` + `fsync` on every record is the whole point: a cache that loses
        the last N samples on a kill is a cache that makes an interrupted run
        restart further back than it needs to.
        """
        line = json.dumps(record.to_json(), ensure_ascii=False) + "\n"
        with self._lock:
            self._records[record.key] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self._handle is None:
                self._handle = self.path.open("a", encoding="utf-8")
            # One write of one complete line, under the lock: that is what makes
            # a concurrent reader of this file see whole records or nothing.
            self._handle.write(line)
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._records

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def __enter__(self) -> JudgeCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def generation_ids(self) -> set[str]:
        """Every generation with at least one cached sample."""
        return {r.generation_id for r in self._records.values()}

    def samples_for(self, generation_id: str) -> list[CachedSample]:
        """Cached samples for one generation, ordered by sample index."""
        return sorted(
            (r for r in self._records.values() if r.generation_id == generation_id),
            key=lambda r: (r.order, r.sample_idx),
        )
