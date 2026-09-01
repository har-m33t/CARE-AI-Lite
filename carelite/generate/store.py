"""Where generations go, and the cache key that makes a run resumable.

The key is the one build plan v3 section 16 specifies:

    (scenario_id, condition, prompt_id, model_digest, seed, sample_idx)

It is also the `UNIQUE` constraint on `generation` in the frozen schema, so
"completed cells are skipped" and "the database refuses a duplicate" are the
same statement rather than two mechanisms that can disagree.

**Resumability is a property of the write, not of a checkpoint file.** Every
record is committed on its own — one transaction in Postgres, one `fsync`ed line
in the journal — before the next cell starts. A `kill -9` therefore loses at
most the generation in flight, and the next run recomputes exactly that one.
There is no run-level state to reconcile, no partially-written batch, and
nothing to clean up: `completed_keys()` reads what is durably stored and the
runner skips it. `tests/unit/generate/test_runner_resume.py` kills a real
subprocess with `SIGKILL` mid-run and asserts the restart finishes the set
without recomputing what was already stored.

**`generation_id` is derived from the key**, not from a counter or a UUID, so a
recomputed cell lands on the same primary key and a re-run is idempotent rather
than duplicating rows under new ids.

**Metadata that the frozen schema has no column for** — the self-check verdict,
the long-context coverage, the input-screen flags — goes to a sidecar JSONL
beside the run. The schema is a shared contract and this lane does not own it,
so the choice is a sidecar or dropping the data, and dropping it would mean a
run with a silently broken self-check looks exactly like a healthy one.

**Two columns the insert has to name explicitly.** `gate_blocked` (D12) and
`served_by` are both `NOT NULL DEFAULT`, so an insert that omits them succeeds
and writes the default — which is silently wrong for a refused response and for
anything a second serving stack produced. `gate_blocked` is read off the sidecar
payload rather than carried as a second field that could disagree with it;
`served_by` comes from the client that produced the text.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from carelite.config import get_settings

__all__ = [
    "CacheKey",
    "GenerationRecord",
    "GenerationStore",
    "JsonlStore",
    "PostgresStore",
    "generation_id_for",
]


@dataclass(frozen=True, slots=True)
class CacheKey:
    """The v3 section 16 cache key. Hashable, ordered, and JSON round-trippable."""

    scenario_id: str
    condition: str
    prompt_id: str
    model_digest: str
    seed: int
    sample_idx: int

    def as_tuple(self) -> tuple[str, str, str, str, int, int]:
        return (
            self.scenario_id,
            self.condition,
            self.prompt_id,
            self.model_digest,
            self.seed,
            self.sample_idx,
        )


def generation_id_for(key: CacheKey) -> str:
    """A stable primary key derived from the cache key.

    blake2b rather than `hash()` for the same reason `config.seed_for` uses it:
    CPython randomises string hashing per process, so a `hash()`-derived id
    would differ between the run that was killed and the run that resumes it.
    """
    payload = "|".join(str(p) for p in key.as_tuple()).encode("utf-8")
    return "gen-" + hashlib.blake2b(payload, digest_size=16).hexdigest()


@dataclass(slots=True)
class GenerationRecord:
    """One finished cell: the columns of `generation`, plus what has no column."""

    key: CacheKey
    model: str
    temperature: float
    response: str
    latency_ms: int | None = None
    trace: dict[str, Any] | None = None
    served_by: str = "ollama"
    """Which serving stack produced the response, from the client that produced
    it. Defaults to Ollama, which is what every row already in the database is
    and what the schema backfills existing rows to."""

    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def gate_blocked(self) -> bool:
        """Whether the output gate withheld this response (D12).

        Read off the sidecar payload the runner already builds rather than
        duplicated as a second field that could disagree with it.
        """
        return bool(self.extra.get("output_gate_blocked"))

    @property
    def generation_id(self) -> str:
        return generation_id_for(self.key)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["key"] = list(self.key.as_tuple())
        payload["generation_id"] = self.generation_id
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> GenerationRecord:
        obj = json.loads(line)
        raw = obj["key"]
        key = CacheKey(
            scenario_id=str(raw[0]),
            condition=str(raw[1]),
            prompt_id=str(raw[2]),
            model_digest=str(raw[3]),
            seed=int(raw[4]),
            sample_idx=int(raw[5]),
        )
        return cls(
            key=key,
            model=str(obj["model"]),
            temperature=float(obj["temperature"]),
            response=str(obj["response"]),
            latency_ms=obj.get("latency_ms"),
            trace=obj.get("trace"),
            # Journals written before a second backend existed have no field
            # here, and every one of those cells was served by Ollama.
            served_by=str(obj.get("served_by") or "ollama"),
            extra=obj.get("extra") or {},
        )


class GenerationStore(Protocol):
    """What the runner needs from a store, and nothing more.

    Two implementations share it, which is what lets the resumability test run
    the real runner loop with no database.
    """

    def completed_keys(self) -> set[CacheKey]: ...

    def record(self, record: GenerationRecord) -> None: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


@dataclass
class JsonlStore:
    """Append-only journal. One `fsync`ed line per generation.

    The store the resumability test drives, and a usable local store in its own
    right for a run against a machine with no Postgres. A torn final line — the
    signature of a process killed mid-write — is skipped on read rather than
    raising, because a journal that cannot be reopened after a crash is not a
    crash-recovery mechanism.
    """

    path: Path
    _handle: Any = field(default=None, init=False, repr=False)

    def completed_keys(self) -> set[CacheKey]:
        return {r.key for r in self.read_all()}

    def read_all(self) -> Iterator[GenerationRecord]:
        path = Path(self.path)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield GenerationRecord.from_json(line)
                except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
                    continue  # a line torn by a kill mid-write

    def record(self, record: GenerationRecord) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._handle is None:
            self._terminate_torn_line(path)
            self._handle = path.open("a", encoding="utf-8")
        self._handle.write(record.to_json() + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    @staticmethod
    def _terminate_torn_line(path: Path) -> None:
        """Close off a final line left unterminated by a killed process.

        A `SIGKILL` between `write` and `fsync` can leave the journal ending
        mid-record with no newline. Appending straight onto that would glue the
        fragment to the next record and lose *both* — the torn cell and a
        perfectly good one. Writing a newline first quarantines the fragment on
        its own line, where `read_all` skips it.
        """
        if not path.exists() or path.stat().st_size == 0:
            return
        with path.open("rb+") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                fh.write(b"\n")
                fh.flush()
                os.fsync(fh.fileno())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------


@dataclass
class PostgresStore:
    """The production store: `generation` and `retrieval_trace`, plus a sidecar.

    Each record commits on its own. Batching would be faster and would mean a
    `kill -9` loses the whole batch, which is the wrong trade for a lane that is
    expected to be interrupted.
    """

    sidecar: Path | None = None

    def __post_init__(self) -> None:
        if self.sidecar is None:
            self.sidecar = get_settings().runs_dir / "generate" / "metadata.jsonl"

    def completed_keys(self) -> set[CacheKey]:
        from carelite.db.connection import fetch_all

        rows = fetch_all(
            "SELECT scenario_id, condition, prompt_id, model_digest, seed, sample_idx "
            "FROM generation"
        )
        return {
            CacheKey(
                scenario_id=str(r["scenario_id"]),
                condition=str(r["condition"]),
                prompt_id=str(r["prompt_id"]),
                model_digest=str(r["model_digest"]),
                seed=int(r["seed"]),
                sample_idx=int(r["sample_idx"]),
            )
            for r in rows
        }

    def record(self, record: GenerationRecord) -> None:
        from carelite.db.connection import transaction

        key = record.key
        with transaction() as conn:
            conn.execute(
                """
                INSERT INTO generation (generation_id, scenario_id, condition, prompt_id,
                                        model, model_digest, seed, temperature, sample_idx,
                                        response, latency_ms, gate_blocked, served_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (scenario_id, condition, prompt_id, model_digest, seed, sample_idx)
                DO NOTHING
                """,
                (
                    record.generation_id,
                    key.scenario_id,
                    key.condition,
                    key.prompt_id,
                    record.model,
                    key.model_digest,
                    key.seed,
                    record.temperature,
                    key.sample_idx,
                    record.response,
                    record.latency_ms,
                    record.gate_blocked,
                    record.served_by,
                ),
            )
            if record.trace is not None:
                t = record.trace
                conn.execute(
                    """
                    INSERT INTO retrieval_trace (generation_id, retrieved_ids, scores,
                                                 crag_grade, route_taken, fell_back_to_b,
                                                 hyde_passage, latency_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (generation_id) DO NOTHING
                    """,
                    (
                        record.generation_id,
                        list(t.get("retrieved_ids") or []),
                        list(t.get("scores") or []),
                        t.get("crag_grade"),
                        t.get("route_taken"),
                        bool(t.get("fell_back_to_b")),
                        t.get("hyde_passage"),
                        t.get("latency_ms"),
                    ),
                )
        self._write_sidecar(record)

    def _write_sidecar(self, record: GenerationRecord) -> None:
        if not record.extra or self.sidecar is None:
            return
        path = Path(self.sidecar)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"generation_id": record.generation_id, **record.extra}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def close(self) -> None:
        return None
