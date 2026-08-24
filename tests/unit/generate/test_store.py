"""The cache key, the derived id, and a journal that survives being killed."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from carelite.generate.store import (
    CacheKey,
    GenerationRecord,
    JsonlStore,
    generation_id_for,
)


def _key(sample: int = 0) -> CacheKey:
    return CacheKey(
        scenario_id="SC-001",
        condition="C",
        prompt_id="condition_c.v1",
        model_digest="sha256:abc",
        seed=12345,
        sample_idx=sample,
    )


def test_the_cache_key_is_the_six_fields_the_build_plan_names() -> None:
    assert _key().as_tuple() == ("SC-001", "C", "condition_c.v1", "sha256:abc", 12345, 0)


def test_generation_id_is_derived_from_the_key_and_stable() -> None:
    """It must not depend on process state: the run that resumes has to land on
    the same primary key as the run that was killed."""
    assert generation_id_for(_key()) == generation_id_for(_key())
    assert generation_id_for(_key(0)) != generation_id_for(_key(1))
    assert generation_id_for(_key()).startswith("gen-")


def test_a_changed_digest_is_a_different_cell() -> None:
    """Tags are mutable; the digest is the identity. Re-pulled weights under the
    same tag must produce a new cell rather than joining the old column."""
    other = replace(_key(), model_digest="sha256:def")
    assert generation_id_for(other) != generation_id_for(_key())


def test_records_round_trip_through_json(tmp_path: Path) -> None:
    record = GenerationRecord(
        key=_key(),
        model="gemma4:12b",
        temperature=0.7,
        response="A reply.",
        latency_ms=42,
        trace={"retrieved_ids": ["kb-1"], "scores": [0.9]},
        extra={"self_check_passed": True},
    )
    back = GenerationRecord.from_json(record.to_json())
    assert back.key == record.key
    assert back.response == record.response
    assert back.trace == record.trace
    assert back.extra == record.extra
    assert back.generation_id == record.generation_id


def test_the_journal_reports_what_is_stored(tmp_path: Path) -> None:
    store = JsonlStore(path=tmp_path / "g.jsonl")
    assert store.completed_keys() == set()
    for i in range(3):
        store.record(GenerationRecord(key=_key(i), model="m", temperature=0.7, response=f"r{i}"))
    store.close()
    assert JsonlStore(path=tmp_path / "g.jsonl").completed_keys() == {_key(i) for i in range(3)}


def test_a_line_torn_by_a_kill_is_skipped_not_fatal(tmp_path: Path) -> None:
    """The signature of `kill -9` mid-write. A journal that cannot be reopened
    after a crash is not a crash-recovery mechanism."""
    path = tmp_path / "g.jsonl"
    store = JsonlStore(path=path)
    store.record(GenerationRecord(key=_key(0), model="m", temperature=0.7, response="r0"))
    store.close()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": ["SC-001", "C", "condition_c.v1"]})[:40])  # torn mid-write

    reopened = JsonlStore(path=path)
    assert reopened.completed_keys() == {_key(0)}


def test_a_record_written_after_a_torn_line_is_not_glued_to_it(tmp_path: Path) -> None:
    """Appending onto an unterminated fragment would lose the torn cell *and*
    the good one that followed it. Both have to survive as separate lines."""
    path = tmp_path / "g.jsonl"
    JsonlStore(path=path).record(
        GenerationRecord(key=_key(0), model="m", temperature=0.7, response="r0")
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"key": ["SC-001", "C"')

    store = JsonlStore(path=path)
    store.record(GenerationRecord(key=_key(1), model="m", temperature=0.7, response="r1"))
    store.close()
    assert JsonlStore(path=path).completed_keys() == {_key(0), _key(1)}
