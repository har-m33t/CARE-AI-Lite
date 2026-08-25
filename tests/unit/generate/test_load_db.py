"""The loader against a live database. Marked `db`; excluded from `make check`.

**Everything this module writes, it deletes.** One throwaway `prompt_version`
row and a handful of `generation` rows, all under ids prefixed `zz-test-`,
removed in a `finally`. An earlier lane in this project silently overwrote all
475 chunk embeddings from an unscoped test, so nothing here touches a row it did
not create and no statement in it is unqualified.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from carelite.generate.load import JournalRefusal, load_journals
from carelite.generate.store import CacheKey, GenerationRecord, JsonlStore

pytestmark = pytest.mark.db

TEST_PROMPT_ID = "zz-test-load-prompt.v1"
TEST_DIGEST = "zz-test-load-digest"


def _a_real_scenario_id() -> str:
    from carelite.db.connection import fetch_one

    row = fetch_one("SELECT scenario_id FROM scenario ORDER BY scenario_id LIMIT 1")
    if row is None:
        pytest.skip("no scenarios loaded; run `python -m carelite.scenarios.load` first")
    return str(row["scenario_id"])


@pytest.fixture
def registered_prompt() -> Iterator[str]:
    """A `prompt_version` row for conditions C and B, torn down afterwards."""
    from carelite.db.connection import transaction

    with transaction() as conn:
        conn.execute(
            "INSERT INTO prompt_version (prompt_id, condition, text) VALUES (%s, %s, %s) "
            "ON CONFLICT (prompt_id) DO NOTHING",
            (TEST_PROMPT_ID, "C,B", "throwaway"),
        )
    try:
        yield TEST_PROMPT_ID
    finally:
        with transaction() as conn:
            conn.execute(
                "DELETE FROM generation WHERE prompt_id = %s AND model_digest = %s",
                (TEST_PROMPT_ID, TEST_DIGEST),
            )
            conn.execute("DELETE FROM prompt_version WHERE prompt_id = %s", (TEST_PROMPT_ID,))


def _record(
    scenario: str,
    prompt_id: str,
    sample_idx: int,
    *,
    condition: str = "C",
    trace: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> GenerationRecord:
    return GenerationRecord(
        key=CacheKey(
            scenario_id=scenario,
            condition=condition,
            prompt_id=prompt_id,
            model_digest=TEST_DIGEST,
            seed=987654321,
            sample_idx=sample_idx,
        ),
        model="zz-test:0b",
        temperature=0.7,
        response=f"A throwaway response written by a test, sample {sample_idx}.",
        latency_ms=7,
        trace=trace,
        extra=extra or {},
    )


def _journal(path: Path, records: list[GenerationRecord]) -> Path:
    store = JsonlStore(path=path)
    for record in records:
        store.record(record)
    store.close()
    return path


def test_a_journal_loads_with_its_trace_and_reloading_is_a_no_op(
    tmp_path: Path, registered_prompt: str
) -> None:
    """Idempotence comes from the table's constraints, not from bookkeeping."""
    from carelite.db.connection import fetch_one

    scenario = _a_real_scenario_id()
    trace = {
        "retrieved_ids": ["zz-test-ref-a", "zz-test-ref-b"],
        "scores": [0.5, 0.25],
        "crag_grade": "relevant",
        "route_taken": "mixed",
        "fell_back_to_b": False,
        "hyde_passage": "a throwaway hypothetical passage",
        "latency_ms": 3,
    }
    records = [
        _record(scenario, registered_prompt, 0, trace=trace),
        _record(
            scenario,
            registered_prompt,
            1,
            trace={**trace, "crag_grade": "none", "fell_back_to_b": True},
        ),
    ]
    path = _journal(tmp_path / "g.jsonl", records)
    sidecar = tmp_path / "metadata.jsonl"

    first = load_journals([path], sidecar=sidecar, batch_size=1)
    assert first.generations_inserted == 2
    assert first.generations_already_present == 0
    assert first.traces_inserted == 2

    row = fetch_one(
        "SELECT response FROM generation WHERE generation_id = %s", (records[0].generation_id,)
    )
    assert row is not None and str(row["response"]) == records[0].response

    stored_trace = fetch_one(
        "SELECT crag_grade, fell_back_to_b, retrieved_ids, hyde_passage "
        "FROM retrieval_trace WHERE generation_id = %s",
        (records[1].generation_id,),
    )
    assert stored_trace is not None
    assert stored_trace["crag_grade"] == "none"
    assert stored_trace["fell_back_to_b"] is True
    assert list(stored_trace["retrieved_ids"]) == ["zz-test-ref-a", "zz-test-ref-b"]

    second = load_journals([path], sidecar=sidecar, batch_size=1)
    assert second.generations_inserted == 0
    assert second.generations_already_present == 2
    assert second.traces_inserted == 0
    assert second.traces_already_present == 2

    total = fetch_one(
        "SELECT count(*) AS n FROM generation WHERE prompt_id = %s AND model_digest = %s",
        (registered_prompt, TEST_DIGEST),
    )
    assert total is not None and int(total["n"]) == 2


def test_an_interrupted_load_finishes_on_the_next_run(
    tmp_path: Path, registered_prompt: str
) -> None:
    """A partial load is a prefix, and the rerun adds only what is missing.

    Simulated by loading two of three records, then loading all three. There is
    no manifest and nothing to reconcile: the second run inserts one row and
    reports the other two as already present.
    """
    from carelite.db.connection import fetch_one

    scenario = _a_real_scenario_id()
    records = [_record(scenario, registered_prompt, i) for i in range(3)]

    partial = _journal(tmp_path / "partial.jsonl", records[:2])
    load_journals([partial], write_sidecar=False, batch_size=1)

    full = _journal(tmp_path / "full.jsonl", records)
    report = load_journals([full], write_sidecar=False, batch_size=1)
    assert report.generations_inserted == 1
    assert report.generations_already_present == 2

    total = fetch_one(
        "SELECT count(*) AS n FROM generation WHERE prompt_id = %s AND model_digest = %s",
        (registered_prompt, TEST_DIGEST),
    )
    assert total is not None and int(total["n"]) == 3


def test_an_unregistered_prompt_is_refused_before_anything_is_written(
    tmp_path: Path, registered_prompt: str
) -> None:
    from carelite.db.connection import fetch_one

    scenario = _a_real_scenario_id()
    path = _journal(
        tmp_path / "g.jsonl",
        [
            _record(scenario, registered_prompt, 0),
            _record(scenario, "zz-test-never-registered.v1", 1),
        ],
    )
    with pytest.raises(JournalRefusal, match="is not registered"):
        load_journals([path], write_sidecar=False)

    # The good record in the same file was not written either.
    total = fetch_one(
        "SELECT count(*) AS n FROM generation WHERE model_digest = %s", (TEST_DIGEST,)
    )
    assert total is not None and int(total["n"]) == 0


def test_a_prompt_registered_for_another_condition_is_refused(
    tmp_path: Path, registered_prompt: str
) -> None:
    """`condition_a.v1` is registered as 'A,A2'; membership, not equality."""
    scenario = _a_real_scenario_id()
    path = _journal(
        tmp_path / "g.jsonl",
        [_record(scenario, registered_prompt, 0, condition="D")],
    )
    with pytest.raises(JournalRefusal, match="but a record claims condition 'D'"):
        load_journals([path], write_sidecar=False)


def test_an_unknown_scenario_is_refused(tmp_path: Path, registered_prompt: str) -> None:
    path = _journal(tmp_path / "g.jsonl", [_record("SC-NOTREAL", registered_prompt, 0)])
    with pytest.raises(JournalRefusal, match="is not in the scenario bank"):
        load_journals([path], write_sidecar=False)


def test_a_cell_already_stored_under_a_different_id_is_refused(
    tmp_path: Path, registered_prompt: str
) -> None:
    """Otherwise the generation insert is a silent no-op and the trace FK aborts."""
    scenario = _a_real_scenario_id()
    record = _record(scenario, registered_prompt, 0)
    load_journals([_journal(tmp_path / "first.jsonl", [record])], write_sidecar=False)

    obj = json.loads(record.to_json())
    obj["generation_id"] = "zz-test-a-different-id"
    other = tmp_path / "second.jsonl"
    other.write_text(json.dumps(obj) + "\n", encoding="utf-8")

    with pytest.raises(JournalRefusal, match="already stored under generation_id"):
        load_journals([other], write_sidecar=False)
