"""The journal -> Postgres loader, driven with no database.

`database=False` is what makes this run in `make check`: every check that
does not need Postgres — parsing, internal consistency, cross-file collisions,
the digest gate, the report — is exercised here, and `test_load_db.py` covers
the foreign keys, the idempotent insert and the resumable batching against a
live table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carelite.generate.load import (
    JournalRefusal,
    blocked_generation_ids,
    collect,
    load_journals,
    load_metadata,
)
from carelite.generate.store import CacheKey, GenerationRecord, JsonlStore, generation_id_for


def _record(
    scenario: str = "SC-001",
    condition: str = "C",
    sample_idx: int = 0,
    *,
    response: str = "A steady, ordinary reply.",
    trace: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    model: str = "gemma4:12b",
    digest: str = "d" * 64,
) -> GenerationRecord:
    key = CacheKey(
        scenario_id=scenario,
        condition=condition,
        prompt_id=f"condition_{condition.lower()}.v1",
        model_digest=digest,
        seed=12345,
        sample_idx=sample_idx,
    )
    return GenerationRecord(
        key=key,
        model=model,
        temperature=0.7,
        response=response,
        latency_ms=11,
        trace=trace,
        extra={
            "condition": condition,
            "scenario_id": scenario,
            "sample_idx": sample_idx,
            "split": "holdout",
            **(extra or {}),
        },
    )


def _journal(path: Path, records: list[GenerationRecord]) -> Path:
    store = JsonlStore(path=path)
    for record in records:
        store.record(record)
    store.close()
    return path


def test_it_reads_a_journal_and_keeps_the_files_generation_id(tmp_path: Path) -> None:
    path = _journal(tmp_path / "g.jsonl", [_record(), _record(sample_idx=1)])
    sources = collect([path])
    assert len(sources) == 2
    assert [s.generation_id for s in sources] == [generation_id_for(s.key) for s in sources]
    assert not any(s.id_drifted for s in sources)


def test_a_generation_id_that_the_deriving_function_disagrees_with_is_kept_and_reported(
    tmp_path: Path,
) -> None:
    """The file is authoritative. Drift is reported, never silently corrected.

    The judge keys rubric scores to the id the run produced. If
    `generation_id_for` ever changes, recomputing would orphan every score
    already written against the old id, so the loader preserves what it read and
    says loudly that the two disagree.
    """
    path = tmp_path / "g.jsonl"
    obj = json.loads(_record().to_json())
    obj["generation_id"] = "gen-somethingelse"
    path.write_text(json.dumps(obj) + "\n", encoding="utf-8")

    (source,) = collect([path])
    assert source.generation_id == "gen-somethingelse"
    assert source.id_drifted

    report = load_journals([path], dry_run=True, database=False)
    assert len(report.id_drift) == 1
    assert "gen-somethingelse" in report.summary()


def test_a_torn_line_is_refused_rather_than_skipped(tmp_path: Path) -> None:
    """`JsonlStore.read_all` skips it by design; a load must not."""
    path = _journal(tmp_path / "g.jsonl", [_record()])
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"key": ["SC-002", "C", "cond')

    assert len(list(JsonlStore(path=path).read_all())) == 1  # the store tolerates it
    with pytest.raises(JournalRefusal, match="not valid JSON"):
        collect([path])


def test_an_empty_response_is_refused(tmp_path: Path) -> None:
    path = _journal(tmp_path / "g.jsonl", [_record(response="   ")])
    with pytest.raises(JournalRefusal, match="empty response"):
        collect([path])


def test_extra_disagreeing_with_its_own_key_is_refused(tmp_path: Path) -> None:
    record = _record(condition="C")
    record.extra["condition"] = "B"
    path = _journal(tmp_path / "g.jsonl", [record])
    with pytest.raises(JournalRefusal, match="disagrees with key"):
        collect([path])


def test_one_generation_id_over_two_different_records_is_refused(tmp_path: Path) -> None:
    """`ON CONFLICT DO NOTHING` would swallow the second one silently."""
    first = _record(response="one")
    second = _record(response="two")
    a = tmp_path / "a.jsonl"
    obj = json.loads(second.to_json())
    obj["generation_id"] = first.generation_id
    _journal(a, [first])
    with a.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj) + "\n")

    with pytest.raises(JournalRefusal, match="claimed by two different records"):
        collect([a])


def test_one_cache_key_under_two_ids_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    record = _record()
    obj = json.loads(record.to_json())
    other = dict(obj)
    other["generation_id"] = "gen-adifferentid"
    path.write_text(
        json.dumps(obj) + "\n" + json.dumps(other) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(JournalRefusal, match="appears under two ids"):
        collect([path])


def test_the_same_record_in_two_files_collapses_instead_of_colliding(tmp_path: Path) -> None:
    """Passing a glob and the directory containing it is not a data problem."""
    record = _record()
    a = _journal(tmp_path / "a.jsonl", [record])
    b = _journal(tmp_path / "b.jsonl", [record])
    sources = collect([a, b])
    assert len(sources) == 1

    report = load_journals([a, b], dry_run=True, database=False)
    assert report.records == 1
    assert report.duplicates_collapsed == 1


def test_one_model_tag_carrying_two_digests_is_refused(tmp_path: Path) -> None:
    """The digest gate: two sets of weights under one name is what it is for."""
    path = _journal(
        tmp_path / "g.jsonl",
        [
            _record(sample_idx=0, digest="a" * 64),
            _record(sample_idx=1, digest="b" * 64),
        ],
    )
    with pytest.raises(JournalRefusal, match="carries 2 digests"):
        collect([path])


def test_two_models_with_different_digests_are_fine(tmp_path: Path) -> None:
    """A2 is a second family, so a load legitimately spans two digests."""
    path = _journal(
        tmp_path / "g.jsonl",
        [
            _record(condition="A", model="gemma4:12b", digest="a" * 64),
            _record(condition="A2", model="qwen3.5:9b", digest="b" * 64),
        ],
    )
    assert len(collect([path])) == 2


def test_a_bad_crag_grade_is_refused_before_the_check_constraint_sees_it(tmp_path: Path) -> None:
    path = _journal(
        tmp_path / "g.jsonl",
        [_record(trace={"crag_grade": "sort-of", "retrieved_ids": [], "scores": []})],
    )
    with pytest.raises(JournalRefusal, match="crag_grade"):
        collect([path])


def test_a_trace_whose_ids_and_scores_disagree_in_length_is_refused(tmp_path: Path) -> None:
    path = _journal(
        tmp_path / "g.jsonl",
        [_record(trace={"crag_grade": "relevant", "retrieved_ids": ["x", "y"], "scores": [0.5]})],
    )
    with pytest.raises(JournalRefusal, match="retrieved_ids but"):
        collect([path])


def test_the_report_names_lc_as_a_partial_record(tmp_path: Path) -> None:
    """D11: 39 of 180 cells over 13 of 60 scenarios is not a usable sample."""
    path = _journal(
        tmp_path / "g.jsonl",
        [_record(condition="LC"), _record(condition="B")],
    )
    report = load_journals([path], dry_run=True, database=False)
    summary = report.summary()
    assert "PARTIAL RECORD (D11)" in summary
    lc_line = next(line for line in summary.splitlines() if line.strip().startswith("LC"))
    b_line = next(line for line in summary.splitlines() if line.strip().startswith("B "))
    assert "PARTIAL" in lc_line and "PARTIAL" not in b_line


def test_the_report_surfaces_the_fallback_rate_the_stats_lane_needs(tmp_path: Path) -> None:
    records = [
        _record(
            sample_idx=i,
            trace={
                "crag_grade": "none" if i == 0 else "relevant",
                "fell_back_to_b": i == 0,
                "retrieved_ids": [],
                "scores": [],
            },
        )
        for i in range(3)
    ]
    report = load_journals([_journal(tmp_path / "g.jsonl", records)], dry_run=True, database=False)
    assert report.fell_back_to_b["C"] == 1
    assert report.crag_grades == {"none": 1, "relevant": 2}
    assert "intention-to-treat" in report.summary()


def test_gate_blocked_cells_are_loaded_flagged_not_dropped(tmp_path: Path) -> None:
    """They are in the journal, not missing from it. Dropping them makes a hole."""
    records = [
        _record(sample_idx=0),
        _record(
            sample_idx=1,
            extra={"output_gate_blocked": True, "output_gate_flags": ["output.clinical_dosing"]},
        ),
    ]
    path = _journal(tmp_path / "g.jsonl", records)
    report = load_journals([path], dry_run=True, database=False)
    assert report.records == 2
    assert report.gate_blocked["C"] == 1
    assert "output gate blocked 1 cell" in report.summary()


def test_the_sidecar_round_trips_and_a_second_load_does_not_duplicate_it(tmp_path: Path) -> None:
    blocked = _record(
        sample_idx=1,
        extra={"output_gate_blocked": True, "output_gate_flags": ["output.phi_leak"]},
    )
    path = _journal(tmp_path / "g.jsonl", [_record(sample_idx=0), blocked])
    sidecar = tmp_path / "metadata.jsonl"

    for _ in range(2):
        load_journals([path], sidecar=sidecar, database=False)

    assert len(sidecar.read_text(encoding="utf-8").splitlines()) == 2
    meta = load_metadata(sidecar)
    assert set(meta) == {_record(sample_idx=0).generation_id, blocked.generation_id}
    assert meta[blocked.generation_id]["split"] == "holdout"
    assert blocked_generation_ids(sidecar) == {blocked.generation_id}


def test_the_sidecar_keeps_rows_it_did_not_write(tmp_path: Path) -> None:
    """A merge, not a truncate: another run's metadata survives a load."""
    sidecar = tmp_path / "metadata.jsonl"
    sidecar.write_text(
        json.dumps({"generation_id": "gen-fromanotherrun", "split": "train"}) + "\n",
        encoding="utf-8",
    )
    load_journals([_journal(tmp_path / "g.jsonl", [_record()])], sidecar=sidecar, database=False)
    assert "gen-fromanotherrun" in load_metadata(sidecar)


def test_a_directory_contributes_its_journals(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _journal(run / "generations-a.jsonl", [_record(condition="A")])
    _journal(run / "generations-b.jsonl", [_record(condition="B")])
    (run / "run.log").write_text("not a journal\n", encoding="utf-8")
    report = load_journals([run], dry_run=True, database=False)
    assert report.records == 2
    assert set(report.by_condition) == {"A", "B"}


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(JournalRefusal, match="no such file"):
        collect([tmp_path / "nope.jsonl"])


def test_a_dry_run_writes_no_sidecar(tmp_path: Path) -> None:
    sidecar = tmp_path / "metadata.jsonl"
    load_journals(
        [_journal(tmp_path / "g.jsonl", [_record()])],
        sidecar=sidecar,
        dry_run=True,
        database=False,
    )
    assert not sidecar.exists()
