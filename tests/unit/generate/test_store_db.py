"""`PostgresStore` against a live database. Marked `db`; excluded from `make check`.

**Everything this module writes, it deletes.** It inserts one throwaway
`prompt_version` row and one `generation` row under ids prefixed `zz-test-`,
and removes both in a `finally`. An earlier lane in this project silently
overwrote all 475 chunk embeddings from an unscoped test, so nothing here
touches a row it did not create, and no statement in it is unqualified.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carelite.generate.store import CacheKey, GenerationRecord, PostgresStore

pytestmark = pytest.mark.db

TEST_PROMPT_ID = "zz-test-prompt.v1"


def _a_real_scenario_id() -> str:
    from carelite.db.connection import fetch_one

    row = fetch_one("SELECT scenario_id FROM scenario ORDER BY scenario_id LIMIT 1")
    if row is None:
        pytest.skip("no scenarios loaded; run `python -m carelite.scenarios.load` first")
    return str(row["scenario_id"])


def test_a_generation_round_trips_and_is_skipped_on_the_second_pass(tmp_path: Path) -> None:
    from carelite.db.connection import transaction

    scenario_id = _a_real_scenario_id()
    key = CacheKey(
        scenario_id=scenario_id,
        condition="C",
        prompt_id=TEST_PROMPT_ID,
        model_digest="sha256:zztest",
        seed=987654321,
        sample_idx=0,
    )
    record = GenerationRecord(
        key=key,
        model="zz-test:0b",
        temperature=0.7,
        response="A throwaway response written by a test.",
        latency_ms=7,
        trace={
            "retrieved_ids": ["zz-test-ref"],
            "scores": [0.5],
            "crag_grade": "relevant",
            "route_taken": "mixed",
            "fell_back_to_b": False,
            "hyde_passage": None,
            "latency_ms": 3,
        },
        extra={"self_check_passed": True},
    )
    store = PostgresStore(sidecar=tmp_path / "metadata.jsonl")

    try:
        with transaction() as conn:
            conn.execute(
                "INSERT INTO prompt_version (prompt_id, condition, text, git_sha) "
                "VALUES (%s, 'C', 'throwaway test prompt', 'deadbeef') "
                "ON CONFLICT (prompt_id) DO NOTHING",
                (TEST_PROMPT_ID,),
            )

        store.record(record)
        assert key in store.completed_keys()

        # The unique constraint is the cache key, so a repeat is a no-op rather
        # than a second row.
        store.record(record)
        from carelite.db.connection import fetch_one

        count = fetch_one(
            "SELECT count(*) AS n FROM generation WHERE generation_id = %s",
            (record.generation_id,),
        )
        assert count is not None and int(count["n"]) == 1

        trace_row = fetch_one(
            "SELECT route_taken, crag_grade FROM retrieval_trace WHERE generation_id = %s",
            (record.generation_id,),
        )
        assert trace_row is not None
        assert trace_row["route_taken"] == "mixed"

        sidecar = (tmp_path / "metadata.jsonl").read_text(encoding="utf-8")
        assert record.generation_id in sidecar
    finally:
        with transaction() as conn:
            conn.execute(
                "DELETE FROM retrieval_trace WHERE generation_id = %s", (record.generation_id,)
            )
            conn.execute("DELETE FROM generation WHERE generation_id = %s", (record.generation_id,))
            conn.execute("DELETE FROM prompt_version WHERE prompt_id = %s", (TEST_PROMPT_ID,))


def test_registration_is_idempotent_and_refuses_an_edited_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Editing a prompt under an unchanged id is the one mistake that quietly
    mixes two experiments in one results table. `register()` raises instead.

    Driven through a throwaway `prompt_id` so the real `prompt_version` rows are
    neither written nor read by this test.
    """
    from carelite.db.connection import transaction
    from carelite.generate import prompts

    def rows(text: str) -> list[dict[str, str]]:
        return [
            {
                "prompt_id": TEST_PROMPT_ID,
                "condition": "C",
                "text": text,
                "git_sha": prompts.blob_sha(text),
            }
        ]

    try:
        monkeypatch.setattr(prompts, "registered_rows", lambda ids=None: rows("first text"))
        assert prompts.register() == 1
        assert prompts.register() == 0, "a second registration of the same text inserts nothing"

        monkeypatch.setattr(prompts, "registered_rows", lambda ids=None: rows("edited text"))
        with pytest.raises(prompts.PromptDriftError, match="already registered"):
            prompts.register()
    finally:
        with transaction() as conn:
            conn.execute("DELETE FROM prompt_version WHERE prompt_id = %s", (TEST_PROMPT_ID,))
