"""The three new statements, against a live Postgres. Marked `db`, not in `make check`.

`fetch_arm`, `fetch_arm_meta` and `fetch_paired_scores` each open their own
connection, so what is exercised here is their SQL run on the test's own
rolled-back connection — the same pattern `test_store_db.py` uses for
`UPSERT_SQL`. That is where a wrong column name, a missing `served_by`, or an
inner join that should be a left join actually lives.

Every fixture row is created and rolled back, so this neither depends on nor
disturbs the 1,119 generations the study reads from.

Run with: make test-db
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest

from carelite.db import connect
from carelite.eval.judge.arms import _ARM_SQL, MixedBackendError, assert_single_backend, pair_cells
from carelite.eval.judge.backend_equivalence import _PAIRED_SQL, compare_backends
from carelite.eval.judge.score_arm import ARM_META_SQL, meta_from_rows
from carelite.types import RUBRIC_DIMENSIONS

from .conftest import RESPONSE, SCENARIO

pytestmark = pytest.mark.db

SCENARIO_ID = "sc-arms-test"
PROMPT_ID = "prompt-arms-test"
RATER_ID = "arms-test-judge-median"


def _insert_generation(
    conn: psycopg.Connection, generation_id: str, *, served_by: str, sample_idx: int, seed: int
) -> None:
    conn.execute(
        "INSERT INTO generation (generation_id, scenario_id, condition, prompt_id, model, "
        "model_digest, seed, temperature, sample_idx, response, served_by) "
        "VALUES (%s, %s, 'LC', %s, 'gemma4:12b', %s, %s, 0.7, %s, %s, %s)",
        (
            generation_id,
            SCENARIO_ID,
            PROMPT_ID,
            f"digest-{served_by}",
            seed,
            sample_idx,
            RESPONSE,
            served_by,
        ),
    )


def _insert_score(conn: psycopg.Connection, generation_id: str, value: int) -> None:
    columns = ", ".join(RUBRIC_DIMENSIONS)
    placeholders = ", ".join(["%s"] * len(RUBRIC_DIMENSIONS))
    conn.execute(
        f"INSERT INTO rubric_score (generation_id, rater_type, rater_id, sample_idx, {columns}) "
        f"VALUES (%s, 'llm_judge', %s, 0, {placeholders})",
        (generation_id, RATER_ID, *([value] * len(RUBRIC_DIMENSIONS))),
    )


@pytest.fixture
def two_backends() -> Iterator[psycopg.Connection]:
    """One LC cell produced by both stacks, at the same seed. Rolled back."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO scenario (scenario_id, text, challenge_type, emotion_intensity, "
            "encounter_phase, literacy_signal, equity_stratum, split) "
            "VALUES (%s, %s, 'fear', 4, 'explanation', 'low', false, 'holdout')",
            (SCENARIO_ID, SCENARIO),
        )
        conn.execute(
            "INSERT INTO prompt_version (prompt_id, condition, text) VALUES (%s, 'LC', 'x')",
            (PROMPT_ID,),
        )
        _insert_generation(conn, "gen-arms-ollama", served_by="ollama", sample_idx=0, seed=7)
        _insert_generation(conn, "gen-arms-vllm", served_by="vllm", sample_idx=0, seed=7)
        _insert_score(conn, "gen-arms-ollama", 4)
        _insert_score(conn, "gen-arms-vllm", 5)
        try:
            yield conn
        finally:
            conn.rollback()


def _arm(conn: psycopg.Connection, served_by: str) -> list[dict[str, Any]]:
    """`_ARM_SQL` with the split clause `fetch_arm` appends, narrowed to this
    test's own scenario so the study's real LC rows in the same database do not
    join the result."""
    sql = (
        _ARM_SQL
        + "  AND sc.split = %(split)s\n  AND g.scenario_id = %(scenario_id)s\n"
        + "ORDER BY g.scenario_id, g.sample_idx"
    )
    return [
        dict(r)
        for r in conn.execute(
            sql,
            {
                "condition": "LC",
                "served_by": served_by,
                "split": "holdout",
                "scenario_id": SCENARIO_ID,
            },
        ).fetchall()
    ]


class TestArmSelection:
    def test_each_backend_selects_only_its_own_rows(self, two_backends: psycopg.Connection) -> None:
        ollama = _arm(two_backends, "ollama")
        vllm = _arm(two_backends, "vllm")
        assert [r["generation_id"] for r in ollama] == ["gen-arms-ollama"]
        assert [r["generation_id"] for r in vllm] == ["gen-arms-vllm"]
        assert assert_single_backend(ollama, what="LC arm") == "ollama"
        assert assert_single_backend(vllm, what="LC arm") == "vllm"

    def test_selecting_on_the_condition_alone_is_the_pooled_query_the_guard_refuses(
        self, two_backends: psycopg.Connection
    ) -> None:
        """This is the query D13 says not to run, run deliberately, to prove the
        guard catches what a bare condition filter returns."""
        pooled = [
            dict(r)
            for r in two_backends.execute(
                "SELECT * FROM generation WHERE condition = 'LC' AND scenario_id = %s",
                (SCENARIO_ID,),
            ).fetchall()
        ]
        assert len(pooled) == 2
        with pytest.raises(MixedBackendError, match="ollama=1, vllm=1"):
            assert_single_backend(pooled, what="condition LC arm")


class TestArmMetadata:
    def test_metadata_survives_a_condition_with_no_retrieval_trace(
        self, two_backends: psycopg.Connection
    ) -> None:
        """LC never retrieves (D7), so the LEFT JOIN is load-bearing: an inner
        join here would return nothing at all for this arm."""
        rows = two_backends.execute(
            ARM_META_SQL, {"ids": ["gen-arms-ollama", "gen-arms-vllm"]}
        ).fetchall()
        meta = meta_from_rows(rows)
        assert set(meta) == {"gen-arms-ollama", "gen-arms-vllm"}
        assert meta["gen-arms-vllm"].served_by == "vllm"
        assert meta["gen-arms-vllm"].n_retrieved == 0
        assert meta["gen-arms-vllm"].fell_back_to_b is False
        assert meta["gen-arms-ollama"].split == "holdout"


class TestPairedScores:
    def test_the_paired_query_returns_one_scored_cell_per_stack(
        self, two_backends: psycopg.Connection
    ) -> None:
        dims = ", ".join(f"rs.{d}" for d in RUBRIC_DIMENSIONS)
        # The rater id is this test's own, so only the fixture rows can match.
        sql = _PAIRED_SQL.format(dims=dims)

        def side(backend: str) -> list[dict[str, Any]]:
            return [
                dict(r)
                for r in two_backends.execute(
                    sql, {"condition": "LC", "served_by": backend, "rater_id": RATER_ID}
                ).fetchall()
            ]

        left, right = side("ollama"), side("vllm")
        assert len(left) == 1 and len(right) == 1
        paired = pair_cells(left, right)
        assert paired.n_pairs == 1
        assert paired.left_backend == "ollama"
        assert paired.right_backend == "vllm"

        unit = "SC/LC/0"
        report = compare_backends(
            {unit: {d: left[0][d] for d in RUBRIC_DIMENSIONS}},
            {unit: {d: right[0][d] for d in RUBRIC_DIMENSIONS}},
            scenario_of={unit: "SC"},
            left_backend="ollama",
            right_backend="vllm",
            n_boot=50,
        )
        assert report.n_cell_pairs == 1
        # One pair resolves nothing, and the report must not pretend otherwise.
        assert report.supports_equivalence_claim is False
        assert report.poolable is False
