"""Persistence round-trips. Marked `db`, excluded from `make check`.

Every test here creates its own scenario, prompt version and generation rows and
rolls the whole transaction back, so it neither depends on nor disturbs the
scenario bank the `carelite-scenarios` lane loads into the same database.

Run with: make test-db
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest

from carelite.db import connect
from carelite.eval.human.blinding import Assignment
from carelite.eval.judge import LLMJudge, ReplayClient
from carelite.eval.judge.store import MEDIAN_RATER_SUFFIX, median_rater_id
from carelite.types import RUBRIC_DIMENSIONS

from .conftest import RESPONSE, SCENARIO, judge_json

pytestmark = pytest.mark.db

SCENARIO_ID = "sc-judge-test"
PROMPT_ID = "prompt-judge-test"
GENERATION_ID = "gen-judge-test"


@pytest.fixture
def rolled_back() -> Iterator[psycopg.Connection]:
    """A connection with fixture rows, rolled back at the end. Leaves no trace."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO scenario (scenario_id, text, challenge_type, emotion_intensity, "
            "encounter_phase, literacy_signal, equity_stratum, split) "
            "VALUES (%s, %s, 'fear', 4, 'explanation', 'low', true, 'holdout')",
            (SCENARIO_ID, SCENARIO),
        )
        conn.execute(
            "INSERT INTO prompt_version (prompt_id, condition, text) VALUES (%s, 'C', 'x')",
            (PROMPT_ID,),
        )
        conn.execute(
            "INSERT INTO generation (generation_id, scenario_id, condition, prompt_id, model, "
            "model_digest, seed, temperature, sample_idx, response) "
            "VALUES (%s, %s, 'C', %s, 'gemma4:12b', 'sha256:test', 1, 0.7, 0, %s)",
            (GENERATION_ID, SCENARIO_ID, PROMPT_ID, RESPONSE),
        )
        try:
            yield conn
        finally:
            conn.rollback()


def _result(scores: dict[str, int] | int = 4, n: int = 1):
    judge = LLMJudge(
        client=ReplayClient(outputs=[judge_json(scores)] * n),
        temperature=0.7 if n > 1 else 0.0,
        n_samples=n,
        rater_id="gpt-oss:20b",
    )
    return judge.score_text(
        generation_id=GENERATION_ID, scenario_text=SCENARIO, response_text=RESPONSE
    )


def _upsert(conn: psycopg.Connection, result, sample_idx: int, rater_id: str) -> None:
    """The store module's SQL, executed on the test's own rolled-back connection."""
    from carelite.eval.judge.store import _UPSERT, _params

    score = result.to_rubric_score(rater_id=rater_id)
    conn.execute(_UPSERT, _params(score, sample_idx))


class TestRubricScoreRoundTrip:
    def test_aggregate_row_persists_with_spans_and_raw_ritualistic(
        self, rolled_back: psycopg.Connection
    ) -> None:
        result = _result({**dict.fromkeys(RUBRIC_DIMENSIONS, 4), "ritualistic": 5})
        _upsert(rolled_back, result, 0, median_rater_id("gpt-oss:20b"))

        row = rolled_back.execute(
            "SELECT * FROM rubric_score WHERE generation_id = %s", (GENERATION_ID,)
        ).fetchone()
        assert row is not None
        assert row["rater_type"] == "llm_judge"
        assert row["rater_id"].endswith(MEDIAN_RATER_SUFFIX)
        # Raw, unflipped: the column is higher-is-worse by contract.
        assert row["ritualistic"] == 5
        assert row["de"] == 4
        assert set(row["evidence_spans"]) == set(RUBRIC_DIMENSIONS)
        assert all(span in RESPONSE for span in row["evidence_spans"].values())

    def test_a_rejected_dimension_lands_as_null_not_a_number(
        self, rolled_back: psycopg.Connection
    ) -> None:
        """The grounding rule, all the way to the column."""
        judge = LLMJudge(
            client=ReplayClient(outputs=[judge_json(4, spans={"de": "never said this"})]),
            temperature=0.0,
            n_samples=1,
            rater_id="gpt-oss:20b",
        )
        result = judge.score_text(
            generation_id=GENERATION_ID, scenario_text=SCENARIO, response_text=RESPONSE
        )
        _upsert(rolled_back, result, 0, median_rater_id("gpt-oss:20b"))

        row = rolled_back.execute(
            "SELECT de, name, evidence_spans FROM rubric_score WHERE generation_id = %s",
            (GENERATION_ID,),
        ).fetchone()
        assert row["de"] is None
        assert row["name"] == 4
        assert "de" not in row["evidence_spans"]

    def test_upsert_is_idempotent(self, rolled_back: psycopg.Connection) -> None:
        """Re-running persistence after an interrupted run must not duplicate-key."""
        result = _result(4)
        rater = median_rater_id("gpt-oss:20b")
        _upsert(rolled_back, result, 0, rater)
        _upsert(rolled_back, result, 0, rater)
        count = rolled_back.execute(
            "SELECT count(*) AS n FROM rubric_score WHERE generation_id = %s", (GENERATION_ID,)
        ).fetchone()
        assert count["n"] == 1

    def test_samples_and_median_coexist_under_the_unique_key(
        self, rolled_back: psycopg.Connection
    ) -> None:
        """Why the aggregate needs its own rater id.

        `rubric_score` is unique on (generation_id, rater_type, rater_id,
        sample_idx). Storing the median at sample_idx 0 under the judge's own id
        would collide with sample 0 — and since the median usually equals sample
        0, the collision would look like it worked.
        """
        from carelite.eval.judge.store import _UPSERT, _params

        result = _result({**dict.fromkeys(RUBRIC_DIMENSIONS, 3), "de": 5}, n=5)
        for sample, score in zip(
            result.samples, result.per_sample_rubric_scores(rater_id="gpt-oss:20b"), strict=True
        ):
            rolled_back.execute(_UPSERT, _params(score, sample.sample_idx))
        _upsert(rolled_back, result, 0, median_rater_id("gpt-oss:20b"))

        rows = rolled_back.execute(
            "SELECT rater_id, sample_idx FROM rubric_score WHERE generation_id = %s "
            "ORDER BY rater_id, sample_idx",
            (GENERATION_ID,),
        ).fetchall()
        assert len(rows) == 6
        assert sum(1 for r in rows if r["rater_id"].endswith(MEDIAN_RATER_SUFFIX)) == 1


class TestRatingAssignmentRoundTrip:
    def test_assignment_persists_and_unblinds_by_join(
        self, rolled_back: psycopg.Connection
    ) -> None:
        """The blinding key is a table, so unblinding is a join."""
        from carelite.eval.human.store import _ASSIGNMENT_UPSERT

        assignment = Assignment(
            rater_id="R01",
            generation_id=GENERATION_ID,
            display_order=7,
            blind_label="R01-042",
            is_calibration=False,
        )
        rolled_back.execute(
            _ASSIGNMENT_UPSERT,
            {
                "rater_id": assignment.rater_id,
                "generation_id": assignment.generation_id,
                "display_order": assignment.display_order,
                "blind_label": assignment.blind_label,
                "is_calibration": assignment.is_calibration,
            },
        )

        row = rolled_back.execute(
            "SELECT ra.blind_label, g.condition FROM rating_assignment ra "
            "JOIN generation g USING (generation_id) WHERE ra.rater_id = %s",
            ("R01",),
        ).fetchone()
        assert row["blind_label"] == "R01-042"
        assert row["condition"] == "C"

    def test_calibration_items_have_no_generation_row_to_reference(
        self, rolled_back: psycopg.Connection
    ) -> None:
        """Documents the schema constraint that keeps calibration out of the table.

        `rating_assignment.generation_id` is a foreign key to `generation`, and
        calibration items are fixtures. `store_assignments` therefore filters
        them out by default rather than failing mid-batch on the constraint.
        """
        from carelite.eval.human.store import _ASSIGNMENT_UPSERT

        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            rolled_back.execute(
                _ASSIGNMENT_UPSERT,
                {
                    "rater_id": "R01",
                    "generation_id": "CAL-01",
                    "display_order": 1,
                    "blind_label": "R01-C01",
                    "is_calibration": True,
                },
            )
