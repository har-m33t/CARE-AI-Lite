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
    from carelite.eval.judge.store import UPSERT_SQL, upsert_params

    score = result.to_rubric_score(rater_id=rater_id)
    conn.execute(UPSERT_SQL, upsert_params(score, sample_idx))


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
        from carelite.eval.judge.store import UPSERT_SQL, upsert_params

        result = _result({**dict.fromkeys(RUBRIC_DIMENSIONS, 3), "de": 5}, n=5)
        for sample, score in zip(
            result.samples, result.per_sample_rubric_scores(rater_id="gpt-oss:20b"), strict=True
        ):
            rolled_back.execute(UPSERT_SQL, upsert_params(score, sample.sample_idx))
        _upsert(rolled_back, result, 0, median_rater_id("gpt-oss:20b"))

        rows = rolled_back.execute(
            "SELECT rater_id, sample_idx FROM rubric_score WHERE generation_id = %s "
            "ORDER BY rater_id, sample_idx",
            (GENERATION_ID,),
        ).fetchall()
        assert len(rows) == 6
        assert sum(1 for r in rows if r["rater_id"].endswith(MEDIAN_RATER_SUFFIX)) == 1


def _write_study(conn: psycopg.Connection, assignment: Assignment) -> None:
    """The store module's study statement, on the test's own rolled-back connection."""
    from carelite.eval.human.store import _ASSIGNMENT_UPSERT

    conn.execute(
        _ASSIGNMENT_UPSERT,
        {
            "rater_id": assignment.rater_id,
            "generation_id": assignment.generation_id,
            "display_order": assignment.display_order,
            "blind_label": assignment.blind_label,
        },
    )


def _write_calibration(conn: psycopg.Connection, assignment: Assignment) -> None:
    """The store module's calibration statement, likewise."""
    from carelite.eval.human.store import _CALIBRATION_UPSERT

    conn.execute(
        _CALIBRATION_UPSERT,
        {
            "rater_id": assignment.rater_id,
            "calibration_id": assignment.calibration_id,
            "display_order": assignment.display_order,
            "blind_label": assignment.blind_label,
        },
    )


def _rejected(
    conn: psycopg.Connection, constraint: str, values: str, params: tuple[object, ...] = ()
) -> None:
    """Assert one `rating_assignment` row is refused by the named CHECK constraint.

    `values` is the tail of the VALUES list, after the rater id. The insert runs
    inside a savepoint that always rolls back, because a constraint violation
    aborts the surrounding transaction and would otherwise poison every
    statement after it — including the fixture's own cleanup.
    """
    sql = (
        "INSERT INTO rating_assignment (rater_id, generation_id, calibration_id, "
        f"display_order, blind_label, is_calibration) VALUES ('R01', {values})"
    )
    with (
        conn.transaction(force_rollback=True),
        pytest.raises(psycopg.errors.CheckViolation, match=constraint),
    ):
        conn.execute(sql, params)


class TestRatingAssignmentRoundTrip:
    def test_assignment_persists_and_unblinds_by_join(
        self, rolled_back: psycopg.Connection
    ) -> None:
        """The blinding key is a table, so unblinding is a join."""
        _write_study(rolled_back, Assignment.for_generation("R01", GENERATION_ID, 7, "R01-042"))

        row = rolled_back.execute(
            "SELECT ra.blind_label, ra.is_calibration, ra.calibration_id, g.condition "
            "FROM rating_assignment ra JOIN generation g USING (generation_id) "
            "WHERE ra.rater_id = %s",
            ("R01",),
        ).fetchone()
        assert row["blind_label"] == "R01-042"
        assert row["condition"] == "C"
        assert row["is_calibration"] is False
        assert row["calibration_id"] is None

    def test_a_calibration_assignment_persists_without_a_generation_row(
        self, rolled_back: psycopg.Connection
    ) -> None:
        """The case the schema was amended for, now actually exercised.

        `calibration_id` is deliberately not a foreign key, so a fixture with no
        `generation` row stores cleanly — which is what makes a rater's packet
        recoverable from the database rather than only from the export file.
        """
        _write_calibration(rolled_back, Assignment.for_calibration("R01", "CAL-01", 1, "R01-C01"))

        row = rolled_back.execute(
            "SELECT generation_id, calibration_id, is_calibration, blind_label "
            "FROM rating_assignment WHERE rater_id = %s AND calibration_id = %s",
            ("R01", "CAL-01"),
        ).fetchone()
        assert row["generation_id"] is None
        assert row["calibration_id"] == "CAL-01"
        assert row["is_calibration"] is True
        assert row["blind_label"] == "R01-C01"

    def test_a_calibration_row_never_joins_to_a_generation(
        self, rolled_back: psycopg.Connection
    ) -> None:
        """The unblinding join must return study rows only, silently and correctly.

        This is the quiet-defect check. A calibration item that unblinded to
        something — or to a NULL that a downstream group-by kept — would land in
        the agreement computation as an extra unit, which raises alpha rather
        than raising an error.
        """
        _write_study(rolled_back, Assignment.for_generation("R01", GENERATION_ID, 7, "R01-042"))
        _write_calibration(rolled_back, Assignment.for_calibration("R01", "CAL-01", 1, "R01-C01"))

        rows = rolled_back.execute(
            "SELECT ra.blind_label FROM rating_assignment ra "
            "JOIN generation g USING (generation_id) WHERE ra.rater_id = %s",
            ("R01",),
        ).fetchall()
        assert [r["blind_label"] for r in rows] == ["R01-042"]

    def test_both_targets_null_is_rejected(self, rolled_back: psycopg.Connection) -> None:
        _rejected(rolled_back, "one_target", "NULL, NULL, 1, 'R01-001', FALSE")

    def test_both_targets_set_is_rejected(self, rolled_back: psycopg.Connection) -> None:
        _rejected(rolled_back, "one_target", "%s, 'CAL-01', 1, 'R01-001', TRUE", (GENERATION_ID,))

    def test_the_flag_disagreeing_with_the_target_is_rejected(
        self, rolled_back: psycopg.Connection
    ) -> None:
        """Both directions. This is the constraint the old `store.py` violated.

        The old statement wrote `'CAL-01'` into `generation_id` with
        `is_calibration = true` and `calibration_id` left NULL — the first of
        these two rows, near enough.
        """
        _rejected(rolled_back, "flag_agrees", "%s, NULL, 1, 'R01-001', TRUE", (GENERATION_ID,))
        _rejected(rolled_back, "flag_agrees", "NULL, 'CAL-01', 1, 'R01-C01', FALSE")

    def test_both_upserts_are_idempotent(self, rolled_back: psycopg.Connection) -> None:
        """Re-exporting a packet must not duplicate-key on either unique constraint."""
        study = Assignment.for_generation("R01", GENERATION_ID, 7, "R01-042")
        cal = Assignment.for_calibration("R01", "CAL-01", 1, "R01-C01")
        for _ in range(2):
            _write_study(rolled_back, study)
            _write_calibration(rolled_back, cal)

        count = rolled_back.execute(
            "SELECT count(*) AS n FROM rating_assignment WHERE rater_id = %s", ("R01",)
        ).fetchone()
        assert count["n"] == 2
