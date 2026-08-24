"""The harness's SQL against the frozen schema, without needing Postgres.

This file exists because of a specific failure. `rating_assignment` was amended
to give calibration items their own nullable `calibration_id` column, and this
lane never consumed the amendment — `store.py` went on writing `'CAL-01'` into
`generation_id`. The only test that could see it was marked `db`, so `make check`
deselected it and the harness stayed broken and green for as long as nobody
happened to have a database running.

The lesson is not "run the db tests more often". It is that a schema-level
invariant guarded only by a test that usually does not run is not guarded. So
the same agreement is checked three times, at three costs:

1. `Assignment.__post_init__` — the rules as Python, so a bad assignment cannot
   be constructed. Tested in `test_human_blinding.TestAssignmentInvariant`.
2. This file — the harness's SQL read as text against `schema.sql` read as text.
   No database, no fixtures, runs on every `make check`. It cannot prove the
   statements execute, only that they name the columns the schema has and the
   constraints it declares, which is exactly the class of drift that happened.
3. `test_store_db.py` — the real round trip, marked `db`. The only one that
   proves Postgres agrees, and the only one that usually does not run.

Layer 2 is the cheap one that would have caught this, so it is the one that is
kept honest: it reads the real schema file rather than a copy of it, and fails
if `rating_assignment` stops declaring what the harness assumes.
"""

from __future__ import annotations

import re
from pathlib import Path

from carelite.eval.human.store import _ASSIGNMENT_UPSERT, _CALIBRATION_UPSERT

SCHEMA = Path(__file__).resolve().parents[3] / "carelite" / "db" / "schema.sql"


def rating_assignment_ddl() -> str:
    """The `CREATE TABLE rating_assignment (...)` body, from the frozen schema."""
    text = SCHEMA.read_text(encoding="utf-8")
    match = re.search(
        r"CREATE TABLE IF NOT EXISTS rating_assignment\s*\((.*?)\n\);", text, re.DOTALL
    )
    assert match is not None, "rating_assignment is not declared in schema.sql"
    return match.group(1)


class TestSchemaStillDeclaresWhatTheHarnessAssumes:
    def test_the_two_targets_are_separate_columns(self) -> None:
        ddl = rating_assignment_ddl()
        assert re.search(r"\bgeneration_id\s+TEXT\s+REFERENCES generation", ddl)
        # Deliberately NOT a foreign key: calibration items are fixtures.
        assert re.search(r"\bcalibration_id\s+TEXT", ddl)
        assert not re.search(r"\bcalibration_id\s+TEXT\s+REFERENCES", ddl)

    def test_generation_id_is_nullable(self) -> None:
        """A calibration row leaves it NULL. If NOT NULL comes back, we are broken."""
        ddl = rating_assignment_ddl()
        line = next(ln for ln in ddl.splitlines() if "generation_id" in ln)
        assert "NOT NULL" not in line

    def test_both_check_constraints_are_present(self) -> None:
        ddl = rating_assignment_ddl()
        assert "num_nonnulls(generation_id, calibration_id) = 1" in ddl
        assert "is_calibration = (calibration_id IS NOT NULL)" in ddl

    def test_the_two_unique_constraints_are_separate(self) -> None:
        """Why two upserts: one statement can only name one conflict target."""
        ddl = rating_assignment_ddl()
        assert "UNIQUE (rater_id, generation_id)" in ddl
        assert "UNIQUE (rater_id, calibration_id)" in ddl


class TestUpsertsMatchTheSchema:
    def test_study_upsert_writes_generation_and_never_calibration(self) -> None:
        sql = " ".join(_ASSIGNMENT_UPSERT.split())
        assert "ON CONFLICT (rater_id, generation_id)" in sql
        assert "%(generation_id)s" in sql
        assert "%(calibration_id)s" not in sql
        # The regression itself: a study row must pin the flag false and the
        # calibration column NULL, so both CHECKs hold by construction.
        assert "NULL, %(display_order)s, %(blind_label)s, FALSE" in sql

    def test_calibration_upsert_writes_calibration_and_never_generation(self) -> None:
        sql = " ".join(_CALIBRATION_UPSERT.split())
        assert "ON CONFLICT (rater_id, calibration_id)" in sql
        assert "%(calibration_id)s" in sql
        assert "%(generation_id)s" not in sql
        assert "NULL, %(calibration_id)s, %(display_order)s, %(blind_label)s, TRUE" in sql

    def test_neither_upsert_lets_the_flag_be_passed_in(self) -> None:
        """`is_calibration` is decided by which statement runs, not by a parameter.

        The broken version took it as a bind parameter, which is how a row could
        claim `is_calibration = true` while its id sat in `generation_id`. With
        the value literal in each statement, that row cannot be written by this
        module at all.
        """
        for sql in (_ASSIGNMENT_UPSERT, _CALIBRATION_UPSERT):
            assert "%(is_calibration)s" not in sql

    def test_every_bind_parameter_is_supplied_by_store_assignments(self) -> None:
        """Catches a parameter added to the SQL and not to the dict that feeds it."""
        source = Path(__file__).resolve().parents[3] / "carelite" / "eval" / "human" / "store.py"
        supplied = set(re.findall(r'"(\w+)":', source.read_text(encoding="utf-8")))
        for sql in (_ASSIGNMENT_UPSERT, _CALIBRATION_UPSERT):
            assert set(re.findall(r"%\((\w+)\)s", sql)) <= supplied
