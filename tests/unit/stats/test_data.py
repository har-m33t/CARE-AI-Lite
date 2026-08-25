"""The database read, and the reshaping it feeds.

`to_long` is pure and is tested directly. The queries themselves need Postgres
and are marked `db`; they are excluded from `make check` by design and are
read-only, so they add no cleanup burden.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from carelite.eval.judge.store import MEDIAN_RATER_SUFFIX
from carelite.stats.data import (
    JUDGE_SAMPLES_SQL,
    SCORES_SQL,
    load_judge_samples,
    load_scores,
    to_long,
)
from carelite.stats.measures import attach_quality
from carelite.types import RUBRIC_DIMENSIONS


def _wide_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "generation_id": "g1",
        "scenario_id": "SC-000",
        "condition": "A",
        "sample_idx": 0,
        "rater_type": "llm_judge",
        "rater_id": "gpt-oss:20b-median",
        "split": "holdout",
        "equity_stratum": False,
        "fell_back_to_b": False,
    }
    row.update(dict.fromkeys(RUBRIC_DIMENSIONS, 3))
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Reshaping
# ---------------------------------------------------------------------------


def test_to_long_produces_one_row_per_dimension() -> None:
    long = to_long(pd.DataFrame([_wide_row()]))
    assert len(long) == len(RUBRIC_DIMENSIONS)
    assert set(long["dimension"]) == set(RUBRIC_DIMENSIONS)
    assert long["raw"].tolist() == [3.0] * len(RUBRIC_DIMENSIONS)


def test_to_long_keeps_every_identifying_column() -> None:
    long = to_long(pd.DataFrame([_wide_row()]))
    for column in ("generation_id", "scenario_id", "condition", "rater_type", "split"):
        assert column in long.columns
        assert long[column].nunique() == 1


def test_to_long_keeps_a_null_score_as_missing_rather_than_dropping_the_row() -> None:
    """§10: an ungrounded judge score is missing for that dimension only."""
    long = to_long(pd.DataFrame([_wide_row(explore=None)]))
    assert len(long) == len(RUBRIC_DIMENSIONS)
    explore = long[long["dimension"] == "explore"]["raw"].iloc[0]
    assert math.isnan(explore)


def test_to_long_handles_a_frame_carrying_only_some_dimensions() -> None:
    row = {
        "generation_id": "g1",
        "scenario_id": "SC-000",
        "condition": "A",
        "name": 4,
        "ritualistic": 5,
    }
    long = to_long(pd.DataFrame([row]))
    assert set(long["dimension"]) == {"name", "ritualistic"}


def test_to_long_rejects_a_frame_with_no_rubric_columns() -> None:
    with pytest.raises(KeyError, match="none of the rubric dimensions"):
        to_long(pd.DataFrame([{"generation_id": "g1"}]))


def test_to_long_output_feeds_attach_quality_directly() -> None:
    """The two halves of the pipeline have to fit; assert the seam, not each side."""
    long = to_long(pd.DataFrame([_wide_row(ritualistic=5)]))
    scored = attach_quality(long).set_index("dimension")["quality"]
    assert scored["ritualistic"] == 1.0
    assert scored["name"] == 3.0


# ---------------------------------------------------------------------------
# The queries as text: the parts that are easy to get wrong
# ---------------------------------------------------------------------------


def test_the_score_query_restricts_to_one_split() -> None:
    """§6: confirmatory analyses run on the held-out split only."""
    assert "WHERE sc.split = %(split)s" in SCORES_SQL


def test_the_score_query_takes_only_the_judge_median_row() -> None:
    """Both the per-sample rows and the median row exist; taking both doubles n."""
    assert "rs.rater_id LIKE %(median_pattern)s" in SCORES_SQL
    assert "rs.rater_type <> 'llm_judge'" in SCORES_SQL


def test_the_sample_query_takes_only_the_non_median_rows() -> None:
    assert "rs.rater_id NOT LIKE %(median_pattern)s" in JUDGE_SAMPLES_SQL
    assert "rs.rater_type = 'llm_judge'" in JUDGE_SAMPLES_SQL


def test_the_two_queries_partition_the_judge_rows_between_them() -> None:
    """One takes `LIKE`, the other `NOT LIKE`, on the same suffix."""
    assert MEDIAN_RATER_SUFFIX == "-median"
    assert "LIKE %(median_pattern)s" in SCORES_SQL
    assert "NOT LIKE %(median_pattern)s" in JUDGE_SAMPLES_SQL


def test_the_score_query_left_joins_the_retrieval_trace() -> None:
    """Conditions other than C have no trace; that absence is not missing data."""
    assert "LEFT JOIN retrieval_trace" in SCORES_SQL
    assert "COALESCE(rt.fell_back_to_b, FALSE)" in SCORES_SQL


def test_the_score_query_selects_all_eleven_dimensions() -> None:
    for dimension in RUBRIC_DIMENSIONS:
        assert f"rs.{dimension}" in SCORES_SQL


# ---------------------------------------------------------------------------
# Against a live database
# ---------------------------------------------------------------------------


@pytest.mark.db
def test_the_score_query_runs_against_the_real_schema() -> None:
    """Read-only: proves the joins and column names match `db/schema.sql`.

    With `generation` and `rubric_score` empty this returns nothing, which is
    the point — it is the query being validated, not the data.
    """
    frame = load_scores()
    assert isinstance(frame, pd.DataFrame)
    for column in ("generation_id", "scenario_id", "condition", "dimension", "raw"):
        assert column in frame.columns


@pytest.mark.db
def test_the_judge_sample_query_runs_against_the_real_schema() -> None:
    frame = load_judge_samples()
    assert isinstance(frame, pd.DataFrame)
    assert "rater_sample_idx" in frame.columns


@pytest.mark.db
def test_the_loaded_holdout_reads_back_as_one_row_per_generation_and_dimension() -> None:
    """The shape the analysis depends on, asserted against the real table.

    939 judged generations x 11 dimensions = 10,329 long rows, and every judge
    row is an aggregate `-median` row so no generation appears twice. This is
    the assertion that would have caught the `dict_row` bug in `_read`: under it
    the frame had the right number of rows and one distinct generation id.
    """
    from carelite.stats.report import run_analysis

    frame = load_scores()
    assert not frame.empty, "rubric_score is empty; run `python -m carelite.eval.judge.load`"
    assert frame["generation_id"].nunique() > 1, "every cell is its own column name — see _read"
    assert frame.shape[0] == frame["generation_id"].nunique() * len(RUBRIC_DIMENSIONS)
    assert frame["raw"].notna().any()

    per_generation = frame.groupby(["generation_id", "dimension"]).size()
    assert per_generation.max() == 1, "a generation is scored twice; the median filter is leaking"

    report = run_analysis(n_boot=50)
    assert not report.empty
    assert report.n_generations == frame["generation_id"].nunique() - 39  # D11 drops LC


# ---------------------------------------------------------------------------
# The dict_row regression
# ---------------------------------------------------------------------------


class _FakeCursor:
    """A psycopg-shaped cursor whose rows are dicts, like `carelite.db.connect`'s.

    This is the exact shape that broke `_read`. `carelite.db.connect` sets
    psycopg's `dict_row` factory, and `pandas.read_sql` iterates each row to
    build a column -- iterating a dict yields its KEYS. Every cell came back as
    the name of its own column: a frame of the right size, full of strings,
    which every downstream `to_numeric(errors="coerce")` turns silently into
    NaN. Nothing raises. The analysis simply finds no data and reports that as
    though it were a fact about the study.

    It was invisible for as long as `rubric_score` was empty, which is why it
    survived to the run. This double is how it stays fixed without Postgres.
    """

    class _Column:
        def __init__(self, name: str) -> None:
            self.name = name

    def __init__(self, columns: list[str], rows: list[tuple[object, ...]]) -> None:
        self._columns = columns
        self._rows = rows
        self.executed: tuple[str, object] | None = None

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    @property
    def description(self) -> list[_FakeCursor._Column]:
        return [self._Column(c) for c in self._columns]

    def execute(self, sql: str, params: object = None) -> None:
        self.executed = (sql, params)

    def fetchall(self) -> list[dict[str, object]]:
        return [dict(zip(self._columns, row, strict=True)) for row in self._rows]


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _FakeCursor:
        return self._cursor


def test_a_dict_row_connection_yields_values_not_column_names() -> None:
    """The regression test for the bug described on `_FakeCursor`."""
    from carelite.stats.data import _read

    cursor = _FakeCursor(
        ["generation_id", "condition", "name", "understand"],
        [("gen-1", "A", 4, 5), ("gen-2", "B", 2, 1)],
    )
    frame = _read("SELECT 1", {}, _FakeConnection(cursor))

    assert list(frame.columns) == ["generation_id", "condition", "name", "understand"]
    assert frame["generation_id"].tolist() == ["gen-1", "gen-2"]
    assert frame["name"].tolist() == [4, 2]
    # The precise symptom the bug produced, asserted directly.
    assert frame["condition"].tolist() != ["condition", "condition"]


def test_a_tuple_row_connection_still_works() -> None:
    """A caller may pass a connection without the dict factory; both shapes are handled."""
    from carelite.stats.data import _frame_from_cursor

    class _TupleCursor(_FakeCursor):
        def fetchall(self) -> list[tuple[object, ...]]:  # type: ignore[override]
            return self._rows

    cursor = _TupleCursor(["generation_id", "name"], [("gen-1", 4)])
    cursor.execute("SELECT 1")
    frame = _frame_from_cursor(cursor)
    assert frame["generation_id"].tolist() == ["gen-1"]
    assert frame["name"].tolist() == [4]


def test_an_empty_result_keeps_its_columns() -> None:
    """An empty read must be shaped, so the empty-frame path is not a special case."""
    from carelite.stats.data import _read

    frame = _read("SELECT 1", {}, _FakeConnection(_FakeCursor(["generation_id", "name"], [])))
    assert frame.empty
    assert list(frame.columns) == ["generation_id", "name"]
