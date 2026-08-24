"""Reading results out of Postgres, in the one shape the rest of this package uses.

Every number this package reports traces to a query in this module and a test in
`tests/unit/stats/`. The queries are written out in full rather than assembled
from fragments, so a reader can run one by hand against the database and get the
frame the analysis ran on.

**The long format.** One row per (generation, rater, dimension), carrying the
scenario stratification columns alongside. Wide-per-dimension would be more
compact and is what the table looks like; long is what makes it impossible to
average across dimensions without going through
`carelite.stats.measures.attach_quality`, because there is no set of eleven
columns sitting next to each other inviting a `.mean(axis=1)`.

**Three joins that are not optional.**

*`scenario`*, because the confirmatory analyses run on the held-out split only
(pre-registration §6: "All confirmatory analyses below run on the 60-scenario
held-out split only; the 40 train scenarios are for prompt and retrieval
development and are never scored as evaluation data"). `split` is a filter with
a default of `holdout`, not an optional convenience.

*`retrieval_trace`*, because sensitivity analysis (b) needs
`fell_back_to_b` per generation and it is only knowable from that table. A
LEFT JOIN with `COALESCE(..., FALSE)`: conditions other than C have no
retrieval trace at all, and their absence of a fallback is not missing data.

*the judge's median row*, because `carelite.eval.judge.store` writes both the
per-sample rows (under the judge's own rater id, `sample_idx` 0..4) and the
aggregate median (under `"<rater_id>-median"`, `sample_idx` 0). Selecting
`rater_type = 'llm_judge'` without discriminating returns both and
double-counts every generation -- and for a single-pass full run the median
equals sample 0, so the duplicate would be invisible in the means and would
simply halve every standard error. `load_scores` takes the median rows and
nothing else by default; `load_judge_samples` is the separate entry point for
the per-sample rows that the self-consistency sensitivity analysis needs.

**`equity_kind` is not in the database.** `carelite/db/schema.sql` stores the
eight frozen `Scenario` fields; `equity_kind` stays in `scenarios/bank.jsonl`
(`carelite/scenarios/load.py`: "the schema is not mine to extend").
`attach_equity_kind` joins it in from the bank, which is the frozen artefact the
holdout digest covers, so the join is against registered data rather than a
re-derivation.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import pandas as pd

from carelite.eval.judge.store import MEDIAN_RATER_SUFFIX
from carelite.types import RUBRIC_DIMENSIONS, RaterType, Split

__all__ = [
    "JUDGE_SAMPLES_SQL",
    "SCORES_SQL",
    "attach_equity_kind",
    "load_judge_samples",
    "load_scores",
    "to_long",
]


_DIMENSION_COLUMNS = ", ".join(f"rs.{d}" for d in RUBRIC_DIMENSIONS)

#: One row per (generation, rater). Judge rows are the aggregate median rows
#: only -- see the module docstring for why that matters.
SCORES_SQL = f"""
SELECT
    g.generation_id,
    g.scenario_id,
    g.condition,
    g.sample_idx,
    g.model,
    g.model_digest,
    g.prompt_id,
    rs.rater_type,
    rs.rater_id,
    rs.sample_idx        AS rater_sample_idx,
    {_DIMENSION_COLUMNS},
    sc.split,
    sc.challenge_type,
    sc.emotion_intensity,
    sc.encounter_phase,
    sc.literacy_signal,
    sc.equity_stratum,
    COALESCE(rt.fell_back_to_b, FALSE) AS fell_back_to_b,
    rt.crag_grade
FROM rubric_score rs
JOIN generation g       ON g.generation_id = rs.generation_id
JOIN scenario   sc      ON sc.scenario_id  = g.scenario_id
LEFT JOIN retrieval_trace rt ON rt.generation_id = g.generation_id
WHERE sc.split = %(split)s
  AND (
        rs.rater_type <> 'llm_judge'
        OR rs.rater_id LIKE %(median_pattern)s
      )
ORDER BY g.scenario_id, g.condition, g.sample_idx, rs.rater_type, rs.rater_id
"""

#: The judge's individual self-consistency samples, which `SCORES_SQL`
#: deliberately excludes. Used only by sensitivity analysis (c).
JUDGE_SAMPLES_SQL = f"""
SELECT
    g.generation_id,
    g.scenario_id,
    g.condition,
    rs.rater_id,
    rs.sample_idx AS rater_sample_idx,
    {_DIMENSION_COLUMNS}
FROM rubric_score rs
JOIN generation g  ON g.generation_id = rs.generation_id
JOIN scenario   sc ON sc.scenario_id  = g.scenario_id
WHERE sc.split = %(split)s
  AND rs.rater_type = 'llm_judge'
  AND rs.rater_id NOT LIKE %(median_pattern)s
ORDER BY g.scenario_id, g.condition, rs.rater_id, rs.sample_idx
"""


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Melt a one-row-per-rating frame into one row per (rating, dimension).

    Pure: no database, no configuration. This is the function the reshaping
    tests exercise, so the shape the analysis depends on is verified without a
    live Postgres.

    Rows whose score is NULL are kept, with `raw` as NaN. Pre-registration §10
    makes an ungrounded judge score missing *for that dimension only*, and
    dropping the row here would erase the distinction between "not scored" and
    "not attempted".
    """
    present = [d for d in RUBRIC_DIMENSIONS if d in wide.columns]
    if not present:
        raise KeyError(
            f"frame carries none of the rubric dimensions {RUBRIC_DIMENSIONS}; "
            f"got columns {list(wide.columns)}"
        )
    id_columns = [c for c in wide.columns if c not in present]
    long = wide.melt(
        id_vars=id_columns,
        value_vars=present,
        var_name="dimension",
        value_name="raw",
    )
    long["raw"] = pd.to_numeric(long["raw"], errors="coerce")
    sort_keys = [c for c in ("scenario_id", "condition", "generation_id", "rater_id") if c in long]
    if sort_keys:
        long = long.sort_values([*sort_keys, "dimension"], kind="mergesort")
    return long.reset_index(drop=True)


def _read(sql: str, params: dict[str, Any], conn: Any | None) -> pd.DataFrame:
    """`pandas.read_sql` over a psycopg connection.

    pandas warns that it only *tests* SQLAlchemy connectables; psycopg3 satisfies
    the DBAPI2 interface `read_sql` uses and is what the rest of the project
    connects with (`carelite.db.connect` registers the pgvector adapters), so the
    warning is suppressed narrowly here rather than by adding a second connection
    stack for this package alone.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pandas only supports SQLAlchemy connectable",
            category=UserWarning,
        )
        if conn is not None:
            return pd.read_sql(sql, conn, params=params)
        from carelite.db import connect

        with connect() as opened:
            return pd.read_sql(sql, opened, params=params)


def load_scores(
    *,
    split: Split | str = Split.HOLDOUT,
    conn: Any | None = None,
    rater_types: Sequence[RaterType | str] | None = None,
) -> pd.DataFrame:
    """Long-format scores for one split, judge rows deduplicated to the median row.

    `conn` is an open psycopg connection, for tests that want a transaction they
    can roll back; `None` opens one from `carelite.db.connect`.

    Returns an empty frame with the right columns when there are no rows -- which
    is the current state of the database, and is a legitimate answer rather than
    an error, so a caller can build and exercise the whole pipeline against it.
    """
    frame = _read(
        SCORES_SQL,
        {"split": str(split), "median_pattern": f"%{MEDIAN_RATER_SUFFIX}"},
        conn,
    )
    if frame.empty:
        columns = [
            "generation_id",
            "scenario_id",
            "condition",
            "sample_idx",
            "model",
            "model_digest",
            "prompt_id",
            "rater_type",
            "rater_id",
            "rater_sample_idx",
            "split",
            "challenge_type",
            "emotion_intensity",
            "encounter_phase",
            "literacy_signal",
            "equity_stratum",
            "fell_back_to_b",
            "crag_grade",
            "dimension",
            "raw",
        ]
        return pd.DataFrame(columns=columns)
    if rater_types is not None:
        wanted = {str(r) for r in rater_types}
        frame = frame[frame["rater_type"].astype(str).isin(wanted)]
    return to_long(frame)


def load_judge_samples(
    *,
    split: Split | str = Split.HOLDOUT,
    conn: Any | None = None,
) -> pd.DataFrame:
    """The judge's per-sample rows, long format. Input to sensitivity analysis (c)."""
    frame = _read(
        JUDGE_SAMPLES_SQL,
        {"split": str(split), "median_pattern": f"%{MEDIAN_RATER_SUFFIX}"},
        conn,
    )
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "generation_id",
                "scenario_id",
                "condition",
                "rater_id",
                "rater_sample_idx",
                "dimension",
                "raw",
            ]
        )
    return to_long(frame)


def attach_equity_kind(long: pd.DataFrame, *, bank_path: str | None = None) -> pd.DataFrame:
    """Join `equity_kind` in from `scenarios/bank.jsonl`.

    The axis is `ses`, `lep` or `racial_ethnic`, and `None` outside the stratum.
    `DECISIONS.md` D5 governs how `racial_ethnic` may be described -- see
    `carelite.stats.subgroups.RACIAL_ETHNIC_DESCRIPTION`, which is emitted with
    any result broken down by this column.
    """
    from carelite.scenarios.bank import load_bank

    mapping = {r.scenario_id: r.equity_kind for r in load_bank(bank_path)}
    out = long.copy()
    out["equity_kind"] = out["scenario_id"].map(mapping)
    return out
