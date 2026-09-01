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

*`scenario`*, because the analysis runs on the held-out split only
(analysis plan §6: "All confirmatory analyses below run on the 60-scenario
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

**`served_by` is selected on every score row, and is not decoration.** After
`DECISIONS.md` D13 the condition label does not identify an arm: `condition =
'LC'` matches the 180-cell vLLM arm and D11's 39 Ollama cells alike. The column
is the only thing in the frame that separates them, so it is read here and
`carelite.stats.arms` is what acts on it. A frame built without it cannot be
resolved into arms and the guard there refuses rather than guessing.

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
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from carelite.eval.judge.store import MEDIAN_RATER_SUFFIX
from carelite.types import RUBRIC_DIMENSIONS, RaterType, Split

__all__ = [
    "DROPPED_CONDITIONS",
    "GENERATION_COUNTS_SQL",
    "JUDGE_SAMPLES_SQL",
    "SCORES_SQL",
    "SERVING_BACKENDS",
    "DataInventory",
    "attach_equity_kind",
    "drop_dropped_conditions",
    "inventory",
    "load_generation_counts",
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
    g.served_by,
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
    g.gate_blocked,
    COALESCE(rt.fell_back_to_b, FALSE) AS fell_back_to_b,
    rt.crag_grade,
    COALESCE(array_length(rt.retrieved_ids, 1), 0) AS n_retrieved
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

#: The `served_by` vocabulary, which is `carelite/db/schema.sql`'s CHECK
#: constraint and not this module's to widen. It is restated here so the count
#: breakdown can print a zero for a backend that produced nothing, rather than
#: omitting the row and leaving "no vLLM generations" indistinguishable from
#: "nobody looked". `tests/unit/stats/test_headline.py` reads the constraint out
#: of the schema and fails if the two disagree.
SERVING_BACKENDS: tuple[str, ...] = ("ollama", "vllm")

#: One row per (condition, serving stack, split), counted from `generation`
#: itself with no exclusion applied. This is deliberately **not** the frame the
#: analysis runs on: it counts the LC cells D11 dropped and the gate-blocked
#: cells D12 flags, because "how many generations does this study have" and "how
#: many generations did the primary comparison run on" are different questions
#: that a single number has been asked to answer before.
#:
#: `served_by` is in the grouping because condition LC may yet be re-run under a
#: second serving stack. Two backends serve different artifacts of the same
#: model family, so a count pooled across them would hide a confound rather than
#: report one.
#:
#: `rubric_score` is joined through a DISTINCT subselect: a generation has one
#: row per rater and dimension set, so a plain join would multiply every count
#: it touches.
GENERATION_COUNTS_SQL = """
SELECT
    g.condition,
    g.served_by,
    sc.split,
    COUNT(*)                                             AS n_generations,
    COUNT(*) FILTER (WHERE g.gate_blocked)               AS n_gate_blocked,
    COUNT(*) FILTER (WHERE s.generation_id IS NOT NULL)  AS n_scored,
    COUNT(DISTINCT g.scenario_id)                        AS n_scenarios
FROM generation g
JOIN scenario sc ON sc.scenario_id = g.scenario_id
LEFT JOIN (SELECT DISTINCT generation_id FROM rubric_score) s
       ON s.generation_id = g.generation_id
GROUP BY g.condition, g.served_by, sc.split
ORDER BY sc.split, g.condition, g.served_by
"""


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Melt a one-row-per-rating frame into one row per (rating, dimension).

    Pure: no database, no configuration. This is the function the reshaping
    tests exercise, so the shape the analysis depends on is verified without a
    live Postgres.

    Rows whose score is NULL are kept, with `raw` as NaN. Analysis plan §10
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


def _frame_from_cursor(cur: Any) -> pd.DataFrame:
    """Build a frame from an executed cursor, whatever row factory it carries.

    **This is not the obvious `pandas.read_sql` call, and the reason is a bug
    that was live in this module until the results table had rows in it.**
    `carelite.db.connect` sets psycopg's `dict_row` factory, so each row is a
    `dict`. `read_sql` iterates whatever the cursor yields and builds columns
    from it -- and iterating a `dict` yields its *keys*. Every cell came back as
    the name of its own column: 10,329 rows, `generation_id.nunique() == 1`,
    `raw` entirely NaN.

    The failure was invisible for as long as `rubric_score` was empty, because an
    empty result set is empty under either reading. It would have been invisible
    afterwards too -- a frame of the right shape, full of strings, that every
    downstream `to_numeric(errors="coerce")` turns quietly into NaN. Nothing
    would have raised; the analysis would simply have found no data anywhere and
    said so in a way that looked like a finding about the study.

    So the rows are fetched explicitly and the column names come from
    `cur.description`, which is authoritative regardless of row factory. Both
    row shapes are handled, because a caller may pass a tuple-row connection.
    """
    rows = cur.fetchall()
    columns = [d.name for d in cur.description] if cur.description else []
    if not rows:
        return pd.DataFrame(columns=columns)
    if isinstance(rows[0], dict):
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _read(sql: str, params: dict[str, Any], conn: Any | None) -> pd.DataFrame:
    """Run one query and return it as a frame. See `_frame_from_cursor`."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pandas only supports SQLAlchemy connectable",
            category=UserWarning,
        )
        if conn is not None:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return _frame_from_cursor(cur)
        from carelite.db import connect

        with connect() as opened, opened.cursor() as cur:
            cur.execute(sql, params)
            return _frame_from_cursor(cur)


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
            "served_by",
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
            "gate_blocked",
            "fell_back_to_b",
            "crag_grade",
            "n_retrieved",
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


def load_generation_counts(*, conn: Any | None = None) -> pd.DataFrame:
    """Row counts straight out of `generation`, grouped by condition and backend.

    No split filter and no exclusion: this is the census the headline block
    quotes, and every decision that narrows it -- the holdout restriction, D11's
    dropped condition, D12's gate-blocked rows -- is applied downstream and
    reported as its own number. See `GENERATION_COUNTS_SQL`.

    An empty frame with the right columns comes back when the table is empty,
    which is a legitimate answer and not an error.
    """
    frame = _read(GENERATION_COUNTS_SQL, {}, conn)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "condition",
                "served_by",
                "split",
                "n_generations",
                "n_gate_blocked",
                "n_scored",
                "n_scenarios",
            ]
        )
    return frame


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


# ---------------------------------------------------------------------------
# Conditions dropped wholesale -- of which, after D13, there are none
# ---------------------------------------------------------------------------

#: **Empty, and that is the decision.** D11 dropped condition LC entirely: it had
#: been stopped at 39 of 180 cells over 13 of 60 scenarios, never randomised for
#: partial analysis, so it was not a small arm but a non-arm. D13 re-opened it —
#: all 180 cells were generated under vLLM — and the exclusion moved from the
#: condition to the `(condition, served_by)` pair, because a rule that still
#: dropped `LC` would now discard the arm D13 exists to restore.
#:
#: The selection rule now lives in `carelite.stats.arms.EXCLUDED_ARMS`, and
#: `restrict_to_analysis_arms` is what the report calls. `drop_dropped_conditions`
#: is kept as the general utility it always was, for a caller that wants to
#: exclude a condition by name.
DROPPED_CONDITIONS: tuple[str, ...] = ()


def drop_dropped_conditions(
    long: pd.DataFrame,
    *,
    conditions: Sequence[str] = DROPPED_CONDITIONS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove whole conditions by name. Returns the frame and what was removed.

    The counts come back rather than being logged, because "n cells over m
    scenarios were excluded" is a sentence the results document has to contain
    and a number nobody should have to re-derive to write it.

    `conditions` defaults to `DROPPED_CONDITIONS`, which is empty after D13.
    Excluding a serving stack within a condition is a different operation and is
    `carelite.stats.arms.restrict_to_analysis_arms`.
    """
    if long.empty or "condition" not in long.columns:
        return long, {}
    wanted = {str(c) for c in conditions}
    mask = long["condition"].astype(str).isin(wanted)
    removed: dict[str, int] = {}
    if mask.any():
        counts = long.loc[mask].groupby(long.loc[mask, "condition"].astype(str))["generation_id"]
        removed = {str(k): int(v) for k, v in counts.nunique().items()}
    return long[~mask], removed


# ---------------------------------------------------------------------------
# What the analysis actually had to work with
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DataInventory:
    """What came out of the database, before any analysis decision was applied.

    Every exclusion this package makes is a decision someone could disagree
    with, so the inventory is computed once from the unfiltered read and
    rendered at the top of the report. A reader who wants a different set of
    exclusions can see exactly how many rows each one costs.
    """

    n_rows: int
    n_generations: int
    n_scenarios: int
    by_condition: dict[str, int] = field(default_factory=dict)
    n_gate_blocked: int = 0
    gate_blocked_by_scenario: dict[str, int] = field(default_factory=dict)
    gate_blocked_by_condition: dict[str, int] = field(default_factory=dict)
    n_fell_back: int = 0
    n_retrieved_cells: int = 0
    n_incomplete_generations: int = 0
    missing_by_dimension: dict[str, int] = field(default_factory=dict)
    dropped_conditions: dict[str, int] = field(default_factory=dict)
    rater_types: tuple[str, ...] = ()

    def render(self) -> str:
        conditions = ", ".join(f"{k} {v}" for k, v in sorted(self.by_condition.items())) or "-"
        missing = (
            ", ".join(f"{k} {v}" for k, v in sorted(self.missing_by_dimension.items()) if v) or "-"
        )
        blocked_scen = (
            ", ".join(
                f"{k} {v}"
                for k, v in sorted(
                    self.gate_blocked_by_scenario.items(), key=lambda kv: (-kv[1], kv[0])
                )
            )
            or "-"
        )
        blocked_cond = (
            ", ".join(f"{k} {v}" for k, v in sorted(self.gate_blocked_by_condition.items())) or "-"
        )
        lines = [
            "DATA INVENTORY — what was read, before any exclusion",
            f"  {self.n_generations} scored generations over {self.n_scenarios} scenarios; "
            f"raters: {', '.join(self.rater_types) or '(none)'}",
            f"  by condition: {conditions}",
            "",
            f"  gate-blocked (D12):      {self.n_gate_blocked} generations "
            f"[by scenario: {blocked_scen}] [by condition: {blocked_cond}]",
            f"  CRAG fell back to B:     {self.n_fell_back} generations "
            f"({self.n_retrieved_cells} actually retrieved)",
            f"  incomplete on >= 1 dim:  {self.n_incomplete_generations} generations "
            f"[missing per dimension: {missing}]",
        ]
        if self.dropped_conditions:
            dropped = ", ".join(
                f"{k} {v} cells" for k, v in sorted(self.dropped_conditions.items())
            )
            lines.append(f"  excluded selections:     {dropped}")
        return "\n".join(lines)


def inventory(long: pd.DataFrame, *, dropped: dict[str, int] | None = None) -> DataInventory:
    """Count everything the report has to declare, from one unfiltered frame.

    Computed on the long frame, so `missing_by_dimension` counts (generation,
    dimension) cells the judge did not score rather than whole rows -- which is
    the resolution the missing-data policy operates at.
    """
    if long.empty:
        return DataInventory(n_rows=0, n_generations=0, n_scenarios=0)

    def _unique_flag(column: str) -> int:
        if column not in long.columns:
            return 0
        flag = long[column].astype("boolean").fillna(False)
        return int(long.loc[flag, "generation_id"].nunique())

    per_generation = long.groupby("generation_id", observed=True)["raw"]
    n_incomplete = int((per_generation.apply(lambda s: s.isna().any())).sum())

    missing = {
        str(dim): int(group["raw"].isna().sum())
        for dim, group in long.groupby("dimension", observed=True)
    }

    blocked_scenario: dict[str, int] = {}
    blocked_condition: dict[str, int] = {}
    if "gate_blocked" in long.columns:
        flag = long["gate_blocked"].astype("boolean").fillna(False)
        blocked = long.loc[flag].drop_duplicates(subset=["generation_id"])
        blocked_scenario = {
            str(k): int(v) for k, v in blocked.groupby("scenario_id", observed=True).size().items()
        }
        blocked_condition = {
            str(k): int(v) for k, v in blocked.groupby("condition", observed=True).size().items()
        }

    n_fell_back = _unique_flag("fell_back_to_b")
    retrieval_cells = 0
    if "fell_back_to_b" in long.columns and "crag_grade" in long.columns:
        traced = long[long["crag_grade"].notna()]
        retrieval_cells = int(traced["generation_id"].nunique()) - n_fell_back

    return DataInventory(
        n_rows=int(long.shape[0]),
        n_generations=int(long["generation_id"].nunique()),
        n_scenarios=int(long["scenario_id"].nunique()),
        by_condition={
            str(k): int(v)
            for k, v in long.groupby("condition", observed=True)["generation_id"].nunique().items()
        },
        n_gate_blocked=_unique_flag("gate_blocked"),
        gate_blocked_by_scenario=blocked_scenario,
        gate_blocked_by_condition=blocked_condition,
        n_fell_back=n_fell_back,
        n_retrieved_cells=max(0, retrieval_cells),
        n_incomplete_generations=n_incomplete,
        missing_by_dimension=missing,
        dropped_conditions=dict(dropped or {}),
        rater_types=tuple(sorted({str(r) for r in long["rater_type"].dropna()})),
    )
