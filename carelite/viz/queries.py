"""Database (and ablation-JSON) plumbing: build the tidy DataFrame each figure
in `carelite.viz.figures` expects, from live Postgres data plus whatever the
retrieval ablation harness has written out.

**Every statistical computation here delegates to `carelite.stats` or
`carelite.eval.judge`, never reimplements them.** This module's own job is
narrow: run the SQL, reshape rows into the long-format frame
`carelite.stats.measures` expects (`LONG_COLUMNS`), and flatten the rich
result objects those packages return (`PairedEffects`, `FriedmanResult`,
`PairwiseResult`, `Label`, `DimensionValidity`) into the plain columns
`carelite.viz.figures` reads. Two small statistics are the deliberate
exception, and both are documented at their definition: a caller-supplied
"mean" statistic handed to `carelite.stats.effects.bootstrap_ci` (that
function is explicitly designed to take an arbitrary statistic callable — the
mean is not a competing implementation of the resampling engine, just an
argument to it), and judge self-consistency's inter-sample variance, which is
a plain, non-controversial `pandas.groupby(...).var()` with no test, p-value,
or effect size attached, computed directly from persisted `rubric_score` rows
because reconstructing the judge lane's richer `JudgeResult` objects from the
database alone is out of this lane's scope.

Every function here either returns a `DataFrame` or raises `DataUnavailable`
with a human-readable reason — never a bare crash on an empty table — so
`carelite.viz.reproduce.run()` can report "figure N skipped: <reason>" for
exactly the tables that are still zero rows (`generation` and `rubric_score`
are both 0 rows as of this writing; see `docs/preregistration.md` on why that
is expected and not a bug in this lane).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from carelite.db.connection import fetch_all
from carelite.eval.judge.agreement import Metric, krippendorff_alpha, paired_series, spearman_rho
from carelite.eval.judge.validation import classify_dimension
from carelite.eval.rubric.dimensions import to_quality
from carelite.stats.effects import DEFAULT_N_BOOT, DEFAULT_SEED, BootstrapCI, bootstrap_ci
from carelite.stats.evidence import EvidenceStatus, RaterScope, label_for
from carelite.stats.measures import MEASURES, cell_means, measure, paired_matrix
from carelite.stats.primary import (
    CONFIRMATORY_FAMILY,
    FRIEDMAN_CONDITIONS,
    Hypothesis,
    friedman_across_conditions,
    run_family,
)
from carelite.types import RUBRIC_DIMENSIONS, Condition

__all__ = [
    "DataUnavailable",
    "ablation_table_df",
    "effect_sizes_df",
    "equity_subgroup_df",
    "judge_agreement_df",
    "judge_self_consistency_df",
    "load_long_scores",
    "negative_control_df",
    "retrieval_quality_df",
    "rubric_scores_df",
]


class DataUnavailable(RuntimeError):
    """Raised when a figure's underlying data does not exist yet (or not
    enough of it does) — distinct from a bug, so `carelite.viz.reproduce` can
    report a clean skip rather than a stack trace for the ordinary "the
    holdout run has not happened yet" case."""


_FRIEDMAN_CONDITION_STRS: tuple[str, ...] = tuple(str(c) for c in FRIEDMAN_CONDITIONS)


# ---------------------------------------------------------------------------
# Raw scores, long format
# ---------------------------------------------------------------------------


def load_long_scores(rater_type: str | None = "llm_judge") -> pd.DataFrame:
    """Every rubric score, melted into the long format
    `carelite.stats.measures` operates on, plus `equity_stratum` and
    `challenge_type` for subgroup queries.

    Columns: `generation_id, scenario_id, condition, sample_idx, rater_type,
    rater_id, equity_stratum, challenge_type, dimension, raw`.

    `rater_type=None` pulls every rater type (used only where a caller filters
    itself, e.g. building judge vs. human frames side by side).
    """
    where = "" if rater_type is None else "WHERE r.rater_type = %(rater_type)s"
    sql = f"""
        SELECT g.generation_id, g.scenario_id, g.condition, g.sample_idx,
               r.rater_type, r.rater_id,
               s.equity_stratum, s.challenge_type,
               r.name, r.understand, r.respect, r.support, r.explore,
               r.ib, r.epp, r.de, r.ie, r.naturalness, r.ritualistic
        FROM generation g
        JOIN scenario s ON s.scenario_id = g.scenario_id
        JOIN rubric_score r ON r.generation_id = g.generation_id
        {where}
    """
    rows = fetch_all(sql, {"rater_type": rater_type} if rater_type is not None else None)
    if not rows:
        raise DataUnavailable(
            f"no rubric_score rows for rater_type={rater_type!r} — the holdout run has not "
            "produced scored generations yet (docs/preregistration.md; generation and "
            "rubric_score are 0 rows as of this writing)"
        )
    wide = pd.DataFrame(rows)
    id_vars = [
        "generation_id",
        "scenario_id",
        "condition",
        "sample_idx",
        "rater_type",
        "rater_id",
        "equity_stratum",
        "challenge_type",
    ]
    long = wide.melt(
        id_vars=id_vars, value_vars=list(RUBRIC_DIMENSIONS), var_name="dimension", value_name="raw"
    )
    return long


# ---------------------------------------------------------------------------
# Shared helper: bootstrap CI on a plain mean
# ---------------------------------------------------------------------------


def _mean_ci(
    values: np.ndarray, *, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED
) -> BootstrapCI:
    """`carelite.stats.effects.bootstrap_ci` with the mean as the statistic.

    Not a second bootstrap implementation — `bootstrap_ci` takes an arbitrary
    `(resampled_units) -> float` callable by design; this is that callable.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return BootstrapCI(math.nan, math.nan, 0.95, n_boot, 0, 0, seed)
    units = arr.reshape(-1, 1)
    return bootstrap_ci(units, lambda u: float(np.mean(u[:, 0])), n_boot=n_boot, seed=seed)


def _judge_statuses(long: pd.DataFrame) -> dict[str, EvidenceStatus] | None:
    """Best-effort per-dimension judge-validation status for gating `Label`s.

    `None` (everything demoted) unless the caller has human rows alongside
    the judge rows to validate against; that mirrors
    `carelite.eval.judge.validation`'s own "no validation study yet" default
    rather than inventing a more permissive one here.
    """
    if "rater_type" not in long.columns:
        return None
    if "human" not in set(long["rater_type"]):
        return None
    try:
        agreement = judge_agreement_df()
    except DataUnavailable:
        return None
    return {row["dimension"]: EvidenceStatus(row["status"]) for _, row in agreement.iterrows()}


# ---------------------------------------------------------------------------
# Figure 1 — per-condition rubric scores
# ---------------------------------------------------------------------------


def rubric_scores_df(
    long: pd.DataFrame,
    *,
    rater_type: str = "llm_judge",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Build `fig_rubric_scores`'s input: one row per condition x dimension."""
    statuses = _judge_statuses(long)
    friedman = friedman_across_conditions(long, rater_type=rater_type, statuses=statuses)
    friedman_by_dim = {f.measure_key: f for f in friedman}

    records: list[dict[str, object]] = []
    for dim in RUBRIC_DIMENSIONS:
        m = measure(dim)
        cells = cell_means(long, m)
        cells = cells[cells["rater_type"] == rater_type]
        confirmatory_dim = friedman_by_dim[dim].label.is_confirmatory
        for condition, group in cells.groupby("condition", observed=True):
            ci = _mean_ci(group["value"].to_numpy(), n_boot=n_boot, seed=seed)
            records.append(
                {
                    "condition": str(condition),
                    "dimension": dim,
                    "mean": float(np.mean(group["value"])),
                    "ci_lo": ci.low,
                    "ci_hi": ci.high,
                    "n": int(group.shape[0]),
                    "confirmatory": bool(
                        str(condition) in _FRIEDMAN_CONDITION_STRS and confirmatory_dim
                    ),
                }
            )
    if not records:
        raise DataUnavailable("no scenario x condition cells available for any rubric dimension")
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Figure 2 — effect sizes forest plot
# ---------------------------------------------------------------------------


def effect_sizes_df(
    long: pd.DataFrame,
    *,
    hypotheses: tuple[Hypothesis, ...] = CONFIRMATORY_FAMILY,
    rater_type: str = "llm_judge",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Build `fig_effect_sizes`'s input from `carelite.stats.primary.run_family`.

    Defaults to the eight pre-specified hypotheses
    (`carelite.stats.primary.CONFIRMATORY_FAMILY`) — the registered comparison
    set the forest plot exists to make legible. Pass
    `carelite.stats.primary.dimension_expansion()` (or a concatenation) for a
    broader, mostly-exploratory view.
    """
    statuses = _judge_statuses(long)
    family = run_family(
        long,
        hypotheses=hypotheses,
        rater_type=rater_type,
        statuses=statuses,
        include_friedman=False,
        n_boot=n_boot,
        seed=seed,
    )
    records: list[dict[str, object]] = []
    for r in family.results:
        h = r.hypothesis
        rb = r.effects.rank_biserial
        records.append(
            {
                "comparison": f"{h.left} vs {h.right}",
                "dimension": h.measure_key,
                "effect": rb.point,
                "ci_lo": rb.ci.low,
                "ci_hi": rb.ci.high,
                "n": r.n_scenarios,
                "p_value": r.p_holm,
                "confirmatory": r.label.is_confirmatory,
            }
        )
    if not records:
        raise DataUnavailable("no paired scenarios available for any pre-specified comparison")
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Figure 3 — ablation table (loaded from JSON, not the database)
# ---------------------------------------------------------------------------

#: `carelite.retrieval.ablation` has no fixed --out convention; this is this lane's
#: own convention for where to look, overridable by the `path` argument.
DEFAULT_ABLATION_PATH = Path("runs") / "ablation" / "ablation.json"

_ABLATION_REQUIRED = (
    "label",
    "n_turns",
    "on_domain_precision",
    "off_domain_rejection_rate",
    "fallback_rate",
    "skipped_rate",
    "mean_latency_ms",
)


def ablation_table_df(path: Path | None = None) -> pd.DataFrame:
    """Load the R0-R9 ablation table for `fig_ablation_table`.

    Deliberately does **not** accept the blended `context_precision` column
    `carelite.retrieval.ablation.AblationRow.to_dict()` emits as of this
    writing (see `carelite.viz.figures.fig_ablation_table`'s module note): the
    gate as currently computed fails structurally because half the probe
    turns are deliberately off-domain and contribute zeros no non-CRAG row can
    recover from. This loader requires the split `on_domain_precision` /
    `off_domain_rejection_rate` columns and raises `DataUnavailable` rather
    than silently substituting the blended one.
    """
    p = path or DEFAULT_ABLATION_PATH
    if not p.exists():
        raise DataUnavailable(
            f"no ablation output at {p} — run `python -m carelite.retrieval.ablation --out {p}` "
            "once carelite-retrieval lands the on/off-domain precision split"
        )
    rows = json.loads(p.read_text())
    if not rows:
        raise DataUnavailable(f"{p} contains no ablation rows")
    missing = [c for c in _ABLATION_REQUIRED if c not in rows[0]]
    if missing:
        raise DataUnavailable(
            f"{p} is missing {missing} — looks like the pre-split AblationRow schema; "
            "this figure needs the on_domain_precision/off_domain_rejection_rate split "
            "(see the coordination note in carelite.viz.figures.fig_ablation_table)"
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 4 — judge-vs-human agreement
# ---------------------------------------------------------------------------


def judge_agreement_df() -> pd.DataFrame:
    """Krippendorff's alpha / Spearman's rho vs. a human consensus, per dimension.

    Reuses `carelite.eval.judge.agreement.{krippendorff_alpha,spearman_rho,
    paired_series}` and `carelite.eval.judge.validation.classify_dimension`
    directly — the pre-specified threshold and the agreement arithmetic both
    stay owned by the judge lane. What is built locally is only the
    generation-id-keyed dict this module's DB rows produce, standing in for
    `carelite.eval.judge.judge.JudgeResult.scores()` /
    `carelite.eval.human.reliability.human_consensus`, which this lane does
    not have a live database row shape to reconstruct end-to-end.
    """
    judge_rows = fetch_all(
        "SELECT generation_id, name, understand, respect, support, explore, ib, epp, de, ie, "
        "naturalness, ritualistic FROM rubric_score WHERE rater_type = 'llm_judge' AND sample_idx = 0"
    )
    human_rows = fetch_all(
        "SELECT generation_id, name, understand, respect, support, explore, ib, epp, de, ie, "
        "naturalness, ritualistic FROM rubric_score WHERE rater_type = 'human'"
    )
    if not judge_rows or not human_rows:
        raise DataUnavailable(
            "judge-vs-human agreement needs both llm_judge and human rubric_score rows; "
            "no human rating has occurred yet (docs/limitations.md §4)"
        )

    judge_by_gen: dict[str, dict[str, int | None]] = {
        r["generation_id"]: dict(r) for r in judge_rows
    }

    human_by_gen: dict[str, list[dict[str, int | None]]] = {}
    for r in human_rows:
        human_by_gen.setdefault(r["generation_id"], []).append(dict(r))

    def _consensus(dim: str, gid: str) -> int | None:
        raters = human_by_gen.get(gid, [])
        vals: list[int] = []
        for r in raters:
            v = r.get(dim)
            if v is not None:
                vals.append(v)
        if not vals:
            return None
        vals.sort()
        mid = len(vals) // 2
        return vals[mid] if len(vals) % 2 else round((vals[mid - 1] + vals[mid]) / 2)

    records: list[dict[str, object]] = []
    for dim in RUBRIC_DIMENSIONS:
        left = {
            gid: (to_quality(dim, v) if (v := row.get(dim)) is not None else None)
            for gid, row in judge_by_gen.items()
        }
        right = {
            gid: (to_quality(dim, v) if (v := _consensus(dim, gid)) is not None else None)
            for gid in human_by_gen
        }
        xs, ys, kept = paired_series(left, right)
        alpha = krippendorff_alpha([xs, ys], metric=Metric.ORDINAL) if xs else math.nan
        rho, _p = spearman_rho(xs, ys) if xs else (math.nan, math.nan)
        status = classify_dimension(alpha, rho, len(kept))
        records.append(
            {
                "dimension": dim,
                "alpha": alpha,
                "rho": rho,
                "n_units": len(kept),
                "status": status.value,
            }
        )
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Figure 5 — judge self-consistency
# ---------------------------------------------------------------------------


def judge_self_consistency_df() -> pd.DataFrame:
    """Inter-sample variance across the 5-sample, temperature-0.7 validation
    subset, per dimension. A plain `groupby().var()`/`.max()-min()` over
    persisted `rubric_score` rows — no test, effect size, or p-value, so it is
    computed directly here rather than through `carelite.stats`.
    """
    rows = fetch_all(
        "SELECT generation_id, sample_idx, name, understand, respect, support, explore, "
        "ib, epp, de, ie, naturalness, ritualistic FROM rubric_score "
        "WHERE rater_type = 'llm_judge' AND sample_idx > 0"
    )
    if not rows:
        raise DataUnavailable(
            "no multi-sample (sample_idx > 0) llm_judge rows — the 5-sample self-consistency "
            "validation run has not happened yet (docs/preregistration.md §9)"
        )
    wide = pd.DataFrame(rows)
    long = wide.melt(
        id_vars=["generation_id", "sample_idx"],
        value_vars=list(RUBRIC_DIMENSIONS),
        var_name="dimension",
        value_name="raw",
    )
    long["quality"] = [
        (to_quality(dim, int(v)) if pd.notna(v) else math.nan)
        for dim, v in zip(long["dimension"], long["raw"], strict=True)
    ]
    long = long.dropna(subset=["quality"])

    per_gen = long.groupby(["dimension", "generation_id"])["quality"].agg(
        n="count", var="var", rng=lambda s: s.max() - s.min()
    )
    per_gen = per_gen[per_gen["n"] >= 2]
    if per_gen.empty:
        raise DataUnavailable("no generation has >= 2 admitted samples on any dimension")

    records: list[dict[str, object]] = []
    for dim, group in per_gen.groupby("dimension"):
        records.append(
            {
                "dimension": dim,
                "n_generations": int(group.shape[0]),
                "mean_variance": float(group["var"].mean()),
                "mean_sd": float(np.sqrt(group["var"]).mean()),
                "mean_range": float(group["rng"].mean()),
                "pct_unanimous": float((group["rng"] == 0).mean()),
                "pct_range_ge_2": float((group["rng"] >= 2).mean()),
            }
        )
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Figure 6 — retrieval quality
# ---------------------------------------------------------------------------


def retrieval_quality_df(ablation_df: pd.DataFrame | None) -> pd.DataFrame:
    """Combine ablation-derived precision panels with a DB-derived CRAG
    fallback-rate-by-stratum panel. `ablation_df` is `ablation_table_df()`'s
    output, or `None` to omit the first two panels entirely.
    """
    panels: list[pd.DataFrame] = []
    if ablation_df is not None and not ablation_df.empty:
        panels.append(
            pd.DataFrame(
                {
                    "panel": "on_domain_precision",
                    "label": ablation_df["label"],
                    "value": ablation_df["on_domain_precision"],
                    "n": ablation_df["n_turns"],
                    "gate": ablation_df["on_domain_precision"].apply(
                        lambda v: None if pd.isna(v) else bool(v > 0.7)
                    ),
                }
            )
        )
        panels.append(
            pd.DataFrame(
                {
                    "panel": "off_domain_rejection_rate",
                    "label": ablation_df["label"],
                    "value": ablation_df["off_domain_rejection_rate"],
                    "n": ablation_df["n_turns"],
                    "gate": None,
                }
            )
        )

    rows = fetch_all(
        "SELECT s.equity_stratum, t.fell_back_to_b "
        "FROM generation g "
        "JOIN scenario s ON s.scenario_id = g.scenario_id "
        "JOIN retrieval_trace t ON t.generation_id = g.generation_id "
        "WHERE g.condition = 'C'"
    )
    if rows:
        df = pd.DataFrame(rows)
        df["stratum"] = df["equity_stratum"].map({True: "equity", False: "non_equity"})
        grouped = df.groupby("stratum")["fell_back_to_b"].agg(value="mean", n="count").reset_index()
        panels.append(
            pd.DataFrame(
                {
                    "panel": "fallback_rate",
                    "label": grouped["stratum"],
                    "value": grouped["value"],
                    "n": grouped["n"],
                    "gate": None,
                }
            )
        )

    if not panels:
        raise DataUnavailable(
            "no ablation data and no condition-C retrieval_trace rows — nothing to plot"
        )
    return pd.concat(panels, ignore_index=True)


# ---------------------------------------------------------------------------
# Figure 7 — equity subgroup
# ---------------------------------------------------------------------------

_EQUITY_MEASURES: tuple[str, ...] = (
    "nurse_composite",
    "four_habits_composite",
    "naturalness",
    "ritualistic",
)


def equity_subgroup_df(
    long: pd.DataFrame,
    *,
    dims: tuple[str, ...] = _EQUITY_MEASURES,
    rater_type: str = "llm_judge",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Build `fig_equity_subgroup`'s input: pre-specified secondary analysis,
    the equity stratum vs. everything else, restricted to {A, B, C}
    (`docs/preregistration.md` §8.4)."""
    if "equity_stratum" not in long.columns:
        raise DataUnavailable("long score frame has no equity_stratum column")

    statuses = _judge_statuses(long)
    records: list[dict[str, object]] = []
    for stratum_name, stratum_value in (("equity", True), ("non_equity", False)):
        subset = long[long["equity_stratum"] == stratum_value]
        if subset.empty:
            continue
        friedman = {
            f.measure_key: f
            for f in friedman_across_conditions(
                subset, dimensions=dims, rater_type=rater_type, statuses=statuses
            )
        }
        for dim in dims:
            m = measure(dim)
            cells = cell_means(subset, m)
            cells = cells[cells["rater_type"] == rater_type]
            confirmatory_dim = stratum_name == "equity" and friedman[dim].label.is_confirmatory
            for condition, group in cells.groupby("condition", observed=True):
                if str(condition) not in _FRIEDMAN_CONDITION_STRS:
                    continue
                ci = _mean_ci(group["value"].to_numpy(), n_boot=n_boot, seed=seed)
                records.append(
                    {
                        "condition": str(condition),
                        "dimension": dim,
                        "stratum": stratum_name,
                        "mean": float(np.mean(group["value"])),
                        "ci_lo": ci.low,
                        "ci_hi": ci.high,
                        "n": int(group.shape[0]),
                        "confirmatory": bool(confirmatory_dim),
                    }
                )
    if not records:
        raise DataUnavailable("no equity/non-equity cells available for the requested measures")
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Figure 8 — negative control B vs D
# ---------------------------------------------------------------------------


def negative_control_df(
    long: pd.DataFrame,
    *,
    rater_type: str = "llm_judge",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Build `fig_negative_control`'s input: B vs. D means per dimension, with
    the pre-specified composite outcome (§4 outcome 7) flagged confirmatory
    via `carelite.stats.evidence.label_for` and everything else exploratory."""
    statuses = _judge_statuses(long)
    scope = RaterScope.from_rater_types([rater_type])
    dims = (*RUBRIC_DIMENSIONS, "nurse_composite", "four_habits_composite")

    records: list[dict[str, object]] = []
    for dim in dims:
        m = MEASURES[dim] if dim in MEASURES else measure(dim)
        cells = cell_means(long, m)
        cells = cells[cells["rater_type"] == rater_type]
        matrix = paired_matrix(cells, (Condition.B, Condition.D), rater_type=rater_type)
        label = label_for(
            m,
            prespecified=(dim == "nurse_composite"),
            rater_scope=scope,
            statuses=statuses,
        )
        for cond_enum, col in ((Condition.B, "B"), (Condition.D, "D")):
            group = cells[cells["condition"] == str(cond_enum)]
            values = (
                matrix[col].to_numpy()
                if col in matrix.columns and not matrix.empty
                else group["value"].to_numpy()
            )
            ci = _mean_ci(values, n_boot=n_boot, seed=seed)
            records.append(
                {
                    "dimension": dim,
                    "condition": col,
                    "mean": float(np.mean(values)) if values.size else math.nan,
                    "ci_lo": ci.low,
                    "ci_hi": ci.high,
                    "n": int(values.size),
                    "confirmatory": bool(label.is_confirmatory),
                }
            )
    if not records:
        raise DataUnavailable("no B/D cells available for the negative-control check")
    return pd.DataFrame.from_records(records)
