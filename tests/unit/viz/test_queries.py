"""Tests for the pure (non-database) parts of `carelite.viz.queries`.

Functions that take a `long` score DataFrame as a parameter (`rubric_scores_df`,
`effect_sizes_df`, `equity_subgroup_df`, `negative_control_df`) are exercised
here against a small hand-built long-format frame — no Postgres required,
since the database round trip lives entirely in `load_long_scores` and the
handful of functions that call `carelite.db.connection.fetch_all` directly
(`judge_agreement_df`, `judge_self_consistency_df`, and the fallback-rate half
of `retrieval_quality_df`), which are covered separately under
`@pytest.mark.db` and excluded from `make check` by design.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from carelite.types import RUBRIC_DIMENSIONS
from carelite.viz.queries import (
    DataUnavailable,
    _judge_statuses,
    _mean_ci,
    ablation_table_df,
    effect_sizes_df,
    equity_subgroup_df,
    negative_control_df,
    retrieval_quality_df,
    rubric_scores_df,
)

CONDITIONS = ("A", "A2", "B", "C", "LC", "D")


def _synthetic_long_df(n_scenarios: int = 8) -> pd.DataFrame:
    """A small, fully-crossed long-format frame matching
    `carelite.stats.measures.LONG_COLUMNS`, plus `equity_stratum`.

    Raw scores are a deterministic function of scenario/condition/dimension so
    conditions are not identical (bootstrap CIs and effect sizes come out
    non-degenerate) but every scenario has every condition (paired tests have
    something to pair).
    """
    rng = np.random.default_rng(0)
    rows = []
    condition_shift = {"A": 0, "A2": 0, "B": 1, "C": 1, "LC": 1, "D": -1}
    for s in range(n_scenarios):
        scenario_id = f"sc-{s:03d}"
        equity = s % 3 == 0
        for cond in CONDITIONS:
            for sample_idx in range(3):
                for dim in RUBRIC_DIMENSIONS:
                    base = 3 + condition_shift[cond]
                    noise = rng.integers(-1, 2)
                    raw = int(min(5, max(1, base + noise)))
                    rows.append(
                        {
                            "generation_id": f"gen-{scenario_id}-{cond}-{sample_idx}",
                            "scenario_id": scenario_id,
                            "condition": cond,
                            "sample_idx": sample_idx,
                            "rater_type": "llm_judge",
                            "rater_id": "judge-1",
                            "equity_stratum": equity,
                            "challenge_type": "adherence_barrier",
                            "dimension": dim,
                            "raw": raw,
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_long() -> pd.DataFrame:
    return _synthetic_long_df()


@pytest.fixture(scope="module")
def synthetic_long_with_degenerate_dimension() -> pd.DataFrame:
    """`_synthetic_long_df()` with `naturalness` pinned to a single value —
    sd 0 on the `to_quality()` scale, below `carelite.stats.instrument
    .MIN_SD` — so `carelite.stats.instrument.classify` calls it DEGENERATE,
    mirroring `naturalness`/`ritualistic` on the real `ie` holdout run.
    """
    df = _synthetic_long_df().copy()
    df.loc[df["dimension"] == "naturalness", "raw"] = 3
    return df


# ---------------------------------------------------------------------------
# _mean_ci
# ---------------------------------------------------------------------------


def test_mean_ci_empty_is_all_nan() -> None:
    ci = _mean_ci(np.array([]))
    assert math.isnan(ci.low) and math.isnan(ci.high) and ci.n_units == 0


def test_mean_ci_single_value_collapses_to_a_point() -> None:
    # A single unit resampled with replacement is always itself: the interval
    # is degenerate (low == high == the value), not NaN. carelite.stats.effects
    # .bootstrap_ci only returns NaN bounds for zero units or a statistic that
    # comes back non-finite on every replicate.
    ci = _mean_ci(np.array([3.0]))
    assert ci.low == pytest.approx(3.0)
    assert ci.high == pytest.approx(3.0)


def test_mean_ci_brackets_the_mean_for_varying_data() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ci = _mean_ci(values, n_boot=500)
    assert ci.low <= float(np.mean(values)) <= ci.high


# ---------------------------------------------------------------------------
# _judge_statuses
# ---------------------------------------------------------------------------


def test_judge_statuses_is_none_without_human_rows(synthetic_long: pd.DataFrame) -> None:
    assert _judge_statuses(synthetic_long) is None


# ---------------------------------------------------------------------------
# rubric_scores_df
# ---------------------------------------------------------------------------


def test_rubric_scores_df_covers_every_dimension_and_condition(
    synthetic_long: pd.DataFrame,
) -> None:
    df = rubric_scores_df(synthetic_long, n_boot=200)
    assert set(df["dimension"]) == set(RUBRIC_DIMENSIONS)
    assert set(df["condition"]) == set(CONDITIONS)
    assert {"mean", "ci_lo", "ci_hi", "n", "confirmatory"} <= set(df.columns)


def test_rubric_scores_df_confirmatory_only_within_omnibus_conditions(
    synthetic_long: pd.DataFrame,
) -> None:
    df = rubric_scores_df(synthetic_long, n_boot=200)
    outside_omnibus = df[df["condition"].isin(["A2", "LC", "D"])]
    assert not outside_omnibus["confirmatory"].any()


def test_rubric_scores_df_flags_a_degenerate_dimension(
    synthetic_long_with_degenerate_dimension: pd.DataFrame,
) -> None:
    df = rubric_scores_df(synthetic_long_with_degenerate_dimension, n_boot=200)
    assert "degenerate" in df.columns
    naturalness = df[df["dimension"] == "naturalness"]
    assert naturalness["degenerate"].all()
    other = df[df["dimension"] != "naturalness"]
    assert not other["degenerate"].any()


# ---------------------------------------------------------------------------
# effect_sizes_df
# ---------------------------------------------------------------------------


def test_effect_sizes_df_covers_the_confirmatory_family(synthetic_long: pd.DataFrame) -> None:
    df = effect_sizes_df(synthetic_long, n_boot=200)
    # docs/preregistration.md §3-4: primary + 7 secondary = 8 rows, ALWAYS —
    # including secondary3_nurse_C_vs_LC, which D11 retired by decision.
    # run_pairwise returns None for it before touching the data (it is not
    # "computed on 13 scenarios", it is not computed at all), and run_family
    # still counts it toward family_size so the seven comparisons that did
    # run are not made easier to pass by its absence. Dropping the row here
    # would silently shrink the family a reader sees to 7, which is exactly
    # the same hazard D12 flagged for gate-blocked generations — so the row
    # stays, explicitly marked `not_computed`.
    assert len(df) == 8
    assert {
        "comparison",
        "dimension",
        "effect",
        "ci_lo",
        "ci_hi",
        "n",
        "p_value",
        "confirmatory",
        "not_computed",
        "not_computed_reason",
        "not_testable",
        "testability_note",
    } <= set(df.columns)
    computed = df[~df["not_computed"]]
    assert len(computed) == 7
    assert computed["effect"].between(-1.0, 1.0).all()


def test_effect_sizes_df_marks_the_d11_retired_comparison_not_computed(
    synthetic_long: pd.DataFrame,
) -> None:
    df = effect_sizes_df(synthetic_long, n_boot=200)
    retired = df[df["comparison"] == "C vs LC"]
    assert len(retired) == 1
    row = retired.iloc[0]
    assert bool(row["not_computed"]) is True
    assert "D11" in row["not_computed_reason"]
    assert math.isnan(row["effect"])
    assert math.isnan(row["p_value"])
    # A retired-by-decision comparison is not additionally "confirmatory".
    assert bool(row["confirmatory"]) is False


def test_effect_sizes_df_marks_a_degenerate_measure_not_testable(
    synthetic_long_with_degenerate_dimension: pd.DataFrame,
) -> None:
    df = effect_sizes_df(synthetic_long_with_degenerate_dimension, n_boot=200)
    naturalness_row = df[(df["dimension"] == "naturalness") & (df["comparison"] == "A vs B")].iloc[
        0
    ]
    assert bool(naturalness_row["not_testable"]) is True
    assert naturalness_row["testability_note"]
    # An unrelated, discriminating measure is unaffected.
    nurse_row = df[(df["dimension"] == "nurse_composite") & (df["comparison"] == "A vs B")].iloc[0]
    assert bool(nurse_row["not_testable"]) is False


# ---------------------------------------------------------------------------
# equity_subgroup_df
# ---------------------------------------------------------------------------


def test_equity_subgroup_df_has_both_strata(synthetic_long: pd.DataFrame) -> None:
    df = equity_subgroup_df(synthetic_long, n_boot=200)
    assert set(df["stratum"]) == {"equity", "non_equity"}
    assert set(df["condition"]) <= {"A", "B", "C"}


def test_equity_subgroup_df_non_equity_never_confirmatory(synthetic_long: pd.DataFrame) -> None:
    df = equity_subgroup_df(synthetic_long, n_boot=200)
    non_equity = df[df["stratum"] == "non_equity"]
    assert not non_equity["confirmatory"].any()


def test_equity_subgroup_df_flags_a_degenerate_dimension(
    synthetic_long_with_degenerate_dimension: pd.DataFrame,
) -> None:
    df = equity_subgroup_df(synthetic_long_with_degenerate_dimension, n_boot=200)
    assert "degenerate" in df.columns
    assert df[df["dimension"] == "naturalness"]["degenerate"].all()


# ---------------------------------------------------------------------------
# negative_control_df
# ---------------------------------------------------------------------------


def test_negative_control_df_only_composite_confirmatory(synthetic_long: pd.DataFrame) -> None:
    df = negative_control_df(synthetic_long, n_boot=200)
    assert set(df["condition"]) == {"B", "D"}
    confirmatory_dims = set(df[df["confirmatory"]]["dimension"])
    assert confirmatory_dims <= {"nurse_composite"}


def test_negative_control_df_flags_a_degenerate_dimension(
    synthetic_long_with_degenerate_dimension: pd.DataFrame,
) -> None:
    df = negative_control_df(synthetic_long_with_degenerate_dimension, n_boot=200)
    assert "degenerate" in df.columns
    assert df[df["dimension"] == "naturalness"]["degenerate"].all()
    assert not df[df["dimension"] == "understand"]["degenerate"].any()


# ---------------------------------------------------------------------------
# ablation_table_df
# ---------------------------------------------------------------------------


def test_ablation_table_df_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DataUnavailable):
        ablation_table_df(tmp_path / "nonexistent.json")


def test_ablation_table_df_rejects_pre_split_schema(tmp_path: Path) -> None:
    path = tmp_path / "ablation.json"
    path.write_text(json.dumps([{"label": "R9", "context_precision": 0.55, "n_turns": 10}]))
    with pytest.raises(DataUnavailable):
        ablation_table_df(path)


def test_ablation_table_df_loads_valid_split_schema(tmp_path: Path) -> None:
    path = tmp_path / "ablation.json"
    path.write_text(
        json.dumps(
            [
                {
                    "label": "R9",
                    "n_turns": 43,
                    "on_domain_precision": 0.81,
                    "off_domain_rejection_rate": 0.9,
                    "fallback_rate": 0.05,
                    "skipped_rate": 0.0,
                    "mean_latency_ms": 5174.0,
                }
            ]
        )
    )
    df = ablation_table_df(path)
    assert len(df) == 1
    assert df.iloc[0]["on_domain_precision"] == 0.81


def test_ablation_table_df_empty_list_raises(tmp_path: Path) -> None:
    path = tmp_path / "ablation.json"
    path.write_text("[]")
    with pytest.raises(DataUnavailable):
        ablation_table_df(path)


# ---------------------------------------------------------------------------
# retrieval_quality_df
#
# `fetch_all` is monkeypatched rather than left to hit a real socket: this
# lane's tests must run headless with no live Postgres, and an unpatched call
# here would either hang trying to connect or silently depend on whatever
# database happens to be reachable in a given environment.
# ---------------------------------------------------------------------------


def test_retrieval_quality_df_with_no_ablation_and_no_db_rows_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("carelite.viz.queries.fetch_all", lambda *a, **k: [])
    with pytest.raises(DataUnavailable):
        retrieval_quality_df(None)


def test_retrieval_quality_df_builds_fallback_panel_from_db_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_rows = [
        {"equity_stratum": True, "fell_back_to_b": True},
        {"equity_stratum": True, "fell_back_to_b": False},
        {"equity_stratum": False, "fell_back_to_b": False},
    ]
    monkeypatch.setattr("carelite.viz.queries.fetch_all", lambda *a, **k: fake_rows)
    df = retrieval_quality_df(None)
    assert set(df["panel"]) == {"fallback_rate"}
    equity_row = df[df["label"] == "equity"].iloc[0]
    assert equity_row["value"] == pytest.approx(0.5)


def test_retrieval_quality_df_adds_retrieval_contrast_panel_when_long_supplied(
    monkeypatch: pytest.MonkeyPatch, synthetic_long: pd.DataFrame
) -> None:
    # carelite.stats.sensitivity.retrieval_contrast reports B vs C twice
    # (offered vs. retrieved); both belong on the figure when a score frame
    # is available to compute them from, not just the fallback-rate panel.
    c_generations = synthetic_long.loc[
        synthetic_long["condition"] == "C", ["generation_id", "equity_stratum"]
    ].drop_duplicates()
    fake_rows = [
        {
            "generation_id": gid,
            "equity_stratum": eq,
            "fell_back_to_b": bool(i % 3 == 0),  # ~1/3 fallback, mirrors the real run's shape
        }
        for i, (gid, eq) in enumerate(
            zip(c_generations["generation_id"], c_generations["equity_stratum"], strict=True)
        )
    ]
    monkeypatch.setattr("carelite.viz.queries.fetch_all", lambda *a, **k: fake_rows)
    df = retrieval_quality_df(None, synthetic_long, rater_type="llm_judge")
    assert "retrieval_contrast" in set(df["panel"])
    contrast = df[df["panel"] == "retrieval_contrast"]
    assert set(contrast["label"]) <= {"offered", "retrieved"}
    assert {"ci_lo", "ci_hi", "not_testable"} <= set(contrast.columns)


def test_retrieval_quality_df_omits_retrieval_contrast_panel_without_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_rows = [{"generation_id": "g1", "equity_stratum": True, "fell_back_to_b": False}]
    monkeypatch.setattr("carelite.viz.queries.fetch_all", lambda *a, **k: fake_rows)
    df = retrieval_quality_df(None, None)
    assert "retrieval_contrast" not in set(df["panel"])
