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


# ---------------------------------------------------------------------------
# effect_sizes_df
# ---------------------------------------------------------------------------


def test_effect_sizes_df_covers_the_confirmatory_family(synthetic_long: pd.DataFrame) -> None:
    df = effect_sizes_df(synthetic_long, n_boot=200)
    assert len(df) == 8  # docs/preregistration.md §3-4: primary + 7 secondary
    assert {
        "comparison",
        "dimension",
        "effect",
        "ci_lo",
        "ci_hi",
        "n",
        "p_value",
        "confirmatory",
    } <= set(df.columns)
    assert df["effect"].between(-1.0, 1.0).all()


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


# ---------------------------------------------------------------------------
# negative_control_df
# ---------------------------------------------------------------------------


def test_negative_control_df_only_composite_confirmatory(synthetic_long: pd.DataFrame) -> None:
    df = negative_control_df(synthetic_long, n_boot=200)
    assert set(df["condition"]) == {"B", "D"}
    confirmatory_dims = set(df[df["confirmatory"]]["dimension"])
    assert confirmatory_dims <= {"nurse_composite"}


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
