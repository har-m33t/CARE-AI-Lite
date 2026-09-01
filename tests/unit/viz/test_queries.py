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

from carelite.stats.primary import CONFIRMATORY_FAMILY, Hypothesis
from carelite.types import RUBRIC_DIMENSIONS, Condition
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
    # docs/preregistration.md §3-4: primary + 7 secondary = 8 rows, ALWAYS.
    # All eight are computed under D13, which generated the full 180-cell LC
    # arm and so restored secondary3_nurse_C_vs_LC to the family. D11 had
    # retired that comparison and this lane rendered it NOT COMPUTED; the row
    # count did not change then and does not change now, because the frame is
    # always one row per planned hypothesis whether or not it ran.
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
        "caveats",
    } <= set(df.columns)
    computed = df[~df["not_computed"]]
    assert len(computed) == 8
    assert computed["effect"].between(-1.0, 1.0).all()


def test_effect_sizes_df_computes_c_vs_lc_under_d13(synthetic_long: pd.DataFrame) -> None:
    """D13 restored secondary outcome 3. It is a computed row, not a marker."""
    df = effect_sizes_df(synthetic_long, n_boot=200)
    rows = df[df["comparison"] == "C vs LC"]
    assert len(rows) == 1
    row = rows.iloc[0]
    assert bool(row["not_computed"]) is False
    assert not math.isnan(row["effect"])
    assert int(row["n"]) > 0


def test_effect_sizes_df_carries_the_c_vs_lc_caveats(synthetic_long: pd.DataFrame) -> None:
    """Both D13 caveats reach the frame, so the figure can mark the row.

    Asserted on substance rather than wording: the serving-stack confound and
    the D7 reduced-form qualification are the two things a reader of this
    comparison must not be without, and `carelite.stats.primary.Hypothesis
    .caveats` is where they are written once.
    """
    df = effect_sizes_df(synthetic_long, n_boot=200)
    caveats = df[df["comparison"] == "C vs LC"].iloc[0]["caveats"]
    assert caveats
    lowered = caveats.lower()
    assert "confounded by serving stack" in lowered
    assert "vllm" in lowered and "ollama" in lowered
    assert "d7" in lowered
    # No other comparison in the family carries one, so the mark means something.
    others = df[df["comparison"] != "C vs LC"]
    assert not others["caveats"].astype(bool).any()


def test_effect_sizes_df_marks_a_hypothesis_retired_by_decision_not_computed(
    synthetic_long: pd.DataFrame,
) -> None:
    """The not-computed mechanism itself, independent of which comparison uses it.

    D11 retired `secondary3_nurse_C_vs_LC` and D13 restored it; no hypothesis
    carries a `not_computable_reason` right now. The machinery must stay tested
    anyway — the next decision of that shape must not have to rebuild it — so
    this drives it with a hypothesis constructed here rather than with whichever
    comparison happens to be retired.
    """
    retired = Hypothesis(
        key="test_only_retired",
        measure_key="nurse_composite",
        left=Condition.B,
        right=Condition.C,
        expected_higher=Condition.C,
        description="Fixture-only hypothesis exercising the retired-by-decision path.",
        not_computable_reason="RETIRED BY DECISION in this test; never computed from the data.",
    )
    df = effect_sizes_df(synthetic_long, hypotheses=(retired,), n_boot=200)
    assert len(df) == 1
    row = df.iloc[0]
    assert bool(row["not_computed"]) is True
    assert "RETIRED BY DECISION" in row["not_computed_reason"]
    assert math.isnan(row["effect"])
    assert math.isnan(row["p_value"])
    assert int(row["n"]) == 0
    # A retired-by-decision comparison is not additionally "confirmatory".
    assert bool(row["confirmatory"]) is False


def test_effect_sizes_df_marks_a_comparison_with_no_paired_data_not_computed(
    synthetic_long: pd.DataFrame,
) -> None:
    """The other not-computed branch: planned, not retired, and no pairs exist.

    A comparison the run simply did not produce must still occupy its row, for
    the reason D12 gave for gate-blocked generations — a silently missing row is
    indistinguishable from a comparison nobody planned.
    """
    without_d = synthetic_long[synthetic_long["condition"] != "D"]
    b_vs_d = next(h for h in CONFIRMATORY_FAMILY if h.key == "secondary7_nurse_B_vs_D")
    df = effect_sizes_df(without_d, hypotheses=(b_vs_d,), n_boot=200)
    assert len(df) == 1
    row = df.iloc[0]
    assert bool(row["not_computed"]) is True
    assert "no paired data" in row["not_computed_reason"]
    assert "retired" not in row["not_computed_reason"].lower()
    assert math.isnan(row["effect"])


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
