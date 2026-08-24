"""Fixture-driven tests for every figure function in `carelite.viz.figures`.

No database, no model: every test here drives a figure with a hand-built
DataFrame from `conftest.py` and checks the result, per the lane brief
("Each figure is a function taking a DataFrame and returning a Figure, so
tests can drive them with fixtures and no database").
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from carelite.viz.figures import (
    _dim_label,
    fig_ablation_table,
    fig_effect_sizes,
    fig_equity_subgroup,
    fig_judge_agreement,
    fig_judge_consistency,
    fig_negative_control,
    fig_retrieval_quality,
    fig_rubric_scores,
)
from carelite.viz.style import save_figure


def _footer_text(fig: Figure) -> str:
    return "\n".join(t.get_text() for t in fig.texts)


def _assert_saves_cleanly(fig: Figure, name: str, tmp_path: Path) -> None:
    paths = save_figure(fig, name, tmp_path)
    assert len(paths) == 2
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


# ---------------------------------------------------------------------------
# Figure 1 — headline rubric scores
# ---------------------------------------------------------------------------


def test_fig_rubric_scores_returns_figure_with_provenance(
    rubric_scores_fixture, tmp_path: Path
) -> None:
    fig = fig_rubric_scores(rubric_scores_fixture)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 11  # 9 adherence panels + 2 naturalness/ritual panels
    footer = _footer_text(fig)
    assert "n =" in footer
    assert "test:" in footer
    _assert_saves_cleanly(fig, "rubric_scores", tmp_path)


def test_fig_rubric_scores_names_reverse_coding_in_the_figure(rubric_scores_fixture) -> None:
    fig = fig_rubric_scores(rubric_scores_fixture)
    all_text = _footer_text(fig) + " ".join(t.get_text() for ax in fig.axes for t in [ax.title])
    assert "reverse" in all_text.lower() or "REVERSE" in all_text


def test_fig_rubric_scores_handles_undefined_ci_without_raising(
    rubric_scores_fixture_with_missing_ci,
) -> None:
    fig = fig_rubric_scores(rubric_scores_fixture_with_missing_ci)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Figure 2 — effect size forest plot
# ---------------------------------------------------------------------------


def test_fig_effect_sizes_returns_figure_ordered_by_effect(
    effect_sizes_fixture, tmp_path: Path
) -> None:
    fig = fig_effect_sizes(effect_sizes_fixture)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    ylabels = [lbl.get_text() for lbl in ax.get_yticklabels()]
    assert len(ylabels) == len(effect_sizes_fixture)
    # ordered ascending by effect (bottom-to-top plotting convention)
    sorted_labels = (
        effect_sizes_fixture.assign(
            label=effect_sizes_fixture["dimension"].map(_dim_label)
            + "  :  "
            + effect_sizes_fixture["comparison"]
        )
        .sort_values("effect")["label"]
        .tolist()
    )
    assert ylabels == sorted_labels
    footer = _footer_text(fig)
    assert "n =" in footer or "scenarios" in footer
    _assert_saves_cleanly(fig, "effect_sizes", tmp_path)


def test_fig_effect_sizes_marks_exploratory_rows_distinctly(effect_sizes_fixture) -> None:
    fig = fig_effect_sizes(effect_sizes_fixture)
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    legend_text = " ".join(t.get_text() for t in legend.get_texts())
    assert "confirmatory" in legend_text.lower()
    assert "exploratory" in legend_text.lower()


# ---------------------------------------------------------------------------
# Figure 3 — ablation table
# ---------------------------------------------------------------------------


def test_fig_ablation_table_renders_every_row(ablation_fixture, tmp_path: Path) -> None:
    fig = fig_ablation_table(ablation_fixture)
    assert isinstance(fig, Figure)
    footer = _footer_text(fig)
    assert "exploratory" in footer.lower() or "EXPLORATORY" in footer
    _assert_saves_cleanly(fig, "ablation_table", tmp_path)


def test_fig_ablation_table_shows_latency_caveat(ablation_fixture) -> None:
    fig = fig_ablation_table(ablation_fixture)
    all_text = _footer_text(fig) + " ".join(t.get_text() for t in fig.texts)
    assert "shared" in all_text.lower() and "daemon" in all_text.lower()


def test_fig_ablation_table_rejects_the_blended_precision_column() -> None:
    # Coordination note: the old blended `context_precision` column must not
    # be silently accepted as a substitute for the on/off-domain split.
    blended = pd.DataFrame(
        [
            {
                "label": "R9",
                "note": "full stack",
                "n_turns": 43,
                "n_scored": 40,
                "mean_retrieved": 4.0,
                "context_precision": 0.55,  # the old, blended column
                "fallback_rate": 0.1,
                "skipped_rate": 0.0,
                "mean_latency_ms": 5000.0,
            }
        ]
    )
    with pytest.raises(KeyError):
        fig_ablation_table(blended)


# ---------------------------------------------------------------------------
# Figure 4 — judge-vs-human agreement
# ---------------------------------------------------------------------------


def test_fig_judge_agreement_labels_exploratory_dimensions(
    judge_agreement_fixture, tmp_path: Path
) -> None:
    fig = fig_judge_agreement(judge_agreement_fixture)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    ylabels = [lbl.get_text() for lbl in ax.get_yticklabels()]
    exploratory_labels = [lbl for lbl in ylabels if "EXPLORATORY" in lbl]
    assert len(exploratory_labels) >= 1
    _assert_saves_cleanly(fig, "judge_agreement", tmp_path)


def test_fig_judge_agreement_survives_nan_coefficients(judge_agreement_fixture) -> None:
    # `ritualistic` in the fixture has NaN alpha/rho (undefined coefficient).
    fig = fig_judge_agreement(judge_agreement_fixture)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Figure 5 — judge self-consistency
# ---------------------------------------------------------------------------


def test_fig_judge_consistency_renders_all_dimensions(
    judge_consistency_fixture, tmp_path: Path
) -> None:
    fig = fig_judge_consistency(judge_consistency_fixture)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert len(ax.get_xticklabels()) == len(judge_consistency_fixture)
    _assert_saves_cleanly(fig, "judge_consistency", tmp_path)


# ---------------------------------------------------------------------------
# Figure 6 — retrieval quality
# ---------------------------------------------------------------------------


def test_fig_retrieval_quality_has_three_panels(retrieval_quality_fixture, tmp_path: Path) -> None:
    fig = fig_retrieval_quality(retrieval_quality_fixture)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
    _assert_saves_cleanly(fig, "retrieval_quality", tmp_path)


def test_fig_retrieval_quality_does_not_plot_bare_latency(retrieval_quality_fixture) -> None:
    # Rule 5: latency is either order-of-magnitude with a caveat, or absent.
    # This figure's contract has no latency column at all — confirm no axis
    # label anywhere claims to plot raw milliseconds.
    fig = fig_retrieval_quality(retrieval_quality_fixture)
    all_labels = " ".join(ax.get_ylabel() + " " + ax.get_title() for ax in fig.axes)
    assert "ms" not in all_labels.lower()


# ---------------------------------------------------------------------------
# Figure 7 — equity subgroup
# ---------------------------------------------------------------------------


def test_fig_equity_subgroup_renders_one_axis_per_dimension(
    equity_subgroup_fixture, tmp_path: Path
) -> None:
    fig = fig_equity_subgroup(equity_subgroup_fixture)
    assert isinstance(fig, Figure)
    n_dims = equity_subgroup_fixture["dimension"].nunique()
    assert len(fig.axes) == n_dims
    footer = _footer_text(fig)
    assert "PRE-SPECIFIED" in footer
    _assert_saves_cleanly(fig, "equity_subgroup", tmp_path)


# ---------------------------------------------------------------------------
# Figure 8 — negative control
# ---------------------------------------------------------------------------


def test_fig_negative_control_flags_overlapping_ci(
    negative_control_fixture, tmp_path: Path
) -> None:
    fig = fig_negative_control(negative_control_fixture)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # the fixture makes `name` overlap and everything else separate; the
    # "no separation" annotation must appear at least once.
    texts = [t.get_text() for t in ax.texts]
    assert any("no separation" in t for t in texts)
    _assert_saves_cleanly(fig, "negative_control", tmp_path)


def test_fig_negative_control_marks_only_composite_as_confirmatory(
    negative_control_fixture,
) -> None:
    fig = fig_negative_control(negative_control_fixture)
    assert isinstance(fig, Figure)
    n_confirmatory = negative_control_fixture[negative_control_fixture["confirmatory"]][
        "dimension"
    ].unique()
    assert list(n_confirmatory) == ["nurse_composite"]
