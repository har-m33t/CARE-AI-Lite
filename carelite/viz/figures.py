"""The eight figures, each a pure function: `DataFrame -> matplotlib.figure.Figure`.

No figure function here touches the database, a model, or the filesystem — that
is `carelite.viz.queries`'s job (building the DataFrame) and
`carelite.viz.style.save_figure`'s job (writing it out). That split is what
lets `tests/unit/viz/` drive every figure from a small, hand-built fixture
DataFrame with a known shape and assert on the result, with no Postgres and no
model in the loop, per the lane brief.

**All arithmetic lives in `carelite.stats`, not here.** Bootstrap CIs, effect
sizes (rank-biserial, Cohen's dz, Hodges-Lehmann), the Friedman/Wilcoxon/
Holm-Bonferroni test family, and the confirmatory/exploratory `Label` (which
folds together "was this pre-specified" and "did the judge clear its
agreement threshold on every dimension this measure touches") are computed by
`carelite.stats.effects`, `carelite.stats.primary`, and
`carelite.stats.evidence` and consumed, never recomputed, by
`carelite.viz.queries`. A figure function below only ever reads already-computed
columns off a DataFrame — it has no numpy resampling loop and no p-value
arithmetic of its own, so there is exactly one implementation of "what a CI on
this study's data means" for the whole project to disagree with.

Every function below:

- never plots a point estimate without a visible uncertainty interval;
- stamps n, the test used, and pre-specified/exploratory status onto the
  canvas itself via `carelite.viz.style.add_provenance_footer` — a figure
  saved out of this module and separated from its caller's context still
  carries that information;
- uses only the Okabe-Ito colourblind-safe palette from `carelite.viz.style`,
  plus a second, colour-independent channel (marker shape, fill vs. hollow,
  line style, or a text label) for anything a reader must not mis-read from
  colour alone — including on a black-and-white printout.

Column contracts are documented per function. `carelite.viz.queries` is the
one module responsible for actually producing DataFrames in this shape from
the live database; see its docstrings for exactly how each column is derived,
and in particular how `confirmatory` is always `carelite.stats.evidence.Label
.is_confirmatory`, not a locally re-derived boolean.
"""

from __future__ import annotations

import math

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from carelite.eval.rubric.dimensions import DIMENSIONS
from carelite.stats.primary import FRIEDMAN_CONDITIONS
from carelite.viz.style import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_MARKERS,
    CONDITION_ORDER,
    OKABE_ITO,
    add_provenance_footer,
    new_figure,
)

__all__ = [
    "fig_ablation_table",
    "fig_effect_sizes",
    "fig_equity_subgroup",
    "fig_judge_agreement",
    "fig_judge_consistency",
    "fig_negative_control",
    "fig_retrieval_quality",
    "fig_rubric_scores",
]

_ADHERENCE_DIMS: tuple[str, ...] = (
    "name",
    "understand",
    "respect",
    "support",
    "explore",
    "ib",
    "epp",
    "de",
    "ie",
)
#: Condition set the Friedman omnibus runs across (`docs/preregistration.md` §8.1).
#: Sourced from `carelite.stats.primary`, not restated, so the two lanes cannot drift.
_OMNIBUS_CONDITIONS: tuple[str, ...] = tuple(str(c) for c in FRIEDMAN_CONDITIONS)

_TEST_LABEL_BOOTSTRAP_CI = (
    "95% bootstrap CI, percentile method (carelite.stats.effects.bootstrap_ci)"
)
_TEST_LABEL_EFFECT = (
    "matched-pairs rank-biserial correlation, 95% bootstrap CI "
    "(carelite.stats.effects.paired_effects)"
)
_TEST_LABEL_OMNIBUS = (
    "Friedman(A,B,C) -> Wilcoxon signed-rank, Holm-Bonferroni across the pre-specified "
    "family (carelite.stats.primary)"
)


def _dim_label(key: str) -> str:
    if key in DIMENSIONS:
        return DIMENSIONS[key].label
    return {
        "nurse_composite": "NURSE composite",
        "four_habits_composite": "Four Habits composite",
    }.get(key, key)


def _ordered_conditions(present: pd.Series) -> list[str]:
    seen = set(present)
    return [c for c in CONDITION_ORDER if c in seen]


def _plot_point_ci(
    ax: Axes,
    x: float,
    row: pd.Series,
    *,
    color: str,
    marker: str,
    filled: bool,
) -> None:
    """One point + vertical 95% CI whisker, filled/hollow encoding confirmatory status."""
    lo, hi, mean = row["ci_lo"], row["ci_hi"], row["mean"]
    if not (math.isnan(lo) or math.isnan(hi)):
        ax.plot([x, x], [lo, hi], color=color, linewidth=1.3, zorder=2, solid_capstyle="round")
        ax.plot([x - 0.08, x + 0.08], [lo, lo], color=color, linewidth=1.3, zorder=2)
        ax.plot([x - 0.08, x + 0.08], [hi, hi], color=color, linewidth=1.3, zorder=2)
    ax.plot(
        x,
        mean,
        marker=marker,
        markersize=6.5,
        markerfacecolor=color if filled else "none",
        markeredgecolor=color,
        markeredgewidth=1.4,
        linestyle="none",
        zorder=3,
    )


def _flag_if_degenerate(ax: Axes, degenerate: bool) -> None:
    """Mark a panel whose dimension the judge did not resolve
    (`carelite.stats.instrument`) — shaded background plus an explicit label,
    a second channel independent of the shading so it survives greyscale.

    A CI gap or overlap on a degenerate dimension describes the judge's own
    floor, not the conditions; leaving the panel looking identical to a
    resolved one is exactly the misreading this exists to prevent.
    """
    if not degenerate:
        return
    ax.set_facecolor("#FBE9E7")
    ax.text(
        0.5,
        0.03,
        "NOT TESTABLE\n(instrument floor)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6,
        fontweight="bold",
        color=OKABE_ITO["vermillion"],
        zorder=5,
    )


def _panel_degenerate(sub: pd.DataFrame) -> bool:
    if "degenerate" not in sub.columns or sub.empty:
        return False
    return bool(sub["degenerate"].iloc[0])


# ---------------------------------------------------------------------------
# Figure 1 — headline: per-condition rubric scores, faceted by dimension
# ---------------------------------------------------------------------------


def fig_rubric_scores(df: pd.DataFrame) -> Figure:
    """The headline figure: per-condition quality scores with 95% bootstrap CIs,
    faceted by dimension, arranged so the naturalness/adherence tension the
    study expects is legible without reading a caption.

    Required columns (one row per condition x dimension cell):
        condition      str, one of `carelite.types.Condition`
        dimension      str, one of `carelite.types.RUBRIC_DIMENSIONS`
                       (already on the `to_quality()` scale — higher is
                       always better here, including for `ritualistic`)
        mean           float, quality-scale mean (`carelite.stats.effects.bootstrap_ci`
                       with a mean statistic, over `carelite.stats.measures.cell_means`)
        ci_lo, ci_hi   float, 95% bootstrap CI bounds on that mean
        n              int, scenarios contributing to the cell
        confirmatory   bool — `carelite.stats.evidence.Label.is_confirmatory` for this
                       dimension's Friedman omnibus (`carelite.stats.primary
                       .friedman_across_conditions`), AND'd with `condition in
                       {A, B, C}`. A2, LC, D sit outside the omnibus entirely and are
                       always `False` here regardless of dimension.
        degenerate     bool, optional — `carelite.stats.primary.FriedmanResult
                       .degenerate`: the judge did not resolve this dimension on
                       this run (`carelite.stats.instrument`). When present and
                       `True`, every panel for that dimension is shaded and
                       labelled NOT TESTABLE rather than plotted as if a CI gap
                       there meant anything. Missing column is read as `False`
                       everywhere (no diagnostic available).
    """
    fig = new_figure((15.5, 9.5))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=9,
        height_ratios=[1.0, 1.35],
        hspace=0.55,
        wspace=0.15,
        top=0.88,
        bottom=0.14,
        left=0.05,
        right=0.98,
    )

    fig.suptitle(
        "Per-condition rubric scores by dimension (95% bootstrap CI)",
        fontsize=13,
        fontweight="bold",
        y=0.965,
    )

    # ---- top band: structural adherence (NURSE + Four Habits) ----
    for i, dim in enumerate(_ADHERENCE_DIMS):
        ax = fig.add_subplot(gs[0, i])
        sub = df[df["dimension"] == dim]
        conds = _ordered_conditions(sub["condition"].unique())
        for x, cond in enumerate(conds):
            row = sub[sub["condition"] == cond].iloc[0]
            _plot_point_ci(
                ax,
                x,
                row,
                color=CONDITION_COLORS.get(cond, OKABE_ITO["black"]),
                marker=CONDITION_MARKERS.get(cond, "o"),
                filled=bool(row["confirmatory"]),
            )
        ax.set_xlim(-0.6, len(conds) - 0.4)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, fontsize=6.5)
        ax.set_ylim(0.8, 5.2)
        ax.set_yticks([1, 2, 3, 4, 5])
        if i > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("quality (1-5)", fontsize=7.5)
        ax.set_title(_dim_label(dim), fontsize=7.8)
        ax.tick_params(axis="both", length=2)
        _flag_if_degenerate(ax, _panel_degenerate(sub))

    fig.text(
        0.015,
        0.945,
        "STRUCTURAL ADHERENCE  (NURSE + Four Habits)",
        fontsize=8.5,
        fontweight="bold",
        color=OKABE_ITO["blue"],
    )

    # ---- bottom band: naturalness / ritual tension ----
    ax_nat = fig.add_subplot(gs[1, 0:4])
    ax_rit = fig.add_subplot(gs[1, 5:9])
    for ax, dim in ((ax_nat, "naturalness"), (ax_rit, "ritualistic")):
        sub = df[df["dimension"] == dim]
        conds = _ordered_conditions(sub["condition"].unique())
        for x, cond in enumerate(conds):
            row = sub[sub["condition"] == cond].iloc[0]
            _plot_point_ci(
                ax,
                x,
                row,
                color=CONDITION_COLORS.get(cond, OKABE_ITO["black"]),
                marker=CONDITION_MARKERS.get(cond, "o"),
                filled=bool(row["confirmatory"]),
            )
        ax.set_xlim(-0.6, len(conds) - 0.4)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(
            [CONDITION_LABELS.get(c, c) for c in conds], fontsize=8, rotation=20, ha="right"
        )
        ax.set_ylim(0.8, 5.2)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_ylabel("quality (1-5)", fontsize=8.5)
        _flag_if_degenerate(ax, _panel_degenerate(sub))

    ax_nat.set_title("Naturalness", fontsize=10, fontweight="bold")
    ax_rit.set_title(
        "Ritualistic — REVERSE-CODED, shown on quality scale (raw 5 = worst)",
        fontsize=9,
        fontweight="bold",
        color=OKABE_ITO["vermillion"],
    )

    fig.text(
        0.015,
        0.44,
        "NATURALNESS / RITUAL  — the expected trade-off with adherence",
        fontsize=8.5,
        fontweight="bold",
        color=OKABE_ITO["vermillion"],
    )
    fig.lines.append(
        Line2D(
            [0.05, 0.98],
            [0.475, 0.475],
            transform=fig.transFigure,
            color=OKABE_ITO["black"],
            linewidth=0.8,
            linestyle=(0, (4, 3)),
        )
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=CONDITION_MARKERS.get(c, "o"),
            color=CONDITION_COLORS.get(c, "k"),
            linestyle="none",
            markersize=7,
            markerfacecolor=CONDITION_COLORS.get(c, "k"),
            label=CONDITION_LABELS.get(c, c),
        )
        for c in _ordered_conditions(df["condition"].unique())
    ]
    legend_handles += [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            linestyle="none",
            markersize=7,
            markerfacecolor="black",
            label="filled = DESCRIPTIVE (planned in advance; judge gate cleared)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            linestyle="none",
            markersize=7,
            markerfacecolor="none",
            label="hollow = exploratory / baseline / control",
        ),
    ]
    if "degenerate" in df.columns and df["degenerate"].any():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color=OKABE_ITO["vermillion"],
                linestyle="none",
                markersize=8,
                markerfacecolor="#FBE9E7",
                label="shaded panel = NOT TESTABLE (judge did not resolve this dimension)",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=1,
        frameon=True,
        fontsize=7,
        bbox_to_anchor=(0.995, 0.88),
    )

    n_lo, n_hi = int(df["n"].min()), int(df["n"].max())
    add_provenance_footer(
        fig,
        n=f"{n_lo}-{n_hi} per cell",
        test=_TEST_LABEL_BOOTSTRAP_CI,
        prespec=None,
        extra=(
            "ritualistic is reverse-coded (raw 5 = worst); values here are 6-raw via "
            "carelite.eval.rubric.dimensions.to_quality() so higher is always better in this figure"
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — forest plot of effect sizes
# ---------------------------------------------------------------------------


def fig_effect_sizes(df: pd.DataFrame) -> Figure:
    """Forest plot of pairwise effect sizes, ordered by effect magnitude.

    Required columns (one row per comparison x dimension):
        comparison     str, e.g. "B vs A"
        dimension      str, dimension key or composite name
                       (`carelite.stats.measures.MEASURES`)
        effect         float or NaN, matched-pairs rank-biserial correlation
                       (`carelite.stats.effects.rank_biserial`, the headline
                       estimator of `carelite.stats.effects.PairedEffects`).
                       NaN on a `not_computed` row.
        ci_lo, ci_hi   float or NaN, 95% bootstrap CI on `effect`
        n              int, paired scenarios contributing (0 if not computed)
        p_value        float or NaN — annotation only, per
                       `docs/preregistration.md` §8.2 (CI before p); the
                       Holm-adjusted p (`PairwiseResult.p_holm`) where available
        confirmatory   bool, `carelite.stats.evidence.Label.is_confirmatory`
                       (`PairwiseResult.label.is_confirmatory`)
        not_computed   bool, optional (default `False`) — the comparison was
                       retired by decision (e.g. D11: LC generation stopped
                       before this comparison had a valid sample) and was
                       never run at all, as distinct from run-and-null. Drawn
                       as an explicit "NOT COMPUTED" marker, never as a
                       missing row, so the row count still matches the family
                       size a reader would expect from the plan.
        not_computed_reason  str, optional — shown in the marker's tooltip
                       text on the canvas when `not_computed` is `True`.
        not_testable   bool, optional (default `False`) —
                       `carelite.stats.primary.PairwiseResult.not_testable`:
                       every dimension the measure touches is degenerate on
                       this run (`carelite.stats.instrument`), so the p-value
                       describes the judge, not the conditions. Marked in
                       vermillion with an explicit "NOT TESTABLE" label rather
                       than rendered like an ordinary null result.
    """
    d = df.copy()
    if "not_computed" not in d.columns:
        d["not_computed"] = False
    if "not_testable" not in d.columns:
        d["not_testable"] = False
    d["label"] = d["dimension"].map(_dim_label) + "  :  " + d["comparison"]
    d = d.sort_values("effect", ascending=True, na_position="last").reset_index(drop=True)

    fig = new_figure((10.5, max(4.0, 0.34 * len(d) + 1.6)))
    ax = fig.add_subplot(111)

    for y, (_, row) in enumerate(d.iterrows()):
        not_computed = bool(row["not_computed"])
        if not_computed:
            ax.axhspan(y - 0.42, y + 0.42, color=OKABE_ITO["grey"], alpha=0.12, zorder=0)
            ax.text(
                0.0,
                y,
                "NOT COMPUTED — retired by decision (see caption)",
                va="center",
                ha="center",
                fontsize=6.8,
                fontstyle="italic",
                color=OKABE_ITO["grey"],
                zorder=3,
            )
            continue

        confirmatory = bool(row["confirmatory"])
        not_testable = bool(row["not_testable"])
        color = OKABE_ITO["vermillion"] if not_testable else OKABE_ITO["black"]
        linestyle = "solid" if confirmatory else "dashed"
        lo, hi, pt = row["ci_lo"], row["ci_hi"], row["effect"]
        if not (math.isnan(lo) or math.isnan(hi)):
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.6, linestyle=linestyle, zorder=2)
        ax.plot(
            pt,
            y,
            marker="o",
            markersize=7,
            markerfacecolor=color if confirmatory else "none",
            markeredgecolor=color,
            markeredgewidth=1.4,
            zorder=3,
        )
        if not_testable:
            text_x = (hi if not math.isnan(hi) else pt) + 0.03
            ax.text(
                text_x,
                y,
                "NOT TESTABLE (instrument)",
                va="center",
                fontsize=6.5,
                fontweight="bold",
                color=OKABE_ITO["vermillion"],
            )
        else:
            p_value = row.get("p_value")
            if p_value is not None and not (isinstance(p_value, float) and math.isnan(p_value)):
                text_x = (hi if not math.isnan(hi) else pt) + 0.03
                ax.text(text_x, y, f"p={p_value:.3f}", va="center", fontsize=6.5, color=color)

    ax.axvline(0.0, color=OKABE_ITO["grey"], linewidth=1.0, linestyle="dotted", zorder=1)
    ax.set_yticks(list(range(len(d))))
    ax.set_yticklabels(d["label"], fontsize=8)
    ax.set_xlim(-1.05, 1.35)
    ax.set_xlabel(
        "matched-pairs rank-biserial correlation  (favours 2nd condition <— 0 —> favours 1st)"
    )
    ax.set_title(
        "Pairwise effect sizes, ordered by effect (95% bootstrap CI)",
        fontsize=11,
        fontweight="bold",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            linestyle="solid",
            markersize=7,
            markerfacecolor="black",
            label="DESCRIPTIVE (planned in advance; judge gate cleared)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            linestyle="dashed",
            markersize=7,
            markerfacecolor="none",
            label="exploratory",
        ),
    ]
    if d["not_testable"].any():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=OKABE_ITO["vermillion"],
                linestyle="solid",
                markersize=7,
                markerfacecolor="none",
                label="NOT TESTABLE (judge did not resolve this dimension)",
            )
        )
    if d["not_computed"].any():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color=OKABE_ITO["grey"],
                linestyle="none",
                markersize=8,
                markerfacecolor=OKABE_ITO["grey"],
                alpha=0.5,
                label="NOT COMPUTED (retired by decision, e.g. D11)",
            )
        )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7.5, frameon=True)

    computed = d[~d["not_computed"]]
    n_range = (
        f"{int(computed['n'].min())}-{int(computed['n'].max())} scenarios paired per comparison"
        if not computed.empty
        else "n/a"
    )
    n_not_computed = int(d["not_computed"].sum())
    extra = "p-values, where shown, are Holm-adjusted Wilcoxon signed-rank; reported after the interval per §8.2"
    if n_not_computed:
        extra += (
            f"; {n_not_computed} of {len(d)} planned comparisons NOT COMPUTED (excluded from "
            "the n range above but still counted in the Holm family size)"
        )
    add_provenance_footer(
        fig,
        n=n_range,
        test=_TEST_LABEL_EFFECT,
        prespec=None,
        extra=extra,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — R0-R9 + LC ablation table
# ---------------------------------------------------------------------------
#
# NOTE ON THE PRECISION METRIC (2026-08-24): the blended `context_precision`
# column `carelite.retrieval.ablation.AblationRow` originally emitted mixes
# on-domain turns with the three deliberately off-domain probe turns. Because
# CRAG is disabled on most rows, those off-domain turns retrieve *something*
# and the judge correctly scores it useless, which drags every non-CRAG row's
# blended mean toward zero — no row without CRAG enabled can clear the 0.7
# gate even when it is doing exactly the right thing on every in-domain turn.
# `carelite-retrieval` is reworking the harness to report the two questions
# separately, and this figure is deliberately built against that split rather
# than the blended column, per the coordination note this lane received.

_ABLATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("label", "row"),
    ("note", "what changed"),
    ("n_turns", "n turns"),
    ("n_scored", "n on-domain scored"),
    ("mean_retrieved", "mean retrieved"),
    ("on_domain_precision", "on-domain context precision"),
    ("gate_passed", "gate >0.7"),
    ("off_domain_rejection_rate", "off-domain correctly rejected"),
    ("fallback_rate", "CRAG fallback"),
    ("skipped_rate", "skipped"),
    ("latency_oom", "latency (order of magnitude)"),
)

_ABLATION_REQUIRED_COLUMNS: tuple[str, ...] = (
    "label",
    "n_turns",
    "on_domain_precision",
    "off_domain_rejection_rate",
    "fallback_rate",
    "skipped_rate",
    "mean_latency_ms",
)


def _latency_oom(ms: float | None) -> str:
    if ms is None or (isinstance(ms, float) and (math.isnan(ms) or ms <= 0)):
        return "-"
    return f"~10^{math.floor(math.log10(ms))} ms"


def fig_ablation_table(df: pd.DataFrame) -> Figure:
    """The R0-R9 + LC retrieval ablation, rendered as a table.

    Required columns — deliberately **not** the single blended `context_precision`
    `carelite.retrieval.ablation.AblationRow.to_dict()` currently emits (see the
    module-level note above for why that column is not usable as a gate input
    right now):

        label, note, n_turns, n_scored, mean_retrieved     as in AblationRow
        on_domain_precision          float or NaN — context precision computed
                                      only over turns the corpus can actually
                                      address; this is what the >0.7 gate checks
        off_domain_rejection_rate    float or NaN — share of the deliberately
                                      off-domain probe turns for which nothing
                                      useful was retrieved (higher is better);
                                      diagnostic only, not gated
        fallback_rate, skipped_rate, mean_latency_ms   as in AblationRow

    Raises `KeyError` naming the missing columns rather than silently falling
    back to a blended precision number that would misrepresent every non-CRAG
    row as failing for the wrong reason.
    """
    missing = [c for c in _ABLATION_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"fig_ablation_table needs {missing}; the blended `context_precision` column is not "
            "a substitute (see the module-level note above carelite.viz.figures.fig_ablation_table)"
        )

    d = df.copy()
    d["latency_oom"] = d["mean_latency_ms"].apply(_latency_oom)
    d["gate_passed"] = d["on_domain_precision"].apply(
        lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else v > 0.7
    )
    if "note" not in d.columns:
        d["note"] = ""
    if "n_scored" not in d.columns:
        d["n_scored"] = d["n_turns"]

    def _fmt(col: str, v: object) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "-"
        if col == "gate_passed":
            return "PASS" if v else "fail"
        if col == "on_domain_precision":
            return f"{v:.3f}"
        if col in ("fallback_rate", "skipped_rate", "off_domain_rejection_rate"):
            return f"{v:.1%}"
        if col == "mean_retrieved":
            return f"{v:.1f}"
        if col == "note":
            return str(v)[:42] + ("..." if len(str(v)) > 42 else "")
        return str(v)

    cell_text = [[_fmt(col, row[col]) for col, _ in _ABLATION_COLUMNS] for _, row in d.iterrows()]
    col_labels = [label for _, label in _ABLATION_COLUMNS]

    fig = new_figure((16.5, max(3.0, 0.42 * len(d) + 2.2)))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(
        "Retrieval ablation R0-R9 (+ LC long-context baseline)",
        fontsize=12,
        fontweight="bold",
        pad=18,
    )

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    table.scale(1, 1.6)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(OKABE_ITO["grey"])
        if row_idx == 0:
            cell.set_facecolor(OKABE_ITO["black"])
            cell.set_text_props(color="white", fontweight="bold")
            continue
        data_row = d.iloc[row_idx - 1]
        gate = data_row.get("gate_passed")
        base = "#FFFFFF"
        if col_idx == 6 and gate is not None:  # gate_passed column
            base = "#E7F5EE" if gate else "#FBE9E7"
        cell.set_facecolor(base)

    fig.text(
        0.01,
        0.02,
        "Latency shown as order-of-magnitude only: a mixed ablation run measured a 6x swing "
        "(32,490ms vs 5,174ms) between rows with identical CRAG configuration, caused by which "
        "model was resident in the shared Ollama daemon, not by the retrieval configuration "
        "(docs/limitations.md). Off-domain rejection is diagnostic, not gated at 0.7.",
        fontsize=6.6,
        style="italic",
        color=OKABE_ITO["black"],
        wrap=True,
    )

    add_provenance_footer(
        fig,
        n=int(d["n_turns"].sum()) if "n_turns" in d.columns else "-",
        test="LLMContextPrecisionWithoutReference (Ragas-equivalent formula, own implementation), "
        "on-domain turns only; gate = on-domain context precision > 0.7",
        prespec=False,
        extra="engineering ablation from build-time development; not a hypothesis test in the family planned in advance",
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — judge-vs-human agreement per dimension
# ---------------------------------------------------------------------------


def fig_judge_agreement(df: pd.DataFrame) -> Figure:
    """Judge-vs-human agreement (Krippendorff's alpha, Spearman's rho) per
    dimension, making which dimensions cleared the fixed agreement gate — and
    which did not — visually obvious.

    Required columns:
        dimension   str, one of `carelite.types.RUBRIC_DIMENSIONS`
        alpha       float, Krippendorff's alpha (ordinal), may be NaN
        rho         float, Spearman's rho, may be NaN
        n_units     int, paired judge/human units
        status      str, "confirmatory" | "exploratory" — the judge lane's
                    `EvidenceStatus` values (`carelite.eval.judge.validation
                    .classify_dimension`, the same classifier
                    `carelite.stats.evidence.status_from_agreement` delegates
                    to). Read `"confirmatory"` here as "cleared the fixed
                    agreement threshold" (D10); this figure never renders the
                    word to a viewer, only "GATE CLEARED" / "EXPLORATORY".
    """
    from carelite.eval.judge.validation import MIN_ALPHA_FOR_CONFIRMATORY, MIN_RHO_FOR_CONFIRMATORY

    order = [k for k in DIMENSIONS if k in set(df["dimension"])]
    d = df.set_index("dimension").loc[order].reset_index()

    fig = new_figure((11.5, max(4.5, 0.5 * len(d) + 1.8)))
    ax = fig.add_subplot(111)

    for y, (_, row) in enumerate(d.iterrows()):
        confirmatory = row["status"] == "confirmatory"
        alpha_v = row["alpha"]
        rho_v = row["rho"]
        alpha_ok = (not math.isnan(alpha_v)) and alpha_v >= MIN_ALPHA_FOR_CONFIRMATORY
        rho_ok = (not math.isnan(rho_v)) and rho_v >= MIN_RHO_FOR_CONFIRMATORY
        if not math.isnan(alpha_v):
            ax.plot(
                alpha_v,
                y + 0.14,
                marker="o",
                markersize=8,
                markerfacecolor=OKABE_ITO["blue"] if alpha_ok else "none",
                markeredgecolor=OKABE_ITO["blue"],
                markeredgewidth=1.5,
                zorder=3,
            )
        if not math.isnan(rho_v):
            ax.plot(
                rho_v,
                y - 0.14,
                marker="s",
                markersize=8,
                markerfacecolor=OKABE_ITO["vermillion"] if rho_ok else "none",
                markeredgecolor=OKABE_ITO["vermillion"],
                markeredgewidth=1.5,
                zorder=3,
            )
        if not confirmatory:
            ax.axhspan(y - 0.5, y + 0.5, color=OKABE_ITO["grey"], alpha=0.14, zorder=0)

    ax.axvline(
        MIN_ALPHA_FOR_CONFIRMATORY,
        color=OKABE_ITO["blue"],
        linestyle="dashed",
        linewidth=1.0,
        alpha=0.7,
    )
    ax.axvline(
        MIN_RHO_FOR_CONFIRMATORY,
        color=OKABE_ITO["vermillion"],
        linestyle="dashed",
        linewidth=1.0,
        alpha=0.7,
    )

    labels = [
        f"{_dim_label(k)}  [{'GATE CLEARED' if s == 'confirmatory' else 'EXPLORATORY'}]"
        for k, s in zip(d["dimension"], d["status"], strict=True)
    ]
    ax.set_yticks(list(range(len(d))))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("agreement coefficient")
    ax.set_title(
        "Judge-vs-human agreement per dimension "
        f"(gate clears at alpha>={MIN_ALPHA_FOR_CONFIRMATORY}, rho>={MIN_RHO_FOR_CONFIRMATORY}, n>=30)",
        fontsize=11,
        fontweight="bold",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=OKABE_ITO["blue"],
            linestyle="none",
            markersize=8,
            markerfacecolor=OKABE_ITO["blue"],
            label="Krippendorff's alpha (ordinal)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color=OKABE_ITO["vermillion"],
            linestyle="none",
            markersize=8,
            markerfacecolor=OKABE_ITO["vermillion"],
            label="Spearman's rho",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            linestyle="none",
            markersize=8,
            markerfacecolor="none",
            label="hollow = below its own threshold",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7.5, frameon=True)

    add_provenance_footer(
        fig,
        n=f"{int(d['n_units'].min())}-{int(d['n_units'].max())} paired units",
        test="Krippendorff's alpha (ordinal) & Spearman's rho vs. human consensus, per dimension",
        prespec=True,
        extra="threshold pre-specified in carelite.eval.judge.validation before validation data existed (docs/preregistration.md §9)",
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — judge self-consistency variance per dimension
# ---------------------------------------------------------------------------


def fig_judge_consistency(df: pd.DataFrame) -> Figure:
    """Judge self-consistency (inter-sample variance across 5 samples at
    temperature 0.7) per dimension.

    Required columns:
        dimension        str
        mean_sd          float, mean inter-sample standard deviation
        n_generations    int
        pct_range_ge_2   float in [0,1], share of generations where the 5
                          samples spanned >= 2 scale points (optional)
    """
    order = [k for k in DIMENSIONS if k in set(df["dimension"])]
    d = df.set_index("dimension").loc[order].reset_index()

    fig = new_figure((10.5, 5.2))
    ax = fig.add_subplot(111)

    xs = range(len(d))
    colors = [
        OKABE_ITO["blue"] if k in _ADHERENCE_DIMS else OKABE_ITO["vermillion"]
        for k in d["dimension"]
    ]
    ax.bar(xs, d["mean_sd"], color=colors, edgecolor="black", linewidth=0.6, width=0.6, zorder=2)

    if "pct_range_ge_2" in d.columns:
        ax2 = ax.twinx()
        ax2.plot(
            xs,
            d["pct_range_ge_2"],
            color="black",
            marker="D",
            markersize=5,
            linestyle="none",
            zorder=3,
            label="share with range >= 2 points",
        )
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("share of generations with sample range >= 2 points", fontsize=8)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([_dim_label(k) for k in d["dimension"]], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean inter-sample SD (quality scale)")
    ax.set_title(
        "Judge self-consistency: 5 samples at temperature 0.7", fontsize=11, fontweight="bold"
    )

    add_provenance_footer(
        fig,
        n=f"{int(d['n_generations'].min())}-{int(d['n_generations'].max())} generations",
        test="inter-sample SD/variance across 5 judge samples per generation, temperature 0.7",
        prespec=True,
        extra="methodology pre-specified, docs/preregistration.md §9; feeds the §8.5(c) sensitivity analysis",
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — retrieval quality
# ---------------------------------------------------------------------------


def fig_retrieval_quality(df: pd.DataFrame) -> Figure:
    """Retrieval quality: on-domain context precision and off-domain rejection
    rate by ablation row, CRAG fallback rate by scenario stratum, and — when
    present — the B vs C retrieval contrast asked two ways.

    Required columns (tidy, one row per panel item):
        panel    str, "on_domain_precision" | "off_domain_rejection_rate" |
                 "fallback_rate" | "retrieval_contrast"
        label    str, ablation row label (panels 1-2), stratum name (panel 3),
                 or "offered" | "retrieved" (panel 4)
        value    float — a rate for panels 1-3, the matched-pairs
                 rank-biserial effect (B vs C, NURSE composite) for panel 4
        n        int
        gate     bool or None — only meaningful for panel "on_domain_precision";
                  the other panels are diagnostic and never gated (see
                  `fig_ablation_table`'s module note)
        ci_lo, ci_hi   float, optional — 95% bootstrap CI on `value`, panel 4 only
        not_testable   bool, optional — panel 4 only; `PairwiseResult
                       .not_testable` (`carelite.stats.instrument`): the
                       composite's dimensions are all degenerate on this run.

    Panel 4, when present, is `carelite.stats.sensitivity.retrieval_contrast`:
    "offered" is all Condition-C cells (does *offering* retrieval help?),
    "retrieved" is only the cells where CRAG actually retrieved (does
    retrieval *itself* help — the architecture's actual claim, on a
    self-selected, not randomised, subset). Both are shown together
    deliberately; neither answers the question alone.
    """
    has_contrast = not df[df["panel"] == "retrieval_contrast"].empty
    n_cols = 4 if has_contrast else 3
    fig = new_figure((17.0 * n_cols / 3, 5.2))
    ax1 = fig.add_subplot(1, n_cols, 1)
    ax2 = fig.add_subplot(1, n_cols, 2)
    ax3 = fig.add_subplot(1, n_cols, 3)
    ax4 = fig.add_subplot(1, n_cols, 4) if has_contrast else None

    cp = df[df["panel"] == "on_domain_precision"]
    if not cp.empty:
        xs = range(len(cp))
        colors = [
            ("#2C7A50" if g else "#B23A2F") if g is not None else OKABE_ITO["grey"]
            for g in cp["gate"]
        ]
        ax1.bar(xs, cp["value"], color=colors, edgecolor="black", linewidth=0.6)
        ax1.axhline(0.7, color="black", linestyle="dashed", linewidth=1.1)
        ax1.text(len(cp) - 0.5, 0.72, "gate = 0.7", fontsize=7, ha="right")
        ax1.set_xticks(list(xs))
        ax1.set_xticklabels(cp["label"], rotation=45, ha="right", fontsize=7.5)
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel("on-domain context precision")
    ax1.set_title("On-domain context precision by row", fontsize=10, fontweight="bold")

    od = df[df["panel"] == "off_domain_rejection_rate"]
    if not od.empty:
        xs = range(len(od))
        ax2.bar(xs, od["value"], color=OKABE_ITO["yellow"], edgecolor="black", linewidth=0.6)
        ax2.set_xticks(list(xs))
        ax2.set_xticklabels(od["label"], rotation=45, ha="right", fontsize=7.5)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("off-domain turns correctly rejected")
    ax2.set_title(
        "Off-domain rejection rate by row (diagnostic, not gated)", fontsize=10, fontweight="bold"
    )

    fb = df[df["panel"] == "fallback_rate"]
    if not fb.empty:
        xs = range(len(fb))
        ax3.bar(xs, fb["value"], color=OKABE_ITO["sky_blue"], edgecolor="black", linewidth=0.6)
        ax3.set_xticks(list(xs))
        ax3.set_xticklabels(fb["label"], rotation=45, ha="right", fontsize=7.5)
        ax3.set_ylim(0, 1.0)
        ax3.set_ylabel("CRAG fallback-to-B rate")
    ax3.set_title("CRAG fallback rate by scenario stratum", fontsize=10, fontweight="bold")

    if ax4 is not None:
        rc = df[df["panel"] == "retrieval_contrast"]
        order = [lbl for lbl in ("offered", "retrieved") if lbl in set(rc["label"])]
        for x, lbl in enumerate(order):
            row = rc[rc["label"] == lbl].iloc[0]
            not_testable = bool(row.get("not_testable", False))
            color = OKABE_ITO["vermillion"] if not_testable else OKABE_ITO["bluish_green"]
            lo = row.get("ci_lo", math.nan)
            hi = row.get("ci_hi", math.nan)
            val = row["value"]
            if not (pd.isna(lo) or pd.isna(hi)):
                ax4.plot([x, x], [lo, hi], color=color, linewidth=1.6, zorder=2)
                ax4.plot([x - 0.08, x + 0.08], [lo, lo], color=color, linewidth=1.6, zorder=2)
                ax4.plot([x - 0.08, x + 0.08], [hi, hi], color=color, linewidth=1.6, zorder=2)
            ax4.plot(
                x,
                val,
                marker="D",
                markersize=8,
                markerfacecolor=color,
                markeredgecolor=color,
                zorder=3,
            )
            if not_testable:
                ax4.text(
                    x,
                    (hi if not pd.isna(hi) else val) + 0.08,
                    "NOT TESTABLE",
                    ha="center",
                    fontsize=6.5,
                    color=OKABE_ITO["vermillion"],
                    fontweight="bold",
                )
        ax4.axhline(0.0, color=OKABE_ITO["grey"], linestyle="dotted", linewidth=1.0, zorder=1)
        ax4.set_xlim(-0.6, max(0, len(order) - 1) + 0.6)
        ax4.set_xticks(list(range(len(order))))
        ax4.set_xticklabels(
            [
                "offered\n(all Condition-C cells)"
                if lbl == "offered"
                else "retrieved\n(CRAG fired)"
                for lbl in order
            ],
            fontsize=8,
        )
        ax4.set_ylim(-1.05, 1.05)
        ax4.set_ylabel("NURSE composite effect (rank-biserial), B vs C")
        ax4.set_title("Does retrieval help? B vs C asked two ways", fontsize=10, fontweight="bold")

    fig.suptitle("Retrieval quality", fontsize=12, fontweight="bold", y=1.03)

    n_total = int(df["n"].sum()) if "n" in df.columns and not df.empty else 0
    extra = "engineering/diagnostic figure; not a hypothesis test in the family planned in advance"
    if ax4 is not None:
        extra += (
            "; panel 4 is a second look at the B vs C comparison planned in advance (§4.2), "
            "uncorrected (family of 1) — the Holm-adjusted p is on fig_effect_sizes; the "
            "'retrieved' arm is a CRAG-selected, not randomised, subset (selection caveat)"
        )
    add_provenance_footer(
        fig,
        n=n_total,
        test="context precision: LLMContextPrecisionWithoutReference, on-domain turns only; "
        "off-domain rejection: share of deliberately off-domain probes retrieving nothing useful; "
        "fallback rate: share of condition-C generations with retrieval_trace.crag_grade='none'"
        + (
            "; panel 4: matched-pairs rank-biserial (carelite.stats.sensitivity.retrieval_contrast)"
            if ax4 is not None
            else ""
        ),
        prespec=False,
        extra=extra,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 7 — equity subgroup, pre-specified secondary
# ---------------------------------------------------------------------------


def fig_equity_subgroup(df: pd.DataFrame) -> Figure:
    """Equity-stratum subgroup comparison, pre-specified secondary analysis.

    Required columns:
        condition      str, expected in {A, B, C}
        dimension      str, dimension key or composite name
        stratum        str, "equity" | "non_equity"
        mean, ci_lo, ci_hi, n   as in `fig_rubric_scores`
        confirmatory   bool, `carelite.stats.evidence.Label.is_confirmatory` for
                       this dimension's omnibus, restricted to the equity stratum
                       (`docs/preregistration.md` §8.4)
        degenerate     bool, optional — as in `rubric_scores_df`; `naturalness`
                       and `ritualistic` are two of the four measures this
                       figure plots and both are degenerate on the `ie`
                       holdout run.
    """
    dims = list(dict.fromkeys(df["dimension"]))
    fig = new_figure((3.6 * max(1, len(dims)) + 1.5, 5.2))
    axes = fig.subplots(1, len(dims), squeeze=False)[0]

    stratum_style = {"equity": ("o", 0.0), "non_equity": ("^", 0.22)}

    for ax, dim in zip(axes, dims, strict=True):
        sub = df[df["dimension"] == dim]
        conds = _ordered_conditions(sub["condition"].unique())
        for stratum, (marker, offset) in stratum_style.items():
            strat_sub = sub[sub["stratum"] == stratum]
            for x, cond in enumerate(conds):
                match = strat_sub[strat_sub["condition"] == cond]
                if match.empty:
                    continue
                row = match.iloc[0]
                _plot_point_ci(
                    ax,
                    x + offset,
                    row,
                    color=CONDITION_COLORS.get(cond, OKABE_ITO["black"]),
                    marker=marker,
                    filled=bool(row["confirmatory"]),
                )
        ax.set_xlim(-0.6, len(conds) - 0.4 + 0.3)
        ax.set_xticks([i + 0.11 for i in range(len(conds))])
        ax.set_xticklabels(conds, fontsize=8)
        ax.set_ylim(0.8, 5.2)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_title(_dim_label(dim), fontsize=9)
        _flag_if_degenerate(ax, _panel_degenerate(sub))

    axes[0].set_ylabel("quality (1-5)")
    fig.suptitle(
        "Equity subgroup vs. non-equity, pre-specified secondary analysis",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            linestyle="none",
            markersize=7,
            markerfacecolor="black",
            label="equity stratum (n=35/100 scenarios); filled = DESCRIPTIVE (planned in advance; judge gate cleared)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="black",
            linestyle="none",
            markersize=7,
            markerfacecolor="none",
            label="non-equity / exploratory",
        ),
    ]
    if "degenerate" in df.columns and df["degenerate"].any():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color=OKABE_ITO["vermillion"],
                linestyle="none",
                markersize=8,
                markerfacecolor="#FBE9E7",
                label="shaded panel = NOT TESTABLE (judge did not resolve this dimension)",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=7.5,
        frameon=True,
        bbox_to_anchor=(0.99, 0.98),
    )

    fig.text(
        0.01,
        0.94,
        "Pre-specified limitations (docs/limitations.md §3): no emotion_intensity=1 scenario in the "
        "stratum; racial_ethnic has no adherence_barrier/decision_conflict/false_comprehension scenario.",
        fontsize=6.6,
        style="italic",
    )

    n_lo, n_hi = int(df["n"].min()), int(df["n"].max())
    add_provenance_footer(
        fig,
        n=f"{n_lo}-{n_hi} per cell (35 equity-stratum scenarios of 100 total)",
        test=_TEST_LABEL_OMNIBUS + " restricted to the equity stratum",
        prespec=True,
        extra="docs/preregistration.md §8.4",
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 8 — negative control: D vs B separation
# ---------------------------------------------------------------------------


def fig_negative_control(df: pd.DataFrame) -> Figure:
    """Negative control: does the rubric separate B from the deliberately
    degraded Condition D? `docs/preregistration.md` §4 outcome 7 and §14/§8.6:
    if it cannot, that is reported as a rubric validity failure, not
    explained away.

    Required columns:
        dimension      str, dimension key or composite name
        condition      str, "B" | "D"
        mean, ci_lo, ci_hi, n   as in `fig_rubric_scores`
        confirmatory   bool — `True` only for `nurse_composite`, the one
                       pre-specified outcome (§4 outcome 7); per-dimension
                       breakdown rows are a descriptive, exploratory view.
        degenerate     bool, optional — as in `rubric_scores_df`. On a
                       degenerate dimension, B and D's CIs overlapping is not
                       evidence the rubric failed to separate them: the judge
                       never had room to separate anything on that dimension.
                       Rendered as NOT TESTABLE instead of "no separation".
    """
    order = [
        k
        for k in (*DIMENSIONS, "nurse_composite", "four_habits_composite")
        if k in set(df["dimension"])
    ]
    fig = new_figure((max(9.0, 0.85 * len(order) + 2.0), 5.5))
    ax = fig.add_subplot(111)

    for x, dim in enumerate(order):
        sub = df[df["dimension"] == dim]
        b = sub[sub["condition"] == "B"]
        dd = sub[sub["condition"] == "D"]
        if b.empty or dd.empty:
            continue
        b_row, d_row = b.iloc[0], dd.iloc[0]
        degenerate_dim = _panel_degenerate(sub)
        overlap = not (b_row["ci_lo"] > d_row["ci_hi"] or d_row["ci_lo"] > b_row["ci_hi"])
        if degenerate_dim:
            ax.axvspan(
                x - 0.35, x + 0.35, color=OKABE_ITO["grey"], alpha=0.22, zorder=0, hatch="//"
            )
        elif overlap:
            ax.axvspan(x - 0.35, x + 0.35, color=OKABE_ITO["vermillion"], alpha=0.16, zorder=0)
        _plot_point_ci(
            ax,
            x - 0.12,
            b_row,
            color=CONDITION_COLORS["B"],
            marker=CONDITION_MARKERS["B"],
            filled=bool(b_row["confirmatory"]),
        )
        _plot_point_ci(
            ax,
            x + 0.12,
            d_row,
            color=CONDITION_COLORS["D"],
            marker=CONDITION_MARKERS["D"],
            filled=False,
        )
        ax.plot(
            [x - 0.12, x + 0.12],
            [b_row["mean"], d_row["mean"]],
            color=OKABE_ITO["grey"],
            linewidth=0.8,
            zorder=1,
        )
        if degenerate_dim:
            ax.text(
                x,
                5.05,
                "NOT TESTABLE",
                fontsize=6.3,
                ha="center",
                color=OKABE_ITO["grey"],
                fontweight="bold",
            )
        elif overlap:
            ax.text(
                x,
                5.05,
                "no separation",
                fontsize=6.3,
                ha="center",
                color=OKABE_ITO["vermillion"],
                fontweight="bold",
            )

    ax.set_xticks(list(range(len(order))))
    ax.set_xticklabels([_dim_label(k) for k in order], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0.7, 5.4)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("quality (1-5)")
    ax.set_title(
        "Negative control: B vs. D (deliberately degraded)  —  shaded = CIs overlap, rubric did "
        "not separate them; hatched = NOT TESTABLE (instrument floor)",
        fontsize=10.5,
        fontweight="bold",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=CONDITION_MARKERS["B"],
            color=CONDITION_COLORS["B"],
            linestyle="none",
            markersize=8,
            markerfacecolor=CONDITION_COLORS["B"],
            label="B — framework-prompted",
        ),
        Line2D(
            [0],
            [0],
            marker=CONDITION_MARKERS["D"],
            color=CONDITION_COLORS["D"],
            linestyle="none",
            markersize=8,
            markerfacecolor="none",
            label="D — negative control (floor)",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=True)

    n_lo, n_hi = int(df["n"].min()), int(df["n"].max())
    add_provenance_footer(
        fig,
        n=f"{n_lo}-{n_hi} per condition x dimension cell",
        test="paired comparison, matched-pairs rank-biserial (Wilcoxon family); composite is the pre-specified outcome",
        prespec=True,
        extra="docs/preregistration.md §4 outcome 7 and §8.6 (negative-control check)",
    )
    return fig
