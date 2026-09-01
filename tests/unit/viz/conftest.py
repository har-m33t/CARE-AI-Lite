"""Fixture DataFrames for the `carelite-viz` lane.

Every fixture here matches a documented column contract in
`carelite.viz.figures` exactly, hand-built with a known shape so a figure
test can assert on the result without a database or a model in the loop —
per the lane brief: "Each figure is a function taking a DataFrame and
returning a Figure, so tests can drive them with fixtures and no database."

Values are deliberately synthetic and mostly arbitrary; what matters is the
*shape* (columns, dtypes, which conditions/dimensions are present) and a few
deliberately chosen values that exercise edge cases: NaN CIs (n=1 cell), a
condition D row plotted hollow, a non-confirmatory dimension, overlapping
B/D confidence intervals (the negative-control "no separation" case).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from carelite.eval.rubric.dimensions import DIMENSIONS
from carelite.types import RUBRIC_DIMENSIONS

ALL_DIMENSIONS = list(RUBRIC_DIMENSIONS)
ALL_CONDITIONS = ["A", "A2", "B", "C", "LC", "D"]
OMNIBUS_CONDITIONS = {"A", "B", "C"}


#: On the `ie` holdout run these two are degenerate (sd 0.49, 0.16): the judge
#: put the vast majority of its scores on a single value. Mirrored in fixtures
#: below so the NOT TESTABLE marking has something real to exercise.
DEGENERATE_DIMENSIONS = {"naturalness", "ritualistic"}


@pytest.fixture
def rubric_scores_fixture() -> pd.DataFrame:
    rows = []
    for dim_i, dim in enumerate(ALL_DIMENSIONS):
        # every third dimension is exploratory (judge validation failed on it)
        dim_confirmatory = dim_i % 3 != 0
        for cond_i, cond in enumerate(ALL_CONDITIONS):
            mean = 2.0 + ((dim_i + cond_i) % 4) * 0.7
            rows.append(
                {
                    "condition": cond,
                    "dimension": dim,
                    "mean": mean,
                    "ci_lo": mean - 0.4,
                    "ci_hi": mean + 0.4,
                    "n": 12 + cond_i,
                    "confirmatory": cond in OMNIBUS_CONDITIONS and dim_confirmatory,
                    "degenerate": dim in DEGENERATE_DIMENSIONS,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def rubric_scores_fixture_with_missing_ci() -> pd.DataFrame:
    """One cell (n=1) has an undefined bootstrap interval — NaN ci_lo/ci_hi."""
    rows = []
    for dim in ALL_DIMENSIONS:
        for cond in ["A", "B", "C"]:
            rows.append(
                {
                    "condition": cond,
                    "dimension": dim,
                    "mean": 3.0,
                    "ci_lo": math.nan if (dim == "naturalness" and cond == "D") else 2.5,
                    "ci_hi": math.nan if (dim == "naturalness" and cond == "D") else 3.5,
                    "n": 1 if (dim == "naturalness" and cond == "D") else 20,
                    "confirmatory": True,
                }
            )
    # add the single-sample D/naturalness cell itself
    rows.append(
        {
            "condition": "D",
            "dimension": "naturalness",
            "mean": 3.0,
            "ci_lo": math.nan,
            "ci_hi": math.nan,
            "n": 1,
            "confirmatory": False,
        }
    )
    return pd.DataFrame(rows)


#: The shape `carelite.viz.queries.effect_sizes_df` returns, in its column order.
EFFECT_SIZE_COLUMNS = [
    "dimension",
    "comparison",
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
]

#: Paraphrases of the two caveats `carelite.stats.primary` attaches to
#: `secondary3_nurse_C_vs_LC` — a fixture, not the authority; the real strings
#: live on `Hypothesis.caveats`. Both open with their claim in a headline
#: sentence, which is the convention the figure's inline mark relies on.
C_VS_LC_CAVEATS = (
    "CONFOUNDED BY SERVING STACK. The LC arm was served by vLLM and condition C by Ollama, "
    "so they differ in artifact, quantisation, sampling defaults, hardware and realised "
    "context pack, and no analysis can separate those from the architecture."
    " | "
    "THE REDUCED FORM OF THE QUESTION (D7). The corpus does not fit the window, so LC is a "
    "fixed query-independent sample rather than the whole corpus, and any selection rule is "
    "itself a form of retrieval."
)

#: A generic retired-by-decision reason. Deliberately not D11's: D13 restored
#: C vs LC, and a fixture that keeps naming the one comparison that used to be
#: retired makes the mechanism look like it belongs to that comparison.
RETIRED_BY_DECISION_REASON = (
    "RETIRED BY DECISION (fixture). Not computed from the data at all, as distinct from "
    "computed and null."
)

_INSTRUMENT_NOTE = (
    "INSTRUMENT-LIMITED: every dimension of this measure is degenerate on this run — "
    "the judge concentrated its scores on one value."
)


@pytest.fixture
def effect_sizes_fixture() -> pd.DataFrame:
    """Mirrors the real eight-hypothesis family as it stands under D13.

    All eight comparisons compute, including `nurse_composite`/`C vs LC`, which
    D11 had retired and D13 restored by generating the full 180-cell LC arm
    under vLLM. That row carries the two caveats D13 attached to it, and the
    two `naturalness`/`ritualistic` rows are instrument-limited
    (`not_testable`) the way they are on the `ie` holdout run.

    Every number here is invented. The real C-vs-LC effect does not exist yet.
    """
    comparisons = [
        ("nurse_composite", "A vs B", 0.62, 0.41, 0.79, 60, 0.0012, True, False, "", False, "", ""),
        (
            "four_habits_composite",
            "A vs B",
            0.55,
            0.30,
            0.74,
            60,
            0.004,
            True,
            False,
            "",
            False,
            "",
            "",
        ),
        ("nurse_composite", "B vs C", 0.18, -0.05, 0.39, 58, 0.11, True, False, "", False, "", ""),
        (
            "nurse_composite",
            "C vs LC",
            0.29,
            0.02,
            0.52,
            60,
            0.048,
            False,
            False,
            "",
            False,
            "",
            C_VS_LC_CAVEATS,
        ),
        (
            "naturalness",
            "A vs B",
            0.47,
            0.20,
            0.68,
            60,
            0.006,
            True,
            False,
            "",
            True,
            _INSTRUMENT_NOTE,
            "",
        ),
        (
            "ritualistic",
            "A vs B",
            0.51,
            0.25,
            0.71,
            60,
            0.003,
            True,
            False,
            "",
            True,
            _INSTRUMENT_NOTE,
            "",
        ),
        ("nurse_composite", "A vs A2", 0.04, -0.22, 0.29, 59, 0.81, True, False, "", False, "", ""),
        (
            "nurse_composite",
            "B vs D",
            0.71,
            0.52,
            0.86,
            60,
            0.0001,
            True,
            False,
            "",
            False,
            "",
            "",
        ),
        # exploratory extras: not in the family planned in advance
        ("de", "A vs C", 0.22, -0.10, 0.48, 55, math.nan, False, False, "", False, "", ""),
        ("epp", "B vs LC", -0.05, -0.30, 0.20, 55, math.nan, False, False, "", False, "", ""),
    ]
    return pd.DataFrame(comparisons, columns=EFFECT_SIZE_COLUMNS)


@pytest.fixture
def effect_sizes_fixture_with_not_computed_row(
    effect_sizes_fixture: pd.DataFrame,
) -> pd.DataFrame:
    """`effect_sizes_fixture` plus one comparison retired by decision.

    No hypothesis carries a `not_computable_reason` under D13, so nothing in
    the live family exercises the NOT COMPUTED rendering any more. It is still
    needed — a comparison with no paired data must render as an explicit marker
    rather than vanish (D12) — so the row is supplied here, deliberately as an
    anonymous comparison, so the test does not encode which decision retired
    what this month.
    """
    retired = pd.DataFrame(
        [
            (
                "nurse_composite",
                "B vs LC",
                math.nan,
                math.nan,
                math.nan,
                0,
                math.nan,
                False,
                True,
                RETIRED_BY_DECISION_REASON,
                False,
                "",
                "",
            )
        ],
        columns=EFFECT_SIZE_COLUMNS,
    )
    return pd.concat([effect_sizes_fixture, retired], ignore_index=True)


@pytest.fixture
def effect_sizes_fixture_caveated_and_not_computed(
    effect_sizes_fixture: pd.DataFrame,
) -> pd.DataFrame:
    """The live state of the database while the LC arm is still being scored.

    `secondary3_nurse_C_vs_LC` is restored to the family by D13 and carries its
    two caveats, but the 180 vLLM cells have not been scored yet, so the
    comparison has no paired data and no effect estimate. Both marks apply to
    the same row at once, and the figure has to render that without either one
    swallowing the other.
    """
    df = effect_sizes_fixture.copy()
    row = df["comparison"] == "C vs LC"
    df.loc[row, ["effect", "ci_lo", "ci_hi", "p_value"]] = math.nan
    df.loc[row, "n"] = 0
    df.loc[row, "not_computed"] = True
    df.loc[row, "not_computed_reason"] = "no paired data available for this comparison on this run"
    return df


@pytest.fixture
def ablation_fixture() -> pd.DataFrame:
    labels = [f"R{i}" for i in range(10)] + ["LC"]
    rows = []
    for i, label in enumerate(labels):
        is_lc = label == "LC"
        rows.append(
            {
                "label": label,
                "note": f"row {label}: incremental component {i}"
                if not is_lc
                else "long-context baseline",
                "n_turns": 43,
                "n_scored": 0 if is_lc else 40,
                "mean_retrieved": math.nan if is_lc else 4.0 + (i % 3),
                "on_domain_precision": math.nan if is_lc else round(0.4 + i * 0.03, 3),
                "off_domain_rejection_rate": math.nan if is_lc else round(0.2 + i * 0.06, 3),
                "fallback_rate": 0.0 if is_lc else round(max(0.0, 0.3 - i * 0.02), 3),
                "skipped_rate": 0.0,
                "mean_latency_ms": 32_490.0
                if label == "R8"
                else (5_174.0 if label == "R9" else 900.0 + i * 50),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def judge_agreement_fixture() -> pd.DataFrame:
    rows = []
    for i, dim in enumerate(ALL_DIMENSIONS):
        if dim == "naturalness":
            alpha, rho, n_units, status = 0.31, 0.28, 32, "exploratory"
        elif dim == "ritualistic":
            alpha, rho, n_units, status = math.nan, math.nan, 5, "exploratory"
        else:
            alpha, rho, n_units, status = 0.72 + i * 0.005, 0.6 + i * 0.01, 40, "confirmatory"
        rows.append(
            {"dimension": dim, "alpha": alpha, "rho": rho, "n_units": n_units, "status": status}
        )
    return pd.DataFrame(rows)


@pytest.fixture
def judge_consistency_fixture() -> pd.DataFrame:
    rows = []
    for i, dim in enumerate(ALL_DIMENSIONS):
        rows.append(
            {
                "dimension": dim,
                "mean_variance": 0.1 + 0.05 * i,
                "mean_sd": round(math.sqrt(0.1 + 0.05 * i), 3),
                "n_generations": 30,
                "pct_range_ge_2": round(0.05 + 0.02 * i, 3),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def retrieval_quality_fixture() -> pd.DataFrame:
    labels = [f"R{i}" for i in range(10)]
    precision_rows = pd.DataFrame(
        {
            "panel": "on_domain_precision",
            "label": labels,
            "value": [0.4 + i * 0.035 for i in range(10)],
            "n": [40] * 10,
            "gate": [(0.4 + i * 0.035) > 0.7 for i in range(10)],
        }
    )
    off_domain_rows = pd.DataFrame(
        {
            "panel": "off_domain_rejection_rate",
            "label": labels,
            "value": [0.2 + i * 0.06 for i in range(10)],
            "n": [3] * 10,
            "gate": [None] * 10,
        }
    )
    fallback_rows = pd.DataFrame(
        {
            "panel": "fallback_rate",
            "label": ["equity", "non_equity"],
            "value": [0.22, 0.15],
            "n": [35, 65],
            "gate": [None, None],
        }
    )
    return pd.concat([precision_rows, off_domain_rows, fallback_rows], ignore_index=True)


@pytest.fixture
def retrieval_quality_with_contrast_fixture(
    retrieval_quality_fixture: pd.DataFrame,
) -> pd.DataFrame:
    """`retrieval_quality_fixture` plus the panel `carelite.stats.sensitivity
    .retrieval_contrast` feeds: B vs C on the NURSE composite, asked as
    "offered" (all Condition-C cells) and "retrieved" (CRAG-fired cells
    only). `retrieved` is marked `not_testable` here purely to exercise that
    rendering path — it is not a claim about the real run.
    """
    contrast_rows = pd.DataFrame(
        {
            "panel": ["retrieval_contrast", "retrieval_contrast"],
            "label": ["offered", "retrieved"],
            "value": [0.18, 0.34],
            "n": [58, 39],
            "gate": [None, None],
            "ci_lo": [-0.05, 0.02],
            "ci_hi": [0.39, 0.61],
            "not_testable": [False, True],
        }
    )
    return pd.concat([retrieval_quality_fixture, contrast_rows], ignore_index=True)


@pytest.fixture
def equity_subgroup_fixture() -> pd.DataFrame:
    dims = ["nurse_composite", "four_habits_composite", "naturalness", "ritualistic"]
    rows = []
    for dim in dims:
        for stratum in ("equity", "non_equity"):
            for cond in ("A", "B", "C"):
                mean = 3.0 if stratum == "equity" else 3.4
                rows.append(
                    {
                        "condition": cond,
                        "dimension": dim,
                        "stratum": stratum,
                        "mean": mean,
                        "ci_lo": mean - 0.5,
                        "ci_hi": mean + 0.5,
                        "n": 12 if stratum == "equity" else 22,
                        "confirmatory": stratum == "equity",
                        "degenerate": dim in DEGENERATE_DIMENSIONS,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def negative_control_fixture() -> pd.DataFrame:
    rows = []
    for dim in [*ALL_DIMENSIONS, "nurse_composite", "four_habits_composite"]:
        # deliberately give one dimension (name) fully overlapping CIs — the
        # "rubric did not separate them" case the figure needs to flag.
        if dim == "name":
            b_mean, d_mean = 3.2, 3.1
        else:
            b_mean, d_mean = 4.1, 2.3
        degenerate = dim in DEGENERATE_DIMENSIONS
        rows.append(
            {
                "dimension": dim,
                "condition": "B",
                "mean": b_mean,
                "ci_lo": b_mean - 0.4,
                "ci_hi": b_mean + 0.4,
                "n": 60,
                "confirmatory": dim == "nurse_composite",
                "degenerate": degenerate,
            }
        )
        rows.append(
            {
                "dimension": dim,
                "condition": "D",
                "mean": d_mean,
                "ci_lo": d_mean - 0.4,
                "ci_hi": d_mean + 0.4,
                "n": 60,
                "confirmatory": False,
                "degenerate": degenerate,
            }
        )
    return pd.DataFrame(rows)


assert set(DIMENSIONS) == set(ALL_DIMENSIONS)  # sanity: fixtures track the real rubric
