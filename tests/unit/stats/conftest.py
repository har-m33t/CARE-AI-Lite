"""Synthetic score frames with properties known in advance.

None of these are drawn from the database. `generation` and `rubric_score` are
empty and held-out generation is blocked in code until OSF registration
(`carelite/generate/runner.py`), so every statistical routine in this package is
verified against data whose right answer was decided before the routine ran.
That is the stronger test anyway: a Friedman implementation that agrees with a
hand-computed example is verified, one that merely runs on real data is not.

`make_long` builds a frame in exactly the shape `carelite.stats.data.load_scores`
returns, so a test that passes here is testing the code path the database feeds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

from carelite.types import RUBRIC_DIMENSIONS, Condition, RaterType


def make_long(
    *,
    scores: Mapping[tuple[str, str, int], Mapping[str, int]],
    rater_type: str = str(RaterType.LLM_JUDGE),
    rater_id: str = "gpt-oss:20b-median",
    equity_scenarios: Sequence[str] = (),
    fell_back: Sequence[tuple[str, str, int]] = (),
    gate_blocked: Sequence[tuple[str, str, int]] = (),
    split: str = "holdout",
    served_by: Mapping[tuple[str, str, int], str] | str | None = None,
    generation_id_suffix: str = "",
) -> pd.DataFrame:
    """Build a long score frame from `{(scenario, condition, sample): {dim: raw}}`.

    Raw scale throughout, exactly as the database stores it: `ritualistic` is
    higher-is-worse here, and every test that cares about polarity is testing
    whether the package flips it.

    `served_by` mirrors `generation.served_by`. The default reproduces the state
    D13 left the database in — condition LC served by vLLM, every other condition
    by Ollama — so a fixture that says nothing about backends still has the
    shape the arm guard reads. Pass a mapping to place a cell on a specific
    stack, or a string to put the whole frame on one.

    `generation_id_suffix` distinguishes two rows for the same cell, which is what
    the same scenario and sample produced by two serving stacks looks like.
    """
    fallback_set = set(fell_back)
    blocked_set = set(gate_blocked)
    rows: list[dict[str, object]] = []
    for (scenario, condition, sample), dims in scores.items():
        generation_id = f"{scenario}-{condition}-{sample}{generation_id_suffix}"
        if isinstance(served_by, str):
            backend = served_by
        elif served_by is None:
            backend = "vllm" if condition == str(Condition.LC) else "ollama"
        else:
            backend = served_by[(scenario, condition, sample)]
        for dimension in RUBRIC_DIMENSIONS:
            if dimension not in dims:
                continue
            rows.append(
                {
                    "generation_id": generation_id,
                    "scenario_id": scenario,
                    "condition": condition,
                    "sample_idx": sample,
                    "served_by": backend,
                    "rater_type": rater_type,
                    "rater_id": rater_id,
                    "rater_sample_idx": 0,
                    "split": split,
                    "challenge_type": "emotional_cue",
                    "emotion_intensity": 3,
                    "encounter_phase": "explanation",
                    "literacy_signal": "unmarked",
                    "equity_stratum": scenario in set(equity_scenarios),
                    "gate_blocked": (scenario, condition, sample) in blocked_set,
                    "fell_back_to_b": (scenario, condition, sample) in fallback_set,
                    "crag_grade": "none"
                    if (scenario, condition, sample) in fallback_set
                    else ("relevant" if condition == "C" else None),
                    "n_retrieved": 0 if (scenario, condition, sample) in fallback_set else 4,
                    "dimension": dimension,
                    "raw": dims[dimension],
                }
            )
    return pd.DataFrame(rows, columns=list(_LONG_COLUMNS))


#: The columns `carelite.stats.data.load_scores` returns, so an empty fixture
#: frame has the same shape as an empty database read rather than no shape at all.
_LONG_COLUMNS: tuple[str, ...] = (
    "generation_id",
    "scenario_id",
    "condition",
    "sample_idx",
    "served_by",
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
)


def constant_scores(
    dimensions: Sequence[str],
    value: int,
) -> dict[str, int]:
    return dict.fromkeys(dimensions, value)


@pytest.fixture
def nurse_dimensions() -> tuple[str, ...]:
    from carelite.stats.measures import NURSE_DIMENSIONS

    return NURSE_DIMENSIONS


@pytest.fixture
def separated_ab(nurse_dimensions: tuple[str, ...]) -> pd.DataFrame:
    """20 scenarios where B beats A on every NURSE dimension, by a clean margin.

    Known in advance: every paired difference is negative for (A - B), so the
    rank-biserial for A vs B is exactly -1.0, the Wilcoxon is as significant as
    n = 20 allows, and the registered direction (B higher) is met.
    """
    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(20):
        scenario = f"SC-{i:03d}"
        for sample in range(3):
            scores[(scenario, str(Condition.A), sample)] = constant_scores(nurse_dimensions, 2)
            scores[(scenario, str(Condition.B), sample)] = constant_scores(nurse_dimensions, 4)
    return make_long(scores=scores)


@pytest.fixture
def three_condition_long(nurse_dimensions: tuple[str, ...]) -> pd.DataFrame:
    """A, B, C on 12 scenarios with a strict A < B < C ordering in every block.

    Every block ranks identically, so the Friedman statistic is the maximum the
    design allows: with n blocks and k = 3, sum of squared rank totals is
    n^2 (1 + 4 + 9) and chi-square is exactly 2n.
    """
    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(12):
        scenario = f"SC-{i:03d}"
        for condition, value in ((Condition.A, 2), (Condition.B, 3), (Condition.C, 4)):
            for sample in range(3):
                scores[(scenario, str(condition), sample)] = constant_scores(
                    nurse_dimensions, value
                )
    return make_long(scores=scores)


@pytest.fixture
def noisy_mixed_frame() -> pd.DataFrame:
    """40 scenarios x {A, B} x 3 samples with variance components known by construction.

    Scenario intercepts ~ N(0, 0.9^2), a condition effect of exactly +0.5 for B,
    residual ~ N(0, 0.35^2). The continuous target for each generation is then
    written as five NURSE dimension integers whose mean reproduces it to the
    nearest 0.2 -- so the frame is a real long score frame on the real 1-5
    rubric scale, and the composite the model fits is the composite the analysis
    plan defines, not a float smuggled past it. Quantisation to the 0.2 grid
    adds variance 0.2^2/12 = 0.0033, two orders below the 0.1225 residual, so
    the components stay recoverable.
    """
    rng = np.random.default_rng(20260822)
    from carelite.stats.measures import NURSE_DIMENSIONS

    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(40):
        scenario = f"SC-{i:03d}"
        offset = rng.normal(0.0, 0.9)
        for condition, effect in ((Condition.A, 0.0), (Condition.B, 0.5)):
            for sample in range(3):
                target = 3.0 + offset + effect + rng.normal(0.0, 0.35)
                scores[(scenario, str(condition), sample)] = _as_five_integers(
                    target, NURSE_DIMENSIONS
                )
    return make_long(scores=scores)


def _as_five_integers(target: float, dimensions: Sequence[str]) -> dict[str, int]:
    """Five 1-5 integers whose mean is `target` rounded to the nearest 0.2."""
    k = len(dimensions)
    total = round(float(np.clip(target, 1.0, 5.0)) * k)
    total = int(np.clip(total, k, 5 * k))
    base, remainder = divmod(total, k)
    values = [base + (1 if j < remainder else 0) for j in range(k)]
    return dict(zip(dimensions, values, strict=True))
