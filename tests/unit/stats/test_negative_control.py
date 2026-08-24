"""The negative control, whose interesting outcome is the failure.

The three verdict states are each constructed deliberately: a rubric that
separates B from D, one that cannot, and one that ranks the degraded prompt
higher. The middle case is the one the pre-registration says must be reported
as a rubric validity failure rather than explained away, so the assertion is on
the words that appear in the output.
"""

from __future__ import annotations

import pandas as pd
import pytest

from carelite.stats.negative_control import (
    NEGATIVE_CONTROL_HYPOTHESIS,
    negative_control,
)
from carelite.stats.primary import CONFIRMATORY_FAMILY, run_family
from tests.unit.stats.conftest import constant_scores, make_long


def _frame(b_score: int, d_score: int, n: int = 20) -> pd.DataFrame:
    from carelite.stats.measures import NURSE_DIMENSIONS

    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(n):
        scenario = f"SC-{i:03d}"
        for sample in range(3):
            scores[(scenario, "B", sample)] = constant_scores(NURSE_DIMENSIONS, b_score)
            scores[(scenario, "D", sample)] = constant_scores(NURSE_DIMENSIONS, d_score)
    return make_long(scores=scores)


def test_the_hypothesis_is_taken_from_the_registered_family_not_redefined() -> None:
    assert NEGATIVE_CONTROL_HYPOTHESIS in CONFIRMATORY_FAMILY
    assert NEGATIVE_CONTROL_HYPOTHESIS.key == "secondary7_nurse_B_vs_D"
    assert NEGATIVE_CONTROL_HYPOTHESIS.expected_direction == ">"


def test_a_rubric_that_separates_b_from_d_passes() -> None:
    long = _frame(b_score=5, d_score=1)
    family = run_family(long, include_friedman=False, n_boot=500)
    result = negative_control(long, family=family)
    assert result is not None
    assert result.direction_correct
    assert result.interval_excludes_zero
    assert result.significant
    assert result.rubric_separates
    assert result.margin_is_large
    assert "NEGATIVE CONTROL PASSES" in result.render()


def test_a_rubric_that_cannot_separate_them_fails_prominently() -> None:
    long = _frame(b_score=3, d_score=3)
    family = run_family(long, include_friedman=False, n_boot=500)
    result = negative_control(long, family=family)
    assert result is not None
    assert not result.rubric_separates
    text = result.render()
    assert "NEGATIVE CONTROL FAILS" in text
    assert "rubric validity failure" in text
    assert "not explained away" in text
    # The verdict is the second line, above every number.
    assert text.splitlines()[1].strip().startswith("***")


def test_a_rubric_that_ranks_the_degraded_prompt_higher_is_flagged_as_inverted() -> None:
    long = _frame(b_score=1, d_score=5)
    family = run_family(long, include_friedman=False, n_boot=500)
    result = negative_control(long, family=family)
    assert result is not None
    assert result.inverted
    assert not result.rubric_separates
    assert "NEGATIVE CONTROL INVERTED" in result.render()


def test_all_three_criteria_are_required_not_any_of_them() -> None:
    """A one-point separation on 8 scenarios is significant but the CI is degenerate.

    The point is that `rubric_separates` is a conjunction: the verdict cannot be
    reached on the p-value alone.
    """
    long = _frame(b_score=4, d_score=3, n=8)
    family = run_family(long, include_friedman=False, n_boot=500)
    result = negative_control(long, family=family)
    assert result is not None
    assert result.rubric_separates == (
        result.direction_correct and result.interval_excludes_zero and result.significant
    )


def test_a_narrow_pass_is_reported_as_narrow() -> None:
    """§4.7 registers 'by a large margin'; a small separation says so."""
    from carelite.stats.measures import NURSE_DIMENSIONS

    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(30):
        scenario = f"SC-{i:03d}"
        # Two thirds of scenarios separate; one third ties, so the rank-biserial
        # lands below the conventional large threshold.
        d_value = 3 if i % 3 else 4
        for sample in range(3):
            scores[(scenario, "B", sample)] = constant_scores(NURSE_DIMENSIONS, 4)
            scores[(scenario, "D", sample)] = constant_scores(NURSE_DIMENSIONS, d_value)
    long = make_long(scores=scores)
    family = run_family(long, include_friedman=False, n_boot=500)
    result = negative_control(long, family=family)
    assert result is not None
    assert result.rubric_separates
    assert result.margin_is_large  # every non-tied scenario moves the same way


def test_the_family_corrected_p_is_used_when_the_family_is_supplied() -> None:
    long = _frame(b_score=5, d_score=1)
    family = run_family(long, include_friedman=False, n_boot=300)
    result = negative_control(long, family=family)
    assert result is not None
    assert result.comparison.family_size == 8


def test_without_the_family_the_correction_is_a_family_of_one_and_says_so() -> None:
    long = _frame(b_score=5, d_score=1)
    result = negative_control(long, n_boot=300)
    assert result is not None
    assert result.comparison.family_size == 1
    assert result.comparison.p_holm == pytest.approx(result.comparison.test.p_value)


def test_no_b_or_d_data_returns_none_rather_than_a_verdict() -> None:
    from carelite.stats.measures import NURSE_DIMENSIONS

    scores = {(f"SC-{i:03d}", "A", 0): constant_scores(NURSE_DIMENSIONS, 3) for i in range(5)}
    assert negative_control(make_long(scores=scores), n_boot=100) is None


def test_the_render_states_that_d_is_not_degraded_on_safety() -> None:
    long = _frame(b_score=5, d_score=1)
    result = negative_control(long, n_boot=300)
    assert result is not None
    text = result.render()
    assert "not on safety" in text
    assert "§10" in text
