"""Friedman, Wilcoxon and Holm against examples computed by hand.

Every expected value in this file is derived from the definition of the statistic
and written into the test as arithmetic, not copied from a run of the code it is
testing. A test that asserts a routine returns what it returned yesterday
catches a regression; it does not catch a routine that was wrong on the day it
was written, and a silent stats bug is indistinguishable from a result.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from carelite.stats.evidence import EvidenceStatus
from carelite.stats.instrument import instrument_report
from carelite.stats.primary import (
    CONFIRMATORY_FAMILY,
    PRESPECIFIED_HYPOTHESES,
    dimension_expansion,
    friedman_across_conditions,
    friedman_omnibus,
    holm_bonferroni,
    run_family,
    run_pairwise,
    wilcoxon_paired,
)
from carelite.types import RUBRIC_DIMENSIONS, Condition
from tests.unit.stats.conftest import constant_scores, make_long

# ---------------------------------------------------------------------------
# Friedman
# ---------------------------------------------------------------------------


def test_friedman_matches_the_hand_computed_chi_square() -> None:
    """Four blocks, three conditions, identical ranking in every block.

    Ranks are (1, 2, 3) in each of the four blocks, so the rank totals are
    R = (4, 8, 12) and the statistic is

        chi2 = 12 / (n k (k+1)) * sum(R^2) - 3 n (k+1)
             = 12 / (4*3*4) * (16 + 64 + 144) - 3*4*4
             = 0.25 * 224 - 48
             = 8.0

    on df = 2, whose upper tail is exp(-8/2) = exp(-4).
    """
    matrix = pd.DataFrame(
        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [5.0, 6.0, 7.0], [0.0, 1.0, 2.0]],
        columns=["A", "B", "C"],
    )
    statistic, p_value, df, n_blocks = friedman_omnibus(matrix)
    assert statistic == pytest.approx(8.0)
    assert df == 2
    assert n_blocks == 4
    assert p_value == pytest.approx(math.exp(-4.0))


def test_friedman_maximum_statistic_is_twice_the_block_count() -> None:
    """With k = 3 and every block ranked identically, chi2 = 2n exactly.

    sum(R^2) = n^2 (1 + 4 + 9) = 14 n^2, so
    chi2 = 12/(3n*4) * 14 n^2 - 12 n = 14n/... = 2n.
    """
    for n in (5, 12, 30):
        matrix = pd.DataFrame(
            {"A": np.ones(n), "B": np.full(n, 2.0), "C": np.full(n, 3.0)},
        )
        statistic, _, _, _ = friedman_omnibus(matrix)
        assert statistic == pytest.approx(2.0 * n)


def test_friedman_is_undefined_rather_than_zero_when_every_block_is_tied() -> None:
    matrix = pd.DataFrame({"A": [3.0] * 6, "B": [3.0] * 6, "C": [3.0] * 6})
    statistic, p_value, _, _ = friedman_omnibus(matrix)
    assert math.isnan(statistic)
    assert math.isnan(p_value)


def test_friedman_needs_three_conditions() -> None:
    matrix = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})
    statistic, _, df, _ = friedman_omnibus(matrix)
    assert math.isnan(statistic)
    assert df == 1


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank
# ---------------------------------------------------------------------------


def test_wilcoxon_matches_the_hand_computed_exact_p() -> None:
    """Differences (1, 2, 3, 4, 5, -6): no ties, so the exact null applies.

    Signed ranks: W+ = 1+2+3+4+5 = 15, W- = 6, statistic = min = 6.
    Under the null every sign pattern is equally likely, so P(W <= 6) counts the
    subsets of {1..6} summing to at most 6: 1 + 1 + 1 + 2 + 2 + 3 + 4 = 14 of 64.
    Two-sided p = 2 * 14/64 = 0.4375.
    """
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -6.0])
    y = np.zeros(6)
    result = wilcoxon_paired(x, y)
    assert result.statistic == pytest.approx(6.0)
    assert result.p_value == pytest.approx(0.4375)
    assert result.method == "exact"
    assert result.n_pairs == 6
    assert result.n_nonzero == 6


def test_wilcoxon_drops_zero_differences_from_the_ranking() -> None:
    """Two tied pairs contribute nothing; the surviving four are the whole test."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0])
    y = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0])
    result = wilcoxon_paired(x, y)
    assert result.n_pairs == 6
    assert result.n_nonzero == 4
    # Four positive differences, no negatives: the smallest exact p at n = 4.
    assert result.p_value == pytest.approx(2.0 / 16.0)


def test_wilcoxon_is_undefined_when_every_pair_is_tied() -> None:
    result = wilcoxon_paired(np.array([3.0, 3.0]), np.array([3.0, 3.0]))
    assert math.isnan(result.p_value)
    assert result.method == "undefined"


def test_wilcoxon_switches_to_the_normal_approximation_with_ties() -> None:
    x = np.array([1.0, 1.0, 2.0, 3.0, -1.0, 4.0])
    result = wilcoxon_paired(x, np.zeros(6))
    assert result.method == "approx"


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------


def test_holm_matches_the_hand_computed_step_down() -> None:
    """p = (0.01, 0.02, 0.03, 0.04), m = 4.

    Multipliers descend 4, 3, 2, 1 down the sorted list: 0.04, 0.06, 0.06, 0.04,
    then the running maximum enforces monotonicity, giving 0.04, 0.06, 0.06, 0.06.
    """
    assert holm_bonferroni([0.01, 0.02, 0.03, 0.04]) == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_holm_preserves_input_order_not_sorted_order() -> None:
    assert holm_bonferroni([0.04, 0.01, 0.03, 0.02]) == pytest.approx([0.06, 0.04, 0.06, 0.06])


def test_holm_monotonicity_carries_a_large_p_down_the_list() -> None:
    """0.5 at rank 2 of 2 gets multiplied by 1, but cannot fall below its predecessor."""
    assert holm_bonferroni([0.001, 0.5]) == pytest.approx([0.002, 0.5])


def test_holm_caps_at_one() -> None:
    assert holm_bonferroni([0.4, 0.5, 0.6]) == pytest.approx([1.0, 1.0, 1.0])


def test_holm_family_size_overrides_the_number_of_p_values() -> None:
    """A test that could not be computed still consumes its share of the family."""
    assert holm_bonferroni([0.01], family_size=8) == pytest.approx([0.08])


def test_holm_passes_nan_through_without_letting_it_win_a_rank() -> None:
    adjusted = holm_bonferroni([0.01, math.nan, 0.02])
    assert math.isnan(adjusted[1])
    # Family size is still 3, so the smallest p is multiplied by 3.
    assert adjusted[0] == pytest.approx(0.03)


def test_holm_is_never_less_conservative_than_the_raw_p() -> None:
    rng = np.random.default_rng(3)
    p_values = list(rng.uniform(0, 1, size=25))
    adjusted = holm_bonferroni(p_values)
    assert all(a >= p - 1e-12 for a, p in zip(adjusted, p_values, strict=True))


def test_holm_agrees_with_statsmodels_on_a_random_family() -> None:
    """Cross-check against an independent implementation of the same procedure."""
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(11)
    p_values = list(rng.uniform(0, 0.3, size=40))
    _, expected, _, _ = multipletests(p_values, method="holm")
    assert holm_bonferroni(p_values) == pytest.approx(list(expected))


# ---------------------------------------------------------------------------
# The correction family
# ---------------------------------------------------------------------------


def test_the_confirmatory_family_is_the_eight_registered_hypotheses() -> None:
    """One primary (§3) and seven secondary (§4). Not per-dimension, not more."""
    assert len(CONFIRMATORY_FAMILY) == 8
    assert CONFIRMATORY_FAMILY is PRESPECIFIED_HYPOTHESES
    assert sum(1 for h in CONFIRMATORY_FAMILY if h.role == "primary") == 1
    assert all(h.prespecified for h in CONFIRMATORY_FAMILY)


def test_the_family_spans_dimensions_which_is_what_holm_corrects_across() -> None:
    """§8.1: a dimension is not corrected in isolation from the others."""
    measures = {h.measure_key for h in CONFIRMATORY_FAMILY}
    assert measures == {
        "nurse_composite",
        "four_habits_composite",
        "naturalness",
        "ritualistic",
    }


def test_the_registered_pairs_are_exactly_the_five_in_section_4() -> None:
    pairs = {(str(h.left), str(h.right)) for h in CONFIRMATORY_FAMILY}
    assert pairs == {("A", "B"), ("B", "C"), ("C", "LC"), ("A", "A2"), ("B", "D")}


def test_registered_directions_match_the_preregistration() -> None:
    expected = {
        "primary_nurse_A_vs_B": "<",  # B higher
        "secondary1_four_habits_A_vs_B": "<",
        "secondary2_nurse_B_vs_C": "<",  # C higher
        "secondary3_nurse_C_vs_LC": ">",  # C higher
        "secondary4_naturalness_A_vs_B": ">",  # A higher — the against-the-system one
        "secondary5_ritualistic_A_vs_B": ">",  # A higher on quality
        "secondary6_nurse_A_vs_A2": "=",  # no difference registered
        "secondary7_nurse_B_vs_D": ">",  # B higher
    }
    assert {h.key: h.expected_direction for h in CONFIRMATORY_FAMILY} == expected


def test_dimension_expansion_is_exploratory_and_excludes_the_registered_cells() -> None:
    expanded = dimension_expansion()
    assert expanded, "expected the exploratory expansion to be non-empty"
    assert all(not h.prespecified for h in expanded)
    assert all(h.role == "exploratory" for h in expanded)
    # Five registered pairs x eleven dimensions, minus the two dimension-level
    # cells §4 already registers (naturalness A-B and ritualistic A-B).
    assert len(expanded) == 5 * len(RUBRIC_DIMENSIONS) - 2
    registered = {(h.left, h.right, h.measure_key) for h in CONFIRMATORY_FAMILY}
    assert not any((h.left, h.right, h.measure_key) in registered for h in expanded)


def test_holm_is_applied_across_the_whole_family_not_per_dimension(
    separated_ab: pd.DataFrame,
) -> None:
    """The adjusted p must reflect m = 8, not m = 1 for its own dimension."""
    family = run_family(separated_ab, include_friedman=False, n_boot=200)
    primary = family.by_key("primary_nurse_A_vs_B")
    assert primary is not None
    assert primary.family_size == 8
    # Only one comparison has data, so it is the smallest p and takes the full
    # multiplier. Anything else means the family was scoped wrongly.
    assert primary.p_holm == pytest.approx(min(1.0, 8 * primary.test.p_value))


def test_an_uncomputable_test_still_consumes_its_share_of_the_family(
    separated_ab: pd.DataFrame,
) -> None:
    family = run_family(separated_ab, include_friedman=False, n_boot=200)
    # Only A and B exist in this frame, and only on the NURSE dimensions, so
    # seven of the eight registered tests have no data.
    assert len(family.results) == 1
    assert family.family_size == 8
    assert all(r.family_size == 8 for r in family.results)
    assert any("no paired data for" in note for note in family.notes)


# ---------------------------------------------------------------------------
# Results carry their own reporting status
# ---------------------------------------------------------------------------


def test_results_are_exploratory_when_the_judge_validation_has_not_run(
    separated_ab: pd.DataFrame,
) -> None:
    family = run_family(separated_ab, include_friedman=False, n_boot=200)
    assert family.confirmatory == ()
    for result in family.results:
        assert not result.label.is_confirmatory
        assert "judge validation study has not run" in result.label.tag()


def test_a_demoted_constituent_dimension_demotes_the_composite(
    separated_ab: pd.DataFrame,
) -> None:
    statuses = dict.fromkeys(RUBRIC_DIMENSIONS, EvidenceStatus.CONFIRMATORY)
    statuses["explore"] = EvidenceStatus.EXPLORATORY
    family = run_family(separated_ab, include_friedman=False, statuses=statuses, n_boot=200)
    primary = family.by_key("primary_nurse_A_vs_B")
    assert primary is not None
    assert not primary.label.is_confirmatory
    assert "explore" in primary.label.tag()


def test_a_fully_validated_composite_can_be_confirmatory(
    separated_ab: pd.DataFrame,
) -> None:
    statuses = dict.fromkeys(RUBRIC_DIMENSIONS, EvidenceStatus.CONFIRMATORY)
    family = run_family(separated_ab, include_friedman=False, statuses=statuses, n_boot=200)
    primary = family.by_key("primary_nurse_A_vs_B")
    assert primary is not None
    assert primary.label.is_confirmatory


# ---------------------------------------------------------------------------
# End to end on a frame with a known answer
# ---------------------------------------------------------------------------


def test_a_clean_separation_gives_a_perfect_rank_biserial(
    separated_ab: pd.DataFrame,
) -> None:
    """A scores 2 and B scores 4 on every dimension of all 20 scenarios.

    Every paired difference (A - B) is exactly -2, so W+ = 0 and the
    rank-biserial is exactly -1: A is lower than B in every scenario, which is
    the registered direction for the primary outcome.
    """
    family = run_family(separated_ab, include_friedman=False, n_boot=500)
    primary = family.by_key("primary_nurse_A_vs_B")
    assert primary is not None
    assert primary.effects.rank_biserial.point == pytest.approx(-1.0)
    assert primary.n_scenarios == 20
    assert primary.observed_direction == "<"
    assert primary.direction_as_registered is True
    assert primary.effects.hodges_lehmann.point == pytest.approx(-2.0)


def test_friedman_runs_on_all_eleven_dimensions(
    three_condition_long: pd.DataFrame,
) -> None:
    results = friedman_across_conditions(three_condition_long)
    assert len(results) == len(RUBRIC_DIMENSIONS)
    assert [r.measure_key for r in results] == list(RUBRIC_DIMENSIONS)
    scored = [r for r in results if not math.isnan(r.statistic)]
    # Only the NURSE dimensions carry scores in this fixture.
    assert len(scored) == 5
    for result in scored:
        assert result.n_blocks == 12
        assert result.statistic == pytest.approx(24.0)  # 2n with k = 3
        assert result.p_value == pytest.approx(stats.chi2.sf(24.0, 2))


def test_render_puts_the_effect_size_before_the_p_value(
    separated_ab: pd.DataFrame,
) -> None:
    """§8.2 is an ordering requirement, so it is asserted as one."""
    family = run_family(separated_ab, include_friedman=False, n_boot=200)
    text = family.by_key("primary_nurse_A_vs_B").render()  # type: ignore[union-attr]
    assert text.index("effect (rank-biserial)") < text.index("then p:")
    assert "95% bootstrap CI" in text


def test_family_render_states_what_the_correction_was_applied_across(
    separated_ab: pd.DataFrame,
) -> None:
    family = run_family(separated_ab, include_friedman=False, n_boot=200)
    text = family.render()
    assert "Holm-Bonferroni across" in text
    assert "not per dimension" in text
    assert "family size m = 8" in text


def test_pairwise_returns_none_rather_than_inventing_a_comparison() -> None:
    empty = pd.DataFrame(
        columns=[
            "generation_id",
            "scenario_id",
            "condition",
            "sample_idx",
            "rater_type",
            "rater_id",
            "dimension",
            "raw",
        ]
    )
    assert run_pairwise(empty, CONFIRMATORY_FAMILY[0], n_boot=50) is None


def test_a_scenario_missing_one_condition_is_dropped_and_counted(
    nurse_dimensions: tuple[str, ...],
) -> None:
    from tests.unit.stats.conftest import constant_scores, make_long

    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(6):
        scenario = f"SC-{i:03d}"
        scores[(scenario, str(Condition.A), 0)] = constant_scores(nurse_dimensions, 2)
        if i < 4:  # two scenarios have no Condition B cell at all
            scores[(scenario, str(Condition.B), 0)] = constant_scores(nurse_dimensions, 4)
    result = run_pairwise(make_long(scores=scores), CONFIRMATORY_FAMILY[0], n_boot=100)
    assert result is not None
    assert result.n_scenarios == 4
    assert result.n_dropped == 2


# ---------------------------------------------------------------------------
# D13: secondary outcome 3 runs again, and cannot be read without its caveats
# ---------------------------------------------------------------------------


class TestSecondaryOutcomeThree:
    """C vs LC is computable again, and carries two qualifications in the output.

    D11 retired this comparison; D13 generated all 180 LC cells under vLLM and
    restored it. What is pinned here is that restoring it did not quietly turn it
    into a clean architectural comparison: the arm was served by a different
    stack from condition C, and under D7 it is the reduced form of the question.
    Both sentences must reach the rendered result and the result's own label, not
    only a docstring.
    """

    @pytest.fixture
    def c_and_lc(self, nurse_dimensions: tuple[str, ...]) -> pd.DataFrame:
        """20 scenarios with a complete LC arm, as D13 left the database."""
        scores: dict[tuple[str, str, int], dict[str, int]] = {}
        for i in range(20):
            scenario = f"SC-{i:03d}"
            for sample in range(3):
                scores[(scenario, "C", sample)] = constant_scores(nurse_dimensions, 4)
                scores[(scenario, "LC", sample)] = constant_scores(nurse_dimensions, 2)
        return make_long(scores=scores)

    def test_the_hypothesis_is_no_longer_retired(self) -> None:
        hypothesis = next(h for h in PRESPECIFIED_HYPOTHESES if h.key == "secondary3_nurse_C_vs_LC")
        assert not hypothesis.retired_by_decision
        assert hypothesis.not_computable_reason == ""
        assert "D13" in hypothesis.description

    def test_no_hypothesis_in_the_family_is_retired_any_more(self) -> None:
        assert [h.key for h in CONFIRMATORY_FAMILY if h.retired_by_decision] == []

    def test_it_computes_and_keeps_the_family_at_eight(self, c_and_lc: pd.DataFrame) -> None:
        family = run_family(c_and_lc, n_boot=200)
        result = family.by_key("secondary3_nurse_C_vs_LC")
        assert result is not None
        assert result.n_scenarios == 20
        assert result.family_size == len(PRESPECIFIED_HYPOTHESES) == 8

    def test_the_serving_stack_confound_is_in_the_rendered_result(
        self, c_and_lc: pd.DataFrame
    ) -> None:
        result = run_pairwise(
            c_and_lc,
            next(h for h in PRESPECIFIED_HYPOTHESES if h.key == "secondary3_nurse_C_vs_LC"),
            n_boot=200,
        )
        assert result is not None
        text = result.render()
        assert "CONFOUNDED BY SERVING STACK" in text
        # The caveat precedes the number it qualifies, for the same reason the
        # instrument verdict does.
        assert text.index("CONFOUNDED BY SERVING STACK") < text.index("then p:")

    def test_the_reduced_form_caveat_names_d7(self, c_and_lc: pd.DataFrame) -> None:
        result = run_pairwise(
            c_and_lc,
            next(h for h in PRESPECIFIED_HYPOTHESES if h.key == "secondary3_nurse_C_vs_LC"),
            n_boot=200,
        )
        assert result is not None
        text = result.render()
        assert "REDUCED FORM" in text
        assert "151/471 chunks" in text
        assert "D7" in text

    def test_the_caveats_demote_the_label_itself(self, c_and_lc: pd.DataFrame) -> None:
        """Structure, not prose: the confound reaches `effect-sizes.csv`'s label column."""
        result = run_pairwise(
            c_and_lc,
            next(h for h in PRESPECIFIED_HYPOTHESES if h.key == "secondary3_nurse_C_vs_LC"),
            n_boot=200,
        )
        assert result is not None
        assert not result.label.cleared_gate
        tag = result.label.tag()
        assert tag.startswith("EXPLORATORY")
        assert "CONFOUNDED BY SERVING STACK" in tag

    def test_a_comparison_with_no_caveats_is_unchanged(self, separated_ab: pd.DataFrame) -> None:
        """The mechanism must not add noise to the seven comparisons that have none."""
        result = run_pairwise(separated_ab, CONFIRMATORY_FAMILY[0], n_boot=200)
        assert result is not None
        assert "!!!" not in result.render()

    def test_a_missing_comparison_still_reads_as_missing(
        self, nurse_dimensions: tuple[str, ...]
    ) -> None:
        """With no LC rows at all, the family says so rather than reporting a null."""
        scores = {
            (f"SC-{i:03d}", "A", s): constant_scores(nurse_dimensions, 3)
            for i in range(10)
            for s in range(3)
        }
        family = run_family(make_long(scores=scores), n_boot=200)
        notes = " ".join(family.notes)
        assert "no paired data for" in notes
        assert "secondary3_nurse_C_vs_LC" in notes
        assert family.family_size == 8


# ---------------------------------------------------------------------------
# An untestable comparison must not render as a null one
# ---------------------------------------------------------------------------


class TestInstrumentLimitedRendering:
    @pytest.fixture
    def flat_naturalness(self, nurse_dimensions: tuple[str, ...]) -> pd.DataFrame:
        """A and B over 20 scenarios where `naturalness` is the same value throughout."""
        scores: dict[tuple[str, str, int], dict[str, int]] = {}
        for i in range(20):
            scenario = f"SC-{i:03d}"
            for condition in ("A", "B"):
                for sample in range(3):
                    scores[(scenario, condition, sample)] = {"naturalness": 3}
        return make_long(scores=scores)

    def test_the_verdict_precedes_the_p_value(self, flat_naturalness: pd.DataFrame) -> None:
        """Ordering is the mechanism. A reader who meets the p-value first has misread it."""
        report = instrument_report(flat_naturalness)
        hypothesis = next(
            h for h in PRESPECIFIED_HYPOTHESES if h.key == "secondary4_naturalness_A_vs_B"
        )
        result = run_pairwise(
            flat_naturalness, hypothesis, n_boot=200, discrimination=report.statuses
        )
        assert result is not None
        assert result.not_testable

        text = result.render()
        assert text.index("INSTRUMENT-LIMITED") < text.index("then p:")
        assert "NOT TESTABLE" in text
        assert "no significant difference" not in text.lower().replace(
            "as 'no significant difference'", ""
        )

    def test_the_pair_count_the_test_rests_on_is_shown(
        self, flat_naturalness: pd.DataFrame
    ) -> None:
        """Degenerate is not the same as flat, so the evidence is quantified."""
        report = instrument_report(flat_naturalness)
        hypothesis = next(
            h for h in PRESPECIFIED_HYPOTHESES if h.key == "secondary4_naturalness_A_vs_B"
        )
        result = run_pairwise(
            flat_naturalness, hypothesis, n_boot=200, discrimination=report.statuses
        )
        assert result is not None
        assert "the test rests on 0 of 20 scenarios (20 tied exactly)" in result.render()

    def test_without_the_diagnostic_no_claim_is_made(self, flat_naturalness: pd.DataFrame) -> None:
        """`discrimination=None` must leave the result unlabelled, not silently clean."""
        hypothesis = next(
            h for h in PRESPECIFIED_HYPOTHESES if h.key == "secondary4_naturalness_A_vs_B"
        )
        result = run_pairwise(flat_naturalness, hypothesis, n_boot=200)
        assert result is not None
        assert result.testability is None
        assert not result.not_testable


# ---------------------------------------------------------------------------
# The pooling guard reaches the comparison, not only the frame builder
# ---------------------------------------------------------------------------


class TestBackendGuardInAComparison:
    """A comparison built on a pooled arm must not quietly produce a number.

    `restrict_to_analysis_arms` is the selection the report uses, but a caller
    can assemble a family from a frame of its own. The LC arm is the one place
    that matters, because it is the one condition served by two stacks.
    """

    @pytest.fixture
    def pooled_lc(self, nurse_dimensions: tuple[str, ...]) -> pd.DataFrame:
        arm: dict[tuple[str, str, int], dict[str, int]] = {}
        for i in range(10):
            scenario = f"SC-{i:03d}"
            for sample in range(3):
                arm[(scenario, "C", sample)] = constant_scores(nurse_dimensions, 4)
                arm[(scenario, "LC", sample)] = constant_scores(nurse_dimensions, 3)
        stale = {
            (f"SC-{i:03d}", "LC", sample): constant_scores(nurse_dimensions, 1)
            for i in range(3)
            for sample in range(3)
        }
        return pd.concat(
            [
                make_long(scores=arm),
                make_long(scores=stale, served_by="ollama", generation_id_suffix="-ollama"),
            ],
            ignore_index=True,
        )

    def test_a_pooled_lc_arm_raises_rather_than_scoring(self, pooled_lc: pd.DataFrame) -> None:
        from carelite.stats.arms import MixedBackendError

        hypothesis = next(h for h in PRESPECIFIED_HYPOTHESES if h.key == "secondary3_nurse_C_vs_LC")
        with pytest.raises(MixedBackendError, match="C vs LC"):
            run_pairwise(pooled_lc, hypothesis, n_boot=100)

    def test_a_comparison_that_does_not_touch_lc_is_unaffected(
        self, pooled_lc: pd.DataFrame, nurse_dimensions: tuple[str, ...]
    ) -> None:
        """The guard is about the one ambiguous condition, not about every frame.

        The A-vs-B rows are appended to the frame whose LC arm is pooled, so the
        assertion is that the guard is scoped to the comparison rather than to
        the whole frame.
        """
        scores = {
            (f"SC-{i:03d}", c, s): constant_scores(nurse_dimensions, 3 if c == "A" else 4)
            for i in range(6)
            for c in ("A", "B")
            for s in range(3)
        }
        frame = pd.concat([pooled_lc, make_long(scores=scores)], ignore_index=True)
        assert run_pairwise(frame, CONFIRMATORY_FAMILY[0], n_boot=100) is not None

    def test_a_frame_without_the_column_is_labelled_not_refused(
        self, nurse_dimensions: tuple[str, ...]
    ) -> None:
        """A legacy frame cannot confirm the arm, and the label says exactly that."""
        scores: dict[tuple[str, str, int], dict[str, int]] = {}
        for i in range(10):
            scenario = f"SC-{i:03d}"
            for sample in range(3):
                scores[(scenario, "C", sample)] = constant_scores(nurse_dimensions, 4)
                scores[(scenario, "LC", sample)] = constant_scores(nurse_dimensions, 3)
        frame = make_long(scores=scores).drop(columns=["served_by"])
        hypothesis = next(h for h in PRESPECIFIED_HYPOTHESES if h.key == "secondary3_nurse_C_vs_LC")
        result = run_pairwise(frame, hypothesis, n_boot=100)
        assert result is not None
        assert "no `served_by` column" in result.label.tag()
