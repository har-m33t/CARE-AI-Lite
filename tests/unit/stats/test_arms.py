"""The pooling guard, tested by building the pooled frame on purpose.

`DECISIONS.md` D13 says the LC analysis arm is `served_by = 'vllm'`, 180 cells,
and nothing else. A docstring saying so is worth less than a test that fails when
the 39 Ollama cells are pooled into it, so the frames below are the mistake, and
the assertions are that the mistake raises rather than producing a number.
"""

from __future__ import annotations

import pandas as pd
import pytest

from carelite.stats.arms import (
    AMBIGUOUS_WITHOUT_BACKEND,
    EXCLUDED_ARMS,
    LC_ANALYSIS_BACKEND,
    LC_EQUIVALENCE_BACKEND,
    MixedBackendError,
    assert_single_backend_per_condition,
    backends_by_condition,
    restrict_to_analysis_arms,
)

from .conftest import constant_scores, make_long


def _lc_frame(nurse_dimensions: tuple[str, ...]) -> pd.DataFrame:
    """C on 12 scenarios, LC under both stacks: vLLM everywhere, Ollama on 4.

    This is the real shape in miniature — a complete vLLM arm alongside a partial
    Ollama record covering a third of the scenarios.
    """
    arm: dict[tuple[str, str, int], dict[str, int]] = {}
    arm_served: dict[tuple[str, str, int], str] = {}
    for i in range(12):
        scenario = f"SC-{i:03d}"
        for sample in range(3):
            arm[(scenario, "C", sample)] = constant_scores(nurse_dimensions, 4)
            arm_served[(scenario, "C", sample)] = "ollama"
            arm[(scenario, "LC", sample)] = constant_scores(nurse_dimensions, 3)
            arm_served[(scenario, "LC", sample)] = LC_ANALYSIS_BACKEND

    sample_only: dict[tuple[str, str, int], dict[str, int]] = {}
    sample_served: dict[tuple[str, str, int], str] = {}
    for i in range(4):
        scenario = f"SC-{i:03d}"
        for sample in range(3):
            sample_only[(scenario, "LC", sample)] = constant_scores(nurse_dimensions, 2)
            sample_served[(scenario, "LC", sample)] = LC_EQUIVALENCE_BACKEND

    return pd.concat(
        [
            make_long(scores=arm, served_by=arm_served),
            # A different generation id for the same cell, which is what two
            # stacks producing the same scenario/sample actually looks like.
            make_long(
                scores=sample_only,
                served_by=sample_served,
                generation_id_suffix="-ollama",
            ),
        ],
        ignore_index=True,
    )


class TestTheExclusionRule:
    def test_the_only_excluded_selection_is_the_ollama_lc_sample(self) -> None:
        assert [(a.condition, a.served_by) for a in EXCLUDED_ARMS] == [
            ("LC", LC_EQUIVALENCE_BACKEND)
        ]
        assert "D13" in EXCLUDED_ARMS[0].reason

    def test_the_vllm_lc_arm_survives_and_the_ollama_sample_does_not(
        self, nurse_dimensions: tuple[str, ...]
    ) -> None:
        frame = _lc_frame(nurse_dimensions)
        kept, selection = restrict_to_analysis_arms(frame)

        lc = kept[kept["condition"] == "LC"]
        assert set(lc["served_by"]) == {LC_ANALYSIS_BACKEND}
        assert lc["generation_id"].nunique() == 12 * 3
        assert lc["scenario_id"].nunique() == 12

        assert selection.excluded_counts == {f"LC/{LC_EQUIVALENCE_BACKEND}": 4 * 3}
        assert selection.exclusions[0].n_scenarios == 4
        assert "D13" in selection.exclusions[0].reason

    def test_the_exclusion_is_by_condition_and_backend_not_by_condition(
        self, nurse_dimensions: tuple[str, ...]
    ) -> None:
        """A rule that dropped `LC` would discard the arm D13 exists to restore."""
        frame = _lc_frame(nurse_dimensions)
        kept, _ = restrict_to_analysis_arms(frame)
        assert (kept["condition"] == "LC").any()

    def test_the_render_names_the_stack_behind_every_arm(
        self, nurse_dimensions: tuple[str, ...]
    ) -> None:
        _, selection = restrict_to_analysis_arms(_lc_frame(nurse_dimensions))
        text = selection.render()
        assert f"LC: {LC_ANALYSIS_BACKEND} 36" in text
        assert "EXCLUDED LC/ollama: 12 generations over 4 scenarios" in text


class TestThePoolingGuard:
    def test_a_pooled_arm_raises_rather_than_producing_a_number(
        self, nurse_dimensions: tuple[str, ...]
    ) -> None:
        pooled = _lc_frame(nurse_dimensions)  # both stacks still present
        with pytest.raises(MixedBackendError) as exc:
            assert_single_backend_per_condition(pooled)
        message = str(exc.value)
        assert "LC" in message
        assert "ollama" in message and "vllm" in message
        assert "D13" in message

    def test_restrict_refuses_a_pool_it_cannot_resolve(
        self, nurse_dimensions: tuple[str, ...]
    ) -> None:
        """Condition C under two stacks is not covered by any exclusion, so it raises."""
        scores: dict[tuple[str, str, int], dict[str, int]] = {}
        served: dict[tuple[str, str, int], str] = {}
        for i in range(4):
            scenario = f"SC-{i:03d}"
            scores[(scenario, "C", 0)] = constant_scores(nurse_dimensions, 4)
            served[(scenario, "C", 0)] = "ollama" if i % 2 else "vllm"
        frame = make_long(scores=scores, served_by=served)
        with pytest.raises(MixedBackendError, match="pools serving stacks"):
            restrict_to_analysis_arms(frame)

    def test_the_pool_can_be_inspected_deliberately(
        self, nurse_dimensions: tuple[str, ...]
    ) -> None:
        frame = _lc_frame(nurse_dimensions)
        kept, selection = restrict_to_analysis_arms(frame, excluded=(), strict=False)
        assert kept.shape == frame.shape
        assert selection.backends["LC"] == {
            LC_ANALYSIS_BACKEND: 36,
            LC_EQUIVALENCE_BACKEND: 12,
        }

    def test_lc_without_a_served_by_column_is_refused_not_guessed(
        self, nurse_dimensions: tuple[str, ...]
    ) -> None:
        """The column is the only thing separating the arm from the sample."""
        scores = {
            ("SC-000", "LC", 0): constant_scores(nurse_dimensions, 3),
            ("SC-000", "C", 0): constant_scores(nurse_dimensions, 4),
        }
        frame = make_long(scores=scores).drop(columns=["served_by"])
        with pytest.raises(MixedBackendError, match="no `served_by` column"):
            assert_single_backend_per_condition(frame)

    def test_a_backend_free_frame_without_lc_is_accepted(self, separated_ab: pd.DataFrame) -> None:
        """Every pre-D13 row is Ollama, which is what the schema backfills to."""
        frame = separated_ab.drop(columns=["served_by"])
        assert assert_single_backend_per_condition(frame) == {}
        kept, selection = restrict_to_analysis_arms(frame)
        assert kept.shape == frame.shape
        assert selection.served_by_present is False
        assert selection.exclusions == ()

    def test_an_empty_frame_is_not_an_error(self) -> None:
        empty = pd.DataFrame(columns=["generation_id", "scenario_id", "condition", "served_by"])
        assert assert_single_backend_per_condition(empty) == {}
        kept, selection = restrict_to_analysis_arms(empty)
        assert kept.empty
        assert selection.exclusions == ()

    def test_single_stack_conditions_report_the_stack_they_ran_on(
        self, separated_ab: pd.DataFrame
    ) -> None:
        assert assert_single_backend_per_condition(separated_ab) == {
            "A": "ollama",
            "B": "ollama",
        }


def test_ambiguous_conditions_are_derived_from_the_exclusion_table() -> None:
    """One list, so a second excluded arm cannot be added without the guard knowing."""
    assert {a.condition for a in EXCLUDED_ARMS} == AMBIGUOUS_WITHOUT_BACKEND


def test_backends_by_condition_counts_generations_not_rows(
    nurse_dimensions: tuple[str, ...],
) -> None:
    frame = _lc_frame(nurse_dimensions)
    counts = backends_by_condition(frame)
    # 12 scenarios x 3 samples, counted as generations rather than as the five
    # dimension rows each one contributes.
    assert counts["C"] == {"ollama": 36}
    assert counts["LC"][LC_ANALYSIS_BACKEND] == 36
