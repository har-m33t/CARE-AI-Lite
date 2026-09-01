"""The arm guard: what is an analysis arm, and what cannot be pooled into one.

D13 completed condition LC under vLLM and kept the 39 Ollama cells from D11 as a
paired equivalence sample. Both sets carry `condition = 'LC'`, so a query that
selects on the condition alone returns 219 rows spanning two serving stacks and
two coverage regimes. These tests pin the refusal.

No database here. Everything that decides whether a selection is legitimate is a
pure function of the rows, which is why it can be tested without Postgres; the
fetchers that go to the table are exercised in `test_store_db.py`.
"""

from __future__ import annotations

import pytest

from carelite.eval.judge.arms import (
    LC_ANALYSIS_BACKEND,
    LC_EQUIVALENCE_BACKEND,
    SERVING_BACKENDS,
    MixedBackendError,
    UnpairableCells,
    assert_single_backend,
    backends_in,
    is_partial_record,
    pair_cells,
)


def cell(
    scenario_id: str = "SC-001",
    *,
    sample_idx: int = 0,
    served_by: str = "ollama",
    seed: int = 11,
    condition: str = "LC",
    prompt_id: str = "condition_lc.v1",
) -> dict[str, object]:
    return {
        "generation_id": f"gen-{scenario_id}-{condition}-{sample_idx}-{served_by}",
        "scenario_id": scenario_id,
        "condition": condition,
        "sample_idx": sample_idx,
        "served_by": served_by,
        "seed": seed,
        "prompt_id": prompt_id,
    }


class TestBackendVocabulary:
    def test_the_vocabulary_is_the_schema_check_constraint(self) -> None:
        """Widening this set is a schema change, not a judge-lane change."""
        from pathlib import Path

        schema = Path("carelite/db/schema.sql").read_text(encoding="utf-8")
        for backend in SERVING_BACKENDS:
            assert f"'{backend}'" in schema
        assert "served_by IN ('ollama', 'vllm')" in schema

    def test_the_two_lc_backends_are_named_and_distinct(self) -> None:
        assert LC_ANALYSIS_BACKEND == "vllm"
        assert LC_EQUIVALENCE_BACKEND == "ollama"
        assert LC_ANALYSIS_BACKEND != LC_EQUIVALENCE_BACKEND


class TestPartialRecord:
    """Partiality is a property of (condition, backend), not of the label `LC`."""

    def test_the_ollama_lc_cells_are_the_d11_partial_record(self) -> None:
        assert is_partial_record("LC", "ollama") is True

    def test_the_vllm_lc_cells_are_a_complete_arm(self) -> None:
        assert is_partial_record("LC", "vllm") is False

    def test_a_journal_with_no_backend_recorded_predates_vllm(self) -> None:
        """Every row written before `served_by` existed was served by Ollama."""
        assert is_partial_record("LC") is True

    def test_no_other_condition_is_a_partial_record(self) -> None:
        for condition in ("A", "A2", "B", "C", "D"):
            assert is_partial_record(condition, "ollama") is False
            assert is_partial_record(condition, "vllm") is False


class TestSingleBackendGuard:
    def test_a_single_backend_selection_returns_its_backend(self) -> None:
        rows = [cell("SC-001"), cell("SC-004")]
        assert assert_single_backend(rows, what="LC arm") == "ollama"

    def test_an_empty_selection_is_not_a_backend_error(self) -> None:
        assert assert_single_backend([], what="LC arm") is None

    def test_two_backends_in_one_selection_is_refused(self) -> None:
        rows = [cell("SC-001"), cell("SC-001", served_by="vllm")]
        with pytest.raises(MixedBackendError) as excinfo:
            assert_single_backend(rows, what="LC arm")
        message = str(excinfo.value)
        assert "LC arm" in message
        # The counts are in the message: a reader must be able to see which
        # stack contributed what without re-running the query.
        assert "ollama=1" in message
        assert "vllm=1" in message

    def test_the_counts_are_reported_without_raising(self) -> None:
        rows = [cell("SC-001"), cell("SC-004"), cell("SC-001", served_by="vllm")]
        assert backends_in(rows) == {"ollama": 2, "vllm": 1}

    def test_an_unknown_backend_is_refused_rather_than_counted(self) -> None:
        with pytest.raises(ValueError, match="tgi"):
            backends_in([cell("SC-001") | {"served_by": "tgi"}])


class TestPairing:
    def test_cells_pair_on_scenario_condition_and_sample(self) -> None:
        left = [cell("SC-001", sample_idx=i) for i in range(3)]
        right = [cell("SC-001", sample_idx=i, served_by="vllm") for i in range(3)]
        paired = pair_cells(left, right)
        assert paired.n_pairs == 3
        assert not paired.left_only
        assert not paired.right_only
        assert [p.key.sample_idx for p in paired.pairs] == [0, 1, 2]

    def test_unpaired_cells_are_reported_not_dropped_silently(self) -> None:
        left = [cell("SC-001"), cell("SC-004")]
        right = [cell("SC-001", served_by="vllm"), cell("SC-009", served_by="vllm")]
        paired = pair_cells(left, right)
        assert paired.n_pairs == 1
        assert [k.scenario_id for k in paired.left_only] == ["SC-004"]
        assert [k.scenario_id for k in paired.right_only] == ["SC-009"]

    def test_a_seed_mismatch_breaks_the_pairing_claim_and_raises(self) -> None:
        """The pair is only a paired observation if the cell is the same cell."""
        left = [cell("SC-001", seed=11)]
        right = [cell("SC-001", served_by="vllm", seed=12)]
        with pytest.raises(UnpairableCells, match="seed"):
            pair_cells(left, right)

    def test_a_prompt_mismatch_also_raises(self) -> None:
        left = [cell("SC-001")]
        right = [cell("SC-001", served_by="vllm", prompt_id="condition_lc.v2")]
        with pytest.raises(UnpairableCells, match="prompt_id"):
            pair_cells(left, right)

    def test_both_sides_must_be_single_backend(self) -> None:
        left = [cell("SC-001"), cell("SC-001", sample_idx=1, served_by="vllm")]
        right = [cell("SC-001", served_by="vllm")]
        with pytest.raises(MixedBackendError):
            pair_cells(left, right)

    def test_the_two_sides_must_not_be_the_same_backend(self) -> None:
        """Pairing a backend against itself is not a backend comparison."""
        left = [cell("SC-001")]
        right = [cell("SC-001")]
        with pytest.raises(UnpairableCells, match="same serving stack"):
            pair_cells(left, right)

    def test_a_duplicate_cell_on_one_side_is_refused(self) -> None:
        left = [cell("SC-001"), cell("SC-001")]
        right = [cell("SC-001", served_by="vllm")]
        with pytest.raises(UnpairableCells, match="duplicate"):
            pair_cells(left, right)

    def test_the_pairing_records_which_backend_is_which_side(self) -> None:
        paired = pair_cells([cell("SC-001")], [cell("SC-001", served_by="vllm")])
        assert paired.left_backend == "ollama"
        assert paired.right_backend == "vllm"
        assert paired.scenario_ids == ("SC-001",)
