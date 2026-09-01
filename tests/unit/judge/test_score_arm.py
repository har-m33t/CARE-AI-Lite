"""Judging one arm out of the table. The instrument must be the holdout run's.

The 939 existing scores came from `holdout.py`'s stack. If this module reached
the model any other way — a different rater id, a different regime, a different
grounding path — the new LC arm would not be comparable to any other arm, and the
comparison D13 re-opened would be the one thing this work cannot deliver. These
tests pin that it reuses the same pieces rather than restating them.

The database path is covered in `test_store_db.py`; here the row-to-metadata
translation is exercised directly, which is where a column can go missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from carelite.eval.judge import ReplayClient
from carelite.eval.judge.arms import Arm
from carelite.eval.judge.holdout import rows_for
from carelite.eval.judge.score_arm import ARM_META_SQL, meta_from_rows, score_arm
from carelite.types import Condition, Generation

from .conftest import RESPONSE, SCENARIO, judge_json


def db_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "generation_id": "gen-lc-1",
        "scenario_id": "SC-001",
        "condition": "LC",
        "sample_idx": 0,
        "model": "gemma4:12b",
        "model_digest": "vllm:google/gemma-4-12B-it@707f0a3",
        "gate_blocked": False,
        "served_by": "vllm",
        "split": "holdout",
        "fell_back_to_b": False,
        "crag_grade": None,
        "route_taken": None,
        "n_retrieved": 0,
    }
    row.update(overrides)
    return row


class TestMetadata:
    def test_a_generation_row_becomes_its_experimental_identity(self) -> None:
        meta = meta_from_rows([db_row()])["gen-lc-1"]
        assert meta.scenario_id == "SC-001"
        assert meta.condition == "LC"
        assert meta.split == "holdout"
        assert meta.served_by == "vllm"
        assert meta.model_digest.startswith("vllm:")

    def test_a_condition_that_never_retrieves_has_no_trace_and_that_is_not_missing_data(
        self,
    ) -> None:
        """LC is query-independent by D7, so it has no `retrieval_trace` row. The
        LEFT JOIN's nulls must land as "did not retrieve", not as an error."""
        meta = meta_from_rows([db_row()])["gen-lc-1"]
        assert meta.fell_back_to_b is False
        assert meta.crag_grade is None
        assert meta.n_retrieved == 0

    def test_the_gate_flag_is_read_from_the_column_not_a_sidecar(self) -> None:
        """D12 gave `gate_blocked` a column precisely so a WHERE can find it."""
        meta = meta_from_rows([db_row(gate_blocked=True)])["gen-lc-1"]
        assert meta.output_gate_blocked is True
        assert meta.output_gate_flags == ()

    def test_the_query_left_joins_the_trace(self) -> None:
        """An inner join would return nothing for the arm this module exists for."""
        assert "LEFT JOIN retrieval_trace" in ARM_META_SQL

    def test_the_query_selects_the_serving_stack(self) -> None:
        assert "g.served_by" in ARM_META_SQL


class TestEmittedRows:
    def test_a_vllm_lc_row_is_not_stamped_as_a_partial_record(self) -> None:
        """D13: the vLLM arm is 180 cells over all 60 scenarios. `holdout.rows_for`
        decides this from `(condition, served_by)`, and this is the caller that
        makes the vLLM half of that reachable."""
        meta = meta_from_rows([db_row()])
        result = _one_result()
        rows = rows_for([result], meta)
        assert rows[0]["served_by"] == "vllm"
        assert rows[0]["partial_condition"] is False

    def test_an_ollama_lc_row_stays_stamped_partial(self) -> None:
        meta = meta_from_rows([db_row(served_by="ollama")])
        rows = rows_for([_one_result()], meta)
        assert rows[0]["partial_condition"] is True


class TestScoringPath:
    def test_the_arm_is_judged_at_temperature_zero_in_one_pass(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The full-run regime, from `settings.experiment`, exactly as the 939
        existing scores were produced. A second regime here would make the LC
        arm incomparable to every other arm."""
        monkeypatch.setattr(
            "carelite.eval.judge.store.fetch_scenario_texts",
            lambda ids: {"SC-001": SCENARIO},
        )
        monkeypatch.setattr(
            "carelite.eval.judge.score_arm.fetch_arm_meta",
            lambda ids: meta_from_rows([db_row()]),
        )
        arm = Arm(
            condition="LC",
            served_by="vllm",
            split="holdout",
            generations=(_generation(),),
            gate_blocked_ids=frozenset(),
        )
        run, rows, _meta = score_arm(
            arm,
            cache_path=tmp_path / "cache.jsonl",
            client=ReplayClient(outputs=[judge_json(4)]),
        )
        assert run.n_judged == 1
        assert not run.errors
        assert rows[0]["temperature"] == 0.0
        assert rows[0]["rater_id"] == "holdout-judge"
        assert rows[0]["judge_model"] == "replay"

    def test_the_same_cache_file_makes_a_second_pass_free(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Resumability is the existing mechanism, not a new one."""
        monkeypatch.setattr(
            "carelite.eval.judge.store.fetch_scenario_texts",
            lambda ids: {"SC-001": SCENARIO},
        )
        monkeypatch.setattr(
            "carelite.eval.judge.score_arm.fetch_arm_meta",
            lambda ids: meta_from_rows([db_row()]),
        )
        arm = Arm(
            condition="LC",
            served_by="vllm",
            split="holdout",
            generations=(_generation(),),
            gate_blocked_ids=frozenset(),
        )
        cache = tmp_path / "cache.jsonl"
        first, _, _ = score_arm(arm, cache_path=cache, client=ReplayClient(outputs=[judge_json(4)]))
        assert first.n_called == 1 and first.n_from_cache == 0
        # An empty replay client: a second call would raise, so a clean run proves
        # the cache served it.
        second, _, _ = score_arm(arm, cache_path=cache, client=ReplayClient(outputs=[]))
        assert second.n_called == 0 and second.n_from_cache == 1
        assert not second.errors


def _generation() -> Generation:
    return Generation(
        generation_id="gen-lc-1",
        scenario_id="SC-001",
        condition=Condition.LC,
        prompt_id="condition_lc.v1",
        model="gemma4:12b",
        model_digest="vllm:google/gemma-4-12B-it@707f0a3",
        seed=1251920662,
        temperature=0.7,
        sample_idx=0,
        response=RESPONSE,
    )


def _one_result():  # type: ignore[no-untyped-def]
    from carelite.eval.judge import LLMJudge, OptionOrder

    judge = LLMJudge(
        client=ReplayClient(outputs=[judge_json(4)]),
        temperature=0.0,
        n_samples=1,
        order=OptionOrder.ASCENDING,
        rater_id="holdout-judge",
    )
    return judge.score(_generation(), SCENARIO)
