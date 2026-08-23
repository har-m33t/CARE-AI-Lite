"""The batch runner: one bad item must not end an eight-hour run."""

from __future__ import annotations

from pathlib import Path

import pytest

from carelite.eval.judge import (
    JudgeCache,
    LLMJudge,
    OptionOrder,
    ReplayClient,
    RunProgress,
    judge_generations,
)
from carelite.eval.judge.client import JudgeCallError
from carelite.types import Condition, Generation

from .conftest import RESPONSE, SCENARIO, judge_json

SCENARIOS = {"sc-0001": SCENARIO}


def gen(idx: int, scenario_id: str = "sc-0001") -> Generation:
    return Generation(
        generation_id=f"gen-{idx:04d}",
        scenario_id=scenario_id,
        condition=Condition.C,
        prompt_id="p1",
        model="gemma4:12b",
        model_digest="sha256:deadbeef",
        seed=idx,
        temperature=0.7,
        sample_idx=0,
        response=RESPONSE,
    )


def judge(outputs: list[str], cache: JudgeCache | None = None) -> LLMJudge:
    return LLMJudge(
        client=ReplayClient(outputs=outputs),
        temperature=0.0,
        n_samples=1,
        order=OptionOrder.ASCENDING,
        cache=cache,
    )


class TestJudgeGenerations:
    def test_happy_path(self) -> None:
        run = judge_generations([gen(1), gen(2)], SCENARIOS, judge([judge_json(4)] * 2))
        assert run.n_judged == 2
        assert run.errors == []
        assert run.complete_rate == 1.0
        assert run.n_called == 2 and run.n_from_cache == 0

    def test_a_model_failure_is_recorded_and_the_run_continues(self) -> None:
        """ReplayClient runs dry on the second item; the third must still be judged."""
        outputs = [judge_json(4)]
        run = judge_generations([gen(1), gen(2), gen(3)], SCENARIOS, judge(outputs))
        assert run.n_judged == 1
        assert [e.generation_id for e in run.errors] == ["gen-0002", "gen-0003"]
        assert run.errors[0].error_type == JudgeCallError.__name__

    def test_a_missing_scenario_is_an_error_not_an_empty_context(self) -> None:
        """Judging a reply without the turn it answers would produce a wrong number."""
        run = judge_generations([gen(1, "sc-9999")], SCENARIOS, judge([judge_json(4)]))
        assert run.n_judged == 0
        assert run.errors[0].error_type == "KeyError"
        assert "sc-9999" in run.errors[0].message

    def test_continue_on_error_off_reraises(self) -> None:
        with pytest.raises(JudgeCallError):
            judge_generations([gen(1)], SCENARIOS, judge([]), continue_on_error=False)

    def test_progress_callback_fires_for_failures_too(self) -> None:
        seen: list[RunProgress] = []
        judge_generations(
            [gen(1), gen(2)], SCENARIOS, judge([judge_json(4)]), on_progress=seen.append
        )
        assert [p.index for p in seen] == [1, 2]
        assert [p.ok for p in seen] == [True, False]
        assert seen[-1].fraction == 1.0

    def test_complete_rate_counts_fully_grounded_results(self) -> None:
        outputs = [judge_json(4), judge_json(4, spans={"de": "never said this"})]
        run = judge_generations([gen(1), gen(2)], SCENARIOS, judge(outputs))
        assert run.n_judged == 2
        assert run.complete_rate == 0.5

    def test_an_interrupted_run_resumes_without_recalling_the_model(self, tmp_path: Path) -> None:
        path = tmp_path / "run.jsonl"
        items = [gen(i) for i in range(1, 6)]

        with JudgeCache(path) as cache:
            # Only three outputs available: the run dies partway through.
            first = judge_generations(items, SCENARIOS, judge([judge_json(4)] * 3, cache))
        assert first.n_judged == 3 and len(first.errors) == 2

        with JudgeCache(path) as cache:
            second = judge_generations(items, SCENARIOS, judge([judge_json(4)] * 2, cache))
        assert second.n_judged == 5
        assert second.errors == []
        assert second.n_from_cache == 3
        assert second.n_called == 2

    def test_results_are_addressable_by_generation(self) -> None:
        run = judge_generations([gen(1), gen(2)], SCENARIOS, judge([judge_json(4)] * 2))
        assert sorted(run.by_generation()) == ["gen-0001", "gen-0002"]
