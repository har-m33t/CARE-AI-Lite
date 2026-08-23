"""The cache is what makes an interrupted 1,080-generation run resumable."""

from __future__ import annotations

from pathlib import Path

from carelite.eval.judge import JudgeCache, LLMJudge, OptionOrder, ReplayClient, cache_key
from carelite.eval.judge.cache import CachedSample

from .conftest import RESPONSE, SCENARIO, judge_json

BASE = {
    "generation_id": "gen-0001",
    "model": "gpt-oss:20b",
    "digest": "sha256:abc",
    "prompt_version": "judge-prompt-1.0.0",
    "rubric_version": "1.0.0",
    "temperature": 0.7,
    "sample_idx": 0,
    "order": "ascending",
}


class TestCacheKey:
    def test_same_inputs_give_the_same_key(self) -> None:
        assert cache_key(**BASE) == cache_key(**BASE)

    def test_float_repr_drift_does_not_split_a_key(self) -> None:
        """0.7 and 0.7000000000000001 must not silently double an eight-hour run."""
        assert cache_key(**{**BASE, "temperature": 0.7000000000000001}) == cache_key(**BASE)

    def test_every_component_changes_the_key(self) -> None:
        variants = {
            "generation_id": "gen-0002",
            "model": "other:20b",
            "digest": "sha256:def",
            "prompt_version": "judge-prompt-2.0.0",
            "rubric_version": "1.1.0",
            "temperature": 0.0,
            "sample_idx": 1,
            "order": "descending",
        }
        base = cache_key(**BASE)
        for field, value in variants.items():
            assert cache_key(**{**BASE, field: value}) != base, field


def _sample(idx: int, raw: str = "{}") -> CachedSample:
    params = {**BASE, "sample_idx": idx}
    return CachedSample(key=cache_key(**params), seed=idx, raw_output=raw, **params)


class TestJudgeCache:
    def test_roundtrip_through_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "judge.jsonl"
        with JudgeCache(path) as cache:
            cache.put(_sample(0, '{"a": 1}'))
        reopened = JudgeCache(path)
        assert len(reopened) == 1
        assert reopened.get(_sample(0).key) is not None
        assert reopened.get(_sample(0).key).raw_output == '{"a": 1}'

    def test_missing_file_is_an_empty_cache(self, tmp_path: Path) -> None:
        assert len(JudgeCache(tmp_path / "nope.jsonl")) == 0

    def test_truncated_final_line_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        """A process killed mid-write must not cost the whole cache file."""
        path = tmp_path / "judge.jsonl"
        with JudgeCache(path) as cache:
            cache.put(_sample(0))
            cache.put(_sample(1))
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"key": "half-writ')
        reopened = JudgeCache(path)
        assert len(reopened) == 2
        assert reopened.corrupt_lines == 1

    def test_records_survive_a_kill_because_writes_are_flushed(self, tmp_path: Path) -> None:
        path = tmp_path / "judge.jsonl"
        cache = JudgeCache(path)
        cache.put(_sample(0))
        # No close(): simulates the process going away.
        assert len(JudgeCache(path)) == 1


class TestResumability:
    def _judge(self, cache: JudgeCache, outputs: list[str]) -> LLMJudge:
        return LLMJudge(
            client=ReplayClient(outputs=outputs),
            temperature=0.7,
            n_samples=5,
            order=OptionOrder.ASCENDING,
            cache=cache,
        )

    def test_a_second_run_makes_no_model_calls(self, tmp_path: Path) -> None:
        path = tmp_path / "judge.jsonl"
        outputs = [judge_json(4)] * 5

        with JudgeCache(path) as cache:
            first = self._judge(cache, outputs)
            first.score_text(
                generation_id="gen-0001", scenario_text=SCENARIO, response_text=RESPONSE
            )
            assert len(first.client.calls) == 5

        with JudgeCache(path) as cache:
            # An empty ReplayClient: any model call at all raises.
            second = self._judge(cache, [])
            result = second.score_text(
                generation_id="gen-0001", scenario_text=SCENARIO, response_text=RESPONSE
            )
        assert second.client.calls == []
        assert all(s.from_cache for s in result.samples)
        assert result.complete

    def test_resume_only_redoes_the_missing_samples(self, tmp_path: Path) -> None:
        """The realistic interruption: three of five samples landed."""
        path = tmp_path / "judge.jsonl"
        with JudgeCache(path) as cache:
            partial = self._judge(cache, [judge_json(4)] * 3)
            for idx in range(3):
                partial.judge_sample(
                    generation_id="gen-0001",
                    scenario_text=SCENARIO,
                    response_text=RESPONSE,
                    sample_idx=idx,
                )

        with JudgeCache(path) as cache:
            resumed = self._judge(cache, [judge_json(2)] * 2)
            result = resumed.score_text(
                generation_id="gen-0001", scenario_text=SCENARIO, response_text=RESPONSE
            )
        assert len(resumed.client.calls) == 2
        assert [s.from_cache for s in result.samples] == [True, True, True, False, False]
        assert result.dimensions["de"].raw_scores == (4, 4, 4, 2, 2)

    def test_a_rubric_version_change_misses_the_cache(self, tmp_path: Path) -> None:
        """A rubric edit must re-judge, not blend two rubrics in one table."""
        path = tmp_path / "judge.jsonl"
        with JudgeCache(path) as cache:
            first = self._judge(cache, [judge_json(4)] * 5)
            first.score_text(generation_id="g", scenario_text=SCENARIO, response_text=RESPONSE)

        with JudgeCache(path) as cache:
            second = self._judge(cache, [judge_json(2)] * 5)
            second.rubric_version = "9.9.9"
            second.score_text(generation_id="g", scenario_text=SCENARIO, response_text=RESPONSE)
        assert len(second.client.calls) == 5

    def test_reversed_order_is_cached_separately(self, tmp_path: Path) -> None:
        """The positional-bias arm must not be served the ascending arm's answers."""
        path = tmp_path / "judge.jsonl"
        with JudgeCache(path) as cache:
            asc = self._judge(cache, [judge_json(4)] * 5)
            asc.score_text(generation_id="g", scenario_text=SCENARIO, response_text=RESPONSE)

            desc = LLMJudge(
                client=ReplayClient(outputs=[judge_json(2)] * 5),
                temperature=0.7,
                n_samples=5,
                order=OptionOrder.DESCENDING,
                cache=cache,
            )
            result = desc.score_text(
                generation_id="g", scenario_text=SCENARIO, response_text=RESPONSE
            )
        assert len(desc.client.calls) == 5
        assert result.dimensions["de"].score == 2
