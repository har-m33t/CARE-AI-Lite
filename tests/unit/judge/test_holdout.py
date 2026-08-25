"""The holdout judging path: regime, resumability, and what travels with a score."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from carelite.eval.judge.cache import CachedSample, JudgeCache, cache_key
from carelite.eval.judge.holdout import (
    build_manifest,
    judge_holdout,
    load_holdout,
    rows_for,
)
from carelite.types import RUBRIC_DIMENSIONS, Condition


def _row(
    scenario: str,
    condition: str,
    sample_idx: int = 0,
    *,
    split: str = "holdout",
    fell_back: bool | None = None,
    crag: str | None = None,
) -> dict[str, Any]:
    trace = None
    if fell_back is not None:
        trace = {
            "fell_back_to_b": fell_back,
            "crag_grade": crag,
            "retrieved_ids": [] if fell_back else ["kb-x", "kb-y"],
            "route_taken": "informational",
        }
    return {
        "key": [scenario, condition, "p.v1", "sha256:gen", 42, sample_idx],
        "model": "gemma4:12b",
        "temperature": 0.7,
        "response": "I hear how frightening the waiting is. The biopsy is what answers it.",
        "latency_ms": 1000,
        "trace": trace,
        "extra": {"split": split, "condition": condition, "sample_idx": sample_idx},
    }


def _journal(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    p = tmp_path / "generations-x.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_a_train_row_in_a_holdout_journal_is_refused(tmp_path: Path) -> None:
    """Checked on the record, not the filename: a filename says what someone
    meant, a record says what the row is."""
    journal = _journal(tmp_path, [_row("SC-001", "A", split="train")])
    with pytest.raises(RuntimeError, match="expected 'holdout'"):
        load_holdout([journal])


def test_a_duplicate_generation_id_is_refused(tmp_path: Path) -> None:
    journal = _journal(tmp_path, [_row("SC-001", "A"), _row("SC-001", "A")])
    with pytest.raises(RuntimeError, match="duplicate generation_id"):
        load_holdout([journal])


def test_the_fallback_flag_survives_loading(tmp_path: Path) -> None:
    journal = _journal(
        tmp_path,
        [
            _row("SC-001", "C", fell_back=True, crag="none"),
            _row("SC-002", "C", fell_back=False, crag="relevant"),
        ],
    )
    _, _, meta = load_holdout([journal])
    by_scenario = {m.scenario_id: m for m in meta.values()}
    assert by_scenario["SC-001"].fell_back_to_b is True
    assert by_scenario["SC-001"].crag_grade == "none"
    assert by_scenario["SC-001"].n_retrieved == 0
    assert by_scenario["SC-002"].fell_back_to_b is False
    assert by_scenario["SC-002"].n_retrieved == 2


# ---------------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------------


class _Client:
    """Returns a well-formed judgement, recording the temperature it was asked for."""

    model = "gpt-oss:20b"
    digest = "sha256:judge"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def chat(self, messages: Any, *, temperature: float, seed: int | None = None) -> str:
        with self._lock:
            self.calls.append({"temperature": temperature, "seed": seed})
        return json.dumps(
            {
                "scores": {
                    k: {
                        "score": 3,
                        "span": "I hear how frightening the waiting is.",
                        "rationale": "r",
                    }
                    for k in RUBRIC_DIMENSIONS
                },
                "safety_flags": [],
            }
        )


def test_the_holdout_is_judged_single_pass_at_temperature_zero(tmp_path: Path) -> None:
    """The five-sample regime belongs to the validation subset. Running it over
    939 generations is a different project, not a longer job."""
    journal = _journal(tmp_path, [_row("SC-001", "A"), _row("SC-002", "B")])
    gens, texts, _ = load_holdout([journal])
    client = _Client()
    run = judge_holdout(gens, texts, cache_path=tmp_path / "c.jsonl", client=client, workers=1)
    assert run.n_judged == 2
    assert len(client.calls) == 2, "one call per generation, not five"
    assert {c["temperature"] for c in client.calls} == {0.0}


def test_a_resumed_run_calls_the_model_only_for_what_is_missing(tmp_path: Path) -> None:
    journal = _journal(tmp_path, [_row("SC-001", "A"), _row("SC-002", "B")])
    gens, texts, _ = load_holdout([journal])
    cache = tmp_path / "c.jsonl"

    first = _Client()
    judge_holdout(gens, texts, cache_path=cache, client=first, workers=1)
    assert len(first.calls) == 2

    second = _Client()
    run = judge_holdout(gens, texts, cache_path=cache, client=second, workers=1)
    assert second.calls == [], "everything was cached; nothing should have been called"
    assert run.n_from_cache == 2
    assert run.n_called == 0
    assert run.n_judged == 2


def test_one_failing_generation_does_not_end_the_run(tmp_path: Path) -> None:
    journal = _journal(tmp_path, [_row("SC-001", "A"), _row("SC-002", "B")])
    gens, texts, _ = load_holdout([journal])

    class _Flaky(_Client):
        def chat(self, messages: Any, *, temperature: float, seed: int | None = None) -> str:
            if len(self.calls) == 0:
                self.calls.append({"temperature": temperature, "seed": seed})
                raise ConnectionError("daemon dropped")
            return super().chat(messages, temperature=temperature, seed=seed)

    run = judge_holdout(gens, texts, cache_path=tmp_path / "c.jsonl", client=_Flaky())
    assert run.n_judged == 1
    assert len(run.errors) == 1
    assert run.errors[0]["error_type"] == "ConnectionError"


def test_a_generation_with_no_scenario_text_is_an_error_not_a_guess(tmp_path: Path) -> None:
    journal = _journal(tmp_path, [_row("SC-001", "A")])
    gens, _, _ = load_holdout([journal])
    run = judge_holdout(gens, {}, cache_path=tmp_path / "c.jsonl", client=_Client())
    assert run.n_judged == 0
    assert run.errors[0]["error_type"] == "KeyError"


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_parallel_workers_judge_every_generation_exactly_once(tmp_path: Path) -> None:
    rows = [_row(f"SC-{i:03d}", "A") for i in range(1, 41)]
    journal = _journal(tmp_path, rows)
    gens, texts, _ = load_holdout([journal])
    client = _Client()
    run = judge_holdout(gens, texts, cache_path=tmp_path / "c.jsonl", client=client, workers=8)
    assert run.n_judged == 40
    assert len(client.calls) == 40
    assert len({r.generation_id for r in run.results}) == 40


def test_the_cache_file_survives_concurrent_writers(tmp_path: Path) -> None:
    """Every line must be a whole record. A torn line is a lost sample, and at
    939 judgements on rented time that is the expensive kind of loss."""
    rows = [_row(f"SC-{i:03d}", "A") for i in range(1, 41)]
    journal = _journal(tmp_path, rows)
    gens, texts, _ = load_holdout([journal])
    cache_path = tmp_path / "c.jsonl"
    judge_holdout(gens, texts, cache_path=cache_path, client=_Client(), workers=8)

    lines = cache_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 40
    for line in lines:
        json.loads(line)  # raises if a write interleaved
    reloaded = JudgeCache(cache_path)
    assert len(reloaded) == 40
    assert reloaded.corrupt_lines == 0


def test_the_cache_is_locked_for_concurrent_puts(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "c.jsonl")

    def put(i: int) -> None:
        cache.put(
            CachedSample(
                key=cache_key(
                    generation_id=f"g{i}",
                    model="m",
                    digest="d",
                    prompt_version="p",
                    rubric_version="r",
                    temperature=0.0,
                    sample_idx=0,
                    order="ascending",
                ),
                generation_id=f"g{i}",
                model="m",
                digest="d",
                prompt_version="p",
                rubric_version="r",
                temperature=0.0,
                sample_idx=0,
                order="ascending",
                seed=i,
                raw_output="{}" * 200,
            )
        )

    threads = [threading.Thread(target=put, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    cache.close()
    assert len(JudgeCache(tmp_path / "c.jsonl")) == 50


# ---------------------------------------------------------------------------
# What travels with a score
# ---------------------------------------------------------------------------


def test_every_emitted_row_carries_the_fallback_flag(tmp_path: Path) -> None:
    """69 of 180 condition-C cells fell back to B. The flag is not derivable
    from the score, so if it does not survive here the distinction is lost and
    the attenuation is absorbed silently into the effect size."""
    journal = _journal(
        tmp_path,
        [
            _row("SC-001", "C", fell_back=True, crag="none"),
            _row("SC-002", "C", fell_back=False, crag="relevant"),
        ],
    )
    gens, texts, meta = load_holdout([journal])
    run = judge_holdout(gens, texts, cache_path=tmp_path / "c.jsonl", client=_Client())
    rows = rows_for(run.results, meta)
    assert len(rows) == 2
    assert {r["fell_back_to_b"] for r in rows} == {True, False}
    assert all("crag_grade" in r and "condition" in r for r in rows)


def test_scores_are_emitted_on_the_raw_scale(tmp_path: Path) -> None:
    """`rubric_score` stores raw. Emitting a quality column beside it is how a
    reverse coding gets applied twice."""
    journal = _journal(tmp_path, [_row("SC-001", "A")])
    gens, texts, meta = load_holdout([journal])
    run = judge_holdout(gens, texts, cache_path=tmp_path / "c.jsonl", client=_Client())
    row = rows_for(run.results, meta)[0]
    assert row["ritualistic"] == 3  # as returned, not 6 - 3
    assert not any(k.endswith("_quality") for k in row)


def test_lc_rows_are_stamped_partial(tmp_path: Path) -> None:
    journal = _journal(tmp_path, [_row("SC-001", "LC"), _row("SC-002", "A")])
    gens, texts, meta = load_holdout([journal])
    run = judge_holdout(gens, texts, cache_path=tmp_path / "c.jsonl", client=_Client())
    rows = {r["condition"]: r for r in rows_for(run.results, meta)}
    assert rows["LC"]["partial_condition"] is True
    assert rows["A"]["partial_condition"] is False


def test_the_manifest_records_the_fallback_share_and_the_lc_caveat(tmp_path: Path) -> None:
    journal = _journal(
        tmp_path,
        [
            _row("SC-001", "C", fell_back=True, crag="none"),
            _row("SC-002", "C", fell_back=False, crag="relevant"),
            _row("SC-003", "LC"),
        ],
    )
    gens, texts, meta = load_holdout([journal])
    run = judge_holdout(gens, texts, cache_path=tmp_path / "c.jsonl", client=_Client())
    manifest = build_manifest(run, rows_for(run.results, meta), meta)

    assert manifest["condition_c_fallback"]["n_fell_back_to_b"] == 1
    assert manifest["condition_c_fallback"]["share"] == 0.5
    assert manifest["condition_lc"]["partial"] is True
    assert manifest["condition_lc"]["planned_cells"] == 180
    assert manifest["reporting"]["descriptive_only"] is True
    assert manifest["regime"]["temperature"] == 0.0
    assert manifest["regime"]["samples"] == 1
    # The validation study's limits must reach whoever reads these scores.
    assert "ritualistic" in manifest["judge_caveats"]["degenerate_on_validation"]
    assert "naturalness" in manifest["judge_caveats"]["low_discrimination_on_validation"]


def test_skip_lc_is_available_but_not_the_default(tmp_path: Path) -> None:
    journal = _journal(tmp_path, [_row("SC-001", "LC"), _row("SC-002", "A")])
    gens, _, _ = load_holdout([journal])
    assert {g.condition for g in gens} == {Condition.LC, Condition.A}
    kept = [g for g in gens if g.condition is not Condition.LC]
    assert len(kept) == 1
