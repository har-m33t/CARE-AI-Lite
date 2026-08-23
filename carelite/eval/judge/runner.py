"""Batch judging over many generations. Resumable, cached, and hard to kill.

The full run is 1,080 generations. At local-inference speed that is hours, and
the failure modes are boring and certain: one generation produces output the
parser cannot use, Ollama drops a connection while reloading a model, the laptop
sleeps. None of those should cost the other 1,079.

So this module has exactly two responsibilities beyond the loop:

1. **Never let one item end the run.** Every per-generation exception is caught,
   recorded as a `JudgeError` with its traceback type, and the loop continues.
   Errors are returned in the run object, not logged and forgotten — a run that
   reports "1,076 judged, 4 errors" is usable; one that reports "1,076 judged"
   is a silent hole in the results.
2. **Resume from wherever it stopped.** Resumption is a property of the cache,
   not of this module: `LLMJudge.judge_sample` consults `JudgeCache` per sample,
   so a re-run after an interruption re-judges only what is genuinely missing.
   `JudgeRun.n_from_cache` is how you check that actually happened, rather than
   discovering it by watching a progress bar move too fast.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from carelite.eval.judge.judge import JudgeResult, LLMJudge
from carelite.types import Generation

__all__ = ["JudgeError", "JudgeRun", "RunProgress", "judge_generations"]


@dataclass(frozen=True, slots=True)
class JudgeError:
    """One generation that could not be judged, and why."""

    generation_id: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class RunProgress:
    """Passed to the `on_progress` callback after each generation."""

    index: int
    total: int
    generation_id: str
    ok: bool
    elapsed_s: float

    @property
    def fraction(self) -> float:
        return self.index / self.total if self.total else 1.0


@dataclass
class JudgeRun:
    """The outcome of a batch. Errors are first-class, not a log line."""

    results: list[JudgeResult] = field(default_factory=list)
    errors: list[JudgeError] = field(default_factory=list)
    #: Samples served from the cache. Non-zero on a resumed run.
    n_from_cache: int = 0
    #: Samples that required a model call.
    n_called: int = 0
    elapsed_s: float = 0.0

    @property
    def n_judged(self) -> int:
        return len(self.results)

    @property
    def complete_rate(self) -> float:
        """Share of judged generations where all eleven dimensions survived grounding."""
        if not self.results:
            return float("nan")
        return sum(1 for r in self.results if r.complete) / len(self.results)

    def by_generation(self) -> dict[str, JudgeResult]:
        return {r.generation_id: r for r in self.results}


def judge_generations(
    generations: Iterable[Generation],
    scenario_texts: Mapping[str, str],
    judge: LLMJudge,
    *,
    on_progress: Callable[[RunProgress], None] | None = None,
    continue_on_error: bool = True,
) -> JudgeRun:
    """Judge every generation, skipping work already in the cache.

    Args:
        generations: The generations to score. Order is preserved.
        scenario_texts: `scenario_id -> patient turn`. A generation whose
            scenario is missing is recorded as an error rather than judged
            against an empty context — judging a reply with no idea what it is
            replying to would produce a number, and the number would be wrong.
        judge: A configured `LLMJudge`, usually from `for_full_run` or
            `for_validation`.
        on_progress: Called after each generation. Use it for a progress bar;
            it must not raise.
        continue_on_error: Leave on for real runs. Turn it off in tests when a
            failure should be loud.

    Returns:
        A `JudgeRun` carrying results, errors, and cache statistics.
    """
    items: Sequence[Generation] = list(generations)
    run = JudgeRun()
    started = time.monotonic()

    for index, generation in enumerate(items, start=1):
        item_started = time.monotonic()
        ok = False
        try:
            scenario_text = scenario_texts.get(generation.scenario_id)
            if scenario_text is None:
                raise KeyError(
                    f"no scenario text for {generation.scenario_id!r}; "
                    "refusing to judge a response without the turn it answers"
                )
            result = judge.score(generation, scenario_text)
            run.results.append(result)
            run.n_from_cache += sum(1 for s in result.samples if s.from_cache)
            run.n_called += sum(1 for s in result.samples if not s.from_cache)
            ok = True
        except Exception as exc:
            run.errors.append(
                JudgeError(
                    generation_id=generation.generation_id,
                    error_type=type(exc).__name__,
                    message=str(exc)[:500],
                )
            )
            if not continue_on_error:
                raise
        finally:
            if on_progress is not None:
                on_progress(
                    RunProgress(
                        index=index,
                        total=len(items),
                        generation_id=generation.generation_id,
                        ok=ok,
                        elapsed_s=time.monotonic() - item_started,
                    )
                )

    run.elapsed_s = time.monotonic() - started
    return run
