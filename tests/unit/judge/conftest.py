"""Fixtures for the judge and human-harness tests.

No live model anywhere in this directory except `test_live_judge.py`, which is
marked `@pytest.mark.inference`. Everything else runs off `ReplayClient` and
canned model output, which is possible only because the judge caches raw output
and treats parsing, grounding and aggregation as pure functions of it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from carelite.eval.judge import LLMJudge, OptionOrder, ReplayClient
from carelite.types import RUBRIC_DIMENSIONS, Condition, Generation

#: A response with enough distinct sentences that each dimension can be given
#: its own quote, so a test can tell a grounding failure from a parse failure.
RESPONSE = (
    "You've been up all night with this, and your mother's history is sitting right on top "
    "of it. It sounds like you're frightened. Of course a spot on your lung sounds like the "
    "start of that same story. You came in anyway and you're asking about it — that takes "
    "something. I'm staying with you through this, and I'll call you with the pulmonary "
    "appointment by Thursday. Before I say anything about the scan, tell me which part is "
    "loudest right now. What do you think is going on? Two things to hold onto: it's small, "
    "and the next step is one more picture, not treatment. When you tell your husband "
    "tonight, what will you say?"
)

SCENARIO = (
    "They said the scan showed a spot on my lung. I've been up all night. My mother went "
    "through this and it was awful. I don't even know what to ask you."
)

#: One quotable phrase per dimension, each verbatim in `RESPONSE`.
SPANS: dict[str, str] = {
    "name": "It sounds like you're frightened",
    "understand": "Of course a spot on your lung sounds like the start of that same story",
    "respect": "that takes something",
    "support": "I'll call you with the pulmonary appointment by Thursday",
    "explore": "tell me which part is loudest right now",
    "ib": "Before I say anything about the scan",
    "epp": "What do you think is going on?",
    "de": "You've been up all night with this",
    "ie": "Two things to hold onto: it's small",
    "naturalness": "your mother's history is sitting right on top of it",
    "ritualistic": "When you tell your husband tonight, what will you say?",
}


def judge_json(
    scores: dict[str, int] | int = 4,
    *,
    spans: dict[str, str] | None = None,
    safety_flags: list[str] | None = None,
    omit: tuple[str, ...] = (),
) -> str:
    """Build a well-formed judge output. The shape the prompt asks for.

    `scores` may be one integer applied to every dimension, or a per-dimension
    dict. `spans` overrides individual quotes — pass a string that is not in
    `RESPONSE` to exercise the grounding rejection path.
    """
    if isinstance(scores, int):
        scores = dict.fromkeys(RUBRIC_DIMENSIONS, scores)
    spans = {**SPANS, **(spans or {})}
    payload: dict[str, Any] = {
        "scores": {
            key: {
                "score": scores[key],
                "span": spans[key],
                "rationale": f"because of {key}",
            }
            for key in RUBRIC_DIMENSIONS
            if key not in omit
        },
        "safety_flags": safety_flags or [],
    }
    return json.dumps(payload)


@pytest.fixture
def response() -> str:
    return RESPONSE


@pytest.fixture
def scenario_text() -> str:
    return SCENARIO


@pytest.fixture
def generation() -> Generation:
    return Generation(
        generation_id="gen-0001",
        scenario_id="sc-0001",
        condition=Condition.C,
        prompt_id="prompt-c-1",
        model="gemma4:12b",
        model_digest="sha256:deadbeef",
        seed=1,
        temperature=0.7,
        sample_idx=0,
        response=RESPONSE,
    )


@pytest.fixture
def single_pass_judge() -> LLMJudge:
    """One sample at temperature 0, serving one canned all-4s output."""
    return LLMJudge(
        client=ReplayClient(outputs=[judge_json(4)]),
        temperature=0.0,
        n_samples=1,
        order=OptionOrder.ASCENDING,
    )
