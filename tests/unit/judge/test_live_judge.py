"""The only tests here that touch a model. Marked `inference`, excluded from `make check`.

Kept deliberately small: inference is hardware-serialised on one Ollama daemon
that other lanes are also using, so this is a smoke test of the seam — does
`gpt-oss:20b` return something the parser and the grounding rule can consume —
not a measurement. Every judgement about the judge's *quality* comes from the
validation study, which runs off cached output and needs no model at all.

Run with: .venv/bin/pytest -m inference tests/unit/judge -q
"""

from __future__ import annotations

import pytest

from carelite.config import get_settings
from carelite.eval.judge import LLMJudge, OllamaChatClient
from carelite.types import RUBRIC_DIMENSIONS

from .conftest import RESPONSE, SCENARIO

pytestmark = pytest.mark.inference


@pytest.fixture(scope="module")
def client() -> OllamaChatClient:
    return OllamaChatClient()


def test_judge_model_is_a_different_family_from_the_generator() -> None:
    """v3 §13 independence, asserted rather than assumed.

    Needs no model, but lives here because it is the claim the live test is
    evidence about: a judge sharing a family with the generator would be scoring
    its own dialect.
    """
    models = get_settings().models
    assert models.judge.tag == "gpt-oss:20b"
    assert models.generator.tag == "gemma4:12b"
    assert models.judge.tag.split(":")[0] != models.generator.tag.split(":")[0]


def test_live_single_pass_produces_grounded_scores(client: OllamaChatClient) -> None:
    """One real call. Asserts shape and grounding, never a particular score."""
    judge = LLMJudge.for_full_run(client)
    result = judge.score_text(
        generation_id="live-smoke-0001", scenario_text=SCENARIO, response_text=RESPONSE
    )

    assert set(result.dimensions) == set(RUBRIC_DIMENSIONS)
    assert result.judge_model == "gpt-oss:20b"
    assert result.temperature == 0.0

    scored = [k for k, d in result.dimensions.items() if d.score is not None]
    assert len(scored) >= 6, f"only {len(scored)} of 11 dimensions survived grounding"

    # Whatever it scored, every surviving span must be verbatim in the response.
    for key, span in result.evidence_spans().items():
        assert span in RESPONSE, f"{key} span is not verbatim: {span!r}"

    assert all(1 <= result.dimensions[k].score <= 5 for k in scored)
