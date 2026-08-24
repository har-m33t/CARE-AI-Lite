"""The seam: the CLI picks up the real engine without a line changing in it."""

from __future__ import annotations

import json
from pathlib import Path

from carelite.cli.engine import resolve_engine
from carelite.generate.engine import CareliteEngine, build_engine
from carelite.generate.graph import GraphDeps, build_graph
from carelite.types import Condition, GuidanceEngine, GuidanceRequest

from .conftest import FakeClient

VERDICT = json.dumps({"faults": [], "verdict": "pass", "revised": ""})


def _engine(client: FakeClient) -> CareliteEngine:
    return CareliteEngine(deps=GraphDeps(client=client), graph=build_graph(prefer_langgraph=False))


def test_the_cli_resolves_the_real_engine_not_the_stub() -> None:
    """`carelite/cli/engine.py` imports `build_engine` by name and falls back to
    its fixture stub on ImportError. Nothing under `carelite/cli/` is edited to
    switch the system on; this asserts the switch actually happened."""
    engine = resolve_engine()
    assert isinstance(engine, CareliteEngine)
    assert type(engine).__module__ == "carelite.generate.engine"


def test_build_engine_satisfies_the_frozen_protocol() -> None:
    assert isinstance(build_engine(), GuidanceEngine)


def test_an_unseeded_request_gets_a_deterministic_seed() -> None:
    """A clinician who reruns a turn to see whether the suggestion was a fluke
    should get an honest answer, so the seed is derived rather than left to the
    model's default."""
    client = FakeClient(reply=lambda p, i: "A steady reply." if i % 2 == 0 else VERDICT)
    engine = _engine(client)
    request = GuidanceRequest(
        utterance="I am frightened about the scan result.", condition=Condition.B
    )
    engine.guide(request)
    engine.guide(request)
    seeds = [c["seed"] for c in client.calls]
    assert seeds[0] == seeds[2], "the same turn in the same condition must reuse its seed"
    assert seeds[0] != 0


def test_an_explicit_seed_is_honoured() -> None:
    client = FakeClient(reply=lambda p, i: "A steady reply." if i % 2 == 0 else VERDICT)
    _engine(client).guide(
        GuidanceRequest(utterance="Why do I need this test?", condition=Condition.A, seed=4242)
    )
    assert client.calls[0]["seed"] == 4242


def test_a_blocked_turn_returns_a_response_rather_than_raising() -> None:
    """`carelite ask` exits 3 by reading the safety verdicts off the response.
    An exception here would take the terminal down instead."""
    client = FakeClient()
    response = _engine(client).guide(
        GuidanceRequest(
            utterance="I have been thinking I want to die and I have worked out how.",
            condition=Condition.B,
        )
    )
    assert response.text == ""
    assert response.input_safety is not None
    assert response.input_safety.allowed is False
    assert response.input_safety.red_flag is True
    assert client.prompts_seen == []


def test_the_response_names_the_prompt_version_and_the_digest() -> None:
    client = FakeClient(reply=lambda p, i: "A steady reply." if i % 2 == 0 else VERDICT)
    response = _engine(client).guide(
        GuidanceRequest(utterance="What does this result mean for me?", condition=Condition.B)
    )
    assert response.prompt_version == "condition_b.v1"
    assert response.model_digest == client.digest
    assert response.condition is Condition.B


def test_one_engine_reuses_its_collaborators_across_turns(tmp_path: Path) -> None:
    """An interactive session must not rebuild the client, the embedder or the
    long-context pack on every turn."""
    client = FakeClient(reply=lambda p, i: "A steady reply." if i % 2 == 0 else VERDICT)
    engine = _engine(client)
    before = engine.deps
    engine.guide(GuidanceRequest(utterance="Why another test?", condition=Condition.A))
    engine.guide(GuidanceRequest(utterance="And what happens next?", condition=Condition.A))
    assert engine.deps is before
    assert len(client.prompts_seen) == 2
