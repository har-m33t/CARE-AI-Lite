"""Mid-turn resumption, and the credential the checkpoint must not contain.

Two properties, and they pull in opposite directions. A checkpoint has to hold
enough of the turn to continue it, and it must not hold the open model client —
which under `CARELITE_BACKEND=vllm` carries a bearer token, and which cannot
survive a `SIGKILL` in any encoding anyway.

Nothing here needs Postgres: the resumption test drives the same graph through
LangGraph's in-memory saver with the same serialiser the Postgres one gets, so
what is under test is the serde and the resume path rather than a database.
"""

from __future__ import annotations

from typing import Any

import pytest

from carelite.generate import graph as graph_mod
from carelite.generate.checkpoint import (
    LIVE_DEPS_MARKER,
    LiveDepsSerde,
    UnboundLiveDeps,
    bind_live_deps,
    clear_live_deps,
    graph_checkpointer,
)
from carelite.generate.graph import GraphDeps, InputPolicy, initial_state
from carelite.types import Condition, EncounterPhase, GuidanceRequest

from .conftest import FakeClient

pytest.importorskip("langgraph", reason="the orchestration extra is not installed")

UTTERANCE = "I do not understand why I need another test. Nobody explains anything to me."
SECRET = "sk-carelite-not-a-real-token-0123456789"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _no_leftover_binding() -> Any:
    clear_live_deps()
    yield
    clear_live_deps()


# ---------------------------------------------------------------------------
# What the checkpoint holds
# ---------------------------------------------------------------------------


def test_the_live_collaborators_are_never_encoded() -> None:
    """No client state — and so no credential a client is holding — reaches the
    checkpoint, whether the deps are the value itself or nested in the input
    mapping LangGraph writes to its `__start__` channel."""
    serde = LiveDepsSerde()
    client = FakeClient()
    client.api_key = SECRET  # type: ignore[attr-defined]
    deps = GraphDeps(client=client)

    for value in (deps, {"deps": deps, "utterance": UTTERANCE}):
        _type, payload = serde.dumps_typed(value)
        assert SECRET.encode() not in payload
        assert b"FakeClient" not in payload
        assert LIVE_DEPS_MARKER.encode() in payload


def test_the_marker_reads_back_as_this_process_live_deps() -> None:
    serde = LiveDepsSerde()
    deps = GraphDeps(client=FakeClient())
    encoded = serde.dumps_typed({"deps": deps})
    bind_live_deps(deps)
    assert serde.loads_typed(encoded)["deps"] is deps


def test_reading_a_checkpoint_with_nothing_bound_raises_rather_than_guessing() -> None:
    """A default `GraphDeps` here would silently swap a fake client for a real
    one, or a configured corpus pack for a freshly built one."""
    serde = LiveDepsSerde()
    encoded = serde.dumps_typed({"deps": GraphDeps(client=FakeClient())})
    clear_live_deps()
    with pytest.raises(UnboundLiveDeps):
        serde.loads_typed(encoded)


def test_the_rest_of_the_turn_still_round_trips() -> None:
    """A resumed turn has to generate against the same request and the same
    retrieved passages, so everything that is not a live collaborator must
    survive the encoding intact."""
    serde = LiveDepsSerde()
    request = GuidanceRequest(
        utterance=UTTERANCE,
        condition=Condition.C,
        encounter_phase=EncounterPhase.EXPLANATION,
        seed=11,
    )
    assert serde.loads_typed(serde.dumps_typed(request)) == request

    from carelite.generate.conditions import spec_for

    spec = spec_for(Condition.C)
    assert serde.loads_typed(serde.dumps_typed(spec)) == spec


def test_an_unencodable_object_raises_rather_than_being_pickled() -> None:
    """`pickle_fallback` is what would write an arbitrary live object — and a
    token it happened to hold — into a database row. It is not enabled."""
    serde = LiveDepsSerde()
    with pytest.raises(Exception) as caught:
        serde.dumps_typed(lambda: None)
    assert "pickle" not in str(caught.value).lower()


# ---------------------------------------------------------------------------
# Resumption
# ---------------------------------------------------------------------------


def _saver() -> Any:
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver(serde=LiveDepsSerde())


def _request(condition: Condition = Condition.B) -> GuidanceRequest:
    return GuidanceRequest(
        utterance=UTTERANCE,
        condition=condition,
        encounter_phase=EncounterPhase.EXPLANATION,
        seed=20260901,
    )


def test_an_interrupted_turn_resumes_at_its_last_completed_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property the long-context condition needs: a turn that dies in the
    self-check does not re-run the generation it already paid for."""
    real_self_check = graph_mod.NODES["self_check"]
    attempts = {"n": 0}

    def flaky(state: Any) -> Any:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("interrupted mid-turn")
        return real_self_check(state)

    monkeypatch.setitem(graph_mod.NODES, "self_check", flaky)

    saver = _saver()
    compiled = graph_mod.to_langgraph(checkpointer=saver)
    client = FakeClient()
    deps = GraphDeps(client=client, input_policy=InputPolicy.CURATED_BANK)

    with pytest.raises(RuntimeError):
        compiled.invoke(initial_state(_request(), deps=deps), thread_id="cell-1")
    assert len(client.prompts_seen) == 1, "the generation should have run exactly once"

    final = compiled.invoke(initial_state(_request(), deps=deps), thread_id="cell-1")
    assert final["text"]
    assert len(client.prompts_seen) == 2, (
        "the resumed turn re-ran the generation instead of continuing from the checkpoint"
    )
    assert isinstance(final["latency_ms"], int)


def test_a_resumed_turn_gets_the_live_deps_back_across_a_cleared_binding() -> None:
    """Stands in for the cross-process case: a second run holds a different
    `GraphDeps`, and the checkpointed turn must be continued against that one
    rather than against a reconstructed corpse of the first."""
    saver = _saver()
    compiled = graph_mod.to_langgraph(checkpointer=saver)

    first = GraphDeps(client=FakeClient(), input_policy=InputPolicy.CURATED_BANK)
    compiled.invoke(initial_state(_request(Condition.A), deps=first), thread_id="cell-2")

    clear_live_deps()
    second_client = FakeClient(reply=lambda p, i: "A different, steady reply.")
    second = GraphDeps(client=second_client, input_policy=InputPolicy.CURATED_BANK)
    final = compiled.invoke(initial_state(_request(Condition.A), deps=second), thread_id="cell-2")
    assert final["deps"] is second
    assert final["text"] == "A different, steady reply."


def test_a_checkpointed_graph_still_agrees_with_the_executor() -> None:
    """Attaching a checkpointer must not change what a turn produces."""
    from carelite.generate.graph import build_graph

    plain = dict(
        build_graph(prefer_langgraph=False).invoke(
            initial_state(_request(), deps=GraphDeps(client=FakeClient()))
        )
    )
    checkpointed = dict(
        graph_mod.to_langgraph(checkpointer=_saver()).invoke(
            initial_state(_request(), deps=GraphDeps(client=FakeClient())),
            thread_id="cell-3",
        )
    )
    for field in sorted(set(plain) - {"deps", "started_at", "latency_ms"}):
        assert plain[field] == checkpointed[field], field


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_checkpointing_is_off_unless_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CARELITE_GRAPH_CHECKPOINT", raising=False)
    with graph_checkpointer() as saver:
        assert saver is None


def test_a_machine_with_no_database_still_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The guard the brief asks for: an unreachable database degrades to the
    per-cell resume that produced every existing row, and says so."""
    monkeypatch.setenv("CARELITE_GRAPH_CHECKPOINT", "1")
    with graph_checkpointer(database_url="postgresql://127.0.0.1:1/nope") as saver:
        assert saver is None
    assert "without mid-turn checkpointing" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Through the runner
# ---------------------------------------------------------------------------


def test_the_runner_resumes_a_failed_cell_mid_graph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The runner already retries a cell it could not finish. With a
    checkpointer it retries the part that did not finish, which for condition
    LC is the difference between one prefill of ~119,500 tokens and two."""
    from carelite.generate.runner import run
    from carelite.generate.store import JsonlStore
    from carelite.types import Scenario, Split

    real_self_check = graph_mod.NODES["self_check"]
    attempts = {"n": 0}

    def flaky(state: Any) -> Any:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("interrupted mid-turn")
        return real_self_check(state)

    monkeypatch.setitem(graph_mod.NODES, "self_check", flaky)

    scenario = Scenario(
        scenario_id="SC-T00",
        text=UTTERANCE,
        challenge_type="frustration_with_care",
        emotion_intensity=3,
        encounter_phase=EncounterPhase.EXPLANATION,
        literacy_signal="low",
        equity_stratum=False,
        split=Split.HOLDOUT,
    )
    client = FakeClient()
    deps = GraphDeps(client=client, input_policy=InputPolicy.CURATED_BANK)
    compiled = graph_mod.to_langgraph(checkpointer=_saver())
    store = JsonlStore(path=tmp_path / "generations.jsonl")

    kwargs: dict[str, Any] = {
        "store": store,
        "scenarios": [scenario],
        "conditions": [Condition.B],
        "samples": 1,
        "deps": deps,
        "graph": compiled,
        "digests": {Condition.B: "sha256:fakedigest"},
    }
    first = run(**kwargs)
    assert first.failed == 1 and first.generated == 0

    second = run(**kwargs)
    assert second.generated == 1
    assert len(client.prompts_seen) == 2, (
        "the retry re-ran the generation instead of resuming after it: "
        f"{len(client.prompts_seen)} model calls for one cell with a self-check"
    )
    store.close()
