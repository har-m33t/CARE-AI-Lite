"""`served_by` from the client to the insert, without changing anything between.

The column exists because `model` and `model_digest` cannot tell two serving
stacks apart on their own: a GGUF and a set of HF safetensors are different
artifacts of the same model family, with different quantisation and different
sampling defaults. Pooling them into one analysis arm without checking they
agree is the confound the column makes visible.

So the value has to survive the whole path — client, graph state, record,
insert — and it has to be the client's own claim rather than something a caller
labels a row with, because a caller that has to remember eventually does not.
"""

from __future__ import annotations

from typing import Any

import pytest

from carelite.generate.graph import GraphDeps, InputPolicy, build_graph, initial_state
from carelite.generate.model import GenerationOutput
from carelite.generate.store import CacheKey, GenerationRecord, JsonlStore, PostgresStore
from carelite.safety.fencing import FencedPrompt
from carelite.types import Condition, EncounterPhase, GuidanceRequest

from .conftest import FakeClient

UTTERANCE = "I do not understand why I need another test. Nobody explains anything to me."


class VLLMFakeClient(FakeClient):
    """A `FakeClient` that claims to be the other serving stack."""

    served_by = "vllm"

    def generate(self, prompt: FencedPrompt, **kwargs: Any) -> GenerationOutput:
        out = super().generate(prompt, **kwargs)
        return GenerationOutput(
            text=out.text,
            model=out.model,
            model_digest=out.model_digest,
            latency_ms=out.latency_ms,
            num_ctx=out.num_ctx,
            prompt_chars=out.prompt_chars,
            served_by="vllm",
        )


def _key(digest: str = "sha256:aaa") -> CacheKey:
    return CacheKey(
        scenario_id="SC-001",
        condition="LC",
        prompt_id="condition_lc.v1",
        model_digest=digest,
        seed=11,
        sample_idx=0,
    )


def _run(client: Any) -> dict[str, Any]:
    request = GuidanceRequest(
        utterance=UTTERANCE,
        condition=Condition.B,
        encounter_phase=EncounterPhase.EXPLANATION,
        seed=7,
    )
    deps = GraphDeps(client=client, input_policy=InputPolicy.CURATED_BANK)
    return dict(build_graph(prefer_langgraph=False).invoke(initial_state(request, deps=deps)))


# ---------------------------------------------------------------------------
# Client to state
# ---------------------------------------------------------------------------


def test_the_graph_carries_the_clients_own_claim() -> None:
    assert _run(FakeClient())["served_by"] == "ollama"
    assert _run(VLLMFakeClient())["served_by"] == "vllm"


def test_a_failed_generation_still_records_which_stack_refused_it() -> None:
    """A halted cell is written to the report and, for a gate block, to the
    table. A row that cannot say which backend produced it is not usable in a
    backend comparison."""
    state = _run(VLLMFakeClient(fail_with="the pod is unreachable"))
    assert state["halted"] is True
    assert state["served_by"] == "vllm"


# ---------------------------------------------------------------------------
# Record to store
# ---------------------------------------------------------------------------


def test_a_record_defaults_to_ollama() -> None:
    """Every one of the 939 rows already in the database was served by Ollama,
    and the backfill in the schema says the same thing."""
    assert GenerationRecord(key=_key(), model="gemma4:12b", temperature=0.7, response="x").served_by


def test_served_by_round_trips_through_the_journal(tmp_path: Any) -> None:
    store = JsonlStore(path=tmp_path / "generations.jsonl")
    store.record(
        GenerationRecord(
            key=_key(),
            model="gemma4:12b",
            temperature=0.7,
            response="A steady reply.",
            served_by="vllm",
        )
    )
    store.close()
    (back,) = list(JsonlStore(path=tmp_path / "generations.jsonl").read_all())
    assert back.served_by == "vllm"


def test_an_older_journal_line_without_the_field_reads_back_as_ollama(tmp_path: Any) -> None:
    path = tmp_path / "generations.jsonl"
    path.write_text(
        '{"key": ["SC-001", "LC", "condition_lc.v1", "sha256:aaa", 11, 0], '
        '"model": "gemma4:12b", "temperature": 0.7, "response": "x", '
        '"latency_ms": 1, "trace": null, "extra": {}}\n',
        encoding="utf-8",
    )
    (back,) = list(JsonlStore(path=path).read_all())
    assert back.served_by == "ollama"


def test_the_insert_names_the_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """A unit-level check on the SQL rather than a live insert: `served_by` and
    `gate_blocked` are both `NOT NULL` with defaults, so an insert that omits
    them succeeds and silently writes the wrong thing."""
    statements: list[tuple[str, tuple[Any, ...]]] = []

    class _Conn:
        def execute(self, sql: str, params: tuple[Any, ...]) -> None:
            statements.append((sql, params))

    class _Transaction:
        def __enter__(self) -> _Conn:
            return _Conn()

        def __exit__(self, *exc: Any) -> bool:
            return False

    import carelite.db.connection as connection

    monkeypatch.setattr(connection, "transaction", lambda: _Transaction())

    store = PostgresStore(sidecar=None)
    store.record(
        GenerationRecord(
            key=_key(),
            model="gemma4:12b",
            temperature=0.7,
            response="A steady reply.",
            served_by="vllm",
            extra={"output_gate_blocked": True},
        )
    )
    (sql, params) = statements[0]
    assert "served_by" in sql
    assert "gate_blocked" in sql
    assert "vllm" in params
    assert True in params


# ---------------------------------------------------------------------------
# The uniqueness key
# ---------------------------------------------------------------------------


def test_a_vllm_rerun_of_an_ollama_cell_is_a_new_row_not_a_collision() -> None:
    """The v3 section 16 key includes `model_digest`, and the two backends
    cannot produce the same one: Ollama's is a `sha256:` blob hash, the vLLM
    client's is `vllm:<repo>@<revision>`. So a paired re-run under the second
    stack inserts alongside the first rather than being swallowed by the
    `ON CONFLICT DO NOTHING`."""
    from carelite.generate.store import generation_id_for

    ollama = _key("sha256:1f2e3d4c5b6a7988990a0b1c2d3e4f50")
    vllm = _key("vllm:google/gemma-4-12b-it@b6c0f4d5e9a1c37f")
    assert ollama != vllm
    assert generation_id_for(ollama) != generation_id_for(vllm)
