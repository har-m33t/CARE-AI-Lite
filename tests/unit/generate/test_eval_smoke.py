"""`make eval-smoke`: does it actually catch the breaks it exists to catch?

The smoke target's whole value is failing before a 1,080-cell run does, so the
thing worth testing is not that it runs — it is that each named breakage
*fails* it. Every test below plants one fault and asserts the audit refuses.

Lives under `tests/unit/generate/` because `carelite/eval/smoke.py` exercises
the generation seam and this lane owns that directory; `carelite/eval/`'s three
subpackages belong to other lanes and nothing here touches them.

No model, no database, no index: the graph is driven with a fake client, a
fake retrieval function and a prebuilt corpus pack, which is what lets the real
audit run against synthetic rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from carelite.eval import smoke as smoke_mod
from carelite.eval.smoke import PreflightRefusal, SmokeResult, smoke
from carelite.generate.conditions import SPEC
from carelite.generate.graph import GraphDeps, InputPolicy, build_graph
from carelite.generate.longcontext import CorpusPack
from carelite.generate.model import DIGEST_UNAVAILABLE
from carelite.retrieval import RetrievalFlags
from carelite.retrieval.pipeline import RetrievalResult
from carelite.types import (
    Condition,
    CRAGGrade,
    EvidenceTier,
    RetrievalTrace,
    RetrievedItem,
    Route,
    Split,
    Theme,
)

from .conftest import FakeClient

DIGESTS = {c: f"sha256:{c.value.lower()}" for c in Condition}
_VERDICT = json.dumps({"faults": [], "verdict": "pass", "revised": ""})


def _item(ref_id: str = "KB-001") -> RetrievedItem:
    return RetrievedItem(
        ref_id=ref_id,
        kind="kb_entry",
        text="Naming an emotion before explaining anything is what the evidence supports.",
        score=0.9,
        theme=Theme.EMPATHY,
        evidence_tier=EvidenceTier.STRONG,
        citation="Smith 2019",
    )


def _pack() -> CorpusPack:
    return CorpusPack(
        items=(_item("KB-001"), _item("KB-002")),
        n_kb_included=2,
        n_kb_total=2,
        n_chunks_included=4,
        n_chunks_total=4,
        est_tokens=500,
        budget_tokens=100_000,
    )


def _retrieve_fn(hits: bool = True, route: Route = Route.INFORMATIONAL) -> Any:
    def fn(_utterance: str, **_kwargs: Any) -> RetrievalResult:
        return RetrievalResult(
            flags=RetrievalFlags(),
            trace=RetrievalTrace(
                route=route,
                queries=["what does the evidence say"],
                retrieved=[_item()] if hits else [],
                crag_grade=CRAGGrade.RELEVANT if hits else CRAGGrade.NONE,
                latency_ms=3,
            ),
        )

    return fn


def _is_self_check(prompt: Any) -> bool:
    """One shared client serves all six conditions and only three of them make
    a second call, so the call *index* drifts out of phase. The self-check
    prompt is the one carrying a fenced `DRAFT_RESPONSE`, which is a property
    of the prompt rather than of the call order."""
    return "DRAFT_RESPONSE" in prompt.user


@dataclass
class SmokeClient(FakeClient):
    """A fake whose replies vary the way a real generator's would.

    `FakeClient.reply` sees only the prompt, and conditions A and A2 assemble
    byte-identical prompts — they share `condition_a.v1` and differ only in
    which model runs it. A fake that ignored the model tag would make them
    produce identical text, which the audit correctly calls a broken run. So
    this one answers as a function of `(model_tag, seed, prompt)`.
    """

    def generate(self, prompt: Any, **kwargs: Any) -> Any:
        out = super().generate(prompt, **kwargs)
        if _is_self_check(prompt):
            return replace(out, text=_VERDICT)
        text = f"A steady reply for {kwargs['model_tag']} seed={kwargs['seed']} ({len(prompt)})."
        return replace(out, text=text)


def _deps(**kwargs: Any) -> GraphDeps:
    client = kwargs.pop("client", None) or SmokeClient()
    return GraphDeps(
        client=client,
        input_policy=InputPolicy.CURATED_BANK,
        corpus_pack=kwargs.pop("corpus_pack", _pack()),
        retrieve_fn=kwargs.pop("retrieve_fn", _retrieve_fn()),
        **kwargs,
    )


def _smoke(tmp_path: Path, **kwargs: Any) -> SmokeResult:
    kwargs.setdefault("deps", _deps())
    kwargs.setdefault("digests", DIGESTS)
    return smoke(
        journal=tmp_path / "smoke.jsonl",
        n_scenarios=kwargs.pop("n_scenarios", 3),
        graph=build_graph(prefer_langgraph=False),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_a_healthy_pipeline_passes_all_six_conditions(tmp_path: Path) -> None:
    result = _smoke(tmp_path)
    assert result.ok, result.render()
    assert result.report.failed == 0
    assert set(result.by_condition()) == {c.value for c in SPEC}
    assert result.report.generated == 3 * len(SPEC)


def test_it_defaults_to_the_train_split(tmp_path: Path) -> None:
    """A wiring check exercises the same code path either way, so there is no
    reason for it to spend held-out scenarios."""
    result = _smoke(tmp_path)
    assert result.report.split_counts == {"train": 3 * len(SPEC)}
    assert all(r.extra["split"] == "train" for r in result.records)


def test_it_never_writes_to_the_generation_store(tmp_path: Path) -> None:
    journal = tmp_path / "smoke.jsonl"
    _smoke(tmp_path)
    assert journal.exists()
    assert journal.parent == tmp_path


def test_running_it_twice_generates_twice(tmp_path: Path) -> None:
    """The cache would otherwise turn the second run into a no-op, and a smoke
    run that skips every cell reports success without having tested anything."""
    first = _smoke(tmp_path)
    second = _smoke(tmp_path)
    assert first.report.generated == second.report.generated == 3 * len(SPEC)
    assert second.report.skipped == 0
    assert second.ok, second.render()


def test_a_dry_run_plans_the_grid_and_calls_no_model(tmp_path: Path) -> None:
    client = FakeClient()
    result = _smoke(tmp_path, deps=_deps(client=client), dry_run=True)
    assert result.report.planned == 3 * len(SPEC)
    assert client.prompts_seen == []
    assert result.ok


# ---------------------------------------------------------------------------
# What it must refuse
# ---------------------------------------------------------------------------


def test_it_fails_when_condition_c_retrieves_nothing(tmp_path: Path) -> None:
    """The measured failure this module was asked to surface: a router sending
    a turn to `emotional_only` leaves C retrieving nothing, and C is then B."""
    result = _smoke(tmp_path, deps=_deps(retrieve_fn=_retrieve_fn(hits=False)))
    assert not result.ok
    assert any("condition C retrieved no evidence" in f for f in result.failures)


def test_it_fails_when_retrieval_leaks_into_a_no_retrieval_condition(tmp_path: Path) -> None:
    result = _smoke(tmp_path)
    assert result.ok
    # Forge the leak on a stored row and re-audit: the conditions would no
    # longer differ only by configuration.
    for record in result.records:
        if record.key.condition == Condition.B.value:
            record.trace = {"retrieved_ids": ["KB-001"], "scores": [0.9]}
    forged = SmokeResult(report=result.report, records=result.records)
    smoke_mod._audit(forged, scenarios=[], samples=1)
    assert any("configured with no retrieval" in f for f in forged.failures)


def test_it_fails_when_a_generation_fails(tmp_path: Path) -> None:
    result = _smoke(tmp_path, deps=_deps(client=FakeClient(fail_with="daemon unreachable")))
    assert not result.ok
    assert result.report.generated == 0
    assert any("cell failed" in f for f in result.failures)
    assert any("produced no rows at all" in f for f in result.failures)


def test_it_fails_when_a_row_has_no_model_digest(tmp_path: Path) -> None:
    result = _smoke(tmp_path, digests={c: DIGEST_UNAVAILABLE for c in Condition})
    assert not result.ok
    assert any("no model digest" in f for f in result.failures)


def test_it_fails_when_the_long_context_pack_is_empty(tmp_path: Path) -> None:
    empty = CorpusPack(
        items=(),
        n_kb_included=0,
        n_kb_total=0,
        n_chunks_included=0,
        n_chunks_total=0,
        est_tokens=0,
        budget_tokens=100_000,
    )
    result = _smoke(tmp_path, deps=_deps(corpus_pack=empty))
    assert not result.ok
    assert any("corpus pack" in f for f in result.failures)


def test_it_fails_when_the_self_check_does_not_run_where_it_should(tmp_path: Path) -> None:
    """Draft on every call: the self-check gets a draft where it expects JSON,
    cannot parse a verdict, and reports itself unavailable."""
    plain = FakeClient(reply=lambda p, i: f"Not JSON, just a draft ({len(p)}).")
    result = _smoke(tmp_path, deps=_deps(client=plain))
    assert not result.ok
    assert any("self-check is configured on" in f for f in result.failures)


def test_it_fails_when_two_conditions_produce_identical_text(tmp_path: Path) -> None:
    """A constant reply is what a run looks like when the per-condition
    configuration never reaches the model."""
    constant = FakeClient(
        reply=lambda p, i: _VERDICT if _is_self_check(p) else "The same reply every time."
    )
    result = _smoke(tmp_path, deps=_deps(client=constant))
    assert not result.ok
    assert any("produced identical text" in f for f in result.failures)


def test_a_break_before_the_first_generation_is_a_failure_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        smoke_mod,
        "scenarios_for_split",
        lambda split: (_ for _ in ()).throw(RuntimeError("no bank")),
    )
    assert smoke_mod.main(["--journal", str(tmp_path / "s.jsonl")]) == 1


# ---------------------------------------------------------------------------
# What it must report
# ---------------------------------------------------------------------------


def test_the_route_distribution_is_in_the_rendered_report(tmp_path: Path) -> None:
    """Reported on every run, because a router collapsing C onto B leaves a run
    that finishes with the right row count, right latencies and no errors."""
    result = _smoke(tmp_path)
    rendered = result.render()
    assert "routes over" in rendered
    assert "route per scenario:" in rendered
    assert len(result.routes_by_scenario()) == 3
    assert set(result.routes_by_scenario().values()) <= {
        r.value for r in (Route.EMOTIONAL_ONLY, Route.INFORMATIONAL, Route.MIXED)
    }


def test_an_emotional_only_route_is_called_out_by_name(tmp_path: Path) -> None:
    result = _smoke(tmp_path)
    for record in result.records:
        record.extra["context_note"] = {"route": Route.EMOTIONAL_ONLY.value}
    forged = SmokeResult(report=result.report, records=result.records)
    smoke_mod._audit_routes(forged)
    assert any("emotional_only" in w and "condition C is condition B" in w for w in forged.warnings)
    assert any("every scenario routed emotional_only" in f for f in forged.failures)


def test_the_render_names_every_condition_and_its_prompt(tmp_path: Path) -> None:
    rendered = _smoke(tmp_path).render()
    for condition, spec in SPEC.items():
        assert condition.value in rendered
        assert spec.prompt_id in rendered


def test_the_default_split_is_train() -> None:
    assert smoke.__kwdefaults__["split"] is Split.TRAIN


def test_either_split_is_reachable(tmp_path: Path) -> None:
    """D10 retired the registration gate; holdout is now a plain option that
    the default simply does not choose."""
    result = _smoke(tmp_path, split=Split.HOLDOUT, n_scenarios=2)
    assert result.report.split_counts == {"holdout": 2 * len(SPEC)}
    assert all(r.extra["split"] == "holdout" for r in result.records)


def test_the_smoke_cli_still_exits_2_when_it_refuses_before_generating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exit-2 convention outlives the gate that introduced it: a script has
    to be able to tell "refused, nothing ran" from "ran, some cells failed"."""
    monkeypatch.setattr(
        smoke_mod,
        "run",
        lambda **_kw: (_ for _ in ()).throw(PreflightRefusal("no digest for gemma4:12b")),
    )
    assert smoke_mod.main(["--journal", str(tmp_path / "s.jsonl")]) == 2
