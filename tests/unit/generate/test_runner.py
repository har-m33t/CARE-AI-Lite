"""The experiment plan: its size, its determinism, and what it skips."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carelite.config import get_settings, seed_for
from carelite.generate.conditions import SPEC
from carelite.generate.graph import GraphDeps, build_graph
from carelite.generate.runner import build_plan, run
from carelite.generate.store import JsonlStore
from carelite.types import Condition, Scenario

from .conftest import FakeClient

DIGESTS = {c: f"sha256:{c.value.lower()}" for c in Condition}


def _plan(scenarios: list[Scenario], samples: int = 3) -> list[Any]:
    return build_plan(scenarios, list(SPEC), samples=samples, digests=DIGESTS)


def test_the_full_plan_is_the_size_the_brief_states(holdout_like: list[Scenario]) -> None:
    """60 holdout x 6 conditions x 3 samples = 1,080."""
    settings = get_settings()
    assert settings.experiment.n_scenarios_holdout == 60
    assert settings.experiment.samples_per_cell == 3
    assert 60 * len(SPEC) * 3 == 1080
    assert len(_plan(holdout_like)) == len(holdout_like) * 6 * 3


def test_seeds_come_only_from_config_seed_for(holdout_like: list[Scenario]) -> None:
    for cell in _plan(holdout_like):
        assert cell.seed == seed_for(
            cell.scenario.scenario_id, cell.spec.condition.value, cell.sample_idx
        )


def test_the_plan_is_identical_across_processes(holdout_like: list[Scenario]) -> None:
    """`config.seed_for` uses blake2b rather than `hash()` because CPython
    randomises string hashing per process. A plan that differed between the run
    that was killed and the run that resumes it would make the cache key
    meaningless."""
    first = [c.key.as_tuple() for c in _plan(holdout_like)]
    code = (
        "import json;"
        "from carelite.generate.runner import build_plan;"
        "from carelite.generate.conditions import SPEC;"
        "from carelite.types import Condition, EncounterPhase, Scenario, Split;"
        "ss=[Scenario(scenario_id=s['scenario_id'], text=s['text'],"
        " challenge_type=s['challenge_type'], emotion_intensity=s['emotion_intensity'],"
        " encounter_phase=s['encounter_phase'], literacy_signal=s['literacy_signal'],"
        " equity_stratum=s['equity_stratum'], split=s['split'])"
        " for s in json.loads(input())];"
        "d={c: 'sha256:'+c.value.lower() for c in Condition};"
        "print(json.dumps([list(c.key.as_tuple())"
        " for c in build_plan(ss, list(SPEC), samples=3, digests=d)]))"
    )
    import subprocess
    import sys

    from carelite.config import REPO_ROOT

    payload = json.dumps([json.loads(s.model_dump_json()) for s in holdout_like])
    out = subprocess.run(
        [sys.executable, "-c", code],
        input=payload,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
        env={"PYTHONHASHSEED": "random", "PATH": "/usr/bin:/bin"},
    )
    assert [tuple(row) for row in json.loads(out.stdout)] == first


def test_the_key_is_the_six_fields_and_carries_the_digest(
    holdout_like: list[Scenario],
) -> None:
    cell = _plan(holdout_like)[0]
    key = cell.key
    assert key.prompt_id == cell.spec.prompt_id
    assert key.model_digest == DIGESTS[cell.spec.condition]
    assert key.condition == cell.spec.condition.value


def test_a_dry_run_counts_without_generating(holdout_like: list[Scenario], tmp_path: Path) -> None:
    client = FakeClient()
    report = run(
        store=JsonlStore(path=tmp_path / "g.jsonl"),
        scenarios=holdout_like,
        conditions=[Condition.A],
        samples=2,
        deps=GraphDeps(client=client),
        digests=DIGESTS,
        dry_run=True,
    )
    assert report.planned == len(holdout_like) * 2
    assert report.generated == 0
    assert client.prompts_seen == []


def _run_once(scenarios: list[Scenario], journal: Path, **kwargs: Any) -> Any:
    verdict = json.dumps({"faults": [], "verdict": "pass", "revised": ""})
    client = kwargs.pop("client", None) or FakeClient(
        reply=lambda p, i: "A steady, ordinary reply." if i % 2 == 0 else verdict
    )
    store = JsonlStore(path=journal)
    report = run(
        store=store,
        scenarios=scenarios,
        deps=GraphDeps(client=client),
        graph=build_graph(prefer_langgraph=False),
        digests=DIGESTS,
        **kwargs,
    )
    store.close()
    return report, client


def test_completed_cells_are_skipped_on_a_second_pass(
    holdout_like: list[Scenario], tmp_path: Path
) -> None:
    journal = tmp_path / "g.jsonl"
    report, client = _run_once(
        holdout_like, journal, conditions=[Condition.A, Condition.B], samples=1
    )
    assert report.generated == len(holdout_like) * 2
    assert report.skipped == 0
    calls_first = len(client.prompts_seen)

    report2, client2 = _run_once(
        holdout_like, journal, conditions=[Condition.A, Condition.B], samples=1
    )
    assert report2.generated == 0
    assert report2.skipped == len(holdout_like) * 2
    assert client2.prompts_seen == []
    assert calls_first > 0


def test_every_stored_row_carries_a_model_digest(
    holdout_like: list[Scenario], tmp_path: Path
) -> None:
    """Tags are mutable; the digest is the real identity, so it is on the row."""
    journal = tmp_path / "g.jsonl"
    _run_once(holdout_like, journal, conditions=[Condition.A], samples=1)
    for record in JsonlStore(path=journal).read_all():
        assert record.key.model_digest == DIGESTS[Condition.A]
        assert record.model == SPEC[Condition.A].model_tag


def test_a_gate_blocked_generation_is_stored_and_flagged(
    holdout_like: list[Scenario], tmp_path: Path
) -> None:
    """No row would make the run look complete while a systematic gate
    interaction sat in the gap, and would mean regenerating the cell forever."""
    journal = tmp_path / "g.jsonl"
    blocked = FakeClient(reply=lambda p, i: "You should take 20 mg twice a day.")
    report, _ = _run_once(
        holdout_like, journal, conditions=[Condition.A], samples=1, client=blocked
    )
    assert report.gate_blocked == len(holdout_like)
    assert report.generated == len(holdout_like)
    records = list(JsonlStore(path=journal).read_all())
    assert all(r.extra["output_gate_blocked"] is True for r in records)
    assert all("output.clinical_dosing" in r.extra["output_gate_flags"] for r in records)


def test_a_generation_failure_is_counted_and_left_for_the_next_run(
    holdout_like: list[Scenario], tmp_path: Path
) -> None:
    journal = tmp_path / "g.jsonl"
    report, _ = _run_once(
        holdout_like,
        journal,
        conditions=[Condition.A],
        samples=1,
        client=FakeClient(fail_with="daemon unreachable"),
    )
    assert report.failed == len(holdout_like)
    assert report.generated == 0
    assert JsonlStore(path=journal).completed_keys() == set()


def test_the_self_check_verdict_reaches_the_sidecar(
    holdout_like: list[Scenario], tmp_path: Path
) -> None:
    journal = tmp_path / "g.jsonl"
    _run_once(holdout_like, journal, conditions=[Condition.B], samples=1)
    records = list(JsonlStore(path=journal).read_all())
    assert records
    for record in records:
        assert record.extra["self_check_available"] is True
        assert record.extra["self_check_passed"] is True


def test_require_committed_refuses_an_uncommitted_prompt(
    holdout_like: list[Scenario], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from carelite.generate import prompts

    monkeypatch.setattr(prompts, "verify_committed", lambda ids=None: {"condition_a.v1": False})
    with pytest.raises(RuntimeError, match="not committed"):
        run(
            store=JsonlStore(path=tmp_path / "g.jsonl"),
            scenarios=holdout_like,
            conditions=[Condition.A],
            samples=1,
            deps=GraphDeps(client=FakeClient()),
            digests=DIGESTS,
            require_committed=True,
            dry_run=True,
        )
