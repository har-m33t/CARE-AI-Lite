"""Train and holdout in one store: the two claims `runner.py` makes, asserted.

The module docstring there argues that a train run and a holdout run cannot
contaminate each other, on two grounds. Both are the kind of claim that is
obviously true right up until someone adds a field to the cache key, so neither
is left to the reader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carelite.config import seed_for
from carelite.generate.graph import GraphDeps, build_graph
from carelite.generate.runner import build_plan, run, scenarios_for_split
from carelite.generate.store import JsonlStore
from carelite.types import Condition, EncounterPhase, Scenario, Split

from .conftest import FakeClient

DIGESTS = {c: f"sha256:{c.value.lower()}" for c in Condition}
VERDICT = json.dumps({"faults": [], "verdict": "pass", "revised": ""})


def _scenario(scenario_id: str, split: Split) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        text=f"I still do not understand what the plan is, {scenario_id}.",
        challenge_type="frustration_with_care",
        emotion_intensity=3,
        encounter_phase=EncounterPhase.EXPLANATION,
        literacy_signal="low",
        equity_stratum=False,
        split=split,
    )


def _run(scenarios: list[Scenario], journal: Path, **kwargs: Any) -> Any:
    client = FakeClient(reply=lambda p, i: "A steady, ordinary reply." if i % 2 == 0 else VERDICT)
    store = JsonlStore(path=journal)
    report = run(
        store=store,
        scenarios=scenarios,
        conditions=[Condition.A],
        samples=1,
        deps=GraphDeps(client=client),
        graph=build_graph(prefer_langgraph=False),
        digests=DIGESTS,
        **kwargs,
    )
    store.close()
    return report, client


# ---------------------------------------------------------------------------
# Claim 1: the two splits cannot collide in the cache key
# ---------------------------------------------------------------------------


def test_the_bank_splits_are_a_partition_of_distinct_ids() -> None:
    """The whole no-collision argument rests on `scenario_id` being unique across
    the bank, so that is what gets checked rather than assumed."""
    train = scenarios_for_split(Split.TRAIN)
    holdout = scenarios_for_split(Split.HOLDOUT)
    train_ids = {s.scenario_id for s in train}
    holdout_ids = {s.scenario_id for s in holdout}

    assert len(train_ids) == len(train) == 40
    assert len(holdout_ids) == len(holdout) == 60
    assert train_ids.isdisjoint(holdout_ids)
    assert all(s.split is Split.TRAIN for s in train)
    assert all(s.split is Split.HOLDOUT for s in holdout)


def test_no_train_cell_can_produce_a_holdout_cell_s_key() -> None:
    conditions = list(Condition)
    train = build_plan(scenarios_for_split(Split.TRAIN), conditions, samples=3, digests=DIGESTS)
    holdout = build_plan(scenarios_for_split(Split.HOLDOUT), conditions, samples=3, digests=DIGESTS)
    train_keys = {c.key for c in train}
    holdout_keys = {c.key for c in holdout}

    assert len(train_keys) == 40 * 6 * 3
    assert len(holdout_keys) == 60 * 6 * 3 == 1080
    assert train_keys.isdisjoint(holdout_keys)


def test_a_run_of_one_split_ignores_the_other_split_s_rows(tmp_path: Path) -> None:
    """One store can hold both. A resumed run reads every key in it and simply
    finds that most are not in its plan, which is a no-op."""
    journal = tmp_path / "both.jsonl"
    train = [_scenario("SC-T01", Split.TRAIN), _scenario("SC-T02", Split.TRAIN)]
    holdout = [_scenario("SC-H01", Split.HOLDOUT), _scenario("SC-H02", Split.HOLDOUT)]

    train_report, _ = _run(train, journal)
    assert train_report.generated == 2

    holdout_report, holdout_client = _run(holdout, journal)
    assert holdout_report.skipped == 0, "the train rows must not satisfy a holdout cell"
    assert holdout_report.generated == 2
    assert len(holdout_client.prompts_seen) == 2

    again, again_client = _run(train, journal)
    assert again.generated == 0
    assert again.skipped == 2
    assert again_client.prompts_seen == []
    assert len(JsonlStore(path=journal).completed_keys()) == 4


# ---------------------------------------------------------------------------
# Claim 2: the seed is a property of the cell, not of the run
# ---------------------------------------------------------------------------


def test_the_split_is_never_an_input_to_the_seed() -> None:
    """A seed is `(scenario_id, condition, sample_idx)`. If the split reached
    `seed_for`, moving a scenario between splits would silently invalidate every
    cached generation it had."""
    scenario_id = scenarios_for_split(Split.TRAIN)[0].scenario_id
    as_train = build_plan(
        [_scenario(scenario_id, Split.TRAIN)], [Condition.C], samples=3, digests=DIGESTS
    )
    as_holdout = build_plan(
        [_scenario(scenario_id, Split.HOLDOUT)], [Condition.C], samples=3, digests=DIGESTS
    )
    assert [c.seed for c in as_train] == [c.seed for c in as_holdout]
    assert [c.key for c in as_train] == [c.key for c in as_holdout]
    assert as_train[0].seed == seed_for(scenario_id, "C", 0)


# ---------------------------------------------------------------------------
# The flag says what was asked for; the row says what it is
# ---------------------------------------------------------------------------


def test_split_and_scenarios_cannot_both_be_given() -> None:
    with pytest.raises(ValueError, match="not both"):
        run(
            store=JsonlStore(path=Path("unused.jsonl")),
            scenarios=[_scenario("SC-T01", Split.TRAIN)],
            split=Split.HOLDOUT,
            conditions=[Condition.A],
            samples=1,
            deps=GraphDeps(client=FakeClient()),
            digests=DIGESTS,
            dry_run=True,
        )


def test_each_row_records_the_split_of_its_own_scenario(tmp_path: Path) -> None:
    journal = tmp_path / "mixed.jsonl"
    mixed = [_scenario("SC-T01", Split.TRAIN), _scenario("SC-H01", Split.HOLDOUT)]
    report, _ = _run(mixed, journal)

    assert report.split_counts == {"train": 1, "holdout": 1}
    assert set(report.split_label.split("+")) == {"train", "holdout"}

    by_id = {r.key.scenario_id: r for r in JsonlStore(path=journal).read_all()}
    assert by_id["SC-T01"].extra["split"] == "train"
    assert by_id["SC-H01"].extra["split"] == "holdout"


def test_the_route_distribution_is_reported(tmp_path: Path) -> None:
    """A router that quietly sent every turn down `emotional_only` would turn
    condition C into condition B, and the run would look entirely healthy."""
    report, _ = _run([_scenario("SC-T01", Split.TRAIN)], tmp_path / "g.jsonl")
    assert sum(report.routes.values()) == report.generated
    assert "routes over" in report.route_summary()
