"""Train and holdout in one store: the two claims `runner.py` makes, asserted.

The module docstring there argues that a train run and a holdout run cannot
contaminate each other, on two grounds. Both are the kind of claim that is
obviously true right up until someone adds a field to the cache key, so neither
is left to the reader.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from carelite.config import seed_for
from carelite.generate import runner as runner_mod
from carelite.generate.conditions import spec_for
from carelite.generate.graph import GraphDeps, build_graph
from carelite.generate.model import DIGEST_UNAVAILABLE, GenerationClient
from carelite.generate.runner import (
    Cell,
    PreflightRefusal,
    RunReport,
    _build_parser,
    assert_digests_resolved,
    build_plan,
    run,
    scenarios_for_split,
)
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


def _run_by_split(journal: Path, **kwargs: Any) -> Any:
    """Drive `run` the way the CLI does — by split, not by scenario list.

    The gate deliberately spares an explicit `scenarios` list, so a helper that
    passes one (as `_run` above does) could never exercise it.
    """
    client = FakeClient(reply=lambda p, i: "A steady, ordinary reply." if i % 2 == 0 else VERDICT)
    store = JsonlStore(path=journal)
    try:
        return run(
            store=store,
            deps=GraphDeps(client=client),
            graph=build_graph(prefer_langgraph=False),
            digests=DIGESTS,
            **kwargs,
        )
    finally:
        store.close()


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


# ---------------------------------------------------------------------------
# The held-out gate
# ---------------------------------------------------------------------------


class TestWhatTheSplitFlagDoes:
    """D10 retired the registration gate. What remains is a plain selector.

    These used to assert a refusal. The decision they enforced was reversed —
    this is a local proof of concept with no audience, so the gate was blocking
    the remaining work in exchange for credibility nobody will be asked to
    extend. What is worth keeping is the evidence that removing it did not take
    anything else with it: both splits still run, both are still labelled, and
    the option-abbreviation fix outlives the flag it was written for.
    """

    def test_both_splits_generate(self, tmp_path: Path) -> None:
        holdout = _run_by_split(
            tmp_path / "h.jsonl", split=Split.HOLDOUT, conditions=[Condition.A], samples=1, limit=2
        )
        train = _run_by_split(
            tmp_path / "t.jsonl", split=Split.TRAIN, conditions=[Condition.A], samples=1, limit=2
        )
        assert holdout.generated == train.generated == 2
        assert holdout.split_counts == {"holdout": 60}
        assert train.split_counts == {"train": 40}

    def test_no_flag_asserts_something_about_a_registration(self) -> None:
        """The gate was removed rather than defeated with its own override, so
        the flag went with it: a flag nobody should ever pass is worse than no
        flag. Its name was an assertion, and passing it after D10 would have
        asserted something false in order to unblock a run."""
        options = {
            option for action in _build_parser()._actions for option in action.option_strings
        }
        assert not [o for o in options if "prereg" in o or "registration" in o]

    def test_options_still_cannot_be_abbreviated(self) -> None:
        """Kept on its own merits after the flag it was written for went away.
        argparse would otherwise let `--req` pass `--require-committed` and
        `--reg` pass `--register-prompts`, either of which changes what a run
        writes."""
        parser = _build_parser()
        for abbreviation in ("--req", "--reg", "--sp"):
            with pytest.raises(SystemExit):
                parser.parse_args([abbreviation])
        assert parser.parse_args(["--require-committed"]).require_committed is True


# ---------------------------------------------------------------------------
# The digest gate
# ---------------------------------------------------------------------------


def test_a_run_refuses_to_key_itself_on_an_unresolved_digest(tmp_path: Path) -> None:
    """Tags are mutable, so the digest is a row's only real claim about which
    weights produced it. `DIGEST_UNAVAILABLE` is an honest record of not knowing
    and a useless thing to discover after 1,080 generations: every affected cell
    has to be regenerated, because its key was never the key it should be."""
    with pytest.raises(PreflightRefusal, match="did not report a digest"):
        assert_digests_resolved({Condition.A: "sha256:real", Condition.B: DIGEST_UNAVAILABLE})

    assert_digests_resolved({Condition.A: "sha256:real"})  # the healthy case is silent


def test_the_digest_gate_is_reached_when_the_runner_resolves_them_itself(
    tmp_path: Path,
) -> None:
    """The check belongs before the first cell, not after the last. Under the
    test guardrail in `model.py` a real client resolves nothing, which is
    exactly the shape of a wedged daemon."""
    store = JsonlStore(path=tmp_path / "g.jsonl")
    with pytest.raises(PreflightRefusal, match="did not report a digest"):
        run(
            store=store,
            scenarios=scenarios_for_split(Split.TRAIN)[:1],
            conditions=[Condition.A],
            samples=1,
            deps=GraphDeps(client=GenerationClient()),
            graph=build_graph(prefer_langgraph=False),
        )
    store.close()


# ---------------------------------------------------------------------------
# Telling starvation from a crash
# ---------------------------------------------------------------------------


def test_the_progress_line_timestamps_every_cell_and_times_the_previous_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """A sibling lane's run went silent for forty minutes under daemon
    contention and was nearly killed as hung. It was queued, not hung, and it
    finished on its own — so the runner deliberately does not abort. What was
    missing was any way to tell the two apart without reconstructing it by hand
    from `wc -l` on a cache file.

    `on_cell` fires before a cell runs, so the wall clock on the last line is
    when the cell still in flight started, and the gap between two lines is how
    long the cell between them took. `tail -1` plus the current time then
    answers "stuck, or just slow?".
    """
    cells = [
        Cell(
            scenario=_scenario("SC-P01", Split.TRAIN),
            spec=spec_for(Condition.A),
            sample_idx=i,
            seed=1,
            model_digest="sha256:x",
        )
        for i in range(3)
    ]

    def fake_run(**kwargs: Any) -> RunReport:
        on_cell = kwargs["on_cell"]
        for index, cell in enumerate(cells, start=1):
            on_cell(cell, index, len(cells))
        return RunReport(planned=3, generated=3)

    monkeypatch.setattr(runner_mod, "run", fake_run)
    assert runner_mod.main(["--store", "jsonl", "--journal", str(tmp_path / "g.jsonl")]) == 0

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[")]
    assert len(lines) == 3
    for index, line in enumerate(lines, start=1):
        assert line.startswith(f"[{index}/3] ")
        # HH:MM:SS, so a reader can compare the last line against the clock.
        assert re.search(r"\[\d/3\] \d{2}:\d{2}:\d{2} SC-P01 A sample=", line)
        assert "elapsed " in line
    # The first cell has no predecessor to have timed; every later one does.
    assert "prev " not in lines[0]
    assert all("prev " in line for line in lines[1:])


def test_duration_stays_readable_across_the_scales_a_long_run_spans() -> None:
    assert runner_mod._duration(8.42) == "8.4s"
    assert runner_mod._duration(150) == "2.5m"
    assert runner_mod._duration(40 * 60) == "40.0m"
    assert runner_mod._duration(3 * 3600) == "3.0h"
