"""Round-tripping the bank through Postgres. Requires a live database.

Excluded from `make check` by the `db` marker; run with `make test-db` once the
schema is applied.
"""

from __future__ import annotations

import pytest

from carelite.scenarios.bank import load_bank
from carelite.scenarios.load import fetch_scenarios, upsert_scenarios
from carelite.types import Split

pytestmark = pytest.mark.db


def test_upsert_writes_all_one_hundred() -> None:
    assert upsert_scenarios() == 100


def test_upsert_is_idempotent() -> None:
    upsert_scenarios()
    upsert_scenarios()
    assert len(fetch_scenarios()) == 100


def test_round_trip_preserves_every_frozen_field() -> None:
    upsert_scenarios()
    stored = {s.scenario_id: s for s in fetch_scenarios()}
    for record in load_bank():
        assert stored[record.scenario_id] == record.to_scenario()


def test_splits_survive_the_round_trip() -> None:
    upsert_scenarios()
    assert len(fetch_scenarios(Split.TRAIN)) == 40
    assert len(fetch_scenarios(Split.HOLDOUT)) == 60


def test_the_equity_subgroup_query_from_the_build_plan_returns_rows() -> None:
    # v3 Part II motivates the relational store with exactly this filter.
    upsert_scenarios()
    equity_holdout = [s for s in fetch_scenarios(Split.HOLDOUT) if s.equity_stratum]
    assert len(equity_holdout) >= 15
