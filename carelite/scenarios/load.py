"""Load the scenario bank into the `scenario` table.

Only the eight frozen columns are written. `equity_kind`, `hard_case` and
`curator_note` stay in `scenarios/bank.jsonl` -- they are curation and review
metadata, and the schema is not mine to extend.

Two invariants are enforced here rather than trusted:

* the stratum audit passes before anything is written, so a bank with a hole in
  it never reaches Postgres and never silently produces a lopsided run;
* the held-out checksum verifies before anything is written, so a tampered
  held-out set cannot be loaded and generated against.

Run it after `make db-up`::

    .venv/bin/python -m carelite.scenarios.load
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from carelite.db import transaction
from carelite.scenarios.audit import assert_full_coverage
from carelite.scenarios.bank import CuratedScenario, load_bank
from carelite.scenarios.freeze import verify_holdout
from carelite.types import Scenario, Split

__all__ = ["fetch_scenarios", "main", "upsert_scenarios"]

_UPSERT = """
INSERT INTO scenario (
    scenario_id, text, challenge_type, emotion_intensity,
    encounter_phase, literacy_signal, equity_stratum, split
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (scenario_id) DO UPDATE SET
    text              = EXCLUDED.text,
    challenge_type    = EXCLUDED.challenge_type,
    emotion_intensity = EXCLUDED.emotion_intensity,
    encounter_phase   = EXCLUDED.encounter_phase,
    literacy_signal   = EXCLUDED.literacy_signal,
    equity_stratum    = EXCLUDED.equity_stratum,
    split             = EXCLUDED.split
"""


def upsert_scenarios(records: Sequence[CuratedScenario] | None = None) -> int:
    """Validate, then write all 100 rows. Returns the number written."""
    rows = list(load_bank()) if records is None else list(records)
    assert_full_coverage(rows)
    verify_holdout(rows)

    with transaction() as conn:
        for record in rows:
            s = record.to_scenario()
            conn.execute(
                _UPSERT,
                (
                    s.scenario_id,
                    s.text,
                    s.challenge_type,
                    s.emotion_intensity,
                    str(s.encounter_phase),
                    s.literacy_signal,
                    s.equity_stratum,
                    str(s.split),
                ),
            )
    return len(rows)


def fetch_scenarios(split: Split | None = None) -> list[Scenario]:
    """Read scenarios back out of Postgres, optionally restricted to one split."""
    sql = "SELECT * FROM scenario"
    params: tuple[object, ...] = ()
    if split is not None:
        sql += " WHERE split = %s"
        params = (str(split),)
    sql += " ORDER BY scenario_id"
    with transaction() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [Scenario.model_validate(dict(row)) for row in rows]


def main() -> int:  # pragma: no cover - exercised against a live database
    n = upsert_scenarios()
    print(f"loaded {n} scenarios into the scenario table")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
