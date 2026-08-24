"""Persist blinded assignments and returned human ratings.

One schema note that shapes this module. `rating_assignment` names one of two
kinds of target in two separate columns. `generation_id` is a foreign key to
`generation` and carries study responses. `calibration_id` is a plain nullable
text column, deliberately *not* a foreign key, and carries the five calibration
responses — which are fixtures in `carelite.eval.rubric.calibration`, not
generated: no scenario row, no prompt version, no model digest. Giving them
`generation` rows to satisfy an FK would put five fabricated rows in the table
every analysis query reads, which is why the column was added instead.

Three CHECK constraints hold that shape together, and this module writes to
satisfy them rather than defending against them:

- `rating_assignment_one_target` — exactly one of the two ids is set;
- `rating_assignment_calibration_flag_agrees` — `is_calibration` is true for
  precisely the rows with a `calibration_id`;
- `UNIQUE (rater_id, generation_id)` and `UNIQUE (rater_id, calibration_id)` —
  which are two separate constraints, so the two kinds need two upserts. A
  single statement cannot name both as its conflict target.

Calibration assignments are now persisted by default. The earlier behaviour —
holding them out of the table — was a workaround for the FK that no longer
exists, and holding them out cost something real: a rater's packet could not be
rebuilt from the database alone, so a lost export meant a lost calibration
record. What must still never happen is a calibration id entering the *results*,
and that is enforced further up, in `ingest_ratings` and `unblind`, which route
calibration ratings to the ingestion report and never to `rubric_score`.
"""

from __future__ import annotations

from collections.abc import Sequence

from carelite.db import connect
from carelite.eval.human.blinding import Assignment
from carelite.eval.judge.store import store_rubric_scores
from carelite.types import RubricScore

__all__ = [
    "fetch_assignments",
    "fetch_human_scores",
    "store_assignments",
    "store_human_scores",
]

#: Study responses. `calibration_id` is left NULL and `is_calibration` FALSE, so
#: both CHECK constraints hold by construction rather than by hope.
_ASSIGNMENT_UPSERT = """
INSERT INTO rating_assignment
    (rater_id, generation_id, calibration_id, display_order, blind_label, is_calibration)
VALUES (%(rater_id)s, %(generation_id)s, NULL, %(display_order)s, %(blind_label)s, FALSE)
ON CONFLICT (rater_id, generation_id) DO UPDATE SET
    display_order = EXCLUDED.display_order,
    blind_label = EXCLUDED.blind_label
"""

#: Calibration fixtures. The mirror image: `generation_id` NULL, `is_calibration`
#: TRUE, and a different conflict target because the two UNIQUE constraints are
#: separate and one statement can only name one of them.
_CALIBRATION_UPSERT = """
INSERT INTO rating_assignment
    (rater_id, generation_id, calibration_id, display_order, blind_label, is_calibration)
VALUES (%(rater_id)s, NULL, %(calibration_id)s, %(display_order)s, %(blind_label)s, TRUE)
ON CONFLICT (rater_id, calibration_id) DO UPDATE SET
    display_order = EXCLUDED.display_order,
    blind_label = EXCLUDED.blind_label
"""


def store_assignments(
    assignments: Sequence[Assignment],
    *,
    include_calibration: bool = True,
) -> int:
    """Write the blinding key to `rating_assignment`. Returns rows written.

    Do this **before** the packet goes out. The assignment table is the only
    record of which blind label meant which generation; a packet emailed to a
    rater before its assignments are stored is a packet whose ratings cannot be
    unblinded if the process that built it is gone.

    Study rows and calibration rows take different statements, because they land
    in different columns and collide on different unique constraints. Which one
    an assignment takes is decided by `is_calibration`, which `Assignment`
    already guarantees agrees with the id it carries — so this function does not
    re-check it, and a hand-built assignment that disagreed would have failed at
    construction.

    Args:
        include_calibration: Persist calibration rows too. Default on; turn it
            off only to write the study half alone.
    """
    rows = [a for a in assignments if include_calibration or not a.is_calibration]
    if not rows:
        return 0
    with connect() as conn:
        with conn.cursor() as cur:
            for a in rows:
                if a.is_calibration:
                    cur.execute(
                        _CALIBRATION_UPSERT,
                        {
                            "rater_id": a.rater_id,
                            "calibration_id": a.calibration_id,
                            "display_order": a.display_order,
                            "blind_label": a.blind_label,
                        },
                    )
                else:
                    cur.execute(
                        _ASSIGNMENT_UPSERT,
                        {
                            "rater_id": a.rater_id,
                            "generation_id": a.generation_id,
                            "display_order": a.display_order,
                            "blind_label": a.blind_label,
                        },
                    )
        conn.commit()
    return len(rows)


def fetch_assignments(rater_id: str | None = None) -> list[Assignment]:
    """Read the blinding key back. This is the unblinding join, as a query.

    Columns are named rather than `SELECT *`. The star is what let this reader
    keep working, silently, through the schema amendment that added
    `calibration_id`: it returned rows that simply had no calibration in them and
    nothing raised. Naming the columns means a future amendment breaks the read
    instead of quietly narrowing it.
    """
    sql = (
        "SELECT rater_id, generation_id, calibration_id, display_order, blind_label, "
        "is_calibration FROM rating_assignment"
    )
    params: dict[str, object] = {}
    if rater_id is not None:
        sql += " WHERE rater_id = %(rater_id)s"
        params["rater_id"] = rater_id
    sql += " ORDER BY rater_id, display_order"
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        Assignment(
            rater_id=row["rater_id"],
            generation_id=row["generation_id"],
            display_order=row["display_order"],
            blind_label=row["blind_label"],
            is_calibration=row["is_calibration"],
            calibration_id=row["calibration_id"],
        )
        for row in rows
    ]


def store_human_scores(scores: Sequence[RubricScore]) -> int:
    """Write validated human ratings to `rubric_score`.

    `sample_idx` is 0 for every human row: the column exists for the judge's
    self-consistency samples, and a human rating a response twice is a
    *different rater id* (the test-retest convention), not a second sample of
    the same one. Storing a retest as `sample_idx=1` would make it invisible to
    every query that filters on the rater.
    """
    return store_rubric_scores([(score, 0) for score in scores])


def fetch_human_scores(rater_id: str | None = None) -> list[RubricScore]:
    """Read human ratings back, for reliability analysis."""
    from carelite.types import RaterType

    sql = "SELECT * FROM rubric_score WHERE rater_type = 'human'"
    params: dict[str, object] = {}
    if rater_id is not None:
        sql += " AND rater_id = %(rater_id)s"
        params["rater_id"] = rater_id
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        RubricScore(
            generation_id=row["generation_id"],
            rater_type=RaterType.HUMAN,
            rater_id=row["rater_id"],
            name=row["name"],
            understand=row["understand"],
            respect=row["respect"],
            support=row["support"],
            explore=row["explore"],
            ib=row["ib"],
            epp=row["epp"],
            de=row["de"],
            ie=row["ie"],
            naturalness=row["naturalness"],
            ritualistic=row["ritualistic"],
            safety_flags=list(row["safety_flags"] or []),
            evidence_spans=dict(row["evidence_spans"] or {}),
        )
        for row in rows
    ]
