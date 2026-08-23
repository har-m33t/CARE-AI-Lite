"""Persist blinded assignments and returned human ratings.

One schema note that shapes this module. `rating_assignment.generation_id` is a
foreign key to `generation`, and the five calibration responses are fixtures in
`carelite.eval.rubric.calibration` — they are not generated, have no scenario
row, no prompt version and no model digest, and inserting them as generations to
satisfy the constraint would put five fabricated rows in the table every
analysis query reads.

So calibration assignments are **not persisted by default**. They live in the
`BlindedPacket` and in the ingestion report, which is where they are used: they
gate whether a rater is calibrated, and they never enter the results. The
`is_calibration` column stays available for a future amendment that gives
calibration items real generation rows; until then `store_assignments` will
refuse them rather than fail on the foreign key halfway through a batch. That
constraint is reported to the foundation lane rather than worked around here.
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

_ASSIGNMENT_UPSERT = """
INSERT INTO rating_assignment (rater_id, generation_id, display_order, blind_label, is_calibration)
VALUES (%(rater_id)s, %(generation_id)s, %(display_order)s, %(blind_label)s, %(is_calibration)s)
ON CONFLICT (rater_id, generation_id) DO UPDATE SET
    display_order = EXCLUDED.display_order,
    blind_label = EXCLUDED.blind_label,
    is_calibration = EXCLUDED.is_calibration
"""


def store_assignments(
    assignments: Sequence[Assignment],
    *,
    include_calibration: bool = False,
) -> int:
    """Write the blinding key to `rating_assignment`. Returns rows written.

    Do this **before** the packet goes out. The assignment table is the only
    record of which blind label meant which generation; a packet emailed to a
    rater before its assignments are stored is a packet whose ratings cannot be
    unblinded if the process that built it is gone.

    Args:
        include_calibration: Persist calibration rows too. Only safe once
            calibration items have `generation` rows — see the module docstring.
    """
    rows = [a for a in assignments if include_calibration or not a.is_calibration]
    if not rows:
        return 0
    with connect() as conn:
        with conn.cursor() as cur:
            for a in rows:
                cur.execute(
                    _ASSIGNMENT_UPSERT,
                    {
                        "rater_id": a.rater_id,
                        "generation_id": a.generation_id,
                        "display_order": a.display_order,
                        "blind_label": a.blind_label,
                        "is_calibration": a.is_calibration,
                    },
                )
        conn.commit()
    return len(rows)


def fetch_assignments(rater_id: str | None = None) -> list[Assignment]:
    """Read the blinding key back. This is the unblinding join, as a query."""
    sql = "SELECT * FROM rating_assignment"
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
