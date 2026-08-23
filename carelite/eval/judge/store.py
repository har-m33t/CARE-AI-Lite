"""Persist judge output to `rubric_score`, and read generations back out.

Two conventions this module fixes, because the schema allows several and a
results table with mixed conventions is unanalysable:

**`sample_idx` means self-consistency sample.** The five validation samples are
stored individually at `sample_idx` 0-4 under the judge's own rater id, so the
inter-sample variance can be recomputed from the table instead of being trusted
from a report. A full-run pass is one sample and lands at `sample_idx` 0.

**The aggregate gets its own rater id.** The median across five samples is
stored under `"<rater_id>-median"` at `sample_idx` 0. It cannot share the
judge's rater id, because `rubric_score`'s unique key is
`(generation_id, rater_type, rater_id, sample_idx)` and the median would collide
with sample 0 — which, since the median often equals sample 0, would look like
it worked. Analyses that want one row per generation filter on the `-median`
rater id; analyses that want the raw samples filter it out.

Writes are `ON CONFLICT DO UPDATE`, so re-running the persistence step after an
interrupted run is a no-op rather than a duplicate-key crash.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from carelite.db import connect
from carelite.eval.judge.judge import JudgeResult
from carelite.types import Condition, Generation, RaterType, RubricScore

__all__ = [
    "MEDIAN_RATER_SUFFIX",
    "fetch_generations",
    "fetch_scenario_texts",
    "median_rater_id",
    "store_judge_result",
    "store_rubric_scores",
]

#: Appended to the judge's rater id for the aggregate row. See the module docstring.
MEDIAN_RATER_SUFFIX = "-median"


def median_rater_id(rater_id: str) -> str:
    return f"{rater_id}{MEDIAN_RATER_SUFFIX}"


_UPSERT = """
INSERT INTO rubric_score (
    generation_id, rater_type, rater_id, sample_idx,
    name, understand, respect, support, explore,
    ib, epp, de, ie, naturalness, ritualistic,
    safety_flags, evidence_spans
) VALUES (
    %(generation_id)s, %(rater_type)s, %(rater_id)s, %(sample_idx)s,
    %(name)s, %(understand)s, %(respect)s, %(support)s, %(explore)s,
    %(ib)s, %(epp)s, %(de)s, %(ie)s, %(naturalness)s, %(ritualistic)s,
    %(safety_flags)s, %(evidence_spans)s
)
ON CONFLICT (generation_id, rater_type, rater_id, sample_idx) DO UPDATE SET
    name = EXCLUDED.name,
    understand = EXCLUDED.understand,
    respect = EXCLUDED.respect,
    support = EXCLUDED.support,
    explore = EXCLUDED.explore,
    ib = EXCLUDED.ib,
    epp = EXCLUDED.epp,
    de = EXCLUDED.de,
    ie = EXCLUDED.ie,
    naturalness = EXCLUDED.naturalness,
    ritualistic = EXCLUDED.ritualistic,
    safety_flags = EXCLUDED.safety_flags,
    evidence_spans = EXCLUDED.evidence_spans
"""


def _params(score: RubricScore, sample_idx: int) -> dict[str, object]:
    return {
        "generation_id": score.generation_id,
        "rater_type": str(score.rater_type),
        "rater_id": score.rater_id,
        "sample_idx": sample_idx,
        "name": score.name,
        "understand": score.understand,
        "respect": score.respect,
        "support": score.support,
        "explore": score.explore,
        "ib": score.ib,
        "epp": score.epp,
        "de": score.de,
        "ie": score.ie,
        "naturalness": score.naturalness,
        "ritualistic": score.ritualistic,
        "safety_flags": list(score.safety_flags),
        "evidence_spans": json.dumps(score.evidence_spans),
    }


def store_rubric_scores(scores: Sequence[tuple[RubricScore, int]]) -> int:
    """Upsert `(score, sample_idx)` pairs. Returns the number of rows written."""
    if not scores:
        return 0
    with connect() as conn:
        with conn.cursor() as cur:
            for score, sample_idx in scores:
                cur.execute(_UPSERT, _params(score, sample_idx))
        conn.commit()
    return len(scores)


def store_judge_result(result: JudgeResult, *, store_samples: bool = True) -> int:
    """Persist one `JudgeResult`: the aggregate row, plus each sample.

    `store_samples=False` is for the full run, where the single sample and the
    aggregate are the same numbers and storing both would double the table for
    no information.
    """
    rater_id = result.rater_id or result.judge_model
    rows: list[tuple[RubricScore, int]] = [
        (result.to_rubric_score(rater_id=median_rater_id(rater_id)), 0)
    ]
    if store_samples:
        for sample, score in zip(
            result.samples, result.per_sample_rubric_scores(rater_id=rater_id), strict=True
        ):
            rows.append((score, sample.sample_idx))
    return store_rubric_scores(rows)


def fetch_generations(
    *,
    condition: str | None = None,
    split: str | None = None,
    limit: int | None = None,
    already_judged_by: str | None = None,
) -> list[Generation]:
    """Load generations to judge.

    `already_judged_by` excludes generations that already have an aggregate row
    from that rater id. That is a *coarse* resume — the judge cache resumes at
    sample granularity — but it is the cheap one to run before a long batch,
    and it means re-running the whole pipeline after a completed judging pass
    does no work at all.
    """
    where: list[str] = []
    params: dict[str, object] = {}
    if condition is not None:
        where.append("g.condition = %(condition)s")
        params["condition"] = condition
    if split is not None:
        where.append("sc.split = %(split)s")
        params["split"] = split
    if already_judged_by is not None:
        where.append(
            "NOT EXISTS (SELECT 1 FROM rubric_score rs "
            "WHERE rs.generation_id = g.generation_id "
            "AND rs.rater_type = 'llm_judge' AND rs.rater_id = %(judged_by)s)"
        )
        params["judged_by"] = median_rater_id(already_judged_by)

    sql = "SELECT g.* FROM generation g JOIN scenario sc USING (scenario_id)"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY g.generation_id"
    if limit is not None:
        sql += " LIMIT %(limit)s"
        params["limit"] = limit

    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [
        Generation(
            generation_id=row["generation_id"],
            scenario_id=row["scenario_id"],
            condition=Condition(row["condition"]),
            prompt_id=row["prompt_id"],
            model=row["model"],
            model_digest=row["model_digest"],
            seed=row["seed"],
            temperature=row["temperature"],
            sample_idx=row["sample_idx"],
            response=row["response"],
            latency_ms=row["latency_ms"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


def fetch_scenario_texts(scenario_ids: Iterable[str] | None = None) -> dict[str, str]:
    """`scenario_id -> text`, for the judge's context."""
    if scenario_ids is None:
        sql, params = "SELECT scenario_id, text FROM scenario", {}
    else:
        ids = list(scenario_ids)
        if not ids:
            return {}
        sql = "SELECT scenario_id, text FROM scenario WHERE scenario_id = ANY(%(ids)s)"
        params = {"ids": ids}
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return {row["scenario_id"]: row["text"] for row in rows}


def fetch_judge_scores(rater_id: str, *, aggregate_only: bool = True) -> list[RubricScore]:
    """Read judge rows back out, for re-analysis without re-judging."""
    sql = "SELECT * FROM rubric_score WHERE rater_type = 'llm_judge' AND rater_id = %(rater_id)s"
    target = median_rater_id(rater_id) if aggregate_only else rater_id
    with connect() as conn:
        rows = conn.execute(sql, {"rater_id": target}).fetchall()
    return [
        RubricScore(
            generation_id=row["generation_id"],
            rater_type=RaterType.LLM_JUDGE,
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
