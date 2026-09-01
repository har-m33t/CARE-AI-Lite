"""Which rows are an analysis arm, and the guard that stops two backends becoming one.

**The hazard this module exists for.** After D13 the `generation` table holds 219
rows carrying `condition = 'LC'`: 180 served by vLLM, which are the analysis arm,
and 39 served by Ollama, which are D11's partial record kept only as a paired
equivalence sample. `SELECT ... WHERE condition = 'LC'` returns all 219. That
selection is wrong in two independent ways at once — it pools two serving stacks
that D13 explicitly declined to pool, and it double-counts 13 of the 60 scenarios
— and nothing about the returned rows makes either visible.

So the fetchers here take `served_by` as a **required** keyword. There is no
function in this lane that returns condition LC across backends, and
`assert_single_backend` re-checks the returned rows rather than trusting the
`WHERE`, because a query is a claim about what came back and a check is the
evidence.

**Partiality moved with it.** `holdout.py` used to stamp `partial_condition:
true` on every LC row, which was correct when LC was 39 cells and is wrong now
that it is also 180. Partiality is a property of `(condition, served_by)`, not of
the label `LC`, and `is_partial_record` is the one place that says so. A journal
written before `served_by` existed has no backend field and every one of those
cells was served by Ollama, so the default preserves the old behaviour exactly.

**What a pair is.** The 39 Ollama cells and their vLLM counterparts share
scenario, condition, sample index, seed and prompt id, which is what makes them
paired observations of the same cell rather than two samples from one arm.
`pair_cells` verifies the seed and the prompt id rather than assuming them: if
they differ, the pairing claim is false and the comparison downstream would be
measuring something nobody named.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from carelite.types import Condition, Generation

__all__ = [
    "LC_ANALYSIS_BACKEND",
    "LC_EQUIVALENCE_BACKEND",
    "PARTIAL_RECORDS",
    "SERVING_BACKENDS",
    "Arm",
    "CellKey",
    "MixedBackendError",
    "PairedCell",
    "PairedCells",
    "UnpairableCells",
    "assert_single_backend",
    "backends_in",
    "fetch_arm",
    "is_partial_record",
    "lc_analysis_arm",
    "lc_equivalence_sample",
    "pair_cells",
]

#: `carelite/db/schema.sql`'s CHECK constraint, restated so a value outside it is
#: named here rather than rejected by Postgres three layers down. Widening this
#: is a schema change and the schema is frozen: it is not this lane's to edit.
SERVING_BACKENDS: tuple[str, ...] = ("ollama", "vllm")

#: The LC analysis arm (D13): 180 cells over all 60 held-out scenarios.
LC_ANALYSIS_BACKEND = "vllm"

#: The LC equivalence sample (D11): 39 cells over 13 scenarios, never randomised
#: for partial analysis. It is not an arm and is never pooled into one.
LC_EQUIVALENCE_BACKEND = "ollama"

#: `(condition, served_by)` selections that are a partial record rather than a
#: complete arm. One entry, named to its decision, so a second hole cannot be
#: added without a reader seeing why the first one is there.
PARTIAL_RECORDS: frozenset[tuple[str, str]] = frozenset(
    {(Condition.LC.value, LC_EQUIVALENCE_BACKEND)}
)


class MixedBackendError(RuntimeError):
    """A selection spans two serving stacks. D13 says do not pool; this refuses."""


class UnpairableCells(RuntimeError):
    """Two sets cannot be treated as paired observations of the same cells."""


def is_partial_record(condition: str, served_by: str | None = None) -> bool:
    """Is this `(condition, backend)` a partial record rather than a complete arm?

    `served_by=None` means a row that predates the column, which is Ollama —
    the same default `carelite.generate.store.GenerationRecord` applies and the
    same one the schema backfills to.
    """
    return (str(condition), str(served_by or "ollama")) in PARTIAL_RECORDS


def backends_in(rows: Iterable[Mapping[str, object]]) -> dict[str, int]:
    """`served_by -> count` over rows, refusing a value outside the schema's set.

    Raises:
        ValueError: on a backend the schema's CHECK constraint does not permit.
            A row carrying one means something wrote past the constraint, and
            counting it would launder that into a summary.
    """
    counts: dict[str, int] = {}
    for row in rows:
        backend = str(row.get("served_by") or "ollama")
        if backend not in SERVING_BACKENDS:
            raise ValueError(
                f"unknown serving stack {backend!r}; schema.sql permits {list(SERVING_BACKENDS)}"
            )
        counts[backend] = counts.get(backend, 0) + 1
    return dict(sorted(counts.items()))


def assert_single_backend(rows: Sequence[Mapping[str, object]], *, what: str) -> str | None:
    """The one backend these rows came from, or a refusal naming the split.

    Returns `None` for an empty selection: no rows is not a pooling error, and
    conflating the two would make "nothing matched" read as "something is
    wrong with the data".
    """
    counts = backends_in(rows)
    if not counts:
        return None
    if len(counts) > 1:
        breakdown = ", ".join(f"{k}={v}" for k, v in counts.items())
        raise MixedBackendError(
            f"{what} spans {len(counts)} serving stacks ({breakdown}). D13: the two "
            f"stacks serve different artifacts of the same model family and realised "
            f"different context packs, so they are not one arm. Select one "
            f"`served_by` value, or use the paired equivalence sample."
        )
    return next(iter(counts))


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, order=True)
class CellKey:
    """What makes two generations the same experimental cell."""

    scenario_id: str
    condition: str
    sample_idx: int


@dataclass(frozen=True, slots=True)
class PairedCell:
    """One cell observed under two serving stacks."""

    key: CellKey
    left: Mapping[str, object]
    right: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PairedCells:
    """The paired sample, plus what did not pair. Both halves are reported."""

    left_backend: str
    right_backend: str
    pairs: tuple[PairedCell, ...]
    left_only: tuple[CellKey, ...]
    right_only: tuple[CellKey, ...]

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(sorted({p.key.scenario_id for p in self.pairs}))

    @property
    def n_scenarios(self) -> int:
        return len(self.scenario_ids)


#: Fields that must agree for two rows to be the same cell under two stacks. The
#: response cannot be on this list — at temperature 0.7 two sampling
#: implementations produce different text from the same seed, and that difference
#: is the thing being measured, not a defect in the pairing.
_PAIR_IDENTITY_FIELDS = ("seed", "prompt_id")


def _key_of(row: Mapping[str, object]) -> CellKey:
    return CellKey(
        scenario_id=str(row["scenario_id"]),
        condition=str(row["condition"]),
        sample_idx=int(str(row["sample_idx"])),
    )


def _index(rows: Sequence[Mapping[str, object]], side: str) -> dict[CellKey, Mapping[str, object]]:
    out: dict[CellKey, Mapping[str, object]] = {}
    for row in rows:
        key = _key_of(row)
        if key in out:
            raise UnpairableCells(
                f"duplicate cell on the {side} side: {key.scenario_id}/{key.condition}"
                f"/sample {key.sample_idx} appears twice"
            )
        out[key] = row
    return out


def pair_cells(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
) -> PairedCells:
    """Match two single-backend selections cell by cell.

    Raises:
        MixedBackendError: if either side spans two stacks.
        UnpairableCells: if both sides are the same stack, if a cell appears
            twice on one side, or if a matched pair disagrees on seed or prompt
            id — in which case it is not the same cell and calling it a paired
            observation would be false.
    """
    left_backend = assert_single_backend(left, what="the left side of a backend pairing")
    right_backend = assert_single_backend(right, what="the right side of a backend pairing")
    if left_backend is not None and left_backend == right_backend:
        raise UnpairableCells(
            f"both sides are the same serving stack ({left_backend!r}); pairing a "
            f"backend against itself measures sampling noise, not the backend"
        )

    left_index = _index(left, "left")
    right_index = _index(right, "right")
    shared = sorted(set(left_index) & set(right_index))

    pairs: list[PairedCell] = []
    for key in shared:
        a, b = left_index[key], right_index[key]
        for field in _PAIR_IDENTITY_FIELDS:
            if a.get(field) != b.get(field):
                raise UnpairableCells(
                    f"{key.scenario_id}/{key.condition}/sample {key.sample_idx}: "
                    f"{field} differs between stacks ({a.get(field)!r} vs "
                    f"{b.get(field)!r}), so these are not the same cell"
                )
        pairs.append(PairedCell(key=key, left=a, right=b))

    return PairedCells(
        left_backend=left_backend or "",
        right_backend=right_backend or "",
        pairs=tuple(pairs),
        left_only=tuple(sorted(set(left_index) - set(right_index))),
        right_only=tuple(sorted(set(right_index) - set(left_index))),
    )


# ---------------------------------------------------------------------------
# The arm itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Arm:
    """One condition under one serving stack, with the coverage that qualifies it."""

    condition: str
    served_by: str
    split: str | None
    generations: tuple[Generation, ...]
    #: Generations the output safety gate refused (D12). Kept in the arm and
    #: flagged, never silently dropped: a missing row is indistinguishable from
    #: a cell that never ran.
    gate_blocked_ids: frozenset[str]

    @property
    def n_cells(self) -> int:
        return len(self.generations)

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(sorted({g.scenario_id for g in self.generations}))

    @property
    def n_scenarios(self) -> int:
        return len(self.scenario_ids)

    @property
    def is_partial(self) -> bool:
        return is_partial_record(self.condition, self.served_by)

    def summary(self) -> str:
        label = "PARTIAL RECORD" if self.is_partial else "arm"
        return (
            f"{self.condition} / {self.served_by}: {self.n_cells} cells over "
            f"{self.n_scenarios} scenarios ({label}), "
            f"{len(self.gate_blocked_ids)} gate-blocked"
        )


_ARM_SQL = """
SELECT g.*, sc.split AS scenario_split
FROM generation g
JOIN scenario sc USING (scenario_id)
WHERE g.condition = %(condition)s
  AND g.served_by = %(served_by)s
"""


def fetch_arm(
    *,
    condition: str,
    served_by: str,
    split: str | None = "holdout",
    unjudged_by: str | None = None,
) -> Arm:
    """Load one condition under one serving stack. `served_by` is not optional.

    Args:
        condition: the condition label, e.g. `"LC"`.
        served_by: `"ollama"` or `"vllm"`. Required, because after D13 the
            condition alone does not identify an arm.
        split: restrict to one split, or `None` for all of them.
        unjudged_by: a judge rater id; excludes generations that already carry
            that rater's aggregate row. This is the coarse resume that makes
            re-running a completed judging pass free. The judge cache resumes
            at sample granularity underneath it.
    """
    from carelite.db import connect
    from carelite.eval.judge.store import median_rater_id

    if served_by not in SERVING_BACKENDS:
        raise ValueError(
            f"unknown serving stack {served_by!r}; schema.sql permits {list(SERVING_BACKENDS)}"
        )

    sql = _ARM_SQL
    params: dict[str, object] = {"condition": condition, "served_by": served_by}
    if split is not None:
        sql += "  AND sc.split = %(split)s\n"
        params["split"] = split
    if unjudged_by is not None:
        sql += (
            "  AND NOT EXISTS (SELECT 1 FROM rubric_score rs "
            "WHERE rs.generation_id = g.generation_id "
            "AND rs.rater_type = 'llm_judge' AND rs.rater_id = %(judged_by)s)\n"
        )
        params["judged_by"] = median_rater_id(unjudged_by)
    sql += "ORDER BY g.scenario_id, g.sample_idx"

    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    # The WHERE said one backend; this proves one came back.
    assert_single_backend(rows, what=f"condition {condition} arm")

    return Arm(
        condition=condition,
        served_by=served_by,
        split=split,
        generations=tuple(
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
        ),
        gate_blocked_ids=frozenset(
            str(row["generation_id"]) for row in rows if row["gate_blocked"]
        ),
    )


def lc_analysis_arm(*, split: str | None = "holdout", unjudged_by: str | None = None) -> Arm:
    """The LC arm as D13 defines it: `served_by = 'vllm'`, and nothing else."""
    return fetch_arm(
        condition=Condition.LC.value,
        served_by=LC_ANALYSIS_BACKEND,
        split=split,
        unjudged_by=unjudged_by,
    )


def lc_equivalence_sample(*, split: str | None = "holdout") -> Arm:
    """D11's 39 Ollama LC cells. An equivalence sample, never an analysis arm."""
    return fetch_arm(
        condition=Condition.LC.value,
        served_by=LC_EQUIVALENCE_BACKEND,
        split=split,
    )
