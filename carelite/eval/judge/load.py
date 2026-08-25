"""Load a JSONL judge-score journal into the `rubric_score` table.

    python -m carelite.eval.judge.load runs/judge-holdout/rubric_scores.jsonl
    python -m carelite.eval.judge.load runs/judge-holdout --dry-run

**Why this exists.** `carelite.eval.judge.holdout` writes its scores to
`rubric_scores.jsonl` and has no Postgres path; `carelite.eval.judge.store` can
write `rubric_score` but is only ever called with live `JudgeResult` objects.
Nothing carried the completed holdout judging run from the first into the second,
so `rubric_score` sat empty while 939 judged rows sat in a file, and every
downstream step -- `carelite.stats`, the figures, `make reproduce` -- reads the
table. This module is that bridge. It is the score-side twin of
`carelite.generate.load` and deliberately follows its shape.

**The rater id is a convention, and the convention decides whether the analysis
sees anything at all.** `store.py` fixes it: per-sample rows go under the judge's
own rater id at `sample_idx` 0..4, and the one-row-per-generation aggregate goes
under `"<rater_id>-median"`, because `rubric_score`'s unique key would otherwise
make the aggregate collide with sample 0. `carelite.stats.data` selects judge
rows with `rater_id LIKE '%-median'` for exactly that reason. This loader
therefore writes the aggregate id by default: the holdout run is **single-pass at
temperature 0**, which is the case `store_judge_result(store_samples=False)`
exists for -- the single sample and the aggregate are the same numbers, and
storing both would double the table for no information. The `-median` suffix
names the aggregate *partition*, not an averaging operation; the median of one
sample is that sample. `--no-aggregate-suffix` writes the file's plain rater id
instead, for a caller who wants the samples partition, and is the wrong choice
for a full run because the analysis will not see the rows.

**Incomplete rows land with NULLs.** 50 of the 939 holdout rows are missing at
least one dimension, because the judge rejected an ungrounded score rather than
keeping it (v3 §13). The schema's `BETWEEN 1 AND 5` CHECKs permit NULL, so the
absence is representable and is written as an absence. Dropping the row and
zero-filling the dimension are both wrong in ways that are invisible downstream:
one shrinks n silently, the other invents data at the bottom of the scale. The
file's `complete` and `n_dimensions_scored` are checked against the actual null
count -- a disagreement means something upstream already dropped or filled -- and
survive into the report and the sidecar, since the table has no column for them.

**Gate-blocked and LC rows load.** 17 generations were refused by the output
safety gate (D12) and 39 LC cells are a partial record over 13 of 60 scenarios
(D11). Both are loaded and both are reported. Excluding them is the analysis's
job, through a `WHERE` it can state and defend; a loader that silently filtered
would leave the analysis unable to tell an excluded cell from one that never ran.

**It refuses rather than lands a corrupted table.** A `generation_id` with no
`generation` row, two rows claiming one `(generation_id, rater_type, rater_id,
sample_idx)` with different scores, a dimension outside 1-5, a `rater_type`
outside the schema's CHECK set, a score with no evidence span, a row whose
identity metadata disagrees with the `generation` row it points at. Every check
runs over the whole input before a single row is written, and a failure names
every offending record rather than the first one.

**Writes are `ON CONFLICT DO UPDATE`**, sharing `store.UPSERT_SQL` rather than
restating it, and are committed in batches, so an interrupted load resumes by
being run again. That also means re-running with a *different* judge run under
the same rater id would overwrite silently, so the report says up front how many
target rows already exist and `--dry-run` shows it before anything is written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

from carelite.config import get_settings
from carelite.eval.judge.store import (
    MEDIAN_RATER_SUFFIX,
    UPSERT_SQL,
    median_rater_id,
    upsert_params,
)
from carelite.types import RUBRIC_DIMENSIONS, Condition, RaterType, RubricScore

__all__ = [
    "PARTIAL_CONDITIONS",
    "LoadReport",
    "ScoreRecord",
    "ScoreRefusal",
    "collect",
    "default_sidecar_path",
    "incomplete_generation_ids",
    "load_score_journals",
    "load_score_metadata",
    "read_journal",
    "resolve_rater_id",
]

#: Conditions that are a partial record rather than a complete condition (D11).
PARTIAL_CONDITIONS = frozenset({Condition.LC.value})

#: `rubric_score.rater_type` carries a CHECK constraint. Validating here means a
#: bad value is named with its file and line instead of aborting a batch.
RATER_TYPES = frozenset(r.value for r in RaterType)

#: How many individual problems a refusal lists before it stops enumerating.
MAX_REPORTED_PROBLEMS = 40

#: `RETURNING (xmax = 0)` distinguishes an insert from an update on an upsert:
#: a freshly inserted tuple has `xmax` 0, an updated one carries the deleting
#: transaction id. Without it the loader could only report "n rows touched",
#: which cannot tell a first load from an overwrite of somebody else's run.
_UPSERT_RETURNING = UPSERT_SQL + "\nRETURNING (xmax = 0) AS inserted"


class ScoreRefusal(RuntimeError):
    """The input would land a table nobody could trust. Nothing was written.

    Raised only from the validation passes, which all run before the first
    insert, so a refusal always means the database is exactly as it was.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        shown = self.problems[:MAX_REPORTED_PROBLEMS]
        rest = len(self.problems) - len(shown)
        body = "\n".join(f"  - {p}" for p in shown)
        if rest > 0:
            body += f"\n  - ... and {rest} more"
        super().__init__(f"refusing to load {len(self.problems)} problem(s):\n{body}")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    """One journal line: the score to write, plus everything the table cannot hold.

    `score.rater_id` is the id the row will be written under, which is not
    necessarily the id in the file -- see `resolve_rater_id`. `source_rater_id`
    keeps the file's own value so the report can say what was rewritten.
    """

    score: RubricScore
    sample_idx: int
    source_rater_id: str
    origin: str
    meta: Mapping[str, Any]

    @property
    def key(self) -> tuple[str, str, str, int]:
        """The `rubric_score` unique key this row will occupy."""
        return (
            self.score.generation_id,
            str(self.score.rater_type),
            self.score.rater_id,
            self.sample_idx,
        )

    @property
    def scored_dimensions(self) -> list[str]:
        return [d for d in RUBRIC_DIMENSIONS if getattr(self.score, d) is not None]

    @property
    def null_dimensions(self) -> list[str]:
        return [d for d in RUBRIC_DIMENSIONS if getattr(self.score, d) is None]


def resolve_rater_id(file_rater_id: str, *, aggregate: bool, override: str | None = None) -> str:
    """The rater id a row is written under. See the module docstring.

    `aggregate=True` appends `store.MEDIAN_RATER_SUFFIX` unless the id already
    carries it, so running the loader over its own output's convention is a
    no-op rather than producing `holdout-judge-median-median`.
    """
    base = override if override is not None else file_rater_id
    if not aggregate:
        return base
    return base if base.endswith(MEDIAN_RATER_SUFFIX) else median_rater_id(base)


def _as_int(value: Any) -> int | None:
    """An int, or None if the value is not one. `bool` is not an int here.

    `isinstance(True, int)` is True in Python, so a JSON `true` in a dimension
    column would otherwise sail through as the score 1 -- which is inside the
    schema's CHECK and therefore invisible once written.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def read_journal(
    path: Path, *, aggregate: bool = True, rater_id: str | None = None
) -> Iterator[ScoreRecord]:
    """Parse every line of a score journal, strictly.

    A malformed line means the file is not the complete record of the judging
    run, and silently loading n-1 rows is precisely the failure this module
    exists to refuse -- so a bad line raises rather than being skipped.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            origin = f"{path}:{lineno}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScoreRefusal([f"{origin}: not valid JSON ({exc.msg})"]) from exc
            if not isinstance(obj, dict):
                raise ScoreRefusal([f"{origin}: expected a JSON object"])
            yield _to_record(obj, origin=origin, aggregate=aggregate, rater_id=rater_id)


def _to_record(
    obj: Mapping[str, Any], *, origin: str, aggregate: bool, rater_id: str | None
) -> ScoreRecord:
    """Build a `ScoreRecord` without validating it. `_check_internal` does that.

    Parsing is deliberately permissive so that a wrong value reaches the checks
    and is reported with its file and line, rather than raising a pydantic error
    naming a field and nothing else.
    """
    file_rater_id = str(obj.get("rater_id") or "")
    rater_type_raw = str(obj.get("rater_type") or "")
    dimensions = {d: _as_int(obj.get(d)) for d in RUBRIC_DIMENSIONS}
    spans_raw = obj.get("evidence_spans")
    spans = {str(k): str(v) for k, v in spans_raw.items()} if isinstance(spans_raw, dict) else {}
    flags_raw = obj.get("safety_flags")
    flags = [str(f) for f in flags_raw] if isinstance(flags_raw, list) else []

    score = RubricScore(
        generation_id=str(obj.get("generation_id") or ""),
        # An unknown rater_type must survive parsing to be *reported*, so the
        # enum is only consulted when the value is one it holds.
        rater_type=(
            RaterType(rater_type_raw) if rater_type_raw in RATER_TYPES else RaterType.LLM_JUDGE
        ),
        rater_id=resolve_rater_id(file_rater_id, aggregate=aggregate, override=rater_id),
        safety_flags=flags,
        evidence_spans=spans,
        **dimensions,
    )
    meta = {k: v for k, v in obj.items() if k not in RUBRIC_DIMENSIONS}
    meta["_raw_rater_type"] = rater_type_raw
    meta["_raw_dimensions"] = {d: obj.get(d) for d in RUBRIC_DIMENSIONS}
    meta["_raw_spans"] = spans_raw
    meta["_raw_flags"] = flags_raw
    return ScoreRecord(
        score=score,
        sample_idx=int(obj["sample_idx"]) if isinstance(obj.get("sample_idx"), int) else -1,
        source_rater_id=file_rater_id,
        origin=origin,
        meta=meta,
    )


def expand_paths(paths: Iterable[Path | str]) -> list[Path]:
    """Accept files or directories; a directory contributes its `*.jsonl`.

    Duplicate paths collapse, because passing a glob and the directory that
    contains it is an easy way to name the same file twice, and calling that a
    duplicate-key collision would be refusing the user's shell.
    """
    seen: dict[Path, None] = {}
    for entry in paths:
        p = Path(entry)
        if p.is_dir():
            for child in sorted(p.glob("*.jsonl")):
                seen.setdefault(child.resolve(), None)
        else:
            if not p.exists():
                raise ScoreRefusal([f"{p}: no such file"])
            seen.setdefault(p.resolve(), None)
    if not seen:
        raise ScoreRefusal(["no score journal files found"])
    return list(seen)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_internal(rec: ScoreRecord, problems: list[str]) -> None:
    """What one record can be wrong about on its own."""
    if not rec.score.generation_id:
        problems.append(f"{rec.origin}: missing generation_id")
    if not rec.source_rater_id:
        problems.append(f"{rec.origin}: missing rater_id")
    raw_type = rec.meta.get("_raw_rater_type")
    if raw_type not in RATER_TYPES:
        problems.append(
            f"{rec.origin}: rater_type {raw_type!r} is not one of {sorted(RATER_TYPES)}"
        )
    if rec.sample_idx < 0:
        problems.append(
            f"{rec.origin}: sample_idx {rec.meta.get('sample_idx')!r} is not a non-negative integer"
        )

    raw_dims: Mapping[str, Any] = rec.meta.get("_raw_dimensions") or {}
    for dim in RUBRIC_DIMENSIONS:
        raw = raw_dims.get(dim)
        value = getattr(rec.score, dim)
        if raw is not None and value is None:
            problems.append(
                f"{rec.origin}: dimension {dim} is {raw!r}, which is not an integer 1-5 or null"
            )
        elif value is not None and not 1 <= value <= 5:
            problems.append(f"{rec.origin}: dimension {dim} is {value}, outside the 1-5 scale")

    if rec.meta.get("_raw_spans") is not None and not isinstance(rec.meta["_raw_spans"], dict):
        problems.append(f"{rec.origin}: evidence_spans is not an object")
    if rec.meta.get("_raw_flags") is not None and not isinstance(rec.meta["_raw_flags"], list):
        problems.append(f"{rec.origin}: safety_flags is not an array")

    # v3 §13: a score without a locatable span is invalid. The judge already
    # nulls an ungrounded dimension, so a scored dimension with no span means
    # the file did not come from that path and its scores are not grounded.
    # Only the judge is held to this; a deterministic scorer quotes nothing.
    if str(rec.score.rater_type) == RaterType.LLM_JUDGE.value:
        for dim in rec.scored_dimensions:
            if not str(rec.score.evidence_spans.get(dim, "")).strip():
                problems.append(
                    f"{rec.origin}: dimension {dim} scored {getattr(rec.score, dim)} with no "
                    f"evidence span (v3 §13: an ungrounded score is not a score)"
                )

    # `complete` and `n_dimensions_scored` restate what the nulls already say.
    # That redundancy is only worth carrying if disagreement is an error, and it
    # is the one that detects a row already dropped or zero-filled upstream.
    n_scored = len(rec.scored_dimensions)
    declared = rec.meta.get("n_dimensions_scored")
    if isinstance(declared, int) and not isinstance(declared, bool) and declared != n_scored:
        problems.append(
            f"{rec.origin}: n_dimensions_scored={declared} but {n_scored} dimension(s) "
            f"carry a value"
        )
    complete = rec.meta.get("complete")
    if isinstance(complete, bool) and complete != (n_scored == len(RUBRIC_DIMENSIONS)):
        problems.append(
            f"{rec.origin}: complete={complete} but {n_scored} of "
            f"{len(RUBRIC_DIMENSIONS)} dimension(s) carry a value"
        )

    condition = rec.meta.get("condition")
    if condition is not None and str(condition) not in {c.value for c in Condition}:
        problems.append(f"{rec.origin}: unknown condition {condition!r}")
    partial = rec.meta.get("partial_condition")
    if isinstance(partial, bool) and partial != (str(condition) in PARTIAL_CONDITIONS):
        problems.append(
            f"{rec.origin}: partial_condition={partial} disagrees with condition {condition!r}"
        )


def _same_scores(a: ScoreRecord, b: ScoreRecord) -> bool:
    return (
        all(getattr(a.score, d) == getattr(b.score, d) for d in RUBRIC_DIMENSIONS)
        and list(a.score.safety_flags) == list(b.score.safety_flags)
        and a.score.evidence_spans == b.score.evidence_spans
    )


def _check_across_records(records: Sequence[ScoreRecord], problems: list[str]) -> list[ScoreRecord]:
    """What only the whole input can be wrong about. Returns the deduplicated set.

    Two rows on one unique key do not fail at insert -- `ON CONFLICT DO UPDATE`
    resolves them by letting the last one win, silently. If they carry the same
    scores that is a harmless repeat and one is kept; if they disagree, the table
    would record whichever line the file happened to end with, so it is refused.
    """
    by_key: dict[tuple[str, str, str, int], ScoreRecord] = {}
    kept: list[ScoreRecord] = []
    for rec in records:
        prior = by_key.get(rec.key)
        if prior is not None:
            if _same_scores(prior, rec):
                continue  # the same row reached us twice; keep one
            problems.append(
                f"rubric_score key {rec.key} claimed by two rows with different scores: "
                f"{prior.origin} and {rec.origin}"
            )
            continue
        by_key[rec.key] = rec
        kept.append(rec)
    return kept


def _collect(
    paths: Iterable[Path | str], *, aggregate: bool, rater_id: str | None
) -> tuple[list[ScoreRecord], int]:
    """The deduplicated records, and how many lines they were read from."""
    files = expand_paths(paths)
    problems: list[str] = []
    records: list[ScoreRecord] = []
    for path in files:
        for rec in read_journal(path, aggregate=aggregate, rater_id=rater_id):
            _check_internal(rec, problems)
            records.append(rec)
    kept = _check_across_records(records, problems)
    if problems:
        raise ScoreRefusal(problems)
    return kept, len(records)


def collect(
    paths: Iterable[Path | str], *, aggregate: bool = True, rater_id: str | None = None
) -> list[ScoreRecord]:
    """Read and validate every journal. Raises `ScoreRefusal` on any problem.

    Nothing here touches the database, so this is also what `--dry-run` runs and
    what a test can drive with no Postgres.
    """
    return _collect(paths, aggregate=aggregate, rater_id=rater_id)[0]


def check_against_database(records: Sequence[ScoreRecord], report: LoadReport) -> None:
    """Every foreign key and every identity claim, checked before the first write.

    The `generation_id` foreign key would fail at insert anyway; failing here
    turns "batch 3 of 5 aborted" into one message naming every offending row
    while the table is still untouched. The identity checks have no such
    backstop: a score journal loaded against the wrong database would attach
    real numbers to the wrong cells and nothing in the schema would object.
    """
    from carelite.db.connection import fetch_all

    problems: list[str] = []
    gen_ids = sorted({r.score.generation_id for r in records})

    rows = fetch_all(
        "SELECT g.generation_id, g.scenario_id, g.condition, g.gate_blocked, sc.split "
        "FROM generation g JOIN scenario sc USING (scenario_id) "
        "WHERE g.generation_id = ANY(%s)",
        (gen_ids,),
    )
    known = {str(r["generation_id"]): r for r in rows}
    for rec in records:
        row = known.get(rec.score.generation_id)
        if row is None:
            problems.append(
                f"{rec.origin}: generation_id {rec.score.generation_id} has no `generation` row"
            )
            continue
        for label, column in (("scenario_id", "scenario_id"), ("condition", "condition")):
            claimed = rec.meta.get(label)
            if claimed is not None and str(claimed) != str(row[column]):
                problems.append(
                    f"{rec.origin}: row claims {label}={claimed!r} but generation "
                    f"{rec.score.generation_id} is {str(row[column])!r} in the table"
                )
        claimed_split = rec.meta.get("split")
        if claimed_split is not None and str(claimed_split) != str(row["split"]):
            report.split_mismatch += 1
        if bool(row["gate_blocked"]) != bool(rec.meta.get("output_gate_blocked", False)):
            report.gate_flag_mismatch += 1

    if problems:
        raise ScoreRefusal(problems)

    # Not a refusal: an existing row on a target key is the idempotent case the
    # upsert is for. It is counted and printed because the *other* reading --
    # a different judging run overwriting this one under the same rater id --
    # looks identical from here and is the thing a reader should notice.
    keys = fetch_all(
        "SELECT generation_id, rater_type, rater_id, sample_idx FROM rubric_score "
        "WHERE generation_id = ANY(%s)",
        (gen_ids,),
    )
    existing = {
        (str(r["generation_id"]), str(r["rater_type"]), str(r["rater_id"]), int(r["sample_idx"]))
        for r in keys
    }
    report.targets_already_present = sum(1 for r in records if r.key in existing)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class LoadReport:
    """What was read, what was written, and what a reader has to know about it."""

    files: list[Path] = field(default_factory=list)
    records: int = 0
    duplicates_collapsed: int = 0
    rows_inserted: int = 0
    rows_updated: int = 0
    targets_already_present: int = 0
    rater_ids: Counter[str] = field(default_factory=Counter)
    rater_id_rewritten: Counter[str] = field(default_factory=Counter)
    by_condition: Counter[str] = field(default_factory=Counter)
    scenarios_by_condition: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    rows_with_null_dimension: int = 0
    nulls_by_dimension: Counter[str] = field(default_factory=Counter)
    incomplete_by_condition: Counter[str] = field(default_factory=Counter)
    gate_blocked: Counter[str] = field(default_factory=Counter)
    partial_rows: Counter[str] = field(default_factory=Counter)
    judge_models: Counter[str] = field(default_factory=Counter)
    prompt_versions: Counter[str] = field(default_factory=Counter)
    rubric_versions: Counter[str] = field(default_factory=Counter)
    temperatures: Counter[str] = field(default_factory=Counter)
    split_mismatch: int = 0
    gate_flag_mismatch: int = 0
    spans_checked: int = 0
    spans_not_found: Counter[str] = field(default_factory=Counter)
    sidecar: Path | None = None
    dry_run: bool = False

    @property
    def rows_written(self) -> int:
        return self.rows_inserted + self.rows_updated

    def summary(self) -> str:
        lines: list[str] = []
        verb = "would load" if self.dry_run else "loaded"
        lines.append(
            f"{verb} {self.records} judge score row(s) from {len(self.files)} file(s): "
            f"{self.rows_inserted} inserted, {self.rows_updated} updated"
        )
        if self.duplicates_collapsed:
            lines.append(f"  duplicate lines collapsed: {self.duplicates_collapsed}")
        for rater_id in sorted(self.rater_ids):
            lines.append(f"  rater_id {rater_id}: {self.rater_ids[rater_id]} row(s)")
        for src, n in sorted(self.rater_id_rewritten.items()):
            lines.append(f"    (rewritten from the file's {src!r} on {n} row(s))")
        if self.targets_already_present:
            lines.append(
                f"  {self.targets_already_present} target row(s) already existed and were "
                f"overwritten by this load. If that was a *different* judging run under the "
                f"same rater id, its numbers are gone."
            )
        for condition in sorted(self.by_condition):
            n = self.by_condition[condition]
            scenarios = len(self.scenarios_by_condition[condition])
            note = ""
            if condition in PARTIAL_CONDITIONS:
                note = "  <- PARTIAL RECORD (D11): not a usable sample, never randomised"
            lines.append(f"  {condition:<3} {n:>4} rows  {scenarios:>2} scenarios{note}")
        if self.rows_with_null_dimension:
            per = " ".join(f"{d}={self.nulls_by_dimension[d]}" for d in RUBRIC_DIMENSIONS)
            by_cond = " ".join(f"{c}={n}" for c, n in sorted(self.incomplete_by_condition.items()))
            lines.append(
                f"  {self.rows_with_null_dimension} row(s) incomplete on at least one "
                f"dimension [{by_cond}]: written as NULL, not dropped and not zero-filled"
            )
            lines.append(f"    nulls per dimension: {per}")
        if self.gate_blocked:
            per = " ".join(f"{c}={n}" for c, n in sorted(self.gate_blocked.items()))
            lines.append(
                f"  {sum(self.gate_blocked.values())} row(s) score a generation the output "
                f"gate refused [{per}] (D12): loaded, not filtered. Exclude them in the "
                f"analysis with `generation.gate_blocked`."
            )
        if self.partial_rows:
            per = " ".join(f"{c}={n}" for c, n in sorted(self.partial_rows.items()))
            lines.append(
                f"  {sum(self.partial_rows.values())} row(s) belong to a partial condition "
                f"[{per}] (D11): loaded and marked, but not a complete arm."
            )
        for label, counter in (
            ("judge model", self.judge_models),
            ("judge prompt", self.prompt_versions),
            ("rubric version", self.rubric_versions),
            ("judge temperature", self.temperatures),
        ):
            if counter:
                shown = " ".join(f"{k}={n}" for k, n in sorted(counter.items()))
                lines.append(f"  {label}: {shown}")
            if len(counter) > 1:
                lines.append(
                    f"    WARNING: {len(counter)} distinct values of {label} in one load; "
                    f"these rows did not all come from one judging configuration"
                )
        if self.split_mismatch:
            lines.append(
                f"  WARNING: {self.split_mismatch} row(s) record a split that disagrees with "
                f"`scenario.split`. The table's split is what the analysis filters on."
            )
        if self.gate_flag_mismatch:
            lines.append(
                f"  WARNING: {self.gate_flag_mismatch} row(s) record an output-gate flag that "
                f"disagrees with `generation.gate_blocked`."
            )
        if self.spans_checked:
            missing = sum(self.spans_not_found.values())
            rate = 100.0 * (self.spans_checked - missing) / self.spans_checked
            lines.append(
                f"  span re-check: {self.spans_checked - missing}/{self.spans_checked} "
                f"cited spans located verbatim in the stored response ({rate:.1f}%)"
            )
            if missing:
                per = " ".join(f"{d}={n}" for d, n in sorted(self.spans_not_found.items()))
                lines.append(f"    not located: {per}")
        if self.sidecar is not None:
            lines.append(f"  sidecar: {self.sidecar}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The sidecar
# ---------------------------------------------------------------------------

#: What the sidecar keeps per row. `rubric_score` has no column for any of it,
#: and `complete`/`n_dimensions_scored` in particular are the difference between
#: "the judge declined to score this" and "this was never judged".
SIDECAR_FIELDS = (
    "complete",
    "n_dimensions_scored",
    "judge_model",
    "judge_digest",
    "prompt_version",
    "rubric_version",
    "temperature",
    "condition",
    "scenario_id",
    "split",
    "partial_condition",
    "output_gate_blocked",
)


def default_sidecar_path() -> Path:
    return get_settings().runs_dir / "judge" / "score_metadata.jsonl"


def load_score_metadata(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """The sidecar, keyed by `generation_id`. Empty dict if it does not exist."""
    target = Path(path) if path is not None else default_sidecar_path()
    if not target.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            gid = obj.get("generation_id")
            if gid:
                out[str(gid)] = obj
    return out


def incomplete_generation_ids(path: Path | None = None) -> set[str]:
    """Generations the judge could not score on every dimension.

    Recoverable from the table as well -- an incomplete row has a NULL
    dimension -- but this says so without a query, and carries the judge's own
    count rather than one re-derived from the row.
    """
    return {gid for gid, meta in load_score_metadata(path).items() if meta.get("complete") is False}


def _write_sidecar(records: Sequence[ScoreRecord], target: Path) -> None:
    """Merge these rows' metadata into the sidecar and rewrite it atomically.

    Merged rather than appended so re-running does not double every line, and
    rather than truncated so a sidecar holding another run's rows survives.
    `os.replace` is atomic on POSIX, so an interrupted write leaves the previous
    sidecar intact instead of half a file.
    """
    merged = load_score_metadata(target)
    for rec in records:
        payload = {k: rec.meta[k] for k in SIDECAR_FIELDS if k in rec.meta}
        if not payload:
            continue
        merged[rec.score.generation_id] = {
            "generation_id": rec.score.generation_id,
            "rater_id": rec.score.rater_id,
            "sample_idx": rec.sample_idx,
            **payload,
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for gid in sorted(merged):
            fh.write(json.dumps(merged[gid], ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


# ---------------------------------------------------------------------------
# The load
# ---------------------------------------------------------------------------


def _summarise(records: Sequence[ScoreRecord], report: LoadReport) -> None:
    for rec in records:
        report.rater_ids[rec.score.rater_id] += 1
        if rec.score.rater_id != rec.source_rater_id:
            report.rater_id_rewritten[rec.source_rater_id] += 1
        condition = str(rec.meta.get("condition") or "?")
        report.by_condition[condition] += 1
        scenario_id = rec.meta.get("scenario_id")
        if scenario_id:
            report.scenarios_by_condition[condition].add(str(scenario_id))
        nulls = rec.null_dimensions
        if nulls:
            report.rows_with_null_dimension += 1
            report.incomplete_by_condition[condition] += 1
            for dim in nulls:
                report.nulls_by_dimension[dim] += 1
        if rec.meta.get("output_gate_blocked"):
            report.gate_blocked[condition] += 1
        if condition in PARTIAL_CONDITIONS:
            report.partial_rows[condition] += 1
        for key, counter in (
            ("judge_model", report.judge_models),
            ("prompt_version", report.prompt_versions),
            ("rubric_version", report.rubric_versions),
            ("temperature", report.temperatures),
        ):
            value = rec.meta.get(key)
            if value is not None:
                counter[str(value)] += 1


def _verify_spans(records: Sequence[ScoreRecord], report: LoadReport) -> None:
    """Re-locate every cited span in the response as the table stores it.

    Not a refusal. The judge grounded each span against the response *as it was
    presented*, and this checks it against the response *as it was stored*; a
    divergence is a provenance signal worth printing, not evidence that the
    scores are wrong.
    """
    from carelite.db.connection import fetch_all
    from carelite.eval.judge.grounding import locate

    gen_ids = sorted({r.score.generation_id for r in records})
    responses = {
        str(row["generation_id"]): str(row["response"])
        for row in fetch_all(
            "SELECT generation_id, response FROM generation WHERE generation_id = ANY(%s)",
            (gen_ids,),
        )
    }
    for rec in records:
        response = responses.get(rec.score.generation_id)
        if response is None:
            continue
        for dim in rec.scored_dimensions:
            span = rec.score.evidence_spans.get(dim, "")
            report.spans_checked += 1
            if locate(span, response) is None:
                report.spans_not_found[dim] += 1


def _insert_batch(batch: Sequence[ScoreRecord], report: LoadReport) -> None:
    from carelite.db.connection import transaction

    with transaction() as conn:
        for rec in batch:
            row = conn.execute(
                _UPSERT_RETURNING, upsert_params(rec.score, rec.sample_idx)
            ).fetchone()
            if row is not None and row["inserted"]:
                report.rows_inserted += 1
            else:
                report.rows_updated += 1


def load_score_journals(
    paths: Iterable[Path | str],
    *,
    aggregate: bool = True,
    rater_id: str | None = None,
    sidecar: Path | None = None,
    write_sidecar: bool = True,
    dry_run: bool = False,
    batch_size: int = 200,
    database: bool = True,
    verify_spans: bool = False,
) -> LoadReport:
    """Read the journals, refuse anything wrong, and upsert what they hold.

    Args:
        paths: score journal files, or directories contributing their `*.jsonl`.
        aggregate: write under `"<rater_id>-median"`, the one-row-per-generation
            partition `carelite.stats.data` selects. See the module docstring.
        rater_id: override the file's rater id before the suffix is applied.
        sidecar: where the non-column metadata goes; defaults to
            `default_sidecar_path()`.
        write_sidecar: set False to load rows only.
        dry_run: validate and report, touching neither the table nor the sidecar.
        batch_size: rows per committed transaction. Smaller means an interrupted
            load loses less work; it never affects correctness, because the
            upsert makes a re-run of any prefix a no-op.
        database: touch Postgres at all. False validates and writes the sidecar
            without opening a connection, which is what lets the unit tests
            drive the whole path in `make check`.
        verify_spans: re-locate every cited span in the stored response and
            report the rate. Requires the database; never refuses.

    Raises:
        ScoreRefusal: before anything is written, listing every problem found.
    """
    files = expand_paths(paths)
    records, total_lines = _collect(files, aggregate=aggregate, rater_id=rater_id)
    report = LoadReport(files=files, records=len(records), dry_run=dry_run)
    report.duplicates_collapsed = total_lines - len(records)
    _summarise(records, report)

    if database:
        check_against_database(records, report)
        if verify_spans:
            _verify_spans(records, report)
    if dry_run:
        return report

    if database:
        step = max(1, batch_size)
        for start in range(0, len(records), step):
            _insert_batch(records[start : start + step], report)

    if write_sidecar:
        target = Path(sidecar) if sidecar is not None else default_sidecar_path()
        _write_sidecar(records, target)
        report.sidecar = target
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carelite.eval.judge.load",
        description=(
            "Load JSONL judge score journals into `rubric_score`. Idempotent: "
            "safe to re-run, safe to interrupt, and it refuses rather than "
            "landing a table it cannot vouch for."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "journals",
        nargs="+",
        type=Path,
        help="score journal files, or directories whose *.jsonl will be loaded",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report; write nothing to the table or the sidecar",
    )
    parser.add_argument(
        "--rater-id",
        default=None,
        help="override the file's rater id (before the aggregate suffix is applied)",
    )
    parser.add_argument(
        "--no-aggregate-suffix",
        action="store_true",
        help=(
            "write the plain rater id instead of '<rater_id>-median'. The samples "
            "partition; carelite.stats.data will NOT see these rows"
        ),
    )
    parser.add_argument("--sidecar", type=Path, default=None)
    parser.add_argument(
        "--no-sidecar",
        action="store_true",
        help="load rows only; drops `complete` and `n_dimensions_scored`",
    )
    parser.add_argument(
        "--verify-spans",
        action="store_true",
        help="re-locate every cited evidence span in the stored response and report the rate",
    )
    parser.add_argument("--batch-size", type=int, default=200)
    return parser


def _fail(message: str) -> NoReturn:
    print(f"carelite.eval.judge.load: {message}", file=sys.stderr)
    raise SystemExit(2)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    args = _build_parser().parse_args(argv)
    try:
        report = load_score_journals(
            args.journals,
            aggregate=not args.no_aggregate_suffix,
            rater_id=args.rater_id,
            sidecar=args.sidecar,
            write_sidecar=not args.no_sidecar,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            verify_spans=args.verify_spans,
        )
    except ScoreRefusal as exc:
        _fail(str(exc))
    print(report.summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
