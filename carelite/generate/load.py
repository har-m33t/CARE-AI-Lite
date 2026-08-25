"""Load a JSONL generation journal into Postgres.

    python -m carelite.generate.load runs/holdout/*.jsonl
    python -m carelite.generate.load runs/holdout --dry-run

**Why this exists.** `JsonlStore` and `PostgresStore` are two implementations of
one `GenerationStore` Protocol, and until now nothing carried a run from the
first into the second. The holdout run was written with `--store jsonl` on
purpose — an earlier attempt lost ~863 generations when Postgres sat on a
container disk that was restarted — so the durable artifact of the experiment is
a set of journal files, and every downstream step (judging, statistics, figures,
`make reproduce`) reads the `generation` table. This module is that bridge.

**The journal is authoritative for `generation_id`.** The id downstream lanes key
rubric scores to is the one the run actually produced, not one recomputed later
from a function that may have drifted. `generation_id_for()` is still evaluated
for every record and any divergence is reported, but the file's id is what is
written. On the 939-row holdout journal the two agree on every record; that
agreement is an observation, not an assumption the loader depends on.

**Idempotence is the table's, not the application's.** Every insert is
`ON CONFLICT DO NOTHING` against the constraints the frozen schema already
declares — the `generation_id` primary key and the v3 §16 `UNIQUE (scenario_id,
condition, prompt_id, model_digest, seed, sample_idx)` — so re-running the loader
over the same files, or over a superset of them, adds only what is missing. Rows
are committed in batches, so an interrupted load resumes by simply being run
again. There is no manifest of what was loaded and nothing to reconcile.

**It refuses rather than lands a corrupted table.** Silently-wrong input is the
failure this guards: a record whose `extra` disagrees with its own key, two files
claiming one `generation_id` for different content, one model tag carrying two
digests, a `prompt_id` registered against a different condition, a scenario that
is not in the bank, an existing row whose id does not match the one the file
would attach a retrieval trace to. Every check runs over the whole input before
a single row is written, and a failure names every offending record rather than
the first one. A partial load is safe; a mixed-provenance table is not.

**The retrieval trace is load-bearing and is never dropped.** Condition C fell
back to B on 69 of its 180 holdout cells (CRAG graded 111 `relevant`, 69 `none`),
so on 38% of the holdout C received no retrieved context and is materially
identical to B. The statistics lane needs `fell_back_to_b` and `crag_grade` to
split C-vs-B into intention-to-treat and actually-retrieved, so `retrieval_trace`
is written in the same transaction as the generation row it belongs to.

**What the schema has no column for goes to the sidecar.** `extra` carries the
self-check verdict, the input-screen flags, the context window, the split, and —
the reason this matters here — `output_gate_blocked`. Seventeen holdout cells had
their generated text refused by the output safety gate (B 2, C 2, A 3, A2 7,
D 3). Those rows are *present* in the journal, flagged, not missing from it, and
loading them unflagged would let the analysis score refused text as ordinary
output. `generation` has no column for the flag and this lane does not own the
schema, so the flag goes to a sidecar keyed by `generation_id` — the same
sidecar `PostgresStore` writes, at the same default path — and
`blocked_generation_ids()` is the supported way to read it back. That is weaker
than a column and it is said plainly here so no later reader has to infer it.

**Condition LC is a partial record, not a condition** (D11). 39 of 180 cells
exist, covering 13 of 60 scenarios, never randomised for partial analysis. The
loader stores them and reports them as partial; it never counts LC toward
coverage and nothing here should be read as making LC a usable sample.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

from carelite.config import get_settings
from carelite.generate.model import DIGEST_UNAVAILABLE
from carelite.generate.store import CacheKey, GenerationRecord, generation_id_for
from carelite.types import Condition

__all__ = [
    "JournalRefusal",
    "LoadReport",
    "SourceRecord",
    "blocked_generation_ids",
    "collect",
    "default_sidecar_path",
    "load_journals",
    "load_metadata",
    "read_journal",
]

#: Conditions that are a partial record rather than a complete condition (D11).
PARTIAL_CONDITIONS = frozenset({Condition.LC.value})

#: `retrieval_trace.crag_grade` carries a CHECK constraint. Validating here means
#: a bad grade is named with its file and line instead of aborting a batch.
CRAG_GRADES = frozenset({"relevant", "ambiguous", "none"})

#: How many individual problems a refusal lists before it stops enumerating.
MAX_REPORTED_PROBLEMS = 40

_INSERT_GENERATION = """
    INSERT INTO generation (generation_id, scenario_id, condition, prompt_id,
                            model, model_digest, seed, temperature, sample_idx,
                            response, latency_ms)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT DO NOTHING
    RETURNING generation_id
"""

_INSERT_TRACE = """
    INSERT INTO retrieval_trace (generation_id, retrieved_ids, scores, crag_grade,
                                 route_taken, fell_back_to_b, hyde_passage, latency_ms)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (generation_id) DO NOTHING
    RETURNING generation_id
"""


class JournalRefusal(RuntimeError):
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
class SourceRecord:
    """One journal line, with where it came from and what its id should be.

    `GenerationRecord` deliberately derives its `generation_id` from the key.
    That is right for a runner, which is producing the id, and wrong for a
    loader, which is preserving one. So the file's id is carried alongside the
    record and the derived one is kept only to be compared.
    """

    record: GenerationRecord
    generation_id: str
    derived_id: str
    origin: str

    @property
    def key(self) -> CacheKey:
        return self.record.key

    @property
    def id_drifted(self) -> bool:
        return self.generation_id != self.derived_id


def read_journal(path: Path) -> Iterator[SourceRecord]:
    """Parse every line of a journal, strictly.

    `JsonlStore.read_all` skips a line it cannot parse, which is correct for its
    job: a journal that cannot be reopened after a `kill -9` is not a
    crash-recovery mechanism. It is wrong for this one. A malformed line during a
    load means the file is not the complete record of the run, and silently
    loading n-1 rows is precisely the failure mode this module exists to refuse.
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
                raise JournalRefusal([f"{origin}: not valid JSON ({exc.msg})"]) from exc
            if not isinstance(obj, dict):
                raise JournalRefusal([f"{origin}: expected a JSON object"])
            try:
                # `from_json` is the single definition of the record shape, so it
                # parses the line rather than this module re-deriving the fields.
                # The second `json.loads` above is what gives us the file's own
                # `generation_id`, which `GenerationRecord` does not carry.
                record = GenerationRecord.from_json(line)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise JournalRefusal([f"{origin}: malformed record ({exc})"]) from exc
            derived = generation_id_for(record.key)
            file_id = obj.get("generation_id")
            yield SourceRecord(
                record=record,
                generation_id=str(file_id) if file_id else derived,
                derived_id=derived,
                origin=origin,
            )


def expand_paths(paths: Iterable[Path | str]) -> list[Path]:
    """Accept files or directories; a directory contributes its `*.jsonl`.

    Duplicate paths collapse, because passing a glob and the directory that
    contains it is an easy way to load the same file twice, and a loader that
    calls that a duplicate-id collision would be refusing the user's shell rather
    than a real problem.
    """
    seen: dict[Path, None] = {}
    for entry in paths:
        p = Path(entry)
        if p.is_dir():
            for child in sorted(p.glob("*.jsonl")):
                seen.setdefault(child.resolve(), None)
        else:
            if not p.exists():
                raise JournalRefusal([f"{p}: no such file"])
            seen.setdefault(p.resolve(), None)
    if not seen:
        raise JournalRefusal(["no journal files found"])
    return list(seen)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_internal(src: SourceRecord, problems: list[str]) -> None:
    """What one record can be wrong about on its own."""
    key = src.key
    rec = src.record

    if key.condition not in {c.value for c in Condition}:
        problems.append(f"{src.origin}: unknown condition {key.condition!r}")
    if not rec.response.strip():
        problems.append(f"{src.origin}: empty response for {src.generation_id}")
    if key.sample_idx < 0:
        problems.append(f"{src.origin}: negative sample_idx {key.sample_idx}")
    if not key.prompt_id:
        problems.append(f"{src.origin}: empty prompt_id")
    if not key.model_digest:
        problems.append(f"{src.origin}: empty model_digest")

    # `extra` restates three of the key's fields. That redundancy is only worth
    # carrying if disagreement is an error, so it is one.
    for field_name, expected in (
        ("condition", key.condition),
        ("scenario_id", key.scenario_id),
        ("sample_idx", key.sample_idx),
    ):
        if field_name in rec.extra and rec.extra[field_name] != expected:
            problems.append(
                f"{src.origin}: extra[{field_name!r}]={rec.extra[field_name]!r} "
                f"disagrees with key {expected!r}"
            )

    trace = rec.trace
    if trace is not None:
        grade = trace.get("crag_grade")
        if grade is not None and grade not in CRAG_GRADES:
            problems.append(
                f"{src.origin}: crag_grade {grade!r} is not one of {sorted(CRAG_GRADES)}"
            )
        ids = trace.get("retrieved_ids") or []
        scores = trace.get("scores") or []
        if len(ids) != len(scores):
            problems.append(
                f"{src.origin}: trace has {len(ids)} retrieved_ids but {len(scores)} scores"
            )


def _check_across_records(
    sources: Sequence[SourceRecord], problems: list[str]
) -> list[SourceRecord]:
    """What only the whole input can be wrong about. Returns the deduplicated set.

    Two distinct failures hide here and neither surfaces at insert time. A
    `generation_id` reused for different content would be swallowed by
    `ON CONFLICT DO NOTHING`, leaving a table that is quietly missing a row. A
    cache key reused under two different ids would insert one and drop the other
    the same way. Both are caught before anything is written.
    """
    by_id: dict[str, SourceRecord] = {}
    by_key: dict[CacheKey, SourceRecord] = {}
    digest_by_model: dict[str, dict[str, str]] = defaultdict(dict)
    kept: list[SourceRecord] = []

    for src in sources:
        prior = by_id.get(src.generation_id)
        if prior is not None:
            if _same_content(prior, src):
                continue  # the same record reached us twice; keep one
            problems.append(
                f"generation_id {src.generation_id} claimed by two different records: "
                f"{prior.origin} and {src.origin}"
            )
            continue
        prior_key = by_key.get(src.key)
        if prior_key is not None:
            problems.append(
                f"cache key {src.key.as_tuple()} appears under two ids: "
                f"{prior_key.generation_id} ({prior_key.origin}) and "
                f"{src.generation_id} ({src.origin})"
            )
            continue
        by_id[src.generation_id] = src
        by_key[src.key] = src
        digest_by_model[src.record.model][src.key.model_digest] = src.origin
        kept.append(src)

    # A tag serving two digests within one run means two different sets of
    # weights were recorded under one name. That is exactly what recording the
    # digest was for, so it is a refusal rather than a note.
    for model, digests in digest_by_model.items():
        if len(digests) > 1:
            where = ", ".join(f"{d[:12]}… ({o})" for d, o in sorted(digests.items()))
            problems.append(f"model tag {model!r} carries {len(digests)} digests: {where}")

    return kept


def _same_content(a: SourceRecord, b: SourceRecord) -> bool:
    return (
        a.key == b.key
        and a.record.model == b.record.model
        and a.record.response == b.record.response
        and a.record.temperature == b.record.temperature
    )


def _collect(paths: Iterable[Path | str]) -> tuple[list[SourceRecord], int]:
    """The deduplicated records, and how many lines they were read from."""
    files = expand_paths(paths)
    problems: list[str] = []
    sources: list[SourceRecord] = []
    for path in files:
        for src in read_journal(path):
            _check_internal(src, problems)
            sources.append(src)
    kept = _check_across_records(sources, problems)
    if problems:
        raise JournalRefusal(problems)
    return kept, len(sources)


def collect(paths: Iterable[Path | str]) -> list[SourceRecord]:
    """Read and validate every journal. Raises `JournalRefusal` on any problem.

    Nothing here touches the database, so this is also what `--dry-run` runs and
    what a test can drive with no Postgres.
    """
    return _collect(paths)[0]


def check_against_database(sources: Sequence[SourceRecord]) -> None:
    """Every foreign key and every pre-existing row, checked before the first write.

    The foreign keys would fail at insert anyway; failing here instead turns
    "batch 3 of 5 aborted" into one message naming every offending record while
    the table is still untouched. The pre-existing-row checks have no such
    backstop: an id that already belongs to a different cell, or a cell that
    already exists under a different id, are both absorbed silently by
    `ON CONFLICT DO NOTHING`.
    """
    from carelite.db.connection import fetch_all

    problems: list[str] = []

    scenario_ids = sorted({s.key.scenario_id for s in sources})
    known_scenarios = {
        str(r["scenario_id"])
        for r in fetch_all(
            "SELECT scenario_id FROM scenario WHERE scenario_id = ANY(%s)", (scenario_ids,)
        )
    }
    for missing in sorted(set(scenario_ids) - known_scenarios):
        problems.append(f"scenario {missing!r} is not in the scenario bank")

    prompt_ids = sorted({s.key.prompt_id for s in sources})
    registered = {
        str(r["prompt_id"]): str(r["condition"])
        for r in fetch_all(
            "SELECT prompt_id, condition FROM prompt_version WHERE prompt_id = ANY(%s)",
            (prompt_ids,),
        )
    }
    for missing in sorted(set(prompt_ids) - set(registered)):
        problems.append(
            f"prompt_version {missing!r} is not registered (run the runner with --register-prompts)"
        )

    # `prompt_version.condition` is comma-joined for a prompt shared between
    # conditions: `condition_a.v1` is registered as 'A,A2' because A2 is
    # condition A on a second model family.
    seen_pairs = {(s.key.prompt_id, s.key.condition) for s in sources}
    for prompt_id, condition in sorted(seen_pairs):
        declared = registered.get(prompt_id)
        if declared is None:
            continue
        if condition not in {c.strip() for c in declared.split(",")}:
            problems.append(
                f"prompt_version {prompt_id!r} is registered for {declared!r} "
                f"but a record claims condition {condition!r}"
            )

    gen_ids = sorted({s.generation_id for s in sources})
    existing = {
        str(r["generation_id"]): r
        for r in fetch_all(
            "SELECT generation_id, scenario_id, condition, prompt_id, model_digest, "
            "seed, sample_idx, response FROM generation WHERE generation_id = ANY(%s)",
            (gen_ids,),
        )
    }
    for src in sources:
        row = existing.get(src.generation_id)
        if row is None:
            continue
        stored = CacheKey(
            scenario_id=str(row["scenario_id"]),
            condition=str(row["condition"]),
            prompt_id=str(row["prompt_id"]),
            model_digest=str(row["model_digest"]),
            seed=int(row["seed"]),
            sample_idx=int(row["sample_idx"]),
        )
        if stored != src.key:
            problems.append(
                f"{src.origin}: generation_id {src.generation_id} already belongs to a "
                f"different cell in the table {stored.as_tuple()}"
            )
        elif str(row["response"]) != src.record.response:
            problems.append(
                f"{src.origin}: generation_id {src.generation_id} is already stored with a "
                f"different response"
            )

    # The mirror case: the cell exists, under an id the journal does not use.
    # Left alone, the generation insert is a no-op and the retrieval trace is
    # attached to an id with no row, so the FK aborts a batch that had looked fine.
    key_rows = fetch_all(
        "SELECT generation_id, scenario_id, condition, prompt_id, model_digest, seed, "
        "sample_idx FROM generation WHERE scenario_id = ANY(%s)",
        (scenario_ids,),
    )
    stored_by_key = {
        CacheKey(
            scenario_id=str(r["scenario_id"]),
            condition=str(r["condition"]),
            prompt_id=str(r["prompt_id"]),
            model_digest=str(r["model_digest"]),
            seed=int(r["seed"]),
            sample_idx=int(r["sample_idx"]),
        ): str(r["generation_id"])
        for r in key_rows
    }
    for src in sources:
        stored_id = stored_by_key.get(src.key)
        if stored_id is not None and stored_id != src.generation_id:
            problems.append(
                f"{src.origin}: cell {src.key.as_tuple()} is already stored under "
                f"generation_id {stored_id}, not {src.generation_id}"
            )

    if problems:
        raise JournalRefusal(problems)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class LoadReport:
    """What was read, what was written, and what a reader has to know about it."""

    files: list[Path] = field(default_factory=list)
    records: int = 0
    duplicates_collapsed: int = 0
    generations_inserted: int = 0
    generations_already_present: int = 0
    traces_inserted: int = 0
    traces_already_present: int = 0
    id_drift: list[str] = field(default_factory=list)
    unpinned_digests: int = 0
    by_condition: Counter[str] = field(default_factory=Counter)
    scenarios_by_condition: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    samples_by_condition: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    digests: dict[str, str] = field(default_factory=dict)
    crag_grades: Counter[str] = field(default_factory=Counter)
    fell_back_to_b: Counter[str] = field(default_factory=Counter)
    gate_blocked: Counter[str] = field(default_factory=Counter)
    gate_blocked_ids: list[str] = field(default_factory=list)
    sidecar: Path | None = None
    dry_run: bool = False

    def summary(self) -> str:
        lines: list[str] = []
        verb = "would load" if self.dry_run else "loaded"
        lines.append(
            f"{verb} {self.records} generations from {len(self.files)} file(s): "
            f"{self.generations_inserted} inserted, "
            f"{self.generations_already_present} already present"
        )
        if self.duplicates_collapsed:
            lines.append(f"  duplicate lines collapsed: {self.duplicates_collapsed}")
        lines.append(
            f"  retrieval traces: {self.traces_inserted} inserted, "
            f"{self.traces_already_present} already present"
        )
        for condition in sorted(self.by_condition):
            n = self.by_condition[condition]
            scenarios = len(self.scenarios_by_condition[condition])
            samples = len(self.samples_by_condition[condition])
            note = ""
            if condition in PARTIAL_CONDITIONS:
                note = "  <- PARTIAL RECORD (D11): not a usable sample, never randomised"
            lines.append(
                f"  {condition:<3} {n:>4} cells  {scenarios:>2} scenarios  {samples} sample idx{note}"
            )
        for model, digest in sorted(self.digests.items()):
            lines.append(f"  model {model} -> {digest[:16]}…")
        if self.unpinned_digests:
            lines.append(
                f"  WARNING: {self.unpinned_digests} row(s) record "
                f"model_digest={DIGEST_UNAVAILABLE!r}: the weights behind them cannot be identified"
            )
        if self.crag_grades:
            grades = " ".join(f"{g}={n}" for g, n in sorted(self.crag_grades.items()))
            lines.append(f"  CRAG grades: {grades}")
        for condition in sorted(self.fell_back_to_b):
            n = self.fell_back_to_b[condition]
            total = self.by_condition[condition]
            pct = 100.0 * n / total if total else 0.0
            lines.append(
                f"  {condition} fell back to B on {n}/{total} cells ({pct:.0f}%): those cells "
                f"received no retrieved context. Split {condition}-vs-B into "
                f"intention-to-treat and actually-retrieved."
            )
        if self.gate_blocked:
            per = " ".join(f"{c}={n}" for c, n in sorted(self.gate_blocked.items()))
            lines.append(
                f"  output gate blocked {sum(self.gate_blocked.values())} cell(s) [{per}]: "
                f"loaded and flagged in the sidecar, NOT dropped. "
                f"Read them with carelite.generate.load.blocked_generation_ids()."
            )
        if self.id_drift:
            lines.append(
                f"  WARNING: {len(self.id_drift)} record(s) carry a generation_id that "
                f"generation_id_for() does not reproduce. The file's id was used."
            )
            for line in self.id_drift[:5]:
                lines.append(f"    {line}")
        if self.sidecar is not None:
            lines.append(f"  sidecar: {self.sidecar}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The sidecar
# ---------------------------------------------------------------------------


def default_sidecar_path() -> Path:
    """Where `PostgresStore` would have written it. Same path, same shape.

    A run stored to Postgres directly writes its `extra` here. Loading a journal
    is reconstructing that run, so the reconstruction lands in the same place
    rather than inventing a second location downstream code would have to learn.
    """
    return get_settings().runs_dir / "generate" / "metadata.jsonl"


def load_metadata(path: Path | None = None) -> dict[str, dict[str, Any]]:
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


def blocked_generation_ids(path: Path | None = None) -> set[str]:
    """Generations whose text the output safety gate refused.

    The supported way for the judging and statistics lanes to find them. These
    rows are in `generation` like any other — the gate refused the text, it did
    not prevent the cell from being generated — and `generation` has no column
    to say so, so this reads the sidecar. Scoring them alongside unflagged output
    is a measurement of something other than what the rubric claims.
    """
    return {gid for gid, meta in load_metadata(path).items() if meta.get("output_gate_blocked")}


def _write_sidecar(sources: Sequence[SourceRecord], target: Path) -> None:
    """Merge these records' `extra` into the sidecar and rewrite it atomically.

    Merged rather than appended so that running the loader twice does not double
    every line, and rather than truncated so that a sidecar already holding a
    different run's rows survives. `os.replace` is atomic on POSIX, so an
    interrupted write leaves the previous sidecar intact instead of a half file.
    """
    merged = load_metadata(target)
    for src in sources:
        if not src.record.extra:
            continue
        merged[src.generation_id] = {"generation_id": src.generation_id, **src.record.extra}
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


def _summarise(sources: Sequence[SourceRecord], report: LoadReport) -> None:
    for src in sources:
        key = src.key
        report.by_condition[key.condition] += 1
        report.scenarios_by_condition[key.condition].add(key.scenario_id)
        report.samples_by_condition[key.condition].add(key.sample_idx)
        report.digests[src.record.model] = key.model_digest
        if key.model_digest == DIGEST_UNAVAILABLE:
            report.unpinned_digests += 1
        if src.id_drifted:
            report.id_drift.append(
                f"{src.origin}: file={src.generation_id} derived={src.derived_id}"
            )
        trace = src.record.trace
        if trace is not None:
            grade = trace.get("crag_grade")
            if grade:
                report.crag_grades[str(grade)] += 1
            if trace.get("fell_back_to_b"):
                report.fell_back_to_b[key.condition] += 1
        if src.record.extra.get("output_gate_blocked"):
            report.gate_blocked[key.condition] += 1
            report.gate_blocked_ids.append(src.generation_id)


def _insert_batch(batch: Sequence[SourceRecord], report: LoadReport) -> None:
    from carelite.db.connection import transaction

    with transaction() as conn:
        for src in batch:
            key = src.key
            rec = src.record
            inserted = conn.execute(
                _INSERT_GENERATION,
                (
                    src.generation_id,
                    key.scenario_id,
                    key.condition,
                    key.prompt_id,
                    rec.model,
                    key.model_digest,
                    key.seed,
                    rec.temperature,
                    key.sample_idx,
                    rec.response,
                    rec.latency_ms,
                ),
            ).fetchone()
            if inserted is None:
                report.generations_already_present += 1
            else:
                report.generations_inserted += 1
            if rec.trace is None:
                continue
            t = rec.trace
            trace_inserted = conn.execute(
                _INSERT_TRACE,
                (
                    src.generation_id,
                    list(t.get("retrieved_ids") or []),
                    list(t.get("scores") or []),
                    t.get("crag_grade"),
                    t.get("route_taken"),
                    bool(t.get("fell_back_to_b")),
                    t.get("hyde_passage"),
                    t.get("latency_ms"),
                ),
            ).fetchone()
            if trace_inserted is None:
                report.traces_already_present += 1
            else:
                report.traces_inserted += 1


def load_journals(
    paths: Iterable[Path | str],
    *,
    sidecar: Path | None = None,
    write_sidecar: bool = True,
    dry_run: bool = False,
    batch_size: int = 200,
    database: bool = True,
) -> LoadReport:
    """Read the journals, refuse anything wrong, and write what is missing.

    Args:
        paths: journal files, or directories contributing their `*.jsonl`.
        sidecar: where `extra` goes; defaults to `default_sidecar_path()`.
        write_sidecar: set False to load rows only.
        dry_run: validate and report, touching neither the table nor the sidecar.
        batch_size: rows per committed transaction. Smaller means an interrupted
            load loses less work; it never affects correctness, because the
            constraints make a re-run of any prefix a no-op.
        database: touch Postgres at all. False validates the journals and writes
            the sidecar without opening a connection, which is what lets the
            unit tests drive the whole path in `make check`.

    Raises:
        JournalRefusal: before anything is written, listing every problem found.
    """
    files = expand_paths(paths)
    sources, total_lines = _collect(files)
    report = LoadReport(files=files, records=len(sources), dry_run=dry_run)
    report.duplicates_collapsed = total_lines - len(sources)
    _summarise(sources, report)

    if database:
        check_against_database(sources)
    if dry_run:
        return report

    if database:
        for start in range(0, len(sources), max(1, batch_size)):
            _insert_batch(sources[start : start + max(1, batch_size)], report)

    if write_sidecar:
        target = Path(sidecar) if sidecar is not None else default_sidecar_path()
        _write_sidecar(sources, target)
        report.sidecar = target
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carelite.generate.load",
        description=(
            "Load JSONL generation journals into the `generation` and "
            "`retrieval_trace` tables. Idempotent: safe to re-run, safe to "
            "interrupt, and it refuses rather than landing a table it cannot "
            "vouch for."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "journals",
        nargs="+",
        type=Path,
        help="journal files, or directories whose *.jsonl will be loaded",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report; write nothing to the table or the sidecar",
    )
    parser.add_argument("--sidecar", type=Path, default=None, help="where `extra` is written")
    parser.add_argument(
        "--no-sidecar",
        action="store_true",
        help="load rows only; drops the output-gate flags and the self-check verdicts",
    )
    parser.add_argument("--batch-size", type=int, default=200)
    return parser


def _fail(message: str) -> NoReturn:
    print(f"carelite.generate.load: {message}", file=sys.stderr)
    raise SystemExit(2)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    args = _build_parser().parse_args(argv)
    try:
        report = load_journals(
            args.journals,
            sidecar=args.sidecar,
            write_sidecar=not args.no_sidecar,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
        )
    except JournalRefusal as exc:
        _fail(str(exc))
    print(report.summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
