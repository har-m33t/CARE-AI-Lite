"""`make reproduce` / `python -m carelite.repro` — regenerate every figure and table from the
database.

**No inference happens here.** This module reads `CARELITE_DATABASE_URL` and nothing else; every
number and figure it produces is already sitting in Postgres by the time it runs. Generating that
data — corpus fetch, knowledge-base extraction, index build, the 1,080-generation holdout run, LLM
judging — is a separate, multi-hour set of steps documented in `REPRODUCE.md`. This entry point is
the fast, idempotent last mile: run it as many times as you like against the same database state
and get the same output, in seconds to low minutes rather than hours.

**Owned by `carelite-repro`, wired to `make reproduce` in `Makefile`.** `carelite/stats/` (owned by
`carelite-stats`) and `carelite/viz/` (owned by `carelite-viz`) are the modules this entry point is
meant to call once they exist — as of this writing neither has landed, which this module reports
rather than hides or works around by duplicating their job here. The integration contract this
module looks for, so either lane can land independently and be picked up automatically the next
time this runs:

    carelite.stats.reproduce.run(output_dir: Path) -> list[Path]   # tables written, e.g. CSV/markdown
    carelite.viz.reproduce.run(output_dir: Path) -> list[Path]     # figures written, e.g. PNG/PDF

A lane landing a different shape should either adopt this contract or tell `carelite-repro` what it
actually exposes so this module can be updated to match — not the other way around; this module
should not need to know a stats or viz implementation's internals.

Until both land, this module still does real, checkable work: it verifies the schema is loaded,
reports how far each pipeline stage has actually progressed (row counts, not assumptions), and
writes that status where a human or CI can find it. A `make reproduce` on a fresh clone that has
only run `make db-up` should complete cleanly and say plainly "0 generations, nothing to
reproduce yet" rather than crash — the definition of done is a clean run, not a run against
complete data, and those are different guarantees.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carelite.config import get_settings
from carelite.db.connection import check_database, fetch_all

# Stage name -> the table whose row count best answers "has this stage run".
# Order matters: it's the pipeline order, and the report is read top to bottom.
PIPELINE_STAGES: tuple[tuple[str, str], ...] = (
    ("corpus fetch", "paper"),
    ("chunking", "chunk"),
    ("knowledge base", "kb_entry"),
    ("kb provenance", "kb_entry_source"),
    ("graph layer", "graph_edge"),
    ("scenario bank", "scenario"),
    ("prompt versions", "prompt_version"),
    ("generation", "generation"),
    ("retrieval traces", "retrieval_trace"),
    ("rubric scoring", "rubric_score"),
    ("human rating assignment", "rating_assignment"),
)

#: Full holdout run per `docs/preregistration.md` §6: 60 scenarios x 6 conditions x 3 samples.
EXPECTED_HOLDOUT_GENERATIONS = 60 * 6 * 3

#: The two downstream lanes this module hands off to once they exist. See the module docstring
#: for the call contract each is expected to expose.
_DOWNSTREAM_MODULES: tuple[tuple[str, str], ...] = (
    ("carelite.stats.reproduce", "tables"),
    ("carelite.viz.reproduce", "figures"),
)


@dataclass
class StageStatus:
    stage: str
    table: str
    n_rows: int | None  # None means the table itself is missing
    error: str | None = None


@dataclass
class DownstreamResult:
    module: str
    kind: str  # "tables" | "figures"
    available: bool
    written: list[Path] = field(default_factory=list)
    error: str | None = None


@dataclass
class ReproReport:
    db_ok: bool
    db_errors: list[str]
    stages: list[StageStatus] = field(default_factory=list)
    downstream: list[DownstreamResult] = field(default_factory=list)

    @property
    def holdout_generation_count(self) -> int | None:
        for s in self.stages:
            if s.table == "generation":
                return s.n_rows
        return None


def _stage_counts() -> list[StageStatus]:
    statuses: list[StageStatus] = []
    for stage_name, table in PIPELINE_STAGES:
        try:
            # `table` is always one of the fixed names in PIPELINE_STAGES above, never
            # user-controlled input, so string interpolation here is not a SQL-injection risk.
            row = fetch_all(f"SELECT COUNT(*) AS n FROM {table}")
            statuses.append(StageStatus(stage=stage_name, table=table, n_rows=row[0]["n"]))
        except Exception as exc:  # table missing, or a real connection problem already reported
            statuses.append(StageStatus(stage=stage_name, table=table, n_rows=None, error=str(exc)))
    return statuses


def _run_downstream(
    output_dir: Path, modules: tuple[tuple[str, str], ...] = _DOWNSTREAM_MODULES
) -> list[DownstreamResult]:
    """Call each downstream lane's `run(output_dir)` if the module exists and exposes it.

    Deliberately tolerant: an ImportError means the lane hasn't landed yet (expected, currently
    true for both), an AttributeError means it landed with a different shape than the documented
    contract (worth surfacing, not worth crashing over), and any other exception is the lane's own
    code failing, which is reported with its message rather than swallowed.

    `modules` is a parameter (not always `_DOWNSTREAM_MODULES`) so tests can point this at a fake
    module without needing `carelite.stats`/`carelite.viz` to exist.
    """
    results: list[DownstreamResult] = []
    for module_path, kind in modules:
        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            results.append(
                DownstreamResult(module=module_path, kind=kind, available=False, error=str(exc))
            )
            continue
        run_fn: Callable[[Path], list[Path]] | None = getattr(mod, "run", None)
        if run_fn is None:
            results.append(
                DownstreamResult(
                    module=module_path,
                    kind=kind,
                    available=True,
                    error=f"{module_path} has no run(output_dir) callable — contract mismatch, "
                    "see carelite/repro.py's module docstring",
                )
            )
            continue
        try:
            written = list(run_fn(output_dir))
            results.append(
                DownstreamResult(module=module_path, kind=kind, available=True, written=written)
            )
        except Exception as exc:  # the lane's own code failed; surface, don't hide
            results.append(
                DownstreamResult(
                    module=module_path,
                    kind=kind,
                    available=True,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def build_report() -> ReproReport:
    db = check_database()
    report = ReproReport(db_ok=db["connected"] and not db["errors"], db_errors=db["errors"])
    if not db["connected"]:
        return report  # nothing else is queryable
    report.stages = _stage_counts()

    settings = get_settings()
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    report.downstream = _run_downstream(settings.runs_dir / "repro")
    return report


def _fmt_stage(s: StageStatus) -> str:
    if s.error is not None and s.n_rows is None:
        return f"  {'MISSING':>9}  {s.stage} (`{s.table}`)"
    return f"  {s.n_rows:>9,}  {s.stage} (`{s.table}`)"


def render_report(report: ReproReport) -> str:
    lines: list[str] = ["carelite reproduce", "=" * 40]

    if not report.db_ok:
        lines.append("DATABASE UNREACHABLE OR SCHEMA MISSING")
        for err in report.db_errors:
            lines.append(f"  {err}")
        lines.append("")
        lines.append("Run `make db-up && make db-check` first — see REPRODUCE.md section 3.")
        return "\n".join(lines)

    lines.append("database: connected, schema present")
    lines.append("")
    lines.append("pipeline stage row counts:")
    for s in report.stages:
        lines.append(_fmt_stage(s))

    n_gen = report.holdout_generation_count
    lines.append("")
    if n_gen is None:
        lines.append("generation table is missing — schema is out of date, re-run `make db-up`.")
    elif n_gen == 0:
        lines.append(
            "0 generations. Nothing to reproduce yet — this is expected on a fresh clone that has "
            "only had the schema applied. See REPRODUCE.md section 7 for the multi-hour inference "
            "run that populates this table (no registration gate applies: DECISIONS.md D10 dropped "
            "OSF registration, so results are descriptive rather than gated on it)."
        )
    elif n_gen < EXPECTED_HOLDOUT_GENERATIONS:
        lines.append(
            f"{n_gen:,} of the expected {EXPECTED_HOLDOUT_GENERATIONS:,} holdout generations "
            f"present (60 scenarios x 6 conditions x 3 samples, docs/preregistration.md SS6). "
            "The full run has not completed — figures and tables below reflect a partial run."
        )
    else:
        lines.append(
            f"{n_gen:,} generations present — the full holdout run "
            f"({EXPECTED_HOLDOUT_GENERATIONS:,} expected) appears complete."
        )

    lines.append("")
    lines.append("downstream (tables and figures):")
    for d in report.downstream:
        if not d.available:
            lines.append(f"  [pending]  {d.module} not yet built (owned by a different lane)")
        elif d.error:
            lines.append(f"  [error]    {d.module}: {d.error}")
        else:
            lines.append(f"  [ok]       {d.module}: wrote {len(d.written)} {d.kind}")
            for p in d.written:
                lines.append(f"               {p}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="carelite.repro",
        description="Regenerate every figure and table from the database. No inference; DB-only.",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the report as JSON instead of the text summary"
    )
    args = parser.parse_args(argv)

    report = build_report()

    settings = get_settings()
    status_path = settings.runs_dir / "repro" / "status.md"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(render_report(report) + "\n")

    if args.json:
        import dataclasses
        import json

        def _default(o: Any) -> Any:
            if dataclasses.is_dataclass(o) and not isinstance(o, type):
                return dataclasses.asdict(o)
            if isinstance(o, Path):
                return str(o)
            raise TypeError(f"not JSON-serializable: {o!r}")

        print(json.dumps(report, default=_default, indent=2))
    else:
        print(render_report(report))

    return 0 if report.db_ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
