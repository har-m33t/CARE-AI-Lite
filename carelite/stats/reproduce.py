"""`make reproduce`'s hook into this lane: write every analysis table to a directory.

`carelite/repro.py` documents the integration contract and looks for it by name:

    carelite.stats.reproduce.run(output_dir: Path) -> list[Path]

That module is `carelite-repro`'s and this one adopts its contract rather than
asking it to accommodate a different shape, which is what its docstring asks a
landing lane to do.

**No inference, and no recomputation of anything upstream.** This reads the
database and runs the same `carelite.stats.report.run_analysis` that
`python -m carelite.stats` runs. There is exactly one analysis in this package
and this is a second way of spelling it, not a second implementation — the
tables below are rendered from one `AnalysisReport`, so a number in a CSV and
the same number in the printed report cannot disagree.

**What gets written, and why these four.**

`analysis.txt` is the full rendered report, which is the artefact that carries
the reasoning: the exclusions, the instrument table, the ordering that keeps a
degenerate dimension's p-value from being read as a null result. The CSVs are
for figures and spreadsheets and are deliberately not a substitute for it.

`effect-sizes.csv` carries all three point estimators with their intervals for
every comparison, plus `not_testable` and `label`. A consumer that plots
`effect` without reading `not_testable` will draw a naturalness effect that
looks real and is not, so the column is in the table rather than in a note.

`instrument-resolution.csv` is the per-dimension spread that decides
`not_testable`. It is the evidence for the most consequential call in the
analysis and belongs in machine-readable form so a reader can move the
threshold themselves.

`data-inventory.csv` is what was excluded and what each exclusion cost.

**An empty database is a clean run, not a crash.** `make reproduce` on a fresh
clone that has only run `make db-up` must complete and say so. The report
renders its structure with no numbers in it and the CSVs come back with headers
and no rows, which is a truthful description of that state.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carelite.stats.effects import DEFAULT_N_BOOT
from carelite.stats.primary import PairwiseResult
from carelite.stats.report import AnalysisReport, run_analysis
from carelite.types import Split

__all__ = ["EFFECT_SIZE_COLUMNS", "INSTRUMENT_COLUMNS", "run", "write_tables"]

#: Column order for `effect-sizes.csv`. Effect estimates precede p-values, which
#: is the same §8.2 ordering every renderer in this package enforces -- a CSV is
#: a table like any other and the rule does not stop applying because it is
#: comma-separated.
EFFECT_SIZE_COLUMNS: tuple[str, ...] = (
    "key",
    "role",
    "comparison",
    "measure",
    "n_scenarios",
    "n_nonzero",
    "rank_biserial",
    "rank_biserial_ci_low",
    "rank_biserial_ci_high",
    "cohens_dz",
    "cohens_dz_ci_low",
    "cohens_dz_ci_high",
    "hodges_lehmann",
    "hodges_lehmann_ci_low",
    "hodges_lehmann_ci_high",
    "p_value",
    "p_holm",
    "family_size",
    "significant",
    "not_testable",
    "predicted_direction",
    "observed_direction",
    "planned_in_advance",
    "label",
)

#: Column order for `instrument-resolution.csv`.
INSTRUMENT_COLUMNS: tuple[str, ...] = (
    "dimension",
    "n_scored",
    "n_missing",
    "distinct_values",
    "mean_quality",
    "mean_raw",
    "sd",
    "modal_value_quality",
    "modal_share",
    "discrimination",
    "why",
)


@dataclass(frozen=True, slots=True)
class _Table:
    name: str
    columns: tuple[str, ...]
    rows: list[dict[str, Any]]


def _effect_rows(results: Sequence[PairwiseResult], role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        e = r.effects
        rows.append(
            {
                "key": r.hypothesis.key,
                "role": role,
                "comparison": r.hypothesis.pair_label,
                "measure": r.hypothesis.measure_key,
                "n_scenarios": r.n_scenarios,
                "n_nonzero": r.test.n_nonzero,
                "rank_biserial": e.rank_biserial.point,
                "rank_biserial_ci_low": e.rank_biserial.ci.low,
                "rank_biserial_ci_high": e.rank_biserial.ci.high,
                "cohens_dz": e.cohens_dz.point,
                "cohens_dz_ci_low": e.cohens_dz.ci.low,
                "cohens_dz_ci_high": e.cohens_dz.ci.high,
                "hodges_lehmann": e.hodges_lehmann.point,
                "hodges_lehmann_ci_low": e.hodges_lehmann.ci.low,
                "hodges_lehmann_ci_high": e.hodges_lehmann.ci.high,
                "p_value": r.test.p_value,
                "p_holm": r.p_holm,
                "family_size": r.family_size,
                "significant": r.significant(),
                # Deliberately adjacent to `significant`: a consumer reading one
                # without the other will report an untestable comparison as a
                # null one, which is the error this whole lane is arranged to
                # prevent.
                "not_testable": r.not_testable,
                "predicted_direction": r.hypothesis.expected_direction,
                "observed_direction": r.observed_direction,
                "planned_in_advance": r.hypothesis.prespecified,
                "label": r.label.tag(),
            }
        )
    return rows


def _tables(report: AnalysisReport) -> list[_Table]:
    effects = _effect_rows(report.primary.results, role="primary_family")
    if report.equity is not None:
        effects.extend(_effect_rows(report.equity.family.results, role="equity_subgroup"))
    if report.retrieval is not None:
        for label, result in (
            ("retrieval_offered", report.retrieval.offered),
            ("retrieval_retrieved_only", report.retrieval.retrieved),
        ):
            if result is not None:
                effects.extend(_effect_rows([result], role=label))

    instrument_rows: list[dict[str, Any]] = []
    if report.instrument is not None:
        for d in report.instrument.distributions:
            instrument_rows.append(
                {
                    "dimension": d.dimension,
                    "n_scored": d.n_scored,
                    "n_missing": d.n_missing,
                    "distinct_values": d.distinct,
                    "mean_quality": d.mean_quality,
                    "mean_raw": d.mean_raw,
                    "sd": d.sd,
                    "modal_value_quality": d.modal_value_quality,
                    "modal_share": d.modal_share,
                    "discrimination": d.discrimination.value,
                    "why": d.why,
                }
            )

    inventory_rows: list[dict[str, Any]] = []
    if report.inventory is not None:
        inv = report.inventory
        inventory_rows = [
            {"item": "scored_generations", "count": inv.n_generations, "detail": ""},
            {"item": "scenarios", "count": inv.n_scenarios, "detail": ""},
            {
                "item": "gate_blocked",
                "count": inv.n_gate_blocked,
                "detail": "by scenario: "
                + "; ".join(f"{k}={v}" for k, v in sorted(inv.gate_blocked_by_scenario.items())),
            },
            {
                "item": "crag_fell_back_to_b",
                "count": inv.n_fell_back,
                "detail": f"{inv.n_retrieved_cells} cells actually retrieved",
            },
            {
                "item": "incomplete_generations",
                "count": inv.n_incomplete_generations,
                "detail": "missing per dimension: "
                + "; ".join(f"{k}={v}" for k, v in sorted(inv.missing_by_dimension.items())),
            },
            *[
                {
                    "item": f"dropped_condition_{name}",
                    "count": count,
                    "detail": "DECISIONS.md D11",
                }
                for name, count in sorted(inv.dropped_conditions.items())
            ],
        ]

    return [
        _Table("effect-sizes.csv", EFFECT_SIZE_COLUMNS, effects),
        _Table("instrument-resolution.csv", INSTRUMENT_COLUMNS, instrument_rows),
        _Table("data-inventory.csv", ("item", "count", "detail"), inventory_rows),
    ]


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_tables(report: AnalysisReport, output_dir: Path) -> list[Path]:
    """Render one `AnalysisReport` to the directory. Pure given the report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    text_path = output_dir / "analysis.txt"
    text_path.write_text(report.render() + "\n", encoding="utf-8")
    written.append(text_path)

    for table in _tables(report):
        written.append(_write_csv(output_dir / table.name, table.columns, table.rows))
    return written


def run(
    output_dir: Path,
    *,
    split: Split | str = Split.HOLDOUT,
    n_boot: int = DEFAULT_N_BOOT,
) -> list[Path]:
    """`carelite.repro`'s entry point. Reads the database, writes the tables.

    Returns the paths written, which is what `carelite/repro.py` reports. Raises
    nothing it can help: a database that is reachable but empty produces a
    report that says so and CSVs with headers and no rows, because "0
    generations, nothing to reproduce yet" is a clean run.
    """
    return write_tables(run_analysis(split=split, n_boot=n_boot), Path(output_dir))
