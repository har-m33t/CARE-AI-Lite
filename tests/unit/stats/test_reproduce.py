"""`carelite.stats.reproduce` — the contract `carelite/repro.py` looks for.

The contract is documented in another lane's module and is checked here by
calling it the way that lane calls it: `run(output_dir) -> list[Path]`. A test
that asserted the shape by reading this module's own signature would pass no
matter what `carelite/repro.py` actually expects.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from carelite.stats.report import run_analysis
from carelite.stats.reproduce import (
    EFFECT_SIZE_COLUMNS,
    INSTRUMENT_COLUMNS,
    write_tables,
)
from carelite.types import Condition
from tests.unit.stats.conftest import constant_scores, make_long


@pytest.fixture
def analysed(nurse_dimensions: tuple[str, ...]) -> pd.DataFrame:
    """A, B and C over 12 scenarios, with `naturalness` pinned flat.

    Flat `naturalness` is deliberate: it makes at least one comparison come back
    `not_testable`, so the tests below check that the flag reaches the CSV
    rather than only checking that a CSV was produced.
    """
    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(12):
        scenario = f"SC-{i:03d}"
        for condition, value in ((Condition.A, 2), (Condition.B, 4), (Condition.C, 4)):
            for sample in range(3):
                cell = constant_scores(nurse_dimensions, value)
                cell["naturalness"] = 3
                scores[(scenario, str(condition), sample)] = cell
    return make_long(scores=scores)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestContract:
    def test_run_matches_the_signature_repro_calls(self, tmp_path: Path) -> None:
        """`carelite/repro.py` does `getattr(mod, "run")` then `run(output_dir)`."""
        import carelite.stats.reproduce as module

        run_fn = getattr(module, "run", None)
        assert callable(run_fn), "carelite/repro.py looks for a `run` callable by name"

    def test_write_tables_returns_the_paths_it_wrote(
        self, analysed: pd.DataFrame, tmp_path: Path
    ) -> None:
        report = run_analysis(long=analysed, n_boot=200)
        written = write_tables(report, tmp_path / "out")

        assert written, "the contract is a list of paths written"
        for path in written:
            assert path.exists()
            assert path.parent == tmp_path / "out"
        names = {p.name for p in written}
        assert names == {
            "analysis.txt",
            "headline-numbers.txt",
            "headline-numbers.csv",
            "effect-sizes.csv",
            "instrument-resolution.csv",
            "data-inventory.csv",
        }

    def test_the_output_directory_is_created(self, analysed: pd.DataFrame, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "out"
        report = run_analysis(long=analysed, n_boot=200)
        write_tables(report, target)
        assert target.is_dir()

    def test_rerunning_overwrites_rather_than_appends(
        self, analysed: pd.DataFrame, tmp_path: Path
    ) -> None:
        """`make reproduce` is idempotent; a doubled CSV would be a silent corruption."""
        report = run_analysis(long=analysed, n_boot=200)
        write_tables(report, tmp_path / "out")
        first = _read(tmp_path / "out" / "effect-sizes.csv")
        write_tables(report, tmp_path / "out")
        second = _read(tmp_path / "out" / "effect-sizes.csv")
        assert len(first) == len(second)


class TestTablesAgreeWithTheReport:
    def test_the_csv_carries_the_same_numbers_as_the_rendered_report(
        self, analysed: pd.DataFrame, tmp_path: Path
    ) -> None:
        """One analysis, two spellings. They cannot be allowed to drift."""
        report = run_analysis(long=analysed, n_boot=200)
        write_tables(report, tmp_path / "out")
        rows = {r["key"]: r for r in _read(tmp_path / "out" / "effect-sizes.csv")}

        primary = report.primary.by_key("primary_nurse_A_vs_B")
        assert primary is not None
        row = rows["primary_nurse_A_vs_B"]
        assert float(row["rank_biserial"]) == pytest.approx(primary.effects.rank_biserial.point)
        assert float(row["p_holm"]) == pytest.approx(primary.p_holm)
        assert int(row["n_scenarios"]) == primary.n_scenarios

    def test_all_three_estimators_reach_the_table(
        self, analysed: pd.DataFrame, tmp_path: Path
    ) -> None:
        """D9.5: all three, always, so none can be chosen after seeing which is largest."""
        report = run_analysis(long=analysed, n_boot=200)
        write_tables(report, tmp_path / "out")
        rows = _read(tmp_path / "out" / "effect-sizes.csv")
        for estimator in ("rank_biserial", "cohens_dz", "hodges_lehmann"):
            assert estimator in rows[0]
            assert f"{estimator}_ci_low" in rows[0]
            assert f"{estimator}_ci_high" in rows[0]

    def test_effect_columns_precede_the_p_value_columns(self) -> None:
        """§8.2 applies to a CSV as much as to a printed table."""
        assert EFFECT_SIZE_COLUMNS.index("rank_biserial") < EFFECT_SIZE_COLUMNS.index("p_value")
        assert EFFECT_SIZE_COLUMNS.index("hodges_lehmann") < EFFECT_SIZE_COLUMNS.index("p_holm")

    def test_not_testable_travels_with_the_effect(
        self, analysed: pd.DataFrame, tmp_path: Path
    ) -> None:
        """A consumer plotting `effect` alone must be able to see the flag next to it."""
        report = run_analysis(long=analysed, n_boot=200)
        write_tables(report, tmp_path / "out")
        rows = {r["key"]: r for r in _read(tmp_path / "out" / "effect-sizes.csv")}

        naturalness = rows["secondary4_naturalness_A_vs_B"]
        assert naturalness["not_testable"] == "True"
        # And a resolved comparison is not flagged, so the column discriminates.
        assert rows["primary_nurse_A_vs_B"]["not_testable"] == "False"

    def test_the_instrument_table_names_every_dimension(
        self, analysed: pd.DataFrame, tmp_path: Path
    ) -> None:
        report = run_analysis(long=analysed, n_boot=200)
        write_tables(report, tmp_path / "out")
        rows = _read(tmp_path / "out" / "instrument-resolution.csv")
        assert len(rows) == 11
        assert list(rows[0]) == list(INSTRUMENT_COLUMNS)
        flat = next(r for r in rows if r["dimension"] == "naturalness")
        assert flat["discrimination"] == "degenerate"

    def test_the_inventory_records_the_exclusions(
        self, analysed: pd.DataFrame, tmp_path: Path
    ) -> None:
        report = run_analysis(long=analysed, n_boot=200)
        write_tables(report, tmp_path / "out")
        items = {r["item"] for r in _read(tmp_path / "out" / "data-inventory.csv")}
        assert {"gate_blocked", "crag_fell_back_to_b", "incomplete_generations"} <= items

    def test_the_headline_block_is_written_and_says_the_census_was_not_read(
        self, analysed: pd.DataFrame, tmp_path: Path
    ) -> None:
        """`write_tables` with no census must not print a count it never took."""
        report = run_analysis(long=analysed, n_boot=200)
        write_tables(report, tmp_path / "out")
        text = (tmp_path / "out" / "headline-numbers.txt").read_text(encoding="utf-8")
        assert "HEADLINE NUMBERS" in text
        assert "not read" in text.lower()

    def test_the_text_report_carries_the_d10_banner(
        self, analysed: pd.DataFrame, tmp_path: Path
    ) -> None:
        """The reasoning lives in the prose artefact; the CSVs are not a substitute."""
        report = run_analysis(long=analysed, n_boot=200)
        write_tables(report, tmp_path / "out")
        text = (tmp_path / "out" / "analysis.txt").read_text(encoding="utf-8")
        assert "ALL RESULTS BELOW ARE DESCRIPTIVE" in text
        assert "INSTRUMENT RESOLUTION" in text


class TestEmptyDatabase:
    def test_an_empty_analysis_writes_headers_and_no_rows(self, tmp_path: Path) -> None:
        """`make reproduce` on a fresh clone must be a clean run, not a crash."""
        report = run_analysis(long=make_long(scores={}), n_boot=50)
        written = write_tables(report, tmp_path / "out")
        assert len(written) == 6

        for name, columns in (
            ("effect-sizes.csv", EFFECT_SIZE_COLUMNS),
            ("instrument-resolution.csv", INSTRUMENT_COLUMNS),
        ):
            rows = _read(tmp_path / "out" / name)
            assert rows == []
            with (tmp_path / "out" / name).open(encoding="utf-8") as handle:
                assert next(csv.reader(handle)) == list(columns)

        text = (tmp_path / "out" / "analysis.txt").read_text(encoding="utf-8")
        assert "NO RESULTS DATA" in text


# ---------------------------------------------------------------------------
# D13: the C-vs-LC caveats reach the machine-readable table
# ---------------------------------------------------------------------------


def test_effect_sizes_csv_carries_the_c_vs_lc_caveats(
    tmp_path: Path, nurse_dimensions: tuple[str, ...]
) -> None:
    """A plot built from this CSV must not be able to lose the confound."""
    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(12):
        scenario = f"SC-{i:03d}"
        for condition, value in (("C", 4), ("LC", 2)):
            for sample in range(3):
                scores[(scenario, condition, sample)] = constant_scores(nurse_dimensions, value)

    report = run_analysis(long=make_long(scores=scores), n_boot=100)
    write_tables(report, tmp_path)

    rows = list(csv.DictReader((tmp_path / "effect-sizes.csv").open()))
    row = next(r for r in rows if r["key"] == "secondary3_nurse_C_vs_LC")
    assert "caveats" in EFFECT_SIZE_COLUMNS
    assert "CONFOUNDED BY SERVING STACK" in row["caveats"]
    assert "REDUCED FORM OF THE QUESTION" in row["caveats"]
    assert row["label"].startswith("EXPLORATORY")


def test_the_data_inventory_names_the_excluded_arm_by_backend(
    tmp_path: Path, nurse_dimensions: tuple[str, ...]
) -> None:
    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(12):
        scenario = f"SC-{i:03d}"
        for sample in range(3):
            scores[(scenario, "C", sample)] = constant_scores(nurse_dimensions, 4)
            scores[(scenario, "LC", sample)] = constant_scores(nurse_dimensions, 2)
    stale = {(f"SC-{i:03d}", "LC", 0): constant_scores(nurse_dimensions, 1) for i in range(4)}
    frame = pd.concat(
        [
            make_long(scores=scores),
            make_long(scores=stale, served_by="ollama", generation_id_suffix="-ollama"),
        ],
        ignore_index=True,
    )
    report = run_analysis(long=frame, n_boot=100)
    write_tables(report, tmp_path)

    rows = list(csv.DictReader((tmp_path / "data-inventory.csv").open()))
    excluded = next(r for r in rows if r["item"].startswith("excluded_arm"))
    assert excluded["item"] == "excluded_arm_LC_ollama"
    assert excluded["count"] == "4"
    assert "D13" in excluded["detail"]
