"""`carelite.stats.headline` — the numbers prose quotes, re-derived every run.

The failure this module exists to prevent is a document that carries a figure
forward from memory. So the tests here are mostly about *structure*: that a
number cannot be emitted without its qualification attached, that the count
breakdown separates the serving stacks rather than pooling them, and that the
two headline estimates are the same objects the report already computed rather
than a second derivation that could drift from it.

Known answers throughout. `flip_orientation` is checked against arithmetic done
by hand; the counts are checked against a frame whose totals were decided
before the function ran.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import pandas as pd
import pytest

from carelite.stats.data import SERVING_BACKENDS
from carelite.stats.headline import (
    HEADLINE_COLUMNS,
    GenerationCounts,
    HeadlineNumbers,
    flip_orientation,
    headline_numbers,
    write_headline,
)
from carelite.stats.report import run_analysis
from carelite.types import Condition
from tests.unit.stats.conftest import constant_scores, make_long

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "carelite" / "db" / "schema.sql"


@pytest.fixture
def analysed(nurse_dimensions: tuple[str, ...]) -> pd.DataFrame:
    """A and B over 12 scenarios, B cleanly higher, so B-vs-A is positive."""
    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(12):
        scenario = f"SC-{i:03d}"
        for condition, value in ((Condition.A, 2), (Condition.B, 4)):
            for sample in range(3):
                cell = constant_scores(nurse_dimensions, value)
                cell["naturalness"] = 3
                scores[(scenario, str(condition), sample)] = cell
    return make_long(scores=scores)


@pytest.fixture
def counts_frame() -> pd.DataFrame:
    """The shape `load_generation_counts` returns, with totals fixed in advance.

    180 each for A and B on Ollama, 39 LC on Ollama, 12 LC on vLLM: 411 rows,
    two backends, and an LC arm split across both — which is precisely the
    confound the backend breakdown exists to make visible.
    """
    return pd.DataFrame(
        [
            ("A", "ollama", "holdout", 180, 3, 180, 60),
            ("B", "ollama", "holdout", 180, 2, 180, 60),
            ("LC", "ollama", "holdout", 39, 0, 39, 13),
            ("LC", "vllm", "holdout", 12, 1, 0, 4),
        ],
        columns=[
            "condition",
            "served_by",
            "split",
            "n_generations",
            "n_gate_blocked",
            "n_scored",
            "n_scenarios",
        ],
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestOrientation:
    """A vs B and B vs A are the same difference with opposite signs."""

    def test_flip_negates_the_point_and_swaps_the_bounds(self) -> None:
        """Hand-computed: the run's primary A-vs-B shift, reoriented."""
        flipped = flip_orientation(-0.667, -0.833, -0.467)
        assert flipped == (pytest.approx(0.667), pytest.approx(0.467), pytest.approx(0.833))

    def test_flipping_twice_is_the_identity(self) -> None:
        once = flip_orientation(-0.667, -0.833, -0.467)
        assert flip_orientation(*once) == (
            pytest.approx(-0.667),
            pytest.approx(-0.833),
            pytest.approx(-0.467),
        )

    def test_a_nan_estimate_stays_nan_rather_than_becoming_a_number(self) -> None:
        point, low, high = flip_orientation(math.nan, math.nan, math.nan)
        assert math.isnan(point) and math.isnan(low) and math.isnan(high)


class TestServingBackends:
    def test_the_vocabulary_matches_the_schema_check_constraint(self) -> None:
        """`served_by` is a frozen contract; this fails if the schema gains a backend.

        The breakdown prints a zero for every backend in the vocabulary, so a
        backend the schema allows but this module has never heard of would go
        missing from the count rather than showing up as absent.
        """
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        match = re.search(r"served_by\s+IN\s+\(([^)]*)\)", sql)
        assert match is not None, "schema.sql no longer constrains served_by"
        in_schema = tuple(sorted(re.findall(r"'([a-z]+)'", match.group(1))))
        assert in_schema == tuple(sorted(SERVING_BACKENDS))


class TestGenerationCounts:
    def test_totals_are_the_sum_of_the_breakdown(self, counts_frame: pd.DataFrame) -> None:
        counts = GenerationCounts.from_frame(counts_frame)
        assert counts.total == 411
        assert sum(counts.by_condition.values()) == counts.total
        assert sum(counts.by_backend.values()) == counts.total

    def test_the_backend_breakdown_prints_a_zero_rather_than_omitting_a_backend(self) -> None:
        """A pooled count across two serving stacks would hide the confound."""
        frame = pd.DataFrame(
            [("A", "ollama", "holdout", 180, 0, 180, 60)],
            columns=[
                "condition",
                "served_by",
                "split",
                "n_generations",
                "n_gate_blocked",
                "n_scored",
                "n_scenarios",
            ],
        )
        counts = GenerationCounts.from_frame(frame)
        assert counts.by_backend == {"ollama": 180, "vllm": 0}

    def test_a_condition_split_across_backends_is_reported_split(
        self, counts_frame: pd.DataFrame
    ) -> None:
        counts = GenerationCounts.from_frame(counts_frame)
        assert counts.by_condition["LC"] == 51
        assert counts.by_condition_and_backend[("LC", "ollama")] == 39
        assert counts.by_condition_and_backend[("LC", "vllm")] == 12
        assert counts.split_across_backends == ("LC",)

    def test_scored_and_gate_blocked_are_counted_separately_from_the_total(
        self, counts_frame: pd.DataFrame
    ) -> None:
        counts = GenerationCounts.from_frame(counts_frame)
        assert counts.n_scored == 399
        assert counts.n_gate_blocked == 6

    def test_an_empty_frame_is_a_read_that_found_nothing(self) -> None:
        """Different from a read that did not happen. Both are represented."""
        counts = GenerationCounts.from_frame(pd.DataFrame())
        assert counts.read is True
        assert counts.total == 0
        assert GenerationCounts.not_read().read is False


class TestEveryNumberCarriesItsQualification:
    def test_no_row_is_emitted_without_a_qualifier(
        self, analysed: pd.DataFrame, counts_frame: pd.DataFrame
    ) -> None:
        """The structural invariant this whole module exists for."""
        report = run_analysis(long=analysed, n_boot=200)
        head = headline_numbers(report, counts=GenerationCounts.from_frame(counts_frame))
        assert head.rows
        for row in head.rows:
            assert row.qualifier.strip(), f"{row.key} carries a bare number"

    def test_the_qualifier_column_sits_immediately_after_the_value(self) -> None:
        """Same rule `effect-sizes.csv` applies to `not_testable`: adjacency."""
        assert HEADLINE_COLUMNS.index("qualifier") == HEADLINE_COLUMNS.index("value") + 1

    def test_the_effect_columns_precede_the_p_value(self) -> None:
        """§8.2 holds in this table too."""
        assert HEADLINE_COLUMNS.index("value") < HEADLINE_COLUMNS.index("p_value")
        assert HEADLINE_COLUMNS.index("ci_high") < HEADLINE_COLUMNS.index("p_value")

    def test_the_text_block_prints_the_qualifier_under_every_number(
        self, analysed: pd.DataFrame, counts_frame: pd.DataFrame
    ) -> None:
        report = run_analysis(long=analysed, n_boot=200)
        head = headline_numbers(report, counts=GenerationCounts.from_frame(counts_frame))
        text = head.render()
        for row in head.rows:
            assert row.qualifier.splitlines()[0] in text

    def test_the_block_carries_the_d10_scope_statement(
        self, analysed: pd.DataFrame, counts_frame: pd.DataFrame
    ) -> None:
        report = run_analysis(long=analysed, n_boot=200)
        text = headline_numbers(report, counts=GenerationCounts.from_frame(counts_frame)).render()
        assert "ALL RESULTS BELOW ARE DESCRIPTIVE" in text
        assert "EXPLORATORY" in text

    def test_the_primary_estimates_are_labelled_exploratory_in_the_row_itself(
        self, analysed: pd.DataFrame, counts_frame: pd.DataFrame
    ) -> None:
        """Not only in the surrounding prose — a CSV consumer sees it too."""
        report = run_analysis(long=analysed, n_boot=200)
        head = headline_numbers(report, counts=GenerationCounts.from_frame(counts_frame))
        for key in ("primary_b_vs_a_coefficient", "primary_b_vs_a_hodges_lehmann"):
            row = head.by_key(key)
            assert row is not None
            assert "EXPLORATORY" in row.qualifier


class TestTheNumbersAreTheReportsOwn:
    def test_the_coefficient_is_the_fitted_mixed_model_effect(
        self, analysed: pd.DataFrame, counts_frame: pd.DataFrame
    ) -> None:
        """Not recomputed here. A second derivation is a second thing to drift."""
        report = run_analysis(long=analysed, n_boot=200)
        head = headline_numbers(report, counts=GenerationCounts.from_frame(counts_frame))
        model = next(m for m in report.mixed if m.measure_key == "nurse_composite")
        effect = next(e for e in model.effects if e.term == "B vs A")

        row = head.by_key("primary_b_vs_a_coefficient")
        assert row is not None
        assert row.value == pytest.approx(effect.coefficient)
        assert row.ci_low == pytest.approx(effect.ci_low)
        assert row.ci_high == pytest.approx(effect.ci_high)
        assert row.p_value == pytest.approx(effect.p_value)

    def test_the_hodges_lehmann_is_the_family_estimate_reoriented(
        self, analysed: pd.DataFrame, counts_frame: pd.DataFrame
    ) -> None:
        """The §8.1 family computes A vs B; the headline reports B vs A."""
        report = run_analysis(long=analysed, n_boot=200)
        head = headline_numbers(report, counts=GenerationCounts.from_frame(counts_frame))
        primary = report.primary.by_key("primary_nurse_A_vs_B")
        assert primary is not None
        hl = primary.effects.hodges_lehmann

        row = head.by_key("primary_b_vs_a_hodges_lehmann")
        assert row is not None
        assert row.value == pytest.approx(-hl.point)
        assert row.ci_low == pytest.approx(-hl.ci.high)
        assert row.ci_high == pytest.approx(-hl.ci.low)
        assert "reorient" in row.qualifier.lower()

    def test_the_two_estimates_agree_in_sign(
        self, analysed: pd.DataFrame, counts_frame: pd.DataFrame
    ) -> None:
        """Both are B relative to A. Opposite signs would mean one is misoriented."""
        report = run_analysis(long=analysed, n_boot=200)
        head = headline_numbers(report, counts=GenerationCounts.from_frame(counts_frame))
        coefficient = head.by_key("primary_b_vs_a_coefficient")
        shift = head.by_key("primary_b_vs_a_hodges_lehmann")
        assert coefficient is not None and shift is not None
        assert coefficient.value is not None and shift.value is not None
        assert coefficient.value > 0 and shift.value > 0

    def test_the_analysed_count_is_not_the_database_count(
        self, analysed: pd.DataFrame, counts_frame: pd.DataFrame
    ) -> None:
        """The distinction the `Additional_builds.md` drift collapsed."""
        report = run_analysis(long=analysed, n_boot=200)
        head = headline_numbers(report, counts=GenerationCounts.from_frame(counts_frame))
        total = head.by_key("generations_total")
        analysed_row = head.by_key("generations_analysed")
        assert total is not None and analysed_row is not None
        assert total.value == 411
        assert analysed_row.value == report.n_generations
        assert "not the number any comparison ran on" in total.qualifier


class TestCountsNotRead:
    def test_a_count_that_was_not_read_says_so_rather_than_reading_zero(
        self, analysed: pd.DataFrame
    ) -> None:
        """`not read` and `0` are different claims and must not render alike."""
        report = run_analysis(long=analysed, n_boot=200)
        head = headline_numbers(report, counts=None)
        row = head.by_key("generations_total")
        assert row is not None
        assert row.value is None
        assert "not read" in row.display.lower()
        assert "not read" in head.render().lower()


class TestWriting:
    def test_both_artefacts_are_written(
        self, analysed: pd.DataFrame, counts_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        report = run_analysis(long=analysed, n_boot=200)
        head = headline_numbers(report, counts=GenerationCounts.from_frame(counts_frame))
        written = write_headline(head, tmp_path / "out")
        assert {p.name for p in written} == {"headline-numbers.txt", "headline-numbers.csv"}
        for path in written:
            assert path.exists()

    def test_the_csv_and_the_text_carry_the_same_numbers(
        self, analysed: pd.DataFrame, counts_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        report = run_analysis(long=analysed, n_boot=200)
        head = headline_numbers(report, counts=GenerationCounts.from_frame(counts_frame))
        write_headline(head, tmp_path / "out")
        rows = {r["key"]: r for r in _rows(tmp_path / "out" / "headline-numbers.csv")}

        assert list(rows) == [r.key for r in head.rows]
        coefficient = head.by_key("primary_b_vs_a_coefficient")
        assert coefficient is not None and coefficient.value is not None
        assert float(rows["primary_b_vs_a_coefficient"]["value"]) == pytest.approx(
            coefficient.value
        )
        assert rows["generations_total"]["value"] == "411"

    def test_every_csv_row_carries_a_qualifier(
        self, analysed: pd.DataFrame, counts_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        report = run_analysis(long=analysed, n_boot=200)
        head = headline_numbers(report, counts=GenerationCounts.from_frame(counts_frame))
        write_headline(head, tmp_path / "out")
        for row in _rows(tmp_path / "out" / "headline-numbers.csv"):
            assert row["qualifier"].strip()

    def test_rerunning_overwrites_rather_than_appends(
        self, analysed: pd.DataFrame, counts_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        report = run_analysis(long=analysed, n_boot=200)
        head = headline_numbers(report, counts=GenerationCounts.from_frame(counts_frame))
        write_headline(head, tmp_path / "out")
        first = _rows(tmp_path / "out" / "headline-numbers.csv")
        write_headline(head, tmp_path / "out")
        assert len(_rows(tmp_path / "out" / "headline-numbers.csv")) == len(first)


class TestEmptyDatabase:
    def test_an_empty_analysis_is_a_clean_run(self, tmp_path: Path) -> None:
        """`make reproduce` on a fresh clone must complete and say so."""
        report = run_analysis(long=make_long(scores={}), n_boot=50)
        head = headline_numbers(report, counts=GenerationCounts.from_frame(pd.DataFrame()))
        written = write_headline(head, tmp_path / "out")
        assert len(written) == 2

        rows = {r.key: r for r in head.rows}
        assert rows["generations_total"].value == 0
        assert rows["primary_b_vs_a_coefficient"].value is None
        assert "not computed" in rows["primary_b_vs_a_coefficient"].display.lower()
        assert isinstance(head.render(), str)


class TestNoLongContextArmIsAssumed:
    """A later package adds C vs LC. Nothing here may pretend those rows exist."""

    def test_the_block_names_no_c_vs_lc_result(
        self, analysed: pd.DataFrame, counts_frame: pd.DataFrame
    ) -> None:
        head: HeadlineNumbers = headline_numbers(
            run_analysis(long=analysed, n_boot=200),
            counts=GenerationCounts.from_frame(counts_frame),
        )
        assert not any("lc" in r.key.split("_") for r in head.rows)
