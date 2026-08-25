"""Unit tests for `carelite.repro`.

`build_report()` and `_stage_counts()` need a live database and are exercised by
`test_repro_db.py` under `@pytest.mark.db`. Everything here is pure: given a `ReproReport` or a
fake downstream module, does the rendering and the contract-checking logic do the right thing.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from carelite.repro import (
    DownstreamResult,
    ReproReport,
    StageStatus,
    _run_downstream,
    render_report,
)


def test_render_report_when_db_unreachable() -> None:
    report = ReproReport(db_ok=False, db_errors=["OperationalError: connection refused"])
    text = render_report(report)
    assert "UNREACHABLE" in text
    assert "connection refused" in text
    assert "make db-up" in text


def test_render_report_zero_generations_is_not_an_error() -> None:
    report = ReproReport(
        db_ok=True,
        db_errors=[],
        stages=[
            StageStatus(stage="scenario bank", table="scenario", n_rows=100),
            StageStatus(stage="generation", table="generation", n_rows=0),
        ],
    )
    text = render_report(report)
    assert "0 generations" in text
    assert "Nothing to reproduce yet" in text
    assert "100" in text  # scenario count still surfaced


def test_render_report_partial_run_says_so() -> None:
    report = ReproReport(
        db_ok=True,
        db_errors=[],
        stages=[StageStatus(stage="generation", table="generation", n_rows=500)],
    )
    text = render_report(report)
    assert "500" in text
    assert "939" in text  # EXPECTED_HOLDOUT_GENERATIONS_ACTUAL, not the original 1,080
    assert "partial run" in text


def test_render_report_complete_run_at_the_d11_figure() -> None:
    # 939, not 1,080 -- DECISIONS.md D11 stopped Condition LC at 39/180 cells by decision.
    report = ReproReport(
        db_ok=True,
        db_errors=[],
        stages=[StageStatus(stage="generation", table="generation", n_rows=939)],
    )
    text = render_report(report)
    assert "complete" in text
    assert "D11" in text
    assert "1,080" in text  # named as the superseded original figure, for contrast


def test_render_report_more_than_d11_expected_flags_for_a_look() -> None:
    report = ReproReport(
        db_ok=True,
        db_errors=[],
        stages=[StageStatus(stage="generation", table="generation", n_rows=1080)],
    )
    text = render_report(report)
    assert "1,080" in text
    assert "exceeding" in text


def test_render_report_missing_table_shown_distinctly_from_zero_rows() -> None:
    report = ReproReport(
        db_ok=True,
        db_errors=[],
        stages=[StageStatus(stage="graph layer", table="graph_edge", n_rows=None, error="boom")],
    )
    text = render_report(report)
    assert "MISSING" in text


def test_render_report_downstream_pending_when_module_absent() -> None:
    report = ReproReport(
        db_ok=True,
        db_errors=[],
        stages=[StageStatus(stage="generation", table="generation", n_rows=0)],
        downstream=[
            DownstreamResult(
                module="carelite.stats.reproduce", kind="tables", available=False, error="no module"
            )
        ],
    )
    text = render_report(report)
    assert "[pending]" in text
    assert "carelite.stats.reproduce" in text


def test_render_report_downstream_ok_lists_written_paths() -> None:
    report = ReproReport(
        db_ok=True,
        db_errors=[],
        stages=[StageStatus(stage="generation", table="generation", n_rows=0)],
        downstream=[
            DownstreamResult(
                module="carelite.viz.reproduce",
                kind="figures",
                available=True,
                written=[Path("figures/effect_sizes.png")],
            )
        ],
    )
    text = render_report(report)
    assert "[ok]" in text
    assert "effect_sizes.png" in text


def test_run_downstream_missing_module_reports_unavailable(tmp_path: Path) -> None:
    results = _run_downstream(tmp_path, modules=(("carelite._does_not_exist", "tables"),))
    assert len(results) == 1
    assert results[0].available is False
    assert results[0].error is not None


def test_run_downstream_module_without_run_reports_contract_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = types.ModuleType("carelite._fake_no_run")
    monkeypatch.setitem(sys.modules, "carelite._fake_no_run", fake)
    results = _run_downstream(tmp_path, modules=(("carelite._fake_no_run", "tables"),))
    assert results[0].available is True
    assert results[0].error is not None
    assert "run(output_dir)" in results[0].error


def test_run_downstream_calls_run_and_collects_written_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = types.ModuleType("carelite._fake_ok")
    written = [tmp_path / "a.csv", tmp_path / "b.csv"]
    fake.run = lambda output_dir: written  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "carelite._fake_ok", fake)
    results = _run_downstream(tmp_path, modules=(("carelite._fake_ok", "tables"),))
    assert results[0].available is True
    assert results[0].error is None
    assert results[0].written == written


def test_run_downstream_run_raising_is_reported_not_propagated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = types.ModuleType("carelite._fake_raises")

    def _boom(output_dir: Path) -> list[Path]:
        raise ValueError("bad output_dir")

    fake.run = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "carelite._fake_raises", fake)
    results = _run_downstream(tmp_path, modules=(("carelite._fake_raises", "tables"),))
    assert results[0].available is True
    assert results[0].error is not None
    assert "bad output_dir" in results[0].error
