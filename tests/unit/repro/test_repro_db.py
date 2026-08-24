"""DB-backed tests for `carelite.repro`. Requires a live Postgres with the schema applied."""

from __future__ import annotations

import pytest

from carelite.repro import PIPELINE_STAGES, build_report, render_report

pytestmark = pytest.mark.db


def test_build_report_against_live_schema() -> None:
    report = build_report()
    assert report.db_ok is True
    assert report.db_errors == []
    # Every pipeline stage table is queried, and the schema (loaded by `make db-up`) means none
    # of them should come back as MISSING even when their row count is legitimately zero.
    seen_tables = {s.table for s in report.stages}
    assert seen_tables == {table for _, table in PIPELINE_STAGES}
    for stage in report.stages:
        assert stage.error is None, f"{stage.table} should exist in a schema-applied database"
        assert stage.n_rows is not None
        assert stage.n_rows >= 0


def test_build_report_renders_without_raising() -> None:
    report = build_report()
    text = render_report(report)
    assert "carelite reproduce" in text
    assert "database: connected" in text
