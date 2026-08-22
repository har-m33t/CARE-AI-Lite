"""Fixtures for the `carelite-cli` test lane."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from carelite.cli.stub import StubEngine


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def stub_engine() -> StubEngine:
    return StubEngine()
