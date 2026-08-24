"""Fixtures for the `carelite-cli` test lane."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import carelite.cli.main as cli_main
from carelite.cli.stub import StubEngine


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    """A `CliRunner` wired to `StubEngine`, regardless of what
    `carelite.cli.engine.resolve_engine()` would otherwise resolve to.

    `resolve_engine()` now finds the real, wave-3 `CareliteEngine`
    (`carelite/generate/engine.py`, landed in `73071d5`) — a live Ollama call
    and a live HuggingFace Hub retrieval fetch on every turn. This lane's
    Definition of Done is explicit that "no test requires a live model," and
    every command test in this package goes through this fixture, so pinning
    the engine here once is what keeps that true regardless of which real
    engine gets registered under us.
    """
    monkeypatch.setattr(cli_main, "resolve_engine", StubEngine)
    return CliRunner()


@pytest.fixture
def stub_engine() -> StubEngine:
    return StubEngine()
