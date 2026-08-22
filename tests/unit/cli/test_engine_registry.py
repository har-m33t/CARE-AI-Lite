"""Tests for the wave-3 registration seam: `carelite.cli.engine.resolve_engine`.

`carelite-orchestrator` swaps in the real engine by adding
`carelite/generate/engine.py` with a `build_engine() -> GuidanceEngine`
factory — without ever touching `carelite/cli/`. These tests pin that
contract: today (no such module) we get the stub; once it exists and is
importable, we get whatever it builds.
"""

from __future__ import annotations

import sys
import types

import pytest

from carelite.cli.engine import resolve_engine
from carelite.cli.stub import StubEngine
from carelite.types import GuidanceRequest, GuidanceResponse


def test_resolves_to_stub_when_no_real_engine_is_registered():
    # `carelite.generate` does not exist yet in this checkout.
    assert "carelite.generate.engine" not in sys.modules
    engine = resolve_engine()
    assert isinstance(engine, StubEngine)


def test_picks_up_a_registered_engine_without_cli_changes(monkeypatch: pytest.MonkeyPatch):
    class _FakeRealEngine:
        def guide(self, request: GuidanceRequest) -> GuidanceResponse:
            return GuidanceResponse(text="real engine spoke", condition=request.condition)

    fake_instance = _FakeRealEngine()

    generate_pkg = types.ModuleType("carelite.generate")
    engine_mod = types.ModuleType("carelite.generate.engine")
    engine_mod.build_engine = lambda: fake_instance  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "carelite.generate", generate_pkg)
    monkeypatch.setitem(sys.modules, "carelite.generate.engine", engine_mod)

    engine = resolve_engine()
    assert engine is fake_instance
    response = engine.guide(GuidanceRequest(utterance="hello"))
    assert response.text == "real engine spoke"


def test_rejects_a_registered_engine_that_does_not_satisfy_the_protocol(
    monkeypatch: pytest.MonkeyPatch,
):
    class _NotAnEngine:
        pass

    generate_pkg = types.ModuleType("carelite.generate")
    engine_mod = types.ModuleType("carelite.generate.engine")
    engine_mod.build_engine = lambda: _NotAnEngine()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "carelite.generate", generate_pkg)
    monkeypatch.setitem(sys.modules, "carelite.generate.engine", engine_mod)

    with pytest.raises(TypeError):
        resolve_engine()
