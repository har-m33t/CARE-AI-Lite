"""Tests for the wave-3 registration seam: `carelite.cli.engine.resolve_engine`.

`carelite-orchestrator` swaps in the real engine by adding
`carelite/generate/engine.py` with a `build_engine() -> GuidanceEngine`
factory — without ever touching `carelite/cli/`. These tests pin that
contract: if the module cannot be imported we get the stub; once it exists
and is importable, we get whatever it builds.

`carelite/generate/engine.py` landed for real in `73071d5`, so the "cannot be
imported" half of the contract is no longer this checkout's natural state —
it has to be simulated. `test_resolves_to_stub_when_no_real_engine_is_registered`
does that by making `carelite.generate.engine` look absent from `sys.modules`
for the duration of the test (setting it to `None`, which is what makes
`import` raise `ImportError` regardless of whether the real module has
already been imported elsewhere in this test session and cached), rather than
by asserting the module is genuinely missing.
"""

from __future__ import annotations

import sys
import types

import pytest

from carelite.cli.engine import resolve_engine
from carelite.cli.stub import StubEngine
from carelite.types import GuidanceRequest, GuidanceResponse


def test_resolves_to_stub_when_no_real_engine_is_registered(monkeypatch: pytest.MonkeyPatch):
    # `carelite/generate/engine.py` is real now (`73071d5`) and may already be
    # cached in `sys.modules` from elsewhere in this test session, so we
    # cannot rely on it genuinely being absent. Forcing the entry to `None`
    # makes `import carelite.generate.engine` raise `ImportError` regardless
    # of any prior import — the same "missing or unimportable" case
    # `resolve_engine()` is contracted to fall back on.
    monkeypatch.setitem(sys.modules, "carelite.generate.engine", None)
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
