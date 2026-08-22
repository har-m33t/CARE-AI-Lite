"""The seam `carelite-orchestrator` swaps in wave 3 — by registration, not by
editing anything under `carelite/cli/`.

`resolve_engine()` is the single place the CLI obtains a `GuidanceEngine`.
Wave 1 (this lane) ships only `StubEngine`. Wave 3 adds
`carelite/generate/engine.py` exposing a zero-argument factory:

    def build_engine() -> GuidanceEngine: ...

Once that module exists and importable, `resolve_engine()` picks it up
automatically on the next run — no file under `carelite/cli/` changes. If the
module is missing (wave 1/2) or raises `ImportError` for any reason (e.g. an
optional dependency not installed), we fall back to `StubEngine` rather than
crashing the terminal app.
"""

from __future__ import annotations

from carelite.cli.stub import StubEngine
from carelite.types import GuidanceEngine


def resolve_engine() -> GuidanceEngine:
    """Return the real engine if `carelite.generate.engine` is registered,
    else the wave-1 stub."""
    try:
        from carelite.generate.engine import build_engine
    except ImportError:
        return StubEngine()

    engine = build_engine()
    if not isinstance(engine, GuidanceEngine):
        raise TypeError(
            "carelite.generate.engine.build_engine() must return an object satisfying the "
            "GuidanceEngine protocol (a .guide(request) -> GuidanceResponse method)"
        )
    return engine
