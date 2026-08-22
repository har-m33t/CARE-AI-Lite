"""CARELite AI terminal interface.

Everything here depends on the `GuidanceEngine` Protocol in `carelite.types`
and never on a concrete engine. See `carelite.cli.engine.resolve_engine` for
the seam `carelite-orchestrator` uses to swap in the real system in wave 3.
"""
