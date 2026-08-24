"""Every figure for the CARELite AI results, regenerated from the database.

`carelite.viz.reproduce.run(output_dir)` is the entry point `carelite/repro.py`
(`make reproduce`) calls. `carelite.viz.figures` holds the eight pure
`DataFrame -> matplotlib.figure.Figure` functions the fixture-driven tests in
`tests/unit/viz/` exercise directly, with no database or model involved.
`carelite.viz.queries` builds those DataFrames from Postgres (and the
retrieval ablation's JSON output) via `carelite.stats`, never recomputing its
statistics. `carelite.viz.style` is the shared colourblind-safe palette and
figure furniture (the n/test/pre-specified footer every figure carries).
"""

from __future__ import annotations

__all__: list[str] = []
