"""carelite.generate — the running system, and the six conditions it runs as.

Everything else in this repository is a component. This package is where they
become an assistant: the state machine that takes a patient turn from the
terminal through safety, routing, retrieval, generation, verification and the
output gate, and the runner that drives the same machinery over the held-out
evaluation set.

    from carelite.generate.engine import build_engine     # what the CLI resolves
    from carelite.generate.graph import build_graph       # the state machine
    from carelite.generate.conditions import SPEC         # the six conditions
    from carelite.generate.runner import run              # inference lane III

Nothing is re-exported here. `engine`, `graph` and `runner` each pull in a
different slice of the system, and an aggregating `__init__` would make
importing the cheapest of them cost all three — which matters because
`carelite.cli.engine.resolve_engine()` imports `carelite.generate.engine` on
every CLI invocation and falls back to the stub on `ImportError`. A heavy
`__init__` would turn one missing optional dependency into a terminal that
silently runs on fixtures.

Two rules hold everywhere in this package.

**Every prompt is assembled by `carelite.safety.fencing.assemble`.** The patient
turn and the retrieved passages are untrusted — the first arrives from a
terminal, the second carries LLM-generated contextual prefixes over a corpus
that is itself a poisoning vector — and neither ever enters a system prompt.

**Every prompt is a versioned file in `carelite/prompts/`.** No system text is
built by string concatenation at call time. See that directory's README for the
format and for what is deliberately shared between conditions.
"""

from __future__ import annotations
