"""`build_engine()` — the factory `carelite.cli.engine.resolve_engine()` looks for.

The CLI does not import anything from this package by name. It tries
`from carelite.generate.engine import build_engine` and falls back to its stub
on `ImportError`, so the real system is switched on by this module existing and
nothing under `carelite/cli/` changes. That is also why the imports at the top
of this file are cheap: pulling sentence-transformers or a database connection
in at import time would turn a missing optional dependency into a CLI that
silently runs on the stub.

`CareliteEngine.guide` is one call into the state machine. It holds a single
`GraphDeps` for the life of the process so an interactive session reuses one
Ollama client, one embedder and — under the long-context condition — one built
corpus pack, instead of rebuilding them on every turn.

Nothing here raises for a blocked turn. A red flag, an injection block or a
withheld output all return a `GuidanceResponse` whose safety verdicts say so,
which is what `carelite ask` reads to exit 3.

**`CARELITE_ENGINE=stub` pins the CLI to the fixture engine.** Setting it makes
this module refuse to import, which `resolve_engine()` already handles — its
contract is that *any* `ImportError` falls back to `StubEngine`. It exists
because the real engine calls a local model on every turn, so a test or a demo
that wants deterministic fixture output needs a way to ask for one that does not
involve deleting this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

ENGINE_ENV_VAR = "CARELITE_ENGINE"

if os.environ.get(ENGINE_ENV_VAR, "").strip().lower() == "stub":
    raise ImportError(
        f"{ENGINE_ENV_VAR}=stub: the real engine is switched off, so "
        "carelite.cli.engine.resolve_engine() falls back to StubEngine. Unset "
        f"{ENGINE_ENV_VAR} to use the real pipeline."
    )

from carelite.config import seed_for  # noqa: E402
from carelite.generate.graph import (  # noqa: E402
    GraphDeps,
    build_graph,
    initial_state,
    to_guidance_response,
)
from carelite.types import GuidanceRequest, GuidanceResponse  # noqa: E402

__all__ = ["ENGINE_ENV_VAR", "NUM_PREDICT", "CareliteEngine", "build_engine"]

#: Output budget for one clinician turn. A turn is a few sentences; 512 tokens is
#: roughly four times that, which leaves room for a model that warms up before it
#: says anything without inviting the wall of text the rubric's `ie` dimension
#: penalises and a bedside reader will not use.
NUM_PREDICT = 512


@dataclass
class CareliteEngine:
    """The real `GuidanceEngine`. One compiled graph, one set of collaborators."""

    deps: GraphDeps = field(default_factory=GraphDeps)
    graph: Any = field(default_factory=build_graph)

    def guide(self, request: GuidanceRequest) -> GuidanceResponse:
        """Run one turn through the state machine.

        A request that arrives without a seed gets one from `config.seed_for`
        rather than from the model's default. An interactive turn has no
        scenario id, so the utterance stands in for one — which means repeating
        the same sentence in the same condition gives the same answer, and a
        clinician who reruns a turn to see whether the suggestion was a fluke
        gets an honest answer to that question.
        """
        if request.seed is None:
            request = request.model_copy(
                update={"seed": seed_for(request.utterance, request.condition.value, 0)}
            )
        state = initial_state(request, deps=self.deps)
        final = self.graph.invoke(state)
        return to_guidance_response(final)


def build_engine() -> CareliteEngine:
    """Zero-argument factory. The exact name `resolve_engine()` imports."""
    return CareliteEngine(deps=GraphDeps(num_predict=NUM_PREDICT))
