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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from carelite.config import seed_for
from carelite.generate.graph import (
    GraphDeps,
    build_graph,
    initial_state,
    to_guidance_response,
)
from carelite.types import GuidanceRequest, GuidanceResponse

__all__ = ["NUM_PREDICT", "CareliteEngine", "build_engine"]

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
