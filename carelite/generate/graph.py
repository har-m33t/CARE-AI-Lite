"""The state machine: safety -> route -> retrieve -> generate -> self-check -> gate.

    from carelite.generate.graph import GraphDeps, build_graph, initial_state

    graph = build_graph()
    final = graph.invoke(initial_state(request, deps=GraphDeps()))

Six nodes and four conditional edges. **No node branches on a `Condition`
value.** Every branch reads a field of the `ConditionSpec` in the state, which
is what makes "the six conditions share one code path" a property of the code
rather than a claim about it — `tests/unit/generate/test_graph.py` asserts that
the module contains no comparison against a member of `Condition`.

```
        START
          |
   safety_screen ----- blocked ------> END
          |
        route
          |
   spec.use_retrieval ? retrieve : generate
          |
       generate ------- failed ------> END
          |
   spec.self_check ? self_check : output_gate
          |
     output_gate
          |
         END
```

**LangGraph.** The topology below is declared once, in `NODES` and `EDGES`. When
`langgraph` is importable, `to_langgraph()` compiles that same declaration into a
`StateGraph`; otherwise `build_graph()` returns a small executor that walks the
same declaration. `langgraph` is now an `orchestration` extra in
`pyproject.toml` and `build_graph()` prefers it, so the compiled `StateGraph` is
what runs on a machine that has it. The executor is not dead code: it is what
produced every generation in the database before the extra existed, and a
machine without the extra still runs the whole system on it.

The two are driven by one topology on purpose — a fallback that reimplemented
the flow would be a second program to keep in step with the first — and
`tests/unit/generate/test_graph_langgraph.py` runs a fixed set of seeded states
through both compilers and asserts the final states are identical on every
branch. That is what makes "one topology, two compilers" a property of the code
rather than a claim in this docstring.

**Why the two agree on the merge.** The executor applies a node's return with
`dict.update`; LangGraph merges it into channels. Those coincide here because
`TurnState` declares no reducers, so every field is a last-write-wins channel and
the topology gives each field exactly one writer per step. The one field that
accumulates — `errors` — is written as a whole new list built from the previous
value rather than as an appended fragment, which is correct under both and would
be correct under a reducer as well. There is one difference and it is not in the
state: the executor mutates the mapping it was handed and returns it, while
LangGraph returns a fresh one. Callers take the return value, and the runner
copies it, so nothing depends on the aliasing.

**What the safety layer does at each end.** The input screen runs before
anything else and a red flag ends the turn — the tool escalates rather than
coaching, and a red-flag turn never reaches a model. The output gate runs after
generation and after any revision, so a self-check that introduced a clinical
recommendation is still caught. Every prompt in between is built by
`fencing.assemble`, so the patient turn and the retrieved passages are inside
data fences in the user turn and the system prompt is git-tracked template text.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypedDict

from carelite.generate import prompts
from carelite.generate.backend import default_client
from carelite.generate.conditions import ConditionSpec, spec_for
from carelite.generate.model import DIGEST_UNAVAILABLE, GenerationError, ModelClient
from carelite.generate.selfcheck import SelfCheckResult, run_self_check
from carelite.safety import fencing, screen_input, screen_output
from carelite.types import (
    GuidanceRequest,
    GuidanceResponse,
    RetrievalTrace,
    RetrievedItem,
    SafetyVerdict,
)

__all__ = [
    "EDGES",
    "END",
    "NODES",
    "GraphDeps",
    "InputPolicy",
    "TurnState",
    "build_graph",
    "initial_state",
    "to_guidance_response",
    "to_langgraph",
]

END = "__end__"


class InputPolicy(StrEnum):
    """Where the utterance came from, which decides what the PHI screen means.

    `TERMINAL` is the default and the safe one: a turn typed at the bedside may
    contain a real identifier, so a PHI hit redacts the text before it reaches a
    model and marks the turn as not-for-storage.

    `CURATED_BANK` is for the frozen evaluation scenarios. They are synthetic,
    they are committed to a public repository, and they are already rows in the
    `scenario` table, so the storage rule has nothing left to protect. It
    matters because the PHI name detector fires on two held-out scenarios — one
    a false positive on "thank you doctor. Everything", one on the synthetic
    clinician name "Dr. Aziz" — and redacting them would mean generating against
    a mangled version of a scenario whose text is frozen. Under this policy the
    screen still runs and its flags are still recorded on the result; only the
    redaction and the storage veto are lifted.
    """

    TERMINAL = "terminal"
    CURATED_BANK = "curated_bank"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class TurnState(TypedDict, total=False):
    """Everything one turn accumulates. A `TypedDict` so LangGraph can own it."""

    request: GuidanceRequest
    spec: ConditionSpec
    deps: GraphDeps

    utterance: str
    input_safety: SafetyVerdict
    may_persist: bool
    halted: bool
    halt_reason: str

    trace: RetrievalTrace | None
    context: list[RetrievedItem]
    context_note: dict[str, Any]

    prompt_id: str
    system_text: str
    draft: str
    text: str
    model: str
    model_digest: str
    served_by: str
    num_ctx: int
    prompt_chars: int

    self_check: SelfCheckResult | None
    self_check_passed: bool
    output_safety: SafetyVerdict | None

    started_at: float
    latency_ms: int
    errors: list[str]


@dataclass
class GraphDeps:
    """Injectable collaborators. Every one defaults to the real thing.

    A unit test builds a `GraphDeps` of fakes and drives the whole graph with no
    daemon and no database; the ablation harness reuses one embedder and one
    loaded cross-encoder across many turns instead of rebuilding them per call.
    """

    client: ModelClient = field(default_factory=default_client)
    """Ollama, or a remote vLLM server under `CARELITE_BACKEND=vllm`. The swap
    is the whole of the backend change: nothing else in this module, in
    `conditions.py` or in `runner.py` knows which stack it is running on."""

    embedder: Any | None = None
    retrieval_generator: Any | None = None
    grader_client: Any | None = None
    reranker: Any | None = None
    corpus_pack: Any | None = None
    """A prebuilt `longcontext.CorpusPack`. The pack is one database read of the
    entire corpus, so the runner builds it once for all long-context cells
    rather than 180 times."""

    input_policy: InputPolicy = InputPolicy.TERMINAL
    num_predict: int = 512
    retrieve_fn: Callable[..., Any] | None = None
    """Overrides `carelite.retrieval.retrieve_detailed`. For tests only."""


def initial_state(request: GuidanceRequest, *, deps: GraphDeps | None = None) -> TurnState:
    spec = spec_for(request.condition)
    deps = deps if deps is not None else GraphDeps()
    return TurnState(
        request=request,
        spec=spec,
        deps=deps,
        utterance=request.utterance,
        # Set here rather than only in `generate`, so a turn the safety screen
        # ends before any model is reached still says which stack the run was
        # configured against.
        served_by=deps.client.served_by,
        may_persist=True,
        halted=False,
        context=[],
        context_note={},
        trace=None,
        self_check=None,
        self_check_passed=True,
        output_safety=None,
        started_at=time.monotonic(),
        errors=[],
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def safety_screen(state: TurnState) -> dict[str, Any]:
    """Red flag, injection, PHI. The only node that can end a turn before a model."""
    deps = state["deps"]
    verdict = screen_input(state["request"].utterance)

    if not verdict.allowed and (verdict.red_flag or verdict.injection_detected):
        return {
            "input_safety": verdict,
            "halted": True,
            "halt_reason": verdict.reason or "input screen blocked this turn",
            "text": "",
            "may_persist": False,
        }

    if verdict.phi_detected and deps.input_policy is InputPolicy.TERMINAL:
        from carelite.safety import phi

        return {
            "input_safety": verdict,
            "utterance": verdict.redacted_text or phi.redact(state["request"].utterance),
            "may_persist": False,
        }

    # Either nothing fired, or a PHI hit under the curated-bank policy: the
    # flags are kept on the verdict either way, so the screen's opinion is in
    # the record even where the policy declines to act on it.
    return {"input_safety": verdict, "may_persist": True}


def route(state: TurnState) -> dict[str, Any]:
    """Classify the turn. Recorded for every condition; acted on only where retrieval runs.

    The deterministic lexicon classifier, never the LLM router: an LLM in the
    router adds per-turn variance to a controlled comparison. `retrieve` routes
    again internally through the same function and reaches the same answer.
    """
    from carelite.retrieval import classify

    decision = classify(state["utterance"])
    note = dict(state.get("context_note") or {})
    note["route"] = decision.route.value
    note["route_should_retrieve"] = decision.should_retrieve
    return {"context_note": note}


def retrieve(state: TurnState) -> dict[str, Any]:
    """Run the hybrid pipeline and take its trace. Only reached when configured."""
    from carelite.retrieval import RetrievalFlags, preset
    from carelite.retrieval.pipeline import retrieve_detailed

    spec = state["spec"]
    deps = state["deps"]
    flags = preset(spec.retrieval_preset) if spec.retrieval_preset else RetrievalFlags()
    fn = deps.retrieve_fn or retrieve_detailed

    try:
        result = fn(
            state["utterance"],
            encounter_phase=state["request"].encounter_phase,
            flags=flags,
            embedder=deps.embedder,
            generator=deps.retrieval_generator,
            grader_client=deps.grader_client,
            reranker=deps.reranker,
        )
    except Exception as exc:
        # Retrieval is one leg of one condition. Losing it degrades that
        # condition to its own CRAG-fallback behaviour, which is a defined
        # state, and the error is recorded rather than ending the turn.
        errors = [*state.get("errors", []), f"retrieval failed: {type(exc).__name__}: {exc}"]
        return {"errors": errors, "context": [], "trace": None}

    trace = result.trace
    return {
        "trace": trace,
        # `trace.retrieved` is empty exactly when no evidence may be used —
        # emotional-only route, or a CRAG `NONE` verdict. Reading it directly is
        # what makes that invariant load-bearing here.
        "context": list(trace.retrieved),
    }


def _long_context_items(state: TurnState) -> tuple[list[RetrievedItem], dict[str, Any]]:
    from carelite.generate import longcontext

    deps = state["deps"]
    pack = deps.corpus_pack
    if pack is None:
        pack = longcontext.build_pack()
        deps.corpus_pack = pack
    return list(pack.items), {"long_context": pack.coverage}


def generate(state: TurnState) -> dict[str, Any]:
    """Assemble the fenced prompt and produce the draft turn."""
    spec = state["spec"]
    deps = state["deps"]
    request = state["request"]

    context = list(state.get("context") or [])
    note = dict(state.get("context_note") or {})
    if spec.use_long_context:
        try:
            context, extra = _long_context_items(state)
            note.update(extra)
        except Exception as exc:
            return {
                "halted": True,
                "halt_reason": f"could not build the long-context pack: {exc}",
                "errors": [*state.get("errors", []), str(exc)],
                "text": "",
            }

    template = prompts.load(spec.prompt_id)
    system_text = prompts.assembled_text(spec.prompt_id)
    prompt = fencing.assemble(
        system=system_text,
        task=template.task,
        utterance=state["utterance"],
        retrieved=context,
        history=list(request.history),
    )

    seed = request.seed if request.seed is not None else 0
    try:
        out = deps.client.generate(
            prompt,
            model_tag=spec.model_tag,
            seed=seed,
            temperature=request.temperature,
            num_predict=deps.num_predict,
            window=spec.context_window,
        )
    except GenerationError as exc:
        return {
            "halted": True,
            "halt_reason": str(exc),
            "errors": [*state.get("errors", []), str(exc)],
            "text": "",
            "prompt_id": spec.prompt_id,
            "system_text": system_text,
            "model": spec.model_tag,
            "model_digest": DIGEST_UNAVAILABLE,
            "served_by": deps.client.served_by,
            "context": context,
            "context_note": note,
        }

    return {
        "prompt_id": spec.prompt_id,
        "system_text": system_text,
        "draft": out.text,
        "text": out.text,
        "model": out.model,
        "model_digest": out.model_digest,
        "served_by": out.served_by,
        "num_ctx": out.num_ctx,
        "prompt_chars": out.prompt_chars,
        "context": context,
        "context_note": note,
    }


def self_check(state: TurnState) -> dict[str, Any]:
    """The CoVe-style verification pass. Only reached when configured."""
    spec = state["spec"]
    deps = state["deps"]
    request = state["request"]
    result = run_self_check(
        state["draft"],
        utterance=state["utterance"],
        retrieved=list(state.get("context") or []),
        client=deps.client,
        model_tag=spec.model_tag,
        seed=request.seed if request.seed is not None else 0,
        window=spec.context_window,
    )
    return {
        "self_check": result,
        "self_check_passed": result.passed,
        "text": result.text,
    }


def output_gate(state: TurnState) -> dict[str, Any]:
    """The last check before a human sees anything.

    The system text is passed so the verbatim-overlap check runs: it is the only
    instruction-leak check that does not depend on someone having anticipated the
    phrasing of the attack.
    """
    verdict = screen_output(state["text"], system_prompt=state.get("system_text"))
    if verdict.allowed:
        return {"output_safety": verdict}
    return {
        "output_safety": verdict,
        "halted": True,
        "halt_reason": verdict.reason or "output gate withheld this response",
        "text": "",
    }


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

NODES: dict[str, Callable[[TurnState], dict[str, Any]]] = {
    "safety_screen": safety_screen,
    "route": route,
    "retrieve": retrieve,
    "generate": generate,
    "self_check": self_check,
    "output_gate": output_gate,
}

START = "safety_screen"


def _after_safety(state: TurnState) -> str:
    return END if state.get("halted") else "route"


def _after_route(state: TurnState) -> str:
    return "retrieve" if state["spec"].use_retrieval else "generate"


def _after_generate(state: TurnState) -> str:
    if state.get("halted"):
        return END
    return "self_check" if state["spec"].self_check else "output_gate"


#: node -> either a fixed successor or a predicate choosing one. Declared once;
#: both `build_graph()` and `to_langgraph()` compile this and nothing else.
EDGES: dict[str, str | Callable[[TurnState], str]] = {
    "safety_screen": _after_safety,
    "route": _after_route,
    "retrieve": "generate",
    "generate": _after_generate,
    "self_check": "output_gate",
    "output_gate": END,
}

#: The complete set of targets each conditional edge can choose between. Kept
#: beside `EDGES` because LangGraph needs it declared and because it is the
#: readable statement of what the branches actually are.
CONDITIONAL_TARGETS: dict[str, tuple[str, ...]] = {
    "safety_screen": (END, "route"),
    "route": ("retrieve", "generate"),
    "generate": (END, "self_check", "output_gate"),
}


class _Executor:
    """Walks `NODES`/`EDGES`. The fallback for when `langgraph` is not installed."""

    def __init__(self, max_steps: int = 32) -> None:
        self.max_steps = max_steps

    def invoke(self, state: TurnState, *, thread_id: str | None = None) -> TurnState:
        """Run one turn. `thread_id` is accepted and ignored.

        The executor has nowhere to checkpoint to, so it cannot resume a turn
        part-way. Taking the argument anyway means the runner has one call
        shape for both compilers rather than a branch on which one it holds.
        """
        current = START
        for _ in range(self.max_steps):
            if current == END:
                break
            state.update(NODES[current](state))  # type: ignore[typeddict-item]
            edge = EDGES[current]
            current = edge(state) if callable(edge) else edge
        else:  # pragma: no cover - the topology is acyclic
            raise RuntimeError("turn graph did not terminate")
        state["latency_ms"] = int((time.monotonic() - state["started_at"]) * 1000)
        return state


@dataclass
class _CompiledLangGraph:
    """A compiled `StateGraph` behind the same `invoke` the executor offers.

    Two things live here rather than in the graph itself.

    **`latency_ms` is stamped after the walk.** It is a column on `generation`
    and it is not the product of any node — every node can be skipped by some
    branch, so there is nowhere in the topology to set it that every path
    reaches. The executor set it after its loop and the compiled graph, having
    no such loop, silently wrote `NULL` into that column on every row it would
    have produced.

    **Resumption is decided here.** With a checkpointer attached, a turn that
    was interrupted part-way is continued from its last completed node instead
    of restarted, which for the long-context condition is the difference
    between re-prefilling ~119,500 tokens and not.
    """

    app: Any
    checkpointed: bool = False

    def invoke(self, state: TurnState, *, thread_id: str | None = None) -> TurnState:
        started = time.monotonic()
        if not self.checkpointed or thread_id is None:
            final = dict(self.app.invoke(state))
            final["latency_ms"] = int((time.monotonic() - state["started_at"]) * 1000)
            return final  # type: ignore[return-value]

        from carelite.generate.checkpoint import bind_live_deps

        # The checkpoint deliberately does not contain the live collaborators —
        # see `checkpoint.py` — so they are handed back to the deserialiser here.
        bind_live_deps(state["deps"])
        config = {"configurable": {"thread_id": thread_id}}
        pending = tuple(self.app.get_state(config).next)
        if pending:
            # Resumed: the clock restarts, because `started_at` was taken from a
            # `time.monotonic()` in a process that is now gone and monotonic
            # clocks are not comparable across processes. The recorded latency
            # is the resumed portion, which is the only interval this process
            # actually measured.
            final = dict(self.app.invoke(None, config=config))
            final["latency_ms"] = int((time.monotonic() - started) * 1000)
            return final  # type: ignore[return-value]
        final = dict(self.app.invoke(state, config=config))
        final["latency_ms"] = int((time.monotonic() - state["started_at"]) * 1000)
        return final  # type: ignore[return-value]


def to_langgraph(*, checkpointer: Any | None = None) -> Any:
    """Compile `NODES`/`EDGES` into a LangGraph `StateGraph`.

    Raises `ImportError` when `langgraph` is not installed. It is the
    `orchestration` extra rather than a core dependency, so that a machine
    without it still runs the system on `_Executor`.

    `checkpointer` is a LangGraph saver — `carelite.generate.checkpoint`
    builds the Postgres one. Passing it makes an interrupted turn resumable at
    its last completed node; the returned object then requires a `thread_id` on
    `invoke` to know which turn it is resuming.
    """
    from langgraph.graph import END as LG_END
    from langgraph.graph import START as LG_START
    from langgraph.graph import StateGraph

    builder: Any = StateGraph(TurnState)
    for name, fn in NODES.items():
        builder.add_node(name, fn)
    builder.add_edge(LG_START, START)
    for name, edge in EDGES.items():
        if callable(edge):
            mapping = {t: (LG_END if t == END else t) for t in CONDITIONAL_TARGETS[name]}
            builder.add_conditional_edges(name, edge, mapping)
        elif edge == END:
            builder.add_edge(name, LG_END)
        else:
            builder.add_edge(name, edge)
    app = builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()
    return _CompiledLangGraph(app=app, checkpointed=checkpointer is not None)


def build_graph(*, prefer_langgraph: bool = True, checkpointer: Any | None = None) -> Any:
    """The compiled turn graph. LangGraph when available, the executor otherwise."""
    if prefer_langgraph:
        try:
            return to_langgraph(checkpointer=checkpointer)
        except ImportError:
            pass
    return _Executor()


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


def to_guidance_response(state: TurnState) -> GuidanceResponse:
    """Project a finished state onto the frozen `GuidanceResponse` contract."""
    return GuidanceResponse(
        text=state.get("text", ""),
        condition=state["request"].condition,
        trace=state.get("trace"),
        input_safety=state.get("input_safety"),
        output_safety=state.get("output_safety"),
        self_check_passed=bool(state.get("self_check_passed", True)),
        model=state.get("model"),
        model_digest=state.get("model_digest"),
        prompt_version=state.get("prompt_id"),
        latency_ms=state.get("latency_ms"),
    )
