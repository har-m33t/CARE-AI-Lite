"""The LangGraph Postgres checkpointer, and the one thing it must never store.

    from carelite.generate.checkpoint import graph_checkpointer

    with graph_checkpointer() as saver:
        graph = build_graph(checkpointer=saver)

**What this buys, stated narrowly.** The runner is already resumable *between*
cells: every generation commits on its own and `completed_keys()` asks the store
what is durably there, so a `kill -9` costs the cell in flight and nothing else
(`store.py`). A checkpointer makes that cell resumable too — the graph continues
from its last completed node instead of restarting at the safety screen. For the
A/A2/D group that saves about six seconds and is not worth having. For the
long-context condition, where a turn that dies in `self_check` would otherwise
re-prefill roughly 119,500 tokens, it is the difference D11 measured in minutes.

**The live collaborators are deliberately not in the checkpoint.** `TurnState`
carries a `GraphDeps`, and a `GraphDeps` holds an open model client — including,
under `CARELITE_BACKEND=vllm`, one carrying a bearer token. LangGraph's default
serialiser can be told to fall back to `pickle`, which would write that token
into a database table as a side effect of a resumability feature. It is not
enabled here and must not be. `LiveDepsSerde` substitutes the deps out of the
object graph before encoding, leaving a marker string in their place, and hands
back the collaborators the calling process is holding when the checkpoint is
read. A resumed run therefore reuses one live client rather than reconstituting
a dead one, which is also the only correct behaviour: a socket does not survive
a `SIGKILL` no matter how it was encoded.

**Everything else in the state does round-trip.** The pydantic models
(`GuidanceRequest`, `SafetyVerdict`, `RetrievalTrace`, `RetrievedItem`) and the
dataclasses (`ConditionSpec`, `SelfCheckResult`) are handled by LangGraph's
msgpack serialiser, so a resumed turn generates against the same request and the
same retrieved passages the interrupted one did.

**A machine without a database still runs.** `graph_checkpointer()` yields
`None` when it is switched off, when `langgraph-checkpoint-postgres` is absent,
or when the connection fails; `build_graph(checkpointer=None)` is exactly the
path that produced every generation in the database today. Checkpointing is an
optimisation on a resume that already worked, so failing to get one is a
degradation and not an error.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

__all__ = [
    "CHECKPOINT_ENV",
    "LIVE_DEPS_MARKER",
    "LiveDepsSerde",
    "UnboundLiveDeps",
    "bind_live_deps",
    "checkpointing_requested",
    "clear_live_deps",
    "graph_checkpointer",
]

#: Set to `1` to checkpoint the graph to Postgres. Off by default: the study's
#: existing rows were produced without it and a run must not change shape on the
#: strength of an environment variable nobody set on purpose.
CHECKPOINT_ENV = "CARELITE_GRAPH_CHECKPOINT"

#: Written in place of the live collaborators. A string rather than a type tag,
#: because LangGraph checkpoints the whole input mapping as one value on its
#: `__start__` channel: the `GraphDeps` is reached nested inside a dict, where a
#: serialiser only gets to choose how the dict as a whole is encoded. So the
#: substitution happens on the object graph before it is handed to the encoder.
LIVE_DEPS_MARKER = "__carelite_live_deps_not_checkpointed__"

_TRUE = {"1", "true", "yes", "on"}

#: The collaborators the current process is holding, handed back when a
#: checkpoint written by an earlier process is read. One slot, because one run
#: has one `GraphDeps`; a registry keyed by anything would only invite the
#: question of what to do when the key is stale.
_LIVE: dict[str, Any] = {}


def checkpointing_requested() -> bool:
    return os.environ.get(CHECKPOINT_ENV, "").strip().lower() in _TRUE


def bind_live_deps(deps: Any) -> None:
    """Register the collaborators to restore when a checkpoint is deserialised."""
    _LIVE["deps"] = deps


def clear_live_deps() -> None:
    _LIVE.pop("deps", None)


class UnboundLiveDeps(RuntimeError):
    """A checkpoint was read before its process registered live collaborators.

    Raised rather than papered over with a default `GraphDeps`, which would
    silently swap a fake client for a real one in a test, or a configured
    long-context pack for a freshly built one in a run.
    """


def _strip_live(obj: Any) -> Any:
    """Replace every `GraphDeps` in the object graph with `LIVE_DEPS_MARKER`.

    Shallow by construction: LangGraph hands over channel values, which are the
    turn state's fields and — on the `__start__` channel — the input mapping.
    Dicts, lists and tuples are the only containers that appear there.
    """
    from carelite.generate.graph import GraphDeps

    if isinstance(obj, GraphDeps):
        bind_live_deps(obj)
        return LIVE_DEPS_MARKER
    if isinstance(obj, dict):
        return {k: _strip_live(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_live(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_strip_live(v) for v in obj)
    return obj


def _restore_live(obj: Any) -> Any:
    """The inverse: hand back the collaborators this process is holding."""
    if isinstance(obj, str):
        if obj != LIVE_DEPS_MARKER:
            return obj
        deps = _LIVE.get("deps")
        if deps is None:
            raise UnboundLiveDeps(
                "a checkpointed turn was read before this process registered its "
                "collaborators. Call carelite.generate.checkpoint.bind_live_deps(deps) "
                "with the GraphDeps the run is using before resuming a turn."
            )
        return deps
    if isinstance(obj, dict):
        return {k: _restore_live(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_live(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_restore_live(v) for v in obj)
    return obj


class LiveDepsSerde:
    """LangGraph's serialiser, with the live collaborators taken out of it.

    Delegates the encoding to `JsonPlusSerializer`. Two departures, both
    deliberate:

    * a `GraphDeps` is substituted out of the object graph before encoding and
      read back as whatever `bind_live_deps` last registered in this process;
    * `pickle_fallback` is never enabled, so an object the msgpack encoder does
      not understand raises here instead of being pickled into a database row.
      That is the guard on the credential: there is no code path that turns an
      unrecognised object into bytes on disk.
    """

    def __init__(self) -> None:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        self.inner = JsonPlusSerializer()

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return self.inner.dumps_typed(_strip_live(obj))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return _restore_live(self.inner.loads_typed(data))


@contextmanager
def graph_checkpointer(
    *, enabled: bool | None = None, database_url: str | None = None
) -> Iterator[Any | None]:
    """A Postgres checkpointer, or `None` and a line on stderr saying why.

    Args:
        enabled: override the `CARELITE_GRAPH_CHECKPOINT` environment switch.
        database_url: override `settings.database_url`.

    Yields `None` — never raises — when checkpointing is off, when
    `langgraph-checkpoint-postgres` is not installed, or when the database
    cannot be reached. Every one of those leaves the runner on the resume path
    that produced all 939 existing rows.
    """
    if enabled is None:
        enabled = checkpointing_requested()
    if not enabled:
        yield None
        return

    from carelite.config import get_settings

    url = database_url or get_settings().database_url
    stack = ExitStack()
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        saver = stack.enter_context(PostgresSaver.from_conn_string(url))
        saver.serde = LiveDepsSerde()
        saver.setup()
    except Exception as exc:
        # Opening the checkpointer is separated from using it so that a failure
        # in the caller's body is never mistaken for a failure to connect, and
        # so this generator yields exactly once on every path.
        stack.close()
        print(
            f"carelite.generate.checkpoint: could not open the checkpointer "
            f"({type(exc).__name__}: {exc}); running without mid-turn checkpointing "
            "(a run is still resumable per cell).",
            file=sys.stderr,
        )
        yield None
        return

    with stack:
        yield saver
