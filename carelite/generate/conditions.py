"""The six experimental conditions, as configuration over one code path.

    from carelite.generate.conditions import SPEC
    SPEC[Condition.C].use_retrieval   # True
    SPEC[Condition.B].use_retrieval   # False

There is one graph, one generator call, one self-check and one output gate.
A condition is a `ConditionSpec` handed to that machinery, and nothing in
`graph.py` branches on a `Condition` value. That is the property the whole
comparison rests on: if condition C ran through code that condition B did not,
a difference between them would be a difference between two programs rather
than between two configurations of one.

**What each condition varies, and what is held constant.**

| | prompt | model | retrieval | corpus stuffed | self-check |
|---|---|---|---|---|---|
| A | `condition_a.v1` | generator | no | no | no |
| A2 | `condition_a.v1` | generator_alt | no | no | no |
| B | `condition_b.v1` | generator | no | no | yes |
| C | `condition_c.v1` | generator | yes | no | yes |
| LC | `condition_lc.v1` | long_context | no | yes | yes |
| D | `condition_d.v1` | generator | no | no | no |

Held constant across all six: the shared `constraints.v1` block, the fencing
data notice, the generation temperature, the seed derivation, the task line's
position after the data blocks, and the output gate.

**The self-check is on for B, C and LC and off for A, A2 and D, and that is a
choice with a cost worth stating.** It means A vs B does not isolate the prompt
text — it compares the bare model against the framework *pipeline*, of which the
verification pass is a part. The alternative, giving the bare baseline a
verification pass, would make "bare" untrue in a more damaging way, since a
model that checks and repairs its own draft is not a bare model. D omits it
because a negative control that gets a repair pass is a partly-repaired control.
`ConditionSpec.with_(self_check=...)` exists so the sensitivity analysis that
this choice obliges can actually be run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from carelite.config import ModelSpec, get_settings
from carelite.types import Condition

__all__ = ["SPEC", "ConditionSpec", "spec_for"]


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    """One experimental condition, fully described by configuration.

    Frozen: a run must not be able to mutate the configuration it is being
    measured under. Derive with `.with_(...)` for a sensitivity analysis.
    """

    condition: Condition
    prompt_id: str
    model_role: str
    """Attribute name on `config.Models`. The tag itself is never written here:
    the frozen contract is the single source of truth for the roster, and a tag
    copied into this module would be a second one."""

    use_retrieval: bool = False
    use_long_context: bool = False
    self_check: bool = True
    retrieval_preset: str | None = None
    """Named `carelite.retrieval.flags` preset, or `None` for the production
    default (`RetrievalFlags()`), which is the full stack. Only condition C
    reads this; it is here so an ablation run can drive condition C through a
    named row of the R0-R9 ladder without a second code path."""

    note: str = ""

    def model_spec(self) -> ModelSpec:
        return getattr(get_settings().models, self.model_role)

    @property
    def model_tag(self) -> str:
        return self.model_spec().tag

    @property
    def context_window(self) -> int:
        return self.model_spec().context_window

    def with_(self, **changes: object) -> ConditionSpec:
        """A derived spec. For sensitivity analyses, never for a production run."""
        return replace(self, **changes)  # type: ignore[arg-type]


_SPECS: tuple[ConditionSpec, ...] = (
    ConditionSpec(
        condition=Condition.A,
        prompt_id="condition_a.v1",
        model_role="generator",
        self_check=False,
        note="bare model: the task and the shared constraints, no framework, no retrieval",
    ),
    ConditionSpec(
        condition=Condition.A2,
        prompt_id="condition_a.v1",
        model_role="generator_alt",
        self_check=False,
        note="condition A on the second model family; the prompt row is shared with A",
    ),
    ConditionSpec(
        condition=Condition.B,
        prompt_id="condition_b.v1",
        model_role="generator",
        note="framework-prompted, no retrieval",
    ),
    ConditionSpec(
        condition=Condition.C,
        prompt_id="condition_c.v1",
        model_role="generator",
        use_retrieval=True,
        note="framework + hybrid retrieval: the full pipeline",
    ),
    ConditionSpec(
        condition=Condition.LC,
        prompt_id="condition_lc.v1",
        model_role="long_context",
        use_long_context=True,
        note="long-context baseline: corpus stuffed into the window, no retrieval",
    ),
    ConditionSpec(
        condition=Condition.D,
        prompt_id="condition_d.v1",
        model_role="generator",
        self_check=False,
        note="deliberately degraded prompt: the negative control from build plan v3 section 14",
    ),
)

#: Every condition, keyed by `Condition`. Read-only.
SPEC: Mapping[Condition, ConditionSpec] = MappingProxyType({s.condition: s for s in _SPECS})


def spec_for(condition: Condition) -> ConditionSpec:
    try:
        return SPEC[condition]
    except KeyError:  # pragma: no cover - Condition is a closed enum
        raise ValueError(f"no specification for condition {condition!r}") from None
