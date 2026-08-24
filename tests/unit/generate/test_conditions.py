"""The six conditions differ by configuration and by nothing else."""

from __future__ import annotations

import pytest

from carelite.generate.conditions import SPEC, spec_for
from carelite.types import Condition


def test_all_six_conditions_are_specified() -> None:
    assert set(SPEC) == set(Condition)
    assert len(SPEC) == 6


def test_only_c_retrieves() -> None:
    retrieving = {c for c, s in SPEC.items() if s.use_retrieval}
    assert retrieving == {Condition.C}


def test_only_lc_stuffs_the_corpus() -> None:
    stuffing = {c for c, s in SPEC.items() if s.use_long_context}
    assert stuffing == {Condition.LC}
    # A long-context condition that also retrieved would not be a baseline for
    # retrieval; it would be the same condition twice.
    assert not SPEC[Condition.LC].use_retrieval


def test_the_cross_model_baseline_changes_the_model_and_nothing_else() -> None:
    a, a2 = SPEC[Condition.A], SPEC[Condition.A2]
    differing = {
        field
        for field in ("prompt_id", "model_role", "use_retrieval", "use_long_context", "self_check")
        if getattr(a, field) != getattr(a2, field)
    }
    assert differing == {"model_role"}


def test_b_and_c_differ_only_in_prompt_and_retrieval() -> None:
    b, c = SPEC[Condition.B], SPEC[Condition.C]
    differing = {
        field
        for field in ("prompt_id", "model_role", "use_retrieval", "use_long_context", "self_check")
        if getattr(b, field) != getattr(c, field)
    }
    assert differing == {"prompt_id", "use_retrieval"}


def test_self_check_is_on_for_the_framework_conditions_only() -> None:
    """A and A2 are bare; D is the negative control. Both would stop being what
    they are if a verification pass repaired their drafts."""
    assert {c for c, s in SPEC.items() if s.self_check} == {
        Condition.B,
        Condition.C,
        Condition.LC,
    }


def test_model_roles_resolve_against_the_frozen_roster() -> None:
    assert SPEC[Condition.A].model_tag == SPEC[Condition.B].model_tag
    assert SPEC[Condition.A2].model_tag != SPEC[Condition.A].model_tag
    for spec in SPEC.values():
        assert spec.model_tag
        assert spec.context_window > 0


def test_a_spec_cannot_be_mutated_in_place() -> None:
    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass raises FrozenInstanceError
        SPEC[Condition.C].use_retrieval = False  # type: ignore[misc]


def test_with_derives_a_variant_without_touching_the_registry() -> None:
    variant = spec_for(Condition.B).with_(self_check=False)
    assert variant.self_check is False
    assert SPEC[Condition.B].self_check is True
