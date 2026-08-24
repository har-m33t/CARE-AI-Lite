"""Ablation rows are configuration, not code edits."""

from __future__ import annotations

import dataclasses

import pytest

from carelite.config import get_settings
from carelite.retrieval.flags import ENV_PREFIX, PRESETS, RetrievalFlags, preset


def test_defaults_come_from_the_frozen_contract() -> None:
    settings = get_settings().retrieval
    flags = RetrievalFlags()
    assert flags.rrf_k == settings.rrf_k
    assert flags.rerank_top_n == settings.rerank_top_n
    assert flags.dense_top_k == settings.dense_top_k
    assert flags.crag_relevance_threshold == settings.crag_relevance_threshold
    assert flags.n_framework_queries == settings.n_framework_queries


def test_hyde_default_follows_the_frozen_setting() -> None:
    """`hyde_enabled` is the one retrieval boolean the frozen config already
    owns; this module must never contradict it."""
    assert RetrievalFlags().hyde is get_settings().retrieval.hyde_enabled


def test_every_ablation_row_is_present() -> None:
    for name in ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "LC"):
        assert name in PRESETS


def test_the_ladder_adds_one_component_at_a_time() -> None:
    """A difference between adjacent rows must be attributable to a single
    component. R8 is the deliberate exception (R9 minus CRAG)."""
    stages = ("query_expansion", "lexical", "graph", "hyde", "rerank", "tier_weighting", "crag")
    for lower, upper in zip(
        ["R0", "R1", "R2", "R3", "R4", "R5", "R6"],
        ["R1", "R2", "R3", "R4", "R5", "R6", "R7"],
        strict=True,
    ):
        changed = [s for s in stages if getattr(PRESETS[lower], s) != getattr(PRESETS[upper], s)]
        assert len(changed) == 1, f"{lower}->{upper} changed {changed}"


def test_r0_is_the_dense_only_baseline() -> None:
    r0 = preset("R0")
    assert r0.dense and not r0.lexical and not r0.graph
    assert not r0.query_expansion and not r0.hyde and not r0.rerank and not r0.crag


def test_r9_is_the_full_stack() -> None:
    r9 = preset("R9")
    for name in (
        "router",
        "query_expansion",
        "hyde",
        "dense",
        "lexical",
        "graph",
        "rerank",
        "tier_weighting",
        "crag",
    ):
        assert getattr(r9, name) is True, name


def test_r8_is_r9_without_crag() -> None:
    """The pair that isolates the gate's contribution."""
    r8, r9 = preset("R8"), preset("R9")
    assert r8.crag is False and r9.crag is True
    for name in (
        "router",
        "query_expansion",
        "hyde",
        "dense",
        "lexical",
        "graph",
        "rerank",
        "tier_weighting",
    ):
        assert getattr(r8, name) == getattr(r9, name), name


def test_lc_does_not_retrieve() -> None:
    lc = preset("LC")
    assert lc.long_context is True
    assert not (lc.dense or lc.lexical or lc.graph)


def test_all_legs_off_is_rejected_outside_long_context() -> None:
    with pytest.raises(ValueError, match="at least one retrieval leg"):
        RetrievalFlags(
            _explicit=frozenset({"dense", "lexical", "graph"}),
            dense=False,
            lexical=False,
            graph=False,
        )


def test_flags_are_immutable() -> None:
    """A pipeline run must not be able to mutate the configuration it is being
    measured under."""
    flags = RetrievalFlags()
    with pytest.raises(dataclasses.FrozenInstanceError):
        flags.rerank = False  # type: ignore[misc]


def test_with_marks_changes_explicit_so_config_cannot_overwrite_them() -> None:
    derived = RetrievalFlags().with_(hyde=False)
    assert derived.hyde is False


def test_environment_override(monkeypatch) -> None:
    """An ablation is still configuration even from a shell."""
    monkeypatch.setenv(f"{ENV_PREFIX}RERANK", "false")
    assert RetrievalFlags().rerank is False


def test_explicit_kwarg_beats_the_environment(monkeypatch) -> None:
    monkeypatch.setenv(f"{ENV_PREFIX}RERANK", "false")
    assert RetrievalFlags().with_(rerank=True).rerank is True


def test_bad_environment_value_fails_loudly(monkeypatch) -> None:
    monkeypatch.setenv(f"{ENV_PREFIX}RERANK", "maybe")
    with pytest.raises(ValueError, match="not a boolean"):
        RetrievalFlags()


def test_unknown_preset_raises() -> None:
    with pytest.raises(KeyError, match="unknown ablation preset"):
        preset("R99")


def test_preset_lookup_is_case_insensitive() -> None:
    assert preset("r9").label == "R9"
