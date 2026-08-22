"""Contract tests. These pin the frozen interface every lane builds against;
a lane breaking one of these has violated the ownership rule."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from carelite.config import get_settings, seed_for
from carelite.types import RUBRIC_DIMENSIONS, Condition, KBEntry, RubricScore, Theme


def test_all_conditions_present():
    assert {c.value for c in Condition} == {"A", "A2", "B", "C", "LC", "D"}


def test_rubric_dimensions_match_score_model():
    fields = set(RubricScore.model_fields)
    assert set(RUBRIC_DIMENSIONS) <= fields
    assert len(RUBRIC_DIMENSIONS) == 11


def test_kb_entry_requires_a_source(kb_entry: KBEntry):
    with pytest.raises(ValidationError):
        kb_entry.model_copy(update={"source_paper_ids": []}).model_validate(
            kb_entry.model_dump() | {"source_paper_ids": []}
        )


def test_kb_entry_requires_substantive_span(kb_entry: KBEntry):
    with pytest.raises(ValidationError):
        KBEntry(**(kb_entry.model_dump() | {"verbatim_span": "short"}))


def test_seed_is_stable_across_processes():
    """Regression guard: `hash()` is per-process randomised, blake2b is not."""
    assert seed_for("sc-0001", "C", 0) == seed_for("sc-0001", "C", 0)
    assert seed_for("sc-0001", "C", 0) != seed_for("sc-0001", "C", 1)
    # Pinned literal — if this changes, every cached generation key changes.
    assert seed_for("sc-0001", "C", 0) == 666998405


def test_embedding_dim_within_pgvector_index_ceiling():
    assert get_settings().retrieval.embedding_dim <= 2000


def test_themes_are_the_seven_readme_categories():
    assert len(Theme) == 7
