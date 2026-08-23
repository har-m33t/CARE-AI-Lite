"""The rubric definition, and its agreement with the document raters read.

`docs/rubric.md` and `carelite/eval/rubric/dimensions.py` are the same rubric
in two forms. Humans score against the document; the LLM judge is prompted
against the module. If they drift apart, human and judge scores stop being
comparable — which is the entire point of the v3 §13 validation study.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from carelite.eval.rubric.dimensions import (
    DIMENSIONS,
    RUBRIC_VERSION,
    SCALE_MAX,
    SCALE_MIN,
    Framework,
    dimension,
    to_quality_scores,
)
from carelite.types import RUBRIC_DIMENSIONS

REPO_ROOT = Path(__file__).resolve().parents[3]
RUBRIC_DOC = REPO_ROOT / "docs" / "rubric.md"

ALL_DIMS = pytest.mark.parametrize("key", RUBRIC_DIMENSIONS)


# ------------------------------------------------------- the frozen contract ---


def test_dimensions_match_the_frozen_contract_exactly_and_in_order() -> None:
    assert tuple(DIMENSIONS) == RUBRIC_DIMENSIONS


def test_the_frameworks_partition_the_dimensions_as_five_four_two() -> None:
    counts = {f: sum(1 for d in DIMENSIONS.values() if d.framework is f) for f in Framework}
    assert counts == {Framework.NURSE: 5, Framework.FOUR_HABITS: 4, Framework.SECONDARY: 2}


def test_unknown_dimension_raises_a_useful_error() -> None:
    with pytest.raises(KeyError, match="nurse"):
        dimension("nurse")


# ----------------------------------------------------------------- anchors ---


@ALL_DIMS
def test_every_dimension_has_a_definition_a_source_and_three_anchors(key: str) -> None:
    d = dimension(key)
    assert len(d.definition) > 120, f"{key}: definition is too thin to rate against"
    assert len(d.source) > 40, f"{key}: no named source"
    assert d.question.endswith(("?", "?)")), f"{key}: the rating prompt must be a question"
    for level, anchor in ((1, d.anchor_1), (3, d.anchor_3), (5, d.anchor_5)):
        assert len(anchor) > 30, f"{key}: anchor {level} carries no example text"
    assert d.anchor_1 != d.anchor_3 != d.anchor_5


@ALL_DIMS
def test_every_source_names_a_real_citation(key: str) -> None:
    """Anchors are only reproducible if a rater can go and read the original."""
    source = dimension(key).source
    assert re.search(r"\b(19|20)\d{2}\b", source), f"{key}: source has no year"


def test_to_quality_scores_preserves_unscored_dimensions() -> None:
    raw = {key: None for key in RUBRIC_DIMENSIONS}
    raw["ritualistic"] = 5
    raw["de"] = 4
    out = to_quality_scores(raw)
    assert out["ritualistic"] == 1  # reverse-coded
    assert out["de"] == 4
    assert out["name"] is None


def test_to_quality_scores_ignores_non_rubric_keys() -> None:
    out = to_quality_scores({"ritualistic": 5, "generation_id": None, "rater_id": None})
    assert out == {"ritualistic": 1}


# ------------------------------------------------- the document raters read ---


def test_the_rubric_document_exists() -> None:
    assert RUBRIC_DOC.is_file(), f"{RUBRIC_DOC} is the artifact human raters score against"


def test_the_document_carries_the_reverse_coding_warning() -> None:
    """A rater who misses this inverts the study's headline secondary finding."""
    text = RUBRIC_DOC.read_text(encoding="utf-8")
    assert "REVERSE CODING" in text
    assert "5 IS THE WORST SCORE" in text
    assert "to_quality" in text


def test_the_document_documents_every_dimension_with_anchors() -> None:
    """Each dimension gets its own section carrying a 1, a 3 and a 5 anchor."""
    text = RUBRIC_DOC.read_text(encoding="utf-8")
    headings = re.findall(r"^## \d+\. `(\w+)` ", text, flags=re.MULTILINE)
    assert tuple(headings) == RUBRIC_DIMENSIONS, headings

    sections = re.split(r"^## \d+\. `\w+` ", text, flags=re.MULTILINE)[1:]
    for key, section in zip(RUBRIC_DIMENSIONS, sections, strict=True):
        assert "**Definition.**" in section, f"{key}: no definition in the document"
        assert "**Source.**" in section, f"{key}: no named source in the document"
        for level in (SCALE_MIN, 3, SCALE_MAX):
            assert f"| **{level}**" in section, f"{key}: no anchor row for level {level}"


def test_the_document_and_the_module_agree_on_the_version() -> None:
    text = RUBRIC_DOC.read_text(encoding="utf-8")
    assert f"Rubric version {RUBRIC_VERSION}" in text


def test_the_document_points_at_the_calibration_set() -> None:
    text = RUBRIC_DOC.read_text(encoding="utf-8")
    for item_id in ("CAL-01", "CAL-02", "CAL-03", "CAL-04", "CAL-05"):
        assert item_id in text
