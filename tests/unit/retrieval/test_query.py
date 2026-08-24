"""Query construction translates patient language into the corpus's register."""

from __future__ import annotations

import pytest

from carelite.retrieval.query import (
    MAX_LEXICAL_TERMS,
    THEME_CUES,
    THEME_LEXICAL,
    THEME_QUERIES,
    MetadataFilter,
    build_queries,
    detect_themes,
)
from carelite.types import EncounterPhase, Theme


def test_every_theme_has_cues_queries_and_lexical_terms() -> None:
    """A theme missing from any table would silently never be retrievable."""
    for theme in Theme:
        assert THEME_CUES[theme], theme
        assert THEME_QUERIES[theme], theme
        assert THEME_LEXICAL[theme], theme


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("I'm scared this is cancer.", Theme.EMPATHY),
        ("Do I have to choose between surgery and radiation?", Theme.ACTIVATION_SDM),
        ("I need an interpreter, my English is not good enough.", Theme.EQUITY),
        ("They keep sending a different doctor.", Theme.TRUST_CONTINUITY),
        ("You said all that so fast, I didn't follow any of it.", Theme.TEACH_BACK),
        ("It's all jargon and big words to me.", Theme.PLAIN_LANGUAGE),
        ("I am furious about how this was handled.", Theme.EMOTION_RESPONSE),
    ],
)
def test_theme_detection(utterance: str, expected: Theme) -> None:
    assert expected in detect_themes(utterance)


def test_theme_detection_is_deterministic() -> None:
    """Ties break in `Theme` declaration order, so a scenario's queries are a
    fixed property of its text rather than something that can drift."""
    utterance = "I'm scared and confused about my options."
    assert detect_themes(utterance) == detect_themes(utterance)


def test_lexical_queries_respect_the_and_of_terms_ceiling() -> None:
    """`websearch_to_tsquery` ANDs every content word — "teach-back method for
    confirming patient comprehension" returns zero rows against the live
    corpus while "teach-back" alone returns 15. Long lexical queries do not
    degrade gracefully, they collapse to nothing."""
    qs = build_queries(
        "I really do not understand what the doctor explained about my treatment options today."
    )
    for lexical in qs.lexical_queries:
        assert len(lexical.split()) <= MAX_LEXICAL_TERMS, lexical


def test_dense_queries_are_framework_language_not_the_utterance() -> None:
    utterance = "I'm scared this is cancer and nobody explains anything to me."
    qs = build_queries(utterance)
    assert utterance not in qs.dense_queries
    assert any("empath" in q for q in qs.dense_queries)


def test_query_count_honours_the_frozen_setting() -> None:
    from carelite.config import get_settings

    n = get_settings().retrieval.n_framework_queries
    qs = build_queries("I'm scared this is cancer.")
    assert len(qs.dense_queries) == n


def test_unexpanded_mode_is_the_naive_baseline() -> None:
    """R0 embeds the raw utterance and nothing else."""
    utterance = "I'm scared this is cancer."
    qs = build_queries(utterance, expand=False)
    assert qs.dense_queries == (utterance,)
    assert qs.expanded is False
    assert qs.metadata.is_empty


def test_equity_relevance_is_inferred_from_the_theme() -> None:
    qs = build_queries("I need an interpreter, my English is not good enough.")
    assert qs.metadata.equity_relevant is True


def test_non_equity_turn_does_not_set_the_equity_filter() -> None:
    qs = build_queries("I'm scared this is cancer.")
    assert qs.metadata.equity_relevant is None


def test_encounter_phase_lands_on_the_filter() -> None:
    qs = build_queries("What happens next?", encounter_phase=EncounterPhase.PLANNING)
    assert qs.metadata.encounter_phase is EncounterPhase.PLANNING


def test_all_queries_deduplicates_for_the_evidence_panel() -> None:
    qs = build_queries("I'm scared this is cancer.")
    assert len(set(qs.all_queries)) == len(qs.all_queries)


def test_metadata_filter_describe_is_readable() -> None:
    mf = MetadataFilter(themes=(Theme.EMPATHY,), encounter_phase=EncounterPhase.OPENING)
    described = mf.describe()
    assert "empathy" in described and "opening" in described
    assert MetadataFilter().describe() == "(none)"


def test_fallback_queries_when_no_theme_is_detected() -> None:
    """An unrecognised turn still issues well-formed queries rather than an
    empty one — but note this is exactly the path that makes an off-domain
    turn *look* on-topic downstream, which is why `crag.py` grades against
    the utterance and not against these."""
    qs = build_queries("The parking garage was closed this morning.")
    assert qs.dense_queries
    assert qs.themes == ()
