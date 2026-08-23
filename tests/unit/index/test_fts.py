"""Unit tests for carelite.index.fts.

`to_tsquery_sql` mode validation is pure logic and part of `make check`. The
framework-term-survives-stemming claims in fts.py's docstring are checked
against the live corpus (`@pytest.mark.db`) — this is the specific regression
test the brief asks for: "NURSE", "teach-back", "Four Habits" must remain
exact-matchable and must not be stemmed into uselessness.
"""

from __future__ import annotations

import pytest

from carelite.index.fts import search_chunks, search_kb_entries, to_tsquery_sql


def test_to_tsquery_sql_known_modes():
    assert to_tsquery_sql("websearch") == "websearch_to_tsquery"
    assert to_tsquery_sql("plain") == "plainto_tsquery"
    assert to_tsquery_sql("phrase") == "phraseto_tsquery"
    assert to_tsquery_sql("raw") == "to_tsquery"


def test_to_tsquery_sql_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown tsquery mode"):
        to_tsquery_sql("drop table;--")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# @pytest.mark.db — the framework-term regression suite.
# ---------------------------------------------------------------------------


@pytest.mark.db
def test_hyphenated_teach_back_survives_stemming_and_is_exact_matchable():
    # Short and keyword-focused deliberately: websearch_to_tsquery ANDs every
    # content word, so a full natural sentence here would AND-explode to zero
    # hits even though "teach-back" itself is well represented (see fts.py's
    # module docstring for how this was caught).
    hits = search_chunks("teach-back method", top_k=10)
    assert hits, "teach-back must be retrievable; corpus is known to contain it"
    assert any("teach-back" in h.text.lower() for h in hits)


@pytest.mark.db
def test_multiword_four_habits_model_survives_as_a_phrase():
    hits = search_chunks("Four Habits Model", top_k=10, mode="phrase")
    assert hits
    assert any("four habits" in h.text.lower() for h in hits)


@pytest.mark.db
def test_nurse_acronym_is_not_dropped_by_english_stemming():
    """'NURSE' stems to the same lexeme as 'nurse'/'nursing' -- a precision
    cost documented in fts.py, not a recall failure. This test asserts the
    recall half: at least one hit for the literal word 'NURSE' exists, and
    at least one of the top hits is plausibly about the framework (co-occurs
    with empathy language) rather than nursing-the-profession."""
    hits = search_chunks("NURSE statements empathic response", top_k=10)
    assert hits
    assert any("nurse" in h.text.lower() for h in hits)
    assert any("nurse" in h.text.lower() and "empath" in h.text.lower() for h in hits), (
        "expected at least one hit where NURSE co-occurs with empathy language"
    )


@pytest.mark.db
def test_spikes_framework_term_is_exact_matchable():
    hits = search_chunks("SPIKES framework delivering bad news", top_k=10)
    assert hits
    assert any("spikes" in h.text.lower() for h in hits)


@pytest.mark.db
def test_search_chunks_ranks_by_cover_density_descending():
    hits = search_chunks("teach-back", top_k=10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.db
def test_search_kb_entries_returns_empty_list_gracefully_when_table_is_sparse():
    """kb_entry is empty until carelite-kb finishes; this must not raise."""
    hits = search_kb_entries("teach-back", top_k=10)
    assert isinstance(hits, list)  # [] today; still a list once populated


@pytest.mark.db
def test_search_chunks_returns_empty_list_for_a_query_matching_nothing():
    hits = search_chunks("zzz_definitely_not_in_the_corpus_zzz", top_k=5)
    assert hits == []
