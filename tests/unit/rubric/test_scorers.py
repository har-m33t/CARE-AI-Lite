"""Deterministic scorers, checked against hand-worked examples.

Every expected value below was computed by hand and the arithmetic is shown in
the test, so a failure says which of the two is wrong. Where a measure is a
heuristic (syllable counting, message segmentation) the tests pin the cases the
heuristic is meant to get right and document the cases it does not.
"""

from __future__ import annotations

import pytest

from carelite.eval.rubric.scorers import (
    SCORER_VERSION,
    flesch_kincaid_grade,
    hedge_density,
    jargon_density,
    jargon_terms_found,
    message_count,
    pseudo_teach_back_phrases,
    question_count,
    response_length,
    ritual_markers,
    ritualistic_proxy,
    score_text,
    sentence_count,
    sentences,
    syllable_count,
    teach_back_phrases,
    teach_back_present,
    words,
)

# --------------------------------------------------------- tokenisation ---


def test_words_ignores_numbers_and_punctuation() -> None:
    assert words("Take 2 pills, twice daily.") == ["Take", "pills", "twice", "daily"]


def test_sentences_splits_on_terminators() -> None:
    assert sentences("I'm sorry. That is hard! What now?") == [
        "I'm sorry.",
        "That is hard!",
        "What now?",
    ]


def test_abbreviations_do_not_end_a_sentence() -> None:
    assert sentence_count("Dr. Ruiz will call you. She is on call.") == 2


def test_each_list_line_is_its_own_unit() -> None:
    text = "Two things:\n- Take the pill each morning.\n- Call if you get a fever."
    assert sentences(text) == [
        "Two things:",
        "Take the pill each morning.",
        "Call if you get a fever.",
    ]


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("the", 1),  # silent-e correction must not take it to zero
        ("cat", 1),
        ("fast", 1),
        ("doctor", 2),  # o | o
        ("benign", 2),  # e | i
        ("hospitalization", 6),  # o i a i a io
        ("", 0),
    ],
)
def test_syllable_count_on_hand_checked_words(word: str, expected: int) -> None:
    assert syllable_count(word) == expected


def test_syllable_heuristic_undercounts_adjacent_vowel_syllables() -> None:
    """Documented drift: 'im-me-di-ate' is 4, the vowel-group heuristic says 3.

    Pinned deliberately. Flesch-Kincaid is used comparatively here, across
    conditions scored by the identical function, so a consistent bias is
    tolerable and an undocumented one is not.
    """
    assert syllable_count("immediate") == 3


# ----------------------------------------------------------- readability ---


def test_flesch_kincaid_matches_hand_arithmetic() -> None:
    """10 words, 2 sentences, 10 syllables.

    0.39*(10/2) + 11.8*(10/10) - 15.59 = 1.95 + 11.8 - 15.59 = -1.84
    """
    text = "The cat sat on the mat. The dog ran fast."
    assert response_length(text) == 10
    assert sentence_count(text) == 2
    assert flesch_kincaid_grade(text) == pytest.approx(-1.84, abs=1e-9)


def test_clinical_prose_reads_harder_than_plain_prose() -> None:
    plain = "It is small. The next step is one more picture, not treatment."
    clinical = (
        "The subcentimetre pulmonary nodule demonstrates smooth margins without spiculation, "
        "consistent with a benign etiology warranting radiographic surveillance."
    )
    assert flesch_kincaid_grade(clinical) > flesch_kincaid_grade(plain)
    assert flesch_kincaid_grade(plain) < 6.0  # the plain-language target


def test_empty_text_does_not_divide_by_zero() -> None:
    assert flesch_kincaid_grade("") == 0.0
    assert jargon_density("") == 0.0
    assert hedge_density("") == 0.0


# --------------------------------------------------------------- jargon ---


def test_jargon_density_is_hits_over_words() -> None:
    """4 words, 2 jargon hits -> 0.5."""
    text = "The nodule is benign."
    assert response_length(text) == 4
    assert jargon_terms_found(text) == ["nodule", "benign"]
    assert jargon_density(text) == pytest.approx(0.5)


def test_multiword_jargon_counts_as_one_hit() -> None:
    """4 words, 1 hit ('serial imaging') -> 0.25."""
    text = "We recommend serial imaging."
    assert jargon_terms_found(text) == ["serial imaging"]
    assert jargon_density(text) == pytest.approx(0.25)


def test_jargon_matching_is_case_insensitive_and_word_bounded() -> None:
    assert jargon_terms_found("Benign findings.") == ["benign"]
    # "prognosis" must not be found inside a longer token
    assert jargon_terms_found("prognostication") == []


def test_ambiguous_everyday_words_are_deliberately_not_jargon() -> None:
    """Conservative by design: a bag-of-words matcher cannot disambiguate these."""
    assert jargon_terms_found("The result was negative and the news is positive.") == []


# --------------------------------------------------------------- hedging ---


def test_hedge_density_is_hits_over_words() -> None:
    """5 words, 2 hedges ('might', 'somewhat') -> 0.4."""
    text = "This might be somewhat serious."
    assert response_length(text) == 5
    assert hedge_density(text) == pytest.approx(0.4)


def test_a_flatly_certain_sentence_has_zero_hedges() -> None:
    assert hedge_density("The scan is clear.") == 0.0


# ------------------------------------------------------------- questions ---


def test_question_count_counts_interrogative_sentences() -> None:
    assert question_count("How are you? I'm here. What happens next?") == 2


def test_question_count_is_zero_for_pure_information() -> None:
    assert question_count("The next step is a CT scan in three months.") == 0


# ------------------------------------------------------------ teach-back ---


@pytest.mark.parametrize(
    "text",
    [
        "Can you tell me in your own words what you'll do?",
        "What's your understanding of the plan?",
        "When you tell your husband tonight, what are you going to say?",
        "How would you explain this to your daughter?",
        "Can you repeat that back to me?",
        "I want to make sure I've explained this well.",
    ],
)
def test_genuine_teach_back_is_detected(text: str) -> None:
    assert teach_back_present(text), text
    assert teach_back_phrases(text)  # and the span is available for grounding


@pytest.mark.parametrize(
    "text",
    [
        "Does that make sense?",
        "Do you understand?",
        "Do you have any questions?",
        "Is that clear?",
    ],
)
def test_closed_comprehension_checks_are_not_teach_back(text: str) -> None:
    """The literature contrasts these with teach-back directly: a patient who
    did not understand will usually still answer yes."""
    assert not teach_back_present(text), text
    assert pseudo_teach_back_phrases(text), text


def test_teach_back_spans_are_verbatim_in_the_response() -> None:
    text = "I've given you a lot. When you tell your husband tonight, what will you say?"
    for span in teach_back_phrases(text):
        assert span in text


# --------------------------------------------------------- message count ---


def test_message_count_counts_declaratives_and_list_items() -> None:
    """2 declaratives + 2 bullets + 0 (the question is excluded) = 4."""
    text = (
        "I'm sorry. This is hard.\n"
        "- Take the pill each morning.\n"
        "- Call if you get a fever.\n"
        "Any questions?"
    )
    assert message_count(text) == 4
    assert question_count(text) == 1


def test_a_multi_sentence_list_item_is_still_one_message() -> None:
    text = "1. Take the pill each morning. It works best with food."
    assert message_count(text) == 1


def test_message_count_flags_the_three_key_message_ceiling() -> None:
    over = (
        "It is a nodule. It is small. We will scan again in three months. "
        "If it grows we biopsy. If it is stable we stop."
    )
    assert message_count(over) > 3


# ------------------------------------------------------------- length ---


def test_response_length_is_a_word_count() -> None:
    assert response_length("One two three.") == 3


# ------------------------------------------------------ ritual markers ---


def test_framework_labels_are_the_strongest_ritual_signal() -> None:
    text = "**Naming:** It sounds like you are frightened."
    m = ritual_markers(text)
    assert m.framework_labels
    assert all(span in text for span in m.framework_labels)
    assert ritualistic_proxy(text) >= 3


def test_stock_stems_alone_raise_the_proxy_without_any_template() -> None:
    text = (
        "I hear you. That must be really difficult. I want you to know we are here for you "
        "every step of the way."
    )
    m = ritual_markers(text)
    assert not m.framework_labels
    assert not m.scaffold_lines
    assert len(m.stock_phrases) >= 4
    assert ritualistic_proxy(text) == 3


def test_a_specific_unformulaic_response_scores_the_best_possible_ritual() -> None:
    text = (
        "Up all night. And your mother's whole story sitting right on top of this one. "
        "Tell me which part is loudest right now."
    )
    assert ritual_markers(text).total == 0
    assert ritualistic_proxy(text) == 1


def test_the_word_nurse_meaning_the_clinician_is_not_a_framework_label() -> None:
    """Case-sensitive on purpose: the mnemonic, not the person."""
    text = "If you cannot sleep before then, my nurse can reach me."
    assert ritual_markers(text).framework_labels == ()
    assert ritualistic_proxy(text) == 1


def test_a_next_steps_heading_is_not_framework_leakage() -> None:
    """Ordinary clinical practice, deliberately excluded from the label patterns."""
    assert ritual_markers("Next steps: one more scan in three months.").framework_labels == ()


def test_ritual_markers_are_verbatim_spans() -> None:
    text = (
        "**Naming:** It sounds like you are anxious.\n"
        "**Supporting:** Please know that we are here for you."
    )
    m = ritual_markers(text)
    for span in (*m.framework_labels, *m.stock_phrases, *m.scaffold_lines):
        assert span in text
    assert m.longest_span() in text


# ----------------------------------------------------------- aggregate ---


def test_score_text_reports_every_measure_and_pins_its_version() -> None:
    text = (
        "That is a rough night, I am sorry. The nodule is small. "
        "What is your understanding of where that leaves you?"
    )
    s = score_text(text)
    assert s.scorer_version == SCORER_VERSION
    assert s.word_count == response_length(text)
    assert s.question_count == 1
    assert s.teach_back_present is True
    assert s.jargon_terms == ("nodule",)
    assert s.ritualistic_proxy_score == 1


def test_score_text_is_a_pure_function_of_the_text() -> None:
    text = "It is small, and the next thing is one more picture."
    assert score_text(text) == score_text(text)
