"""Unit tests for carelite.kb.spans.

The normalisation these tests pin down is the load-bearing part of the KB's
provenance guarantee: too strict and real quotes from real PDFs get rejected,
too loose and the check stops meaning anything. Both failure directions are
tested here — the "should still match" cases and the "must not match" cases
carry equal weight.
"""

from __future__ import annotations

from carelite.kb.spans import locate_span, normalize, normalized_text, surrounding_context


class TestNormalize:
    def test_collapses_whitespace_runs(self) -> None:
        assert normalized_text("teach   back\n\n  works") == "teach back works"

    def test_folds_ligatures(self) -> None:
        # PDF extraction routinely emits the ﬁ/ﬂ ligature codepoints.
        assert normalized_text("conﬁrm the ﬂow") == "confirm the flow"

    def test_folds_curly_quotes_and_dashes(self) -> None:
        assert normalized_text("the patient’s view — clearly") == "the patient's view - clearly"  # noqa: RUF001 - the curly glyph is the input under test

    def test_joins_hyphenated_line_break(self) -> None:
        assert normalized_text("com-\nmunication skills") == "communication skills"

    def test_keeps_real_hyphens(self) -> None:
        assert normalized_text("teach-back is low-risk") == "teach-back is low-risk"

    def test_drops_zero_width_characters(self) -> None:
        assert normalized_text("teach​back") == "teachback"

    def test_is_case_insensitive(self) -> None:
        assert normalized_text("Teach-Back") == normalized_text("teach-back")

    def test_offsets_map_back_to_source(self) -> None:
        source = "The  patient’s  understanding"  # noqa: RUF001 - the curly glyph is the input under test
        norm = normalize(source)
        idx = norm.text.index("understanding")
        start, end = norm.source_slice(idx, idx + len("understanding"))
        assert source[start:end] == "understanding"

    def test_offsets_survive_ligature_expansion(self) -> None:
        # 'ﬁ' is one source character that becomes two normalised ones; the
        # offset map has to stay consistent across that.
        source = "we conﬁrm understanding"
        norm = normalize(source)
        idx = norm.text.index("understanding")
        start, end = norm.source_slice(idx, idx + len("understanding"))
        assert source[start:end] == "understanding"


class TestLocateSpan:
    DOC = (
        "Results\n\nTeach-back involves asking patients to explain in their own\n"
        "words what a health provider has just told them. Any misunder-\n"
        "standings are then clariﬁed by the health provider."
    )

    def test_finds_span_across_a_line_break(self) -> None:
        match = locate_span(
            "Teach-back involves asking patients to explain in their own words", self.DOC
        )
        assert match is not None
        assert match.source_text.startswith("Teach-back involves")

    def test_recovers_the_exact_source_text_not_the_claim(self) -> None:
        # The claimed span says 'clarified'; the document says 'clariﬁed'.
        # What comes back must be what the document says.
        match = locate_span("are then clarified by the health provider", self.DOC)
        assert match is not None
        assert "clariﬁed" in match.source_text
        assert self.DOC[match.start : match.end] == match.source_text

    def test_finds_span_across_hyphenated_break(self) -> None:
        match = locate_span("Any misunderstandings are then", self.DOC)
        assert match is not None
        assert "misunder-\nstandings" in match.source_text

    def test_returns_none_for_text_not_present(self) -> None:
        assert locate_span("Teach-back reduced thirty-day readmissions by 45%", self.DOC) is None

    def test_returns_none_when_a_clause_is_dropped(self) -> None:
        # A "quote" that removes words is not a quote. This is the case a
        # similarity-scoring validator would wrongly accept.
        assert locate_span("Teach-back involves asking patients to explain what", self.DOC) is None

    def test_returns_none_when_two_sentences_are_welded_together(self) -> None:
        welded = "words what a health provider has just told them. Teach-back is effective."
        assert locate_span(welded, self.DOC) is None

    def test_empty_span_is_not_a_match(self) -> None:
        assert locate_span("   ", self.DOC) is None

    def test_reuses_a_prenormalised_document(self) -> None:
        norm = normalize(self.DOC)
        match = locate_span("in their own words", self.DOC, normalized_document=norm)
        assert match is not None

    def test_rejects_a_prenormalised_document_from_different_text(self) -> None:
        import pytest

        norm = normalize("a completely different document about something else")
        with pytest.raises(ValueError, match="different text"):
            locate_span("in their own words", self.DOC, normalized_document=norm)


class TestLayoutGlueMatching:
    """The second-pass match, and the near-misses it deliberately does not reach.

    Every case here is drawn from the real corpus. Measuring the 26 spans the
    validator could not locate showed they were three different things wearing
    one label, and only one of the three is the validator's fault:

    - 12 were a word the PDF extractor split or joined at a column break. The
      model quoted the word as the printed page shows it. That is a rendering
      difference and the glue pass now reaches it.
    - 6 were the model altering characters that really are in the text — an
      inline superscript reference marker, or punctuation inside a statistics
      string. Reaching those would mean folding digits and punctuation, which
      is where a provenance check turns into similarity scoring.
    - 8 were genuine misquotation: substituted words, dropped content words,
      and non-adjacent sentences welded together.

    The tests below hold that line from both sides.
    """

    # Real extraction artefacts: the page reads one word, the text file does not.
    SPLIT = (
        "By investing time and effort in show ing presence and empathy, whether at the beginning"
    )
    JOINED = (
        "patients tend to prefer an active collaborative decisionmaking role and greater levels"
    )
    LOST_SPACE = "reflected in the perception of those inthe sdm group that they hada greater role"

    def test_matches_a_word_the_extractor_split_across_a_column_break(self) -> None:
        match = locate_span(
            "By investing time and effort in showing presence and empathy, whether at", self.SPLIT
        )
        assert match is not None
        assert match.via == "glued"
        # What is stored is still the paper's own text, artefact and all.
        assert "show ing" in match.source_text
        assert self.SPLIT[match.start : match.end] == match.source_text

    def test_matches_a_hyphen_the_extractor_dropped_at_a_line_break(self) -> None:
        match = locate_span(
            "patients tend to prefer an active collaborative decision-making role and greater",
            self.JOINED,
        )
        assert match is not None
        assert match.via == "glued"

    def test_matches_a_space_the_extractor_dropped(self) -> None:
        match = locate_span(
            "reflected in the perception of those in the SDM group that they had a greater role",
            self.LOST_SPACE,
        )
        assert match is not None
        assert match.via == "glued"

    def test_a_strict_match_never_reports_itself_as_glued(self) -> None:
        doc = "Teach-back involves asking patients to explain in their own words."
        match = locate_span("Teach-back involves asking patients to explain", doc)
        assert match is not None
        assert match.via == "exact"

    # --- the near-misses the glue pass must NOT reach ----------------------

    def test_does_not_reach_a_dropped_reference_marker(self) -> None:
        # The source carries an inlined superscript citation; the model dropped it.
        doc = "We have shown a relationship between trust and patient-centered communication17;"
        assert (
            locate_span(
                "We have shown a relationship between trust and patient-centered communication;",
                doc,
            )
            is None
        )

    def test_does_not_reach_altered_punctuation_in_a_statistics_string(self) -> None:
        # 'B = 0.374; β' and 'B = 0.374, β' are different readings of a result.
        doc = "significant in both groups (B = 0.861, β = 0.720; B = 0.374; β = 0.562; p < .001)."
        assert (
            locate_span(
                "significant in both groups (B = 0.861, β = 0.720; B = 0.374, β = 0.562; p < .001).",
                doc,
            )
            is None
        )

    def test_does_not_reach_a_substituted_content_word(self) -> None:
        doc = "clinicians should be mindful of language barriers, prior negative experiences of racism"
        assert (
            locate_span(
                "clinicians should be mindful of language barriers, prior even experiences of racism",
                doc,
            )
            is None
        )

    def test_does_not_reach_a_dropped_content_word(self) -> None:
        doc = "Advanced empathic communication skills training did not extensively cover complex cases"
        assert (
            locate_span(
                "Advanced communication skills training did not extensively cover complex cases",
                doc,
            )
            is None
        )

    def test_glue_does_not_weld_across_intervening_text(self) -> None:
        # Deleting spaces must not let two separated fragments join up.
        doc = "the plan was agreed. Several paragraphs later, the patient was satisfied."
        assert locate_span("the plan was agreed the patient was satisfied here now", doc) is None

    def test_glued_match_must_begin_on_a_word_boundary(self) -> None:
        # 'therapist' contains 'the rapist' once spaces are deleted; a glued
        # match starting partway through a word is never returned.
        doc = "the therapist explained the plan"
        match = locate_span("he rapist explained", doc)
        assert match is None


class TestSurroundingContext:
    def test_returns_text_either_side_snapped_to_word_boundaries(self) -> None:
        doc = "alpha beta gamma delta epsilon zeta eta theta"
        start = doc.index("delta")
        before, after = surrounding_context(doc, start, start + len("delta"), window=12)
        assert not before.endswith("gam")
        assert "gamma" in before
        assert "epsilon" in after
