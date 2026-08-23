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


class TestSurroundingContext:
    def test_returns_text_either_side_snapped_to_word_boundaries(self) -> None:
        doc = "alpha beta gamma delta epsilon zeta eta theta"
        start = doc.index("delta")
        before, after = surrounding_context(doc, start, start + len("delta"), window=12)
        assert not before.endswith("gam")
        assert "gamma" in before
        assert "epsilon" in after
