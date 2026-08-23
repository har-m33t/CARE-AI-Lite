"""Unit tests for carelite.kb.review.

Rendering and sign-off parsing are tested without a database, against
synthetic `ReviewRow` objects. The parsing tests carry the weight: sign-off
round-trips through a Markdown file a human has edited by hand, so the parser
has to cope with a file that has been reflowed, partially ticked, or ticked
with a capital X — and must never read an unticked entry as approved.
"""

from __future__ import annotations

import hashlib

import pytest

from carelite.kb.papers import PaperText
from carelite.kb.review import ReviewRow, parse_signoff, render_digest

SPAN = "patients receiving teach-back demonstrated significantly higher recall"
PAPER_TEXT = (
    "Results\n\nAcross the pooled analysis, "
    + SPAN
    + " of their discharge instructions at thirty days than those who did not."
)


@pytest.fixture(autouse=True)
def _synthetic_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render the digest against a fixture paper, not the fetched corpus.

    `make check` must not depend on `data/pdfs/` being populated — those files
    are gitignored and absent on a fresh clone, and a digest test that silently
    passes because every paper is missing would test nothing at all.
    """
    paper = PaperText(
        paper_id="10-1371-journal-pone-0231350",
        source_path="/tmp/fixture.xml",
        text=PAPER_TEXT,
        text_sha256=hashlib.sha256(PAPER_TEXT.encode()).hexdigest(),
    )
    monkeypatch.setattr("carelite.kb.review.load_paper_texts", lambda: {paper.paper_id: paper})


def _row(
    entry_id: str = "kb-teach_back-0123456789",
    theme: str = "teach_back",
    paper_ids: list[str] | None = None,
    verified: bool = False,
) -> ReviewRow:
    return ReviewRow(
        entry_id=entry_id,
        theme=theme,
        finding="Teach-back improved recall across health literacy levels.",
        practical_takeaway="Ask the patient to restate the plan in their own words.",
        example_behavior="Inviting a restatement of the plan before closing.",
        evidence_tier="strong",
        action_type="generation",
        verbatim_span=SPAN,
        encounter_phase=["explanation"],
        equity_relevant=False,
        human_verified=verified,
        paper_ids=paper_ids or ["10-1371-journal-pone-0231350"],
    )


class TestRenderDigest:
    def test_shows_the_span_inside_its_surrounding_context(self) -> None:
        out = render_digest([_row()])
        assert f">>> {SPAN} <<<" in out
        assert "Across the pooled analysis" in out
        assert "thirty days" in out

    def test_includes_every_field_a_reviewer_needs(self) -> None:
        out = render_digest([_row()])
        assert "kb-teach_back-0123456789" in out
        assert "Teach-back improved recall" in out
        assert "restate the plan in their own words" in out
        assert "**Finding.**" in out
        assert "**Takeaway.**" in out

    def test_shows_the_study_design_beside_the_claimed_tier(self) -> None:
        out = render_digest([_row()])
        assert "systematic review" in out
        assert "claimed tier **strong**" in out

    def test_unverified_entries_render_an_empty_checkbox(self) -> None:
        assert "- [ ] `kb-teach_back-0123456789`" in render_digest([_row()])

    def test_already_verified_entries_render_a_ticked_checkbox(self) -> None:
        assert "- [x] `kb-teach_back-0123456789`" in render_digest([_row(verified=True)])

    def test_flags_a_single_source_theme(self) -> None:
        rows = [
            _row(entry_id="kb-teach_back-0000000001"),
            _row(entry_id="kb-teach_back-0000000002"),
        ]
        out = render_digest(rows)
        assert "single-source" in out

    def test_does_not_flag_a_theme_with_several_sources(self) -> None:
        rows = [
            _row(entry_id="kb-empathy-0000000001", theme="empathy", paper_ids=["10-1370-afm-348"]),
            _row(
                entry_id="kb-empathy-0000000002",
                theme="empathy",
                paper_ids=["10-1371-journal-pone-0247259"],
            ),
        ]
        out = render_digest(rows)
        assert "| empathy | 2 | 2 |" in out

    def test_warns_when_a_span_no_longer_appears_in_the_paper(self) -> None:
        """A digest must never present an unlocatable span as reviewable."""
        row = _row()
        row.verbatim_span = "a sentence that appears in no paper in this corpus at all"
        out = render_digest([row])
        assert "WARNING" in out
        assert "Do not sign off" in out

    def test_coverage_table_counts_entries_per_theme(self) -> None:
        rows = [
            _row(entry_id="kb-teach_back-0000000001"),
            _row(entry_id="kb-empathy-1", theme="empathy"),
        ]
        out = render_digest(rows)
        assert "| teach_back | 1 |" in out
        assert "| empathy | 1 |" in out


class TestParseSignoff:
    DIGEST = """# Knowledge Base Review Digest

## teach_back

- [x] `kb-teach_back-0000000001`

  **Finding.** something

- [ ] `kb-teach_back-0000000002`

  **Finding.** something else

- [X] `kb-empathy-0000000003`
"""

    def test_reads_ticked_entries_as_approved(self) -> None:
        approved, _ = parse_signoff(self.DIGEST)
        assert "kb-teach_back-0000000001" in approved

    def test_accepts_a_capital_x(self) -> None:
        approved, _ = parse_signoff(self.DIGEST)
        assert "kb-empathy-0000000003" in approved

    def test_never_reads_an_unticked_entry_as_approved(self) -> None:
        approved, unticked = parse_signoff(self.DIGEST)
        assert "kb-teach_back-0000000002" not in approved
        assert "kb-teach_back-0000000002" in unticked

    def test_ignores_prose_that_is_not_a_checkbox(self) -> None:
        approved, unticked = parse_signoff("Some notes about kb-teach_back-0000000001 here.")
        assert approved == []
        assert unticked == []

    def test_a_rendered_digest_round_trips(self) -> None:
        rows = [
            _row(entry_id="kb-teach_back-0000000001", verified=True),
            _row(entry_id="kb-empathy-2", theme="empathy"),
        ]
        approved, unticked = parse_signoff(render_digest(rows))
        assert approved == ["kb-teach_back-0000000001"]
        assert unticked == ["kb-empathy-2"]


@pytest.mark.db
class TestSignoffAgainstPostgres:
    def test_record_signoff_returns_zero_for_an_empty_list(self) -> None:
        from carelite.kb.review import record_signoff

        assert record_signoff([]) == 0
