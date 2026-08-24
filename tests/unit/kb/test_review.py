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
from carelite.kb.review import EntryAudit, ReviewRow, parse_signoff, render_digest

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
    tier: str = "strong",
) -> ReviewRow:
    return ReviewRow(
        entry_id=entry_id,
        theme=theme,
        finding="Teach-back improved recall across health literacy levels.",
        practical_takeaway="Ask the patient to restate the plan in their own words.",
        example_behavior="Inviting a restatement of the plan before closing.",
        evidence_tier=tier,
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

    def test_shows_the_study_design_beside_the_stored_tier(self) -> None:
        out = render_digest([_row()])
        assert "systematic review" in out
        assert "evidence tier **strong**" in out

    def test_an_uncorrected_entry_shows_no_downgrade_note(self) -> None:
        audit = {"kb-teach_back-0123456789": EntryAudit("strong", "strong", "exact")}
        out = render_digest([_row()], audit=audit)
        assert "corrected down" not in out

    def test_a_corrected_tier_shows_the_models_original_claim(self) -> None:
        """The reviewer must be able to see the overreach, not just the fix.

        A digest that printed only the corrected tier would be hiding the most
        interesting thing the pipeline did to the entry, and a reviewer could
        not tell a well-judged correction from a wrong one.
        """
        audit = {"kb-teach_back-0123456789": EntryAudit("strong", "emerging", "exact")}
        out = render_digest([_row(tier="emerging")], audit=audit)
        assert "Evidence tier **emerging**" in out
        assert "corrected down from the model's claim of *strong*" in out
        assert "Evidence tier corrected on 1 of 1 entries" in out

    def test_a_glued_span_match_is_flagged_for_the_reviewer(self) -> None:
        audit = {"kb-teach_back-0123456789": EntryAudit("strong", "strong", "glued")}
        out = render_digest([_row()], audit=audit)
        assert "layout-glue normalisation" in out

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

    def test_flags_two_entries_whose_takeaways_restate_each_other(self) -> None:
        """Both entries are valid; counting both as evidence is what is wrong.

        The validator cannot reject these — real span, real source, well-formed
        entry — so the only place the overlap can be caught is a human's eye,
        and the digest has to point at it.
        """
        a = _row(entry_id="kb-teach_back-0000000001")
        b = _row(entry_id="kb-teach_back-0000000002")
        b.practical_takeaway = "Ask the patient to restate the plan in their own words please."
        out = render_digest([a, b])
        assert "Entries that may restate each other" in out
        assert "kb-teach_back-0000000001" in out
        assert "same paper" in out

    def test_distinct_takeaways_produce_no_overlap_section(self) -> None:
        a = _row(entry_id="kb-teach_back-0000000001")
        b = _row(entry_id="kb-empathy-0000000002", theme="empathy")
        b.practical_takeaway = "Name the emotion you have just heard before moving to the plan."
        out = render_digest([a, b])
        assert "Entries that may restate each other" not in out

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
    SEED_PAPER = "10.1/kb-review-signoff-test"
    SEED_ENTRY = "kb-teach_back-0000000042"

    @pytest.fixture
    def _seeded(self):
        """One entry of our own to sign off, removed afterwards.

        The database holds the real knowledge base and other lanes' work, so
        this must never tick a row it did not create.
        """
        from carelite.db import connect

        with connect(autocommit=True) as conn:
            conn.execute(
                "INSERT INTO paper (paper_id, doi, apa_citation, evidence_tier) "
                "VALUES (%s, %s, %s, 'strong') ON CONFLICT (paper_id) DO NOTHING",
                (self.SEED_PAPER, self.SEED_PAPER, "review signoff test paper"),
            )
            conn.execute(
                "INSERT INTO kb_entry (entry_id, theme, finding, practical_takeaway, "
                "example_behavior, evidence_tier, action_type, verbatim_span) "
                "VALUES (%s, 'teach_back', %s, %s, %s, 'strong', 'generation', %s) "
                "ON CONFLICT (entry_id) DO NOTHING",
                (
                    self.SEED_ENTRY,
                    "seeded finding",
                    "Ask the patient to restate the plan in their own words.",
                    "Inviting a restatement before closing.",
                    SPAN,
                ),
            )
            conn.execute(
                "INSERT INTO kb_entry_source (entry_id, paper_id) VALUES (%s, %s) "
                "ON CONFLICT DO NOTHING",
                (self.SEED_ENTRY, self.SEED_PAPER),
            )
        yield
        with connect(autocommit=True) as conn:
            conn.execute("DELETE FROM kb_entry WHERE entry_id = %s", (self.SEED_ENTRY,))
            conn.execute("DELETE FROM paper WHERE paper_id = %s", (self.SEED_PAPER,))

    def test_record_signoff_returns_zero_for_an_empty_list(self) -> None:
        from carelite.kb.review import record_signoff

        assert record_signoff([]) == 0

    def test_a_ticked_digest_flips_human_verified_and_an_unticked_one_does_not(
        self, _seeded, tmp_path
    ) -> None:
        """The gate, end to end: a reviewer's tick is what sets `human_verified`.

        This is the only place in the pipeline where a human's judgment enters,
        so the round-trip is worth testing in the form the reviewer performs it
        — editing the Markdown file and running the sign-off command — rather
        than by calling `record_signoff` directly with an id.
        """
        from carelite.db import connect
        from carelite.kb.review import apply_signoff

        def verified() -> bool:
            with connect() as conn:
                row = conn.execute(
                    "SELECT human_verified FROM kb_entry WHERE entry_id = %s", (self.SEED_ENTRY,)
                ).fetchone()
            assert row is not None
            return bool(row["human_verified"])

        assert verified() is False

        digest = tmp_path / "digest.md"
        digest.write_text(f"- [ ] `{self.SEED_ENTRY}`\n", encoding="utf-8")
        result = apply_signoff(digest, reviewer="tester")
        assert result["approved"] == 0
        assert verified() is False, "an unticked entry must never be recorded as reviewed"

        digest.write_text(f"- [x] `{self.SEED_ENTRY}`\n", encoding="utf-8")
        result = apply_signoff(digest, reviewer="tester")
        assert result["approved"] == 1
        assert result["rows_updated"] == 1
        assert verified() is True

    def test_reloading_an_entry_does_not_undo_its_signoff(self, _seeded) -> None:
        """The failure this lane most needs to not have: a quiet un-verify.

        Adding `human_verified` to the loader's ON CONFLICT clause would break
        nothing visibly and would silently discard every review decision on the
        next load. `test_upsert_never_touches_human_verified` checks the SQL;
        this checks the behaviour against a live database.
        """
        from carelite.db import connect
        from carelite.kb.load import load_entries
        from carelite.kb.review import record_signoff
        from carelite.kb.validate import ValidatedEntry
        from carelite.types import ActionType, EvidenceTier, KBEntry, Theme

        assert record_signoff([self.SEED_ENTRY]) == 1

        entry = KBEntry(
            entry_id=self.SEED_ENTRY,
            theme=Theme.TEACH_BACK,
            finding="seeded finding, reloaded",
            practical_takeaway="Ask the patient to restate the plan in their own words.",
            example_behavior="Inviting a restatement before closing.",
            evidence_tier=EvidenceTier.STRONG,
            action_type=ActionType.GENERATION,
            verbatim_span=SPAN,
            source_paper_ids=[self.SEED_PAPER],
        )
        load_entries(
            [
                ValidatedEntry(
                    entry=entry,
                    paper_id=self.SEED_PAPER,
                    span_start=0,
                    span_end=len(SPAN),
                    span_was_exact=True,
                    paper_sha256=hashlib.sha256(b"x").hexdigest(),
                    claimed_tier=EvidenceTier.STRONG,
                )
            ]
        )

        with connect() as conn:
            row = conn.execute(
                "SELECT finding, human_verified FROM kb_entry WHERE entry_id = %s",
                (self.SEED_ENTRY,),
            ).fetchone()
        assert row is not None
        assert row["finding"] == "seeded finding, reloaded", "the reload should have happened"
        assert row["human_verified"] is True, "and must not have cleared the sign-off"
