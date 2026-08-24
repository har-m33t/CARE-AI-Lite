"""Unit tests for carelite.kb.load.

Parameter shaping and the SQL's shape are plain unit tests, in `make check`.
Anything opening a connection is `@pytest.mark.db`.

The assertion that matters most here is the negative one:
`test_upsert_never_touches_human_verified`. Adding `human_verified` to the
`ON CONFLICT` clause would be a one-word change that breaks nothing visibly,
passes every other test, and silently un-verifies the whole knowledge base the
next time the loader runs.
"""

from __future__ import annotations

import hashlib

import pytest

from carelite.kb.load import (
    _UPSERT_ENTRY_SQL,
    entry_params,
    load_entries,
    orphaned_entries,
)
from carelite.kb.validate import ValidatedEntry
from carelite.types import ActionType, EncounterPhase, EvidenceTier, KBEntry, Theme

ENTRY_ID = "kb-teach_back-0000000001"
PAPER_ID = "10.1/kb-load-test-paper"


def _entry(entry_id: str = ENTRY_ID, paper_id: str = PAPER_ID) -> KBEntry:
    return KBEntry(
        entry_id=entry_id,
        theme=Theme.TEACH_BACK,
        finding="Teach-back improved recall across health literacy levels.",
        practical_takeaway="Ask the patient to restate the plan in their own words.",
        example_behavior="Inviting a restatement of the plan before closing.",
        evidence_tier=EvidenceTier.STRONG,
        action_type=ActionType.GENERATION,
        verbatim_span="patients receiving teach-back demonstrated significantly higher recall",
        source_paper_ids=[paper_id],
        encounter_phase=[EncounterPhase.EXPLANATION, EncounterPhase.CLOSING],
        equity_relevant=True,
    )


def _validated(entry: KBEntry | None = None) -> ValidatedEntry:
    entry = entry or _entry()
    return ValidatedEntry(
        entry=entry,
        paper_id=entry.source_paper_ids[0],
        span_start=10,
        span_end=10 + len(entry.verbatim_span),
        span_was_exact=True,
        paper_sha256=hashlib.sha256(b"x").hexdigest(),
        claimed_tier=entry.evidence_tier,
    )


class TestEntryParams:
    def test_enums_are_flattened_to_their_values(self) -> None:
        params = entry_params(_entry())
        assert params["theme"] == "teach_back"
        assert params["evidence_tier"] == "strong"
        assert params["action_type"] == "generation"
        assert params["encounter_phase"] == ["explanation", "closing"]

    def test_every_column_in_the_insert_has_a_parameter(self) -> None:
        import re

        named = set(re.findall(r"%\((\w+)s?\)s", _UPSERT_ENTRY_SQL))
        named = {n.rstrip(")") for n in named}
        assert named <= set(entry_params(_entry()))

    def test_empty_list_fields_are_empty_lists_not_none(self) -> None:
        # The schema declares TEXT[] NOT NULL DEFAULT '{}'; None would fail.
        params = entry_params(_entry())
        assert params["nurse_component"] == []
        assert params["four_habits"] == []


class TestUpsertSql:
    def test_upsert_never_touches_human_verified(self) -> None:
        conflict_clause = _UPSERT_ENTRY_SQL.split("DO UPDATE SET", 1)[1]
        assignments = [
            line for line in conflict_clause.splitlines() if "=" in line and "--" not in line
        ]
        assert assignments
        assert not any("human_verified" in line for line in assignments)

    def test_upsert_is_keyed_on_entry_id(self) -> None:
        assert "ON CONFLICT (entry_id)" in _UPSERT_ENTRY_SQL


@pytest.mark.db
class TestAgainstPostgres:
    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Removes only rows this test file created. The database is shared
        with the real corpus and with other lanes, so a blanket delete would
        destroy their work."""
        yield
        from carelite.db import connect

        with connect(autocommit=True) as conn:
            conn.execute("DELETE FROM kb_entry WHERE entry_id LIKE 'kb-teach_back-00000%'")
            conn.execute("DELETE FROM paper WHERE paper_id = %s", (PAPER_ID,))

    def _seed_paper(self) -> None:
        from carelite.db import connect

        with connect(autocommit=True) as conn:
            conn.execute(
                """
                INSERT INTO paper (paper_id, doi, apa_citation, evidence_tier)
                VALUES (%s, %s, %s, 'strong')
                ON CONFLICT (paper_id) DO NOTHING
                """,
                (PAPER_ID, PAPER_ID, "test paper"),
            )

    def test_every_loaded_span_is_a_literal_slice_of_its_source_paper(self) -> None:
        """The provenance guarantee, asserted against the database itself.

        Everything else in this lane checks candidates on their way *in*. This
        checks what is actually sitting in `kb_entry` right now, joined to
        `kb_entry_source` and then to the paper text on disk, using plain
        substring containment with no normalisation of any kind. That is the
        claim the write-up makes about the knowledge base, so it is worth
        testing in the form the claim is made: every stored span is text that
        appears, character for character, in the paper the entry cites.

        Skips rather than fails on an empty table, so it is a real assertion
        after a load and not an obstacle before one.
        """
        from carelite.db import connect
        from carelite.kb.papers import load_paper_texts

        with connect() as conn:
            rows = conn.execute(
                """
                SELECT e.entry_id, e.verbatim_span,
                       array_agg(s.paper_id ORDER BY s.paper_id) AS paper_ids
                FROM kb_entry e
                JOIN kb_entry_source s USING (entry_id)
                GROUP BY e.entry_id
                """
            ).fetchall()

        if not rows:
            pytest.skip("kb_entry is empty; run `python -m carelite.kb.load` first")

        papers = load_paper_texts()
        unlocatable = [
            r["entry_id"]
            for r in rows
            if not any(
                r["verbatim_span"] in papers[pid].text for pid in r["paper_ids"] if pid in papers
            )
        ]
        assert not unlocatable, (
            f"{len(unlocatable)} of {len(rows)} loaded entries carry a span that is not "
            f"a literal substring of any paper they cite: {unlocatable[:5]}"
        )

    def test_no_loaded_entry_is_missing_its_source_link(self) -> None:
        """A `kb_entry` row with no `kb_entry_source` row satisfies the schema
        perfectly well and is exactly the untraceable entry this lane exists to
        prevent, so it is checked separately from the span itself."""
        from carelite.db import connect

        with connect() as conn:
            n = conn.execute("SELECT count(*) AS n FROM kb_entry").fetchone()
        if not n or n["n"] == 0:
            pytest.skip("kb_entry is empty; run `python -m carelite.kb.load` first")
        assert orphaned_entries() == []

    def test_entry_and_source_link_are_both_written(self) -> None:
        from carelite.db import connect

        self._seed_paper()
        result = load_entries([_validated()])
        assert result.entries_written == 1
        assert result.sources_written == 1

        with connect() as conn:
            row = conn.execute(
                "SELECT human_verified FROM kb_entry WHERE entry_id = %s", (ENTRY_ID,)
            ).fetchone()
            assert row is not None
            assert row["human_verified"] is False

            link = conn.execute(
                "SELECT paper_id FROM kb_entry_source WHERE entry_id = %s", (ENTRY_ID,)
            ).fetchone()
            assert link is not None
            assert link["paper_id"] == PAPER_ID

    def test_reload_does_not_clear_human_verified(self) -> None:
        from carelite.db import connect

        self._seed_paper()
        load_entries([_validated()])
        with connect(autocommit=True) as conn:
            conn.execute(
                "UPDATE kb_entry SET human_verified = TRUE WHERE entry_id = %s", (ENTRY_ID,)
            )

        load_entries([_validated()])

        with connect() as conn:
            row = conn.execute(
                "SELECT human_verified FROM kb_entry WHERE entry_id = %s", (ENTRY_ID,)
            ).fetchone()
        assert row is not None
        assert row["human_verified"] is True

    def test_an_entry_whose_paper_is_absent_is_skipped_not_orphaned(self) -> None:
        entry = _entry(entry_id="kb-teach_back-0000000002", paper_id="10.1/no-such-paper")
        result = load_entries([_validated(entry)])

        assert result.entries_written == 0
        assert entry.entry_id in result.skipped_unknown_paper
        assert entry.entry_id not in orphaned_entries()

    def test_every_loaded_entry_has_a_source_link(self) -> None:
        self._seed_paper()
        load_entries([_validated()])
        assert orphaned_entries() == []
