"""Unit tests for carelite.kb.papers.

The evidence-tier judgment is the thing to protect here. It is a human call
this lane makes on the corpus's behalf, and the invariant that keeps it
auditable is that tier is *derived* from a recorded study design and never
assigned directly — so raising a paper's tier requires changing what the table
claims that paper's design is, which a reviewer can check against its Methods.
"""

from __future__ import annotations

import re

import pytest

from carelite.kb.papers import (
    DESIGN_TIER,
    PAPER_META,
    strongest_tier,
    tier_at_most,
    tier_ceiling,
)
from carelite.types import EvidenceTier


class TestDesignTierMapping:
    def test_every_recorded_design_has_a_tier(self) -> None:
        for meta in PAPER_META.values():
            assert meta.design in DESIGN_TIER

    def test_tier_is_derived_not_stored(self) -> None:
        for meta in PAPER_META.values():
            assert meta.evidence_tier == DESIGN_TIER[meta.design]

    def test_a_protocol_is_never_stronger_than_emerging(self) -> None:
        assert DESIGN_TIER["study protocol"] is EvidenceTier.EMERGING

    def test_protocols_in_the_corpus_are_all_emerging(self) -> None:
        protocols = [m for m in PAPER_META.values() if m.design == "study protocol"]
        assert protocols  # the corpus does contain protocols; guard against a silent empty pass
        assert all(m.evidence_tier is EvidenceTier.EMERGING for m in protocols)

    def test_reviews_and_trials_are_strong(self) -> None:
        assert DESIGN_TIER["systematic review and meta-analysis"] is EvidenceTier.STRONG
        assert DESIGN_TIER["randomized controlled trial"] is EvidenceTier.STRONG

    def test_paper_ids_are_unique(self) -> None:
        assert len({m.paper_id for m in PAPER_META.values()}) == len(PAPER_META)


class TestTierComparison:
    def test_strongest_tier_of_a_mixed_set(self) -> None:
        assert (
            strongest_tier([EvidenceTier.EMERGING, EvidenceTier.STRONG, EvidenceTier.MODERATE])
            is EvidenceTier.STRONG
        )

    def test_strongest_tier_of_nothing_is_none(self) -> None:
        assert strongest_tier([]) is None

    def test_two_moderate_sources_do_not_make_a_strong_ceiling(self) -> None:
        assert (
            strongest_tier([EvidenceTier.MODERATE, EvidenceTier.MODERATE]) is EvidenceTier.MODERATE
        )

    @pytest.mark.parametrize(
        ("claimed", "ceiling", "expected"),
        [
            (EvidenceTier.EMERGING, EvidenceTier.STRONG, True),
            (EvidenceTier.STRONG, EvidenceTier.STRONG, True),
            (EvidenceTier.STRONG, EvidenceTier.EMERGING, False),
            (EvidenceTier.MODERATE, EvidenceTier.EMERGING, False),
        ],
    )
    def test_tier_at_most(
        self, claimed: EvidenceTier, ceiling: EvidenceTier, expected: bool
    ) -> None:
        assert tier_at_most(claimed, ceiling) is expected

    def test_tier_ceiling_ignores_unknown_papers(self) -> None:
        assert tier_ceiling(["not-a-paper-id"]) is None


class TestCitations:
    """Citations are derived from Crossref, not composed. The tests check for the
    shape a derived reference has and the absence of the placeholder it replaced.

    A fabricated citation would be the same failure this lane exists to prevent,
    one field over — and it would be an invisible one, because a plausible
    reference reads exactly like a real reference.
    """

    def test_no_paper_still_carries_the_fetch_placeholder(self) -> None:
        for meta in PAPER_META.values():
            assert "[citation pending]" not in meta.apa_citation

    def test_every_apa_citation_carries_its_own_doi(self) -> None:
        """The one part of a reference that can be checked mechanically."""
        for paper_id, meta in PAPER_META.items():
            assert "https://doi.org/" in meta.apa_citation, paper_id

    def test_every_apa_citation_names_an_author_a_year_and_a_journal(self) -> None:
        for paper_id, meta in PAPER_META.items():
            assert re.match(r"^[A-ZÇÖÅÆØÉ]", meta.apa_citation), paper_id
            assert f"({meta.year})" in meta.apa_citation, paper_id
            # author (year). Title. Journal…  — four sentence-final stops minimum
            assert meta.apa_citation.count(". ") >= 3, paper_id

    def test_short_citation_year_matches_the_recorded_year(self) -> None:
        for paper_id, meta in PAPER_META.items():
            assert f"({meta.year})" in meta.short_citation, paper_id

    def test_years_are_plausible(self) -> None:
        for paper_id, meta in PAPER_META.items():
            assert 1990 <= meta.year <= 2026, paper_id


@pytest.mark.db
class TestSyncPaperMetadata:
    """The design has to reach the table, not just the module.

    This was a real defect and a quiet one: `PAPER_META` knew every paper's
    design while all 33 `paper` rows sat at `design IS NULL`,
    `evidence_tier = 'emerging'` and a placeholder citation. The digest, which
    imports this module, printed "randomized controlled trial"; the CLI's
    evidence panel, which reads the table, would have shown a clinician
    "[citation pending]". Nothing failed — the two just disagreed.
    """

    def test_sync_writes_design_tier_citation_and_year(self) -> None:
        from carelite.db.connection import transaction
        from carelite.kb.papers import placeholder_counts, sync_paper_metadata

        result = sync_paper_metadata()
        assert result.rows_updated > 0

        counts = placeholder_counts()
        assert counts["null_design"] == 0
        assert counts["pending_citation"] == 0

        with transaction() as conn:
            rows = conn.execute(
                "SELECT paper_id, design, evidence_tier, apa_citation, year FROM paper"
            ).fetchall()
        for row in rows:
            meta = PAPER_META.get(row["paper_id"])
            if meta is None:
                continue
            assert row["design"] == meta.design
            assert row["evidence_tier"] == meta.evidence_tier.value
            assert row["apa_citation"] == meta.apa_citation
            assert row["year"] == meta.year

    def test_sync_is_idempotent(self) -> None:
        from carelite.kb.papers import sync_paper_metadata

        first = sync_paper_metadata()
        second = sync_paper_metadata()
        assert first.rows_updated == second.rows_updated
