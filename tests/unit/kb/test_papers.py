"""Unit tests for carelite.kb.papers.

The evidence-tier judgment is the thing to protect here. It is a human call
this lane makes on the corpus's behalf, and the invariant that keeps it
auditable is that tier is *derived* from a recorded study design and never
assigned directly — so raising a paper's tier requires changing what the table
claims that paper's design is, which a reviewer can check against its Methods.
"""

from __future__ import annotations

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
