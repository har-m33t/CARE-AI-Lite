"""Unit tests for carelite.kb.validate — the provenance gate.

`TestFabricatedSpanIsRejected` is the reason this file exists. Everything else
here protects it: a validator that accepted fabrications would still pass every
other test in this suite while silently making the whole knowledge base
worthless, so the fabrication case is tested from several directions (invented
sentence, plausible paraphrase, dropped clause, right-paper-wrong-paper) rather
than once.

None of these tests touch Postgres or a model. Papers are synthetic fixtures
built with `PaperText`, so the suite runs in `make check`.
"""

from __future__ import annotations

import hashlib

import pytest

from carelite.kb.extract import CandidateEntry
from carelite.kb.papers import PaperMeta, PaperText
from carelite.kb.validate import (
    RejectedCandidate,
    ValidatedEntry,
    entry_id_for,
    takeaway_is_actionable,
    validate_candidate,
    validate_candidates,
)
from carelite.types import EvidenceTier, Theme

TEACH_BACK_TEXT = (
    "Methods\n\n"
    "We searched five databases for studies of teach-back in adult populations.\n\n"
    "Results\n\n"
    "Teach-back involves asking patients to explain in their own words what a health "
    "provider has just told them. Any misunderstandings are then clariﬁed by the health "
    "provider and understanding is checked again. Across the included studies, teach-back "
    "was associated with improved disease-speciﬁc knowledge and self-eﬃcacy, and no study "
    "reported harm arising from its use.\n\n"
    "Discussion\n\n"
    "Implementation was inconsistent across settings."
)

PROTOCOL_TEXT = (
    "This protocol describes a cluster randomised trial of shared decision-making in "
    "bipolar disorder. Recruitment is ongoing and no outcome data are reported here. "
    "The primary outcome will be medication adherence at twelve months, measured by "
    "pill count and self-report at each scheduled study visit."
)


def _paper(paper_id: str, text: str, design: str) -> PaperText:
    return PaperText(
        paper_id=paper_id,
        source_path=f"/tmp/{paper_id}.xml",
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        meta=PaperMeta(paper_id=paper_id, design=design, short_citation=f"{paper_id} (test)"),
    )


@pytest.fixture
def papers() -> dict[str, PaperText]:
    return {
        "review-paper": _paper("review-paper", TEACH_BACK_TEXT, "systematic review"),
        "protocol-paper": _paper("protocol-paper", PROTOCOL_TEXT, "study protocol"),
    }


def _candidate(**overrides: object) -> CandidateEntry:
    base: dict[str, object] = {
        "theme": "teach_back",
        "finding": "Teach-back is associated with improved disease-specific knowledge.",
        "practical_takeaway": (
            "Ask the patient to explain the plan back in their own words, and re-explain "
            "differently when the answer is incomplete."
        ),
        "example_behavior": "Inviting the patient to describe the medication change in their own words.",
        "evidence_tier": "strong",
        "action_type": "generation",
        "verbatim_span": (
            "Teach-back involves asking patients to explain in their own words what a "
            "health provider has just told them"
        ),
        "source_paper_ids": ["review-paper"],
    }
    base.update(overrides)
    return CandidateEntry.model_validate(base)


class TestFabricatedSpanIsRejected:
    """The central guarantee. A span that is not in the paper must fail."""

    def test_wholly_invented_span_is_rejected(self, papers: dict[str, PaperText]) -> None:
        candidate = _candidate(
            verbatim_span=(
                "Teach-back reduced thirty-day hospital readmissions by 45% in a "
                "multicentre randomised trial of 12,000 patients"
            )
        )
        result = validate_candidate(candidate, papers=papers)

        assert isinstance(result, RejectedCandidate)
        assert any("not found in review-paper" in r for r in result.reasons)

    def test_plausible_paraphrase_is_rejected(self, papers: dict[str, PaperText]) -> None:
        # Every word is drawn from the paper's vocabulary and the claim is even
        # true; the sentence is not in the paper. That is enough to reject.
        candidate = _candidate(
            verbatim_span=(
                "Patients are asked by health providers to explain in their own words "
                "what they have just been told, and misunderstandings are clarified"
            )
        )
        result = validate_candidate(candidate, papers=papers)

        assert isinstance(result, RejectedCandidate)
        assert any("not found" in r for r in result.reasons)

    def test_span_with_a_dropped_clause_is_rejected(self, papers: dict[str, PaperText]) -> None:
        candidate = _candidate(
            verbatim_span=(
                "Teach-back involves asking patients to explain what a health provider "
                "has just told them"
            )
        )
        result = validate_candidate(candidate, papers=papers)

        assert isinstance(result, RejectedCandidate)
        assert any("not found" in r for r in result.reasons)

    def test_span_from_a_different_paper_is_rejected(self, papers: dict[str, PaperText]) -> None:
        # The span is real — it is just not in the paper the entry cites.
        candidate = _candidate(
            source_paper_ids=["protocol-paper"],
            evidence_tier="emerging",
            verbatim_span=(
                "Teach-back involves asking patients to explain in their own words what a "
                "health provider has just told them"
            ),
        )
        result = validate_candidate(candidate, papers=papers)

        assert isinstance(result, RejectedCandidate)
        assert any("not found in protocol-paper" in r for r in result.reasons)

    def test_fabrication_survives_a_valid_batch(self, papers: dict[str, PaperText]) -> None:
        good = _candidate()
        bad = _candidate(verbatim_span="Teach-back eliminated all medication errors entirely.")
        report = validate_candidates([good, bad], papers=papers)

        assert len(report.accepted) == 1
        assert len(report.rejected) == 1
        assert report.rejection_rate == 0.5
        assert "fabricated span (not in source paper)" in report.reason_counts()


class TestAcceptedEntries:
    def test_a_well_formed_candidate_is_accepted(self, papers: dict[str, PaperText]) -> None:
        result = validate_candidate(_candidate(), papers=papers)

        assert isinstance(result, ValidatedEntry)
        assert result.entry.theme is Theme.TEACH_BACK
        assert result.entry.evidence_tier is EvidenceTier.STRONG
        assert result.paper_id == "review-paper"

    def test_stored_span_is_the_exact_source_substring(self, papers: dict[str, PaperText]) -> None:
        # The candidate spells 'clarified'; the paper has the ligature form.
        # What is stored must be the paper's bytes, not the model's.
        candidate = _candidate(
            verbatim_span=(
                "Any misunderstandings are then clarified by the health provider and "
                "understanding is checked again"
            )
        )
        result = validate_candidate(candidate, papers=papers)

        assert isinstance(result, ValidatedEntry)
        assert result.entry.verbatim_span in TEACH_BACK_TEXT
        assert "clariﬁed" in result.entry.verbatim_span
        assert result.span_was_exact is False

    def test_span_offsets_point_at_the_stored_span(self, papers: dict[str, PaperText]) -> None:
        result = validate_candidate(_candidate(), papers=papers)
        assert isinstance(result, ValidatedEntry)
        assert TEACH_BACK_TEXT[result.span_start : result.span_end] == result.entry.verbatim_span

    def test_entry_id_is_deterministic(self, papers: dict[str, PaperText]) -> None:
        first = validate_candidate(_candidate(), papers=papers)
        second = validate_candidate(_candidate(), papers=papers)
        assert isinstance(first, ValidatedEntry) and isinstance(second, ValidatedEntry)
        assert first.entry_id == second.entry_id

    def test_equity_theme_forces_the_equity_flag(self, papers: dict[str, PaperText]) -> None:
        candidate = _candidate(theme="equity", equity_relevant=False)
        result = validate_candidate(candidate, papers=papers)
        assert isinstance(result, ValidatedEntry)
        assert result.entry.equity_relevant is True


class TestVocabulary:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Teach-Back", Theme.TEACH_BACK),
            ("shared decision making", Theme.ACTIVATION_SDM),
            ("Emotion Recognition and Response", Theme.EMOTION_RESPONSE),
        ],
    )
    def test_theme_aliases_are_tolerated(
        self, papers: dict[str, PaperText], raw: str, expected: Theme
    ) -> None:
        result = validate_candidate(_candidate(theme=raw), papers=papers)
        assert isinstance(result, ValidatedEntry)
        assert result.entry.theme is expected

    def test_unknown_theme_is_rejected_with_a_readable_reason(
        self, papers: dict[str, PaperText]
    ) -> None:
        result = validate_candidate(_candidate(theme="bedside manner"), papers=papers)
        assert isinstance(result, RejectedCandidate)
        assert any("unknown theme" in r for r in result.reasons)

    def test_unknown_action_type_is_rejected(self, papers: dict[str, PaperText]) -> None:
        result = validate_candidate(_candidate(action_type="summarisation"), papers=papers)
        assert isinstance(result, RejectedCandidate)
        assert any("unknown action type" in r for r in result.reasons)


class TestTierAgainstDesign:
    """An overclaimed tier is corrected, not fatal — but the claim is kept.

    This is the one check in the validator that repairs rather than rejects,
    and the reason is that it is the only one with a derivable right answer:
    the study design is recorded, so the tier it supports is known. The
    entry's span, theme, finding and takeaway are untouched by the model's
    tier error, and discarding four correct fields to punish a fifth would
    manufacture a knowledge-base shortfall out of a field the pipeline can fix.
    What must not happen is the correction being silent, so both values
    survive on the result and the review digest prints them together.
    """

    def test_a_strong_claim_off_a_protocol_is_downgraded_not_rejected(
        self, papers: dict[str, PaperText]
    ) -> None:
        candidate = _candidate(
            source_paper_ids=["protocol-paper"],
            evidence_tier="strong",
            theme="activation_sdm",
            verbatim_span=(
                "The primary outcome will be medication adherence at twelve months, "
                "measured by pill count and self-report at each scheduled study visit"
            ),
        )
        result = validate_candidate(candidate, papers=papers)

        assert isinstance(result, ValidatedEntry)
        assert result.entry.evidence_tier is EvidenceTier.EMERGING
        assert result.claimed_tier is EvidenceTier.STRONG
        assert result.design_ceiling is EvidenceTier.EMERGING
        assert result.tier_downgraded is True

    def test_a_tier_within_the_ceiling_is_left_alone(self, papers: dict[str, PaperText]) -> None:
        result = validate_candidate(_candidate(evidence_tier="moderate"), papers=papers)
        assert isinstance(result, ValidatedEntry)
        assert result.tier_downgraded is False
        assert result.claimed_tier is result.entry.evidence_tier

    def test_the_report_counts_every_downgrade(self, papers: dict[str, PaperText]) -> None:
        overclaimed = _candidate(
            source_paper_ids=["protocol-paper"],
            evidence_tier="strong",
            theme="activation_sdm",
            verbatim_span=(
                "The primary outcome will be medication adherence at twelve months, "
                "measured by pill count and self-report at each scheduled study visit"
            ),
        )
        report = validate_candidates([overclaimed], papers=papers)
        assert len(report.downgraded) == 1
        assert report.downgrade_counts() == {"strong -> emerging": 1}

    def test_protocol_supports_an_emerging_claim(self, papers: dict[str, PaperText]) -> None:
        candidate = _candidate(
            source_paper_ids=["protocol-paper"],
            evidence_tier="emerging",
            theme="activation_sdm",
            verbatim_span=(
                "The primary outcome will be medication adherence at twelve months, "
                "measured by pill count and self-report at each scheduled study visit"
            ),
        )
        result = validate_candidate(candidate, papers=papers)
        assert isinstance(result, ValidatedEntry)

    def test_a_cautious_claim_below_the_ceiling_is_allowed(
        self, papers: dict[str, PaperText]
    ) -> None:
        result = validate_candidate(_candidate(evidence_tier="moderate"), papers=papers)
        assert isinstance(result, ValidatedEntry)


class TestSpanQuality:
    def test_a_span_shorter_than_the_evidence_floor_is_rejected(
        self, papers: dict[str, PaperText]
    ) -> None:
        # Present in the paper, but far too generic to prove anything.
        result = validate_candidate(_candidate(verbatim_span="Implementation was"), papers=papers)
        assert isinstance(result, RejectedCandidate)
        assert any("evidence floor" in r or "not found" in r for r in result.reasons)


class TestActionability:
    @pytest.mark.parametrize(
        "takeaway",
        [
            "Clinicians should receive communication skills training before practising.",
            "Medical schools should embed empathy training across the whole curriculum.",
            "Further research is needed to establish the effect on patient outcomes here.",
            "Health systems should invest in a communication training programme for staff.",
        ],
    )
    def test_training_and_policy_claims_are_not_actionable(self, takeaway: str) -> None:
        ok, reason = takeaway_is_actionable(takeaway)
        assert ok is False
        assert reason

    @pytest.mark.parametrize(
        "takeaway",
        [
            "Ask the patient to restate the plan in their own words before closing the visit.",
            "Name the emotion you hear before moving on to any clinical information at all.",
            "Elicit what matters most to the patient before recommending a treatment option.",
        ],
    )
    def test_encounter_level_moves_are_actionable(self, takeaway: str) -> None:
        ok, reason = takeaway_is_actionable(takeaway)
        assert ok is True
        assert reason is None

    def test_a_slogan_is_not_actionable(self) -> None:
        ok, _ = takeaway_is_actionable("Be more empathetic.")
        assert ok is False

    def test_non_actionable_takeaway_rejects_the_entry(self, papers: dict[str, PaperText]) -> None:
        candidate = _candidate(
            practical_takeaway="Clinicians should receive communication skills training."
        )
        result = validate_candidate(candidate, papers=papers)
        assert isinstance(result, RejectedCandidate)
        assert any("training" in r for r in result.reasons)

    def test_a_script_as_example_behaviour_is_rejected(self, papers: dict[str, PaperText]) -> None:
        candidate = _candidate(
            example_behavior='"Can you tell me in your own words what we just discussed?"'
        )
        result = validate_candidate(candidate, papers=papers)
        assert isinstance(result, RejectedCandidate)
        assert any("script" in r for r in result.reasons)


class TestBatchBehaviour:
    def test_duplicate_spans_are_counted_once(self, papers: dict[str, PaperText]) -> None:
        report = validate_candidates([_candidate(), _candidate()], papers=papers)
        assert len(report.accepted) == 1
        assert "duplicate of an accepted entry" in report.reason_counts()

    def test_report_counts_themes(self, papers: dict[str, PaperText]) -> None:
        report = validate_candidates([_candidate()], papers=papers)
        assert report.theme_counts() == {"teach_back": 1}

    def test_all_failure_reasons_are_collected_not_just_the_first(
        self, papers: dict[str, PaperText]
    ) -> None:
        candidate = _candidate(
            theme="not a theme",
            verbatim_span="a sentence this paper never contained anywhere in its text",
            practical_takeaway="Clinicians should attend a communication training course.",
        )
        result = validate_candidate(candidate, papers=papers)
        assert isinstance(result, RejectedCandidate)
        assert len(result.reasons) >= 3


class TestEntryId:
    def test_id_encodes_the_theme_and_is_stable_across_whitespace(self) -> None:
        a = entry_id_for(Theme.EMPATHY, "p1", "the same  sentence")
        b = entry_id_for(Theme.EMPATHY, "p1", "the same\nsentence")
        assert a == b
        assert a.startswith("kb-empathy-")

    def test_different_papers_give_different_ids(self) -> None:
        a = entry_id_for(Theme.EMPATHY, "p1", "the same sentence")
        b = entry_id_for(Theme.EMPATHY, "p2", "the same sentence")
        assert a != b
