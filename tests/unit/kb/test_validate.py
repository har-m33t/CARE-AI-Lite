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
        meta=PaperMeta(
            paper_id=paper_id,
            design=design,
            short_citation=f"{paper_id} (test)",
            year=2020,
            apa_citation=f"Test, A. (2020). {paper_id}. Journal of Fixtures, 1(1), 1-2.",
        ),
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


class TestTierFromDesign:
    """Tier is derived from the recorded design, in both directions.

    The first version of this check treated the design tier as a *ceiling*: it
    lowered an overclaim and left an underclaim alone. That is half a check, and
    the half it omitted let the same defect back in through the other door —
    four papers ended up carrying entries at more than one tier and one carried
    entries at all three, because wherever the model happened to say `emerging`
    about a randomised trial, `emerging` survived. Tier then recorded the
    model's confidence rather than the study design, which is exactly the
    property the check exists to remove. `README.md` defines evidence strength
    as a property of the source, so `test_two_entries_from_one_paper_agree` is
    the invariant that matters here; the directional tests are how it is met.

    Correcting rather than rejecting is still right, for the reason it always
    was: unlike a fabricated span, an ill-judged tier has a derivable right
    answer, and the entry's other fields are untouched by the error. What must
    not happen is the correction being silent, so both values survive on the
    result and the digest prints them together.
    """

    def _protocol_candidate(self, tier: str) -> CandidateEntry:
        return _candidate(
            source_paper_ids=["protocol-paper"],
            evidence_tier=tier,
            theme="activation_sdm",
            verbatim_span=(
                "The primary outcome will be medication adherence at twelve months, "
                "measured by pill count and self-report at each scheduled study visit"
            ),
        )

    def test_a_strong_claim_off_a_protocol_is_corrected_down_not_rejected(
        self, papers: dict[str, PaperText]
    ) -> None:
        result = validate_candidate(self._protocol_candidate("strong"), papers=papers)

        assert isinstance(result, ValidatedEntry)
        assert result.entry.evidence_tier is EvidenceTier.EMERGING
        assert result.claimed_tier is EvidenceTier.STRONG
        assert result.design_tier is EvidenceTier.EMERGING
        assert result.tier_corrected is True
        assert result.tier_direction == "down"

    def test_an_underclaim_off_a_systematic_review_is_corrected_up(
        self, papers: dict[str, PaperText]
    ) -> None:
        """The half the ceiling version missed, and the reason for this rewrite."""
        result = validate_candidate(_candidate(evidence_tier="emerging"), papers=papers)

        assert isinstance(result, ValidatedEntry)
        assert result.entry.evidence_tier is EvidenceTier.STRONG
        assert result.claimed_tier is EvidenceTier.EMERGING
        assert result.tier_corrected is True
        assert result.tier_direction == "up"

    def test_a_cautious_moderate_claim_is_also_raised_to_the_design(
        self, papers: dict[str, PaperText]
    ) -> None:
        result = validate_candidate(_candidate(evidence_tier="moderate"), papers=papers)
        assert isinstance(result, ValidatedEntry)
        assert result.entry.evidence_tier is EvidenceTier.STRONG

    def test_a_correct_claim_is_left_alone(self, papers: dict[str, PaperText]) -> None:
        result = validate_candidate(_candidate(evidence_tier="strong"), papers=papers)
        assert isinstance(result, ValidatedEntry)
        assert result.tier_corrected is False
        assert result.tier_direction == "unchanged"

    def test_two_entries_from_one_paper_agree(self, papers: dict[str, PaperText]) -> None:
        """The invariant. One paper, one design, one evidence strength."""
        first = _candidate(evidence_tier="emerging")
        second = _candidate(
            evidence_tier="strong",
            verbatim_span=(
                "Any misunderstandings are then clarified by the health provider and "
                "understanding is checked again"
            ),
        )
        report = validate_candidates([first, second], papers=papers)

        assert len(report.accepted) == 2
        assert {e.entry.evidence_tier for e in report.accepted} == {EvidenceTier.STRONG}
        assert report.tier_consistency() == {"review-paper": {"strong"}}

    def test_the_report_counts_every_correction_with_its_direction(
        self, papers: dict[str, PaperText]
    ) -> None:
        report = validate_candidates(
            [self._protocol_candidate("strong"), _candidate(evidence_tier="emerging")],
            papers=papers,
        )
        assert len(report.tier_corrected) == 2
        assert report.tier_correction_counts() == {
            "strong -> emerging (down)": 1,
            "emerging -> strong (up)": 1,
        }


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


class TestRejectionReporting:
    """The rejection counts are a published number, so the buckets have to be right.

    `format_report`'s totals appear in the review digest and in the write-up as
    the measured fabrication rate. They are produced by matching a needle against
    each rejection reason in order, which is fragile in a specific way: a short
    generic needle earlier in the list absorbs every later reason containing it.
    `"span is"` matched both "span is 6 words … below the evidence floor" and
    "span is about a clinician-inward practice", so one bucket over-counted and
    another never appeared at all — a reporting bug that says the pipeline
    rejected something for a reason it did not.
    """

    def test_every_rejection_reason_maps_to_its_own_bucket(
        self, papers: dict[str, PaperText]
    ) -> None:
        short_span = _candidate(verbatim_span="Implementation was inconsistent across settings")
        inward = _candidate(
            source_paper_ids=["protocol-paper"],
            theme="emotion_response",
            verbatim_span=(
                "The primary outcome will be medication adherence at twelve months, "
                "measured by pill count and self-report at each scheduled study visit"
            ),
            practical_takeaway=(
                "Practise mindfulness so that you notice the patient's emotional changes."
            ),
        )
        report = validate_candidates([short_span, inward], papers=papers)
        buckets = report.reason_counts()

        assert buckets.get("span too short to be evidence") == 1
        assert buckets.get("out of scope: clinician-inward practice") == 1

    def test_the_bucket_counts_add_up_to_the_rejections(self, papers: dict[str, PaperText]) -> None:
        """The property the collision broke: nothing may be counted twice or lost."""
        candidates = [
            _candidate(verbatim_span="a sentence that is nowhere in either fixture paper at all"),
            _candidate(verbatim_span="Implementation was inconsistent across settings"),
            _candidate(theme="not_a_theme"),
            _candidate(practical_takeaway="Clinicians should be mindful of the patient's needs."),
        ]
        report = validate_candidates(candidates, papers=papers)
        assert sum(report.reason_counts().values()) == len(report.rejected)


class TestAspirationIsNotAction:
    """The shape the D3 equity re-extraction produced, and why it is rejected.

    `DECISIONS.md` D3 revised the extraction prompt so that where a passage
    reports a disparity, the takeaway must name the compensating *move* rather
    than the awareness — and warned that a model told to find compensating moves
    will find them whether or not the passage supports one. What the revised
    prompt actually produced was the original awareness statement with an active
    verb bolted on: "be mindful of the empathy gap" became "proactively work to
    bridge the empathy gap".

    That clears the verb whitelist, because `work`, `address` and `provide` are
    real verbs, and it clears the attitude filter, because it is not asking
    anyone to bear something in mind. It is still not a move: it names an
    outcome to bring about, and no observer could say whether a clinician did
    it in a given conversation.
    """

    @pytest.mark.parametrize(
        "takeaway",
        [
            "Clinicians should proactively work to bridge the empathy gap for patients from "
            "lower socioeconomic backgrounds.",
            "Clinicians should proactively identify and address potential barriers to care "
            "for patients from lower socioeconomic backgrounds to ensure equitable empathy.",
            "Clinicians should actively work to provide consistent, high-quality empathetic "
            "communication to all patients regardless of their background.",
        ],
    )
    def test_an_outcome_to_bring_about_is_not_actionable(self, takeaway: str) -> None:
        ok, why = takeaway_is_actionable(takeaway)
        assert ok is False
        assert why is not None
        assert "outcome to bring about" in why

    @pytest.mark.parametrize(
        "takeaway",
        [
            # The equity entry the review named as the model of a real
            # compensating move. It must survive: the point of the rule is to
            # separate this from the aspiration, not to empty the theme.
            "Clinicians should actively check their internal biases regarding a patient's "
            "ability to follow medical advice or their specific pain management needs.",
            "Ask the patient to explain the plan back in their own words, and re-explain "
            "differently when the answer is incomplete.",
            "Ask what actually gets in the way of taking the medication before assuming "
            "the patient has chosen not to.",
        ],
    )
    def test_a_named_move_still_passes(self, takeaway: str) -> None:
        ok, why = takeaway_is_actionable(takeaway)
        assert ok is True, why

    def test_the_rule_is_not_equity_specific(self, papers: dict[str, PaperText]) -> None:
        """An aspiration is unactionable wherever it appears, not only under `equity`."""
        candidate = _candidate(
            theme="teach_back",
            practical_takeaway=(
                "Clinicians should work towards a consistently high standard of "
                "comprehension checking in every consultation."
            ),
        )
        result = validate_candidate(candidate, papers=papers)
        assert isinstance(result, RejectedCandidate)
        assert any("outcome to bring about" in r for r in result.reasons)


class TestPromptVersionFilter:
    """An experimental prompt variant must not join the base by finishing.

    The extraction cache is append-only and the validator reads all of it, so
    the moment a new variant's first window lands its candidates are in the next
    load — before anyone has looked at whether the variant works. `DECISIONS.md`
    D3 approved the equity variant *with a guard*, and a variant's output should
    reach the knowledge base when the guard has been applied, not when the
    inference finishes.
    """

    def _cache(self, tmp_path: object) -> str:
        from pathlib import Path

        from carelite.kb.extract import WindowResult, append_cache

        path = Path(str(tmp_path)) / "extraction.jsonl"
        for version in ("kb-extract-v1", "kb-extract-equity-v1"):
            append_cache(
                WindowResult(
                    paper_id="review-paper",
                    window_index=0,
                    paper_sha256="x",
                    prompt_version=version,
                    model="test",
                    candidates=[_candidate(finding=f"finding from {version}")],
                ),
                path,
            )
        return str(path)

    def test_no_filter_reads_every_variant(self, tmp_path: object) -> None:
        from carelite.kb.validate import candidates_from_cache

        assert len(candidates_from_cache(self._cache(tmp_path))) == 2

    def test_a_filter_excludes_the_experimental_variant(self, tmp_path: object) -> None:
        from carelite.kb.validate import candidates_from_cache

        kept = candidates_from_cache(self._cache(tmp_path), prompt_versions=["kb-extract-v1"])
        assert len(kept) == 1
        assert "kb-extract-v1" in kept[0].finding
