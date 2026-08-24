"""Unit tests for carelite.kb.scope — the checks that survive a real span.

Every span quoted below is a real one from the loaded knowledge base, and each
was a defect a content review found after automated validation had passed the
entry. That is the point of the file: these are the failures that look fine to a
provenance check, because the quote is genuinely in the paper. The paper is
simply saying something other than what the entry claims, or saying it about
somebody else's study.

The false-positive tests matter as much as the true-positive ones and are kept
next to them deliberately. A filter tuned only on the entries it should catch is
a filter with an unmeasured cost, and the first drafts of three of these rules
were rejecting good entries at a rate nobody would have noticed from the
defect list alone.
"""

from __future__ import annotations

import pytest

from carelite.kb.scope import (
    LOW_OVERLAP_THRESHOLD,
    clinician_inward,
    out_of_scope,
    second_hand_ceiling,
    second_hand_evidence,
    takeaway_span_overlap,
    training_transfer,
    unevidenced_comparison,
    unevidenced_direction,
)
from carelite.types import EvidenceTier


class TestTrainingTransfer:
    """TAXONOMY.md §1: a course improving trainees' scores is not a bedside entry."""

    @pytest.mark.parametrize(
        ("span", "why"),
        [
            (
                "Following the course, scores on all three domains of burnout (emotional "
                "exhaustion, depersonalization, and personal achievement) and empathy "
                "improved significantly.",
                "the outcome is the clinician's own burnout and empathy scores",
            ),
            (
                "the involvement of expert patients in educational activities increased "
                "empathy scores in the EG on both scales (BEES and JSE-HPS)",
                "an educational activity moving an instrument score",
            ),
            (
                "Programmes that improved empathy provided personalised feedback compared "
                "to no or general feedback and used simulated patients in role play "
                "instead of peer role-play.",
                "programme design, not a communicative act",
            ),
            (
                "Trainees' scores improved in 8 of 11 of the SPIKES and NURSE-based coded "
                "behaviors",
                "coded behaviours scored on trainees",
            ),
            (
                "The training had a positive impact on reported use of patient-centered CS "
                "among both physicians (greatest gains in SDM and patient education skills) "
                "and patients",
                "reported use of a skill after training, not an effect on a patient",
            ),
        ],
    )
    def test_training_transfer_spans_are_out_of_scope(self, span: str, why: str) -> None:
        assert training_transfer(span) is not None, why

    def test_a_span_whose_learner_cohort_is_only_a_pronoun_needs_its_finding(self) -> None:
        """`They demonstrated significant improvement in several empathic skills`.

        Read alone that sentence names no learner and no course — `They` is
        doing the work, and the antecedent is a paragraph away. The entry's
        finding supplies it, which is the same reason the finding is searched
        for context at all.
        """
        span = (
            "They demonstrated significant improvement in several empathic skills, and "
            "in clarifying skill"
        )
        assert (
            training_transfer(
                span,
                "Nurses demonstrated significant improvement in several empathic skills "
                "after a training program involving didactic teaching and role-playing.",
            )
            is not None
        )

    def test_the_finding_carries_the_training_context_when_the_span_does_not(self) -> None:
        """`Specifically, the domains of … achieved statistical significance` alone is neutral.

        The entry built on it is not: its own finding says intervention
        physicians scored higher than controls. An entry is a training-transfer
        entry if its *claim* is one, however neutrally the sentence it quotes
        happens to read.
        """
        span = (
            "Specifically, the domains of 'Conveyed clear information' and 'Know patient's "
            "medical history,' achieved statistical significance."
        )
        assert training_transfer(span) is None
        assert (
            training_transfer(
                span,
                "Intervention physicians scored significantly higher in the domain of "
                "'Conveyed clear information' compared to control physicians.",
            )
            is not None
        )

    def test_a_self_rated_skill_survey_is_out_of_scope(self) -> None:
        span = (
            "The lowest-rated CS were: allowing the patient to share their narrative "
            "thread, summarizing the patient's history from the provider, and assessing "
            "patient understanding"
        )
        found = training_transfer(span)
        assert found is not None
        assert found.code == "self_rated_skill"

    @pytest.mark.parametrize(
        "span",
        [
            # A training study reporting a *patient* outcome is exactly the link
            # §1 asks for, and belongs in the knowledge base.
            "Patients whose physicians completed the course reported higher satisfaction "
            "with the consultation than those whose physicians did not.",
            # No training anywhere; ordinary bedside prose that mentions skills.
            "Patient-centered communication requires physicians and other health care "
            "providers to have the communication skills to elicit patients' true wishes "
            "and to recognize and respond to both their needs and emotional concerns.",
            # A finding about what empathy does in the room.
            "The empathetic relationship of the health professionals with their health "
            "care users reinforces their cooperation towards designing a therapeutic plan.",
        ],
    )
    def test_bedside_findings_are_not_flagged(self, span: str) -> None:
        assert training_transfer(span) is None


class TestClinicianInward:
    """TAXONOMY.md §2: regulating the clinician's inner state, with no patient in it."""

    def test_a_mindfulness_outcome_with_no_patient_facing_act_is_out_of_scope(self) -> None:
        span = (
            "The evidence suggests that over 90% of the studies reported a significant "
            "positive effect of brief mindfulness-based interventions on at least one "
            "health-related outcome"
        )
        takeaway = (
            "Clinicians should practice mindfulness to better notice and respond to "
            "emotional changes in patients during the encounter."
        )
        found = clinician_inward(span, takeaway)
        assert found is not None
        assert "mindfulness" in found.detail

    def test_an_inward_practice_tied_to_a_communicative_act_is_kept(self) -> None:
        span = (
            "Clinicians who paused to steady themselves before the conversation were more "
            "likely to acknowledge the emotion the patient had just expressed."
        )
        assert clinician_inward(span, "Pause before responding to a distressed patient.") is None


class TestUnevidencedComparison:
    """A finding that claims a result its span never reports."""

    def test_a_methods_sentence_cannot_support_a_comparative_result(self) -> None:
        finding = (
            "A shared decision-making process, where clinicians and patients negotiate a "
            "treatment regimen, resulted in significantly better controller and "
            "long-acting beta-agonist adherence compared to usual care or clinician-only "
            "decision making."
        )
        span = (
            "In shared decision making (SDM), nonphysician clinicians and patients "
            "negotiated a treatment regimen that accommodated patient goals and "
            "preferences."
        )
        assert unevidenced_comparison(finding, span) is not None

    def test_a_span_that_does_report_the_result_is_kept(self) -> None:
        finding = (
            "Negotiating treatment decisions significantly improves adherence compared "
            "with usual care."
        )
        span = (
            "Negotiating patients' treatment decisions significantly improves adherence to "
            "asthma pharmacotherapy and clinical outcomes."
        )
        assert unevidenced_comparison(finding, span) is None

    def test_a_qualitative_causal_paraphrase_is_not_a_mismatch(self) -> None:
        """The rule that cost three good entries before the veto covered causation.

        A finding saying jargon "can lead to confusion" over a span saying
        translated words are "often leading to patient confusion" is a
        paraphrase. Only a *comparative* claim — A beat B — needs the span to
        report an outcome.
        """
        finding = (
            "Verbatim translation of complex clinical terms can lead to patient confusion "
            "and misunderstandings because the chosen words may not convey the intended "
            "meaning when translated."
        )
        span = (
            "Interpreters in our evaluation described that often in serious illness "
            "communication, the words that clinicians choose do not convey the same "
            "meaning when translated verbatim, often leading to patient confusion and "
            "misunderstandings."
        )
        assert unevidenced_comparison(finding, span) is None


class TestUnevidencedDirection:
    """A decrease read out of a character the PDF extractor could not map."""

    SPAN = (
        "Patient understanding of their disease. # 12% in re-admission rates for heart "
        "failure patients 1 years post-TB implementation."
    )

    def test_a_direction_carried_only_by_a_mangled_glyph_is_rejected(self) -> None:
        finding = (
            "Implementing teach-back was associated with improved patient understanding "
            "and lower hospital readmission rates."
        )
        found = unevidenced_direction(finding, "Implement teach-back.", self.SPAN)
        assert found is not None
        assert "mangled glyph" in found.detail

    def test_the_same_span_with_the_arrow_intact_is_fine(self) -> None:
        """Whichever way the corpus lane's extraction fix lands, the check holds.

        If the arrow survives extraction the direction is in the span and the
        entry passes; if it does not, the entry fails. Either outcome is honest,
        which is why the rule is written on the glyph rather than on the entry.
        """
        span = self.SPAN.replace("#", "↓")
        assert (
            unevidenced_direction("…lower readmission rates.", "Implement teach-back.", span)
            is None
        )

    def test_two_numbers_and_a_comparison_are_not_a_mangled_glyph(self) -> None:
        span = (
            "Retention of discharge instructions in the TB group compared to the control "
            "group (recall rate 82.1% vs 70.0%; p<0.05)."
        )
        finding = "Teach-back was associated with significantly higher recall of instructions."
        assert unevidenced_direction(finding, "Use teach-back.", span) is None


class TestSecondHandEvidence:
    """Whether the span is reporting a study that is not the one it comes from."""

    @pytest.mark.parametrize(
        "span",
        [
            "A second RCT reported that teach-back significantly improved comprehension of "
            "post-ED care among patients with limited health literacy",
            "One study in children with asthma found that teach-back was associated with "
            "increased patient-centered communication (OR = 4.97) [40].",
            "Studies have shown that less than half the information provided about "
            "medication and diet is accurately recalled by patients [15, 16].",
            "A frequently observed barrier to patient understanding is the continued use "
            "of medical terminology by doctors [12-14].",
            "Motivational interviewing is instrumental in fostering the adoption of "
            "healthy behaviors among hypertensive patients (18). It also aids in the "
            "effective management of blood pressure (19).",
            "Research highlights that BD patients tend to prefer an active collaborative "
            "decision-making role [15].",
        ],
    )
    def test_relayed_evidence_is_detected(self, span: str) -> None:
        assert second_hand_evidence(span) is not None

    @pytest.mark.parametrize(
        ("span", "why"),
        [
            (
                "The framework includes 5 counseling steps: (1) assess the risk behavior, "
                "(2) advise change, (3) agree on goals and an action plan.",
                "an enumeration, not a citation list",
            ),
            (
                "Dissatisfaction with the self-introduction was zero, compared to 66% (289) "
                "of patients who reported being highly satisfied in this area.",
                "a count after a percentage, not a reference number",
            ),
            (
                "Teach-back involves asking patients to explain in their own words what a "
                "health provider has just told them.",
                "the paper's own prose",
            ),
        ],
    )
    def test_first_hand_prose_is_not_flagged(self, span: str, why: str) -> None:
        assert second_hand_evidence(span) is None, why

    def test_a_synthesis_paper_may_still_carry_moderate(self) -> None:
        assert second_hand_ceiling("systematic review") is EvidenceTier.MODERATE
        assert second_hand_ceiling("meta-analysis") is EvidenceTier.MODERATE

    def test_anything_else_relaying_a_study_drops_to_emerging(self) -> None:
        """An RCT's introduction citing other work is narrative review, whatever the RCT is."""
        assert second_hand_ceiling("randomized controlled trial") is EvidenceTier.EMERGING
        assert second_hand_ceiling("cross-sectional survey") is EvidenceTier.EMERGING
        assert second_hand_ceiling(None) is EvidenceTier.EMERGING


class TestTakeawaySpanOverlap:
    """Reported to a reader, never enforced. The test is that it stays a report."""

    def test_a_drifting_takeaway_scores_low(self) -> None:
        overlap = takeaway_span_overlap(
            "Clinicians should actively listen to and incorporate the patient's personal "
            "narrative of their illness to better understand their perspective.",
            "the involvement of expert patients in educational activities increased "
            "empathy scores in the EG on both scales (BEES and JSE-HPS).",
        )
        assert overlap < LOW_OVERLAP_THRESHOLD

    def test_a_faithful_takeaway_scores_high(self) -> None:
        overlap = takeaway_span_overlap(
            "Ask the patient to explain the plan back in their own words.",
            "Teach-back involves asking patients to explain in their own words what a "
            "health provider has just told them.",
        )
        assert overlap > LOW_OVERLAP_THRESHOLD * 3

    def test_a_good_paraphrase_can_also_score_low(self) -> None:
        """Why this must not become a gate.

        Both of these are real, and the entry is a good one: the takeaway is the
        advice the span supports, worded as an instruction rather than as a
        finding. Rejecting on overlap would throw it away.
        """
        overlap = takeaway_span_overlap(
            "When a patient expresses an emotion through their tone, acknowledge the "
            "emotion before moving to the logical answer.",
            "These responses need to be addressed with empathetic responses instead of "
            "rational answers.",
        )
        assert overlap < LOW_OVERLAP_THRESHOLD

    def test_an_empty_takeaway_does_not_divide_by_zero(self) -> None:
        assert takeaway_span_overlap("", "any span at all") == 0.0


class TestOutOfScope:
    def test_returns_the_first_failure_and_none_for_a_good_entry(self) -> None:
        assert (
            out_of_scope(
                "Teach-back involves asking patients to explain in their own words what a "
                "health provider has just told them.",
                "Teach-back is asking a patient to state understanding back.",
                "Ask the patient to explain the plan back in their own words.",
            )
            is None
        )
        found = out_of_scope(
            "Trainees' scores improved in 8 of 11 of the SPIKES and NURSE-based coded behaviors",
            "The SPIKES protocol improved clinician competence.",
            "Use the SPIKES framework when delivering difficult news.",
        )
        assert found is not None
        assert found.code == "training_transfer"
