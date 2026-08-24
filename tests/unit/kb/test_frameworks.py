"""Unit tests for carelite.kb.frameworks — the behavior-to-framework mapping.

Two hazards shape this file, and most of the tests exist for the second one.

The first is ordinary precision: a component has to be assigned on the act the
entry prescribes, matched against the rubric's own definition. Those are the
`test_*_is_assigned` cases.

The second is the one worth naming. A knowledge base where all nine components
have entries looks better than one where two have none, so every pressure on a
mapping like this runs toward widening patterns until the table fills up. Every
false positive below was found by reading the assignments this module made
against the entries it made them on, and each is pinned so the pattern that
produced it cannot come back. `test_respect_and_support_have_no_entries_in_this_corpus`
pins the outcome that matters most: two components are empty, that is a finding
about the evidence base, and a future change that quietly fills them should have
to argue with a failing test first.
"""

from __future__ import annotations

import pytest

from carelite.kb.frameworks import (
    FOUR_HABITS_COMPONENTS,
    NURSE_COMPONENTS,
    component_counts,
    map_components,
    map_entries,
    prescribed_act,
    unmapped_entries,
)


def _nurse(takeaway: str, behavior: str = "") -> tuple[str, ...]:
    return map_components(takeaway, behavior).nurse


def _habits(takeaway: str, behavior: str = "") -> tuple[str, ...]:
    return map_components(takeaway, behavior).four_habits


class TestTheMappingReadsTheActNotTheTopic:
    """The matched surface is what the clinician does, never what the study found."""

    def test_prescribed_act_excludes_finding_and_span(self) -> None:
        text = prescribed_act("Ask the patient to explain the plan back.", "Uses teach-back.")
        assert "explain the plan back" in text
        assert "teach-back" in text.lower()

    def test_an_empathy_study_can_prescribe_a_comprehension_check(self) -> None:
        """The case that makes topic-matching wrong.

        An entry drawn from an empathy trial whose prescribed act is a
        comprehension check instantiates Invest in the End, not Demonstrate
        Empathy. Matching the finding would get this exactly backwards.
        """
        mapping = map_components(
            "Ask the patient to state the plan back in their own words.",
            "Checking that the patient can repeat the dose change.",
        )
        assert "ie" in mapping.four_habits
        assert "de" not in mapping.four_habits

    def test_every_assignment_carries_the_text_that_produced_it(self) -> None:
        mapping = map_components("Use the teach-back method to confirm understanding.", "")
        assert "ie" in mapping.evidence
        assert "teach" in mapping.evidence["ie"].lower()


class TestNurse:
    def test_naming_an_emotion_is_assigned(self) -> None:
        assert "name" in _nurse("Acknowledge the patient's emotional state directly.")

    def test_distressing_information_is_not_an_emotion_being_named(self) -> None:
        """`distress\\w*` matched "distressing information", where the emotion word
        is an adjective on a noun and no emotion is being named at all."""
        assert "name" not in _nurse(
            "Identify specific pieces of distressing information the patient found online."
        )

    def test_validating_an_emotion_is_understanding(self) -> None:
        assert "understand" in _nurse("Validate the patient's worries and experiences.")

    def test_checking_patient_comprehension_is_not_understanding(self) -> None:
        """NURSE Understanding legitimises an emotion. The word "understand" in
        this corpus almost always means the patient's comprehension instead."""
        assert "understand" not in _nurse(
            "Confirm that the patient understands their discharge instructions."
        )

    def test_a_backchannel_affirmation_is_not_respecting(self) -> None:
        """The false positive that gave `respect` its only two entries.

        NURSE Respecting credits the patient for something they did or endured.
        "Verbal affirmations to show you are listening" is a backchannel cue —
        a different act that happens to share a word.
        """
        assert "respect" not in _nurse(
            "Use nonverbal cues and verbal affirmations to show the patient you are listening."
        )

    def test_crediting_the_patient_would_be_respecting(self) -> None:
        """The pattern still works; the corpus simply contains no entry like this."""
        assert "respect" in _nurse(
            "Acknowledge how hard the patient has worked to keep the appointments."
        )

    def test_a_collaborative_stance_alone_is_not_supporting(self) -> None:
        """The false positive that gave `support` all four of its entries.

        The rubric requires partnership *made concrete*: who does what, by when,
        and how the patient reaches someone. A stance is not that.
        """
        assert "support" not in _nurse(
            "Shift from a top-down authoritative style to a collaborative partnership."
        )

    def test_partnership_with_concrete_follow_through_is_supporting(self) -> None:
        assert "support" in _nurse(
            "Tell the patient how to reach the team if things change before the next visit."
        )

    def test_an_open_question_about_goals_is_not_exploring(self) -> None:
        """Exploring opens a door onto the *emotion*. An open question about
        goals or preferences is Elicit the Patient's Perspective."""
        mapping = map_components(
            "Use open-ended questions to invite the patient to share their perspective.", ""
        )
        assert "explore" not in mapping.nurse
        assert "epp" in mapping.four_habits

    def test_an_open_question_about_a_concern_is_exploring(self) -> None:
        assert "explore" in _nurse(
            "Ask an open-ended question about the concerns the patient has just raised."
        )


class TestFourHabits:
    def test_rapport_at_the_opening_is_invest_in_the_beginning(self) -> None:
        assert "ib" in _habits("Establish rapport by greeting the patient warmly.")

    def test_a_closing_check_is_not_invest_in_the_beginning(self) -> None:
        """The leak a negative lookahead could not close, because the closing
        marker sits *before* the match: "…before ending the interaction to
        ensure all concerns are addressed"."""
        mapping = map_components(
            "Ask the patient if there is anything else on their mind before ending the "
            "interaction to ensure all concerns are addressed in one go.",
            "",
        )
        assert "ib" not in mapping.four_habits
        assert "ie" in mapping.four_habits

    def test_eliciting_the_patients_own_model_is_epp(self) -> None:
        assert "epp" in _habits("Ask the patient what matters most to them.")

    def test_naming_the_patients_perspective_without_asking_is_not_epp(self) -> None:
        """The rubric's question is whether the response *asks*. Stating your own
        understanding of the patient's feelings is a NURSE move, not this one."""
        assert "epp" not in _habits(
            "Explicitly state your understanding of the patient's feelings."
        )

    def test_an_empathic_response_is_demonstrate_empathy(self) -> None:
        assert "de" in _habits("Use empathetic responses before moving to clinical detail.")

    def test_a_measured_empathy_outcome_is_not_demonstrate_empathy(self) -> None:
        """`de` is a holistic judgement about the response, not a topic tag.
        The word "empathy" describing what a study measured must not pull an
        entry in — and the takeaway is where that distinction shows."""
        assert "de" not in _habits("Arrange the clinic list so consultations are not rushed.")

    def test_teach_back_is_invest_in_the_end(self) -> None:
        assert "ie" in _habits("Use the teach-back method to confirm understanding.")

    def test_involving_the_patient_in_the_decision_is_invest_in_the_end(self) -> None:
        """On the rubric's own wording: "the patient involved in the decision,
        and a clear next step" is part of this habit."""
        assert "ie" in _habits("Involve patients in the decision about their treatment plan.")

    def test_checking_the_clinicians_own_understanding_is_not_invest_in_the_end(self) -> None:
        """The comprehension check has to be a check on the *patient's*
        understanding. "State your understanding of the patient's feelings" is
        the clinician's, and is a NURSE move."""
        assert "ie" not in _habits("Explicitly state your understanding of the patient's feelings.")


class TestCoverageGapsSurvive:
    """Zeros are results. They must be reportable, and they must not be tuned away."""

    def test_an_entry_instantiating_nothing_gets_empty_lists(self) -> None:
        mapping = map_components(
            "Request in-person interpreters whenever possible.",
            "Booking an in-person interpreter rather than a telephone line.",
        )
        assert mapping.is_empty
        assert mapping.nurse == ()
        assert mapping.four_habits == ()

    def test_component_counts_reports_zeros_rather_than_omitting_them(self) -> None:
        counts = component_counts(
            map_entries([("kb-x-1", "teach_back", "Use teach-back to confirm understanding.", "")])
        )
        assert set(counts) == set(NURSE_COMPONENTS + FOUR_HABITS_COMPONENTS)
        assert counts["ie"] == 1
        assert counts["respect"] == 0

    def test_unmapped_entries_are_surfaced_not_hidden(self) -> None:
        mapped = map_entries(
            [
                ("kb-x-1", "teach_back", "Use teach-back to confirm understanding.", ""),
                ("kb-x-2", "equity", "Request an in-person interpreter.", ""),
            ]
        )
        assert [e.entry_id for e in unmapped_entries(mapped)] == ["kb-x-2"]

    @pytest.mark.db
    def test_respect_and_support_have_no_entries_in_this_corpus(self) -> None:
        """The finding this whole module has to be trusted not to erase.

        Neither NURSE Respecting nor NURSE Supporting is prescribed anywhere in
        the 33 papers, so the judge scores two dimensions the knowledge base
        cannot ground. If a future change fills these, it should have to argue
        with this test rather than quietly improve a table.
        """
        from carelite.db.connection import transaction

        with transaction() as conn:
            rows = conn.execute("SELECT nurse_component, four_habits FROM kb_entry").fetchall()
        assigned = {c for r in rows for c in (r["nurse_component"] or [])}
        assigned |= {c for r in rows for c in (r["four_habits"] or [])}
        assert "respect" not in assigned
        assert "support" not in assigned

    @pytest.mark.db
    def test_every_loaded_entry_agrees_with_the_derivation(self) -> None:
        """The stored value must be re-derivable, which is what makes it checkable."""
        from carelite.db.connection import transaction

        with transaction() as conn:
            rows = conn.execute(
                "SELECT practical_takeaway, example_behavior, nurse_component, four_habits "
                "FROM kb_entry"
            ).fetchall()
        for row in rows:
            mapping = map_components(row["practical_takeaway"], row["example_behavior"])
            assert list(mapping.nurse) == list(row["nurse_component"] or [])
            assert list(mapping.four_habits) == list(row["four_habits"] or [])
