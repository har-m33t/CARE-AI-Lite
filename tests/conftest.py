"""Shared fixtures. FROZEN — foundation lane only.

Lanes add their own fixtures in `tests/unit/<lane>/conftest.py`.
"""

from __future__ import annotations

import pytest

from carelite.types import (
    ActionType,
    Condition,
    EncounterPhase,
    EvidenceTier,
    GuidanceRequest,
    KBEntry,
    Scenario,
    Split,
    Theme,
)


@pytest.fixture
def kb_entry() -> KBEntry:
    return KBEntry(
        entry_id="kb-0001",
        theme=Theme.TEACH_BACK,
        finding="Teach-back improved comprehension across health literacy levels.",
        practical_takeaway="Ask the patient to restate the plan in their own words.",
        example_behavior="'I want to make sure I explained that well — how would you describe it?'",
        evidence_tier=EvidenceTier.STRONG,
        action_type=ActionType.GENERATION,
        verbatim_span="patients receiving teach-back demonstrated significantly higher recall",
        source_paper_ids=["paper-0001"],
        encounter_phase=[EncounterPhase.EXPLANATION],
        nurse_component=["understand"],
    )


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(
        scenario_id="sc-0001",
        text="I don't understand why I need another test. Nobody explains anything to me.",
        challenge_type="frustration_with_care",
        emotion_intensity=4,
        encounter_phase=EncounterPhase.EXPLANATION,
        literacy_signal="low",
        equity_stratum=True,
        split=Split.HOLDOUT,
    )


@pytest.fixture
def request_c() -> GuidanceRequest:
    return GuidanceRequest(
        utterance="I'm scared this is cancer.",
        condition=Condition.C,
        encounter_phase=EncounterPhase.EXPLANATION,
        seed=1,
    )
