"""The 100-scenario evaluation bank: loading, validation, and the two splits.

`scenarios/bank.jsonl` is the canonical artifact. One JSON object per line, one
synthetic patient utterance per object. Everything in it was written by hand for
this study; **no line is a real patient utterance and none ever may be.**

Each record carries the eight fields of `carelite.types.Scenario` -- which are
exactly the columns of the `scenario` table -- plus three curation fields the
frozen schema does not model:

* ``equity_kind``   which documented disparity axis the scenario represents
                    (``ses`` / ``lep`` / ``racial_ethnic``), or ``None``. The
                    schema stores only the boolean ``equity_stratum``; the axis
                    is what a human reviewer needs in order to review it.
* ``hard_case``     tags for the difficulty the scenario is meant to create
                    (``blocking_bait``, ``false_comprehension``, ``prognosis``,
                    ``family_override``, ``misplaced_blame``, ...). Descriptive,
                    not a stratification factor.
* ``curator_note``  one sentence on what makes the turn hard to answer well.

Only the eight frozen fields reach Postgres. `to_scenario()` performs that
narrowing, so the curation metadata cannot leak into the analysis by accident.

**Split discipline.** `train_scenarios()` is the only set Sprint 9's prompt
optimisation may ever see. `holdout_scenarios()` is write-once: its content is
checksummed in `carelite.scenarios.freeze` and any edit fails a unit test. The
naming here is deliberate -- `for_optimisation()` and `for_final_evaluation()`
are aliases that say out loud which set a caller is asking for.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from carelite.config import REPO_ROOT
from carelite.types import EncounterPhase, Scenario, Split

__all__ = [
    "BANK_PATH",
    "CHALLENGE_TYPES",
    "EQUITY_KINDS",
    "EXPECTED_HOLDOUT",
    "EXPECTED_TOTAL",
    "EXPECTED_TRAIN",
    "LITERACY_SIGNALS",
    "BankError",
    "CuratedScenario",
    "by_id",
    "equity_scenarios",
    "for_final_evaluation",
    "for_optimisation",
    "holdout_scenarios",
    "load_bank",
    "scenarios",
    "train_scenarios",
]

BANK_PATH: Path = REPO_ROOT / "scenarios" / "bank.jsonl"

#: Ten communication challenges, ten scenarios each. `challenge_type` is a free
#: TEXT column in the frozen schema, so this tuple is the controlled vocabulary.
CHALLENGE_TYPES: tuple[str, ...] = (
    "emotional_cue",
    "prognosis_request",
    "jargon_question",
    "false_comprehension",
    "family_override",
    "misplaced_blame",
    "decision_conflict",
    "adherence_barrier",
    "trust_rupture",
    "information_overload",
)

#: `unmarked` is the plurality on purpose: a literacy signal is a *marked*
#: feature of an utterance, and a bank in which every patient displays one would
#: not resemble a clinic.
LITERACY_SIGNALS: tuple[str, ...] = (
    "unmarked",
    "low_health_literacy",
    "numeracy_gap",
    "high_health_fluency",
)

#: The three disparities the corpus documents (README theme 7): the SES empathy
#: gap, emotional blocking of minority patients, and lower-quality conversations
#: with limited-English-proficiency patients.
EQUITY_KINDS: tuple[str, ...] = ("ses", "lep", "racial_ethnic")

EXPECTED_TOTAL = 100
EXPECTED_TRAIN = 40
EXPECTED_HOLDOUT = 60


class BankError(RuntimeError):
    """The bank file is missing, malformed, or violates a structural invariant."""


class CuratedScenario(BaseModel):
    """One bank record: the frozen `Scenario` fields plus curation metadata.

    `extra="forbid"` matters. A typo in a field name would otherwise be silently
    dropped and the scenario would quietly lose its stratum assignment.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    text: str = Field(min_length=10)
    challenge_type: str
    emotion_intensity: int = Field(ge=1, le=5)
    encounter_phase: EncounterPhase
    literacy_signal: str
    equity_stratum: bool
    split: Split
    equity_kind: str | None = None
    hard_case: list[str] = Field(default_factory=list)
    curator_note: str = ""

    @field_validator("challenge_type")
    @classmethod
    def _known_challenge(cls, v: str) -> str:
        if v not in CHALLENGE_TYPES:
            raise ValueError(f"unknown challenge_type {v!r}; expected one of {CHALLENGE_TYPES}")
        return v

    @field_validator("literacy_signal")
    @classmethod
    def _known_literacy(cls, v: str) -> str:
        if v not in LITERACY_SIGNALS:
            raise ValueError(f"unknown literacy_signal {v!r}; expected one of {LITERACY_SIGNALS}")
        return v

    @field_validator("equity_kind")
    @classmethod
    def _known_equity_kind(cls, v: str | None) -> str | None:
        if v is not None and v not in EQUITY_KINDS:
            raise ValueError(f"unknown equity_kind {v!r}; expected one of {EQUITY_KINDS}")
        return v

    def model_post_init(self, __context: object) -> None:
        # The boolean is what the schema stores and what the analysis groups by;
        # the axis is what a reviewer reads. They must not drift apart.
        if self.equity_stratum != (self.equity_kind is not None):
            raise ValueError(
                f"{self.scenario_id}: equity_stratum={self.equity_stratum} but "
                f"equity_kind={self.equity_kind!r}; the two must agree"
            )

    def to_scenario(self) -> Scenario:
        """Narrow to the frozen contract. Curation metadata is dropped here."""
        return Scenario(
            scenario_id=self.scenario_id,
            text=self.text,
            challenge_type=self.challenge_type,
            emotion_intensity=self.emotion_intensity,
            encounter_phase=self.encounter_phase,
            literacy_signal=self.literacy_signal,
            equity_stratum=self.equity_stratum,
            split=self.split,
        )


@lru_cache(maxsize=1)
def load_bank(path: str | None = None) -> tuple[CuratedScenario, ...]:
    """Parse and validate `scenarios/bank.jsonl`.

    Raises `BankError` on anything structurally wrong: a bad line, a duplicate
    id, or a count that is not 100 / 40 / 60. Stratum *coverage* is a separate
    and stricter question -- see `carelite.scenarios.audit`.
    """
    src = Path(path) if path else BANK_PATH
    if not src.exists():
        raise BankError(f"scenario bank not found at {src}")

    records: list[CuratedScenario] = []
    for lineno, raw in enumerate(src.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BankError(f"{src}:{lineno}: not valid JSON: {exc}") from exc
        try:
            records.append(CuratedScenario.model_validate(payload))
        except Exception as exc:
            raise BankError(f"{src}:{lineno}: invalid scenario record: {exc}") from exc

    ids = [r.scenario_id for r in records]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise BankError(f"{src}: duplicate scenario_ids: {dupes}")

    if len(records) != EXPECTED_TOTAL:
        raise BankError(f"{src}: expected {EXPECTED_TOTAL} scenarios, found {len(records)}")

    n_train = sum(1 for r in records if r.split is Split.TRAIN)
    n_holdout = len(records) - n_train
    if (n_train, n_holdout) != (EXPECTED_TRAIN, EXPECTED_HOLDOUT):
        raise BankError(
            f"{src}: split must be {EXPECTED_TRAIN} train / {EXPECTED_HOLDOUT} holdout, "
            f"found {n_train} / {n_holdout}"
        )

    return tuple(sorted(records, key=lambda r: r.scenario_id))


def scenarios() -> tuple[Scenario, ...]:
    """All 100, narrowed to the frozen contract."""
    return tuple(r.to_scenario() for r in load_bank())


def train_scenarios() -> tuple[CuratedScenario, ...]:
    """The 40 training scenarios. The only set prompt optimisation may see."""
    return tuple(r for r in load_bank() if r.split is Split.TRAIN)


def holdout_scenarios() -> tuple[CuratedScenario, ...]:
    """The 60 held-out scenarios. WRITE-ONCE -- see `carelite.scenarios.freeze`.

    Reading these during Sprint 9 (prompt optimisation) invalidates the held-out
    result. `carelite.scenarios.leakage` exists to catch it when it happens
    anyway.
    """
    return tuple(r for r in load_bank() if r.split is Split.HOLDOUT)


#: Aliases that make a caller's intent legible at the call site.
for_optimisation = train_scenarios
for_final_evaluation = holdout_scenarios


def equity_scenarios() -> tuple[CuratedScenario, ...]:
    """The pre-specified equity subgroup (build plan v3 §14, secondary analysis)."""
    return tuple(r for r in load_bank() if r.equity_stratum)


def by_id(scenario_id: str) -> CuratedScenario:
    for record in load_bank():
        if record.scenario_id == scenario_id:
            return record
    raise KeyError(scenario_id)
