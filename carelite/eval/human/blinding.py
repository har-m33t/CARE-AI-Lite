"""Blinded export for human rating. **Unblinding must be a join, never a guess.**

The protocol in build plan v3 §12 is short and every clause is load-bearing:
condition labels stripped, presentation order randomised per rater, rubric
distributed before rating, five calibration responses scored and discussed
first. This module implements the first two and the assignment record that makes
the rest recoverable.

The shape of the design is: the rater sees a `BlindItem`, which carries a
meaningless label and two blocks of text and *nothing else*. The mapping from
that label back to a condition lives only in `Assignment`, which goes to the
`rating_assignment` table. So there is exactly one way to learn what condition a
rating belongs to — join on `(rater_id, blind_label)` — and no way to infer it
from the export. That is what `assert_blinded` enforces, and it is checked
rather than trusted because the failure is silent: an export that leaks
condition through the ordering still produces sixty perfectly plausible ratings.

Three leaks this guards against, all of which are easy to ship by accident:

1. **The label.** `blind_label` is derived from the rater's shuffled position,
   never from the condition or the generation id.
2. **The payload.** `generation_id` never leaves this module. Generation ids in
   this study encode scenario and condition, so exporting one is exporting the
   answer.
3. **The order.** A packet built without shuffling presents A, then B, then C in
   blocks. A rater notices that by item ten. `assert_blinded` fails a packet
   whose conditions arrive in runs.

Randomisation is seeded from `settings.experiment.base_seed` and the rater id,
so an export is reproducible — a rater who loses their file gets the same one
back, with the same labels, and their partial ratings still join.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from carelite.config import get_settings
from carelite.eval.rubric.calibration import CALIBRATION_SCENARIO, CALIBRATION_SET
from carelite.types import Condition

__all__ = [
    "Assignment",
    "BlindItem",
    "BlindedPacket",
    "BlindingViolation",
    "RateableItem",
    "assert_blinded",
    "build_packet",
    "rater_seed",
    "unblind",
]

_TAG_RE = re.compile(r"[^A-Za-z0-9]+")


class BlindingViolation(RuntimeError):
    """The export could reveal a condition. A programming error, never caught."""


@dataclass(frozen=True, slots=True)
class RateableItem:
    """One response to be rated, with the metadata that must NOT be exported."""

    generation_id: str
    scenario_text: str
    response_text: str
    condition: Condition | str
    scenario_id: str = ""


@dataclass(frozen=True, slots=True)
class BlindItem:
    """What the rater actually receives. Deliberately impoverished.

    No generation id, no condition, no scenario id, no model name. If a field
    is added here, ask what a rater could infer from it before adding it.
    """

    blind_label: str
    display_order: int
    is_calibration: bool
    scenario_text: str
    response_text: str


@dataclass(frozen=True, slots=True)
class Assignment:
    """One row of `rating_assignment`: the key that unblinds one rating.

    Note the FK: `rating_assignment.generation_id` references `generation`, so
    calibration items — which are fixtures in `carelite.eval.rubric.calibration`,
    not generated responses — have no row to reference. Calibration assignments
    are therefore kept in the packet and not persisted by default; see
    `carelite.eval.human.store`.
    """

    rater_id: str
    generation_id: str
    display_order: int
    blind_label: str
    is_calibration: bool = False


@dataclass(frozen=True, slots=True)
class BlindedPacket:
    """Everything one rater gets, plus the private key to their randomisation."""

    rater_id: str
    seed: int
    items: tuple[BlindItem, ...]
    assignments: tuple[Assignment, ...]
    #: Kept for the audit in `assert_blinded`; never exported to the rater.
    _conditions: Mapping[str, str] = field(default_factory=dict, repr=False)

    @property
    def study_items(self) -> tuple[BlindItem, ...]:
        return tuple(i for i in self.items if not i.is_calibration)

    @property
    def calibration_items(self) -> tuple[BlindItem, ...]:
        return tuple(i for i in self.items if i.is_calibration)

    def label_to_generation(self) -> dict[str, str]:
        """The join key, for ingestion. Not part of the rater-facing export."""
        return {a.blind_label: a.generation_id for a in self.assignments}


def _tag(rater_id: str) -> str:
    """A short, filename-safe form of the rater id for use in labels."""
    cleaned = _TAG_RE.sub("", rater_id).upper()
    return cleaned[:8] or "RATER"


def rater_seed(rater_id: str) -> int:
    """Deterministic per-rater randomisation seed.

    blake2b rather than `hash()` for the same reason `carelite.config.seed_for`
    uses it: CPython randomises string hashing per process, so `hash()` would
    give a different presentation order every time the export was regenerated
    and the labels in a rater's half-finished spreadsheet would stop matching.
    """
    base = get_settings().experiment.base_seed
    digest = hashlib.blake2b(rater_id.encode("utf-8"), digest_size=8).digest()
    return (base + int.from_bytes(digest, "big")) % (2**31 - 1)


def build_packet(
    rater_id: str,
    items: Sequence[RateableItem],
    *,
    include_calibration: bool = True,
    check: bool = True,
) -> BlindedPacket:
    """Build one rater's blinded packet.

    Calibration items come first, in the fixed order `CALIBRATION_SET` defines —
    that order is pedagogical (CAL-01 and CAL-02 set the two axes raters most
    often conflate) and shuffling it would break the teaching sequence. They are
    labelled `…-C01`, are openly identified as calibration, and are rated and
    discussed before the study items are touched.

    Study items are shuffled with the rater's own seed, so two raters see the
    same sixty responses in different orders and any order effect is spread
    across conditions instead of aligned with them.

    Args:
        rater_id: Stable identifier. Also seeds the shuffle.
        items: The study responses. Order here is irrelevant; it is discarded.
        include_calibration: Off only for a test-retest second occasion, where
            the rater has already been calibrated.
        check: Run `assert_blinded` before returning. Leave on.

    Raises:
        BlindingViolation: if the packet could reveal a condition.
        ValueError: on duplicate generation ids in `items`.
    """
    seen = [i.generation_id for i in items]
    if len(set(seen)) != len(seen):
        raise ValueError("duplicate generation_id in items; each response is rated once per rater")

    tag = _tag(rater_id)
    blind_items: list[BlindItem] = []
    assignments: list[Assignment] = []
    order = 0

    if include_calibration:
        for idx, cal in enumerate(CALIBRATION_SET, start=1):
            order += 1
            label = f"{tag}-C{idx:02d}"
            blind_items.append(
                BlindItem(
                    blind_label=label,
                    display_order=order,
                    is_calibration=True,
                    scenario_text=CALIBRATION_SCENARIO,
                    response_text=cal.response,
                )
            )
            assignments.append(
                Assignment(
                    rater_id=rater_id,
                    generation_id=cal.item_id,
                    display_order=order,
                    blind_label=label,
                    is_calibration=True,
                )
            )

    shuffled = list(items)
    random.Random(rater_seed(rater_id)).shuffle(shuffled)

    for idx, item in enumerate(shuffled, start=1):
        order += 1
        label = f"{tag}-{idx:03d}"
        blind_items.append(
            BlindItem(
                blind_label=label,
                display_order=order,
                is_calibration=False,
                scenario_text=item.scenario_text,
                response_text=item.response_text,
            )
        )
        assignments.append(
            Assignment(
                rater_id=rater_id,
                generation_id=item.generation_id,
                display_order=order,
                blind_label=label,
                is_calibration=False,
            )
        )

    packet = BlindedPacket(
        rater_id=rater_id,
        seed=rater_seed(rater_id),
        items=tuple(blind_items),
        assignments=tuple(assignments),
        _conditions={i.generation_id: str(i.condition) for i in items},
    )
    if check:
        assert_blinded(packet, items)
    return packet


def assert_blinded(packet: BlindedPacket, items: Sequence[RateableItem]) -> None:
    """Fail loudly if the packet could tell a rater which condition they are reading.

    Three checks, in increasing subtlety. The third one — that conditions do not
    arrive in blocks — is the one that catches a packet built with the shuffle
    accidentally removed, which is otherwise invisible in a code review because
    every individual field looks correctly stripped.
    """
    by_id = {i.generation_id: i for i in items}

    for item in packet.items:
        haystack = f"{item.blind_label}\n{item.scenario_text}\n{item.response_text}"
        for generation_id in by_id:
            if generation_id and generation_id in haystack:
                raise BlindingViolation(
                    f"generation id {generation_id!r} appears in the rater-facing export"
                )
        for condition in Condition:
            token = f"condition {condition.value}".casefold()
            if token in item.blind_label.casefold():
                raise BlindingViolation(f"blind label {item.blind_label!r} names a condition")

    study = [a for a in packet.assignments if not a.is_calibration]
    conditions = [str(by_id[a.generation_id].condition) for a in study if a.generation_id in by_id]
    distinct = set(conditions)
    # Below four items per condition the run count is uninformative — three
    # items from three conditions give three runs however well they were
    # shuffled — so the check would fail correct packets. Real packets are
    # sixty items over three conditions.
    if len(distinct) > 1 and len(conditions) >= 4 * len(distinct):
        runs = 1 + sum(1 for a, b in pairwise(conditions) if a != b)
        # Perfectly blocked presentation gives exactly one run per condition.
        if runs <= len(distinct):
            raise BlindingViolation(
                f"conditions are presented in blocks ({runs} runs across {len(conditions)} "
                "items); the presentation order was not randomised"
            )


def unblind(
    assignments: Sequence[Assignment],
    ratings_by_label: Mapping[str, Mapping[str, int | None]],
) -> dict[str, Mapping[str, int | None]]:
    """Join returned ratings back to generation ids. The only unblinding path.

    Raises `KeyError` naming the offending label if a rating arrives under a
    label this rater was never assigned — which means either a typo in a
    returned spreadsheet or ratings from the wrong rater's packet, and both are
    worth stopping for.
    """
    lookup = {a.blind_label: a.generation_id for a in assignments}
    out: dict[str, Mapping[str, int | None]] = {}
    for label, scores in ratings_by_label.items():
        if label not in lookup:
            raise KeyError(f"rating returned under unknown blind label {label!r}")
        out[lookup[label]] = scores
    return out
