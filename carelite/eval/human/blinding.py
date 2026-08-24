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

An assignment names one of two kinds of target, in two separate columns:
`generation_id` for a study response and `calibration_id` for one of the five
calibration fixtures, with `is_calibration` agreeing with whichever is set. The
split exists because calibration items are not generations and must never be
joined as if they were; `Assignment` and the schema both enforce it. The
one-way property above is unaffected — a calibration item is openly labelled as
calibration to the rater, so there is nothing about it to blind.

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

    An assignment points at **exactly one** of two kinds of thing, and which one
    it is decides the column it lands in:

    - a study response, in `generation_id`, an FK to `generation`;
    - a calibration response, in `calibration_id`, which is deliberately *not* a
      foreign key — calibration items are fixtures in
      `carelite.eval.rubric.calibration`, with no scenario row, prompt version
      or model digest, and giving them `generation` rows to satisfy an FK would
      put five fabricated rows in the table every analysis query reads.

    `is_calibration` is not free-floating: it must agree with which of the two
    ids is set. The schema enforces all three rules
    (`rating_assignment_one_target`, `rating_assignment_calibration_flag_agrees`)
    and `__post_init__` enforces the same three here, so an assignment that
    could not be stored cannot be constructed either. That duplication is
    deliberate — the database check only fires where a database is running, and
    the harness is developed and tested mostly where one is not.

    Prefer `for_generation` and `for_calibration` over the raw constructor.
    """

    rater_id: str
    generation_id: str | None
    display_order: int
    blind_label: str
    is_calibration: bool = False
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        n_targets = sum(x is not None for x in (self.generation_id, self.calibration_id))
        if n_targets != 1:
            raise ValueError(
                f"assignment {self.blind_label!r} must name exactly one target, got "
                f"{n_targets} (generation_id={self.generation_id!r}, "
                f"calibration_id={self.calibration_id!r})"
            )
        if self.is_calibration != (self.calibration_id is not None):
            raise ValueError(
                f"assignment {self.blind_label!r} has is_calibration="
                f"{self.is_calibration} but calibration_id={self.calibration_id!r}; "
                "the flag and the target must agree"
            )

    @classmethod
    def for_generation(
        cls, rater_id: str, generation_id: str, display_order: int, blind_label: str
    ) -> Assignment:
        """A study item: goes to `generation_id`, `is_calibration` false."""
        return cls(
            rater_id=rater_id,
            generation_id=generation_id,
            display_order=display_order,
            blind_label=blind_label,
            is_calibration=False,
        )

    @classmethod
    def for_calibration(
        cls, rater_id: str, calibration_id: str, display_order: int, blind_label: str
    ) -> Assignment:
        """A calibration fixture: goes to `calibration_id`, `is_calibration` true."""
        return cls(
            rater_id=rater_id,
            generation_id=None,
            display_order=display_order,
            blind_label=blind_label,
            is_calibration=True,
            calibration_id=calibration_id,
        )

    @property
    def target_id(self) -> str:
        """Whichever id this assignment carries. Use when the kind does not matter.

        Use `generation_id` — and let it be `None` for calibration — wherever the
        kind *does* matter, which is anywhere the value is about to be treated
        as a result. A calibration id that reaches `rubric_score` or a
        Krippendorff unit list is a silent contamination, not an error.
        """
        target = self.generation_id if self.calibration_id is None else self.calibration_id
        assert target is not None  # guaranteed by __post_init__
        return target


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
        """The join key, for ingestion. Not part of the rater-facing export.

        **Study items only.** Calibration labels are absent rather than mapped
        to their `calibration_id`, because everything downstream of this mapping
        treats its values as generation ids: a `CAL-01` in here becomes a
        `rubric_score` row with no generation, and then a Krippendorff unit that
        every rater scored against a published answer key. That inflates alpha
        instead of failing, which is why it is excluded here rather than
        filtered by each caller. Use `label_to_calibration` for the other half.
        """
        return {
            a.blind_label: a.generation_id for a in self.assignments if a.generation_id is not None
        }

    def label_to_calibration(self) -> dict[str, str]:
        """The calibration half of the join. Feeds `ingest.calibration_check` only."""
        return {
            a.blind_label: a.calibration_id
            for a in self.assignments
            if a.calibration_id is not None
        }


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
                Assignment.for_calibration(
                    rater_id=rater_id,
                    calibration_id=cal.item_id,
                    display_order=order,
                    blind_label=label,
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
            Assignment.for_generation(
                rater_id=rater_id,
                generation_id=item.generation_id,
                display_order=order,
                blind_label=label,
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

    **Calibration ratings are dropped, not unblinded.** They have no generation
    to join to, and a calibration id sitting in a dict of generation ids becomes
    a fake unit in the alpha computation — every rater scored those five items
    against a consensus they were then shown, so including them raises agreement
    on exactly the responses the study is not about. Their labels are still
    *recognised*, so a genuinely unknown label keeps raising: the distinction
    this function has to preserve is "not a result" versus "not this rater's".
    `ingest.ingest_ratings` routes them to `IngestReport.calibration`, which is
    where a calibration rating is meant to end up.
    """
    known = {a.blind_label for a in assignments}
    lookup = {a.blind_label: a.generation_id for a in assignments if a.generation_id is not None}
    out: dict[str, Mapping[str, int | None]] = {}
    for label, scores in ratings_by_label.items():
        if label not in known:
            raise KeyError(f"rating returned under unknown blind label {label!r}")
        generation_id = lookup.get(label)
        if generation_id is None:  # a calibration label; see the docstring
            continue
        out[generation_id] = scores
    return out
