"""Synthetic raters, so the harness is proven working before a real rater exists.

Human rating happens in sprint 10; the harness is built now. The risk that
creates is specific and expensive: a blinding bug, an off-by-one in the label
join, or a reversed `ritualistic` column is discovered *after* two paid raters
have spent a weekend on sixty responses, and the only fix is to ask them to do
it again. Exercising every step against generated raters costs nothing and
removes that risk.

The generator is not a fixture that returns plausible numbers. It has a latent
truth per response, and each rater observes it through a controllable amount of
noise, so the tests can assert on the *direction* of the reliability
coefficients rather than merely that they compute:

- low noise must produce a high alpha; if it does not, the alpha implementation
  is wrong;
- high noise must produce an alpha near zero; if a "reliable" number comes out
  of raters who are guessing, the implementation is agreeing with itself
  somewhere it should not be;
- a rater with a constant offset must show low alpha and high rho, which is the
  signature the study needs to be able to tell apart from real disagreement;
- a rater who has `ritualistic` backwards must be caught by
  `calibration_check`, since that is the one error that produces entirely
  normal-looking data.

Everything is seeded. A test that fails on one run and passes on the next is
worse than no test.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from carelite.eval.human.blinding import BlindedPacket
from carelite.eval.rubric.calibration import CALIBRATION_SET
from carelite.eval.rubric.dimensions import SCALE_MAX, SCALE_MIN
from carelite.types import RUBRIC_DIMENSIONS

__all__ = [
    "synthetic_ratings",
    "synthetic_retest_ratings",
    "synthetic_truth",
]


def _clamp(value: int) -> int:
    return max(SCALE_MIN, min(SCALE_MAX, value))


def synthetic_truth(
    generation_ids: Sequence[str],
    *,
    seed: int = 1234,
) -> dict[str, dict[str, int]]:
    """A latent "true" score per generation per dimension.

    Deliberately spread across the whole 1-5 range. A truth concentrated on 3s
    would give every rater near-perfect agreement for free and would make the
    alpha tests pass regardless of whether alpha works — ordinal alpha divides
    by expected disagreement, and with no variance there is none.
    """
    rng = random.Random(seed)
    return {
        gid: {key: rng.randint(SCALE_MIN, SCALE_MAX) for key in RUBRIC_DIMENSIONS}
        for gid in generation_ids
    }


def synthetic_ratings(
    packet: BlindedPacket,
    truth: Mapping[str, Mapping[str, int]],
    *,
    noise: float = 0.4,
    bias: int = 0,
    seed: int = 7,
    reverse_ritualistic: bool = False,
    skip_rate: float = 0.0,
) -> list[dict[str, Any]]:
    """Generate one rater's returned rows, in the shape `ingest_ratings` expects.

    Rows are keyed by `blind_label` and contain nothing else identifying, which
    is the point: the synthetic rater is given exactly what a real rater is
    given, so a leak in the export would show up here as the generator having
    access to something it should not.

    Args:
        packet: The rater's blinded packet.
        truth: Latent scores per generation. Calibration items use the agreed
            consensus as their truth, so a well-behaved synthetic rater passes
            `calibration_check` and a misbehaving one fails it.
        noise: Probability of moving a score one point away from truth. 0.0 is a
            perfect rater; ~0.4 is a plausible good one; 1.5+ is a guesser.
        bias: Constant offset added to every score. Models systematic leniency
            or severity — the failure mode that shows as low alpha with high rho.
        seed: Reproducibility. Vary it per rater, not the noise.
        reverse_ritualistic: Score `ritualistic` the wrong way round. Models the
            one rater error that produces plausible-looking data, and the reason
            `calibration_check` reports a signed bias per dimension.
        skip_rate: Probability of leaving a cell blank, to exercise the
            missing-data path through ingestion and alpha.
    """
    rng = random.Random(seed)
    consensus = {c.item_id: c.consensus for c in CALIBRATION_SET}
    lookup = {a.blind_label: a for a in packet.assignments}

    rows: list[dict[str, Any]] = []
    for item in packet.items:
        assignment = lookup[item.blind_label]
        source: Mapping[str, int] | None = (
            consensus.get(assignment.generation_id)
            if assignment.is_calibration
            else truth.get(assignment.generation_id)
        )
        if source is None:
            continue

        row: dict[str, Any] = {"blind_label": item.blind_label}
        for key in RUBRIC_DIMENSIONS:
            if skip_rate and rng.random() < skip_rate:
                row[key] = ""
                continue
            value = source[key]
            if reverse_ritualistic and key == "ritualistic":
                value = SCALE_MIN + SCALE_MAX - value
            drift = 0
            remaining = noise
            while remaining > 0:
                if rng.random() < min(1.0, remaining):
                    drift += rng.choice((-1, 1))
                remaining -= 1.0
            row[key] = _clamp(value + drift + bias)
        row["safety_flags"] = ""
        row["notes"] = ""
        rows.append(row)
    return rows


def synthetic_retest_ratings(
    first_pass: Sequence[Mapping[str, Any]],
    retest_packet: BlindedPacket,
    original_packet: BlindedPacket,
    *,
    instability: float = 0.5,
    seed: int = 99,
) -> list[dict[str, Any]]:
    """A second occasion by the same rater, relabelled for the retest packet.

    Built from the first pass rather than from the latent truth, because that is
    what a retest actually measures: how close a person lands to *their own*
    earlier judgement, not to the truth. A rater who is consistently wrong
    should come out of this with a high intra-rater reliability, and the
    write-up needs that to be visible rather than hidden by a generator that
    quietly regenerates from truth each time.

    Labels differ between the two packets — the retest is reshuffled — so rows
    are re-keyed through the generation id.
    """
    rng = random.Random(seed)
    first_by_generation: dict[str, Mapping[str, Any]] = {}
    original_lookup = {a.blind_label: a.generation_id for a in original_packet.assignments}
    for row in first_pass:
        generation_id = original_lookup.get(str(row.get("blind_label", "")))
        if generation_id:
            first_by_generation[generation_id] = row

    retest_lookup = {a.blind_label: a for a in retest_packet.assignments}

    rows: list[dict[str, Any]] = []
    for item in retest_packet.items:
        assignment = retest_lookup[item.blind_label]
        earlier = first_by_generation.get(assignment.generation_id)
        if earlier is None:
            continue
        retest_row: dict[str, Any] = {"blind_label": item.blind_label}
        for key in RUBRIC_DIMENSIONS:
            raw = earlier.get(key)
            if raw is None or str(raw).strip() == "":
                retest_row[key] = ""
                continue
            value = int(raw)
            if rng.random() < instability:
                value += rng.choice((-1, 1))
            retest_row[key] = _clamp(value)
        retest_row["safety_flags"] = ""
        retest_row["notes"] = ""
        rows.append(retest_row)
    return rows
