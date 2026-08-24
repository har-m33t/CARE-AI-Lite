"""Drive the whole human-rating harness against synthetic raters, end to end.

    python -m carelite.eval.human.dry_run

The unit tests already assert each step. This is the other kind of evidence: one
pass through the real pipeline in the order a real rating round runs it —
packets built, blinding asserted, sheets returned, ingested, unblinded, alpha
computed, consensus formed, judge compared, retest measured — on the study's
actual sixty responses rather than on fixtures. A harness whose parts each pass
a unit test can still be wrong at the joins, and the joins are where the
expensive failure lives: a blinding leak or a reversed column is discovered
*after* two paid raters have spent a weekend on sixty responses.

**What is real here and what is not.** The responses, the packets, the blinding,
the ingestion, the coefficients and the thresholds are the production code
paths. The *ratings* are generated. So every number this module prints is a
statement about the instrument, never about the judge: `judge_human_validity`
run against synthetic raters measures whether the function computes agreement,
not whether the judge agrees with anyone. Human rating is sprint 10 and until
then the §13 validity question has no answer, which is what
`carelite.eval.judge.study --stage report` records.

Two truth models, and the contrast between them is the point:

* `independent` — the raters share a latent truth drawn at random, unrelated
  to anything the judge did. They therefore agree with *each other* (a real
  inter-rater alpha, which is what exercises the harness) while agreeing with
  the *judge* only by chance. So `judge_human_validity` must land near zero and
  every dimension must come back `exploratory`. This is the null control: if a
  dimension ever clears the threshold here, the threshold is broken.
* `judge_anchored` — the raters observe the judge's own scores through noise,
  so judge-human agreement must be high and dimensions must clear the
  threshold. This is the positive control, and it is what makes the null
  control mean anything: a `classify_dimension` hardwired to return
  `exploratory` would pass the null control perfectly.

Neither is evidence about the judge. Both are evidence that the instrument
which will one day measure the judge responds to signal and not to noise.

**The calibration-contamination check is the reason this module exists.** The
five calibration items are scored by every rater against a consensus the raters
are then shown, so they arrive as near-unanimous units. Routed into the study
unit list they do not fail — they *raise* Krippendorff's alpha, and a harness
defect that inflates the headline reliability number is the one defect that
never announces itself. `da38cd1` fixed it by keeping calibration in
`calibration_id`; `contamination_check` re-derives the alpha the old behaviour
would have produced and prints both, so the fix is demonstrated on live numbers
rather than asserted in a test that could be deleted.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carelite.eval.human.blinding import (
    BlindedPacket,
    RateableItem,
    assert_blinded,
    build_packet,
)
from carelite.eval.human.ingest import CalibrationCheck, calibration_check, ingest_ratings
from carelite.eval.human.reliability import (
    human_consensus,
    inter_rater_alpha,
    intra_rater_reliability,
)
from carelite.eval.human.synthetic import (
    synthetic_ratings,
    synthetic_retest_ratings,
    synthetic_truth,
)
from carelite.eval.rubric.calibration import CALIBRATION_SET
from carelite.types import RUBRIC_DIMENSIONS, RaterType, RubricScore

__all__ = [
    "STANDARD_PANEL",
    "DryRunResult",
    "RaterProfile",
    "TruthModel",
    "contamination_check",
    "dry_run",
    "main",
]


@dataclass(frozen=True, slots=True)
class RaterProfile:
    """One synthetic rater: how noisy, how biased, and whether they are broken."""

    rater_id: str
    noise: float
    bias: int = 0
    seed: int = 7
    reverse_ritualistic: bool = False
    skip_rate: float = 0.0


#: Two competent raters plus one deliberately broken one. The broken rater is
#: not scored into the panel — they exist so `calibration_check` has something
#: to catch, and catching the reversed `ritualistic` before a rating round is
#: the single highest-value thing the calibration step does.
STANDARD_PANEL: tuple[RaterProfile, ...] = (
    RaterProfile("R01", noise=0.4, seed=11),
    RaterProfile("R02", noise=0.5, bias=0, seed=23, skip_rate=0.03),
)

REVERSED_RATER = RaterProfile("R09-broken", noise=0.3, seed=31, reverse_ritualistic=True)


class TruthModel:
    """Where the synthetic raters' latent truth comes from."""

    INDEPENDENT = "independent"
    JUDGE_ANCHORED = "judge_anchored"


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------


def anchored_truth(
    judge_scores: Mapping[str, Mapping[str, int | None]],
    *,
    seed: int = 4242,
) -> tuple[dict[str, dict[str, int]], int]:
    """Latent truth taken from the judge's raw scores, filled where it has none.

    Returns `(truth, n_filled)`. A dimension the judge rejected for want of a
    locatable span has no anchor, so it falls back to the independent random
    draw and is counted. That is also what happens with real raters — a human
    has an opinion on a dimension the judge declined to score — and silently
    dropping those cells would make the positive control easier than the thing
    it is standing in for.

    Raw scale throughout, matching what a rater writes on the sheet and what
    `rubric_score` stores. `ritualistic` stays higher-is-worse here; every
    comparison downstream canonicalises through `to_quality` exactly once.
    """
    fallback = synthetic_truth(sorted(judge_scores), seed=seed)
    truth: dict[str, dict[str, int]] = {}
    n_filled = 0
    for gid, scores in judge_scores.items():
        row: dict[str, int] = {}
        for key in RUBRIC_DIMENSIONS:
            value = scores.get(key)
            if value is None:
                value = fallback[gid][key]
                n_filled += 1
            row[key] = int(value)
        truth[gid] = row
    return truth, n_filled


# ---------------------------------------------------------------------------
# Contamination
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContaminationCheck:
    """What the calibration items would have done to alpha had they leaked in.

    Three arms over the same `inter_rater_alpha`, differing only in the unit list:

    * `clean` — what the harness actually reports. Study generations only.
    * `leaked_observed` — the five calibration ratings re-keyed as generations,
      exactly as `unblind()` did before `da38cd1`, using the ratings the
      synthetic raters actually produced.
    * `leaked_converged` — the same leak, but with every rater scoring the
      calibration items at the agreed consensus.

    The third arm is the one that matters, and the second exists to show why.
    The synthetic raters apply the same noise to a calibration item as to a
    study item, so the observed leak adds five units of ordinary character and
    barely moves the coefficient — which would make the defect look harmless.
    Real raters score the calibration set, are *shown the consensus*, and
    discuss it before touching a study item, so their calibration ratings
    converge toward that published answer key. Near-unanimous units are what
    inflate Krippendorff's alpha, and `leaked_converged` is the faithful model
    of the failure. Reporting only `leaked_observed` would understate a defect
    by mismodelling it, which is the same error one level down.
    """

    clean: Mapping[str, float]
    leaked_observed: Mapping[str, float]
    leaked_converged: Mapping[str, float]
    n_clean_units: int
    n_leaked_units: int

    def inflation(self, arm: Mapping[str, float]) -> dict[str, float]:
        """Per-dimension `arm - clean`. Positive means the defect flattered alpha."""
        return {
            key: arm[key] - self.clean[key]
            for key in self.clean
            if not math.isnan(self.clean[key]) and not math.isnan(arm[key])
        }

    @property
    def max_inflation(self) -> float:
        """Worst-case inflation across dimensions, from the converged arm."""
        values = list(self.inflation(self.leaked_converged).values())
        return max(values) if values else math.nan


def _as_rubric_rows(
    rater_id: str, items: Mapping[str, Mapping[str, int | None]]
) -> list[RubricScore]:
    """Calibration ratings shaped as the `rubric_score` rows the defect produced.

    The calibration id lands in `generation_id`. That is the bug, reproduced
    deliberately so the comparison runs through the production
    `inter_rater_alpha` rather than through a reverted copy of it.
    """
    rows: list[RubricScore] = []
    for item_id, values in items.items():
        rows.append(
            RubricScore(
                generation_id=item_id,
                rater_type=RaterType.HUMAN,
                rater_id=rater_id,
                name=values.get("name"),
                understand=values.get("understand"),
                respect=values.get("respect"),
                support=values.get("support"),
                explore=values.get("explore"),
                ib=values.get("ib"),
                epp=values.get("epp"),
                de=values.get("de"),
                ie=values.get("ie"),
                naturalness=values.get("naturalness"),
                ritualistic=values.get("ritualistic"),
            )
        )
    return rows


def contamination_check(
    panel_scores: Sequence[RubricScore],
    calibration_by_rater: Mapping[str, Mapping[str, Mapping[str, int | None]]],
) -> ContaminationCheck:
    """Recompute alpha with and without the calibration items as study units."""
    clean = inter_rater_alpha(panel_scores)

    observed: list[RubricScore] = list(panel_scores)
    converged: list[RubricScore] = list(panel_scores)
    consensus = {c.item_id: c.consensus for c in CALIBRATION_SET}

    for rater_id, items in calibration_by_rater.items():
        observed.extend(_as_rubric_rows(rater_id, items))
        # Post-discussion limit: every rater lands on the published consensus.
        converged.extend(
            _as_rubric_rows(
                rater_id,
                {item_id: dict(consensus[item_id]) for item_id in items if item_id in consensus},
            )
        )

    leaked_observed = inter_rater_alpha(observed)
    leaked_converged = inter_rater_alpha(converged)

    return ContaminationCheck(
        clean={k: v.alpha for k, v in clean.items()},
        leaked_observed={k: v.alpha for k, v in leaked_observed.items()},
        leaked_converged={k: v.alpha for k, v in leaked_converged.items()},
        n_clean_units=clean["name"].n_units,
        n_leaked_units=leaked_converged["name"].n_units,
    )


# ---------------------------------------------------------------------------
# The dry run
# ---------------------------------------------------------------------------


@dataclass
class DryRunResult:
    """Everything the pass produced, in a shape that serialises."""

    truth_model: str
    n_items: int
    n_raters: int
    packets: dict[str, BlindedPacket] = field(default_factory=dict)
    panel_scores: list[RubricScore] = field(default_factory=list)
    calibration_by_rater: dict[str, dict[str, dict[str, int | None]]] = field(default_factory=dict)
    consensus: dict[str, dict[str, int | None]] = field(default_factory=dict)
    inter_rater: dict[str, Any] = field(default_factory=dict)
    intra_rater: dict[str, Any] = field(default_factory=dict)
    contamination: ContaminationCheck | None = None
    calibration_checks: dict[str, CalibrationCheck] = field(default_factory=dict)
    broken_rater_check: CalibrationCheck | None = None
    ingest_errors: list[str] = field(default_factory=list)
    n_truth_filled: int = 0


def dry_run(
    items: Sequence[RateableItem],
    *,
    truth_model: str = TruthModel.INDEPENDENT,
    judge_scores: Mapping[str, Mapping[str, int | None]] | None = None,
    panel: Sequence[RaterProfile] = STANDARD_PANEL,
) -> DryRunResult:
    """One full pass: packets -> synthetic sheets -> ingestion -> coefficients.

    Args:
        items: The study responses to be rated.
        truth_model: `independent` (null control) or `judge_anchored` (positive
            control, requires `judge_scores`).
        judge_scores: `generation_id -> {dimension: raw score}` from
            `JudgeResult.scores()`. Raw scale.
        panel: The synthetic raters. Two is the floor for a defensible alpha.
    """
    ids = [i.generation_id for i in items]
    if truth_model == TruthModel.JUDGE_ANCHORED:
        if judge_scores is None:
            raise ValueError("judge_anchored truth needs judge_scores")
        truth, n_filled = anchored_truth({gid: judge_scores[gid] for gid in ids})
    else:
        truth, n_filled = synthetic_truth(ids), 0

    result = DryRunResult(
        truth_model=truth_model,
        n_items=len(items),
        n_raters=len(panel),
        n_truth_filled=n_filled,
    )

    first_rows: list[Mapping[str, Any]] = []
    first_packet: BlindedPacket | None = None

    for profile in panel:
        packet = build_packet(profile.rater_id, items)
        # Belt and braces: `build_packet` already checks, and this is the one
        # failure whose cost is measured in other people's weekends.
        assert_blinded(packet, items)
        result.packets[profile.rater_id] = packet

        rows = synthetic_ratings(
            packet,
            truth,
            noise=profile.noise,
            bias=profile.bias,
            seed=profile.seed,
            reverse_ritualistic=profile.reverse_ritualistic,
            skip_rate=profile.skip_rate,
        )
        report = ingest_ratings(profile.rater_id, rows, packet.assignments)
        result.ingest_errors.extend(
            f"{profile.rater_id}:{e.blind_label}:{e.field}:{e.problem}" for e in report.errors
        )
        result.panel_scores.extend(report.scores)
        result.calibration_by_rater[profile.rater_id] = dict(report.calibration)
        result.calibration_checks[profile.rater_id] = calibration_check(
            profile.rater_id, report.calibration
        )
        if first_packet is None:
            first_packet, first_rows = packet, list(rows)

    result.consensus = human_consensus(result.panel_scores)
    result.inter_rater = {
        key: {
            "alpha": value.alpha,
            "rho": value.rho,
            "n_units": value.n_units,
            "n_observers": value.n_observers,
        }
        for key, value in inter_rater_alpha(result.panel_scores).items()
    }
    result.contamination = contamination_check(result.panel_scores, result.calibration_by_rater)

    # The broken rater never joins the panel; the point is that the calibration
    # step catches them before they would have.
    broken_packet = build_packet(REVERSED_RATER.rater_id, items)
    broken_rows = synthetic_ratings(
        broken_packet,
        truth,
        noise=REVERSED_RATER.noise,
        seed=REVERSED_RATER.seed,
        reverse_ritualistic=True,
    )
    broken_report = ingest_ratings(REVERSED_RATER.rater_id, broken_rows, broken_packet.assignments)
    result.broken_rater_check = calibration_check(
        REVERSED_RATER.rater_id, broken_report.calibration
    )

    # Single-rater fallback: the same rater, a second occasion, reshuffled.
    if first_packet is not None:
        retest_packet = build_packet("R01-t2", items, include_calibration=False)
        retest_rows = synthetic_retest_ratings(first_rows, retest_packet, first_packet)
        retest_report = ingest_ratings("R01-t2", retest_rows, retest_packet.assignments)
        occasion_1 = [s for s in result.panel_scores if s.rater_id == panel[0].rater_id]
        result.intra_rater = {
            key: {"alpha": value.alpha, "rho": value.rho, "n_units": value.n_units}
            for key, value in intra_rater_reliability(occasion_1, retest_report.scores).items()
        }

    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(result: DryRunResult) -> str:
    """Plain-text summary. Instrument check first, so it is never read as a result."""
    lines: list[str] = []
    lines.append(f"HUMAN HARNESS DRY RUN — truth model: {result.truth_model}")
    lines.append(
        "  These are SYNTHETIC raters. Every number below is a statement about the "
        "harness, not about the judge and not about any human."
    )
    lines.append(
        f"  {result.n_items} study items x {result.n_raters} raters; "
        f"{len(result.panel_scores)} rubric_score rows; "
        f"{len(result.ingest_errors)} ingest errors"
    )
    if result.n_truth_filled:
        lines.append(
            f"  {result.n_truth_filled} latent cells had no judge anchor (the judge rejected "
            "the span) and fell back to the independent draw."
        )
    lines.append("")

    c = result.contamination
    if c is not None:
        lines.append("CALIBRATION CONTAMINATION (da38cd1 regression check)")
        lines.append(
            f"  study units: {c.n_clean_units} clean vs {c.n_leaked_units} if the five "
            "calibration items were routed as generations"
        )
        lines.append(
            "  'converged' is the faithful arm: real raters are shown the consensus and "
            "discuss it, so their calibration ratings become near-unanimous units."
        )
        lines.append(
            f"  worst alpha inflation the defect would have produced: {c.max_inflation:+.3f}"
        )
        lines.append(
            f"  {'dimension':<14}{'clean':>9}{'observed':>10}{'converged':>11}{'inflation':>11}"
        )
        conv = c.inflation(c.leaked_converged)
        for key in RUBRIC_DIMENSIONS:
            lines.append(
                f"  {key:<14}{c.clean[key]:>9.3f}{c.leaked_observed[key]:>10.3f}"
                f"{c.leaked_converged[key]:>11.3f}{conv.get(key, math.nan):>+11.3f}"
            )
        lines.append("")

    lines.append("INTER-RATER ALPHA (synthetic panel)")
    lines.append(f"  {'dimension':<14}{'alpha':>9}{'rho':>9}{'n':>6}")
    for key in RUBRIC_DIMENSIONS:
        row = result.inter_rater.get(key, {})
        lines.append(
            f"  {key:<14}{row.get('alpha', math.nan):>9.3f}"
            f"{row.get('rho', math.nan):>9.3f}{row.get('n_units', 0):>6}"
        )
    lines.append("")

    if result.intra_rater:
        lines.append("INTRA-RATER RELIABILITY (single-rater fallback, two occasions)")
        lines.append(
            "  Weaker evidence than inter-rater agreement: it measures whether one "
            "person is stable, not whether two people agree."
        )
        for key in RUBRIC_DIMENSIONS:
            row = result.intra_rater.get(key, {})
            lines.append(f"  {key:<14}{row.get('alpha', math.nan):>9.3f}{row.get('n_units', 0):>6}")
        lines.append("")

    lines.append("CALIBRATION CHECK")
    for rater_id, check in result.calibration_checks.items():
        lines.append(
            f"  {rater_id}: {check.n_items} items, flagged={list(check.flagged) or 'none'}, "
            f"ritualistic bias {check.bias['ritualistic']:+.2f}"
        )
    broken = result.broken_rater_check
    if broken is not None:
        caught = "ritualistic" in broken.flagged
        lines.append(
            f"  {REVERSED_RATER.rater_id} (ritualistic reversed on purpose): "
            f"flagged={list(broken.flagged)}, ritualistic bias "
            f"{broken.bias['ritualistic']:+.2f} -> {'CAUGHT' if caught else 'MISSED'}"
        )
    return "\n".join(lines)


def to_json(result: DryRunResult) -> dict[str, Any]:
    """JSON-safe view. NaN becomes null; JSON has no NaN."""

    def clean(value: Any) -> Any:
        if isinstance(value, float):
            return None if math.isnan(value) else round(value, 6)
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        return value

    c = result.contamination
    return {
        "truth_model": result.truth_model,
        "n_items": result.n_items,
        "n_raters": result.n_raters,
        "n_rubric_score_rows": len(result.panel_scores),
        "n_ingest_errors": len(result.ingest_errors),
        "ingest_errors": result.ingest_errors[:20],
        "n_truth_filled": result.n_truth_filled,
        "inter_rater": clean(result.inter_rater),
        "intra_rater": clean(result.intra_rater),
        "contamination": None
        if c is None
        else {
            "n_clean_units": c.n_clean_units,
            "n_leaked_units": c.n_leaked_units,
            "clean": clean(dict(c.clean)),
            "leaked_observed": clean(dict(c.leaked_observed)),
            "leaked_converged": clean(dict(c.leaked_converged)),
            "inflation_observed": clean(c.inflation(c.leaked_observed)),
            "inflation_converged": clean(c.inflation(c.leaked_converged)),
            "max_inflation": clean(c.max_inflation),
        },
        "calibration_checks": {
            rater: {"flagged": list(check.flagged), "bias": clean(dict(check.bias))}
            for rater, check in result.calibration_checks.items()
        },
        "broken_rater_caught": (
            result.broken_rater_check is not None
            and "ritualistic" in result.broken_rater_check.flagged
        ),
        "n_calibration_items": len(CALIBRATION_SET),
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(
        prog="carelite.eval.human.dry_run",
        description="Exercise the blinded human-rating harness against synthetic raters.",
    )
    parser.add_argument("--items", type=int, default=60)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    items = [
        RateableItem(
            generation_id=f"gen-dry-{i:03d}",
            scenario_text=f"Synthetic patient turn {i}.",
            response_text=f"Synthetic clinician response {i}. It says something about the plan.",
            condition=("A", "A2", "B", "C", "LC", "D")[i % 6],
        )
        for i in range(args.items)
    ]
    result = dry_run(items)
    text = render(result)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(to_json(result), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
