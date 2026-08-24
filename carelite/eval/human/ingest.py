"""Read returned rating sheets, validate them, and unblind them.

A returned spreadsheet is untrusted in the ordinary sense — not adversarial, but
hand-edited, and hand-edited data arrives with blank cells, a `4.5` where an
integer was asked for, a `0` for "not applicable", a row someone pasted twice,
and a label with a trailing space from a copy-paste. Every one of those has a
plausible silent interpretation and every silent interpretation corrupts a
reliability coefficient in a way that cannot be seen afterwards.

So this module refuses instead of guessing. Errors are collected per row rather
than raised on the first one, because a rater who made the same mistake in
twelve rows should be told about all twelve, once, rather than twelve times in
sequence. `IngestReport.ok` is the gate: nothing reaches `rubric_score` until it
is true.

The one thing that *is* interpreted: a blank cell means "not rated", not zero.
It becomes `None`, which flows through Krippendorff's alpha as genuine missing
data — the coefficient handles it correctly, and it is the honest encoding of a
rater who skipped an item.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carelite.eval.human.blinding import Assignment
from carelite.eval.rubric.calibration import CALIBRATION_SET
from carelite.eval.rubric.dimensions import SCALE_MAX, SCALE_MIN
from carelite.types import RUBRIC_DIMENSIONS, RaterType, RubricScore

__all__ = [
    "CalibrationCheck",
    "IngestReport",
    "RowError",
    "calibration_check",
    "ingest_ratings",
    "read_csv",
    "read_json",
]

#: Mean absolute deviation from consensus, per dimension, above which a rater is
#: flagged for re-discussion before their study ratings are used. One scale
#: point of average disagreement on an anchored five-point scale is not noise.
CALIBRATION_MAD_LIMIT = 1.0


@dataclass(frozen=True, slots=True)
class RowError:
    """One problem with one returned row."""

    blind_label: str
    field: str
    problem: str


@dataclass
class IngestReport:
    """The outcome of reading one rater's returned sheet."""

    rater_id: str
    scores: list[RubricScore] = field(default_factory=list)
    calibration: dict[str, dict[str, int | None]] = field(default_factory=dict)
    errors: list[RowError] = field(default_factory=list)
    #: Labels the rater was assigned but did not return a row for.
    missing_labels: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def n_rated(self) -> int:
        return len(self.scores)


def read_csv(path: Path | str) -> list[dict[str, Any]]:
    """Read a returned rating sheet. Blank cells stay blank; nothing is coerced here."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path | str) -> list[dict[str, Any]]:
    """Read ratings returned as JSON: a list of row objects, or label -> scores."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [{"blind_label": label, **scores} for label, scores in payload.items()]
    return list(payload)


def _parse_score(raw: Any) -> tuple[int | None, str | None]:
    """`(value, problem)`. A blank cell is `(None, None)` — missing, not zero."""
    if raw is None:
        return None, None
    text = str(raw).strip()
    if text == "" or text.upper() in {"NA", "N/A", "-"}:
        return None, None
    try:
        value = float(text)
    except ValueError:
        return None, f"{text!r} is not a number"
    if value != int(value):
        return None, f"{text!r} is not a whole number; the scale has no half points"
    number = int(value)
    if not SCALE_MIN <= number <= SCALE_MAX:
        return None, f"{number} is outside the {SCALE_MIN}-{SCALE_MAX} scale"
    return number, None


def _parse_flags(raw: Any) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def ingest_ratings(
    rater_id: str,
    rows: Iterable[Mapping[str, Any]],
    assignments: Sequence[Assignment],
) -> IngestReport:
    """Validate and unblind one rater's returned rows.

    Args:
        rater_id: Whose sheet this is. Checked against the assignments so a
            packet returned by the wrong person is caught rather than silently
            attributed.
        rows: Returned rows, each carrying `blind_label` and eleven dimensions.
        assignments: That rater's `Assignment` records — the only unblinding key.

    Returns:
        An `IngestReport`. Study ratings become `RubricScore` rows with
        `rater_type=human`; calibration ratings are kept separately, because
        calibration items are fixtures rather than generations and must never
        enter the results table.
    """
    mine = [a for a in assignments if a.rater_id == rater_id]
    if not mine:
        return IngestReport(
            rater_id=rater_id,
            errors=[RowError("", "rater_id", f"no assignments recorded for rater {rater_id!r}")],
        )

    by_label = {a.blind_label: a for a in mine}
    report = IngestReport(rater_id=rater_id)
    seen: set[str] = set()

    for row in rows:
        label = str(row.get("blind_label", "")).strip()
        if not label:
            report.errors.append(RowError("", "blind_label", "row has no blind_label"))
            continue
        assignment = by_label.get(label)
        if assignment is None:
            report.errors.append(
                RowError(label, "blind_label", f"{label!r} was not assigned to {rater_id!r}")
            )
            continue
        if label in seen:
            report.errors.append(RowError(label, "blind_label", "duplicate row for this label"))
            continue
        seen.add(label)

        values: dict[str, int | None] = {}
        for key in RUBRIC_DIMENSIONS:
            value, problem = _parse_score(row.get(key))
            if problem is not None:
                report.errors.append(RowError(label, key, problem))
            values[key] = value

        if assignment.is_calibration:
            # Keyed by `calibration_id`, which is what `calibration_check`
            # looks up in `CALIBRATION_SET`. These never become `RubricScore`
            # rows: a calibration item has no generation, and the raters were
            # shown its consensus scores, so it would be an agreement unit whose
            # answer was published in advance.
            assert assignment.calibration_id is not None  # Assignment invariant
            report.calibration[assignment.calibration_id] = values
            continue

        assert assignment.generation_id is not None  # Assignment invariant
        report.scores.append(
            RubricScore(
                generation_id=assignment.generation_id,
                rater_type=RaterType.HUMAN,
                rater_id=rater_id,
                name=values["name"],
                understand=values["understand"],
                respect=values["respect"],
                support=values["support"],
                explore=values["explore"],
                ib=values["ib"],
                epp=values["epp"],
                de=values["de"],
                ie=values["ie"],
                naturalness=values["naturalness"],
                ritualistic=values["ritualistic"],
                safety_flags=_parse_flags(row.get("safety_flags")),
            )
        )

    report.missing_labels = sorted(set(by_label) - seen)
    return report


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibrationCheck:
    """How far one rater sat from the calibration consensus, per dimension."""

    rater_id: str
    n_items: int
    #: Dimension -> mean absolute deviation from consensus.
    mad: Mapping[str, float]
    #: Dimension -> mean signed deviation. A consistent sign is a systematic
    #: leniency or severity offset, which is worth a conversation before rating
    #: rather than an alpha correction afterwards.
    bias: Mapping[str, float]
    flagged: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.flagged


def calibration_check(
    rater_id: str, calibration: Mapping[str, Mapping[str, int | None]]
) -> CalibrationCheck:
    """Compare a rater's calibration scores against the agreed consensus.

    Run before the discussion, not after: the point is to know which dimensions
    to spend the discussion on. Deviation on `ritualistic` in particular is
    worth checking for a *sign* pattern — a rater who has the reverse coding
    backwards produces a large, consistent, one-directional error there and
    normal-looking numbers everywhere else.
    """
    consensus = {c.item_id: c.consensus for c in CALIBRATION_SET}
    deviations: dict[str, list[float]] = {k: [] for k in RUBRIC_DIMENSIONS}
    n_items = 0

    for item_id, scores in calibration.items():
        agreed = consensus.get(item_id)
        if agreed is None:
            continue
        n_items += 1
        for key in RUBRIC_DIMENSIONS:
            value = scores.get(key)
            if value is None:
                continue
            deviations[key].append(float(value) - float(agreed[key]))

    mad = {
        key: (sum(abs(d) for d in vals) / len(vals) if vals else float("nan"))
        for key, vals in deviations.items()
    }
    bias = {
        key: (sum(vals) / len(vals) if vals else float("nan")) for key, vals in deviations.items()
    }
    flagged = tuple(
        key for key, value in mad.items() if value == value and value > CALIBRATION_MAD_LIMIT
    )
    return CalibrationCheck(rater_id=rater_id, n_items=n_items, mad=mad, bias=bias, flagged=flagged)
