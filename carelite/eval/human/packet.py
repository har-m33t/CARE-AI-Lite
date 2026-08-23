"""Rater-facing artefacts: instructions, the calibration packet, the rating sheet.

v3 §12 requires the written rubric with anchored examples to be distributed
*before* rating starts, and five calibration responses to be scored and
discussed first. This module generates both from the same data the judge is
prompted with and the reverse-coding tests assert against, so a rater and the
`gpt-oss:20b` judge are demonstrably reading one rubric rather than two
descriptions of it that have drifted apart.

Everything here is generated, never transcribed. `docs/rubric.md` is the prose
document a human reads end to end; these instructions are the working reference
that ships inside the packet, and both trace to
`carelite.eval.rubric.dimensions`.

The rating sheet is CSV because raters use spreadsheets. Its first columns are
the label and the two texts; the eleven dimension columns follow in rubric
order, with `ritualistic` last and its reverse coding stated in the header
comment, in the instructions, and in the calibration discussion. Three
statements is not redundancy — a rater who reverses that one column produces
data that looks entirely normal.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from pathlib import Path

from carelite.eval.human.blinding import BlindedPacket, BlindItem
from carelite.eval.rubric.calibration import CALIBRATION_SCENARIO, CALIBRATION_SET
from carelite.eval.rubric.dimensions import DIMENSIONS, RUBRIC_VERSION
from carelite.types import RUBRIC_DIMENSIONS

__all__ = [
    "RATING_SHEET_COLUMNS",
    "calibration_answer_key",
    "calibration_worksheet",
    "rater_instructions",
    "rating_sheet_csv",
    "write_packet",
]

#: Column order of the returned rating sheet. `blind_label` is the join key and
#: must survive a round trip through a spreadsheet unedited.
RATING_SHEET_COLUMNS: tuple[str, ...] = (
    "blind_label",
    *RUBRIC_DIMENSIONS,
    "safety_flags",
    "notes",
)


def rater_instructions() -> str:
    """The written rubric raters get before they score anything.

    Generated from `dimensions.py`. If a definition or an anchor changes, this
    document changes with it and `RUBRIC_VERSION` moves, which is the signal to
    re-calibrate rather than to carry old ratings forward.
    """
    out: list[str] = []
    out.append(f"# CARELite rating instructions (rubric v{RUBRIC_VERSION})")
    out.append("")
    out.append("## What you are rating")
    out.append("")
    out.append(
        "Each item is one **response**: a single clinician turn, addressed to the patient, "
        "replying to the patient turn shown above it. Rate only what is in that response. Do "
        "not give credit for what the clinician would plausibly say next, and do not mark a "
        "response down for being one turn rather than a whole visit."
    )
    out.append("")
    out.append(
        "You will not be told which system produced any response, and the order you see them "
        "in is specific to you. Both are deliberate. If you think you have worked out the "
        "pattern, say so — it is a finding about the responses, not a reason to change how you "
        "score."
    )
    out.append("")
    out.append("## How to score")
    out.append("")
    out.append(
        "1. Every dimension gets a whole number from 1 to 5. Anchors are given for 1, 3 and 5; "
        "2 and 4 sit between them."
    )
    out.append(
        "2. **Score the dimensions independently.** A response can be excellent on one and "
        "entirely absent on another. Resist the pull to make the eleven numbers agree — "
        "calibration item CAL-03 is a good response that scores 2 on `respect`, because the "
        "respecting move simply is not in it."
    )
    out.append(
        "3. **Naturalness is not quality.** A blunt, unempathic response can be completely "
        "natural speech. CAL-01 is exactly that, and scores 1s on NURSE with a 4 on "
        "naturalness."
    )
    out.append(
        "4. If you cannot decide between two adjacent scores, pick the lower one and note why. "
        "Consistent hesitation is data; a coin flip is not."
    )
    out.append("")
    out.append("## Reverse coding — read this twice")
    out.append("")
    out.append(
        "Ten dimensions are scored so that **5 is good**. `ritualistic` is the exception: for "
        "`ritualistic`, **5 is the worst score** and means the response is a script with the "
        "communication framework showing. 1 means no ritual at all."
    )
    out.append("")
    out.append(
        "A response can honestly score 4-5 on the NURSE dimensions *and* 5 on `ritualistic` at "
        "the same time. That is not a contradiction — it is the pattern this study exists to "
        "measure. Do not soften either number to make them agree. CAL-02 is that case."
    )
    out.append("")
    out.append("## The eleven dimensions")
    out.append("")

    for key in RUBRIC_DIMENSIONS:
        dim = DIMENSIONS[key]
        out.append(f"### `{key}` — {dim.label}")
        out.append("")
        out.append(f"*Framework:* {dim.framework}")
        out.append("")
        out.append(f"**Ask yourself:** {dim.question}")
        out.append("")
        out.append(dim.definition)
        out.append("")
        if dim.reverse_coded:
            out.append("> **REVERSE-CODED. 5 is the worst score on this dimension.**")
            out.append("")
        out.append(f"- **1** — {dim.anchor_1}")
        out.append(f"- **3** — {dim.anchor_3}")
        out.append(f"- **5** — {dim.anchor_5}")
        out.append("")
        out.append(f"*Source:* {dim.source}")
        out.append("")

    out.append("## Before you start")
    out.append("")
    out.append(
        "Score the five calibration responses first, on your own, then go through them with "
        "the study lead. All five answer the same patient turn, so you are comparing like with "
        "like. The point is not to test you — it is to surface the four disagreements that "
        "otherwise show up as noise across sixty ratings."
    )
    out.append("")
    return "\n".join(out)


def calibration_worksheet() -> str:
    """The five calibration responses, without their answers.

    Handed out first. The consensus scores are in `calibration_answer_key` and
    are not shown to the rater until they have committed their own numbers —
    a calibration set read alongside its answers teaches nothing.
    """
    out: list[str] = [
        "# Calibration set — score these five before anything else",
        "",
        "All five responses answer the same patient turn:",
        "",
        f"> {CALIBRATION_SCENARIO}",
        "",
        "Score each on all eleven dimensions, then bring your scores to the discussion.",
        "Remember that `ritualistic` runs the other way: 5 is the worst score.",
        "",
    ]
    for cal in CALIBRATION_SET:
        out.append(f"## {cal.item_id}")
        out.append("")
        out.append(_quote(cal.response))
        out.append("")
    return "\n".join(out)


def calibration_answer_key() -> str:
    """Consensus scores, per-dimension reasoning, and the arguments that settled them.

    Read out during the discussion. The `disagreements` field is the valuable
    part: it records where the consensus was genuinely argued and what rule
    resolved it, which is what a new rater needs and what a bare table of
    numbers cannot convey.
    """
    out: list[str] = [
        f"# Calibration answer key (rubric v{RUBRIC_VERSION})",
        "",
        "Do not distribute before the raters have committed their own scores.",
        "",
    ]
    for cal in CALIBRATION_SET:
        out.append(f"## {cal.item_id} — {cal.archetype}")
        out.append("")
        out.append(f"**Teaching point.** {cal.teaching_point}")
        out.append("")
        out.append("| dimension | consensus | why |")
        out.append("|---|---|---|")
        for key in RUBRIC_DIMENSIONS:
            reason = cal.rationales[key].replace("|", "/")
            out.append(f"| `{key}` | {cal.consensus[key]} | {reason} |")
        out.append("")
        if cal.evidence_spans:
            out.append("**Evidence quoted from the response:**")
            out.append("")
            for key, span in cal.evidence_spans.items():
                out.append(f"- `{key}`: “{span}”")
            out.append("")
        if cal.disagreements:
            out.append("**Where this was argued:**")
            out.append("")
            for note in cal.disagreements:
                out.append(f"- {note}")
            out.append("")
    return "\n".join(out)


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def rating_sheet_csv(items: Sequence[BlindItem]) -> str:
    """The blank rating sheet, one row per item, dimensions as empty columns.

    Includes the two texts inline so the rater never has to hold two files open
    and never has to match a label by hand — the commonest source of a
    mislabelled rating, and one that silently swaps two conditions.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "blind_label",
            "display_order",
            "is_calibration",
            "patient_turn",
            "clinician_response",
            *RUBRIC_DIMENSIONS,
            "safety_flags",
            "notes",
        ]
    )
    for item in items:
        writer.writerow(
            [
                item.blind_label,
                item.display_order,
                "yes" if item.is_calibration else "no",
                item.scenario_text,
                item.response_text,
                *[""] * len(RUBRIC_DIMENSIONS),
                "",
                "",
            ]
        )
    return buffer.getvalue()


def write_packet(packet: BlindedPacket, out_dir: Path) -> dict[str, Path]:
    """Write one rater's complete packet to disk. Returns the paths written.

    The answer key is deliberately **not** written here. It goes to the study
    lead, not into the rater's directory, and putting it in the same function
    that builds the rater's folder is how it ends up in the same zip file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "instructions": out_dir / "rater_instructions.md",
        "calibration": out_dir / "calibration_worksheet.md",
        "sheet": out_dir / f"ratings_{packet.rater_id}.csv",
    }
    paths["instructions"].write_text(rater_instructions(), encoding="utf-8")
    paths["calibration"].write_text(calibration_worksheet(), encoding="utf-8")
    paths["sheet"].write_text(rating_sheet_csv(packet.items), encoding="utf-8")
    return paths
