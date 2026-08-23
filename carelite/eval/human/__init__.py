"""Blinded human-rating harness. Built now, run in sprint 10.

Human rating is the bottleneck this project cannot control — v3 §12 says to
start recruiting in sprint 1 rather than sprint 10 — so the harness is built and
proven before a rater exists. "Proven" here means exercised end to end against
`synthetic.py`: packets built, blinding asserted, rows returned, ingested,
unblinded, and Krippendorff's alpha computed, with the tests asserting that low
noise produces high alpha and high noise produces alpha near zero. A harness
that merely imports is not evidence of anything.

The pipeline::

    items = [RateableItem(...), ...]                 # study responses
    packet = build_packet("R01", items)              # blinded, per-rater shuffle
    store_assignments(packet.assignments)            # the unblinding key, first
    write_packet(packet, out_dir)                    # instructions + calibration + sheet
    ...                                              # a human does the work
    report = ingest_ratings("R01", read_csv(path), packet.assignments)
    alpha = inter_rater_alpha([*r1.scores, *r2.scores])

Two things to know before touching any of it.

**Unblinding is a join.** The rater-facing export carries a meaningless label
and two blocks of text. The condition lives only in `rating_assignment`. See
`blinding.assert_blinded`, which fails a packet that leaks a condition through
its labels, its payload, or its ordering.

**`ritualistic` is reverse-coded**, and a rater who gets that backwards produces
data that looks completely normal. It is stated in the instructions, in the
calibration discussion, and in the sheet header, and `ingest.calibration_check`
reports a signed per-dimension bias specifically so the reversal shows up before
sixty ratings are collected under it.
"""

from carelite.eval.human.blinding import (
    Assignment,
    BlindedPacket,
    BlindingViolation,
    BlindItem,
    RateableItem,
    assert_blinded,
    build_packet,
    rater_seed,
    unblind,
)
from carelite.eval.human.ingest import (
    CalibrationCheck,
    IngestReport,
    RowError,
    calibration_check,
    ingest_ratings,
    read_csv,
    read_json,
)
from carelite.eval.human.packet import (
    RATING_SHEET_COLUMNS,
    calibration_answer_key,
    calibration_worksheet,
    rater_instructions,
    rating_sheet_csv,
    write_packet,
)
from carelite.eval.human.reliability import (
    human_consensus,
    inter_rater_alpha,
    intra_rater_reliability,
    reliability_matrix,
    scores_by_rater,
)
from carelite.eval.human.synthetic import (
    synthetic_ratings,
    synthetic_retest_ratings,
    synthetic_truth,
)

__all__ = [
    "RATING_SHEET_COLUMNS",
    "Assignment",
    "BlindItem",
    "BlindedPacket",
    "BlindingViolation",
    "CalibrationCheck",
    "IngestReport",
    "RateableItem",
    "RowError",
    "assert_blinded",
    "build_packet",
    "calibration_answer_key",
    "calibration_check",
    "calibration_worksheet",
    "human_consensus",
    "ingest_ratings",
    "inter_rater_alpha",
    "intra_rater_reliability",
    "rater_instructions",
    "rater_seed",
    "rating_sheet_csv",
    "read_csv",
    "read_json",
    "reliability_matrix",
    "scores_by_rater",
    "synthetic_ratings",
    "synthetic_retest_ratings",
    "synthetic_truth",
    "unblind",
    "write_packet",
]
