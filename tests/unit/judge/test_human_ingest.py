"""Ingestion refuses to guess. Every ambiguity is an error, not an interpretation."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from carelite.eval.human import (
    build_packet,
    calibration_check,
    ingest_ratings,
    rater_instructions,
    rating_sheet_csv,
    read_csv,
    read_json,
)
from carelite.eval.human.packet import calibration_answer_key, calibration_worksheet, write_packet
from carelite.eval.rubric.calibration import CALIBRATION_SET
from carelite.types import RUBRIC_DIMENSIONS, RaterType

from .test_human_blinding import items


def full_row(label: str, value: int = 4) -> dict[str, object]:
    return {"blind_label": label, **dict.fromkeys(RUBRIC_DIMENSIONS, value)}


class TestIngest:
    def test_valid_rows_become_human_rubric_scores(self) -> None:
        packet = build_packet("R01", items(6))
        study = [a for a in packet.assignments if not a.is_calibration]
        report = ingest_ratings("R01", [full_row(a.blind_label) for a in study], packet.assignments)

        assert report.ok
        assert report.n_rated == 6
        assert all(s.rater_type is RaterType.HUMAN for s in report.scores)
        assert all(s.rater_id == "R01" for s in report.scores)
        assert sorted(s.generation_id for s in report.scores) == sorted(
            a.generation_id for a in study
        )

    def test_calibration_rows_are_kept_out_of_the_results(self) -> None:
        """Calibration items are fixtures; they must never enter `rubric_score`."""
        packet = build_packet("R01", items(6))
        rows = [full_row(a.blind_label) for a in packet.assignments]
        report = ingest_ratings("R01", rows, packet.assignments)
        assert report.n_rated == 6
        assert set(report.calibration) == {c.item_id for c in CALIBRATION_SET}

    def test_a_blank_cell_is_missing_not_zero(self) -> None:
        packet = build_packet("R01", items(2))
        label = next(a.blind_label for a in packet.assignments if not a.is_calibration)
        row = {**full_row(label), "naturalness": ""}
        report = ingest_ratings("R01", [row], packet.assignments)
        assert report.ok
        assert report.scores[0].naturalness is None
        assert report.scores[0].de == 4

    def test_out_of_range_scores_are_errors(self) -> None:
        packet = build_packet("R01", items(2))
        label = next(a.blind_label for a in packet.assignments if not a.is_calibration)
        report = ingest_ratings("R01", [{**full_row(label), "de": 7}], packet.assignments)
        assert not report.ok
        assert report.errors[0].field == "de"
        assert "outside" in report.errors[0].problem

    def test_half_points_are_errors(self) -> None:
        packet = build_packet("R01", items(2))
        label = next(a.blind_label for a in packet.assignments if not a.is_calibration)
        report = ingest_ratings("R01", [{**full_row(label), "ib": "3.5"}], packet.assignments)
        assert not report.ok
        assert "whole number" in report.errors[0].problem

    def test_unknown_label_is_refused(self) -> None:
        packet = build_packet("R01", items(2))
        report = ingest_ratings("R01", [full_row("R99-001")], packet.assignments)
        assert not report.ok
        assert "was not assigned" in report.errors[0].problem

    def test_duplicate_rows_are_refused(self) -> None:
        packet = build_packet("R01", items(2))
        label = next(a.blind_label for a in packet.assignments if not a.is_calibration)
        report = ingest_ratings("R01", [full_row(label), full_row(label)], packet.assignments)
        assert not report.ok
        assert "duplicate" in report.errors[0].problem

    def test_every_bad_row_is_reported_not_just_the_first(self) -> None:
        packet = build_packet("R01", items(4))
        study = [a for a in packet.assignments if not a.is_calibration]
        rows = [{**full_row(a.blind_label), "de": 9} for a in study]
        report = ingest_ratings("R01", rows, packet.assignments)
        assert len(report.errors) == 4

    def test_unreturned_labels_are_listed(self) -> None:
        packet = build_packet("R01", items(6))
        study = [a for a in packet.assignments if not a.is_calibration]
        report = ingest_ratings("R01", [full_row(study[0].blind_label)], packet.assignments)
        assert len(report.missing_labels) == 10  # 5 calibration + 5 unrated study items

    def test_a_packet_returned_by_the_wrong_rater_is_caught(self) -> None:
        packet = build_packet("R01", items(2))
        report = ingest_ratings("R02", [], packet.assignments)
        assert not report.ok
        assert "no assignments recorded" in report.errors[0].problem


class TestRoundTripThroughFiles:
    def test_csv_sheet_round_trips(self, tmp_path: Path) -> None:
        packet = build_packet("R01", items(6))
        sheet = rating_sheet_csv(packet.items)

        # Fill it in the way a rater would: open in a spreadsheet, type numbers.
        rows = list(csv.DictReader(io.StringIO(sheet)))
        for row in rows:
            for key in RUBRIC_DIMENSIONS:
                row[key] = "3"
        out = tmp_path / "returned.csv"
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        report = ingest_ratings("R01", read_csv(out), packet.assignments)
        assert report.ok
        assert report.n_rated == 6

    def test_json_ratings_keyed_by_label(self, tmp_path: Path) -> None:
        import json

        packet = build_packet("R01", items(2))
        label = next(a.blind_label for a in packet.assignments if not a.is_calibration)
        path = tmp_path / "r.json"
        path.write_text(json.dumps({label: dict.fromkeys(RUBRIC_DIMENSIONS, 5)}))
        report = ingest_ratings("R01", read_json(path), packet.assignments)
        assert report.ok and report.n_rated == 1

    def test_write_packet_produces_the_three_rater_files_and_no_answer_key(
        self, tmp_path: Path
    ) -> None:
        packet = build_packet("R01", items(6))
        paths = write_packet(packet, tmp_path)
        assert set(paths) == {"instructions", "calibration", "sheet"}
        assert all(p.exists() for p in paths.values())
        # The answer key must not land in the rater's folder.
        written = "\n".join(p.read_text() for p in paths.values())
        assert "consensus" not in written.lower()


class TestGeneratedDocuments:
    def test_instructions_cover_every_dimension_and_anchor(self) -> None:
        text = rater_instructions()
        from carelite.eval.rubric.dimensions import DIMENSIONS

        for key in RUBRIC_DIMENSIONS:
            assert f"`{key}`" in text
            assert DIMENSIONS[key].anchor_5 in text

    def test_instructions_state_the_reverse_coding_prominently(self) -> None:
        text = rater_instructions()
        assert "5 is the worst score" in text
        assert "REVERSE-CODED" in text

    def test_worksheet_hides_the_consensus_and_the_key_shows_it(self) -> None:
        worksheet = calibration_worksheet()
        key = calibration_answer_key()
        for cal in CALIBRATION_SET:
            # The response is blockquoted line by line, so match on its lines.
            for line in filter(None, cal.response.splitlines()):
                assert f"> {line}" in worksheet
            assert cal.item_id in worksheet
            assert cal.teaching_point in key
        assert "consensus" not in worksheet.lower()


class TestCalibrationCheck:
    def test_a_rater_who_matches_consensus_is_not_flagged(self) -> None:
        scores = {c.item_id: dict(c.consensus) for c in CALIBRATION_SET}
        check = calibration_check("R01", scores)
        assert check.ok
        assert check.n_items == 5
        assert all(v == 0.0 for v in check.mad.values())

    def test_a_reversed_ritualistic_column_is_caught(self) -> None:
        """The one rater error that otherwise produces entirely normal data."""
        scores = {
            c.item_id: {**dict(c.consensus), "ritualistic": 6 - c.consensus["ritualistic"]}
            for c in CALIBRATION_SET
        }
        check = calibration_check("R01", scores)
        assert "ritualistic" in check.flagged
        assert not check.ok
        # And the signed bias identifies it as a direction error, not noise.
        assert abs(check.bias["ritualistic"]) > 1.0

    def test_a_lenient_rater_shows_a_consistent_positive_bias(self) -> None:
        scores = {
            c.item_id: {k: min(5, v + 1) for k, v in c.consensus.items()} for c in CALIBRATION_SET
        }
        check = calibration_check("R01", scores)
        assert all(b >= 0 for b in check.bias.values())
