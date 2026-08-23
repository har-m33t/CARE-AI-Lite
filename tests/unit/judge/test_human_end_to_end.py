"""The human harness, exercised end to end on synthetic raters.

Human rating happens in sprint 10. The harness is built now, so the risk is that
a blinding bug, a label-join off-by-one, or a reversed `ritualistic` column is
discovered *after* two paid raters have spent a weekend on sixty responses. This
module is the insurance: every step from packet construction to Krippendorff's
alpha runs here against generated raters, and the assertions are on the
*direction* of the coefficients, not merely that they compute.

If low-noise raters did not produce a high alpha, or guessing raters did not
produce an alpha near zero, the reliability layer would be agreeing with itself
somewhere it should not be.
"""

from __future__ import annotations

import csv
import io
import math

import pytest

from carelite.eval.human import (
    RateableItem,
    build_packet,
    calibration_check,
    human_consensus,
    ingest_ratings,
    inter_rater_alpha,
    intra_rater_reliability,
    rating_sheet_csv,
    scores_by_rater,
    synthetic_ratings,
    synthetic_retest_ratings,
    synthetic_truth,
)
from carelite.eval.judge import LLMJudge, ReplayClient, judge_human_validity
from carelite.eval.rubric.dimensions import to_quality
from carelite.types import RUBRIC_DIMENSIONS, Condition

from .conftest import RESPONSE, SCENARIO, judge_json

N_ITEMS = 60
CONDITIONS = (Condition.A, Condition.B, Condition.C)


def study_items(n: int = N_ITEMS) -> list[RateableItem]:
    return [
        RateableItem(
            generation_id=f"gen-{i:04d}",
            scenario_text=f"Patient turn {i % 20}.",
            response_text=f"Clinician response number {i}.",
            condition=CONDITIONS[i % 3],
            scenario_id=f"sc-{i % 20:04d}",
        )
        for i in range(n)
    ]


def run_rater(rater_id: str, items, truth, **kwargs):
    """Packet -> synthetic rows -> ingestion. The whole rater-facing loop."""
    packet = build_packet(rater_id, items)
    rows = synthetic_ratings(packet, truth, **kwargs)
    report = ingest_ratings(rater_id, rows, packet.assignments)
    return packet, rows, report


class TestFullPipeline:
    def test_packet_to_alpha_on_two_synthetic_raters(self) -> None:
        """The complete sprint-10 workflow, with nobody in the loop."""
        items = study_items()
        truth = synthetic_truth([i.generation_id for i in items], seed=1)

        _, _, r1 = run_rater("R01", items, truth, noise=0.3, seed=11)
        _, _, r2 = run_rater("R02", items, truth, noise=0.3, seed=22)

        assert r1.ok and r2.ok
        assert r1.n_rated == r2.n_rated == N_ITEMS

        alpha = inter_rater_alpha([*r1.scores, *r2.scores])
        assert set(alpha) == set(RUBRIC_DIMENSIONS)
        for key in RUBRIC_DIMENSIONS:
            assert alpha[key].n_units == N_ITEMS
            assert alpha[key].n_observers == 2
            assert alpha[key].alpha > 0.6, key

    def test_low_noise_gives_high_alpha_and_guessing_gives_near_zero(self) -> None:
        """The direction check. Without it, alpha here would prove nothing."""
        items = study_items()
        truth = synthetic_truth([i.generation_id for i in items], seed=2)

        _, _, careful_1 = run_rater("C1", items, truth, noise=0.0, seed=1)
        _, _, careful_2 = run_rater("C2", items, truth, noise=0.0, seed=2)
        careful = inter_rater_alpha([*careful_1.scores, *careful_2.scores])

        _, _, guess_1 = run_rater("G1", items, truth, noise=6.0, seed=3)
        _, _, guess_2 = run_rater("G2", items, truth, noise=6.0, seed=4)
        guessing = inter_rater_alpha([*guess_1.scores, *guess_2.scores])

        assert careful["de"].alpha == pytest.approx(1.0)
        assert abs(guessing["de"].alpha) < 0.25
        assert careful["naturalness"].alpha > guessing["naturalness"].alpha

    def test_a_lenient_rater_shows_low_alpha_with_high_rho(self) -> None:
        """The signature the study must be able to tell apart from real disagreement."""
        items = study_items()
        truth = synthetic_truth([i.generation_id for i in items], seed=3)
        _, _, strict = run_rater("S1", items, truth, noise=0.0, seed=5)
        _, _, lenient = run_rater("S2", items, truth, noise=0.0, bias=1, seed=6)

        result = inter_rater_alpha([*strict.scores, *lenient.scores])["de"]
        assert result.rho > 0.9
        assert result.alpha < result.rho

    def test_missing_cells_flow_through_as_missing_data(self) -> None:
        items = study_items()
        truth = synthetic_truth([i.generation_id for i in items], seed=4)
        _, _, r1 = run_rater("M1", items, truth, noise=0.2, seed=7, skip_rate=0.2)
        _, _, r2 = run_rater("M2", items, truth, noise=0.2, seed=8)

        assert r1.ok  # a blank cell is legal, not an error
        assert any(s.de is None for s in r1.scores)
        alpha = inter_rater_alpha([*r1.scores, *r2.scores])
        # Fewer paired units than items, but a real coefficient nonetheless.
        assert 0 < alpha["de"].n_units < N_ITEMS
        assert alpha["de"].alpha > 0.6

    def test_three_raters_report_three_observers(self) -> None:
        items = study_items()
        truth = synthetic_truth([i.generation_id for i in items], seed=5)
        reports = [run_rater(f"T{i}", items, truth, noise=0.3, seed=100 + i)[2] for i in range(3)]
        alpha = inter_rater_alpha([s for r in reports for s in r.scores])
        assert alpha["de"].n_observers == 3
        # Rho is nan with three raters rather than an averaged pairwise number.
        assert math.isnan(alpha["de"].rho)


class TestReverseCodingSurvivesTheRoundTrip:
    def test_ritualistic_stays_raw_from_sheet_to_rubric_score(self) -> None:
        items = study_items(6)
        truth = {i.generation_id: dict.fromkeys(RUBRIC_DIMENSIONS, 5) for i in items}
        _packet, _rows, report = run_rater("R01", items, truth, noise=0.0, seed=1)
        assert all(s.ritualistic == 5 for s in report.scores)
        # And `to_quality` is what turns that into "worst", nowhere else.
        assert to_quality("ritualistic", 5) == 1

    def test_a_rater_with_the_scale_backwards_is_caught_at_calibration(self) -> None:
        """Caught before sixty ratings are collected under the error."""
        items = study_items(6)
        truth = synthetic_truth([i.generation_id for i in items], seed=6)
        packet = build_packet("BAD", items)
        rows = synthetic_ratings(packet, truth, noise=0.0, seed=1, reverse_ritualistic=True)
        report = ingest_ratings("BAD", rows, packet.assignments)
        check = calibration_check("BAD", report.calibration)
        assert "ritualistic" in check.flagged
        assert not check.ok


class TestConsensus:
    def test_median_across_raters_stays_on_the_scale(self) -> None:
        items = study_items(20)
        truth = synthetic_truth([i.generation_id for i in items], seed=7)
        reports = [run_rater(f"R{i}", items, truth, noise=0.5, seed=200 + i)[2] for i in range(3)]
        consensus = human_consensus([s for r in reports for s in r.scores])

        assert len(consensus) == 20
        for scores in consensus.values():
            for value in scores.values():
                assert value is None or (isinstance(value, int) and 1 <= value <= 5)

    def test_min_raters_gate_blanks_thinly_rated_dimensions(self) -> None:
        items = study_items(10)
        truth = synthetic_truth([i.generation_id for i in items], seed=8)
        _, _, only_one = run_rater("R01", items, truth, noise=0.0, seed=1)
        consensus = human_consensus(only_one.scores, min_raters=2)
        assert all(v is None for scores in consensus.values() for v in scores.values())

    def test_consensus_feeds_the_judge_validity_computation(self) -> None:
        """The join the whole §13 validity section depends on.

        The judge is scored on the same generation ids the raters saw, so the
        two halves of the study meet on `generation_id` with no re-keying.
        """
        items = study_items(40)
        truth = synthetic_truth([i.generation_id for i in items], seed=9)
        reports = [run_rater(f"R{i}", items, truth, noise=0.3, seed=300 + i)[2] for i in range(2)]
        consensus = human_consensus([s for r in reports for s in r.scores])

        judge_results = []
        for item in items:
            scores = {k: truth[item.generation_id][k] for k in RUBRIC_DIMENSIONS}
            judge = LLMJudge(
                client=ReplayClient(outputs=[judge_json(scores)]), temperature=0.0, n_samples=1
            )
            judge_results.append(
                judge.score_text(
                    generation_id=item.generation_id,
                    scenario_text=SCENARIO,
                    response_text=RESPONSE,
                )
            )

        validity = judge_human_validity(judge_results, consensus)
        assert validity["de"].agreement.n_units == 40
        # A judge that reproduces the latent truth agrees strongly with raters
        # who observe it through modest noise.
        assert validity["de"].agreement.alpha > 0.7
        assert validity["ritualistic"].agreement.alpha > 0.7


class TestSingleRaterFallback:
    """v3 §12 option 3: one rater, twice, at least two weeks apart."""

    def _two_occasions(self, *, instability: float, n: int = N_ITEMS):
        items = study_items(n)
        truth = synthetic_truth([i.generation_id for i in items], seed=10)

        first_packet = build_packet("R01", items)
        first_rows = synthetic_ratings(first_packet, truth, noise=0.4, seed=41)
        first = ingest_ratings("R01", first_rows, first_packet.assignments)

        # The retest is independently reshuffled and carries its own rater id.
        retest_packet = build_packet("R01-t2", items, include_calibration=False)
        retest_rows = synthetic_retest_ratings(
            first_rows, retest_packet, first_packet, instability=instability, seed=42
        )
        second = ingest_ratings("R01-t2", retest_rows, retest_packet.assignments)
        return first, second, first_packet, retest_packet

    def test_retest_is_reshuffled_not_replayed_in_order(self) -> None:
        """A second pass in the same order is a memory test, not a retest."""
        _, _, first_packet, retest_packet = self._two_occasions(instability=0.3)
        first_order = [a.generation_id for a in first_packet.assignments if not a.is_calibration]
        retest_order = [a.generation_id for a in retest_packet.assignments]
        assert first_order != retest_order
        assert sorted(first_order) == sorted(retest_order)

    def test_a_stable_rater_reports_high_intra_rater_reliability(self) -> None:
        first, second, _, _ = self._two_occasions(instability=0.0)
        alpha = intra_rater_reliability(first.scores, second.scores)
        assert alpha["de"].n_observers == 2
        assert alpha["de"].alpha == pytest.approx(1.0)

    def test_an_unstable_rater_reports_lower_reliability(self) -> None:
        stable, stable_2, _, _ = self._two_occasions(instability=0.0)
        drifty, drifty_2, _, _ = self._two_occasions(instability=1.0)
        stable_alpha = intra_rater_reliability(stable.scores, stable_2.scores)["de"].alpha
        drifty_alpha = intra_rater_reliability(drifty.scores, drifty_2.scores)["de"].alpha
        assert drifty_alpha < stable_alpha
        assert drifty_alpha < 0.8

    def test_a_consistently_wrong_rater_still_scores_high_here(self) -> None:
        """Why this is reported as intra-rater reliability and never as agreement.

        Test-retest measures stability against oneself. A rater who is
        systematically wrong in the same direction both times looks excellent on
        this metric, and the write-up must not let the number sit in a column
        beside inter-rater alpha without a label.
        """
        items = study_items(30)
        truth = synthetic_truth([i.generation_id for i in items], seed=11)

        p1 = build_packet("W01", items)
        rows1 = synthetic_ratings(p1, truth, noise=0.0, bias=2, seed=1)
        first = ingest_ratings("W01", rows1, p1.assignments)

        p2 = build_packet("W01-t2", items, include_calibration=False)
        rows2 = synthetic_retest_ratings(rows1, p2, p1, instability=0.0, seed=2)
        second = ingest_ratings("W01-t2", rows2, p2.assignments)

        retest = intra_rater_reliability(first.scores, second.scores)["de"].alpha
        assert retest == pytest.approx(1.0)

    def test_the_same_rater_id_twice_is_refused(self) -> None:
        """Two occasions sharing an id would collapse into one observer."""
        items = study_items(10)
        truth = synthetic_truth([i.generation_id for i in items], seed=12)
        _, _, report = run_rater("R01", items, truth, noise=0.2, seed=1)
        with pytest.raises(ValueError, match="different rater ids"):
            intra_rater_reliability(report.scores, report.scores)


class TestSyntheticRatersUseOnlyWhatARaterSees:
    def test_rows_carry_nothing_but_the_label_and_the_scores(self) -> None:
        """A leak in the export would surface here as extra keys in the rows."""
        items = study_items(6)
        truth = synthetic_truth([i.generation_id for i in items], seed=13)
        packet = build_packet("R01", items)
        rows = synthetic_ratings(packet, truth, noise=0.2, seed=1)
        allowed = {"blind_label", *RUBRIC_DIMENSIONS, "safety_flags", "notes"}
        for row in rows:
            assert set(row) <= allowed
            assert not str(row["blind_label"]).startswith("gen-")

    def test_the_returned_sheet_matches_the_blank_sheet_columns(self) -> None:
        items = study_items(6)
        packet = build_packet("R01", items)
        header = next(csv.reader(io.StringIO(rating_sheet_csv(packet.items))))
        for key in RUBRIC_DIMENSIONS:
            assert key in header
        assert header[0] == "blind_label"


class TestScoresByRater:
    def test_indexes_by_rater_then_generation(self) -> None:
        items = study_items(6)
        truth = synthetic_truth([i.generation_id for i in items], seed=14)
        _, _, r1 = run_rater("R01", items, truth, noise=0.0, seed=1)
        indexed = scores_by_rater(r1.scores)
        assert set(indexed) == {"R01"}
        assert len(indexed["R01"]) == 6
        assert set(next(iter(indexed["R01"].values()))) == set(RUBRIC_DIMENSIONS)
