"""Known-answer tests for the degenerate-dimension diagnostic.

Every distribution here is constructed so the right answer is arithmetic rather
than judgement: a dimension built to be constant must come back degenerate, a
dimension built bimodal must not, and the standard deviations are computable by
hand from the counts.

The bimodal case is the one that matters most. It is the case that broke the
first version of this module -- a modal-share rule called it degenerate -- so it
is pinned here, with the reasoning in the assertion, to stop the rule drifting
back.
"""

from __future__ import annotations

import pandas as pd
import pytest

from carelite.stats.instrument import (
    MIN_SCORED_FOR_CLASSIFICATION,
    MIN_SD,
    RITUAL_MECHANISM_DIMENSIONS,
    Discrimination,
    describe_dimensions,
    instrument_report,
    measure_testability,
    threshold_sensitivity,
)
from carelite.stats.measures import NURSE_COMPOSITE, measure
from carelite.types import Condition


def _frame(values_by_dimension: dict[str, list[int]]) -> pd.DataFrame:
    """A minimal long frame: `{dimension: [raw, raw, ...]}`, one row per value."""
    rows: list[dict[str, object]] = []
    for dimension, values in values_by_dimension.items():
        for i, raw in enumerate(values):
            rows.append(
                {
                    "generation_id": f"gen-{i:04d}",
                    "scenario_id": f"SC-{i % 20:03d}",
                    "condition": str(Condition.A),
                    "rater_type": "llm_judge",
                    "rater_id": "judge-median",
                    "dimension": dimension,
                    "raw": raw,
                }
            )
    return pd.DataFrame(rows)


class TestClassification:
    def test_a_constant_dimension_is_degenerate(self) -> None:
        """sd is exactly 0, which is the clearest possible case."""
        frame = _frame({"understand": [3] * 100})
        (dist,) = describe_dimensions(frame, dimensions=["understand"])
        assert dist.sd == 0.0
        assert dist.modal_share == 1.0
        assert dist.distinct == 1
        assert dist.discrimination is Discrimination.DEGENERATE

    def test_a_nearly_constant_dimension_is_degenerate(self) -> None:
        """99 of 100 on one value: this is `ritualistic` on the real run."""
        frame = _frame({"understand": [1] * 99 + [3]})
        (dist,) = describe_dimensions(frame, dimensions=["understand"])
        assert dist.sd < MIN_SD
        assert dist.discrimination is Discrimination.DEGENERATE

    def test_a_bimodal_dimension_is_NOT_degenerate_despite_a_high_modal_share(
        self,
    ) -> None:
        """The case that broke the first rule. Concentration is not degeneracy.

        77 scores of 1 and 23 of 5 puts 77% of the mass on one value — over any
        plausible modal-share cut — while using the full range of the scale and
        carrying a standard deviation near 1.7. A paired test on this dimension
        has plenty to rank. `name` on the holdout run is exactly this shape.
        """
        frame = _frame({"name": [1] * 77 + [5] * 23})
        (dist,) = describe_dimensions(frame, dimensions=["name"])
        assert dist.modal_share == pytest.approx(0.77)
        assert dist.modal_share > 0.75, "the modal-share rule would have flagged this"
        assert dist.sd > MIN_SD
        assert dist.discrimination is Discrimination.DISCRIMINATING

    def test_a_spread_dimension_is_discriminating(self) -> None:
        frame = _frame({"explore": [1, 2, 3, 4, 5] * 20})
        (dist,) = describe_dimensions(frame, dimensions=["explore"])
        assert dist.distinct == 5
        assert dist.discrimination is Discrimination.DISCRIMINATING

    def test_too_few_items_is_unknown_not_a_verdict(self) -> None:
        """An unmeasured distribution must not be reported as either answer."""
        frame = _frame({"understand": [3] * (MIN_SCORED_FOR_CLASSIFICATION - 1)})
        (dist,) = describe_dimensions(frame, dimensions=["understand"])
        assert dist.discrimination is Discrimination.UNKNOWN
        assert not dist.is_degenerate
        assert "not classified" in dist.why


class TestReverseCoding:
    def test_ritualistic_is_described_on_the_quality_scale(self) -> None:
        """A raw 1 on `ritualistic` is quality 5. Both means are reported.

        The whole package aggregates `quality`, so the diagnostic must describe
        `quality` too — a table that mixed the two scales would call the same
        dimension a floor and a ceiling in different rows.
        """
        frame = _frame({"ritualistic": [1] * 100})
        (dist,) = describe_dimensions(frame, dimensions=["ritualistic"])
        assert dist.mean_raw == pytest.approx(1.0)
        assert dist.mean_quality == pytest.approx(5.0)
        assert dist.modal_value_quality == 5

    def test_reversal_does_not_change_the_verdict(self) -> None:
        """Degeneracy is a property of spread, and 6 - x has the spread of x."""
        raw = [1] * 90 + [2] * 10
        (forward,) = describe_dimensions(_frame({"understand": raw}), dimensions=["understand"])
        (reversed_,) = describe_dimensions(_frame({"ritualistic": raw}), dimensions=["ritualistic"])
        assert forward.sd == pytest.approx(reversed_.sd)
        assert forward.discrimination is reversed_.discrimination


class TestThresholdGuard:
    def test_a_wide_gap_reports_stable(self) -> None:
        """Constant vs fully-spread: no cut in the range can separate them differently."""
        frame = _frame({"understand": [3] * 100, "explore": [1, 2, 3, 4, 5] * 20})
        distributions = describe_dimensions(frame, dimensions=["understand", "explore"])
        stable, changes = threshold_sensitivity(distributions)
        assert stable
        assert changes == []

    def test_a_dimension_sitting_on_the_cut_reports_unstable(self) -> None:
        """The guard has to be able to fail, or it is not a guard.

        A dimension whose sd lands inside the swept range moves in and out of
        the degenerate set as the cut moves, and the report must say so instead
        of quoting the default as though it were settled.
        """
        borderline = [1] * 85 + [3] * 15  # sd near 0.72, inside the 0.60-0.95 sweep
        distributions = describe_dimensions(
            _frame({"understand": borderline}), dimensions=["understand"]
        )
        assert 0.60 < distributions[0].sd < 0.95
        stable, changes = threshold_sensitivity(distributions)
        assert not stable
        assert changes, "an unstable classification must name the cuts where it moves"


class TestMeasureTestability:
    def test_a_composite_with_one_degenerate_dimension_is_still_testable(self) -> None:
        statuses = dict.fromkeys(NURSE_COMPOSITE.dimensions, Discrimination.DISCRIMINATING)
        statuses["support"] = Discrimination.DEGENERATE
        result = measure_testability(NURSE_COMPOSITE, statuses)
        assert result.testable
        assert result.attenuated_by == ("support",)
        assert "ATTENUATED" in result.note

    def test_a_composite_with_every_dimension_degenerate_is_not_testable(self) -> None:
        statuses = dict.fromkeys(NURSE_COMPOSITE.dimensions, Discrimination.DEGENERATE)
        result = measure_testability(NURSE_COMPOSITE, statuses)
        assert not result.testable
        assert "INSTRUMENT-LIMITED" in result.note

    def test_a_single_dimension_measure_follows_its_one_dimension(self) -> None:
        statuses = {"naturalness": Discrimination.DEGENERATE}
        result = measure_testability(measure("naturalness"), statuses)
        assert not result.testable
        assert result.degenerate_dimensions == ("naturalness",)

    def test_a_clean_measure_carries_no_note(self) -> None:
        statuses = dict.fromkeys(NURSE_COMPOSITE.dimensions, Discrimination.DISCRIMINATING)
        result = measure_testability(NURSE_COMPOSITE, statuses)
        assert result.testable
        assert result.note == ""
        assert result.attenuated_by == ()


class TestRitualMechanism:
    def test_both_dimensions_degenerate_makes_the_prediction_untestable(self) -> None:
        """v3's naturalness prediction is carried jointly by outcome and mechanism."""
        frame = _frame({"naturalness": [3] * 100, "ritualistic": [1] * 100})
        report = instrument_report(frame, dimensions=list(RITUAL_MECHANISM_DIMENSIONS))
        assert not report.ritual_mechanism_testable
        text = report.render()
        assert "THE RITUAL MECHANISM CANNOT BE TESTED" in text
        assert "null result about" in text

    def test_one_surviving_dimension_keeps_the_prediction_testable(self) -> None:
        frame = _frame({"naturalness": [1, 2, 3, 4, 5] * 20, "ritualistic": [1] * 100})
        report = instrument_report(frame, dimensions=list(RITUAL_MECHANISM_DIMENSIONS))
        assert report.ritual_mechanism_testable
        assert "CANNOT BE TESTED" not in report.render()


class TestRender:
    def test_the_report_is_labelled_post_hoc(self) -> None:
        """It was invented after the data existed and the output has to say so."""
        report = instrument_report(_frame({"understand": [3] * 100}))
        assert not report.prespecified
        assert "NOT PLANNED IN ADVANCE" in report.render()

    def test_the_floor_mechanism_is_stated_with_the_verdict(self) -> None:
        """Why a bigger sample does not help belongs next to the finding."""
        text = instrument_report(_frame({"understand": [3] * 100})).render()
        assert "0.878" in text
        assert "More scenarios would narrow the intervals around the same floor." in text

    def test_every_dimension_appears_with_its_numbers(self) -> None:
        """The cut must be movable by a reader without re-running anything."""
        frame = _frame({"understand": [3] * 100, "explore": [1, 2, 3, 4, 5] * 20})
        text = instrument_report(frame, dimensions=["understand", "explore"]).render()
        for dimension in ("understand", "explore"):
            assert dimension in text
        assert "sd" in text and "modal" in text

    def test_an_empty_frame_does_not_raise(self) -> None:
        report = instrument_report(pd.DataFrame(columns=["dimension", "raw"]))
        assert report.degenerate == ()
        assert isinstance(report.render(), str)
