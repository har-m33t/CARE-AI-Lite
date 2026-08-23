"""Every build-plan v3 §13 metric, per dimension."""

from __future__ import annotations

import math
import random

import pytest

from carelite.eval.judge import (
    LLMJudge,
    OptionOrder,
    ReplayClient,
    build_validation_report,
    classify_dimension,
    judge_among_raters_alpha,
    judge_human_validity,
    positional_bias,
    sample_spans_for_review,
    self_consistency,
    span_grounding_audit,
    span_support_rate,
)
from carelite.eval.judge.validation import (
    MIN_ALPHA_FOR_CONFIRMATORY,
    MIN_RHO_FOR_CONFIRMATORY,
    MIN_UNITS_FOR_CONFIRMATORY,
    N_SPANS_TO_REVIEW,
    EvidenceStatus,
    SpanReviewVerdict,
)
from carelite.types import RUBRIC_DIMENSIONS

from .conftest import RESPONSE, SCENARIO, judge_json


def judged(
    outputs: list[str],
    *,
    generation_id: str = "gen-0001",
    order: OptionOrder = OptionOrder.ASCENDING,
):
    judge = LLMJudge(
        client=ReplayClient(outputs=outputs),
        temperature=0.7,
        n_samples=len(outputs),
        order=order,
    )
    return judge.score_text(
        generation_id=generation_id, scenario_text=SCENARIO, response_text=RESPONSE
    )


def flat(score: int, **overrides: int) -> dict[str, int]:
    return {**dict.fromkeys(RUBRIC_DIMENSIONS, score), **overrides}


# ---------------------------------------------------------------- 1. stability


class TestSelfConsistency:
    def test_unanimous_samples_report_zero_variance(self) -> None:
        stats = self_consistency([judged([judge_json(4)] * 5)])
        assert stats["de"].n_generations == 1
        assert stats["de"].mean_variance == 0.0
        assert stats["de"].pct_unanimous == 1.0
        assert stats["de"].pct_range_ge_2 == 0.0

    def test_disagreeing_samples_report_variance_and_range(self) -> None:
        result = judged([judge_json(flat(3, de=v)) for v in (1, 2, 3, 4, 5)])
        stats = self_consistency([result])
        assert stats["de"].mean_variance == pytest.approx(2.5)
        assert stats["de"].mean_range == 4.0
        assert stats["de"].pct_range_ge_2 == 1.0
        # Other dimensions were unanimous in the same run.
        assert stats["name"].mean_variance == 0.0

    def test_single_pass_results_contribute_nothing(self) -> None:
        """n=1 must not be reported as evidence of perfect stability."""
        stats = self_consistency([judged([judge_json(4)])])
        assert stats["de"].n_generations == 0
        assert math.isnan(stats["de"].mean_variance)

    def test_per_dimension_not_pooled(self) -> None:
        """The whole point of §13: naturalness may be unstable while `ib` is not."""
        result = judged([judge_json(flat(3, naturalness=v)) for v in (1, 5, 1, 5, 3)])
        stats = self_consistency([result])
        assert stats["naturalness"].mean_variance > stats["ib"].mean_variance


# ------------------------------------------------------------ 2. positional bias


class TestPositionalBias:
    def test_identical_scores_in_both_orders_show_no_bias(self) -> None:
        asc = [judged([judge_json(4)], generation_id=f"g{i}") for i in range(3)]
        desc = [
            judged([judge_json(4)], generation_id=f"g{i}", order=OptionOrder.DESCENDING)
            for i in range(3)
        ]
        bias = positional_bias(asc, desc)
        assert bias["de"].n_paired == 3
        assert bias["de"].mean_signed_delta == 0.0
        assert bias["de"].pct_shift_ge_1 == 0.0

    def test_a_lenient_reversed_arm_shows_a_positive_delta(self) -> None:
        asc = [judged([judge_json(flat(3))], generation_id=f"g{i}") for i in range(4)]
        desc = [
            judged([judge_json(flat(5))], generation_id=f"g{i}", order=OptionOrder.DESCENDING)
            for i in range(4)
        ]
        bias = positional_bias(asc, desc)
        assert bias["de"].mean_signed_delta == pytest.approx(2.0)
        assert bias["de"].mean_abs_delta == pytest.approx(2.0)
        assert bias["de"].pct_shift_ge_1 == 1.0

    def test_delta_is_on_the_quality_scale_so_ritualistic_points_the_same_way(self) -> None:
        """Reversing the anchors made the judge score the response as *better*.

        On the raw scale a friendlier `ritualistic` score is a *lower* number,
        so a raw-scale table would show this dimension moving opposite to its
        ten neighbours and read as a bug in the reversed arm.
        """
        asc = [
            judged([judge_json(flat(3, ritualistic=4))], generation_id=f"g{i}") for i in range(3)
        ]
        desc = [
            judged(
                [judge_json(flat(5, ritualistic=2))],
                generation_id=f"g{i}",
                order=OptionOrder.DESCENDING,
            )
            for i in range(3)
        ]
        bias = positional_bias(asc, desc)
        assert bias["de"].mean_signed_delta > 0
        assert bias["ritualistic"].mean_signed_delta > 0

    def test_a_rejected_score_removes_the_pair_rather_than_scoring_zero_delta(self) -> None:
        asc = [judged([judge_json(4, spans={"de": "not in the response"})], generation_id="g0")]
        desc = [judged([judge_json(4)], generation_id="g0", order=OptionOrder.DESCENDING)]
        bias = positional_bias(asc, desc)
        assert bias["de"].n_paired == 0
        assert bias["name"].n_paired == 1

    def test_no_reversed_arm_yet_reports_nan_not_zero(self) -> None:
        bias = positional_bias([judged([judge_json(4)])], [])
        assert bias["de"].n_paired == 0
        assert math.isnan(bias["de"].mean_signed_delta)


# ------------------------------------------------------------- 3. span grounding


class TestSpanGroundingAudit:
    def test_all_spans_locatable(self) -> None:
        audit = span_grounding_audit([judged([judge_json(4)])])
        assert audit.n_attempted == len(RUBRIC_DIMENSIONS)
        assert audit.n_rejected == 0
        assert audit.admitted_rate == 1.0
        assert audit.exact_rate == 1.0
        assert audit.presented_only_rate == 0.0

    def test_fabrication_and_non_compliance_are_counted_separately(self) -> None:
        """`span_not_found` is a hallucination rate; `missing_span` is compliance.

        Rolling them together would hide which of the two problems we have, and
        they have different fixes — one is a prompt change, the other is a
        model-capability limit.
        """
        import json

        raw = json.dumps(
            {
                "scores": {
                    "de": {"score": 4, "span": "a quote the model invented"},
                    "name": {"score": 4},
                    **{
                        k: {"score": 3, "span": "It sounds like you're frightened"}
                        for k in RUBRIC_DIMENSIONS
                        if k not in {"de", "name"}
                    },
                }
            }
        )
        audit = span_grounding_audit([judged([raw])])
        assert audit.reasons["span_not_found"] == 1
        assert audit.reasons["missing_span"] == 1
        assert audit.n_rejected == 2
        assert audit.per_dimension["de"] == 0.0
        assert audit.per_dimension["respect"] == 1.0

    def test_typography_normalised_matches_lower_the_exact_rate(self) -> None:
        quoted = "It sounds like you\u2019re frightened"  # curly apostrophe
        audit = span_grounding_audit([judged([judge_json(4, spans={"name": quoted})])])
        assert audit.admitted_rate == 1.0
        assert audit.exact_rate < 1.0


class TestSpanReview:
    def _results(self, n: int = 6) -> list:
        return [judged([judge_json(4)], generation_id=f"gen-{i:03d}") for i in range(n)]

    def test_draws_the_requested_number(self) -> None:
        results = self._results()
        items = sample_spans_for_review(
            results, {r.generation_id: RESPONSE for r in results}, n=N_SPANS_TO_REVIEW
        )
        assert len(items) == N_SPANS_TO_REVIEW

    def test_is_stratified_across_dimensions(self) -> None:
        """An unstratified draw leaves dimensions with zero spans reviewed.

        "Spans support their scores 87% of the time" is not a useful sentence if
        the failures were concentrated in a dimension the sample happened to miss.
        """
        results = self._results()
        items = sample_spans_for_review(results, {r.generation_id: RESPONSE for r in results}, n=22)
        counts = {k: sum(1 for i in items if i.dimension == k) for k in RUBRIC_DIMENSIONS}
        assert set(counts.values()) == {2}

    def test_is_reproducible_and_seed_sensitive(self) -> None:
        results = self._results()
        responses = {r.generation_id: RESPONSE for r in results}
        a = [i.item_id for i in sample_spans_for_review(results, responses, n=15, seed=1)]
        b = [i.item_id for i in sample_spans_for_review(results, responses, n=15, seed=1)]
        c = [i.item_id for i in sample_spans_for_review(results, responses, n=15, seed=2)]
        assert a == b
        assert a != c

    def test_rejected_dimensions_are_not_offered_for_review(self) -> None:
        results = [judged([judge_json(4, spans={"de": "invented"})], generation_id="g0")]
        items = sample_spans_for_review(results, {"g0": RESPONSE}, n=30)
        assert all(i.dimension != "de" for i in items)

    def test_reviewer_gets_the_whole_response(self) -> None:
        results = self._results(1)
        items = sample_spans_for_review(results, {"gen-000": RESPONSE}, n=1)
        assert items[0].response == RESPONSE
        assert items[0].span in RESPONSE


class TestSpanSupportRate:
    def test_rate_and_interval(self) -> None:
        verdicts = [SpanReviewVerdict(f"i{i}", "de", supports=i < 27) for i in range(30)]
        report = span_support_rate(verdicts)
        assert report.n_reviewed == 30
        assert report.n_supported == 27
        assert report.support_rate == pytest.approx(0.9)
        assert 0.0 < report.ci_low < 0.9 < report.ci_high < 1.0

    def test_perfect_score_still_has_an_upper_bounded_interval(self) -> None:
        """At n=30 a bare 100% reads far more precise than it is."""
        report = span_support_rate([SpanReviewVerdict(f"i{i}", "de", True) for i in range(30)])
        assert report.support_rate == 1.0
        assert report.ci_low < 1.0

    def test_per_dimension_breakdown(self) -> None:
        verdicts = [
            SpanReviewVerdict("a", "naturalness", False),
            SpanReviewVerdict("b", "naturalness", False),
            SpanReviewVerdict("c", "ib", True),
        ]
        report = span_support_rate(verdicts)
        assert report.per_dimension["naturalness"] == 0.0
        assert report.per_dimension["ib"] == 1.0

    def test_empty_review_is_nan_not_zero(self) -> None:
        assert math.isnan(span_support_rate([]).support_rate)


# ------------------------------------------------------------------ 4. validity


def _agreeing_corpus(n: int, noise: int = 0, seed: int = 5):
    """n judged generations plus a human consensus that tracks them."""
    rng = random.Random(seed)
    results = []
    consensus: dict[str, dict[str, int | None]] = {}
    for i in range(n):
        gid = f"gen-{i:03d}"
        scores = {k: rng.randint(1, 5) for k in RUBRIC_DIMENSIONS}
        results.append(judged([judge_json(scores)], generation_id=gid))
        consensus[gid] = {
            k: max(
                1, min(5, v + (rng.choice((-1, 1)) if noise and rng.random() < noise / 10 else 0))
            )
            for k, v in scores.items()
        }
    return results, consensus


class TestJudgeHumanValidity:
    def test_perfect_agreement_is_confirmatory(self) -> None:
        results, consensus = _agreeing_corpus(40)
        validity = judge_human_validity(results, consensus)
        assert validity["de"].agreement.n_units == 40
        assert validity["de"].agreement.alpha == pytest.approx(1.0)
        assert validity["de"].status is EvidenceStatus.CONFIRMATORY

    def test_random_human_scores_land_exploratory(self) -> None:
        results, _ = _agreeing_corpus(60)
        rng = random.Random(99)
        junk = {r.generation_id: {k: rng.randint(1, 5) for k in RUBRIC_DIMENSIONS} for r in results}
        validity = judge_human_validity(results, junk)
        assert validity["naturalness"].status is EvidenceStatus.EXPLORATORY
        assert validity["naturalness"].agreement.alpha < MIN_ALPHA_FOR_CONFIRMATORY

    def test_small_samples_are_exploratory_however_good_the_coefficient(self) -> None:
        results, consensus = _agreeing_corpus(MIN_UNITS_FOR_CONFIRMATORY - 1)
        validity = judge_human_validity(results, consensus)
        assert validity["de"].agreement.alpha == pytest.approx(1.0)
        assert validity["de"].status is EvidenceStatus.EXPLORATORY

    def test_no_human_data_yet_is_exploratory_everywhere(self) -> None:
        results, _ = _agreeing_corpus(40)
        validity = judge_human_validity(results, {})
        assert all(v.status is EvidenceStatus.EXPLORATORY for v in validity.values())
        assert all(v.agreement.n_units == 0 for v in validity.values())

    def test_ritualistic_agreement_survives_the_reverse_coding(self) -> None:
        """Both sides are canonicalised with `to_quality`, so agreement is positive.

        If one side were raw and the other quality-coded, this would come out
        strongly negative and could be misreported as "the judge disagrees about
        ritual". Pinning it here is the regression test for that mistake.
        """
        results, consensus = _agreeing_corpus(40)
        validity = judge_human_validity(results, consensus)
        assert validity["ritualistic"].agreement.alpha == pytest.approx(1.0)
        assert validity["ritualistic"].agreement.rho > 0.9


class TestJudgeAmongRaters:
    def test_a_judge_that_behaves_like_a_rater_barely_moves_alpha(self) -> None:
        results, consensus = _agreeing_corpus(40)
        humans = {"R01": consensus, "R02": consensus}
        with_judge = judge_among_raters_alpha(results, humans)
        assert with_judge["de"] == pytest.approx(1.0)

    def test_a_judge_that_disagrees_drags_the_panel_down(self) -> None:
        results, consensus = _agreeing_corpus(40)
        rng = random.Random(3)
        humans = {
            "R01": {g: {k: rng.randint(1, 5) for k in RUBRIC_DIMENSIONS} for g in consensus},
        }
        humans["R02"] = humans["R01"]
        assert judge_among_raters_alpha(results, humans)["de"] < 0.5


class TestPreSpecifiedThreshold:
    def test_both_gates_must_clear(self) -> None:
        n = MIN_UNITS_FOR_CONFIRMATORY
        assert classify_dimension(0.9, 0.9, n) is EvidenceStatus.CONFIRMATORY
        assert (
            classify_dimension(MIN_ALPHA_FOR_CONFIRMATORY - 0.01, 0.9, n)
            is EvidenceStatus.EXPLORATORY
        )
        assert (
            classify_dimension(0.9, MIN_RHO_FOR_CONFIRMATORY - 0.01, n)
            is EvidenceStatus.EXPLORATORY
        )

    def test_boundary_values_pass(self) -> None:
        assert (
            classify_dimension(
                MIN_ALPHA_FOR_CONFIRMATORY, MIN_RHO_FOR_CONFIRMATORY, MIN_UNITS_FOR_CONFIRMATORY
            )
            is EvidenceStatus.CONFIRMATORY
        )

    def test_undefined_agreement_is_not_evidence_of_agreement(self) -> None:
        assert classify_dimension(math.nan, 0.9, 100) is EvidenceStatus.EXPLORATORY
        assert classify_dimension(0.9, math.nan, 100) is EvidenceStatus.EXPLORATORY


# -------------------------------------------------------------------- report


class TestValidationReport:
    def test_assembles_every_metric(self) -> None:
        results, consensus = _agreeing_corpus(40)
        reversed_arm = [
            judged([judge_json(4)], generation_id=r.generation_id, order=OptionOrder.DESCENDING)
            for r in results
        ]
        verdicts = [SpanReviewVerdict(f"i{i}", "de", supports=i < 28) for i in range(30)]

        report = build_validation_report(
            validation_results=results,
            responses={r.generation_id: RESPONSE for r in results},
            human_consensus=consensus,
            reversed_results=reversed_arm,
            span_verdicts=verdicts,
            generator_model="gemma4:12b",
        )
        assert report.n_generations == 40
        assert set(report.self_consistency) == set(RUBRIC_DIMENSIONS)
        assert set(report.positional_bias) == set(RUBRIC_DIMENSIONS)
        assert set(report.validity) == set(RUBRIC_DIMENSIONS)
        assert report.grounding.admitted_rate == 1.0
        assert report.span_support is not None

        rendered = report.render()
        assert "INDEPENDENCE" in rendered
        assert "gemma4:12b" in rendered
        for key in RUBRIC_DIMENSIONS:
            assert key in rendered

    def test_report_names_the_exploratory_dimensions(self) -> None:
        results, _ = _agreeing_corpus(40)
        report = build_validation_report(
            validation_results=results,
            responses={r.generation_id: RESPONSE for r in results},
            human_consensus=None,
        )
        assert set(report.exploratory_dimensions) == set(RUBRIC_DIMENSIONS)
        assert report.confirmatory_dimensions == []
        assert "EXPLORATORY" in report.render()

    def test_unrun_spot_check_is_stated_not_omitted(self) -> None:
        results, _ = _agreeing_corpus(3)
        report = build_validation_report(
            validation_results=results,
            responses={r.generation_id: RESPONSE for r in results},
        )
        assert "NOT YET RUN" in report.render()
