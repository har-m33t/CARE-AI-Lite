"""The three §8.5 reruns, and the flip detection that is the point of them.

The fixtures here are built so that a conclusion is *known in advance* to move
under one rerun and not under another. A sensitivity analysis that cannot detect
a planted flip cannot be trusted to detect a real one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from carelite.stats.primary import run_family
from carelite.stats.sensitivity import (
    DEFAULT_MAX_PCT_RANGE_GE_2,
    backend_equivalence_check,
    compare_conclusions,
    conclusions,
    read_backend_equivalence,
    retrieval_contrast,
    run_all_sensitivity,
    scenario_judge_consistency,
    sensitivity_crag_fallback,
    sensitivity_gate_blocked,
    sensitivity_judge_consistency,
    sensitivity_rater_type,
)
from tests.unit.stats.conftest import constant_scores, make_long


@pytest.fixture
def base_long(nurse_dimensions: tuple[str, ...]) -> pd.DataFrame:
    """B beats A on all 20 scenarios: a conclusion that should survive a rerun."""
    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(20):
        scenario = f"SC-{i:03d}"
        for sample in range(3):
            scores[(scenario, "A", sample)] = constant_scores(nurse_dimensions, 2)
            scores[(scenario, "B", sample)] = constant_scores(nurse_dimensions, 4)
    return make_long(scores=scores)


# ---------------------------------------------------------------------------
# What counts as a conclusion
# ---------------------------------------------------------------------------


def test_conclusions_reduce_a_family_to_significance_and_direction(
    base_long: pd.DataFrame,
) -> None:
    family = run_family(base_long, include_friedman=False, n_boot=200)
    reduced = conclusions(family)
    primary = reduced["primary_nurse_A_vs_B"]
    assert primary.direction == "<"
    assert primary.significant is True
    assert primary.n_scenarios == 20


def test_an_identical_rerun_produces_no_flips(base_long: pd.DataFrame) -> None:
    family = run_family(base_long, include_friedman=False, n_boot=200)
    assert compare_conclusions(family, family) == ()


def test_a_reversed_direction_is_a_flip(
    base_long: pd.DataFrame, nurse_dimensions: tuple[str, ...]
) -> None:
    reversed_scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(20):
        scenario = f"SC-{i:03d}"
        for sample in range(3):
            reversed_scores[(scenario, "A", sample)] = constant_scores(nurse_dimensions, 4)
            reversed_scores[(scenario, "B", sample)] = constant_scores(nurse_dimensions, 2)
    base = run_family(base_long, include_friedman=False, n_boot=200)
    variant = run_family(make_long(scores=reversed_scores), include_friedman=False, n_boot=200)
    flips = compare_conclusions(base, variant)
    assert len(flips) == 1
    assert "direction moved < -> >" in flips[0].what


def test_a_lost_significance_is_a_flip(
    base_long: pd.DataFrame, nurse_dimensions: tuple[str, ...]
) -> None:
    """Three scenarios cannot reach significance after a family-of-eight correction."""
    small: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(3):
        scenario = f"SC-{i:03d}"
        small[(scenario, "A", 0)] = constant_scores(nurse_dimensions, 2)
        small[(scenario, "B", 0)] = constant_scores(nurse_dimensions, 4)
    base = run_family(base_long, include_friedman=False, n_boot=200)
    variant = run_family(make_long(scores=small), include_friedman=False, n_boot=200)
    flips = compare_conclusions(base, variant)
    assert any("lost significance" in f.what for f in flips)


def test_a_comparison_that_vanishes_from_the_rerun_is_a_flip(
    base_long: pd.DataFrame, nurse_dimensions: tuple[str, ...]
) -> None:
    base = run_family(base_long, include_friedman=False, n_boot=200)
    only_a = {(f"SC-{i:03d}", "A", 0): constant_scores(nurse_dimensions, 2) for i in range(20)}
    variant = run_family(make_long(scores=only_a), include_friedman=False, n_boot=200)
    flips = compare_conclusions(base, variant)
    assert any("not computable in the rerun" in f.what for f in flips)


def test_a_shrinking_effect_that_stays_significant_is_not_called_a_flip(
    nurse_dimensions: tuple[str, ...],
) -> None:
    def frame(gap: int) -> pd.DataFrame:
        scores: dict[tuple[str, str, int], dict[str, int]] = {}
        for i in range(20):
            scenario = f"SC-{i:03d}"
            scores[(scenario, "A", 0)] = constant_scores(nurse_dimensions, 2)
            scores[(scenario, "B", 0)] = constant_scores(nurse_dimensions, 2 + gap)
        return make_long(scores=scores)

    base = run_family(frame(3), include_friedman=False, n_boot=200)
    variant = run_family(frame(1), include_friedman=False, n_boot=200)
    assert compare_conclusions(base, variant) == ()


# ---------------------------------------------------------------------------
# (a) rater type
# ---------------------------------------------------------------------------


def test_rater_sensitivity_cannot_run_with_one_rater_type(base_long: pd.DataFrame) -> None:
    base = run_family(base_long, include_friedman=False, n_boot=200)
    assert sensitivity_rater_type(base_long, base, n_boot=200) == ()


def test_rater_sensitivity_splits_judge_from_human(
    base_long: pd.DataFrame, nurse_dimensions: tuple[str, ...]
) -> None:
    """Humans rate the opposite way here, so the human-only rerun must flip."""
    human_scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(20):
        scenario = f"SC-{i:03d}"
        human_scores[(scenario, "A", 0)] = constant_scores(nurse_dimensions, 5)
        human_scores[(scenario, "B", 0)] = constant_scores(nurse_dimensions, 1)
    combined = pd.concat(
        [base_long, make_long(scores=human_scores, rater_type="human", rater_id="R1")]
    )
    base = run_family(combined, include_friedman=False, rater_type="llm_judge", n_boot=200)
    runs = sensitivity_rater_type(combined, base, n_boot=200)
    assert {r.name for r in runs} == {"(a) human-only ratings", "(a) llm_judge-only ratings"}
    human_run = next(r for r in runs if "human" in r.name)
    assert not human_run.conclusions_hold
    assert any("direction moved" in f.what for f in human_run.flips)
    assert human_run.caveats, "the human sample is smaller than the holdout; say so"


# ---------------------------------------------------------------------------
# (b) CRAG fallback
# ---------------------------------------------------------------------------


def test_crag_sensitivity_excludes_fallback_generations(
    nurse_dimensions: tuple[str, ...],
) -> None:
    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    fell_back: list[tuple[str, str, int]] = []
    for i in range(20):
        scenario = f"SC-{i:03d}"
        for sample in range(3):
            scores[(scenario, "B", sample)] = constant_scores(nurse_dimensions, 3)
            # Fallback samples look exactly like B; the real retrieval sample is better.
            value = 3 if sample < 2 else 5
            scores[(scenario, "C", sample)] = constant_scores(nurse_dimensions, value)
            if sample < 2:
                fell_back.append((scenario, "C", sample))
    long = make_long(scores=scores, fell_back=fell_back)
    base = run_family(long, include_friedman=False, n_boot=200)
    run = sensitivity_crag_fallback(long, base, n_boot=200)

    assert "the 40 generations where" in run.specification
    assert any("40 generations excluded" in n for n in run.family.notes)
    # With fallback turns in, C's cell mean is (3+3+5)/3; with them out it is 5.
    rerun = run.family.by_key("secondary2_nurse_B_vs_C")
    assert rerun is not None
    assert rerun.effects.hodges_lehmann.point == pytest.approx(-2.0)
    original = base.by_key("secondary2_nurse_B_vs_C")
    assert original is not None
    assert original.effects.hodges_lehmann.point == pytest.approx(-2.0 / 3.0)


def test_crag_sensitivity_is_a_no_op_when_nothing_fell_back(
    base_long: pd.DataFrame,
) -> None:
    base = run_family(base_long, include_friedman=False, n_boot=200)
    run = sensitivity_crag_fallback(base_long, base, n_boot=200)
    assert run.conclusions_hold
    assert "the 0 generations where" in run.specification


def test_crag_sensitivity_survives_a_frame_without_the_column(
    base_long: pd.DataFrame,
) -> None:
    base = run_family(base_long, include_friedman=False, n_boot=200)
    run = sensitivity_crag_fallback(base_long.drop(columns=["fell_back_to_b"]), base, n_boot=200)
    assert run.conclusions_hold


# ---------------------------------------------------------------------------
# (c) judge self-consistency
# ---------------------------------------------------------------------------


def _judge_samples(unstable: set[str], partially_unstable: set[str] = frozenset()) -> pd.DataFrame:
    """Five samples per (generation, dimension) across two conditions.

    A scenario in `unstable` spans three scale points in both conditions, so its
    `pct_range_ge_2` is 1.0. A scenario in `partially_unstable` does so in one
    condition only, giving 0.5 — which is what makes a threshold sweep able to
    separate them, and is the realistic case.
    """
    rows: list[dict[str, object]] = []
    for i in range(20):
        scenario = f"SC-{i:03d}"
        for condition in ("A", "B"):
            wobbles = scenario in unstable or (scenario in partially_unstable and condition == "A")
            for sample_idx in range(5):
                raw = (1 if sample_idx % 2 else 4) if wobbles else 2
                rows.append(
                    {
                        "generation_id": f"{scenario}-{condition}-0",
                        "scenario_id": scenario,
                        "condition": condition,
                        "rater_id": "gpt-oss:20b",
                        "rater_sample_idx": sample_idx,
                        "dimension": "name",
                        "raw": raw,
                    }
                )
    return pd.DataFrame(rows)


def test_scenario_consistency_flags_only_the_unstable_scenarios() -> None:
    samples = _judge_samples(unstable={"SC-000", "SC-001"})
    consistency = scenario_judge_consistency(samples).set_index("scenario_id")
    assert consistency.loc["SC-000", "pct_range_ge_2"] == pytest.approx(1.0)
    assert consistency.loc["SC-005", "pct_range_ge_2"] == pytest.approx(0.0)
    assert consistency.loc["SC-000", "mean_range"] == pytest.approx(3.0)


def test_consistency_ignores_single_sample_items() -> None:
    """A full-run single pass carries no stability information at all."""
    one_sample = _judge_samples(unstable=set())
    one_sample = one_sample[one_sample["rater_sample_idx"] == 0]
    assert scenario_judge_consistency(one_sample).empty


def test_consistency_uses_the_quality_scale_for_ritualistic() -> None:
    """The range is the same either way; the values must still be flipped."""
    rows = [
        {
            "generation_id": "g1",
            "scenario_id": "SC-000",
            "condition": "A",
            "rater_id": "j",
            "rater_sample_idx": idx,
            "dimension": "ritualistic",
            "raw": raw,
        }
        for idx, raw in enumerate([5, 5, 1])
    ]
    consistency = scenario_judge_consistency(pd.DataFrame(rows))
    assert consistency.iloc[0]["mean_range"] == pytest.approx(4.0)


def test_judge_consistency_rerun_excludes_the_flagged_scenarios(
    base_long: pd.DataFrame,
) -> None:
    base = run_family(base_long, include_friedman=False, n_boot=200)
    samples = _judge_samples(unstable={f"SC-{i:03d}" for i in range(5)})
    run, exclusion = sensitivity_judge_consistency(base_long, base, samples, n_boot=200)
    assert len(exclusion.excluded) == 5
    rerun = run.family.by_key("primary_nurse_A_vs_B")
    assert rerun is not None
    assert rerun.n_scenarios == 15


def test_the_consistency_threshold_is_not_pre_specified_and_says_so(
    base_long: pd.DataFrame,
) -> None:
    """§8.5(c) fixes the analysis but not the numeric cut for 'poor'."""
    base = run_family(base_long, include_friedman=False, n_boot=200)
    samples = _judge_samples(unstable={"SC-000"})
    run, exclusion = sensitivity_judge_consistency(base_long, base, samples, n_boot=200)
    assert exclusion.threshold_prespecified is False
    assert exclusion.threshold == DEFAULT_MAX_PCT_RANGE_GE_2
    assert "not" in exclusion.reason and "pre-registration §8.5(c)" in exclusion.reason
    assert any("implementation choice" in c for c in run.caveats)


def test_a_stricter_threshold_excludes_more(base_long: pd.DataFrame) -> None:
    """Four scenarios wobble in both conditions, three in one. The cut separates them."""
    base = run_family(base_long, include_friedman=False, n_boot=200)
    samples = _judge_samples(
        unstable={f"SC-{i:03d}" for i in range(4)},
        partially_unstable={f"SC-{i:03d}" for i in range(10, 13)},
    )
    _, lax = sensitivity_judge_consistency(
        base_long, base, samples, max_pct_range_ge_2=0.75, n_boot=200
    )
    _, strict = sensitivity_judge_consistency(
        base_long, base, samples, max_pct_range_ge_2=0.25, n_boot=200
    )
    assert len(lax.excluded) == 4
    assert len(strict.excluded) == 7
    assert set(lax.excluded).issubset(set(strict.excluded))


# ---------------------------------------------------------------------------
# The whole report
# ---------------------------------------------------------------------------


def test_the_report_says_what_it_could_not_run(base_long: pd.DataFrame) -> None:
    base = run_family(base_long, include_friedman=False, n_boot=200)
    report = run_all_sensitivity(base_long, base, n_boot=200)
    assert len(report.not_runnable) == 2
    assert any("§8.5(a)" in m for m in report.not_runnable)
    assert any("§8.5(c)" in m for m in report.not_runnable)
    text = report.render()
    assert "NOT RUNNABLE" in text


def test_the_report_leads_with_the_flips(
    base_long: pd.DataFrame, nurse_dimensions: tuple[str, ...]
) -> None:
    """A conclusion that flips is the finding, so it goes above the tables."""
    human_scores: dict[tuple[str, str, int], dict[str, int]] = {}
    for i in range(20):
        scenario = f"SC-{i:03d}"
        human_scores[(scenario, "A", 0)] = constant_scores(nurse_dimensions, 5)
        human_scores[(scenario, "B", 0)] = constant_scores(nurse_dimensions, 1)
    combined = pd.concat(
        [base_long, make_long(scores=human_scores, rater_type="human", rater_id="R1")]
    )
    base = run_family(combined, include_friedman=False, rater_type="llm_judge", n_boot=200)
    report = run_all_sensitivity(combined, base, n_boot=200)
    assert not report.conclusions_hold
    text = report.render()
    assert "CONCLUSION(S) MOVE UNDER SENSITIVITY ANALYSIS" in text
    assert text.index("is the finding") < text.index("(a) human-only ratings")


def test_a_stable_study_says_so_plainly(base_long: pd.DataFrame) -> None:
    base = run_family(base_long, include_friedman=False, n_boot=200)
    report = run_all_sensitivity(base_long, base, n_boot=200)
    assert report.conclusions_hold
    assert "Every conclusion holds" in report.render()


def test_an_empty_rerun_is_not_reported_as_conclusions_holding() -> None:
    """No data is not evidence of robustness."""
    from tests.unit.stats.conftest import make_long

    empty = make_long(scores={})
    base = run_family(empty, include_friedman=False, n_boot=50)
    report = run_all_sensitivity(empty, base, n_boot=50)
    assert report.nothing_to_compare
    text = report.render()
    assert "nothing was tested for robustness" in text
    assert "Every conclusion holds" not in text


# ---------------------------------------------------------------------------
# (d) the output safety gate — D12
# ---------------------------------------------------------------------------


@pytest.fixture
def blocked_long(nurse_dimensions: tuple[str, ...]) -> pd.DataFrame:
    """20 scenarios x {A, B} x 3 samples, with one scenario's cells gate-refused.

    SC-000 carries the refusals, concentrated the way the real run's are on
    SC-029: all three A samples and all three B samples. Excluding them removes
    that scenario from the paired comparison entirely, which is the behaviour
    the rerun has to demonstrate rather than assert.
    """
    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    blocked: list[tuple[str, str, int]] = []
    for i in range(20):
        scenario = f"SC-{i:03d}"
        for sample in range(3):
            scores[(scenario, "A", sample)] = constant_scores(nurse_dimensions, 2)
            scores[(scenario, "B", sample)] = constant_scores(nurse_dimensions, 4)
            if scenario == "SC-000":
                blocked.append((scenario, "A", sample))
                blocked.append((scenario, "B", sample))
    return make_long(scores=scores, gate_blocked=blocked)


def test_gate_blocked_rerun_excludes_the_refused_generations(
    blocked_long: pd.DataFrame,
) -> None:
    base = run_family(blocked_long, n_boot=200)
    run = sensitivity_gate_blocked(blocked_long, base, n_boot=200)

    assert "6 generations excluded" in " ".join(run.family.notes)
    primary = run.family.by_key("primary_nurse_A_vs_B")
    assert primary is not None
    # SC-000 loses both arms, so the paired comparison drops from 20 to 19.
    assert primary.n_scenarios == 19


def test_the_gate_blocked_rerun_names_the_affected_scenarios(
    blocked_long: pd.DataFrame,
) -> None:
    """A concentrated exclusion has to be visible as concentrated."""
    base = run_family(blocked_long, n_boot=200)
    run = sensitivity_gate_blocked(blocked_long, base, n_boot=200)
    assert "SC-000" in run.specification


def test_the_gate_blocked_rerun_is_labelled_not_planned_in_advance(
    blocked_long: pd.DataFrame,
) -> None:
    """D12 postdates the plan; the rerun must not borrow the plan's standing."""
    base = run_family(blocked_long, n_boot=200)
    run = sensitivity_gate_blocked(blocked_long, base, n_boot=200)
    assert not run.prespecified
    text = run.render()
    assert "NOT PLANNED IN ADVANCE" in text
    assert "PREFERRED READING" in run.name
    assert any("category error" in caveat for caveat in run.caveats)


def test_the_gate_blocked_rerun_survives_a_frame_without_the_column(
    base_long: pd.DataFrame,
) -> None:
    base = run_family(base_long, n_boot=200)
    run = sensitivity_gate_blocked(base_long.drop(columns=["gate_blocked"]), base, n_boot=200)
    assert "0 generations excluded" in " ".join(run.family.notes)
    assert run.conclusions_hold


# ---------------------------------------------------------------------------
# Retrieval, asked two ways
# ---------------------------------------------------------------------------


@pytest.fixture
def retrieval_long(nurse_dimensions: tuple[str, ...]) -> pd.DataFrame:
    """B and C over 20 scenarios, where C only beats B on the cells that retrieved.

    Constructed so the two readings must differ and the direction of the
    difference is known in advance: on the 10 scenarios where CRAG fired, C
    scores 4 against B's 3; on the 10 where it fell back, C scores 3, exactly
    matching B. Pooling therefore halves the effect, which is the attenuation
    the two-way report exists to expose.
    """
    scores: dict[tuple[str, str, int], dict[str, int]] = {}
    fell_back: list[tuple[str, str, int]] = []
    for i in range(20):
        scenario = f"SC-{i:03d}"
        retrieved = i < 10
        for sample in range(3):
            scores[(scenario, "B", sample)] = constant_scores(nurse_dimensions, 3)
            scores[(scenario, "C", sample)] = constant_scores(
                nurse_dimensions, 4 if retrieved else 3
            )
            if not retrieved:
                fell_back.append((scenario, "C", sample))
    return make_long(scores=scores, fell_back=fell_back)


def test_the_two_readings_answer_different_questions(
    retrieval_long: pd.DataFrame,
) -> None:
    """Pooling fallback cells attenuates the effect; the split shows by how much."""
    contrast = retrieval_contrast(retrieval_long, n_boot=200)

    assert contrast.n_offered_cells == 60
    assert contrast.n_fallback_cells == 30
    assert contrast.n_retrieved_cells == 30

    assert contrast.offered is not None and contrast.retrieved is not None
    # Pooled: 10 scenarios tie exactly, so only half the pairs carry any signal.
    assert contrast.offered.test.n_nonzero == 10
    assert contrast.offered.n_scenarios == 20
    # Retrieval-only: the tied scenarios lose their C cell and leave the pairing.
    assert contrast.retrieved.n_scenarios == 10
    assert contrast.retrieved.test.n_nonzero == 10
    # The hypothesis is B vs C, so the effect is B - C and C scoring higher makes
    # it negative. Both readings agree on direction; the retrieval-only one is the
    # undiluted magnitude, and the pooled one is attenuated by the tied fallbacks —
    # here to exactly half, since exactly half the scenarios fell back.
    offered_shift = contrast.offered.effects.hodges_lehmann.point
    retrieved_shift = contrast.retrieved.effects.hodges_lehmann.point
    assert offered_shift < 0 and retrieved_shift < 0
    assert retrieved_shift == pytest.approx(-1.0)
    assert offered_shift == pytest.approx(-0.5)
    assert abs(retrieved_shift) > abs(offered_shift)


def test_the_retrieval_only_reading_is_labelled_a_selected_subgroup(
    retrieval_long: pd.DataFrame,
) -> None:
    """CRAG chose the subset. That confound cannot be removed, only declared."""
    contrast = retrieval_contrast(retrieval_long, n_boot=200)
    assert contrast.retrieved is not None
    assert "not a randomised subgroup" in contrast.retrieved.label.tag()
    text = contrast.render()
    assert "not randomised" in contrast.selection_caveat
    assert "UNCORRECTED" in text
    assert "does offering retrieval help" in text.lower()
    assert "does retrieval help" in text.lower()


def test_the_retrieval_p_values_are_uncorrected_not_nan(
    retrieval_long: pd.DataFrame,
) -> None:
    """A family of zero renders `nan`, which reads as a broken number rather than a choice."""
    contrast = retrieval_contrast(retrieval_long, n_boot=200)
    assert contrast.offered is not None
    assert contrast.offered.family_size == 1
    assert contrast.offered.p_holm == contrast.offered.test.p_value


# ---------------------------------------------------------------------------
# (e) backend equivalence — D13, consumed rather than recomputed
# ---------------------------------------------------------------------------


class _FakeEquivalence:
    """Stands in for `carelite.eval.judge.backend_equivalence.BackendEquivalence`.

    The stats lane must not recompute this measurement, so the test asserts on
    what it does with a result rather than on a number of its own. A stub with
    the two attributes this package reads is the whole contract.
    """

    poolable = False

    def render(self) -> str:
        return "Backend equivalence, condition LC: ollama vs vllm\n  39 paired cells"


class TestBackendEquivalenceFraming:
    def test_it_reports_the_judge_lane_result_verbatim(self) -> None:
        check = backend_equivalence_check(_FakeEquivalence())
        text = check.render()
        assert "39 paired cells" in text
        assert check.available
        assert check.poolable is False

    def test_every_limit_is_printed_with_the_number(self) -> None:
        text = backend_equivalence_check(_FakeEquivalence()).render()
        assert "13 of the 60 held-out scenarios" in text
        assert "never randomised" in text
        assert "does NOT license" in text
        assert "does NOT isolate" in text
        assert "not a rerun of the §8.1 family" in text
        assert "EXPLORATORY" in text

    def test_agreement_does_not_authorise_pooling(self) -> None:
        """Even a poolable verdict does not reverse D13's decision about the arm."""

        class _Poolable(_FakeEquivalence):
            poolable = True

        text = backend_equivalence_check(_Poolable()).render()
        assert "does not reverse it" in text
        assert "served_by = 'vllm'" in text

    def test_an_absent_check_is_absence_not_agreement(self) -> None:
        check = backend_equivalence_check(None)
        assert not check.available
        assert check.poolable is None
        assert "not evidence that they agree" in check.render()

    def test_it_is_not_a_run_and_cannot_flip_a_conclusion(self, separated_ab: pd.DataFrame) -> None:
        base = run_family(separated_ab, n_boot=100)
        report = run_all_sensitivity(
            separated_ab,
            base,
            n_boot=100,
            backend_equivalence=backend_equivalence_check(_FakeEquivalence()),
        )
        assert report.backend_equivalence is not None
        assert all("backend" not in run.name for run in report.runs)
        assert report.flips == ()
        assert "(e) backend equivalence" in report.render()

    def test_a_database_failure_is_reported_not_raised(self, monkeypatch) -> None:
        import carelite.eval.judge.backend_equivalence as be

        def _boom(**_: object) -> None:
            raise RuntimeError("no database configured")

        monkeypatch.setattr(be, "run_backend_equivalence", _boom)
        check = read_backend_equivalence()
        assert not check.available
        assert "no database configured" in check.unavailable_reason
