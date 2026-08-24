"""The §13 study driver: subset selection, the holdout gate, and the controls.

The expensive parts of this lane are the model calls, and none of them are
exercised here. What is exercised is everything that decides whether those
calls were spent on the right thing — which scenarios, which split, and whether
the threshold that classifies the results can actually return both answers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from carelite.eval.human.dry_run import (
    STANDARD_PANEL,
    TruthModel,
    anchored_truth,
    dry_run,
)
from carelite.eval.judge.study import (
    N_SUBSET_SCENARIOS,
    _require_train,
    agreement_against_synthetic,
    load_generations,
    replay_from_cache,
    select_subset,
)
from carelite.eval.judge.validation import (
    MIN_ALPHA_FOR_CONFIRMATORY,
    MIN_UNITS_FOR_CONFIRMATORY,
    EvidenceStatus,
)
from carelite.types import RUBRIC_DIMENSIONS, Condition, Scenario, Split

# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


def test_the_subset_spans_every_challenge_type() -> None:
    """The reason `--limit 60` is not used.

    `build_plan` is scenario-major and bank ids are allocated in blocks of ten
    per challenge type, so the first sixty cells are three challenge types. A
    judge's failure modes are content-shaped; a sample that cannot see eight of
    the ten challenge types cannot find them.
    """
    scenarios, _ = select_subset()
    assert len({s.challenge_type for s in scenarios}) == N_SUBSET_SCENARIOS


def test_the_subset_is_sixty_responses_over_all_six_conditions() -> None:
    scenarios, cells = select_subset()
    assert len(cells) == len(scenarios) * len(Condition) == 60
    assert {c.condition for c in cells} == {c.value for c in Condition}


def test_the_subset_clears_the_threshold_s_minimum_unit_count() -> None:
    """Sixty is not arbitrary: below `MIN_UNITS_FOR_CONFIRMATORY` every dimension
    is demoted for a reason that has nothing to do with the judge."""
    _, cells = select_subset()
    assert len(cells) >= MIN_UNITS_FOR_CONFIRMATORY


def test_the_subset_spreads_encounter_phase() -> None:
    scenarios, _ = select_subset()
    counts: dict[str, int] = {}
    for s in scenarios:
        counts[str(s.encounter_phase)] = counts.get(str(s.encounter_phase), 0) + 1
    # Ten scenarios over five phases; no phase may dominate.
    assert max(counts.values()) <= 3


def test_selection_is_deterministic() -> None:
    assert [s.scenario_id for s in select_subset()[0]] == [
        s.scenario_id for s in select_subset()[0]
    ]


def test_every_selected_scenario_is_train() -> None:
    scenarios, _ = select_subset()
    assert all(s.split is Split.TRAIN for s in scenarios)


# ---------------------------------------------------------------------------
# The holdout gate
# ---------------------------------------------------------------------------


def _scenario(scenario_id: str, split: Split) -> Scenario:
    from carelite.types import EncounterPhase

    return Scenario(
        scenario_id=scenario_id,
        text="Patient turn.",
        challenge_type="emotional_cue",
        emotion_intensity=3,
        encounter_phase=EncounterPhase.EXPLANATION,
        literacy_signal="none",
        equity_stratum=False,
        split=split,
    )


def test_require_train_refuses_a_holdout_scenario() -> None:
    """The OSF pre-registration is not submitted. This is the gate that cannot
    be delegated, so the study module checks it too rather than trusting the
    runner's flag."""
    with pytest.raises(RuntimeError, match="train split only"):
        _require_train([_scenario("SC-999", Split.HOLDOUT)])


def test_require_train_checks_the_record_not_the_flag() -> None:
    mixed = [_scenario("SC-001", Split.TRAIN), _scenario("SC-002", Split.HOLDOUT)]
    with pytest.raises(RuntimeError, match="SC-002"):
        _require_train(mixed)


def test_loading_a_journal_refuses_a_holdout_row(tmp_path: Path) -> None:
    """The journal is written by another lane and read here. A holdout row in it
    is not something to filter out quietly — it means something upstream is
    generating data that must not exist yet."""
    journal = tmp_path / "j.jsonl"
    journal.write_text(
        json.dumps(
            {
                "key": ["SC-061", "C", "cond.c.v1", "sha256:x", 7, 0],
                "model": "gemma4:12b",
                "temperature": 0.7,
                "response": "text",
                "extra": {"split": "holdout"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="must not read held-out data"):
        load_generations(journal)


def test_loading_a_journal_accepts_train_rows(tmp_path: Path) -> None:
    journal = tmp_path / "j.jsonl"
    journal.write_text(
        json.dumps(
            {
                "key": ["SC-061", "C", "cond.c.v1", "sha256:x", 7, 0],
                "model": "gemma4:12b",
                "temperature": 0.7,
                "response": "a response",
                "extra": {"split": "train"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    generations, _, responses = load_generations(journal)
    assert len(generations) == 1
    assert responses[generations[0].generation_id] == "a response"


# ---------------------------------------------------------------------------
# Cache-only replay
# ---------------------------------------------------------------------------


def test_replay_calls_no_model_and_skips_uncached_generations(tmp_path: Path) -> None:
    """`--stage report` must never start an inference run by accident."""
    from carelite.eval.judge.prompt import OptionOrder
    from carelite.types import Generation

    generation = Generation(
        generation_id="gen-nothing-cached",
        scenario_id="SC-002",
        condition=Condition.C,
        prompt_id="cond.c.v1",
        model="gemma4:12b",
        model_digest="sha256:x",
        seed=1,
        temperature=0.7,
        sample_idx=0,
        response="a response",
    )
    results = replay_from_cache(
        [generation],
        {"SC-002": "turn"},
        order=OptionOrder.ASCENDING,
        cache_path=tmp_path / "empty.jsonl",
    )
    assert results == []


# ---------------------------------------------------------------------------
# The calibration-contamination regression (da38cd1)
# ---------------------------------------------------------------------------


def _rateables(n: int) -> list:
    from carelite.eval.human.blinding import RateableItem

    return [
        RateableItem(
            generation_id=f"gen-{i:03d}",
            scenario_text=f"turn {i}",
            response_text=f"response {i}",
            condition=("A", "A2", "B", "C", "LC", "D")[i % 6],
        )
        for i in range(n)
    ]


def test_calibration_items_never_enter_the_study_unit_list() -> None:
    """The defect `da38cd1` fixed. Calibration ratings must not become units."""
    result = dry_run(_rateables(30))
    unit_ids = {s.generation_id for s in result.panel_scores}
    assert all(not uid.startswith("CAL-") for uid in unit_ids)
    assert result.calibration_by_rater[STANDARD_PANEL[0].rater_id]


def test_leaking_calibration_inflates_alpha_on_every_dimension() -> None:
    """Why the defect was dangerous rather than merely wrong.

    Raters are shown the calibration consensus and discuss it, so their
    calibration ratings converge on a published answer key. Leaked into the unit
    list they are near-unanimous units, and near-unanimous units *raise*
    Krippendorff's alpha. A harness defect that flatters the headline
    reliability number is the one that never announces itself.
    """
    result = dry_run(_rateables(30))
    check = result.contamination
    assert check is not None
    inflation = check.inflation(check.leaked_converged)
    assert inflation, "no dimension produced a comparable coefficient"
    assert all(delta > 0 for delta in inflation.values()), inflation
    assert check.n_leaked_units > check.n_clean_units


def test_the_contamination_check_uses_the_production_alpha() -> None:
    """Both arms differ only in their unit list, so the delta is attributable."""
    result = dry_run(_rateables(30))
    check = result.contamination
    assert check is not None
    assert set(check.clean) == set(RUBRIC_DIMENSIONS)
    assert set(check.leaked_converged) == set(RUBRIC_DIMENSIONS)


# ---------------------------------------------------------------------------
# Truth models
# ---------------------------------------------------------------------------


def test_anchored_truth_fills_the_cells_the_judge_rejected() -> None:
    """A human has an opinion on a dimension the judge declined to score.
    Dropping those cells would make the positive control easier than the thing
    it stands in for."""
    judge_scores = {"gen-a": dict.fromkeys(RUBRIC_DIMENSIONS, 4)}
    judge_scores["gen-a"]["naturalness"] = None
    truth, n_filled = anchored_truth(judge_scores)
    assert n_filled == 1
    assert truth["gen-a"]["naturalness"] is not None
    assert truth["gen-a"]["name"] == 4


def test_anchored_truth_keeps_ritualistic_on_the_raw_scale() -> None:
    """Raw is the interchange format: it is what the schema stores and what a
    rater writes. The one reversal happens at comparison time, in `to_quality`."""
    judge_scores = {"gen-a": {**dict.fromkeys(RUBRIC_DIMENSIONS, 3), "ritualistic": 5}}
    truth, _ = anchored_truth(judge_scores)
    assert truth["gen-a"]["ritualistic"] == 5


# ---------------------------------------------------------------------------
# The two controls on the threshold
# ---------------------------------------------------------------------------


def _fake_results(n: int) -> Sequence:
    """Judge results with fixed, spread scores. No model, no cache."""
    from carelite.eval.judge.judge import DimensionResult, JudgeResult

    results = []
    for i in range(n):
        dimensions = {}
        for j, key in enumerate(RUBRIC_DIMENSIONS):
            score = 1 + ((i + j) % 5)
            dimensions[key] = DimensionResult(
                dimension=key,
                score=score,
                raw_scores=(score,),
                median=float(score),
                variance=None,
                score_range=0,
                span=f"span {i}",
                span_source="response",
                span_exact=True,
                rationale="",
                rejections=(),
            )
        results.append(
            JudgeResult(
                generation_id=f"gen-{i:03d}",
                judge_model="gpt-oss:20b",
                judge_digest="gpt-oss:20b",
                prompt_version="judge-prompt-1.0.0",
                rubric_version="1.0.0",
                temperature=0.7,
                n_samples_requested=5,
                order="ascending",  # type: ignore[arg-type]
                dimensions=dimensions,
                samples=(),
            )
        )
    return results


def test_the_null_control_demotes_every_dimension() -> None:
    """Raters unrelated to the judge must not clear the threshold. If they do,
    the threshold is broken and every downstream `EvidenceStatus` is worthless."""
    results = _fake_results(40)
    responses = {r.generation_id: "response text" for r in results}
    arms = agreement_against_synthetic(results, responses)
    assert arms["null_control"]["n_confirmatory"] == 0


def test_the_positive_control_can_reach_confirmatory() -> None:
    """Without this, "all exploratory" is unfalsifiable: a `classify_dimension`
    hardwired to return `exploratory` passes the null control perfectly."""
    results = _fake_results(40)
    responses = {r.generation_id: "response text" for r in results}
    arms = agreement_against_synthetic(results, responses)
    assert arms["positive_control"]["n_confirmatory"] >= 6


def test_the_published_status_is_reproducible_from_the_published_numbers() -> None:
    """A reader must be able to check the verdict against the rule without
    rerunning anything."""
    from carelite.eval.judge.validation import classify_dimension

    results = _fake_results(40)
    responses = {r.generation_id: "response text" for r in results}
    arms = agreement_against_synthetic(results, responses)
    for arm in ("null_control", "positive_control"):
        for key, row in arms[arm]["dimensions"].items():
            alpha = float("nan") if row["alpha"] is None else row["alpha"]
            rho = float("nan") if row["rho"] is None else row["rho"]
            expected = classify_dimension(alpha, rho, row["n_units"])
            assert row["status"] == str(expected), f"{arm}:{key}"


def test_the_threshold_recorded_in_the_artifact_is_the_pre_specified_one() -> None:
    arms = agreement_against_synthetic(_fake_results(35), {})
    assert arms["threshold"]["min_alpha"] == MIN_ALPHA_FOR_CONFIRMATORY
    assert arms["threshold"]["min_units"] == MIN_UNITS_FOR_CONFIRMATORY


def test_too_few_units_is_exploratory_whatever_the_coefficient() -> None:
    results = _fake_results(MIN_UNITS_FOR_CONFIRMATORY - 5)
    responses = {r.generation_id: "response text" for r in results}
    arms = agreement_against_synthetic(results, responses)
    statuses = {row["status"] for row in arms["positive_control"]["dimensions"].values()}
    assert statuses == {str(EvidenceStatus.EXPLORATORY)}


def test_a_dimension_needs_both_coefficients_not_just_alpha() -> None:
    from carelite.eval.judge.validation import classify_dimension

    assert (
        classify_dimension(MIN_ALPHA_FOR_CONFIRMATORY + 0.1, 0.2, 60) is EvidenceStatus.EXPLORATORY
    )


# ---------------------------------------------------------------------------
# The harness end to end
# ---------------------------------------------------------------------------


def test_the_dry_run_ingests_without_errors_and_catches_the_broken_rater() -> None:
    result = dry_run(_rateables(30))
    assert result.ingest_errors == []
    assert result.broken_rater_check is not None
    assert "ritualistic" in result.broken_rater_check.flagged


def test_the_judge_anchored_panel_needs_judge_scores() -> None:
    with pytest.raises(ValueError, match="judge_anchored"):
        dry_run(_rateables(5), truth_model=TruthModel.JUDGE_ANCHORED)


def test_intra_rater_reliability_is_reported_for_the_single_rater_fallback() -> None:
    result = dry_run(_rateables(30))
    assert set(result.intra_rater) == set(RUBRIC_DIMENSIONS)
    assert result.intra_rater["name"]["n_units"] > 0
