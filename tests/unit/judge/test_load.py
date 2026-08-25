"""The JSONL -> `rubric_score` bridge.

Everything here except `TestAgainstPostgres` runs with `database=False`, which is
the whole reason `load_score_journals` has that switch: the parsing, the
refusals, the null handling and the report are pure functions of the file and are
verified in `make check` with no Postgres anywhere.

The invariants under test are the ones that corrupt the results table silently
rather than loudly. An incomplete row dropped or zero-filled, a row landed under
a rater id the analysis does not select, two rows resolved by last-writer-wins --
none of those raise, and all of them produce a table that looks fine and reports
the wrong numbers.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from carelite.eval.judge.load import (
    LoadReport,
    ScoreRefusal,
    collect,
    incomplete_generation_ids,
    load_score_journals,
    load_score_metadata,
    resolve_rater_id,
)
from carelite.eval.judge.store import MEDIAN_RATER_SUFFIX
from carelite.types import RUBRIC_DIMENSIONS

RESPONSE = (
    "It sounds like you're frightened. That takes something. Tell me which part "
    "is loudest right now."
)

#: One quotable span per dimension. Every one is verbatim in `RESPONSE`, because
#: a scored dimension with no span is a refusal and the fixtures must not trip it
#: by accident.
SPANS = dict.fromkeys(RUBRIC_DIMENSIONS, "It sounds like you're frightened")


def row(
    generation_id: str = "gen-1",
    *,
    scores: dict[str, int | None] | int = 4,
    condition: str = "C",
    **overrides: Any,
) -> dict[str, Any]:
    """A well-formed journal line, in `holdout.rows_for`'s shape."""
    if isinstance(scores, int):
        values: dict[str, int | None] = dict.fromkeys(RUBRIC_DIMENSIONS, scores)
    else:
        values = {d: scores.get(d, 4) for d in RUBRIC_DIMENSIONS}
    n_scored = sum(1 for v in values.values() if v is not None)
    out: dict[str, Any] = {
        "generation_id": generation_id,
        "rater_type": "llm_judge",
        "rater_id": "holdout-judge",
        "sample_idx": 0,
        "safety_flags": [],
        "evidence_spans": {d: SPANS[d] for d, v in values.items() if v is not None},
        "judge_model": "gpt-oss:20b",
        "judge_digest": "gpt-oss:20b",
        "prompt_version": "judge-prompt-1.0.0",
        "rubric_version": "1.0.0",
        "temperature": 0.0,
        "n_dimensions_scored": n_scored,
        "complete": n_scored == len(RUBRIC_DIMENSIONS),
        "scenario_id": "SC-001",
        "condition": condition,
        "generation_sample_idx": 0,
        "split": "holdout",
        "fell_back_to_b": False,
        "crag_grade": None,
        "route": "informational",
        "n_retrieved": 0,
        "generator_model": "gemma4:12b",
        "generator_digest": "sha256:test",
        "partial_condition": condition == "LC",
        "output_gate_blocked": False,
        "output_gate_flags": [],
    }
    out.update(values)
    out.update(overrides)
    return out


def journal(tmp_path: Path, *rows: dict[str, Any], name: str = "scores.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def load(path: Path, **kwargs: Any) -> LoadReport:
    """The offline path: validate, summarise, write no rows and open no connection."""
    kwargs.setdefault("database", False)
    kwargs.setdefault("write_sidecar", False)
    return load_score_journals([path], **kwargs)


# ---------------------------------------------------------------------------
# The rater id convention
# ---------------------------------------------------------------------------


class TestRaterId:
    def test_aggregate_suffix_is_the_default(self, tmp_path: Path) -> None:
        """`carelite.stats.data` selects `rater_id LIKE '%-median'` and nothing else."""
        report = load(journal(tmp_path, row()))
        assert list(report.rater_ids) == [f"holdout-judge{MEDIAN_RATER_SUFFIX}"]

    def test_suffix_is_not_doubled(self) -> None:
        assert resolve_rater_id("holdout-judge-median", aggregate=True) == "holdout-judge-median"

    def test_no_aggregate_writes_the_samples_partition(self, tmp_path: Path) -> None:
        report = load(journal(tmp_path, row()), aggregate=False)
        assert list(report.rater_ids) == ["holdout-judge"]

    def test_override_replaces_the_file_id_before_the_suffix(self, tmp_path: Path) -> None:
        report = load(journal(tmp_path, row()), rater_id="second-judge")
        assert list(report.rater_ids) == ["second-judge-median"]

    def test_one_row_per_generation(self, tmp_path: Path) -> None:
        """The count the analysis depends on: no per-sample duplicate beside it."""
        records = collect([journal(tmp_path, row("gen-1"), row("gen-2"), row("gen-3"))])
        assert len({r.key for r in records}) == 3
        assert {r.score.generation_id for r in records} == {"gen-1", "gen-2", "gen-3"}


# ---------------------------------------------------------------------------
# Incomplete rows
# ---------------------------------------------------------------------------


class TestIncompleteRows:
    def test_missing_dimension_lands_as_none_not_zero(self, tmp_path: Path) -> None:
        records = collect([journal(tmp_path, row(scores={"naturalness": None}))])
        assert len(records) == 1
        assert records[0].score.naturalness is None
        assert records[0].score.understand == 4

    def test_incomplete_row_is_not_dropped(self, tmp_path: Path) -> None:
        report = load(journal(tmp_path, row("gen-1"), row("gen-2", scores={"explore": None})))
        assert report.records == 2
        assert report.rows_with_null_dimension == 1
        assert report.nulls_by_dimension["explore"] == 1

    def test_declared_completeness_must_match_the_nulls(self, tmp_path: Path) -> None:
        """The check that catches a row already zero-filled or dropped upstream."""
        bad = row(scores={"explore": None})
        bad["n_dimensions_scored"] = 11
        with pytest.raises(ScoreRefusal, match="n_dimensions_scored=11"):
            load(journal(tmp_path, bad))

    def test_complete_flag_must_match_the_nulls(self, tmp_path: Path) -> None:
        bad = row(scores={"explore": None})
        bad["complete"] = True
        bad["n_dimensions_scored"] = 10
        with pytest.raises(ScoreRefusal, match="complete=True"):
            load(journal(tmp_path, bad))

    def test_report_names_the_incomplete_rows(self, tmp_path: Path) -> None:
        report = load(journal(tmp_path, row(scores={"explore": None})))
        assert "not dropped and not zero-filled" in report.summary()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    @pytest.mark.parametrize("value", [0, 6, -1, 99])
    def test_dimension_outside_the_scale(self, tmp_path: Path, value: int) -> None:
        with pytest.raises(ScoreRefusal, match="outside the 1-5 scale"):
            load(journal(tmp_path, row(scores={"ib": value})))

    @pytest.mark.parametrize("value", ["3", 3.5, True, []])
    def test_dimension_that_is_not_an_integer(self, tmp_path: Path, value: Any) -> None:
        """`True` matters: `isinstance(True, int)` would let it land as the score 1."""
        bad = row()
        bad["ib"] = value
        with pytest.raises(ScoreRefusal, match="not an integer 1-5 or null"):
            load(journal(tmp_path, bad))

    def test_rater_type_outside_the_check_set(self, tmp_path: Path) -> None:
        with pytest.raises(ScoreRefusal, match="rater_type 'oracle' is not one of"):
            load(journal(tmp_path, row(rater_type="oracle")))

    def test_score_without_an_evidence_span(self, tmp_path: Path) -> None:
        """v3 §13: a score with no locatable span is not a score."""
        bad = row()
        del bad["evidence_spans"]["respect"]
        with pytest.raises(ScoreRefusal, match="no evidence span"):
            load(journal(tmp_path, bad))

    def test_blank_evidence_span_is_no_span(self, tmp_path: Path) -> None:
        bad = row()
        bad["evidence_spans"]["respect"] = "   "
        with pytest.raises(ScoreRefusal, match="no evidence span"):
            load(journal(tmp_path, bad))

    def test_two_rows_one_key_with_different_scores(self, tmp_path: Path) -> None:
        with pytest.raises(ScoreRefusal, match="two rows with different scores"):
            load(journal(tmp_path, row("gen-1", scores=4), row("gen-1", scores=2)))

    def test_two_identical_rows_collapse(self, tmp_path: Path) -> None:
        report = load(journal(tmp_path, row("gen-1"), row("gen-1")))
        assert report.records == 1
        assert report.duplicates_collapsed == 1

    def test_malformed_json_line(self, tmp_path: Path) -> None:
        path = tmp_path / "scores.jsonl"
        path.write_text(json.dumps(row()) + "\n{not json\n", encoding="utf-8")
        with pytest.raises(ScoreRefusal, match="not valid JSON"):
            load(path)

    def test_missing_generation_id(self, tmp_path: Path) -> None:
        with pytest.raises(ScoreRefusal, match="missing generation_id"):
            load(journal(tmp_path, row(generation_id="")))

    def test_negative_sample_idx(self, tmp_path: Path) -> None:
        with pytest.raises(ScoreRefusal, match="non-negative integer"):
            load(journal(tmp_path, row(sample_idx=-3)))

    def test_unknown_condition(self, tmp_path: Path) -> None:
        with pytest.raises(ScoreRefusal, match="unknown condition"):
            load(journal(tmp_path, row(condition="Z", partial_condition=False)))

    def test_partial_flag_disagreeing_with_the_condition(self, tmp_path: Path) -> None:
        with pytest.raises(ScoreRefusal, match="partial_condition"):
            load(journal(tmp_path, row(condition="C", partial_condition=True)))

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ScoreRefusal, match="no such file"):
            load(tmp_path / "absent.jsonl")

    def test_every_problem_is_reported_not_just_the_first(self, tmp_path: Path) -> None:
        with pytest.raises(ScoreRefusal) as exc:
            load(journal(tmp_path, row("gen-1", scores={"ib": 9}), row("gen-2", scores={"de": 0})))
        assert len(exc.value.problems) == 2


# ---------------------------------------------------------------------------
# Rows the loader must not filter
# ---------------------------------------------------------------------------


class TestRowsThatMustNotBeFiltered:
    def test_gate_blocked_rows_load(self, tmp_path: Path) -> None:
        """D12: exclusion is the analysis's job, through a WHERE it can defend."""
        report = load(journal(tmp_path, row("gen-1"), row("gen-2", output_gate_blocked=True)))
        assert report.records == 2
        assert report.gate_blocked["C"] == 1
        assert "loaded, not filtered" in report.summary()

    def test_lc_rows_load_and_stay_marked(self, tmp_path: Path) -> None:
        """D11: LC is a partial record. Present, marked, never a complete arm."""
        report = load(journal(tmp_path, row("gen-1"), row("gen-2", condition="LC")))
        assert report.records == 2
        assert report.by_condition["LC"] == 1
        assert report.partial_rows["LC"] == 1
        assert "PARTIAL RECORD (D11)" in report.summary()


# ---------------------------------------------------------------------------
# The sidecar
# ---------------------------------------------------------------------------


class TestSidecar:
    def test_completeness_survives_the_table_having_no_column(self, tmp_path: Path) -> None:
        sidecar = tmp_path / "meta.jsonl"
        load(
            journal(tmp_path, row("gen-1"), row("gen-2", scores={"ie": None})),
            write_sidecar=True,
            sidecar=sidecar,
        )
        meta = load_score_metadata(sidecar)
        assert meta["gen-1"]["complete"] is True
        assert meta["gen-2"]["n_dimensions_scored"] == 10
        assert incomplete_generation_ids(sidecar) == {"gen-2"}

    def test_rewriting_merges_rather_than_doubling(self, tmp_path: Path) -> None:
        sidecar = tmp_path / "meta.jsonl"
        first = journal(tmp_path, row("gen-1"), name="a.jsonl")
        second = journal(tmp_path, row("gen-2"), name="b.jsonl")
        load(first, write_sidecar=True, sidecar=sidecar)
        load(second, write_sidecar=True, sidecar=sidecar)
        assert set(load_score_metadata(sidecar)) == {"gen-1", "gen-2"}

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        sidecar = tmp_path / "meta.jsonl"
        report = load(journal(tmp_path, row()), write_sidecar=True, sidecar=sidecar, dry_run=True)
        assert not sidecar.exists()
        assert report.rows_written == 0
        assert "would load" in report.summary()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


class TestReading:
    def test_a_directory_contributes_its_jsonl(self, tmp_path: Path) -> None:
        journal(tmp_path, row("gen-1"), name="a.jsonl")
        journal(tmp_path, row("gen-2"), name="b.jsonl")
        report = load_score_journals([tmp_path], database=False, write_sidecar=False)
        assert report.records == 2

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "scores.jsonl"
        path.write_text("\n" + json.dumps(row()) + "\n\n", encoding="utf-8")
        assert load(path).records == 1

    def test_spans_and_flags_survive(self, tmp_path: Path) -> None:
        records = collect([journal(tmp_path, row(safety_flags=["advice"]))])
        assert records[0].score.safety_flags == ["advice"]
        assert records[0].score.evidence_spans["name"] == SPANS["name"]

    def test_mixed_judge_configurations_are_flagged(self, tmp_path: Path) -> None:
        report = load(
            journal(tmp_path, row("gen-1"), row("gen-2", prompt_version="judge-prompt-2.0.0"))
        )
        assert "did not all come from one judging configuration" in report.summary()


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------


SCENARIO_ID = "sc-judge-load-test"
PROMPT_ID = "prompt-judge-load-test"


@pytest.fixture
def committed_generations() -> Iterator[list[str]]:
    """Two committed generations, deleted afterwards.

    `load_score_journals` opens its own connections through
    `carelite.db.transaction`, so this cannot ride on a rolled-back connection
    the way `test_store_db` does. Deleting the scenario cascades to `generation`
    and from there to `rubric_score`, so the teardown leaves nothing behind.
    """
    from carelite.db import connect

    ids = ["gen-judge-load-1", "gen-judge-load-2"]
    with connect() as conn:
        conn.execute(
            "INSERT INTO scenario (scenario_id, text, challenge_type, emotion_intensity, "
            "encounter_phase, literacy_signal, equity_stratum, split) "
            "VALUES (%s, 'x', 'fear', 4, 'explanation', 'low', true, 'holdout')",
            (SCENARIO_ID,),
        )
        conn.execute(
            "INSERT INTO prompt_version (prompt_id, condition, text) VALUES (%s, 'C', 'x')",
            (PROMPT_ID,),
        )
        for i, gid in enumerate(ids):
            conn.execute(
                "INSERT INTO generation (generation_id, scenario_id, condition, prompt_id, "
                "model, model_digest, seed, temperature, sample_idx, response) "
                "VALUES (%s, %s, 'C', %s, 'gemma4:12b', 'sha256:test', 1, 0.0, %s, %s)",
                (gid, SCENARIO_ID, PROMPT_ID, i, RESPONSE),
            )
        conn.commit()
    try:
        yield ids
    finally:
        with connect() as conn:
            conn.execute("DELETE FROM scenario WHERE scenario_id = %s", (SCENARIO_ID,))
            conn.execute("DELETE FROM prompt_version WHERE prompt_id = %s", (PROMPT_ID,))
            conn.commit()


@pytest.mark.db
class TestAgainstPostgres:
    def _journal(self, tmp_path: Path, ids: list[str]) -> Path:
        return journal(
            tmp_path,
            row(ids[0], scenario_id=SCENARIO_ID),
            row(ids[1], scores={"naturalness": None}, scenario_id=SCENARIO_ID),
        )

    def test_round_trip_keeps_the_null(
        self, tmp_path: Path, committed_generations: list[str]
    ) -> None:
        from carelite.db import fetch_all

        report = load_score_journals(
            [self._journal(tmp_path, committed_generations)], write_sidecar=False
        )
        assert report.rows_inserted == 2
        rows = {
            r["generation_id"]: r
            for r in fetch_all(
                "SELECT * FROM rubric_score WHERE generation_id = ANY(%s)",
                (committed_generations,),
            )
        }
        assert set(rows) == set(committed_generations)
        assert rows[committed_generations[1]]["naturalness"] is None
        assert rows[committed_generations[1]]["understand"] == 4
        assert rows[committed_generations[0]]["rater_id"].endswith(MEDIAN_RATER_SUFFIX)
        assert rows[committed_generations[0]]["evidence_spans"]["name"] == SPANS["name"]

    def test_the_analysis_filter_sees_exactly_one_row_per_generation(
        self, tmp_path: Path, committed_generations: list[str]
    ) -> None:
        """`carelite.stats.data`'s WHERE clause, asserted from this side of it."""
        from carelite.db import fetch_all

        load_score_journals([self._journal(tmp_path, committed_generations)], write_sidecar=False)
        rows = fetch_all(
            "SELECT generation_id, count(*) AS n FROM rubric_score "
            "WHERE rater_type = 'llm_judge' AND rater_id LIKE %s "
            "AND generation_id = ANY(%s) GROUP BY generation_id",
            (f"%{MEDIAN_RATER_SUFFIX}", committed_generations),
        )
        assert len(rows) == 2
        assert {int(r["n"]) for r in rows} == {1}

    def test_reloading_updates_rather_than_duplicating(
        self, tmp_path: Path, committed_generations: list[str]
    ) -> None:
        from carelite.db import fetch_all

        path = self._journal(tmp_path, committed_generations)
        load_score_journals([path], write_sidecar=False)
        second = load_score_journals([path], write_sidecar=False)
        assert second.rows_inserted == 0
        assert second.rows_updated == 2
        assert second.targets_already_present == 2
        total = fetch_all(
            "SELECT count(*) AS n FROM rubric_score WHERE generation_id = ANY(%s)",
            (committed_generations,),
        )
        assert int(total[0]["n"]) == 2

    def test_unknown_generation_id_is_refused_before_anything_is_written(
        self, tmp_path: Path, committed_generations: list[str]
    ) -> None:
        from carelite.db import fetch_all

        path = journal(
            tmp_path,
            row(committed_generations[0], scenario_id=SCENARIO_ID),
            row("gen-does-not-exist", scenario_id=SCENARIO_ID),
        )
        with pytest.raises(ScoreRefusal, match="has no `generation` row"):
            load_score_journals([path], write_sidecar=False)
        rows = fetch_all(
            "SELECT count(*) AS n FROM rubric_score WHERE generation_id = ANY(%s)",
            (committed_generations,),
        )
        assert int(rows[0]["n"]) == 0

    def test_identity_mismatch_is_refused(
        self, tmp_path: Path, committed_generations: list[str]
    ) -> None:
        """A journal loaded against the wrong database attaches real numbers to
        the wrong cells, and no constraint in the schema would object."""
        path = journal(
            tmp_path,
            row(committed_generations[0], scenario_id=SCENARIO_ID, condition="A"),
        )
        with pytest.raises(ScoreRefusal, match="claims condition"):
            load_score_journals([path], write_sidecar=False)
