"""Parsing tolerates packaging and refuses to invent content."""

from __future__ import annotations

import json

import pytest

from carelite.eval.judge.parsing import JudgeParseError, extract_json_object, parse_judge_output
from carelite.types import RUBRIC_DIMENSIONS

GOOD = json.dumps(
    {
        "scores": {
            key: {"score": 3, "span": "a quoted phrase", "rationale": "why"}
            for key in RUBRIC_DIMENSIONS
        },
        "safety_flags": [],
    }
)


class TestExtractJsonObject:
    def test_bare_object(self) -> None:
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_object(self) -> None:
        assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_object_after_a_preamble(self) -> None:
        assert extract_json_object('Here are my scores.\n{"a": 1}') == {"a": 1}

    def test_object_before_trailing_commentary(self) -> None:
        assert extract_json_object('{"a": 1}\nLet me know if you need more.') == {"a": 1}

    def test_thinking_channel_is_stripped_first(self) -> None:
        """A `{` inside private reasoning must not be mistaken for the answer."""
        raw = '<think>maybe {"a": 99} is right</think>\n{"a": 1}'
        assert extract_json_object(raw) == {"a": 1}

    def test_braces_inside_a_quoted_span_do_not_end_the_object(self) -> None:
        raw = '{"span": "he said {this} to me", "score": 2}'
        assert extract_json_object(raw)["score"] == 2

    def test_escaped_quote_inside_a_span(self) -> None:
        raw = '{"span": "she said \\"no\\" twice", "score": 2}'
        assert extract_json_object(raw)["span"] == 'she said "no" twice'

    def test_empty_output_raises(self) -> None:
        with pytest.raises(JudgeParseError):
            extract_json_object("   ")

    def test_prose_only_raises(self) -> None:
        with pytest.raises(JudgeParseError):
            extract_json_object("I am unable to score this response.")


class TestParseJudgeOutput:
    def test_full_wellformed_output(self) -> None:
        parsed = parse_judge_output(GOOD)
        assert set(parsed.dimensions) == set(RUBRIC_DIMENSIONS)
        assert parsed.get("de").score == 3
        assert parsed.get("de").span == "a quoted phrase"

    def test_dimensions_at_top_level_without_a_container(self) -> None:
        raw = json.dumps({key: {"score": 2, "span": "x"} for key in RUBRIC_DIMENSIONS})
        assert parse_judge_output(raw).get("name").score == 2

    @pytest.mark.parametrize("alias", ["evidence", "quote", "verbatim", "evidence_span"])
    def test_span_key_aliases(self, alias: str) -> None:
        raw = json.dumps({"scores": {"name": {"score": 4, alias: "the quote"}}})
        assert parse_judge_output(raw).get("name").span == "the quote"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(4, 4), ("4", 4), ("4/5", 4), (4.0, 4), ("Score: 2", 2), (0, 0), (7, 7)],
    )
    def test_score_coercion(self, value: object, expected: int) -> None:
        """Out-of-range values survive coercion; they are rejected later, by name."""
        raw = json.dumps({"scores": {"name": {"score": value, "span": "q"}}})
        assert parse_judge_output(raw).get("name").score == expected

    def test_a_spelled_out_number_is_not_guessed(self) -> None:
        raw = json.dumps({"scores": {"name": {"score": "four", "span": "q"}}})
        assert parse_judge_output(raw).get("name").score is None

    def test_missing_dimension_stays_missing(self) -> None:
        raw = json.dumps({"scores": {"name": {"score": 4, "span": "q"}}})
        parsed = parse_judge_output(raw)
        assert parsed.get("naturalness").score is None
        assert parsed.get("naturalness").span is None

    def test_dimension_labels_are_accepted_as_keys(self) -> None:
        raw = json.dumps({"scores": {"Invest in the Beginning": {"score": 5, "span": "q"}}})
        assert parse_judge_output(raw).get("ib").score == 5

    def test_unknown_keys_are_dropped_and_recorded(self) -> None:
        raw = json.dumps({"scores": {"empathy_overall": {"score": 5}, "name": {"score": 1}}})
        parsed = parse_judge_output(raw)
        assert "empathy_overall" in parsed.unknown_keys
        assert "empathy_overall" not in parsed.dimensions

    def test_safety_flags_are_bounded(self) -> None:
        raw = json.dumps({"scores": {}, "safety_flags": ["x" * 200] * 40})
        flags = parse_judge_output(raw).safety_flags
        assert len(flags) <= 8
        assert all(len(f) <= 64 for f in flags)

    def test_bare_number_is_a_score_with_no_span(self) -> None:
        raw = json.dumps({"scores": {"name": 3}})
        parsed = parse_judge_output(raw)
        assert parsed.get("name").score == 3
        assert parsed.get("name").span is None
