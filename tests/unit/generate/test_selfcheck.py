"""The verification pass, including what it does when it cannot run."""

from __future__ import annotations

import json

from carelite.generate.selfcheck import parse_verdict, run_self_check
from carelite.safety.fencing import SENTINEL

from .conftest import FakeClient


def test_a_clean_verdict_parses() -> None:
    parsed = parse_verdict(json.dumps({"faults": [], "verdict": "pass", "revised": ""}))
    assert parsed == (True, (), "")


def test_a_fenced_code_block_still_parses() -> None:
    """Not every model in the roster honours a constrained decode. A parser that
    only accepted a bare object would silently disable the check on the rest."""
    raw = (
        "Here is the result:\n```json\n"
        + json.dumps(
            {"faults": ["1. mentions a dose"], "verdict": "revise", "revised": "A repaired turn."}
        )
        + "\n```\n"
    )
    parsed = parse_verdict(raw)
    assert parsed is not None
    passed, faults, revised = parsed
    assert passed is False
    assert faults == ("1. mentions a dose",)
    assert revised == "A repaired turn."


def test_an_unparseable_reply_is_not_a_verdict() -> None:
    assert parse_verdict("I think the draft is fine.") is None
    assert parse_verdict('{"verdict": "maybe"}') is None
    assert parse_verdict("{not json at all}") is None


def test_the_draft_is_fenced_as_untrusted() -> None:
    client = FakeClient(reply=lambda p, i: json.dumps({"faults": [], "verdict": "pass"}))
    draft = "Some draft text that a model produced and that must not be trusted."
    result = run_self_check(
        draft,
        utterance="I am worried about the results.",
        client=client,
        model_tag="fake:1b",
        seed=1,
    )
    prompt = client.prompts_seen[0]
    assert draft not in prompt.system
    assert f"{SENTINEL}_DRAFT_RESPONSE_BEGIN" in prompt.user
    assert result.passed is True
    assert result.text == draft


def test_verification_runs_at_temperature_zero() -> None:
    client = FakeClient(reply=lambda p, i: json.dumps({"faults": [], "verdict": "pass"}))
    run_self_check("draft", utterance="u", client=client, model_tag="fake:1b", seed=9)
    assert client.calls[0]["temperature"] == 0.0
    assert client.calls[0]["json_format"] is True


def test_a_failed_check_returns_the_draft_and_says_it_did_not_run() -> None:
    client = FakeClient(fail_with="daemon unreachable")
    result = run_self_check("draft", utterance="u", client=client, model_tag="fake:1b", seed=1)
    assert result.available is False
    assert result.text == "draft"
    assert result.reason is not None and "daemon unreachable" in result.reason
    # `passed=True` with `available=False` is the honest pair: nothing was
    # found because nothing was looked for. A caller must read both.
    assert result.passed is True
    assert result.as_record()["self_check_available"] is False


def test_an_unparseable_reply_degrades_the_same_way() -> None:
    client = FakeClient(reply=lambda p, i: "looks good to me")
    result = run_self_check("draft", utterance="u", client=client, model_tag="fake:1b", seed=1)
    assert result.available is False
    assert result.text == "draft"


def test_a_revise_verdict_with_no_replacement_keeps_the_draft_but_records_the_fault() -> None:
    client = FakeClient(
        reply=lambda p, i: json.dumps({"faults": ["3. unsupported claim"], "verdict": "revise"})
    )
    result = run_self_check("draft", utterance="u", client=client, model_tag="fake:1b", seed=1)
    assert result.text == "draft"
    assert result.revised is False
    assert result.passed is False
    assert result.faults == ("3. unsupported claim",)
