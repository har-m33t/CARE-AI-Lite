"""The prompts are experimental apparatus, so they get tested like apparatus."""

from __future__ import annotations

import subprocess

import pytest

from carelite.generate import prompts
from carelite.generate.conditions import SPEC
from carelite.types import Condition


def test_every_prompt_id_matches_its_filename() -> None:
    for prompt_id, template in prompts.load_all().items():
        assert template.path is not None
        assert template.path.name == f"{prompt_id}.md"


def test_every_condition_resolves_to_a_prompt() -> None:
    for condition in Condition:
        template = prompts.load(SPEC[condition].prompt_id)
        assert condition.value in template.conditions


def test_a_and_a2_share_one_prompt() -> None:
    """The cross-model baseline varies the model and nothing else.

    Sharing the `prompt_id` is the mechanical form of that claim: there is one
    row in `prompt_version`, so the two conditions cannot drift apart in an edit.
    """
    assert SPEC[Condition.A].prompt_id == SPEC[Condition.A2].prompt_id
    assert SPEC[Condition.A].model_role != SPEC[Condition.A2].model_role


def test_c_and_lc_contain_b_verbatim() -> None:
    """`extends` makes "C is B plus retrieval" a property of the files."""
    framework = prompts.load("condition_b.v1").system
    assert framework in prompts.assembled_text("condition_c.v1")
    assert framework in prompts.assembled_text("condition_lc.v1")


def test_every_condition_carries_the_shared_constraints() -> None:
    """Including the degraded control: D is degraded on communication, not safety."""
    constraints = prompts.load("constraints.v1").system
    for condition in Condition:
        assert constraints in prompts.assembled_text(SPEC[condition].prompt_id), condition


def test_the_degraded_control_actually_differs_from_b() -> None:
    b = prompts.assembled_text("condition_b.v1")
    d = prompts.assembled_text("condition_d.v1")
    assert b != d
    # The negative control has to be degraded on what the rubric measures. If
    # it told the model to do emotion work it would not be a control.
    assert "Do not dwell on how the patient is feeling" in d
    assert "naming the feeling" in b and "naming the feeling" not in d


def test_no_prompt_claims_the_evidence_base_is_human_verified() -> None:
    """DECISIONS.md D4. The claim is not true and must not appear in a prompt."""
    forbidden = (
        "human-verified",
        "human verified",
        "clinician-reviewed",
        "clinician reviewed",
        "expert-reviewed",
        "peer-reviewed by",
        "verified by a clinician",
    )
    for prompt_id in prompts.load_all():
        text = prompts.assembled_text(prompt_id).lower()
        for phrase in forbidden:
            assert phrase not in text, f"{prompt_id} claims {phrase!r}"


def test_the_project_positions_are_stated_in_every_condition() -> None:
    """Not a diagnostic tool; no clinical recommendations. Enforced by the gate,
    stated in the prompt, and held constant across the manipulation."""
    for condition in Condition:
        text = prompts.assembled_text(SPEC[condition].prompt_id).lower()
        assert "you are not a diagnostic tool" in text, condition
        assert "do not give medical advice" in text, condition


def test_blob_sha_matches_git_hash_object() -> None:
    """The recorded `git_sha` has to be resolvable with real git, or it is decoration."""
    text = prompts.assembled_text("condition_b.v1")
    try:
        out = subprocess.run(
            ["git", "hash-object", "--stdin"],
            input=text.encode("utf-8"),
            capture_output=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        pytest.skip("git is not available")
    assert out.stdout.decode().strip() == prompts.blob_sha(text)


def test_registered_rows_carry_condition_and_sha() -> None:
    rows = {r["prompt_id"]: r for r in prompts.registered_rows()}
    assert rows["condition_a.v1"]["condition"] == "A,A2"
    assert rows["condition_c.v1"]["condition"] == "C"
    for row in rows.values():
        assert len(row["git_sha"]) == 40
        assert row["text"] == prompts.assembled_text(row["prompt_id"])


def test_unknown_prompt_id_names_what_exists() -> None:
    with pytest.raises(prompts.PromptError) as exc:
        prompts.load("condition_z.v9")
    assert "condition_b.v1" in str(exc.value)


def test_extends_chain_is_not_circular() -> None:
    for prompt_id in prompts.load_all():
        prompts.assembled_text(prompt_id)  # would raise on a cycle


def test_every_system_prompt_has_a_task_line() -> None:
    for condition in Condition:
        assert prompts.load(SPEC[condition].prompt_id).task.strip()
