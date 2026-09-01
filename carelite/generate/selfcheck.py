"""The verification pass: Self-RAG's reflection idea, CoVe's shape, no critic.

Self-RAG trains a critic to emit reflection tokens. There is no training data
for one here and building one would be a project in itself, so what is adopted
is the idea rather than the implementation: after the draft exists, the model is
asked a fixed list of questions about it and repairs only what comes back badly.
That is Chain-of-Verification with the question-planning step removed.

**The questions are fixed, and that is the design.** CoVe has the model plan its
own verification questions. A planned list varies from turn to turn, which would
put a source of variance inside the independent variable of a controlled
comparison — condition C would differ from condition C. The list in
`carelite/prompts/selfcheck.v1.md` is versioned like any other prompt.

**The revision instruction is narrow on purpose.** The prompt tells the model to
repair the faults and change nothing else. A verification pass that rewrites
freely is a second generator, and the conditions that have one would then differ
from the conditions that do not by two things instead of one.

**Failure is recorded, not swallowed.** If the daemon is down, or the reply is
not JSON, the draft is returned untouched with `available=False` and a reason.
Marking every draft failed would fabricate a result; marking every draft passed
without saying so would hide a condition running without a component it is
supposed to have. The runner writes `available` alongside the generation so a
run with a broken self-check is visible in the data rather than in a log.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from carelite.generate import prompts
from carelite.generate.model import GenerationError, ModelClient
from carelite.safety import fencing

__all__ = ["SELF_CHECK_PROMPT_ID", "SelfCheckResult", "parse_verdict", "run_self_check"]

SELF_CHECK_PROMPT_ID = "selfcheck.v1"

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class SelfCheckResult:
    """What the verification pass concluded, and what text to use afterwards."""

    text: str
    """The turn to carry forward: the revision if there was one, else the draft."""

    passed: bool
    """True when the check ran and found no fault. False when it found one —
    whether or not a revision replaced the draft. A run that could not check
    reports `passed=True` with `available=False`; read both."""

    revised: bool
    faults: tuple[str, ...] = ()
    available: bool = True
    reason: str | None = None

    def as_record(self) -> dict[str, Any]:
        """Flat form for the runner's per-generation metadata."""
        return {
            "self_check_available": self.available,
            "self_check_passed": self.passed,
            "self_check_revised": self.revised,
            "self_check_faults": list(self.faults),
            "self_check_reason": self.reason,
        }


def parse_verdict(raw: str) -> tuple[bool, tuple[str, ...], str] | None:
    """`(passed, faults, revised_text)` from the model's reply, or `None`.

    Tolerant of a fenced code block or prose around the object, because a
    constrained decode is not available on every model in the roster and a
    parser that only accepts a bare object would silently disable the check on
    whichever model wraps its JSON.
    """
    match = _JSON_OBJECT.search(raw)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in {"pass", "revise"}:
        return None
    raw_faults = obj.get("faults") or []
    faults = (
        tuple(str(f).strip() for f in raw_faults if str(f).strip())
        if (isinstance(raw_faults, list))
        else ()
    )
    revised = str(obj.get("revised") or "").strip()
    return verdict == "pass", faults, revised


def run_self_check(
    draft: str,
    *,
    utterance: str,
    retrieved: Sequence[Any] = (),
    client: ModelClient,
    model_tag: str,
    seed: int,
    window: int = 8192,
    num_predict: int = 700,
) -> SelfCheckResult:
    """Verify one draft. Never raises; see the module docstring on failure.

    `draft` is model output and therefore untrusted — a generation produced from
    an adversarial patient turn can carry an instruction as easily as it can
    carry an empathic reply — so it goes into the prompt through
    `extra_untrusted`, inside a fence, exactly like the patient turn does.

    Temperature is 0 regardless of the generation temperature: the verification
    step is measurement apparatus, and apparatus that varies run to run is not
    apparatus.
    """
    template = prompts.load(SELF_CHECK_PROMPT_ID)
    prompt = fencing.assemble(
        system=prompts.assembled_text(SELF_CHECK_PROMPT_ID),
        task=template.task,
        utterance=utterance,
        retrieved=list(retrieved),
        extra_untrusted=(("DRAFT_RESPONSE", draft),),
    )

    try:
        out = client.generate(
            prompt,
            model_tag=model_tag,
            seed=seed,
            temperature=0.0,
            num_predict=num_predict,
            window=window,
            json_format=True,
        )
    except GenerationError as exc:
        return SelfCheckResult(
            text=draft,
            passed=True,
            revised=False,
            available=False,
            reason=f"self-check did not run: {exc}",
        )

    parsed = parse_verdict(out.text)
    if parsed is None:
        return SelfCheckResult(
            text=draft,
            passed=True,
            revised=False,
            available=False,
            reason="self-check reply was not a parseable verdict object",
        )

    passed, faults, revised_text = parsed
    if passed or not revised_text:
        # A "revise" verdict with no replacement text is a fault the check
        # found and could not repair. The draft stands, and `passed` records
        # that it was faulted, so the run does not quietly look clean.
        return SelfCheckResult(text=draft, passed=passed, revised=False, faults=faults)
    return SelfCheckResult(text=revised_text, passed=False, revised=True, faults=faults)
