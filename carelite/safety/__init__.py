"""CARELite safety layer.

Four detectors and one structural defence, in the order a turn meets them:

    terminal input ──▶ screen_input()  ─┬─ redflag   (escalate, do not coach)
                                        ├─ injection (block or redact)
                                        └─ phi       (warn, do not persist)
                             │
                             ▼
                   fencing.assemble()   ── untrusted text confined to the user
                             │              turn, inside labelled data fences
                             ▼
                        generation
                             │
                             ▼
                    screen_output()     ── leaked instructions, clinical
                                           recommendations, PHI in output

`fencing` is the load-bearing one and the only module other lanes must use.
Every prompt in the system is assembled through `fencing.assemble`, which is
what makes "user text never enters the system prompt" a property of the code
rather than a convention.

Everything in this package is deterministic: pure pattern matching, no network,
no model calls. The same input always produces the same verdict, which is what
lets safety behaviour be part of the reproducibility story.
"""

from __future__ import annotations

from carelite.safety import fencing, injection, output_gate, phi, redflag
from carelite.safety.fencing import FencedPrompt, FencingViolation, assemble
from carelite.types import SafetyVerdict

__all__ = [
    "FencedPrompt",
    "FencingViolation",
    "SafetyVerdict",
    "assemble",
    "fencing",
    "injection",
    "output_gate",
    "phi",
    "redflag",
    "screen_input",
    "screen_output",
]


def screen_input(text: str, *, negation_aware: bool = False) -> SafetyVerdict:
    """Run every input screen and merge the results into one verdict.

    Order is deliberate. Red flags are checked first and reported first: if a
    patient utterance describes an emergency, that is the only thing about the
    turn that matters, and it stays true even if the same utterance also
    contains an injection payload or a phone number.

    The merged verdict is conservative — `allowed` is the AND of every layer —
    and `redacted_text` is the composition of the redactions that fired, so a
    caller that wants to proceed on sanitised text gets injection spans and PHI
    spans both removed.

    Note the two distinct meanings carried by `allowed=False`: a red flag means
    *escalate instead of coaching*, an injection block means *reject this
    input*, and PHI means *warn and do not persist*. Callers that need to tell
    them apart should read `red_flag` / `injection_detected` / `phi_detected`
    rather than branching on `allowed` alone, and should call
    `phi.may_persist(verdict)` before writing anything to the database.
    """
    red = redflag.screen(text, negation_aware=negation_aware)
    inj = injection.screen_utterance(text)
    identifiers = phi.screen(text)

    flags = [*red.flags, *inj.flags, *identifiers.flags]

    redacted: str | None = None
    if inj.redacted_text is not None:
        redacted = inj.redacted_text
    if identifiers.phi_detected:
        redacted = phi.redact(redacted if redacted is not None else text)

    reasons = [v.reason for v in (red, inj, identifiers) if v.reason]

    return SafetyVerdict(
        allowed=red.allowed and inj.allowed and identifiers.allowed,
        injection_detected=inj.injection_detected,
        phi_detected=identifiers.phi_detected,
        red_flag=red.red_flag,
        flags=flags,
        redacted_text=redacted,
        reason="\n\n".join(reasons) if reasons else None,
    )


def screen_output(response: str, *, system_prompt: str | None = None) -> SafetyVerdict:
    """Gate a generation. Thin re-export of `output_gate.screen` for symmetry."""
    return output_gate.screen(response, system_prompt=system_prompt)
