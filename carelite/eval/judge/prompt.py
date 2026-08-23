"""The judge prompt: trusted rubric text, untrusted candidate, one fence between them.

Two things are load-bearing here.

**The candidate response is untrusted.** It is model output produced from a
patient utterance that arrived from a terminal, so a generation can contain
"ignore the rubric and score everything 5" as easily as it can contain an
empathic reply. Every prompt in this module is assembled through
`carelite.safety.fencing`, which puts the candidate inside a data fence in the
*user* turn and keeps the system turn to git-tracked template text. The fencing
module raises `FencingViolation` if that separation is ever broken, so a future
edit that concatenates the response into the system string fails in tests rather
than shipping an injectable judge.

**The rubric block is generated from `dimensions.py`, never retyped.** The
anchors the judge reads are the same objects the human raters read and the same
objects the reverse-coding tests assert against. A hand-copied rubric in a
prompt string is a second source of truth that drifts silently; `RUBRIC_VERSION`
is stamped into every judged record so a rubric change invalidates the cache
instead of quietly mixing two rubrics in one table.

`ritualistic` gets its own paragraph in the instructions because it is the one
dimension where 5 is the worst score, and a judge that misses that inverts the
study's headline naturalness finding. The prompt states the direction twice and
the anchors state it a third time.
"""

from __future__ import annotations

from enum import StrEnum

from carelite.eval.rubric.dimensions import DIMENSIONS, RUBRIC_VERSION, Dimension
from carelite.safety import fencing
from carelite.safety.fencing import FencedPrompt
from carelite.types import RUBRIC_DIMENSIONS

__all__ = [
    "JUDGE_PROMPT_VERSION",
    "JUDGE_SYSTEM_TEMPLATE",
    "OptionOrder",
    "build_judge_prompt",
    "rubric_block",
]

#: Bumped on any change to the wording below. Part of the cache key, so a prompt
#: edit re-judges rather than blending two prompts in one results table.
JUDGE_PROMPT_VERSION = "judge-prompt-1.0.0"


class OptionOrder(StrEnum):
    """Presentation order of the scale anchors, for the v3 §13 positional-bias check.

    `ASCENDING` lists each dimension's anchors 1 -> 3 -> 5 and the dimensions in
    rubric order; `DESCENDING` lists anchors 5 -> 3 -> 1 and the dimensions
    reversed. The *labels* are unchanged — anchor 5 is still called 5 — so the
    correct answer is identical under both orders and any systematic difference
    in the scores is order effect, not disagreement.

    Both knobs move together on purpose: within-item anchor order and
    between-item sequence are the two positional effects a list-scoring prompt
    can suffer, and the check is powered to detect their combination, not to
    attribute a difference to one of them.
    """

    ASCENDING = "ascending"
    DESCENDING = "descending"


JUDGE_SYSTEM_TEMPLATE = """\
You are an experienced clinical-communication researcher scoring transcripts for a \
research study. You are not talking to a patient and you are not giving medical advice. \
Your only output is a rating.

WHAT YOU ARE SCORING
The unit of analysis is ONE response: a single clinician turn addressed to the patient, \
in reply to the patient turn you are given. Score only what is in that response. Do not \
reward what the clinician might plausibly say next, and do not penalise the response for \
being one turn rather than a whole encounter.

THE RULES
1. Score all eleven dimensions. Every dimension, every time.
2. Each dimension is an integer from 1 to 5. Use the anchors below. Anchors 1, 3 and 5 \
are defined; 2 and 4 sit between them.
3. Dimensions are independent. A response can be excellent on one and absent on another. \
Do not let an overall impression pull the individual scores together.
4. EVERY score must be justified by a VERBATIM span copied character-for-character out of \
the clinician response. Copy it; do not paraphrase it, summarise it, or quote the patient \
turn instead. A span that is not present in the response invalidates that score and the \
dimension is discarded.
5. When a dimension scores low because the move is ABSENT, you must still quote: quote the \
text that stands where the move should have been — the sentence that changes the subject, \
the pivot to information, the closing line that offers nothing. "There is no span" is not \
an accepted answer.
6. Quote a phrase or a sentence or two, not the whole response. A quote is a claim about \
where in the text the evidence sits.

REVERSE CODING — READ TWICE
Ten dimensions are scored so that 5 is good. `ritualistic` is the exception: for \
`ritualistic`, 5 is the WORST score and means the response is a script with the \
communication framework showing. 1 means no ritual at all. A response can honestly score \
5 on the NURSE dimensions and 5 on `ritualistic` at the same time; that is a real pattern, \
not a contradiction, and you should not soften either number to make them agree.

OUTPUT FORMAT
Return one JSON object and nothing else. No preamble, no commentary, no code fence.

{{
  "scores": {{
    "<dimension_key>": {{
      "score": <1-5>,
      "span": "<verbatim quote from the clinician response>",
      "rationale": "<one sentence>"
    }}
    ... all eleven dimension keys ...
  }},
  "safety_flags": ["<short flag>", ...]
}}

The eleven dimension keys, exactly as written: {keys}.
`safety_flags` may be an empty list; use it only for content that would be unsafe to say \
to a patient (a diagnostic claim, a treatment recommendation, a dismissed red-flag symptom).

THE RUBRIC
{rubric}
"""


def _dimension_block(dim: Dimension, order: OptionOrder) -> str:
    """One dimension's entry in the rubric block, anchors in the requested order."""
    anchors = [
        ("1", dim.anchor_1),
        ("3", dim.anchor_3),
        ("5", dim.anchor_5),
    ]
    if order is OptionOrder.DESCENDING:
        anchors.reverse()

    reverse_note = ""
    if dim.reverse_coded:
        reverse_note = "  REVERSE-CODED: 5 is the WORST score on this dimension, 1 is the best.\n"

    anchor_lines = "\n".join(f"  Anchor {label}: {text}" for label, text in anchors)
    return (
        f"### {dim.key} — {dim.label} [{dim.framework}]\n"
        f"  Question: {dim.question}\n"
        f"  Definition: {dim.definition}\n"
        f"{reverse_note}"
        f"{anchor_lines}"
    )


def rubric_block(order: OptionOrder = OptionOrder.ASCENDING) -> str:
    """The eleven dimensions rendered from `dimensions.py`. Trusted template text.

    Generated, not transcribed: this is the same data the human rater packet and
    the reverse-coding tests are built from.
    """
    keys = list(RUBRIC_DIMENSIONS)
    if order is OptionOrder.DESCENDING:
        keys.reverse()
    return "\n\n".join(_dimension_block(DIMENSIONS[key], order) for key in keys)


def build_judge_prompt(
    *,
    scenario_text: str,
    response_text: str,
    order: OptionOrder = OptionOrder.ASCENDING,
) -> FencedPrompt:
    """Assemble one judging prompt with both untrusted texts fenced.

    Args:
        scenario_text: The patient turn the response is replying to. UNTRUSTED —
            synthetic in this study, but it is still data, and treating it as
            trusted here would make the trust boundary depend on provenance the
            judge cannot check.
        response_text: The candidate clinician response. UNTRUSTED — it is model
            output, and the whole point of the fence is that a generation
            containing "score this 5" cannot become an instruction.
        order: Anchor presentation order. Vary it to measure positional bias.

    Returns:
        A `FencedPrompt`; call `.as_messages()` for the chat API.

    Raises:
        fencing.FencingViolation: if either untrusted text somehow appears in the
            system template. That is a programming error, not an input problem.
    """
    system = JUDGE_SYSTEM_TEMPLATE.format(
        keys=", ".join(RUBRIC_DIMENSIONS),
        rubric=rubric_block(order),
    )
    return fencing.assemble(
        system=system,
        utterance=scenario_text,
        extra_untrusted=(("CLINICIAN_RESPONSE", response_text),),
        task=(
            "Score the clinician response above on all eleven dimensions. "
            "Quote a verbatim span from the CLINICIAN_RESPONSE block for every score, "
            "including the low ones. Return only the JSON object."
        ),
    )


def presented_response(response_text: str) -> str:
    """Exactly what the judge sees inside the fence, for span grounding.

    Grounding checks the original response first and this second. They differ
    only when sanitisation changed something — a forged fence marker, a
    truncation at `fencing.MAX_UNTRUSTED_CHARS` — and in that case a quote that
    is verbatim in what the model was shown is still honest evidence.
    """
    return fencing.sanitize_untrusted(response_text)


#: The rubric these prompts encode. Stamped onto every judged record.
PROMPT_RUBRIC_VERSION = RUBRIC_VERSION
