"""Turn whatever `gpt-oss:20b` emitted into eleven `(score, span, rationale)` triples.

The judge is asked for strict JSON and is called with Ollama's `format="json"`,
so the happy path is one `json.loads`. This module exists for the other path. A
20B local model wraps JSON in a fence, prefixes it with a sentence of analysis,
emits a `<think>` block, spells a score `"4/5"`, and names the span field
`evidence` instead of `span` — none of which is a reason to throw away a whole
generation's scores when the intent is unambiguous.

The tolerance is strictly syntactic. This module will repair *packaging*: locate
the JSON object inside surrounding prose, accept a documented set of key
aliases, coerce `"4"` and `"4/5"` to `4`. It will not invent content. A missing
dimension stays missing, an unparseable score stays missing, and both become a
`SpanRejection.NO_SCORE` downstream rather than a default value. The line to
hold is that nothing here may produce a score the model did not state, because
a fabricated score is indistinguishable from a real one once it reaches the
database.

Unknown keys are dropped and recorded in `ParsedJudgeOutput.unknown_keys` — a
judge that starts inventing dimensions is a prompt regression worth seeing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from carelite.eval.rubric.dimensions import DIMENSIONS, SCALE_MAX, SCALE_MIN

__all__ = [
    "JudgeParseError",
    "ParsedDimension",
    "ParsedJudgeOutput",
    "extract_json_object",
    "parse_judge_output",
]

#: Accepted spellings of the field holding the score.
_SCORE_KEYS = ("score", "rating", "value", "points")
#: Accepted spellings of the field holding the verbatim quote.
_SPAN_KEYS = ("span", "evidence", "evidence_span", "quote", "verbatim", "excerpt")
#: Accepted spellings of the field holding the free-text justification.
_RATIONALE_KEYS = ("rationale", "reason", "justification", "explanation", "why")

#: Wrappers a model puts the eleven dimensions inside instead of at top level.
_CONTAINER_KEYS = ("scores", "dimensions", "ratings", "rubric", "results")

#: Reasoning channels some models emit before the answer. Stripped before the
#: brace scan so a `{` inside the model's private thinking cannot be mistaken
#: for the start of the answer.
_THINK_RE = re.compile(r"<(think|thinking|analysis|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)

#: `4`, `4/5`, `4.0`, `score: 4`, `four` is *not* accepted — a model that spells
#: numbers is a prompt problem, and silently guessing is worse than a rejection.
_SCORE_RE = re.compile(r"-?\d+(?:\.\d+)?")

#: Dimension labels ("Naming", "Invest in the Beginning") accepted alongside the
#: canonical keys, since a judge reading the rubric block sometimes echoes them.
_LABEL_TO_KEY = {d.label.split("(")[0].strip().casefold(): d.key for d in DIMENSIONS.values()}

_MAX_SAFETY_FLAGS = 8
_MAX_FLAG_CHARS = 64
_MAX_RATIONALE_CHARS = 1_000


class JudgeParseError(ValueError):
    """No JSON object could be recovered from the model's output at all.

    Distinct from "the JSON was fine but a dimension was missing": this means
    the sample is unusable and should be retried or recorded as an error, not
    partially salvaged.
    """


@dataclass(frozen=True, slots=True)
class ParsedDimension:
    """One dimension as the model stated it. Nothing is validated here yet."""

    score: int | None
    span: str | None
    rationale: str = ""


@dataclass(slots=True)
class ParsedJudgeOutput:
    """The model's answer, unpacked but not yet judged for admissibility."""

    dimensions: dict[str, ParsedDimension] = field(default_factory=dict)
    safety_flags: list[str] = field(default_factory=list)
    #: Keys that looked like dimensions but are not in `RUBRIC_DIMENSIONS`.
    unknown_keys: list[str] = field(default_factory=list)

    def get(self, key: str) -> ParsedDimension:
        """The dimension, or an all-`None` placeholder if the model omitted it."""
        return self.dimensions.get(key, ParsedDimension(score=None, span=None))


# ---------------------------------------------------------------------------
# Locating the JSON
# ---------------------------------------------------------------------------


def extract_json_object(raw: str) -> dict[str, Any]:
    """Recover the first complete JSON object in `raw`.

    Handles a bare object, a fenced object, an object after a preamble, and an
    object followed by trailing commentary. The scan is brace-balanced and
    string-aware, so a `{` or `}` inside a quoted evidence span — which is
    common, spans are prose — does not terminate the object early.
    """
    if not raw or not raw.strip():
        raise JudgeParseError("judge returned empty output")

    text = _THINK_RE.sub(" ", raw)

    # Strip code fences wholesale; the brace scan handles the rest.
    text = re.sub(r"```(?:json|JSON)?", " ", text)

    start = text.find("{")
    while start >= 0:
        candidate = _balanced_object(text, start)
        if candidate is not None:
            try:
                loaded = json.loads(candidate)
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, dict):
                return loaded
        start = text.find("{", start + 1)

    raise JudgeParseError(f"no JSON object found in judge output (first 200 chars: {raw[:200]!r})")


def _balanced_object(text: str, start: int) -> str | None:
    """The substring from `start` to its matching close brace, or `None`."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


def _first_key(obj: dict[str, Any], names: tuple[str, ...]) -> Any:
    lowered = {str(k).casefold(): v for k, v in obj.items()}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def _coerce_score(value: Any) -> int | None:
    """`4`, `4.0`, `"4"`, `"4/5"` -> 4. Anything else -> `None`.

    Out-of-range values survive coercion deliberately: `0` and `7` are recorded
    as stated and rejected later with `SCORE_OUT_OF_RANGE`, which is a more
    useful signal than a parse failure that looks like a formatting problem.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value)
    if isinstance(value, str):
        match = _SCORE_RE.search(value)
        if match is None:
            return None
        try:
            return round(float(match.group()))
        except ValueError:  # pragma: no cover - regex already guarantees a number
            return None
    return None


def _coerce_text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):  # a model occasionally returns ["quote"]
        value = " ".join(str(v) for v in value)
    text = str(value).strip()
    if not text:
        return None
    return text[:limit]


def _coerce_dimension(value: Any) -> ParsedDimension:
    """One dimension's payload, in any of the shapes models actually emit."""
    if isinstance(value, dict):
        return ParsedDimension(
            score=_coerce_score(_first_key(value, _SCORE_KEYS)),
            span=_coerce_text(_first_key(value, _SPAN_KEYS), 4_000),
            rationale=_coerce_text(_first_key(value, _RATIONALE_KEYS), _MAX_RATIONALE_CHARS) or "",
        )
    # A bare number is a score with no span, which is a grounding failure rather
    # than a parse failure — it is recorded as such and rejected downstream.
    return ParsedDimension(score=_coerce_score(value), span=None)


def _coerce_flags(value: Any) -> list[str]:
    """Safety flags are model-authored strings landing in a TEXT[] column.

    Bounded in count and length before they get anywhere near the database.
    """
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    flags: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            flags.append(text[:_MAX_FLAG_CHARS])
        if len(flags) >= _MAX_SAFETY_FLAGS:
            break
    return flags


def parse_judge_output(raw: str) -> ParsedJudgeOutput:
    """Full parse: locate the JSON, unwrap the container, coerce each dimension.

    Raises `JudgeParseError` only when no JSON object exists at all. Every other
    problem is represented as a missing or malformed field, so the grounding
    layer — not this one — decides what is admissible.
    """
    obj = extract_json_object(raw)

    container: dict[str, Any] = obj
    for key in _CONTAINER_KEYS:
        nested = _first_key(obj, (key,))
        if isinstance(nested, dict):
            container = nested
            break

    out = ParsedJudgeOutput()
    for raw_key, value in container.items():
        key = str(raw_key).strip().casefold()
        canonical_key = key if key in DIMENSIONS else _LABEL_TO_KEY.get(key)
        if canonical_key is None:
            if key not in {*_CONTAINER_KEYS, "safety_flags", "safety", "flags", "notes"}:
                out.unknown_keys.append(str(raw_key))
            continue
        out.dimensions[canonical_key] = _coerce_dimension(value)

    out.safety_flags = _coerce_flags(
        _first_key(obj, ("safety_flags", "safety", "flags"))
        or _first_key(container, ("safety_flags", "safety", "flags"))
    )
    return out


def score_in_range(score: int | None) -> bool:
    """True if `score` is on the 1-5 rubric scale."""
    return score is not None and SCALE_MIN <= score <= SCALE_MAX
