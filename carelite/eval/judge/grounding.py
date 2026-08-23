"""Verbatim evidence-span grounding. **The judge's integrity check.**

Build plan v3 §13 makes one non-negotiable demand of the judge: *every score
requires a verbatim evidence span from the response*. This module is where that
demand is enforced, and the enforcement is deliberately blunt — a score whose
span cannot be located in the text it claims to quote is **rejected**, not kept
with a warning. A rejected dimension comes back as `None`, which propagates to a
NULL in `rubric_score` and is visibly missing in the analysis, rather than
becoming a plausible-looking number backed by a quote the model invented.

Why this matters more than it looks: an unusable score that is *absent* costs
one cell. An unusable score that is *present* contaminates every mean it enters,
and nothing downstream can tell it apart from a real one.

Matching is verbatim but not byte-pedantic. Models reliably reproduce the words
and unreliably reproduce the typography — a curly apostrophe becomes straight, a
line break becomes a space, an em dash becomes a hyphen. Insisting on byte
equality would reject honest quotes and reward nothing, so `locate` matches on a
canonical form and then returns the **original slice**, which is what gets stored.
What it will not do is match a paraphrase: the word sequence must be present.

Grounding runs against two texts, in order:

1. the original `generation.response`, so the stored span is verbatim in the
   row the analysis reads;
2. failing that, the *presented* text — `fencing.sanitize_untrusted(response)`,
   which is what the judge actually saw. These differ only when sanitisation
   changed something (a forged fence marker, a truncation), and a quote that is
   verbatim in what the model was shown is honest evidence even so. Spans found
   only there are flagged `source="presented"` and counted separately in the
   validation report.

Pure and deterministic: no I/O, no model calls, no configuration.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "MAX_SPAN_CHARS",
    "MIN_SPAN_CHARS",
    "GroundedSpan",
    "SpanRejection",
    "canonical",
    "ground_span",
    "locate",
]

#: Shorter than this and a "span" is not evidence — "I" or "we" occurs in almost
#: any response and would let a model satisfy the grounding rule with noise. The
#: one exception is a response that is itself shorter than the floor, which is
#: allowed to be quoted whole.
MIN_SPAN_CHARS = 8

#: Longer than this and a "span" is not a citation but a copy of the response.
#: A quote is a claim about *where* in the text the evidence sits; a 1,500-char
#: block makes no such claim. Responses shorter than the ceiling may still be
#: quoted in full — the cap only bites on genuinely long ones.
MAX_SPAN_CHARS = 600


class SpanRejection(StrEnum):
    """Why a dimension's score was thrown away.

    These are counted per-reason in the validation report because they mean very
    different things. `NOT_FOUND` is the judge inventing a quote — a fabrication
    rate. `MISSING` is the judge declining to quote — a compliance rate. Rolling
    them into one "ungrounded" number would hide which failure we have.
    """

    #: No span field, or a blank one.
    MISSING = "missing_span"
    #: Below `MIN_SPAN_CHARS` and not the whole response.
    TOO_SHORT = "span_too_short"
    #: Above `MAX_SPAN_CHARS`: a copy of the response, not a citation.
    TOO_LONG = "span_too_long"
    #: The span is not in the response, in any normalisation. A hallucinated quote.
    NOT_FOUND = "span_not_found"
    #: The response itself is empty, so no span could ever be grounded.
    EMPTY_RESPONSE = "empty_response"
    #: The model returned no score for this dimension at all.
    NO_SCORE = "no_score"
    #: A score outside the 1-5 rubric scale.
    SCORE_OUT_OF_RANGE = "score_out_of_range"


@dataclass(frozen=True, slots=True)
class GroundedSpan:
    """A span that was actually found, with its location in the source text."""

    #: The **original** slice, exactly as it appears in `source_text`. This is
    #: what gets persisted, not the string the model emitted.
    text: str
    start: int
    end: int
    #: `"response"` (found in the stored generation) or `"presented"` (found only
    #: in the sanitised text the judge was shown).
    source: str
    #: True if the model's string was byte-identical to the slice. False means
    #: it matched only after typography canonicalisation.
    exact: bool


# ---------------------------------------------------------------------------
# Canonicalisation with an offset map
# ---------------------------------------------------------------------------

#: Typography that models rewrite without changing a single word. Folded before
#: matching so an honest quote is not rejected for a straightened apostrophe.
_FOLD: dict[str, str] = {
    "‘": "'",  # noqa: RUF001
    "’": "'",  # noqa: RUF001
    "‚": "'",  # noqa: RUF001
    "‛": "'",  # noqa: RUF001
    "ʼ": "'",  # noqa: RUF001
    "´": "'",  # noqa: RUF001
    "`": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "«": '"',
    "»": '"',
    "–": "-",  # noqa: RUF001
    "—": "-",
    "―": "-",
    "−": "-",  # noqa: RUF001
    "…": "...",
    " ": " ",  # noqa: RUF001
}

#: Rendered as nothing. Dropped outright so they cannot break a match.
_INVISIBLE = frozenset("­᠎​‌‍‎‏‪‫‬‭‮⁠⁡⁢⁣⁤⁦⁧⁨⁩﻿")


def canonical(text: str) -> tuple[str, list[int], list[int]]:
    """Canonical form of `text`, plus a map back to original offsets.

    Returns `(canon, starts, ends)` where `canon[i]` was produced by the original
    slice `text[starts[i]:ends[i]]`. The map is what makes it possible to return
    the *original* bytes for a match found in canonical space — without it we
    would have to store the model's own rendering of the quote, which is a
    different string from the one in the database.

    Canonicalisation is: NFKC, fold the typography in `_FOLD`, drop invisibles,
    collapse every whitespace run to one space, lowercase. Nothing here can
    reorder or delete a word, so a paraphrase still fails to match.
    """
    canon: list[str] = []
    starts: list[int] = []
    ends: list[int] = []

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        if ch in _INVISIBLE:
            i += 1
            continue

        if ch.isspace():
            j = i
            while j < n and (text[j].isspace() or text[j] in _INVISIBLE):
                j += 1
            canon.append(" ")
            starts.append(i)
            ends.append(j)
            i = j
            continue

        folded = _FOLD.get(ch)
        if folded is None:
            folded = unicodedata.normalize("NFKC", ch)
            folded = "".join(_FOLD.get(c, c) for c in folded)
        folded = folded.lower()

        # One original character may canonicalise to several (e.g. "…" -> "...").
        # Every product maps back to the same original slice, which keeps the
        # offset map total without pretending to a precision we do not have.
        for c in folded:
            canon.append(c)
            starts.append(i)
            ends.append(i + 1)
        i += 1

    return "".join(canon), starts, ends


def locate(needle: str, haystack: str, *, source: str = "response") -> GroundedSpan | None:
    """Find `needle` in `haystack`, tolerating typography. `None` if absent.

    Tries byte-exact first, so the common case costs one `str.find` and reports
    `exact=True`. Falls back to canonical matching, and returns the original
    slice of `haystack` rather than the caller's string.
    """
    probe = needle.strip()
    if not probe or not haystack:
        return None

    idx = haystack.find(probe)
    if idx >= 0:
        return GroundedSpan(
            text=haystack[idx : idx + len(probe)],
            start=idx,
            end=idx + len(probe),
            source=source,
            exact=True,
        )

    canon_hay, starts, ends = canonical(haystack)
    canon_needle, _, _ = canonical(probe)
    canon_needle = canon_needle.strip()
    if not canon_needle:
        return None

    pos = canon_hay.find(canon_needle)
    if pos < 0:
        return None

    start = starts[pos]
    end = ends[pos + len(canon_needle) - 1]
    return GroundedSpan(text=haystack[start:end], start=start, end=end, source=source, exact=False)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def ground_span(
    span: str | None,
    response: str,
    presented: str | None = None,
) -> tuple[GroundedSpan | None, SpanRejection | None]:
    """Apply the v3 §13 grounding rule to one claimed evidence span.

    Returns `(grounded, None)` if the span is admissible, or `(None, reason)` if
    the score it supports must be rejected. Exactly one of the two is ever set.

    Args:
        span: What the judge claimed to quote. `None` or blank is a rejection.
        response: The stored `generation.response`. Checked first, so an
            admissible span is verbatim in the row the analysis will read.
        presented: The sanitised text the judge was actually shown, if it
            differs. Checked second; a match here is real evidence but is
            flagged so the validation report can count how often it happens.
    """
    if not response.strip():
        return None, SpanRejection.EMPTY_RESPONSE
    if span is None or not span.strip():
        return None, SpanRejection.MISSING

    probe = span.strip()
    if len(probe) > MAX_SPAN_CHARS:
        return None, SpanRejection.TOO_LONG
    # A response too short to contain an eight-character quote may be quoted
    # whole; a response long enough to have one must supply one.
    if len(probe) < MIN_SPAN_CHARS and len(response.strip()) >= MIN_SPAN_CHARS:
        return None, SpanRejection.TOO_SHORT

    found = locate(probe, response, source="response")
    if found is not None:
        return found, None

    if presented is not None and presented != response:
        found = locate(probe, presented, source="presented")
        if found is not None:
            return found, None

    return None, SpanRejection.NOT_FOUND
