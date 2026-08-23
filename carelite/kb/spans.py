"""Verbatim-span location: normalise, then find, then recover the true source text.

This module is the mechanism behind the KB's provenance guarantee. An entry
claims a `verbatim_span` quoted from a paper; the validator's job is to prove
that quote exists. Proving it needs normalisation, because the same sentence
survives a PDF round-trip in several forms:

    "confi rm the patient's understanding"   (ligature split by the extractor)
    "confirm the patient's understanding"    (curly apostrophe)
    "con-\\nfirm the patient's under-\\nstanding"  (line-broken hyphenation)
    "confirm  the patient's\\nunderstanding"  (whatever whitespace the column had)

All four are the same sentence and all four must match. None of that is
fuzzy matching: every transform here is glyph-level and reversible in meaning.
Nothing collapses different words together, nothing drops content words,
nothing tolerates a missing or altered clause. A span that still cannot be
found after this is not a formatting variant — it is text that is not in the
paper, and the validator must reject it.

The important design choice is `locate_span`. It matches on the normalised
form but returns the **character offsets into the original text**, so the
caller can replace the model's paraphrase-shaped quote with the exact source
substring. What lands in `kb_entry.verbatim_span` is therefore literally a
slice of the extracted paper, not the model's rendering of one. That turns
"we checked the quote" into "the quote is the source", which is a materially
stronger guarantee and makes the review digest trustworthy by construction.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

#: Glyph folds NFKC does not perform. Quotation marks and dashes are the two
#: things a PDF extractor and a language model most reliably disagree about,
#: and neither carries meaning that distinguishes one claim from another.
_CHAR_FOLDS: dict[str, str] = {
    "‘": "'",  # left single quote  # noqa: RUF001 - the ambiguous glyph is the point: this table folds it
    "’": "'",  # right single quote / apostrophe  # noqa: RUF001 - the ambiguous glyph is the point: this table folds it
    "‚": "'",  # noqa: RUF001 - the ambiguous glyph is the point: this table folds it
    "‛": "'",  # noqa: RUF001 - the ambiguous glyph is the point: this table folds it
    "′": "'",  # prime  # noqa: RUF001 - the ambiguous glyph is the point: this table folds it
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "„": '"',
    "‟": '"',
    "″": '"',
    "‐": "-",  # hyphen  # noqa: RUF001 - the ambiguous glyph is the point: this table folds it
    "‑": "-",  # non-breaking hyphen  # noqa: RUF001 - the ambiguous glyph is the point: this table folds it
    "‒": "-",  # figure dash  # noqa: RUF001 - the ambiguous glyph is the point: this table folds it
    "–": "-",  # en dash  # noqa: RUF001 - the ambiguous glyph is the point: this table folds it
    "—": "-",  # em dash
    "―": "-",  # horizontal bar
    "−": "-",  # minus sign  # noqa: RUF001 - the ambiguous glyph is the point: this table folds it
    " ": " ",  # nbsp  # noqa: RUF001 - the ambiguous glyph is the point: this table folds it
    " ": " ",  # narrow nbsp  # noqa: RUF001 - the ambiguous glyph is the point: this table folds it
    " ": " ",  # figure space  # noqa: RUF001 - the ambiguous glyph is the point: this table folds it
    " ": " ",  # thin space  # noqa: RUF001 - the ambiguous glyph is the point: this table folds it
}

#: Removed outright: they are invisible and carry no textual content.
_DROPPED = {
    "­",  # soft hyphen
    "​",  # zero-width space
    "‌",  # zero-width non-joiner
    "‍",  # zero-width joiner
    "﻿",  # BOM / zero-width nbsp
}


def _fold_char(ch: str) -> str:
    """One character -> its normalised form, which may be zero or many characters.

    NFKC handles the ligatures (ﬁ -> 'fi', ﬂ -> 'fl') that PDF
    extraction leaves behind; the explicit table above handles what NFKC
    deliberately leaves alone. Case folding happens here too, so the offset
    bookkeeping in `normalize` stays correct even for the handful of
    characters whose lowercase form is longer than their uppercase one
    (U+0130, for instance).
    """
    if ch in _DROPPED:
        return ""
    ch = _CHAR_FOLDS.get(ch, ch)
    return unicodedata.normalize("NFKC", ch).lower()


@dataclass(frozen=True)
class NormalizedText:
    """Normalised text plus a map back to where each character came from.

    `offsets[i]` is the index in `source` of the character that produced
    `text[i]`. `source_slice` uses it to recover an original-text span for a
    match found in the normalised form.
    """

    text: str
    offsets: tuple[int, ...]
    source: str

    def source_slice(self, start: int, end: int) -> tuple[int, int]:
        """Normalised `[start, end)` -> original `[a, b)`.

        The end bound maps through the *last included* character rather than
        through `end` itself, because `end` may sit past the final normalised
        character or inside a run of collapsed whitespace.
        """
        if not (0 <= start < end <= len(self.text)):
            raise ValueError(f"span [{start}, {end}) out of range for {len(self.text)} characters")
        return self.offsets[start], self.offsets[end - 1] + 1


def normalize(text: str) -> NormalizedText:
    """Fold glyphs, join hyphenated line breaks, collapse whitespace, lowercase.

    Hyphenation joining removes a character the source really contains: a
    hyphen between an alphabetic character and a whitespace run followed by a
    lowercase letter is dropped, so `con-\\nfirm` becomes `confirm`. That rule
    also fires on constructions like `pre- and post-course`, which normalises
    to the non-word `preand post-course`.

    That is deliberate and harmless, because normalisation here only has to be
    *consistent*, not *correct*. Both the claimed span and the paper go through
    this same function, so both mangle identically and still match each other;
    and what gets stored is recovered from the original text through
    `NormalizedText.offsets`, never from the normalised form. The only risk a
    more aggressive fold carries is a false positive, and dropping a hyphen
    cannot turn one claim into a different one. Failing to join a real
    line-break hyphenation, by contrast, would reject a genuine quote — the
    error this module must not make.
    """
    buf: list[str] = []
    offsets: list[int] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch.isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            joins_hyphenation = (
                len(buf) >= 2
                and buf[-1] == "-"
                and buf[-2].isalpha()
                and j < n
                and text[j].islower()
            )
            if joins_hyphenation:
                buf.pop()
                offsets.pop()
            elif buf and buf[-1] != " ":
                buf.append(" ")
                offsets.append(i)
            i = j
            continue

        folded = _fold_char(ch)
        for out_ch in folded:
            buf.append(out_ch)
            offsets.append(i)
        i += 1

    while buf and buf[-1] == " ":
        buf.pop()
        offsets.pop()

    return NormalizedText(text="".join(buf), offsets=tuple(offsets), source=text)


def normalized_text(text: str) -> str:
    """`normalize(text).text`, for callers that do not need the offset map."""
    return normalize(text).text


@dataclass(frozen=True)
class SpanMatch:
    """Where a span was found, and what the source actually says there.

    `source_text` is the authoritative form: an exact substring of the paper.
    `exact` records whether the claimed span already matched the source
    byte-for-byte, or only after normalisation — a distinction the review
    digest surfaces so a human can see how much cleanup was applied.
    """

    start: int
    end: int
    source_text: str
    exact: bool

    @property
    def length(self) -> int:
        return self.end - self.start


def locate_span(
    span: str,
    document: str,
    *,
    normalized_document: NormalizedText | None = None,
) -> SpanMatch | None:
    """Find `span` in `document`. Returns `None` if it is not there.

    `normalized_document` lets a caller normalise a paper once and reuse it
    across every candidate entry from that paper, which matters: normalising
    a 100 KB paper per candidate would dominate validation runtime.

    Returning `None` rather than raising is deliberate — a missing span is an
    expected outcome of LLM-assisted extraction, not an exceptional one, and
    the validator turns it into a rejection with a reason.
    """
    needle = normalized_text(span)
    if not needle:
        return None

    haystack = normalized_document if normalized_document is not None else normalize(document)
    if haystack.source is not document and haystack.source != document:
        raise ValueError(
            "normalized_document was built from different text than `document`; "
            "the offsets would point into the wrong string and the recovered "
            "span would be a fabrication with a provenance check attached to it"
        )

    idx = haystack.text.find(needle)
    if idx < 0:
        return None

    start, end = haystack.source_slice(idx, idx + len(needle))
    return SpanMatch(
        start=start,
        end=end,
        source_text=document[start:end],
        exact=span in document,
    )


def surrounding_context(
    document: str, start: int, end: int, *, window: int = 320
) -> tuple[str, str]:
    """The text immediately before and after a located span, for the review digest.

    Snapped outward to whitespace so the reviewer never sees a context
    fragment that starts or ends mid-word.
    """
    left = max(0, start - window)
    right = min(len(document), end + window)

    while left > 0 and not document[left].isspace():
        left -= 1
    while right < len(document) and not document[right - 1].isspace():
        right += 1

    return document[left:start].strip(), document[end:right].strip()


__all__ = [
    "NormalizedText",
    "SpanMatch",
    "locate_span",
    "normalize",
    "normalized_text",
    "surrounding_context",
]
