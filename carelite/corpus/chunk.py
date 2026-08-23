"""carelite.corpus.chunk — semantic chunking to ~512 tokens with ~64 overlap.

Sizes come from `settings.retrieval.chunk_target_tokens` /
`chunk_overlap_tokens`. Token counts are a whitespace-word approximation —
no BPE tokenizer dependency is declared for this lane — good enough to keep
chunks in the right ballpark; not a token-exact count.

Chunking works over sentence/heading "units" and only ever closes a chunk
between units, so it never splits mid-sentence. It force-breaks (no overlap
carried across) at a detected section heading, and otherwise closes once the
running token count would exceed the target, carrying the trailing ~overlap
tokens' worth of sentences into the next chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from carelite.config import get_settings
from carelite.types import Chunk

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“(])")

_KNOWN_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "methods",
    "method",
    "materials and methods",
    "results",
    "result",
    "discussion",
    "discussions",
    "conclusion",
    "conclusions",
    "limitations",
    "acknowledgements",
    "acknowledgments",
    "funding",
}
_NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*[.:)]?\s+[A-Z][A-Za-z ,'-]{2,60}$")


def _approx_tokens(text: str) -> int:
    """Word-count approximation of token length (no tokenizer dependency here)."""
    return len(text.split())


def _looks_like_heading(paragraph: str) -> bool:
    stripped = paragraph.strip()
    if not stripped or "\n" in stripped or len(stripped) > 80:
        return False
    bare = stripped.strip(":.").strip()
    if bare.lower() in _KNOWN_HEADINGS:
        return True
    if stripped.isupper() and 1 <= len(stripped.split()) <= 10:
        return True
    return bool(_NUMBERED_HEADING_RE.match(stripped))


def _split_sentences(paragraph: str) -> list[str]:
    flat = re.sub(r"\s+", " ", paragraph).strip()
    if not flat:
        return []
    return [p.strip() for p in _SENTENCE_SPLIT_RE.split(flat) if p.strip()]


@dataclass
class _Unit:
    text: str
    is_heading: bool


def _units(text: str) -> list[_Unit]:
    paragraphs = re.split(r"\n\s*\n", text)
    units: list[_Unit] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if _looks_like_heading(para):
            units.append(_Unit(para, is_heading=True))
            continue
        units.extend(_Unit(s, is_heading=False) for s in _split_sentences(para))
    return units


def chunk_text(
    paper_id: str,
    text: str,
    *,
    target_tokens: int | None = None,
    overlap_tokens: int | None = None,
    min_chunk_tokens: int | None = None,
) -> list[Chunk]:
    """Chunk one paper's cleaned text into `Chunk` objects, in reading order.

    `chunk_id` is `f"{paper_id}::{ordinal:04d}"` — stable across re-runs and
    lets `carelite.corpus.load` recover a monotonic per-paper ordinal without
    a separate field on the frozen `Chunk` model.
    """
    retrieval = get_settings().retrieval
    target = target_tokens or retrieval.chunk_target_tokens
    overlap = overlap_tokens if overlap_tokens is not None else retrieval.chunk_overlap_tokens
    min_tokens = min_chunk_tokens if min_chunk_tokens is not None else max(1, target // 5)

    units = _units(text)
    if not units:
        return []

    chunks_text: list[str] = []
    # Parallel to chunks_text: True if that chunk's first unit was a section
    # heading. A chunk that opens a new section is never merged into its
    # predecessor by the tiny-tail cleanup below, even if it's short — a short
    # final section (e.g. a one-line "Conclusion") is a real boundary, not a
    # fragment left over from a token-overflow split.
    chunk_starts_section: list[bool] = []
    buffer: list[_Unit] = []
    buffer_tokens = 0

    def flush(*, carry_overlap: bool) -> list[_Unit]:
        nonlocal buffer, buffer_tokens
        chunks_text.append(" ".join(u.text for u in buffer))
        chunk_starts_section.append(bool(buffer) and buffer[0].is_heading)
        carry: list[_Unit] = []
        if carry_overlap and overlap > 0:
            carry_tokens = 0
            for u in reversed(buffer):
                t = _approx_tokens(u.text)
                if carry_tokens + t > overlap and carry:
                    break
                carry.insert(0, u)
                carry_tokens += t
        buffer = []
        buffer_tokens = 0
        return carry

    i = 0
    made_progress = False  # True once >=1 *new* unit has been appended since the last flush
    while i < len(units):
        unit = units[i]
        unit_tokens = _approx_tokens(unit.text)

        if unit.is_heading and buffer:
            carry = flush(carry_overlap=False)  # fresh start at section boundaries
            buffer.extend(carry)
            buffer_tokens = sum(_approx_tokens(u.text) for u in buffer)
            made_progress = False
            continue

        if (
            made_progress
            and buffer
            and buffer_tokens + unit_tokens > target
            and buffer_tokens >= min_tokens
        ):
            carry = flush(carry_overlap=True)
            buffer.extend(carry)
            buffer_tokens = sum(_approx_tokens(u.text) for u in buffer)
            made_progress = False
            continue

        buffer.append(unit)
        buffer_tokens += unit_tokens
        made_progress = True
        i += 1

    if buffer:
        chunks_text.append(" ".join(u.text for u in buffer))
        chunk_starts_section.append(bool(buffer) and buffer[0].is_heading)

    # A near-empty trailing chunk (e.g. all that's left after the final flush's
    # overlap carry) reads better merged into its predecessor than standalone
    # — unless it's the start of its own section, which is a real boundary.
    if (
        len(chunks_text) >= 2
        and not chunk_starts_section[-1]
        and _approx_tokens(chunks_text[-1]) < min_tokens
    ):
        tail = chunks_text.pop()
        chunks_text[-1] = f"{chunks_text[-1]} {tail}"

    return [
        Chunk(chunk_id=f"{paper_id}::{ordinal:04d}", paper_id=paper_id, text=t)
        for ordinal, t in enumerate(chunks_text)
    ]


def ordinal_of(chunk: Chunk) -> int:
    """Recover the monotonic ordinal encoded in `chunk_id` by `chunk_text`."""
    try:
        return int(chunk.chunk_id.rsplit("::", 1)[-1])
    except ValueError as e:
        raise ValueError(f"chunk_id {chunk.chunk_id!r} has no recoverable ordinal") from e
