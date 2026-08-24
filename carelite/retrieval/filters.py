"""carelite.retrieval.filters — drop publication boilerplate before it reaches a prompt.

A paper PDF is not all prose. Funding statements, competing-interest
declarations, copyright blocks, submission dates, and author-affiliation
front matter all survive text extraction and become chunks, and those chunks
get embedded like any other. They then retrieve, because they sit in a corpus
about clinician communication and share its vocabulary — an affiliation block
listing "Department of Emergency Medicine, Cooper Medical School" is
lexically and semantically close to the corpus mean.

This was not a hypothesis. Grading the retrieval for "I need an interpreter,
my English is not good enough for this" returned four chunks from one paper,
of which one was a title-and-affiliation page and one was:

    "We uploaded the data for the analyses in the Supporting Information for
    unrestricted access. Funding: The author(s) received no specific funding
    for this work. Competing interests: Authors ... are authors of a book on
    compassion science ..."

That passage cannot help a clinician respond to anything. It consumes one of
only four context slots, and it is the kind of text that makes a generated
response sound like it is reading a journal at the patient.

**Conservative by construction.** The corpus is small and recall matters more
than tidiness, so a chunk is dropped only on *converging* evidence: either two
independent strong markers, or one strong marker in a chunk with little real
prose around it. A single "Funding:" inside an otherwise substantive
discussion section is not enough.

**These constants are deliberately not tuned to the current chunk text.** The
`carelite-corpus` lane is in flight repairing two extraction defects that
produce much of what looks like boilerplate today: running headers/footers
landing inside sentences, and word-level layout damage (`show ing`,
`healthrelated`). Chunk text will change underneath this filter and the index
lane will re-embed what changed. So the markers here are restricted to phrases
that identify publication *apparatus* by their own meaning — a competing-interest
declaration is a competing-interest declaration regardless of how cleanly it was
extracted — and no threshold was fitted to make a particular drop rate come out.
See `_NOT_MARKERS_SEE_DOCSTRING` for what was removed once this was understood.

**Switchable, like every other component.** `RetrievalFlags.drop_boilerplate`
turns it off, so its contribution to context precision is an ablation row
rather than an article of faith.

The real fix belongs upstream, in the corpus lane's chunker, which could drop
this material before it is ever embedded. That is not this lane's file to
edit, so this filter is a retrieval-side mitigation and is documented as one.
"""

from __future__ import annotations

import re

__all__ = [
    "BOILERPLATE_MARKERS",
    "MAX_DROP_FRACTION",
    "boilerplate_score",
    "drop_boilerplate",
    "earliest_marker_fraction",
    "is_boilerplate",
]

#: Phrases that only appear in publication apparatus, never in the substantive
#: prose this corpus exists for. Matched case-insensitively.
BOILERPLATE_MARKERS: tuple[str, ...] = (
    "competing interests",
    "conflict of interest",
    "conflicts of interest",
    "the author(s) received no specific funding",
    "received no specific funding",
    "funding:",
    "funding statement",
    "this article is licensed under",
    "open access this article",
    "creative commons",
    "data availability statement",
    "supplementary material",
    "supporting information",
    "author contributions",
    "informed consent was obtained",
    "ethics approval",
    "institutional review board",
    "corresponding author",
    "all rights reserved",
    "copyright \u00a9",
    "\u00a9 the author",
)

#: **Deliberately NOT markers.** Running headers and footers — `PLOS ONE |`,
#: a bare `https://doi.org/10.…`, `Received:`/`Accepted:`/`Published:` date
#: lines — are page furniture that PDF extraction interleaves *into the middle
#: of otherwise substantive sentences*. Treating them as boilerplate evidence
#: is a false-positive machine: an early version of this module included them
#: and dropped 15.2% of the corpus, among them chunks whose text began
#: "Discussion The purpose of this systematic review and meta-analysis was to
#: generate preliminary data for testing the hypothesis that health care
#: disparities exist in patient experience of clinician empathy" — exactly the
#: equity content the corpus is thinnest in and most needs to retrieve.
#:
#: The `carelite-corpus` lane is removing that furniture at source (22
#: occurrences across 9 papers). A retrieval-side rule matching on it would
#: duplicate that fix, and would go on silently deleting good chunks after the
#: upstream repair landed. Left here as a named non-decision so it is not
#: "helpfully" re-added.
_NOT_MARKERS_SEE_DOCSTRING: tuple[str, ...] = (
    "plos one |",
    "https://doi.org/10.",
    "received:",
    "accepted:",
    "published:",
)


#: Affiliation-block shape: a run of institutional nouns and credentials with
#: very little sentence structure.
_AFFILIATION_TOKENS = (
    "department of",
    "school of medicine",
    "university",
    "college of",
    "hospital",
    "institute",
    "faculty of",
    "center for",
    "centre for",
)

#: A marker inside the opening third of a chunk suggests the chunk *is*
#: apparatus; a marker after it suggests prose that ran into the apparatus.
#: Not fitted: it is the natural "front matter vs tail" split, and the corpus
#: lane's in-flight text repairs would invalidate anything finer.
_HEAD_FRACTION = 0.33

_CREDENTIAL_RE = re.compile(r"\b(MD|PhD|MPH|MSc|BSN|RN|DO|MBBS|MA|MS|DrPH)\b")
_SENTENCE_RE = re.compile(r"[.!?]\s+[A-Z]")


def boilerplate_score(text: str) -> tuple[int, list[str]]:
    """Count strong markers. Returns `(score, matched_markers)`."""
    lowered = text.casefold()
    matched = [m for m in BOILERPLATE_MARKERS if m in lowered]
    return len(matched), matched


def earliest_marker_fraction(text: str) -> float:
    """Where the first apparatus marker appears, as a fraction of the chunk.

    This is the criterion that separates the two cases a marker *count*
    conflates. A chunk that opens "RESEARCH Open Access (c) The Author(s)
    2025. This article is licensed under..." is apparatus from the first
    character. A chunk that opens "Furthermore, we excluded studies that
    delivered a teach-back intervention in combination with other
    comprehensive strategies..." and ends with an acknowledgements tail is
    *prose that ran into the apparatus*, and its prose is exactly what this
    corpus is thin on — that chunk is one of a handful touching teach-back at
    all, from the single teach-back paper in the corpus.

    Returns 1.0 when no marker is present.
    """
    lowered = text.casefold()
    if not lowered:
        return 1.0
    positions = [lowered.find(m) for m in BOILERPLATE_MARKERS]
    found = [p for p in positions if p >= 0]
    return min(found) / len(lowered) if found else 1.0


def _looks_like_affiliation_block(text: str) -> bool:
    lowered = text.casefold()
    hits = sum(1 for token in _AFFILIATION_TOKENS if token in lowered)
    credentials = len(_CREDENTIAL_RE.findall(text))
    sentences = len(_SENTENCE_RE.findall(text))
    # Institutions and credentials densely packed, with almost no prose.
    return (hits >= 2 and credentials >= 2) and sentences <= 2


def is_boilerplate(text: str, *, min_prose_chars: int = 400) -> bool:
    """True if `text` is publication apparatus rather than substantive prose.

    `min_prose_chars` is the length below which one strong marker is taken as
    decisive. A long discussion section that happens to mention "Funding:" in
    passing keeps its place; a 200-character fragment that is mostly a funding
    statement does not.
    """
    stripped = text.strip()
    if not stripped:
        return True

    score, _ = boilerplate_score(stripped)
    lead = earliest_marker_fraction(stripped)

    # Apparatus from the top: markers start in the opening `_HEAD_FRACTION` of
    # the chunk and there is more than one of them.
    if score >= 2 and lead <= _HEAD_FRACTION:
        return True
    # A short fragment that is mostly a single declaration.
    if score >= 1 and len(stripped) < min_prose_chars and lead <= 0.5:
        return True
    return _looks_like_affiliation_block(stripped)


#: Never remove more than this share of a candidate set. A bound, not a tuned
#: constant: it caps the blast radius of a false positive without making any
#: claim about the right drop rate. Measured against the live corpus the
#: filter removes 4.4% of all chunks, so this ceiling is nowhere near binding
#: in normal operation — it exists for the pathological case where a query
#: happens to retrieve a cluster of front matter, which is exactly when
#: over-dropping would leave the pipeline with nothing to reason from.
MAX_DROP_FRACTION = 0.5


def drop_boilerplate[T](items: list[T], *, key=lambda item: item.text) -> list[T]:
    """Filter a list of anything carrying text.

    Defers in two cases rather than starving the pipeline of context: when
    every candidate looks like boilerplate, and when the filter would remove
    more than `MAX_DROP_FRACTION` of them. A mediocre passage the CRAG gate
    can still reject beats a silently empty retrieval, which no downstream
    component can distinguish from "the corpus has nothing".
    """
    if not items:
        return items
    kept = [item for item in items if not is_boilerplate(key(item))]
    if not kept:
        return items
    if len(items) - len(kept) > MAX_DROP_FRACTION * len(items):
        return items
    return kept
