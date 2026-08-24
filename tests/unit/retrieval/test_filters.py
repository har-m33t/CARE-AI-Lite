"""Publication apparatus must not occupy one of only four context slots.

Every constant in `filters.py` is deliberately *not* tuned to the current
chunk text: the carelite-corpus lane is in flight repairing extraction defects
(running headers/footers landing inside sentences, and word-level layout
damage), so the text these rules see will change. The rules therefore key on
phrases that identify apparatus by their own meaning.
"""

from __future__ import annotations

from carelite.retrieval.filters import (
    MAX_DROP_FRACTION,
    boilerplate_score,
    drop_boilerplate,
    earliest_marker_fraction,
    is_boilerplate,
)

from .conftest import make_item

FRONT_MATTER = (
    "RESEARCH Open Access © The Author(s) 2025. Open Access This article is licensed "
    "under a Creative Commons Attribution 4.0 International License, which permits use, "
    "sharing, adaptation, distribution and reproduction in any medium."
)

FUNDING_BLOCK = (
    "We uploaded the data for the analyses in the Supporting Information for unrestricted "
    "access. Funding: The author(s) received no specific funding for this work. Competing "
    "interests: Authors are authors of a book on compassion science."
)

REAL_PROSE = (
    "Clinicians frequently miss empathic opportunities when patients express fear about a "
    "diagnosis. Naming the emotion before providing further information is associated with "
    "improved patient experience and greater disclosure in later turns of the consultation. "
    "Training interventions that target this behaviour show durable effects at six months."
)

PROSE_WITH_APPARATUS_TAIL = (
    "Furthermore, we excluded studies that delivered a teach-back intervention in "
    "combination with other comprehensive strategies so that the independent contribution "
    "of teach-back could be estimated. This restriction reduced the pool of eligible "
    "studies considerably but improves the interpretability of the pooled estimate. "
    "Supporting information S1 Appendix. Author contributions: conceptualisation."
)


def test_front_matter_is_dropped() -> None:
    assert is_boilerplate(FRONT_MATTER)


def test_funding_block_is_dropped() -> None:
    assert is_boilerplate(FUNDING_BLOCK)


def test_real_prose_is_kept() -> None:
    assert not is_boilerplate(REAL_PROSE)


def test_prose_with_an_apparatus_tail_is_kept() -> None:
    """The case that marker *counting* gets wrong. This chunk is one of a
    handful in the corpus touching teach-back at all — from the single
    teach-back paper — and an earlier count-based rule dropped it."""
    assert not is_boilerplate(PROSE_WITH_APPARATUS_TAIL)


def test_marker_position_separates_the_two_cases() -> None:
    assert earliest_marker_fraction(FRONT_MATTER) < 0.33
    assert earliest_marker_fraction(PROSE_WITH_APPARATUS_TAIL) > 0.33


def test_running_headers_are_not_treated_as_markers() -> None:
    """`PLOS ONE |` and bare DOIs are page furniture that extraction
    interleaves into good sentences. The carelite-corpus lane removes them at
    source; matching on them here dropped 15.2% of the corpus, including
    substantive equity content."""
    footer_prose = (
        "Discussion The purpose of this systematic review and meta-analysis was to "
        "generate preliminary data for testing the hypothesis that health care "
        "disparities exist in patient experience of clinician empathy. PLOS ONE | "
        "https://doi.org/10.1371/journal.pone.0247259 February 2021"
    )
    score, matched = boilerplate_score(footer_prose)
    assert score == 0, matched
    assert not is_boilerplate(footer_prose)


def test_a_single_marker_in_long_prose_is_not_enough() -> None:
    assert not is_boilerplate(REAL_PROSE + " Funding: none declared.")


def test_a_short_fragment_that_is_mostly_a_declaration_is_dropped() -> None:
    assert is_boilerplate("Competing interests: none declared.")


def test_empty_text_is_boilerplate() -> None:
    assert is_boilerplate("")
    assert is_boilerplate("   ")


def test_filter_defers_rather_than_returning_nothing() -> None:
    """A silently empty retrieval is indistinguishable downstream from "the
    corpus has nothing", which is a much worse failure than one weak passage
    the CRAG gate can still reject."""
    items = [make_item("a", FRONT_MATTER), make_item("b", FUNDING_BLOCK)]
    assert drop_boilerplate(items) == items


def test_filter_bounds_its_own_blast_radius() -> None:
    items = [
        make_item("a", FRONT_MATTER),
        make_item("b", FUNDING_BLOCK),
        make_item("c", REAL_PROSE),
    ]
    # Dropping 2 of 3 exceeds MAX_DROP_FRACTION, so the filter defers entirely.
    assert MAX_DROP_FRACTION == 0.5
    assert drop_boilerplate(items) == items


def test_filter_removes_a_minority_of_apparatus() -> None:
    items = [make_item(f"p{i}", REAL_PROSE) for i in range(4)] + [make_item("junk", FRONT_MATTER)]
    kept = drop_boilerplate(items)
    assert [i.ref_id for i in kept] == ["p0", "p1", "p2", "p3"]


def test_empty_input_is_returned_unchanged() -> None:
    assert drop_boilerplate([]) == []
