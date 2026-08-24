"""Unit tests for carelite.corpus.extract, using tiny synthetic PDFs built
with pymupdf itself so nothing here depends on a real paper being present."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from carelite.corpus import extract


def _make_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for page_text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), page_text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def test_extract_pdf_reports_failure_for_unopenable_file(tmp_path):
    bad = tmp_path / "not_a_pdf.pdf"
    bad.write_bytes(b"this is not a pdf at all")

    result = extract.extract_pdf(bad)

    assert not result.ok
    assert result.text == ""
    assert result.failures
    assert "could not open" in result.failures[0].reason


def test_extract_pdf_strips_repeated_running_header_and_footer(tmp_path):
    header = "CARE Journal of Communication"
    footer = "Confidential draft - do not distribute"
    pages = [
        f"{header}\nBody paragraph one on page {i}.\nMore body text here.\n{footer}"
        for i in range(1, 5)
    ]
    pdf = _make_pdf(tmp_path / "multi.pdf", pages)

    result = extract.extract_pdf(pdf)

    assert result.ok
    assert header not in result.text
    assert footer not in result.text
    assert "Body paragraph one on page 1" in result.text
    assert "Body paragraph one on page 4" in result.text


def test_extract_pdf_does_not_strip_headers_on_short_documents(tmp_path):
    # Fewer than 3 pages -> nothing to detect repetition against; a line that
    # happens to appear on both pages should NOT be treated as noise.
    pages = ["Shared Title\nReal content on page one.", "Shared Title\nReal content on page two."]
    pdf = _make_pdf(tmp_path / "short.pdf", pages)

    result = extract.extract_pdf(pdf)

    assert "Shared Title" in result.text


def test_extract_pdf_strips_references_section(tmp_path):
    body = "Findings suggest empathy improves comprehension.\n\nReferences\nSmith J. 2020. Some Journal."
    pdf = _make_pdf(tmp_path / "refs.pdf", [body])

    result = extract.extract_pdf(pdf)

    assert "Findings suggest empathy" in result.text
    assert "Smith J. 2020" not in result.text
    assert "Some Journal" not in result.text


def test_extract_pdf_strips_figure_and_table_captions(tmp_path):
    body = "Real finding text here.\nFigure 1: Study flow diagram.\nTable 2: Baseline characteristics.\nMore real text."
    pdf = _make_pdf(tmp_path / "captions.pdf", [body])

    result = extract.extract_pdf(pdf)

    assert "Real finding text here" in result.text
    assert "More real text" in result.text
    assert "Figure 1" not in result.text
    assert "Table 2" not in result.text


def test_extract_pdf_records_failure_for_empty_page(tmp_path):
    doc = pymupdf.open()
    doc.new_page()  # a page with no text at all
    pdf = tmp_path / "empty.pdf"
    doc.save(pdf)
    doc.close()

    result = extract.extract_pdf(pdf)

    assert not result.ok
    assert result.failures
    assert "no usable text" in result.failures[0].reason


def test_extract_corpus_aggregates_results_and_failures(tmp_path):
    _make_pdf(tmp_path / "good.pdf", ["Some real body text about patient communication."])
    (tmp_path / "bad.pdf").write_bytes(b"garbage")

    results, failures = extract.extract_corpus(tmp_path)

    assert len(results) == 2
    assert len(failures) == 1
    assert failures[0].source_path.endswith("bad.pdf")


def test_extract_corpus_default_dir_uses_settings(monkeypatch, tmp_path):
    _make_pdf(tmp_path / "only.pdf", ["Content here."])
    monkeypatch.setattr(extract, "get_settings", lambda: type("S", (), {"pdf_dir": tmp_path})())

    results, failures = extract.extract_corpus()

    assert len(results) == 1
    assert not failures


# ---------------------------------------------------------------------------
# JATS full-text XML (PMC / Europe PMC fullTextXML)
# ---------------------------------------------------------------------------

_JATS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <title-group><article-title>Empathy in Clinical Encounters</article-title></title-group>
      <abstract><p>This study examines empathy in clinical encounters.</p></abstract>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Introduction</title>
      <p>Empathy improves patient outcomes.</p>
      <fig><label>Figure 1</label><caption><p>Study flow diagram.</p></caption></fig>
      <p>Second sentence of the introduction.</p>
    </sec>
    <sec>
      <title>Methods</title>
      <p>We interviewed patients about their care.</p>
      <table-wrap><label>Table 1</label><caption><p>Baseline characteristics.</p></caption></table-wrap>
    </sec>
  </body>
  <back>
    <ref-list>
      <ref><citation>Smith J. 2020. Some Journal.</citation></ref>
    </ref-list>
  </back>
</article>
"""


def _make_xml(path: Path, content: str = _JATS_SAMPLE) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_extract_xml_jats_pulls_title_and_body_text(tmp_path):
    xml_path = _make_xml(tmp_path / "sample.xml")

    result = extract.extract_xml_jats(xml_path)

    assert result.ok
    assert result.source_kind == "xml"
    assert result.title == "Empathy in Clinical Encounters"
    assert "Empathy improves patient outcomes." in result.text
    assert "Second sentence of the introduction." in result.text
    assert "We interviewed patients about their care." in result.text


def test_extract_xml_jats_prepends_abstract(tmp_path):
    xml_path = _make_xml(tmp_path / "sample.xml")
    result = extract.extract_xml_jats(xml_path)
    assert "Abstract" in result.text
    assert "This study examines empathy in clinical encounters." in result.text
    # abstract precedes the body
    assert result.text.index("Abstract") < result.text.index("Introduction")


def test_extract_xml_jats_excludes_figure_and_table_captions_structurally(tmp_path):
    xml_path = _make_xml(tmp_path / "sample.xml")
    result = extract.extract_xml_jats(xml_path)
    assert "Study flow diagram" not in result.text
    assert "Baseline characteristics" not in result.text
    assert "Figure 1" not in result.text
    assert "Table 1" not in result.text


def test_extract_xml_jats_excludes_reference_list(tmp_path):
    xml_path = _make_xml(tmp_path / "sample.xml")
    result = extract.extract_xml_jats(xml_path)
    assert "Smith J. 2020" not in result.text


_JATS_SUPPLEMENTARY_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <title-group><article-title>A Study of Care Communication</article-title></title-group>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Results</title>
      <p>Patients reported higher satisfaction with structured communication.</p>
    </sec>
    <sec>
      <title>Supporting information</title>
      <supplementary-material id="s001" position="float">
        <label>S1 File</label>
        <caption><title>Minimal data set.</title><p>(PDF)</p></caption>
        <media xlink:href="s001.pdf"/>
      </supplementary-material>
      <supplementary-material id="s002" position="float">
        <label>S1 Graphical abstract</label>
        <caption><p>(TIF)</p></caption>
        <media xlink:href="s002.tif"/>
      </supplementary-material>
    </sec>
  </body>
</article>
"""


def test_extract_xml_jats_excludes_supplementary_material_structurally(tmp_path):
    """Regression test: PLOS-style <supplementary-material> wraps each
    attachment's format tag ("(PDF)", "(TIF)") in a <caption>, which used to
    leak out as a near-empty block — and then got misread as a section
    heading by chunk.py's ALL-CAPS check, becoming its own junk chunk."""
    xml_path = _make_xml(tmp_path / "supplementary.xml", _JATS_SUPPLEMENTARY_SAMPLE)
    result = extract.extract_xml_jats(xml_path)

    assert "(PDF)" not in result.text
    assert "(TIF)" not in result.text
    assert "Minimal data set" not in result.text
    assert "S1 File" not in result.text
    assert "Patients reported higher satisfaction" in result.text


def test_extract_xml_jats_preserves_section_headings_for_chunk_boundaries(tmp_path):
    """Headings come through as their own paragraph block, so
    carelite.corpus.chunk's heading-boundary detection still works unmodified
    on XML-derived text — the Chunk contract needs no changes."""
    xml_path = _make_xml(tmp_path / "sample.xml")
    result = extract.extract_xml_jats(xml_path)

    from carelite.corpus.chunk import chunk_text

    chunks = chunk_text("paper-x", result.text, target_tokens=1000, overlap_tokens=0)
    assert len(chunks) >= 2  # forced section breaks at "Introduction" and "Methods"
    assert any(c.text.startswith("Introduction") for c in chunks)
    assert any(c.text.startswith("Methods") for c in chunks)


def test_extract_xml_jats_reports_failure_for_malformed_xml(tmp_path):
    xml_path = tmp_path / "broken.xml"
    xml_path.write_bytes(b"<article><unclosed>")

    result = extract.extract_xml_jats(xml_path)

    assert not result.ok
    assert result.failures
    assert "could not parse" in result.failures[0].reason


def test_extract_xml_jats_reports_failure_for_no_usable_text(tmp_path):
    xml_path = _make_xml(tmp_path / "thin.xml", "<?xml version='1.0'?><article><body/></article>")

    result = extract.extract_xml_jats(xml_path)

    assert not result.ok
    assert "no usable text" in result.failures[0].reason


def test_extract_source_dispatches_on_suffix(tmp_path):
    pdf_result = extract.extract_source(_make_pdf(tmp_path / "a.pdf", ["Some content."]))
    assert pdf_result.source_kind == "pdf"

    xml_result = extract.extract_source(_make_xml(tmp_path / "b.xml"))
    assert xml_result.source_kind == "xml"


def test_extract_source_reports_failure_for_unsupported_suffix(tmp_path):
    other = tmp_path / "notes.txt"
    other.write_text("plain text")

    result = extract.extract_source(other)

    assert not result.ok
    assert "unsupported file type" in result.failures[0].reason


def test_iter_source_files_picks_up_both_pdf_and_xml(tmp_path):
    _make_pdf(tmp_path / "a.pdf", ["content"])
    _make_xml(tmp_path / "b.xml")
    (tmp_path / "ignore.txt").write_text("not a source file")

    found = list(extract.iter_source_files(tmp_path))
    assert {p.name for p in found} == {"a.pdf", "b.xml"}


def test_extract_corpus_handles_a_mixed_pdf_and_xml_directory(tmp_path):
    _make_pdf(tmp_path / "a.pdf", ["PDF content about empathy."])
    _make_xml(tmp_path / "b.xml")

    results, failures = extract.extract_corpus(tmp_path)

    assert len(results) == 2
    assert not failures
    kinds = {r.source_kind for r in results}
    assert kinds == {"pdf", "xml"}


# ---------------------------------------------------------------------------
# Running headers/footers that are more than one or two lines, and the
# per-page pagination line that varies too much to be caught by repetition.
#
# The original `edge_lines=2` default only ever looked at the last two lines
# of a page, so a real multi-line footer block (journal name, short title,
# DOI line, date, page number — each its own pymupdf "text"-mode line) only
# had its last line or two stripped. The rest survived into the extracted
# text and, once a page break happened to fall mid-sentence, landed as
# garbage words inside that sentence — see
# `test_extract_pdf_removes_footer_that_would_otherwise_land_mid_sentence`
# below, which reproduces the escalated corpus defect
# (10-1371-journal-pone-0247259) directly.
# ---------------------------------------------------------------------------


def test_extract_pdf_strips_multiline_running_footer_block(tmp_path):
    """A footer block wider than the old edge_lines=2 window must be
    detected in full, not just its last line or two."""
    footer_lines = (
        "JOURNAL OF CARE\nExample Study\nJOURNAL OF CARE | https://doi.org/10.1/x\nJanuary 1, 2020"
    )
    pages = [f"Body text discussing empathy on page {i}.\n{footer_lines}" for i in range(1, 6)]
    pdf = _make_pdf(tmp_path / "multiline_footer.pdf", pages)

    result = extract.extract_pdf(pdf)

    assert result.ok
    for line in footer_lines.splitlines():
        assert line not in result.text
    assert "Body text discussing empathy on page 1" in result.text


def test_extract_pdf_removes_footer_that_would_otherwise_land_mid_sentence(tmp_path):
    """Regression test for the escalated corpus defect: a multi-line running
    footer sits between the end of one page's text and the page break, right
    where a sentence continues onto the next page. Once every footer line is
    correctly recognised as noise, no fragment of it survives to land inside
    the sentence when its internal newlines are later flattened to spaces
    (as carelite.corpus.chunk's sentence splitter does)."""
    footer_lines = (
        "CARE JOURNAL\nEmpathy Disparities\nCARE JOURNAL | https://doi.org/10.9/y\nMarch 1, 2021"
    )
    pages = [
        f"Prior sentence ends here.\nLow SES was associated with lower empathy (mean\n{footer_lines}",
        "CARE difference = -0.87 [95% CI -1.72 to -0.02]).",
        f"Filler page three text.\n{footer_lines}",
        f"Filler page four text.\n{footer_lines}",
    ]
    pdf = _make_pdf(tmp_path / "midsentence_footer.pdf", pages)

    result = extract.extract_pdf(pdf)

    assert result.ok
    for line in footer_lines.splitlines():
        assert line not in result.text
    flattened = " ".join(result.text.split())
    assert "CARE JOURNAL" not in flattened
    assert "Empathy Disparities" not in flattened


def test_extract_pdf_strips_standalone_pagination_line(tmp_path):
    pages = [f"Body text on page {i} about patient communication.\n{i} / 4" for i in range(1, 5)]
    pdf = _make_pdf(tmp_path / "paginated.pdf", pages)

    result = extract.extract_pdf(pdf)

    assert result.ok
    assert not re.search(r"^\s*\d+\s*/\s*4\s*$", result.text, re.MULTILINE)


def test_extract_pdf_does_not_strip_pagination_look_alikes_in_running_text(tmp_path):
    """A ratio embedded in a sentence must survive -- only a *standalone*
    "N / M" line (the actual page-number footer) is treated as noise."""
    pages = [
        f"On page {i}, 18/22 participants completed the survey about care." for i in range(1, 4)
    ]
    pdf = _make_pdf(tmp_path / "ratio.pdf", pages)

    result = extract.extract_pdf(pdf)

    assert "18/22" in result.text


def test_extract_pdf_does_not_strip_pagination_on_short_documents(tmp_path):
    # Fewer than 3 pages -> no repetition/pagination heuristics apply at all,
    # matching the existing header/footer detector's own threshold.
    pages = ["Body text here.\n1 / 2", "More body text.\n2 / 2"]
    pdf = _make_pdf(tmp_path / "short_paginated.pdf", pages)

    result = extract.extract_pdf(pdf)

    assert "1 / 2" in result.text


# ---------------------------------------------------------------------------
# Word-level layout artefacts: PDF column-break splits and dropped-space
# glues. See the module-level note above `_rejoin_split_words` in
# extract.py for why the two directions are handled so differently.
# ---------------------------------------------------------------------------


def test_rejoin_split_words_joins_when_both_fragments_are_not_real_words():
    text = "The approach was highly collabora tive throughout the study."
    fixed = extract._rejoin_split_words(text)
    assert "collaborative" in fixed
    assert "collabora tive" not in fixed


def test_rejoin_split_words_joins_a_second_example():
    text = "Results were assessed sta tistically across all sites."
    fixed = extract._rejoin_split_words(text)
    assert "statistically" in fixed
    assert "sta tistically" not in fixed


def test_rejoin_split_words_leaves_genuine_two_word_phrases_alone():
    """Both fragments read as real words on their own ("set", "out"), so the
    ambiguity can't be resolved from text alone -- must NOT be merged into
    "setout", even though that concatenation happens to also be a real
    (obscure) dictionary word. This is the deliberately conservative side of
    the rule: an unresolved ambiguity is left as-is rather than guessed at."""
    text = "The study protocol was set out in advance."
    fixed = extract._rejoin_split_words(text)
    assert "set out" in fixed
    assert "setout" not in fixed


def test_rejoin_split_words_does_not_cross_a_newline():
    """A word split across a genuine page/line boundary (an explicit
    newline, not a same-line space) is a different phenomenon -- the
    page-join itself -- and is not this function's job."""
    text = "collabora\ntive work continued."
    fixed = extract._rejoin_split_words(text)
    assert "collaborative" not in fixed


def test_rejoin_split_words_requires_minimum_merged_length():
    """Guards against short coincidental dictionary collisions -- e.g. a
    split citation fragment ("ad" from "Mahmoudir-ad", followed by "et" from
    "et al.") that would otherwise merge into the unrelated real word
    "adet"."""
    text = "Mahmoudir-ad et al. (2015) reported similar findings."
    fixed = extract._rejoin_split_words(text)
    assert "ad et" in fixed
    assert "adet" not in fixed


def test_fix_glued_words_inserts_dropped_space():
    text = "This gap was noted inthe SDM group, and they hada greater role."
    fixed = extract._fix_glued_words(text)
    assert "in the SDM group" in fixed
    assert "had a greater" in fixed
    assert "inthe" not in fixed
    assert "hada" not in fixed


def test_fix_glued_words_preserves_sentence_initial_capitalisation():
    text = "Inthe control group, outcomes were similar."
    fixed = extract._fix_glued_words(text)
    assert fixed.startswith("In the control group")


def test_fix_glued_words_leaves_open_class_compounds_untouched():
    """Deliberate scope limit, not an oversight: an open-class glued
    compound like "healthrelated" or "decisionmaking" is not in the curated
    lookup table and is left alone. See the module-level note above
    `_rejoin_split_words` in extract.py for why a general detector for this
    direction was built, measured against the full corpus, and rejected --
    it "fixed" real words and names into nonsense far more often than it
    fixed genuine artefacts."""
    text = "Several healthrelated outcomes and decisionmaking patterns were noted."
    fixed = extract._fix_glued_words(text)
    assert "healthrelated" in fixed
    assert "decisionmaking" in fixed


def test_extract_pdf_applies_layout_artefact_fixes_end_to_end(tmp_path):
    body = "This gap was noted inthe control group, which was highly collabora tive."
    pdf = _make_pdf(tmp_path / "artefacts.pdf", [body])

    result = extract.extract_pdf(pdf)

    assert "in the control group" in result.text
    assert "collaborative" in result.text
