"""Unit tests for carelite.corpus.extract, using tiny synthetic PDFs built
with pymupdf itself so nothing here depends on a real paper being present."""

from __future__ import annotations

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
