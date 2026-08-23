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
    assert failures[0].pdf_path.endswith("bad.pdf")


def test_extract_corpus_default_dir_uses_settings(monkeypatch, tmp_path):
    _make_pdf(tmp_path / "only.pdf", ["Content here."])
    monkeypatch.setattr(extract, "get_settings", lambda: type("S", (), {"pdf_dir": tmp_path})())

    results, failures = extract.extract_corpus()

    assert len(results) == 1
    assert not failures
