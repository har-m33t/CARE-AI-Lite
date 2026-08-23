"""Unit tests for carelite.corpus.pipeline.reload_corpus.

Exercises the orchestration logic (which papers get extracted, how failures
are handled, that stale chunks are replaced not just added to) with the
Postgres-touching functions monkeypatched to an in-memory fake — so this
stays in `make check`'s scope rather than needing `@pytest.mark.db`. The
real SQL behind `replace_paper_chunks`/`upsert_papers` is covered separately
in `test_load.py`'s db-marked tests.
"""

from __future__ import annotations

from carelite.corpus import pipeline
from carelite.types import EvidenceTier, Paper


class _FakeStore:
    def __init__(self) -> None:
        self.papers: dict[str, Paper] = {}
        self.chunks_by_paper: dict[str, list[str]] = {}
        self.upsert_papers_calls: list[list[Paper]] = []
        self.replace_calls: list[tuple[str, list[str]]] = []

    def upsert_papers(self, papers):
        papers = list(papers)
        self.upsert_papers_calls.append(papers)
        for p in papers:
            self.papers[p.paper_id] = p
        return len(papers)

    def replace_paper_chunks(self, paper_id, chunks):
        chunk_texts = [c.text for c in chunks]
        self.replace_calls.append((paper_id, chunk_texts))
        self.chunks_by_paper[paper_id] = chunk_texts
        return len(chunks)


def _paper(paper_id: str, pdf_path: str) -> Paper:
    return Paper(
        paper_id=paper_id,
        doi=None,
        apa_citation="[citation pending]",
        evidence_tier=EvidenceTier.EMERGING,
        pdf_path=pdf_path,
    )


def test_reload_corpus_extracts_chunks_and_loads_each_paper(monkeypatch, tmp_path):
    store = _FakeStore()
    monkeypatch.setattr(pipeline, "upsert_papers", store.upsert_papers)
    monkeypatch.setattr(pipeline, "replace_paper_chunks", store.replace_paper_chunks)

    pdf_dir = tmp_path
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Real content about patient communication here.")
    pdf_path = pdf_dir / "a.pdf"
    doc.save(pdf_path)
    doc.close()

    paper = _paper("paper-a", str(pdf_path))
    monkeypatch.setattr(pipeline, "manifest_papers", lambda source_dir=None: [paper])

    report = pipeline.reload_corpus(pdf_dir)

    assert report.papers == 1
    assert report.chunks >= 1
    assert not report.extraction_failures
    assert store.upsert_papers_calls == [[paper]]
    assert "paper-a" in store.chunks_by_paper
    assert store.chunks_by_paper["paper-a"]  # at least one real chunk stored


def test_reload_corpus_clears_chunks_and_reports_failure_for_bad_extraction(monkeypatch, tmp_path):
    store = _FakeStore()
    monkeypatch.setattr(pipeline, "upsert_papers", store.upsert_papers)
    monkeypatch.setattr(pipeline, "replace_paper_chunks", store.replace_paper_chunks)

    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"not a real pdf")
    paper = _paper("paper-bad", str(bad_pdf))
    monkeypatch.setattr(pipeline, "manifest_papers", lambda source_dir=None: [paper])

    report = pipeline.reload_corpus(tmp_path)

    assert report.papers == 1
    assert report.chunks == 0
    assert len(report.extraction_failures) == 1
    assert "paper-bad" in report.extraction_failures[0]
    # chunks explicitly cleared (empty replace), not left untouched
    assert store.replace_calls == [("paper-bad", [])]


def test_reload_corpus_handles_multiple_papers_independently(monkeypatch, tmp_path):
    store = _FakeStore()
    monkeypatch.setattr(pipeline, "upsert_papers", store.upsert_papers)
    monkeypatch.setattr(pipeline, "replace_paper_chunks", store.replace_paper_chunks)

    import pymupdf

    good_pdf = tmp_path / "good.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Good content sentence about communication research.")
    doc.save(good_pdf)
    doc.close()

    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"garbage")

    papers = [_paper("paper-good", str(good_pdf)), _paper("paper-bad", str(bad_pdf))]
    monkeypatch.setattr(pipeline, "manifest_papers", lambda source_dir=None: papers)

    report = pipeline.reload_corpus(tmp_path)

    assert report.papers == 2
    assert report.chunks >= 1
    assert len(report.extraction_failures) == 1
    assert store.chunks_by_paper["paper-good"]
    assert store.chunks_by_paper["paper-bad"] == []


def test_main_reports_extraction_failures_via_nonzero_exit(monkeypatch, tmp_path):
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"garbage")
    paper = _paper("paper-bad", str(bad_pdf))

    monkeypatch.setattr(pipeline, "manifest_papers", lambda source_dir=None: [paper])
    monkeypatch.setattr(pipeline, "upsert_papers", lambda papers: len(list(papers)))
    monkeypatch.setattr(pipeline, "replace_paper_chunks", lambda paper_id, chunks: len(chunks))

    exit_code = pipeline.main([str(tmp_path)])
    assert exit_code == 1


def test_main_returns_zero_on_a_clean_reload(monkeypatch, tmp_path):
    import pymupdf

    good_pdf = tmp_path / "good.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Content sentence about clinician communication.")
    doc.save(good_pdf)
    doc.close()
    paper = _paper("paper-good", str(good_pdf))

    monkeypatch.setattr(pipeline, "manifest_papers", lambda source_dir=None: [paper])
    monkeypatch.setattr(pipeline, "upsert_papers", lambda papers: len(list(papers)))
    monkeypatch.setattr(pipeline, "replace_paper_chunks", lambda paper_id, chunks: len(chunks))

    exit_code = pipeline.main([str(tmp_path)])
    assert exit_code == 0
