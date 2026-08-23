"""Unit tests for carelite.corpus.fetch — no network, no filesystem side effects
outside pytest's tmp_path."""

from __future__ import annotations

import csv
from unittest.mock import patch

import pytest

from carelite.corpus import fetch
from carelite.types import EvidenceTier


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        json_data: dict | None = None,
        content: bytes = b"",
        headers: dict | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self._content = content
        self.headers = headers or {}

    def json(self) -> dict:
        return self._json_data

    def iter_content(self, chunk_size: int):
        # single chunk is enough for these fixtures
        if self._content:
            yield self._content


def test_slug_normalizes_doi_to_filesystem_safe_string():
    assert fetch.slug("10.1370/afm.348") == "10-1370-afm-348"
    assert fetch.slug("10.1186/S12909-023-04010-Z") == "10-1186-s12909-023-04010-z"


def test_dest_for_uses_year_and_slug():
    from pathlib import Path

    dest = fetch.dest_for(Path("/x"), "10.1/abc", "2020")
    assert dest.name == "2020_10-1-abc.pdf"


def test_dest_for_uses_nd_when_year_missing():
    from pathlib import Path

    dest = fetch.dest_for(Path("/x"), "10.1/abc", "")
    assert dest.name == "nd_10-1-abc.pdf"


def test_download_pdf_rejects_non_pdf_content(tmp_path):
    dest = tmp_path / "out.pdf"
    resp = FakeResponse(
        status_code=200, content=b"<html>paywall</html>", headers={"content-type": "text/html"}
    )
    with patch("carelite.corpus.fetch.requests.get", return_value=resp):
        ok, note = fetch.download_pdf("http://example.com/x.pdf", dest, {})
    assert ok is False
    assert "not a PDF" in note
    assert not dest.exists()


def test_download_pdf_accepts_real_pdf_magic_bytes(tmp_path):
    dest = tmp_path / "out.pdf"
    resp = FakeResponse(status_code=200, content=b"%PDF-1.4 fake pdf body")
    with patch("carelite.corpus.fetch.requests.get", return_value=resp):
        ok, _note = fetch.download_pdf("http://example.com/x.pdf", dest, {})
    assert ok is True
    assert dest.exists()
    assert dest.read_bytes().startswith(b"%PDF")


def test_download_pdf_reports_http_error(tmp_path):
    dest = tmp_path / "out.pdf"
    resp = FakeResponse(status_code=404)
    with patch("carelite.corpus.fetch.requests.get", return_value=resp):
        ok, note = fetch.download_pdf("http://example.com/x.pdf", dest, {})
    assert ok is False
    assert "404" in note
    assert not dest.exists()


def test_fetch_all_skips_duplicates_without_any_network_call(tmp_path):
    manifest = [
        ("a.pdf", "10.1/aaa", "2020", ""),
        ("a_dup.pdf", "10.1/aaa", "2020", "a.pdf"),  # duplicate_of set -> never fetched
    ]
    with patch("carelite.corpus.fetch.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(status_code=200, json_data={})
        result = fetch.fetch_all(
            "me@example.com", out_dir=tmp_path, manifest=manifest, sleep_seconds=0, log=False
        )
    assert result.total_unique == 1  # the duplicate row never enters `todo`
    # unpaywall returns no OA location -> pmc_pdf_url also called once (2 requests) then failure recorded
    assert result.failed  # no OA link found for the one real row
    assert mock_get.call_count == 2  # unpaywall + pmc idconv, never a third call for the dup


def test_fetch_all_is_idempotent_and_skips_existing_files(tmp_path):
    manifest = [("a.pdf", "10.1/aaa", "2020", "")]
    dest = fetch.dest_for(tmp_path, "10.1/aaa", "2020")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"%PDF-1.4 already here")

    with patch("carelite.corpus.fetch.requests.get") as mock_get:
        result = fetch.fetch_all(
            "me@example.com", out_dir=tmp_path, manifest=manifest, sleep_seconds=0, log=False
        )
    assert result.skipped_existing == 1
    assert result.downloaded == 0
    mock_get.assert_not_called()


def test_fetch_all_writes_manual_needed_csv_for_no_doi_and_failures(tmp_path):
    manifest = [
        ("no_doi.pdf", "", "", ""),
        ("fails.pdf", "10.1/fails", "2021", ""),
    ]
    with patch("carelite.corpus.fetch.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(status_code=404)
        result = fetch.fetch_all(
            "me@example.com", out_dir=tmp_path, manifest=manifest, sleep_seconds=0, log=False
        )

    assert result.manual_csv_path is not None
    assert result.manual_csv_path.exists()
    with open(result.manual_csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    files = {r["original_file"] for r in rows}
    assert "no_doi.pdf" in files
    assert "fails.pdf" in files


def test_fetch_all_downloads_a_real_pdf(tmp_path):
    manifest = [("a.pdf", "10.1/aaa", "2020", "")]

    def fake_get(url, **kwargs):
        if "unpaywall" in url:
            return FakeResponse(
                status_code=200, json_data={"best_oa_location": {"url_for_pdf": "http://oa/x.pdf"}}
            )
        return FakeResponse(status_code=200, content=b"%PDF-1.4 real content here")

    with patch("carelite.corpus.fetch.requests.get", side_effect=fake_get):
        result = fetch.fetch_all(
            "me@example.com", out_dir=tmp_path, manifest=manifest, sleep_seconds=0, log=False
        )
    assert result.downloaded == 1
    assert not result.failed
    dest = fetch.dest_for(tmp_path, "10.1/aaa", "2020")
    assert dest.exists()


def test_manifest_papers_builds_provisional_paper_stubs(tmp_path):
    manifest = [
        ("a.pdf", "10.1/aaa", "2020", ""),
        ("a_dup.pdf", "10.1/aaa", "2020", "a.pdf"),
        ("b.pdf", "10.1/bbb", "2021", ""),  # never downloaded -> excluded
    ]
    dest = fetch.dest_for(tmp_path, "10.1/aaa", "2020")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"%PDF-1.4")

    papers = fetch.manifest_papers(tmp_path, manifest=manifest)
    assert len(papers) == 1  # only the row with a PDF on disk, dup and missing excluded
    paper = papers[0]
    assert paper.doi == "10.1/aaa"
    assert paper.paper_id == fetch.slug("10.1/aaa")
    assert paper.evidence_tier == EvidenceTier.EMERGING
    assert paper.pdf_path == str(dest)


def test_main_requires_email_when_none_configured(monkeypatch):
    monkeypatch.setattr(fetch, "get_settings", lambda: type("S", (), {"unpaywall_email": ""})())
    with pytest.raises(SystemExit):
        fetch.main(["--out", "/tmp/whatever"])
