"""Unit tests for carelite.corpus.fetch — no network, no filesystem side effects
outside pytest's tmp_path."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from tenacity import stop_after_attempt, wait_none

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
        self.content = content
        self.headers = headers or {}

    def json(self) -> dict:
        return self._json_data

    def iter_content(self, chunk_size: int):
        # single chunk is enough for these fixtures
        if self.content:
            yield self.content


# ---------------------------------------------------------------------------
# slug / dest_for / existing_dest
# ---------------------------------------------------------------------------


def test_slug_normalizes_doi_to_filesystem_safe_string():
    assert fetch.slug("10.1370/afm.348") == "10-1370-afm-348"
    assert fetch.slug("10.1186/S12909-023-04010-Z") == "10-1186-s12909-023-04010-z"


def test_dest_for_uses_year_and_slug():
    dest = fetch.dest_for(Path("/x"), "10.1/abc", "2020")
    assert dest.name == "2020_10-1-abc.pdf"


def test_dest_for_uses_nd_when_year_missing():
    dest = fetch.dest_for(Path("/x"), "10.1/abc", "")
    assert dest.name == "nd_10-1-abc.pdf"


def test_dest_for_honours_explicit_extension():
    dest = fetch.dest_for(Path("/x"), "10.1/abc", "2020", "xml")
    assert dest.name == "2020_10-1-abc.xml"


def test_existing_dest_finds_either_pdf_or_xml(tmp_path):
    assert fetch.existing_dest(tmp_path, "10.1/abc", "2020") is None
    xml_dest = fetch.dest_for(tmp_path, "10.1/abc", "2020", "xml")
    xml_dest.write_bytes(b"<article/>")
    assert fetch.existing_dest(tmp_path, "10.1/abc", "2020") == xml_dest


# ---------------------------------------------------------------------------
# download_source / download_pdf (%PDF and XML content guards)
# ---------------------------------------------------------------------------


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


def test_download_source_accepts_real_xml(tmp_path):
    dest = tmp_path / "out.xml"
    resp = FakeResponse(status_code=200, content=b"<?xml version='1.0'?><article/>")
    with patch("carelite.corpus.fetch.requests.get", return_value=resp):
        ok, _note = fetch.download_source("http://example.com/x.xml", dest, {}, "xml")
    assert ok is True
    assert dest.exists()


def test_download_source_rejects_html_dressed_up_as_xml(tmp_path):
    """The XML guard must catch the same failure mode the %PDF guard exists
    for: a landing/error page returned where real content was expected."""
    dest = tmp_path / "out.xml"
    resp = FakeResponse(
        status_code=200,
        content=b"<!DOCTYPE html><html><body>not found</body></html>",
        headers={"content-type": "text/html"},
    )
    with patch("carelite.corpus.fetch.requests.get", return_value=resp):
        ok, note = fetch.download_source("http://example.com/x.xml", dest, {}, "xml")
    assert ok is False
    assert "HTML" in note
    assert not dest.exists()


def test_download_source_rejects_content_that_is_neither_pdf_nor_xml(tmp_path):
    dest = tmp_path / "out.xml"
    resp = FakeResponse(status_code=200, content=b"just some plain text, not a document")
    with patch("carelite.corpus.fetch.requests.get", return_value=resp):
        ok, note = fetch.download_source("http://example.com/x", dest, {}, "xml")
    assert ok is False
    assert "not XML" in note


# ---------------------------------------------------------------------------
# The resolver chain: no NCBI /pmc/articles/{pmcid}/pdf/ scraping anywhere
# ---------------------------------------------------------------------------


def test_blocked_pmc_article_pdf_url_pattern_is_never_requested():
    """Regression guard for the 19/34 HTTP 403s: NCBI blocks programmatic
    fetches of pmc/articles/{pmcid}/pdf/. Drive the full resolver chain (every
    resolver returns a PMCID but no OA hit, so every resolver actually runs)
    and assert that URL pattern is never among the requests actually made —
    the PMC OA Web Service is the sanctioned replacement, used instead."""
    assert not hasattr(fetch, "pmc_pdf_url")  # renamed to pmc_idconv_pmcid (metadata-only)

    requested_urls: list[str] = []

    def fake_get(url, **kwargs):
        requested_urls.append(url)
        if "idconv" in url:
            return FakeResponse(status_code=200, json_data={"records": [{"pmcid": "PMC123"}]})
        return FakeResponse(status_code=200, json_data={})  # every other resolver misses

    with patch("carelite.corpus.fetch.requests.get", side_effect=fake_get):
        list(fetch.resolve_candidates("10.1/aaa", "me@example.com", {}))

    assert requested_urls  # sanity: the chain actually ran
    assert not any(
        "articles/PMC123/pdf" in u or re.search(r"articles/\{?pmcid\}?/pdf", u)
        for u in requested_urls
    )


def test_resolve_candidates_yields_unpaywall_first():
    def fake_get(url, **kwargs):
        if "unpaywall" in url:
            return FakeResponse(
                status_code=200, json_data={"best_oa_location": {"url_for_pdf": "http://oa/x.pdf"}}
            )
        raise AssertionError(f"should not call {url} before consuming the first candidate")

    with patch("carelite.corpus.fetch.requests.get", side_effect=fake_get):
        candidates = fetch.resolve_candidates("10.1/aaa", "me@example.com", {})
        first = next(candidates)
    assert first == fetch.Resolution("http://oa/x.pdf", "pdf")


def test_resolve_candidates_yields_europepmc_xml_next_when_open_access_there():
    def fake_get(url, **kwargs):
        if "unpaywall" in url:
            return FakeResponse(status_code=200, json_data={})
        if "europepmc" in url:
            return FakeResponse(
                status_code=200,
                json_data={"resultList": {"result": [{"pmcid": "PMC123", "isOpenAccess": "Y"}]}},
            )
        raise AssertionError(f"should not reach {url} once Europe PMC XML hits")

    with patch("carelite.corpus.fetch.requests.get", side_effect=fake_get):
        resolution = next(fetch.resolve_candidates("10.1/aaa", "me@example.com", {}))
    assert resolution.kind == "xml"
    assert resolution.url == fetch.europepmc_fulltext_xml_url("PMC123")


def test_resolve_candidates_yields_pmc_oa_web_service_next():
    oa_fcgi_xml = (
        b'<OA><records><record id="PMC123">'
        b'<link format="pdf" href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/x.pdf"/>'
        b"</record></records></OA>"
    )

    def fake_get(url, **kwargs):
        if "unpaywall" in url:
            return FakeResponse(status_code=200, json_data={})
        if "europepmc" in url:
            # found in Europe PMC's index, but not marked open access there
            return FakeResponse(
                status_code=200,
                json_data={"resultList": {"result": [{"pmcid": "PMC123", "isOpenAccess": "N"}]}},
            )
        if "oa.fcgi" in url:
            return FakeResponse(status_code=200, content=oa_fcgi_xml)
        raise AssertionError(f"should not reach {url} once the OA Web Service hits")

    with patch("carelite.corpus.fetch.requests.get", side_effect=fake_get):
        resolution = next(fetch.resolve_candidates("10.1/aaa", "me@example.com", {}))
    assert resolution == fetch.Resolution(
        "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/x.pdf", "pdf"
    )


def test_resolve_candidates_falls_through_to_idconv_when_europepmc_has_no_pmcid():
    oa_fcgi_xml = (
        b'<OA><records><record id="PMC999">'
        b'<link format="pdf" href="https://example.org/oa/y.pdf"/>'
        b"</record></records></OA>"
    )

    def fake_get(url, **kwargs):
        if "unpaywall" in url:
            return FakeResponse(status_code=200, json_data={})
        if "europepmc" in url:
            return FakeResponse(status_code=200, json_data={"resultList": {"result": []}})
        if "idconv" in url:
            return FakeResponse(status_code=200, json_data={"records": [{"pmcid": "PMC999"}]})
        if "oa.fcgi" in url:
            return FakeResponse(status_code=200, content=oa_fcgi_xml)
        raise AssertionError(f"should not reach {url}")

    with patch("carelite.corpus.fetch.requests.get", side_effect=fake_get):
        resolution = next(fetch.resolve_candidates("10.1/aaa", "me@example.com", {}))
    assert resolution == fetch.Resolution("https://example.org/oa/y.pdf", "pdf")


def test_resolve_candidates_yields_openalex_next():
    def fake_get(url, **kwargs):
        if "unpaywall" in url or "europepmc" in url or "idconv" in url:
            return FakeResponse(status_code=200, json_data={})
        if "openalex" in url:
            return FakeResponse(
                status_code=200, json_data={"best_oa_location": {"pdf_url": "http://oax/z.pdf"}}
            )
        raise AssertionError(f"should not reach {url} once OpenAlex hits")

    with patch("carelite.corpus.fetch.requests.get", side_effect=fake_get):
        resolution = next(fetch.resolve_candidates("10.1/aaa", "me@example.com", {}))
    assert resolution == fetch.Resolution("http://oax/z.pdf", "pdf")


def test_resolve_candidates_yields_semantic_scholar_last():
    def fake_get(url, **kwargs):
        if "semanticscholar" in url:
            return FakeResponse(
                status_code=200, json_data={"openAccessPdf": {"url": "http://s2/w.pdf"}}
            )
        return FakeResponse(status_code=200, json_data={})

    with patch("carelite.corpus.fetch.requests.get", side_effect=fake_get):
        resolution = next(fetch.resolve_candidates("10.1/aaa", "me@example.com", {}))
    assert resolution == fetch.Resolution("http://s2/w.pdf", "pdf")


def test_resolve_candidates_is_empty_when_every_resolver_misses():
    with patch("carelite.corpus.fetch.requests.get", return_value=FakeResponse(status_code=404)):
        candidates = list(fetch.resolve_candidates("10.1/nowhere", "me@example.com", {}))
    assert candidates == []


def test_resolve_candidates_yields_multiple_when_more_than_one_resolver_hits():
    """The whole point of the generator over a single resolve_source lookup:
    when Unpaywall's link turns out not to work, the caller needs a second
    candidate to fall through to."""

    def fake_get(url, **kwargs):
        if "unpaywall" in url:
            return FakeResponse(
                status_code=200, json_data={"best_oa_location": {"url_for_pdf": "http://oa/x.pdf"}}
            )
        if "europepmc" in url:
            return FakeResponse(
                status_code=200,
                json_data={"resultList": {"result": [{"pmcid": "PMC1", "isOpenAccess": "Y"}]}},
            )
        return FakeResponse(status_code=200, json_data={})

    with patch("carelite.corpus.fetch.requests.get", side_effect=fake_get):
        candidates = list(fetch.resolve_candidates("10.1/aaa", "me@example.com", {}))
    assert len(candidates) == 2
    assert candidates[0] == fetch.Resolution("http://oa/x.pdf", "pdf")
    assert candidates[1].kind == "xml"


def test_resolve_candidates_treats_a_resolver_network_error_as_a_miss():
    """One resolver raising shouldn't sink the whole chain for this DOI."""
    import requests as requests_module

    def fake_get(url, **kwargs):
        if "unpaywall" in url:
            raise requests_module.ConnectionError("dns failure")
        if "europepmc" in url:
            return FakeResponse(
                status_code=200,
                json_data={"resultList": {"result": [{"pmcid": "PMC1", "isOpenAccess": "Y"}]}},
            )
        return FakeResponse(status_code=200, json_data={})

    with patch("carelite.corpus.fetch.requests.get", side_effect=fake_get):
        resolution = next(fetch.resolve_candidates("10.1/aaa", "me@example.com", {}))
    assert resolution.kind == "xml"


# ---------------------------------------------------------------------------
# HTTP 429: retried with backoff, honouring Retry-After
# ---------------------------------------------------------------------------


def test_retry_after_seconds_parses_a_numeric_header():
    resp = FakeResponse(headers={"Retry-After": "3"})
    assert fetch._retry_after_seconds(resp) == 3.0


def test_retry_after_seconds_returns_none_for_missing_or_unparseable_header():
    assert fetch._retry_after_seconds(FakeResponse(headers={})) is None
    assert (
        fetch._retry_after_seconds(FakeResponse(headers={"Retry-After": "Wed, 01 Jan 2026"}))
        is None
    )


def test_get_retries_429_then_succeeds(monkeypatch):
    # Zero-wait retry policy so this test doesn't actually sleep.
    fast_get = fetch._get.retry_with(wait=wait_none(), stop=stop_after_attempt(4))
    monkeypatch.setattr(fetch, "_get", fast_get)

    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            return FakeResponse(status_code=429, headers={"Retry-After": "0"})
        return FakeResponse(status_code=200, json_data={"ok": True})

    with patch("carelite.corpus.fetch.requests.get", side_effect=fake_get):
        resp = fetch._get("http://example.com/thing", headers={}, timeout=5)

    assert calls["n"] == 3
    assert resp.status_code == 200


def test_get_gives_up_after_persistent_429(monkeypatch):
    fast_get = fetch._get.retry_with(wait=wait_none(), stop=stop_after_attempt(3))
    monkeypatch.setattr(fetch, "_get", fast_get)

    with (
        patch(
            "carelite.corpus.fetch.requests.get",
            return_value=FakeResponse(status_code=429, headers={}),
        ),
        pytest.raises(fetch._RateLimited),
    ):
        fetch._get("http://example.com/thing", headers={}, timeout=5)


def test_get_does_not_retry_non_429_errors():
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        return FakeResponse(status_code=403)

    with patch("carelite.corpus.fetch.requests.get", side_effect=fake_get):
        resp = fetch._get("http://example.com/thing", headers={}, timeout=5)
    assert resp.status_code == 403
    assert calls["n"] == 1  # no retry for a real access-control response


# ---------------------------------------------------------------------------
# fetch_all: dedup, idempotency, manual CSV, end-to-end download
# ---------------------------------------------------------------------------


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
    assert result.failed  # every resolver misses for the one real row -> reported, not crashed
    # unpaywall, europepmc, idconv, openalex, semantic scholar: 5 calls, never a 6th for the dup
    assert mock_get.call_count == 5


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


def test_fetch_all_skips_existing_xml_too(tmp_path):
    """Idempotency must recognise an already-fetched XML full text, not just a PDF."""
    manifest = [("a.pdf", "10.1/aaa", "2020", "")]
    dest = fetch.dest_for(tmp_path, "10.1/aaa", "2020", "xml")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"<article/>")

    with patch("carelite.corpus.fetch.requests.get") as mock_get:
        result = fetch.fetch_all(
            "me@example.com", out_dir=tmp_path, manifest=manifest, sleep_seconds=0, log=False
        )
    assert result.skipped_existing == 1
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


def test_fetch_all_falls_through_when_the_first_candidates_download_fails(tmp_path):
    """The real bug this guards against: Unpaywall resolves a DOI to a URL
    that returns HTTP 403 (observed live for 10.1177/08258597241245022 ->
    journals.sagepub.com, gated for programmatic clients). fetch_all must not
    give up there — it should fall through to the next sanctioned resolver
    (Europe PMC here) and succeed."""
    manifest = [("a.pdf", "10.1/aaa", "2020", "")]

    def fake_get(url, **kwargs):
        if "unpaywall" in url:
            return FakeResponse(
                status_code=200,
                json_data={"best_oa_location": {"url_for_pdf": "http://gated.example/x.pdf"}},
            )
        if "gated.example" in url:
            return FakeResponse(status_code=403)
        if "europepmc" in url and "fullTextXML" not in url:
            return FakeResponse(
                status_code=200,
                json_data={"resultList": {"result": [{"pmcid": "PMC1", "isOpenAccess": "Y"}]}},
            )
        if "fullTextXML" in url:
            return FakeResponse(status_code=200, content=b"<?xml version='1.0'?><article/>")
        raise AssertionError(f"should not reach {url}")

    with patch("carelite.corpus.fetch.requests.get", side_effect=fake_get):
        result = fetch.fetch_all(
            "me@example.com", out_dir=tmp_path, manifest=manifest, sleep_seconds=0, log=False
        )

    assert result.downloaded == 1
    assert not result.failed
    dest = fetch.dest_for(tmp_path, "10.1/aaa", "2020", "xml")
    assert dest.exists()


def test_fetch_all_reports_every_attempt_when_all_candidates_fail(tmp_path):
    manifest = [("a.pdf", "10.1/aaa", "2020", "")]

    def fake_get(url, **kwargs):
        if "unpaywall" in url:
            return FakeResponse(
                status_code=200, json_data={"best_oa_location": {"url_for_pdf": "http://oa/x.pdf"}}
            )
        if "oa/x.pdf" in url:
            return FakeResponse(status_code=403)
        return FakeResponse(status_code=200, json_data={})  # every other resolver misses

    with patch("carelite.corpus.fetch.requests.get", side_effect=fake_get):
        result = fetch.fetch_all(
            "me@example.com", out_dir=tmp_path, manifest=manifest, sleep_seconds=0, log=False
        )

    assert result.downloaded == 0
    assert len(result.failed) == 1
    note = result.failed[0][2]
    assert "403" in note  # the one real attempt's failure reason is preserved, not swallowed
    assert not fetch.dest_for(tmp_path, "10.1/aaa", "2020").exists()  # no leftover partial file


def test_fetch_all_downloads_europepmc_xml_when_that_is_the_only_hit(tmp_path):
    manifest = [("a.pdf", "10.1/aaa", "2020", "")]

    def fake_get(url, **kwargs):
        if "unpaywall" in url:
            return FakeResponse(status_code=200, json_data={})
        if "europepmc" in url and "fullTextXML" not in url:
            return FakeResponse(
                status_code=200,
                json_data={"resultList": {"result": [{"pmcid": "PMC1", "isOpenAccess": "Y"}]}},
            )
        if "fullTextXML" in url:
            return FakeResponse(status_code=200, content=b"<?xml version='1.0'?><article/>")
        raise AssertionError(f"should not reach {url}")

    with patch("carelite.corpus.fetch.requests.get", side_effect=fake_get):
        result = fetch.fetch_all(
            "me@example.com", out_dir=tmp_path, manifest=manifest, sleep_seconds=0, log=False
        )
    assert result.downloaded == 1
    dest = fetch.dest_for(tmp_path, "10.1/aaa", "2020", "xml")
    assert dest.exists()


# ---------------------------------------------------------------------------
# manifest_papers
# ---------------------------------------------------------------------------


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


def test_manifest_papers_finds_xml_full_text_too(tmp_path):
    manifest = [("a.pdf", "10.1/aaa", "2020", "")]
    dest = fetch.dest_for(tmp_path, "10.1/aaa", "2020", "xml")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"<article/>")

    papers = fetch.manifest_papers(tmp_path, manifest=manifest)
    assert len(papers) == 1
    assert papers[0].pdf_path == str(dest)


def test_main_requires_email_when_none_configured(monkeypatch):
    monkeypatch.setattr(fetch, "get_settings", lambda: type("S", (), {"unpaywall_email": ""})())
    with pytest.raises(SystemExit):
        fetch.main(["--out", "/tmp/whatever"])
