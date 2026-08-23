"""carelite.corpus.fetch — rebuild the CARELite paper corpus as real PDFs (or,
where only that is open, full-text XML).

Refactored from the original `data/fetch_corpus.py` (now a thin shim over
this module). Behaviour preserved: the embedded, self-contained DOI
manifest; the `%PDF`/XML content-sniff guard so a paywall's HTML landing
page never gets saved as a real document; `duplicate_of` dedup; idempotent
re-runs (existing output — PDF or XML — is skipped); rows with no DOI, plus
any runtime failures, land in `_manual_needed.csv`.

Resolution chain per DOI, first hit wins, in order:

1. **Unpaywall** — the primary OA index (requires `--email` per their ToS).
2. **Europe PMC** (`ebi.ac.uk/europepmc/webservices/rest`) — REST search by
   DOI, explicitly built for programmatic access. When the article is OA
   there, its `fullTextXML` endpoint is preferred *ahead of* a PDF: it is
   clean JATS XML with real section/paragraph/reference structure, so
   `carelite.corpus.extract` doesn't have to heuristically strip running
   headers, footers, or ligature damage the way it must for a PDF.
3. **PMC OA Web Service** (`ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi`) — NCBI's
   own sanctioned bulk-download API for the PMC open-access subset. This
   replaces the old fallback of building
   `pmc/articles/{pmcid}/pdf/` directly and scraping it: that URL pattern
   returns HTTP 403 for programmatic clients (verified against this
   manifest — 19/34 failures were exactly this) because it's gated for
   interactive browser use, not because the content is closed. The correct
   response to a 403 is to use the API the content owner actually offers
   for automated access, not to retry past the block or spoof a browser
   User-Agent — this module does neither. NCBI PMC's ID-converter
   (`pmc/utils/idconv`) is still used, but only as a metadata lookup to get
   a PMCID for the OA Web Service call, never to build the blocked URL.
4. **OpenAlex** (`api.openalex.org`) — `best_oa_location.pdf_url`.
5. **Semantic Scholar** (`api.semanticscholar.org`) — `openAccessPdf.url`.

A DOI that resolves nowhere in this chain is genuinely not open-access
through any of these sources and is reported in `_manual_needed.csv`, same
as before — that file is a legitimate output, not a failure of the script.

Requests are rate-limited with `time.sleep(1)` between manifest rows, and
every HTTP call additionally retries HTTP 429 with exponential backoff
(honouring `Retry-After` when the server sends one) via `_get`.

Default output directory is `settings.pdf_dir` (`data/pdfs/`, gitignored —
fetched documents are mixed-copyright and must never be committed).
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
import time
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - dependency is declared in pyproject
    sys.exit("Missing dependency. Run:  uv sync")

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from carelite.config import get_settings
from carelite.types import EvidenceTier, Paper

# ---------------------------------------------------------------------------
# Embedded manifest: [original_filename, doi, year, duplicate_of]
# duplicate_of is set when this file is a byte-identical copy of another
# entry in this same list (same DOI) — those are skipped, not re-downloaded.
# 5 papers had no recoverable DOI and are listed for manual lookup: fp3606284
# (.pdf/_1.pdf), Tanaffos17241.pdf, ppa2137.pdf, i15245012174317.pdf (DOI
# prefix present but article number was truncated in extraction).
# ---------------------------------------------------------------------------
ManifestRow = tuple[str, str, str, str]

MANIFEST: list[ManifestRow] = [
    ("0030415.pdf", "10.1370/afm.348", "2005", ""),
    ("0030415_1.pdf", "10.1370/afm.348", "2005", "0030415.pdf"),
    ("10_1177_08258597241245022.pdf", "10.1177/08258597241245022", "2014", ""),
    ("10_1177_10732748241236327.pdf", "10.1177/10732748241236327", "2024", ""),
    ("10_1177_2150132720922714.pdf", "10.1177/2150132720922714", "2020", ""),
    ("10_1177_2333392819882871.pdf", "10.1177/2333392819882871", "2017", ""),
    ("11606_2012_Article_2157.pdf", "10.1007/s11606-012-2157-7", "", ""),
    ("11606_2016_Article_3597.pdf", "10.1007/s11606-016-3597-2", "", ""),
    ("12885_2017_Article_3238.pdf", "10.1186/s12885-017-3238-0", "2017", ""),
    ("12888_2018_Article_1686.pdf", "10.1186/s12888-018-1686-y", "2018", ""),
    ("12888_2023_Article_4948.pdf", "10.1186/s12888-023-04948-w", "2023", ""),
    (
        "12888_2023_Article_4948_1.pdf",
        "10.1186/s12888-023-04948-w",
        "2023",
        "12888_2023_Article_4948.pdf",
    ),
    ("12909_2023_Article_4010.pdf", "10.1186/s12909-023-04010-z", "2023", ""),
    (
        "12909_2023_Article_4010_1.pdf",
        "10.1186/s12909-023-04010-z",
        "2023",
        "12909_2023_Article_4010.pdf",
    ),
    ("12909_2025_Article_6710.pdf", "10.1186/s12909-025-06710-0", "2025", ""),
    ("12909_2025_Article_7797.pdf", "10.1186/s12909-025-07797-1", "2025", ""),
    (
        "12909_2025_Article_7797_1.pdf",
        "10.1186/s12909-025-07797-1",
        "2025",
        "12909_2025_Article_7797.pdf",
    ),
    ("12913_2024_Article_11647.pdf", "10.1186/s12913-024-11647-z", "2024", ""),
    ("12913_2025_Article_13506.pdf", "10.1186/s12913-025-13506-x", "2025", ""),
    ("AJRCCM1816566.pdf", "10.1164/rccm.200906-0907OC", "", ""),
    ("bmjopen153.pdf", "10.1136/bmjopen-2024-091143", "2025", ""),
    ("bmjopen2018023666.pdf", "10.1136/bmjopen-2018-023666", "2019", ""),
    ("fcvm111457039.pdf", "10.3389/fcvm.2024.1457039", "2024", ""),
    ("fphar141283135.pdf", "10.3389/fphar.2023.1283135", "", ""),
    ("healthcare0800026.pdf", "10.3390/healthcare8010026", "2019", ""),
    ("i15245012174317.pdf", "", "", ""),  # DOI truncated in extraction — manual
    ("ijgc2023004693.pdf", "10.1136/ijgc-2023-004693", "2023", ""),
    ("main.pdf", "10.1016/j.pecinn.2025.100426", "2025", ""),
    ("main_1.pdf", "10.1016/j.abd.2025.501228", "2025", ""),
    ("main_2.pdf", "10.1016/j.pecinn.2025.100426", "2025", "main.pdf"),
    ("main_3.pdf", "10.1016/j.jpainsymman.2020.07.022", "2020", ""),
    ("main_4.pdf", "10.1016/j.pecinn.2025.100436", "2025", ""),
    ("main_5.pdf", "10.1016/j.pecinn.2025.100399", "2025", ""),
    ("nihms1057661.pdf", "10.7326/M19-1152", "2020", ""),
    ("nihms1581950.pdf", "10.1016/j.pec.2020.03.019", "2021", ""),
    (
        "nihms1581950_1.pdf",
        "10.1016/j.pec.2020.03.019",
        "2021",
        "nihms1581950.pdf",
    ),
    ("nihms1849896.pdf", "10.1016/j.jcomdis.2022.106274", "2022", ""),
    ("nihms1959154.pdf", "10.1016/j.jpainsymman.2022.11.029", "2022", ""),
    ("nihms250897.pdf", "10.1002/jhm.861", "2007", ""),
    ("nihms305491.pdf", "10.1016/j.pec.2011.04.023", "2011", ""),
    ("nihms648836.pdf", "10.1016/j.pec.2014.11.024", "2014", ""),
    ("nihms725148.pdf", "10.1016/j.pec.2015.09.001", "2015", ""),
    ("nihms856268.pdf", "10.1177/0272989X10364247", "", ""),
    ("pharmacy0600018.pdf", "10.3390/pharmacy6010018", "2018", ""),
    (
        "pharmacy0600018_1.pdf",
        "10.3390/pharmacy6010018",
        "2018",
        "pharmacy0600018.pdf",
    ),
    ("pmr_2025_0005.pdf", "10.1089/pmr.2025.0005", "2025", ""),
    ("pone_0230672.pdf", "10.1371/journal.pone.0230672", "2020", ""),
    ("pone_0231350.pdf", "10.1371/journal.pone.0231350", "2019", ""),
    ("pone_0247259.pdf", "10.1371/journal.pone.0247259", "2021", ""),
    ("pone_0304180.pdf", "10.1371/journal.pone.0304180", "2024", ""),
    ("prbm12457.pdf", "10.2147/PRBM.S208427", "2017", ""),
    ("Tanaffos17241.pdf", "", "", ""),  # no DOI recovered — manual
    ("ppa2137.pdf", "", "", ""),  # no DOI recovered — manual
    ("fp3606284.pdf", "", "", ""),  # no DOI recovered — manual
    ("fp3606284_1.pdf", "", "", "fp3606284.pdf"),
]

UA_TEMPLATE = "CARELite-corpus-rebuild/1.0 (mailto:{email})"


class _RateLimited(requests.RequestException):
    """Raised internally on HTTP 429 to trigger a tenacity retry.

    Subclasses `requests.RequestException` so it's caught by the same
    `except requests.RequestException` handling every caller already uses,
    with a message that reports the real cause instead of a generic error.
    """

    def __init__(self, retry_after: float | None) -> None:
        self.retry_after = retry_after
        suffix = f", honouring Retry-After: {retry_after}s" if retry_after else ""
        super().__init__(f"HTTP 429 (rate limited{suffix})")


def _retry_after_seconds(resp: requests.Response) -> float | None:
    value = resp.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # HTTP-date form; fall back to exponential backoff instead


def _wait_honouring_retry_after(retry_state):
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, _RateLimited) and exc.retry_after is not None:
        return exc.retry_after
    return wait_exponential(multiplier=1, min=2, max=30)(retry_state)


@retry(
    retry=retry_if_exception_type(_RateLimited),
    wait=_wait_honouring_retry_after,
    stop=stop_after_attempt(5),
    reraise=True,
)
def _get(url: str, **kwargs: Any) -> requests.Response:
    """`requests.get`, transparently retrying HTTP 429 with backoff.

    Every other status code (including 403/404 — a real access control or a
    dead link, not a transient condition) is returned as-is for the caller
    to interpret; only 429 is retried here.
    """
    r = requests.get(url, **kwargs)
    if r.status_code == 429:
        raise _RateLimited(_retry_after_seconds(r))
    return r


def slug(doi: str) -> str:
    """Stable filesystem/paper_id slug for a DOI, e.g. 10.1370/afm.348 -> 10-1370-afm-348."""
    return re.sub(r"[^A-Za-z0-9]+", "-", doi).strip("-").lower()


def unpaywall_pdf_url(doi: str, email: str, headers: dict[str, str]) -> str | None:
    r = _get(
        f"https://api.unpaywall.org/v2/{doi}",
        params={"email": email},
        headers=headers,
        timeout=30,
    )
    if r.status_code != 200:
        return None
    loc = (r.json() or {}).get("best_oa_location") or {}
    return loc.get("url_for_pdf") or loc.get("url")


def pmc_idconv_pmcid(doi: str, headers: dict[str, str]) -> str | None:
    """DOI -> PMCID via NCBI's ID-converter. A metadata lookup, not a document
    fetch — used only to feed the PMC OA Web Service, never to build the
    `pmc/articles/{pmcid}/pdf/` URL (that pattern 403s for programmatic
    clients; see the module docstring)."""
    r = _get(
        "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
        params={"ids": doi, "format": "json", "tool": "carelite", "email": "noreply@example.com"},
        headers=headers,
        timeout=30,
    )
    if r.status_code != 200:
        return None
    records = (r.json() or {}).get("records") or []
    pmcid = records[0].get("pmcid") if records else None
    return str(pmcid) if pmcid else None


def europepmc_lookup(doi: str, headers: dict[str, str]) -> dict[str, object] | None:
    """Europe PMC REST search by DOI — explicitly built for programmatic access."""
    r = _get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": f"DOI:{doi}", "format": "json", "resultType": "core"},
        headers=headers,
        timeout=30,
    )
    if r.status_code != 200:
        return None
    results = ((r.json() or {}).get("resultList") or {}).get("result") or []
    if not results:
        return None
    result = results[0]
    return {"pmcid": result.get("pmcid"), "is_oa": result.get("isOpenAccess") == "Y"}


def europepmc_fulltext_xml_url(pmcid: str) -> str:
    """Europe PMC's own full-text endpoint: clean JATS XML, preferred over a
    PDF when available — real section/paragraph structure means
    `carelite.corpus.extract` doesn't have to heuristically undo header/
    footer/ligature damage the way it must for a PDF."""
    return f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"


def pmc_oa_pdf_url(pmcid: str, headers: dict[str, str]) -> str | None:
    """The PMC OA Web Service: NCBI's sanctioned API for the open-access
    subset, offered *instead of* scraping the article-view /pdf/ URL."""
    r = _get(
        "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
        params={"id": pmcid},
        headers=headers,
        timeout=30,
    )
    if r.status_code != 200:
        return None
    try:
        root = ElementTree.fromstring(r.content)
    except ElementTree.ParseError:
        return None
    for link in root.iter("link"):
        if link.get("format") != "pdf":
            continue
        href = link.get("href")
        if not href:
            continue
        # NCBI's OA service returns ftp:// links; the same tree is also
        # served over https at an identical path, which `requests` can fetch.
        return href.replace("ftp://", "https://", 1) if href.startswith("ftp://") else href
    return None


def openalex_pdf_url(doi: str, headers: dict[str, str]) -> str | None:
    r = _get(f"https://api.openalex.org/works/doi:{doi}", headers=headers, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json() or {}
    best = data.get("best_oa_location") or {}
    return best.get("pdf_url") or (data.get("open_access") or {}).get("oa_url")


def semantic_scholar_pdf_url(doi: str, headers: dict[str, str]) -> str | None:
    r = _get(
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
        params={"fields": "openAccessPdf"},
        headers=headers,
        timeout=30,
    )
    if r.status_code != 200:
        return None
    pdf = (r.json() or {}).get("openAccessPdf") or {}
    return pdf.get("url")


@dataclass
class Resolution:
    """One resolved source for a DOI: a URL and what kind of document it is."""

    url: str
    kind: str  # "pdf" | "xml"


def _safe_lookup(fn: Any, *args: Any) -> Any:
    """Call a resolver lookup, treating a network error as a miss (None)
    rather than aborting the whole candidate chain for this DOI. `_get`
    already retries HTTP 429 internally; what's left here (timeouts,
    connection resets) is rare enough that one resolver failing outright
    shouldn't stop the others from being tried."""
    try:
        return fn(*args)
    except requests.RequestException:
        return None


def resolve_candidates(doi: str, email: str, headers: dict[str, str]) -> Iterator[Resolution]:
    """Yield every candidate source for this DOI, in priority order, lazily.

    Order: Unpaywall -> Europe PMC (XML preferred when OA there) -> PMC OA
    Web Service -> OpenAlex -> Semantic Scholar.

    A resolver returning *some* URL does not guarantee that URL actually
    serves the document — Unpaywall's `best_oa_location` is sometimes a
    publisher's gated landing page, not the PDF itself (observed directly:
    Unpaywall resolves 10.1177/08258597241245022 to
    `journals.sagepub.com/doi/pdf/...`, which 403s for a programmatic
    client). This is a generator, not a single lookup, so a caller can
    attempt each candidate's download and fall through to the next only on
    an actual failure, trying every sanctioned source before concluding a
    DOI is genuinely not open-access. Because it's a generator, resolvers
    further down the list are only ever called if an earlier candidate's
    *download* failed — the common case (first candidate just works) costs
    exactly one resolver call, same as before.
    """
    url = _safe_lookup(unpaywall_pdf_url, doi, email, headers)
    if url:
        yield Resolution(url, "pdf")

    epmc = _safe_lookup(europepmc_lookup, doi, headers) or {}
    pmcid = epmc.get("pmcid")
    if epmc.get("is_oa") and pmcid:
        yield Resolution(europepmc_fulltext_xml_url(str(pmcid)), "xml")

    if not pmcid:
        pmcid = _safe_lookup(pmc_idconv_pmcid, doi, headers)
    if pmcid:
        pdf_url = _safe_lookup(pmc_oa_pdf_url, str(pmcid), headers)
        if pdf_url:
            yield Resolution(pdf_url, "pdf")

    url = _safe_lookup(openalex_pdf_url, doi, headers)
    if url:
        yield Resolution(url, "pdf")

    url = _safe_lookup(semantic_scholar_pdf_url, doi, headers)
    if url:
        yield Resolution(url, "pdf")


_XML_LOOKS_LIKE_HTML_RE = re.compile(rb"^\s*<(!doctype\s+html|html)\b", re.IGNORECASE)


def download_source(
    url: str, dest: pathlib.Path, headers: dict[str, str], kind: str
) -> tuple[bool, str]:
    """Stream to disk; abort and report if the response doesn't match `kind`.

    `kind="pdf"` requires the `%PDF` magic bytes (unchanged from the
    original guard). `kind="xml"` requires something that looks like XML and
    explicitly rejects an HTML landing/error page dressed up as a hit — the
    same failure mode the PDF guard exists to catch.
    """
    r = _get(url, headers=headers, timeout=90, stream=True, allow_redirects=True)
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    it = r.iter_content(65536)
    try:
        first = next(it)
    except StopIteration:
        return False, "empty response"

    if kind == "pdf":
        if not first.startswith(b"%PDF"):
            ctype = r.headers.get("content-type", "?")
            return False, f"not a PDF (got {ctype})"
    else:
        head = first.lstrip()
        if _XML_LOOKS_LIKE_HTML_RE.match(head):
            ctype = r.headers.get("content-type", "?")
            return False, f"not XML, got an HTML page (content-type {ctype})"
        if not head.startswith((b"<?xml", b"<")):
            ctype = r.headers.get("content-type", "?")
            return False, f"not XML (got {ctype})"

    with open(dest, "wb") as fh:
        fh.write(first)
        for piece in it:
            fh.write(piece)
    return True, f"{dest.stat().st_size // 1024} KB"


def download_pdf(url: str, dest: pathlib.Path, headers: dict[str, str]) -> tuple[bool, str]:
    """Backwards-compatible PDF-only entry point over `download_source`."""
    return download_source(url, dest, headers, "pdf")


@dataclass
class FetchResult:
    """Outcome of a full manifest pass, returned by `fetch_all` for callers/tests."""

    out_dir: pathlib.Path
    total_unique: int
    downloaded: int
    skipped_existing: int
    failed: list[tuple[str, str, str]] = field(default_factory=list)  # (orig_file, doi, note)
    manual: list[ManifestRow] = field(default_factory=list)  # raw rows with no DOI recovered
    manual_csv_path: pathlib.Path | None = None

    @property
    def ok(self) -> int:
        return self.downloaded + self.skipped_existing


def dest_for(out_dir: pathlib.Path, doi: str, year: str, ext: str = "pdf") -> pathlib.Path:
    return out_dir / f"{year or 'nd'}_{slug(doi)}.{ext}"


def existing_dest(out_dir: pathlib.Path, doi: str, year: str) -> pathlib.Path | None:
    """Whichever of the PDF/XML destinations for this DOI already exists, if any."""
    for ext in ("pdf", "xml"):
        candidate = dest_for(out_dir, doi, year, ext)
        if candidate.exists():
            return candidate
    return None


def fetch_all(
    email: str,
    out_dir: pathlib.Path | str | None = None,
    *,
    manifest: Iterable[ManifestRow] = MANIFEST,
    sleep_seconds: float = 1.0,
    log: bool = True,
) -> FetchResult:
    """Resolve and download every unique, non-manual DOI in `manifest`.

    Idempotent: files already present in `out_dir` are skipped without a
    network call. `sleep_seconds` is injectable so tests can run this loop
    without actually pausing.
    """
    settings = get_settings()
    resolved_out = pathlib.Path(out_dir) if out_dir is not None else settings.pdf_dir
    resolved_out.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": UA_TEMPLATE.format(email=email)}

    rows = list(manifest)
    todo = [row for row in rows if row[1] and not row[3]]  # has DOI, not a duplicate
    manual = [row for row in rows if not row[1]]

    if log:
        print("CARELite corpus rebuild")
        print(
            f"  {len(rows)} original files -> {len(todo)} unique papers to fetch, "
            f"{len(manual)} need manual lookup\n"
        )
        print(f"Downloading into: {resolved_out.resolve()}\n")

    downloaded, skipped = 0, 0
    failed: list[tuple[str, str, str]] = []

    for i, (orig_file, doi, year, _dup) in enumerate(todo, 1):
        already = existing_dest(resolved_out, doi, year)
        if already is not None:
            if log:
                print(f"[{i:>2}/{len(todo)}] skip (exists)   {already.name}")
            skipped += 1
            continue

        # Try each candidate source in turn: a resolver returning a URL does
        # not guarantee that URL actually serves the document (Unpaywall's OA
        # location is sometimes a gated publisher landing page), so a failed
        # download falls through to the next sanctioned resolver instead of
        # giving up after the first one.
        good = False
        note = "no open-access source found"
        dest: pathlib.Path | None = None
        attempts: list[str] = []
        for resolution in resolve_candidates(doi, email, headers):
            candidate_dest = dest_for(resolved_out, doi, year, resolution.kind)
            try:
                good, note = download_source(
                    resolution.url, candidate_dest, headers, resolution.kind
                )
            except requests.RequestException as e:
                good, note = False, str(e)

            if good:
                dest = candidate_dest
                break
            candidate_dest.unlink(missing_ok=True)
            attempts.append(f"{resolution.kind}:{note}")

        if good and dest is not None:
            downloaded += 1
            if log:
                print(f"[{i:>2}/{len(todo)}] ok  {note:>9}   {dest.name}")
        else:
            failure_note = "; ".join(attempts) if attempts else note
            failed.append((orig_file, doi, failure_note))
            if log:
                print(f"[{i:>2}/{len(todo)}] FAIL            {doi} - {failure_note}")

        time.sleep(sleep_seconds)  # be polite to the free APIs

    manual_rows = [(f, d, "no DOI recovered from original file") for f, d, _y, _dup in manual]
    manual_rows += failed

    manual_csv_path = None
    if manual_rows:
        manual_csv_path = write_manual_needed_csv(resolved_out, manual_rows)
        if log:
            print(f"\n{len(manual_rows)} papers need manual retrieval - see {manual_csv_path}")
            print(
                "For each: search the title on Google Scholar, or if you have the DOI, "
                "paste it into https://doi.org/<DOI>"
            )

    if log:
        print(f"\nDone: {downloaded + skipped}/{len(todo)} papers available ({downloaded} new).")
        print(f"PDFs are in: {resolved_out.resolve()}")

    return FetchResult(
        out_dir=resolved_out,
        total_unique=len(todo),
        downloaded=downloaded,
        skipped_existing=skipped,
        failed=failed,
        manual=manual,
        manual_csv_path=manual_csv_path,
    )


def write_manual_needed_csv(
    out_dir: pathlib.Path, rows: list[tuple[str, str, str]]
) -> pathlib.Path:
    manual_csv = out_dir / "_manual_needed.csv"
    with open(manual_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["original_file", "doi_or_blank", "note"])
        w.writerows(rows)
    return manual_csv


def manifest_papers(
    out_dir: pathlib.Path | str | None = None,
    *,
    manifest: Iterable[ManifestRow] = MANIFEST,
    evidence_tier: EvidenceTier = EvidenceTier.EMERGING,
) -> list[Paper]:
    """Build minimal `Paper` stubs, one per unique DOI, for rows whose PDF exists on disk.

    PROVISIONAL: `apa_citation` and `evidence_tier` are not derivable from the
    manifest alone — a real citation and a defensible evidence-tier judgment
    (study design vs. finding strength) are a KB/human review call, not
    something the fetch pipeline should assert. This produces a placeholder
    good enough to satisfy the `paper` table's NOT NULL constraints and to
    unblock downstream lanes; `evidence_tier` defaults to the weakest tier
    (`emerging`) rather than guessing high, and `apa_citation` is a DOI
    placeholder pending a real citation pass.
    """
    settings = get_settings()
    resolved_out = pathlib.Path(out_dir) if out_dir is not None else settings.pdf_dir

    papers: list[Paper] = []
    for _orig_file, doi, year, dup in manifest:
        if not doi or dup:
            continue
        dest = existing_dest(resolved_out, doi, year)
        if dest is None:
            continue
        papers.append(
            Paper(
                paper_id=slug(doi),
                doi=doi,
                apa_citation=f"[citation pending] DOI: {doi}",
                year=int(year) if year else None,
                evidence_tier=evidence_tier,
                pdf_path=str(dest),  # may be a .pdf or a .xml full-text file
            )
        )
    return papers


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Rebuild the CARELite paper corpus as real PDFs.")
    ap.add_argument(
        "--email",
        default=None,
        help=(
            "Contact email required by the Unpaywall API. Defaults to "
            "settings.unpaywall_email (CARELITE_UNPAYWALL_EMAIL / .env) if not given."
        ),
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output folder (default: settings.pdf_dir, i.e. data/pdfs/)",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    email = args.email or get_settings().unpaywall_email
    if not email:
        build_arg_parser().error(
            "--email is required (or set CARELITE_UNPAYWALL_EMAIL / unpaywall_email in .env)"
        )

    # Failures are reported via _manual_needed.csv, not treated as a fatal exit code.
    fetch_all(email, out_dir=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
