"""carelite.corpus.fetch — rebuild the CARELite paper corpus as real PDFs.

Refactored from the original `data/fetch_corpus.py` (now a thin shim over
this module). Behaviour is preserved exactly:

- The DOI manifest is embedded below, self-contained.
- Resolution order per DOI: Unpaywall -> NCBI PMC ID-converter -> PMC PDF URL.
- Every download is checked for a real "%PDF" magic-byte header before being
  kept, so a paywall's HTML page never gets saved with a .pdf extension.
- `duplicate_of` marks byte-identical papers so they are fetched once.
- Re-running is safe: existing output files are skipped.
- Requests are rate-limited with `time.sleep(1)` between manifest rows.
- Rows with no DOI, plus any runtime failures, land in `_manual_needed.csv`.

Default output directory is `settings.pdf_dir` (`data/pdfs/`, gitignored —
PDFs are mixed-copyright and must never be committed).
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

try:
    import requests
except ImportError:  # pragma: no cover - dependency is declared in pyproject
    sys.exit("Missing dependency. Run:  uv sync")

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


def slug(doi: str) -> str:
    """Stable filesystem/paper_id slug for a DOI, e.g. 10.1370/afm.348 -> 10-1370-afm-348."""
    return re.sub(r"[^A-Za-z0-9]+", "-", doi).strip("-").lower()


def unpaywall_pdf_url(doi: str, email: str, headers: dict[str, str]) -> str | None:
    r = requests.get(
        f"https://api.unpaywall.org/v2/{doi}",
        params={"email": email},
        headers=headers,
        timeout=30,
    )
    if r.status_code != 200:
        return None
    loc = (r.json() or {}).get("best_oa_location") or {}
    return loc.get("url_for_pdf") or loc.get("url")


def pmc_pdf_url(doi: str, headers: dict[str, str]) -> str | None:
    """Fallback: resolve DOI -> PMCID via NCBI, then build the PMC PDF URL."""
    r = requests.get(
        "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
        params={"ids": doi, "format": "json", "tool": "carelite", "email": "noreply@example.com"},
        headers=headers,
        timeout=30,
    )
    if r.status_code != 200:
        return None
    records = (r.json() or {}).get("records") or []
    pmcid = records[0].get("pmcid") if records else None
    return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/" if pmcid else None


def download_pdf(url: str, dest: pathlib.Path, headers: dict[str, str]) -> tuple[bool, str]:
    """Stream to disk; abort and report if the response isn't actually a PDF."""
    r = requests.get(url, headers=headers, timeout=90, stream=True, allow_redirects=True)
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    it = r.iter_content(65536)
    try:
        first = next(it)
    except StopIteration:
        return False, "empty response"
    if not first.startswith(b"%PDF"):
        ctype = r.headers.get("content-type", "?")
        return False, f"not a PDF (got {ctype})"
    with open(dest, "wb") as fh:
        fh.write(first)
        for piece in it:
            fh.write(piece)
    return True, f"{dest.stat().st_size // 1024} KB"


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


def dest_for(out_dir: pathlib.Path, doi: str, year: str) -> pathlib.Path:
    return out_dir / f"{year or 'nd'}_{slug(doi)}.pdf"


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
        dest = dest_for(resolved_out, doi, year)
        if dest.exists():
            if log:
                print(f"[{i:>2}/{len(todo)}] skip (exists)   {dest.name}")
            skipped += 1
            continue

        url = None
        try:
            url = unpaywall_pdf_url(doi, email, headers) or pmc_pdf_url(doi, headers)
        except requests.RequestException as e:
            failed.append((orig_file, doi, f"lookup failed: {e}"))
            if log:
                print(f"[{i:>2}/{len(todo)}] LOOKUP ERROR    {doi}")
            continue

        if not url:
            failed.append((orig_file, doi, "no open-access PDF found"))
            if log:
                print(f"[{i:>2}/{len(todo)}] no OA link      {doi}")
            time.sleep(sleep_seconds)
            continue

        try:
            good, note = download_pdf(url, dest, headers)
        except requests.RequestException as e:
            good, note = False, str(e)

        if good:
            downloaded += 1
            if log:
                print(f"[{i:>2}/{len(todo)}] ok  {note:>9}   {dest.name}")
        else:
            dest.unlink(missing_ok=True)
            failed.append((orig_file, doi, note))
            if log:
                print(f"[{i:>2}/{len(todo)}] FAIL            {doi} - {note}")

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
        dest = dest_for(resolved_out, doi, year)
        if not dest.exists():
            continue
        papers.append(
            Paper(
                paper_id=slug(doi),
                doi=doi,
                apa_citation=f"[citation pending] DOI: {doi}",
                year=int(year) if year else None,
                evidence_tier=evidence_tier,
                pdf_path=str(dest),
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
