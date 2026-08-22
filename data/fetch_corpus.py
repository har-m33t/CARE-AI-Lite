#!/usr/bin/env python3
"""
fetch_corpus.py — rebuild the CARELite paper corpus as real PDFs.

Self-contained: the manifest (original filename -> DOI) is embedded below,
so this is the only file you need. Drop it anywhere in your IDE/repo and run.

WHY THIS EXISTS
    The files Claude's project knowledge held were not real PDFs — they were
    zip archives of page images + extracted text (Claude's internal reading
    format). This script re-fetches the real, original PDFs from open-access
    sources using the DOIs recovered from that extracted text.

SETUP (one time)
    pip install requests

RUN
    python fetch_corpus.py --email you@example.com

    --email is required by the Unpaywall API (it's just used to identify
    traffic to them, per their terms — no account needed, nothing is sent
    anywhere else).

OUTPUT
    ./carelite_pdfs/<year>_<doi-slug>.pdf   — one file per unique paper
    ./carelite_pdfs/_manual_needed.csv      — anything that couldn't be
                                               auto-resolved, with the DOI
                                               (or note) to look up by hand

NOTES
    - Byte-identical duplicate papers (7 pairs in the original set) are
      fetched once, not twice — see `duplicate_of` in the embedded manifest.
    - Every download is checked for a real "%PDF" header before being kept,
      so a paywall's HTML page never gets saved with a .pdf extension.
    - Safe to re-run: already-downloaded files are skipped.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
import time

try:
    import requests
except ImportError:
    sys.exit("Missing dependency. Run:  pip install requests")


# ---------------------------------------------------------------------------
# Embedded manifest: [original_filename, doi, year, duplicate_of]
# duplicate_of is set when this file is a byte-identical copy of another
# entry in this same list (same DOI) — those are skipped, not re-downloaded.
# 5 papers had no recoverable DOI and are listed at the bottom for manual
# lookup: fp3606284(.pdf/_1.pdf), Tanaffos17241.pdf, ppa2137.pdf,
# i15245012174317.pdf (DOI prefix present but article number was truncated
# in extraction).
# ---------------------------------------------------------------------------
MANIFEST = [
    ["0030415.pdf", "10.1370/afm.348", "2005", ""],
    ["0030415_1.pdf", "10.1370/afm.348", "2005", "0030415.pdf"],
    ["10_1177_08258597241245022.pdf", "10.1177/08258597241245022", "2014", ""],
    ["10_1177_10732748241236327.pdf", "10.1177/10732748241236327", "2024", ""],
    ["10_1177_2150132720922714.pdf", "10.1177/2150132720922714", "2020", ""],
    ["10_1177_2333392819882871.pdf", "10.1177/2333392819882871", "2017", ""],
    ["11606_2012_Article_2157.pdf", "10.1007/s11606-012-2157-7", "", ""],
    ["11606_2016_Article_3597.pdf", "10.1007/s11606-016-3597-2", "", ""],
    ["12885_2017_Article_3238.pdf", "10.1186/s12885-017-3238-0", "2017", ""],
    ["12888_2018_Article_1686.pdf", "10.1186/s12888-018-1686-y", "2018", ""],
    ["12888_2023_Article_4948.pdf", "10.1186/s12888-023-04948-w", "2023", ""],
    ["12888_2023_Article_4948_1.pdf", "10.1186/s12888-023-04948-w", "2023", "12888_2023_Article_4948.pdf"],
    ["12909_2023_Article_4010.pdf", "10.1186/s12909-023-04010-z", "2023", ""],
    ["12909_2023_Article_4010_1.pdf", "10.1186/s12909-023-04010-z", "2023", "12909_2023_Article_4010.pdf"],
    ["12909_2025_Article_6710.pdf", "10.1186/s12909-025-06710-0", "2025", ""],
    ["12909_2025_Article_7797.pdf", "10.1186/s12909-025-07797-1", "2025", ""],
    ["12909_2025_Article_7797_1.pdf", "10.1186/s12909-025-07797-1", "2025", "12909_2025_Article_7797.pdf"],
    ["12913_2024_Article_11647.pdf", "10.1186/s12913-024-11647-z", "2024", ""],
    ["12913_2025_Article_13506.pdf", "10.1186/s12913-025-13506-x", "2025", ""],
    ["AJRCCM1816566.pdf", "10.1164/rccm.200906-0907OC", "", ""],
    ["bmjopen153.pdf", "10.1136/bmjopen-2024-091143", "2025", ""],
    ["bmjopen2018023666.pdf", "10.1136/bmjopen-2018-023666", "2019", ""],
    ["fcvm111457039.pdf", "10.3389/fcvm.2024.1457039", "2024", ""],
    ["fphar141283135.pdf", "10.3389/fphar.2023.1283135", "", ""],
    ["healthcare0800026.pdf", "10.3390/healthcare8010026", "2019", ""],
    ["i15245012174317.pdf", "", "", ""],  # DOI truncated in extraction — manual
    ["ijgc2023004693.pdf", "10.1136/ijgc-2023-004693", "2023", ""],
    ["main.pdf", "10.1016/j.pecinn.2025.100426", "2025", ""],
    ["main_1.pdf", "10.1016/j.abd.2025.501228", "2025", ""],
    ["main_2.pdf", "10.1016/j.pecinn.2025.100426", "2025", "main.pdf"],
    ["main_3.pdf", "10.1016/j.jpainsymman.2020.07.022", "2020", ""],
    ["main_4.pdf", "10.1016/j.pecinn.2025.100436", "2025", ""],
    ["main_5.pdf", "10.1016/j.pecinn.2025.100399", "2025", ""],
    ["nihms1057661.pdf", "10.7326/M19-1152", "2020", ""],
    ["nihms1581950.pdf", "10.1016/j.pec.2020.03.019", "2021", ""],
    ["nihms1581950_1.pdf", "10.1016/j.pec.2020.03.019", "2021", "nihms1581950.pdf"],
    ["nihms1849896.pdf", "10.1016/j.jcomdis.2022.106274", "2022", ""],
    ["nihms1959154.pdf", "10.1016/j.jpainsymman.2022.11.029", "2022", ""],
    ["nihms250897.pdf", "10.1002/jhm.861", "2007", ""],
    ["nihms305491.pdf", "10.1016/j.pec.2011.04.023", "2011", ""],
    ["nihms648836.pdf", "10.1016/j.pec.2014.11.024", "2014", ""],
    ["nihms725148.pdf", "10.1016/j.pec.2015.09.001", "2015", ""],
    ["nihms856268.pdf", "10.1177/0272989X10364247", "", ""],
    ["pharmacy0600018.pdf", "10.3390/pharmacy6010018", "2018", ""],
    ["pharmacy0600018_1.pdf", "10.3390/pharmacy6010018", "2018", "pharmacy0600018.pdf"],
    ["pmr_2025_0005.pdf", "10.1089/pmr.2025.0005", "2025", ""],
    ["pone_0230672.pdf", "10.1371/journal.pone.0230672", "2020", ""],
    ["pone_0231350.pdf", "10.1371/journal.pone.0231350", "2019", ""],
    ["pone_0247259.pdf", "10.1371/journal.pone.0247259", "2021", ""],
    ["pone_0304180.pdf", "10.1371/journal.pone.0304180", "2024", ""],
    ["prbm12457.pdf", "10.2147/PRBM.S208427", "2017", ""],
    ["Tanaffos17241.pdf", "", "", ""],       # no DOI recovered — manual
    ["ppa2137.pdf", "", "", ""],             # no DOI recovered — manual
    ["fp3606284.pdf", "", "", ""],           # no DOI recovered — manual
    ["fp3606284_1.pdf", "", "", "fp3606284.pdf"],
]

UA_TEMPLATE = "CARELite-corpus-rebuild/1.0 (mailto:{email})"
OUT_DIR = pathlib.Path("carelite_pdfs")


def slug(doi: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", doi).strip("-").lower()


def unpaywall_pdf_url(doi: str, email: str, headers: dict) -> str | None:
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


def pmc_pdf_url(doi: str, headers: dict) -> str | None:
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


def download_pdf(url: str, dest: pathlib.Path, headers: dict) -> tuple[bool, str]:
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
        for chunk in it:
            fh.write(chunk)
    return True, f"{dest.stat().st_size // 1024} KB"


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild the CARELite paper corpus as real PDFs.")
    ap.add_argument("--email", required=True, help="required by the Unpaywall API")
    ap.add_argument("--out", default=str(OUT_DIR), help="output folder (default: ./carelite_pdfs)")
    args = ap.parse_args()

    headers = {"User-Agent": UA_TEMPLATE.format(email=args.email)}

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [row for row in MANIFEST if row[1] and not row[3]]  # has DOI, not a duplicate
    manual = [row for row in MANIFEST if not row[1]]

    print("CARELite corpus rebuild")
    print(f"  {len(MANIFEST)} original files -> {len(todo)} unique papers to fetch, "
          f"{len(manual)} need manual lookup\n")
    print(f"Downloading into: {out_dir.resolve()}\n")

    ok, failed = 0, []
    for i, (orig_file, doi, year, _dup) in enumerate(todo, 1):
        dest = out_dir / f"{year or 'nd'}_{slug(doi)}.pdf"
        if dest.exists():
            print(f"[{i:>2}/{len(todo)}] skip (exists)   {dest.name}")
            ok += 1
            continue

        url = None
        try:
            url = unpaywall_pdf_url(doi, args.email, headers) or pmc_pdf_url(doi, headers)
        except requests.RequestException as e:
            failed.append((orig_file, doi, f"lookup failed: {e}"))
            print(f"[{i:>2}/{len(todo)}] LOOKUP ERROR    {doi}")
            continue

        if not url:
            failed.append((orig_file, doi, "no open-access PDF found"))
            print(f"[{i:>2}/{len(todo)}] no OA link      {doi}")
            time.sleep(1)
            continue

        try:
            good, note = download_pdf(url, dest, headers)
        except requests.RequestException as e:
            good, note = False, str(e)

        if good:
            ok += 1
            print(f"[{i:>2}/{len(todo)}] ok  {note:>9}   {dest.name}")
        else:
            dest.unlink(missing_ok=True)
            failed.append((orig_file, doi, note))
            print(f"[{i:>2}/{len(todo)}] FAIL            {doi} - {note}")

        time.sleep(1)  # be polite to the free APIs

    manual_rows = [(f, d, "no DOI recovered from original file") for f, d, _y, _dup in manual]
    manual_rows += [(f, d, why) for f, d, why in failed]

    if manual_rows:
        manual_csv = out_dir / "_manual_needed.csv"
        with open(manual_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["original_file", "doi_or_blank", "note"])
            w.writerows(manual_rows)
        print(f"\n{len(manual_rows)} papers need manual retrieval - see {manual_csv}")
        print("For each: search the title on Google Scholar, or if you have the DOI, "
              "paste it into https://doi.org/<DOI>")

    print(f"\nDone: {ok}/{len(todo)} papers downloaded automatically.")
    print(f"PDFs are in: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())