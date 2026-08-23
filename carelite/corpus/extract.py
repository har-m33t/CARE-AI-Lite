"""carelite.corpus.extract — PDF -> cleaned text with pymupdf.

Strips running headers/footers (lines that repeat near the top/bottom of
many pages), the reference list, and figure/table captions, so downstream
chunking doesn't index citation noise. Extraction failures (a PDF that
won't open, or that yields no usable text) are recorded on the result
rather than silently producing an empty chunk stream.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from carelite.config import get_settings

REFERENCES_HEADING_RE = re.compile(
    r"^\s*(references|bibliography|works cited|literature cited)\.?\s*$",
    re.IGNORECASE,
)
FIGURE_CAPTION_RE = re.compile(
    r"^\s*(figure|fig\.?|table|supplementary\s+table)\s+[sS]?\d+[.:)]", re.IGNORECASE
)
_BLANK_RUN_RE = re.compile(r"\n{3,}")


@dataclass
class ExtractionFailure:
    pdf_path: str
    reason: str


@dataclass
class ExtractedPaper:
    pdf_path: str
    text: str
    page_count: int
    title: str | None = None
    author: str | None = None
    failures: list[ExtractionFailure] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures and bool(self.text.strip())


def _page_lines(page: pymupdf.Page) -> list[str]:
    return page.get_text("text").splitlines()


def _detect_running_headers_footers(pages_lines: list[list[str]], edge_lines: int = 2) -> set[str]:
    """Lines that repeat near the top/bottom of most pages are headers/footers.

    A one- or two-page PDF has nothing to detect repetition against, so this
    is a no-op below `min_pages_for_detection`.
    """
    n_pages = len(pages_lines)
    if n_pages < 3:
        return set()

    counts: Counter[str] = Counter()
    for lines in pages_lines:
        non_empty = [ln.strip() for ln in lines if ln.strip()]
        if not non_empty:
            continue
        candidates = set(non_empty[:edge_lines]) | set(non_empty[-edge_lines:])
        for c in candidates:
            if len(c) < 120:  # a repeated full sentence is not a header/footer
                counts[c] += 1

    threshold = max(2, int(n_pages * 0.4))
    return {line for line, c in counts.items() if c >= threshold}


def _strip_references_section(text: str) -> str:
    """Cut everything from the first standalone 'References' heading onward."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if REFERENCES_HEADING_RE.match(line.strip()):
            return "\n".join(lines[:i]).rstrip()
    return text


def _strip_figure_captions(text: str) -> str:
    kept = [ln for ln in text.splitlines() if not FIGURE_CAPTION_RE.match(ln.strip())]
    return "\n".join(kept)


def _collapse_blank_runs(text: str) -> str:
    return _BLANK_RUN_RE.sub("\n\n", text).strip()


def extract_pdf(pdf_path: Path | str) -> ExtractedPaper:
    """Extract cleaned text from one PDF. Never raises; failures are recorded."""
    pdf_path = Path(pdf_path)
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:  # pymupdf raises various backend-specific errors
        return ExtractedPaper(
            pdf_path=str(pdf_path),
            text="",
            page_count=0,
            failures=[ExtractionFailure(str(pdf_path), f"could not open: {type(e).__name__}: {e}")],
        )

    failures: list[ExtractionFailure] = []
    title: str | None = None
    author: str | None = None
    text = ""
    page_count = doc.page_count

    try:
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip() or None
        author = (meta.get("author") or "").strip() or None

        pages_lines = [_page_lines(doc[page_index]) for page_index in range(doc.page_count)]
        noise = _detect_running_headers_footers(pages_lines)
        cleaned_pages = []
        for lines in pages_lines:
            kept = [ln for ln in lines if ln.strip() not in noise]
            cleaned_pages.append("\n".join(kept))
        text = "\n\n".join(cleaned_pages)
    except Exception as e:
        failures.append(
            ExtractionFailure(str(pdf_path), f"text extraction failed: {type(e).__name__}: {e}")
        )
    finally:
        doc.close()

    if text.strip():
        text = _strip_references_section(text)
        text = _strip_figure_captions(text)
        text = _collapse_blank_runs(text)

    if not text.strip() and not failures:
        failures.append(ExtractionFailure(str(pdf_path), "extraction produced no usable text"))

    return ExtractedPaper(
        pdf_path=str(pdf_path),
        text=text,
        page_count=page_count,
        title=title,
        author=author,
        failures=failures,
    )


def iter_pdfs(pdf_dir: Path | str) -> Iterator[Path]:
    return iter(sorted(Path(pdf_dir).glob("*.pdf")))


def extract_corpus(
    pdf_dir: Path | str | None = None,
) -> tuple[list[ExtractedPaper], list[ExtractionFailure]]:
    """Extract every PDF in `pdf_dir` (default: settings.pdf_dir).

    Returns `(results, failures)` — `failures` is the flattened list across
    all papers, so a caller can report them without walking `results` again.
    """
    resolved = Path(pdf_dir) if pdf_dir is not None else get_settings().pdf_dir
    results: list[ExtractedPaper] = []
    failures: list[ExtractionFailure] = []
    for pdf_path in iter_pdfs(resolved):
        r = extract_pdf(pdf_path)
        results.append(r)
        failures.extend(r.failures)
    return results, failures


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Extract cleaned text from every PDF in a directory.")
    ap.add_argument("pdf_dir", nargs="?", default=None, help="default: settings.pdf_dir")
    args = ap.parse_args(argv)

    results, failures = extract_corpus(args.pdf_dir)
    print(f"Extracted {len(results)} PDFs, {len(failures)} failure(s).")
    for f in failures:
        print(f"  FAIL  {f.pdf_path}: {f.reason}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
