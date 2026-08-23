"""carelite.corpus.extract — PDF or JATS full-text XML -> cleaned text.

Two source kinds, one output shape (`ExtractedPaper.text`, paragraphs
separated by blank lines, so `carelite.corpus.chunk` needs no changes for
either):

- **PDF** (`extract_pdf`, via pymupdf): strips running headers/footers
  (lines that repeat near the top/bottom of many pages), the reference
  list, and figure/table captions — all heuristic, since a PDF has no
  structure beyond page geometry.
- **XML** (`extract_xml_jats`): PMC/Europe PMC full-text is JATS XML with
  real `<sec>`/`<p>`/`<title>` structure. References (`<back>`), figures,
  and tables are skipped by construction — walking only `<front>/abstract`
  and `<body>`, and never descending into `<fig>`/`<table-wrap>` — rather
  than pattern-matched after the fact, so this source is more reliable, not
  less, than the PDF path for the same cleanup job.

`extract_source` dispatches on file suffix; `extract_corpus` picks up every
`.pdf` and `.xml` file in a directory. Extraction failures (a file that
won't open/parse, or that yields no usable text) are recorded on the
result rather than silently producing an empty chunk stream.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ElementTree
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
    source_path: str
    reason: str


@dataclass
class ExtractedPaper:
    source_path: str
    text: str
    page_count: int
    source_kind: str = "pdf"  # "pdf" | "xml"
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
            source_path=str(pdf_path),
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
        source_path=str(pdf_path),
        text=text,
        page_count=page_count,
        source_kind="pdf",
        title=title,
        author=author,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# JATS full-text XML (PMC / Europe PMC fullTextXML)
# ---------------------------------------------------------------------------

_SKIP_SUBTREES = {
    "fig",
    "fig-group",
    "table-wrap",
    "table-wrap-group",
    "disp-formula",
    "disp-formula-group",
    "graphic",
    "media",
    "ref-list",
    "ack",
    # PLOS-style "Supporting information" sections wrap each attachment in
    # <supplementary-material><label>S1 File</label><caption><p>(PDF)</p>
    # </caption><media .../></supplementary-material>. Without this, the
    # caption's format tag ("(PDF)", "(TIF)", ...) is the only text inside
    # the subtree and leaks out as its own near-empty block — worse, it then
    # gets misread as a section heading by chunk.py's ALL-CAPS heuristic
    # (str.isupper() ignores the parens) and ends up as its own standalone
    # chunk. Excluding the element structurally, like <fig>/<table-wrap>,
    # is correct where a regex on "(PDF)" would also strip legitimate prose
    # that happens to end a sentence with an all-caps parenthetical.
    "supplementary-material",
}


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _elem_text(elem: ElementTree.Element) -> str:
    """Flatten an element's own + descendant text, collapsing internal whitespace."""
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()


def _walk_jats_paragraphs(elem: ElementTree.Element) -> list[str]:
    """Recursively collect heading/paragraph text blocks from a JATS `<body>` or `<sec>`.

    Never descends into `_SKIP_SUBTREES` (figures, tables, formulas, ref
    lists, acknowledgments) — the XML equivalent of the PDF path's caption
    and reference-list stripping, done structurally instead of by regex.
    """
    blocks: list[str] = []
    for child in elem:
        tag = _local_tag(child.tag)
        if tag in _SKIP_SUBTREES:
            continue
        if tag == "sec":
            title_elem = child.find("title")
            if title_elem is not None:
                title_text = _elem_text(title_elem)
                if title_text:
                    blocks.append(title_text)
            blocks.extend(_walk_jats_paragraphs(child))
        elif tag == "title":
            continue  # emitted by the parent <sec> branch above; skip here
        elif tag == "p":
            p_text = _elem_text(child)
            if p_text:
                blocks.append(p_text)
        else:
            # Any other structural wrapper (e.g. <boxed-text>, <sec> variants) — recurse.
            blocks.extend(_walk_jats_paragraphs(child))
    return blocks


def extract_xml_jats(xml_path: Path | str) -> ExtractedPaper:
    """Extract cleaned text from one JATS full-text XML file. Never raises;
    failures are recorded, matching `extract_pdf`'s contract."""
    xml_path = Path(xml_path)
    try:
        root = ElementTree.fromstring(xml_path.read_bytes())
    except Exception as e:  # malformed XML, or not XML at all
        return ExtractedPaper(
            source_path=str(xml_path),
            text="",
            page_count=0,
            source_kind="xml",
            failures=[
                ExtractionFailure(str(xml_path), f"could not parse XML: {type(e).__name__}: {e}")
            ],
        )

    failures: list[ExtractionFailure] = []
    title: str | None = None
    blocks: list[str] = []

    try:
        title_elem = root.find(".//article-title")
        if title_elem is not None:
            title = _elem_text(title_elem) or None

        abstract_elem = root.find(".//abstract")
        if abstract_elem is not None:
            abstract_blocks = _walk_jats_paragraphs(abstract_elem)
            if abstract_blocks:
                blocks.append("Abstract")
                blocks.extend(abstract_blocks)

        body_elem = root.find(".//body")
        if body_elem is not None:
            blocks.extend(_walk_jats_paragraphs(body_elem))
    except Exception as e:
        failures.append(
            ExtractionFailure(str(xml_path), f"text extraction failed: {type(e).__name__}: {e}")
        )

    text = "\n\n".join(b for b in blocks if b.strip())
    if text.strip():
        text = _collapse_blank_runs(text)

    if not text.strip() and not failures:
        failures.append(ExtractionFailure(str(xml_path), "extraction produced no usable text"))

    return ExtractedPaper(
        source_path=str(xml_path),
        text=text,
        page_count=0,  # not a meaningful concept for XML
        source_kind="xml",
        title=title,
        author=None,
        failures=failures,
    )


def extract_source(path: Path | str) -> ExtractedPaper:
    """Dispatch to `extract_pdf` or `extract_xml_jats` by file suffix."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".xml":
        return extract_xml_jats(path)
    return ExtractedPaper(
        source_path=str(path),
        text="",
        page_count=0,
        failures=[ExtractionFailure(str(path), f"unsupported file type: {suffix or '(none)'}")],
    )


def iter_pdfs(pdf_dir: Path | str) -> Iterator[Path]:
    return iter(sorted(Path(pdf_dir).glob("*.pdf")))


def iter_source_files(source_dir: Path | str) -> Iterator[Path]:
    """Every `.pdf` and `.xml` file in `source_dir`, sorted for deterministic order."""
    source_dir = Path(source_dir)
    return iter(sorted((*source_dir.glob("*.pdf"), *source_dir.glob("*.xml"))))


def extract_corpus(
    source_dir: Path | str | None = None,
) -> tuple[list[ExtractedPaper], list[ExtractionFailure]]:
    """Extract every `.pdf`/`.xml` file in `source_dir` (default: settings.pdf_dir).

    Returns `(results, failures)` — `failures` is the flattened list across
    all papers, so a caller can report them without walking `results` again.
    """
    resolved = Path(source_dir) if source_dir is not None else get_settings().pdf_dir
    results: list[ExtractedPaper] = []
    failures: list[ExtractionFailure] = []
    for path in iter_source_files(resolved):
        r = extract_source(path)
        results.append(r)
        failures.extend(r.failures)
    return results, failures


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract cleaned text from every .pdf/.xml file in a directory."
    )
    ap.add_argument("source_dir", nargs="?", default=None, help="default: settings.pdf_dir")
    args = ap.parse_args(argv)

    results, failures = extract_corpus(args.source_dir)
    print(f"Extracted {len(results)} document(s), {len(failures)} failure(s).")
    for f in failures:
        print(f"  FAIL  {f.source_path}: {f.reason}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
