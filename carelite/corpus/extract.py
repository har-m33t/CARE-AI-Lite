"""carelite.corpus.extract — PDF or JATS full-text XML -> cleaned text.

Two source kinds, one output shape (`ExtractedPaper.text`, paragraphs
separated by blank lines, so `carelite.corpus.chunk` needs no changes for
either):

- **PDF** (`extract_pdf`, via pymupdf): strips running headers/footers
  (lines that repeat near the top/bottom of many pages, plus a page's own
  "N / M" pagination line), the reference list, and figure/table captions —
  all heuristic, since a PDF has no structure beyond page geometry.
- **XML** (`extract_xml_jats`): PMC/Europe PMC full-text is JATS XML with
  real `<sec>`/`<p>`/`<title>` structure. References (`<back>`), figures,
  and tables are skipped by construction — walking only `<front>/abstract`
  and `<body>`, and never descending into `<fig>`/`<table-wrap>` — rather
  than pattern-matched after the fact, so this source is more reliable, not
  less, than the PDF path for the same cleanup job.

Both paths finish with `_fix_layout_artefacts`, which repairs two distinct
word-level PDF extraction artefacts — see that function's docstring and
`_rejoin_split_words`/`_fix_glued_words` for what is and is not attempted,
and why: a general "any glued word" detector was built and measured against
the full corpus during development and rejected for its false-positive
rate (it "fixed" real words like "healthcare" and names like "Pearson"
into nonsense); what shipped is deliberately narrower.

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
from functools import lru_cache
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


def _detect_running_headers_footers(pages_lines: list[list[str]], edge_lines: int = 8) -> set[str]:
    """Lines that repeat near the top/bottom of most pages are headers/footers.

    A one- or two-page PDF has nothing to detect repetition against, so this
    is a no-op below `min_pages_for_detection`.

    `edge_lines` was originally 2, which only ever caught a single-line
    footer. Real running-header/footer blocks are commonly 4-9 lines
    (journal name, short title, DOI, date, page number — each pymupdf
    "text"-mode line is one of these), and multi-column layouts can push
    that block several lines further in from the literal top/bottom of the
    per-page line list (a sidebar column's tail can sort after it in
    reading order). At `edge_lines=2` most of a PLOS ONE-style footer
    survived uncleaned and could land mid-sentence once a page break falls
    inside a sentence — see `extract_pdf`'s docstring. 8 was chosen by
    measuring the known footer/header blocks in this corpus (PLOS ONE: 4
    lines + pagination; BMJ Open: ~9-line "Downloaded from ... by guest"
    watermark) and checking it doesn't start pulling in a repeated
    multi-page table's column headers as if they were boilerplate — see
    `tests/unit/corpus/test_extract.py` for both directions pinned.
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


#: A page's own "N / M" pagination line (e.g. "5 / 16"). Unlike the rest of
#: a running footer this text is *different on every page*, so it can never
#: clear `_detect_running_headers_footers`'s repeated-line threshold — it
#: needs its own pattern-based check. Validated against the whole corpus:
#: every match is a genuine page number (the exact same three PLOS ONE
#: papers that have a "PLOS ONE | https://doi.org/..." footer, each ratio
#: matching `<page index> / <page count>`), and no other paper's body text
#: matches this pattern at all — no fraction, ratio, or score written on a
#: line by itself. Still gated on `n_pages >= 3` for consistency with the
#: header/footer detector above, even though the pattern itself carries no
#: repetition requirement.
_PAGE_NUMBER_RE = re.compile(r"^\d{1,4}\s*/\s*\d{1,4}$")


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


# ---------------------------------------------------------------------------
# Word-level layout artefacts: PDF column breaks and dropped line-wrap spaces
# ---------------------------------------------------------------------------
#
# pymupdf's line-based text extraction occasionally does the opposite thing
# to a word depending on layout: a multi-column page can split one word
# across two "lines" ("collabora" / "tive"), landing in the extracted text
# as two space-separated fragments once lines are joined; a line-wrap can
# instead drop the boundary entirely, landing as one glued token
# ("healthrelated"). Both corrupt the token stream that gets embedded and
# full-text indexed — "healthrelated" doesn't match a search for
# "health-related", and a stray fragment like "tive" tokenises as noise.
#
# A general "is this token a real word" detector was the first approach
# tried for *both* directions, backed by a large English wordlist
# (`_wordlist.txt`, derived from the system dictionary). It works cleanly
# for the split-and-rejoin direction (`_rejoin_split_words` below) because
# requiring *both* space-separated fragments to be absent from the
# wordlist, with their concatenation present, is a strict enough gate that
# it produced zero false positives across the full corpus at introduction.
#
# It does not work for the glued direction. Run the mirror-image rule
# (single token absent from the wordlist; unique two-way split where both
# halves ARE present) against this corpus and the false-positive dictionary
# gap dominates the output: "healthcare" -> "health"/"care",
# "checklist" -> "check"/"list", "Pearson" -> "Pears"/"on",
# "asking" -> "as"/"king", "became" -> "be"/"came" — genuine words, names,
# and inflected forms that simply happen not to be one of the ~210k lemmas
# in a general dictionary, each one a real corruption if "fixed". No
# frequency threshold or corpus-derived vocabulary closed that gap either
# (a real word used once in 33 papers is indistinguishable, by that
# signal, from a genuine artefact used once) — see the commit history for
# the measurements. Shipping that rule would trade a bounded, cosmetic
# defect for an unbounded, silent one, which is the wrong trade.
#
# So the glued direction (`_fix_glued_words` below) is instead a small,
# hand-verified lookup table of closed-class function-word pairs
# ("inthe" -> "in the"), the one sub-pattern where every member of the
# closed set could be checked for collisions against real content. It has
# high precision and deliberately low recall — an open-class glue like
# "healthrelated" is left untouched rather than guessed at. A future,
# properly scoped fix for that case would use pymupdf's word/character
# bounding boxes (an anomalously small or negative gap between adjacent
# glyph runs is real evidence of a layout-induced split, independent of
# any dictionary) rather than text-only heuristics.

_WORDLIST_PATH = Path(__file__).parent / "_wordlist.txt"


@lru_cache(maxsize=1)
def _load_wordlist() -> frozenset[str]:
    """General-English wordlist backing `_rejoin_split_words` (lowercase,
    length >= 2, derived from the system dictionary). Loaded once per
    process. Missing file degrades to "no rejoin fixes applied", not a
    crash — this is a best-effort cleanup pass, not a required input.
    """
    try:
        with _WORDLIST_PATH.open(encoding="utf-8") as f:
            return frozenset(line.strip() for line in f if line.strip())
    except OSError:
        return frozenset()


_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z]+")
_MIN_FRAGMENT_LEN = 2
_MIN_REJOINED_LEN = 5


def _rejoin_split_words(text: str) -> str:
    """Repair a PDF column-break word split: "collabora tive" -> "collaborative".

    Fires only when *both* space-separated fragments are absent from the
    wordlist but their concatenation is present. That's the strict
    direction on purpose: a pair where either side independently reads as
    a real word (e.g. "set out", "a bout") is left alone, because that
    ambiguity can't be resolved from text alone and a wrong merge is worse
    than a surviving split — see the module-level note above for what this
    traded off against. Restricted to the same line (a literal space or
    tab, not a newline): a sentence that genuinely spans a page break is a
    different phenomenon (the page-join itself, not a word split) and
    isn't touched here. The length-5 floor on the merged word is a cheap
    extra guard against short coincidental collisions (e.g. a split
    citation fragment like "ad et" from "Mahmoudir-ad et al." very nearly
    matched "adet" during testing) rather than a load-bearing one — the
    both-fragments-absent gate is what does the real work.

    Scans consecutive alpha tokens explicitly rather than doing a single
    `re.sub` over a two-token pattern: `re.sub` only returns *non-overlapping*
    matches, so a naive "word word" regex silently skips every other
    adjacent pair (in "sta tistically across", it pairs "assessed"/"sta"
    first and so never even looks at "sta"/"tistically" together) — this
    was caught by `test_rejoin_split_words_joins_a_second_example` during
    development.
    """
    words = _load_wordlist()
    if not words:
        return text

    tokens = list(_ALPHA_TOKEN_RE.finditer(text))
    merge_at: set[int] = set()
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if b.start() - a.end() != 1 or text[a.end() : b.start()] not in (" ", "\t"):
            continue  # must be same line, separated by exactly one space/tab
        a_word, b_word = a.group(), b.group()
        if len(a_word) < _MIN_FRAGMENT_LEN or len(b_word) < _MIN_FRAGMENT_LEN:
            continue
        whole = a_word + b_word
        if (
            len(whole) >= _MIN_REJOINED_LEN
            and whole.lower() in words
            and a_word.lower() not in words
            and b_word.lower() not in words
        ):
            merge_at.add(i)

    if not merge_at:
        return text

    out: list[str] = []
    last = 0
    i = 0
    n = len(tokens)
    while i < n:
        if i in merge_at:
            a, b = tokens[i], tokens[i + 1]
            out.append(text[last : a.start()])
            out.append(a.group() + b.group())
            last = b.end()
            i += 2
        else:
            i += 1
    out.append(text[last:])
    return "".join(out)


#: Hand-verified PDF line-wraps that glued a closed-class function word
#: straight onto its neighbour with no separator at all — distinct from the
#: column-break *split* `_rejoin_split_words` handles above. Each key was
#: checked against the full corpus before being added: none collides with a
#: genuine word, name, or abbreviation (e.g. "isa"/"wasa" were considered
#: and dropped — both read as plausible names/typos and neither actually
#: occurs, so there was nothing to gain by keeping them). This is
#: deliberately a fixed lookup, not a general rule; see the module-level
#: note above for why a general glued-word detector was rejected.
_GLUE_FIXES: dict[str, str] = {
    "inthe": "in the", "ofthe": "of the", "tothe": "to the", "andthe": "and the",
    "forthe": "for the", "withthe": "with the", "onthe": "on the", "atthe": "at the",
    "bythe": "by the", "fromthe": "from the", "asthe": "as the", "inthis": "in this",
    "ofthis": "of this", "tothis": "to this", "onthis": "on this", "atthis": "at this",
    "hada": "had a", "thatthe": "that the", "whenthe": "when the", "ifthe": "if the",
    "thisis": "this is", "amongthe": "among the", "overthe": "over the",
    "underthe": "under the", "sincethe": "since the", "whilethe": "while the",
    "beforethe": "before the", "afterthe": "after the", "aboutthe": "about the",
    "aroundthe": "around the", "betweenthe": "between the", "duringthe": "during the",
    "throughthe": "through the", "acrossthe": "across the", "withinthe": "within the",
    "withoutthe": "without the", "towardsthe": "towards the", "towardthe": "toward the",
}  # fmt: skip
_GLUE_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _GLUE_FIXES) + r")\b", re.IGNORECASE)


def _fix_glued_words(text: str) -> str:
    """Repair the fixed, hand-verified glued-word patterns in `_GLUE_FIXES`."""

    def repl(m: re.Match[str]) -> str:
        matched = m.group(0)
        fixed = _GLUE_FIXES[matched.lower()]
        if matched[0].isupper():
            fixed = fixed[0].upper() + fixed[1:]
        return fixed

    return _GLUE_RE.sub(repl, text)


def _fix_layout_artefacts(text: str) -> str:
    """Repair word-level PDF/XML layout artefacts. See `_rejoin_split_words`
    and `_fix_glued_words` for the two patterns this covers, and the
    module-level note above them for the (larger) pattern this deliberately
    does not attempt.
    """
    text = _fix_glued_words(text)
    text = _rejoin_split_words(text)
    return text


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
        strip_pagination = len(pages_lines) >= 3
        cleaned_pages = []
        for lines in pages_lines:
            kept = []
            for ln in lines:
                stripped = ln.strip()
                if stripped in noise:
                    continue
                if strip_pagination and _PAGE_NUMBER_RE.match(stripped):
                    continue
                kept.append(ln)
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
        text = _fix_layout_artefacts(text)

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
        text = _fix_layout_artefacts(text)

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
