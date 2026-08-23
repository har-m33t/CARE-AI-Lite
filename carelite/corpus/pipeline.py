"""carelite.corpus.pipeline — extract -> chunk -> load, tied together.

Not one of the wave-1 brief's five modules, but a direct consequence of
having them: after an extraction or chunking bug fix, the corpus on disk
needs to be re-extracted, re-chunked, and pushed back into Postgres without
leaving stale rows behind. `reload_corpus` is that one operation; `load`,
`extract`, and `chunk` stay independently usable and independently tested.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from carelite.corpus.chunk import chunk_text
from carelite.corpus.extract import extract_source
from carelite.corpus.fetch import manifest_papers
from carelite.corpus.load import replace_paper_chunks, upsert_papers
from carelite.types import Paper


@dataclass
class ReloadReport:
    papers: int = 0
    chunks: int = 0
    extraction_failures: list[str] = field(default_factory=list)


def reload_corpus(source_dir: Path | str | None = None) -> ReloadReport:
    """Extract every paper currently on disk, re-chunk it, and push the
    result into Postgres.

    Papers are upserted; each paper's chunks are *replaced* via
    `carelite.corpus.load.replace_paper_chunks` (delete existing rows for
    that paper_id, then insert the fresh set) rather than merely upserted,
    so a chunk set that shrank since the last load — the point of this
    function existing — doesn't leave stale rows behind. A paper whose
    extraction fails has its chunk rows cleared (empty replace) rather than
    left at a stale, possibly-junk previous state, and is reported rather
    than silently skipped.
    """
    papers: list[Paper] = manifest_papers(source_dir)
    upsert_papers(papers)

    report = ReloadReport(papers=len(papers))
    for paper in papers:
        result = extract_source(paper.pdf_path or "")
        if not result.ok:
            reasons = "; ".join(f.reason for f in result.failures) or "no usable text"
            report.extraction_failures.append(f"{paper.paper_id}: {reasons}")
            replace_paper_chunks(paper.paper_id, [])
            continue

        chunks = chunk_text(paper.paper_id, result.text)
        replace_paper_chunks(paper.paper_id, chunks)
        report.chunks += len(chunks)

    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract -> chunk -> load the corpus currently on disk into Postgres."
    )
    ap.add_argument("source_dir", nargs="?", default=None, help="default: settings.pdf_dir")
    args = ap.parse_args(argv)

    report = reload_corpus(args.source_dir)
    print(f"Loaded {report.papers} papers, {report.chunks} chunks.")
    if report.extraction_failures:
        print(f"{len(report.extraction_failures)} extraction failure(s):")
        for f in report.extraction_failures:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
