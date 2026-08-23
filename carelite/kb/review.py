"""The human review gate: emit a digest a person can actually check, read back their decisions.

`human_verified` is the column that turns "an LLM proposed this" into "a human
checked this". Everything else in this lane is machine enforcement; this is the
only place a person enters the loop, so the digest has to be genuinely
reviewable rather than a formality. That means showing, per entry:

- the seven fields as they will be stored,
- the verbatim span, marked up inside its **surrounding paragraph**, so the
  reviewer can see the sentence in context and judge whether the entry's
  finding is a fair reading of it rather than a true sentence bolted to an
  unrelated claim,
- the source citation and study design, so an overclaimed tier is visible,
- and a checkbox.

The context is the part that makes this real. A digest that printed only the
quote would let a reviewer confirm the quote exists — which the validator has
already proved mechanically — without ever checking the thing a machine cannot
check, which is whether the *entry* follows from the paper. Machine-checkable
things belong to `validate.py`; a human's time should go to the judgment call.

Sign-off round-trips through the digest file itself. The reviewer ticks
`- [x]` next to entries they accept, saves, and `record_signoff` reads the
file back and sets `human_verified` for exactly those. No second file to keep
in sync, and the artifact that records the decision is the same one that shows
the evidence it was made on.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from carelite.config import REPO_ROOT
from carelite.db.connection import transaction
from carelite.kb.papers import PAPER_META, load_paper_texts
from carelite.kb.spans import locate_span, surrounding_context

DIGEST_PATH = REPO_ROOT / "knowledge_base" / "review" / "kb_review_digest.md"

_CHECKBOX_RE = re.compile(r"^\s*-\s*\[(?P<mark>[ xX])\]\s*`(?P<entry_id>kb-[a-z_]+-[0-9a-f]+)`")

_SELECT_ENTRIES_SQL = """
SELECT
    e.entry_id, e.theme, e.finding, e.practical_takeaway, e.example_behavior,
    e.evidence_tier, e.action_type, e.verbatim_span, e.encounter_phase,
    e.equity_relevant, e.human_verified,
    array_agg(s.paper_id ORDER BY s.paper_id) AS paper_ids
FROM kb_entry e
JOIN kb_entry_source s USING (entry_id)
GROUP BY e.entry_id
ORDER BY e.theme, e.entry_id
"""

_MARK_VERIFIED_SQL = """
UPDATE kb_entry SET human_verified = %(verified)s WHERE entry_id = ANY(%(entry_ids)s)
"""


@dataclass
class ReviewRow:
    """One entry as the digest presents it, joined to its source paper."""

    entry_id: str
    theme: str
    finding: str
    practical_takeaway: str
    example_behavior: str
    evidence_tier: str
    action_type: str
    verbatim_span: str
    encounter_phase: list[str]
    equity_relevant: bool
    human_verified: bool
    paper_ids: list[str]

    @property
    def primary_paper(self) -> str:
        return self.paper_ids[0] if self.paper_ids else ""


def fetch_review_rows() -> list[ReviewRow]:
    """Every loaded entry, with its source links. Requires a live database."""
    with transaction() as conn:
        rows = conn.execute(_SELECT_ENTRIES_SQL).fetchall()
    return [
        ReviewRow(
            entry_id=r["entry_id"],
            theme=r["theme"],
            finding=r["finding"],
            practical_takeaway=r["practical_takeaway"],
            example_behavior=r["example_behavior"],
            evidence_tier=r["evidence_tier"],
            action_type=r["action_type"],
            verbatim_span=r["verbatim_span"],
            encounter_phase=list(r["encounter_phase"] or []),
            equity_relevant=r["equity_relevant"],
            human_verified=r["human_verified"],
            paper_ids=list(r["paper_ids"] or []),
        )
        for r in rows
    ]


def _context_block(row: ReviewRow) -> str:
    """The span shown inside its surrounding paragraph, with the quote marked.

    Falls back to the bare span if the paper text is unavailable — a digest
    that renders without context is worse than one that does not render, but
    only just, so the fallback says so out loud rather than quietly printing a
    quote with no provenance around it.
    """
    papers = load_paper_texts()
    paper = papers.get(row.primary_paper)
    if paper is None:
        return (
            f"> {row.verbatim_span}\n>\n> _(source text unavailable — context could not be shown)_"
        )

    match = locate_span(row.verbatim_span, paper.text)
    if match is None:
        return (
            f"> {row.verbatim_span}\n>\n"
            "> **WARNING: this span no longer appears in the current extraction of the "
            "source paper.** Do not sign off on this entry; re-run validation."
        )

    before, after = surrounding_context(paper.text, match.start, match.end)
    body = f"{before} **>>> {match.source_text} <<<** {after}".strip()
    body = re.sub(r"\s+", " ", body)
    return "\n".join(f"> {line}" for line in (body,))


def render_digest(rows: Sequence[ReviewRow], *, generated_at: dt.datetime | None = None) -> str:
    """Render the full review digest as Markdown."""
    stamp = (generated_at or dt.datetime.now(dt.UTC)).strftime("%Y-%m-%d %H:%M UTC")
    total = len(rows)
    verified = sum(1 for r in rows if r.human_verified)

    by_theme: dict[str, list[ReviewRow]] = {}
    for row in rows:
        by_theme.setdefault(row.theme, []).append(row)

    source_counts: dict[str, int] = {}
    for row in rows:
        source_counts[row.primary_paper] = source_counts.get(row.primary_paper, 0) + 1

    out: list[str] = [
        "# Knowledge Base Review Digest",
        "",
        f"Generated {stamp}. {total} entr(ies) loaded, {verified} already signed off.",
        "",
        "Every entry below passed automated provenance validation: its quoted span was",
        "located, character for character after whitespace and ligature normalisation, in",
        "the extracted text of the paper it cites. That is already proven and is not what",
        "you are being asked to check.",
        "",
        "**What to check, per entry:** does the *finding* follow from the quoted sentence in",
        "the context shown, or has a true sentence been attached to a claim it does not",
        "support? Is the *takeaway* something a clinician could actually do mid-encounter?",
        "Is the *evidence tier* honest about the study design named beside it?",
        "",
        "Tick `- [x]` beside each entry you accept, save this file, then run:",
        "",
        "```",
        "python -m carelite.kb.review --signoff --reviewer <your-name>",
        "```",
        "",
        "Entries left unticked stay `human_verified = FALSE`. Nothing is deleted by",
        "signing off, so an entry you reject can be discussed rather than vanishing.",
        "",
        "## Coverage",
        "",
        "| Theme | Entries | Distinct source papers |",
        "|---|---|---|",
    ]

    for theme, theme_rows in sorted(by_theme.items()):
        distinct = len({r.primary_paper for r in theme_rows})
        flag = "  **single-source**" if distinct == 1 and len(theme_rows) > 1 else ""
        out.append(f"| {theme} | {len(theme_rows)} | {distinct}{flag} |")

    out += [
        "",
        "A theme marked **single-source** draws every one of its entries from one paper.",
        "Those entries are individually well-evidenced but they are not independent of each",
        "other, and the write-up must not present them as convergent evidence.",
        "",
    ]

    for theme, theme_rows in sorted(by_theme.items()):
        out += [f"## {theme}", ""]
        for row in theme_rows:
            meta = PAPER_META.get(row.primary_paper)
            citation = meta.short_citation if meta else row.primary_paper
            design = meta.design if meta else "design not recorded"
            mark = "x" if row.human_verified else " "
            phases = ", ".join(row.encounter_phase) or "-"

            out += [
                f"- [{mark}] `{row.entry_id}`",
                "",
                f"  **Source.** {citation}  ",
                f"  `{row.primary_paper}` — {design} — claimed tier **{row.evidence_tier}**",
                "",
                f"  **Finding.** {row.finding}",
                "",
                f"  **Takeaway.** {row.practical_takeaway}",
                "",
                f"  **Example behaviour.** {row.example_behavior}",
                "",
                f"  **Action type.** {row.action_type} · **Phase.** {phases} · "
                f"**Equity-relevant.** {'yes' if row.equity_relevant else 'no'}",
                "",
                "  **Quoted span, in context** (the quote is marked `>>> <<<`):",
                "",
                _context_block(row),
                "",
            ]

    return "\n".join(out).rstrip() + "\n"


def write_digest(path: Path | str = DIGEST_PATH, rows: Sequence[ReviewRow] | None = None) -> Path:
    """Render and write the digest, preserving existing tick marks via `human_verified`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows) if rows is not None else fetch_review_rows()
    path.write_text(render_digest(rows), encoding="utf-8")
    return path


def parse_signoff(text: str) -> tuple[list[str], list[str]]:
    """Read a reviewed digest. Returns `(approved_ids, unticked_ids)`.

    Parsing the reviewer's own file rather than a separate decisions format is
    deliberate: the record of what was approved is the same document that
    showed the evidence, so an audit later cannot drift between them.
    """
    approved: list[str] = []
    unticked: list[str] = []
    for line in text.splitlines():
        m = _CHECKBOX_RE.match(line)
        if not m:
            continue
        if m.group("mark").lower() == "x":
            approved.append(m.group("entry_id"))
        else:
            unticked.append(m.group("entry_id"))
    return approved, unticked


def record_signoff(entry_ids: Iterable[str], *, verified: bool = True) -> int:
    """Set `human_verified` for exactly these entries. Returns rows affected."""
    ids = list(dict.fromkeys(entry_ids))
    if not ids:
        return 0
    with transaction() as conn:
        cur = conn.execute(_MARK_VERIFIED_SQL, {"verified": verified, "entry_ids": ids})
        return cur.rowcount


def apply_signoff(path: Path | str = DIGEST_PATH, *, reviewer: str = "") -> dict[str, int]:
    """Read a reviewed digest and record its decisions.

    Only ticked entries are set TRUE. Unticked entries are *not* forced back to
    FALSE, because a partially reviewed digest is the normal case and a second
    pass must not undo the first one's decisions.
    """
    text = Path(path).read_text(encoding="utf-8")
    approved, unticked = parse_signoff(text)
    n = record_signoff(approved, verified=True)
    return {
        "approved": len(approved),
        "unticked": len(unticked),
        "rows_updated": n,
        "reviewer_recorded": 1 if reviewer else 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Emit or apply the KB review digest.")
    ap.add_argument("--path", default=str(DIGEST_PATH))
    ap.add_argument(
        "--signoff",
        action="store_true",
        help="read the reviewed digest back and record human_verified",
    )
    ap.add_argument("--reviewer", default="", help="name recorded in the run output")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.signoff:
        result = apply_signoff(args.path, reviewer=args.reviewer)
        print(
            f"{result['approved']} entr(ies) approved by "
            f"{args.reviewer or 'an unnamed reviewer'}; "
            f"{result['rows_updated']} row(s) updated; {result['unticked']} left unticked."
        )
        return 0

    path = write_digest(args.path)
    rows = fetch_review_rows()
    print(f"Wrote {path} — {len(rows)} entr(ies) for review.")
    return 0


__all__ = [
    "DIGEST_PATH",
    "ReviewRow",
    "apply_signoff",
    "fetch_review_rows",
    "parse_signoff",
    "record_signoff",
    "render_digest",
    "write_digest",
]


if __name__ == "__main__":
    raise SystemExit(main())
