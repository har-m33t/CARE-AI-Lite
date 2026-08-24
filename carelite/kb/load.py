"""Import validated entries into `kb_entry` and `kb_entry_source`.

Only output of `carelite.kb.validate` should ever reach this module. It does
not re-check provenance — it takes `ValidatedEntry` objects, which by
construction carry a span that was located in a real paper.

Two properties of the upsert are load-bearing.

**`human_verified` is never written here.** The column defaults FALSE in the
frozen schema and the `ON CONFLICT` clause deliberately omits it, so re-running
the loader after a review gate cannot silently un-verify entries a human
already signed off. Sign-off is recorded by `carelite.kb.review` and nothing
else. Getting this backwards would be quiet and expensive: the entries would
still be there, still correct, and no longer marked as reviewed.

**`kb_entry_source` is rewritten, not merged.** An entry's sources are a
complete statement about what backs it, so a re-load replaces the set rather
than accumulating rows from a previous extraction that may no longer apply.

**A re-load is not only additive, so it has to prune.** An upsert alone leaves
behind every entry the previous run produced and this one did not, and both
things that remove an entry are routine. A validator rule tightens, and
thirteen entries stop qualifying. A paper's extraction is corrected — the
corpus lane repairing `inthe` to `in the` — and the span changes, which changes
the `entry_id` derived from it, so the same finding arrives under a new id and
the old row survives as a duplicate quoting text the paper no longer contains.
Left alone, the knowledge base accumulates exactly the entries the pipeline has
decided against, and every one of them is still reachable by retrieval.

`prune_entries` is therefore part of a full reload and refuses one case: it will
not silently delete a row whose `human_verified` is TRUE. Nothing in this corpus
is (`DECISIONS.md` D4), but an entry a person has read is a decision, and a rule
change should not be able to erase one without saying so.

`kb_entry.embedding` is left NULL. Embedding belongs to the index lane, which
reads these rows after the review gate.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from carelite.db.connection import transaction
from carelite.kb.validate import ValidatedEntry
from carelite.types import KBEntry

_UPSERT_ENTRY_SQL = """
INSERT INTO kb_entry (
    entry_id, theme, finding, practical_takeaway, example_behavior,
    evidence_tier, action_type, verbatim_span,
    encounter_phase, nurse_component, four_habits, equity_relevant
)
VALUES (
    %(entry_id)s, %(theme)s, %(finding)s, %(practical_takeaway)s, %(example_behavior)s,
    %(evidence_tier)s, %(action_type)s, %(verbatim_span)s,
    %(encounter_phase)s, %(nurse_component)s, %(four_habits)s, %(equity_relevant)s
)
ON CONFLICT (entry_id) DO UPDATE SET
    theme = EXCLUDED.theme,
    finding = EXCLUDED.finding,
    practical_takeaway = EXCLUDED.practical_takeaway,
    example_behavior = EXCLUDED.example_behavior,
    evidence_tier = EXCLUDED.evidence_tier,
    action_type = EXCLUDED.action_type,
    verbatim_span = EXCLUDED.verbatim_span,
    encounter_phase = EXCLUDED.encounter_phase,
    nurse_component = EXCLUDED.nurse_component,
    four_habits = EXCLUDED.four_habits,
    equity_relevant = EXCLUDED.equity_relevant
-- human_verified is intentionally NOT updated: re-loading must never
-- discard a review decision. See carelite.kb.review.
"""

_DELETE_SOURCES_SQL = "DELETE FROM kb_entry_source WHERE entry_id = %(entry_id)s"

_INSERT_SOURCE_SQL = """
INSERT INTO kb_entry_source (entry_id, paper_id)
VALUES (%(entry_id)s, %(paper_id)s)
ON CONFLICT (entry_id, paper_id) DO NOTHING
"""

_COUNT_SQL = """
SELECT
    count(*) AS n_entries,
    count(*) FILTER (WHERE human_verified) AS n_verified
FROM kb_entry
"""

#: Entries whose source paper is missing from `paper` would violate the
#: `kb_entry_source` foreign key. Checking first turns a raw psycopg
#: ForeignKeyViolation mid-transaction into a named, actionable failure.
_KNOWN_PAPERS_SQL = "SELECT paper_id FROM paper"


def entry_params(entry: KBEntry) -> dict[str, object]:
    return {
        "entry_id": entry.entry_id,
        "theme": entry.theme.value,
        "finding": entry.finding,
        "practical_takeaway": entry.practical_takeaway,
        "example_behavior": entry.example_behavior,
        "evidence_tier": entry.evidence_tier.value,
        "action_type": entry.action_type.value,
        "verbatim_span": entry.verbatim_span,
        "encounter_phase": [p.value for p in entry.encounter_phase],
        "nurse_component": list(entry.nurse_component),
        "four_habits": list(entry.four_habits),
        "equity_relevant": entry.equity_relevant,
    }


@dataclass
class LoadResult:
    entries_written: int = 0
    sources_written: int = 0
    skipped_unknown_paper: tuple[str, ...] = ()

    def __str__(self) -> str:
        out = (
            f"{self.entries_written} entr(ies) upserted, "
            f"{self.sources_written} source link(s) written."
        )
        if self.skipped_unknown_paper:
            out += (
                f" Skipped {len(self.skipped_unknown_paper)} entr(ies) whose source paper "
                f"is not in the `paper` table: {', '.join(self.skipped_unknown_paper[:5])}"
            )
        return out


def load_entries(validated: Iterable[ValidatedEntry]) -> LoadResult:
    """Upsert validated entries and their source links in one transaction.

    An entry whose source paper is absent from `paper` is skipped and named
    rather than loaded without provenance — a `kb_entry` row with no
    `kb_entry_source` row is exactly the untraceable entry this lane exists to
    prevent, and it would satisfy the schema perfectly well.
    """
    items = list(validated)
    result = LoadResult()
    if not items:
        return result

    with transaction() as conn:
        known = {row["paper_id"] for row in conn.execute(_KNOWN_PAPERS_SQL).fetchall()}

        skipped: list[str] = []
        for item in items:
            sources = [p for p in item.entry.source_paper_ids if p in known]
            if not sources:
                skipped.append(item.entry_id)
                continue

            conn.execute(_UPSERT_ENTRY_SQL, entry_params(item.entry))
            result.entries_written += 1

            conn.execute(_DELETE_SOURCES_SQL, {"entry_id": item.entry_id})
            for paper_id in sources:
                conn.execute(_INSERT_SOURCE_SQL, {"entry_id": item.entry_id, "paper_id": paper_id})
                result.sources_written += 1

        result.skipped_unknown_paper = tuple(skipped)

    return result


@dataclass
class PruneResult:
    deleted: tuple[str, ...] = ()
    kept_verified: tuple[str, ...] = ()

    def __str__(self) -> str:
        out = f"{len(self.deleted)} stale entr(ies) deleted."
        if self.kept_verified:
            out += (
                f" {len(self.kept_verified)} kept because a human had verified them: "
                f"{', '.join(self.kept_verified[:5])}"
            )
        return out


def prune_entries(keep: Iterable[str], *, dry_run: bool = False) -> PruneResult:
    """Delete `kb_entry` rows that the current pipeline no longer produces.

    `keep` is the full set of entry ids the run just validated. Anything else in
    the table is from a previous run under different rules or against different
    paper text, and leaving it there would make the knowledge base the union of
    every extraction ever run rather than the output of this one.

    `kb_entry_source` rows go with it via `ON DELETE CASCADE`; so does anything
    else keyed on `entry_id`. A human-verified row is reported and kept.
    """
    keep_ids = set(keep)
    with transaction() as conn:
        rows = conn.execute("SELECT entry_id, human_verified FROM kb_entry").fetchall()
        stale = [r["entry_id"] for r in rows if r["entry_id"] not in keep_ids]
        verified = {r["entry_id"] for r in rows if r["human_verified"]}

        deletable = sorted(e for e in stale if e not in verified)
        protected = sorted(e for e in stale if e in verified)

        if deletable and not dry_run:
            conn.execute("DELETE FROM kb_entry WHERE entry_id = ANY(%s)", (deletable,))

    return PruneResult(deleted=tuple(deletable), kept_verified=tuple(protected))


def kb_counts() -> dict[str, int]:
    """`(n_entries, n_verified)` — the number the review gate moves."""
    with transaction() as conn:
        row = conn.execute(_COUNT_SQL).fetchone()
    return {"n_entries": row["n_entries"], "n_verified": row["n_verified"]} if row else {}


def orphaned_entries() -> list[str]:
    """Entry ids with no row in `kb_entry_source`. Should always be empty."""
    with transaction() as conn:
        rows = conn.execute(
            """
            SELECT e.entry_id
            FROM kb_entry e
            LEFT JOIN kb_entry_source s USING (entry_id)
            WHERE s.entry_id IS NULL
            ORDER BY e.entry_id
            """
        ).fetchall()
    return [r["entry_id"] for r in rows]


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from carelite.kb.extract import CACHE_PATH
    from carelite.kb.validate import candidates_from_cache, format_report, validate_candidates

    ap = argparse.ArgumentParser(description="Validate cached candidates and load survivors.")
    ap.add_argument("--cache", default=str(CACHE_PATH))
    ap.add_argument("--dry-run", action="store_true", help="validate and report, write nothing")
    ap.add_argument(
        "--no-prune",
        action="store_true",
        help="leave entries the current rules no longer produce in place (they will be stale)",
    )
    ap.add_argument(
        "--sync-papers",
        action="store_true",
        help="also write design, evidence tier, citation and year onto the `paper` rows",
    )
    ap.add_argument(
        "--prompt-version",
        action="append",
        dest="prompt_versions",
        help="only load candidates extracted with this prompt version; repeatable. "
        "An experimental variant should reach the knowledge base when its guard has "
        "been applied, not when its inference finishes.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    candidates = candidates_from_cache(args.cache, prompt_versions=args.prompt_versions)
    report = validate_candidates(candidates)
    print(format_report(report))

    if args.dry_run:
        return 0

    if args.sync_papers:
        from carelite.kb.papers import sync_paper_metadata

        print(sync_paper_metadata())

    result = load_entries(report.accepted)
    print(result)

    if not args.no_prune:
        pruned = prune_entries(e.entry_id for e in report.accepted)
        print(pruned)
    orphans = orphaned_entries()
    if orphans:
        print(f"  WARNING  {len(orphans)} entr(ies) with no source link: {orphans[:5]}")
        return 1
    print(f"  counts: {kb_counts()}")
    return 0


__all__ = [
    "LoadResult",
    "PruneResult",
    "entry_params",
    "kb_counts",
    "load_entries",
    "orphaned_entries",
    "prune_entries",
]


if __name__ == "__main__":
    raise SystemExit(main())
