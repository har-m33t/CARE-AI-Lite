"""The review digest: an optional aid, and the record of what nobody has checked.

This module was written as a required gate. `DECISIONS.md` D4 removed the gate
rather than tick it falsely, and the distinction matters more than it sounds.
The knowledge base's provenance claim is now, exactly: **LLM-assisted extraction
with automated verbatim-span validation, and no human verification.**
`human_verified` stays FALSE on every entry because that is true, and nothing in
this module may be used to make it look otherwise. `record_signoff` still works,
for whoever does a review later; it is a tool, not a step the pipeline is
waiting on.

So the digest's job changed. It no longer asks a reviewer to complete something.
It shows what was checked mechanically, states plainly what was not, and gives
anyone who wants to check the rest everything they need to. Per entry:

- the seven fields as they are stored,
- the verbatim span, marked up inside its **surrounding paragraph**, so a reader
  can see the sentence in context and judge whether the entry's finding is a
  fair reading of it rather than a true sentence bolted to an unrelated claim,
- the source citation and study design, so the derived tier can be argued with,
- whether the span relays somebody else's study,
- and a checkbox, unticked, that a later reviewer may use.

The context is what makes the document worth anything. A digest that printed
only the quote would let a reader confirm the quote exists — which the validator
has already proved mechanically — without ever reaching the thing a machine
cannot reach, which is whether the *entry* follows from the paper. That gap is
the whole of what D4 declines to claim, and the digest should point straight at
it rather than around it.

Sign-off, if it ever happens, round-trips through the digest file itself: tick
`- [x]`, save, and `record_signoff` sets `human_verified` for exactly those
entries. No second file to keep in sync, and the artifact recording the decision
is the same one that showed the evidence it was made on.
"""

from __future__ import annotations

import datetime as dt
import difflib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from carelite.config import REPO_ROOT
from carelite.db.connection import transaction
from carelite.eval.rubric.dimensions import DIMENSIONS
from carelite.kb.frameworks import FOUR_HABITS_COMPONENTS, NURSE_COMPONENTS
from carelite.kb.papers import PAPER_META, load_paper_texts
from carelite.kb.scope import LOW_OVERLAP_THRESHOLD, takeaway_span_overlap
from carelite.kb.spans import locate_span, surrounding_context

#: Ranking used only to say which way a tier correction went in the digest.
_TIER_ORDER = {"emerging": 0, "moderate": 1, "strong": 2}

DIGEST_PATH = REPO_ROOT / "knowledge_base" / "review" / "kb_review_digest.md"

_CHECKBOX_RE = re.compile(r"^\s*-\s*\[(?P<mark>[ xX])\]\s*`(?P<entry_id>kb-[a-z_]+-[0-9a-f]+)`")

_SELECT_ENTRIES_SQL = """
SELECT
    e.entry_id, e.theme, e.finding, e.practical_takeaway, e.example_behavior,
    e.evidence_tier, e.action_type, e.verbatim_span, e.encounter_phase,
    e.nurse_component, e.four_habits,
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


@dataclass(frozen=True)
class EntryAudit:
    """What the pipeline decided about an entry, for a reader to see.

    Three of the validator's behaviours change or stretch what the model
    produced, and a digest that hid them would be decoration:

    - `claimed_tier` is the evidence tier the extraction model asserted. The
      stored tier is derived from the source's recorded study design, so the
      two differ whenever the model misjudged the design — in either
      direction. A reader should see the model's judgment, not only the
      corrected value.
    - `second_hand` is set when the span relays a study that is not the paper
      it was quoted from. Those entries carry a capped tier for a different
      reason than everything else, and the digest says so where it shows them.
    - `span_via` is how much normalisation locating the quote required.
      ``"glued"`` means it matched only after spaces and hyphens were deleted
      from both sides — almost always a word the PDF extractor split across a
      column break, but the loosest match the validator makes and therefore
      the one most worth an eye.

    None of it is stored in `kb_entry`; the schema is frozen and this lane does
    not amend it. All of it is re-derived from the extraction cache at digest
    time, which keeps one source of truth instead of a sidecar file that drifts.
    """

    claimed_tier: str
    stored_tier: str
    span_via: str
    second_hand: str = ""

    @property
    def tier_corrected(self) -> bool:
        return self.claimed_tier != self.stored_tier


def build_audit() -> dict[str, EntryAudit]:
    """Re-validate the extraction cache to recover what the pipeline corrected.

    Returns an empty map when the cache is unavailable — a machine that has the
    database but not the extraction cache can still produce a digest, it just
    cannot show the model's original claim, and `render_digest` says so rather
    than implying nothing was corrected.
    """
    try:
        from carelite.kb.extract import CACHE_PATH, read_cache
        from carelite.kb.validate import validate_candidates

        candidates = [c for r in read_cache(CACHE_PATH) for c in r.candidates]
        if not candidates:
            return {}
        report = validate_candidates(candidates)
    except Exception:  # no cache, no papers on disk, no corpus extraction
        return {}

    return {
        e.entry_id: EntryAudit(
            claimed_tier=e.claimed_tier.value,
            stored_tier=e.entry.evidence_tier.value,
            span_via=e.span_match_via,
            second_hand=str(e.second_hand) if e.second_hand else "",
        )
        for e in report.accepted
    }


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
    nurse_component: list[str] = field(default_factory=list)
    four_habits: list[str] = field(default_factory=list)

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
            nurse_component=list(r["nurse_component"] or []),
            four_habits=list(r["four_habits"] or []),
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


#: Takeaway similarity above which two entries are worth a reader's attention.
#: Tuned against the loaded knowledge base: at 0.72 it surfaces a handful of
#: pairs, each the same paper making the same point through two different
#: sentences. Lower and it flags any two entries that share a theme vocabulary;
#: higher and it misses genuine restatements.
OVERLAP_THRESHOLD = 0.72

#: The threshold used *within* one (theme, paper) group. Lower than the global
#: one on purpose: two entries drawn from the same paper about the same theme
#: already share their subject, so the question is not "are these related" —
#: they are, by construction — but "do these say the same thing".
#:
#: Set at 0.47, and the calibration is worth stating because the first value
#: repeated the very defect this section exists to fix. At 0.58 it reported 17
#: of 114 entries as clustered. The case that exposed it: *"Brief the
#: interpreter on the goals and specific content of the conversation before the
#: patient enters the room"* and *"provide the interpreter with advanced
#: preparation and specific context before the encounter"* — plainly one piece
#: of advice, and they score **0.478**, so 0.58 called them independent. Reading
#: the groups that appear as the threshold drops confirms the same thing at
#: scale: at 0.47 the `teach_back` cluster grows from 6 entries to 10, which
#: matches what reading them shows, because nearly every Talevski teach_back
#: entry says "use teach-back to confirm the patient understood their discharge
#: instructions". The honest figure is 38 of 114, not 17.
#:
#: **Over-grouping is the safer error here and the threshold is set accordingly.**
#: Nothing in this section rejects an entry; it tells a reader which entries are
#: not independent of each other. A cluster a reader disagrees with costs them a
#: moment; a restatement this misses goes into the write-up as convergent
#: evidence, which is the failure being corrected.
CLUSTER_THRESHOLD = 0.47

#: A theme drawing this share or more of its entries from a single paper is
#: single-source in practice, whatever its distinct-paper count says.
DOMINANCE_THRESHOLD = 0.60


def overlapping_pairs(
    rows: Sequence[ReviewRow], *, threshold: float = OVERLAP_THRESHOLD
) -> list[tuple[ReviewRow, ReviewRow, float]]:
    """Entry pairs whose takeaways say close to the same thing, anywhere in the base.

    Kept for cross-paper restatement, which `redundancy_clusters` will not see:
    two different papers reaching the same advice is a genuinely different
    situation from one paper being quoted nine times, and it is the one case
    where the duplication does not overstate the evidence.
    """
    out: list[tuple[ReviewRow, ReviewRow, float]] = []
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            ratio = difflib.SequenceMatcher(
                None, a.practical_takeaway.lower(), b.practical_takeaway.lower()
            ).ratio()
            if ratio >= threshold:
                out.append((a, b, ratio))
    return sorted(out, key=lambda t: -t[2])


#: Numbers of the shape a paper reports a result in: `6.99`, `4.30`, `82.1%`.
#: Two entries quoting the same statistics block are quoting one result, however
#: differently their takeaways are worded.
_STATISTIC = re.compile(r"(?<![A-Za-z0-9.])\d+\.\d+")


def _statistics(row: ReviewRow) -> frozenset[str]:
    return frozenset(_STATISTIC.findall(row.verbatim_span))


@dataclass(frozen=True)
class RedundancyCluster:
    """Entries from one paper, in one theme, that are making one point."""

    theme: str
    paper_id: str
    rows: tuple[ReviewRow, ...]
    shared_statistics: frozenset[str] = frozenset()

    @property
    def size(self) -> int:
        return len(self.rows)


def redundancy_clusters(
    rows: Sequence[ReviewRow], *, threshold: float = CLUSTER_THRESHOLD
) -> list[RedundancyCluster]:
    """Group entries that restate each other, within one paper and one theme.

    A pairwise scan was the first version of this and it under-reported the
    problem badly. Pairs are what you find when the duplication is a pair; the
    actual shape in this corpus is a *cluster* — ten `activation_sdm` entries
    from one motivational-interviewing meta-analysis, nine of which amount to
    "use motivational interviewing", three of them quoting the same statistics
    block. A pairwise check at a threshold high enough not to drown in
    theme vocabulary sees a few of those ten and calls it six pairs across the
    whole base, which reads as a minor tidying job rather than as one paper
    counted nine times.

    Clustering inside a (theme, paper) group fixes the comparison. Two entries
    from one paper about one theme already share their subject, so a lower
    threshold is meaningful there and misleading globally. Single-linkage:
    entries chain through each other, because A restating B and B restating C
    is one point made three times whatever A and C look like side by side.

    A shared statistics block joins a cluster regardless of similarity — two
    takeaways can be worded quite differently and still be quoting the same
    `MD = 6.99`.
    """
    groups: dict[tuple[str, str], list[ReviewRow]] = {}
    for row in rows:
        groups.setdefault((row.theme, row.primary_paper), []).append(row)

    clusters: list[RedundancyCluster] = []
    for (theme, paper_id), members in groups.items():
        if len(members) < 2:
            continue
        parent = list(range(len(members)))

        def find(i: int, parent: list[int] = parent) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int, parent: list[int] = parent) -> None:
            ri, rj = find(i, parent), find(j, parent)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        stats = [_statistics(m) for m in members]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                ratio = difflib.SequenceMatcher(
                    None,
                    members[i].practical_takeaway.lower(),
                    members[j].practical_takeaway.lower(),
                ).ratio()
                if ratio >= threshold or (stats[i] and stats[i] & stats[j]):
                    union(i, j)

        buckets: dict[int, list[int]] = {}
        for i in range(len(members)):
            buckets.setdefault(find(i), []).append(i)
        for indices in buckets.values():
            if len(indices) < 2:
                continue
            shared: frozenset[str] = frozenset()
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    shared |= stats[indices[a]] & stats[indices[b]]
            clusters.append(
                RedundancyCluster(
                    theme=theme,
                    paper_id=paper_id,
                    rows=tuple(members[i] for i in indices),
                    shared_statistics=shared,
                )
            )
    return sorted(clusters, key=lambda c: (-c.size, c.theme, c.paper_id))


@dataclass(frozen=True)
class ThemeCoverage:
    """How independent a theme's entries actually are."""

    theme: str
    n_entries: int
    n_papers: int
    dominant_paper: str
    dominant_count: int

    @property
    def dominant_share(self) -> float:
        return self.dominant_count / self.n_entries if self.n_entries else 0.0

    @property
    def single_source_in_effect(self) -> bool:
        """One paper carries most of the theme, even though others appear in it.

        The distinct-paper count is the number the coverage table used to report
        on its own, and it flatters a theme badly: `teach_back` at 17 entries
        over 4 papers looks like a four-paper theme and is in practice one
        paper's, with three entries from elsewhere. A write-up that treats those
        17 as convergent evidence is wrong about its own knowledge base.
        """
        return self.n_entries > 1 and self.dominant_share >= DOMINANCE_THRESHOLD


def theme_coverage(rows: Sequence[ReviewRow]) -> list[ThemeCoverage]:
    """Per-theme entry counts, source counts, and how concentrated the sources are."""
    by_theme: dict[str, list[ReviewRow]] = {}
    for row in rows:
        by_theme.setdefault(row.theme, []).append(row)

    out: list[ThemeCoverage] = []
    for theme, theme_rows in sorted(by_theme.items()):
        counts: dict[str, int] = {}
        for row in theme_rows:
            counts[row.primary_paper] = counts.get(row.primary_paper, 0) + 1
        dominant, dominant_count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        out.append(
            ThemeCoverage(
                theme=theme,
                n_entries=len(theme_rows),
                n_papers=len(counts),
                dominant_paper=dominant,
                dominant_count=dominant_count,
            )
        )
    return out


def render_digest(
    rows: Sequence[ReviewRow],
    *,
    generated_at: dt.datetime | None = None,
    audit: Mapping[str, EntryAudit] | None = None,
) -> str:
    """Render the full digest as Markdown."""
    stamp = (generated_at or dt.datetime.now(dt.UTC)).strftime("%Y-%m-%d %H:%M UTC")
    audit = {} if audit is None else audit
    total = len(rows)
    verified = sum(1 for r in rows if r.human_verified)
    audits = [audit.get(r.entry_id) for r in rows]
    corrected = sum(1 for x in audits if x and x.tier_corrected)
    glued = sum(1 for x in audits if x and x.span_via == "glued")
    second_hand = sum(1 for x in audits if x and x.second_hand)

    by_theme: dict[str, list[ReviewRow]] = {}
    for row in rows:
        by_theme.setdefault(row.theme, []).append(row)

    out: list[str] = [
        "# Knowledge Base Digest",
        "",
        f"Generated {stamp}. {total} entr(ies) loaded, {verified} carrying a recorded review.",
        "",
        "## What has been checked, and what has not",
        "",
        "This document is **not a gate**. `DECISIONS.md` D4 dropped the human-verification",
        "requirement rather than tick it without anyone having read anything, so the claim",
        "the knowledge base makes is now exactly this: **LLM-assisted extraction with",
        "automated verbatim-span validation, and no human verification.** `human_verified`",
        "is FALSE on every entry because that is the true record, not because a review is",
        "pending.",
        "",
        "**What is mechanically established.** Every entry's quoted span was located in the",
        "extracted text of the paper it cites, and what is stored is a literal slice of that",
        "source rather than the model's rendering of it — the marked text in each context",
        "block is the document's own characters. Matching folds only rendering differences:",
        "ligatures, quotation glyphs, dashes, hyphenated line breaks, whitespace, case, and",
        "(where flagged) the spaces and hyphens a PDF extractor invents at a column break.",
        "Candidates were rejected for a fabricated span, a span too short to carry evidence,",
        "a takeaway that named no action, a subject matter `TAXONOMY.md` excludes, or a",
        "finding the span does not report; the rejection counts are enumerated by",
        "`carelite.kb.validate`, and the fabrication rate was measured rather than estimated.",
        "",
        "**What is not established, by anyone.** Whether each *finding* actually follows",
        "from its *span* — whether a true sentence has been attached to a claim it does not",
        "support. No automated check can do this and no person has done it. Any result that",
        "depends on knowledge base quality inherits that limitation.",
        "",
        "So the entries below are worth reading in the same order of suspicion the pipeline",
        "would: the finding against the quoted sentence in its context, then the takeaway",
        "against the finding, then the tier against the design.",
        "",
        "## What the pipeline decided, and why you are being told",
        "",
        "**Evidence tier is derived from the study design, not from the model, on all",
        f"{total} entries; {corrected} differ from what the model claimed.** The extraction",
        "model judges a tier from the passage in front of it, and it misses in both",
        "directions — calling a survey `strong` because the result sounded confident, or a",
        "randomised trial `emerging` because the sentence was hedged. An earlier version of",
        "this pipeline only capped overclaims, which left four papers carrying entries at",
        "more than one tier and one carrying entries at all three. `README.md` defines",
        "evidence strength as a property of the source, so that was incoherent: two entries",
        "citing one paper cannot honestly carry different strengths. The tier now comes from",
        "the recorded design outright. Each corrected entry prints the model's own claim",
        "beside the stored value, so nothing is laundered — **if you think a correction went",
        "the wrong way, that is a finding about the design label, and the design label is in",
        "`carelite/kb/papers.py` where it can be argued with.**",
        "",
        f"**{second_hand} entr(ies) quote a span that relays somebody else's study.** A",
        "systematic review's summary of a trial is a legitimate quotation, but that trial is",
        "not in this corpus, and stamping the entry `strong` because the *review* is strong",
        "asserts we hold evidence we do not hold. Those entries are marked `second-hand`",
        "below and their tier is capped at what this corpus can vouch for: `moderate` when a",
        "systematic review or meta-analysis is doing the relaying, `emerging` otherwise.",
        "This is the one place two entries from one paper may honestly differ in tier.",
        "",
        f"**{glued} entr(ies) matched only after layout-glue normalisation.** Their quote was",
        "located only once spaces and hyphens were deleted from both sides, which almost",
        "always means the PDF text extractor split a word across a column break (`show ing`)",
        "or joined one across a line break (`healthrelated`). These are marked. The stored",
        "span is still the paper's own text, artefact and all — but they are the loosest",
        "matches the validator makes, so they are worth an eye first.",
        "",
        "## If you do review these",
        "",
        "Nothing requires it, and nothing downstream is waiting on it. If you want the",
        "record to show that a person read an entry, tick `- [x]` beside it, save this file,",
        "and run:",
        "",
        "```",
        "python -m carelite.kb.review --signoff --reviewer <your-name>",
        "```",
        "",
        "Only ticked entries change. Unticked entries stay `human_verified = FALSE`, which",
        "is what they should be if nobody has read them. Signing off deletes nothing, so an",
        "entry you would reject can be discussed rather than vanishing.",
        "",
        "## Coverage",
        "",
        "| Theme | Entries | Source papers | Largest single source |",
        "|---|---|---|---|",
    ]

    coverage = theme_coverage(rows)
    for cov in coverage:
        if cov.n_papers == 1 and cov.n_entries > 1:
            flag = " **single-source**"
        elif cov.single_source_in_effect:
            flag = " **single-source in effect**"
        else:
            flag = ""
        out.append(
            f"| {cov.theme} | {cov.n_entries} | {cov.n_papers} | "
            f"{cov.dominant_count}/{cov.n_entries} ({cov.dominant_share:.0%}) "
            f"`{cov.dominant_paper}`{flag} |"
        )

    effective = [c for c in coverage if c.single_source_in_effect and c.n_papers > 1]
    out += [
        "",
        "A theme marked **single-source** draws every entry from one paper. A theme marked",
        "**single-source in effect** draws most of them from one paper while listing several",
        f"— the distinct-paper count flatters it. {'Here that is: ' if effective else ''}"
        + (
            "; ".join(
                f"`{c.theme}` ({c.dominant_count} of {c.n_entries} from `{c.dominant_paper}`)"
                for c in effective
            )
            + "."
            if effective
            else "No theme is currently in that position."
        ),
        "Entries in either kind of theme are individually evidenced but are not independent",
        "of each other, and the write-up must not present them as convergent evidence.",
        "",
    ]

    out += _framework_coverage_section(rows)

    clusters = redundancy_clusters(rows)
    if clusters:
        clustered = sum(c.size for c in clusters)
        out += [
            "## Entries that restate each other",
            "",
            f"{clustered} entr(ies) fall into {len(clusters)} cluster(s) where one paper "
            "is making one",
            "point through several quoted sentences. Every entry in them is valid",
            "— real span, real source — but they are not independent evidence, and counting",
            "them as separate support overstates the knowledge base. Clustering is done within",
            "a single (theme, paper) group, because two entries from one paper about one theme",
            "already share their subject; the question is whether they share their *claim*.",
            "Entries quoting the same statistics block are grouped whatever their wording.",
            "",
        ]
        for cluster in clusters:
            meta = PAPER_META.get(cluster.paper_id)
            citation = meta.short_citation if meta else cluster.paper_id
            stats = (
                f" — all quoting {', '.join(sorted(cluster.shared_statistics))}"
                if cluster.shared_statistics
                else ""
            )
            out += [
                f"- **{cluster.size} entries**, `{cluster.theme}`, {citation}{stats}",
            ]
            out += [f"  - `{r.entry_id}` — {r.practical_takeaway}" for r in cluster.rows]
        out.append("")

    cross_paper = [
        (a, b, ratio)
        for a, b, ratio in overlapping_pairs(rows)
        if a.primary_paper != b.primary_paper
    ]
    if cross_paper:
        out += [
            "## Near-duplicate takeaways across different papers",
            "",
            "Unlike the clusters above, these are two papers arriving at the same advice, which",
            "is convergent evidence rather than double-counting. Listed so a reader can see the",
            "wording is nearly identical and decide whether both entries earn their place.",
            "",
        ]
        for left, right, ratio in cross_paper:
            out += [
                f"- {ratio:.0%} similar — `{left.entry_id}` and `{right.entry_id}`",
                f"  - {left.practical_takeaway}",
                f"  - {right.practical_takeaway}",
            ]
        out.append("")

    low_overlap = [
        (row, takeaway_span_overlap(row.practical_takeaway, row.verbatim_span)) for row in rows
    ]
    low_overlap = [(r, o) for r, o in low_overlap if o < LOW_OVERLAP_THRESHOLD]
    if low_overlap:
        out += [
            "## Takeaways that share little vocabulary with their span",
            "",
            f"{len(low_overlap)} entr(ies) whose takeaway repeats almost none of the words in",
            "the sentence it cites. **This is a pointer, not a defect** — a good takeaway often",
            "paraphrases rather than echoes, and several entries here are fine. It is listed",
            "because the one thing nobody has checked is whether a takeaway is *supported by*",
            "its span or merely sits next to it, and low vocabulary overlap is the cheapest",
            "place to start looking.",
            "",
        ]
        for row, overlap in sorted(low_overlap, key=lambda t: t[1]):
            out.append(f"- {overlap:.0%} — `{row.entry_id}` — {row.practical_takeaway}")
        out.append("")

    for theme, theme_rows in sorted(by_theme.items()):
        out += [f"## {theme}", ""]
        for row in theme_rows:
            meta = PAPER_META.get(row.primary_paper)
            citation = meta.short_citation if meta else row.primary_paper
            design = meta.design if meta else "design not recorded"
            mark = "x" if row.human_verified else " "
            phases = ", ".join(row.encounter_phase) or "-"
            entry_audit = audit.get(row.entry_id)

            if entry_audit and entry_audit.tier_corrected:
                direction = (
                    "up"
                    if _TIER_ORDER[entry_audit.claimed_tier] < _TIER_ORDER[row.evidence_tier]
                    else "down"
                )
                tier_line = (
                    f"  `{row.primary_paper}` — {design}  \n"
                    f"  Evidence tier **{row.evidence_tier}**, derived from that design "
                    f"— corrected {direction} from the model's claim of "
                    f"*{entry_audit.claimed_tier}*"
                )
            else:
                tier_line = (
                    f"  `{row.primary_paper}` — {design} — evidence tier "
                    f"**{row.evidence_tier}**, derived from that design"
                )
            if entry_audit and entry_audit.second_hand:
                tier_line += (
                    f"  \n  **Second-hand:** the span {entry_audit.second_hand}. The evidence "
                    "belongs to a study outside this corpus, so the tier is capped at what "
                    "this corpus can vouch for rather than set by the design above."
                )

            out += [
                f"- [{mark}] `{row.entry_id}`",
                "",
                f"  **Source.** {citation}  ",
                tier_line,
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
            if entry_audit and entry_audit.span_via == "glued":
                out += [
                    "  > _Located only after layout-glue normalisation: the extracted text "
                    "splits or joins a word that the printed page does not. Check the marked "
                    "quote reads as a sentence._",
                    "",
                ]

    return "\n".join(out).rstrip() + "\n"


def _framework_coverage_section(rows: Sequence[ReviewRow]) -> list[str]:
    """Which rubric components the knowledge base actually grounds, zeros included.

    This is the behavior-to-framework mapping reported as coverage, and the
    zeros are the reason it is worth printing. A component with no entries says
    the corpus does not prescribe that move, which is a statement about the
    evidence base and belongs in the write-up next to the theme counts. It is
    not a defect in the mapping and must not be read as one.
    """
    nurse_counts = dict.fromkeys(NURSE_COMPONENTS, 0)
    habit_counts = dict.fromkeys(FOUR_HABITS_COMPONENTS, 0)
    for row in rows:
        for key in row.nurse_component:
            nurse_counts[key] = nurse_counts.get(key, 0) + 1
        for key in row.four_habits:
            habit_counts[key] = habit_counts.get(key, 0) + 1

    unmapped = [r for r in rows if not r.nurse_component and not r.four_habits]
    empty = [k for k, n in {**nurse_counts, **habit_counts}.items() if n == 0]

    out = [
        "## Framework coverage",
        "",
        "Which of the eleven scored dimensions each entry instantiates, derived by",
        "`carelite.kb.frameworks` from the act the entry prescribes — its takeaway and example",
        "behaviour — and never from its finding or its span, which say what a study *measured*.",
        "The patterns are anchored to the definitions in `carelite.eval.rubric.dimensions`, so",
        "any single assignment can be checked by reading the entry beside the definition.",
        "",
        "| Framework | Component | Entries |",
        "|---|---|---|",
    ]
    for key, count in nurse_counts.items():
        label = DIMENSIONS[key].label if key in DIMENSIONS else key
        out.append(f"| NURSE | `{key}` {label} | {count} |")
    for key, count in habit_counts.items():
        label = DIMENSIONS[key].label if key in DIMENSIONS else key
        out.append(f"| Four Habits | `{key}` {label} | {count} |")

    out += [
        "",
        f"**{len(unmapped)} of {len(rows)} entries instantiate none of the nine.** That is not a",
        "backlog to be worked off. Plenty of well-evidenced communication advice — request an",
        "in-person interpreter, check your own assumptions about this patient — is simply not one",
        "of the nine moves the rubric scores, and an entry forced into a component it does not",
        "perform would put a false edge in the graph and a false row in the coverage table.",
        "",
    ]
    if empty:
        named = ", ".join(f"`{k}`" for k in empty)
        out += [
            f"**{named} have no entries at all.** This corpus does not prescribe those moves. The",
            "NURSE Respecting move — crediting the patient for something specific they have done or",
            "endured — and the Supporting move — a partnership statement made concrete with who does",
            "what and how to reach someone — appear in the rubric because the source literature",
            "describes them, but nothing in these 33 papers turns either into a finding with a",
            "quotable span and an actionable takeaway. **Those are results about the evidence base,",
            "not gaps in this mapping**, and the write-up should report them as such: the judge",
            "scores dimensions the knowledge base cannot ground, so a system built on these entries",
            "has no evidential basis for two of the eleven things it is measured on.",
            "",
        ]

    equity_flagged = [r for r in rows if r.equity_relevant]
    equity_theme = [r for r in rows if r.theme == "equity"]
    cross = [r for r in equity_flagged if r.theme != "equity"]
    out += [
        f"**Equity reaches further than its theme.** {len(equity_flagged)} entries are flagged",
        f"`equity_relevant` while the `equity` theme itself holds {len(equity_theme)};",
        f"{len(cross)} of them sit under another theme "
        f"({', '.join(sorted({r.theme for r in cross}))}).",
        "That asymmetry is the design working: an interpreter finding is a `plain_language` entry",
        "that is also an equity one, and it is reachable from both directions. Reporting only the",
        "theme count understates what the base holds on equity; reporting only the flag would",
        "overstate how much of it is *about* a disparity. Both numbers belong in the write-up.",
        "",
    ]
    return out


def write_digest(path: Path | str = DIGEST_PATH, rows: Sequence[ReviewRow] | None = None) -> Path:
    """Render and write the digest, preserving existing tick marks via `human_verified`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows) if rows is not None else fetch_review_rows()
    path.write_text(render_digest(rows, audit=build_audit()), encoding="utf-8")
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
    "CLUSTER_THRESHOLD",
    "DIGEST_PATH",
    "DOMINANCE_THRESHOLD",
    "EntryAudit",
    "RedundancyCluster",
    "ReviewRow",
    "ThemeCoverage",
    "apply_signoff",
    "build_audit",
    "fetch_review_rows",
    "overlapping_pairs",
    "parse_signoff",
    "record_signoff",
    "redundancy_clusters",
    "render_digest",
    "theme_coverage",
    "write_digest",
]


if __name__ == "__main__":
    raise SystemExit(main())
