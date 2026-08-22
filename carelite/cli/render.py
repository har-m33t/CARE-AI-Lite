"""Rich rendering for the CARELite CLI.

Everything here is a pure function of the frozen `carelite.types` models
(`GuidanceResponse`, `RetrievalTrace`, `RetrievedItem`, `SafetyVerdict`) so it
can be exercised in tests without a terminal. `rich.console.Console` degrades
automatically when writing to a pipe or a narrow terminal (no ANSI, folded
text), which is what "graceful when piped" means here — we do not special
case it beyond using `Console`'s own detection and `overflow="fold"` /
`no_wrap=False` on any column that might carry long text.
"""

from __future__ import annotations

from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from carelite.types import (
    Condition,
    EvidenceTier,
    GuidanceResponse,
    RetrievalTrace,
    RetrievedItem,
    SafetyVerdict,
)

DISCLAIMER = (
    "CARELite AI is a research prototype. It is NOT clinical software and provides NO "
    "diagnostic or treatment advice. Guidance below is communication coaching only; use "
    "clinical judgment."
)

_TIER_STYLE = {
    EvidenceTier.STRONG: "bold green",
    EvidenceTier.MODERATE: "bold yellow",
    EvidenceTier.EMERGING: "bold dark_orange",
}


def tier_style(tier: EvidenceTier | None) -> str:
    if tier is None:
        return "dim"
    return _TIER_STYLE.get(tier, "white")


def tier_label(tier: EvidenceTier | None) -> Text:
    if tier is None:
        return Text("unrated", style="dim")
    return Text(tier.value, style=tier_style(tier))


def render_header(console: Console) -> None:
    console.print(Panel(DISCLAIMER, style="bold white on red", expand=True))


def render_safety(console: Console, verdict: SafetyVerdict | None, label: str) -> None:
    """Render one safety verdict. Silent when there is nothing to say."""
    if verdict is None:
        return
    if not verdict.allowed:
        body = Text()
        body.append(verdict.reason or "Blocked by the safety screen.", style="bold red")
        if verdict.flags:
            body.append(f"\nflags: {', '.join(verdict.flags)}", style="red")
        console.print(
            Panel(body, title=f"[bold red]BLOCKED — {label} safety[/]", border_style="red")
        )
        return
    # Allowed, but something is worth surfacing (PHI redaction, control flags, etc).
    if verdict.flags or verdict.phi_detected or verdict.red_flag:
        body = Text(verdict.reason or "", style="yellow")
        if verdict.redacted_text:
            body.append(f"\nredacted: {verdict.redacted_text}", style="yellow")
        if verdict.flags:
            body.append(f"\nflags: {', '.join(verdict.flags)}", style="yellow")
        console.print(
            Panel(body, title=f"[bold yellow]{label} safety note[/]", border_style="yellow")
        )


def _retrieved_table(items: list[RetrievedItem]) -> Table:
    """Compact one-row-per-item table. Kept to a handful of narrow columns
    plus one flexible, truncating (never letter-wrapping) citation column, so
    it still reads on an 80-column terminal."""
    table = Table(show_lines=False, expand=True, pad_edge=False)
    table.add_column("ref", style="cyan", no_wrap=True, width=11)
    table.add_column("theme", no_wrap=True, max_width=16)
    table.add_column("tier", no_wrap=True, width=9)
    table.add_column("score", justify="right", no_wrap=True, width=6)
    table.add_column("citation", no_wrap=True, overflow="ellipsis", ratio=1)

    for item in items:
        table.add_row(
            item.ref_id,
            item.theme.value if item.theme else "-",
            tier_label(item.evidence_tier),
            f"{item.score:.3f}",
            item.citation or "-",
        )
    return table


def _rank_table(items: list[RetrievedItem]) -> Table:
    """Per-source rank breakdown, shown separately from the main table (only
    in --explain / /why) so the main table never has to shrink to fit it."""
    table = Table(show_lines=False, expand=True, pad_edge=False, title="rank breakdown")
    table.add_column("ref", style="cyan", no_wrap=True, width=11)
    table.add_column("dense", justify="right", no_wrap=True)
    table.add_column("lexical", justify="right", no_wrap=True)
    table.add_column("graph hops", justify="right", no_wrap=True)
    table.add_column("rerank", justify="right", no_wrap=True)

    for item in items:
        table.add_row(
            item.ref_id,
            str(item.dense_rank) if item.dense_rank is not None else "-",
            str(item.lexical_rank) if item.lexical_rank is not None else "-",
            str(item.graph_hops) if item.graph_hops is not None else "-",
            f"{item.rerank_score:.3f}" if item.rerank_score is not None else "-",
        )
    return table


def render_evidence_panel(
    console: Console,
    trace: RetrievalTrace | None,
    condition: Condition,
    expanded: bool = False,
) -> None:
    if trace is None:
        console.print(
            Panel(
                f"No retrieval trace — condition {condition.value} does not retrieve "
                "(framework-only or bare-model condition). Guidance is not evidence-grounded.",
                title="evidence",
                border_style="dim",
            )
        )
        return

    header = Text()
    header.append(f"route: {trace.route.value}   ", style="bold")
    header.append(f"CRAG: {trace.crag_grade.value}   ", style="bold")
    if trace.latency_ms is not None:
        header.append(f"retrieval latency: {trace.latency_ms}ms", style="dim")
    console.print(header)

    if trace.fell_back_to_b:
        console.print(
            Panel(
                "Retrieval was not relevant enough to ground this response (CRAG grade: "
                f"{trace.crag_grade.value}). The guidance above falls back to the framework "
                "alone — treat it as UNSUPPORTED by retrieved evidence.",
                title="[bold white on dark_orange3] FELL BACK TO CONDITION B [/]",
                border_style="dark_orange3",
            )
        )

    if trace.retrieved:
        console.print(_retrieved_table(trace.retrieved))
    else:
        console.print(Text("no items retrieved", style="dim"))

    if expanded:
        if trace.retrieved:
            console.print(_rank_table(trace.retrieved))
        detail = Text()
        detail.append("queries:\n", style="bold")
        for q in trace.queries:
            detail.append(f"  - {q}\n")
        if trace.hyde_passage:
            detail.append("HyDE passage:\n", style="bold")
            detail.append(f"  {trace.hyde_passage}\n")
        console.print(Padding(detail, (0, 0, 0, 1)))


def render_turn(console: Console, response: GuidanceResponse, expanded: bool = False) -> None:
    render_safety(console, response.input_safety, "input")
    if response.input_safety is not None and not response.input_safety.allowed:
        # A blocked turn must explain itself and stop there — there is no
        # guidance text and no evidence to show.
        return

    title = f"guidance — condition {response.condition.value}"
    console.print(Panel(response.text or "(empty response)", title=title))

    render_safety(console, response.output_safety, "output")
    if response.output_safety is not None and not response.output_safety.allowed:
        return
    if response.self_check_passed is False:
        console.print(Text("self-check: FAILED", style="bold red"))

    render_evidence_panel(console, response.trace, response.condition, expanded=expanded)


def render_retrieval_probe(
    console: Console, trace: RetrievalTrace | None, query: str, expanded: bool = False
) -> None:
    console.print(Text(f"query: {query}", style="bold"))
    render_evidence_panel(console, trace, Condition.C, expanded=expanded)
