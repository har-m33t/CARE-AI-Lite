"""`carelite` — the Typer + Rich terminal app.

Every command builds a `GuidanceRequest`, calls `GuidanceEngine.guide()` via
`carelite.cli.engine.resolve_engine()`, and renders the `GuidanceResponse`
through `carelite.cli.render`. No command imports a concrete engine — that is
the seam `carelite-orchestrator` swaps in wave 3 without touching this file.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from carelite.cli import render
from carelite.cli.engine import resolve_engine
from carelite.types import Condition, GuidanceRequest, GuidanceResponse

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="CARELite AI — evidence-grounded guidance for clinician-patient communication. "
    "Research prototype: not clinical software, no diagnostic or treatment advice.",
)
db_app = typer.Typer(add_completion=False, help="Database readiness checks.")
app.add_typer(db_app, name="db")

console = Console()


def _blocked(response: GuidanceResponse) -> bool:
    if response.input_safety is not None and not response.input_safety.allowed:
        return True
    return response.output_safety is not None and not response.output_safety.allowed


@app.command()
def ask(
    utterance: Annotated[str, typer.Argument(help="Patient utterance to respond to.")],
    condition: Annotated[
        Condition, typer.Option("--condition", "-c", help="Experimental condition.")
    ] = Condition.C,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json", help="Emit the raw GuidanceResponse JSON instead of the rich panel."
        ),
    ] = False,
) -> None:
    """One-shot: get guidance for a single utterance."""
    engine = resolve_engine()
    request = GuidanceRequest(utterance=utterance, condition=condition)
    response = engine.guide(request)

    if json_output:
        typer.echo(response.model_dump_json(indent=2))
    else:
        render.render_header(console)
        render.render_turn(console, response)

    raise typer.Exit(code=3 if _blocked(response) else 0)


@app.command()
def retrieve(
    query: Annotated[str, typer.Argument(help="Query to probe retrieval with.")],
    explain: Annotated[
        bool,
        typer.Option(
            "--explain",
            help="Show full detail: expanded queries, HyDE passage, per-source rank breakdown.",
        ),
    ] = False,
) -> None:
    """Probe retrieval only: route, queries, fused scores, CRAG grade — no generation."""
    engine = resolve_engine()
    # Retrieval only happens under condition C; the probe forces it so there
    # is always a trace to show, regardless of the CLI's default condition.
    request = GuidanceRequest(utterance=query, condition=Condition.C)
    response = engine.guide(request)
    render.render_header(console)
    render.render_retrieval_probe(console, response.trace, query=query, expanded=explain)


@app.command()
def chat() -> None:
    """Interactive session. /quit to exit, /condition <A|A2|B|C|LC|D> to switch condition,
    /why to expand the last turn's evidence panel."""
    engine = resolve_engine()
    render.render_header(console)
    console.print(
        "[dim]/quit to exit  ·  /condition <A|A2|B|C|LC|D> to switch  ·  "
        "/why to expand the last turn's evidence[/dim]"
    )

    condition = Condition.C
    history: list[str] = []
    last_response: GuidanceResponse | None = None

    while True:
        try:
            raw = console.input("[bold cyan]patient> [/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        raw = raw.strip()
        if not raw:
            continue
        if raw in ("/quit", "/exit"):
            break
        if raw == "/why":
            if last_response is None:
                console.print("[yellow]no turn yet[/yellow]")
            else:
                render.render_evidence_panel(
                    console, last_response.trace, last_response.condition, expanded=True
                )
            continue
        if raw.startswith("/condition"):
            parts = raw.split(maxsplit=1)
            candidate = parts[1].strip().upper() if len(parts) == 2 else ""
            if candidate in Condition.__members__:
                condition = Condition[candidate]
                console.print(f"[green]condition set to {condition.value}[/green]")
            else:
                choices = ", ".join(c.value for c in Condition)
                console.print(f"[red]unknown condition '{candidate}'; choose from {choices}[/red]")
            continue
        if raw.startswith("/"):
            console.print(f"[red]unknown command: {raw}[/red]")
            continue

        request = GuidanceRequest(utterance=raw, condition=condition, history=list(history))
        response = engine.guide(request)
        render.render_turn(console, response)

        history.append(f"patient: {raw}")
        if response.text:
            history.append(f"assistant: {response.text}")
        last_response = response


@db_app.command("check")
def db_check() -> None:
    """Delegate to `carelite.db.check` — reports Postgres/pgvector readiness.

    Fails gracefully (prints a report, exits non-zero) when Postgres is not
    installed or not reachable; never raises out to the terminal.
    """
    from carelite.db.check import main as run_check

    exit_code = run_check()
    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
