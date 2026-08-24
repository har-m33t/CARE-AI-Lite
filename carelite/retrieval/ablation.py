"""carelite.retrieval.ablation — the R0-R9 harness and the retrieval-quality gate.

Runs the same `pipeline.retrieve_detailed` function once per ablation row,
varying only the `RetrievalFlags`, and emits the table. Because every row goes
through one code path, a difference between two rows is attributable to the
flag that changed and not to two separately-written pipelines that happen to
disagree.

    python -m carelite.retrieval.ablation --max-turns 8

**The ladder.** R0 is dense-only over the raw utterance; each row adds one
component; R9 is the full stack. R8 is the deliberate exception — it is R9
*minus* CRAG, so the pair (R8, R9) isolates what the gate is worth, which is
the number the study most needs (see `crag.py`). LC is the long-context
baseline v3 §3 requires be reported: no retrieval at all, whole corpus
stuffed into the window.

**Scenarios come from the train split only.** `scenarios/holdout.lock` freezes
60 of the 100 scenarios for the actual experiment. Measuring retrieval
configurations against holdout turns and then reporting holdout results would
be tuning on the test set, so `default_turns()` reads `train_scenarios()` and
nothing else.

**Context precision.** The gate is "Ragas context precision > 0.7". The
`ragas` package is not a project dependency, so this module implements the
metric directly rather than importing it: `LLMContextPrecisionWithoutReference`
is, for one turn,

    CP = sum_k [ precision@k * v_k ] / max(1, sum_k v_k)

with `v_k` the judged relevance of the k-th retrieved item and `precision@k`
the fraction of the first k items that are relevant. Verdicts come from the
judge-family model, one cached call per (turn, passage) pair, so the many
passages that recur across rows are judged once. This is the standard formula
computed over our own verdicts — an equivalent implementation, not a call into
Ragas, and it is labelled that way in the emitted table so nobody reports it
as a `ragas` number.

**A turn that correctly retrieves nothing is not a precision failure.** When
CRAG falls back, `retrieved` is empty by design. Scoring that as 0.0 would
punish exactly the behaviour the gate exists to produce, and R9 would score
below R8 for doing the right thing. Such turns are therefore excluded from
the precision mean and counted separately in the `n_scored` column, so the
reader can see how many turns each figure rests on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from carelite.config import get_settings
from carelite.retrieval.flags import PRESETS, RetrievalFlags, preset
from carelite.retrieval.pipeline import RetrievalResult, retrieve_detailed
from carelite.types import Route

__all__ = [
    "ABLATION_ORDER",
    "AblationRow",
    "context_precision",
    "default_turns",
    "format_markdown",
    "long_context_stats",
    "main",
    "run_ablation",
    "run_row",
]

#: Emission order for the table. R8 sits before R9 so the gate's contribution
#: reads as the last delta.
ABLATION_ORDER: tuple[str, ...] = (
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "R7",
    "R8",
    "R9",
    "LC",
)

#: The gate from the lane brief.
CONTEXT_PRECISION_GATE = 0.7


@dataclass
class AblationRow:
    """One row of the emitted table."""

    label: str
    note: str
    config: str
    n_turns: int = 0
    n_scored: int = 0
    mean_retrieved: float = 0.0
    mean_latency_ms: float = 0.0
    fallback_rate: float = 0.0
    skipped_rate: float = 0.0
    context_precision: float | None = None
    routes: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def gate_passed(self) -> bool | None:
        if self.context_precision is None:
            return None
        return self.context_precision > CONTEXT_PRECISION_GATE

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "note": self.note,
            "config": self.config,
            "n_turns": self.n_turns,
            "n_scored": self.n_scored,
            "mean_retrieved": round(self.mean_retrieved, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "fallback_rate": round(self.fallback_rate, 3),
            "skipped_rate": round(self.skipped_rate, 3),
            "context_precision": (
                None if self.context_precision is None else round(self.context_precision, 3)
            ),
            "gate_passed": self.gate_passed,
            "routes": self.routes,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Scenario turns
# ---------------------------------------------------------------------------


def default_turns(limit: int | None = None) -> list[str]:
    """Patient turns from the **train** split. See the module docstring."""
    from carelite.scenarios.bank import train_scenarios

    turns = [s.text for s in train_scenarios()]
    return turns[:limit] if limit else turns


#: Turns the corpus provably cannot address. Included in every ablation run
#: because a table computed only over on-domain turns cannot show what CRAG
#: does — the gate's entire job is visible only when something should be
#: rejected. The `fallback_rate` column is the number to read here.
OFF_DOMAIN_TURNS: tuple[str, ...] = (
    "How do I replace the oil filter on a 2003 Honda Civic?",
    "What is the tax treatment of a Roth IRA conversion?",
    "What were the main causes of the fall of the Western Roman Empire?",
)


# ---------------------------------------------------------------------------
# Context precision
# ---------------------------------------------------------------------------

RELEVANCE_SYSTEM = """You judge whether one reference passage is useful for
responding to one patient turn from a clinical encounter.

The passage comes from a corpus of peer-reviewed papers on clinician-patient
communication. Judge usefulness for THIS turn specifically, not general quality.
A passage that is sound communication advice but unrelated to what this patient
said is not useful. If the patient turn is not about clinical communication at
all, no passage from this corpus is useful.

Reply with JSON only: {"useful": true|false}"""

_RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {"useful": {"type": "boolean"}},
    "required": ["useful"],
}


def _judge_relevance(client: Any, utterance: str, passage: str) -> bool | None:
    result = client.chat(
        system=RELEVANCE_SYSTEM,
        task="Is the passage above useful for responding to this patient turn? JSON only.",
        utterance=utterance,
        extra_untrusted=[("PASSAGE", passage)],
        json_schema=_RELEVANCE_SCHEMA,
        num_predict=200,
    )
    if result is None or not result.text:
        return None
    try:
        return bool(json.loads(result.text)["useful"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def context_precision(
    utterance: str,
    passages: Sequence[str],
    *,
    client: Any,
) -> float | None:
    """`LLMContextPrecisionWithoutReference` for one turn. `None` if unjudgeable.

    Order matters: the metric rewards putting the relevant passages first,
    which is exactly what the reranker is supposed to do, so it is the right
    metric for distinguishing R4 from R5.
    """
    if not passages:
        return None
    verdicts: list[bool] = []
    for passage in passages:
        verdict = _judge_relevance(client, utterance, passage)
        if verdict is None:
            return None
        verdicts.append(verdict)

    total_relevant = sum(verdicts)
    if total_relevant == 0:
        return 0.0
    running = 0
    accumulated = 0.0
    for k, is_relevant in enumerate(verdicts, start=1):
        if is_relevant:
            running += 1
            accumulated += running / k
    return accumulated / total_relevant


# ---------------------------------------------------------------------------
# Long context baseline
# ---------------------------------------------------------------------------


def long_context_stats() -> dict[str, Any]:
    """Corpus-stuffing budget for condition LC.

    Context precision is deliberately **not** computed for LC. LC's "retrieved
    set" is the entire corpus, so its precision is (relevant chunks)/(all
    chunks) — a number that would require hand-labelling all 475 chunks per
    turn to obtain honestly, and which would in any case be near zero by
    construction rather than by measurement. The reviewer-relevant fact about
    LC is whether the corpus fits in the window at all, so that is what this
    reports.
    """
    from carelite.db.connection import fetch_one

    row = fetch_one("SELECT count(*) AS n, coalesce(sum(length(text)), 0) AS chars FROM chunk")
    n_chunks = int(row["n"]) if row else 0
    chars = int(row["chars"]) if row else 0
    # ~4 characters per token is the usual English approximation; this is a
    # budget check, not an exact count.
    est_tokens = chars // 4
    window = get_settings().models.long_context.context_window
    return {
        "n_chunks": n_chunks,
        "est_tokens": est_tokens,
        "context_window": window,
        "fits": est_tokens < window,
        "utilisation": round(est_tokens / window, 3) if window else None,
    }


# ---------------------------------------------------------------------------
# Running a row
# ---------------------------------------------------------------------------


def run_row(
    flags: RetrievalFlags,
    turns: Sequence[str],
    *,
    embedder: Any = None,
    generator: Any = None,
    grader_client: Any = None,
    reranker: Any = None,
    precision_client: Any = None,
    score_precision: bool = True,
) -> AblationRow:
    """Run one ablation configuration over every turn."""
    row = AblationRow(label=flags.label or "?", note=flags.note, config=flags.summary())

    if flags.long_context:
        stats = long_context_stats()
        row.n_turns = len(turns)
        row.mean_retrieved = float(stats["n_chunks"])
        row.notes.append(
            f"no retrieval; {stats['n_chunks']} chunks ~{stats['est_tokens']:,} tokens "
            f"into a {stats['context_window']:,}-token window "
            f"({'fits' if stats['fits'] else 'DOES NOT FIT'}, "
            f"{stats['utilisation']:.1%} utilisation)"
        )
        row.notes.append("context precision not computed for LC — see long_context_stats()")
        return row

    results: list[RetrievalResult] = []
    latencies: list[float] = []
    for turn in turns:
        started = time.monotonic()
        result = retrieve_detailed(
            turn,
            flags=flags,
            embedder=embedder,
            generator=generator,
            grader_client=grader_client,
            reranker=reranker,
        )
        latencies.append((time.monotonic() - started) * 1000)
        results.append(result)

    row.n_turns = len(results)
    row.mean_retrieved = _mean([len(r.trace.retrieved) for r in results])
    row.mean_latency_ms = _mean(latencies)
    row.fallback_rate = _mean([1.0 if r.trace.fell_back_to_b else 0.0 for r in results])
    row.skipped_rate = _mean(
        [1.0 if r.trace.route is Route.EMOTIONAL_ONLY else 0.0 for r in results]
    )
    counts: dict[str, int] = {}
    for r in results:
        counts[r.trace.route.value] = counts.get(r.trace.route.value, 0) + 1
    row.routes = counts

    unavailable = {n for r in results for n in r.leg_notes if "unavailable" in n}
    row.notes.extend(sorted(unavailable))

    if score_precision and precision_client is not None:
        scores: list[float] = []
        for turn, result in zip(turns, results, strict=True):
            if not result.trace.retrieved:
                # Correctly retrieving nothing is not a precision failure.
                continue
            score = context_precision(
                turn, [i.text for i in result.trace.retrieved], client=precision_client
            )
            if score is not None:
                scores.append(score)
        row.n_scored = len(scores)
        row.context_precision = _mean(scores) if scores else None

    return row


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_ablation(
    turns: Sequence[str] | None = None,
    *,
    rows: Sequence[str] = ABLATION_ORDER,
    max_turns: int | None = 8,
    score_precision: bool = True,
    include_off_domain: bool = True,
) -> list[AblationRow]:
    """Run the whole ladder. Shares one embedder, one loaded cross-encoder and
    one judge client across every row, so the model load cost is paid once."""
    if turns is None:
        turns = default_turns(limit=max_turns)
    if include_off_domain:
        turns = [*turns, *OFF_DOMAIN_TURNS]

    from carelite.index.embed import OllamaEmbedder
    from carelite.retrieval.llm import LLMClient

    settings = get_settings()
    embedder = OllamaEmbedder()
    generator = LLMClient()
    judge = LLMClient(model_tag=settings.models.judge.tag)

    # One reranker for every row that reranks; rows that don't never touch it,
    # so torch is not imported unless some row in `rows` sets the flag.
    reranker = None
    if any(PRESETS[name].rerank for name in rows if name in PRESETS):
        from carelite.retrieval.rerank import get_reranker

        reranker = get_reranker()

    out: list[AblationRow] = []
    try:
        for name in rows:
            flags = preset(name)
            out.append(
                run_row(
                    flags,
                    turns,
                    embedder=embedder,
                    generator=generator,
                    grader_client=judge,
                    reranker=reranker,
                    precision_client=judge,
                    score_precision=score_precision,
                )
            )
    finally:
        embedder.close()
        generator.close()
        judge.close()
    return out


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def format_markdown(rows: Sequence[AblationRow]) -> str:
    header = (
        "| row | config | turns | n_ret | fallback | skipped | "
        "ctx precision (Ragas-equiv) | n_scored | latency ms |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for r in rows:
        cp = "n/a" if r.context_precision is None else f"{r.context_precision:.3f}"
        if r.gate_passed is True:
            cp += " PASS"
        elif r.gate_passed is False:
            cp += " FAIL"
        lines.append(
            f"| {r.label} | {r.config} | {r.n_turns} | {r.mean_retrieved:.1f} | "
            f"{r.fallback_rate:.0%} | {r.skipped_rate:.0%} | {cp} | {r.n_scored} | "
            f"{r.mean_latency_ms:.0f} |"
        )
    lines.append("")
    lines.append(f"Gate: context precision > {CONTEXT_PRECISION_GATE}")
    for r in rows:
        for note in r.notes:
            lines.append(f"- **{r.label}**: {note}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="CARELite R0-R9 retrieval ablation")
    parser.add_argument(
        "--max-turns", type=int, default=8, help="train-split scenarios to use (default 8)"
    )
    parser.add_argument("--rows", nargs="*", default=list(ABLATION_ORDER))
    parser.add_argument(
        "--no-precision", action="store_true", help="skip the LLM-judged context precision column"
    )
    parser.add_argument("--out", default="", help="write JSON results here")
    args = parser.parse_args(argv)

    rows = run_ablation(
        rows=args.rows,
        max_turns=args.max_turns,
        score_precision=not args.no_precision,
    )
    table = format_markdown(rows)
    print(table)

    settings = get_settings()
    out_path = args.out or (settings.runs_dir / "retrieval" / "ablation.json")
    from pathlib import Path

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in rows], indent=2))
    print(f"\nwrote {path}")

    full = next((r for r in rows if r.label == "R9"), None)
    if full is not None and full.gate_passed is False:
        print(
            f"\nGATE FAILED: R9 context precision {full.context_precision:.3f} "
            f"is not above {CONTEXT_PRECISION_GATE}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
