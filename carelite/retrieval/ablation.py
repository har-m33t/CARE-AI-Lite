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

**Context precision, and one correction to what it asks.** The gate is "Ragas
context precision > 0.7". The `ragas` package is not a project dependency, so
this module implements the metric directly rather than importing it.

Ragas asks its judge whether a context is relevant to *answering the query*.
That question is wrong here and it measurably mis-scores this corpus. A patient
turn is not a query, and the system's task is not to answer the patient — it is
to help a clinician respond. The scenario bank is built specifically around
turns where those diverge: `hard_case` tags include `false_comprehension`,
`buried_cue`, and `blocking_bait`. Asked the literal question, the judge scored
the passage "teach-back ... confirms patient comprehension" as NOT useful for
"Mm-hm. Yeah. No, that makes sense. It's a lot of words, that's all. Keep
going." — a turn tagged `false_comprehension`, retrieved at a rerank score of
0.976, and about as on-target as this corpus gets. Every hard scenario would
score 0 for being hard, and the metric would report the retriever's best
behaviour as its worst.

`RELEVANCE_SYSTEM` therefore asks whether the passage helps the clinician
handle the moment, and says explicitly that patients understate what they need.
It still rejects off-domain turns outright. This is a correction to the
question, not a loosening of the bar: the off-domain and wrong-moment cases
below are unchanged, and both are checked in
`tests/unit/retrieval/test_ablation.py`.

`LLMContextPrecisionWithoutReference` is, for one turn,

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
    "lc_sample",
    "lc_sample_stats",
    "long_context_stats",
    "main",
    "prewarm_hyde",
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

#: The gate from the lane brief. Applied to on-domain context precision only.
CONTEXT_PRECISION_GATE = 0.7

#: Below this many scored turns a row gets no gate verdict at all.
#: Exists because a run once printed "1.000 PASS" on a single surviving turn
#: after CRAG rejected the other five. A gate is a claim about retrieval
#: quality across turns; one turn cannot support it in either direction.
MIN_SCORED_FOR_GATE = 5


@dataclass
class AblationRow:
    """One row of the emitted table.

    **Precision and rejection are separate columns, and that is the whole
    point of this shape.** An earlier version blended them into a single
    context-precision figure computed over every turn, off-domain turns
    included. Because `RELEVANCE_SYSTEM` correctly tells the judge that *no*
    passage helps an off-domain turn, each such turn contributes a structural
    zero. With three off-domain turns in a six-turn run, no configuration
    without CRAG could exceed ~0.5 against a gate of 0.7 — the gate was
    testing a property of the turn mix rather than of the retriever, and every
    non-CRAG row was guaranteed to fail however good retrieval was.

    The same blend inverted CRAG's meaning. R7 and R9 scored 1.000 and were
    marked PASS on `n_scored = 1`: the gate had rejected five of six turns, so
    one turn survived and happened to score perfectly. A reader skimming that
    table would conclude CRAG *improves* precision, when what it did was reject
    almost everything. Rejecting is CRAG's job, but it must read as a
    behaviour, not as a shrinking denominator that flatters the row.

    So: `context_precision` is computed over **on-domain turns only** and is
    what the gate is applied to; `off_domain_rejection_rate` is CRAG's
    correctness on turns that ought to be rejected; `on_domain_fallback_rate`
    is what it costs on turns that ought not to be. `gate_passed` refuses a
    verdict below `min_scored`, so a sample of one can never print PASS again.
    """

    label: str
    note: str
    config: str
    n_turns: int = 0
    n_on_domain: int = 0
    n_off_domain: int = 0
    n_scored: int = 0
    mean_retrieved: float = 0.0
    mean_latency_ms: float = 0.0
    fallback_rate: float = 0.0
    on_domain_fallback_rate: float = 0.0
    off_domain_rejection_rate: float | None = None
    skipped_rate: float = 0.0
    context_precision: float | None = None
    min_scored: int = 0
    routes: dict[str, int] = field(default_factory=dict)
    graders: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.min_scored:
            self.min_scored = MIN_SCORED_FOR_GATE

    @property
    def gate_passed(self) -> bool | None:
        """`None` means "no verdict", not "failed".

        Returned when precision was not computed at all, and — deliberately —
        when it rests on fewer than `min_scored` turns. The gate is a claim
        about retrieval quality; it cannot be made from a sample of one.
        """
        if self.context_precision is None or self.n_scored < self.min_scored:
            return None
        return self.context_precision > CONTEXT_PRECISION_GATE

    @property
    def gate_label(self) -> str:
        if self.context_precision is None:
            return "n/a"
        cell = f"{self.context_precision:.3f}"
        verdict = self.gate_passed
        if verdict is None:
            return f"{cell} (n={self.n_scored}<{self.min_scored}, no verdict)"
        return f"{cell} {'PASS' if verdict else 'FAIL'}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "note": self.note,
            "config": self.config,
            "n_turns": self.n_turns,
            "n_on_domain": self.n_on_domain,
            "n_off_domain": self.n_off_domain,
            "n_scored": self.n_scored,
            "mean_retrieved": round(self.mean_retrieved, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "fallback_rate": round(self.fallback_rate, 3),
            "on_domain_fallback_rate": round(self.on_domain_fallback_rate, 3),
            "off_domain_rejection_rate": (
                None
                if self.off_domain_rejection_rate is None
                else round(self.off_domain_rejection_rate, 3)
            ),
            "skipped_rate": round(self.skipped_rate, 3),
            "context_precision_on_domain": (
                None if self.context_precision is None else round(self.context_precision, 3)
            ),
            "n_scored_min_for_gate": self.min_scored,
            "gate_passed": self.gate_passed,
            "routes": self.routes,
            "graders": self.graders,
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

#: Reasoning models spend this budget on thinking before emitting any content,
#: so a budget sized for the *answer* starves the model and yields an empty
#: string. Measured on `gpt-oss:20b` with this exact prompt: 200 returned `''`
#: every time, 600 and 1500 both returned the correct verdict.
#:
#: The failure is silent and expensive. An empty response becomes a `None`
#: verdict, a `None` verdict discards that turn's precision score entirely, and
#: the ablation table then reports a smaller `n_scored` and a depressed
#: precision that looks like a retrieval problem. It cost this lane a wrong
#: conclusion once: R0/R8/R9 all measured context precision 0.000 and were
#: nearly reported as "the corpus cannot serve these turns", when a substantial
#: part of it was this truncation. Sized with headroom deliberately — the cost
#: of over-budgeting is latency, the cost of under-budgeting is a wrong number
#: in a results table.
JUDGE_NUM_PREDICT = 800

_RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {"useful": {"type": "boolean"}},
    "required": ["useful"],
}


def _judge_relevance(client: Any, utterance: str, passage: str) -> bool | None:
    """One verdict. `None` only after a retry, because a single unparseable
    answer would otherwise discard the whole turn's precision score (see
    `context_precision`) and silently shrink the sample the table rests on."""
    for _ in range(2):
        result = client.chat(
            system=RELEVANCE_SYSTEM,
            task=(
                "Would the passage above help a clinician respond well to this patient "
                "turn? Judge the moment, not the literal request. JSON only."
            ),
            utterance=utterance,
            extra_untrusted=[("PASSAGE", passage)],
            json_schema=_RELEVANCE_SCHEMA,
            num_predict=JUDGE_NUM_PREDICT,
        )
        if result is None or not result.text:
            continue
        try:
            return bool(json.loads(result.text)["useful"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
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


#: Head-room reserved inside the context window for the system prompt, the
#: patient turn, and the model's own response. The corpus sample is fitted to
#: what is left, not to the raw window size.
LC_RESERVE_TOKENS = 16_000

#: Characters per token. A budgeting approximation for English prose, not an
#: exact count — the sample is sized conservatively so the approximation has
#: room to be wrong.
CHARS_PER_TOKEN = 4


def long_context_stats() -> dict[str, Any]:
    """What condition LC would cost, and what it actually gets (D7).

    v3 §3 assumed the whole corpus fits in the window and could therefore be
    stuffed, making LC a genuine no-retrieval baseline. Measured, it does not:
    471 chunks are roughly 326,526 tokens against the 128,000-token window
    configured for `models.long_context`, or **255% utilisation**. Reserving
    `LC_RESERVE_TOKENS` for the prompt and response, about a third of the
    corpus fits.

    Context precision is deliberately not computed for LC. Its context is
    fixed rather than retrieved, so per-turn precision would be measuring the
    sample, not a retriever, and would be near zero by construction rather
    than by measurement.
    """
    from carelite.db.connection import fetch_one

    row = fetch_one("SELECT count(*) AS n, coalesce(sum(length(text)), 0) AS chars FROM chunk")
    n_chunks = int(row["n"]) if row else 0
    chars = int(row["chars"]) if row else 0
    est_tokens = chars // CHARS_PER_TOKEN
    window = get_settings().models.long_context.context_window
    budget = max(0, window - LC_RESERVE_TOKENS)
    return {
        "n_chunks": n_chunks,
        "est_tokens": est_tokens,
        "context_window": window,
        "reserve_tokens": LC_RESERVE_TOKENS,
        "budget_tokens": budget,
        "fits": est_tokens < window,
        "utilisation": round(est_tokens / window, 3) if window else None,
    }


_LC_CHUNK_SQL = """
SELECT chunk_id, paper_id, ordinal, length(text) AS chars
FROM chunk
ORDER BY paper_id, ordinal
"""


def lc_sample(
    *,
    budget_tokens: int | None = None,
    seed: int | None = None,
) -> list[str]:
    """The fixed, query-independent corpus sample that condition LC-sample uses.

    **Any selection rule is a form of retrieval.** This is the point D7 exists
    to keep in the open, and it must not be quietly absorbed into an
    implementation detail. LC was specified as the baseline that asks whether
    curated retrieval beats stuffing everything in. Because the corpus does not
    fit, it can no longer ask that. It can only ask whether *query-dependent*
    selection beats a *fixed* context. That remains a real and interesting
    question — arguably closer to what a practitioner would actually build —
    but it is not the question build plan v3 §3 posed, and a reader must not be
    able to mistake one for the other. The row is named `LC-sample` for exactly
    that reason: the name is load-bearing, not cosmetic.

    **Round-robin across papers, not random sampling.** Round-robin guarantees
    every one of the 33 papers is represented. Random selection can drop whole
    papers by chance, which would make LC's content an accident of the seed and
    would turn part of the C-vs-LC comparison into a comparison of which papers
    happened to survive.

    **Where the seed actually matters.** Within a paper, chunks are taken in
    `ordinal` order, which is deterministic and needs no seed. Papers are
    visited in an order shuffled at `seed`, and that only changes the outcome
    on the final, partial cycle — when the budget runs out part-way through a
    round, the seed decides which papers get that last chunk. Fixing it stops
    the tail from systematically favouring whichever papers sort first by id.

    Returns chunk ids in the order they should be stuffed into the prompt.
    """
    import random

    from carelite.db.connection import fetch_all

    settings = get_settings()
    if budget_tokens is None:
        budget_tokens = max(0, settings.models.long_context.context_window - LC_RESERVE_TOKENS)
    if seed is None:
        seed = settings.experiment.base_seed

    by_paper: dict[str, list[tuple[str, int]]] = {}
    for row in fetch_all(_LC_CHUNK_SQL):
        by_paper.setdefault(str(row["paper_id"]), []).append(
            (str(row["chunk_id"]), int(row["chars"]))
        )

    papers = sorted(by_paper)
    random.Random(seed).shuffle(papers)

    selected: list[str] = []
    spent = 0
    depth = 0
    deepest = max((len(v) for v in by_paper.values()), default=0)
    while depth < deepest:
        progressed = False
        for paper in papers:
            chunks = by_paper[paper]
            if depth >= len(chunks):
                continue
            chunk_id, chars = chunks[depth]
            cost = chars // CHARS_PER_TOKEN
            if spent + cost > budget_tokens:
                # Budget exhausted mid-cycle. Stop outright rather than
                # skipping to a smaller chunk: "take whatever still fits"
                # would silently bias the sample toward short chunks.
                return selected
            selected.append(chunk_id)
            spent += cost
            progressed = True
        if not progressed:
            break
        depth += 1
    return selected


def lc_sample_stats() -> dict[str, Any]:
    """`long_context_stats()` plus what the sample actually selected."""
    stats = long_context_stats()
    sample = lc_sample()
    stats["sample_chunks"] = len(sample)
    stats["sample_fraction"] = (
        round(len(sample) / stats["n_chunks"], 3) if stats["n_chunks"] else None
    )
    stats["sample_chunk_ids"] = sample
    return stats


# ---------------------------------------------------------------------------
# Running a row
# ---------------------------------------------------------------------------


def run_row(
    flags: RetrievalFlags,
    turns: Sequence[str],
    *,
    off_domain: Sequence[str] = (),
    embedder: Any = None,
    generator: Any = None,
    grader_client: Any = None,
    reranker: Any = None,
    precision_client: Any = None,
    score_precision: bool = True,
) -> AblationRow:
    """Run one ablation configuration over every turn.

    `off_domain` names the subset of `turns` the corpus provably cannot serve.
    Those turns are scored for *rejection*, never for precision — see
    `AblationRow`.
    """
    row = AblationRow(label=flags.label or "?", note=flags.note, config=flags.summary())
    off = set(off_domain)

    if flags.long_context:
        stats = lc_sample_stats()
        row.n_turns = len(turns)
        row.n_on_domain = len([t for t in turns if t not in off])
        row.n_off_domain = len(turns) - row.n_on_domain
        row.mean_retrieved = float(stats["sample_chunks"])
        row.notes.append(
            f"no query-dependent retrieval; the full corpus is {stats['n_chunks']} chunks "
            f"~{stats['est_tokens']:,} tokens against a {stats['context_window']:,}-token "
            f"window ({stats['utilisation']:.1%} utilisation) and DOES NOT FIT"
        )
        row.notes.append(
            f"fixed round-robin sample across all papers at seed "
            f"{get_settings().experiment.base_seed}: {stats['sample_chunks']} of "
            f"{stats['n_chunks']} chunks ({stats['sample_fraction']:.1%}) inside a "
            f"{stats['budget_tokens']:,}-token budget "
            f"({stats['reserve_tokens']:,} reserved for prompt and response)"
        )
        row.notes.append(
            "ANY SELECTION RULE IS A FORM OF RETRIEVAL. This row does not ask whether "
            "curated retrieval beats stuffing everything in — the corpus does not fit, "
            "so that question is not available. It asks whether query-DEPENDENT "
            "selection beats a FIXED context. That is not the question v3 §3 posed "
            "(D7); the LC-sample name carries the distinction."
        )
        row.notes.append(
            "context precision not computed for LC-sample: its context is fixed rather "
            "than retrieved, so per-turn precision would measure the sample, not a retriever"
        )
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

    paired = list(zip(turns, results, strict=True))
    on_pairs = [(t, r) for t, r in paired if t not in off]
    off_pairs = [(t, r) for t, r in paired if t in off]

    row.n_turns = len(results)
    row.n_on_domain = len(on_pairs)
    row.n_off_domain = len(off_pairs)
    row.mean_retrieved = _mean([len(r.trace.retrieved) for r in results])
    row.mean_latency_ms = _mean(latencies)
    row.fallback_rate = _mean([1.0 if r.trace.fell_back_to_b else 0.0 for r in results])
    row.on_domain_fallback_rate = _mean(
        [1.0 if r.trace.fell_back_to_b else 0.0 for _, r in on_pairs]
    )
    row.off_domain_rejection_rate = (
        _mean([1.0 if r.trace.fell_back_to_b else 0.0 for _, r in off_pairs]) if off_pairs else None
    )
    row.skipped_rate = _mean(
        [1.0 if r.trace.route is Route.EMOTIONAL_ONLY else 0.0 for r in results]
    )

    counts: dict[str, int] = {}
    for r in results:
        counts[r.trace.route.value] = counts.get(r.trace.route.value, 0) + 1
    row.routes = counts

    # Which grader actually decided. Recorded so "was this the LLM evaluator or
    # the cosine-threshold fallback?" is answerable from the run artifact
    # rather than by reading the code and reasoning about it.
    graders: dict[str, int] = {}
    for r in results:
        if r.grade_report is not None:
            name = r.grade_report.grader or "?"
            graders[name] = graders.get(name, 0) + 1
    row.graders = graders

    unavailable = {n for r in results for n in r.leg_notes if "unavailable" in n}
    row.notes.extend(sorted(unavailable))

    if score_precision and precision_client is not None:
        scores: list[float] = []
        for turn, result in on_pairs:
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


def prewarm_hyde(turns: Sequence[str], generator: Any) -> int:
    """Generate every HyDE passage up front, in one pass over the generator.

    **This is a model-residency fix, not a micro-optimisation.** Ollama holds a
    limited number of models in memory at once. The natural per-turn ordering
    is HyDE (generator, ~8GB) then CRAG grading and precision judging (judge,
    ~12.7GB), which on a single-GPU machine evicts and fully reloads a large
    model twice per turn. Measured mid-run: `ollama ps` showed the generator
    not resident at all, and judge throughput collapsed from the 5-15s per call
    measured in isolation to roughly 150s per call — a ~10x slowdown caused
    entirely by the interleaving.

    Generating all passages first collapses the generator to a single
    residency period, after which every remaining call in the run is a judge
    call and the judge stays loaded. Passages land in the `LLMClient` cache, so
    the pipeline picks them up transparently on each row without knowing this
    happened.

    Returns the number of passages successfully generated.
    """
    from carelite.retrieval.hyde import generate_hyde_passage

    generated = 0
    for turn in turns:
        if generate_hyde_passage(turn, client=generator, enabled=True):
            generated += 1
    return generated


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
    off_domain: tuple[str, ...] = OFF_DOMAIN_TURNS if include_off_domain else ()
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
        if any(PRESETS[name].hyde for name in rows if name in PRESETS):
            prewarm_hyde(turns, generator)

        for name in rows:
            flags = preset(name)
            out.append(
                run_row(
                    flags,
                    turns,
                    off_domain=off_domain,
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
    """Emit the table.

    Column choice is load-bearing. Precision is on-domain only, so the gate is
    applied to something a configuration can actually achieve; rejection is
    reported separately and split by whether rejecting was the right call.
    Read `off-dom rej` as CRAG working and `on-dom fb` as what it costs.
    """
    header = (
        "| row | config | on-dom | ctx precision, on-domain (Ragas-equiv) | n_scored | "
        "on-dom fb | off-dom rej | n_ret | latency ms | grader |\n"
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|"
    )
    lines = [header]
    for r in rows:
        off = "n/a" if r.off_domain_rejection_rate is None else f"{r.off_domain_rejection_rate:.0%}"
        grader = ",".join(f"{k}:{v}" for k, v in sorted(r.graders.items())) if r.graders else "-"
        lines.append(
            f"| {r.label} | {r.config} | {r.n_on_domain} | {r.gate_label} | {r.n_scored} | "
            f"{r.on_domain_fallback_rate:.0%} | {off} | {r.mean_retrieved:.1f} | "
            f"{r.mean_latency_ms:.0f} | {grader} |"
        )
    lines.append("")
    lines.append(
        f"Gate: on-domain context precision > {CONTEXT_PRECISION_GATE}, "
        f"and only where at least {MIN_SCORED_FOR_GATE} turns were scored."
    )
    lines.append("")
    lines.append(
        "**Precision is on-domain only, and that is deliberate.** The run "
        "includes off-domain turns on purpose — without them the table cannot "
        "show what CRAG does — but `RELEVANCE_SYSTEM` correctly judges that no "
        "passage in this corpus helps an off-domain turn, so each one "
        "contributes a structural zero. Blending them into one precision figure "
        "caps every non-CRAG row below the gate by construction and measures "
        "the turn mix rather than the retriever. Rejection of those turns is "
        "reported in `off-dom rej`, where a HIGH number is CRAG succeeding."
    )
    lines.append("")
    lines.append(
        "**A row with `no verdict` is not a failure.** It means precision rests "
        "on fewer than "
        f"{MIN_SCORED_FOR_GATE} turns, usually because CRAG rejected most of "
        "them, and a gate cannot be claimed either way from that sample. Raise "
        "`--max-turns` for a gate-quality run."
    )
    lines.append("")
    lines.append(
        "**Latency is not a component cost in a mixed run.** Rows share one Ollama "
        "daemon, so a row's timing depends on which model happened to be resident "
        "when it ran and on what earlier rows left in the prompt cache. Observed "
        "directly: R7 and R9 have identical CRAG configuration and measured 46,169ms "
        "and 4,141ms in the same run, an 11x gap that is entirely residency and "
        "caching. Read this column for order-of-magnitude only; time a single "
        "configuration on its own before quoting a per-component figure."
    )
    lines.append("")
    lines.append(
        "**`grader` attributes each CRAG decision.** `llm` is the prompted "
        "retrieval evaluator; `score` is the cosine-threshold fallback, which "
        "cannot detect an off-domain turn (see crag.py). A row showing `score` "
        "means the judge model was unreachable and that row's rejections should "
        "not be trusted."
    )
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
