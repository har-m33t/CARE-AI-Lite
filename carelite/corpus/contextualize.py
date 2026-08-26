"""carelite.corpus.contextualize — the Anthropic contextual-retrieval pass.

`contextualize_chunks` builds the prompt and drives the loop (resumability,
rate-limit-aware retry) against an injected `generate_prefix` callable — that
part is model-agnostic and is exactly what shipped structure-only in the
first wave. `make_ollama_generate_prefix` is the real generator, wired to a
local Ollama daemon and `settings.models.generator` (gemma4:12b); it is now
the default for `run_contextualize_pass`, the entry point that pulls chunks
missing a prefix straight out of Postgres, generates, and persists each one
back immediately (see its docstring for why "immediately" rather than
batched matters for an unattended multi-hour run). `python -m
carelite.corpus.contextualize` runs the whole pending set.

Resumability: chunks that already have a `contextual_prefix` are skipped, so
an interrupted run picks up where it left off with no re-work and no
duplicate model calls. `run_contextualize_pass` also filters at the SQL level
(`WHERE contextual_prefix IS NULL`), so a resumed run doesn't even fetch rows
it would just skip.

**Trust boundary.** `contextual_prefix` is model-generated text that gets
concatenated onto `Chunk.text` (see `carelite.types.Chunk.embedding_text`)
and from there is embedded, retrieved, and fed into the generator's prompt —
the same path as any other untrusted content reaching the model. It is
prepended, not interpolated into instructions, and the source documents are
the fixed 33-paper corpus rather than attacker-reachable input, but a prefix
should still be read as data describing the chunk, never as something to
follow.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from carelite.config import get_settings
from carelite.types import Chunk

# Anthropic's contextual retrieval prompt shape: give the model the whole
# document plus the one chunk, ask for a short situating sentence to prepend
# before embedding. See build_plan Part I §51.
_PROMPT_TEMPLATE = """\
<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short, succinct context (1-2 sentences) to situate this chunk \
within the overall document, for the purpose of improving search retrieval \
of the chunk. Answer only with the succinct context and nothing else."""


class RateLimited(Exception):
    """Raised by a `generate_prefix` implementation on a retryable rate-limit response.

    `contextualize_chunks` retries on this with exponential backoff; any
    other exception is recorded as a per-chunk failure and the loop moves on.
    """


def build_prompt(document_text: str, chunk: Chunk) -> str:
    """The exact prompt sent to the generator model for one chunk. Pure string
    construction — no network/model call, so it is unit-testable on its own."""
    return _PROMPT_TEMPLATE.format(document=document_text, chunk=chunk.text)


GeneratePrefix = Callable[[str, Chunk], str]
"""`(document_text, chunk) -> situating prefix`. Implemented by the inference lane."""


class DocumentTextLookup(Protocol):
    """What `contextualize_chunks` needs from `document_text_by_paper_id`: a
    `.get(paper_id, default=None)` — satisfied by a plain `dict[str, str]`
    (the structure-only wave's shape, and what every existing test still
    passes) and by `_PdfDocumentTextCache`'s lazy, cached lookup alike."""

    def get(self, paper_id: str, default: str | None = None) -> str | None: ...


def _not_wired_up(_document_text: str, _chunk: Chunk) -> str:
    raise NotImplementedError(
        "carelite.corpus.contextualize has no model wired up this wave (structure only). "
        "Pass a `generate_prefix` callable — the inference lane provides the real one."
    )


@dataclass
class ContextualizeReport:
    processed: int = 0
    skipped_already_done: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)  # (chunk_id, reason)

    @property
    def total_failed(self) -> int:
        return len(self.failed)


@retry(
    retry=retry_if_exception_type(RateLimited),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _generate_with_retry(generate_prefix: GeneratePrefix, document_text: str, chunk: Chunk) -> str:
    return generate_prefix(document_text, chunk)


def contextualize_chunks(
    chunks: Iterable[Chunk],
    document_text_by_paper_id: DocumentTextLookup,
    *,
    generate_prefix: GeneratePrefix = _not_wired_up,
    resume: bool = True,
) -> tuple[list[Chunk], ContextualizeReport]:
    """Fill in `Chunk.contextual_prefix` for every chunk that doesn't have one.

    Returns `(chunks, report)` with `chunks` in the same order as the input,
    each either untouched (already had a prefix, or `document_text_by_paper_id`
    was missing its paper), successfully updated, or left as-is with the
    failure recorded in `report.failed`.

    Rate-limit-aware: a `generate_prefix` that raises `RateLimited` is retried
    with exponential backoff (via tenacity) before being counted as a failure.
    Any other exception is recorded immediately and the loop continues — one
    bad chunk must not abort a ~1,500-chunk unattended run.
    """
    report = ContextualizeReport()
    out: list[Chunk] = []

    for chunk in chunks:
        if resume and chunk.contextual_prefix:
            report.skipped_already_done += 1
            out.append(chunk)
            continue

        document_text = document_text_by_paper_id.get(chunk.paper_id)
        if document_text is None:
            report.failed.append(
                (chunk.chunk_id, f"no document text for paper_id={chunk.paper_id!r}")
            )
            out.append(chunk)
            continue

        try:
            prefix = _generate_with_retry(generate_prefix, document_text, chunk)
        except Exception as e:
            report.failed.append((chunk.chunk_id, f"{type(e).__name__}: {e}"))
            out.append(chunk)
            continue

        out.append(chunk.model_copy(update={"contextual_prefix": prefix}))
        report.processed += 1

    return out, report


# ---------------------------------------------------------------------------
# The real generator: Ollama
# ---------------------------------------------------------------------------

_MIN_CONTEXT = 8192

#: Exception type names treated as retryable (transient daemon/network
#: trouble) rather than a hard per-chunk failure. Matched by name rather than
#: `isinstance` against `httpx`/`ollama` exception classes so this doesn't
#: take a hard dependency on either library's exception hierarchy and still
#: works against a fake client in tests that raises a plain `ConnectionError`
#: or `TimeoutError`.
_TRANSIENT_ERROR_NAMES = {
    "ConnectionError",
    "TimeoutError",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "RequestError",
}


def _is_transient_ollama_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True
    return type(exc).__name__ in _TRANSIENT_ERROR_NAMES


def _num_ctx_for(prompt: str, *, num_predict: int, window: int) -> int:
    """`num_ctx` sized from the assembled prompt, floored and capped.

    Same rule as `carelite.generate.model.context_size` (rounded to a power
    of two so a near-identical prompt length doesn't change the allocation),
    reimplemented here rather than imported — `carelite/generate/` is a
    different lane's owned path, and this module has no reason to depend on
    its internals. ~4 chars/token, erring high on purpose.
    """
    needed = len(prompt) // 4 + 1 + num_predict + 256
    size = _MIN_CONTEXT
    while size < needed:
        size *= 2
    return min(max(size, _MIN_CONTEXT), window)


def _invoke_chat(
    client: Any,
    *,
    model: str,
    prompt: str,
    num_ctx: int,
    num_predict: int,
    temperature: float,
) -> str:
    """Call `client.chat`, suppressing visible chain-of-thought where supported.

    `think=False` matters a lot for this specific call, not just as hygiene:
    `gemma4:12b` (the configured generator, see `settings.models.generator`)
    is a thinking model, and a short situating sentence doesn't need a
    scratchpad. Measured against this corpus's own chunks: ~117s cold /
    ~17s warm per call with thinking left on (`num_predict` mostly consumed
    by `thinking`, leaving the visible `content` empty at a modest budget)
    versus roughly 10x faster with it off. Over ~470 chunks that is the
    difference between an unattended run finishing in a couple of hours or
    running most of a day. Mirrors `carelite.generate.model._invoke`: an
    older `ollama` client that rejects the `think` kwarg falls back to a
    plain call, and an empty first answer (occasionally seen paired with
    `think=False`) is retried once with thinking left on rather than
    accepted as the model's answer.
    """
    options = {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx}
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": options,
    }
    for extra in ({"think": False}, {}):
        try:
            response = client.chat(**kwargs, **extra)
        except TypeError:
            response = client.chat(**kwargs)
        text = str(response["message"]["content"])
        if text.strip():
            return text.strip()
    return ""


def make_ollama_generate_prefix(
    *,
    client: Any | None = None,
    model: str | None = None,
    host: str | None = None,
    num_predict: int = 200,
    temperature: float = 0.0,
) -> GeneratePrefix:
    """Build a `generate_prefix` callable backed by a local Ollama daemon.

    Defaults to `settings.models.generator` (gemma4:12b) on
    `settings.ollama_host`; both are overridable for a one-off run against a
    different tag or host. `client` is injectable — pass a fake with a
    `.chat(**kwargs)` method to exercise this in a unit test with no daemon
    involved; production code (and the default) lazily imports and opens a
    real `ollama.Client` on first use, so importing this module never
    requires the `ollama` package's transport to be reachable.

    A transient-looking failure (connection refused, timeout, a 5xx/429 from
    the daemon) is raised as `RateLimited` so `contextualize_chunks`'
    exponential-backoff retry picks it up; anything else propagates and is
    recorded as a per-chunk failure, matching the same policy the rest of
    this module documents.
    """
    settings = get_settings()
    resolved_model = model or settings.models.generator.tag
    resolved_host = host or settings.ollama_host
    window = settings.models.generator.context_window
    _client = client

    def generate(document_text: str, chunk: Chunk) -> str:
        nonlocal _client
        if _client is None:
            import ollama

            _client = ollama.Client(host=resolved_host)

        prompt = build_prompt(document_text, chunk)
        num_ctx = _num_ctx_for(prompt, num_predict=num_predict, window=window)

        try:
            text = _invoke_chat(
                _client,
                model=resolved_model,
                prompt=prompt,
                num_ctx=num_ctx,
                num_predict=num_predict,
                temperature=temperature,
            )
        except Exception as exc:
            if _is_transient_ollama_error(exc):
                raise RateLimited(f"{type(exc).__name__}: {exc}") from exc
            raise

        if not text:
            raise ValueError(
                f"{resolved_model!r} returned an empty prefix for chunk {chunk.chunk_id!r}"
            )
        return text

    return generate


# ---------------------------------------------------------------------------
# Driving the pass against the live corpus
# ---------------------------------------------------------------------------


class _PdfDocumentTextCache:
    """Lazily extracts and caches each paper's full document text, keyed by
    `paper_id`, from its source file on disk (`paper.pdf_path` in Postgres).

    Duck-types the one method `contextualize_chunks` calls on
    `document_text_by_paper_id` (`.get`), so it drops straight into that
    signature without changing it. Extraction happens once per paper on
    first request rather than once per chunk — with chunks processed grouped
    by paper (see `_load_pending_chunks`), that's once per paper for the
    whole run.

    A paper with no `pdf_path` on file, or whose extraction fails, caches a
    `None`/miss rather than raising — every chunk from that paper then fails
    with "no document text for paper_id=...", visible in the report, instead
    of aborting the run.
    """

    def __init__(self) -> None:
        self._text_by_paper: dict[str, str | None] = {}
        self._pdf_path_by_paper: dict[str, str] | None = None

    def _pdf_paths(self) -> dict[str, str]:
        if self._pdf_path_by_paper is None:
            from carelite.db.connection import fetch_all

            rows = fetch_all("SELECT paper_id, pdf_path FROM paper")
            self._pdf_path_by_paper = {r["paper_id"]: r["pdf_path"] for r in rows if r["pdf_path"]}
        return self._pdf_path_by_paper

    def get(self, paper_id: str, default: str | None = None) -> str | None:
        if paper_id not in self._text_by_paper:
            from carelite.corpus.extract import extract_source

            path = self._pdf_paths().get(paper_id)
            if not path:
                self._text_by_paper[paper_id] = None
            else:
                result = extract_source(path)
                self._text_by_paper[paper_id] = result.text if result.ok else None
        value = self._text_by_paper[paper_id]
        return value if value is not None else default


def _load_pending_chunks() -> list[Chunk]:
    """Every chunk with no `contextual_prefix` yet, ordered `paper_id,
    ordinal` — so all of one paper's chunks are contiguous and in document
    order. That grouping is what lets Ollama's prompt-prefix cache carry the
    `<document>` block across consecutive calls: each paper's text is the
    resident prefix for its own run of chunks, and only changes when the
    paper does. Processing in scattered order (e.g. by `chunk_id` globally,
    or interleaved across papers) would re-prefill the whole document on
    nearly every call — see D11 in `DECISIONS.md` for what that costs when
    ignored (33x slower, and the reason condition LC was dropped).
    """
    from carelite.db.connection import fetch_all

    rows = fetch_all(
        "SELECT chunk_id, paper_id, text FROM chunk "
        "WHERE contextual_prefix IS NULL ORDER BY paper_id, ordinal"
    )
    return [Chunk(chunk_id=r["chunk_id"], paper_id=r["paper_id"], text=r["text"]) for r in rows]


def run_contextualize_pass(
    *,
    chunks: Sequence[Chunk] | None = None,
    document_text_by_paper_id: DocumentTextLookup | None = None,
    generate_prefix: GeneratePrefix | None = None,
    persist: Callable[[str, str], None] | None = None,
    on_progress: Callable[[int, int, Chunk], None] | None = None,
) -> ContextualizeReport:
    """Run the contextual-prefix pass, persisting each prefix to Postgres as
    soon as it's generated.

    Every dependency is injectable and defaults to the real thing (live DB,
    live Ollama), so this function is exercised in unit tests with fakes for
    all four — no live database or model required, matching the `db`/
    `inference` marker split the rest of this lane uses.

    Persisting per chunk — rather than collecting the whole batch and writing
    once at the end, which is what a bare loop over `contextualize_chunks`
    would do — is what makes an unattended ~470-chunk run resumable at chunk
    granularity: a crash, a kill, or a `Ctrl-C` partway through leaves every
    already-written prefix in Postgres, and the next run's `_load_pending_chunks`
    (or an injected `chunks`, if a test wants to check something narrower)
    simply won't see it again.

    Delegates the actual generate/retry/record-failure logic to
    `contextualize_chunks`, called once per chunk rather than once for the
    whole list, purely so a persist call can happen after each one — the
    retry policy, resumability skip, and failure bookkeeping are exactly the
    tested behaviour of that function, not reimplemented here.
    """
    if chunks is None:
        chunks = _load_pending_chunks()
    if document_text_by_paper_id is None:
        document_text_by_paper_id = _PdfDocumentTextCache()
    if generate_prefix is None:
        generate_prefix = make_ollama_generate_prefix()
    if persist is None:
        from carelite.corpus.load import update_chunk_prefix

        persist = update_chunk_prefix

    total = ContextualizeReport()
    n = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        out, one = contextualize_chunks(
            [chunk], document_text_by_paper_id, generate_prefix=generate_prefix, resume=True
        )
        total.processed += one.processed
        total.skipped_already_done += one.skipped_already_done
        total.failed.extend(one.failed)

        updated = out[0]
        if one.processed and updated.contextual_prefix:
            persist(updated.chunk_id, updated.contextual_prefix)

        if on_progress:
            on_progress(idx, n, updated)

    return total


def main(argv: list[str] | None = None) -> int:
    """`python -m carelite.corpus.contextualize` — run the pass over every
    chunk currently missing a `contextual_prefix`, printing progress as it
    goes (this runs for hours over the full corpus; see the module
    docstring). `--limit` runs a small slice first for a spot check.
    """
    import argparse

    ap = argparse.ArgumentParser(
        description="Fill in Chunk.contextual_prefix for every chunk that doesn't have one yet."
    )
    ap.add_argument(
        "--model", default=None, help="Ollama tag (default: settings.models.generator.tag)"
    )
    ap.add_argument("--host", default=None, help="Ollama host (default: settings.ollama_host)")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N chunks, then stop")
    args = ap.parse_args(argv)

    chunks = _load_pending_chunks()
    if args.limit is not None:
        chunks = chunks[: args.limit]
    if not chunks:
        print("Nothing to do: every chunk already has a contextual_prefix.")
        return 0

    n_papers = len({c.paper_id for c in chunks})
    print(f"{len(chunks)} chunk(s) across {n_papers} paper(s) need a contextual_prefix.")

    generate_prefix = make_ollama_generate_prefix(model=args.model, host=args.host)
    started = time.monotonic()
    state: dict[str, str | None] = {"paper": None}

    def on_progress(idx: int, total: int, chunk: Chunk) -> None:
        if chunk.paper_id != state["paper"]:
            print(f"-- {chunk.paper_id} --")
            state["paper"] = chunk.paper_id
        status = "ok" if chunk.contextual_prefix else "FAILED"
        elapsed = time.monotonic() - started
        print(f"[{idx}/{total}] {chunk.chunk_id} {status} ({elapsed:.0f}s elapsed)")

    report = run_contextualize_pass(
        chunks=chunks, generate_prefix=generate_prefix, on_progress=on_progress
    )

    elapsed = time.monotonic() - started
    print(
        f"\nDone in {elapsed / 60:.1f} min. processed={report.processed} "
        f"skipped_already_done={report.skipped_already_done} failed={report.total_failed}"
    )
    if report.failed:
        print(f"{report.total_failed} failure(s):")
        for chunk_id, reason in report.failed:
            print(f"  {chunk_id}: {reason}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
