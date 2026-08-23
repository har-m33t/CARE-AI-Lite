"""carelite.corpus.contextualize — the Anthropic contextual-retrieval pass.

STRUCTURE ONLY THIS WAVE. This module builds the prompt and drives the loop
(resumability, rate-limit-aware retry) but does not call any model — the
`generate_prefix` callable is injected by the caller. A later inference lane
wires in the real generator model (see `settings.models.generator`) and runs
this over the full ~1,500-chunk corpus unattended.

Resumability: chunks that already have a `contextual_prefix` are skipped, so
an interrupted run picks up where it left off with no re-work and no
duplicate API calls.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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
    document_text_by_paper_id: dict[str, str],
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
