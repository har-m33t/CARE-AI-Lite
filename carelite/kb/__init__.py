"""The CARELite knowledge base: extraction, provenance validation, load, review.

The pipeline is four stages and one invariant.

    extract  ->  validate  ->  load  ->  review
                    ^
                    |
        every entry's verbatim_span must appear in its source paper

`extract` is LLM-assisted and therefore assumed unreliable: a model asked to
quote a paper will sometimes produce a sentence the paper never contained.
`validate` is what makes the knowledge base trustworthy anyway — it locates
each claimed span in the real extracted text and rejects the entry outright
when it cannot, alongside deriving the evidence tier from the source's study
design and rejecting subject matter `TAXONOMY.md` excludes. `load` writes only
what survived. `review` produces a digest anyone can read the entries against.

**The provenance claim this supports, stated exactly: LLM-assisted extraction
with automated verbatim-span validation, and no human verification.** Not
"hand-authored", and — since `DECISIONS.md` D4 — not "human-verified" either.
The gate was dropped rather than ticked without anyone having read anything, so
`human_verified` is FALSE on every entry as the honest record of that, and
`review` is an available tool rather than a step the pipeline is waiting on.

What that claim covers is precise and worth not overstating in either
direction. Established mechanically: every stored `verbatim_span` is a literal
slice of its cited paper's extracted text; entries were rejected for fabricated
spans, unsupported subject matter, non-actionable takeaways, and findings their
span does not report; the fabrication rate was measured rather than estimated.
Not established by anyone: whether each *finding* follows from its *span*. No
automated check reaches that, and any result depending on knowledge base quality
inherits the gap. That distinction has to survive into the write-up.
"""

from carelite.kb.papers import PAPER_META, PaperMeta, PaperText, load_paper_texts
from carelite.kb.spans import SpanMatch, locate_span, normalize

__all__ = [
    "PAPER_META",
    "PaperMeta",
    "PaperText",
    "SpanMatch",
    "load_paper_texts",
    "locate_span",
    "normalize",
]
