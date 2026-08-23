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
when it cannot. `load` writes only what survived, with `human_verified` left
FALSE; `review` produces the digest a human signs off against and records that
sign-off.

The provenance story this supports is "LLM-assisted extraction, human-verified",
not "hand-authored". That distinction has to survive into the write-up.
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
