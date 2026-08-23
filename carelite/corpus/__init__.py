"""Corpus pipeline: fetch -> extract -> chunk -> contextualize -> load.

Owned by the carelite-corpus lane. Each stage is an independent, importable
module so the pipeline can be run end-to-end or driven stage-by-stage from
tests and from other lanes (e.g. carelite-index consumes `chunk.Chunk`
objects; carelite-kb consumes `extract.ExtractedPaper` text).
"""

from __future__ import annotations
