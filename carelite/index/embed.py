"""carelite.index.embed — Ollama embedding client for `settings.models.embedder`.

Owned by the carelite-index lane. Everything downstream — dense chunk/kb_entry
rows in Postgres (`build.py`), HyDE, and the framework-query construction in
the retrieval lane — depends on two properties this module guarantees:

**Instruction-aware, asymmetric encoding — with an empirical correction.**
`embed_queries` and `embed_documents` are two distinct, independently
configurable code paths (not one `embed()` call with a boolean flag) so that
HyDE (which embeds a hypothetical *document*) and framework-query
construction (which embeds a *query*) can never be mixed up by accident —
that structural separation is what "asymmetric encoding" means here and it
is unconditional.

What is *not* unconditional is the assumption, carried over from
BGE-large/E5-style models, that the query side wants an instruction sentence
prepended ("Represent this sentence for searching relevant passages: ").
That was this module's original default and it was wrong for `bge-m3`
specifically: measured against the live corpus (see the carelite-index final
report), prepending *any* instruction text — even a 7-character `"query: "`
— raised cosine similarity between two topically unrelated one-word queries
from 0.52 to 0.72-0.84, i.e. it compresses the embedding space toward a
generic "this is a search query" direction and destroys the very
discrimination retrieval depends on. The same pattern showed up on full
15-word probe queries: with the original instruction prefix, `chunk`
rankings for four different natural-language questions were nearly
identical (cosine scores flat to 3 decimal places); with no prefix, the same
four queries produced clearly different, topically appropriate top-5s.
`bge-m3`'s own model card documents it as *not* requiring an instruction the
way `bge-large-en-v1.5` does — this is that guidance holding up empirically,
not a surprise. `DEFAULT_QUERY_PREFIX` is therefore `""`, same as
`DEFAULT_DOCUMENT_PREFIX`. Both remain real, independently overridable
fields (`query_prefix=`, `document_prefix=`) rather than hardcoded away,
because a future model swap (or a `carelite-retrieval` experiment that wants
to try one) should not require touching this module's internals — it should
just pass a different prefix and measure.

**A hard dimension check.** `vector(1024)` is frozen in `schema.sql`
(`carelite/db/schema.sql`) and `settings.retrieval.embedding_dim` (also
frozen, `carelite/config.py`) agrees. If the embedder ever returns a
different width — a model swap, a quantization change — every downstream
insert would fail loudly at the database, which is late and confusing. This
module checks the width itself, on the first real response, and raises
`EmbedDimensionError` immediately with both numbers in the message.

**Caching by content hash.** Re-embedding is real GPU/CPU time on a machine
running one Ollama daemon shared by two other inference lanes (`carelite-kb`,
and the judge in a later wave). `EmbedCache` keys on
`sha256(model_digest + prefix + text)`, so re-running `build.py` after an
interruption, or after a corpus reload that leaves 95% of chunks byte-identical,
costs nothing for the unchanged 95%. It is deliberately independent of
`build.py`'s own DB-side resumability (`index_embedding_state`): the cache
survives even a full DB wipe/reload, and `build.py`'s state table survives
even a cleared cache. Belt and suspenders, cheap either way.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from carelite.config import get_settings

__all__ = [
    "DEFAULT_DOCUMENT_PREFIX",
    "DEFAULT_QUERY_PREFIX",
    "EmbedCache",
    "EmbedDimensionError",
    "EmbedError",
    "OllamaEmbedder",
    "hash_text",
]


class EmbedError(RuntimeError):
    """The embedder could not be reached, or returned nothing usable."""


class EmbedDimensionError(EmbedError):
    """The model returned a vector whose width disagrees with `vector(1024)`.

    Raised rather than silently truncating/padding: `settings.retrieval.
    embedding_dim` and the schema's `vector(1024)` are both frozen contracts,
    and a dimension mismatch means the wrong model is configured, not
    something this lane should work around.
    """


#: Prepended to *queries* only (`embed_query`/`embed_queries`). Empty by
#: default for `bge-m3` — see the module docstring's "empirical correction"
#: section for the measurement that overturned the original non-empty
#: default carried over from bge-large/E5 convention. Real, independently
#: overridable field: a future model swap or a `carelite-retrieval`
#: experiment can pass `query_prefix=` without touching this module.
DEFAULT_QUERY_PREFIX = ""

#: Prepended to *documents* (corpus chunks, kb_entry rows, and HyDE's
#: hypothetical passage). Empty by default — bge-m3 measured best with no
#: instruction on either side. Kept as a real, independently overridable
#: field for the same reason as `DEFAULT_QUERY_PREFIX`.
DEFAULT_DOCUMENT_PREFIX = ""


def hash_text(text: str) -> str:
    """Stable content hash used both by `EmbedCache` and by `build.py`'s
    per-row `content_hash` (kept as the same function so the two layers of
    resumability agree on what "unchanged" means)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_key(model_digest: str, prefixed_text: str) -> str:
    return hash_text(f"{model_digest}|{prefixed_text}")


@dataclass
class EmbedCache:
    """Append-only JSONL cache of `prefixed_text -> embedding`, keyed by content hash.

    Same shape and durability contract as `carelite.eval.judge.cache.JudgeCache`:
    flush + fsync on every write so a killed process loses at most the record
    in flight, never the file; a truncated final line from an interrupted run
    is skipped and counted, not fatal.
    """

    path: Path
    corrupt_lines: int = field(default=0, init=False)
    _records: dict[str, list[float]] = field(default_factory=dict, init=False, repr=False)
    _handle: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.load()

    def load(self) -> None:
        self._records.clear()
        self.corrupt_lines = 0
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    key = str(obj["key"])
                    vec = [float(x) for x in obj["embedding"]]
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    self.corrupt_lines += 1
                    continue
                self._records[key] = vec

    def get(self, key: str) -> list[float] | None:
        return self._records.get(key)

    def put(self, key: str, embedding: Sequence[float]) -> None:
        vec = [float(x) for x in embedding]
        self._records[key] = vec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._handle is None:
            self._handle = self.path.open("a", encoding="utf-8")
        self._handle.write(json.dumps({"key": key, "embedding": vec}) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __len__(self) -> int:
        return len(self._records)

    def __enter__(self) -> EmbedCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _batched[T](seq: Sequence[T], size: int) -> list[list[T]]:
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


@dataclass
class OllamaEmbedder:
    """`bge-m3` embeddings over a local Ollama server. The only networked class
    in this module — everything else (`fts.py`, the pure parts of `build.py`)
    stays importable and testable without Ollama running.

    Distinct `embed_queries` / `embed_documents` entry points are the whole
    point (see module docstring); `embed_query` / `embed_document` are the
    single-item conveniences HyDE and the CLI evidence panel actually call.
    """

    model_tag: str = ""
    host: str = ""
    batch_size: int = 16
    max_attempts: int = 5
    cache_path: Path | None = None
    use_cache: bool = True
    query_prefix: str = DEFAULT_QUERY_PREFIX
    document_prefix: str = DEFAULT_DOCUMENT_PREFIX
    expected_dim: int | None = None
    _digest: str = field(default="", init=False, repr=False)
    _verified_dim: int | None = field(default=None, init=False, repr=False)
    _cache: EmbedCache | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        settings = get_settings()
        spec = settings.models.embedder
        self.model_tag = self.model_tag or spec.tag
        self.host = self.host or settings.ollama_host
        self._digest = spec.digest or ""
        if self.expected_dim is None:
            self.expected_dim = settings.retrieval.embedding_dim
        if self.cache_path is None:
            self.cache_path = settings.runs_dir / "index" / "embed_cache.jsonl"
        if self.use_cache:
            self._cache = EmbedCache(Path(self.cache_path))

    # -- identity -----------------------------------------------------------

    @property
    def digest(self) -> str:
        """The model digest, resolved from a live `ollama list` on first use
        and cached thereafter. Falls back to the tag (never empty) so it is
        always safe to use as a cache/state key, matching the convention in
        `carelite.eval.judge.client.ChatClient.digest`."""
        if not self._digest:
            self._digest = self._resolve_digest() or self.model_tag
        return self._digest

    def _resolve_digest(self) -> str:
        import ollama

        try:
            client = ollama.Client(host=self.host)
            resp = client.list()
        except Exception:
            return ""
        wanted = self.model_tag
        wanted_base = wanted.split(":")[0]
        for m in getattr(resp, "models", []) or []:
            name = getattr(m, "model", "") or (m.get("model", "") if isinstance(m, dict) else "")
            if name == wanted or name.split(":")[0] == wanted_base:
                digest = getattr(m, "digest", "") or (
                    m.get("digest", "") if isinstance(m, dict) else ""
                )
                return str(digest or "")
        return ""

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()

    def __enter__(self) -> OllamaEmbedder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- public API -----------------------------------------------------------

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def embed_document(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Instruction-prefixed for retrieval. Used by HyDE's query side and
        by the retrieval lane's framework-query construction."""
        return self._embed_prefixed([f"{self.query_prefix}{t}" for t in texts])

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Used by `build.py` for `chunk.embedding_text` / kb_entry text, and
        by HyDE for the hypothetical passage it generates (a HyDE passage
        stands in for a document, not a question)."""
        return self._embed_prefixed([f"{self.document_prefix}{t}" for t in texts])

    def verify_dimension(self) -> int:
        """Probe the live model once and confirm its output width matches
        `settings.retrieval.embedding_dim` (1024, frozen alongside
        `vector(1024)` in schema.sql). Raises `EmbedDimensionError` on
        mismatch rather than letting a later `INSERT` fail unexplained."""
        vec = self.embed_document("dimension probe")
        dim = len(vec)
        if dim != self.expected_dim:
            raise EmbedDimensionError(
                f"{self.model_tag} returned a {dim}-dim embedding; "
                f"settings.retrieval.embedding_dim (and the frozen vector(1024) "
                f"schema column) expect {self.expected_dim}. Do not work around "
                f"this — stop and reconcile the model or the schema."
            )
        self._verified_dim = dim
        return dim

    # -- internals -----------------------------------------------------------

    def _embed_prefixed(self, prefixed_texts: Sequence[str]) -> list[list[float]]:
        digest = self.digest
        results: list[list[float] | None] = [None] * len(prefixed_texts)
        misses: list[int] = []
        for i, text in enumerate(prefixed_texts):
            cached = self._cache.get(_cache_key(digest, text)) if self._cache is not None else None
            if cached is not None:
                results[i] = cached
            else:
                misses.append(i)

        for batch_idx in _batched(misses, self.batch_size):
            batch_texts = [prefixed_texts[i] for i in batch_idx]
            vectors = self._call_with_retry(batch_texts)
            for i, vec in zip(batch_idx, vectors, strict=True):
                self._check_dim(vec)
                results[i] = vec
                if self._cache is not None:
                    self._cache.put(_cache_key(digest, prefixed_texts[i]), vec)

        return [r for r in results if r is not None]  # mypy: all slots filled above

    def _check_dim(self, vec: Sequence[float]) -> None:
        if self.expected_dim is not None and len(vec) != self.expected_dim:
            raise EmbedDimensionError(
                f"{self.model_tag} returned a {len(vec)}-dim embedding; expected "
                f"{self.expected_dim} (settings.retrieval.embedding_dim / vector(1024))."
            )

    def _call_with_retry(self, batch: Sequence[str]) -> list[list[float]]:
        retryer = Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        return retryer(self._raw_embed, batch)

    def _raw_embed(self, batch: Sequence[str]) -> list[list[float]]:
        import ollama

        client = ollama.Client(host=self.host)
        resp = client.embed(model=self.model_tag, input=list(batch))
        embeddings = getattr(resp, "embeddings", None)
        if embeddings is None and isinstance(resp, dict):
            embeddings = resp.get("embeddings")
        if not embeddings or len(embeddings) != len(batch):
            raise EmbedError(
                f"{self.model_tag} returned {len(embeddings or [])} embeddings "
                f"for a batch of {len(batch)}"
            )
        return [[float(x) for x in e] for e in embeddings]
