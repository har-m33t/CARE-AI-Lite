"""carelite.retrieval.langchain_adapter — hybrid RAG built from LangChain parts,
kept as a second implementation to measure the native pipeline against.

    python -m carelite.retrieval.langchain_adapter --matched --legs --production

**What this is.** A dense retriever backed by `langchain_postgres`'s
`PGVectorStore`, a `rank_bm25`-backed `BM25Retriever` over the same rows, and
LangChain's `EnsembleRetriever` fusing them with reciprocal rank fusion. It
reads the existing `chunk` and `kb_entry` tables and the existing `bge-m3`
embeddings. It creates no tables, writes nothing, and re-embeds nothing.

**What this is not.** It is not the system's retrieval path. Everything in
`pipeline.py` — the adaptive router, framework query expansion, HyDE, the
graph leg, the cross-encoder rerank with evidence-tier weighting, and the CRAG
gate — is absent here, because none of it exists in the LangChain components
this module composes. All 939 generations in the database were produced by the
native stack and condition C continues to run it. The adapter is selected only
by `RetrievalFlags.langchain_adapter`, which defaults to `False`, and turning
it on is study-invalidating for the same reason `crag=False` is: it removes the
gate that stops evidence being injected into turns the corpus cannot address.

**Why it exists.** The project claims hybrid RAG over a PostgreSQL/pgvector
knowledge base using LangChain. Before this module, LangChain was not imported
anywhere. The claim is now true, and the honest form of it is comparative:
hybrid RAG via LangChain is implemented and *measured* against the native
pipeline, with the divergence reported rather than asserted away.

**Two departures from the design document's sketch, both forced by the
libraries as installed.**

`langchain_postgres.PGVector` — the class §4 W2 names — owns its storage. It
writes `langchain_pg_collection` and `langchain_pg_embedding` and can only
search rows it inserted itself, which would mean copying and re-embedding 587
rows into a second corpus. That is the one thing the package was told not to
do. `PGVectorStore` (the v2 class in the same distribution, 0.0.17) takes the
name of an *existing* table plus its id, content and embedding columns, so it
searches `chunk` and `kb_entry` in place. It is used instead.

`EnsembleRetriever` no longer lives in `langchain.retrievers`; under langchain
1.3 it moved to `langchain_classic.retrievers.ensemble`. Same class, same
weighted-RRF implementation.

**Embeddings are the project's own, deliberately.** `CareLiteEmbeddings` wraps
`carelite.index.embed.OllamaEmbedder` rather than using `langchain_ollama`'s
client. `embed.py` documents a measured correction — an instruction prefix on
the query side raised the cosine similarity between two unrelated one-word
queries from 0.52 to 0.84 on `bge-m3`, so both prefixes are empty here. A
second embedding client would be free to disagree about that, and any
divergence it caused would be indistinguishable from a divergence in
retrieval. One client removes the confound.

**The equivalence measurement.** `run_equivalence()` reports comparisons over
`EQUIVALENCE_QUERIES`, a fixed tuple committed in this file:

- *matched*: the adapter against `MATCHED_NATIVE_FLAGS`, a native
  configuration reduced stage-for-stage to what the adapter does — one raw
  query, dense plus lexical, RRF, nothing else. This asks whether LangChain's
  ensemble reproduces the native fusion.
- *legs*: the same comparison with one leg at a time, which is what makes the
  matched number interpretable.
- *production*: the adapter against the full native stack, which is what
  condition C actually runs. This asks how different the study would be if the
  adapter replaced it. A low number here is the expected and correct result;
  the two pipelines are not the same pipeline.

All report mean overlap@k at `k = settings.retrieval.rerank_top_n` and a mean
Spearman rank correlation over a deeper cut. None has a pass threshold, and
none should acquire one: a threshold turns a measurement into a target, and
tuning the adapter until it matched would be reverse-engineering a result to
fit a claim.

**Measured, 2026-09-01, over all 13 queries against the live database (471
chunks, 116 knowledge base entries, `bge-m3`).**

| comparison | mean overlap@4 | mean Spearman | n |
|---|---:|---:|---:|
| dense leg alone | 1.000 | 1.000 | 13/13 |
| lexical leg alone | 0.068 | -0.425 | 11/13 |
| matched (dense + lexical, RRF) | 0.231 | -0.011 | 13/13 |
| production (full native stack) | 0.019 | -0.566 | 13/13 |

**The dense leg is exact.** `PGVectorStore` searching `chunk` and `kb_entry`
in place returns the identical top-4, in the identical order, on every query
including the three off-domain ones. That is the result the adapter had to
produce to be believable at all: same table, same stored vectors, same
embedding client, same cosine operator, therefore the same ranking. It also
means the divergence below is not a defect in the adapter.

**The lexical leg is a different retriever wearing the same name.**
`BM25Retriever` and Postgres full-text search agree on 0.068 of the top-4 and
are *negatively* rank-correlated at -0.425. Two mechanisms, both verified
against the live database rather than inferred:

*Different query.* `query.build_queries` still extracts content words for the
lexical leg even with `query_expansion=False` — "I'm scared this is cancer and
nobody explains anything to me." reaches Postgres as `scared cancer nobody`.
The adapter passes the utterance through untouched, because a drop-in
LangChain retriever has no counterpart to that extraction. The comparison
holds the *input* constant and lets each implementation do what it does, which
is the right shape for "would LangChain give us the same thing"; it does mean
part of the gap is native query construction, not the retriever.

*Different matching.* `websearch_to_tsquery` stems, drops stop words, and ANDs
what remains, with `fusion.lexical_search` shortening the conjunction when it
matches nothing. `BM25Retriever` splits on whitespace, keeps stop words as
terms, and sums per-term scores, so a long chunk containing "the", "for" and
"I" can outrank one containing the single word that mattered. On two queries
the native leg returned nothing at all — no row satisfies the conjunction even
after backoff — and BM25 confidently returned four.

**Against the pipeline condition C actually runs, agreement is 0.019.** Twelve
of the thirteen queries share not one document in the top 4. This is the
expected result and it is worth stating in the direction that makes it useful:
the full stack embeds a HyDE guidance passage rather than the utterance, fuses
a third leg, and reorders everything through a cross-encoder weighted by
evidence tier, so the adapter is not a degraded version of it but a different
retrieval. The number measures what swapping the adapter in would cost the
study. Nothing in this module is a candidate for that swap.

**So the matched figure of 0.231 is one leg exact and one leg unrelated,**
which is the honest reading and is a fact about `rank_bm25` versus Postgres
FTS rather than about LangChain's ensemble. RRF then dilutes rather than
rescues: the fused list is half-built from a lexical ranking the native
pipeline would not have produced.

**None of this argues for changing the study.** It argues the opposite. The
native lexical leg was written against a measured failure — the carelite-index
lane found that framework vocabulary ("NURSE", "teach-back", "SPIKES") is
exact-match vocabulary a paraphrase-tolerant embedder does not rank highly —
and the stemmed, conjunctive, backing-off implementation is the one that
serves that purpose. A drop-in BM25 does not.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from carelite.config import get_settings
from carelite.db.connection import fetch_all
from carelite.retrieval.flags import RetrievalFlags, preset
from carelite.types import CRAGGrade, EvidenceTier, RetrievalTrace, RetrievedItem, Route, Theme

__all__ = [
    "EQUIVALENCE_COMMAND",
    "EQUIVALENCE_QUERIES",
    "MATCHED_NATIVE_FLAGS",
    "CareLiteEmbeddings",
    "CorpusRecord",
    "DenseTableRetriever",
    "EquivalenceReport",
    "LangChainRetrievalAdapter",
    "load_records",
    "main",
    "overlap_at_k",
    "run_equivalence",
    "spearman",
]

#: Reproducible command for the real comparison. Needs a live Postgres with the
#: corpus loaded and a live Ollama serving `bge-m3` and `gpt-oss:20b`; the
#: production half also loads the cross-encoder.
EQUIVALENCE_COMMAND = "python -m carelite.retrieval.langchain_adapter --matched --legs --production"


# ---------------------------------------------------------------------------
# The native configuration the adapter is matched against
# ---------------------------------------------------------------------------

#: The native pipeline reduced to as close to what the adapter does as its
#: flags allow: one utterance, a dense leg and a lexical leg, RRF, and no stage
#: the LangChain composition has no counterpart for. Comparing the adapter
#: against the *full* stack alone would report the cost of the missing reranker
#: and HyDE as though it were a difference between two fusion implementations.
#:
#: "As close as its flags allow" is exact, not a hedge. `query_expansion=False`
#: suppresses the three framework queries but `build_queries` still extracts
#: content words for the lexical leg, so the native side searches
#: `scared cancer nobody` where the adapter searches the whole sentence. There
#: is no flag for that and inventing one would be a code change made to move a
#: number. It is disclosed in the module docstring instead.
MATCHED_NATIVE_FLAGS = RetrievalFlags().with_(
    router=False,
    query_expansion=False,
    hyde=False,
    dense=True,
    lexical=True,
    graph=False,
    rerank=False,
    tier_weighting=False,
    crag=False,
    drop_boilerplate=False,
    metadata_filter=False,
    langchain_adapter=False,
)


#: The fixed query set the reported numbers are computed over.
#:
#: Written out here rather than sampled from the scenario bank on purpose. A
#: comparison whose query set can change is not reproducible, and the scenario
#: bank is another lane's file. Ten on-domain turns span the seven themes and
#: both a plain-language and an equity-relevant register; the last three are
#: `ablation.OFF_DOMAIN_TURNS`, included because the two implementations have
#: the most room to disagree exactly where neither has anything good to return.
EQUIVALENCE_QUERIES: tuple[str, ...] = (
    "I'm scared this is cancer and nobody explains anything to me.",
    "Can you tell me again what the medicine is for? I forgot what you said.",
    "My daughter usually translates for me but she couldn't come today.",
    "I don't understand what any of these words on the paper mean.",
    "Do I really have to decide today, or can I think about it?",
    "Every doctor tells me something different and I don't know who to believe.",
    "I've been feeling like there's no point in any of this treatment.",
    "What are the actual chances this works? Give me a real number.",
    "The last clinic made me feel like I was wasting their time.",
    "How do I know I'm doing the injections right at home?",
    "How do I replace the oil filter on a 2003 Honda Civic?",
    "What is the tax treatment of a Roth IRA conversion?",
    "What were the main causes of the fall of the Western Roman Empire?",
)


# ---------------------------------------------------------------------------
# Corpus records
# ---------------------------------------------------------------------------


def _coerce_theme(value: str | None) -> Theme | None:
    try:
        return Theme(value) if value else None
    except ValueError:
        return None


def _coerce_tier(value: str | None) -> EvidenceTier | None:
    try:
        return EvidenceTier(value) if value else None
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    """One `chunk` or `kb_entry` row, with the provenance the evidence panel needs.

    `index_text` and `text` differ for chunks. The native lexical leg searches
    `chunk.tsv`, a generated column over `contextual_prefix || ' ' || text`, so
    BM25 must be given the same string to be searching the same corpus; but the
    text placed in a prompt is `chunk.text` alone, because the contextual
    prefix is LLM-generated and `schema.sql` marks it a poisoning vector.
    """

    ref_id: str
    kind: str
    index_text: str
    text: str
    paper_id: str | None = None
    citation: str | None = None
    theme: str | None = None
    evidence_tier: str | None = None

    def as_document(self) -> Document:
        """The canonical LangChain document for this row.

        Both legs emit this same object for a given `ref_id`, which is what
        lets `EnsembleRetriever` deduplicate on `id_key="ref_id"` rather than
        on page content — the dense leg's content comes from one column and
        BM25's from the indexed composite, so content-based deduplication
        would treat one row as two.
        """
        return Document(
            id=self.ref_id,
            page_content=self.index_text,
            metadata={"ref_id": self.ref_id, "kind": self.kind, "paper_id": self.paper_id},
        )

    def as_item(self, *, score: float, rank: int | None = None) -> RetrievedItem:
        return RetrievedItem(
            ref_id=self.ref_id,
            kind=self.kind,
            text=self.text,
            score=score,
            dense_rank=rank,
            theme=_coerce_theme(self.theme),
            evidence_tier=_coerce_tier(self.evidence_tier),
            paper_id=self.paper_id,
            citation=self.citation,
        )


_CHUNK_SQL = """
SELECT c.chunk_id AS ref_id,
       coalesce(c.contextual_prefix, '') || ' ' || c.text AS index_text,
       c.text,
       c.paper_id,
       p.apa_citation AS citation,
       p.evidence_tier
FROM chunk c JOIN paper p USING (paper_id)
"""

_KB_SQL = """
SELECT k.entry_id AS ref_id,
       k.finding || ' ' || k.practical_takeaway || ' ' || k.example_behavior AS index_text,
       k.finding || ' ' || k.practical_takeaway || ' ' || k.example_behavior AS text,
       k.theme,
       k.evidence_tier,
       (SELECT s.paper_id FROM kb_entry_source s
         WHERE s.entry_id = k.entry_id ORDER BY s.paper_id LIMIT 1) AS paper_id,
       (SELECT p.apa_citation FROM kb_entry_source s
          JOIN paper p USING (paper_id)
         WHERE s.entry_id = k.entry_id ORDER BY s.paper_id LIMIT 1) AS citation
FROM kb_entry k
"""


def load_records(kinds: Sequence[str] = ("chunk", "kb_entry")) -> dict[str, CorpusRecord]:
    """Every row of the corpus, keyed by `ref_id`.

    587 rows on this database, so loading them whole costs one query per table
    and a few megabytes. BM25 needs the full text set in memory regardless —
    that is what `rank_bm25` is — and holding the same map for the dense leg
    means both legs resolve a hit to the same record, with the same citation
    and the same evidence tier, from the same read.
    """
    out: dict[str, CorpusRecord] = {}
    if "chunk" in kinds:
        for row in fetch_all(_CHUNK_SQL):
            out[str(row["ref_id"])] = CorpusRecord(
                ref_id=str(row["ref_id"]),
                kind="chunk",
                index_text=str(row["index_text"]),
                text=str(row["text"]),
                paper_id=_opt(row["paper_id"]),
                citation=_opt(row["citation"]),
                theme=None,
                evidence_tier=_opt(row["evidence_tier"]),
            )
    if "kb_entry" in kinds:
        for row in fetch_all(_KB_SQL):
            out[str(row["ref_id"])] = CorpusRecord(
                ref_id=str(row["ref_id"]),
                kind="kb_entry",
                index_text=str(row["index_text"]),
                text=str(row["text"]),
                paper_id=_opt(row["paper_id"]),
                citation=_opt(row["citation"]),
                theme=_opt(row["theme"]),
                evidence_tier=_opt(row["evidence_tier"]),
            )
    return out


def _opt(value: Any) -> str | None:
    return str(value) if value is not None else None


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class CareLiteEmbeddings(Embeddings):
    """LangChain `Embeddings` over the project's own `OllamaEmbedder`.

    Not `langchain_ollama.OllamaEmbeddings`: see the module docstring. The
    asymmetric document/query split is preserved, so a hypothetical passage
    embedded through this class takes the same code path it takes natively.
    """

    def __init__(self, embedder: Any | None = None) -> None:
        if embedder is None:
            from carelite.index.embed import OllamaEmbedder

            embedder = OllamaEmbedder()
            self._owned = True
        else:
            self._owned = False
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return list(self._embedder.embed_documents(list(texts)))

    def embed_query(self, text: str) -> list[float]:
        return list(self._embedder.embed_query(text))

    def close(self) -> None:
        if self._owned:
            self._embedder.close()


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------


class DenseTableRetriever(BaseRetriever):
    """A LangChain retriever over one existing pgvector table.

    Thin by design: `PGVectorStore` does the cosine search against the live
    HNSW index, and this wrapper only maps each hit back to its `CorpusRecord`
    so both legs emit the identical canonical document. A hit with no record —
    which should not happen, and would mean the table changed under the
    adapter — is dropped rather than emitted with empty provenance.
    """

    store: Any
    records: dict[str, Any]
    top_k: int = 20

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        docs = self.store.similarity_search(query, k=self.top_k)
        out: list[Document] = []
        for doc in docs:
            ref_id = doc.id or str(doc.metadata.get("ref_id", ""))
            record = self.records.get(ref_id)
            if record is not None:
                out.append(record.as_document())
        return out


def build_dense_retriever(
    table: str,
    *,
    id_column: str,
    content_column: str,
    records: dict[str, CorpusRecord],
    embeddings: Embeddings,
    engine: Any,
    top_k: int,
) -> DenseTableRetriever:
    """A `PGVectorStore` bound to an existing table, wrapped as a retriever.

    `content_column` is a real column, which is why `kb_entry` binds `finding`
    rather than the three-column composite the native dense leg selects: a
    store reads one column and the composite is computed. It changes nothing
    about the ranking, which comes from `kb_entry.embedding` either way, and
    the text that reaches a prompt is rehydrated from the record map.
    """
    from langchain_postgres import PGVectorStore

    store = PGVectorStore.create_sync(
        engine=engine,
        embedding_service=embeddings,
        table_name=table,
        id_column=id_column,
        content_column=content_column,
        embedding_column="embedding",
        metadata_columns=[],
    )
    return DenseTableRetriever(store=store, records=records, top_k=top_k)


def build_bm25_retriever(records: Sequence[CorpusRecord], *, top_k: int) -> BaseRetriever:
    """`rank_bm25` over the same rows the lexical leg searches in Postgres.

    A real divergence to hold in mind when reading the numbers: the native
    lexical leg is Postgres full-text search over a `tsvector`, which stems,
    drops stop words, and ANDs the remaining terms with a documented backoff
    when that conjunction matches nothing. `BM25Retriever` tokenises on
    whitespace and scores every term independently. They are both "the lexical
    leg" and they are not the same function, so a rank disagreement between
    them is expected and is a property of the two libraries rather than a bug
    in either.
    """
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents([r.as_document() for r in records])
    retriever.k = top_k
    return retriever


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class LangChainRetrievalAdapter:
    """Dense + BM25 + `EnsembleRetriever`, returning native `RetrievedItem`s.

    Constructed with its retrievers and record map injected, so the ensemble
    wiring is testable without a database. `build()` is the live constructor.
    """

    def __init__(
        self,
        *,
        records: dict[str, CorpusRecord],
        retrievers: Sequence[BaseRetriever],
        rrf_k: int | None = None,
        top_n: int | None = None,
        engine: Any | None = None,
        embeddings: CareLiteEmbeddings | None = None,
    ) -> None:
        from langchain_classic.retrievers.ensemble import EnsembleRetriever

        settings = get_settings().retrieval
        self.records = records
        self.rrf_k = rrf_k if rrf_k else settings.rrf_k
        self.top_n = top_n if top_n else settings.rerank_top_n
        self._engine = engine
        self._embeddings = embeddings
        # `id_key="ref_id"` rather than content-based deduplication, and
        # `c=rrf_k` so the fusion constant is the frozen contract's 60 — the
        # same value `fusion.rrf_fuse` uses. LangChain's weighted RRF scores
        # `weight / (rank + c)`; equal weights make that the native
        # `1 / (k + rank)` scaled by a constant, and therefore the same
        # ordering. The weights are written out rather than left to the
        # default because "the default happens to be uniform" is not something
        # a comparison should rest on.
        # `list[Any]`: `EnsembleRetriever.retrievers` is declared
        # `list[Runnable[str, list[Document]]]`, and `list` is invariant, so a
        # `list[BaseRetriever]` is rejected despite every element satisfying it.
        legs: list[Any] = list(retrievers)
        self.ensemble = EnsembleRetriever(
            retrievers=legs,
            weights=[1.0 / len(legs)] * len(legs) if legs else [],
            c=self.rrf_k,
            id_key="ref_id",
        )

    @classmethod
    def build(
        cls,
        *,
        flags: RetrievalFlags | None = None,
        embedder: Any | None = None,
    ) -> LangChainRetrievalAdapter:
        """Live constructor: opens a `PGEngine`, binds both tables, loads BM25.

        Call `close()` when done. The engine holds an asyncpg pool and a
        background event loop thread.
        """
        from langchain_postgres import PGEngine
        from sqlalchemy.engine import make_url

        flags = flags or RetrievalFlags()
        records = load_records()
        chunks = [r for r in records.values() if r.kind == "chunk"]
        kb = [r for r in records.values() if r.kind == "kb_entry"]

        embeddings = CareLiteEmbeddings(embedder)
        # `PGEngine` drives SQLAlchemy's async layer, which needs the asyncpg
        # driver named in the URL. The credentials are the same ones
        # `carelite.db.connection` uses; nothing new is read or stored.
        url = make_url(get_settings().database_url).set(drivername="postgresql+asyncpg")
        engine = PGEngine.from_connection_string(url)

        retrievers: list[BaseRetriever] = []
        if flags.dense:
            retrievers.append(
                build_dense_retriever(
                    "chunk",
                    id_column="chunk_id",
                    content_column="text",
                    records=records,
                    embeddings=embeddings,
                    engine=engine,
                    top_k=flags.dense_top_k,
                )
            )
            retrievers.append(
                build_dense_retriever(
                    "kb_entry",
                    id_column="entry_id",
                    content_column="finding",
                    records=records,
                    embeddings=embeddings,
                    engine=engine,
                    top_k=flags.dense_top_k,
                )
            )
        if flags.lexical:
            # One BM25 index per table, mirroring the native rule that `chunk`
            # and `kb_entry` never share a ranked list: a 512-token chunk
            # outscores a one-sentence curated entry on any term-frequency
            # measure, and a merged list starves the knowledge base.
            retrievers.append(build_bm25_retriever(chunks, top_k=flags.lexical_top_k))
            retrievers.append(build_bm25_retriever(kb, top_k=flags.lexical_top_k))

        return cls(
            records=records,
            retrievers=retrievers,
            rrf_k=flags.rrf_k,
            top_n=flags.rerank_top_n,
            engine=engine,
            embeddings=embeddings,
        )

    def retrieve(self, utterance: str, *, top_k: int | None = None) -> list[RetrievedItem]:
        """Fused top-n for one utterance.

        The score is derived from the final ensemble rank rather than read off
        the fusion: `EnsembleRetriever` sorts by its RRF score and then
        discards it, returning bare documents. The value is therefore
        order-consistent and comparable within one call, and is not the same
        quantity as `fusion.rrf_fuse`'s score.
        """
        limit = top_k if top_k else self.top_n
        docs = self.ensemble.invoke(utterance)
        items: list[RetrievedItem] = []
        for doc in docs:
            record = self.records.get(str(doc.metadata.get("ref_id", "")))
            if record is None:
                continue
            rank = len(items) + 1
            items.append(record.as_item(score=1.0 / (self.rrf_k + rank), rank=rank))
            if len(items) >= limit:
                break
        return items

    def result(self, utterance: str, *, top_k: int | None = None) -> Any:
        """A `RetrievalResult` shaped like `pipeline.retrieve_detailed`'s.

        Same return type so `graph.retrieve` and the CLI evidence panel work
        unchanged when the flag selects this path. The trace records the route
        as informational and the grade as relevant because neither the router
        nor the gate ran — which is the honest reading, and is exactly why the
        note below is attached to every result this path produces.
        """
        from carelite.retrieval.pipeline import RetrievalResult

        items = self.retrieve(utterance, top_k=top_k)
        trace = RetrievalTrace(
            route=Route.INFORMATIONAL,
            queries=[utterance],
            retrieved=items,
            crag_grade=CRAGGrade.RELEVANT,
            fell_back_to_b=False,
        )
        return RetrievalResult(
            trace=trace,
            flags=RetrievalFlags().with_(langchain_adapter=True),
            leg_notes=[
                "langchain adapter: dense + BM25 + EnsembleRetriever only — no router, "
                "no query expansion, no HyDE, no graph leg, no rerank, and no CRAG gate, "
                "so nothing here can decline to answer a turn the corpus cannot address"
            ],
            leg_hits={"langchain_ensemble": len(items)},
        )

    def close(self) -> None:
        if self._embeddings is not None:
            self._embeddings.close()
        if self._engine is not None:
            self._engine._run_as_sync(self._engine.close())


# ---------------------------------------------------------------------------
# The two statistics
# ---------------------------------------------------------------------------


def overlap_at_k(native: Sequence[str], adapter: Sequence[str], *, k: int) -> float | None:
    """Fraction of the top-`k` cut the two implementations agree on.

    The denominator is `min(k, len(native), len(adapter))`, not `k`. Charging
    a pipeline for returning three documents when only three exist would
    conflate "these two disagree" with "there was little to retrieve", and the
    off-domain turns in the query set exist precisely to produce short lists.
    Returns `None` when neither side returned anything: two empty lists are
    not agreement, and averaging them in as 1.0 would inflate the headline.
    """
    a, b = list(native)[:k], list(adapter)[:k]
    denominator = min(k, len(a), len(b))
    if denominator == 0:
        return None
    return len(set(a) & set(b)) / denominator


def spearman(native: Sequence[str], adapter: Sequence[str], *, depth: int) -> float | None:
    """Rank correlation over the union of the two top-`depth` lists.

    A document present in one list and absent from the other is assigned rank
    `depth + 1` rather than dropped. Dropping it would compute the correlation
    only over documents both pipelines found, which is the subset where they
    agree by construction — the statistic would rise as the two diverged.

    Returns `None` when the union holds fewer than two documents, or when
    either rank vector has no variance, because the coefficient is undefined
    there rather than zero.
    """
    a, b = list(native)[:depth], list(adapter)[:depth]
    union = sorted(set(a) | set(b))
    if len(union) < 2:
        return None
    absent = float(depth + 1)
    xs = [float(a.index(r) + 1) if r in a else absent for r in union]
    ys = [float(b.index(r) + 1) if r in b else absent for r in union]
    return _pearson(xs, ys)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return cov / math.sqrt(vx * vy)


@dataclass
class EquivalenceReport:
    """Per-query agreement between the two implementations, plus its means.

    Rows whose statistic is `None` are excluded from the mean rather than
    counted as zero, and `n_scored_*` says how many survived — a mean over
    four of thirteen queries is a different claim from a mean over thirteen.
    """

    label: str
    k: int
    depth: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_queries(self) -> int:
        return len(self.rows)

    def _values(self, key: str) -> list[float]:
        return [float(r[key]) for r in self.rows if r.get(key) is not None]

    @property
    def n_scored_overlap(self) -> int:
        return len(self._values("overlap_at_k"))

    @property
    def n_scored_spearman(self) -> int:
        return len(self._values("spearman"))

    @property
    def mean_overlap(self) -> float | None:
        values = self._values("overlap_at_k")
        return sum(values) / len(values) if values else None

    @property
    def mean_spearman(self) -> float | None:
        values = self._values("spearman")
        return sum(values) / len(values) if values else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "k": self.k,
            "depth": self.depth,
            "n_queries": self.n_queries,
            "mean_overlap_at_k": _round(self.mean_overlap),
            "n_scored_overlap": self.n_scored_overlap,
            "mean_spearman": _round(self.mean_spearman),
            "n_scored_spearman": self.n_scored_spearman,
            "rows": self.rows,
            "notes": self.notes,
        }

    def format_markdown(self) -> str:
        lines = [
            f"### {self.label} — overlap@{self.k}, Spearman over top-{self.depth}",
            "",
            f"mean overlap@{self.k} = {_fmt(self.mean_overlap)} "
            f"(n = {self.n_scored_overlap}/{self.n_queries}); "
            f"mean Spearman = {_fmt(self.mean_spearman)} "
            f"(n = {self.n_scored_spearman}/{self.n_queries})",
            "",
            f"| query | overlap@{self.k} | Spearman | n native | n adapter |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in self.rows:
            query = str(row["query"])
            shown = query if len(query) <= 60 else query[:57] + "..."
            lines.append(
                f"| {shown} | {_fmt(row.get('overlap_at_k'))} | {_fmt(row.get('spearman'))} "
                f"| {row.get('n_native')} | {row.get('n_adapter')} |"
            )
        for note in self.notes:
            lines.extend(["", note])
        return "\n".join(lines)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


# ---------------------------------------------------------------------------
# The comparison run
# ---------------------------------------------------------------------------


def compare(
    adapter: LangChainRetrievalAdapter,
    *,
    native_flags: RetrievalFlags,
    label: str,
    queries: Sequence[str] = EQUIVALENCE_QUERIES,
    k: int | None = None,
    depth: int = 20,
    embedder: Any | None = None,
    generator: Any | None = None,
    grader_client: Any | None = None,
    reranker: Any | None = None,
) -> EquivalenceReport:
    """Run both implementations over `queries` and report the two statistics.

    The native side goes through `retrieve_detailed`, the same entry point
    condition C uses, so the comparison is against the real pipeline rather
    than a reimplementation of it.

    **The native side is run at `rerank_top_n = depth`, and that is not
    cosmetic.** Left at its production value of 4, the native list is four
    documents long, so sixteen of the adapter's twenty are absent from it and
    take the `depth + 1` rank; the correlation then measures the length
    mismatch rather than the ordering. Measured: the dense-only comparison,
    where the two implementations agree on the top 4 *perfectly*, reported
    Spearman 0.698 under the truncated native list and 1.000 once both sides
    were read at the same depth. Widening the cut does not change the top-`k`
    prefix — neither `candidates[:top_n]` nor the reranker's `out[:top_n]`
    reorders anything — so `overlap_at_k` is unaffected.

    One consequence worth stating where the CRAG gate is on: it grades `depth`
    passages here rather than 4, so the `fell_back_to_b` this function records
    is the verdict on a twenty-passage context and is not necessarily the
    verdict condition C would reach on its four.
    """
    from carelite.retrieval.pipeline import retrieve_detailed

    settings = get_settings().retrieval
    k = k if k else settings.rerank_top_n
    report = EquivalenceReport(label=label, k=k, depth=depth)
    deep_flags = native_flags.with_(rerank_top_n=depth)

    for query in queries:
        native = retrieve_detailed(
            query,
            flags=deep_flags,
            embedder=embedder,
            generator=generator,
            grader_client=grader_client,
            reranker=reranker,
        )
        # `retrieved + rejected` rather than `retrieved`: a CRAG fallback
        # empties `retrieved` by design, and a turn the gate rejected still
        # has a ranking worth correlating. The two lists are disjoint and the
        # rejected set keeps its order, so this is the full ranked candidate
        # list either way.
        native_ids = [i.ref_id for i in native.trace.retrieved] + [
            i.ref_id for i in native.rejected
        ]
        adapter_ids = [i.ref_id for i in adapter.retrieve(query, top_k=depth)]
        report.rows.append(
            {
                "query": query,
                "overlap_at_k": overlap_at_k(native_ids, adapter_ids, k=k),
                "spearman": spearman(native_ids, adapter_ids, depth=depth),
                "n_native": len(native_ids[:k]),
                "n_adapter": len(adapter_ids[:k]),
                "fell_back_to_b": native.trace.fell_back_to_b,
                "route": native.trace.route.value,
            }
        )
    return report


def compare_legs(
    *,
    queries: Sequence[str] = EQUIVALENCE_QUERIES,
    depth: int = 20,
    embedder: Any | None = None,
) -> dict[str, EquivalenceReport]:
    """The same comparison run one leg at a time.

    Without this the combined number is uninterpretable: a low overlap could
    mean the adapter is wrong, the fusion differs, or one leg differs. Running
    dense alone and lexical alone against native configurations reduced the
    same way attributes it. Each comparison builds its own adapter, because a
    leg is switched off by leaving its retriever out of the ensemble.
    """
    out: dict[str, EquivalenceReport] = {}
    cases = (
        ("dense", preset("LCHAIN").with_(lexical=False), MATCHED_NATIVE_FLAGS.with_(lexical=False)),
        ("lexical", preset("LCHAIN").with_(dense=False), MATCHED_NATIVE_FLAGS.with_(dense=False)),
    )
    for name, adapter_flags, native_flags in cases:
        adapter = LangChainRetrievalAdapter.build(flags=adapter_flags, embedder=embedder)
        try:
            out[name] = compare(
                adapter,
                native_flags=native_flags,
                label=f"{name} leg alone",
                queries=queries,
                depth=depth,
                embedder=embedder,
            )
        finally:
            adapter.close()
    out["dense"].notes.append(
        "PGVectorStore against the existing `chunk` and `kb_entry` tables versus "
        "`fusion.dense_search` against the same rows, with one shared embedding client."
    )
    out["lexical"].notes.append(
        "`rank_bm25` over the indexed text versus Postgres full-text search over the "
        "`tsv` generated column built from the same text."
    )
    return out


def run_equivalence(
    *,
    matched: bool = True,
    production: bool = False,
    legs: bool = False,
    queries: Sequence[str] = EQUIVALENCE_QUERIES,
    depth: int = 20,
) -> dict[str, EquivalenceReport]:
    """Every comparison, sharing one adapter, one embedder and one model client.

    Reproduce with::

        python -m carelite.retrieval.langchain_adapter --matched --legs --production

    Needs a live Postgres holding the corpus and a live Ollama serving
    `bge-m3`; `--production` additionally needs `gpt-oss:20b` for the CRAG
    gate and loads the cross-encoder.
    """
    from carelite.index.embed import OllamaEmbedder

    settings = get_settings()
    embedder = OllamaEmbedder()
    adapter = LangChainRetrievalAdapter.build(embedder=embedder)
    out: dict[str, EquivalenceReport] = {}
    generator: Any = None
    judge: Any = None
    reranker: Any = None
    try:
        if matched:
            out["matched"] = compare(
                adapter,
                native_flags=MATCHED_NATIVE_FLAGS,
                label="adapter vs native, matched stage-for-stage",
                queries=queries,
                depth=depth,
                embedder=embedder,
            )
            out["matched"].notes.append(
                "Native side: "
                + MATCHED_NATIVE_FLAGS.summary()
                + ". This is the comparison that isolates the fusion."
            )
        if legs:
            out.update(compare_legs(queries=queries, depth=depth, embedder=embedder))
        if production:
            from carelite.retrieval.llm import LLMClient
            from carelite.retrieval.rerank import get_reranker

            generator = LLMClient()
            judge = LLMClient(model_tag=settings.models.judge.tag)
            reranker = get_reranker()
            out["production"] = compare(
                adapter,
                native_flags=RetrievalFlags(),
                label="adapter vs the full native stack (what condition C runs)",
                queries=queries,
                depth=depth,
                embedder=embedder,
                generator=generator,
                grader_client=judge,
                reranker=reranker,
            )
            out["production"].notes.append(
                "Native side: "
                + RetrievalFlags().summary()
                + ". A low number here is the expected result, not a defect: the "
                "adapter has no HyDE, no cross-encoder rerank and no CRAG gate, so "
                "this measures what the study would lose, not how well two "
                "implementations of one algorithm agree."
            )
    finally:
        adapter.close()
        embedder.close()
        for client in (generator, judge):
            if client is not None:
                client.close()
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure the LangChain retrieval adapter against the native pipeline."
    )
    parser.add_argument(
        "--matched", action="store_true", help="compare against the stage-matched native config"
    )
    parser.add_argument(
        "--production", action="store_true", help="compare against the full native stack"
    )
    parser.add_argument(
        "--legs", action="store_true", help="also compare the dense and lexical legs separately"
    )
    parser.add_argument("--depth", type=int, default=20, help="rank-correlation depth (default 20)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = parser.parse_args(argv)

    matched = args.matched or not (args.production or args.legs)
    reports = run_equivalence(
        matched=matched, production=args.production, legs=args.legs, depth=args.depth
    )
    if args.json:
        print(json.dumps({k: r.to_dict() for k, r in reports.items()}, indent=2))
    else:
        for report in reports.values():
            print(report.format_markdown())
            print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
