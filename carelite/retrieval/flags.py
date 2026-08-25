"""carelite.retrieval.flags — every component switch, in one place.

The lane brief requires that "each component is independently switchable so
the ablation is real" and that "every component reads its flag from config so
ablations are configuration, not code edits".

`carelite/config.py` is a **frozen contract** and carries only one retrieval
switch (`hyde_enabled`). Rather than request an amendment mid-wave for nine
more booleans, this module layers a `RetrievalFlags` object *over* the frozen
settings:

- `hyde` defaults to `settings.retrieval.hyde_enabled` (the frozen field is
  the authority; this module never contradicts it),
- every other switch defaults to "full stack on",
- all of them can be overridden by environment variable
  (`CARELITE_RETRIEVAL_<NAME>`), so an ablation run is still *configuration*
  rather than a code edit even from a shell,
- and `PRESETS` names the R0-R9 + LC rows of the ablation table as flag sets.

If the foundation lane later moves these onto `config.Retrieval`, this module
becomes a thin adapter and no calling code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

from carelite.config import get_settings

__all__ = [
    "ENV_PREFIX",
    "PRESETS",
    "RetrievalFlags",
    "preset",
]

#: Environment override prefix, matching `Settings.model_config`'s own
#: `CARELITE_` convention with a `RETRIEVAL_` segment so these never collide
#: with a frozen `Settings` field name.
ENV_PREFIX = "CARELITE_RETRIEVAL_"

_TRUE = {"1", "true", "yes", "on", "y", "t"}
_FALSE = {"0", "false", "no", "off", "n", "f"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(f"{ENV_PREFIX}{name.upper()}")
    if raw is None:
        return default
    lowered = raw.strip().casefold()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(
        f"{ENV_PREFIX}{name.upper()}={raw!r} is not a boolean; use one of {sorted(_TRUE | _FALSE)}"
    )


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(f"{ENV_PREFIX}{name.upper()}")
    return default if raw is None else float(raw)


@dataclass(frozen=True, slots=True)
class RetrievalFlags:
    """One immutable switch-set describing a retrieval configuration.

    Constructed with `RetrievalFlags()` for "whatever config says" (the
    production default), or with `preset("R4")` for a named ablation row.
    Frozen because a pipeline run must not be able to mutate the
    configuration it is being measured under; use `.with_(...)` to derive.
    """

    # -- pipeline stages ----------------------------------------------------
    router: bool = True
    """Adaptive routing. Off means every turn retrieves, including
    emotional-only ones — which is exactly the confound R7 exists to size."""

    query_expansion: bool = True
    """Build `n_framework_queries` framework-language queries. Off means the
    raw patient utterance is the only query, i.e. the naive baseline."""

    hyde: bool = True
    """Generate and embed a hypothetical guidance passage. Default is
    overridden from the frozen `settings.retrieval.hyde_enabled`."""

    dense: bool = True
    """pgvector cosine leg. There is no meaningful configuration with every
    leg off; `validate()` enforces at least one."""

    lexical: bool = True
    """Postgres FTS (BM25-ish `ts_rank_cd`) leg."""

    graph: bool = True
    """`graph_edge` traversal leg."""

    rerank: bool = True
    """Cross-encoder rerank to `settings.retrieval.rerank_top_n`. When off,
    `carelite.retrieval.rerank` is never imported, so sentence-transformers
    and torch are never loaded (see `pipeline.py`)."""

    tier_weighting: bool = True
    """Weight rerank scores by `evidence_tier` so strong evidence outranks
    emerging at comparable relevance."""

    crag: bool = True
    """The corrective-RAG relevance gate. Off is a *study-invalidating*
    configuration and exists only so R8 can measure what CRAG is worth;
    never ship it off."""

    drop_boilerplate: bool = True
    """Discard publication apparatus (funding statements, copyright blocks,
    author-affiliation front matter) before reranking. See `filters.py`."""

    crag_filter_items: bool = False
    """Keep only the passages CRAG judged useful, instead of all-or-nothing.

    `LLMGrader` already records *which* passages help this turn
    (`GradeReport.relevant_ids`); the pipeline currently uses only the
    aggregate verdict and keeps all `rerank_top_n` on any non-NONE grade.
    Measured over 20 train turns plus 3 off-domain: of 60 passages placed in
    condition C prompts, 45 were judged useful and **15 (25%) were judged
    useless and injected anyway**. Per-turn useful counts were 2, 3 or 4 — a
    real continuum that the all-or-nothing rule flattens into the 4-or-0
    distribution seen downstream.

    **Default off, deliberately.** Turning it on changes what condition C puts
    in its prompt, and cells have already been generated against the current
    behaviour. Enabling it is a study decision, not a lane decision.

    **It also makes the context-precision gate partly circular**, which must be
    disclosed wherever that number is reported: filtering with the judge and
    then measuring precision with the same judge family on a closely related
    question guarantees the number rises. The generation-quality argument for
    filtering is sound on its own; the precision improvement it produces is
    not independent evidence."""

    metadata_filter: bool = True
    """Apply theme / encounter-phase / equity filters to the kb_entry legs."""

    # -- knobs whose defaults live in the frozen contract -------------------
    rrf_k: int = 0
    dense_top_k: int = 0
    lexical_top_k: int = 0
    graph_top_k: int = 0
    rerank_top_n: int = 0
    n_framework_queries: int = 0
    crag_relevance_threshold: float = 0.0

    crag_ambiguous_ratio: float = 0.6
    """Fraction of `crag_relevance_threshold` below which a grade drops from
    AMBIGUOUS to NONE. See `crag.py` for the measurement behind it."""

    use_llm_router: bool = False
    """Ask the generator model to classify the turn instead of the
    deterministic lexicon. Off by default: an LLM in the router adds
    per-turn variance to a controlled comparison (v3 §14)."""

    use_llm_crag: bool = True
    """Grade retrieved context with the judge-family model rather than a score
    threshold. **On by default, unlike `use_llm_router`, and the asymmetry is
    empirical rather than stylistic.**

    Every score this pipeline produces was measured against the live corpus and
    none of them separates an on-domain turn from an off-domain one — dense
    cosine, cross-encoder-on-the-utterance, and HyDE-passage cosine all fail,
    for the structural reason set out at length in `crag.py`. A score-threshold
    gate therefore passes an off-domain turn straight through: measured, it
    rejected 0 of 6 off-domain turns while the LLM grader rejected 6 of 6.

    Since a gate that cannot reject is not a gate, and Condition C scoring
    below Condition B on unaddressable turns is precisely the confound CRAG
    exists to prevent, the more expensive grader is the default. Determinism is
    pinned by temperature 0, a fixed seed, and a persistent prompt-hash cache;
    the cost is roughly 5-15s per uncached turn on `gpt-oss:20b`."""

    long_context: bool = False
    """Condition LC-sample: no query-dependent retrieval. Handled by
    `ablation.py`, not by `pipeline.py`.

    Not "stuff the whole corpus" — the corpus does not fit (D7). It is a fixed
    round-robin sample across all papers at a pinned seed. Any selection rule
    is a form of retrieval, so this row asks whether query-dependent selection
    beats a fixed context, which is a different question from the one v3 §3
    posed. See `ablation.lc_sample`."""

    label: str = ""
    """Ablation row name (`"R0"`…`"R9"`, `"LC"`). Free-form; carried into the
    ablation table."""

    note: str = ""

    _explicit: frozenset[str] = field(default=frozenset(), repr=False, compare=False)

    def __post_init__(self) -> None:
        settings = get_settings().retrieval
        # Frozen-contract defaults, applied only where the caller left the
        # sentinel. object.__setattr__ because the dataclass is frozen.
        defaults: dict[str, Any] = {
            "rrf_k": settings.rrf_k,
            "dense_top_k": settings.dense_top_k,
            "lexical_top_k": settings.lexical_top_k,
            "graph_top_k": settings.graph_top_k,
            "rerank_top_n": settings.rerank_top_n,
            "n_framework_queries": settings.n_framework_queries,
            "crag_relevance_threshold": settings.crag_relevance_threshold,
        }
        for name, value in defaults.items():
            if not getattr(self, name):
                object.__setattr__(self, name, value)

        # `hyde` is the one boolean the frozen contract already owns.
        if "hyde" not in self._explicit:
            object.__setattr__(self, "hyde", settings.hyde_enabled)

        # Environment overrides win over everything except an explicit kwarg,
        # so a preset can still be nudged from a shell without a code edit.
        for name in _BOOL_FIELDS:
            if name in self._explicit:
                continue
            object.__setattr__(self, name, _env_bool(name, getattr(self, name)))
        for name in _FLOAT_FIELDS:
            if name in self._explicit:
                continue
            object.__setattr__(self, name, _env_float(name, getattr(self, name)))

        self.validate()

    def validate(self) -> None:
        if not (self.dense or self.lexical or self.graph or self.long_context):
            raise ValueError(
                "at least one retrieval leg (dense / lexical / graph) must be enabled; "
                "a configuration with all three off retrieves nothing and is not an "
                "ablation. The one exception is `long_context=True`, which is defined "
                "as the no-retrieval baseline."
            )
        if not 0.0 <= self.crag_relevance_threshold <= 1.0:
            raise ValueError("crag_relevance_threshold must be in [0, 1]")
        if not 0.0 <= self.crag_ambiguous_ratio <= 1.0:
            raise ValueError("crag_ambiguous_ratio must be in [0, 1]")

    def with_(self, **changes: Any) -> RetrievalFlags:
        """Derive a new flag set, marking the changed names as explicit so
        `__post_init__` does not overwrite them from config or environment."""
        explicit = frozenset(self._explicit | set(changes))
        return replace(self, **changes, _explicit=explicit)

    @property
    def legs(self) -> tuple[str, ...]:
        return tuple(name for name in ("dense", "lexical", "graph") if getattr(self, name))

    def summary(self) -> str:
        """One-line human description, used as the ablation table's config column."""
        on = [
            name
            for name in (
                "router",
                "query_expansion",
                "hyde",
                "dense",
                "lexical",
                "graph",
                "rerank",
                "tier_weighting",
                "crag",
                "drop_boilerplate",
            )
            if getattr(self, name)
        ]
        return "+".join(on) if on else "(nothing)"


#: Named explicitly rather than derived from `fields()`: `from __future__
#: import annotations` makes every `f.type` a *string*, so an `is bool` test
#: would silently match nothing and disable every environment override.
_BOOL_FIELDS: tuple[str, ...] = (
    "router",
    "query_expansion",
    "hyde",
    "dense",
    "lexical",
    "graph",
    "rerank",
    "tier_weighting",
    "crag",
    "metadata_filter",
    "drop_boilerplate",
    "use_llm_router",
    "use_llm_crag",
    "long_context",
)

_FLOAT_FIELDS: tuple[str, ...] = ("crag_relevance_threshold", "crag_ambiguous_ratio")


def _f(**kw: Any) -> RetrievalFlags:
    return RetrievalFlags(_explicit=frozenset(kw), **kw)


#: The R0-R9 ablation ladder plus the long-context baseline.
#:
#: R0 is the floor the brief names ("dense-only baseline"); R9 is the full
#: stack. Each intermediate row turns on (or, from R6, turns *off*) exactly
#: one thing relative to a named neighbour, so a difference in the table is
#: attributable to a single component rather than to a bundle of changes.
PRESETS: dict[str, RetrievalFlags] = {
    "R0": _f(
        label="R0",
        note="dense-only baseline: raw utterance, one vector query, no fusion",
        router=False,
        query_expansion=False,
        hyde=False,
        dense=True,
        lexical=False,
        graph=False,
        rerank=False,
        tier_weighting=False,
        crag=False,
        drop_boilerplate=False,
    ),
    "R1": _f(
        label="R1",
        note="R0 + framework query expansion (3 queries, RRF over them)",
        router=False,
        query_expansion=True,
        hyde=False,
        dense=True,
        lexical=False,
        graph=False,
        rerank=False,
        tier_weighting=False,
        crag=False,
    ),
    "R2": _f(
        label="R2",
        note="R1 + BM25 lexical leg (dense+lexical hybrid, RRF)",
        router=False,
        query_expansion=True,
        hyde=False,
        dense=True,
        lexical=True,
        graph=False,
        rerank=False,
        tier_weighting=False,
        crag=False,
    ),
    "R3": _f(
        label="R3",
        note="R2 + graph leg (full three-leg RRF fusion)",
        router=False,
        query_expansion=True,
        hyde=False,
        dense=True,
        lexical=True,
        graph=True,
        rerank=False,
        tier_weighting=False,
        crag=False,
    ),
    "R4": _f(
        label="R4",
        note="R3 + HyDE hypothetical guidance passage on the dense leg",
        router=False,
        query_expansion=True,
        hyde=True,
        dense=True,
        lexical=True,
        graph=True,
        rerank=False,
        tier_weighting=False,
        crag=False,
    ),
    "R5": _f(
        label="R5",
        note="R4 + cross-encoder rerank to top-n (no tier weighting)",
        router=False,
        query_expansion=True,
        hyde=True,
        dense=True,
        lexical=True,
        graph=True,
        rerank=True,
        tier_weighting=False,
        crag=False,
    ),
    "R6": _f(
        label="R6",
        note="R5 + evidence-tier weighting on the rerank score",
        router=False,
        query_expansion=True,
        hyde=True,
        dense=True,
        lexical=True,
        graph=True,
        rerank=True,
        tier_weighting=True,
        crag=False,
    ),
    "R7": _f(
        label="R7",
        note="R6 + CRAG relevance gate (Condition-B fallback on NONE)",
        router=False,
        query_expansion=True,
        hyde=True,
        dense=True,
        lexical=True,
        graph=True,
        rerank=True,
        tier_weighting=True,
        crag=True,
    ),
    "R8": _f(
        label="R8",
        note="full stack MINUS CRAG — isolates what the gate is worth",
        router=True,
        query_expansion=True,
        hyde=True,
        dense=True,
        lexical=True,
        graph=True,
        rerank=True,
        tier_weighting=True,
        crag=False,
    ),
    "R9": _f(
        label="R9",
        note="full stack: adaptive router + expansion + HyDE + 3-leg RRF "
        "+ rerank + tier weighting + CRAG",
        router=True,
        query_expansion=True,
        hyde=True,
        dense=True,
        lexical=True,
        graph=True,
        rerank=True,
        tier_weighting=True,
        crag=True,
    ),
    "LC": _f(
        label="LC-sample",
        note="long-context baseline: no QUERY-DEPENDENT retrieval. A fixed round-robin "
        "sample across all papers at a pinned seed, because the corpus is ~255% of the "
        "context window and does not fit (D7). Any selection rule is a form of retrieval, "
        "so this asks whether query-dependent selection beats a fixed context — not the "
        "question v3 §3 posed.",
        long_context=True,
        router=False,
        query_expansion=False,
        hyde=False,
        dense=False,
        lexical=False,
        graph=False,
        rerank=False,
        tier_weighting=False,
        crag=False,
    ),
}


#: Row labels that differ from their preset key, so a caller can ask for a row
#: by the name it is *reported* under. `LC-sample` is the reported name (D7);
#: `LC` remains accepted because run scripts and ABLATION_ORDER use it.
_LABEL_ALIASES: dict[str, str] = {"LC-SAMPLE": "LC"}


def preset(name: str) -> RetrievalFlags:
    """Look up a named ablation row. Raises on an unknown name rather than
    silently returning the default, so a typo in a run script fails loudly."""
    key = name.upper()
    key = _LABEL_ALIASES.get(key, key)
    try:
        return PRESETS[key]
    except KeyError as exc:
        raise KeyError(
            f"unknown ablation preset {name!r}; expected one of "
            f"{sorted(PRESETS) + sorted(_LABEL_ALIASES)}"
        ) from exc
