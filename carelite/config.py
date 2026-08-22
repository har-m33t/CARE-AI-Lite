"""Runtime configuration. FROZEN INTERFACE — foundation lane only.

Model tags are mutable in Ollama, so every tag is paired with a digest that is
recorded on each generation (v3 §16). `make pin-models` refreshes the digests.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class ModelSpec(BaseModel):
    tag: str
    digest: str | None = None  # filled by `make pin-models`
    context_window: int = 8192
    role: str = ""


class Models(BaseModel):
    """The roster fixed in planning. Judge is a different family from the
    generator so that v3 §13's independence requirement holds."""

    generator: ModelSpec = ModelSpec(
        tag="gemma4:12b", context_window=128_000, role="primary generator (A, B, C, D)"
    )
    generator_alt: ModelSpec = ModelSpec(
        tag="qwen3.5:9b", context_window=128_000, role="cross-model baseline (A2)"
    )
    long_context: ModelSpec = ModelSpec(
        tag="gemma4:12b", context_window=128_000, role="condition LC"
    )
    judge: ModelSpec = ModelSpec(
        tag="gpt-oss:20b", context_window=128_000, role="LLM-as-judge (cross-family)"
    )
    embedder: ModelSpec = ModelSpec(
        tag="bge-m3", context_window=8192, role="instruction-aware embeddings"
    )
    reranker: ModelSpec = ModelSpec(
        tag="BAAI/bge-reranker-v2-m3", role="cross-encoder rerank (sentence-transformers)"
    )


class Retrieval(BaseModel):
    embedding_dim: int = 1024  # bge-m3; pgvector indexed ceiling is 2000
    n_framework_queries: int = 3
    dense_top_k: int = 20
    lexical_top_k: int = 20
    graph_top_k: int = 10
    rrf_k: int = 60
    rerank_top_n: int = 4
    crag_relevance_threshold: float = 0.5
    hyde_enabled: bool = True
    chunk_target_tokens: int = 512
    chunk_overlap_tokens: int = 64


class Experiment(BaseModel):
    n_scenarios_train: int = 40
    n_scenarios_holdout: int = 60
    samples_per_cell: int = 3
    base_seed: int = 20260822
    generation_temperature: float = 0.7
    # v3 §13 self-consistency is measured on the validation subset only; the
    # full run is judged single-pass at temp 0 to keep the lane ~8h not ~35h.
    judge_temperature_full_run: float = 0.0
    judge_samples_full_run: int = 1
    judge_temperature_validation: float = 0.7
    judge_samples_validation: int = 5
    alpha: float = 0.05


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CARELITE_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://localhost:5432/carelite"
    ollama_host: str = "http://localhost:11434"  # localhost only: no egress at inference
    unpaywall_email: str = ""  # required by Unpaywall ToS; keep in .env

    pdf_dir: Path = REPO_ROOT / "data" / "pdfs"
    runs_dir: Path = REPO_ROOT / "runs"
    figures_dir: Path = REPO_ROOT / "figures"

    models: Models = Field(default_factory=Models)
    retrieval: Retrieval = Field(default_factory=Retrieval)
    experiment: Experiment = Field(default_factory=Experiment)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def seed_for(scenario_id: str, condition: str, sample_idx: int) -> int:
    """Deterministic per-cell seed so runs are reproducible and resumable.

    Uses blake2b rather than `hash()`: CPython randomises string hashing per
    process unless PYTHONHASHSEED is pinned, which would make seeds differ
    between runs and quietly destroy the reproducibility guarantee.
    """
    base = get_settings().experiment.base_seed
    key = f"{scenario_id}|{condition}|{sample_idx}".encode()
    h = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    return (base + h) % (2**31 - 1)
