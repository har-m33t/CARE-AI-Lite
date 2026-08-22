-- CARELite AI schema. FROZEN — foundation lane only.
-- Derived from build_plan v3 §9. One database holds corpus, vectors, lexical
-- index, graph edges, and every experimental result, so the analysis queries in
-- §5 are ordinary SQL joins rather than a data-wrangling project.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------- corpus ---

CREATE TABLE IF NOT EXISTS paper (
    paper_id        TEXT PRIMARY KEY,
    doi             TEXT UNIQUE,
    apa_citation    TEXT NOT NULL,
    year            INTEGER,
    design          TEXT,
    evidence_tier   TEXT NOT NULL CHECK (evidence_tier IN ('strong','moderate','emerging')),
    pdf_path        TEXT,                      -- local only; PDFs are never committed
    oa_license      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunk (
    chunk_id            TEXT PRIMARY KEY,
    paper_id            TEXT NOT NULL REFERENCES paper(paper_id) ON DELETE CASCADE,
    ordinal             INTEGER NOT NULL,
    text                TEXT NOT NULL,
    contextual_prefix   TEXT,                  -- LLM-generated; a poisoning vector, treat as untrusted
    embedding           vector(1024),
    tsv                 tsvector GENERATED ALWAYS AS (
                            to_tsvector('english',
                                coalesce(contextual_prefix,'') || ' ' || text)
                        ) STORED,
    UNIQUE (paper_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunk_embedding_hnsw
    ON chunk USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunk_tsv_gin ON chunk USING gin (tsv);

-- -------------------------------------------------------- knowledge base ---

CREATE TABLE IF NOT EXISTS kb_entry (
    entry_id            TEXT PRIMARY KEY,
    theme               TEXT NOT NULL,
    finding             TEXT NOT NULL,
    practical_takeaway  TEXT NOT NULL,
    example_behavior    TEXT NOT NULL,
    evidence_tier       TEXT NOT NULL CHECK (evidence_tier IN ('strong','moderate','emerging')),
    action_type         TEXT NOT NULL CHECK (action_type IN ('detection','generation','reframing')),
    verbatim_span       TEXT NOT NULL,         -- must appear in a source paper; enforced in carelite.kb
    encounter_phase     TEXT[] NOT NULL DEFAULT '{}',
    nurse_component     TEXT[] NOT NULL DEFAULT '{}',
    four_habits         TEXT[] NOT NULL DEFAULT '{}',
    equity_relevant     BOOLEAN NOT NULL DEFAULT FALSE,
    embedding           vector(1024),
    tsv                 tsvector GENERATED ALWAYS AS (
                            to_tsvector('english',
                                finding || ' ' || practical_takeaway || ' ' || example_behavior)
                        ) STORED,
    human_verified      BOOLEAN NOT NULL DEFAULT FALSE,   -- set at the wave-2 review gate
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS kb_entry_embedding_hnsw
    ON kb_entry USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS kb_entry_tsv_gin ON kb_entry USING gin (tsv);
CREATE INDEX IF NOT EXISTS kb_entry_theme_idx ON kb_entry (theme);

CREATE TABLE IF NOT EXISTS kb_entry_source (
    entry_id    TEXT NOT NULL REFERENCES kb_entry(entry_id) ON DELETE CASCADE,
    paper_id    TEXT NOT NULL REFERENCES paper(paper_id) ON DELETE CASCADE,
    PRIMARY KEY (entry_id, paper_id)
);

-- ----------------------------------------------------------------- graph ---
-- Postgres is the source of truth; NetworkX is a derived in-memory view (v3 §8).

CREATE TABLE IF NOT EXISTS graph_edge (
    edge_id         BIGSERIAL PRIMARY KEY,
    source_id       TEXT NOT NULL,
    relation        TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    evidence_tier   TEXT CHECK (evidence_tier IN ('strong','moderate','emerging')),
    paper_id        TEXT REFERENCES paper(paper_id) ON DELETE SET NULL,
    UNIQUE (source_id, relation, target_id)
);

CREATE INDEX IF NOT EXISTS graph_edge_source_idx ON graph_edge (source_id);
CREATE INDEX IF NOT EXISTS graph_edge_target_idx ON graph_edge (target_id);

-- ------------------------------------------------------------ experiment ---

CREATE TABLE IF NOT EXISTS scenario (
    scenario_id         TEXT PRIMARY KEY,
    text                TEXT NOT NULL,
    challenge_type      TEXT NOT NULL,
    emotion_intensity   INTEGER NOT NULL CHECK (emotion_intensity BETWEEN 1 AND 5),
    encounter_phase     TEXT NOT NULL,
    literacy_signal     TEXT NOT NULL,
    equity_stratum      BOOLEAN NOT NULL,
    split               TEXT NOT NULL CHECK (split IN ('train','holdout'))
);

CREATE TABLE IF NOT EXISTS prompt_version (
    prompt_id   TEXT PRIMARY KEY,
    condition   TEXT NOT NULL,
    text        TEXT NOT NULL,
    optimizer   TEXT,                          -- null | 'bootstrap_fewshot' | 'gepa'
    git_sha     TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS generation (
    generation_id   TEXT PRIMARY KEY,
    scenario_id     TEXT NOT NULL REFERENCES scenario(scenario_id) ON DELETE CASCADE,
    condition       TEXT NOT NULL,
    prompt_id       TEXT NOT NULL REFERENCES prompt_version(prompt_id),
    model           TEXT NOT NULL,
    model_digest    TEXT NOT NULL,             -- tags are mutable; digest is the real identity
    seed            BIGINT NOT NULL,
    temperature     REAL NOT NULL,
    sample_idx      INTEGER NOT NULL,
    response        TEXT NOT NULL,
    latency_ms      INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- the v3 §16 cache key: re-running a completed cell is a no-op
    UNIQUE (scenario_id, condition, prompt_id, model_digest, seed, sample_idx)
);

CREATE INDEX IF NOT EXISTS generation_condition_idx ON generation (condition);

CREATE TABLE IF NOT EXISTS retrieval_trace (
    generation_id   TEXT PRIMARY KEY REFERENCES generation(generation_id) ON DELETE CASCADE,
    retrieved_ids   TEXT[] NOT NULL DEFAULT '{}',
    scores          REAL[] NOT NULL DEFAULT '{}',
    crag_grade      TEXT CHECK (crag_grade IN ('relevant','ambiguous','none')),
    route_taken     TEXT,
    fell_back_to_b  BOOLEAN NOT NULL DEFAULT FALSE,
    hyde_passage    TEXT,
    latency_ms      INTEGER
);

CREATE TABLE IF NOT EXISTS rubric_score (
    score_id        BIGSERIAL PRIMARY KEY,
    generation_id   TEXT NOT NULL REFERENCES generation(generation_id) ON DELETE CASCADE,
    rater_type      TEXT NOT NULL CHECK (rater_type IN ('deterministic','llm_judge','human')),
    rater_id        TEXT NOT NULL,
    sample_idx      INTEGER NOT NULL DEFAULT 0,   -- judge self-consistency sample
    -- NURSE
    name        INTEGER CHECK (name        BETWEEN 1 AND 5),
    understand  INTEGER CHECK (understand  BETWEEN 1 AND 5),
    respect     INTEGER CHECK (respect     BETWEEN 1 AND 5),
    support     INTEGER CHECK (support     BETWEEN 1 AND 5),
    explore     INTEGER CHECK (explore     BETWEEN 1 AND 5),
    -- Four Habits
    ib          INTEGER CHECK (ib          BETWEEN 1 AND 5),
    epp         INTEGER CHECK (epp         BETWEEN 1 AND 5),
    de          INTEGER CHECK (de          BETWEEN 1 AND 5),
    ie          INTEGER CHECK (ie          BETWEEN 1 AND 5),
    -- secondary
    naturalness INTEGER CHECK (naturalness BETWEEN 1 AND 5),
    ritualistic INTEGER CHECK (ritualistic BETWEEN 1 AND 5),
    safety_flags    TEXT[] NOT NULL DEFAULT '{}',
    evidence_spans  JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (generation_id, rater_type, rater_id, sample_idx)
);

CREATE INDEX IF NOT EXISTS rubric_score_generation_idx ON rubric_score (generation_id);
CREATE INDEX IF NOT EXISTS rubric_score_rater_idx ON rubric_score (rater_type);

-- Human rating is deferred but the harness is built now: this table records the
-- blinded presentation each rater saw, so unblinding is a join, not a guess.
CREATE TABLE IF NOT EXISTS rating_assignment (
    assignment_id   BIGSERIAL PRIMARY KEY,
    rater_id        TEXT NOT NULL,
    generation_id   TEXT NOT NULL REFERENCES generation(generation_id) ON DELETE CASCADE,
    display_order   INTEGER NOT NULL,
    blind_label     TEXT NOT NULL,             -- what the rater sees instead of the condition
    is_calibration  BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (rater_id, generation_id)
);
