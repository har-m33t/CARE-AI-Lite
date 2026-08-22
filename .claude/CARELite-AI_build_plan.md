# CARELite AI — Build Plan v3

**Local Ollama stack · Publication-grade rigor · Full data-architecture rationale**
August 2026. Supersedes v1 and v2.

---

## Part I — Choosing the RAG architecture

### 1. The decision that governs everything else

Microsoft's GraphRAG work introduced the cleanest framing available for this choice: **local queries** vs **global queries**. Local queries are answered by a small number of text regions ("what does the evidence say about teach-back?"). Global queries require synthesizing across large portions of the corpus and produce an answer stated nowhere explicitly ("what are the recurring themes across these papers?"). Vector RAG wins local; GraphRAG wins global.

**Your production query mix is ~95% local.** A patient says something; the system needs guidance on how to respond. That's a small-number-of-regions retrieval, every time. On that basis alone, full GraphRAG is the wrong architecture and the literature is nearly unanimous that teams should exhaust hybrid retrieval before reaching for a graph.

**But there's a second consideration that flips part of the analysis.**

### 2. Why the standard GraphRAG cost/benefit doesn't apply to you

Every argument against GraphRAG in 2024–2026 centers on the *extraction pipeline*. Original Microsoft GraphRAG indexing ran into the tens of thousands of dollars on large corpora; LazyGraphRAG cut that to roughly 0.1% by deferring summarization to query time. And cost was never the worst part — reported entity-extraction failure rates of 30–40% on typical corpora, plus name-collision errors that poison every downstream query touching the bad node, are the reason graph projects fail.

**You already did the extraction. By hand. In Phase I.**

Your 45-entry knowledge base, 10 themes, and behavior-to-framework mappings *are* a curated knowledge graph. A human read 25 papers and hand-authored the nodes and edges:

```
Paper ──supports──> Behavior ──instantiates──> NURSE component
  │                    │                              │
  │                    └──measured_by──> Outcome       └──belongs_to──> Theme
  └──has──> EvidenceTier                                        │
                                                    Behavior ──appropriate_in──> EncounterPhase
```

This inverts the usual calculus. The expensive, error-prone, quality-limiting step is already complete and was done by the most reliable extractor available — a domain-informed human working from primary sources. The remaining cost of a graph layer is a few hundred lines of code.

**So the answer is neither "GraphRAG" nor "no graph."** It's: skip extraction-based GraphRAG entirely, and build a **curated property-graph layer** over your existing structure. You get the relational retrieval primitive without paying any of the costs that make GraphRAG a bad default.

### 3. RAG variants, assessed against your actual needs

| Architecture | Verdict | Reasoning |
|---|---|---|
| **Hybrid vector + BM25 + RRF + rerank** | **Core. Build first.** | The 2026 production minimum. Framework terms ("NURSE," "teach-back," "Four Habits") are exact-match tokens that dense retrieval alone misses. Everything else is measured against this baseline. |
| **Curated property graph** | **Build. High value, low cost.** | See §2. Answers questions flat retrieval structurally cannot: "which behaviors have outcome evidence rather than expert opinion," "which NURSE components are under-supported." Drives evidence-tier weighting in reranking and powers the provenance display in the UI. ~200 lines with NetworkX. |
| **Microsoft GraphRAG (full)** | **Skip.** | Leiden community detection over LLM-extracted entities, designed for global synthesis over million-token corpora. You have ~25 papers and local queries. Wrong tool at every axis. |
| **LazyGraphRAG** | **Skip, but know why.** | Solves the indexing-cost problem you don't have (your extraction is free and already done) while adding query-time latency you can't afford in an interactive assistant. |
| **RAPTOR (hierarchical)** | **Partial — curated, not clustered.** | RAPTOR recursively clusters and summarizes to build a hierarchy. **You already have the hierarchy**: behavior → theme → framework component, hand-built. Implement RAPTOR's *retrieval* pattern (query can hit any level of the tree) over your *curated* tree. Skip the clustering. Same argument as the graph. |
| **Adaptive RAG (query routing)** | **Build.** | The emerging 2026 default. Routes by query type before retrieval runs. For you it's a quality gain, not just latency: retrieving evidence for a purely emotional turn ("I'm just so scared") is one of the main ways framework-guided systems come out sounding clinical instead of warm. |
| **CRAG (corrective)** | **Build. Non-negotiable.** | Grades retrieved docs; falls back to Condition-B behavior when nothing relevant surfaces. Without it, Condition C injects noise on turns your corpus can't address and can score *below* Condition B — a confound that invalidates the headline comparison. |
| **Self-RAG** | **Partial.** | Adopt the reflection/critique token idea as the self-check node. Skip the fine-tuned-critic implementation — no training data. |
| **HyDE** | **Build.** | Generate a hypothetical *guidance passage* and embed that. Directly attacks the patient-utterance-vs-guidance-document asymmetry, which is the central retrieval problem here. |
| **Contextual Retrieval (Anthropic)** | **Build.** | Prepend an LLM-generated situating sentence to each paper chunk before embedding. Anthropic reported large reductions in retrieval failure. One-time offline pass; free locally. Highest-value preprocessing available. |
| **Agentic / multi-hop RAG** | **Skip.** | Adds variance to a controlled comparison and latency to a conversational interface. Multi-hop is not your query shape. |
| **ColBERT / late interaction** | **Optional ablation.** | Qdrant supports multi-vector natively if you want to test it. Nice-to-have, not core. |
| **Long-context (skip RAG entirely)** | **Report as a baseline.** | Your whole corpus fits in a 256K window. Stuffing it is a *legitimate baseline condition* and a reviewer will ask. Cheap to run, and if RAG doesn't beat it, that's a finding. |

### 4. Target architecture

```
                        patient turn
                             │
                  ┌──────────▼──────────┐
                  │  Adaptive router    │
                  └──┬────────┬─────────┘
       emotional-only│        │informational / mixed
                     │        │
                     │   ┌────▼──────────────────────────┐
                     │   │ Query construction            │
                     │   │  · 3 framework-language queries│
                     │   │  · HyDE guidance passage      │
                     │   │  · metadata filter            │
                     │   └────┬──────────────────────────┘
                     │        │
                     │   ┌────▼─────┬──────────┬─────────────┐
                     │   │  dense   │  BM25    │  graph       │
                     │   │ (pgvector│ (Postgres│ (traversal   │
                     │   │  +instr.)│  FTS)    │  from filter)│
                     │   └────┬─────┴────┬─────┴──────┬───────┘
                     │        └──── RRF fusion ───────┘
                     │                  │
                     │        cross-encoder rerank (top 4)
                     │                  │
                     │           CRAG relevance gate
                     │            ├─ relevant → context
                     │            └─ none → fall back to B
                     │                  │
                     └──────────────────┴──→ generate → self-check → safety → out
```

---

## Part II — The data layer

### 5. You don't need a vector database. You need three stores.

This is the part most RAG tutorials get wrong for research projects. Vector search is one of **four** distinct data problems here, and only one of them is a similarity problem:

| Store | Holds | Access pattern |
|---|---|---|
| **Relational** | 45 KB entries, 25 paper records, 50 scenarios, prompt versions, **every generation**, rubric scores, human ratings, retrieval traces | Joins, aggregation, filtering, `GROUP BY condition` |
| **Vector** | ~2,000 embeddings (45 entries + ~1,500 paper chunks + HyDE cache) | ANN similarity + metadata pre-filter |
| **Lexical** | Same text, inverted index | BM25 exact-term match |
| **Graph** | ~150 nodes, ~400 edges | Traversal, path queries |

**For a rigor-focused project, the relational store is the most important one and the one people neglect.** Your analysis depends on queries like:

> "Mean NURSE-Respect score by condition, restricted to held-out scenarios in the equity stratum, where retrieval grade was 'relevant', grouped by prompt version."

That's a three-way join. If your generations are in CSVs, your scores are in JSON files, and your embeddings are in Chroma, you cannot answer it reliably and your reproducibility story falls apart at exactly the point it matters most.

### 6. Recommendation: PostgreSQL 17 + pgvector as the system of record

**One database holds everything.** Vectors live in the same instance as the experiment results, with ACID guarantees and SQL joins across both.

```sql
-- retrieval + analysis in one query
SELECT g.condition, AVG(s.respect), COUNT(*)
FROM generation g
JOIN rubric_score s USING (generation_id)
JOIN scenario sc USING (scenario_id)
WHERE sc.split = 'holdout' AND sc.equity_stratum
  AND g.prompt_version = 'B_gepa_v4'
GROUP BY g.condition;
```

**Why Postgres over Chroma here**, despite Chroma being the standard prototype recommendation and genuinely fine at this scale:

- Your vector count is trivial. ~2,000 vectors is nothing; pgvector's HNSW index delivers sub-5ms latency well past a million. Vector *performance* is simply not a decision criterion for you, which frees you to choose on other axes.
- Postgres gives you **full-text search built in** — your BM25 layer without a second system.
- **ACID + transactional consistency** between documents and embeddings. Reindex without half-written state.
- SQL is your analysis language. `pandas.read_sql` straight into the stats notebook.
- One backup, one `pg_dump`, one file in the reproducibility archive.
- Sovereignty: everything local, nothing leaves the machine.

**One constraint to know:** pgvector caps standard-precision indexed vectors at 2,000 dimensions. Qwen3-Embedding-0.6B outputs 1024 and supports Matryoshka truncation, so you're comfortably inside it.

**Setup cost:** one `docker compose up`. That's the whole objection.

### 7. Alternatives, honestly

| Option | When it's right for you |
|---|---|
| **Chroma + SQLite** | If Postgres genuinely blocks you. Lower ceiling on analysis queries; you'll write more Python glue. Acceptable, not preferred. |
| **Qdrant** | If you want native sparse vectors and ColBERT multi-vector as first-class features for the ablation study. Best-in-class filtering. Cost: a second system alongside your relational store. Reasonable if late-interaction retrieval is a research question you care about. |
| **LanceDB** | Embedded, file-based, no server, versioned by design — the versioning is genuinely attractive for reproducibility. Newer and smaller community. A defensible contrarian pick. |
| **Weaviate / Milvus / Pinecone** | No. Built for scale and ops problems you will never have. |

### 8. The graph store

**Do not install Neo4j.** At ~150 nodes and ~400 edges, the graph is smaller than most CSVs.

- Edges live in a Postgres table (`source_id`, `relation`, `target_id`, `evidence_tier`, `source_paper`).
- Materialize into **NetworkX** at startup for traversal.
- Postgres remains the single source of truth; the graph is a derived in-memory view.

If you later want Cypher and a browser visualizer, Neo4j Community or Apache AGE (the Postgres graph extension) are drop-in upgrades. You almost certainly won't need them.

### 9. Schema sketch

```
paper(paper_id, apa_citation, year, design, evidence_tier, doi, pdf_path)
kb_entry(entry_id, theme, finding, practical_takeaway, example_behavior,
         encounter_phase[], nurse_component[], four_habits[], equity_relevant)
kb_entry_source(entry_id, paper_id)                    -- provenance
chunk(chunk_id, paper_id, text, contextual_prefix, embedding vector(1024), tsv)
graph_edge(source_id, relation, target_id, evidence_tier, paper_id)

scenario(scenario_id, text, challenge_type, emotion_intensity, encounter_phase,
         literacy_signal, equity_stratum, split)       -- split: train | holdout
prompt_version(prompt_id, condition, text, optimizer, git_sha, created_at)
generation(generation_id, scenario_id, condition, prompt_id, model, seed,
           temperature, sample_idx, response, latency_ms, created_at)
retrieval_trace(generation_id, retrieved_ids[], scores[], crag_grade, route_taken)
rubric_score(generation_id, rater_type, rater_id, name, understand, respect,
             support, explore, ib, epp, de, ie, naturalness, ritualistic,
             safety_flags[], evidence_spans jsonb)
```

`rater_type ∈ {deterministic, llm_judge, human}` in one table is deliberate — it makes judge-vs-human agreement a single self-join instead of a data-wrangling project.

---

## Part III — Restoring publication-grade rigor

Everything v2 cut, reinstated — plus the things v1 didn't have.

### 10. Pre-registration

Register on **OSF** (free, timestamped, public) before generating a single evaluation response. This is the cheapest credibility purchase available and it costs an afternoon.

Specify: primary outcome (composite NURSE adherence, Condition A vs B), secondary outcomes, hypotheses with direction, sample size and its justification, the full analysis plan including corrections, exclusion criteria, and the stopping rule. Everything not listed becomes explicitly **exploratory** — which is fine, and honest, and is the entire point.

Pre-registration is what makes the naturalness result credible if it goes against you. Without it, "Condition A beat B on naturalness" reads as a post-hoc excuse. With it, it reads as a pre-specified secondary outcome that came out the interesting way.

### 11. Power analysis drives n, not convenience

Run it before fixing the scenario count. Paired design, Wilcoxon signed-rank, α = 0.05, power = 0.80.

- Detecting a **large** paired effect (d ≈ 0.8): ~15–20 scenarios
- **Medium** (d ≈ 0.5): ~35–45
- **Small** (d ≈ 0.3): ~90+

You expect large effects on structural adherence (A vs B) and smaller ones on B vs C. **Power for the comparison you care about least is what sets n.** Budget **100 scenarios: 40 train / 60 held-out.** Free local inference means the only real cost is your curation time — and curation is where the quality lives, so don't inflate the count by generating slop.

### 12. Human evaluation — the hardest solo constraint, addressed properly

A single rater is the ceiling on this project's credibility. Options, best first:

1. **Recruit 2 external raters.** Medical students or nursing students. Your prior institutional contacts are the obvious channel; failing that, r/medicalschool, student associations, or Prolific with a screener (roughly $150–250 for 60 responses × 2 raters). This is the single highest-return expenditure in the whole project.
2. **One external rater + you.** Two raters is the floor for a defensible Krippendorff's α.
3. **You only, with intra-rater reliability.** Score the set twice, ≥2 weeks apart, blinded and reshuffled both times. Report test–retest α. Weaker, but it's honest data about rating stability rather than an unexamined assumption.

**Protocol regardless of which:** stratified sample of 60 responses (20 scenarios × 3 conditions), condition labels stripped, presentation order randomized per rater, written rubric with anchored examples distributed *before* rating, and a calibration set of 5 responses scored and discussed first. Compute **Krippendorff's α** (ordinal) for human–human agreement. Report it whatever it is; a low α is a finding about the construct, not a failure to hide.

### 13. LLM-as-judge, validated as a study in its own right

Using a local ~20B judge is a real limitation. Treat validating it as a component study, not a checkbox:

- **Independence:** judge model from a different family than the generator (`gpt-oss:20b` judging `qwen3.6:27b`). Report it prominently.
- **Self-consistency:** 5 samples at temp 0.7, median score, and report inter-sample variance as a stability metric.
- **Grounding:** every score requires a verbatim evidence span from the response. Spot-check 30 spans manually; report the rate at which the cited span actually supports the score.
- **Positional-bias check:** rerun a subset with option order reversed.
- **Validity:** Krippendorff's α and Spearman ρ between judge and human consensus, computed **per rubric dimension** — judges are usually decent at structural items and poor at naturalness, and you want to know which of your numbers to discount.
- **Pre-specified threshold:** commit in the pre-registration to a minimum agreement below which judge-only results are reported as exploratory.

### 14. Statistics

- **Primary:** Friedman omnibus across A/B/C; Wilcoxon signed-rank pairwise; **Holm–Bonferroni** across the pairwise family. Corrections applied across all rubric dimensions, not per-dimension.
- **Effect sizes with 95% bootstrap CIs** on every comparison. Report these first; at n=60 they carry more information than p-values.
- **Variance decomposition:** you have 3 samples per scenario-condition. Use a mixed-effects model (random intercept for scenario) to separate within-scenario generation variance from between-condition effect. Local inference makes this affordable and it's a meaningfully better analysis than treating samples as independent.
- **Subgroup:** equity stratum pre-specified as a secondary analysis; all others exploratory and labeled as such.
- **Sensitivity:** rerun the primary analysis (a) judge-only vs human-only, (b) with and without CRAG-fallback turns, (c) excluding scenarios where judge self-consistency was poor. Report whether conclusions hold.
- **Negative controls:** include a deliberately degraded prompt condition. If your rubric can't distinguish it from Condition B, the rubric isn't measuring what you think.

### 15. Reporting standards

Reporting guidelines for LLM studies in health matured over 2024–2026 — **TRIPOD-LLM** and **CHART** (Chatbot Assessment Reporting Tool) are the relevant ones. Pull the current versions from the EQUATOR Network and complete the checklist as an appendix. Even for a personal project, working through a checklist surfaces omissions you won't otherwise notice, and it's the single clearest signal to a reader that the work was done carefully.

### 16. Reproducibility artifact

- Public repo: code, prompts (all versions, git-tracked), scenario bank, rubric with anchors, rater instructions.
- `pg_dump` of the full results database.
- Pinned environment (`uv.lock`), exact Ollama model tags **and digests** (tags are mutable — pin the digest).
- Every generation cached and keyed by `(scenario, condition, prompt_version, model_digest, seed, sample_idx)`.
- Langfuse traces exported.
- A `make reproduce` target that regenerates every figure and table from the database.

### 17. Limitations to state without being asked

Small corpus (25 papers). Synthetic scenarios, not real patient utterances. Single-turn or short multi-turn, not full encounters. Local-model capability ceiling. Rubric operationalizes NURSE and Four Habits in one particular way among several defensible ones. No patient-reported outcomes — you're measuring adherence to frameworks that are *proxies* for patient experience, and the frameworks' own validation literature is mixed (your Phase I notes already flagged the 4HM course with no significant post-course empathy change). No clinical deployment claim.

---

## Part IV — Build sequence

| Sprint | Deliverable | Gate |
|---|---|---|
| **0. Pre-registration** | OSF registration, power analysis, rubric with anchored examples | Registered and timestamped before any eval data exists |
| **1. Data layer** | Postgres 17 + pgvector via Docker, full schema, Langfuse self-hosted | Schema loaded; a dummy join returns rows |
| **2. Knowledge base migration** | 45 entries + 25 papers into relational tables with full metadata and provenance | Every entry traceable to a source paper via `kb_entry_source` |
| **3. Graph layer** | `graph_edge` populated, NetworkX materialization, traversal queries | Answer "which behaviors have outcome-level evidence" in SQL + traversal |
| **4. Indexing** | Semantic chunking, contextual retrieval pass, instruction-aware embeddings, HNSW index, Postgres FTS | 10 hand-written probes return sensible entries |
| **5. Retrieval** | Hybrid + RRF + rerank + HyDE + query builder + CRAG + adaptive router; R0→R9 ablation incl. long-context baseline | Ragas context precision >0.7; ablation table populated |
| **6. Graph + safety + orchestration** | LangGraph state machine, 3 conditions + long-context baseline, red-flag filter, CoVe post-check | End-to-end on 5 pilots; red-flag test set 100% caught |
| **7. Scenario bank** | 100 curated scenarios, stratified, 40/60 split | Every stratum cell populated; equity stratum reviewed by a second person |
| **8. Judge validation** | Judge implementation + the §13 validation study | Pre-registered agreement threshold met, or judge results demoted to exploratory |
| **9. Prompt optimization** | DSPy modules, BootstrapFewShot, GEPA on the 40 train scenarios | Beats hand-written on train; **held-out untouched** |
| **10. Human evaluation** | 60 blinded responses, 2+ raters, calibration, Krippendorff's α | α computed and reported |
| **11. Full run + analysis** | 60 × 4 conditions × 3 samples, mixed-effects models, corrections, sensitivity analyses | Every claim traces to a pre-specified or explicitly-exploratory test |
| **12. Interface + writeup** | Streamlit with evidence + provenance panel; report with TRIPOD-LLM/CHART checklist | `make reproduce` runs cold on a clean machine |

**Ordering constraints that matter:** Sprint 0 before any eval data. Sprint 8 before Sprint 9 (GEPA optimizes against the rubric; an unvalidated rubric means optimizing toward an unverified target). Sprint 9 never touches held-out.

---

## Part V — Risks

- **Human raters are the bottleneck.** Start recruiting during Sprint 1, not Sprint 10. Everything else is under your control; this isn't.
- **Local judge weakness on naturalness.** Likely your weakest measurement, and naturalness is exactly where Condition B is expected to lose. If judge–human agreement on naturalness is poor, that dimension becomes human-rated only, at n=60. Plan for it.
- **GEPA overfitting.** 40 training scenarios is few. The held-out set is the only defense; treat it as write-once.
- **The graph tempting you into scope creep.** Build the curated property graph. Do not drift into entity extraction, community detection, or "while I'm here, let me index PubMed."
- **Contextual retrieval preprocessing drift.** It's LLM-generated. Spot-check 30 prefixes; a bad situating sentence poisons a chunk permanently.
- **Still not clinical software.** Header disclaimer on the app, statement in the report.

---

## Appendix: sources consulted (August 2026)

- Microsoft Research, *LazyGraphRAG: Setting a New Standard for Quality and Cost* (June 2025) — local/global query distinction, 0.1% indexing cost
- Microsoft Research, GraphRAG + BenchmarkQED — comprehensiveness/diversity win rates on global queries
- TianPan.co (Apr–May 2026), *GraphRAG vs Vector RAG* — entity-extraction failure rates, HippoRAG/PathRAG/OG-RAG cost curves
- Cognee (July 2026), *GraphRAG vs RAG* — DRIFT search, "strengthen the foundation first" framing
- Firecrawl, MarkTechPost, vucense, callsphere (2026) — vector DB comparisons; pgvector/Qdrant/Chroma/LanceDB positioning
- MCP.Directory (May 2026) — pgvector 2,000-dimension index ceiling
- Anthropic, *Contextual Retrieval*
- Agrawal et al., *GEPA*, ICLR 2026 oral (arxiv 2507.19457)
- Qwen, *Qwen3 Embedding* — instruction-aware embeddings, MRL, Apache 2.0
- EQUATOR Network — TRIPOD-LLM, CHART reporting guidelines (verify current versions)