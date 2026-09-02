# CARELite AI
### An Evidence-Based AI Assistant for Patient Communication

*DBMI Summer Internship Project | University of Arizona College of Medicine – Phoenix*
*Department of Biomedical Informatics | June 8 – August 15, 2026*

---

## Project Goal

CARELite AI is being built to address a specific and well-documented problem: clinician communication skills are unevenly developed, unevenly delivered, and unevenly reinforced — and patients pay the difference in adherence, comprehension, satisfaction, and outcomes.

The goal of this project is to design and prototype an AI-assisted communication support system grounded in peer-reviewed evidence. The system is not intended to replace clinical judgment or to script patient encounters. It is intended to do what the human literature has consistently failed to do at scale: provide real-time, specific, low-friction support for the communication behaviors that the research shows actually change outcomes.

This means the system needs to do three things well. It needs to notice when something important is happening in a clinical conversation — an emotional cue, a jargon-heavy explanation, a missed teach-back opportunity. It needs to generate a useful response, prompt, or reframe at the right moment. And it needs to do both of these things without adding cognitive load to an already demanding clinical encounter.

---

## Background

Effective patient communication is not a soft skill. It is a mechanism. The research is clear that how a clinician communicates changes medication adherence, diagnostic costs, patient self-efficacy, readmission rates, and quality of life. It also changes clinician burnout — which in turn degrades communication further, creating a cycle that worsens care over time.

Despite this, most clinicians receive limited communication training after medical school, most training they do receive is brief and does not persist, and most healthcare systems have no infrastructure for reinforcing communication skills at the point of care. The gap between what the evidence recommends and what happens in exam rooms is wide and largely invisible.

CARELite AI is designed to close part of that gap.

---

## Communication Frameworks

The system is built on seven evidence-based communication categories derived from synthesis of the peer-reviewed literature (33 papers retrieved into the corpus — see "Literature Corpus" below). These categories were not selected from existing frameworks — they were identified through pattern analysis across the literature and then validated against the strongest findings in the corpus.

### 1. Empathy — Responsive, Not Just Warm
Empathy is treated here as a behavioral skill, not a personality trait. The evidence shows that high-empathy clinicians use activating, cognitively-oriented communication — not more emotional language. The system supports empathy by detecting missed acknowledgments and generating responses that name emotional states before offering clinical information.

### 2. Emotion Recognition and Response
The most damaging communicative act documented in the literature is blocking a patient's emotional expression — pivoting to information immediately after an emotional cue without acknowledgment. This pattern is measurable, trainable, and falls disproportionately on minority patients. The system flags blocking patterns and generates alternative openings that hold space for emotion before moving forward.

### 3. Patient Activation and Shared Decision-Making
The strongest adherence outcomes in the literature come from plans that were genuinely negotiated around patient goals — not from education, not from information delivery, but from the act of building the plan together. The system prompts goal elicitation before any treatment recommendation is generated and monitors whether patient preferences were actually incorporated.

### 4. Comprehension Confirmation (Teach-Back)
Teach-back is the highest-evidence, lowest-risk communication behavior in the corpus. It works across settings, populations, and health literacy levels, and no study in the reviewed literature found it harmful. The system generates teach-back prompts at every key handoff point and produces re-explanations using different language when a patient's response is incomplete.

### 5. Plain Language and Information Clarity
Telling a patient something is not the same as ensuring they understood it. The system detects clinical jargon, flags responses that exceed three key messages, and generates plain-language alternatives grounded in everyday analogy. The target is understood consent and comprehension, not disclosed information.

### 6. Trust and Relational Continuity
Trust mediates the relationship between communication and outcomes — in several studies, communication's effect on quality of life was only significant through the trust it generated. The system supports trust-building through consistency prompts, transparency about uncertainty, prior-visit callbacks, and active EHR-sharing cues.

### 7. Equity-Aware Communication
Communication quality is not delivered evenly. Low-SES patients experience significantly lower clinician empathy. Minority patients have their emotional expressions blocked more often. Limited-English-proficiency patients receive lower-quality serious illness conversations even when structured guides are used. These are not incidental findings — they are the documented baseline the system is designed to correct, not replicate.

---

## Knowledge Base Structure

Every finding from the literature is stored as a structured entry containing seven fields.

| Field | Description |
|---|---|
| Theme | One of the seven communication categories |
| Source | Full paper citation with DOI |
| Key Finding | Main result in one to two sentences |
| Practical Takeaway | What a clinician should do differently |
| Example Behavior | A specific, observable communication act |
| Evidence Strength | Strong, Moderate, or Emerging |
| AI Action Type | Detection, Generation, or Reframing |

The knowledge base holds 116 entries spanning all seven themes (activation_sdm 40, plain_language 21, teach_back 15, trust_continuity 14, empathy 14, emotion_response 9, equity 3), loaded into PostgreSQL from the retrieved paper corpus.

Entries are produced by **LLM-assisted extraction with automated verbatim-span provenance validation — not by human curation and not by clinician review**. Every entry's quoted span is mechanically confirmed to appear in the extracted text of the paper it cites before the entry is accepted, and an entry whose span cannot be located is rejected as a fabrication rather than repaired. That check is real and enforced in code (`carelite/kb/validate.py`), and it is also narrow: no person has reviewed whether a given finding follows from its quoted span, `human_verified` is `false` on every loaded entry, and any result built on the knowledge base inherits that limitation.

**116 entries is not 116 independent findings.** Roughly a third of them restate one another in different words, so the count must never stand in for convergent evidence — `docs/limitations.md` §2 has the redundancy-cluster accounting. The equity count of 3 is not a gap awaiting more extraction either: a targeted re-extraction with a revised prompt returned zero net new equity entries (`DECISIONS.md` D3's recorded outcome), which is a finding about this corpus rather than a pipeline shortfall. `docs/decisions/README.md` records why the knowledge base is derived this way rather than hand-authored to a planning-time count.

---

## Actionable Behavior System

Every knowledge-base entry is tagged with one of three functional categories — `AI Action Type` in the table above — rather than being organized into a separate, curated master list of behaviors as originally planned. Across the 116 loaded entries that tagging is 73 generation, 26 reframing, 17 detection. A refined, prioritized, non-overlapping behavior list distilled from these tags remains a planned deliverable, not a finished artifact.

**Detection behaviors** — the system monitors the conversation and flags something the clinician might miss. Examples include detecting emotional blocking patterns, flagging jargon, and identifying conversations that end without a teach-back check.

**Generation behaviors** — the system produces a specific response, prompt, or communication element. Examples include generating teach-back prompts at key handoffs, producing activating follow-up questions after emotional acknowledgments, and generating barriers checks after every treatment plan.

**Reframing behaviors** — the system rewrites or corrects something already said. Examples include replacing patient-blame language with barrier-attribution language, offering plain-language alternatives to jargon-heavy explanations, and generating alternative openings when emotional blocking is detected.

---

## How the System Runs

CARELite is a terminal bedside assistant plus the evaluation study that measures it. PostgreSQL 18.6 with pgvector is the system of record: the corpus, the knowledge base, the graph edges, the scenario bank, every generation and every score sit in one database, and `carelite/db/schema.sql` is the authority on its shape. Retrieval is hybrid — dense pgvector search and lexical `tsvector` search fused by reciprocal rank, reranked, then gated by CRAG, which declines to inject evidence it grades irrelevant rather than injecting it anyway.

The evaluation compares six conditions across the 60 held-out scenarios at three samples each: **A** (bare model), **A2** (the same prompt on a second model family, as a cross-model baseline), **B** (framework-prompted, no retrieval), **C** (framework plus retrieval), **LC** (a fixed, query-independent sample of the corpus packed into the context window — the corpus does not fit the window, so this is the reduced long-context baseline `DECISIONS.md` D7 defines), and **D** (a deliberately degraded prompt, as a negative control).

Every model in the roster is open-weight, and no hosted vendor model sits anywhere in the inference path. **What is not true is that everything runs locally.** Conditions A, A2, B, C and D were served by Ollama on a rented Runpod L40S; Condition LC was served by vLLM with prefix caching on a rented A100, because the same cell costs 198 seconds on the first stack and 3.61 seconds warm on the second — a serving-stack difference, not a property of the condition (`DECISIONS.md` D11 and D13). The two passes sat differently against that hardware: the Ollama pass ran the generation loop on the pod itself and wrote journal files that were loaded into the database afterwards, whereas for LC only model serving was remote and the loop, the database, and the analysis stayed on the operator's machine. `generation.served_by` records which stack produced each row, so a comparison that spans both is visibly confounded rather than quietly pooled.

---

## What This Project Is Not

CARELite AI is not a diagnostic tool. It does not make clinical recommendations. It does not replace the clinician's judgment about what a patient needs. It does not generate empathy — it supports the conditions under which a clinician can express it more consistently and more equitably.

The system is also not a script generator. The literature is explicit on this point: communication frameworks that become scripts stop working. The goal is a system that recognizes what kind of moment a clinician is in and supports the right kind of response — not one that tells a clinician exactly what to say.

---

## Project Structure

The tree below is the current layout, not an intended one. Every count in this file is a hand-written
snapshot; `runs/repro/headline-numbers.txt` is the authority, because `make reproduce` writes it from
the database and this file is written by hand.

```
carelite-ai/
│
├── README.md                   # This file
├── REPRODUCE.md                # Cold-start reproduction instructions
├── DECISIONS.md                # Dated record of project-owner decisions (D1-D13)
├── Makefile                    # install / check / db-up / eval-smoke / reproduce
├── pyproject.toml, uv.lock     # Toolchain: Python 3.13, uv-managed
│
├── carelite/                   # The application and evaluation-harness code
│   ├── types.py, config.py     # Frozen contracts: controlled vocabularies, model roster, seeds
│   ├── db/                     # PostgreSQL + pgvector schema and connection helpers
│   ├── corpus/                 # Fetch, extract, chunk the paper corpus
│   ├── kb/                     # Knowledge-base extraction, verbatim-span provenance, load
│   ├── scenarios/              # The 100-scenario bank; the frozen 60-scenario holdout split
│   ├── index/                  # Dense (pgvector) + lexical (tsvector) indexing
│   ├── retrieval/              # Hybrid retrieval: RRF fusion, rerank, CRAG, HyDE, R0-R9 ablations
│   ├── graph/                  # Curated property graph over the knowledge base
│   ├── generate/               # The six conditions; orchestration; Ollama and vLLM backends
│   ├── prompts/                # Versioned prompt files per condition
│   ├── eval/
│   │   ├── rubric/             # The 11-dimension NURSE / Four Habits rubric, deterministic scorers
│   │   ├── judge/              # LLM-as-judge and its validation study
│   │   └── human/              # Blinded human-rating harness (synthetic-rater-exercised)
│   ├── safety/                 # Input/output safety: injection, PHI, red-flag screens
│   ├── stats/                  # Friedman/Wilcoxon/Holm-Bonferroni, mixed-effects, sensitivity
│   ├── viz/                    # Every figure, regenerated from the database
│   ├── cli/                    # The Typer + Rich terminal interface
│   └── repro.py                # `make reproduce` entry point
│
├── data/
│   ├── fetch_corpus.py         # Thin shim -> carelite.corpus.fetch
│   └── pdfs/                   # Retrieved PDFs (gitignored) + _manual_needed.csv
│
├── knowledge_base/
│   ├── TAXONOMY.md             # Seven-theme taxonomy proposal (accepted, DECISIONS.md D1)
│   ├── review/                 # Generated review digest (no entry has been signed off)
│   └── cache/                  # Extraction cache (gitignored)
│
├── scenarios/
│   ├── EQUITY_REVIEW.md        # Equity-stratum review packet and outcome (D2, D5)
│   ├── bank.jsonl              # The 100-scenario bank
│   └── holdout.lock            # Per-record digests behind HOLDOUT_DIGEST
│
├── docs/
│   ├── rubric.md               # The rating rubric humans and the judge are scored against
│   ├── preregistration.md      # Analysis plan; registration dropped by D10, kept as a record
│   ├── limitations.md          # Kept-current limitations record (build plan v3 §17)
│   ├── decisions/              # Dated log of the foundational build decisions
│   └── reporting/              # TRIPOD-LLM and CHART checklists, completed as an appendix
│
├── figures/                    # Empty and unused; every artifact goes to runs/repro/
├── dumps/                      # Local database dumps (gitignored, never committed)
├── runs/                       # Run artifacts: caches, journals, repro output (gitignored)
└── tests/
    ├── unit/                   # One directory per carelite package
    └── security/               # Adversarial input corpus (injection, PHI, red-flag)
```

---

## Literature Corpus

The knowledge base is grounded in 33 retrieved peer-reviewed sources, not the "approximately fifty" earlier planning estimated. The manifest lists 43 unique DOIs; 10 did not resolve to an open-access PDF — nine are `nihms*` manuscripts that both Unpaywall and the NCBI ID-converter report as not licensed for programmatic retrieval, readable on PMC in a browser but not fetchable — and they are not in the corpus.

That loss is not evenly spread across the seven themes; `docs/limitations.md` §1 has the coverage table. It cost the corpus some frequently-cited anchors specifically. Flickinger et al. (2016) on empathy and medication self-efficacy, Yen and Leasure (2019) on teach-back, and Park et al. (2020) on racial disparities in emotion response are all in the unresolved set and are **not** in this project's evidence base, however often they appear in the surrounding literature.

What the retrieved corpus does anchor: Wilson et al. (2010) on shared decision-making and asthma adherence, Talevski et al. (2020) — a systematic review that alone accounts for 12 of the 15 teach-back knowledge-base entries — and Roberts et al. (2021) on the socioeconomic and racial empathy gap. `data/fetch_corpus.py` rebuilds this corpus from its embedded DOI manifest, and `data/pdfs/_manual_needed.csv` is the honest record of what did not resolve.

---

## Status

The four planned deliverables are built with one exception. The structured knowledge base, the prompt
architecture, and the evaluation framework exist and have been run; the refined twenty-to-thirty
behavior list does not, and what stands in its place is the action-type tagging on the knowledge base
described above.

The table below is what is actually built and running, queried against the live database and the
repository on 2026-09-01. **Counts here are a hand-written snapshot and some of them move as scoring
proceeds; `runs/repro/headline-numbers.txt` is the authority** — `make reproduce` writes it from
Postgres with each figure printed beside the qualification it cannot honestly be quoted without, and
it exists because a planning document in this repository was once written from numbers carried
forward in memory. `docs/decisions/README.md` holds the reasoning behind each row and
`docs/limitations.md` holds what each one does not claim.

| Component | Status |
|---|---|
| Corpus retrieval | Built — 33 of 43 manifest DOIs resolved; the rest are recorded as genuinely unavailable in `data/pdfs/_manual_needed.csv`, not silently dropped |
| Knowledge base extraction and provenance validation | Built — 116 entries, every one verbatim-span-validated and **none human-reviewed** (`human_verified = false` on all of them); roughly a third restate one another (`docs/limitations.md` §2) |
| Equity knowledge-base re-extraction (`DECISIONS.md` D3) | Complete — negative result: zero net new equity entries, established as a property of the corpus rather than an unfinished extraction |
| Scenario bank and frozen holdout split | Complete — 100 scenarios (40 train / 60 holdout), checksummed and write-once |
| Dense + lexical index | Built and verified — 471/471 chunks embedded; the retrieval-quality gate passes 12/12 probes, re-run against the live index on 2026-09-01 |
| Hybrid retrieval (RRF, rerank, CRAG, HyDE) | Built, with the R0–R9 ablation harness |
| Curated graph layer | Built — 715 edges over the knowledge base |
| Safety screens (input/output, injection, PHI, red-flag) | Built, and observed refusing real generated text at scale rather than only in tests (`DECISIONS.md` D12) |
| Rubric (11-dimension NURSE / Four Habits) | Built, with anchored examples and a calibration set |
| Generation orchestration (six experimental conditions) | Complete — `generation` holds 1,119 rows: 180 cells for each of the six conditions, plus the 39 partial Ollama LC cells `DECISIONS.md` D11 stopped at, which D13 retains as a paired backend-equivalence sample belonging to no analysis arm. The LC arm is `served_by = 'vllm'` and nothing else |
| LLM-as-judge and its validation study | Built and running against the generated cells; the scored-row count moves, so read `runs/repro/headline-numbers.txt` rather than any number here. The judge-validation study itself has **not** run, which is why every judged result is exploratory. The instrument findings it did surface — `naturalness` and `ritualistic` are measurement failures rather than null results — are in `docs/limitations.md` §4 |
| Human-rating harness | Built, exercised only against synthetic raters — **no real human rating has occurred**, and `rating_assignment` is empty |
| Terminal (CLI) interface | Built |
| Statistical analysis (Friedman/Wilcoxon/Holm-Bonferroni, mixed-effects) | Built and run. `runs/repro/analysis.txt` is regenerated from the database and is where effect sizes and corrected tests live; `carelite/stats/RESULTS.md` is the write-up |
| Figures | Built (`carelite/viz/`) — `make reproduce` writes PNG+PDF pairs into `runs/repro/`, skipping with a stated reason any figure whose source table is still empty rather than failing or silently omitting it |
| Secondary outcome 3 (Condition C vs. Condition LC) | **No result is stated here.** D11 closed this comparison as untestable and D13 re-opened it by completing LC, in the reduced form D7 already fixed — whether query-dependent selection beats a fixed context, not whether retrieval beats stuffing the corpus in. C was served by Ollama and the LC arm by vLLM, so the comparison carries a serving-stack confound wherever it is printed. Whether it has been scored yet is a question for `runs/repro/headline-numbers.txt` |
| OSF pre-registration | **Retired by decision, not pending** — `DECISIONS.md` D10: this is a local proof of concept, not being published or submitted, so registration was dropped rather than completed. `docs/preregistration.md` is kept as a timestamped record of the analysis plan as it stood before evaluation data existed |
| Reporting checklists (TRIPOD-LLM, CHART) | Drafted as a living appendix (`docs/reporting/`); items that depend on results are marked pending, not filled in |
| `make reproduce` | Built. One command regenerates every table and figure from the database — six tables, including `headline-numbers.txt`, and ten figure files, plus a per-stage row-count census. It runs no inference and is safe to repeat |

**Every result this project has produced is descriptive and exploratory.** `DECISIONS.md` D10 dropped
the pre-registration, so nothing here is confirmatory or hypothesis-testing; no dimension has cleared
a judge-agreement threshold, because the judge-validation study has not run; and no human rating
exists. Completing Condition LC under a faster serving stack added a comparison and strengthened
nothing.

---

*CARELite AI | University of Arizona College of Medicine – Phoenix | DBMI Summer Internship 2026*
