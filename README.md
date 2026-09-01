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

The system is built on seven evidence-based communication categories derived from synthesis of the peer-reviewed literature (33 papers retrieved into the corpus as of this writing — see "Literature Corpus" below). These categories were not selected from existing frameworks — they were identified through pattern analysis across the literature and then validated against the strongest findings in the corpus.

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

The knowledge base currently holds 116 entries spanning all seven themes (activation_sdm 40, plain_language 21, teach_back 15, trust_continuity 14, empathy 14, emotion_response 9, equity 3), loaded into PostgreSQL from the retrieved paper corpus. Entries are produced by **LLM-assisted extraction with automated verbatim-span provenance validation, not human curation or clinician review**: every entry's quoted span is mechanically confirmed to appear in the extracted text of the paper it cites before the entry is accepted, and an entry whose span cannot be located is rejected as a fabrication rather than repaired. That check is real and enforced in code (`carelite/kb/validate.py`), and it is also specific about what it does not claim — no person has reviewed whether a given finding follows from its quoted span, `human_verified` is `false` on every loaded entry, and any result built on the knowledge base inherits that limitation. **116 is not 116 independent findings**: roughly a third of the entries restate one another in different words, so an entry count must never stand in for convergent evidence — `docs/limitations.md` §2 has the redundancy-cluster accounting. The equity count of 3 is not a gap awaiting more extraction either — the same section records why a targeted re-extraction attempt returned nothing (two of the entries above), which is itself a finding about this corpus, not a pipeline shortfall. `docs/decisions/README.md` records why the knowledge base is derived this way rather than hand-authored to a planning-time count.

---

## Actionable Behavior System

Every knowledge-base entry is tagged with one of three functional categories — `AI Action Type` in the table above — rather than being organized into a separate, curated master list of behaviors as originally planned. As of this writing that tagging spans the 116 loaded entries: 73 generation, 26 reframing, 17 detection. A refined, prioritized, non-overlapping behavior list distilled from these tags remains a planned deliverable — see "Expected Outcomes" below — not a finished artifact.

**Detection behaviors** — the system monitors the conversation and flags something the clinician might miss. Examples include detecting emotional blocking patterns, flagging jargon, and identifying conversations that end without a teach-back check.

**Generation behaviors** — the system produces a specific response, prompt, or communication element. Examples include generating teach-back prompts at key handoffs, producing activating follow-up questions after emotional acknowledgments, and generating barriers checks after every treatment plan.

**Reframing behaviors** — the system rewrites or corrects something already said. Examples include replacing patient-blame language with barrier-attribution language, offering plain-language alternatives to jargon-heavy explanations, and generating alternative openings when emotional blocking is detected.

---

## Expected Outcomes

By the end of the internship period, this project is expected to produce the following deliverables.

**A structured knowledge base** containing evidence-based entries across all seven communication themes, each linking a specific research finding to a practical clinical behavior and an AI action type.

**A refined behavior list** of twenty to thirty prioritized, non-overlapping, evaluation-ready communication behaviors derived from the master list and grounded in evidence strength.

**A prototype prompt architecture** demonstrating how the system would detect, generate, and reframe in real clinical conversation contexts — including example inputs, expected outputs, and the evidence rationale behind each.

**An evaluation framework** defining what good looks like for each behavior: what the system should produce, how that output would be assessed, and what the evidence-based standard for success is.

**Project documentation** including this README, a literature synthesis, a communication themes framework, evidence summaries for each theme, and a versioned record of all design decisions made during the internship.

---

## What This Project Is Not

CARELite AI is not a diagnostic tool. It does not make clinical recommendations. It does not replace the clinician's judgment about what a patient needs. It does not generate empathy — it supports the conditions under which a clinician can express it more consistently and more equitably.

The system is also not a script generator. The literature is explicit on this point: communication frameworks that become scripts stop working. The goal is a system that recognizes what kind of moment a clinician is in and supports the right kind of response — not one that tells a clinician exactly what to say.

---

## Project Structure

The layout below is the actual current tree, not the intended one — the version this section
described until 2026-08-24 documented four directories (`literature/`, `framework/`, `behaviors/`,
and most of `docs/`'s planned contents) that were never created, alongside a status table that
called the project further along than it was. The build is substantially complete — see "Status"
below for what remains, and read `runs/repro/headline-numbers.txt` rather than any count in this
file, since `make reproduce` writes that file from the database and this one is written by hand.

```
carelite-ai/
│
├── README.md                    # This file
├── REPRODUCE.md                 # Cold-start reproduction instructions
├── DECISIONS.md                 # Dated record of project-owner decisions (D1-D13)
├── Makefile                     # install / check / db-up / eval-smoke / reproduce
├── pyproject.toml, uv.lock      # Toolchain: Python 3.13, uv-managed
│
├── carelite/                    # The application and evaluation-harness code
│   ├── types.py, config.py      # Frozen contracts: controlled vocabularies, model roster, seeds
│   ├── db/                      # PostgreSQL + pgvector schema and connection helpers
│   ├── corpus/                  # Fetch, extract, chunk the paper corpus
│   ├── kb/                      # Knowledge-base extraction, verbatim-span provenance, load
│   ├── scenarios/                # The 100-scenario bank; the frozen 60-scenario holdout split
│   ├── index/                   # Dense (pgvector) + lexical (tsvector) indexing
│   ├── retrieval/                # Hybrid retrieval: RRF fusion, rerank, CRAG, HyDE
│   ├── graph/                     # Curated property graph over the knowledge base (715 edges)
│   ├── generate/                 # The six conditions; orchestration; Ollama and vLLM backends
│   ├── prompts/                  # Versioned prompt files per condition
│   ├── eval/
│   │   ├── rubric/               # The 11-dimension NURSE / Four Habits rubric, deterministic scorers
│   │   ├── judge/                 # LLM-as-judge and its v3 §13 validation study
│   │   └── human/                 # Blinded human-rating harness (synthetic-rater-exercised)
│   ├── safety/                    # Input/output safety: injection, PHI, red-flag screens
│   ├── stats/                     # Friedman/Wilcoxon/Holm-Bonferroni, mixed-effects, sensitivity
│   ├── viz/                       # Every figure, regenerated from the database
│   ├── cli/                       # The Typer + Rich terminal interface
│   └── repro.py                   # `make reproduce` entry point
│
├── data/
│   ├── fetch_corpus.py           # Thin shim -> carelite.corpus.fetch
│   └── pdfs/                     # Retrieved PDFs (gitignored) + _manual_needed.csv
│
├── knowledge_base/
│   ├── TAXONOMY.md               # Seven-theme taxonomy proposal (accepted, DECISIONS.md D1)
│   ├── review/                    # Generated human-review digest (0/116 signed off as of writing)
│   └── cache/                     # Extraction cache (gitignored)
│
├── scenarios/
│   ├── EQUITY_REVIEW.md          # Equity-stratum review packet and outcome (D2, D5)
│   ├── bank.jsonl                 # The 100-scenario bank
│   └── holdout.lock               # Per-record digests behind HOLDOUT_DIGEST
│
├── docs/
│   ├── rubric.md                  # The rating rubric humans and the judge are scored against
│   ├── preregistration.md         # Analysis plan; OSF registration dropped by D10, kept as a record
│   ├── limitations.md             # Kept-current limitations record (build plan v3 §17)
│   ├── decisions/                 # Dated decision log (foundational build decisions)
│   ├── reporting/                 # TRIPOD-LLM and CHART checklists, completed as an appendix
│   └── superpowers/specs/         # Orchestrating-session design records, not project docs
│
├── figures/                       # Present but unused; `make reproduce` writes to runs/repro/
├── dumps/                         # Local database dumps (gitignored, never committed)
├── runs/                          # Run artifacts: caches, journals, repro output (gitignored)
└── tests/
    ├── unit/                      # ~2,560 tests, one directory per carelite package
    └── security/                  # Adversarial input corpus (injection, PHI, red-flag)
```

---

## Literature Corpus

The knowledge base is grounded in 33 retrieved peer-reviewed sources, not the “approximately fifty” earlier planning estimated. The manifest lists 43 unique DOIs; 10 did not resolve to an open-access PDF (nine are `nihms*` manuscripts both Unpaywall and the NCBI ID-converter report as not licensed for programmatic retrieval, readable on PMC in a browser but not fetchable) and are not in the corpus. That loss is not evenly spread across the seven themes — `docs/limitations.md` has the coverage table — and it cost the corpus some frequently-cited anchors specifically: Flickinger et al. (2016) on empathy and medication self-efficacy, Yen and Leasure (2019) on teach-back, and Park et al. (2020) on racial disparities in emotion response are all in the unresolved set and are **not** in this project's evidence base, however often they appear in the surrounding literature. What the retrieved corpus does anchor: Wilson et al. (2010) on shared decision-making and asthma adherence, Talevski et al. (2020) — a systematic review that alone accounts for 13 of the 17 teach-back knowledge-base entries — and Roberts et al. (2021) on the socioeconomic and racial empathy gap. `data/fetch_corpus.py` rebuilds this corpus from its embedded DOI manifest; `data/pdfs/_manual_needed.csv` is the honest record of what did not resolve.

---

## Status

This table describes what is actually built and running today, queried against the live database
and the repository as of 2026-09-01, not a plan. **Counts here are a hand-written snapshot;
`runs/repro/headline-numbers.txt` is the authority** — `make reproduce` writes it from Postgres,
with each figure printed beside the qualification it cannot honestly be quoted without, and it
exists because a planning document in this repository was once written from numbers carried forward
in memory. See `docs/decisions/README.md` for the decisions behind each row and
`docs/limitations.md` for what each one does not claim.

| Component | Status |
|---|---|
| Corpus retrieval | Built — 33 of 43 manifest DOIs resolved; the rest are documented as genuinely unavailable, not silently dropped |
| Knowledge base extraction and provenance validation | Built — 116 entries (`DECISIONS.md` D3's outcome; ~1/3 restate one another, see `docs/limitations.md` §2), verbatim-span-validated; **not human-reviewed** (`human_verified = false` on all of them) |
| Equity knowledge-base re-extraction (`DECISIONS.md` D3) | Complete — negative result: zero net new equity entries; established as a property of the corpus (Roberts is a meta-analysis, Holdsworth is not), not an unfinished extraction |
| Scenario bank and frozen holdout split | Complete — 100 scenarios (40 train / 60 holdout), checksummed and write-once |
| Dense + lexical index | Built and verified — 471/471 chunks embedded, 10/10 retrieval probes passing |
| Hybrid retrieval (RRF, rerank, CRAG, HyDE) | Built |
| Curated graph layer | Built — 715 edges (`belongs_to` 116, `supports` 116, `has` 149, `appropriate_in` 160, `instantiates` 92, `restates` 82) |
| Safety screens (input/output, injection, PHI, red-flag) | Built |
| Rubric (11-dimension NURSE / Four Habits) | Built, with anchored examples and a calibration set |
| Generation orchestration (six experimental conditions) | Complete for all six at 180 cells each. A, A2, B, C and D were served by Ollama; **Condition LC was stopped at 39/180 under Ollama by `DECISIONS.md` D11 for cost, then completed in full under vLLM with prefix caching by `DECISIONS.md` D13** — 3.61 s per warm cell against Ollama's 198 s, a 54.9x difference that belongs to the serving stack and not to the condition. The 39 Ollama LC cells are retained as a paired backend-equivalence sample and belong to no analysis arm; the LC arm is `served_by = 'vllm'` and nothing else |
| LLM-as-judge and its validation study | The Ollama pass is judged: 939/939, zero errors, 206 minutes. **Condition LC's 180 vLLM cells are generated and being scored now, so `rubric_score` holds 939 rows against 1,119 generations** — the gap is exactly that set. Confirmatory status still blocked on human-rating data (unchanged); the n=30 validation study's instrument-check findings are confirmed at n=939 — see `docs/limitations.md` §4 for the `naturalness`/`ritualistic` measurement failure this surfaced |
| Human-rating harness | Built, exercised only against synthetic raters — **no real human rating has occurred** |
| Terminal (CLI) interface | Built |
| Full evaluation run | **`generation` holds 1,119 rows**: 180 each for the six conditions, plus the 39 Ollama LC cells D11 stopped at and D13 retained. Produced in two passes on rented GPU hardware — five conditions under Ollama on 2026-08-25, Condition LC under vLLM on 2026-09-01 (`DECISIONS.md` D11 and D13; `docs/limitations.md` §4). **Completing LC adds one comparison and strengthens nothing**: the primary contrast is unmoved, `ie`/`naturalness`/`ritualistic` stay degenerate, no human rating exists, and **every result is descriptive per `DECISIONS.md` D10 and exploratory for want of a judge-validation comparator** |
| Secondary outcome 3 (Condition C vs. Condition LC) | **No result exists yet.** D11 closed this comparison as untestable and D13 re-opened it by completing LC, in the reduced form `DECISIONS.md` D7 already fixed — whether query-dependent selection beats a fixed context, not whether retrieval beats stuffing the corpus in. The cells are generated and not yet scored, so this row is a seam and not a number |
| OSF pre-registration | **Retired by decision, not pending** — `DECISIONS.md` D10: this is a local proof of concept, not being published or submitted, so registration was dropped rather than completed. `docs/preregistration.md` is kept as a timestamped record of the analysis plan as it stood before evaluation data existed. |
| Statistical analysis (Friedman/Wilcoxon/Holm-Bonferroni, mixed-effects) | Built and run on the five judged conditions; `carelite/stats/RESULTS.md` is the write-up and `runs/repro/analysis.txt` is regenerated from the database. Effect sizes and corrected tests live there, not here |
| Figures | Built (`carelite/viz/`) — `make reproduce` currently writes ten files (five figures as PNG+PDF pairs) into `runs/repro/`, skipping with a stated reason any figure whose source table is still empty rather than failing or silently omitting it |
| Reporting checklists (TRIPOD-LLM, CHART) | Drafted as a living appendix (`docs/reporting/`); items that depend on results are marked pending, not filled in |
| `make reproduce` | Built. Both downstream lanes are wired under the `run(output_dir)` contract, so one command regenerates every table and figure from the database: six tables (including `headline-numbers.txt`, the block any write-up should quote from) and ten figure files, plus a per-stage row-count census. It runs no inference and is safe to repeat |

---

*CARELite AI | University of Arizona College of Medicine – Phoenix | DBMI Summer Internship 2026*