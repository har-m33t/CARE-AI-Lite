# TRIPOD-LLM checklist — completed as an appendix

**Guideline:** Gallifant J, Afshar M, Ameen S, et al. The TRIPOD-LLM reporting guideline for
studies using large language models. *Nature Medicine*. 2025;31(1):60–69.
doi:10.1038/s41591-024-03425-5. Also at the EQUATOR Network
(equator-network.org/reporting-guidelines/the-tripod-llm-reporting-guideline-for-studies-using-large-language-models/)
and the interactive completion tool at tripod-llm.vercel.app. TRIPOD-LLM is maintained as a living
guideline with quarterly expert-panel review; this appendix was completed against the version
published 2025-01-08 and should be re-checked against the interactive tool if registration is
delayed past a quarterly review cycle.

**Applicability to this project.** TRIPOD-LLM's item set is modular by research design (de novo
development, methods, evaluation, or healthcare-settings evaluation) and by LLM task. This project
is a **healthcare-settings evaluation** of an LLM-based system (Conditions A/A2/B/C/LC/D compared
against each other, not against a deployed clinical baseline), so the healthcare-settings-only
items (marked below) apply. The task is closest to **generation/documentation** (producing a
clinician-facing response to a patient utterance) rather than classification or summarization, so
items 7e (comparison to benchmarks) and the summarization-specific item 10 are marked not
applicable.

**Status.** Completed against the current state of the repository as of 2026-08-24. Items that
depend on data that does not exist yet (results, human-rating, judge-validation outcomes) are
marked pending and point to the document that will hold the answer once the run completes.

| # | Item | Status | Where answered / notes |
|---|---|---|---|
| 1 | Identify the study as developing/fine-tuning/evaluating an LLM; specify task, population, outcome | Done | Evaluation (not fine-tuning) of an LLM-based clinician-communication support system; task = generation of a clinician turn in response to a patient utterance; population = synthetic patient-utterance scenarios; outcome = 11-dimension rubric score. `README.md`, `docs/preregistration.md` §1–2. |
| 2 | TRIPOD-LLM for Abstracts (2a–2l) | Pending | No results abstract exists yet; template to be completed once `carelite.repro` produces a results summary. |
| 3a | Healthcare context / use case and rationale | Done | `README.md` "Project Goal" and "Background." |
| 3b | Target population and intended use / users | Done | Clinicians during a patient visit, framework-guided real-time communication support; not diagnostic, not a script generator (`README.md` "What This Project Is Not"). |
| 4 | Study objectives, including development vs. validation | Done | Evaluation study comparing six conditions on a frozen holdout set; `docs/preregistration.md` §1–4. |
| 5a | Data sources per dataset (train/tuning/eval) | Done | Corpus: 33 papers via `carelite/corpus/fetch.py`. Scenarios: 100 synthetic scenarios, 40 train / 60 holdout, `carelite/scenarios/`. No LLM fine-tuning occurs — all generators are used off-the-shelf. |
| 5b | Quantitative/qualitative description of data points | Done | `docs/limitations.md` §1 (corpus theme coverage table), §3 (scenario bank composition). |
| 5c | Oldest/newest text date | Partial | Paper publication years recorded in `paper.year`; oldest/newest to be reported from that column in the results write-up. Not yet tabulated. |
| 5d | Data pre-processing and quality checking | Done | `carelite/kb/spans.py` (span-location normalization), `carelite/kb/validate.py` (provenance gate); `docs/limitations.md` §2. |
| 5e | Missing/imbalanced data handling | Done | `docs/preregistration.md` §10 (exclusion criteria: regeneration on failure, missing judge spans treated as missing not imputed); `docs/limitations.md` §3 (empty equity-stratum cell, not backfilled). |
| 6a | LLM name, version, last training date | Done | `carelite/config.py.Models`; a digest is recorded on every generation rather than a tag (v3 §16), since tags are mutable. The `make pin-models` target that was meant to capture them up front references `carelite.models.pin`, a module that does not exist in this repository — `REPRODUCE.md` §6 gives the manual `curl` equivalent, and `generation.model_digest` is the durable record either way. Training cutoff dates are Ollama model-card metadata, to be captured verbatim in the results appendix at run time. The vLLM-served LC rows identify the model as `vllm:<repo id>@<revision>` rather than by a blob hash, since vLLM serves HF safetensors and has no equivalent — the served commit is pinned with `--revision` and the run refuses to start if none resolves. |
| 6b | LLM development process (architecture, training, fine-tuning) | Not applicable | No LLM in this study is trained or fine-tuned by this project; all are used as distributed. |
| 6c | Text generation details, prompt engineering | Done | `carelite/prompts/` (versioned prompt files per condition), `carelite/generate/` (orchestration). Prompt version recorded per generation. |
| 6d | Initial and post-processed LLM output | Done | `generation.response` stores the raw model output; the CLI/output-safety gate is the only post-processing step and its verdict is stored separately (`safety_verdict`), not merged into the response text. |
| 6e | Classification/probability details | Not applicable | The system generates free text; it does not classify or emit probabilities. |
| 7a | Generative-output quality metrics (consistency, relevance, accuracy) | Done | `docs/rubric.md`'s 11 dimensions; judge self-consistency (`carelite/eval/judge/validation.py`) reported as a stability metric, not a pass/fail gate. |
| 7b | Outcome metrics' relevance to deployment task | Done | `docs/limitations.md` §6: rubric scores adherence to communication frameworks, explicitly not a patient-reported outcome or a deployment claim. |
| 7c | Outcome definition and calculation | Done | `docs/preregistration.md` §3–4 (primary/secondary outcome definitions via `to_quality()`). |
| 7d | Assessor qualifications for subjective outcomes | Pending | Human raters not yet recruited; qualifications (e.g., medical/nursing student status) to be recorded once §12 of `docs/preregistration.md` is executed. |
| 7e | Comparison to other LLMs, humans, benchmarks | Partial | Cross-model baseline (A2) is within-study, not an external benchmark; no external LLM benchmark comparison is in scope. Human comparison is §12 of the pre-registration, pending. |
| 8a–8c | Annotation (labeling process, annotator count, annotator background) | Pending | Rubric scoring is the annotation act here (rater_type ∈ {deterministic, llm_judge, human}); human annotator counts/background pending recruitment. Judge "annotation" is the prompt in `carelite/eval/judge/prompt.py`. |
| 9a | Prompt design process | Done | `carelite/prompts/README.md` and per-condition prompt files; iterated against the 40-scenario train split only, never against holdout. |
| 9b | Data used to develop prompts | Done | Train-split scenarios (40 of 100) exclusively; `docs/preregistration.md` §6 states holdout is never used for development. |
| 10 | Summarization pre-processing | Not applicable | This system does not perform document summarization. |
| 11 | Instruction tuning/alignment strategies | Not applicable | No instruction tuning is performed; all models are used with off-the-shelf instruction tuning as distributed by their publishers. |
| 12 | Compute / cost / inference time / FLOPs | Done | Two serving stacks, recorded per row in `generation.served_by`. **Ollama on a rented Runpod L40S (48 GB)**, four parallel workers by condition: 939 generations, zero failures; judging 939/939 in 206 minutes. Condition LC cost ~33x the other conditions per cell on that stack and was stopped at 39/180 cells (`DECISIONS.md` D11). **vLLM 0.28.0 with `--enable-prefix-caching` on one A100 SXM (80 GB)**, `google/gemma-4-12B-it` pinned at revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`: the same LC cells cost 3.61 s per warm cell (mean of 9; min 2.03, max 6.51) after one 64.31 s cold prefill of 110,653 tokens, completing all 180 in ~21 minutes for ~$1.38 — a 54.9x difference attributable to prefix-cache reuse, not to the condition (`DECISIONS.md` D13). Current census 1,119 generations; read `runs/repro/headline-numbers.txt` rather than this cell for the live figure. See `docs/limitations.md` §4 and `REPRODUCE.md` §7 for the full cost accounting. |
| 13 | IRB / ethics committee | Not applicable | No human-subjects data; all scenarios are synthetic and all corpus papers are already-published, publicly available literature. Human rating (§12 of pre-registration) uses external raters scoring de-identified synthetic text, not patient data — to be confirmed against the recruiting institution's own policy before recruitment begins. |
| 14a | Funding source and role | Pending | Project context: DBMI Summer Internship, University of Arizona College of Medicine – Phoenix. No external funding identified as of this writing. |
| 14b | Conflicts of interest | Done | None declared. |
| 14c | Study protocol location (healthcare settings) | Done | This repository; `docs/preregistration.md` is the protocol. |
| 14d | Registration information (healthcare settings) | Not applicable | **Retired by decision, not pending — `DECISIONS.md` D10.** This is a local proof of concept; OSF registration was dropped rather than completed, and there is no registration URL to report. `docs/preregistration.md` is kept as a timestamped analysis-plan record instead. |
| 14e | Data availability | Done | Public repository (see repo README); no PDFs, database dumps, or real patient data committed, per fleet rule 4. Synthetic scenarios and generated text are committed or reproducible via `make reproduce`. |
| 14f | Code availability | Done | This repository; `REPRODUCE.md` and `make reproduce`. |
| 15 | Patient and public involvement | Not applicable | No patients or public contributors were involved in scenario design, prompt design, or rubric design; all scenarios are synthetic and authored by the project team. Stated plainly rather than omitted. |
| 16a–16d | Participant/data flow, characteristics, outcome counts (healthcare settings) | Pending | No patient/EHR data is used, so 16a/16b/16d reduce to the scenario-bank flow (`carelite/scenarios/audit.py` coverage report) rather than a patient flow diagram; 16c (clinical-outcome variable comparison) is not applicable — this study has no clinical outcome. |
| 17 | LLM performance per pre-specified metrics/human evaluation | Partial | The five Ollama conditions are generated and judged (939 generations, judged 939/939). Condition LC's 180 vLLM cells are generated and **not yet scored**, so no C-vs-LC comparison exists and none should be reported. **Per `DECISIONS.md` D10 no metric here is confirmatory or pre-specified in a registered sense.** The headline instrument finding: `naturalness` and `ritualistic` — the two dimensions carrying build plan v3's most-predicted effect — are measurement failures, not null results (`ritualistic` scored 1 on 99% of 921 scored rows; `naturalness` discrimination ratio 0.68). Six of eleven dimensions discriminate meaningfully (`docs/limitations.md` §4). Statistical write-up (effect sizes, corrected tests) owned by `carelite-stats`, in progress separately. Human evaluation remains not yet conducted. |
| 18 | LLM updating results | Not applicable | No LLM updating (fine-tuning, retraining) occurs during this study. |
| 19a | Overall interpretation, including fairness | Partial | `docs/limitations.md` §4 has the process-level interpretation (what could and could not be measured, and what a reader must not conclude); the statistical interpretation of effect sizes is `carelite-stats`'s write-up, in progress. Fairness: the equity knowledge base holds 3 entries as a property of the corpus rather than the extraction (`DECISIONS.md` D3's outcome), and the `racial_ethnic` scenario axis narrows to a single mechanism (D5) — `docs/limitations.md` §3 states both together as one finding about this evidence base's coverage of disparity vs. remedy. |
| 19b | Limitations and their effect on bias/uncertainty/generalizability | Done (living) | `docs/limitations.md`, kept current. |
| 19c | Known challenges using data for this task/domain (healthcare settings) | Done | `docs/limitations.md` §1 (corpus skew toward training studies), §3 (equity-axis mechanism confound). |
| 19d | Intended use for the implementation evaluated (healthcare settings) | Done | `README.md` "What This Project Is Not"; `docs/limitations.md` §6, "no clinical deployment claim." |
| 19e | Assessing poor-quality/unavailable input data (healthcare settings) | Done | `carelite/safety/` input screens (injection, PHI, red-flag detection) run before generation; CRAG (`crag_grade`) assesses retrieval-input quality specifically. |
| 19f | Whether users are required to interact (healthcare settings) | Done | The system is clinician-facing and requires an active clinician turn-by-turn interaction; it does not act autonomously on a patient. |
| 19g | Next steps for future research | Pending | To be written once `carelite-stats`'s statistical write-up lands; `docs/limitations.md` §4's instrument findings (the `naturalness`/`ritualistic` degeneracy, the variance-bounds-agreement result) are themselves a research direction — a judge or rubric revision that increases discrimination on those two dimensions specifically. |

**Summary:** the holdout run completed 2026-08-25 (939 generations, 939/939 judged), so item 12
moved to Done and items 17/19a moved from Pending to Partial — the process-level record is complete
in `docs/limitations.md` §4, and what remains pending on those two items is `carelite-stats`'s
statistical write-up specifically, not this project's data collection. **Updated 2026-09-01 for
`DECISIONS.md` D13:** Condition LC was completed under a second serving stack, bringing `generation`
to 1,119 rows, and item 12 now reports both stacks. That adds one comparison to the study and
changes nothing about its evidential status — the instrument findings in item 17 stand unaltered,
the judge validation study has still not run, and the new LC cells are not yet scored. Genuinely
still pending:
7d, 8a–8c (human-rater characteristics — recruitment has not occurred), 14a (funding), 16a–16d
(scenario-bank flow in place of a patient-data flow, per that row's own reasoning — could now be
filled in from `carelite/scenarios/audit.py`'s coverage report but has not been), 19g (next steps,
written alongside the statistical results). **Per `DECISIONS.md` D10, every result reported
anywhere in this appendix is descriptive, not confirmatory or pre-specified in a registered
sense**, since OSF registration was dropped by decision (item 14d) rather than merely not yet
completed.
