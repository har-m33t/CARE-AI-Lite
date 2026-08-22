# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

CARELite AI is a research/prototyping project from the DBMI Summer Internship at the University of Arizona College of Medicine – Phoenix (June 8 – August 15, 2026). It is **primarily a documentation and knowledge-engineering repo, not an application**. The deliverables are a structured knowledge base, a behavior list, a prompt architecture, and an evaluation framework — all grounded in a corpus of ~50 peer-reviewed papers on clinician–patient communication.

There is no build system, package manager, test suite, or application code. The only executable file today is `data/fetch_corpus.py`.

## Current state vs. README

`README.md` documents the *intended* layout (`literature/`, `framework/`, `knowledge_base/`, `behaviors/`, `docs/`). **None of those directories exist yet.** The working tree contains only `README.md` and `data/fetch_corpus.py`. When creating those documents, follow the README's structure and file names rather than inventing new ones.

## Corpus retrieval

`data/fetch_corpus.py` rebuilds the paper corpus as real PDFs. The DOI manifest is embedded in the file itself — it is intentionally self-contained and has no repo dependencies.

```bash
pip install requests
python data/fetch_corpus.py --email you@example.com          # -> ./carelite_pdfs/
python data/fetch_corpus.py --email you@example.com --out DIR
```

Behavior worth knowing before editing it:
- Resolution order per DOI is Unpaywall (`--email` is required by their terms) → NCBI ID-converter → PMC PDF URL.
- Downloads are validated against a `%PDF` magic-byte check so paywall HTML is never saved as `.pdf`; failures are unlinked.
- Re-running is safe — existing output files are skipped.
- The manifest's 4th column (`duplicate_of`) marks byte-identical papers so they are fetched once. Rows with an empty DOI are never fetched; they land in `carelite_pdfs/_manual_needed.csv` along with any runtime failures.
- Requests are rate-limited with `time.sleep(1)` to stay polite to the free APIs — keep that if you extend the loop.
- Downloaded PDFs go to `carelite_pdfs/` and are not tracked in git.

## Domain model — the two structures everything hangs off

**Seven communication themes** (defined in README, derived from the literature, not adopted from an existing framework): Empathy; Emotion Recognition and Response; Patient Activation and Shared Decision-Making; Comprehension Confirmation (Teach-Back); Plain Language and Information Clarity; Trust and Relational Continuity; Equity-Aware Communication.

**Knowledge base entries** are seven-field records: Theme, Source (citation + DOI), Key Finding, Practical Takeaway, Example Behavior, Evidence Strength (Strong/Moderate/Emerging), AI Action Type (Detection/Generation/Reframing). Any new KB entry must fill all seven fields and cite a real paper in the corpus.

**Behaviors** are typed by the same three AI action types: *Detection* (flag something the clinician might miss), *Generation* (produce a prompt/response), *Reframing* (rewrite something already said). The master list holds 33; the deliverable is a refined, non-overlapping 20–30.

## Design constraints from the literature

These are project positions, not stylistic preferences — do not write content that violates them:
- Not a diagnostic tool; it makes no clinical recommendations.
- **Not a script generator.** The literature is explicit that communication frameworks which become scripts stop working. Outputs should recognize the *kind of moment* and support a response, never dictate exact wording.
- Equity findings (SES empathy gap, emotional blocking of minority patients, lower-quality LEP conversations) are the baseline the system corrects, not replicates.
- Every claim should trace to a source in the corpus. Prefer citing an anchoring paper over asserting a general finding.

## Prose style

The README is written in a deliberate register: declarative, evidence-first, no marketing language, no bullet-point padding, no hedging filler. Match it in new documentation — full sentences over fragmented bullets, specific findings over generic claims.
