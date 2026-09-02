carelite reproduce
========================================
database: connected, schema present

pipeline stage row counts:
         33  corpus fetch (`paper`)
        471  chunking (`chunk`)
        116  knowledge base (`kb_entry`)
        116  kb provenance (`kb_entry_source`)
        715  graph layer (`graph_edge`)
        100  scenario bank (`scenario`)
          7  prompt versions (`prompt_version`)
      1,119  generation (`generation`)
        180  retrieval traces (`retrieval_trace`)
      1,119  rubric scoring (`rubric_score`)
          0  human rating assignment (`rating_assignment`)

1,119 generations present — the holdout run is complete. All six conditions ran to 1,080 cells, plus the 39 LC cells retained from the run DECISIONS.md D11 stopped. Condition LC was completed under D13 on a serving stack with prefix caching; the retained cells are a backend-equivalence sample and belong to no analysis arm — see docs/limitations.md SS4.

downstream (tables and figures):
  [ok]       carelite.stats.reproduce: wrote 6 tables
               /Users/harmeetsingh/UofA/CARE-AI-Lite/runs/repro/analysis.txt
               /Users/harmeetsingh/UofA/CARE-AI-Lite/runs/repro/headline-numbers.txt
               /Users/harmeetsingh/UofA/CARE-AI-Lite/runs/repro/headline-numbers.csv
               /Users/harmeetsingh/UofA/CARE-AI-Lite/runs/repro/effect-sizes.csv
               /Users/harmeetsingh/UofA/CARE-AI-Lite/runs/repro/instrument-resolution.csv
               /Users/harmeetsingh/UofA/CARE-AI-Lite/runs/repro/data-inventory.csv
  [ok]       carelite.viz.reproduce: wrote 10 figures
               /Users/harmeetsingh/UofA/CARE-AI-Lite/runs/repro/01_rubric_scores.png
               /Users/harmeetsingh/UofA/CARE-AI-Lite/runs/repro/01_rubric_scores.pdf
               /Users/harmeetsingh/UofA/CARE-AI-Lite/runs/repro/02_effect_sizes.png
               /Users/harmeetsingh/UofA/CARE-AI-Lite/runs/repro/02_effect_sizes.pdf
               /Users/harmeetsingh/UofA/CARE-AI-Lite/runs/repro/06_retrieval_quality.png
               /Users/harmeetsingh/UofA/CARE-AI-Lite/runs/repro/06_retrieval_quality.pdf
               /Users/harmeetsingh/UofA/CARE-AI-Lite/runs/repro/07_equity_subgroup.png
               /Users/harmeetsingh/UofA/CARE-AI-Lite/runs/repro/07_equity_subgroup.pdf
               /Users/harmeetsingh/UofA/CARE-AI-Lite/runs/repro/08_negative_control.png
               /Users/harmeetsingh/UofA/CARE-AI-Lite/runs/repro/08_negative_control.pdf
