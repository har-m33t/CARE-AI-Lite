"""The stratified 100-scenario evaluation bank and its frozen train/holdout split.

Everything in `scenarios/bank.jsonl` is synthetic. It was written for this study
and contains no real patient utterance. That is a hard constraint, not a
convention: the repository has a public remote.

Four modules:

* `bank`     load and validate the 100 records; `train_scenarios()` and
             `holdout_scenarios()` are the two splits.
* `audit`    prove every gated stratum cell is populated; fails loudly with a
             full enumeration. `python -m carelite.scenarios.audit`.
* `freeze`   the write-once guarantee on the 60 held-out scenarios.
* `leakage`  detect held-out text copied into an optimised prompt.

The one rule a caller needs: **prompt optimisation sees `train_scenarios()` and
nothing else.**
"""

# `audit` is deliberately NOT re-exported here: it is runnable as
# `python -m carelite.scenarios.audit`, and importing it from this package's
# __init__ makes runpy warn about a double import. Import it directly.
from carelite.scenarios.bank import (
    CuratedScenario,
    equity_scenarios,
    for_final_evaluation,
    for_optimisation,
    holdout_scenarios,
    load_bank,
    scenarios,
    train_scenarios,
)
from carelite.scenarios.freeze import HOLDOUT_DIGEST, verify_holdout
from carelite.scenarios.leakage import assert_no_holdout_leakage, find_leaks

__all__ = [
    "HOLDOUT_DIGEST",
    "CuratedScenario",
    "assert_no_holdout_leakage",
    "equity_scenarios",
    "find_leaks",
    "for_final_evaluation",
    "for_optimisation",
    "holdout_scenarios",
    "load_bank",
    "scenarios",
    "train_scenarios",
    "verify_holdout",
]
