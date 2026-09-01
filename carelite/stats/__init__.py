"""The pre-specified statistical analysis (docs/preregistration.md §8, v3 §14).

The plan is fixed in `docs/preregistration.md` and this package implements it;
it does not design it. Where the plan is silent -- the sidedness of the tests,
the point estimator behind the bootstrap interval, the numeric cut for "poor"
judge self-consistency -- the module that hits the gap says so in its docstring
and carries the choice on its output rather than burying it. Those are
pre-registration amendments to make before registration, not decisions for this
package to take quietly.

Three invariants hold across every module here:

* **`ritualistic` is reverse-coded and `to_quality()` is the only way onto a
  common polarity.** Aggregation reads the `quality` column that
  `carelite.stats.measures.attach_quality` derives, and never `raw`.
* **The three samples in a scenario x condition cell are not independent.** They
  are averaged into a cell mean for the paired tests and absorbed by a random
  intercept for scenario in `carelite.stats.mixed`; they are never counted as
  1,080 units.
* **Every result carries its own reporting status.** Not pre-specified, or the
  judge did not clear §9's threshold on a constituent dimension, and the result
  object says EXPLORATORY -- in the structure, not only in the prose.
"""

from carelite.stats.data import load_judge_samples, load_scores
from carelite.stats.effects import PairedEffects, bootstrap_ci, paired_effects
from carelite.stats.evidence import EvidenceStatus, Label, RaterScope, label_for
from carelite.stats.headline import GenerationCounts, HeadlineNumbers, headline_numbers
from carelite.stats.measures import (
    FOUR_HABITS_COMPOSITE,
    MEASURES,
    NURSE_COMPOSITE,
    Measure,
    attach_quality,
    cell_means,
    measure,
)
from carelite.stats.mixed import fit_random_intercept, variance_components_moments
from carelite.stats.negative_control import negative_control
from carelite.stats.power import build_power_report, detectable_effect, required_n
from carelite.stats.primary import (
    CONFIRMATORY_FAMILY,
    PRESPECIFIED_HYPOTHESES,
    FamilyResult,
    friedman_across_conditions,
    holm_bonferroni,
    run_family,
    wilcoxon_paired,
)
from carelite.stats.report import AnalysisReport, run_analysis
from carelite.stats.sensitivity import run_all_sensitivity
from carelite.stats.subgroups import equity_subgroup, exploratory_subgroup

__all__ = [
    "CONFIRMATORY_FAMILY",
    "FOUR_HABITS_COMPOSITE",
    "MEASURES",
    "NURSE_COMPOSITE",
    "PRESPECIFIED_HYPOTHESES",
    "AnalysisReport",
    "EvidenceStatus",
    "FamilyResult",
    "GenerationCounts",
    "HeadlineNumbers",
    "Label",
    "Measure",
    "PairedEffects",
    "RaterScope",
    "attach_quality",
    "bootstrap_ci",
    "build_power_report",
    "cell_means",
    "detectable_effect",
    "equity_subgroup",
    "exploratory_subgroup",
    "fit_random_intercept",
    "friedman_across_conditions",
    "headline_numbers",
    "holm_bonferroni",
    "label_for",
    "load_judge_samples",
    "load_scores",
    "measure",
    "negative_control",
    "paired_effects",
    "required_n",
    "run_all_sensitivity",
    "run_analysis",
    "run_family",
    "variance_components_moments",
    "wilcoxon_paired",
]
