"""The handful of numbers that end up in prose, re-derived from the database every run.

**Why this module exists.** A planning document in this repository was written
against figures carried forward from memory — "939 generations", "+0.67,
p < 0.001" — and then reasoned backward toward the code that would justify them.
It happened to be right: the database independently holds 939 rows and the
mixed-effects fit independently reports `B vs A +0.6724 [+0.5408, +0.8040],
p = 1.33e-23`. Being right by luck is not a property anyone can rely on twice.

So the small set of figures a write-up actually quotes is emitted by `make
reproduce` as a block read out of Postgres, next to `analysis.txt` and the CSVs.
A document that disagrees with `runs/repro/headline-numbers.txt` is stale, and
the way to find that out is to run `make reproduce` rather than to argue from
recollection.

**Nothing here recomputes anything.** The two estimates come off the
`AnalysisReport` that `carelite.stats.report.run_analysis` already produced —
the same object `analysis.txt` is rendered from — so the headline and the full
report cannot disagree. The only thing this module reads for itself is the
generation census, and it reads that because it is a different question from the
one the analysis frame answers: `generation` holds 939 rows, the primary
comparison ran on 900 of them, and collapsing the two into one number is exactly
the drift this module is here to prevent. Both are printed, each saying what it
is.

**Every number carries its qualification, in the structure and not in the
prose.** `DECISIONS.md` D10 dropped the pre-registration: every result this
project produces is descriptive, and because the judge validation study has not
run, every model-based result is additionally EXPLORATORY. A block that printed
`+0.67, p < 0.001` stripped of that would be worse than no block at all, because
it is precisely the artefact someone pastes into a document. So `HeadlineNumber`
has no constructor path that produces a value without a `qualifier`, the
qualifier column sits immediately beside the value in the CSV, and the rendered
text prints the two together. A consumer can still quote the number alone; it
just cannot do so without having stepped over the sentence saying not to.

**The serving-stack breakdown is not decoration.** `generation.served_by`
records which stack produced a row. Every row in the current run is `'ollama'`,
and condition LC may yet be re-run under vLLM; a count pooled across two
backends would hide that the arm changed underneath it. The breakdown therefore
prints a zero for a backend that produced nothing rather than omitting it, and
names any condition that ended up split across both.

**What this module deliberately does not do.** It computes no C-vs-LC contrast
and no backend-equivalence test. Those belong to a later package and to the
generations it will produce; nothing here assumes those rows exist. The seam
they need is `GenerationCounts.by_condition_and_backend` and
`split_across_backends`, which already report the shape such an analysis would
have to reckon with.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from carelite.stats.data import SERVING_BACKENDS
from carelite.stats.evidence import D10_BANNER
from carelite.stats.measures import NURSE_COMPOSITE
from carelite.stats.report import AnalysisReport
from carelite.types import RUBRIC_DIMENSIONS, Condition

__all__ = [
    "HEADLINE_COLUMNS",
    "PRIMARY_CONTRAST",
    "GenerationCounts",
    "HeadlineNumber",
    "HeadlineNumbers",
    "flip_orientation",
    "headline_numbers",
    "write_headline",
]

#: The contrast the write-up leads with: condition B against condition A on the
#: composite NURSE outcome. Both headline estimates are reported in this
#: orientation, so a positive number always means B scored higher.
PRIMARY_CONTRAST = "B vs A"

#: The §8.1 family computes the primary comparison as `A vs B` (see
#: `carelite.stats.primary.PRESPECIFIED_HYPOTHESES`), while the mixed-effects
#: model contrasts every condition against the reference A. Reporting them side
#: by side without reconciling the orientation would print one estimate at
#: +0.6724 and the other at -0.667 for the same difference.
_PRIMARY_HYPOTHESIS_KEY = "primary_nurse_A_vs_B"

#: Column order for `headline-numbers.csv`. `qualifier` is adjacent to `value`
#: on purpose, the same way `not_testable` sits beside `significant` in
#: `effect-sizes.csv`: a consumer reading one has the other in view. Effect
#: columns precede `p_value`, which is §8.2's ordering and applies to a CSV like
#: any other table.
HEADLINE_COLUMNS: tuple[str, ...] = (
    "key",
    "value",
    "qualifier",
    "label",
    "ci_low",
    "ci_high",
    "p_value",
    "display",
    "section",
    "source",
)


def flip_orientation(
    point: float,
    ci_low: float,
    ci_high: float,
) -> tuple[float, float, float]:
    """Re-express a paired location shift in the opposite orientation.

    `HL(B - A) = -HL(A - B)` exactly, and the percentile interval of a negated
    statistic is the negated interval with its bounds swapped. So this is a
    relabelling rather than a re-estimate, and it is a function with a
    hand-checkable answer rather than a sign flip buried in a format string --
    a silently mis-signed headline would invert the study's main finding.
    """
    return (-point, -ci_high, -ci_low)


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenerationCounts:
    """What `generation` actually holds, before any analysis decision.

    `read` distinguishes "the table is empty" from "nobody asked it". Both are
    real states -- a fresh clone that has only run `make db-up` is the first;
    `write_tables` called from a test with no database is the second -- and a
    block that rendered them alike would report a zero it had not measured.
    """

    rows: tuple[tuple[str, str, str, int, int, int, int], ...] = ()
    read: bool = True

    @classmethod
    def not_read(cls) -> GenerationCounts:
        return cls(rows=(), read=False)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> GenerationCounts:
        """Build from the frame `carelite.stats.data.load_generation_counts` returns."""
        if frame is None or frame.empty:
            return cls(rows=(), read=True)
        records: list[tuple[str, str, str, int, int, int, int]] = []
        for row in frame.itertuples(index=False):
            records.append(
                (
                    str(row.condition),
                    str(row.served_by),
                    str(row.split),
                    int(row.n_generations),
                    int(row.n_gate_blocked),
                    int(row.n_scored),
                    int(row.n_scenarios),
                )
            )
        return cls(rows=tuple(records), read=True)

    @property
    def total(self) -> int:
        return sum(r[3] for r in self.rows)

    @property
    def n_gate_blocked(self) -> int:
        return sum(r[4] for r in self.rows)

    @property
    def n_scored(self) -> int:
        return sum(r[5] for r in self.rows)

    @property
    def by_condition(self) -> dict[str, int]:
        """Counts per condition, in the order `carelite.types.Condition` declares."""
        totals: dict[str, int] = {}
        for condition, _backend, _split, n, *_rest in self.rows:
            totals[condition] = totals.get(condition, 0) + n
        return {k: totals[k] for k in _condition_order(totals)}

    @property
    def by_backend(self) -> dict[str, int]:
        """Counts per serving stack, with a zero for a backend that produced nothing."""
        totals = dict.fromkeys(SERVING_BACKENDS, 0)
        for _condition, backend, _split, n, *_rest in self.rows:
            totals[backend] = totals.get(backend, 0) + n
        return totals

    @property
    def by_condition_and_backend(self) -> dict[tuple[str, str], int]:
        totals: dict[tuple[str, str], int] = {}
        for condition, backend, _split, n, *_rest in self.rows:
            key = (condition, backend)
            totals[key] = totals.get(key, 0) + n
        return totals

    @property
    def gate_blocked_by_condition(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for condition, _backend, _split, _n, blocked, *_rest in self.rows:
            totals[condition] = totals.get(condition, 0) + blocked
        return totals

    @property
    def backends_for(self) -> dict[str, dict[str, int]]:
        """Per condition, what each serving stack contributed."""
        out: dict[str, dict[str, int]] = {}
        for (condition, backend), n in self.by_condition_and_backend.items():
            out.setdefault(condition, {})[backend] = n
        return out

    @property
    def split_across_backends(self) -> tuple[str, ...]:
        """Conditions with generations from more than one serving stack.

        A condition in this tuple is an arm whose rows were produced by two
        different serving stacks, which is a confound to report rather than a
        detail to pool over.
        """
        return tuple(
            condition
            for condition in self.by_condition
            if len([n for n in self.backends_for.get(condition, {}).values() if n > 0]) > 1
        )


def _condition_order(present: Mapping[str, int]) -> list[str]:
    declared = [str(c) for c in Condition]
    ordered = [c for c in declared if c in present]
    ordered.extend(sorted(k for k in present if k not in declared))
    return ordered


# ---------------------------------------------------------------------------
# The numbers themselves
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeadlineNumber:
    """One quotable figure and the sentence it may not be quoted without.

    `qualifier` is a required field. There is no path through this module that
    produces a value with an empty one, which is the mechanism that makes the
    qualification structural rather than editorial.
    """

    key: str
    label: str
    value: float | int | None
    qualifier: str
    display: str
    section: str
    source: str
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None

    def __post_init__(self) -> None:
        if not self.qualifier.strip():
            raise ValueError(f"headline number {self.key!r} has no qualifier")

    def render(self) -> str:
        return f"  {self.label}\n      {self.display}\n      {self.qualifier}"


@dataclass(frozen=True, slots=True)
class HeadlineNumbers:
    """The block, in the order it is printed."""

    rows: tuple[HeadlineNumber, ...] = field(default_factory=tuple)
    counts_read: bool = True

    def by_key(self, key: str) -> HeadlineNumber | None:
        for row in self.rows:
            if row.key == key:
                return row
        return None

    def render(self) -> str:
        lines = [
            "=" * 78,
            "HEADLINE NUMBERS — read from the database, not carried forward from a document",
            "=" * 78,
            "",
            "  Every figure below was queried from Postgres when this file was written. None of "
            "it is transcribed from a plan, a memo, or an earlier run. A number in prose that "
            "disagrees with this block is stale, and this block is the one to trust.",
            "",
            f"  {D10_BANNER}",
            "",
            "  Each figure is printed with the qualification it cannot be quoted without. "
            "Lifting the number and leaving the line beneath it misreports this study.",
        ]
        if not self.counts_read:
            lines.extend(
                [
                    "",
                    "  THE GENERATION CENSUS WAS NOT READ in this invocation, so the counts below "
                    "say `not read` rather than reporting a zero nobody measured.",
                ]
            )
        current = ""
        for row in self.rows:
            if row.section != current:
                current = row.section
                lines.extend(["", "-" * 78, current, ""])
            lines.append(row.render())
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def _count_display(value: int | None) -> str:
    if value is None:
        return "not read from the database in this invocation"
    return f"{value:,}"


_COUNTS = "GENERATIONS — counted from `generation` itself, before any exclusion"
_PRIMARY = "PRIMARY CONTRAST — condition B against condition A, composite NURSE adherence"
_GATE = "WHY EVERY RESULT ABOVE IS EXPLORATORY"


def _count_rows(counts: GenerationCounts | None) -> list[HeadlineNumber]:
    read = counts is not None and counts.read
    total = counts.total if read and counts is not None else None
    scored = counts.n_scored if read and counts is not None else None
    blocked = counts.n_gate_blocked if read and counts is not None else None

    blocked_clause = (
        f"the {blocked} output-gate refusals D12 keeps flagged"
        if blocked is not None
        else "the output-gate refusals D12 keeps flagged"
    )
    rows: list[HeadlineNumber] = [
        HeadlineNumber(
            key="generations_total",
            label="generations in `generation`",
            value=total,
            qualifier=(
                "Every row the table holds, across all splits and all serving stacks. It "
                f"includes the partial LC cells D11 kept as a record and {blocked_clause}, so it "
                "is not the number any comparison ran on — see `generations_analysed` below."
            ),
            display=_count_display(total),
            section=_COUNTS,
            source="carelite.stats.data.GENERATION_COUNTS_SQL",
        )
    ]

    by_condition = counts.by_condition if read and counts is not None else {}
    backends_for = counts.backends_for if read and counts is not None else {}
    gate_by_condition = counts.gate_blocked_by_condition if read and counts is not None else {}
    for condition, n in by_condition.items():
        stacks = ", ".join(
            f"{backend} {backends_for.get(condition, {}).get(backend, 0)}"
            for backend in SERVING_BACKENDS
            if backends_for.get(condition, {}).get(backend, 0)
        )
        parts = [f"By serving stack: {stacks}."]
        refused = gate_by_condition.get(condition, 0)
        if refused:
            parts.append(
                f"{refused} of these were refused by the output gate and are counted here (D12); "
                "sensitivity analysis (d) in analysis.txt is the rerun that excludes them."
            )
        if condition == str(Condition.LC):
            parts.append(
                "D11 stopped LC at 39 of 180 planned cells, covering 13 of 60 scenarios, never "
                "randomised for partial analysis. These rows are a record of what ran and are "
                "excluded from every comparison rather than analysed with a caveat."
            )
        rows.append(
            HeadlineNumber(
                key=f"generations_by_condition.{condition}",
                label=f"condition {condition}",
                value=n,
                qualifier=" ".join(parts),
                display=_count_display(n),
                section=_COUNTS,
                source="carelite.stats.data.GENERATION_COUNTS_SQL",
            )
        )

    by_backend = counts.by_backend if read and counts is not None else {}
    split_arms = counts.split_across_backends if read and counts is not None else ()
    for backend in SERVING_BACKENDS:
        n_backend: int | None = by_backend.get(backend, 0) if read else None
        note = (
            "The two stacks serve different artifacts of the same model family — a GGUF against "
            "HF safetensors — so `model` and `model_digest` alone cannot tell them apart and a "
            "count pooled across backends would hide the difference. A zero is printed rather "
            "than the row omitted, so a stack that produced nothing is visibly nothing rather "
            "than silently absent."
        )
        if split_arms:
            note += (
                " Conditions produced by more than one stack on this run: "
                + ", ".join(split_arms)
                + ". Any arm on that list is pooled across backends and must be checked for "
                "agreement before it is treated as one arm."
            )
        rows.append(
            HeadlineNumber(
                key=f"generations_by_served_by.{backend}",
                label=f"served by {backend}",
                value=n_backend,
                qualifier=note,
                display=_count_display(n_backend),
                section=_COUNTS,
                source="carelite.stats.data.GENERATION_COUNTS_SQL",
            )
        )

    rows.append(
        HeadlineNumber(
            key="generations_scored",
            label="generations with a rubric score",
            value=scored,
            qualifier=(
                "Rows in `generation` with at least one row in `rubric_score`. A generation "
                "without one was produced but never judged, and the difference from the total "
                "above is exactly that set."
            ),
            display=_count_display(scored),
            section=_COUNTS,
            source="carelite.stats.data.GENERATION_COUNTS_SQL",
        )
    )
    return rows


def _analysed_rows(report: AnalysisReport) -> list[HeadlineNumber]:
    return [
        HeadlineNumber(
            key="generations_analysed",
            label="generations the analysis ran on",
            value=int(report.n_generations),
            qualifier=(
                f"The `{report.split}` split with condition LC removed (D11), which is the frame "
                "behind every comparison in analysis.txt. Gate-blocked generations are included "
                "here, as D12 specifies for the base reading. This is a smaller number than the "
                "census above and the two are not interchangeable."
            ),
            display=f"{int(report.n_generations):,}",
            section=_COUNTS,
            source="carelite.stats.report.run_analysis",
        ),
        HeadlineNumber(
            key="scenarios_analysed",
            label="scenarios the analysis ran on",
            value=int(report.n_scenarios),
            qualifier=(
                "Held-out scenarios contributing at least one scored generation. The paired "
                "tests run on scenario-level cell means, so this — not the generation count — is "
                "the n every effect size and interval is computed at."
            ),
            display=f"{int(report.n_scenarios):,}",
            section=_COUNTS,
            source="carelite.stats.report.run_analysis",
        ),
    ]


def _primary_rows(report: AnalysisReport) -> list[HeadlineNumber]:
    rows: list[HeadlineNumber] = []

    model = next((m for m in report.mixed if m.measure_key == NURSE_COMPOSITE.key), None)
    effect = (
        next((e for e in model.effects if e.term == PRIMARY_CONTRAST), None)
        if model is not None
        else None
    )
    if model is not None and effect is not None:
        rows.append(
            HeadlineNumber(
                key="primary_b_vs_a_coefficient",
                label="mixed-effects coefficient, B vs A",
                value=float(effect.coefficient),
                ci_low=float(effect.ci_low),
                ci_high=float(effect.ci_high),
                p_value=float(effect.p_value),
                qualifier=(
                    f"{model.label.tag()}. Rubric quality points, condition B relative to "
                    f"reference A, from `value ~ condition + (1 | scenario)` fitted by REML over "
                    f"{model.n_observations} generations in {model.n_scenarios} scenarios; the "
                    "random intercept is what stops the 3 samples in a cell counting as 3 "
                    "scenarios. Descriptive, not confirmatory: D10 dropped the pre-registration, "
                    "so this is an observation from one local run rather than a hypothesis test. "
                    "The p-value is uncorrected — the Holm family is the eight comparisons in "
                    "analysis.txt, and this model is not a member of it."
                ),
                display=(
                    f"{effect.coefficient:+.4f}  "
                    f"95% CI [{effect.ci_low:+.4f}, {effect.ci_high:+.4f}]  "
                    f"p = {effect.p_value:.4g}"
                ),
                section=_PRIMARY,
                source=(
                    "carelite.stats.mixed.fit_random_intercept(nurse_composite), "
                    f"fixed effect {PRIMARY_CONTRAST!r}"
                ),
            )
        )
    else:
        rows.append(
            HeadlineNumber(
                key="primary_b_vs_a_coefficient",
                label="mixed-effects coefficient, B vs A",
                value=None,
                qualifier=(
                    "NOT COMPUTED on this data: the §8.3 mixed-effects model has no B-vs-A "
                    "contrast to report, which means conditions A and B are not both present "
                    "with enough scenarios to fit it. Not a null result."
                ),
                display="not computed on this data",
                section=_PRIMARY,
                source="carelite.stats.mixed.fit_random_intercept(nurse_composite)",
            )
        )

    primary = report.primary.by_key(_PRIMARY_HYPOTHESIS_KEY)
    if primary is not None and not math.isnan(primary.effects.hodges_lehmann.point):
        hl = primary.effects.hodges_lehmann
        point, low, high = flip_orientation(hl.point, hl.ci.low, hl.ci.high)
        rows.append(
            HeadlineNumber(
                key="primary_b_vs_a_hodges_lehmann",
                label="Hodges-Lehmann shift, B vs A",
                value=point,
                ci_low=low,
                ci_high=high,
                p_value=float(primary.p_holm),
                qualifier=(
                    f"{primary.label.tag()}. The paired location shift in rubric quality points "
                    f"over {primary.n_scenarios} scenario-level cell means, with a 95% bootstrap "
                    "interval. REORIENTED: the §8.1 family computes this comparison as A vs B, so "
                    "the point estimate is negated and the interval bounds swapped to put it in "
                    "the same B-against-A orientation as the coefficient above. The p-value "
                    f"is Holm-adjusted across the family of {primary.family_size}; read it in "
                    "analysis.txt, where the instrument diagnostic that governs it is printed "
                    "first."
                ),
                display=(
                    f"{point:+.3f} rubric points  95% CI [{low:+.3f}, {high:+.3f}]  "
                    f"Holm-adjusted p = {primary.p_holm:.4g}"
                ),
                section=_PRIMARY,
                source=(
                    "carelite.stats.primary.run_family — "
                    f"{_PRIMARY_HYPOTHESIS_KEY}, Hodges-Lehmann, reoriented"
                ),
            )
        )
    else:
        rows.append(
            HeadlineNumber(
                key="primary_b_vs_a_hodges_lehmann",
                label="Hodges-Lehmann shift, B vs A",
                value=None,
                qualifier=(
                    "NOT COMPUTED on this data: the §8.1 family produced no Hodges-Lehmann "
                    "estimate for the primary comparison, which means there were no paired "
                    "A-and-B scenarios to compute a shift over. Not a null result."
                ),
                display="not computed on this data",
                section=_PRIMARY,
                source=f"carelite.stats.primary.run_family — {_PRIMARY_HYPOTHESIS_KEY}",
            )
        )
    return rows


def _gate_rows(report: AnalysisReport) -> list[HeadlineNumber]:
    cleared = len(RUBRIC_DIMENSIONS) - len(report.exploratory_dimensions)
    return [
        HeadlineNumber(
            key="dimensions_clearing_the_judge_gate",
            label="rubric dimensions clearing the judge-agreement threshold",
            value=cleared,
            qualifier=(
                "Either the judge validation study has not run, or no dimension cleared "
                "alpha >= 0.667 and rho >= 0.5 on >= 30 paired units (analysis plan §9). While "
                "this reads 0, every judge-scored result in this block is EXPLORATORY, and that "
                "label is a measured state of the database rather than a hedge — `rating_"
                "assignment` is what has to be populated for it to change."
            ),
            display=f"{cleared} of {len(RUBRIC_DIMENSIONS)}",
            section=_GATE,
            source="carelite.stats.evidence.dimension_statuses",
        )
    ]


def headline_numbers(
    report: AnalysisReport,
    *,
    counts: GenerationCounts | None = None,
) -> HeadlineNumbers:
    """Assemble the block from one `AnalysisReport` and one generation census.

    `counts` is `None` when the census was not read — a caller with no database.
    That renders as `not read` rather than as a zero, because a count nobody
    took and a count that came back empty are different claims.
    """
    resolved = counts if counts is not None else GenerationCounts.not_read()
    rows: list[HeadlineNumber] = [
        *_count_rows(resolved),
        *_analysed_rows(report),
        *_primary_rows(report),
        *_gate_rows(report),
    ]
    return HeadlineNumbers(rows=tuple(rows), counts_read=resolved.read)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _csv_value(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def write_headline(headline: HeadlineNumbers, output_dir: Path) -> list[Path]:
    """Write `headline-numbers.txt` and `headline-numbers.csv`. Returns both paths.

    Two artefacts for the same reason `analysis.txt` and `effect-sizes.csv`
    coexist: the text carries the reasoning a reader needs and the CSV carries
    the numbers a script needs, rendered from one list of rows so they cannot
    disagree.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    text_path = directory / "headline-numbers.txt"
    text_path.write_text(headline.render(), encoding="utf-8")

    csv_path = directory / "headline-numbers.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HEADLINE_COLUMNS))
        writer.writeheader()
        for row in headline.rows:
            writer.writerow(
                {
                    "key": row.key,
                    "value": _csv_value(row.value),
                    "qualifier": row.qualifier,
                    "label": row.label,
                    "ci_low": _csv_value(row.ci_low),
                    "ci_high": _csv_value(row.ci_high),
                    "p_value": _csv_value(row.p_value),
                    "display": row.display,
                    "section": row.section,
                    "source": row.source,
                }
            )
    return [text_path, csv_path]
