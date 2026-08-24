"""Stratum coverage audit for the scenario bank.

Build plan v3 sprint 7 gates on "every stratum cell populated". Five factors are
stratified -- `challenge_type`, `emotion_intensity` (1-5), `encounter_phase`,
`literacy_signal`, `equity_stratum` -- and their full cross product is
10 x 5 x 5 x 4 x 2 = 2000 cells. A 100-scenario bank cannot populate 2000 cells,
so "every cell" has to mean something specific, or it means nothing. This module
fixes it to three concrete claims:

1. **Every level of every factor is populated**, above a stated minimum. An
   unused level is a stratum that exists only in the documentation.
2. **The design cell -- `challenge_type` x `encounter_phase` -- is complete.**
   All 50 combinations carry exactly two scenarios. This is the cross the bank
   is actually built on, and it is the cross the analysis slices by.
3. **Every pairing involving `equity_stratum` or `split` is populated.** Equity
   is a pre-specified secondary analysis and `split` partitions everything, so a
   hole in either turns a planned comparison into an impossible one.

Anything else is reported as a marginal count and not gated.

Two failure modes this is specifically built to catch:

* **Empty cells.** `assert_full_coverage()` raises `StratumCoverageError`
  enumerating every violation, rather than the first one. A partial list makes
  someone fix one hole and rediscover the next on the following run.
* **A confounded equity stratum.** If equity scenarios clustered into one
  challenge type, or into the top of the intensity range, the pre-specified
  equity subgroup effect would be indistinguishable from a topic or intensity
  effect. The `EQUITY_*` checks below exist for that and nothing else.

**One hole is accepted, and it is named.** `ACCEPTED_EMPTY_CELLS` is an allowlist
of cells that are empty as the recorded consequence of a decision a person made,
rather than as an oversight. It holds exactly one entry, it names the decision
that produced it, and the audit prints it on every run instead of hiding it. It
is not a way to lower a gate: a cell not listed there still fails, and the entry
is pinned by a unit test so a second hole cannot be quietly added to it. The
distinction the mechanism is drawing is between *the bank has a gap nobody noticed*
and *the bank has a gap somebody chose*, and only the second one is survivable.

Run it standalone::

    .venv/bin/python -m carelite.scenarios.audit

Exit status 0 means every gate passed or was accepted; 1 means it did not.
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

from carelite.scenarios.bank import (
    CHALLENGE_TYPES,
    EQUITY_KINDS,
    EXPECTED_HOLDOUT,
    EXPECTED_TOTAL,
    EXPECTED_TRAIN,
    LITERACY_SIGNALS,
    CuratedScenario,
    load_bank,
)
from carelite.types import EncounterPhase, Split

__all__ = [
    "ACCEPTED_EMPTY_CELLS",
    "EQUITY_MIN_PER_CHALLENGE",
    "EQUITY_MIN_TOTAL",
    "MIN_PER_EQUITY_KIND",
    "MIN_PER_INTENSITY",
    "MIN_PER_LITERACY",
    "AuditReport",
    "StratumCoverageError",
    "assert_full_coverage",
    "main",
    "run_audit",
]

#: A cell is always keyed by exactly two factor levels.
CellKey = tuple[object, object]
CellTable = dict[CellKey, int]

INTENSITIES: tuple[int, ...] = (1, 2, 3, 4, 5)
PHASES: tuple[EncounterPhase, ...] = tuple(EncounterPhase)

#: Per-challenge-type count. Ten types x ten scenarios is the shape of the bank.
PER_CHALLENGE_TYPE = 10
#: Per-phase count, and per (challenge_type, phase) cell.
PER_PHASE = 20
PER_DESIGN_CELL = 2

#: Intensity 1 is deliberately uncommon: an emotionally flat turn is a real and
#: important case (it is where over-reading emotion becomes the failure), but a
#: bank of communication *challenges* that was mostly flat would be the wrong
#: instrument. Eight is the floor at which the level still supports a descriptive
#: split.
MIN_PER_INTENSITY = 8
#: `unmarked` dominates by design; the three marked signals each need enough
#: mass to be describable.
MIN_PER_LITERACY = 12
MIN_PER_EQUITY_KIND = 8

#: Equity is a pre-specified secondary analysis on the held-out set, so the
#: stratum needs enough scenarios there to say anything. 30-45 of 100 keeps the
#: subgroup analysable without turning the bank into an equity bank.
EQUITY_MIN_TOTAL = 30
EQUITY_MAX_TOTAL = 45
EQUITY_MIN_HOLDOUT = 15
EQUITY_MIN_TRAIN = 8
#: Guards the confound: equity must not be a proxy for one challenge type.
EQUITY_MIN_PER_CHALLENGE = 2

#: Cells that are empty because a person decided they would be, keyed by
#: ``(view name, cell key)`` and valued with the decision that produced them.
#:
#: This exists for exactly one situation and should stay that small. `DECISIONS.md`
#: D2 moved SC-010 out of the equity stratum because its LEP signal was carried by
#: register rather than by a situation, which the equity review packet's own rule 2
#: forbids. SC-010 was the only equity scenario at ``emotion_intensity=1``, so the
#: correction emptied that cell. The gap is genuine and is recorded in
#: `scenarios/EQUITY_REVIEW.md` and in the limitations: the equity stratum now spans
#: intensities 2-5 and the bank cannot measure whether the disparity behaves
#: differently on an emotionally flat turn. Filling it would mean writing a new
#: held-out scenario, which is a far larger protocol amendment than a metadata
#: change and needs its own decision.
#:
#: An empty cell not listed here is still a failure.
ACCEPTED_EMPTY_CELLS: Final[dict[tuple[str, CellKey], str]] = {
    ("equity_stratum x emotion_intensity", (True, 1)): (
        "DECISIONS.md D2 (2026-08-24) -- SC-010 left the equity stratum and was its only "
        "intensity-1 scenario. D5 accepted this as a permanent, pre-specified limitation of "
        "the equity subgroup analysis rather than repairing it: see scenarios/EQUITY_REVIEW.md."
    ),
}


class StratumCoverageError(AssertionError):
    """One or more stratum cells are empty or under-populated."""


@dataclass
class AuditReport:
    """Counts for every gated view, plus the violations found."""

    n_total: int = 0
    n_train: int = 0
    n_holdout: int = 0
    marginals: dict[str, Counter[object]] = field(default_factory=dict)
    cells: dict[str, CellTable] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    #: Findings that would be violations but for an entry in
    #: `ACCEPTED_EMPTY_CELLS`. They do not fail the audit; they are printed on
    #: every run so an accepted gap stays visible rather than becoming invisible.
    exemptions: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def empty_cells(self, view: str) -> list[CellKey]:
        return [key for key, n in self.cells.get(view, {}).items() if n == 0]


def _flag(report: AuditReport, view: str, key: CellKey, message: str) -> None:
    """Record a cell-level finding as a violation, or as an accepted exemption.

    Routing every cell finding through here rather than special-casing the one
    accepted cell is deliberate: the allowlist is a policy with one entry, not a
    hard-coded escape hatch for one gate.
    """
    reason = ACCEPTED_EMPTY_CELLS.get((view, key))
    if reason is None:
        report.violations.append(message)
    else:
        report.exemptions.append(f"{message}\n      ACCEPTED: {reason}")


def _cross(
    records: Sequence[CuratedScenario],
    name: str,
    levels_a: Iterable[object],
    levels_b: Iterable[object],
    key: Callable[[CuratedScenario], CellKey],
    report: AuditReport,
) -> CellTable:
    """Materialise a full two-way table, zeros included.

    Building the table from the declared level lists rather than from the data
    is the whole point: a cell that no scenario populates has to appear as a 0,
    not be absent.
    """
    observed: Counter[CellKey] = Counter(key(r) for r in records)
    table = {(a, b): observed.get((a, b), 0) for a in levels_a for b in levels_b}
    report.cells[name] = table
    return table


def run_audit(records: Sequence[CuratedScenario] | None = None) -> AuditReport:
    """Compute every gated view and collect all violations. Never raises."""
    rows: Sequence[CuratedScenario] = load_bank() if records is None else records
    report = AuditReport()
    v = report.violations

    # ---- counts -----------------------------------------------------------
    report.n_total = len(rows)
    report.n_train = sum(1 for r in rows if r.split is Split.TRAIN)
    report.n_holdout = sum(1 for r in rows if r.split is Split.HOLDOUT)
    if report.n_total != EXPECTED_TOTAL:
        v.append(f"total: expected {EXPECTED_TOTAL} scenarios, found {report.n_total}")
    if report.n_train != EXPECTED_TRAIN:
        v.append(f"split/train: expected {EXPECTED_TRAIN}, found {report.n_train}")
    if report.n_holdout != EXPECTED_HOLDOUT:
        v.append(f"split/holdout: expected {EXPECTED_HOLDOUT}, found {report.n_holdout}")

    ids = [r.scenario_id for r in rows]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        v.append(f"identity: duplicate scenario_ids {dupes}")

    # ---- 1. every level of every factor is populated -----------------------
    by_challenge: Counter[object] = Counter(r.challenge_type for r in rows)
    by_intensity: Counter[object] = Counter(r.emotion_intensity for r in rows)
    by_phase: Counter[object] = Counter(r.encounter_phase for r in rows)
    by_literacy: Counter[object] = Counter(r.literacy_signal for r in rows)
    by_equity: Counter[object] = Counter(r.equity_stratum for r in rows)
    by_equity_kind: Counter[object] = Counter(r.equity_kind for r in rows if r.equity_kind)
    report.marginals = {
        "challenge_type": by_challenge,
        "emotion_intensity": by_intensity,
        "encounter_phase": by_phase,
        "literacy_signal": by_literacy,
        "equity_stratum": by_equity,
        "equity_kind": by_equity_kind,
    }

    for ct in CHALLENGE_TYPES:
        n = by_challenge.get(ct, 0)
        if n != PER_CHALLENGE_TYPE:
            v.append(f"challenge_type/{ct}: expected {PER_CHALLENGE_TYPE}, found {n}")
    unknown = sorted(str(k) for k in by_challenge if k not in CHALLENGE_TYPES)
    if unknown:
        v.append(f"challenge_type: values outside the controlled vocabulary: {unknown}")

    for level in INTENSITIES:
        n = by_intensity.get(level, 0)
        if n < MIN_PER_INTENSITY:
            v.append(
                f"emotion_intensity/{level}: {n} scenarios, minimum {MIN_PER_INTENSITY}"
                + (" -- EMPTY CELL" if n == 0 else "")
            )

    for phase in PHASES:
        n = by_phase.get(phase, 0)
        if n != PER_PHASE:
            v.append(f"encounter_phase/{phase}: expected {PER_PHASE}, found {n}")

    for sig in LITERACY_SIGNALS:
        n = by_literacy.get(sig, 0)
        if n < MIN_PER_LITERACY:
            v.append(
                f"literacy_signal/{sig}: {n} scenarios, minimum {MIN_PER_LITERACY}"
                + (" -- EMPTY CELL" if n == 0 else "")
            )
    unknown = sorted(str(k) for k in by_literacy if k not in LITERACY_SIGNALS)
    if unknown:
        v.append(f"literacy_signal: values outside the controlled vocabulary: {unknown}")

    n_equity = by_equity.get(True, 0)
    if not EQUITY_MIN_TOTAL <= n_equity <= EQUITY_MAX_TOTAL:
        v.append(
            f"equity_stratum/True: {n_equity} scenarios, expected "
            f"{EQUITY_MIN_TOTAL}-{EQUITY_MAX_TOTAL}"
        )
    if by_equity.get(False, 0) == 0:
        v.append("equity_stratum/False: EMPTY CELL -- no non-equity comparison group")

    for kind in EQUITY_KINDS:
        n = by_equity_kind.get(kind, 0)
        if n < MIN_PER_EQUITY_KIND:
            v.append(
                f"equity_kind/{kind}: {n} scenarios, minimum {MIN_PER_EQUITY_KIND}"
                + (" -- EMPTY CELL" if n == 0 else "")
            )

    # ---- 2. the design cell: challenge_type x encounter_phase --------------
    design = _cross(
        rows,
        "challenge_type x encounter_phase",
        CHALLENGE_TYPES,
        PHASES,
        lambda r: (r.challenge_type, r.encounter_phase),
        report,
    )
    for (d_ct, d_phase), n in sorted(design.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        if n == 0:
            _flag(
                report,
                "challenge_type x encounter_phase",
                (d_ct, d_phase),
                f"cell challenge_type={d_ct} x encounter_phase={d_phase}: EMPTY CELL",
            )
        elif n != PER_DESIGN_CELL:
            v.append(
                f"cell challenge_type={d_ct} x encounter_phase={d_phase}: "
                f"{n} scenarios, expected {PER_DESIGN_CELL}"
            )

    # ---- 3. every pairing involving equity_stratum or split ----------------
    eq_phase = _cross(
        rows,
        "equity_stratum x encounter_phase",
        (True, False),
        PHASES,
        lambda r: (r.equity_stratum, r.encounter_phase),
        report,
    )
    for (ep_eq, ep_phase), n in eq_phase.items():
        if n == 0:
            _flag(
                report,
                "equity_stratum x encounter_phase",
                (ep_eq, ep_phase),
                f"cell equity_stratum={ep_eq} x encounter_phase={ep_phase}: EMPTY CELL",
            )

    eq_intensity = _cross(
        rows,
        "equity_stratum x emotion_intensity",
        (True, False),
        INTENSITIES,
        lambda r: (r.equity_stratum, r.emotion_intensity),
        report,
    )
    for (ei_eq, ei_level), n in eq_intensity.items():
        if n == 0:
            _flag(
                report,
                "equity_stratum x emotion_intensity",
                (ei_eq, ei_level),
                f"cell equity_stratum={ei_eq} x emotion_intensity={ei_level}: EMPTY CELL "
                "-- equity would be confounded with intensity",
            )

    eq_literacy = _cross(
        rows,
        "equity_stratum x literacy_signal",
        (True, False),
        LITERACY_SIGNALS,
        lambda r: (r.equity_stratum, r.literacy_signal),
        report,
    )
    for (el_eq, el_sig), n in eq_literacy.items():
        if n == 0:
            _flag(
                report,
                "equity_stratum x literacy_signal",
                (el_eq, el_sig),
                f"cell equity_stratum={el_eq} x literacy_signal={el_sig}: EMPTY CELL",
            )

    eq_challenge = _cross(
        rows,
        "equity_stratum x challenge_type",
        (True,),
        CHALLENGE_TYPES,
        lambda r: (r.equity_stratum, r.challenge_type),
        report,
    )
    for (_, ec_ct), n in sorted(eq_challenge.items(), key=lambda kv: str(kv[0][1])):
        if n < EQUITY_MIN_PER_CHALLENGE:
            v.append(
                f"cell equity_stratum=True x challenge_type={ec_ct}: {n} scenarios, "
                f"minimum {EQUITY_MIN_PER_CHALLENGE} -- equity would be confounded "
                "with challenge type"
            )

    split_challenge = _cross(
        rows,
        "split x challenge_type",
        tuple(Split),
        CHALLENGE_TYPES,
        lambda r: (r.split, r.challenge_type),
        report,
    )
    for (sc_split, sc_ct), n in sorted(
        split_challenge.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))
    ):
        want = 4 if sc_split is Split.TRAIN else 6
        if n != want:
            v.append(
                f"cell split={sc_split} x challenge_type={sc_ct}: {n} scenarios, expected {want}"
            )

    split_phase = _cross(
        rows,
        "split x encounter_phase",
        tuple(Split),
        PHASES,
        lambda r: (r.split, r.encounter_phase),
        report,
    )
    for (sp_split, sp_phase), n in split_phase.items():
        want = 8 if sp_split is Split.TRAIN else 12
        if n != want:
            v.append(
                f"cell split={sp_split} x encounter_phase={sp_phase}: {n} scenarios, expected {want}"
            )

    split_intensity = _cross(
        rows,
        "split x emotion_intensity",
        tuple(Split),
        INTENSITIES,
        lambda r: (r.split, r.emotion_intensity),
        report,
    )
    for (si_split, si_level), n in split_intensity.items():
        if n == 0:
            _flag(
                report,
                "split x emotion_intensity",
                (si_split, si_level),
                f"cell split={si_split} x emotion_intensity={si_level}: EMPTY CELL",
            )

    split_literacy = _cross(
        rows,
        "split x literacy_signal",
        tuple(Split),
        LITERACY_SIGNALS,
        lambda r: (r.split, r.literacy_signal),
        report,
    )
    for (sl_split, sl_sig), n in split_literacy.items():
        if n == 0:
            _flag(
                report,
                "split x literacy_signal",
                (sl_split, sl_sig),
                f"cell split={sl_split} x literacy_signal={sl_sig}: EMPTY CELL",
            )

    split_equity = _cross(
        rows,
        "split x equity_stratum",
        tuple(Split),
        (True, False),
        lambda r: (r.split, r.equity_stratum),
        report,
    )
    n_eq_holdout = split_equity[(Split.HOLDOUT, True)]
    n_eq_train = split_equity[(Split.TRAIN, True)]
    if n_eq_holdout < EQUITY_MIN_HOLDOUT:
        v.append(
            f"cell split=holdout x equity_stratum=True: {n_eq_holdout} scenarios, "
            f"minimum {EQUITY_MIN_HOLDOUT} for the pre-specified subgroup analysis"
        )
    if n_eq_train < EQUITY_MIN_TRAIN:
        v.append(
            f"cell split=train x equity_stratum=True: {n_eq_train} scenarios, "
            f"minimum {EQUITY_MIN_TRAIN}"
        )

    split_kind = _cross(
        rows,
        "split x equity_kind",
        tuple(Split),
        EQUITY_KINDS,
        lambda r: (r.split, r.equity_kind),
        report,
    )
    for (sk_split, sk_kind), n in split_kind.items():
        if n == 0:
            _flag(
                report,
                "split x equity_kind",
                (sk_split, sk_kind),
                f"cell split={sk_split} x equity_kind={sk_kind}: EMPTY CELL",
            )

    return report


def assert_full_coverage(records: Sequence[CuratedScenario] | None = None) -> AuditReport:
    """Run the audit and raise `StratumCoverageError` listing *every* violation."""
    report = run_audit(records)
    if not report.ok:
        detail = "\n".join(f"  - {line}" for line in report.violations)
        raise StratumCoverageError(
            f"scenario bank stratum audit failed ({len(report.violations)} violations):\n{detail}"
        )
    return report


def _format(report: AuditReport) -> str:
    out: list[str] = []
    out.append("CARELite scenario bank -- stratum coverage audit")
    out.append(
        f"  {report.n_total} scenarios | {report.n_train} train | {report.n_holdout} holdout"
    )
    out.append("")
    for name, counter in report.marginals.items():
        rendered = ", ".join(f"{k}={counter[k]}" for k in sorted(counter, key=str))
        out.append(f"  {name:<20} {rendered}")
    out.append("")
    for view, table in report.cells.items():
        empty = [k for k, n in table.items() if n == 0]
        status = "OK" if not empty else f"{len(empty)} EMPTY"
        out.append(f"  {view:<38} {len(table):>3} cells  {status}")
    out.append("")
    if report.exemptions:
        out.append(
            f"  {len(report.exemptions)} accepted gap(s) -- empty by decision, not by oversight:"
        )
        out.extend(f"    - {line}" for line in report.exemptions)
        out.append("")
    if report.ok and report.exemptions:
        out.append(
            f"  PASS -- every gated stratum cell is populated except "
            f"{len(report.exemptions)} accepted above."
        )
    elif report.ok:
        out.append("  PASS -- every gated stratum cell is populated.")
    else:
        out.append(f"  FAIL -- {len(report.violations)} violations:")
        out.extend(f"    - {line}" for line in report.violations)
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Exit 1 on any violation."""
    del argv
    report = run_audit()
    print(_format(report))
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
