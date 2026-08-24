"""`carelite.viz.reproduce.run(output_dir)` — the contract `carelite/repro.py`
looks for (see its module docstring: "carelite.viz.reproduce.run(output_dir:
Path) -> list[Path]").

Regenerates every figure this lane owns from the live database (and, for the
retrieval ablation, from whatever `carelite.retrieval.ablation` has written to
disk — see `carelite.viz.queries.ablation_table_df`), and writes each as a
PNG + PDF pair under `output_dir`.

**Tolerant by design, on purpose.** `generation` and `rubric_score` are 0 rows
as of this writing (`docs/preregistration.md`: held-out generation is blocked
until OSF registration), so every figure that needs scored generations will
raise `carelite.viz.queries.DataUnavailable` right now, every time. That is
the expected, correct state of a fresh clone — not a bug in this module — so
`run()` catches it per figure, keeps going, and reports a clear skip reason
for each one rather than aborting the whole reproduction because one table is
still empty. An unexpected exception (a real bug) is also caught per figure,
with its type and message recorded, so one broken figure cannot hide whether
the other seven succeeded.

A full human-readable status report is written to `output_dir/status.md`
alongside whatever figures actually rendered, and `run()`'s return value is
exactly the list of files that were written — nothing about a skip or an
error appears in that list, so `carelite/repro.py`'s "wrote N figures" count
means what it says.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from carelite.viz import figures
from carelite.viz.queries import (
    DataUnavailable,
    ablation_table_df,
    effect_sizes_df,
    equity_subgroup_df,
    judge_agreement_df,
    judge_self_consistency_df,
    load_long_scores,
    negative_control_df,
    retrieval_quality_df,
    rubric_scores_df,
)
from carelite.viz.style import save_figure

__all__ = ["FigureStatus", "run", "run_with_report"]


@dataclass
class FigureStatus:
    name: str
    written: list[Path]
    skipped_reason: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.skipped_reason is None and self.error is None


def _try_load(
    loader: Callable[..., pd.DataFrame], *args: object, **kwargs: object
) -> tuple[pd.DataFrame | None, str | None]:
    """Run a `carelite.viz.queries` loader, returning `(df, error_message)`.

    `error_message` distinguishes an expected `DataUnavailable` (message as
    raised) from any other exception (`type: message`, so a real bug still
    reads as one), and is `None` on success.
    """
    try:
        return loader(*args, **kwargs), None
    except DataUnavailable as exc:
        return None, str(exc)
    except Exception as exc:  # a real bug in the query layer — report, don't hide
        return None, f"{type(exc).__name__}: {exc}"


def run_with_report(output_dir: Path) -> list[FigureStatus]:
    """Every figure, attempted independently, with a per-figure outcome."""
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses: list[FigureStatus] = []

    long_judge, long_error = _try_load(load_long_scores, rater_type="llm_judge")
    ablation_df, ablation_error = _try_load(ablation_table_df)

    # --- 1. headline rubric scores -----------------------------------------
    if long_judge is None:
        statuses.append(FigureStatus("rubric_scores", [], skipped_reason=long_error))
    else:
        try:
            df = rubric_scores_df(long_judge)
            fig = figures.fig_rubric_scores(df)
            statuses.append(
                FigureStatus("rubric_scores", save_figure(fig, "01_rubric_scores", output_dir))
            )
        except DataUnavailable as exc:
            statuses.append(FigureStatus("rubric_scores", [], skipped_reason=str(exc)))
        except Exception as exc:
            statuses.append(FigureStatus("rubric_scores", [], error=f"{type(exc).__name__}: {exc}"))

    # --- 2. effect sizes forest plot ----------------------------------------
    if long_judge is None:
        statuses.append(FigureStatus("effect_sizes", [], skipped_reason=long_error))
    else:
        try:
            df = effect_sizes_df(long_judge)
            fig = figures.fig_effect_sizes(df)
            statuses.append(
                FigureStatus("effect_sizes", save_figure(fig, "02_effect_sizes", output_dir))
            )
        except DataUnavailable as exc:
            statuses.append(FigureStatus("effect_sizes", [], skipped_reason=str(exc)))
        except Exception as exc:
            statuses.append(FigureStatus("effect_sizes", [], error=f"{type(exc).__name__}: {exc}"))

    # --- 3. ablation table ---------------------------------------------------
    if ablation_df is None:
        statuses.append(FigureStatus("ablation_table", [], skipped_reason=ablation_error))
    else:
        try:
            fig = figures.fig_ablation_table(ablation_df)
            statuses.append(
                FigureStatus("ablation_table", save_figure(fig, "03_ablation_table", output_dir))
            )
        except DataUnavailable as exc:
            statuses.append(FigureStatus("ablation_table", [], skipped_reason=str(exc)))
        except Exception as exc:
            statuses.append(
                FigureStatus("ablation_table", [], error=f"{type(exc).__name__}: {exc}")
            )

    # --- 4. judge-vs-human agreement -----------------------------------------
    try:
        df = judge_agreement_df()
        fig = figures.fig_judge_agreement(df)
        statuses.append(
            FigureStatus("judge_agreement", save_figure(fig, "04_judge_agreement", output_dir))
        )
    except DataUnavailable as exc:
        statuses.append(FigureStatus("judge_agreement", [], skipped_reason=str(exc)))
    except Exception as exc:
        statuses.append(FigureStatus("judge_agreement", [], error=f"{type(exc).__name__}: {exc}"))

    # --- 5. judge self-consistency --------------------------------------------
    try:
        df = judge_self_consistency_df()
        fig = figures.fig_judge_consistency(df)
        statuses.append(
            FigureStatus("judge_consistency", save_figure(fig, "05_judge_consistency", output_dir))
        )
    except DataUnavailable as exc:
        statuses.append(FigureStatus("judge_consistency", [], skipped_reason=str(exc)))
    except Exception as exc:
        statuses.append(FigureStatus("judge_consistency", [], error=f"{type(exc).__name__}: {exc}"))

    # --- 6. retrieval quality ---------------------------------------------------
    try:
        df = retrieval_quality_df(ablation_df)
        fig = figures.fig_retrieval_quality(df)
        statuses.append(
            FigureStatus("retrieval_quality", save_figure(fig, "06_retrieval_quality", output_dir))
        )
    except DataUnavailable as exc:
        statuses.append(FigureStatus("retrieval_quality", [], skipped_reason=str(exc)))
    except Exception as exc:
        statuses.append(FigureStatus("retrieval_quality", [], error=f"{type(exc).__name__}: {exc}"))

    # --- 7. equity subgroup ----------------------------------------------------
    if long_judge is None:
        statuses.append(FigureStatus("equity_subgroup", [], skipped_reason=long_error))
    else:
        try:
            df = equity_subgroup_df(long_judge)
            fig = figures.fig_equity_subgroup(df)
            statuses.append(
                FigureStatus("equity_subgroup", save_figure(fig, "07_equity_subgroup", output_dir))
            )
        except DataUnavailable as exc:
            statuses.append(FigureStatus("equity_subgroup", [], skipped_reason=str(exc)))
        except Exception as exc:
            statuses.append(
                FigureStatus("equity_subgroup", [], error=f"{type(exc).__name__}: {exc}")
            )

    # --- 8. negative control -----------------------------------------------------
    if long_judge is None:
        statuses.append(FigureStatus("negative_control", [], skipped_reason=long_error))
    else:
        try:
            df = negative_control_df(long_judge)
            fig = figures.fig_negative_control(df)
            statuses.append(
                FigureStatus(
                    "negative_control", save_figure(fig, "08_negative_control", output_dir)
                )
            )
        except DataUnavailable as exc:
            statuses.append(FigureStatus("negative_control", [], skipped_reason=str(exc)))
        except Exception as exc:
            statuses.append(
                FigureStatus("negative_control", [], error=f"{type(exc).__name__}: {exc}")
            )

    _write_status_report(output_dir, statuses)
    return statuses


def _write_status_report(output_dir: Path, statuses: list[FigureStatus]) -> None:
    lines = ["carelite.viz.reproduce", "=" * 40, ""]
    for s in statuses:
        if s.written:
            lines.append(f"[ok]      {s.name}: wrote {len(s.written)} files")
        elif s.skipped_reason:
            lines.append(f"[skipped] {s.name}: {s.skipped_reason}")
        else:
            lines.append(f"[error]   {s.name}: {s.error}")
    (output_dir / "status.md").write_text("\n".join(lines) + "\n")


def run(output_dir: Path) -> list[Path]:
    """The contract `carelite/repro.py` calls. Every file successfully written,
    across every figure that had data — see `run_with_report` for per-figure
    detail (skip reasons, errors), also written to `output_dir/status.md`.
    """
    statuses = run_with_report(output_dir)
    written: list[Path] = []
    for s in statuses:
        written.extend(s.written)
    return written
