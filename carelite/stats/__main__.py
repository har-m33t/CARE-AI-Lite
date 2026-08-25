"""`python -m carelite.stats` — run the analysis and print it.

Reads the held-out split from Postgres and renders every §8 analysis. **D10: every
result is descriptive**; the plan in `docs/preregistration.md` still governs the
analysis but was never registered.

With an empty results table it prints the structure with no numbers in it. If the
holdout run has completed and this still happens, the judge scores have not been
loaded — run `python -m carelite.eval.judge.load`.
"""

from __future__ import annotations

import argparse
import sys

from carelite.stats.report import run_analysis
from carelite.types import Split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default=str(Split.HOLDOUT),
        choices=[str(Split.HOLDOUT), str(Split.TRAIN)],
        help="which split to analyse; the planned analyses are holdout-only (§6)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=10_000,
        help="bootstrap replicates for every confidence interval",
    )
    args = parser.parse_args(argv)

    try:
        report = run_analysis(split=args.split, n_boot=args.n_boot)
    except Exception as exc:
        print(f"could not run the analysis: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
