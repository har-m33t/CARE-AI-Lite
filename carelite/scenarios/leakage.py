"""Detects held-out scenario text turning up where it must not.

The checksum in `carelite.scenarios.freeze` catches someone *editing* the
held-out set. It cannot catch the other failure, which is the one Sprint 9
actually risks: **held-out text being copied into an optimised prompt.**

That is not hypothetical. DSPy's `BootstrapFewShot` builds a prompt by embedding
selected training examples *verbatim* in it. If the example pool is drawn from
the whole bank instead of from `train_scenarios()`, held-out utterances end up
inside `prompt_version.text`, and every held-out score from then on is measured
on a prompt that has already seen the answer. The prompt file looks completely
normal. Nothing else in the pipeline notices.

So the guard is a text search. Held-out utterances are shingled into overlapping
8-word n-grams; any artifact that contains one of those n-grams is reporting a
match. Eight words is long enough that a collision on ordinary English is
unlikely and short enough to survive light paraphrasing, truncation, or a change
of quoting style.

The check is one-directional and cheap, so run it on anything that gets written
by an optimiser and read by a generator: prompt versions, few-shot example
files, cached optimiser state.

    from carelite.scenarios.leakage import assert_no_holdout_leakage
    assert_no_holdout_leakage(prompt_text, source="prompts/condition_c_v4.txt")

A match is not automatically fatal -- a legitimately quoted rubric anchor could
in principle collide -- but it always requires a human to look. `find_leaks()`
returns the matches for a caller that wants to decide; `assert_no_holdout_leakage`
raises.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from carelite.scenarios.bank import CuratedScenario, holdout_scenarios

__all__ = [
    "DEFAULT_SHINGLE_N",
    "HoldoutLeakError",
    "Leak",
    "assert_no_holdout_leakage",
    "find_leaks",
    "holdout_shingles",
    "normalise",
    "scan_paths",
]

#: 8 words. Short enough to survive reformatting, long enough that an innocent
#: collision on ordinary English is improbable.
DEFAULT_SHINGLE_N = 8

_WORD = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class Leak:
    """One held-out n-gram found in a scanned artifact."""

    scenario_id: str
    ngram: str
    source: str = ""

    def __str__(self) -> str:
        where = f" in {self.source}" if self.source else ""
        return f"{self.scenario_id}{where}: ...{self.ngram}..."


def normalise(text: str) -> list[str]:
    """Lowercase word tokens. Punctuation, casing and whitespace are discarded.

    Speaker tags such as `[Daughter]` normalise to a bare word, which is the
    right behaviour: a multi-speaker scenario copied without its brackets is
    still the same leaked scenario.
    """
    return _WORD.findall(text.lower())


def _shingles(tokens: Sequence[str], n: int) -> set[str]:
    if len(tokens) < n:
        # Short utterance: use the whole thing rather than emitting nothing.
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def holdout_shingles(
    records: Iterable[CuratedScenario] | None = None, n: int = DEFAULT_SHINGLE_N
) -> dict[str, set[str]]:
    """`scenario_id -> set of n-grams` over the held-out utterances."""
    rows = holdout_scenarios() if records is None else records
    return {r.scenario_id: _shingles(normalise(r.text), n) for r in rows}


def find_leaks(
    text: str,
    source: str = "",
    records: Iterable[CuratedScenario] | None = None,
    n: int = DEFAULT_SHINGLE_N,
) -> list[Leak]:
    """Every held-out n-gram present in `text`, at most one report per scenario."""
    haystack = " ".join(normalise(text))
    if not haystack:
        return []
    leaks: list[Leak] = []
    for scenario_id, grams in holdout_shingles(records, n).items():
        hit = next((g for g in sorted(grams) if g in haystack), None)
        if hit is not None:
            leaks.append(Leak(scenario_id=scenario_id, ngram=hit, source=source))
    return sorted(leaks, key=lambda leak: leak.scenario_id)


class HoldoutLeakError(AssertionError):
    """Held-out scenario text was found in an artifact that must not contain it."""


def assert_no_holdout_leakage(
    text: str,
    source: str = "",
    records: Iterable[CuratedScenario] | None = None,
    n: int = DEFAULT_SHINGLE_N,
) -> None:
    leaks = find_leaks(text, source=source, records=records, n=n)
    if leaks:
        detail = "\n".join(f"  - {leak}" for leak in leaks)
        raise HoldoutLeakError(
            f"held-out scenario text found in {source or 'the scanned text'} "
            f"({len(leaks)} scenario(s)):\n{detail}\n"
            "  Optimisation and few-shot selection must draw only from "
            "carelite.scenarios.bank.train_scenarios()."
        )


def scan_paths(paths: Iterable[Path], n: int = DEFAULT_SHINGLE_N) -> list[Leak]:
    """Scan files for held-out text. Missing paths are skipped, not an error.

    Skipping missing paths is deliberate: this is meant to be pointed at
    `carelite/prompts/` from a test that runs before that lane has landed.
    """
    leaks: list[Leak] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        leaks.extend(find_leaks(content, source=str(path), n=n))
    return leaks
