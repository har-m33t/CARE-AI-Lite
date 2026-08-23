"""The held-out split is write-once. This module is what makes that structural.

Build plan v3, Part V: *"40 training scenarios is few. The held-out set is the
only defense; treat it as write-once."* And sprint 9's gate: *"held-out
untouched"*. A sentence in a document does not enforce that. Two mechanisms
here do:

**1. A checksum over the held-out set.** `HOLDOUT_DIGEST` below is a sha256 over
the canonical serialisation of all 60 held-out records. `verify_holdout()`
recomputes it, and `tests/unit/scenarios/test_freeze.py` calls that on every
`make check`. Editing a held-out utterance, changing one of its stratum values,
moving a scenario across the split boundary, or adding or removing a held-out
scenario all fail that test. The failure names the specific scenario ids that
changed -- `scenarios/holdout.lock` carries a per-record digest so the diff is
readable rather than a single mismatched hex string.

**2. No ordinary path rewrites the lock.** `write_lock()` refuses to run unless
`CARELITE_UNFREEZE_HOLDOUT` is set to an explicit acknowledgement string, and
there is deliberately no Make target and no CLI subcommand that calls it. If
the held-out set genuinely has to change -- a factual error in a scenario, say
-- that is a protocol amendment, and it should cost a deliberate act plus a note
in the limitations record, not a routine regeneration.

**What is frozen, precisely.** The eight fields of `carelite.types.Scenario`
plus `equity_kind`. Those are the evaluation item: change any of them and a
held-out result means something different. `hard_case` and `curator_note` are
curation commentary -- a reviewer may sharpen a note at the wave-2 gate without
that being an edit to the held-out set -- so they are outside the digest, on
purpose. Everything about the *train* split is likewise outside it; the train
scenarios are meant to be worked on.

This guards accidental edits, not a determined adversary: someone can always
recompute the constant. That is fine. The failure mode being defended against
is a well-meaning edit during Sprint 9 that nobody notices.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from carelite.config import REPO_ROOT
from carelite.scenarios.bank import CuratedScenario, load_bank
from carelite.types import Split

__all__ = [
    "HOLDOUT_DIGEST",
    "LOCK_PATH",
    "UNFREEZE_ENV",
    "UNFREEZE_TOKEN",
    "HoldoutTamperError",
    "canonical_record",
    "holdout_digest",
    "read_lock",
    "record_digest",
    "record_digests",
    "verify_holdout",
    "write_lock",
]

LOCK_PATH: Path = REPO_ROOT / "scenarios" / "holdout.lock"

UNFREEZE_ENV: Final = "CARELITE_UNFREEZE_HOLDOUT"
UNFREEZE_TOKEN: Final = "i-understand-this-invalidates-the-holdout"

#: sha256 over the newline-joined per-record digests of the 60 held-out
#: scenarios, ordered by scenario_id. Frozen 2026-08-22, before any generation
#: existed. Do not update this to make a test pass.
# The pragma below is on the value line because detect-secrets matches per line. This is a
# content checksum over public synthetic text, not a credential.
HOLDOUT_DIGEST: Final = (
    "adfedb33cbbb2ec627bff50ae25a572594f0d4e36b5b34fb2804b3408c3600c4"  # pragma: allowlist secret
)

#: The fields that constitute the evaluation item, in a fixed order.
FROZEN_FIELDS: Final[tuple[str, ...]] = (
    "scenario_id",
    "text",
    "challenge_type",
    "emotion_intensity",
    "encounter_phase",
    "literacy_signal",
    "equity_stratum",
    "split",
    "equity_kind",
)


class HoldoutTamperError(AssertionError):
    """The held-out set no longer matches its frozen checksum."""


def canonical_record(record: CuratedScenario) -> str:
    """Deterministic serialisation of one scenario's evaluation-relevant fields.

    Sorted keys, no whitespace variance, `ensure_ascii=False` so the digest is
    over the text as written rather than over an escaping choice.
    """
    payload = {name: getattr(record, name) for name in FROZEN_FIELDS}
    payload["encounter_phase"] = str(payload["encounter_phase"])
    payload["split"] = str(payload["split"])
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def record_digest(record: CuratedScenario) -> str:
    return hashlib.sha256(canonical_record(record).encode("utf-8")).hexdigest()


def _holdout(records: Sequence[CuratedScenario] | None) -> list[CuratedScenario]:
    rows = list(load_bank()) if records is None else list(records)
    return sorted((r for r in rows if r.split is Split.HOLDOUT), key=lambda r: r.scenario_id)


def record_digests(records: Sequence[CuratedScenario] | None = None) -> dict[str, str]:
    """`scenario_id -> digest` for the held-out set, in id order."""
    return {r.scenario_id: record_digest(r) for r in _holdout(records)}


def holdout_digest(records: Sequence[CuratedScenario] | None = None) -> str:
    """The aggregate checksum compared against `HOLDOUT_DIGEST`."""
    joined = "\n".join(f"{sid}  {dig}" for sid, dig in record_digests(records).items())
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def read_lock(path: Path | None = None) -> dict[str, str]:
    """Parse `scenarios/holdout.lock` into `scenario_id -> digest`."""
    src = path or LOCK_PATH
    if not src.exists():
        raise HoldoutTamperError(f"holdout lock file missing at {src}")
    entries: dict[str, str] = {}
    for raw in src.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise HoldoutTamperError(f"{src}: malformed lock line {raw!r}")
        entries[parts[0]] = parts[1]
    return entries


def verify_holdout(records: Sequence[CuratedScenario] | None = None) -> str:
    """Raise `HoldoutTamperError` if the held-out set has changed at all.

    The message names the added, removed and modified scenario ids rather than
    just reporting a hash mismatch, because the first question anyone asks on
    seeing this fail is "which one".
    """
    current = record_digests(records)
    locked = read_lock()

    added = sorted(set(current) - set(locked))
    removed = sorted(set(locked) - set(current))
    modified = sorted(sid for sid in set(current) & set(locked) if current[sid] != locked[sid])

    if added or removed or modified:
        detail: list[str] = []
        if modified:
            detail.append(f"  modified: {', '.join(modified)}")
        if added:
            detail.append(f"  added to holdout: {', '.join(added)}")
        if removed:
            detail.append(f"  removed from holdout: {', '.join(removed)}")
        raise HoldoutTamperError(
            "the held-out scenario set has been edited; it is write-once "
            "(build plan v3, Part V).\n" + "\n".join(detail) + "\n"
            "  If this change is intended it is a protocol amendment: record it in the "
            "limitations, then re-lock deliberately.\n"
            "  Any held-out result produced before this edit is not comparable to one "
            "produced after it."
        )

    digest = holdout_digest(records)
    if digest != HOLDOUT_DIGEST:
        raise HoldoutTamperError(
            "the held-out aggregate checksum does not match the frozen constant in "
            "carelite.scenarios.freeze, although every per-record digest matches the "
            f"lock file. Expected {HOLDOUT_DIGEST}, computed {digest}. Either the lock "
            "file and the constant were updated inconsistently, or the set of held-out "
            "ids changed in a way the lock file was updated to accept."
        )
    return digest


def write_lock(records: Sequence[CuratedScenario] | None = None, path: Path | None = None) -> str:
    """Regenerate the lock file. Deliberately hard to reach.

    Refuses unless the caller has set::

        CARELITE_UNFREEZE_HOLDOUT=i-understand-this-invalidates-the-holdout

    Returns the new aggregate digest, which then has to be pasted into
    `HOLDOUT_DIGEST` by hand. That second manual step is intentional: it forces
    a code change, which forces a diff, which forces a reviewer.
    """
    if os.environ.get(UNFREEZE_ENV) != UNFREEZE_TOKEN:
        raise HoldoutTamperError(
            f"refusing to rewrite the held-out lock file. The held-out set is write-once. "
            f"If you genuinely mean to re-lock it, set {UNFREEZE_ENV}={UNFREEZE_TOKEN} and "
            f"record the amendment in the limitations."
        )
    digests = record_digests(records)
    digest = holdout_digest(records)
    dest = path or LOCK_PATH
    header = [
        "# CARELite held-out scenario lock. Write-once -- see carelite/scenarios/freeze.py.",
        "# One line per held-out scenario: <scenario_id>  <sha256 of its frozen fields>.",
        f"# aggregate: {digest}",
    ]
    body = [f"{sid}  {dig}" for sid, dig in digests.items()]
    dest.write_text("\n".join([*header, *body]) + "\n", encoding="utf-8")
    return digest
