"""`kill -9` mid-run, restart, finish. The requirement, tested as stated.

The runner will be interrupted — 1,080 local generations do not complete in one
sitting — so resumption is not a nicety and is not something to assert by
reading the code. This test starts a real subprocess, kills it with `SIGKILL`
while it is mid-generation, restarts it against the same journal, and checks
three things:

1. the restarted run finishes every cell;
2. it does not recompute cells that were already durably stored;
3. the journal is still readable, including after a line torn by the kill.

No model and no database are involved: the child drives the real runner loop
with a fake client, which is exactly why the loop under test is the real one.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from carelite.config import REPO_ROOT
from carelite.generate.store import JsonlStore

CELLS = 8
DELAY_S = 0.5


def _spawn(journal: Path, calls: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.unit.generate.resume_child",
            str(journal),
            str(calls),
            str(DELAY_S),
            str(CELLS),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_for_lines(path: Path, n: int, timeout_s: float = 30.0) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            if count >= n:
                return count
        time.sleep(0.05)
    raise AssertionError(f"{path} did not reach {n} lines within {timeout_s}s")


def test_the_runner_resumes_after_a_kill_9(tmp_path: Path) -> None:
    journal = tmp_path / "generations.jsonl"
    calls = tmp_path / "calls.jsonl"

    first = _spawn(journal, calls)
    try:
        _wait_for_lines(journal, 2)
        os.kill(first.pid, signal.SIGKILL)
    finally:
        first.wait(timeout=30)

    assert first.returncode != 0, "the child was supposed to be killed, not to finish"
    stored_before = JsonlStore(path=journal).completed_keys()
    calls_before = len(calls.read_text(encoding="utf-8").splitlines())
    assert 0 < len(stored_before) < CELLS, "the kill must land mid-run to test anything"

    second = _spawn(journal, calls)
    out, err = second.communicate(timeout=120)
    assert second.returncode == 0, err.decode()

    stored_after = JsonlStore(path=journal).completed_keys()
    assert len(stored_after) == CELLS, out.decode()
    assert stored_before <= stored_after, "a resumed run must not lose a stored cell"

    # The one cell that was in flight when the kill landed may be recomputed;
    # nothing else may be. Anything more means the resume ignored the store.
    calls_total = len(calls.read_text(encoding="utf-8").splitlines())
    assert calls_total <= CELLS + 1, (
        f"{calls_total} model calls for {CELLS} cells: the restart recomputed "
        f"completed work ({calls_before} calls before the kill)"
    )
    assert calls_total >= CELLS


def test_a_second_run_over_a_finished_journal_generates_nothing(tmp_path: Path) -> None:
    """The idempotence half of the same property: rerunning a completed set is
    a no-op, so a cron-style rerun cannot duplicate rows."""
    journal = tmp_path / "generations.jsonl"
    calls = tmp_path / "calls.jsonl"

    first = _spawn(journal, calls)
    out, err = first.communicate(timeout=120)
    assert first.returncode == 0, err.decode()
    assert b"generated=8" in out

    calls_after_first = len(calls.read_text(encoding="utf-8").splitlines())
    second = _spawn(journal, calls)
    out2, err2 = second.communicate(timeout=120)
    assert second.returncode == 0, err2.decode()
    assert b"generated=0" in out2
    assert b"skipped=8" in out2
    assert len(calls.read_text(encoding="utf-8").splitlines()) == calls_after_first
