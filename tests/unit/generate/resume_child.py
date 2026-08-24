"""The child process `test_runner_resume.py` kills with SIGKILL.

Not named `test_*`, so pytest does not collect it. It runs the real
`carelite.generate.runner.run` loop against a `JsonlStore`, with a fake
generation client that sleeps so the parent has a window in which to kill it.

Every call the fake client makes is appended to a second journal. That is what
lets the test assert the thing that actually matters: after the restart, the
number of model calls across both processes is the number of cells plus at most
the one that was in flight. A resumed run that quietly recomputed everything
would finish with the right journal and the wrong call count.

    python -m tests.unit.generate.resume_child <journal> <calls> <delay_s> <n>
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carelite.generate.graph import GraphDeps, InputPolicy, build_graph
from carelite.generate.model import GenerationClient, GenerationOutput
from carelite.generate.runner import run
from carelite.generate.store import JsonlStore
from carelite.safety.fencing import FencedPrompt
from carelite.types import Condition, EncounterPhase, Scenario, Split

DIGEST = "sha256:resumetest"


@dataclass
class SlowFakeClient(GenerationClient):
    """Records every call durably, then sleeps, then answers."""

    calls_path: Path = field(default_factory=lambda: Path("calls.jsonl"))
    delay_s: float = 0.0

    def resolve_digest(self, model_tag: str) -> str:
        return DIGEST

    def generate(
        self,
        prompt: FencedPrompt,
        *,
        model_tag: str,
        seed: int,
        temperature: float,
        num_predict: int = 512,
        window: int = 8192,
        json_format: bool = False,
    ) -> GenerationOutput:
        with self.calls_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"seed": seed, "pid": os.getpid()}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        time.sleep(self.delay_s)
        return GenerationOutput(
            text=f"A steady reply for seed {seed}.",
            model=model_tag,
            model_digest=DIGEST,
            latency_ms=1,
            num_ctx=8192,
            prompt_chars=len(prompt),
        )


def scenarios(n: int) -> list[Scenario]:
    return [
        Scenario(
            scenario_id=f"SC-R{i:03d}",
            text=(
                "I do not understand why I need another test. "
                f"Nobody explains anything to me, visit number {i}."
            ),
            challenge_type="frustration_with_care",
            emotion_intensity=3,
            encounter_phase=EncounterPhase.EXPLANATION,
            literacy_signal="low",
            equity_stratum=False,
            split=Split.HOLDOUT,
        )
        for i in range(n)
    ]


def main(argv: list[str]) -> int:
    journal, calls, delay, count = argv[0], argv[1], float(argv[2]), int(argv[3])
    store = JsonlStore(path=Path(journal))
    client = SlowFakeClient(calls_path=Path(calls), delay_s=delay)
    deps: Any = GraphDeps(client=client, input_policy=InputPolicy.CURATED_BANK)
    report = run(
        store=store,
        scenarios=scenarios(count),
        conditions=[Condition.A],  # self_check off: one model call per cell
        samples=1,
        deps=deps,
        graph=build_graph(prefer_langgraph=False),
        digests={Condition.A: DIGEST},
    )
    store.close()
    print(report.summary(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
