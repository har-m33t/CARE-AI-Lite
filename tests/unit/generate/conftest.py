"""Fakes for the generation lane.

Everything in `tests/unit/generate/` runs with no Ollama daemon and no
Postgres. `FakeClient` stands in for `GenerationClient` and records every
prompt it was handed, which is how the fencing assertions get made: a test can
check what actually reached the system role rather than trusting that it was
assembled correctly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from carelite.generate.model import GenerationClient, GenerationError, GenerationOutput
from carelite.safety.fencing import FencedPrompt
from carelite.types import EncounterPhase, Scenario, Split


@dataclass
class FakeClient(GenerationClient):
    """A `GenerationClient` that never touches a daemon.

    `reply` receives the assembled prompt and the call index, so a test can make
    the first call return a draft and the second (the self-check) return a
    verdict object.
    """

    reply: Callable[[FencedPrompt, int], str] = lambda prompt, i: "A steady, ordinary reply."
    digest: str = "sha256:fakedigest"
    fail_with: str | None = None
    prompts_seen: list[FencedPrompt] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def resolve_digest(self, model_tag: str) -> str:
        return self.digest

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
        index = len(self.prompts_seen)
        self.prompts_seen.append(prompt)
        self.calls.append(
            {
                "model_tag": model_tag,
                "seed": seed,
                "temperature": temperature,
                "json_format": json_format,
                "window": window,
            }
        )
        if self.fail_with:
            raise GenerationError(self.fail_with)
        return GenerationOutput(
            text=self.reply(prompt, index),
            model=model_tag,
            model_digest=self.digest,
            latency_ms=1,
            num_ctx=8192,
            prompt_chars=len(prompt),
        )


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def holdout_like() -> list[Scenario]:
    """Three scenarios shaped like the bank's, with no safety-screen triggers."""
    return [
        Scenario(
            scenario_id=f"SC-T{i:02d}",
            text=(
                "I do not understand why I need another test. "
                f"Nobody explains anything to me here, visit {i}."
            ),
            challenge_type="frustration_with_care",
            emotion_intensity=3,
            encounter_phase=EncounterPhase.EXPLANATION,
            literacy_signal="low",
            equity_stratum=False,
            split=Split.HOLDOUT,
        )
        for i in range(3)
    ]
