"""The measured safety report, as an executable artifact.

`pytest tests/security/test_metrics.py -s` prints the table. The assertions
below pin every number that a wave-3 gate or a write-up would quote, so the
figures in the report cannot drift away from the figures in the code.
"""

from __future__ import annotations

import pytest

from carelite.safety import injection, output_gate, phi, redflag
from tests.security import corpus

pytestmark = pytest.mark.security


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def measure() -> dict[str, dict[str, float | int]]:
    """Recall on positives and false-positive rate on matched negatives."""
    rf_missed = sum(1 for _, t in corpus.RED_FLAGS if not redflag.is_red_flag(t))
    rf_fp = sum(1 for t in corpus.RED_FLAG_NEGATIVES if redflag.is_red_flag(t))
    rf_hard = sum(1 for t in corpus.RED_FLAG_HARD_NEGATIVES if redflag.is_red_flag(t))

    inj_missed = sum(1 for _, t in corpus.INJECTIONS if not injection.detect(t))
    inj_fp = sum(1 for t in corpus.INJECTION_NEGATIVES if injection.detect(t))

    phi_missed = sum(1 for _, t in corpus.PHI_SAMPLES if not phi.contains_phi(t))
    phi_fp = sum(1 for t in corpus.PHI_NEGATIVES if phi.contains_phi(t))

    gate_missed = sum(1 for _, t in corpus.BAD_OUTPUTS if output_gate.screen(t).allowed)
    gate_fp = sum(1 for t in corpus.GOOD_OUTPUTS if not output_gate.screen(t).allowed)

    return {
        "red_flag": {
            "positives": len(corpus.RED_FLAGS),
            "recall": 1 - _rate(rf_missed, len(corpus.RED_FLAGS)),
            "negatives": len(corpus.RED_FLAG_NEGATIVES),
            "fp_rate": _rate(rf_fp, len(corpus.RED_FLAG_NEGATIVES)),
            "hard_negatives": len(corpus.RED_FLAG_HARD_NEGATIVES),
            "hard_fp_rate": _rate(rf_hard, len(corpus.RED_FLAG_HARD_NEGATIVES)),
        },
        "injection": {
            "positives": len(corpus.INJECTIONS),
            "recall": 1 - _rate(inj_missed, len(corpus.INJECTIONS)),
            "negatives": len(corpus.INJECTION_NEGATIVES),
            "fp_rate": _rate(inj_fp, len(corpus.INJECTION_NEGATIVES)),
        },
        "phi": {
            "positives": len(corpus.PHI_SAMPLES),
            "recall": 1 - _rate(phi_missed, len(corpus.PHI_SAMPLES)),
            "negatives": len(corpus.PHI_NEGATIVES),
            "fp_rate": _rate(phi_fp, len(corpus.PHI_NEGATIVES)),
        },
        "output_gate": {
            "positives": len(corpus.BAD_OUTPUTS),
            "recall": 1 - _rate(gate_missed, len(corpus.BAD_OUTPUTS)),
            "negatives": len(corpus.GOOD_OUTPUTS),
            "fp_rate": _rate(gate_fp, len(corpus.GOOD_OUTPUTS)),
        },
    }


def test_corpus_meets_its_minimum_size() -> None:
    """The brief's floor: 40 injections, 20 PHI samples, 20 red flags, plus negatives."""
    assert len(corpus.INJECTIONS) >= 40
    assert len(corpus.PHI_SAMPLES) >= 20
    assert len(corpus.RED_FLAGS) >= 20
    assert len(corpus.INJECTION_NEGATIVES) >= 20
    assert len(corpus.PHI_NEGATIVES) >= 20
    assert len(corpus.RED_FLAG_NEGATIVES) >= 20


def test_measured_safety_report(capsys: pytest.CaptureFixture[str]) -> None:
    m = measure()
    with capsys.disabled():
        print("\n\n  layer         positives  recall    negatives  FP rate")
        print("  " + "-" * 54)
        for name, row in m.items():
            print(
                f"  {name:<13} {row['positives']:>9}  {row['recall']:>6.1%}  "
                f"{row['negatives']:>9}  {row['fp_rate']:>6.1%}"
            )
        hard = m["red_flag"]
        print(
            f"\n  red-flag hard negatives (denials / third-party / historical): "
            f"{hard['hard_negatives']} samples, {hard['hard_fp_rate']:.1%} flagged."
        )
        print("  That cost is accepted deliberately: recall is the gate, not precision.\n")

    assert m["red_flag"]["recall"] == 1.0
    assert m["red_flag"]["fp_rate"] == 0.0
    assert m["injection"]["recall"] == 1.0
    assert m["injection"]["fp_rate"] == 0.0
    assert m["phi"]["recall"] == 1.0
    assert m["phi"]["fp_rate"] == 0.0
    assert m["output_gate"]["recall"] == 1.0
    assert m["output_gate"]["fp_rate"] == 0.0
