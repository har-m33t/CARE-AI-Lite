"""Output screening: the last check before a generation reaches a human.

Two things must never leave this system.

**Leaked instructions.** If a generation echoes the system prompt, the fencing
contract has failed somewhere upstream and the failure must be visible, not
silently shipped. The gate checks both for leak-shaped phrasing and — when the
caller passes the system text — for verbatim overlap with it, which catches a
successful extraction attack no phrase list would have anticipated.

**Clinical recommendations.** From README, *What This Project Is Not*: "CARELite
AI is not a diagnostic tool. It does not make clinical recommendations." That is
a project position, so it is enforced mechanically rather than left to the
model's judgement. Dosing, drug directives, and diagnostic assertions block the
output.

A third check is advisory rather than blocking. The README is equally explicit
that the system "is also not a script generator" — but the knowledge base
legitimately carries `example_behavior` phrasings, so example wording is normal
output. Only explicit dictation ("say exactly the following", "read this
verbatim") raises the non-blocking `output.script_dictation` flag, which the
study can count without the gate strangling ordinary suggestions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from carelite.safety import phi
from carelite.safety.fencing import SENTINEL
from carelite.safety.normalize import detection_form, normalize_text
from carelite.types import SafetyVerdict

#: Length in words of the shingle used for verbatim system-prompt overlap.
#: Twelve is long enough that ordinary English will not collide by chance and
#: short enough to catch a paraphrase-free quotation of one sentence.
SHINGLE_WORDS = 12


@dataclass(frozen=True)
class GateRule:
    flag: str
    blocking: bool
    description: str
    patterns: tuple[str, ...]


RULES: tuple[GateRule, ...] = (
    GateRule(
        flag="output.system_prompt_leak",
        blocking=True,
        description="the response discloses its own instructions or configuration",
        patterns=(
            re.escape(SENTINEL),
            r"\bmy\s+(system\s+)?(prompt|instructions|directives)\s+(are|is|say|state)\b",
            r"\bhere\s+(are|is)\s+my\s+(system\s+prompt|instructions|rules)\b",
            r"\b(the|my)\s+system\s+prompt\b",
            r"\bi\s+(was|am)\s+instructed\s+to\b",
            r"\bmy\s+initial\s+prompt\b",
            r"\byou\s+are\s+carelite\b",
            r"^\s*(system|assistant)\s*:\s",
            r"<\|[a-z_]{2,20}\|>",
            r"\[/?INST\]",
            r"###\s*(system|instruction)\b",
            r"\bas\s+an\s+ai\s+language\s+model,?\s+(my|i\s+was)\s+"
            r"(instructions|trained|configured)\b",
        ),
    ),
    GateRule(
        flag="output.clinical_dosing",
        blocking=True,
        description="a drug, dose, or administration schedule",
        patterns=(
            r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|milligrams?|micrograms?|grams?|ml|cc|"
            r"units?|iu|tablets?|pills?|capsules?|puffs?|drops?)\b",
            r"\b(?:once|twice|three\s+times|four\s+times|\d+\s*times)\s+(?:a|per|each)\s+"
            r"(?:day|week|night)\b",
            r"\b(?:b\.?i\.?d|t\.?i\.?d|q\.?i\.?d|q\.?h\.?s|p\.?r\.?n|q\d+h|p\.?o\.?)\b(?=[\s.,;])",
            r"\b(?:start|stop|begin|discontinue|increase|decrease|double|halve|switch|taper)\s+"
            r"(?:taking\s+)?(?:your|the|his|her|their)\s+"
            r"(?:dose|dosage|medication|meds|pills?|insulin|statin|metformin|"
            r"lisinopril|prednisone|antibiotics?|inhaler)\b",
            r"\byou\s+should\s+(?:take|start|stop|increase|decrease|switch|try)\s+"
            r"(?:taking\s+)?(?:a\s+|an\s+|the\s+|some\s+)?\w*"
            r"(?:medication|antibiotic|painkiller|insulin|steroid|pill|dose)\w*\b",
            r"\bi\s+(?:recommend|advise|suggest)\s+(?:that\s+you\s+)?"
            r"(?:take|start|stop|increase|decrease)\b",
            r"\b(?:prescribe|prescription\s+for)\b",
        ),
    ),
    GateRule(
        flag="output.diagnosis",
        blocking=True,
        description="a diagnostic assertion about the patient",
        patterns=(
            r"\byou\s+(?:have|are\s+having|likely\s+have|probably\s+have|definitely\s+have|"
            r"appear\s+to\s+have|most\s+likely\s+have)\s+(?:a|an|the)?\s*"
            r"(?:cancer|tumou?r|diabetes|pneumonia|infection|heart\s+attack|stroke|copd|"
            r"asthma|depression|anxiety\s+disorder|hypertension|high\s+blood\s+pressure|"
            r"covid|influenza|flu|ulcer|appendicitis|sepsis|kidney\s+(?:disease|failure)|"
            r"heart\s+failure|arthritis|migraine|clot|embolism|fracture)\b",
            r"\b(?:the|your)\s+diagnosis\s+is\b",
            r"\byou\s+(?:are|have\s+been)\s+diagnosed\s+with\b",
            r"\bthis\s+is\s+(?:definitely|certainly|clearly)\s+(?:a\s+|an\s+)?\w+"
            r"(?:itis|osis|emia|oma|pathy)\b",
            r"\byour\s+(?:test|scan|labs?|results?|bloodwork)\s+(?:shows?|means?|indicates?)\s+"
            r"(?:that\s+)?you\b",
            r"\bit'?s\s+(?:not\s+)?cancer\b",
        ),
    ),
    GateRule(
        flag="output.discourages_care",
        blocking=True,
        description="advice against seeking or continuing clinical care",
        patterns=(
            r"\byou\s+(?:don'?t|do\s+not)\s+need\s+to\s+(?:see|call|go\s+to|visit|worry)\b",
            r"\b(?:no\s+need|there'?s\s+no\s+need)\s+to\s+(?:see|call|go\s+to|worry\s+about)\b",
            r"\bthat'?s\s+nothing\s+to\s+worry\s+about\b",
            r"\bit'?s\s+(?:probably|definitely)\s+nothing\b",
            r"\byou\s+can\s+(?:skip|cancel)\s+(?:the|your)\s+"
            r"(?:appointment|follow[-\s]?up|test|referral|scan)\b",
            r"\bwait\s+and\s+see\s+before\s+(?:calling|going\s+to)\s+(?:911|the\s+er)\b",
        ),
    ),
    GateRule(
        flag="output.script_dictation",
        blocking=False,
        description="dictated wording rather than a supported response",
        patterns=(
            r"\bsay\s+(?:exactly|precisely)\s+(?:this|the\s+following)\b",
            r"\bread\s+(?:this|the\s+following)\s+(?:verbatim|word\s+for\s+word|aloud\s+exactly)\b",
            r"\bword[-\s]for[-\s]word\b",
            r"\buse\s+(?:this|these)\s+exact\s+words?\b",
            r"\bfollow\s+this\s+script\b",
        ),
    ),
)

_COMPILED = tuple(
    (rule, tuple(re.compile(p, re.IGNORECASE | re.MULTILINE) for p in rule.patterns))
    for rule in RULES
)


def _shingles(text: str, n: int = SHINGLE_WORDS) -> set[str]:
    words = detection_form(text).split()
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def verbatim_overlap(response: str, system_prompt: str) -> str | None:
    """The first `SHINGLE_WORDS`-word run the response shares with the system prompt.

    Catches extraction attacks that succeed — the model quoting its instructions
    back — without needing to have anticipated the phrasing of the attack.
    """
    system_shingles = _shingles(system_prompt)
    if not system_shingles:
        return None
    for shingle in _shingles(response):
        if shingle in system_shingles:
            return shingle
    return None


@dataclass(frozen=True)
class GateHit:
    flag: str
    blocking: bool
    span: str
    description: str


def inspect(response: str, *, system_prompt: str | None = None) -> list[GateHit]:
    """Every gate hit in a generation, without forming a verdict."""
    normalized = normalize_text(response)
    hits: list[GateHit] = []

    for rule, patterns in _COMPILED:
        for pattern in patterns:
            m = pattern.search(normalized)
            if m:
                hits.append(
                    GateHit(
                        flag=rule.flag,
                        blocking=rule.blocking,
                        span=m.group(0),
                        description=rule.description,
                    )
                )
                break

    if system_prompt:
        overlap = verbatim_overlap(response, system_prompt)
        if overlap:
            hits.append(
                GateHit(
                    flag="output.system_prompt_verbatim",
                    blocking=True,
                    span=overlap,
                    description="the response quotes the system prompt verbatim",
                )
            )

    phi_hits = phi.find_phi(response)
    if phi_hits:
        hits.append(
            GateHit(
                flag="output.phi_leak",
                blocking=True,
                span=phi_hits[0].span,
                description="the response contains identifiers that must not be emitted",
            )
        )
    return hits


def screen(response: str, *, system_prompt: str | None = None) -> SafetyVerdict:
    """Gate a generation.

    Args:
        response: model output. Untrusted — a model that has been injected is
            an attacker's channel.
        system_prompt: the trusted system text used for this generation. Pass it
            whenever it is available; it enables the verbatim-overlap check,
            which is the only leak check that does not depend on a phrase list.

    Returns:
        `allowed=False` when anything blocking fired. `redacted_text` is
        populated only for a PHI-only hit, where redaction is meaningful; a
        leaked instruction or a clinical recommendation cannot be made safe by
        deleting a span, so those return `None` and the caller must not display
        the generation.
    """
    hits = inspect(response, system_prompt=system_prompt)
    if not hits:
        return SafetyVerdict(allowed=True)

    flags = [h.flag for h in hits]
    blocking = [h for h in hits if h.blocking]

    if not blocking:
        return SafetyVerdict(
            allowed=True,
            flags=flags,
            reason="Advisory: " + "; ".join(sorted({h.description for h in hits})) + ".",
        )

    only_phi = {h.flag for h in blocking} == {"output.phi_leak"}
    return SafetyVerdict(
        allowed=False,
        phi_detected=any(h.flag == "output.phi_leak" for h in hits),
        injection_detected=any("system_prompt" in h.flag for h in hits),
        flags=flags,
        redacted_text=phi.redact(response) if only_phi else None,
        reason=(
            "Response withheld: "
            + "; ".join(sorted({h.description for h in blocking}))
            + ". CARELite supports communication; it does not diagnose, dose, or advise "
            "against care, and it does not disclose its own instructions."
        ),
    )


__all__ = [
    "RULES",
    "SHINGLE_WORDS",
    "GateHit",
    "GateRule",
    "inspect",
    "screen",
    "verbatim_overlap",
]
