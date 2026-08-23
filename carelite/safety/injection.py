"""Prompt-injection screening for untrusted text.

This is the *lexical* layer and it is explicitly the weaker of the two. Any
enumeration of attack phrases is beatable by a phrasing nobody enumerated, so
this module is a tripwire and a telemetry source, not the defence. The defence
is `fencing.py`, which removes the capability rather than guessing at intent.

What this layer buys: it catches the obvious attempts early, before they cost a
retrieval and a generation; it produces `flags` a study can count; and it lets a
poisoned corpus chunk be neutralised in place rather than blocking a whole turn.

Two entry points, with deliberately different policies:

* `screen_utterance` — terminal input. High-confidence attacks **block**
  (`allowed=False`); obfuscation and structural noise are **redacted**
  (`allowed=True` with `redacted_text`), because a patient who happens to type
  a code fence should not lose their turn.
* `screen_retrieved` — corpus text. **Never blocks.** Blocking here would let
  one poisoned chunk take down retrieval; the injected span is redacted and the
  turn proceeds with the rest of the evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from carelite.safety.normalize import (
    detection_forms,
    has_control_chars,
    has_invisibles,
    normalize_text,
    squeezed_forms,
)
from carelite.types import SafetyVerdict

REDACTION = "[redacted: injection attempt]"


class Severity(StrEnum):
    """BLOCK cannot be made safe by removing the span; REDACT can."""

    BLOCK = "block"
    REDACT = "redact"


@dataclass(frozen=True)
class Rule:
    flag: str
    severity: Severity
    description: str
    patterns: tuple[str, ...]
    #: matched against the punctuation-free, separator-free `squeeze` form
    squeezed: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------

RULES: tuple[Rule, ...] = (
    Rule(
        flag="injection.instruction_override",
        severity=Severity.BLOCK,
        description="attempts to cancel or replace the system instructions",
        patterns=(
            r"\b(ignore|disregard|forget|discard|override|bypass|erase)\b[^.!?\n]{0,40}?"
            r"\b(previous|prior|above|earlier|initial|original|all|any|your|the)\b"
            r"[^.!?\n]{0,30}?\b(instruction|instructions|prompt|prompts|rule|rules|"
            r"direction|directions|guideline|guidelines|context|training|programming|"
            r"constraint|constraints|restriction|restrictions|policy|policies)\b",
            r"\bignore\s+everything\b",
            r"\bforget\s+(everything|all)\b",
            r"\b(new|updated|revised|real|actual|true)\s+(instructions?|system\s+prompt|"
            r"directive|task)\s*[:\-]",
            r"\bthese\s+are\s+your\s+new\b",
            r"\bstop\s+being\b[^.\n]{0,30}\b(assistant|carelite|helper)\b",
            r"\byour\s+(real|true|actual)\s+(task|job|purpose|instruction)\b",
            r"\bdisregard\s+(safety|the\s+safety|your\s+safety)\b",
            r"\boverride\s+code\b",
            r"\bsudo\b\s+\w+",
            r"\badmin(istrator)?\s+(mode|override|access)\b",
        ),
        squeezed=(
            "ignoreallpreviousinstructions",
            "ignorepreviousinstructions",
            "disregardallpreviousinstructions",
            "ignoreallpriorinstructions",
            "forgetallpreviousinstructions",
        ),
    ),
    Rule(
        flag="injection.role_reassignment",
        severity=Severity.BLOCK,
        description="attempts to replace the assistant's role or persona",
        patterns=(
            r"\byou\s+are\s+(now|no\s+longer)\b",
            r"\bfrom\s+now\s+on[, ]{0,3}you\b",
            r"\b(act|behave|respond|answer)\s+as\s+(a|an|if\s+you\s+(are|were))\b",
            r"\bpretend\s+(to\s+be|you\s+(are|were))\b",
            r"\brole\s*[- ]?\s*play\s+as\b",
            r"\byou\s+are\s+(an?\s+)?(unrestricted|unfiltered|uncensored|amoral|jailbroken)\b",
            r"\b(dan|do\s+anything\s+now)\s+mode\b",
            r"\b(developer|debug|god|maintenance|test|dev)\s+mode\s+(on|enabled|activated)\b",
            r"\benable\s+(developer|debug|god|dan)\s+mode\b",
            r"\bjail\s*break\b",
            r"\bsimulate\s+(a|an)\s+\w+\s+(that|which|who)\s+(has\s+no|ignores)\b",
            r"\byou\s+are\s+(a\s+)?(doctor|physician|licensed\s+\w+)\s+(and|who)\b"
            r"[^.\n]{0,40}\b(prescrib|diagnos|dose|dosage)\w*",
            r"\bhypothetically[, ]{0,3}if\s+you\s+had\s+no\s+(rules|restrictions|guidelines)\b",
        ),
        squeezed=("youarenowdan", "doanythingnow", "developermodeenabled"),
    ),
    Rule(
        flag="injection.system_prompt_extraction",
        severity=Severity.BLOCK,
        description="attempts to read back the system prompt or hidden configuration",
        patterns=(
            r"\b(repeat|print|show|reveal|display|output|echo|recite|dump|reproduce|list)\b"
            r"[^.!?\n]{0,40}?\b(system\s+prompt|initial\s+prompt|your\s+prompt|your\s+"
            r"instructions|the\s+instructions|your\s+rules|your\s+guidelines|"
            r"everything\s+above|the\s+text\s+above|your\s+configuration|your\s+context)\b",
            r"\bwhat\s+(are|were)\s+(your|the)\s+(exact\s+)?(instructions|rules|system\s+"
            r"prompt|guidelines|directives)\b",
            r"\bwhat\s+(does|did)\s+(your|the)\s+system\s+(prompt|message)\s+say\b",
            r"\btell\s+me\s+(your|the)\s+(system\s+prompt|instructions|prompt\s+template)\b",
            r"\bverbatim\b[^.!?\n]{0,30}\b(above|instructions|prompt)\b",
            r"\b(begin|start)\s+your\s+(reply|answer|response)\s+with\s+"
            r"[\"']?\s*(you\s+are|system)\b",
            r"\bsummari[sz]e\s+your\s+(instructions|system\s+prompt)\b",
            r"\bwhat\s+is\s+written\s+(above|before\s+this)\b",
        ),
        squeezed=(
            "repeatyoursystemprompt",
            "printyourinstructions",
            "whatareyourinstructions",
            "revealyoursystemprompt",
        ),
    ),
    Rule(
        flag="injection.exfiltration",
        severity=Severity.BLOCK,
        description="attempts to route content to an external destination",
        patterns=(
            r"\b(send|post|upload|transmit|forward|exfiltrate|leak)\b[^.\n]{0,40}?"
            r"\b(to|at)\b[^.\n]{0,20}?(https?://|www\.|@[\w.-]+\.\w{2,})",
            r"!\[[^\]]*\]\(\s*https?://",
            r"\bcurl\s+https?://",
            r"\bfetch\s*\(\s*[\"']https?://",
            r"\bappend\s+(the|your)\s+\w+\s+to\s+the\s+url\b",
            r"\bencode\s+(the|your)\s+(instructions|prompt|context)\b[^.\n]{0,30}\burl\b",
        ),
    ),
    Rule(
        flag="injection.delimiter_break",
        severity=Severity.REDACT,
        description="chat-template or fence delimiters embedded in the text",
        patterns=(
            r"<\|[a-z_]{2,20}\|>",
            r"<\/?(s|system|assistant|user|instruction|im_start|im_end)>",
            r"\[/?INST\]",
            r"\[/?SYS\]",
            r"###\s*(system|instruction|assistant|human)\b",
            r"^\s*(system|assistant)\s*:",
            r"\bcarelite[_\s-]?untrusted\b",
            r"<<<+",
            r"```\s*(system|instructions?)\b",
            r"\bend\s+of\s+(patient\s+)?(input|context|data|document)\b",
            r"-{5,}\s*(system|instruction)",
        ),
    ),
    Rule(
        flag="injection.encoded_payload",
        severity=Severity.REDACT,
        description="encoded or obfuscated content that hides its own meaning",
        patterns=(
            r"\b(decode|decrypt|un\s?scramble|de\s?obfuscate)\b[^.\n]{0,30}"
            r"\b(this|the\s+following|below|base\s?64|hex|rot\s?13)\b",
            r"\b(base\s?64|rot\s?13|hex(adecimal)?)\s*(encoded|string|payload|:)",
            r"\brot\s?13\b",
            r"\b[A-Za-z0-9+/]{40,}={0,2}\b",
            r"(?:\\u[0-9a-fA-F]{4}){4,}",
            r"(?:\\x[0-9a-fA-F]{2}){6,}",
            r"(?:&#\d{2,5};){6,}",
            r"\bread\s+(this|the\s+following)\s+backwards\b",
        ),
    ),
    Rule(
        flag="injection.embedded_directive",
        severity=Severity.REDACT,
        description="second-person directives aimed at the model inside quoted data",
        patterns=(
            r"\b(assistant|model|ai|chatbot|llm|system)\s*[,:]\s*(please\s+)?"
            r"(ignore|do|respond|reply|output|say|answer|note)\b",
            r"\bimportant\s+(note|instruction)\s+(to|for)\s+(the\s+)?"
            r"(assistant|ai|model|reader)\b",
            r"\bwhen\s+(summari[sz]ing|answering|responding)[^.\n]{0,30}\b"
            r"(always|be\s+sure\s+to|you\s+must)\b",
            r"\bdo\s+not\s+(mention|disclose|reveal|tell)\b[^.\n]{0,30}"
            r"\b(this|these\s+instructions|the\s+user|anyone)\b",
            r"\bthis\s+(is|takes)\s+(a\s+)?(higher\s+)?priority\s+over\b",
            r"\byour\s+(previous|prior)\s+answer\s+was\s+wrong[^.\n]{0,20}instead\b",
        ),
    ),
    Rule(
        flag="injection.safety_bypass",
        severity=Severity.BLOCK,
        description="explicit requests to drop safety behaviour",
        patterns=(
            r"\b(without|no|skip|remove|drop|turn\s+off|disable)\b[^.\n]{0,25}"
            r"\b(safety|guardrails?|filters?|restrictions?|disclaimers?|warnings?|"
            r"content\s+polic\w+|ethical\s+\w+)\b",
            r"\byou\s+(are\s+)?(allowed|permitted|free)\s+to\s+ignore\b",
            r"\bthere\s+are\s+no\s+(rules|restrictions|limits)\b",
            r"\bthis\s+is\s+(just\s+)?(a\s+)?(test|hypothetical|fiction)[^.\n]{0,30}"
            r"\bso\s+(you\s+can|ignore)\b",
            r"\bi\s+am\s+(the|your)\s+(developer|admin|administrator|creator|"
            r"engineer|operator)\b",
        ),
    ),
)

_COMPILED: tuple[tuple[Rule, tuple[re.Pattern[str], ...]], ...] = tuple(
    (rule, tuple(re.compile(p, re.IGNORECASE | re.MULTILINE) for p in rule.patterns))
    for rule in RULES
)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    rule: Rule
    span: str


def _find_matches(text: str) -> list[Match]:
    """Run every rule over every canonical form of the text."""
    matches: list[Match] = []
    seen: set[str] = set()

    normalized = normalize_text(text)
    haystacks = {normalized, *detection_forms(text)}
    squeezed = squeezed_forms(text)

    for rule, patterns in _COMPILED:
        hit: str | None = None
        for pattern in patterns:
            for hay in haystacks:
                m = pattern.search(hay)
                if m:
                    hit = m.group(0)
                    break
            if hit:
                break
        if hit is None:
            for needle in rule.squeezed:
                if any(needle in s for s in squeezed):
                    hit = needle
                    break
        if hit is not None and rule.flag not in seen:
            seen.add(rule.flag)
            matches.append(Match(rule=rule, span=hit))
    return matches


def _redact(text: str, matches: list[Match]) -> str:
    """Replace matched spans in the *normalised* text.

    Redaction runs on the normalised form because that is the only form whose
    offsets are meaningful for the original string; matches found only in the
    squeezed or de-leeted forms cannot be located precisely, so those fall back
    to flagging without redaction — which is why a squeeze-only hit is always a
    BLOCK rule.
    """
    out = normalize_text(text)
    for match in matches:
        for pattern in dict(_COMPILED)[match.rule]:
            out = pattern.sub(REDACTION, out)
    return out


def detect(text: str) -> list[str]:
    """Flags only. Cheap probe for lanes that want telemetry, not a verdict."""
    return [m.rule.flag for m in _find_matches(text)]


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def screen_utterance(text: str) -> SafetyVerdict:
    """Screen terminal input before it reaches query construction.

    High-confidence attacks block the turn. Structural noise and obfuscation are
    redacted and the turn proceeds on `redacted_text`.
    """
    matches = _find_matches(text)
    flags = [m.rule.flag for m in matches]

    if has_invisibles(text):
        flags.append("injection.invisible_characters")
    if has_control_chars(text):
        flags.append("injection.control_characters")

    if not flags:
        return SafetyVerdict(allowed=True, reason=None)

    blocking = [m for m in matches if m.rule.severity is Severity.BLOCK]
    redacted = _redact(text, matches) if matches else normalize_text(text)

    if blocking:
        reasons = "; ".join(sorted({m.rule.description for m in blocking}))
        return SafetyVerdict(
            allowed=False,
            injection_detected=True,
            flags=flags,
            redacted_text=redacted,
            reason=(
                f"Blocked: the input contains {reasons}. CARELite treats terminal input as "
                "untrusted; rephrase the patient utterance without instructions aimed at the "
                "assistant."
            ),
        )

    return SafetyVerdict(
        allowed=True,
        injection_detected=True,
        flags=flags,
        redacted_text=redacted,
        reason=(
            "Input contained delimiter or encoding artefacts that were removed before the "
            "turn proceeded. The remaining text is treated as quoted patient data."
        ),
    )


def screen_retrieved(text: str, *, ref_id: str | None = None) -> SafetyVerdict:
    """Screen retrieved corpus text. Redacts; never blocks.

    Corpus text is untrusted because contextual prefixes are LLM-generated, so a
    poisoned source PDF becomes a poisoned prefix. Dropping the whole turn would
    hand an attacker a denial-of-service against retrieval, so the injected span
    is redacted and the chunk continues into the (fenced) context block.
    """
    matches = _find_matches(text)
    if not matches:
        return SafetyVerdict(allowed=True, redacted_text=normalize_text(text))

    flags = [m.rule.flag for m in matches]
    where = f" in {ref_id}" if ref_id else ""
    return SafetyVerdict(
        allowed=True,
        injection_detected=True,
        flags=[*flags, "injection.in_retrieved_context"],
        redacted_text=_redact(text, matches),
        reason=(
            f"Retrieved context{where} contained instruction-shaped text "
            f"({', '.join(sorted({m.rule.description for m in matches}))}); the span was "
            "redacted. Corpus contextual prefixes are LLM-generated and are a poisoning vector."
        ),
    )


__all__ = [
    "REDACTION",
    "RULES",
    "Match",
    "Rule",
    "Severity",
    "detect",
    "screen_retrieved",
    "screen_utterance",
]
