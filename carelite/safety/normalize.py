"""Shared text hygiene for the safety layer.

Every detector in this package sees adversarial text. An attacker's cheapest
move is not a cleverer phrase but a cheaper encoding: zero-width joiners inside
a keyword, a fullwidth homoglyph, leetspeak, letter spacing, or a bidi override
that makes the rendered string differ from the bytes. This module collapses all
of that into a small number of canonical forms that the pattern sets match
against, so a detector author writes one plain-English phrase and gets the
obfuscated variants for free.

Everything here is pure, deterministic, and allocation-cheap. No I/O, no
network, no model calls.
"""

from __future__ import annotations

import re
import unicodedata

# Characters that render as nothing (or as direction changes) and therefore let
# an attacker split a keyword without changing what a human sees.
INVISIBLE_CODEPOINTS = (
    "\u00ad"  # soft hyphen
    "\u180e"  # mongolian vowel separator
    "\u200b\u200c\u200d\u200e\u200f"  # zero-width space/non-joiner/joiner, LRM, RLM
    "\u202a\u202b\u202c\u202d\u202e"  # bidi embedding and override
    "\u2060\u2061\u2062\u2063\u2064"  # word joiner, invisible operators
    "\u2066\u2067\u2068\u2069"  # bidi isolates
    "\ufeff"  # BOM / zero-width no-break space
)

_INVISIBLE_RE = re.compile(f"[{re.escape(INVISIBLE_CODEPOINTS)}]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_WS_RE = re.compile(r"[^\S\n]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\n]+")
# Apostrophes are *deleted*, not turned into a separator: "can't" must become
# "cant" so that one written phrase covers the contraction and the elision alike.
_APOSTROPHE_RE = re.compile("['\u2018\u2019\u02bc`\u00b4]")
_SQUEEZE_RE = re.compile(r"[^a-z0-9]+")

# Deliberately conservative: only substitutions that are unambiguous as letter
# stand-ins in the middle of a word. Digits keep their meaning elsewhere, which
# is why de-leeting produces an *additional* form rather than replacing the
# primary one.
_LEET_MAP = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "8": "b",
        "@": "a",
        "$": "s",
        "!": "i",
        "|": "l",
    }
)


def strip_invisibles(text: str) -> str:
    """Drop zero-width and bidi-control characters."""
    return _INVISIBLE_RE.sub("", text)


def has_invisibles(text: str) -> bool:
    """True if the text carries zero-width or bidi-control characters.

    On its own this is a strong obfuscation signal: no clinician types a
    zero-width joiner into a patient utterance.
    """
    return _INVISIBLE_RE.search(text) is not None


def has_control_chars(text: str) -> bool:
    """True if the text carries C0 control characters other than tab/newline."""
    return _CONTROL_RE.search(text) is not None


def normalize_text(text: str) -> str:
    """Canonical, human-readable form. Structure and case are preserved.

    NFKC folds homoglyphs and fullwidth forms onto ASCII; invisibles and stray
    control characters are removed; horizontal whitespace is collapsed. This is
    the form the fencing layer emits and the form structural patterns
    (delimiters, encodings, identifiers) match against.
    """
    out = unicodedata.normalize("NFKC", text)
    out = strip_invisibles(out)
    out = _CONTROL_RE.sub(" ", out)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = _HORIZONTAL_WS_RE.sub(" ", out)
    out = _MULTI_NEWLINE_RE.sub("\n\n", out)
    return out.strip()


def detection_form(text: str) -> str:
    """Lowercased, punctuation-free token stream for phrase matching.

    Runs of three or more identical characters collapse to one, so
    ``"chesttt paiiin"`` matches a ``chest pain`` pattern. Word boundaries
    survive as single spaces; everything else becomes a space.
    """
    out = normalize_text(text).casefold()
    out = _REPEAT_RE.sub(r"\1", out)
    out = _APOSTROPHE_RE.sub("", out)
    out = _NON_ALNUM_RE.sub(" ", out)
    return re.sub(r" +", " ", out).strip()


def deleet(text: str) -> str:
    """Map common digit/symbol letter-substitutions back to letters."""
    return text.translate(_LEET_MAP)


def squeeze(text: str) -> str:
    """All-lowercase, alphanumerics only, no separators at all.

    Defeats letter spacing (``k i l l   m y s e l f``) and separator injection
    (``k.i.l.l-myself``). Matching against this form loses word boundaries, so
    only phrases long enough to be unambiguous should be checked here.
    """
    return _SQUEEZE_RE.sub("", normalize_text(text).casefold())


def detection_forms(text: str) -> tuple[str, ...]:
    """The token-stream forms a phrase detector should try, de-duplicated.

    Returns the plain detection form first and its de-leeted variant second
    when they differ, so a caller that wants the cheapest check can stop early.
    """
    plain = detection_form(text)
    leet = detection_form(deleet(text))
    return (plain,) if plain == leet else (plain, leet)


def squeezed_forms(text: str) -> tuple[str, ...]:
    """The separator-free forms a phrase detector should try, de-duplicated."""
    plain = squeeze(text)
    leet = squeeze(deleet(text))
    return (plain,) if plain == leet else (plain, leet)


def phrase_pattern(phrase: str) -> str:
    """Compile a plain-English phrase into a separator-tolerant regex source.

    ``"chest pain"`` becomes a pattern that also matches ``chest-pain``,
    ``chest   pain`` and ``chest.pain``. Word boundaries are anchored so
    ``"die"`` does not match ``"diet"``.
    """
    words = [re.escape(w) for w in phrase.split()]
    return r"\b" + r"[\s\W_]*".join(words) + r"\b"


def compile_phrases(phrases: tuple[str, ...] | list[str]) -> re.Pattern[str]:
    """One alternation regex over many phrases. Cheaper than a loop of regexes."""
    return re.compile("|".join(phrase_pattern(p) for p in phrases), re.IGNORECASE)


def excerpt(text: str, limit: int = 60) -> str:
    """A short, single-line excerpt safe to put in a `reason` string."""
    flat = normalize_text(text).replace("\n", " ")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
