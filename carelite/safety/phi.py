"""PHI / PII detection for terminal input.

Every scenario in this study is synthetic. That is a design commitment, not an
accident, and this module is what keeps it true: the terminal accepts free text,
so nothing structurally prevents an operator from pasting a real note into it.
When that happens the turn must be **warned about and not persisted**.

Policy, stated precisely because the two halves are easy to conflate:

* `allowed=False` — the turn must not proceed *as typed*. `redacted_text` is
  always populated, so a caller may re-run on the redacted form.
* the `phi.do_not_persist` flag — the generation, the utterance, and the trace
  must not be written to Postgres, cached, or logged. Use `may_persist()`
  rather than re-deriving this from `allowed`.

Coverage targets the HIPAA identifier list where a deterministic detector is
honest about its own reliability: structured identifiers (SSN, MRN, phone,
email, account and policy numbers, IP, URL, dates of birth, ages over 89,
street addresses) are matched on shape and on context words. Names are the weak
spot — no regex recognises a name — so this module matches names only where a
context word ("my name is", an honorific, a `Patient:` field label) or a common
first name followed by a capitalised surname makes the intent unambiguous, and
the module documents that residual gap rather than pretending to close it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from carelite.safety.normalize import normalize_text
from carelite.types import SafetyVerdict

#: Written to the database instead of the turn when PHI is found: nothing.
DO_NOT_PERSIST = "phi.do_not_persist"

_MONTHS = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_NUMERIC_DATE = r"\b(?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01])[/\-.](?:19|20)\d{2}\b"
_WORD_DATE = rf"\b{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}\b"

_STREET_TYPES = (
    r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|way|"
    r"circle|cir|place|pl|terrace|ter|parkway|pkwy|highway|hwy|trail|trl|loop)"
)

# A small gazetteer, not a database. Its only job is to make an unambiguous
# "Firstname Lastname" readable as a name; anything it misses is caught by the
# context-word patterns, and anything it over-matches is reported as an FP.
_COMMON_FIRST_NAMES = (
    "james|john|robert|michael|william|david|richard|joseph|thomas|charles|christopher|"
    "daniel|matthew|anthony|donald|mark|paul|steven|andrew|kenneth|george|joshua|kevin|"
    "brian|edward|ronald|timothy|jason|jeffrey|ryan|jacob|gary|nicholas|eric|jonathan|"
    "stephen|larry|justin|scott|brandon|frank|benjamin|gregory|samuel|raymond|patrick|"
    "alexander|jack|dennis|jerry|tyler|aaron|jose|juan|luis|carlos|miguel|antonio|"
    "mary|patricia|jennifer|linda|elizabeth|barbara|susan|jessica|sarah|karen|nancy|"
    "lisa|margaret|betty|sandra|ashley|dorothy|kimberly|emily|donna|michelle|carol|"
    "amanda|melissa|deborah|stephanie|rebecca|laura|sharon|cynthia|kathleen|amy|shirley|"
    "angela|helen|anna|brenda|pamela|nicole|ruth|katherine|samantha|christine|maria|"
    "rosa|carmen|ana|sofia|aisha|fatima|priya|wei|mei|chen|hiroshi|yuki|omar|ahmed"
)


@dataclass(frozen=True)
class Detector:
    kind: str
    pattern: re.Pattern[str]
    #: `False` for shape-only detectors that need a context word nearby;
    #: those are compiled with the context word inside the pattern already.
    description: str = ""


def _c(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    return re.compile(pattern, flags)


DETECTORS: tuple[Detector, ...] = (
    Detector(
        "ssn",
        _c(r"\b\d{3}[-. ]\d{2}[-. ]\d{4}\b"),
        "social security number",
    ),
    Detector(
        "ssn",
        _c(
            r"\b(?:ssn|social\s+security(?:\s+(?:number|no\.?|#))?)\b\s*[:#]?\s*\d{3}[-. ]?\d{2}[-. ]?\d{4}\b"
        ),
        "social security number with context word",
    ),
    Detector(
        "mrn",
        _c(
            r"\b(?:mrn|m\.r\.n\.|medical\s+record(?:\s+(?:number|no\.?|#))?|"
            r"chart\s+(?:number|no\.?|#)|patient\s+(?:id|number|#))\b"
            r"(?:\s*(?:is|was|=|:|#|number|no\.?)){0,3}\s*[A-Za-z]{0,3}[-]?\d{4,12}\b"
        ),
        "medical record number",
    ),
    Detector(
        "account_or_policy_number",
        _c(
            r"\b(?:member|subscriber|policy|group|insurance|account|claim|medicare|medicaid|"
            r"beneficiary)\s*(?:id|number|no\.?|#)?(?:\s*(?:is|was|=|:|#)){0,2}\s*"
            r"[A-Za-z0-9]*\d[A-Za-z0-9-]{4,}\b"
        ),
        "insurance or account identifier",
    ),
    Detector(
        "license_number",
        _c(
            r"\b(?:driver'?s?\s+licen[cs]e|licen[cs]e\s+(?:number|no\.?|#)|"
            r"passport\s*(?:number|no\.?|#)?)"
            r"(?:\s*(?:is|was|=|:|#|number|no\.?)){0,3}\s*[A-Za-z0-9-]*\d[A-Za-z0-9-]{4,}\b"
        ),
        "licence or passport number",
    ),
    Detector(
        "phone",
        _c(r"(?:\+?1[-. ])?\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b"),
        "telephone or fax number",
    ),
    Detector(
        "email",
        _c(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "email address",
    ),
    Detector(
        "url",
        _c(r"\bhttps?://[^\s<>\"]+"),
        "web address",
    ),
    Detector(
        "ip_address",
        _c(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "IP address",
    ),
    Detector(
        "date_of_birth",
        _c(
            rf"(?:d\.?o\.?b\.?|date\s+of\s+birth|birth\s*date|born(?:\s+on)?|birthday)"
            rf"(?:\s*(?:is|was|=|:|#|on)){{0,3}}\s*(?:{_NUMERIC_DATE[2:-2]}|{_WORD_DATE[2:-2]})"
        ),
        "date of birth",
    ),
    Detector("date", _c(_NUMERIC_DATE), "specific calendar date"),
    Detector("date", _c(_WORD_DATE), "specific calendar date"),
    Detector(
        "age_over_89",
        _c(r"\b(?:9\d|1\d{2})\s*(?:-|\s)?\s*(?:years?\s*[-\s]?old|y\.?o\.?\b|yrs?\s+old)"),
        "age over 89 (a HIPAA identifier on its own)",
    ),
    Detector(
        "street_address",
        _c(rf"\b\d{{1,6}}\s+(?:[A-Za-z0-9'.]+\s+){{0,3}}{_STREET_TYPES}\b\.?"),
        "street address",
    ),
    Detector(
        "zip_code",
        _c(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", re.NOFLAG),
        "state and ZIP code",
    ),
    Detector(
        "zip_code",
        _c(r"\bzip\s*(?:code)?\s*[:#=]?\s*\d{5}(?:-\d{4})?\b"),
        "ZIP code",
    ),
    Detector(
        "credit_card",
        _c(r"\b(?:\d{4}[- ]){3}\d{4}\b"),
        "payment card number",
    ),
    Detector(
        "name",
        _c(
            r"\b(?i:mr|mrs|ms|miss|dr|doctor|nurse|prof(?:essor)?)\.?\s+"
            r"[A-Z][a-z]{1,}(?:\s+[A-Z][a-z]+)?",
            re.NOFLAG,
        ),
        "personal name with honorific",
    ),
    # Case matters for the two name detectors below: a real name is capitalised,
    # and matching case-insensitively turns "my name is not important" and
    # "mark down the dose" into false positives.
    Detector(
        "name",
        _c(
            r"\b(?i:my\s+name\s+is|i'?m\s+called|this\s+is|patient(?:'s)?\s+name|"
            r"name\s+of\s+(?:the\s+)?patient|full\s+name)\s*[:,]?\s*"
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+",
            re.NOFLAG,
        ),
        "personal name introduced by context",
    ),
    Detector(
        "name",
        _c(rf"\b(?i:{_COMMON_FIRST_NAMES})\s+[A-Z][a-z]{{2,}}\b", re.NOFLAG),
        "common first name followed by a surname",
    ),
)


@dataclass(frozen=True)
class PHIHit:
    kind: str
    span: str
    description: str


def find_phi(text: str) -> list[PHIHit]:
    """Every PHI hit in `text`, de-duplicated by (kind, span)."""
    normalized = normalize_text(text)
    hits: list[PHIHit] = []
    seen: set[tuple[str, str]] = set()
    for det in DETECTORS:
        for m in det.pattern.finditer(normalized):
            key = (det.kind, m.group(0).casefold())
            if key in seen:
                continue
            seen.add(key)
            hits.append(PHIHit(kind=det.kind, span=m.group(0), description=det.description))
    return hits


def contains_phi(text: str) -> bool:
    """Cheap boolean probe — used by the output gate and the persistence layer."""
    return bool(find_phi(text))


def redact(text: str) -> str:
    """Replace every PHI span with a typed placeholder.

    Placeholders keep the type (`[REDACTED:phone]`) so a redacted utterance is
    still coachable — "my daughter called from [REDACTED:phone]" retains its
    communicative shape — without carrying the identifier.
    """
    out = normalize_text(text)
    for det in DETECTORS:
        out = det.pattern.sub(f"[REDACTED:{det.kind}]", out)
    return out


def screen(text: str) -> SafetyVerdict:
    """Screen terminal input for PHI.

    Returns `allowed=False` with `redacted_text` populated when anything is
    found. The `phi.do_not_persist` flag is the binding one: even if a caller
    chooses to continue on the redacted text, the turn must not be written to
    the database.
    """
    hits = find_phi(text)
    if not hits:
        return SafetyVerdict(allowed=True)

    kinds = sorted({h.kind for h in hits})
    described = sorted({h.description for h in hits})
    return SafetyVerdict(
        allowed=False,
        phi_detected=True,
        flags=[*(f"phi.{k}" for k in kinds), DO_NOT_PERSIST],
        redacted_text=redact(text),
        reason=(
            "This turn appears to contain protected health information ("
            + ", ".join(described)
            + "). CARELite scenarios are synthetic and real patient data must never reach the "
            "database, so this turn was not saved. A redacted version is available if you want "
            "to continue with it."
        ),
    )


def may_persist(verdict: SafetyVerdict) -> bool:
    """Whether a turn carrying this verdict may be written to the database.

    Call this from the persistence path rather than checking `allowed`: an
    injection block and a PHI block both set `allowed=False`, but only the PHI
    case is about storage.
    """
    return DO_NOT_PERSIST not in verdict.flags and not verdict.phi_detected


__all__ = [
    "DETECTORS",
    "DO_NOT_PERSIST",
    "Detector",
    "PHIHit",
    "contains_phi",
    "find_phi",
    "may_persist",
    "redact",
    "screen",
]
