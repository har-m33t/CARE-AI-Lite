"""Study design and evidence tier for each corpus paper, plus paper text loading.

`carelite.corpus.fetch.manifest_papers()` stamps every paper `emerging` with a
placeholder citation and says why: a real tier "is a KB/human review call, not
something the fetch pipeline should assert". That was the right call to defer,
and this is the lane that makes it.

Two rules keep the judgment auditable rather than asserted:

1. **Design is read off the paper; tier is derived from design.** `DESIGN_TIER`
   is the only place a tier is decided, so the mapping can be argued with once
   instead of thirty-three times. Assigning a paper a stronger tier means
   changing its recorded design label, which is a claim about the paper that a
   reviewer can check against its Methods section.
2. **A protocol never outranks its own results.** Study protocols describe a
   design whose results do not exist yet. `s12888-018-1686-y` proposes a cluster
   RCT and `s12888-023-04948-w` a stepped-wedge cluster RCT; both are recorded
   as `study protocol` and land at `emerging`, because an entry cannot cite a
   result that has not been reported.

This module does **not** write to the `paper` table. That table belongs to the
corpus lane, and `paper.design` / `paper.evidence_tier` are still placeholders
in the database. `PAPER_META` is the KB lane's own view, used to validate an
entry's claimed tier against its source; applying it to the `paper` rows is a
one-line change in `carelite.corpus.load` that the corpus or foundation lane
should make, and this lane reports it rather than reaching across the boundary.

Citations are short-form (`Surname et al. (year)`) and derived from the
documents themselves. Where a first author could not be read out of the file,
the field says so instead of guessing — a fabricated citation would be the same
failure this lane exists to prevent, one field over.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from carelite.corpus.extract import extract_source
from carelite.corpus.fetch import manifest_papers
from carelite.types import EvidenceTier

# ---------------------------------------------------------------------------
# Design -> tier. The single place a tier is decided.
# ---------------------------------------------------------------------------

DESIGN_TIER: dict[str, EvidenceTier] = {
    # Synthesis of multiple controlled studies, or a controlled study itself.
    "systematic review": EvidenceTier.STRONG,
    "systematic review and meta-analysis": EvidenceTier.STRONG,
    "integrative systematic review": EvidenceTier.STRONG,
    "meta-analysis": EvidenceTier.STRONG,
    "randomized controlled trial": EvidenceTier.STRONG,
    "cluster randomized controlled trial": EvidenceTier.STRONG,
    # Comparison exists but allocation is not randomised, or the stimulus is
    # controlled without randomisation of participants.
    "non-randomized controlled trial": EvidenceTier.MODERATE,
    "prospective cohort study": EvidenceTier.MODERATE,
    "retrospective cohort study": EvidenceTier.MODERATE,
    "cross-sectional observational study with standardized patients": EvidenceTier.MODERATE,
    "instrument validation study": EvidenceTier.MODERATE,
    # No comparison group, no results yet, or no systematic search.
    "cross-sectional survey": EvidenceTier.EMERGING,
    "mixed-methods evaluation": EvidenceTier.EMERGING,
    "mixed-methods process evaluation": EvidenceTier.EMERGING,
    "qualitative study": EvidenceTier.EMERGING,
    "pre-post single-arm study": EvidenceTier.EMERGING,
    "curriculum development and pilot evaluation": EvidenceTier.EMERGING,
    "narrative review": EvidenceTier.EMERGING,
    "integrative literature review": EvidenceTier.EMERGING,
    "study protocol": EvidenceTier.EMERGING,
    "expert commentary": EvidenceTier.EMERGING,
}


@dataclass(frozen=True)
class PaperMeta:
    paper_id: str
    design: str
    short_citation: str

    @property
    def evidence_tier(self) -> EvidenceTier:
        return DESIGN_TIER[self.design]


def _meta(paper_id: str, design: str, short_citation: str) -> tuple[str, PaperMeta]:
    if design not in DESIGN_TIER:
        raise KeyError(f"{paper_id}: unmapped design {design!r}")
    return paper_id, PaperMeta(paper_id=paper_id, design=design, short_citation=short_citation)


#: One row per fetched document, keyed by `slug(doi)` — the same `paper_id`
#: the `paper` table already carries.
PAPER_META: dict[str, PaperMeta] = dict(
    [
        _meta(
            "10-1370-afm-348",
            "cross-sectional observational study with standardized patients",
            "Epstein et al. (2005), Patient-centered communication and diagnostic testing",
        ),
        _meta(
            "10-1177-08258597241245022",
            "integrative systematic review",
            "Core competencies for serious illness conversations (2024)",
        ),
        _meta(
            "10-1177-2333392819882871",
            "retrospective cohort study",
            "Miller et al. (2019), Provider-patient communication over time",
        ),
        _meta(
            "10-1186-s12885-017-3238-0",
            "randomized controlled trial",
            "Wuensch et al. (2017), Individualized communication skills training in oncology",
        ),
        _meta(
            "10-2147-prbm-s208427",
            "randomized controlled trial",
            "Expert-patient teaching and empathy in nursing students (2019)",
        ),
        _meta(
            "10-1186-s12888-018-1686-y",
            "study protocol",
            "Samalin et al. (2018), ShareD-BD shared decision-making trial protocol",
        ),
        _meta(
            "10-3390-pharmacy6010018",
            "narrative review",
            "Naughton (2018), Patient-centered communication",
        ),
        _meta(
            "10-1136-bmjopen-2018-023666",
            "randomized controlled trial",
            "Alhassan (2019), 2-day communication skills training and student empathy",
        ),
        _meta(
            "10-1371-journal-pone-0231350",
            "systematic review",
            "Talevski et al. (2020), Teach-back: implementation and impacts",
        ),
        _meta(
            "10-3390-healthcare8010026",
            "integrative literature review",
            "Moudatsou et al. (2020), The role of empathy in health and social care",
        ),
        _meta(
            "10-1016-j-jpainsymman-2020-07-022",
            "expert commentary",
            "Horowitz et al. (2020), MVP model for serious illness conversations",
        ),
        _meta(
            "10-1177-2150132720922714",
            "randomized controlled trial",
            "Motivational interviewing for whole-person lifestyle change (2020)",
        ),
        _meta(
            "10-1371-journal-pone-0230672",
            "instrument validation study",
            "Bellier et al. (2020), French Four Habits Coding Scheme",
        ),
        _meta(
            "10-1371-journal-pone-0247259",
            "systematic review and meta-analysis",
            "Roberts et al. (2021), Socioeconomic, racial and ethnic differences in clinician empathy",
        ),
        _meta(
            "10-1136-ijgc-2023-004693",
            "cross-sectional survey",
            "How to break bad news and how to learn this skill (2023)",
        ),
        _meta(
            "10-1186-s12888-023-04948-w",
            "study protocol",
            "Zhu et al. (2023), Patient Oriented Four Habits Model trial protocol",
        ),
        _meta(
            "10-1186-s12909-023-04010-z",
            "mixed-methods evaluation",
            "Yuen et al. (2023), Serious illness communication skills and multidimensional empathy",
        ),
        _meta(
            "10-1177-10732748241236327",
            "cross-sectional survey",
            "Cakmak et al. (2024), Patient-centered communication and patient engagement",
        ),
        _meta(
            "10-1186-s12913-024-11647-z",
            "mixed-methods process evaluation",
            "Hartford Kvael et al. (2024), Communication course in intermediate care",
        ),
        _meta(
            "10-1371-journal-pone-0304180",
            "curriculum development and pilot evaluation",
            "Supporting professionals in serious illness conversations (2024)",
        ),
        _meta(
            "10-3389-fcvm-2024-1457039",
            "meta-analysis",
            "Xu et al. (2025), Motivational interviewing for hypertension management",
        ),
        _meta(
            "10-1016-j-abd-2025-501228",
            "cross-sectional survey",
            "Patient satisfaction as a quality indicator in dermatological care (2025)",
        ),
        _meta(
            "10-1016-j-pecinn-2025-100399",
            "randomized controlled trial",
            "Empathy training via Kalamazoo Consensus, remote and in-person (2025)",
        ),
        _meta(
            "10-1016-j-pecinn-2025-100426",
            "study protocol",
            "Rower et al. (2025), VR-TALKS virtual reality communication training protocol",
        ),
        _meta(
            "10-1016-j-pecinn-2025-100436",
            "curriculum development and pilot evaluation",
            "Longitudinal empathy-focused communication skills training (2025)",
        ),
        _meta(
            "10-1089-pmr-2025-0005",
            "mixed-methods evaluation",
            "Holdsworth et al. (2025), Serious illness communication with limited English proficient patients",
        ),
        _meta(
            "10-1136-bmjopen-2024-091143",
            "qualitative study",
            "Four Habits Model communication course in intermediate care (2025)",
        ),
        _meta(
            "10-1186-s12909-025-06710-0",
            "randomized controlled trial",
            "Lyhne et al. (2025), On-site communication skills training and burnout",
        ),
        _meta(
            "10-1186-s12909-025-07797-1",
            "systematic review",
            "Peimani et al. (2025), Communication skills training in chronic care",
        ),
        _meta(
            "10-1186-s12913-025-13506-x",
            "cross-sectional survey",
            "Altunisik et al. (2025), Patient-centered communication in oncology outpatients",
        ),
        _meta(
            "10-1007-s11606-016-3597-2",
            "non-randomized controlled trial",
            "Boissy et al. (2016), Communication skills training and patient satisfaction",
        ),
        _meta(
            "10-1164-rccm-200906-0907oc",
            "randomized controlled trial",
            "Wilson et al. (2010), Shared treatment decision making in poorly controlled asthma",
        ),
        _meta(
            "10-3389-fphar-2023-1283135",
            "cross-sectional survey",
            "Shared decision making and medication adherence in COPD/asthma (ANANAS)",
        ),
    ]
)


#: Tiers a paper of this design can legitimately support. A `strong` design may
#: still carry a cautious `moderate` claim — a well-powered RCT reporting a
#: secondary outcome, say — so the check is a ceiling, not an equality.
_TIER_ORDER: dict[EvidenceTier, int] = {
    EvidenceTier.EMERGING: 0,
    EvidenceTier.MODERATE: 1,
    EvidenceTier.STRONG: 2,
}


def strongest_tier(tiers: Iterable[EvidenceTier]) -> EvidenceTier | None:
    """The strongest of several tiers, or `None` when there are none.

    A multi-source entry is as strong as its strongest source: two moderate
    studies agreeing do not make a strong finding, but one strong source among
    them does support a strong claim.
    """
    ranked = list(tiers)
    if not ranked:
        return None
    return max(ranked, key=lambda t: _TIER_ORDER[t])


def tier_ceiling(paper_ids: Iterable[str]) -> EvidenceTier | None:
    """Strongest tier the corpus table says these papers support.

    Callers holding `PaperText` objects should read the ceiling off those
    instead (`strongest_tier(p.meta.evidence_tier for p in ...)`). This lookup
    silently ignores unknown paper ids, which is right for a corpus-wide report
    and wrong for validating a single entry: an entry whose paper is missing
    from the table would get no ceiling at all and pass the check by default.
    """
    return strongest_tier(PAPER_META[p].evidence_tier for p in paper_ids if p in PAPER_META)


def tier_at_most(claimed: EvidenceTier, ceiling: EvidenceTier) -> bool:
    return _TIER_ORDER[claimed] <= _TIER_ORDER[ceiling]


# ---------------------------------------------------------------------------
# Paper text
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperText:
    """One paper's cleaned full text, with the identity of that exact text.

    `text_sha256` exists because `carelite.corpus.extract` is another lane's
    file and is actively being improved. A span validated against one
    extraction is not automatically valid against the next, and recording the
    digest makes that drift visible instead of silent: a cached candidate whose
    paper digest no longer matches is re-validated rather than trusted.
    """

    paper_id: str
    source_path: str
    text: str
    text_sha256: str
    meta: PaperMeta | None = None

    @property
    def design(self) -> str | None:
        return self.meta.design if self.meta else None

    @property
    def short_citation(self) -> str:
        return self.meta.short_citation if self.meta else self.paper_id


@lru_cache(maxsize=1)
def load_paper_texts(source_dir: str | None = None) -> dict[str, PaperText]:
    """Every fetched paper's cleaned text, keyed by `paper_id`.

    Joins `manifest_papers()` (DOI -> paper_id -> on-disk path) with
    `extract_source` (path -> cleaned text). Reuses the corpus lane's
    extraction rather than reimplementing it, so the text this lane validates
    spans against is byte-identical to the text that was chunked and embedded.
    """
    out: dict[str, PaperText] = {}
    for paper in manifest_papers(source_dir):
        if not paper.pdf_path:
            continue
        path = Path(paper.pdf_path)
        if not path.exists():
            continue
        extracted = extract_source(path)
        if not extracted.ok:
            continue
        out[paper.paper_id] = PaperText(
            paper_id=paper.paper_id,
            source_path=str(path),
            text=extracted.text,
            text_sha256=hashlib.sha256(extracted.text.encode("utf-8")).hexdigest(),
            meta=PAPER_META.get(paper.paper_id),
        )
    return out


def unmapped_paper_ids(source_dir: str | None = None) -> list[str]:
    """Fetched papers with no `PAPER_META` row — a corpus change this module missed."""
    return sorted(pid for pid in load_paper_texts(source_dir) if pid not in PAPER_META)


__all__ = [
    "DESIGN_TIER",
    "PAPER_META",
    "PaperMeta",
    "PaperText",
    "load_paper_texts",
    "strongest_tier",
    "tier_at_most",
    "tier_ceiling",
    "unmapped_paper_ids",
]
