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

**This module writes `paper.design`, `paper.evidence_tier`, `paper.year` and
`paper.apa_citation`** — see `sync_paper_metadata`. It did not, originally, on
the argument that the `paper` table belongs to the corpus lane. That argument
produced a knowledge base where the design was known in Python and unknown in
Postgres: all 33 rows sat at `design IS NULL`, `evidence_tier = 'emerging'` (the
fetch pipeline's deliberate placeholder) and `apa_citation = '[citation
pending] DOI: …'`. The digest printed "randomized controlled trial" next to an
entry while the CLI's evidence panel, reading the same paper from the database,
would have shown a clinician "[citation pending]". Every consumer that reads the
table rather than importing this module — the graph lane, the stats lane, the
CLI — saw placeholders. Deferring the write did not keep the boundary clean; it
just moved the judgment somewhere nothing could reach. The tier derivation in
`validate.py` now depends on the design being in the table besides.

Citations are **real**, in two forms. `short_citation` is the reader-facing
`Surname et al. (year), what the paper is` used in the review digest;
`apa_citation` is a full reference and is what lands in the database. Both are
built from Crossref's record for the DOI, fetched once and frozen into the table
above rather than resolved at runtime, so a cold rebuild needs no network and
the values cannot drift under the write-up. `python -m carelite.kb.papers
--refresh-citations` re-derives the block from Crossref for pasting back, which
is how it was generated. Nothing here is hand-composed: a citation invented to
fill a NOT NULL column would be the same failure this lane exists to prevent,
one field over. Article titles keep their published capitalisation rather than
being forced to APA sentence case, because the sentence-casing rule cannot be
applied mechanically without flattening proper nouns (Kalamazoo, Four Habits,
English, SPIKES).
"""

# ruff: noqa: RUF001 - published titles and author names keep the typography the
# journal printed. Replacing an en dash or a curly apostrophe to satisfy a linter
# would make the citation a slightly different string from the real reference.

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from carelite.corpus.extract import extract_source
from carelite.corpus.fetch import manifest_papers
from carelite.db.connection import transaction
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
    year: int
    apa_citation: str

    @property
    def evidence_tier(self) -> EvidenceTier:
        return DESIGN_TIER[self.design]


def _meta(
    paper_id: str, design: str, short_citation: str, year: int, apa_citation: str
) -> tuple[str, PaperMeta]:
    if design not in DESIGN_TIER:
        raise KeyError(f"{paper_id}: unmapped design {design!r}")
    return paper_id, PaperMeta(
        paper_id=paper_id,
        design=design,
        short_citation=short_citation,
        year=year,
        apa_citation=apa_citation,
    )


#: One row per fetched document, keyed by `slug(doi)` — the same `paper_id`
#: the `paper` table already carries.
PAPER_META: dict[str, PaperMeta] = dict(
    [
        _meta(
            "10-1370-afm-348",
            "cross-sectional observational study with standardized patients",
            "Epstein (2005), Patient-centered communication and diagnostic testing",
            2005,
            "Epstein, R. M. (2005). Patient-Centered Communication and Diagnostic Testing. The "
            "Annals of Family Medicine, 3(5), 415-421. https://doi.org/10.1370/afm.348",
        ),
        _meta(
            "10-1177-08258597241245022",
            "integrative systematic review",
            "Pusa et al. (2024), Core competencies for serious illness conversations",
            2024,
            "Pusa, S., Baxter, R., Andersson, S., Fromme, E. K., Paladino, J., & Sandgren, A. "
            "(2024). Core Competencies for Serious Illness Conversations: An Integrative "
            "Systematic Review. Journal of Palliative Care, 39(4), 340-351. "
            "https://doi.org/10.1177/08258597241245022",
        ),
        _meta(
            "10-1177-2333392819882871",
            "retrospective cohort study",
            "Miller & Peck (2019), Provider-patient communication over time",
            2019,
            "Miller, L. R., & Peck, B. M. (2019). Patient-Centered Care: An Examination of "
            "Provider–Patient Communication Over Time. Health Services Research and Managerial "
            "Epidemiology, 6. https://doi.org/10.1177/2333392819882871",
        ),
        _meta(
            "10-1186-s12885-017-3238-0",
            "randomized controlled trial",
            "Wuensch et al. (2017), Individualized communication skills training in oncology",
            2017,
            "Wuensch, A., Goelz, T., Ihorst, G., Terris, D. D., Bertz, H., Bengel, J., Wirsching, "
            "M., & Fritzsche, K. (2017). Effect of individualized communication skills training "
            "on physicians’ discussion of clinical trials in oncology: results from a randomized "
            "controlled trial. BMC Cancer, 17(1). https://doi.org/10.1186/s12885-017-3238-0",
        ),
        _meta(
            "10-2147-prbm-s208427",
            "randomized controlled trial",
            "Ferri et al. (2019), Expert-patient teaching and empathy in nursing students",
            2019,
            "Ferri, P., Rovesti, S., Padula, M. S., D'Amico, R., & Di Lorenzo, R. (2019). Effect "
            "of expert-patient teaching on empathy in nursing students: a randomized controlled "
            "trial. Psychology Research and Behavior Management, 12, 457-467. "
            "https://doi.org/10.2147/prbm.s208427",
        ),
        _meta(
            "10-1186-s12888-018-1686-y",
            "study protocol",
            "Samalin et al. (2018), ShareD-BD shared decision-making trial protocol",
            2018,
            "Samalin, L., Honciuc, M., Boyer, L., de Chazeron, I., Blanc, O., Abbar, M., & "
            "Llorca, P. M. (2018). Efficacy of shared decision-making on treatment adherence of "
            "patients with bipolar disorder: a cluster randomized trial (ShareD-BD). BMC "
            "Psychiatry, 18(1). https://doi.org/10.1186/s12888-018-1686-y",
        ),
        _meta(
            "10-3390-pharmacy6010018",
            "narrative review",
            "Naughton (2018), Patient-centered communication",
            2018,
            "Naughton, C. (2018). Patient-Centered Communication. Pharmacy, 6(1), 18. "
            "https://doi.org/10.3390/pharmacy6010018",
        ),
        _meta(
            "10-1136-bmjopen-2018-023666",
            "randomized controlled trial",
            "Alhassan (2019), 2-day communication skills training and student empathy",
            2019,
            "Alhassan, M. (2019). Effect of a 2-day communication skills training on nursing and "
            "midwifery students’ empathy: a randomised controlled trial. BMJ Open, 9(3), e023666. "
            "https://doi.org/10.1136/bmjopen-2018-023666",
        ),
        _meta(
            "10-1371-journal-pone-0231350",
            "systematic review",
            "Talevski et al. (2020), Teach-back: implementation and impacts",
            2020,
            "Talevski, J., Wong Shee, A., Rasmussen, B., Kemp, G., & Beauchamp, A. (2020). "
            "Teach-back: A systematic review of implementation and impacts. PLOS ONE, 15(4), "
            "e0231350. https://doi.org/10.1371/journal.pone.0231350",
        ),
        _meta(
            "10-3390-healthcare8010026",
            "integrative literature review",
            "Moudatsou et al. (2020), The role of empathy in health and social care",
            2020,
            "Moudatsou, M., Stavropoulou, A., Philalithis, A., & Koukouli, S. (2020). The Role of "
            "Empathy in Health and Social Care Professionals. Healthcare, 8(1), 26. "
            "https://doi.org/10.3390/healthcare8010026",
        ),
        _meta(
            "10-1016-j-jpainsymman-2020-07-022",
            "expert commentary",
            "Horowitz et al. (2020), MVP model for serious illness conversations",
            2020,
            "Horowitz, R. K., Hogan, L. A., & Carroll, T. (2020). MVP–Medical Situation, Values, "
            "and Plan: A Memorable and Useful Model for All Serious Illness Conversations. "
            "Journal of Pain and Symptom Management, 60(5), 1059-1065. "
            "https://doi.org/10.1016/j.jpainsymman.2020.07.022",
        ),
        _meta(
            "10-1177-2150132720922714",
            "randomized controlled trial",
            "Sawyer et al. (2020), Motivational interviewing for whole-person lifestyle change",
            2020,
            "Sawyer, A. T., Wheeler, J., Jennelle, P., Pepe, J., & Robinson, P. S. (2020). A "
            "Randomized Controlled Trial of a Motivational Interviewing Intervention to Improve "
            "Whole-Person Lifestyle. Journal of Primary Care & Community Health, 11. "
            "https://doi.org/10.1177/2150132720922714",
        ),
        _meta(
            "10-1371-journal-pone-0230672",
            "instrument validation study",
            "Bellier et al. (2020), French Four Habits Coding Scheme",
            2020,
            "Bellier, A., Chaffanjon, P., Krupat, E., Francois, P., & Labarère, J. (2020). "
            "Cross-cultural adaptation of the 4-Habits Coding Scheme into French to assess "
            "physician communication skills. PLOS ONE, 15(4), e0230672. "
            "https://doi.org/10.1371/journal.pone.0230672",
        ),
        _meta(
            "10-1371-journal-pone-0247259",
            "systematic review and meta-analysis",
            "Roberts et al. (2021), Socioeconomic, racial and ethnic differences in clinician empathy",
            2021,
            "Roberts, B. W., Puri, N. K., Trzeciak, C. J., Mazzarelli, A. J., & Trzeciak, S. "
            "(2021). Socioeconomic, racial and ethnic differences in patient experience of "
            "clinician empathy: Results of a systematic review and meta-analysis. PLOS ONE, "
            "16(3), e0247259. https://doi.org/10.1371/journal.pone.0247259",
        ),
        _meta(
            "10-1136-ijgc-2023-004693",
            "cross-sectional survey",
            "Herzog et al. (2023), How to break bad news and how to learn this skill",
            2023,
            "Herzog, E. M., Pirmorady Sehouli, A., Boer, J., Pietzner, K., Petru, E., "
            "Heinzelmann, V., Roser, E., Dimitrova, D., Oskay-Özcelik, G., Camara, O., & Sehouli, "
            "J. (2023). How to break bad news and how to learn this skill: results from an "
            "international North-Eastern German Society for Gynecological Oncology (NOGGO) survey "
            "among physicians and medical students with 1089 participants. International Journal "
            "of Gynecological Cancer, 33(12), 1934-1942. https://doi.org/10.1136/ijgc-2023-004693",
        ),
        _meta(
            "10-1186-s12888-023-04948-w",
            "study protocol",
            "Zhu et al. (2023), Patient Oriented Four Habits Model trial protocol",
            2023,
            "Zhu, Y., Li, S., Zhang, R., Bao, L., Zhang, J., Xiao, X., Jiang, D., Chen, W., Hu, "
            "C., Zou, C., Zhang, J., Zhu, Y., Wang, J., Liang, J., & Yang, Q. (2023). Enhancing "
            "doctor-patient relationships in community health care institutions: the Patient "
            "Oriented Four Habits Model (POFHM) trial—a stepped wedge cluster randomized trial "
            "protocol. BMC Psychiatry, 23(1). https://doi.org/10.1186/s12888-023-04948-w",
        ),
        _meta(
            "10-1186-s12909-023-04010-z",
            "mixed-methods evaluation",
            "Yuen et al. (2023), Serious illness communication skills and multidimensional empathy",
            2023,
            "Yuen, J. K., See, C., Cheung, J. T. K., Lum, C. M., Lee, J. S., & Wong, W. T. "
            "(2023). Can teaching serious illness communication skills foster multidimensional "
            "empathy? A mixed-methods study. BMC Medical Education, 23(1). "
            "https://doi.org/10.1186/s12909-023-04010-z",
        ),
        _meta(
            "10-1177-10732748241236327",
            "cross-sectional survey",
            "Çakmak & Uğurluoğlu (2024), Patient-centered communication and patient engagement",
            2024,
            "Çakmak, C., & Uğurluoğlu, Ö. (2024). The Effects of Patient-Centered Communication "
            "on Patient Engagement, Health-Related Quality of Life, Service Quality Perception "
            "and Patient Satisfaction in Patients with Cancer: A Cross-Sectional Study in "
            "Türkiye. Cancer Control, 31. https://doi.org/10.1177/10732748241236327",
        ),
        _meta(
            "10-1186-s12913-024-11647-z",
            "mixed-methods process evaluation",
            "Kvæl et al. (2024), Communication course in intermediate care",
            2024,
            "Kvæl, L. A. H., Gulbrandsen, P., Werner, A., & Bergland, A. (2024). Implementation "
            "of the four habits model in intermediate care services in Norway: a process "
            "evaluation. BMC Health Services Research, 24(1). "
            "https://doi.org/10.1186/s12913-024-11647-z",
        ),
        _meta(
            "10-1371-journal-pone-0304180",
            "curriculum development and pilot evaluation",
            "Gonella et al. (2024), Supporting professionals in serious illness conversations",
            2024,
            "Gonella, S., Di Giulio, P., Riva-Rovedda, F., Stella, L., Rivolta, M. M., "
            "Malinverni, E., Paleologo, M., Di Vella, G., & Dimonte, V. (2024). Supporting health "
            "and social care professionals in serious illness conversations: Development, "
            "validation, and preliminary evaluation of an educational booklet. PLOS ONE, 19(5), "
            "e0304180. https://doi.org/10.1371/journal.pone.0304180",
        ),
        _meta(
            "10-3389-fcvm-2024-1457039",
            "meta-analysis",
            "Xu et al. (2025), Motivational interviewing for hypertension management",
            2025,
            "Xu, J., Gu, X., Gu, J., Zhao, L., Li, M., & Hong, C. (2025). Motivational "
            "interviewing intervention for the management of hypertension: a meta-analysis. "
            "Frontiers in Cardiovascular Medicine, 11. https://doi.org/10.3389/fcvm.2024.1457039",
        ),
        _meta(
            "10-1016-j-abd-2025-501228",
            "cross-sectional survey",
            "Gahona et al. (2025), Patient satisfaction as a quality indicator in dermatological care",
            2025,
            "Gahona, M., Prada, S. A., & Chaparro, D. (2025). Patient satisfaction as a quality "
            "indicator in dermatological care: cross-sectional study in two tertiary institutions "
            "with residency programs. Anais Brasileiros de Dermatologia, 100(6), 501228. "
            "https://doi.org/10.1016/j.abd.2025.501228",
        ),
        _meta(
            "10-1016-j-pecinn-2025-100399",
            "randomized controlled trial",
            "Previti et al. (2025), Empathy training via Kalamazoo Consensus, remote and in-person",
            2025,
            "Previti, G. B., Mazzatenta, C., Bellandi, T., Niccolai, F., Nieri, D., Ungaretti, "
            "V., Cavasini, I., Mazzoni, A., Maiorano, S., Di Paolo, L., D'Elia, V., Torre, M., "
            "Matteucci, L., Miccinesi, G., Marcucci, M., Maielli, M., & Ardis, S. (2025). Empathy "
            "training via Kalamazoo Consensus in remote and in-person medical communication: A "
            "randomized controlled trial. PEC Innovation, 6, 100399. "
            "https://doi.org/10.1016/j.pecinn.2025.100399",
        ),
        _meta(
            "10-1016-j-pecinn-2025-100426",
            "study protocol",
            "Röwer et al. (2025), VR-TALKS virtual reality communication training protocol",
            2025,
            "Röwer, H. A. A., Skvortsova, A., Saab, M. M., Hartigan, I., Bausewein, C., Pereira, "
            "S. M., Hernández-Marrero, P., Hrdlička, J., Rusinová, K., Loučka, M., Hrdličková, "
            "L., Zielina, M., Payne, C., Van Vliet, L. M., Klemmt, M., Afshar, K., & Stiel, S. "
            "(2025). Innovations in communication training for medical and nursing students: "
            "Virtual reality communication tool for application and evaluation with key "
            "stakeholders and students (VR-TALKS) – a study protocol. PEC Innovation, 7, 100426. "
            "https://doi.org/10.1016/j.pecinn.2025.100426",
        ),
        _meta(
            "10-1016-j-pecinn-2025-100436",
            "curriculum development and pilot evaluation",
            "Ward et al. (2025), Longitudinal empathy-focused communication skills training",
            2025,
            "Ward, A., Gilligan, C., Bennett-Weston, A., & Howick, J. (2025). The development, "
            "delivery, and evaluation of novel longitudinal empathy-focused communication skills "
            "training at a UK medical school. PEC Innovation, 7, 100436. "
            "https://doi.org/10.1016/j.pecinn.2025.100436",
        ),
        _meta(
            "10-1089-pmr-2025-0005",
            "mixed-methods evaluation",
            "Holdsworth et al. (2025), Serious illness communication with limited English proficient patients",
            2025,
            "Holdsworth, L. M., Kling, S. M. R., Winget, M., Mui, H. Z., Garvert, D. W., "
            "Veruttipong, D., Seevaratnam, B., Harris, S., & Teuteberg, W. (2025). Quality of "
            "Serious Illness Communication with Hospitalized Limited English Proficient Patients: "
            "A Mixed Methods Study. Palliative Medicine Reports, 6(1). "
            "https://doi.org/10.1089/pmr.2025.0005",
        ),
        _meta(
            "10-1136-bmjopen-2024-091143",
            "qualitative study",
            "Kvæl et al. (2025), Four Habits Model communication course in intermediate care",
            2025,
            "Kvæl, L. A. H., Bye, A., Bergland, A., & Fromholt Olsen, C. (2025). Healthcare "
            "professionals' experiences of the Four Habits Model communication course: a "
            "qualitative and survey approach to evaluate impact in an intermediate care setting. "
            "BMJ Open, 15(3), e091143. https://doi.org/10.1136/bmjopen-2024-091143",
        ),
        _meta(
            "10-1186-s12909-025-06710-0",
            "randomized controlled trial",
            "Antonsen et al. (2025), On-site communication skills training and burnout",
            2025,
            "Antonsen, K. K., Lyhne, J. D., Johnsen, A. T., Eßer-Naumann, S., Poulsen, L. Ø., "
            "Lund, L., Timm, S., & Jensen, L. H. (2025). Assessing the effect of On-site "
            "supportive communication training (On-site SCT) on doctor burnout: a randomized "
            "controlled trial. BMC Medical Education, 25(1). "
            "https://doi.org/10.1186/s12909-025-06710-0",
        ),
        _meta(
            "10-1186-s12909-025-07797-1",
            "systematic review",
            "Peimani et al. (2025), Communication skills training in chronic care",
            2025,
            "Peimani, M., Tanhapour, M., Majlesi, M., & Nasli-Esfahani, E. (2025). Training in "
            "communication skills for healthcare providers in chronic care: a systematic review. "
            "BMC Medical Education, 25(1). https://doi.org/10.1186/s12909-025-07797-1",
        ),
        _meta(
            "10-1186-s12913-025-13506-x",
            "cross-sectional survey",
            "Eskiler et al. (2025), Patient-centered communication in oncology outpatients",
            2025,
            "Eskiler, E., Bilir, C., & Altunisik, R. (2025). Exploring patient centered "
            "communication on health outcomes: Role of trust and online information seeking in "
            "Turkish cancer patients receiving chemotherapy. BMC Health Services Research, 25(1). "
            "https://doi.org/10.1186/s12913-025-13506-x",
        ),
        _meta(
            "10-1007-s11606-016-3597-2",
            "non-randomized controlled trial",
            "Boissy et al. (2016), Communication skills training and patient satisfaction",
            2016,
            "Boissy, A., Windover, A. K., Bokar, D., Karafa, M., Neuendorf, K., Frankel, R. M., "
            "Merlino, J., & Rothberg, M. B. (2016). Communication Skills Training for Physicians "
            "Improves Patient Satisfaction. Journal of General Internal Medicine, 31(7), 755-761. "
            "https://doi.org/10.1007/s11606-016-3597-2",
        ),
        _meta(
            "10-1164-rccm-200906-0907oc",
            "randomized controlled trial",
            "Wilson et al. (2010), Shared treatment decision making in poorly controlled asthma",
            2010,
            "Wilson, S. R., Strub, P., Buist, A. S., Knowles, S. B., Lavori, P. W., Lapidus, J., "
            "& Vollmer, W. M. (2010). Shared Treatment Decision Making Improves Adherence and "
            "Outcomes in Poorly Controlled Asthma. American Journal of Respiratory and Critical "
            "Care Medicine, 181(6), 566-577. https://doi.org/10.1164/rccm.200906-0907oc",
        ),
        _meta(
            "10-3389-fphar-2023-1283135",
            "cross-sectional survey",
            "Achterbosch et al. (2023), Shared decision making and medication adherence in COPD/asthma (ANANAS)",
            2023,
            "Achterbosch, M., Vart, P., van Dijk, L., & van Boven, J. F. M. (2023). Shared "
            "decision making and medication adherence in patients with COPD and/or asthma: the "
            "ANANAS study. Frontiers in Pharmacology, 14. "
            "https://doi.org/10.3389/fphar.2023.1283135",
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


# ---------------------------------------------------------------------------
# Persisting the judgment to `paper`
# ---------------------------------------------------------------------------

_SYNC_SQL = """
UPDATE paper
   SET design        = %(design)s,
       evidence_tier = %(evidence_tier)s,
       apa_citation  = %(apa_citation)s,
       year          = %(year)s
 WHERE paper_id = %(paper_id)s
"""

_PLACEHOLDER_SQL = """
SELECT count(*) FILTER (WHERE design IS NULL)                   AS null_design,
       count(*) FILTER (WHERE apa_citation LIKE '[citation pending]%%') AS pending_citation,
       count(*)                                                  AS n_rows
FROM paper
"""


@dataclass(frozen=True)
class SyncResult:
    rows_updated: int
    unknown_in_db: tuple[str, ...]
    """`PAPER_META` ids with no `paper` row — this module and the corpus disagree."""

    def __str__(self) -> str:
        out = f"{self.rows_updated} paper row(s) updated with design, tier, citation and year."
        if self.unknown_in_db:
            out += (
                f" {len(self.unknown_in_db)} PAPER_META id(s) have no row in `paper`: "
                f"{', '.join(self.unknown_in_db[:5])}"
            )
        return out


def sync_paper_metadata() -> SyncResult:
    """Write this module's design, tier, citation and year onto the `paper` rows.

    An UPDATE rather than an upsert on purpose. Creating a `paper` row is the
    corpus lane's job and depends on things this module does not know — the PDF
    path, the open-access licence, whether the file was actually fetched. If a
    paper is missing from the table, that is a fetch problem to report, not one
    to paper over by inventing a row from a citation.
    """
    with transaction() as conn:
        present = {row["paper_id"] for row in conn.execute("SELECT paper_id FROM paper").fetchall()}
        updated = 0
        for paper_id, meta in PAPER_META.items():
            if paper_id not in present:
                continue
            conn.execute(
                _SYNC_SQL,
                {
                    "paper_id": paper_id,
                    "design": meta.design,
                    "evidence_tier": meta.evidence_tier.value,
                    "apa_citation": meta.apa_citation,
                    "year": meta.year,
                },
            )
            updated += 1
    missing = tuple(sorted(p for p in PAPER_META if p not in present))
    return SyncResult(rows_updated=updated, unknown_in_db=missing)


def placeholder_counts() -> dict[str, int]:
    """How many `paper` rows still carry the fetch pipeline's placeholders."""
    with transaction() as conn:
        row = conn.execute(_PLACEHOLDER_SQL).fetchone()
    return dict(row) if row else {}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Paper design, tier and citation.")
    ap.add_argument(
        "--sync",
        action="store_true",
        help="write design, evidence_tier, apa_citation and year onto the `paper` rows",
    )
    ap.add_argument(
        "--refresh-citations",
        action="store_true",
        help="re-derive the PAPER_META citation fields from Crossref and print them",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.refresh_citations:
        for paper_id in sorted(PAPER_META):
            print(f"{paper_id}\t{crossref_apa(PAPER_META[paper_id])}")
        return 0

    if args.sync:
        print(sync_paper_metadata())
        print(f"  remaining placeholders: {placeholder_counts()}")
        return 0

    print(f"{len(PAPER_META)} papers with a recorded design.")
    for design in sorted({m.design for m in PAPER_META.values()}):
        n = sum(1 for m in PAPER_META.values() if m.design == design)
        print(f"  {n:3d}  {design} -> {DESIGN_TIER[design].value}")
    return 0


def crossref_apa(meta: PaperMeta, *, timeout: float = 25.0) -> str:
    """Re-derive one APA reference from Crossref. Used only by `--refresh-citations`.

    Kept out of every import path that matters: the frozen table above is what
    the pipeline reads, and this exists so a reader can check that the table was
    derived rather than composed.
    """
    import html
    import json
    import re as _re
    import urllib.parse
    import urllib.request

    doi = meta.apa_citation.rsplit("https://doi.org/", 1)[-1]
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    request = urllib.request.Request(url, headers={"User-Agent": "CARELite/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        message = json.load(response)["message"]

    def clean(text: str) -> str:
        return _re.sub(r"\s+", " ", _re.sub(r"<[^>]+>", "", html.unescape(text or ""))).strip()

    names = [
        f"{clean(a.get('family', ''))}, "
        + " ".join(p[0].upper() + "." for p in _re.split(r"[\s\-.]+", a.get("given", "")) if p)
        for a in message.get("author", [])
    ]
    if len(names) > 1:
        authors = ", ".join(names[:-1]) + ", & " + names[-1]
    else:
        authors = names[0] if names else ""
    year = (message.get("issued", {}).get("date-parts") or [[None]])[0][0]
    title = clean((message.get("title") or [""])[0]).rstrip(".")
    location = clean((message.get("container-title") or [""])[0])
    if volume := clean(str(message.get("volume") or "")).replace("Volume ", ""):
        location += f", {volume}"
        if issue := clean(str(message.get("issue") or "")):
            location += f"({issue})"
    if pages := clean(str(message.get("page") or "")):
        location += f", {pages}"
    return f"{authors} ({year}). {title}. {location}. https://doi.org/{message.get('DOI')}"


__all__ = [
    "DESIGN_TIER",
    "PAPER_META",
    "PaperMeta",
    "PaperText",
    "SyncResult",
    "crossref_apa",
    "load_paper_texts",
    "placeholder_counts",
    "strongest_tier",
    "sync_paper_metadata",
    "tier_at_most",
    "tier_ceiling",
    "unmapped_paper_ids",
]


if __name__ == "__main__":
    raise SystemExit(main())
