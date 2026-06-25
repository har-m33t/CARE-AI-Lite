# CARELite AI
### An Evidence-Based AI Assistant for Patient Communication

*DBMI Summer Internship Project | University of Arizona College of Medicine – Phoenix*
*Department of Biomedical Informatics | June 8 – August 15, 2026*

---

## Project Goal

CARELite AI is being built to address a specific and well-documented problem: clinician communication skills are unevenly developed, unevenly delivered, and unevenly reinforced — and patients pay the difference in adherence, comprehension, satisfaction, and outcomes.

The goal of this project is to design and prototype an AI-assisted communication support system grounded in peer-reviewed evidence. The system is not intended to replace clinical judgment or to script patient encounters. It is intended to do what the human literature has consistently failed to do at scale: provide real-time, specific, low-friction support for the communication behaviors that the research shows actually change outcomes.

This means the system needs to do three things well. It needs to notice when something important is happening in a clinical conversation — an emotional cue, a jargon-heavy explanation, a missed teach-back opportunity. It needs to generate a useful response, prompt, or reframe at the right moment. And it needs to do both of these things without adding cognitive load to an already demanding clinical encounter.

---

## Background

Effective patient communication is not a soft skill. It is a mechanism. The research is clear that how a clinician communicates changes medication adherence, diagnostic costs, patient self-efficacy, readmission rates, and quality of life. It also changes clinician burnout — which in turn degrades communication further, creating a cycle that worsens care over time.

Despite this, most clinicians receive limited communication training after medical school, most training they do receive is brief and does not persist, and most healthcare systems have no infrastructure for reinforcing communication skills at the point of care. The gap between what the evidence recommends and what happens in exam rooms is wide and largely invisible.

CARELite AI is designed to close part of that gap.

---

## Communication Frameworks

The system is built on seven evidence-based communication categories derived from synthesis of approximately fifty peer-reviewed studies. These categories were not selected from existing frameworks — they were identified through pattern analysis across the literature and then validated against the strongest findings in the corpus.

### 1. Empathy — Responsive, Not Just Warm
Empathy is treated here as a behavioral skill, not a personality trait. The evidence shows that high-empathy clinicians use activating, cognitively-oriented communication — not more emotional language. The system supports empathy by detecting missed acknowledgments and generating responses that name emotional states before offering clinical information.

### 2. Emotion Recognition and Response
The most damaging communicative act documented in the literature is blocking a patient's emotional expression — pivoting to information immediately after an emotional cue without acknowledgment. This pattern is measurable, trainable, and falls disproportionately on minority patients. The system flags blocking patterns and generates alternative openings that hold space for emotion before moving forward.

### 3. Patient Activation and Shared Decision-Making
The strongest adherence outcomes in the literature come from plans that were genuinely negotiated around patient goals — not from education, not from information delivery, but from the act of building the plan together. The system prompts goal elicitation before any treatment recommendation is generated and monitors whether patient preferences were actually incorporated.

### 4. Comprehension Confirmation (Teach-Back)
Teach-back is the highest-evidence, lowest-risk communication behavior in the corpus. It works across settings, populations, and health literacy levels, and no study in the reviewed literature found it harmful. The system generates teach-back prompts at every key handoff point and produces re-explanations using different language when a patient's response is incomplete.

### 5. Plain Language and Information Clarity
Telling a patient something is not the same as ensuring they understood it. The system detects clinical jargon, flags responses that exceed three key messages, and generates plain-language alternatives grounded in everyday analogy. The target is understood consent and comprehension, not disclosed information.

### 6. Trust and Relational Continuity
Trust mediates the relationship between communication and outcomes — in several studies, communication's effect on quality of life was only significant through the trust it generated. The system supports trust-building through consistency prompts, transparency about uncertainty, prior-visit callbacks, and active EHR-sharing cues.

### 7. Equity-Aware Communication
Communication quality is not delivered evenly. Low-SES patients experience significantly lower clinician empathy. Minority patients have their emotional expressions blocked more often. Limited-English-proficiency patients receive lower-quality serious illness conversations even when structured guides are used. These are not incidental findings — they are the documented baseline the system is designed to correct, not replicate.

---

## Knowledge Base Structure

Every finding from the literature is stored as a structured entry containing seven fields.

| Field | Description |
|---|---|
| Theme | One of the seven communication categories |
| Source | Full paper citation with DOI |
| Key Finding | Main result in one to two sentences |
| Practical Takeaway | What a clinician should do differently |
| Example Behavior | A specific, observable communication act |
| Evidence Strength | Strong, Moderate, or Emerging |
| AI Action Type | Detection, Generation, or Reframing |

The knowledge base currently contains fifteen sample entries spanning all seven themes. It is designed to be extended, versioned, and refined as the project develops.

---

## Actionable Behavior System

The knowledge base feeds into a master list of thirty-three actionable behaviors organized into three functional categories.

**Detection behaviors** — the system monitors the conversation and flags something the clinician might miss. Examples include detecting emotional blocking patterns, flagging jargon, and identifying conversations that end without a teach-back check.

**Generation behaviors** — the system produces a specific response, prompt, or communication element. Examples include generating teach-back prompts at key handoffs, producing activating follow-up questions after emotional acknowledgments, and generating barriers checks after every treatment plan.

**Reframing behaviors** — the system rewrites or corrects something already said. Examples include replacing patient-blame language with barrier-attribution language, offering plain-language alternatives to jargon-heavy explanations, and generating alternative openings when emotional blocking is detected.

---

## Expected Outcomes

By the end of the internship period, this project is expected to produce the following deliverables.

**A structured knowledge base** containing evidence-based entries across all seven communication themes, each linking a specific research finding to a practical clinical behavior and an AI action type.

**A refined behavior list** of twenty to thirty prioritized, non-overlapping, evaluation-ready communication behaviors derived from the master list and grounded in evidence strength.

**A prototype prompt architecture** demonstrating how the system would detect, generate, and reframe in real clinical conversation contexts — including example inputs, expected outputs, and the evidence rationale behind each.

**An evaluation framework** defining what good looks like for each behavior: what the system should produce, how that output would be assessed, and what the evidence-based standard for success is.

**Project documentation** including this README, a literature synthesis, a communication themes framework, evidence summaries for each theme, and a versioned record of all design decisions made during the internship.

---

## What This Project Is Not

CARELite AI is not a diagnostic tool. It does not make clinical recommendations. It does not replace the clinician's judgment about what a patient needs. It does not generate empathy — it supports the conditions under which a clinician can express it more consistently and more equitably.

The system is also not a script generator. The literature is explicit on this point: communication frameworks that become scripts stop working. The goal is a system that recognizes what kind of moment a clinician is in and supports the right kind of response — not one that tells a clinician exactly what to say.

---

## Project Structure

```
carelite-ai/
│
├── README.md                          # This file
├── literature/
│   ├── Literature_Synthesis.md        # Full synthesis across ~50 papers
│   └── Literature_Notes.pdf           # Annotated source citations
│
├── framework/
│   ├── Communication_Themes_Framework.md     # Seven category definitions
│   └── Evidence_Summaries.md                # Per-theme evidence + examples
│
├── knowledge_base/
│   ├── KB_Structure.md                # Field definitions and design rationale
│   └── KB_Sample_Entries.md           # 15 sample entries across all themes
│
├── behaviors/
│   └── Evidence_Based_Communication_Behaviors.md   # Master list of 33 behaviors
│
└── docs/
    ├── project_notes.md               # Running notebook and reflections
    └── changelog.md                   # Version history and design decisions
```

---

## Literature Corpus

The knowledge base is grounded in approximately fifty peer-reviewed sources including randomized controlled trials, systematic reviews, meta-analyses, observational studies, study protocols, and conceptual pieces. Key anchoring papers include Flickinger et al. (2016) on empathy and medication self-efficacy, Wilson et al. (2010) on shared decision-making and asthma adherence, Yen and Leasure (2019) and Talevski et al. (2020) on teach-back, Roberts et al. (2021) on the SES empathy gap, and Park et al. (2020) on racial disparities in emotion response. Full citations are available in the literature notes file.

---

## Status

| Component | Status |
|---|---|
| Literature review and annotation | Complete |
| Communication themes framework | Complete |
| Evidence summaries | Complete |
| Knowledge base structure | Complete |
| Knowledge base sample entries | Complete |
| Master behavior list | Complete |
| Behavior refinement and prioritization | In progress |
| Prompt architecture prototype | Not started |
| Evaluation framework | Not started |

---

*CARELite AI | University of Arizona College of Medicine – Phoenix | DBMI Summer Internship 2026*