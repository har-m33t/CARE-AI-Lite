# Project decisions

Decisions that the build plan routed to the project owner, recorded here with the
reasoning that produced them. A decision listed here is settled; a lane that
disagrees should raise it rather than re-open it in passing.

Decisions were delegated to the orchestrating session on 2026-08-24. One gate in
the build plan is **not** recorded here because it cannot be delegated: OSF
pre-registration. See "Gates that remain with a person" at the end. The knowledge
base sign-off gate was removed rather than delegated — see D4.

---

## D1 — Theme taxonomy: seven, not ten

**Decision: adopt the seven themes already encoded in `carelite.types.Theme` and
defined in `README.md`. The "10 themes" figure in build plan v3 is retired.**

The `carelite-kb` lane's proposal in `knowledge_base/TAXONOMY.md` is accepted in
full, including its recommendation not to add serious-illness conversation as an
eighth theme. Its central observation is the decisive one: the ten-theme number has
no referent. It appears once, in a sentence arguing that the knowledge base is
already a curated knowledge graph, and there is no list, definition, or derivation
of ten themes anywhere in the repository. The choice was never between two
taxonomies — it was between a taxonomy and a number, and adopting ten would have
meant inventing three themes to satisfy a count.

The loaded knowledge base strengthens the case. At 127 entries over seven themes,
three themes already hold ten entries or fewer (emotion_response 10,
trust_continuity 13, equity 3). `settings.retrieval.rerank_top_n` is 4, so a
theme-filtered retrieval in those themes already returns barely more candidates
than the reranker consumes. Splitting finer would produce facets that cannot
function as filters.

Serious-illness conversation remains the largest coherent cluster in the corpus
without a theme of its own. It stays modelled as `EncounterPhase`, per the lane's
argument that what distinguishes it is where in the encounter and at what stakes,
not which communicative function is being performed.

`carelite/types.py` is unchanged by this decision, which is the point: it confirms
a frozen contract rather than amending one.

## D2 — Equity stratum: two scenarios reclassified, five kept

**Decision: `SC-077` and `SC-010` leave the equity stratum. Their text is
unchanged and they stay in the bank. `SC-029`, `SC-045`, `SC-065`, `SC-088`, and
`SC-090` are accepted into the equity stratum as written.**

This follows the two the `carelite-scenarios` lane nominated itself in
`scenarios/EQUITY_REVIEW.md`, on the lane's own stated criteria.

**`SC-077`** — *"…a man at my church said the same thing happened to his
brother."* Community-sourced health information weighed against a prescription is
real, well documented, and **not specific to any group**. The church detail is what
makes the scenario read as coded for one, and no finding in the corpus ties
congregational information-sharing to the disparity the `racial_ethnic` stratum is
measuring. The communication challenge is genuine and worth keeping; counting it as
evidence of a documented disparity would put a scenario in the stratum that is not
measuring what the stratum claims to measure.

**`SC-010`** — *"Yes, thank you doctor. Everything is good. Maybe my son can call
you with the questions."* This fails the review packet's own rule 2: LEP is to be
signalled situationally, never grammatically. Every other `lep` scenario marks
itself with an event — an interpreter who has left (SC-033), a nephew summarising
(SC-050), a phone interpreter laughed at (SC-086). SC-010 marks itself with
*register*: clipped sentences and deferential phrasing. That is closer to the
grammatical marker the rule forbids than to a situational one, and the rule is
right — politeness is not a nationality. It is also partly redundant with SC-050,
which covers the family-member-handles-the-questions pattern situationally.

Both remain valuable non-equity scenarios: SC-077 as an adherence and trust case,
SC-010 as a false-comprehension and deference case with a low-affect closing cue,
which is a difficulty the bank should test.

On the five kept, briefly. **SC-088** is legible as an equity scenario only to a
reader who knows the under-treatment-of-pain literature — that is acceptable,
because the stratum is an analysis label and does not need to be legible to the
model. **SC-065**'s "people like me" is doing real work: it requests a social norm
as a substitute for a preference, which is the documented lower-participation
pattern rather than a caricature of it. **SC-045**'s SES signal is carried by the
six o'clock shift alone and is the weakest of the five, but neither the mother nor
the adolescent reads as a problem rather than a person under pressure. **SC-029**
is the strongest of the seven: the difficulty is unambiguously in the discharge
material. **SC-090**'s warmth reads as a person.

**Timing matters and this is the free moment.** The bank's holdout split is frozen,
so a change to stratum metadata is a protocol amendment — but no evaluation data
exists yet and OSF pre-registration has not happened. Amending now costs nothing;
the same change after registration would have to be declared. The amended bank is
what gets registered.

Resulting counts: equity stratum 37 → 35, `lep` 11 → 10, `racial_ethnic` 10 → 9.
The `carelite-scenarios` lane re-runs its coverage audit and re-freezes.

## D3 — Equity knowledge base entries: re-extract with a revised prompt

**Decision: approved, and sequenced to run once `carelite-corpus` has landed its
extraction fixes, so it runs once against clean text rather than twice.**

The `carelite-kb` lane recommended this rather than doing it, which was the right
call to escalate. Three equity entries out of 127 understates what the corpus
holds, and the lane established that this is not under-sampling — the extractor read
the right pages, and the Roberts meta-analysis scores 162 on equity vocabulary.

The cause is structural and worth stating precisely, because it is a finding in its
own right. The equity literature *describes* a disparity, so a faithful extraction
of it produces an awareness statement — *"clinicians should be mindful of empathy
gaps in patients from lower socioeconomic backgrounds"* — and the actionability
gate rejects awareness statements correctly, because awareness is not something the
system can detect, generate, or reframe. Six of the nine equity rejections share
that one sentence shape.

The revision instructs the extractor that where a passage reports a disparity, the
entry must name the **compensating move**, not the awareness. The surviving Roberts
entry is the model: *check your assumptions about this patient's adherence and pain
needs*, not *be aware that assumptions exist*.

**The risk is real and must be guarded, not assumed away.** A prompt told to find
compensating moves will find them whether or not the passage supports one. The span
requirement does not catch this, because the span can be perfectly genuine while the
takeaway drifts beyond it. The guard is that the takeaway must be supported by the
quoted span rather than merely adjacent to it, and equity entries from the re-run
get read individually rather than sampled.

This bumps `PROMPT_VERSION` and invalidates the cached extraction windows. That cost
is accepted.

---

## D4 — The knowledge base is not human-verified, and will not claim to be

**Decision (2026-08-24, project owner): drop the human-verification gate. The
provenance claim is amended to what is actually true — "LLM-assisted extraction
with automated verbatim-span validation, no human verification" — and the build
proceeds. `human_verified` stays `FALSE` on all 127 entries as the honest record
of that.**

The gate was blocking, and the alternative to blocking on it was ticking it
falsely, which would have put an untrue sentence in the methods section of a
write-up whose whole argument is that its provenance is checkable. Removing the
claim is the honest resolution; the entries are exactly as good or as bad as they
were, and now the documentation says so.

What the knowledge base *can* still claim is not nothing, and the write-up should
state it precisely rather than retreating to a vague disclaimer:

- Every entry's `verbatim_span` was located in the extracted text of the paper it
  cites, and what is stored is a literal slice of that source rather than the
  model's rendering of it. This was verified against the database, and separately
  spot-checked by re-extracting sampled papers from their original files.
- Entries were rejected for a fabricated span, an unsupported evidence tier, a
  non-actionable takeaway, or a span too short to carry evidence. The genuine
  fabrication rate over all candidates was measured, not estimated.
- No human read the entries for whether the *finding* follows from the *span*.
  That is the specific thing an automated check cannot do and that is now not
  claimed.

The review machinery in `carelite/kb/review.py` is kept rather than deleted. It
cost little, it is tested, and it is what a later reviewer would need. It is now
an available tool rather than a required gate, and the digest should stop
describing itself as something that must be completed.

**This becomes a limitations entry, not a footnote.** `docs/limitations.md` must
record it in the same register as the corpus shortfall: the knowledge base is
machine-extracted and machine-validated, a documented share of candidates were
rejected as fabrications, and the surviving entries carry provenance that is
mechanically checkable but semantically unreviewed. Any result that depends on
knowledge base quality inherits that limitation.

---

## Gates that remain with a person

**OSF pre-registration.** Agents draft it; registration is an account-holder
action and an irreversible public act. It must happen before inference lane III
generates any evaluation data — the argument that an against-you naturalness
result is credible rests entirely on the analysis having been fixed in advance.
