# Project decisions

Decisions that the build plan routed to the project owner, recorded here with the
reasoning that produced them. A decision listed here is settled; a lane that
disagrees should raise it rather than re-open it in passing.

Decisions were delegated to the orchestrating session on 2026-08-24. Two gates in
the build plan are **not** recorded here because they cannot be delegated: the
`human_verified` sign-off on knowledge base entries, and OSF pre-registration. See
"Gates that remain with a person" at the end.

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

## Gates that remain with a person

**The knowledge base `human_verified` sign-off.** This field means a person read the
entry. Setting it any other way would make the provenance claim — "LLM-assisted
extraction, human-verified" — false in a write-up that depends on it. It stays
`FALSE` on all 127 entries until a person ticks it. What *can* be delegated is the
work behind it, and that has been done: see `knowledge_base/review/` and the review
findings routed to the `carelite-kb` lane, which reduce the entries needing a human
judgement call to a much smaller set than 127.

**OSF pre-registration.** Agents draft it; registration is an account-holder action
and an irreversible public act. It must happen before inference lane III generates
any evaluation data — the argument that an against-you naturalness result is
credible rests entirely on the analysis having been fixed in advance.
