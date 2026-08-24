# Prompts

Every prompt the system sends is a file in this directory, and every file is
versioned in its own name. A prompt is experimental apparatus: a result produced by
a prompt that exists only in a Python string, or only in the version of the string
that happened to be in the working tree that afternoon, is not reproducible.

## File format

Front matter between `---` lines, then a `[SYSTEM]` section, then an optional
`[TASK]` section. Keys are single-line `key: value`; an indented line continues the
value above it. `carelite.generate.prompts` is the only reader.

| key | meaning |
|---|---|
| `prompt_id` | Primary key in the `prompt_version` table. Must equal the filename without `.md`. |
| `kind` | `system`, `constraints`, or `selfcheck`. |
| `conditions` | Which experimental conditions use this prompt. |
| `extends` | Another `prompt_id` whose `[SYSTEM]` body is prepended to this one. |
| `constraints` | A `kind: constraints` prompt appended to this one. |
| `description` | Prose, for a reader. Not sent to any model. |

`[TASK]` is the trusted one-line instruction that `fencing.assemble` places after
the data blocks, so the last thing the model reads is an instruction from the
trusted channel rather than from the quoted material.

## Versioning

The version lives in the filename and in `prompt_id`: `condition_b.v1`. Editing a
prompt in place after it has generated data is the one thing that silently ruins a
results table, so the rule is that a file whose bytes have been used in a run is
never edited — a change means a new `.v2` file and a new row in `prompt_version`.
`carelite.generate.prompts.register` enforces the near miss: if a `prompt_id`
already exists in the database with different text, it raises rather than writing.

`prompt_version.git_sha` holds the **git blob hash** of the assembled prompt text,
not a commit hash. A commit hash tells you when a prompt was written; a blob hash
tells you what it said, is computable without a repository, and is verifiable
afterwards with `git cat-file -p <sha>`. Reproducibility wants the second one.
`prompts.verify_committed()` confirms the blob is actually in the object database,
which is how "every prompt version is committed" gets checked rather than asserted.

## What is shared, and why that matters

- **`constraints.v1` is attached to all six conditions, identically.** The project
  positions it carries — not a diagnostic tool, no clinical recommendations — are
  properties of CARELite rather than of the manipulation, so they are held constant
  across conditions. That includes the degraded control: **D is degraded on the
  communication dimensions the rubric scores, not on safety.**
- **C and LC `extends: condition_b.v1`.** The claim "C is B plus retrieval" is then
  a fact about the files. There is one copy of the framework wording and the three
  conditions cannot drift apart.
- **A and A2 share `condition_a.v1`.** A2 is condition A on a second model family,
  so it is the same prompt row in `prompt_version`; the generation rows differ only
  in `condition`, `model` and `model_digest`. `tests/unit/generate` asserts the
  shared `prompt_id`, which is the mechanical form of "the cross-model baseline
  varies only the model".
- **The fencing data notice is appended to every system prompt** by
  `carelite.safety.fencing.assemble`, in all six conditions. It is part of the
  constant, not part of the manipulation.

## What none of these prompts claim

Per `DECISIONS.md` D4 the knowledge base is machine-extracted and
machine-validated, with no human verification of whether each finding follows from
its span. No prompt here describes the evidence base as verified, reviewed, or
clinician-approved, and none should be edited to.
