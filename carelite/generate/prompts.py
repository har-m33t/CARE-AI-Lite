"""The prompt registry: versioned files in `carelite/prompts/`, loaded and pinned.

Every prompt this system sends to a model is a git-tracked file. Nothing in
`carelite.generate` builds system text by concatenating strings at call time,
because a prompt that exists only as a runtime expression cannot be quoted in a
methods section and cannot be recovered from a results table six months later.

Three things this module is responsible for.

**Composition without duplication.** `condition_c.v1` and `condition_lc.v1`
declare `extends: condition_b.v1`, so the framework wording exists once. The
sentence "condition C is condition B plus retrieval" is then a fact about the
files rather than an assertion in a write-up, and the two cannot drift apart in
an edit. `constraints.v1` is appended to all six conditions identically, which
is what keeps the project's safety positions out of the manipulated variable —
including for the degraded control, which is degraded on communication quality
and not on safety.

**Content addressing.** `git_sha` on a `prompt_version` row is the git *blob*
hash of the prompt file's own bytes, not a commit hash and not a hash of the
assembled text. A commit hash records when a prompt was written; a blob hash
records what it said, is computable with no repository present, and resolves
afterwards with `git cat-file -p <sha>`. It has to name the file rather than the
assembly, because the assembly — chain plus body plus constraints — is the
content of no file and is therefore a blob git has never stored; the assembled
text is kept in full in `prompt_version.text` instead. `verify_committed()`
checks that every file contributing to a prompt is in the object database, which
is how "every prompt version is committed" becomes something tested rather than
something claimed.

**Refusing silent edits.** `register()` writes a `prompt_version` row, and if a
row with that `prompt_id` already exists carrying different text it raises
`PromptDriftError` instead of leaving the old text in place. An edited prompt
under an unchanged id is the one mistake that quietly mixes two experiments in
one results table; the fix is a new `.v2` file, and the error says so.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

__all__ = [
    "PROMPTS_DIR",
    "PromptDriftError",
    "PromptError",
    "PromptTemplate",
    "assembled_text",
    "blob_sha",
    "load",
    "load_all",
    "register",
    "registered_rows",
    "verify_committed",
]

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)
_SECTION = re.compile(r"^\[([A-Z_]+)\]\s*$", re.MULTILINE)
_KEY_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")


class PromptError(RuntimeError):
    """A prompt file is missing, malformed, or inconsistent with its filename."""


class PromptDriftError(PromptError):
    """A `prompt_id` already exists in the database with different text.

    Raised rather than updating the row. Silently replacing the text of a
    prompt that has already generated rows in `generation` would leave those
    rows joined to a prompt they were not produced by.
    """


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """One prompt file, parsed. `system`/`task` are this file's own sections.

    Use `assembled_text(template)` — or `PromptTemplate.assemble()` — to get the
    text that is actually sent, which is the `extends` chain plus this file's
    system section plus the shared constraints.
    """

    prompt_id: str
    kind: str
    conditions: tuple[str, ...]
    system: str
    task: str
    extends: str | None = None
    constraints: str | None = None
    description: str = ""
    path: Path | None = field(default=None, compare=False)

    def assemble(self) -> str:
        """The full trusted system text, `extends` chain resolved."""
        return assembled_text(self.prompt_id)

    def git_sha(self) -> str:
        """Blob hash of **this file's bytes**. See the module docstring.

        Not the assembled text: the assembly is `extends` chain + body +
        constraints, which is not the content of any file and therefore is not
        a blob git has ever stored. The assembled text is what
        `prompt_version.text` holds in full; `git_sha` is the pointer back into
        history, so it has to name something history contains.
        """
        if self.path is None:  # pragma: no cover - only for synthesised templates
            raise PromptError(f"{self.prompt_id} has no source file to hash")
        return blob_sha(self.path.read_text(encoding="utf-8"))

    def dependencies(self) -> tuple[str, ...]:
        """This prompt and every file that contributes text to its assembly."""
        chain: list[str] = [self.prompt_id]
        node = self
        while node.extends:
            node = load(node.extends)
            chain.append(node.prompt_id)
        if self.constraints:
            chain.append(self.constraints)
        return tuple(chain)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_front_matter(raw: str) -> dict[str, str]:
    """Single-line `key: value` pairs; an indented line continues the value above.

    Deliberately not YAML. The header holds six string keys, and adding a YAML
    parser to read them would make the prompt format depend on a library whose
    version is another thing to pin.
    """
    fields: dict[str, str] = {}
    key: str | None = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        if line[:1].isspace() and key is not None:
            fields[key] = f"{fields[key]} {line.strip()}".strip()
            continue
        match = _KEY_LINE.match(line)
        if not match:
            raise PromptError(f"unparseable front-matter line: {line!r}")
        key = match.group(1)
        fields[key] = match.group(2).strip()
    return fields


def _parse_sections(body: str) -> dict[str, str]:
    """Split a body on `[SECTION]` markers into a name -> text mapping."""
    parts = _SECTION.split(body)
    if len(parts) < 3:
        raise PromptError("prompt body has no [SECTION] marker")
    sections: dict[str, str] = {}
    for name, text in zip(parts[1::2], parts[2::2], strict=True):
        sections[name] = text.strip()
    return sections


def _parse(path: Path) -> PromptTemplate:
    raw = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(raw)
    if not match:
        raise PromptError(f"{path.name}: missing `---` front matter")
    fields = _parse_front_matter(match.group(1))
    sections = _parse_sections(match.group(2))

    prompt_id = fields.get("prompt_id", "")
    expected = path.name.removesuffix(".md")
    if prompt_id != expected:
        raise PromptError(
            f"{path.name}: prompt_id {prompt_id!r} does not match the filename. "
            "The version lives in both, and they have to agree."
        )
    if "SYSTEM" not in sections:
        raise PromptError(f"{path.name}: no [SYSTEM] section")

    conditions = tuple(c.strip() for c in fields.get("conditions", "").split(",") if c.strip())
    return PromptTemplate(
        prompt_id=prompt_id,
        kind=fields.get("kind", "system"),
        conditions=conditions,
        system=sections["SYSTEM"],
        task=sections.get("TASK", ""),
        extends=fields.get("extends") or None,
        constraints=fields.get("constraints") or None,
        description=fields.get("description", ""),
        path=path,
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_all() -> dict[str, PromptTemplate]:
    """Every prompt file in `carelite/prompts/`, keyed by `prompt_id`."""
    if not PROMPTS_DIR.is_dir():  # pragma: no cover - only if the tree is broken
        raise PromptError(f"prompt directory not found: {PROMPTS_DIR}")
    templates: dict[str, PromptTemplate] = {}
    for path in sorted(PROMPTS_DIR.glob("*.v*.md")):
        template = _parse(path)
        templates[template.prompt_id] = template
    if not templates:  # pragma: no cover - only if the tree is broken
        raise PromptError(f"no prompt files found in {PROMPTS_DIR}")
    return templates


def load(prompt_id: str) -> PromptTemplate:
    templates = load_all()
    try:
        return templates[prompt_id]
    except KeyError:
        raise PromptError(f"unknown prompt_id {prompt_id!r}; known: {sorted(templates)}") from None


@lru_cache(maxsize=32)
def assembled_text(prompt_id: str) -> str:
    """The trusted system text as sent: `extends` chain, body, then constraints.

    The fencing data notice is *not* included here. `fencing.assemble` appends
    it, identically for every condition, so it is a constant of the apparatus
    rather than part of any one prompt's text.
    """
    template = load(prompt_id)
    parts: list[str] = []
    seen: list[str] = []
    chain = template
    while chain.extends:
        if chain.extends in seen:
            raise PromptError(f"circular `extends` chain at {chain.prompt_id!r}")
        seen.append(chain.extends)
        chain = load(chain.extends)
        parts.insert(0, chain.system)
    parts.append(template.system)
    if template.constraints:
        parts.append(load(template.constraints).system)
    return "\n\n".join(p.strip() for p in parts if p.strip())


# ---------------------------------------------------------------------------
# Content addressing
# ---------------------------------------------------------------------------


def blob_sha(text: str) -> str:
    """The git blob hash of `text`, computed without invoking git.

    `git hash-object` over the same bytes returns this string, so a `git_sha`
    recorded on a generation can be resolved back to the exact prompt with
    `git cat-file -p <sha>` as long as the file was committed.
    """
    payload = text.encode("utf-8")
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def verify_committed(prompt_ids: list[str] | None = None) -> dict[str, bool]:
    """Whether every file contributing to each prompt is in the object database.

    A prompt's assembled text is its `extends` chain plus its body plus the
    shared constraints, so checking one blob would leave the other two free to
    be edited without the check noticing. `dependencies()` names all of them and
    every one has to be present.

    `False` means some contributing file has been edited since the last commit
    that contained it, so a result generated now could not be recovered from
    history. The runner refuses to start on a `False` under
    `--require-committed`, which is the enforcement behind "every prompt version
    is committed".

    A missing `git` binary, or a directory that is not a repository, yields
    `False` for everything rather than raising: this is a check, and a check
    that takes the process down when it cannot run is worse than one that
    reports it could not.
    """
    ids = prompt_ids if prompt_ids is not None else sorted(load_all())
    out: dict[str, bool] = {}
    for prompt_id in ids:
        out[prompt_id] = all(
            _blob_present(load(dep).git_sha()) for dep in load(prompt_id).dependencies()
        )
    return out


def _blob_present(sha: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{blob}}"],
            cwd=PROMPTS_DIR,
            capture_output=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return False


# ---------------------------------------------------------------------------
# Database registration
# ---------------------------------------------------------------------------


def registered_rows(prompt_ids: list[str] | None = None) -> list[dict[str, str]]:
    """The `prompt_version` rows for these prompts, as plain dicts.

    Separated from `register()` so the row-building logic is testable without a
    database. `condition` carries every condition that uses the prompt, joined
    by commas: `condition_a.v1` is used by both A and A2 and there is exactly
    one row for it, which is the record of the cross-model baseline sharing a
    prompt rather than merely being said to.
    """
    ids = prompt_ids if prompt_ids is not None else sorted(load_all())
    rows: list[dict[str, str]] = []
    for prompt_id in ids:
        template = load(prompt_id)
        rows.append(
            {
                "prompt_id": prompt_id,
                "condition": ",".join(template.conditions),
                # `text` is the assembled text, exactly as sent. `git_sha` is the
                # blob of the source file, which is what history actually holds.
                "text": assembled_text(prompt_id),
                "git_sha": template.git_sha(),
            }
        )
    return rows


def register(prompt_ids: list[str] | None = None) -> int:
    """Upsert `prompt_version` rows. Returns the number of rows inserted.

    Raises `PromptDriftError` if a row exists under the same `prompt_id` with
    different text. `optimizer` is left NULL: nothing here is DSPy-optimised,
    and sprint 9 will write its own rows under its own ids.
    """
    from carelite.db.connection import transaction

    rows = registered_rows(prompt_ids)
    inserted = 0
    with transaction() as conn:
        for row in rows:
            existing = conn.execute(
                "SELECT text FROM prompt_version WHERE prompt_id = %s", (row["prompt_id"],)
            ).fetchone()
            if existing is not None:
                if existing["text"] != row["text"]:
                    raise PromptDriftError(
                        f"prompt_version {row['prompt_id']!r} is already registered with "
                        "different text. A prompt that has generated data is never edited "
                        "in place — add a new .v2 file instead."
                    )
                continue
            conn.execute(
                "INSERT INTO prompt_version (prompt_id, condition, text, optimizer, git_sha) "
                "VALUES (%s, %s, %s, NULL, %s)",
                (row["prompt_id"], row["condition"], row["text"], row["git_sha"]),
            )
            inserted += 1
    return inserted
