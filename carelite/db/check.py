"""`make db-check` / `python -m carelite.db.check` — the wave-0 gate."""

from __future__ import annotations

import sys

from carelite.db.connection import check_database

EXPECTED_TABLES = {
    "paper",
    "chunk",
    "kb_entry",
    "kb_entry_source",
    "graph_edge",
    "scenario",
    "prompt_version",
    "generation",
    "retrieval_trace",
    "rubric_score",
    "rating_assignment",
}


def main() -> int:
    report = check_database()

    def line(ok: bool, label: str) -> str:
        return f"  {'PASS' if ok else 'FAIL'}  {label}"

    print("carelite db check")
    print(line(report["connected"], "connection"))
    print(line(report["vector_extension"], "pgvector extension"))

    missing = EXPECTED_TABLES - set(report["tables"])
    print(line(not missing, f"tables ({len(report['tables'])} present)"))
    if missing:
        print(f"        missing: {', '.join(sorted(missing))}")
    print(line(report["join_ok"], "three-way analysis join"))

    for err in report["errors"]:
        print(f"  error: {err}")

    ok = report["connected"] and report["vector_extension"] and not missing and report["join_ok"]
    print("\nOK" if ok else "\nNOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
