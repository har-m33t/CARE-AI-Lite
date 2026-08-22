"""Postgres connection helpers.

Every lane uses these rather than opening its own connections, so connection
settings, the vector adapter registration, and row factory stay consistent.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import DictRow, dict_row

from carelite.config import get_settings

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@contextmanager
def connect(*, autocommit: bool = False) -> Iterator[psycopg.Connection[DictRow]]:
    """Open a connection with pgvector adapters registered."""
    settings = get_settings()
    conn = psycopg.connect(settings.database_url, row_factory=dict_row, autocommit=autocommit)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction() -> Iterator[psycopg.Connection[DictRow]]:
    """Connection wrapped in an explicit transaction; rolls back on exception."""
    with connect() as conn:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def fetch_all(sql: str, params: Sequence[Any] | dict[str, Any] | None = None) -> list[DictRow]:
    with connect() as conn:
        return conn.execute(sql, params).fetchall()


def fetch_one(sql: str, params: Sequence[Any] | dict[str, Any] | None = None) -> DictRow | None:
    with connect() as conn:
        return conn.execute(sql, params).fetchone()


def apply_schema() -> None:
    """Idempotent: schema.sql is written entirely with IF NOT EXISTS."""
    sql = SCHEMA_PATH.read_text()
    with connect(autocommit=True) as conn:
        conn.execute(sql)


def check_database() -> dict[str, Any]:
    """Wave-0 gate: extension present, tables created, a three-way join runs.

    Returns a report rather than raising, so `carelite db check` can print
    something useful when a piece is missing.
    """
    report: dict[str, Any] = {
        "connected": False,
        "vector_extension": False,
        "tables": [],
        "join_ok": False,
        "errors": [],
    }
    try:
        with connect() as conn:
            report["connected"] = True

            ext = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
            report["vector_extension"] = ext is not None

            rows = conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
            ).fetchall()
            report["tables"] = [r["tablename"] for r in rows]

            # The join shape v3 §5 says the whole design exists to support.
            conn.execute(
                """
                SELECT g.condition, AVG(s.respect) AS mean_respect, COUNT(*) AS n
                FROM generation g
                JOIN rubric_score s USING (generation_id)
                JOIN scenario sc USING (scenario_id)
                WHERE sc.split = 'holdout'
                GROUP BY g.condition
                """
            ).fetchall()
            report["join_ok"] = True
    except Exception as exc:  # surfaced to the CLI, not swallowed
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report
