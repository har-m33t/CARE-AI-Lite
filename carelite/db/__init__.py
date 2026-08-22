"""Database access. FROZEN INTERFACE — foundation lane only."""

from carelite.db.connection import (
    apply_schema,
    check_database,
    connect,
    fetch_all,
    fetch_one,
    transaction,
)

__all__ = [
    "apply_schema",
    "check_database",
    "connect",
    "fetch_all",
    "fetch_one",
    "transaction",
]
