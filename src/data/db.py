"""
Data access helpers for the project database.

    from src.data.db import query, load
    events = load("events")
    df = query("SELECT bank, COUNT(*) FROM events GROUP BY bank")

The database is opened read-only; it is (re)built automatically if the file is missing.
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data/capstone.duckdb"


def connect():
    if not DB.exists():
        from build_database import build      # noqa: WPS433 (local import keeps the helper light)
        build()
    return duckdb.connect(str(DB), read_only=True)


def query(sql, params=None):
    """Run SQL and return a pandas DataFrame."""
    with connect() as con:
        return con.execute(sql, params or []).df()


def load(table, where=None):
    """Load a whole table or view, optionally filtered, e.g. load("prices", "symbol_current = 'TCS'")."""
    return query(f"SELECT * FROM {table}" + (f" WHERE {where}" if where else ""))


def tables():
    return query("SELECT table_name, table_type FROM information_schema.tables ORDER BY table_type, table_name")
