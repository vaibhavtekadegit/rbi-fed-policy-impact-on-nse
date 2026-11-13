import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/data"))
import build_database  # noqa: E402
from db import query  # noqa: E402


def test_build_passes_integrity_checks():
    assert build_database.build()


def test_core_counts():
    assert query("SELECT COUNT(*) AS n FROM events")["n"][0] == 142
    assert query("SELECT COUNT(DISTINCT symbol_current) AS n FROM prices")["n"][0] == 497
    assert query("SELECT COUNT(*) AS n FROM v_event_day0_returns")["n"][0] > 50_000
