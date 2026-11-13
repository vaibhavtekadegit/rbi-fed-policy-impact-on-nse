"""
Build the project's analytical database (DuckDB, a single local file; no server, nothing exposed).

Loads every curated table from the repository into data/capstone.duckdb, adds primary-key and
range checks, creates analysis views, and writes the schema documentation data/db_schema.md.
The database file is rebuilt from the CSV sources in a few seconds, so it is not committed.

Run from the repository root:
    python src/data/build_database.py
Then query it with src/data/db.py (see `query()` and `load()`).
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data/capstone.duckdb"
DOC = ROOT / "data/db_schema.md"

# table name: (source file, primary key, description)
TABLES = {
    "events": ("data/interim/events.csv", ["event_id"],
               "142 RBI and Fed announcements: release time (IST), NSE day 0, decision, rate change, surprise, overlap flags"),
    "stocks": ("data/raw/prices/validation_report.csv", ["symbol"],
               "501 NIFTY 500 constituents (497 equity shares with prices; 3 REITs and 1 placeholder have none): ISIN, company, NSE industry, data checks"),
    "prices": ("data/interim/prices_adjusted.csv.gz", ["symbol_current", "date"],
               "Daily NSE Bhavcopy prices, corporate-action adjusted, with daily return and corporate-action flags"),
    "nifty50": ("data/raw/prices/nifty50_daily.csv", ["date"], "NIFTY 50 index, daily OHLC (market benchmark)"),
    "india_vix": ("data/raw/prices/india_vix_daily.csv", ["date"], "India VIX, daily OHLC (volatility control)"),
    "policy_rates": ("data/interim/policy_rates.csv", ["date"], "Daily RBI and Fed policy rates (BIS), carried forward"),
    "macro_releases": ("data/interim/macro_release_dates.csv", ["release", "release_date"],
                       "U.S. CPI and Employment Situation (BLS) and India CPI release dates with NSE day 0"),
    "corporate_actions": ("data/interim/corporate_actions_parsed.csv", None,
                          "NSE corporate actions with split, bonus and dividend adjustment factors"),
    "sentences": ("data/interim/sentences.csv", ["sent_id"], "Policy sentences from the 142 documents"),
    "gold_labels": ("data/labels/gold_300_labels.csv", ["gold_id"], "Author's gold-standard tone labels (300 sentences)"),
    "llm_labels": ("data/labels/llm_labels.csv", ["sent_id"], "LLM tone labels used to train the author's classifier"),
    "tone_scores": ("data/interim/tone_scores.csv", ["event_id"], "Tone score per announcement from the author's classifier"),
}

VIEWS = {
    "v_event_day0_returns": (
        """SELECT e.event_id, e.bank, e.event_type, e.event_day0, p.symbol_current AS symbol, s.industry,
                  p.ret AS ret_day0, p.complex_action
           FROM events e JOIN prices p ON p.date = e.event_day0
           JOIN stocks s ON s.symbol = p.symbol_current""",
        "Each stock's adjusted return on each announcement's day 0"),
    "v_event_market": (
        """SELECT e.*, n.close AS nifty50_close, v.close AS india_vix_close
           FROM events e LEFT JOIN nifty50 n ON n.date = e.event_day0
           LEFT JOIN india_vix v ON v.date = e.event_day0""",
        "Announcements with the NIFTY 50 and India VIX close on day 0"),
    "v_sentence_labels": (
        """SELECT s.sent_id, s.event_id, s.bank, s.event_type, s.text, g.label AS gold_label, g.partition AS gold_partition,
                  {llm} AS llm_label
           FROM sentences s LEFT JOIN gold_labels g ON g.sent_id = s.sent_id {llm_join}""",
        "Sentences with the author's gold label and the LLM label where available"),
    "v_sector_day0": (
        """SELECT bank, event_type, industry, COUNT(*) AS n, AVG(ret_day0) * 100 AS mean_ret_pct
           FROM v_event_day0_returns GROUP BY ALL ORDER BY bank, event_type, industry""",
        "Mean day-0 raw return by bank, announcement type and sector"),
}


def build():
    if DB.exists():
        DB.unlink()
    con = duckdb.connect(str(DB))
    lines = ["# Database schema (`data/capstone.duckdb`)", "",
             "DuckDB file built by `src/data/build_database.py` from the repository CSVs; query with `src/data/db.py`.", "",
             "| Table | Rows | Primary key | Source | Description |", "|---|---|---|---|---|"]
    loaded = set()
    for name, (src, pk, desc) in TABLES.items():
        path = ROOT / src
        if not path.exists():
            print(f"  skip {name}: {src} not in repo yet")
            continue
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_csv_auto('{path.as_posix()}', header=true, sample_size=-1)")
        n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        if pk:
            dup = con.execute(f"SELECT COUNT(*) FROM (SELECT {', '.join(pk)} FROM {name} GROUP BY ALL HAVING COUNT(*) > 1)").fetchone()[0]
            assert dup == 0, f"{name}: {dup} duplicate keys on {pk}"
        loaded.add(name)
        lines.append(f"| `{name}` | {n:,} | {', '.join(pk) if pk else '—'} | `{src}` | {desc} |")
        print(f"  {name:18} {n:>9,} rows")
    lines += ["", "## Views", "", "| View | Description |", "|---|---|"]
    for name, (sql, desc) in VIEWS.items():
        if name == "v_sentence_labels":
            sql = sql.format(llm="l.label" if "llm_labels" in loaded else "NULL",
                             llm_join="LEFT JOIN llm_labels l ON l.sent_id = s.sent_id" if "llm_labels" in loaded else "")
        con.execute(f"CREATE VIEW {name} AS {sql}")
        lines.append(f"| `{name}` | {desc} |")

    # integrity checks
    checks = {
        "events = 142": con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 142,
        "stocks = 497 with prices": con.execute("SELECT COUNT(DISTINCT symbol_current) FROM prices").fetchone()[0] == 497,
        "every event day 0 is a trading day": con.execute(
            "SELECT COUNT(*) FROM events WHERE event_day0 NOT IN (SELECT DISTINCT date FROM prices)").fetchone()[0] == 0,
        "prices within 2021-04-01 to 2026-09-30": con.execute(
            "SELECT MIN(date) >= DATE '2021-04-01' AND MAX(date) <= DATE '2026-09-30' FROM prices").fetchone()[0],
        "no non-positive adjusted close": con.execute("SELECT COUNT(*) FROM prices WHERE adj_close <= 0").fetchone()[0] == 0,
    }
    lines += ["", "## Integrity checks (run at build time)", ""] + [f"- {'✅' if ok else '❌'} {k}" for k, ok in checks.items()]
    lines += ["", "## Columns", ""]
    for name in loaded:
        cols = con.execute(f"DESCRIBE {name}").fetchall()
        lines.append(f"**{name}**: " + ", ".join(f"`{c[0]}` ({c[1].lower()})" for c in cols))
        lines.append("")
    con.close()
    DOC.write_text("\n".join(lines))
    for k, ok in checks.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {k}")
    print(f"Database: {DB.relative_to(ROOT)} ({DB.stat().st_size / 1e6:.1f} MB) | schema: {DOC.relative_to(ROOT)}")
    return all(checks.values())


if __name__ == "__main__":
    build()
