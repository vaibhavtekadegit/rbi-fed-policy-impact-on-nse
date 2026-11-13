# Database schema (`data/capstone.duckdb`)

DuckDB file built by `src/data/build_database.py` from the repository CSVs; query with `src/data/db.py`.

| Table | Rows | Primary key | Source | Description |
|---|---|---|---|---|
| `events` | 142 | event_id | `data/interim/events.csv` | 142 RBI and Fed announcements: release time (IST), NSE day 0, decision, rate change, surprise, overlap flags |
| `stocks` | 501 | symbol | `data/raw/prices/validation_report.csv` | 501 NIFTY 500 constituents (497 equity shares with prices; 3 REITs and 1 placeholder have none): ISIN, company, NSE industry, data checks |
| `prices` | 599,800 | symbol_current, date | `data/interim/prices_adjusted.csv.gz` | Daily NSE Bhavcopy prices, corporate-action adjusted, with daily return and corporate-action flags |
| `nifty50` | 1,364 | date | `data/raw/prices/nifty50_daily.csv` | NIFTY 50 index, daily OHLC (market benchmark) |
| `india_vix` | 1,363 | date | `data/raw/prices/india_vix_daily.csv` | India VIX, daily OHLC (volatility control) |
| `policy_rates` | 2,022 | date | `data/interim/policy_rates.csv` | Daily RBI and Fed policy rates (BIS), carried forward |
| `macro_releases` | 206 | release, release_date | `data/interim/macro_release_dates.csv` | U.S. CPI and Employment Situation (BLS) and India CPI release dates with NSE day 0 |
| `corporate_actions` | 13,596 | — | `data/interim/corporate_actions_parsed.csv` | NSE corporate actions with split, bonus and dividend adjustment factors |
| `sentences` | 20,205 | sent_id | `data/interim/sentences.csv` | Policy sentences from the 142 documents |
| `gold_labels` | 300 | gold_id | `data/labels/gold_300_labels.csv` | Author's gold-standard tone labels (300 sentences) |

## Views

| View | Description |
|---|---|
| `v_event_day0_returns` | Each stock's adjusted return on each announcement's day 0 |
| `v_event_market` | Announcements with the NIFTY 50 and India VIX close on day 0 |
| `v_sentence_labels` | Sentences with the author's gold label and the LLM label where available |
| `v_sector_day0` | Mean day-0 raw return by bank, announcement type and sector |

## Integrity checks (run at build time)

- ✅ events = 142
- ✅ stocks = 497 with prices
- ✅ every event day 0 is a trading day
- ✅ prices within 2021-04-01 to 2026-09-30
- ✅ no non-positive adjusted close

## Columns

**india_vix**: `date` (date), `open` (double), `high` (double), `low` (double), `close` (double)

**macro_releases**: `release` (varchar), `release_date` (date), `release_datetime_ist` (timestamp with time zone), `source` (varchar), `nse_day0` (date)

**sentences**: `sent_id` (varchar), `event_id` (varchar), `bank` (varchar), `event_type` (varchar), `para_no` (bigint), `sent_no` (bigint), `text` (varchar), `n_words` (bigint)

**events**: `event_id` (varchar), `bank` (varchar), `event_type` (varchar), `release_date` (date), `release_time_ist` (timestamp), `release_time_source` (varchar), `event_day0` (date), `meeting_date` (date), `decision` (varchar), `rate_before` (double), `rate_after` (double), `rate_change_bps` (double), `rate_source` (varchar), `surprise_bps` (double), `surprise_source` (varchar), `overlap_event` (varchar), `overlap_macro` (varchar), `overlap_flag` (bigint), `source_url` (varchar), `file_name` (varchar)

**stocks**: `symbol` (varchar), `isin` (varchar), `company` (varchar), `industry` (varchar), `rows` (bigint), `old_isins` (varchar), `first_date` (date), `last_date` (date), `missing_days` (double), `bad_prices` (double), `big_moves` (double), `symbol_changes` (double)

**prices**: `date` (date), `symbol_current` (varchar), `symbol` (varchar), `isin_current` (varchar), `open` (double), `high` (double), `low` (double), `close` (double), `adj_factor` (double), `adj_open` (double), `adj_high` (double), `adj_low` (double), `adj_close` (double), `volume` (bigint), `value` (double), `ret` (double), `corp_action` (varchar), `complex_action` (boolean)

**nifty50**: `date` (date), `open` (double), `high` (double), `low` (double), `close` (double)

**corporate_actions**: `symbol` (varchar), `ex_date` (date), `action` (varchar), `price_factor` (double), `dividend_rs` (double), `purpose` (varchar)

**policy_rates**: `date` (date), `rbi_rate` (double), `fed_rate` (double)

**gold_labels**: `gold_id` (varchar), `sent_id` (varchar), `event_id` (varchar), `bank` (varchar), `event_type` (varchar), `partition` (varchar), `label` (varchar), `notes` (varchar), `labeled_by` (varchar)
