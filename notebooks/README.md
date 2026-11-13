# Colab notebooks — run order

Each notebook clones the repo, works on a dated branch, and pushes it; open the printed PR link and merge before the next one.
Secrets (Colab key icon): `GITHUB_TOKEN` (all), `ANTHROPIC_API_KEY` (05 only).

| # | Notebook | Produces | Status |
|---|---|---|---|
| 01 | `01_policy_documents_download` | 142 RBI/Fed HTML documents, DGS2 | Done (Oct 20) |
| 02 | `02_prices_corporate_actions` | NSE Bhavcopy per stock, corporate actions, adjusted prices, NIFTY 50, India VIX | Done (Oct 20) |
| 03 | `03_data_finalize` | first `events.csv` | Done (Oct 20); superseded by 04 |
| 04 | `04_part_a_final_data` | BIS rates, macro dates, final `events.csv` (T-bill skipped by decision) | Done (Nov 13) |
| 05 | `05_llm_labeling` | `llm_labels.csv` (Fed v1.0, RBI v1.1, Haiku 5.5) | Done (Nov 13) |
| 06 | `06_tone_classifier` | classifier results, saved model, `tone_scores.csv` | Done (Nov 13, PR #23) |
| 07 | `07_final_analysis` | RQ1–RQ4 final tables, figures, `models/xgb_rq4.json` | **Run next** |

Sentence extraction (`src/tone/sentences.py`), gold sampling (`src/tone/gold_sample.py`) and the database
(`src/data/build_database.py`) are plain scripts and already run; outputs are in the repo.
