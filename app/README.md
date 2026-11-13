---
title: RBI and Fed Announcements and Indian Stocks
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: How NIFTY 500 stocks react to RBI and Fed announcements
---

# RBI and Fed announcements: how might Indian stocks react?

Demonstration app for the QM640 Data Analytics Capstone project
"Local or Global? How Reserve Bank of India and Federal Reserve Announcements Move Indian Stocks" (Vaibhav Tekade, Walsh College).
Code, data and full results: https://github.com/vaibhavtekadegit/rbi-fed-policy-impact-on-nse

## What it does

- **Try a scenario:** choose RBI or Fed, decision or minutes, enter the policy news (Fed 2-year yield surprise or RBI rate
  change), India VIX and the recent NIFTY 50 return, and optionally paste the policy text. The app scores the tone of the
  text with the author's classifier and predicts, for each of 493 NIFTY 500 stocks, the probability of a positive
  abnormal return over the announcement day and the next day. Results are shown by sector and for the most and least
  likely stocks.
- **Past announcements:** replay any of the 142 announcements (October 2021 to September 2026) and compare the predicted
  and actual share of stocks that rose. Test announcements were never used in training.
- **About the model:** data, test metrics and limitations.

## Model card (short)

| Item | Value |
|---|---|
| Model | XGBoost classifier (max depth 4, 500 trees, learning rate 0.03), trained in notebook 07 |
| Training data | 99 announcements, 41,661 stock-announcement rows (Oct 2021 to Mar 2025) |
| Test data | 43 later announcements, 20,296 rows |
| Test accuracy / ROC-AUC | 0.516 / 0.532 (majority-class baseline accuracy 0.503) |
| Tone model | TF-IDF + linear SVM, macro-F1 0.785 on 200 author-labeled sentences |
| Intended use | teaching and research demonstration |
| Not for | trading or investment decisions |

The test results show that the direction of a single stock over two days is close to unpredictable from announcement
features; the app shows this caveat next to every prediction.

## Files

| File | Purpose |
|---|---|
| `app.py` | Gradio interface |
| `model.py` | loads the models and makes predictions (no interface code) |
| `build_app_data.py` | builds `data/` from the repository outputs (run from the repo root) |
| `data/` | stock features on the last price date, the 142 events, past stock-event rows, test metrics |
| `models/` (Space only) | copy of the repo's `models/` folder: `tone_clf_v1.joblib`, `xgb_rq4.json`, `xgb_rq4_features.json` |

## Run locally

```
pip install -r app/requirements.txt
python app/app.py
```

Then open http://127.0.0.1:7860.
