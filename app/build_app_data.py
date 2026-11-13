"""
Build the small data files used by the Gradio app (app/data/).

Run from the repository root after notebook 07:
    python app/build_app_data.py

Outputs
    app/data/stocks_latest.csv      one row per stock: market-model features as of the last price date
                                    (beta, size_log, resid_sd, pre_car), company and sector
    app/data/events.csv             the 142 announcements with the event-level model features
    app/data/past_stock_events.csv.gz  stock-event rows (features and actual CAR) for replaying past announcements
    app/data/app_meta.json          market inputs on the last price date and the reported test metrics

The stock features follow src/analysis/final_analysis.py: market model on the NIFTY 50 over days -130 to -11
(at least 100 valid days), residual standard deviation, log of mean traded value, and the sum of abnormal
returns over the 5 days before day 0. For the "latest" file, day 0 is the session after the last price date.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app/data"
OUT.mkdir(parents=True, exist_ok=True)
EST_START, EST_END, MIN_EST = -130, -11, 100
EVENT_FEATS = ["is_fed", "is_min", "fed_surprise", "rbi_rate_change", "vix_m1", "nifty_pre5", "tone_score", "tone_x_fed"]

# ---------------------------------------------------------------- prices
adj = pd.read_csv(ROOT / "data/interim/prices_adjusted.csv.gz", parse_dates=["date"],
                  usecols=["date", "symbol_current", "ret", "value"])
R = adj.pivot(index="date", columns="symbol_current", values="ret").sort_index() * 100
VAL = adj.pivot(index="date", columns="symbol_current", values="value").reindex(R.index)
nifty = pd.read_csv(ROOT / "data/raw/prices/nifty50_daily.csv", parse_dates=["date"]).set_index("date")
M = (nifty["close"].pct_change() * 100).reindex(R.index)
vix = pd.read_csv(ROOT / "data/raw/prices/india_vix_daily.csv", parse_dates=["date"]).set_index("date")["close"]
info = pd.read_csv(ROOT / "data/raw/prices/validation_report.csv").set_index("symbol")

# ---------------------------------------------------------------- latest stock features (day 0 = next session)
t0 = len(R)
Y = R.values[t0 + EST_START:t0 + EST_END + 1]
X = M.values[t0 + EST_START:t0 + EST_END + 1][:, None]
mask = ~np.isnan(Y) & ~np.isnan(X)
n = mask.sum(0)
Xm, Ym = np.where(mask, X, 0.0), np.where(mask, Y, 0.0)
sx, sy, sxx, sxy = Xm.sum(0), Ym.sum(0), (Xm ** 2).sum(0), (Xm * Ym).sum(0)
with np.errstate(invalid="ignore", divide="ignore"):
    beta = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
    alpha = (sy - beta * sx) / n
    resid = np.where(mask, Y - (alpha + beta * X), np.nan)
    resid_sd = np.sqrt(np.nansum(resid ** 2, 0) / (n - 2))
    v = VAL.values[t0 + EST_START:t0 + EST_END + 1]
    size_log = np.log(np.nanmean(np.where(v > 0, v, np.nan), axis=0))
pre_ar = R.values[t0 - 5:t0] - (alpha + beta * M.values[t0 - 5:t0][:, None])
pre_car = np.nansum(pre_ar, 0)

stocks = pd.DataFrame({"symbol": R.columns, "beta": beta, "size_log": size_log, "resid_sd": resid_sd,
                       "pre_car": pre_car, "est_days": n})
stocks = stocks.join(info[["company", "industry"]], on="symbol").rename(columns={"industry": "sector"})
stocks = stocks[(stocks["est_days"] >= MIN_EST) & stocks[["beta", "size_log", "resid_sd"]].notna().all(axis=1)]
stocks = stocks.drop(columns="est_days").round(6)
stocks.to_csv(OUT / "stocks_latest.csv", index=False)

# ---------------------------------------------------------------- past announcements
ev = pd.read_csv(ROOT / "reports/final/analysis/event_level.csv")
ev = ev.sort_values("order").reset_index(drop=True)
n_train = int(round(0.70 * len(ev)))
ev["partition"] = np.where(ev["order"] < n_train, "TRAIN", "TEST")
ev_out = ev[["event_id", "bank", "event_type", "release_date", "event_day0", "decision", "rate_change_bps", "order",
             "partition", "share_up", "caar", "n_stocks"] + EVENT_FEATS]
ev_out.to_csv(OUT / "events.csv", index=False)

se = pd.read_csv(ROOT / "reports/final/analysis/stock_event_car.csv.gz",
                 usecols=["event_id", "symbol", "sector", "beta", "size_log", "resid_sd", "pre_car", "car"])
se = se[se["car"] != 0].dropna(subset=["beta", "size_log", "resid_sd", "pre_car"])
se["up"] = (se["car"] > 0).astype(int)
se = se.drop(columns="car").round(6)
se.to_csv(OUT / "past_stock_events.csv.gz", index=False, compression="gzip")

# ---------------------------------------------------------------- metadata
perf = pd.read_csv(ROOT / "reports/final/analysis/rq4_performance.csv")
ci = pd.read_csv(ROOT / "reports/final/analysis/rq4_bootstrap_ci.csv").set_index("model")
test = perf[perf["partition"] == "TEST"].set_index("model")
meta = {
    "last_price_date": str(R.index[-1].date()),
    "vix_last": float(vix.loc[:R.index[-1]].iloc[-1]),
    "nifty_pre5_last": float(np.nansum(M.values[t0 - 5:t0])),
    "n_stocks_latest": int(len(stocks)),
    "first_test_event": str(ev.loc[ev["order"] == n_train, "event_id"].iloc[0]),
    "test_metrics": {m: {"accuracy": round(float(test.loc[m, "accuracy"]), 3), "roc_auc": round(float(test.loc[m, "roc_auc"]), 3),
                         "auc_ci": [round(float(ci.loc[m, "auc_ci_low"]), 3), round(float(ci.loc[m, "auc_ci_high"]), 3)]}
                     for m in test.index},
}
(OUT / "app_meta.json").write_text(json.dumps(meta, indent=2))
print(f"stocks_latest: {len(stocks)} stocks; events: {len(ev_out)}; past stock-events: {len(se)}")
print(json.dumps(meta, indent=2))
