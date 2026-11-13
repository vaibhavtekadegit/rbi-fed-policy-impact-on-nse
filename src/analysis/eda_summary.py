"""
Exploratory data analysis and data-quality log for the final report.

Reads the same inputs as src/analysis/final_analysis.py and its outputs in reports/final/analysis/,
and writes reports/final/eda/:
    data_quality.csv          missing values, exclusions and outlier handling, with counts
    exclusions.csv            stock-event candidates and the reason each excluded one was dropped
    eda_events.csv            announcements by bank and type: decisions, CAAR, share of stocks up, tone, overlaps
    eda_car_distribution.csv  distribution of stock-level CAR[0,+1] before and after winsorization
    eda_sectors.csv           stocks and stock-events by sector, mean CAR and share up
    figures/fig_eda_timeline.png  policy rates with announcement dates
    figures/fig_eda_car_hist.png  stock-level CAR[0,+1] histogram with winsorization bounds

Run from the repository root after notebook 07:
    python src/analysis/eda_summary.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "reports/final/analysis"
OUT = ROOT / "reports/final/eda"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
EST_START, EST_END, MIN_EST = -130, -11, 100        # same as final_analysis.py
C_RBI, C_FED = "#1f4e79", "#c55a11"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

# ---------------------------------------------------------------- inputs
adj = pd.read_csv(ROOT / "data/interim/prices_adjusted.csv.gz", parse_dates=["date"],
                  usecols=["date", "symbol_current", "ret", "complex_action"])
R = adj.pivot(index="date", columns="symbol_current", values="ret").sort_index()
CX = adj.pivot(index="date", columns="symbol_current", values="complex_action").reindex(R.index).fillna(False).astype(bool)
nifty = pd.read_csv(ROOT / "data/raw/prices/nifty50_daily.csv", parse_dates=["date"]).set_index("date")
M = nifty["close"].pct_change().reindex(R.index)
val = pd.read_csv(ROOT / "data/raw/prices/validation_report.csv")
ev = pd.read_csv(FINAL / "event_level.csv", parse_dates=["release_date", "event_day0"])
se = pd.read_csv(FINAL / "stock_event_car.csv.gz")
res = json.loads((FINAL / "results.json").read_text())
lo, hi = res["sample"]["winsor_bounds_0_1"]
Rv, Mv, CXv = R.values, M.values, CX.values
n_days = len(R)

# ---------------------------------------------------------------- 1. exclusions (same rules as final_analysis.py)
rows = []
for e in ev.itertuples():
    t0 = int(e.t0)
    e0, e1 = max(t0 + EST_START, 0), t0 + EST_END
    Y, X = Rv[e0:e1 + 1], Mv[e0:e1 + 1][:, None]
    n_est = (~np.isnan(Y) & ~np.isnan(X)).sum(0)
    miss01 = np.isnan(Rv[t0]) | np.isnan(Rv[t0 + 1]) if t0 + 1 < n_days else np.ones(R.shape[1], bool)
    cx = CXv[max(t0 - 1, 0):t0 + 2].any(0)
    reason = np.where(n_est < MIN_EST, "fewer than 100 estimation days (recent listing)",
             np.where(miss01, "no return on day 0 or day +1",
             np.where(cx, "complex corporate action on days -1 to +1", "kept")))
    rows.append(pd.Series(reason).value_counts().rename(e.event_id))
excl = pd.DataFrame(rows).fillna(0).astype(int)
excl_tot = excl.sum().sort_values(ascending=False)
excl_tot.rename_axis("reason").rename("stock_events").to_csv(OUT / "exclusions.csv")
assert excl_tot.get("kept", 0) == len(se), "exclusion rules must reproduce the 61,957 stock-events"

# ---------------------------------------------------------------- 2. data-quality log
n_const, n_priced = len(val), int(val["rows"].gt(0).sum())
car = se["car_raw"]
fed_missing = ev.loc[(ev["bank"] == "FED") & ev["surprise_bps"].isna(), "event_id"].tolist()
dq = [
    ("NIFTY 500 constituents in the index file", n_const, "Universe as of October 2026"),
    ("Constituents without NSE equity prices", n_const - n_priced, "Excluded (3 REITs and 1 placeholder symbol)"),
    ("Stocks with price files", n_priced, "Used"),
    ("Trading days in the price window (Apr 2021 to Sep 2026)", n_days, ""),
    ("Stocks listed after April 2021 (shorter history)", int((val["rows"].between(1, n_days - 1)).sum()),
     "Kept; a stock-event needs at least 100 valid estimation days"),
    ("Missing trading days inside a stock's listed period (total)", int(val["missing_days"].fillna(0).sum()),
     "Left missing (no filling); returns over the gap are not used"),
    ("Stock-days with complex corporate actions", int(adj["complex_action"].sum()),
     "Stock-events with such a day in [-1,+1] excluded"),
    ("Stock-event candidates (142 events x stocks with prices)", int(excl.values.sum()), ""),
] + [(f"Excluded: {k}", int(v), "Dropped") for k, v in excl_tot.items() if k != "kept"] + [
    ("Stock-events kept", int(excl_tot["kept"]), f"{excl_tot['kept'] / excl.values.sum():.1%} of candidates"),
    ("Fed events without a 2-year yield surprise", len(fed_missing),
     f"{', '.join(fed_missing)}: dropped from RQ3 (141 events); in RQ4 set to 0 with a missing-value flag"),
    ("RBI events with a market-based surprise", 0, "India T-bill not used; RBI rate change (bp) used instead, 0 for holds and minutes"),
    ("Stock CARs below the 1st percentile", int((car < lo).sum()), f"Winsorized to {lo:.2f} pp"),
    ("Stock CARs above the 99th percentile", int((car > hi).sum()), f"Winsorized to {hi:.2f} pp"),
    ("Stock CARs exactly zero", int((se["car"] == 0).sum()), "Would be excluded from RQ4 (direction undefined)"),
    ("Announcements with another event or macro release on days -1 to +1", int(ev["overlap_flag"].sum()),
     "Kept in the main analysis; excluded in robustness checks"),
]
pd.DataFrame(dq, columns=["item", "count", "handling"]).to_csv(OUT / "data_quality.csv", index=False)

# ---------------------------------------------------------------- 3. events by bank and type
g = ev.groupby(["bank", "event_type"])
eda_ev = pd.DataFrame({
    "events": g.size(),
    "hikes": g["decision"].apply(lambda x: (x == "hike").sum()),
    "cuts": g["decision"].apply(lambda x: (x == "cut").sum()),
    "holds": g["decision"].apply(lambda x: (x == "hold").sum()),
    "stocks_per_event": g["n_stocks"].mean(),
    "caar_mean": g["caar"].mean(), "caar_sd": g["caar"].std(), "caar_min": g["caar"].min(), "caar_max": g["caar"].max(),
    "abs_caar_mean": g["abs_caar"].mean(), "share_up_mean": g["share_up"].mean(),
    "tone_mean": g["tone_score"].mean(), "tone_sd": g["tone_score"].std(),
    "sentences": g["n_sentences"].sum(), "overlap_events": g["overlap_flag"].sum(),
    "mean_cross_corr": g["rbar"].mean(),
}).round(4).reset_index()
eda_ev.to_csv(OUT / "eda_events.csv", index=False)


# ---------------------------------------------------------------- 4. CAR distribution
def describe(x):
    return {"n": len(x), "mean": x.mean(), "sd": x.std(), "p1": x.quantile(.01), "p25": x.quantile(.25),
            "median": x.median(), "p75": x.quantile(.75), "p99": x.quantile(.99), "min": x.min(), "max": x.max(),
            "skew": x.skew(), "excess_kurtosis": x.kurt(), "share_positive": (x > 0).mean()}


dist = pd.DataFrame({"before winsorization": describe(car), "after winsorization": describe(se["car"]),
                     "event CAAR (142 events)": describe(ev["caar"])}).T.round(4)
dist.index.name = "series"
dist.to_csv(OUT / "eda_car_distribution.csv")

# ---------------------------------------------------------------- 5. sectors
sec = (se.groupby("sector").agg(stocks=("symbol", "nunique"), stock_events=("car", "size"),
                                mean_car=("car", "mean"), share_up=("car", lambda x: (x > 0).mean()))
         .sort_values("stocks", ascending=False).round(4).reset_index())
sec.to_csv(OUT / "eda_sectors.csv", index=False)

# ---------------------------------------------------------------- 6. figures
rates = pd.read_csv(ROOT / "data/interim/policy_rates.csv", parse_dates=["date"]).set_index("date")
rates = rates.reindex(pd.date_range("2021-09-01", "2026-09-30")).ffill()   # carry the last BIS value forward
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.step(rates.index, rates["rbi_rate"], where="post", color=C_RBI, lw=1.6, label="RBI repo rate")
ax.step(rates.index, rates["fed_rate"], where="post", color=C_FED, lw=1.6, label="Fed funds target (mid-point)")
for b, col, y in [("RBI", C_RBI, -0.35), ("FED", C_FED, -0.75)]:
    for t, mk in [("DEC", "|"), ("MIN", "x")]:
        d = ev[(ev["bank"] == b) & (ev["event_type"] == t)]["release_date"]
        ax.scatter(d, np.full(len(d), y), marker=mk, color=col, s=28 if mk == "|" else 14, lw=1,
                   label=f"{'Fed' if b == 'FED' else 'RBI'} {'decisions' if t == 'DEC' else 'minutes'}")
ax.set_ylabel("Policy rate (%)"); ax.set_ylim(-1.1, 7.2)
ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig_eda_timeline.png", dpi=200, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.hist(car.clip(-15, 15), bins=120, color="#7f8c8d", alpha=0.85)
for b in (lo, hi):
    ax.axvline(b, color="black", lw=0.9, ls="--")
ax.text(hi + 0.3, ax.get_ylim()[1] * 0.85, f"99th pct\n{hi:.2f}", fontsize=8)
ax.text(lo - 0.3, ax.get_ylim()[1] * 0.85, f"1st pct\n{lo:.2f}", fontsize=8, ha="right")
ax.set_xlabel("Stock CAR[0,+1], percentage points (values beyond +/-15 shown at the edges)"); ax.set_ylabel("Stock-events")
fig.tight_layout(); fig.savefig(FIG / "fig_eda_car_hist.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- print
pd.set_option("display.width", 200)
print(pd.read_csv(OUT / "data_quality.csv").to_string(index=False))
print(eda_ev.to_string(index=False))
print(dist.to_string())
print(sec.head(8).to_string(index=False))
