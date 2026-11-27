"""
Announcement days up close: which individual announcements moved NIFTY 500 stocks?

For each of the 142 announcements this script computes a cross-sectional t-test of the stock CARs, adjusted for the
average same-day correlation of abnormal returns (Kolari and Pynnonen, 2010), and screens the significant days with
rules that use only repository data:
    1. significant on its own: adjusted p < .05
    2. no other sample announcement and no U.S. CPI, U.S. payroll or India CPI release on days -1 to +1 (overlap_flag = 0)
    3. no large move before the release: |abnormal return on day -1| < 0.5 percentage points
Same-day global news that is not in the repository is not screened; this is stated wherever the table is used.

Run from the repository root after notebook 07:
    python src/analysis/event_cases.py

Outputs (reports/final/analysis/):
    event_cases.csv                one row per announcement with the test and the screens
    figures/fig_event_cases.png    CAAR of every announcement over time, significant and screened days marked
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "reports/final/analysis"
PRE_LIMIT = 0.5
C_RBI, C_FED = "#1f4e79", "#c55a11"

ev = pd.read_csv(FINAL / "event_level.csv", parse_dates=["release_date", "event_day0"])
se = pd.read_csv(FINAL / "stock_event_car.csv.gz", usecols=["event_id", "car"])
g = se.groupby("event_id")["car"].agg(car_mean="mean", car_sd="std", n="size")
ev = ev.merge(g, left_on="event_id", right_index=True)

# Kolari-Pynnonen adjusted cross-sectional t for each announcement
t_raw = ev["car_mean"] / (ev["car_sd"] / np.sqrt(ev["n"]))
ev["t_adj"] = t_raw * np.sqrt((1 - ev["rbar"]) / (1 + (ev["n"] - 1) * ev["rbar"]))
ev["p_adj"] = 2 * stats.t.sf(ev["t_adj"].abs(), ev["n"] - 1)

# NIFTY 50 move on days 0 and +1, and the average abnormal return on day -1
nifty = pd.read_csv(ROOT / "data/raw/prices/nifty50_daily.csv", parse_dates=["date"]).set_index("date")["close"]
r = nifty.pct_change() * 100


def nifty_move(day0):
    i = r.index.get_indexer([day0])[0]
    return r.iloc[i] + r.iloc[i + 1]


ev["nifty_0_1"] = [nifty_move(d) for d in ev["event_day0"]]
ev["ar_day_m1"] = ev["caar[-1,+1]"] - ev["caar[0,+1]"]

ev["significant"] = ev["p_adj"] < .05
ev["no_overlap"] = ev["overlap_flag"] == 0
ev["no_pre_move"] = ev["ar_day_m1"].abs() < PRE_LIMIT
ev["passes_screen"] = ev["significant"] & ev["no_overlap"] & ev["no_pre_move"]

cols = ["event_id", "bank", "event_type", "event_day0", "decision", "rate_change_bps", "fed_surprise", "tone_score",
        "n", "caar", "share_up", "t_adj", "p_adj", "ar_day_m1", "nifty_0_1", "overlap_event", "overlap_macro",
        "significant", "no_overlap", "no_pre_move", "passes_screen"]
out = ev[cols].sort_values("event_day0")
out.round({c: 4 for c in out.select_dtypes("number").columns}).to_csv(FINAL / "event_cases.csv", index=False)

# figure: every announcement's CAAR over time
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(9, 3.6))
for b, col, lab in [("RBI", C_RBI, "RBI"), ("FED", C_FED, "Fed")]:
    d = out[out["bank"] == b]
    ax.scatter(d["event_day0"], d["caar"], s=18, color=col, alpha=0.35, label=f"{lab} announcement")
    s = d[d["significant"] & ~d["passes_screen"]]
    ax.scatter(s["event_day0"], s["caar"], s=46, facecolors="none", edgecolors=col, linewidths=1.3,
               label=f"{lab}: significant, fails a screen")
    k = d[d["passes_screen"]]
    ax.scatter(k["event_day0"], k["caar"], s=60, color=col, edgecolors="black", linewidths=0.8, label=f"{lab}: passes all screens")
ax.axhline(0, color="black", lw=0.7)
ax.set_ylabel("CAAR[0,+1], percentage points")
ax.legend(ncol=3, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2))
fig.tight_layout()
fig.savefig(FINAL / "figures/fig_event_cases.png", dpi=200, bbox_inches="tight")

n_sig = int(out["significant"].sum())
print(f"significant: {n_sig} of {len(out)} ({n_sig / len(out):.1%}); expected by chance at 5%: {0.05 * len(out):.1f}")
print(f"of those, no overlap: {int((out['significant'] & out['no_overlap']).sum())}; also no pre-move: {int(out['passes_screen'].sum())}")
pd.set_option("display.width", 220)
print(out[out["significant"]][["event_id", "decision", "fed_surprise", "rate_change_bps", "caar", "share_up", "p_adj",
                                "ar_day_m1", "nifty_0_1", "no_overlap", "no_pre_move"]].round(3).to_string(index=False))
