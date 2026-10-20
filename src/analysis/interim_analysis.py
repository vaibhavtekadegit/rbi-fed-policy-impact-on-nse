"""
Interim analysis: event study of NIFTY 500 stocks around RBI and Fed announcements.

Reads only files in this repository and writes every table, figure and number used in the
interim report to reports/interim/. Run from the repository root:

    python src/analysis/interim_analysis.py

Steps
1. Load adjusted prices, NIFTY 50, India VIX, events, BIS policy rates, FRED DGS2.
2. Market model per stock and event (estimation window days -130 to -11, >= 100 valid days).
3. Abnormal returns days -5..+5, CAR[0,+1]; exclusions; 1st/99th percentile winsorization.
4. Event-level CAAR; placebo days; RQ1 and RQ2 tests; RQ3 preliminary regressions;
   RQ4 preliminary classifiers (chronological event split); within-event correlation.
"""
import json
import warnings
warnings.filterwarnings("ignore")
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/interim"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
EST_START, EST_END, MIN_EST = -130, -11, 100
PATH_DAYS = range(-5, 6)
RESULTS = {}

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
C_RBI, C_FED = "#1f4e79", "#c55a11"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 1. load
adj = pd.read_csv(ROOT / "data/interim/prices_adjusted.csv.gz", parse_dates=["date"])
dup_rows = int(adj.duplicated(["symbol_current", "date"]).sum())
adj["ret_pct"] = adj["ret"] * 100
R = adj.pivot(index="date", columns="symbol_current", values="ret_pct").sort_index()
CX = adj.pivot(index="date", columns="symbol_current", values="complex_action").reindex(R.index)
CX = CX.fillna(False).astype(bool)
VAL = adj.pivot(index="date", columns="symbol_current", values="value").reindex(R.index)
days = R.index
stocks = R.columns
N_DAYS, N_STOCKS = len(days), len(stocks)

nifty = pd.read_csv(ROOT / "data/raw/prices/nifty50_daily.csv", parse_dates=["date"]).set_index("date")
M = (nifty["close"].pct_change() * 100).reindex(days)
vix = pd.read_csv(ROOT / "data/raw/prices/india_vix_daily.csv", parse_dates=["date"]).set_index("date")["close"]
vix = vix.reindex(days)

info = pd.read_csv(ROOT / "data/raw/prices/validation_report.csv")
sector = info.set_index("symbol")["industry"].reindex(stocks)

ev = pd.read_csv(ROOT / "data/interim/events.csv", parse_dates=["release_date", "event_day0"])


def read_bis(path):
    d = pd.read_csv(path, skiprows=3)
    d = d[["TIME_PERIOD:Period", "OBS_VALUE:Value"]].dropna()
    d.columns = ["date", "rate"]
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["rate"].sort_index()


bis = {"RBI": read_bis(ROOT / "data/raw/rates/bis_cbpol_india.csv"),
       "FED": read_bis(ROOT / "data/raw/rates/bis_cbpol_us.csv")}
dgs2 = pd.read_csv(ROOT / "data/raw/rates/fred_dgs2.csv", parse_dates=["observation_date"])
dgs2 = dgs2.set_index("observation_date")["DGS2"].dropna()


def rbi_rate_from_text(file_name):
    """Policy repo rate after the decision, as stated in the RBI resolution (fallback where BIS has no data)."""
    html = (ROOT / "data/raw/events/rbi/decisions" / file_name).read_text(errors="ignore")
    text = " ".join(re.sub(r"<[^>]+>", " ", html).split())
    for pat in (r"policy repo rate[^.]{0,120}?unchanged at\s*([\d.]+)\s*per\s*cent",
                r"policy repo rate[^.]{0,160}?\bto\s*([\d.]+)\s*per\s*cent",
                r"repo rate[^.]{0,120}?unchanged at\s*([\d.]+)\s*per\s*cent"):
        m = re.search(pat, text, re.I)
        if m:
            return float(m.group(1))
    return np.nan


rb, ra, src = [], [], []
for _, r in ev.iterrows():
    if r["event_type"] != "DEC":
        rb.append(np.nan); ra.append(np.nan); src.append(""); continue
    s = bis[r["bank"]]
    before = s[s.index < r["release_date"]]
    after = s[(s.index > r["release_date"]) & (s.index <= r["release_date"] + pd.Timedelta(days=7))]
    b_val = before.iloc[-1] if len(before) else np.nan
    a_val = after.iloc[0] if len(after) else np.nan
    source = "BIS"
    if np.isnan(a_val) and r["bank"] == "RBI":
        a_val, source = rbi_rate_from_text(r["file_name"]), "BIS + RBI resolution text"
    if np.isnan(b_val) and r["bank"] == "RBI":
        txt = rbi_rate_from_text(r["file_name"])
        b_val, source = (a_val if not np.isnan(a_val) else txt), "RBI resolution text (BIS starts 2021-10-20)"
    rb.append(b_val); ra.append(a_val); src.append(source if not (np.isnan(a_val) or np.isnan(b_val)) else "missing")
ev["rate_before"], ev["rate_after"], ev["rate_source"] = rb, ra, src
ev["rate_change_bps"] = ((ev["rate_after"] - ev["rate_before"]) * 100).round(0)
ev["decision"] = np.where(ev["rate_change_bps"].isna(), None,
                          np.where(ev["rate_change_bps"] > 0, "hike",
                                   np.where(ev["rate_change_bps"] < 0, "cut", "hold")))
ev["bank_label"] = ev["bank"].map({"RBI": "RBI", "FED": "Fed"})
ev = ev.sort_values(["release_date", "bank", "event_type"]).reset_index(drop=True)
ev["order"] = np.arange(len(ev))
day_pos = {d: i for i, d in enumerate(days)}
ev["t0"] = ev["event_day0"].map(day_pos)
ev["vix_m1"] = [vix.iloc[t - 1] if pd.notna(t) else np.nan for t in ev["t0"]]


# ------------------------------------------------------------------ 2. market model per event
Rv, Mv, CXv_, VALv = R.values, M.values, CX.values, VAL.values


def event_block(t0, exclude_complex=True):
    """Return per-stock alpha, beta, n_est, AR path (days -5..+5), CAR[0,+1], size, flags."""
    e0, e1 = max(t0 + EST_START, 0), t0 + EST_END        # first event: window starts at first data day
    Y = Rv[e0:e1 + 1]
    X = Mv[e0:e1 + 1][:, None]
    mask = ~np.isnan(Y) & ~np.isnan(X)
    n = mask.sum(0)
    Xm = np.where(mask, X, 0.0)
    Ym = np.where(mask, Y, 0.0)
    sx, sy = Xm.sum(0), Ym.sum(0)
    sxx, sxy = (Xm ** 2).sum(0), (Xm * Ym).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
        alpha = (sy - beta * sx) / n
    path = np.full((len(PATH_DAYS), Y.shape[1]), np.nan)
    for k, d in enumerate(PATH_DAYS):
        if 0 <= t0 + d < N_DAYS:
            path[k] = Rv[t0 + d] - (alpha + beta * Mv[t0 + d])
    car = path[5] + path[6]                                   # days 0 and +1
    cx = CXv_[max(t0 - 1, 0):t0 + 2].any(0)
    size = np.log(np.nanmean(np.where(VALv[e0:e1 + 1] > 0, VALv[e0:e1 + 1], np.nan), axis=0))
    ok_est = n >= MIN_EST
    ok_evt = ~np.isnan(path[5]) & ~np.isnan(path[6])
    keep = ok_est & ok_evt & (~cx if exclude_complex else True)
    return dict(alpha=alpha, beta=beta, n=n, path=path, car=car, size=size, cx=cx,
                ok_est=ok_est, ok_evt=ok_evt, keep=keep, any_price=(~np.isnan(Rv[e0:t0 + 2])).any(0))


rows, paths, log = [], {}, {"possible": 0, "no_prices": 0, "short_estimation": 0,
                            "missing_event_day_return": 0, "complex_action": 0}
for _, e in ev.iterrows():
    b = event_block(int(e["t0"]))
    log["possible"] += N_STOCKS
    no_px = ~b["any_price"]
    short = b["any_price"] & ~b["ok_est"]
    miss = b["ok_est"] & ~b["ok_evt"]
    cxx = b["ok_est"] & b["ok_evt"] & b["cx"]
    log["no_prices"] += int(no_px.sum()); log["short_estimation"] += int(short.sum())
    log["missing_event_day_return"] += int(miss.sum()); log["complex_action"] += int(cxx.sum())
    k = b["keep"]
    paths[e["event_id"]] = np.nanmean(b["path"][:, k], axis=1)
    rows.append(pd.DataFrame({"event_id": e["event_id"], "symbol": stocks[k], "sector": sector.values[k],
                              "beta": b["beta"][k], "size_log": b["size"][k], "car_0_1": b["car"][k]}))
se = pd.concat(rows, ignore_index=True)
lo, hi = se["car_0_1"].quantile([0.01, 0.99])
se["winsorized"] = (se["car_0_1"] < lo) | (se["car_0_1"] > hi)
se["car_w"] = se["car_0_1"].clip(lo, hi)
log["kept"] = len(se)
log["winsorized"] = int(se["winsorized"].sum())
log["winsor_bounds_pct"] = [round(lo, 3), round(hi, 3)]
log["duplicate_stock_dates"] = dup_rows
RESULTS["cleaning"] = log

evl = se.groupby("event_id").agg(caar=("car_w", "mean"), n_stocks=("car_w", "size"),
                                 share_up=("car_w", lambda x: (x > 0).mean())).reset_index()
ev = ev.merge(evl, on="event_id", how="left")
ev["abs_caar"] = ev["caar"].abs()


def market_car(t0):
    """NIFTY 50 CAR[0,+1] against its own estimation-window mean (constant-mean model)."""
    e0, e1 = max(t0 + EST_START, 0), t0 + EST_END
    mu = np.nanmean(Mv[e0:e1 + 1])
    return (Mv[t0] - mu) + (Mv[t0 + 1] - mu)


ev["mkt_car"] = [market_car(int(t)) for t in ev["t0"]]
ev["abs_mkt_car"] = ev["mkt_car"].abs()
ev[["event_id", "bank", "event_type", "release_date", "event_day0", "decision", "rate_before", "rate_after",
    "rate_change_bps", "rate_source", "surprise_bps", "vix_m1", "overlap_flag", "overlap_event",
    "n_stocks", "caar", "abs_caar", "share_up", "mkt_car"]].to_csv(OUT / "event_level_results.csv", index=False)
se.to_csv(OUT / "stock_event_car.csv.gz", index=False, compression="gzip")

# ------------------------------------------------------------------ 3. placebo days
event_t0 = set(ev["t0"].astype(int))
rng = np.random.default_rng(SEED)
cands = [t for t in range(-EST_START, N_DAYS - 2) if all(abs(t - x) > 2 for x in event_t0)]
plc = []
for t in rng.choice(cands, size=300, replace=False):
    b = event_block(int(t))
    c = np.clip(b["car"][b["keep"]], lo, hi)
    plc.append({"t0": int(t), "date": days[t].date(), "caar": float(np.mean(c)), "n": int(b["keep"].sum()),
                "mkt_car": float(market_car(int(t)))})
plc = pd.DataFrame(plc)
plc["abs_caar"] = plc["caar"].abs()
plc["abs_mkt_car"] = plc["mkt_car"].abs()
plc.to_csv(OUT / "placebo_days.csv", index=False)


# ------------------------------------------------------------------ 4. tests
def t_summary(x, label):
    x = pd.Series(x).dropna()
    t, p = stats.ttest_1samp(x, 0)
    ci = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    d = x.mean() / x.std(ddof=1)
    return {"group": label, "n": len(x), "mean_pp": x.mean(), "sd_pp": x.std(ddof=1), "t": t, "p": p,
            "ci_low": ci[0], "ci_high": ci[1], "cohen_d": d}


rq1 = [t_summary(ev["caar"], "All announcements")]
for bk in ["RBI", "FED"]:
    rq1.append(t_summary(ev.loc[ev["bank"] == bk, "caar"], f"{'Fed' if bk == 'FED' else 'RBI'}: all"))
    for ty, tl in [("DEC", "decisions"), ("MIN", "minutes")]:
        rq1.append(t_summary(ev.loc[(ev["bank"] == bk) & (ev["event_type"] == ty), "caar"],
                             f"{'Fed' if bk == 'FED' else 'RBI'}: {tl}"))
rq1.append(t_summary(ev.loc[ev["overlap_flag"] == 0, "caar"], "All, excluding overlapping events"))
rq1.append(t_summary(plc["caar"], "Placebo (300 random non-event days)"))
for bk, lab in [(None, "Market (NIFTY 50): all"), ("RBI", "Market (NIFTY 50): RBI"), ("FED", "Market (NIFTY 50): Fed")]:
    x = ev["mkt_car"] if bk is None else ev.loc[ev["bank"] == bk, "mkt_car"]
    rq1.append(t_summary(x, lab))
rq1.append(t_summary(plc["mkt_car"], "Market (NIFTY 50): placebo days"))
rq1 = pd.DataFrame(rq1)
rq1.to_csv(OUT / "rq1_tests.csv", index=False)

# magnitude: are announcement days larger than ordinary days?
mw = stats.mannwhitneyu(ev["abs_caar"].dropna(), plc["abs_caar"], alternative="greater")
mw_m = stats.mannwhitneyu(ev["abs_mkt_car"].dropna(), plc["abs_mkt_car"], alternative="greater")
RESULTS["rq1_magnitude_vs_placebo"] = {
    "median_abs_event": float(ev["abs_caar"].median()), "median_abs_placebo": float(plc["abs_caar"].median()),
    "mean_abs_event": float(ev["abs_caar"].mean()), "mean_abs_placebo": float(plc["abs_caar"].mean()),
    "mannwhitney_U": float(mw.statistic), "p_one_sided": float(mw.pvalue),
    "market_median_abs_event": float(ev["abs_mkt_car"].median()),
    "market_median_abs_placebo": float(plc["abs_mkt_car"].median()),
    "market_mw_p_one_sided": float(mw_m.pvalue)}


def welch(a, b, label):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return {"comparison": label, "n_fed": len(a), "n_rbi": len(b), "mean_fed": a.mean(), "mean_rbi": b.mean(),
            "diff": a.mean() - b.mean(), "t": t, "p": p, "cohen_d": (a.mean() - b.mean()) / sp}


F, Rb = ev["bank"] == "FED", ev["bank"] == "RBI"
D, Mn, NO = ev["event_type"] == "DEC", ev["event_type"] == "MIN", ev["overlap_flag"] == 0
rq2 = pd.DataFrame([
    welch(ev.loc[F, "abs_caar"], ev.loc[Rb, "abs_caar"], "All announcements"),
    welch(ev.loc[F & D, "abs_caar"], ev.loc[Rb & D, "abs_caar"], "Decisions only"),
    welch(ev.loc[F & Mn, "abs_caar"], ev.loc[Rb & Mn, "abs_caar"], "Minutes only"),
    welch(ev.loc[F & NO, "abs_caar"], ev.loc[Rb & NO, "abs_caar"], "Excluding overlapping events"),
    welch(ev.loc[F, "abs_mkt_car"], ev.loc[Rb, "abs_mkt_car"], "Market level (NIFTY 50 |CAR|)")])
rq2.to_csv(OUT / "rq2_tests.csv", index=False)

# RQ3 preliminary: surprise (Fed) and raw rate change (RBI decisions)
rq3 = []
fed = ev[F & ev["surprise_bps"].notna()]
for lab, d in [("Fed, all (surprise)", fed), ("Fed, decisions (surprise)", fed[fed["event_type"] == "DEC"])]:
    m = sm.OLS(d["caar"], sm.add_constant(d[["surprise_bps"]])).fit(cov_type="HC1")
    rq3.append({"model": lab, "n": int(m.nobs), "coef_per_bp": m.params["surprise_bps"],
                "se": m.bse["surprise_bps"], "p": m.pvalues["surprise_bps"], "r2": m.rsquared})
fd = ev[F & D & ev["surprise_bps"].notna()].copy()
fd["min"] = 0
pool = ev[F & ev["surprise_bps"].notna()].assign(minutes=lambda x: (x["event_type"] == "MIN").astype(int))
m = sm.OLS(pool["caar"], sm.add_constant(pool[["surprise_bps", "minutes"]])).fit(cov_type="HC1")
rq3.append({"model": "Fed, surprise + minutes indicator (reduced model, Fed part)", "n": int(m.nobs),
            "coef_per_bp": m.params["surprise_bps"], "se": m.bse["surprise_bps"],
            "p": m.pvalues["surprise_bps"], "r2": m.rsquared})
rd = ev[Rb & D & ev["rate_change_bps"].notna() & ev["caar"].notna()]
m = sm.OLS(rd["caar"], sm.add_constant(rd[["rate_change_bps"]])).fit(cov_type="HC1")
rq3.append({"model": "RBI, decisions (raw rate change, robustness)", "n": int(m.nobs),
            "coef_per_bp": m.params["rate_change_bps"], "se": m.bse["rate_change_bps"],
            "p": m.pvalues["rate_change_bps"], "r2": m.rsquared})
rq3 = pd.DataFrame(rq3)
rq3.to_csv(OUT / "rq3_preliminary.csv", index=False)

# within-event correlation (one-way ICC) for RQ4 power
def icc1(df, val):
    g = df.groupby("event_id")[val]
    k = g.size().mean()
    msb = (g.size() * (g.mean() - df[val].mean()) ** 2).sum() / (g.ngroups - 1)
    msw = ((df[val] - g.transform("mean")) ** 2).sum() / (len(df) - g.ngroups)
    return (msb - msw) / (msb + (k - 1) * msw)


pre = []
for _, e in ev.iterrows():
    b = event_block(int(e["t0"]))
    k = b["keep"]
    pre.append(pd.DataFrame({"event_id": e["event_id"], "ar_m5": b["path"][0][k]}))
pre = pd.concat(pre).dropna()
RESULTS["icc"] = {"car_0_1": float(icc1(se, "car_w")), "pre_event_ar_day_m5": float(icc1(pre, "ar_m5"))}

# ------------------------------------------------------------------ 5. RQ4 preliminary models
X = se.merge(ev[["event_id", "order", "bank", "event_type", "surprise_bps", "rate_change_bps", "vix_m1"]],
             on="event_id")
X = X[X["car_w"] != 0].copy()
X["y"] = (X["car_w"] > 0).astype(int)
X["is_fed"] = (X["bank"] == "FED").astype(int)
X["is_min"] = (X["event_type"] == "MIN").astype(int)
X["surprise_missing"] = X["surprise_bps"].isna().astype(int)
X["surprise_bps"] = X["surprise_bps"].fillna(0)
X["rate_change_bps"] = X["rate_change_bps"].fillna(0)
X["sector_code"] = X["sector"].astype("category").cat.codes
num = ["is_fed", "is_min", "surprise_bps", "surprise_missing", "rate_change_bps", "vix_m1", "beta", "size_log"]
n_train_events = int(round(0.70 * len(ev)))
train_ids = set(ev.loc[ev["order"] < n_train_events, "event_id"])
X["partition"] = np.where(X["event_id"].isin(train_ids), "TRAIN", "TEST")
tr, te = X[X["partition"] == "TRAIN"].dropna(subset=num), X[X["partition"] == "TEST"].dropna(subset=num)

maj = int(tr["y"].mean() >= 0.5)
p_maj = np.full(len(te), tr["y"].mean())
sec_tr = pd.get_dummies(tr["sector"], prefix="sec", dtype=float)
sec_te = pd.get_dummies(te["sector"], prefix="sec", dtype=float).reindex(columns=sec_tr.columns, fill_value=0)
sc = StandardScaler().fit(tr[num])
Xtr_l = np.hstack([sc.transform(tr[num]), sec_tr.values])
Xte_l = np.hstack([sc.transform(te[num]), sec_te.values])
logit = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr_l, tr["y"])
p_log = logit.predict_proba(Xte_l)[:, 1]
xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    random_state=SEED, eval_metric="logloss", n_jobs=4)
xgb.fit(tr[num + ["sector_code"]], tr["y"])
p_xgb = xgb.predict_proba(te[num + ["sector_code"]])[:, 1]


def metrics(y, p, label, thr=0.5):
    yhat = (p >= thr).astype(int)
    auc = roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5
    return {"model": label, "accuracy": accuracy_score(y, yhat), "balanced_accuracy": balanced_accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0), "roc_auc": auc, "brier": brier_score_loss(y, p)}


yte = te["y"].values
perf = pd.DataFrame([metrics(yte, p_maj, f"Majority class (always {'rise' if maj else 'fall'})"),
                     metrics(yte, p_log, "Logistic regression"), metrics(yte, p_xgb, "XGBoost")])


def mcnemar(y, pa, pb):
    a, b = (pa >= 0.5) == y, (pb >= 0.5) == y
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    chi = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if n01 + n10 else 0.0
    return n01, n10, chi, stats.chi2.sf(chi, 1)


def block_boot(df, pa, pb, reps=1000):
    ids = df["event_id"].values
    uniq = np.unique(ids)
    ca, cb = ((pa >= 0.5) == df["y"].values), ((pb >= 0.5) == df["y"].values)
    per = pd.DataFrame({"id": ids, "a": ca, "b": cb}).groupby("id").agg(["sum", "size"])
    gains = []
    r = np.random.default_rng(SEED)
    for _ in range(reps):
        s = r.choice(uniq, len(uniq), replace=True)
        sub = per.loc[s]
        gains.append(sub[("b", "sum")].sum() / sub[("b", "size")].sum()
                     - sub[("a", "sum")].sum() / sub[("a", "size")].sum())
    return np.percentile(gains, [2.5, 97.5])


comp = []
for name, base in [("Majority class", p_maj), ("Logistic regression", p_log)]:
    n01, n10, chi, p = mcnemar(yte, base, p_xgb)
    ci = block_boot(te, base, p_xgb)
    gain = accuracy_score(yte, (p_xgb >= .5).astype(int)) - accuracy_score(yte, (base >= .5).astype(int))
    comp.append({"baseline": name, "xgb_correct_only": n01, "baseline_correct_only": n10, "mcnemar_chi2": chi,
                 "p": p, "accuracy_gain": gain, "boot_ci_low": ci[0], "boot_ci_high": ci[1]})
comp = pd.DataFrame(comp)
perf.to_csv(OUT / "rq4_performance.csv", index=False)
comp.to_csv(OUT / "rq4_comparison.csv", index=False)
RESULTS["rq4_split"] = {"train_events": len(train_ids), "test_events": int(te["event_id"].nunique()),
                        "train_rows": len(tr), "test_rows": len(te), "train_share_up": float(tr["y"].mean()),
                        "test_share_up": float(te["y"].mean()),
                        "test_events_by_bank": te.groupby("bank")["event_id"].nunique().to_dict(),
                        "first_test_event": ev.loc[ev["order"] == n_train_events, "event_id"].iloc[0]}
imp = pd.Series(xgb.feature_importances_, index=num + ["sector_code"]).sort_values(ascending=False)
imp.to_csv(OUT / "rq4_xgb_importance.csv")

# ------------------------------------------------------------------ 6. figures
# Figure: policy rates
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.step(bis["RBI"].index, bis["RBI"].values, where="post", color=C_RBI, label="RBI policy repo rate")
ax.step(bis["FED"].index, bis["FED"].values, where="post", color=C_FED, label="Fed policy rate (BIS series)")
for _, e in ev[D].iterrows():
    ax.axvline(e["release_date"], color=C_RBI if e["bank"] == "RBI" else C_FED, alpha=0.12, lw=0.8)
ax.set_ylabel("Per cent per year"); ax.legend(frameon=False, loc="lower right")
save(fig, "fig_policy_rates.png")

# Figure: CAAR per event over time
fig, ax = plt.subplots(figsize=(7, 3.2))
for bk, col in [("RBI", C_RBI), ("FED", C_FED)]:
    d = ev[ev["bank"] == bk]
    ax.bar(d["event_day0"], d["caar"], width=6, color=col, label="RBI" if bk == "RBI" else "Fed")
ax.axhline(0, color="black", lw=0.6)
ax.set_ylabel("CAAR[0,+1], percentage points"); ax.legend(frameon=False)
save(fig, "fig_caar_timeline.png")

# Figure: box plot by bank and type
fig, ax = plt.subplots(figsize=(6, 3.2))
groups = [("RBI", "DEC"), ("RBI", "MIN"), ("FED", "DEC"), ("FED", "MIN")]
data = [ev.loc[(ev["bank"] == b) & (ev["event_type"] == t), "caar"].dropna() for b, t in groups]
bp = ax.boxplot(data, patch_artist=True, widths=0.55)
for patch, (b, _) in zip(bp["boxes"], groups):
    patch.set_facecolor(C_RBI if b == "RBI" else C_FED); patch.set_alpha(0.55)
ax.set_xticks(range(1, 5), ["RBI\ndecisions", "RBI\nminutes", "Fed\ndecisions", "Fed\nminutes"])
ax.axhline(0, color="black", lw=0.6); ax.set_ylabel("CAAR[0,+1], percentage points")
save(fig, "fig_caar_box.png")

# Figure: average AR path
fig, ax = plt.subplots(figsize=(6, 3.2))
for bk, col, lab in [("RBI", C_RBI, "RBI"), ("FED", C_FED, "Fed")]:
    P = np.vstack([paths[i] for i in ev.loc[ev["bank"] == bk, "event_id"]])
    m_, s_ = P.mean(0), P.std(0, ddof=1) / np.sqrt(len(P))
    ax.plot(list(PATH_DAYS), m_, marker="o", color=col, label=lab)
    ax.fill_between(list(PATH_DAYS), m_ - 1.96 * s_, m_ + 1.96 * s_, color=col, alpha=0.15)
ax.axhline(0, color="black", lw=0.6); ax.axvspan(-0.3, 1.3, color="grey", alpha=0.12)
ax.set_xlabel("Trading day relative to announcement (day 0)"); ax.set_ylabel("Average abnormal return, pp")
ax.legend(frameon=False)
save(fig, "fig_ar_path.png")

# Figure: |CAAR| events vs placebo
fig, ax = plt.subplots(figsize=(6, 3.2))
bins = np.linspace(0, max(ev["abs_caar"].max(), plc["abs_caar"].max()), 25)
ax.hist(plc["abs_caar"], bins=bins, density=True, alpha=0.5, color="grey", label="Random non-announcement days")
ax.hist(ev["abs_caar"], bins=bins, density=True, alpha=0.6, color=C_FED, label="Announcement days")
ax.set_xlabel("|CAAR[0,+1]|, percentage points"); ax.set_ylabel("Density"); ax.legend(frameon=False)
save(fig, "fig_abs_caar_placebo.png")

# Figure: Fed surprise vs CAAR
fig, ax = plt.subplots(figsize=(5.5, 3.4))
for ty, mk in [("DEC", "o"), ("MIN", "^")]:
    d = fed[fed["event_type"] == ty]
    ax.scatter(d["surprise_bps"], d["caar"], marker=mk, color=C_FED, alpha=0.75,
               label="Statements" if ty == "DEC" else "Minutes")
b_ = np.polyfit(fed["surprise_bps"], fed["caar"], 1)
xs = np.linspace(fed["surprise_bps"].min(), fed["surprise_bps"].max(), 50)
ax.plot(xs, np.polyval(b_, xs), color="black", lw=1)
ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
ax.set_xlabel("Fed surprise: change in U.S. 2-year yield, bps"); ax.set_ylabel("CAAR[0,+1], pp")
ax.legend(frameon=False)
save(fig, "fig_fed_surprise.png")

# Figure: sector heatmap
sec_tab = se.merge(ev[["event_id", "bank"]], on="event_id").groupby(["sector", "bank"])["car_w"].mean().unstack()
sec_tab = sec_tab.reindex(sec_tab.abs().max(axis=1).sort_values(ascending=False).index)
fig, ax = plt.subplots(figsize=(5.5, 6))
v = np.nanmax(np.abs(sec_tab.values))
im = ax.imshow(sec_tab[["RBI", "FED"]].values, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
ax.set_xticks([0, 1], ["RBI", "Fed"]); ax.set_yticks(range(len(sec_tab)), sec_tab.index, fontsize=8)
for i in range(len(sec_tab)):
    for j, c in enumerate(["RBI", "FED"]):
        ax.text(j, i, f"{sec_tab[c].iloc[i]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, label="Mean CAR[0,+1], pp", shrink=0.6)
save(fig, "fig_sector_heatmap.png")
sec_tab.to_csv(OUT / "sector_mean_car.csv")

# ------------------------------------------------------------------ 7. descriptive numbers
RESULTS["data"] = {
    "trading_days": N_DAYS, "first_day": str(days[0].date()), "last_day": str(days[-1].date()),
    "stocks": N_STOCKS, "events": len(ev), "price_rows": len(adj),
    "events_by_bank_type": {f"{a}_{b}": int(v) for (a, b), v in ev.groupby(["bank", "event_type"]).size().items()},
    "decisions": {f"{a}_{b}": int(v) for (a, b), v in ev.groupby(["bank", "decision"]).size().items()},
    "rate_source": ev.loc[D, "rate_source"].value_counts().to_dict(),
    "fed_surprise_available": int(ev.loc[F, "surprise_bps"].notna().sum()),
    "overlap_events": int(ev["overlap_flag"].sum()),
    "stock_events": len(se), "stocks_per_event_mean": float(ev["n_stocks"].mean()),
    "stocks_per_event_min": int(ev["n_stocks"].min()), "stocks_per_event_max": int(ev["n_stocks"].max()),
    "car_raw_mean": float(se["car_0_1"].mean()), "car_raw_sd": float(se["car_0_1"].std()),
    "car_raw_min": float(se["car_0_1"].min()), "car_raw_max": float(se["car_0_1"].max()),
    "caar_sd": float(ev["caar"].std()), "share_up_mean": float(ev["share_up"].mean()),
    "beta_median": float(se["beta"].median()), "vix_mean": float(ev["vix_m1"].mean()),
    "largest_pos": ev.nlargest(3, "caar")[["event_id", "caar"]].round(3).values.tolist(),
    "largest_neg": ev.nsmallest(3, "caar")[["event_id", "caar"]].round(3).values.tolist()}
with open(OUT / "results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

pd.set_option("display.width", 200)
print(json.dumps(RESULTS, indent=1, default=str))
print(rq1.round(4).to_string(index=False)); print(rq2.round(4).to_string(index=False))
print(rq3.round(5).to_string(index=False)); print(perf.round(4).to_string(index=False))
print(comp.round(4).to_string(index=False)); print(imp.round(3).to_string())
print(sec_tab.round(3).to_string())

# ------------------------------------------------------------------ 8. confusion counts (for worked metric example)
def confusion(y, p):
    yhat = (p >= 0.5).astype(int)
    return {"TP": int(((yhat == 1) & (y == 1)).sum()), "TN": int(((yhat == 0) & (y == 0)).sum()),
            "FP": int(((yhat == 1) & (y == 0)).sum()), "FN": int(((yhat == 0) & (y == 1)).sum())}


RESULTS["confusion"] = {"xgboost": confusion(yte, p_xgb), "logistic": confusion(yte, p_log)}
RESULTS["fed_market_level_surprise"] = {}
_m = sm.OLS(fed["mkt_car"], sm.add_constant(fed[["surprise_bps"]])).fit(cov_type="HC1")
RESULTS["fed_market_level_surprise"] = {"coef": float(_m.params["surprise_bps"]), "p": float(_m.pvalues["surprise_bps"]),
                                        "r2": float(_m.rsquared), "n": int(_m.nobs)}
with open(OUT / "results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(RESULTS["confusion"], RESULTS["fed_market_level_surprise"])
