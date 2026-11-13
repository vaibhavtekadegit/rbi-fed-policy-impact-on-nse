"""
Final analysis: RQ1-RQ4 with policy tone, robustness checks, and the RQ4 prediction model.

Reads only repository files (prices, events, tone scores from src/tone/classifier.py) and writes every
table, figure, model and number used in the final report to reports/final/analysis/. Run from the repo root:

    python src/analysis/final_analysis.py

Sections
 1. Load data; market model per stock and announcement (estimation days -130..-11, >= 100 days).
 2. RQ1  H0: mean CAAR[0,+1] = 0. Event-level t test; BMP standardized test with the Kolari-Pynnonen
         cross-correlation adjustment; stock-level regression with two-way (event, stock) clustered errors.
 3. RQ2  H0: mean |CAAR| equal for Fed and RBI. Welch, permutation, Holm, Hedges g, bootstrap CI, clustered.
 4. RQ3  H0: policy information (Fed 2-year surprise, RBI rate change) and tone do not explain CAAR.
         Reduced vs full model, partial F and robust Wald tests; Fed-only and RBI-only tone checks; stock level.
 5. RQ4  H0: a model with policy, tone and stock features does not beat the majority class for the sign of
         CAR[0,+1]. Chronological 70/30 event split; expanding-window tuning; logistic, XGBoost, MLP;
         train and test metrics; ROC, confusion, calibration, SHAP; tone ablation; McNemar; block bootstrap.
 6. Robustness: windows, overlap filters (cross-bank, macro), no winsorization, LLM tone, earnings filter.
 7. Figures and run metadata (CPU/GPU, versions, runtime).
"""
import json
import os
import platform
import subprocess
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, DMatrix

T_START = time.time()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/final/analysis"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)
SEED = 42
EST_START, EST_END, MIN_EST = -130, -11, 100
PATH_DAYS = list(range(-5, 6))                     # index 5 = day 0
WINDOWS = {"[0,+1]": [5, 6], "[-1,+1]": [4, 5, 6], "[0,+5]": [5, 6, 7, 8, 9, 10]}
B_REPS = 2000
QUICK = os.environ.get("QUICK") == "1"          # smoke test: tiny grids and resamples
if QUICK:
    B_REPS = 50
RES = {}
C_RBI, C_FED = "#1f4e79", "#c55a11"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def jdump(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ============================================================ 1. data and event study
adj = pd.read_csv(ROOT / "data/interim/prices_adjusted.csv.gz", parse_dates=["date"],
                  usecols=["date", "symbol_current", "ret", "value", "complex_action"])
adj["ret_pct"] = adj["ret"] * 100
R = adj.pivot(index="date", columns="symbol_current", values="ret_pct").sort_index()
CX = adj.pivot(index="date", columns="symbol_current", values="complex_action").reindex(R.index).fillna(False).astype(bool)
VAL = adj.pivot(index="date", columns="symbol_current", values="value").reindex(R.index)
days, stocks = R.index, R.columns
N_DAYS, N_STOCKS = len(days), len(stocks)
nifty = pd.read_csv(ROOT / "data/raw/prices/nifty50_daily.csv", parse_dates=["date"]).set_index("date")
M = (nifty["close"].pct_change() * 100).reindex(days)
vix = pd.read_csv(ROOT / "data/raw/prices/india_vix_daily.csv", parse_dates=["date"]).set_index("date")["close"].reindex(days)
sector = pd.read_csv(ROOT / "data/raw/prices/validation_report.csv").set_index("symbol")["industry"].reindex(stocks)

ev = pd.read_csv(ROOT / "data/interim/events.csv", parse_dates=["release_date", "event_day0"])
tone = pd.read_csv(ROOT / "data/interim/tone_scores.csv")
ev = ev.merge(tone[["event_id", "tone_score", "tone_score_llm", "n_sentences"]], on="event_id", how="left")
assert ev["tone_score"].notna().all(), "tone_scores.csv must cover all 142 events (run notebook 06 first)"
ev = ev.sort_values(["release_date", "bank", "event_type"]).reset_index(drop=True)
ev["order"] = np.arange(len(ev))
day_pos = {d: i for i, d in enumerate(days)}
ev["t0"] = ev["event_day0"].map(day_pos).astype(int)
ev["vix_m1"] = [vix.iloc[t - 1] for t in ev["t0"]]
ev["is_fed"] = (ev["bank"] == "FED").astype(int)
ev["is_min"] = (ev["event_type"] == "MIN").astype(int)
ev["fed_surprise"] = np.where(ev["is_fed"] == 1, ev["surprise_bps"], 0.0)          # NaN for the one Fed event without DGS2
ev["rbi_rate_change"] = np.where((ev["bank"] == "RBI") & (ev["event_type"] == "DEC"), ev["rate_change_bps"].fillna(0), 0.0)
ev["tone_x_fed"] = ev["tone_score"] * ev["is_fed"]
ev["overlap_cross_bank"] = ev["overlap_event"].notna().astype(int)
ev["overlap_macro_flag"] = ev["overlap_macro"].notna().astype(int)
EV_START = ev[["event_id", "bank", "event_type", "order"]]

Rv, Mv, CXv, VALv = R.values, M.values, CX.values, VAL.values


def event_block(t0):
    """Market model per stock for one day 0: AR path -5..+5, residual sd, size, keep mask, residual cross-correlation."""
    e0, e1 = max(t0 + EST_START, 0), t0 + EST_END
    Y, X = Rv[e0:e1 + 1], Mv[e0:e1 + 1][:, None]
    mask = ~np.isnan(Y) & ~np.isnan(X)
    n = mask.sum(0)
    Xm, Ym = np.where(mask, X, 0.0), np.where(mask, Y, 0.0)
    sx, sy, sxx, sxy = Xm.sum(0), Ym.sum(0), (Xm ** 2).sum(0), (Xm * Ym).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
        alpha = (sy - beta * sx) / n
        resid = np.where(mask, Y - (alpha + beta * X), np.nan)
        s = np.sqrt(np.nansum(resid ** 2, 0) / (n - 2))
    path = np.full((len(PATH_DAYS), Y.shape[1]), np.nan)
    for k, d in enumerate(PATH_DAYS):
        if 0 <= t0 + d < N_DAYS:
            path[k] = Rv[t0 + d] - (alpha + beta * Mv[t0 + d])
    cx = CXv[max(t0 - 1, 0):t0 + 2].any(0)
    size = np.log(np.nanmean(np.where(VALv[e0:e1 + 1] > 0, VALv[e0:e1 + 1], np.nan), axis=0))
    keep = (n >= MIN_EST) & ~np.isnan(path[5]) & ~np.isnan(path[6]) & ~cx
    return dict(beta=beta, s=s, n=n, path=path, size=size, keep=keep, resid=resid)


def mean_cross_corr(resid):
    """Average pairwise correlation of estimation-window residuals (Kolari-Pynnonen r-bar)."""
    m = ~np.isnan(resid)
    z = np.where(m, (resid - np.nanmean(resid, 0)) / np.nanstd(resid, 0), 0.0)
    num, cnt = z.T @ z, m.T.astype(float) @ m.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = num / cnt
    iu = np.triu_indices_from(c, 1)
    v = c[iu]
    return float(np.nanmean(v[cnt[iu] >= 60]))


rows = []
paths = {}
for _, e in ev.iterrows():
    b = event_block(e["t0"])
    k = b["keep"]
    paths[e["event_id"]] = np.nanmean(b["path"][:, k], axis=1)
    d = {"event_id": e["event_id"], "symbol": stocks[k], "sector": sector.values[k], "beta": b["beta"][k],
         "size_log": b["size"][k], "resid_sd": b["s"][k], "pre_car": np.nansum(b["path"][0:5, k], 0),
         "rbar": mean_cross_corr(b["resid"][:, k])}
    for w, idx in WINDOWS.items():
        d[f"car{w}"] = b["path"][idx][:, k].sum(0)
    rows.append(pd.DataFrame(d))
se = pd.concat(rows, ignore_index=True)
se["car_raw"] = se["car[0,+1]"]
for w in WINDOWS:                                            # winsorize each window at its own 1st/99th percentile
    lo_, hi_ = se[f"car{w}"].quantile([0.01, 0.99])
    se[f"car{w}"] = se[f"car{w}"].clip(lo_, hi_)
se["car"] = se["car[0,+1]"]
se["scar"] = se["car"] / (se["resid_sd"] * np.sqrt(2))      # standardized CAR (BMP)
RES["sample"] = {"stock_events": len(se), "events": int(se["event_id"].nunique()), "stocks": int(se["symbol"].nunique()),
                 "stocks_per_event_mean": float(se.groupby("event_id").size().mean()),
                 "winsor_bounds_0_1": [float(v) for v in se["car_raw"].quantile([0.01, 0.99])]}

agg = {f"caar{w}": (f"car{w}", "mean") for w in WINDOWS}
evl = se.groupby("event_id").agg(n_stocks=("car", "size"), caar=("car", "mean"), caar_raw=("car_raw", "mean"),
                                 share_up=("car", lambda x: (x > 0).mean()), rbar=("rbar", "first"),
                                 scar_mean=("scar", "mean"), scar_sd=("scar", "std"), **agg).reset_index()
ev = ev.merge(evl, on="event_id", how="left")
ev["abs_caar"] = ev["caar"].abs()


def market_car(t0):
    e0, e1 = max(t0 + EST_START, 0), t0 + EST_END
    mu = np.nanmean(Mv[e0:e1 + 1])
    return (Mv[t0] - mu) + (Mv[t0 + 1] - mu)


ev["mkt_car"] = [market_car(t) for t in ev["t0"]]
ev["nifty_pre5"] = [np.nansum(Mv[t - 5:t]) for t in ev["t0"]]
# BMP z per event, then the Kolari-Pynnonen adjustment for cross-sectional correlation of residuals
n_ = ev["n_stocks"]
ev["bmp_z"] = ev["scar_mean"] * np.sqrt(n_) / ev["scar_sd"]
ev["kp_z"] = ev["bmp_z"] * np.sqrt((1 - ev["rbar"]) / (1 + (n_ - 1) * ev["rbar"]))
ev.drop(columns=["scar_mean", "scar_sd"]).to_csv(OUT / "event_level.csv", index=False)
se.to_csv(OUT / "stock_event_car.csv.gz", index=False, compression="gzip")

# placebo days (same procedure on 300 random non-announcement days)
rng = np.random.default_rng(SEED)
event_t0 = set(ev["t0"])
cands = [t for t in range(-EST_START, N_DAYS - 6) if all(abs(t - x) > 2 for x in event_t0)]
plc = []
lo, hi = RES["sample"]["winsor_bounds_0_1"]
for t in rng.choice(cands, size=30 if QUICK else 300, replace=False):
    b = event_block(int(t))
    c = np.clip((b["path"][5] + b["path"][6])[b["keep"]], lo, hi)
    plc.append({"t0": int(t), "date": days[t].date(), "caar": float(np.mean(c)), "mkt_car": float(market_car(int(t)))})
plc = pd.DataFrame(plc)
plc["abs_caar"] = plc["caar"].abs()
plc.to_csv(OUT / "placebo_days.csv", index=False)

# groups used throughout
GROUPS = {"All announcements": ev["bank"].notna(), "RBI: all": ev["bank"] == "RBI", "RBI: decisions": (ev["bank"] == "RBI") & (ev["is_min"] == 0),
          "RBI: minutes": (ev["bank"] == "RBI") & (ev["is_min"] == 1), "Fed: all": ev["bank"] == "FED",
          "Fed: decisions": (ev["bank"] == "FED") & (ev["is_min"] == 0), "Fed: minutes": (ev["bank"] == "FED") & (ev["is_min"] == 1)}


def boot_ci(stat, groups, reps=B_REPS):
    vals, r_ = [], np.random.default_rng(SEED)
    for _ in range(reps):
        vals.append(stat(*[g[r_.integers(0, len(g), len(g))] for g in groups]))
    return np.nanpercentile(vals, [2.5, 97.5])


def clustered_mean(df, y):
    """Mean of y with standard errors clustered by event and by stock (two-way)."""
    g = np.column_stack([pd.factorize(df["event_id"])[0], pd.factorize(df["symbol"])[0]])
    m = sm.OLS(df[y].values, np.ones(len(df))).fit(cov_type="cluster", cov_kwds={"groups": g})
    return m.params[0], m.bse[0], m.pvalues[0]


# ============================================================ 2. RQ1
rq1 = []
for lab, mask in GROUPS.items():
    x = ev.loc[mask, "caar"]
    t, p = stats.ttest_1samp(x, 0)
    ci = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    kz = ev.loc[mask, "kp_z"]
    z_kp = kz.sum() / np.sqrt(len(kz))                         # Stouffer combination of event-level KP statistics
    sub = se[se["event_id"].isin(ev.loc[mask, "event_id"])]
    cm, cse, cp = clustered_mean(sub, "car")
    rq1.append({"group": lab, "K_events": len(x), "N_stock_events": len(sub), "caar_pp": x.mean(), "sd_pp": x.std(ddof=1),
                "ci_low": ci[0], "ci_high": ci[1], "t": t, "p_t": p, "cohen_d": x.mean() / x.std(ddof=1),
                "bmp_kp_z": z_kp, "p_bmp_kp": 2 * stats.norm.sf(abs(z_kp)),
                "share_events_sig_kp": float((kz.abs() > 1.96).mean()),
                "clustered_mean_pp": cm, "clustered_se": cse, "p_clustered": cp,
                "mean_rbar": ev.loc[mask, "rbar"].mean()})
for lab, x in [(f"Placebo days ({len(plc)})", plc["caar"]), ("Market NIFTY 50: all", ev["mkt_car"]), ("Market NIFTY 50: placebo", plc["mkt_car"])]:
    t, p = stats.ttest_1samp(x, 0)
    rq1.append({"group": lab, "K_events": len(x), "caar_pp": x.mean(), "sd_pp": x.std(ddof=1), "t": t, "p_t": p,
                "cohen_d": x.mean() / x.std(ddof=1)})
rq1 = pd.DataFrame(rq1)
rq1.to_csv(OUT / "rq1_tests.csv", index=False)
mw = stats.mannwhitneyu(ev["abs_caar"], plc["abs_caar"], alternative="greater")
RES["rq1_magnitude_vs_placebo"] = {"median_abs_event": ev["abs_caar"].median(), "median_abs_placebo": plc["abs_caar"].median(),
                                   "mannwhitney_p_one_sided": mw.pvalue}

# ============================================================ 3. RQ2
F_, Rb_ = ev["bank"] == "FED", ev["bank"] == "RBI"
DEC_, MIN_ = ev["is_min"] == 0, ev["is_min"] == 1


def compare(a, b, label):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    g = (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (n1 + n2) - 9))
    seg = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    pool, obs, r_ = np.concatenate([a, b]), a.mean() - b.mean(), np.random.default_rng(SEED)
    perm = np.array([(lambda s: s[:n1].mean() - s[n1:].mean())(r_.permutation(pool)) for _ in range(200 if QUICK else 10000)])
    ci = boot_ci(lambda x, y: x.mean() - y.mean(), [a, b])
    return {"comparison": label, "n_fed": n1, "n_rbi": n2, "mean_fed": a.mean(), "mean_rbi": b.mean(), "diff": obs,
            "boot_ci_low": ci[0], "boot_ci_high": ci[1], "welch_t": t, "p_welch": p,
            "p_permutation": (np.sum(np.abs(perm) >= abs(obs)) + 1) / (len(perm) + 1),
            "p_mannwhitney": stats.mannwhitneyu(a, b).pvalue, "hedges_g": g, "g_ci_low": g - 1.96 * seg, "g_ci_high": g + 1.96 * seg}


rq2 = pd.DataFrame([compare(ev.loc[F_, "abs_caar"], ev.loc[Rb_, "abs_caar"], "All announcements"),
                    compare(ev.loc[F_ & DEC_, "abs_caar"], ev.loc[Rb_ & DEC_, "abs_caar"], "Decisions"),
                    compare(ev.loc[F_ & MIN_, "abs_caar"], ev.loc[Rb_ & MIN_, "abs_caar"], "Minutes"),
                    compare(ev.loc[F_ & (ev["overlap_flag"] == 0), "abs_caar"], ev.loc[Rb_ & (ev["overlap_flag"] == 0), "abs_caar"],
                            "Excluding overlapping events")])
order = np.argsort(rq2["p_welch"].values)
holm, run_ = np.empty(len(rq2)), 0
for rank, i in enumerate(order):
    run_ = max(run_, min(1, (len(rq2) - rank) * rq2["p_welch"].values[i]))
    holm[i] = run_
rq2["p_holm"] = holm
rq2.to_csv(OUT / "rq2_tests.csv", index=False)
# event-level interaction model and stock-level two-way clustered model for |CAR|
m_int = smf.ols("abs_caar ~ is_fed * is_min", data=ev).fit(cov_type="HC1")
se2 = se.merge(ev[["event_id", "is_fed", "is_min"]], on="event_id")
se2["abs_car"] = se2["car"].abs()
g2 = np.column_stack([pd.factorize(se2["event_id"])[0], pd.factorize(se2["symbol"])[0]])
m_stock = smf.ols("abs_car ~ is_fed * is_min", data=se2).fit(cov_type="cluster", cov_kwds={"groups": g2})
RES["rq2_models"] = {"event_interaction": {k: {"coef": m_int.params[k], "p": m_int.pvalues[k]} for k in m_int.params.index},
                     "stock_two_way_clustered": {k: {"coef": m_stock.params[k], "p": m_stock.pvalues[k]} for k in m_stock.params.index}}

# ============================================================ 4. RQ3
REDUCED = "caar ~ is_fed + is_min + fed_surprise + rbi_rate_change + vix_m1"
FULL = REDUCED + " + tone_score + tone_x_fed"
d3 = ev.dropna(subset=["fed_surprise"]).copy()
fits = {"Reduced (policy information)": smf.ols(REDUCED, d3).fit(cov_type="HC1"),
        "Full (policy information + tone)": smf.ols(FULL, d3).fit(cov_type="HC1")}
plain_r, plain_f = smf.ols(REDUCED, d3).fit(), smf.ols(FULL, d3).fit()
f_tone = plain_f.compare_f_test(plain_r)                                        # classic partial F
wald_tone = fits["Full (policy information + tone)"].wald_test("tone_score = 0, tone_x_fed = 0", use_f=True, scalar=True)
wald_info = fits["Full (policy information + tone)"].wald_test(
    "fed_surprise = 0, rbi_rate_change = 0, tone_score = 0, tone_x_fed = 0", use_f=True, scalar=True)
fed_d, rbi_d = d3[d3["is_fed"] == 1], d3[d3["is_fed"] == 0]
fits["Fed only: surprise + tone"] = smf.ols("caar ~ is_min + fed_surprise + tone_score + vix_m1", fed_d).fit(cov_type="HC1")
fits["RBI only: rate change + tone"] = smf.ols("caar ~ is_min + rbi_rate_change + tone_score + vix_m1", rbi_d).fit(cov_type="HC1")
fits["Full, tone from LLM labels"] = smf.ols(FULL.replace("tone_score", "tone_llm").replace("tone_x_fed", "tone_llm_x_fed"),
                                             d3.assign(tone_llm=d3["tone_score_llm"], tone_llm_x_fed=d3["tone_score_llm"] * d3["is_fed"])).fit(cov_type="HC1")
fits["Full, market level (NIFTY 50 CAR)"] = smf.ols(FULL.replace("caar ~", "mkt_car ~"), d3).fit(cov_type="HC1")
# stock level with two-way clustering and stock controls
s3 = se.merge(d3[["event_id", "is_fed", "is_min", "fed_surprise", "rbi_rate_change", "vix_m1", "tone_score", "tone_x_fed"]], on="event_id")
g3 = np.column_stack([pd.factorize(s3["event_id"])[0], pd.factorize(s3["symbol"])[0]])
fits["Full, stock level (two-way clustered)"] = smf.ols(FULL.replace("caar ~", "car ~") + " + beta + size_log", s3).fit(
    cov_type="cluster", cov_kwds={"groups": g3})
rq3 = []
for name, m in fits.items():
    for term in m.params.index:
        ci = m.conf_int().loc[term]
        rq3.append({"model": name, "term": term, "coef": m.params[term], "se": m.bse[term], "ci_low": ci[0], "ci_high": ci[1],
                    "p": m.pvalues[term], "n": int(m.nobs), "r2": m.rsquared, "adj_r2": m.rsquared_adj})
rq3 = pd.DataFrame(rq3)
rq3.to_csv(OUT / "rq3_models.csv", index=False)
RES["rq3_tests"] = {"n_events": len(d3),
                    "partial_F_tone": {"F": f_tone[0], "p": f_tone[1], "df_diff": f_tone[2]},
                    "robust_wald_tone": {"F": float(wald_tone.statistic), "p": float(wald_tone.pvalue)},
                    "robust_wald_all_policy_information": {"F": float(wald_info.statistic), "p": float(wald_info.pvalue)},
                    "r2_reduced": plain_r.rsquared, "r2_full": plain_f.rsquared,
                    "adj_r2_reduced": plain_r.rsquared_adj, "adj_r2_full": plain_f.rsquared_adj,
                    "tone_sd": float(ev["tone_score"].std()),
                    "tone_by_group": ev.groupby(["bank", "event_type"])["tone_score"].agg(["mean", "std"]).round(4).reset_index().to_dict("records"),
                    "corr_tone_surprise_fed": float(fed_d["tone_score"].corr(fed_d["fed_surprise"])),
                    "corr_tone_rate_change_rbi_dec": float(rbi_d.loc[rbi_d["is_min"] == 0, "tone_score"].corr(
                        rbi_d.loc[rbi_d["is_min"] == 0, "rbi_rate_change"]))}

# ============================================================ 5. RQ4 prediction
X = se.merge(ev[["event_id", "order", "is_fed", "is_min", "fed_surprise", "rbi_rate_change", "vix_m1", "tone_score",
                 "tone_x_fed", "nifty_pre5"]], on="event_id")
X = X[X["car"] != 0].copy()
X["y"] = (X["car"] > 0).astype(int)
X["fed_surprise_missing"] = X["fed_surprise"].isna().astype(int)
X["fed_surprise"] = X["fed_surprise"].fillna(0)
NUM = ["is_fed", "is_min", "fed_surprise", "fed_surprise_missing", "rbi_rate_change", "vix_m1", "nifty_pre5",
       "tone_score", "tone_x_fed", "beta", "size_log", "resid_sd", "pre_car"]
TONE_FEATS = ["tone_score", "tone_x_fed"]
X = X.dropna(subset=NUM)
n_train_events = int(round(0.70 * len(ev)))
X["partition"] = np.where(X["order"] < n_train_events, "TRAIN", "TEST")
tr, te = X[X["partition"] == "TRAIN"].reset_index(drop=True), X[X["partition"] == "TEST"].reset_index(drop=True)
SECTORS = sorted(tr["sector"].dropna().unique())


def design(df, feats, scaler=None):
    sec = pd.get_dummies(pd.Categorical(df["sector"], categories=SECTORS), prefix="sec", dtype=float)
    num = df[feats].values.astype(float)
    if scaler is None:
        scaler = StandardScaler().fit(num)
    return np.hstack([scaler.transform(num), sec.values]), scaler, feats + list(sec.columns)


def expanding_folds(df, k=4):
    """Chronological blocks of training events: fit on blocks 1..j, validate on block j+1 (j = 1..k-1)."""
    ev_order = np.sort(df["order"].unique())
    blocks = np.array_split(ev_order, k)
    for j in range(1, k):
        fit_ev, val_ev = np.concatenate(blocks[:j]), blocks[j]
        yield np.where(df["order"].isin(fit_ev))[0], np.where(df["order"].isin(val_ev))[0]


FAMILIES = {
    "Logistic regression": (lambda p: LogisticRegression(max_iter=3000, C=p["C"]), [{"C": c} for c in (0.01, 0.1, 1, 10)]),
    "XGBoost": (lambda p: XGBClassifier(**p, subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=SEED,
                                        eval_metric="logloss", n_jobs=-1),
                [{"max_depth": d, "n_estimators": n, "learning_rate": lr, "min_child_weight": w}
                 for d in (2, 3, 4) for n in (200, 500) for lr in (0.03, 0.1) for w in (1, 20)]),
    "Neural network (MLP)": (lambda p: MLPClassifier(hidden_layer_sizes=p["h"], alpha=p["alpha"], early_stopping=True,
                                                     max_iter=200, random_state=SEED),
                             [{"h": h, "alpha": a} for h in ((32,), (64, 32)) for a in (1e-4, 1e-2)]),
}


if QUICK:
    FAMILIES = {k: (v[0], v[1][:1]) for k, v in FAMILIES.items()}


def tune(feats):
    rows_, best = [], {}
    for fam, (make, grid) in FAMILIES.items():
        for params in grid:
            aucs = []
            for fi, vi in expanding_folds(tr):
                Xf, sc, _ = design(tr.iloc[fi], feats)
                Xv, _, _ = design(tr.iloc[vi], feats, sc)
                m = make(params).fit(Xf, tr["y"].values[fi])
                aucs.append(roc_auc_score(tr["y"].values[vi], m.predict_proba(Xv)[:, 1]))
            rows_.append({"family": fam, "params": json.dumps(params), "val_auc_mean": np.mean(aucs), "val_auc_sd": np.std(aucs),
                          "val_auc_folds": json.dumps([round(a, 4) for a in aucs])})
            if fam not in best or np.mean(aucs) > best[fam][1]:
                best[fam] = (params, np.mean(aucs))
    return pd.DataFrame(rows_), best


def metrics(y, p, label, part):
    yhat = (p >= 0.5).astype(int)
    return {"model": label, "partition": part, "n": len(y), "accuracy": accuracy_score(y, yhat),
            "balanced_accuracy": balanced_accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0), "f1": f1_score(y, yhat, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5, "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))}


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


t_tune = time.time()
tuning, best = tune(NUM)
tuning.to_csv(OUT / "rq4_tuning.csv", index=False)
RES["rq4_tuning_seconds"] = time.time() - t_tune
ytr, yte = tr["y"].values, te["y"].values
Xtr, SC, FEAT_NAMES = design(tr, NUM)
Xte, _, _ = design(te, NUM, SC)
preds, perf, fitted = {}, [], {}
p_major = float(ytr.mean())
preds["Majority class"] = (np.full(len(ytr), p_major), np.full(len(yte), p_major))
for fam, (make, _) in FAMILIES.items():
    m = make(best[fam][0]).fit(Xtr, ytr)
    fitted[fam] = m
    preds[fam] = (m.predict_proba(Xtr)[:, 1], m.predict_proba(Xte)[:, 1])
for name, (ptr, pte) in preds.items():
    perf += [metrics(ytr, ptr, name, "TRAIN"), metrics(yte, pte, name, "TEST")]
perf = pd.DataFrame(perf)
perf["ece"] = [ece(ytr if r.partition == "TRAIN" else yte, preds[r.model][0 if r.partition == "TRAIN" else 1]) for r in perf.itertuples()]
perf["val_auc"] = perf["model"].map({f: v[1] for f, v in best.items()})
perf["best_params"] = perf["model"].map({f: json.dumps(v[0]) for f, v in best.items()})
BEST = max(best, key=lambda f: best[f][1])                    # chosen on validation AUC, never on the test set
perf["selected"] = perf["model"] == BEST
perf.to_csv(OUT / "rq4_performance.csv", index=False)

# tone ablation: same family and parameters, tone features removed
NO_TONE = [f for f in NUM if f not in TONE_FEATS]
Xtr_nt, sc_nt, _ = design(tr, NO_TONE)
Xte_nt, _, _ = design(te, NO_TONE, sc_nt)
m_nt = FAMILIES[BEST][0](best[BEST][0]).fit(Xtr_nt, ytr)
p_nt = m_nt.predict_proba(Xte_nt)[:, 1]
p_best = preds[BEST][1]

ids = te["event_id"].values
uniq = np.unique(ids)
idx_by = {u: np.where(ids == u)[0] for u in uniq}


def block_boot(fn, reps=50 if QUICK else 1000):
    r_, out = np.random.default_rng(SEED), []
    for _ in range(reps):
        ii = np.concatenate([idx_by[u] for u in r_.choice(uniq, len(uniq), replace=True)])
        out.append(fn(ii))
    return np.percentile(out, [2.5, 97.5])


def mcnemar(y, pa, pb):
    a, b = (pa >= 0.5) == y, (pb >= 0.5) == y
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    chi = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if n01 + n10 else 0.0
    return n01, n10, chi, stats.chi2.sf(chi, 1)


comp = []
for base_name, p_base in [("Majority class", preds["Majority class"][1]), ("Logistic regression", preds["Logistic regression"][1]),
                          (f"{BEST} without tone", p_nt)]:
    if base_name == BEST:
        continue
    n01, n10, chi, p = mcnemar(yte, p_base, p_best)
    acc_gain = accuracy_score(yte, p_best >= .5) - accuracy_score(yte, p_base >= .5)
    ci_acc = block_boot(lambda ii: accuracy_score(yte[ii], p_best[ii] >= .5) - accuracy_score(yte[ii], p_base[ii] >= .5))
    auc_base = roc_auc_score(yte, p_base) if len(np.unique(p_base)) > 1 else 0.5
    ci_auc = block_boot(lambda ii: roc_auc_score(yte[ii], p_best[ii]) -
                        (roc_auc_score(yte[ii], p_base[ii]) if len(np.unique(p_base)) > 1 else 0.5))
    comp.append({"best_model": BEST, "baseline": base_name, "best_correct_only": n01, "baseline_correct_only": n10,
                 "mcnemar_chi2": chi, "p_mcnemar": p, "accuracy_gain": acc_gain, "acc_gain_ci_low": ci_acc[0],
                 "acc_gain_ci_high": ci_acc[1], "auc_gain": roc_auc_score(yte, p_best) - auc_base,
                 "auc_gain_ci_low": ci_auc[0], "auc_gain_ci_high": ci_auc[1]})
comp = pd.DataFrame(comp)
comp.to_csv(OUT / "rq4_comparison.csv", index=False)
boot4 = []
for name, (_, p_) in preds.items():
    ci_a = block_boot(lambda ii: accuracy_score(yte[ii], p_[ii] >= .5))
    ci_u = block_boot(lambda ii: roc_auc_score(yte[ii], p_[ii]) if len(np.unique(p_[ii])) > 1 else 0.5)
    boot4.append({"model": name, "acc_ci_low": ci_a[0], "acc_ci_high": ci_a[1], "auc_ci_low": ci_u[0], "auc_ci_high": ci_u[1]})
pd.DataFrame(boot4).to_csv(OUT / "rq4_bootstrap_ci.csv", index=False)
cm_best = confusion_matrix(yte, (p_best >= 0.5).astype(int), labels=[0, 1])
RES["rq4"] = {"train_events": int(tr["event_id"].nunique()), "test_events": int(te["event_id"].nunique()),
              "train_rows": len(tr), "test_rows": len(te), "train_share_up": float(ytr.mean()), "test_share_up": float(yte.mean()),
              "first_test_event": ev.loc[ev["order"] == n_train_events, "event_id"].iloc[0],
              "best_model": BEST, "best_params": best[BEST][0], "best_val_auc": best[BEST][1],
              "confusion_best_test": {"TN": int(cm_best[0, 0]), "FP": int(cm_best[0, 1]), "FN": int(cm_best[1, 0]), "TP": int(cm_best[1, 1])},
              "features": FEAT_NAMES}

# save the XGBoost model (used by the app) and the scaler / feature list
xgb_final = fitted["XGBoost"]
xgb_final.save_model(str(ROOT / "models/xgb_rq4.json"))
jdump({"numeric_features": NUM, "sectors": SECTORS, "scaler_mean": SC.mean_.tolist(), "scaler_scale": SC.scale_.tolist(),
       "feature_names": FEAT_NAMES, "params": best["XGBoost"][0], "selected_overall": BEST}, ROOT / "models/xgb_rq4_features.json")

# SHAP values from XGBoost's own tree explainer (pred_contribs), test set
contrib = xgb_final.get_booster().predict(DMatrix(Xte, feature_names=None), pred_contribs=True)[:, :-1]
shap_imp = pd.DataFrame({"feature": FEAT_NAMES, "mean_abs_shap": np.abs(contrib).mean(0)})
shap_num = shap_imp[~shap_imp["feature"].str.startswith("sec_")].copy()
shap_num = pd.concat([shap_num, pd.DataFrame([{"feature": "sector (all dummies)",
                                               "mean_abs_shap": np.abs(contrib[:, [i for i, f in enumerate(FEAT_NAMES) if f.startswith("sec_")]].sum(1)).mean()}])])
shap_num = shap_num.sort_values("mean_abs_shap", ascending=False)
shap_num.to_csv(OUT / "rq4_shap_importance.csv", index=False)

# ============================================================ 6. robustness
rob = []


def rq_summary(e, spec, caar_col="caar", tone_col="tone_score"):
    e = e.copy()
    e["y_"] = e[caar_col]
    e["abs_"] = e["y_"].abs()
    for lab, mask in [("All", e["bank"].notna()), ("RBI", e["bank"] == "RBI"), ("Fed", e["bank"] == "FED")]:
        x = e.loc[mask, "y_"]
        t, p = stats.ttest_1samp(x, 0)
        rob.append({"spec": spec, "test": f"RQ1 mean CAAR, {lab}", "n": len(x), "estimate": x.mean(), "p": p})
    a, b = e.loc[(e["bank"] == "FED") & (e["is_min"] == 0), "abs_"], e.loc[(e["bank"] == "RBI") & (e["is_min"] == 0), "abs_"]
    rob.append({"spec": spec, "test": "RQ2 decisions |CAAR| Fed - RBI", "n": len(a) + len(b), "estimate": a.mean() - b.mean(),
                "p": stats.ttest_ind(a, b, equal_var=False).pvalue})
    d = e.dropna(subset=["fed_surprise"]).assign(t_=lambda z: z[tone_col], tx_=lambda z: z[tone_col] * z["is_fed"])
    m = smf.ols("y_ ~ is_fed + is_min + fed_surprise + rbi_rate_change + vix_m1 + t_ + tx_", d).fit(cov_type="HC1")
    for term, lab in [("fed_surprise", "RQ3 Fed surprise (pp per bp)"), ("rbi_rate_change", "RQ3 RBI rate change (pp per bp)"),
                      ("t_", "RQ3 tone (RBI)"), ("tx_", "RQ3 tone x Fed")]:
        rob.append({"spec": spec, "test": lab, "n": int(m.nobs), "estimate": m.params[term], "p": m.pvalues[term]})


rq_summary(ev, "Baseline: [0,+1], winsorized")
rq_summary(ev, "Window [-1,+1]", "caar[-1,+1]")
rq_summary(ev, "Window [0,+5]", "caar[0,+5]")
rq_summary(ev, "No winsorization", "caar_raw")
rq_summary(ev[ev["overlap_cross_bank"] == 0], "Excluding cross-bank overlaps")
rq_summary(ev[ev["overlap_macro_flag"] == 0], "Excluding U.S./India macro-release overlaps")
rq_summary(ev[ev["overlap_flag"] == 0], "Excluding all overlaps")
rq_summary(ev, "Tone from LLM labels", tone_col="tone_score_llm")
earn = ROOT / "data/interim/results_dates.csv"
if earn.exists():
    rd = pd.read_csv(earn, parse_dates=["date"])
    bad = set()
    for _, e in ev.iterrows():
        win = days[max(e["t0"] - 1, 0):e["t0"] + 2]
        bad |= {(e["event_id"], s) for s in rd.loc[rd["date"].isin(win), "symbol"]}
    keep = ~pd.Series(list(zip(se["event_id"], se["symbol"]))).isin(bad).values
    ce = se[keep].groupby("event_id")["car"].mean().rename("caar_noearn")
    rq_summary(ev.merge(ce, on="event_id"), "Excluding stocks with results on days -1..+1", "caar_noearn")
    RES["earnings_filter"] = {"stock_events_removed": int((~keep).sum())}
else:
    RES["earnings_filter"] = "skipped: data/interim/results_dates.csv not in repo (NSE board-meeting file not downloaded)"
rob = pd.DataFrame(rob)
rob.to_csv(OUT / "robustness.csv", index=False)

# ============================================================ 7. figures
# AR path by bank
fig, ax = plt.subplots(figsize=(6, 3.2))
for bk, col, lab in [("RBI", C_RBI, "RBI"), ("FED", C_FED, "Fed")]:
    P = np.vstack([paths[i] for i in ev.loc[ev["bank"] == bk, "event_id"]])
    m_, s_ = P.mean(0), P.std(0, ddof=1) / np.sqrt(len(P))
    ax.plot(PATH_DAYS, m_, marker="o", color=col, label=lab)
    ax.fill_between(PATH_DAYS, m_ - 1.96 * s_, m_ + 1.96 * s_, color=col, alpha=0.15)
ax.axhline(0, color="black", lw=0.6); ax.axvspan(-0.3, 1.3, color="grey", alpha=0.12)
ax.set_xlabel("Trading day relative to announcement (day 0)"); ax.set_ylabel("Average abnormal return, pp"); ax.legend(frameon=False)
save(fig, "fig_ar_path.png")

# CAAR box by group
fig, ax = plt.subplots(figsize=(6, 3.2))
grp = [("RBI", 0), ("RBI", 1), ("FED", 0), ("FED", 1)]
bp = ax.boxplot([ev.loc[(ev["bank"] == b) & (ev["is_min"] == m), "caar"] for b, m in grp], patch_artist=True, widths=0.55)
for patch, (b, _) in zip(bp["boxes"], grp):
    patch.set_facecolor(C_RBI if b == "RBI" else C_FED); patch.set_alpha(0.55)
ax.set_xticks(range(1, 5), ["RBI\ndecisions", "RBI\nminutes", "Fed\ndecisions", "Fed\nminutes"])
ax.axhline(0, color="black", lw=0.6); ax.set_ylabel("CAAR[0,+1], pp")
save(fig, "fig_caar_box.png")

# tone by group and tone vs CAAR
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
bp = axes[0].boxplot([ev.loc[(ev["bank"] == b) & (ev["is_min"] == m), "tone_score"] for b, m in grp], patch_artist=True, widths=0.55)
for patch, (b, _) in zip(bp["boxes"], grp):
    patch.set_facecolor(C_RBI if b == "RBI" else C_FED); patch.set_alpha(0.55)
axes[0].set_xticks(range(1, 5), ["RBI\ndecisions", "RBI\nminutes", "Fed\ndecisions", "Fed\nminutes"])
axes[0].axhline(0, color="black", lw=0.6); axes[0].set_ylabel("Tone score (H - D) / (H + D + N)")
for bk, col, lab in [("RBI", C_RBI, "RBI"), ("FED", C_FED, "Fed")]:
    d = ev[ev["bank"] == bk]
    axes[1].scatter(d["tone_score"], d["caar"], color=col, alpha=0.7, s=18, label=lab)
    b_ = np.polyfit(d["tone_score"], d["caar"], 1)
    xs = np.linspace(d["tone_score"].min(), d["tone_score"].max(), 20)
    axes[1].plot(xs, np.polyval(b_, xs), color=col, lw=1)
axes[1].axhline(0, color="grey", lw=0.5); axes[1].set_xlabel("Tone score (hawkish > 0)"); axes[1].set_ylabel("CAAR[0,+1], pp")
axes[1].legend(frameon=False)
save(fig, "fig_tone.png")

# Fed surprise vs CAAR
fed_s = ev[(ev["is_fed"] == 1) & ev["fed_surprise"].notna()]
fig, ax = plt.subplots(figsize=(5.5, 3.4))
for ty, mk in [(0, "o"), (1, "^")]:
    d = fed_s[fed_s["is_min"] == ty]
    ax.scatter(d["fed_surprise"], d["caar"], marker=mk, color=C_FED, alpha=0.75, label="Statements" if ty == 0 else "Minutes")
b_ = np.polyfit(fed_s["fed_surprise"], fed_s["caar"], 1)
xs = np.linspace(fed_s["fed_surprise"].min(), fed_s["fed_surprise"].max(), 50)
ax.plot(xs, np.polyval(b_, xs), color="black", lw=1)
ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
ax.set_xlabel("Fed surprise: day-0 change in U.S. 2-year yield, bps"); ax.set_ylabel("CAAR[0,+1], pp"); ax.legend(frameon=False)
save(fig, "fig_fed_surprise.png")

# RQ4: ROC (train and test), calibration, confusion, SHAP
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, part, k in [(axes[0], "Training", 0), (axes[1], "Test", 1)]:
    y_ = ytr if k == 0 else yte
    for name, col in [("Logistic regression", "grey"), ("XGBoost", C_FED), ("Neural network (MLP)", C_RBI)]:
        fpr, tpr, _ = roc_curve(y_, preds[name][k])
        ax.plot(fpr, tpr, color=col, label=f"{name} (AUC {roc_auc_score(y_, preds[name][k]):.3f})")
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=0.7)
    ax.set_title(f"{part} set"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(frameon=False, fontsize=7)
save(fig, "fig_rq4_roc.png")

fig, ax = plt.subplots(figsize=(4.8, 4))
for name, col in [("Logistic regression", "grey"), ("XGBoost", C_FED), ("Neural network (MLP)", C_RBI)]:
    fp, mp = calibration_curve(yte, preds[name][1], n_bins=10, strategy="quantile")
    ax.plot(mp, fp, marker="o", color=col, label=name, ms=3)
ax.plot([0, 1], [0, 1], color="black", ls="--", lw=0.7)
lim = (min(p.min() for _, p in preds.values()) - .02, max(p.max() for _, p in preds.values()) + .02)
ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel("Predicted probability of a rise"); ax.set_ylabel("Observed share of rises"); ax.legend(frameon=False, fontsize=7)
save(fig, "fig_rq4_calibration.png")

fig, ax = plt.subplots(figsize=(3.8, 3.4))
ax.imshow(cm_best, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm_best[i, j]:,}", ha="center", va="center", color="white" if cm_best[i, j] > cm_best.max() / 2 else "black")
ax.set_xticks([0, 1], ["Fall", "Rise"]); ax.set_yticks([0, 1], ["Fall", "Rise"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{BEST}, test set", fontsize=9)
save(fig, "fig_rq4_confusion.png")

fig, ax = plt.subplots(figsize=(5.5, 3.8))
top = shap_num.head(12).iloc[::-1]
ax.barh(top["feature"], top["mean_abs_shap"], color=C_FED)
ax.set_xlabel("Mean |SHAP value| (log-odds), XGBoost, test set")
save(fig, "fig_rq4_shap.png")

# ============================================================ 8. metadata and summary
def gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "none"
    except Exception:
        return "none"


import sklearn, xgboost, statsmodels, scipy  # noqa: E401  (versions only)
cpu = platform.processor() or platform.machine()
try:
    cpu = next(l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
except Exception:
    pass
RES["run"] = {"seconds": round(time.time() - T_START, 1), "cpu": cpu, "cpu_cores": os.cpu_count(), "gpu": gpu_name(),
              "gpu_used": "no (statistical models and XGBoost run on CPU)", "python": platform.python_version(),
              "versions": {"pandas": pd.__version__, "numpy": np.__version__, "statsmodels": statsmodels.__version__,
                           "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "scipy": scipy.__version__},
              "seed": SEED}
jdump(RES, OUT / "results.json")

pd.set_option("display.width", 220)
print("\n== RQ1"); print(rq1[["group", "K_events", "caar_pp", "ci_low", "ci_high", "p_t", "bmp_kp_z", "p_bmp_kp", "p_clustered"]].round(4).to_string(index=False))
print("\n== RQ2"); print(rq2[["comparison", "diff", "boot_ci_low", "boot_ci_high", "p_welch", "p_permutation", "p_holm", "hedges_g"]].round(4).to_string(index=False))
print("\n== RQ3"); print(rq3[rq3["term"] != "Intercept"][["model", "term", "coef", "p", "n", "r2"]].round(4).to_string(index=False))
print(json.dumps({k: RES["rq3_tests"][k] for k in ["partial_F_tone", "robust_wald_tone", "robust_wald_all_policy_information"]}, default=float))
print("\n== RQ4"); print(perf[["model", "partition", "accuracy", "balanced_accuracy", "f1", "roc_auc", "brier", "ece", "val_auc", "selected"]].round(4).to_string(index=False))
print(comp.round(4).to_string(index=False))
print(shap_num.head(8).round(4).to_string(index=False))
print("\n== Robustness"); print(rob.pivot_table(index="test", columns="spec", values="estimate").round(4).to_string())
print("\n", json.dumps(RES["run"], indent=1))
