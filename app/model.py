"""
Prediction logic for the Gradio app (no user interface code here, so it can be tested on its own).

Models (trained in notebooks 06 and 07, loaded read-only):
    tone_clf_v1.joblib   TF-IDF + linear SVM sentence classifier (Hawkish / Dovish / Neutral)
    xgb_rq4.json         XGBoost classifier for the sign of a stock's CAR[0,+1]
    xgb_rq4_features.json  feature order, sector list and the scaler used in training
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def _find_models():
    """Models sit in ../models in the GitHub repo and in ./models in the Hugging Face Space."""
    for d in (HERE / "models", HERE.parent / "models"):
        if (d / "xgb_rq4.json").exists():
            return d
    raise FileNotFoundError("models folder with xgb_rq4.json, xgb_rq4_features.json and tone_clf_v1.joblib not found")


MODELS = _find_models()
SPEC = json.loads((MODELS / "xgb_rq4_features.json").read_text())
NUM, SECTORS = SPEC["numeric_features"], SPEC["sectors"]
MEAN, SCALE = np.array(SPEC["scaler_mean"]), np.array(SPEC["scaler_scale"])
FEATURE_NAMES = SPEC["feature_names"]

XGB = XGBClassifier()
XGB.load_model(str(MODELS / "xgb_rq4.json"))
TONE = joblib.load(MODELS / "tone_clf_v1.joblib")

STOCKS = pd.read_csv(DATA / "stocks_latest.csv")
EVENTS = pd.read_csv(DATA / "events.csv")
META = json.loads((DATA / "app_meta.json").read_text())
_PAST = None


def past_rows():
    global _PAST
    if _PAST is None:
        _PAST = pd.read_csv(DATA / "past_stock_events.csv.gz")
    return _PAST


# ---------------------------------------------------------------- tone
ABBREV = {"u.s", "u.k", "dr", "mr", "mrs", "ms", "shri", "smt", "prof", "no", "nos", "vs", "e.g", "i.e", "etc",
          "a.m", "p.m", "st", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
          "inc", "co", "corp", "ltd", "govt", "approx", "fig", "sec", "fy", "rs", "viz", "cf", "pp", "vol"}


def split_sentences(text):
    """Same rule as src/tone/sentences.py: split on . ? ! before a capital letter, protecting abbreviations."""
    text = " ".join(str(text).split())
    out, start = [], 0
    for m in re.finditer(r"[.?!]['’”\")]*\s+(?=[A-Z“\"(‘'])", text):
        end = m.start() + 1
        prev = re.search(r"([A-Za-z.]+)\.$", text[:end])
        if prev and prev.group(1).lower().rstrip(".") in ABBREV:
            continue
        if re.search(r"\b[A-Z]\.$", text[:end]):
            continue
        out.append(text[start:m.end()].strip())
        start = m.end()
    if text[start:].strip():
        out.append(text[start:].strip())
    return [s for s in out if len(s.split()) >= 6]


def score_tone(text):
    """Tone score (H - D) / (H + D + N) and the sentence-level labels."""
    sents = split_sentences(text)
    if not sents:
        return None, pd.DataFrame(columns=["sentence", "label"])
    labels = TONE.predict(sents)
    h, d = int((labels == "Hawkish").sum()), int((labels == "Dovish").sum())
    return (h - d) / len(labels), pd.DataFrame({"sentence": sents, "label": labels})


# ---------------------------------------------------------------- stock predictions
def design(stock_df, event):
    """Numeric features scaled as in training, then sector dummies, in the saved feature order."""
    df = stock_df.copy()
    df["fed_surprise_missing"] = 0
    for k, v in event.items():
        df[k] = v
    num = (df[NUM].to_numpy(float) - MEAN) / SCALE
    sec = np.column_stack([(df["sector"] == s).to_numpy(float) for s in SECTORS])
    return np.hstack([num, sec])


def event_features(bank, ev_type, fed_surprise, rbi_rate_change, vix, nifty_pre5, tone):
    is_fed = int(bank == "Fed")
    return {"is_fed": is_fed, "is_min": int(ev_type == "Minutes"),
            "fed_surprise": float(fed_surprise) if is_fed else 0.0,
            "rbi_rate_change": float(rbi_rate_change) if (not is_fed and ev_type == "Decision") else 0.0,
            "vix_m1": float(vix), "nifty_pre5": float(nifty_pre5),
            "tone_score": float(tone), "tone_x_fed": float(tone) * is_fed}


def predict_stocks(event, stock_df=None):
    stock_df = STOCKS if stock_df is None else stock_df
    p = XGB.predict_proba(design(stock_df, event))[:, 1].astype("float64")   # float32 shows long decimals
    out = stock_df.assign(prob_up=p)
    return out


def sector_table(pred):
    t = (pred.groupby("sector")
             .agg(stocks=("prob_up", "size"), mean_prob_up=("prob_up", "mean"),
                  predicted_up=("prob_up", lambda x: (x >= 0.5).mean()))
             .sort_values("mean_prob_up", ascending=False).reset_index())
    return t.round(3)


def replay_event(event_id):
    """Predict a past announcement with its real features and compare with what actually happened."""
    e = EVENTS.set_index("event_id").loc[event_id]
    rows = past_rows()
    rows = rows[rows["event_id"] == event_id]
    feats = {k: float(e[k]) for k in ["is_fed", "is_min", "fed_surprise", "rbi_rate_change", "vix_m1", "nifty_pre5",
                                      "tone_score", "tone_x_fed"]}
    feats["fed_surprise_missing"] = int(np.isnan(feats["fed_surprise"]))
    feats["fed_surprise"] = 0.0 if np.isnan(feats["fed_surprise"]) else feats["fed_surprise"]
    pred = predict_stocks(feats, rows)
    acc = float(((pred["prob_up"] >= 0.5).astype(int) == pred["up"]).mean())
    return e, pred, acc
