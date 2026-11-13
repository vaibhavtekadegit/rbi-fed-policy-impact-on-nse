"""The app must reproduce the XGBoost test metrics reported in reports/final/analysis/rq4_performance.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import model as m  # noqa: E402


def test_app_reproduces_test_metrics():
    ys, ps = [], []
    for eid in m.EVENTS.loc[m.EVENTS["partition"] == "TEST", "event_id"]:
        _, pred, _ = m.replay_event(eid)
        ys.append(pred["up"].values)
        ps.append(pred["prob_up"].values)
    y, p = np.concatenate(ys), np.concatenate(ps)
    perf = pd.read_csv(ROOT / "reports/final/analysis/rq4_performance.csv")
    ref = perf[(perf["model"] == "XGBoost") & (perf["partition"] == "TEST")].iloc[0]
    assert len(y) == ref["n"]
    assert abs(roc_auc_score(y, p) - ref["roc_auc"]) < 1e-3
    assert abs(accuracy_score(y, p >= 0.5) - ref["accuracy"]) < 1e-3


def test_tone_score_range():
    score, table = m.score_tone("The Committee decided to raise the policy rate by 25 basis points. "
                                "Growth is expected to moderate somewhat over the coming quarters.")
    assert -1 <= score <= 1 and len(table) == 2
