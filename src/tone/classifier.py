"""
The author's own tone classifier (hawkish / dovish / neutral) and per-announcement tone scores.

Training data : LLM-labeled sentences (data/labels/llm_labels.csv), excluding all 300 gold sentences.
Agreement     : Cohen's kappa between LLM labels and the author's 100 development gold labels (accept if >= .60).
Models        : (1) keyword dictionary, (2) TF-IDF + logistic regression, (3) TF-IDF + linear SVM,
                (4) neural network: a text CNN with embeddings learned from scratch (PyTorch); falls back to a
                two-layer perceptron on TF-IDF if PyTorch is unavailable.
Selection     : 5-fold stratified cross-validation on the training data, macro-F1 (decided before the test).
Test          : once, on the author's 200 evaluation gold sentences: macro-F1 and kappa, overall and per bank.
Output        : models/tone_clf_v1.joblib (+ .pt for the CNN), data/interim/tone_scores.csv,
                reports/final/tone/*.csv and figures.

Run from the repository root:  python src/tone/classifier.py
"""
import json
import platform
import re
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
LABELS = ["Hawkish", "Dovish", "Neutral"]
OUT = ROOT / "reports/final/tone"
MODELS = ROOT / "models"
KAPPA_MIN = 0.60
DEVIATION = ("Author decision (Nov 13): LLM labels used although RBI dev kappa (.589, guideline v1.1) is below .60; "
             "Sonnet 5.5 did not improve it (.577). Reported as a deviation; the classifier is judged on the eval gold set.")

# ---------------------------------------------------------------- keyword dictionary (author's lexicon)
HAWK = ["withdrawal of accommodation", "raise the target range", "increase the policy repo rate", "rate hike",
        "tighten", "tightening", "restrictive", "inflation remains elevated", "elevated inflation", "inflationary pressures",
        "upside risks to inflation", "price pressures", "persistent inflation", "strong growth", "robust", "tight labor",
        "solid pace", "overheating", "further increases", "ongoing increases", "vigilant", "anchor inflation expectations",
        "above target", "firming", "hardening", "upward revision"]
DOVE = ["accommodative", "lower the target range", "reduce the policy repo rate", "rate cut", "easing", "ease",
        "support growth", "policy space", "downside risks", "slowdown", "weak", "weakness", "softening", "moderated",
        "declined", "disinflation", "benign", "below target", "unemployment rate has moved up", "slack", "headwinds",
        "contraction", "subdued", "downward revision", "growth momentum"]


class KeywordDictionary:
    """Counts hawkish and dovish phrases; Neutral when tied or none found."""
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        out = []
        for t in X:
            t = t.lower()
            h = sum(t.count(k) for k in HAWK)
            d = sum(t.count(k) for k in DOVE)
            out.append("Hawkish" if h > d else "Dovish" if d > h else "Neutral")
        return np.array(out)


def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, strip_accents="unicode")


def make_models():
    models = {
        "Keyword dictionary": KeywordDictionary(),
        "TF-IDF + logistic regression": make_pipeline(tfidf(), LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced")),
        "TF-IDF + linear SVM": make_pipeline(tfidf(), LinearSVC(C=0.5, class_weight="balanced")),
    }
    try:
        import torch  # noqa: F401
        models["Neural network (text CNN)"] = TextCNN()
    except ImportError:
        models["Neural network (MLP on TF-IDF)"] = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True),
            MLPClassifier(hidden_layer_sizes=(256, 64), early_stopping=True, max_iter=60, random_state=SEED))
    return models


# ---------------------------------------------------------------- deep learning model (author's own, PyTorch)
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|%")


class TextCNN:
    """Embedding (learned from scratch) → parallel 1D convolutions (widths 2, 3, 4) → max-pool → dropout → softmax."""
    def __init__(self, emb=100, filters=100, widths=(2, 3, 4), max_len=60, epochs=6, batch=64, lr=2e-3, min_count=2):
        self.emb, self.filters, self.widths, self.max_len = emb, filters, widths, max_len
        self.epochs, self.batch, self.lr, self.min_count = epochs, batch, lr, min_count

    def _tok(self, t):
        return TOKEN_RE.findall(t.lower())[: self.max_len]

    def _encode(self, X):
        import torch
        ids = np.zeros((len(X), self.max_len), dtype=np.int64)
        for i, t in enumerate(X):
            toks = [self.vocab.get(w, 1) for w in self._tok(t)]
            ids[i, :len(toks)] = toks
        return torch.from_numpy(ids)

    def fit(self, X, y):
        import torch
        from torch import nn
        torch.manual_seed(SEED)
        counts = {}
        for t in X:
            for w in self._tok(t):
                counts[w] = counts.get(w, 0) + 1
        self.vocab = {w: i + 2 for i, w in enumerate(sorted(w for w, c in counts.items() if c >= self.min_count))}
        self.classes_ = np.array(LABELS)
        yi = torch.tensor([LABELS.index(v) for v in y])
        Xi = self._encode(list(X))
        freq = np.bincount(yi.numpy(), minlength=3)
        weight = torch.tensor(len(yi) / (3 * np.maximum(freq, 1)), dtype=torch.float32)   # class balance

        class Net(nn.Module):
            def __init__(s, vocab, emb, filters, widths):
                super().__init__()
                s.emb = nn.Embedding(vocab, emb, padding_idx=0)
                s.convs = nn.ModuleList([nn.Conv1d(emb, filters, w) for w in widths])
                s.drop = nn.Dropout(0.5)
                s.out = nn.Linear(filters * len(widths), 3)

            def forward(s, x):
                e = s.emb(x).transpose(1, 2)
                h = torch.cat([torch.relu(c(e)).max(dim=2).values for c in s.convs], dim=1)
                return s.out(s.drop(h))

        self.net = Net(len(self.vocab) + 2, self.emb, self.filters, self.widths)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = nn.CrossEntropyLoss(weight=weight)
        g = torch.Generator().manual_seed(SEED)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(yi), generator=g)
            for k in range(0, len(yi), self.batch):
                idx = perm[k:k + self.batch]
                opt.zero_grad()
                loss = lossf(self.net(Xi[idx]), yi[idx])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            out = []
            Xi = self._encode(list(X))
            for k in range(0, len(Xi), 512):
                out.append(torch.softmax(self.net(Xi[k:k + 512]), dim=1).numpy())
        return np.vstack(out)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# ---------------------------------------------------------------- helpers
def macro_f1(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def kappa(y, p):
    return cohen_kappa_score(y, p, labels=LABELS)


def load_data():
    sent = pd.read_csv(ROOT / "data/interim/sentences.csv")
    llm = pd.read_csv(ROOT / "data/labels/llm_labels.csv").drop_duplicates("sent_id", keep="last")
    gold = pd.read_csv(ROOT / "data/labels/gold_300_labels.csv")
    gold = gold.merge(sent[["sent_id", "text"]], on="sent_id")
    train = sent.merge(llm[["sent_id", "label"]], on="sent_id")
    train = train[~train["sent_id"].isin(gold["sent_id"]) & train["label"].isin(LABELS)].reset_index(drop=True)
    return sent, llm, gold, train


def agreement(llm, gold):
    g = gold.merge(llm[["sent_id", "label"]].rename(columns={"label": "llm"}), on="sent_id", how="left")
    rows = []
    for part in ["dev", "eval"]:
        for bank in ["All", "RBI", "FED"]:
            d = g[(g["partition"] == part) & ((g["bank"] == bank) | (bank == "All"))].dropna(subset=["llm"])
            rows.append({"partition": part, "bank": bank, "n": len(d), "kappa": kappa(d["label"], d["llm"]),
                         "accuracy": float((d["label"] == d["llm"]).mean()), "macro_f1": macro_f1(d["label"], d["llm"])})
    return pd.DataFrame(rows), g


def cross_validate(models, train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    X, y = train["text"].values, train["label"].values
    rows = []
    for name, m in models.items():
        scores, t0 = [], time.time()
        for tr, va in skf.split(X, y):
            mm = type(m)() if isinstance(m, KeywordDictionary) else _clone(m)
            mm.fit(X[tr], y[tr])
            scores.append(macro_f1(y[va], mm.predict(X[va])))
        rows.append({"model": name, "cv_macro_f1": np.mean(scores), "cv_sd": np.std(scores, ddof=1),
                     "seconds": round(time.time() - t0, 1)})
        print(f"  CV {name:32} macro-F1 {np.mean(scores):.3f} ± {np.std(scores, ddof=1):.3f}")
    return pd.DataFrame(rows)


def _clone(m):
    if isinstance(m, TextCNN):
        return TextCNN(m.emb, m.filters, m.widths, m.max_len, m.epochs, m.batch, m.lr, m.min_count)
    from sklearn.base import clone
    return clone(m)


def evaluate(fitted, gold_eval):
    rows, cms = [], {}
    for name, m in fitted.items():
        p = m.predict(gold_eval["text"].values)
        for bank in ["All", "RBI", "FED"]:
            mask = (gold_eval["bank"] == bank) | (bank == "All")
            rows.append({"model": name, "bank": bank, "n": int(mask.sum()),
                         "macro_f1": macro_f1(gold_eval["label"][mask], p[mask]),
                         "kappa": kappa(gold_eval["label"][mask], p[mask]),
                         "accuracy": float((gold_eval["label"][mask] == p[mask]).mean())})
        cms[name] = confusion_matrix(gold_eval["label"], p, labels=LABELS)
    return pd.DataFrame(rows), cms


def tone_scores(model, sent, llm):
    sent = sent.copy()
    sent["pred"] = model.predict(sent["text"].values)
    sent = sent.merge(llm[["sent_id", "label"]].rename(columns={"label": "llm"}), on="sent_id", how="left")
    rows = []
    for e, g in sent.groupby("event_id"):
        H, D, N = (g["pred"] == "Hawkish").sum(), (g["pred"] == "Dovish").sum(), (g["pred"] == "Neutral").sum()
        h2, d2 = (g["llm"] == "Hawkish").sum(), (g["llm"] == "Dovish").sum()
        rows.append({"event_id": e, "n_sentences": len(g), "hawkish": int(H), "dovish": int(D), "neutral": int(N),
                     "tone_score": (H - D) / (H + D + N), "tone_score_llm": (h2 - d2) / max(g["llm"].notna().sum(), 1)})
    return pd.DataFrame(rows), sent[["sent_id", "event_id", "pred"]]


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4, 3.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Author's gold label")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    sent, llm, gold, train = load_data()
    print(f"Training sentences (LLM labels, gold excluded): {len(train):,} | {train['label'].value_counts().to_dict()}")

    agr, gl = agreement(llm, gold)
    agr.to_csv(OUT / "agreement_llm_vs_gold.csv", index=False)
    dev = agr[(agr["partition"] == "dev")]
    print(agr.round(3).to_string(index=False))
    accepted = bool((dev["kappa"] >= KAPPA_MIN).all())
    print(f"Kappa rule {'met' if accepted else 'NOT met'} (dev kappa >= {KAPPA_MIN} overall and per bank)")
    if not accepted:
        print(f"  {DEVIATION}")

    models = make_models()
    cv = cross_validate(models, train)
    best = cv.sort_values("cv_macro_f1", ascending=False)["model"].iloc[0]
    cv["selected"] = cv["model"] == best
    cv.to_csv(OUT / "cv_results.csv", index=False)

    fitted = {}
    for name, m in models.items():
        fitted[name] = m.fit(train["text"].values, train["label"].values)
    ev, cms = evaluate(fitted, gold[gold["partition"] == "eval"].reset_index(drop=True))
    ev["selected"] = ev["model"] == best
    ev.to_csv(OUT / "eval_gold_results.csv", index=False)
    print(ev.round(3).to_string(index=False))
    for name, cm in cms.items():
        slug = re.sub(r"[^a-z]+", "_", name.lower()).strip("_")
        plot_confusion(cm, f"{name}: test on 200 gold sentences", OUT / f"confusion_{slug}.png")
        pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(OUT / f"confusion_{slug}.csv")

    model = fitted[best]
    if isinstance(model, TextCNN):
        import torch
        torch.save(model.net.state_dict(), MODELS / "tone_clf_v1_cnn.pt")
        joblib.dump({"type": "TextCNN", "vocab": model.vocab, "params": {k: getattr(model, k) for k in
                     ["emb", "filters", "widths", "max_len"]}}, MODELS / "tone_clf_v1.joblib")
    else:
        joblib.dump(model, MODELS / "tone_clf_v1.joblib")
    scores, preds = tone_scores(model, sent, llm)
    scores.to_csv(ROOT / "data/interim/tone_scores.csv", index=False)
    preds.to_csv(OUT / "sentence_predictions.csv.gz", index=False, compression="gzip")
    meta = {"selected_model": best, "kappa_rule_met": accepted, "llm_labels_used": True,
            "deviation": None if accepted else DEVIATION, "training_sentences": len(train),
            "python": platform.python_version(), "cpu": platform.machine(), "gpu": (__import__("torch").cuda.get_device_name(0) if __import__("importlib").util.find_spec("torch") and __import__("torch").cuda.is_available() else "none"), "seed": SEED,
            "events_scored": len(scores), "tone_score_corr_llm": float(scores["tone_score"].corr(scores["tone_score_llm"]))}
    (OUT / "tone_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    main()
