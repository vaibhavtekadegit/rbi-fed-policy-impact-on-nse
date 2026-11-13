"""Draws docs/architecture.png (end-to-end workflow of the capstone project). Run: python docs/architecture.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

KIND = {  # stage type: (box fill, edge colour, legend label)
    "data": ("#eef3f8", "#1f4e79", "Data and storage"),
    "prep": ("#fdf1e7", "#c55a11", "Preprocessing, EDA and features"),
    "model": ("#f1ecf7", "#5b3a8c", "Modelling and evaluation"),
    "deploy": ("#e9f5ec", "#2e7d32", "Deployment and users"),
}
STAGES = [  # (stage title, kind, boxes, chain the boxes left to right)
    ("1. Data sources", "data",
     ["RBI MPC resolutions and minutes\n62 HTML files (rbi.org.in)", "FOMC statements and minutes\n80 HTML files (federalreserve.gov)",
      "NSE Bhavcopy: 497 stocks,\ncorporate actions, NIFTY 50, India VIX", "BIS policy rates, FRED 2-year\nyield, BLS release dates"], False),
    ("2. Ingestion and\nstorage", "data",
     ["Scripted downloads and\nvalidation checks (Colab 01-04)", "GitHub repository\n(versioned CSV files)",
      "DuckDB database\n(12 tables, 4 views, SQL access)"], True),
    ("3. Preprocessing", "prep",
     ["Corporate-action price\nadjustment, ISIN chaining", "Release time to NSE day 0;\noverlap flags",
      "Missing values: at least 100\nestimation days; missing flag", "Outliers: CAR winsorized at\n1st and 99th percentiles"], True),
    ("4. Exploratory\ndata analysis", "prep",
     ["Events, policy rates and\noverlaps over time", "CAR and CAAR distributions,\nsectors", "Tone by bank and type;\ncross-correlation of returns"], False),
    ("5. Feature\nengineering", "prep",
     ["Market model AR, CAR[0,+1],\nCAAR (days -130 to -11)", "Policy news: Fed 2-year\nsurprise, RBI rate change",
      "Tone: 20,205 sentences, LLM weak\nlabels, own TF-IDF + linear SVM", "Stock features: beta, size, volatility,\npre-CAR; scaling; sector one-hot"], False),
    ("6. Model", "model",
     ["Chronological split\n99 train / 43 test events", "RQ1-RQ3: t, BMP-KP, Welch,\npermutation, OLS partial F",
      "RQ4: logistic regression, XGBoost,\nneural network (MLP)"], True),
    ("7. Evaluation", "model",
     ["Train and test accuracy,\nROC-AUC, Brier, calibration", "McNemar, event-block\nbootstrap, Holm correction",
      "SHAP, tone ablation,\nrobustness checks", "Reports: tables, figures,\nresults.json"], True),
    ("8. Deployment", "deploy",
     ["Saved models\n(tone_clf_v1, xgb_rq4)", "GitHub Actions:\ntest, then deploy",
      "Gradio app on Hugging Face\nSpaces (free CPU)", "Users: investors, analysts,\nstudents, instructor"], True),
]
W, X0, X1, BH, GAP = 15, 2.9, 14.85, 0.78, 1.18
n_st = len(STAGES)
fig, ax = plt.subplots(figsize=(12.5, 11.2))
ax.set_xlim(0, W); ax.set_ylim(-0.75, n_st * GAP + 0.05); ax.axis("off")
centers = {}


def harrow(p, q, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color="#555555", lw=0.9,
                                 linestyle="--" if dashed else "-", shrinkA=1, shrinkB=1))


for i, (title, kind, boxes, chain) in enumerate(STAGES):
    fill, edge, _ = KIND[kind]
    y = (n_st - 1 - i) * GAP + 0.15
    ax.add_patch(FancyBboxPatch((0.15, y), 2.3, BH, boxstyle="round,pad=0.02,rounding_size=0.1", fc=edge, ec=edge))
    ax.text(1.3, y + BH / 2, title, ha="center", va="center", fontsize=9.3, weight="bold", color="white")
    if i < n_st - 1:   # pipeline spine between stages
        ax.add_patch(FancyArrowPatch((1.3, y), (1.3, y - GAP + BH), arrowstyle="-|>", mutation_scale=13, color="#333333", lw=1.4))
    n = len(boxes)
    bw = (X1 - X0 - 0.3 * (n - 1)) / n
    for j, b in enumerate(boxes):
        x = X0 + j * (bw + 0.3)
        ax.add_patch(FancyBboxPatch((x, y), bw, BH, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fill, ec=edge, lw=1.1))
        ax.text(x + bw / 2, y + BH / 2, b, ha="center", va="center", fontsize=7.9)
        centers[(i, j)] = (x, y, bw)
        if chain and j > 0:
            px, _, pw = centers[(i, j - 1)]
            harrow((px + pw, y + BH / 2), (x, y + BH / 2), dashed=(i == n_st - 1 and j in (1, 2)))
    ax.plot([2.45, X0], [y + BH / 2, y + BH / 2], color=edge, lw=0.8, ls=":")

# extra links that are not left-to-right
x, y, w = centers[(0, 0)]
for j in range(4):
    xa, ya, wa = centers[(0, j)]
    harrow((xa + wa / 2, ya), (x + w / 2 + (j - 1.5) * 0.35, y - GAP + BH))

legend = [Patch(fc=f, ec=e, label=l) for f, e, l in KIND.values()] + [
    Line2D([0], [0], color="#333333", lw=1.4, label="Pipeline order (stage to stage)"),
    Line2D([0], [0], color="#555555", lw=0.9, label="Data or results flow"),
    Line2D([0], [0], color="#555555", lw=0.9, ls="--", label="Automated CI/CD on push to main"),
]
ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=4, fontsize=8, frameon=True,
          edgecolor="#bbbbbb", title="Legend", title_fontsize=8.5)
ax.set_title("End-to-end workflow: RBI and Fed announcements and Indian stocks", fontsize=12.5, weight="bold", pad=6)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("architecture.png"), dpi=220, bbox_inches="tight", facecolor="white")
