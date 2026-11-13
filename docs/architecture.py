"""Draws docs/architecture.png (system architecture of the capstone project). Run: python docs/architecture.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

LAYERS = [  # (layer title, fill, edge, boxes)
    ("1. Data sources", "#eef3f8", "#1f4e79",
     ["RBI MPC resolutions\nand minutes (62)", "FOMC statements\nand minutes (80)", "NSE Bhavcopy, corporate\nactions, NIFTY 50, India VIX",
      "BIS policy rates, FRED\n2-year yield, BLS dates"]),
    ("2. Ingestion\n(Colab 01-04)", "#fdf1e7", "#c55a11",
     ["Downloaders and\nvalidation checks", "Corporate-action\nprice adjustment", "Event table: release time,\nday 0, surprise, overlaps"]),
    ("3. Storage", "#f2f2f2", "#555555",
     ["DuckDB database\n(12 tables, 4 views)", "GitHub repository\n(versioned CSV files)"]),
    ("4. Tone pipeline\n(Colab 05-06)", "#fdf1e7", "#c55a11",
     ["Sentence splitter\n(20,205 sentences)", "LLM weak labels\n(Claude Haiku 5.5)",
      "Own classifiers: keywords, TF-IDF\nLR / SVM, text CNN; checked on\n300 author gold labels", "Tone score per\nannouncement"]),
    ("5. Analysis\n(Colab 07)", "#fdf1e7", "#c55a11",
     ["Event study: market model,\nCAR[0,+1], CAAR", "RQ1-RQ3: t, BMP-KP, clustered,\nWelch, permutation, regressions",
      "RQ4: logistic, XGBoost, MLP\n(expanding-window tuning)"]),
    ("6. Outputs and\ndeployment", "#e9f5ec", "#2e7d32",
     ["Reports: tables,\nfigures, results.json", "Users: analysts,\nstudents, instructor", "Gradio app on\nHugging Face Spaces",
      "Saved models\n(tone_clf_v1, xgb_rq4)"]),
]
W, X0, X1, BH, GAP = 14, 2.6, 13.8, 0.85, 1.35
fig, ax = plt.subplots(figsize=(12, 9.6))
ax.set_xlim(0, W); ax.set_ylim(-0.3, len(LAYERS) * GAP + 0.1); ax.axis("off")
centers = {}
for i, (title, fill, edge, boxes) in enumerate(LAYERS):
    y = (len(LAYERS) - 1 - i) * GAP + 0.2
    ax.add_patch(FancyBboxPatch((0.1, y - 0.18), W - 0.2, BH + 0.36, boxstyle="round,pad=0.02", fc="#fafafa", ec="#bbbbbb", lw=0.8))
    ax.text(1.2, y + BH / 2, title, ha="center", va="center", fontsize=9.5, weight="bold", color="#333333")
    n = len(boxes)
    bw = (X1 - X0 - 0.25 * (n - 1)) / n
    for j, b in enumerate(boxes):
        x = X0 + j * (bw + 0.25)
        ax.add_patch(FancyBboxPatch((x, y), bw, BH, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fill, ec=edge, lw=1.1))
        ax.text(x + bw / 2, y + BH / 2, b, ha="center", va="center", fontsize=8.6)
        centers[(i, j)] = (x, y, bw)


def arrow(a, b, side_a="bottom", side_b="top", label=None, rad=0.0):
    xa, ya, wa = centers[a]; xb, yb, wb = centers[b]
    pa = {"bottom": (xa + wa / 2, ya), "right": (xa + wa, ya + BH / 2), "top": (xa + wa / 2, ya + BH), "left": (xa, ya + BH / 2)}[side_a]
    pb = {"top": (xb + wb / 2, yb + BH), "left": (xb, yb + BH / 2), "bottom": (xb + wb / 2, yb), "right": (xb + wb, yb + BH / 2)}[side_b]
    ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="-|>", mutation_scale=11, color="#555555", lw=0.9,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        ax.text((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2 + 0.08, label, ha="center", fontsize=7.5, style="italic", bbox=dict(fc="white", ec="none", pad=0.5), color="#444444")


def route_left(a, b, x_route=2.4, label=None):
    """Elbow connector along the left margin (keeps the line clear of other boxes)."""
    xa, ya, _ = centers[a]; xb, yb, _ = centers[b]
    ax.plot([xa, x_route, x_route], [ya + BH / 2, ya + BH / 2, yb + BH / 2], color="#555555", lw=0.9)
    ax.add_patch(FancyArrowPatch((x_route, yb + BH / 2), (xb, yb + BH / 2), arrowstyle="-|>", mutation_scale=11,
                                 color="#555555", lw=0.9, shrinkB=2))
    if label:
        ax.text(x_route - 0.05, (ya + yb) / 2 + BH / 2, label, rotation=90, ha="right", va="center", fontsize=7.5, style="italic", color="#444444")


for j in range(4):                       # sources to downloaders
    arrow((0, j), (1, 0))
arrow((1, 0), (1, 1), "right", "left"); arrow((1, 1), (1, 2), "right", "left")
arrow((1, 1), (2, 1)); arrow((1, 2), (2, 1))
arrow((2, 1), (2, 0), "left", "right", "rebuilt in seconds")
arrow((2, 1), (3, 0)); ax.text(12.45, 3.93, "tone_scores.csv", fontsize=7.5, style="italic", color="#444444", ha="left")
arrow((3, 0), (3, 1), "right", "left"); arrow((3, 1), (3, 2), "right", "left", "train")
arrow((3, 2), (3, 3), "right", "left")
arrow((3, 3), (2, 1), "top", "bottom")
route_left((2, 0), (4, 0), label="SQL queries")
arrow((4, 0), (4, 1), "right", "left"); arrow((4, 1), (4, 2), "right", "left")
arrow((4, 1), (5, 0)); arrow((4, 2), (5, 3))
arrow((5, 3), (5, 2), "left", "right"); arrow((5, 2), (5, 1), "left", "right"); arrow((5, 0), (5, 1), "right", "left")
ax.set_title("System architecture: RBI and Fed announcements and Indian stocks", fontsize=12, weight="bold", pad=6)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("architecture.png"), dpi=220, bbox_inches="tight", facecolor="white")
