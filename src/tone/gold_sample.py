"""
Draw the stratified gold-standard sample and write the blind labeling workbook.

Design (final synopsis): 300 sentences = 150 RBI + 150 Fed; within each bank 75 from
decisions and 75 from minutes; at most 4 sentences per document; each bank × type
stratum split 25 development / 50 evaluation. Guideline example sentences are excluded.

Outputs
    data/labels/gold_300_key.csv          gold_id ↔ sent_id, partition (no labels)
    data/labels/gold_300_to_label.xlsx    blind workbook for the author (dropdown labels)

Run from the repository root:  python src/tone/gold_sample.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
PER_STRATUM, DEV_PER_STRATUM, CAP_PER_DOC = 75, 25, 4


def draw(sent, exclude):
    rng = np.random.default_rng(SEED)
    pool = sent[~sent["sent_id"].isin(exclude)]
    picks = []
    for (bank, etype), g in pool.groupby(["bank", "event_type"]):
        g = g.iloc[rng.permutation(len(g))]
        taken, per_doc = [], {}
        for _, r in g.iterrows():
            if per_doc.get(r["event_id"], 0) < CAP_PER_DOC:
                taken.append(r)
                per_doc[r["event_id"]] = per_doc.get(r["event_id"], 0) + 1
            if len(taken) == PER_STRATUM:
                break
        t = pd.DataFrame(taken)
        t["partition"] = ["dev"] * DEV_PER_STRATUM + ["eval"] * (PER_STRATUM - DEV_PER_STRATUM)
        picks.append(t)
    gold = pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)
    gold.insert(0, "gold_id", [f"G{i:03d}" for i in range(1, len(gold) + 1)])
    return gold


def workbook(gold, path, guideline_md):
    wb = Workbook()
    ws = wb.active
    ws.title = "Label here"
    head = ["gold_id", "partition", "bank", "doc_type", "sentence", "label", "notes"]
    ws.append(head)
    for _, r in gold.iterrows():
        ws.append([r["gold_id"], r["partition"], r["bank"], "Decision" if r["event_type"] == "DEC" else "Minutes",
                   r["text"], None, None])
    dv = DataValidation(type="list", formula1='"Hawkish,Dovish,Neutral"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Use the list", error="Choose Hawkish, Dovish or Neutral")
    ws.add_data_validation(dv)
    dv.add(f"F2:F{len(gold) + 1}")
    widths = {"A": 9, "B": 10, "C": 7, "D": 10, "E": 110, "F": 12, "G": 30}
    for c, w in widths.items():
        ws.column_dimensions[c].width = w
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E79")
    for row in ws.iter_rows(min_row=2):
        row[4].alignment = Alignment(wrap_text=True, vertical="top")
        row[5].fill = PatternFill("solid", fgColor="FFF2CC")
    ws.freeze_panes = "A2"

    info = wb.create_sheet("Instructions")
    lines = ["How to label (about 2–3 hours; breaks are fine, the file saves your work)",
             "1. Read each sentence on its own and choose Hawkish, Dovish or Neutral in column F (yellow).",
             "2. Follow the guideline (next sheet and data/labels/guideline.md). Do not look up market moves.",
             "3. Do NOT open any LLM label file before you finish: the gold set must be independent.",
             "4. Unsure? Choose your best label and write a short note in column G.",
             "5. Label all 300 rows (dev and eval). Save as gold_300_labeled.xlsx and upload it to the repo",
             "   (data/labels/) or send it to Claude.",
             "",
             "partition: dev = used to check and tune the LLM prompt; eval = held out for the final classifier test."]
    for l in lines:
        info.append([l])
    info.column_dimensions["A"].width = 120
    info["A1"].font = Font(bold=True, size=12)

    gsheet = wb.create_sheet("Guideline")
    for l in guideline_md.splitlines():
        gsheet.append([l])
    gsheet.column_dimensions["A"].width = 160
    wb.move_sheet("Instructions", offset=-1)
    wb.active = 0
    wb.save(path)


if __name__ == "__main__":
    sent = pd.read_csv(ROOT / "data/interim/sentences.csv")
    exclude = set(pd.read_csv(ROOT / "data/labels/guideline_examples.csv")["sent_id"])
    gold = draw(sent, exclude)
    gold[["gold_id", "sent_id", "event_id", "bank", "event_type", "partition"]].to_csv(
        ROOT / "data/labels/gold_300_key.csv", index=False)
    workbook(gold, ROOT / "data/labels/gold_300_to_label.xlsx",
             (ROOT / "data/labels/guideline.md").read_text())
    print(gold.groupby(["bank", "event_type", "partition"]).size().unstack().to_string())
    print("documents covered:", gold["event_id"].nunique(), "| max per document:",
          gold.groupby("event_id").size().max())
