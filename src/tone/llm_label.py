"""
LLM weak labels for every sentence (data preparation only; the analysis classifier is the author's own).

- Prompt = data/labels/guideline.md (definitions, rules, real examples) + a batch of sentences.
- Temperature 0 where the model allows it (newer models reject the setting; then the model default is used and recorded);
  the exact model id returned by the API, prompt version, temperature and time are stored per row.
- Resumable: sentences already in the output file are skipped.

Output: data/labels/llm_labels.csv  (sent_id, label, model, prompt_version, labeled_at)

Usage (Colab): see notebooks/05_llm_labeling.ipynb. Needs the environment variable ANTHROPIC_API_KEY.
"""
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
SENTENCES = ROOT / "data/interim/sentences.csv"
GUIDELINE = ROOT / "data/labels/guideline.md"
OUT = ROOT / "data/labels/llm_labels.csv"
PROMPT_VERSION = "v1.0"
MODEL = os.environ.get("LLM_MODEL", "claude-haiku-5-5")      # cheapest; switch to claude-sonnet-5-5 if kappa < .60
BATCH = 40
LABELS = {"Hawkish", "Dovish", "Neutral"}
USE_TEMPERATURE = True          # set to False automatically if the model rejects the temperature setting

SYSTEM = (
    "You are an annotator of central bank communication. Label each sentence as Hawkish, Dovish or Neutral "
    "strictly following the guideline below. Judge each sentence on its own. "
    "Return only JSON: a list of objects {\"id\": <id>, \"label\": \"Hawkish|Dovish|Neutral\"}, one per input sentence, "
    "in the same order, with no other text.\n\n=== GUIDELINE ===\n")


def call(batch, api_key, max_retries=6):
    global USE_TEMPERATURE
    items = "\n".join(f'{{"id": "{r.sent_id}", "bank": "{r.bank}", "sentence": {json.dumps(r.text)}}}'
                      for r in batch.itertuples())
    body = {"model": MODEL, "max_tokens": 4000,
            "system": [{"type": "text", "text": SYSTEM + GUIDELINE.read_text(),
                        "cache_control": {"type": "ephemeral"}}],          # prompt caching: guideline sent once
            "messages": [{"role": "user", "content": f"Label these {len(batch)} sentences:\n{items}"}]}
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    for attempt in range(max_retries):
        if USE_TEMPERATURE:
            body["temperature"] = 0
        else:
            body.pop("temperature", None)
        r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=180)
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(min(60, 5 * 2 ** attempt))
            continue
        if r.status_code == 400 and USE_TEMPERATURE and "temperature" in r.text:
            USE_TEMPERATURE = False                                     # newer models fix sampling; retry without it
            print("  note: model does not accept a temperature setting; using its default sampling")
            continue
        if r.status_code >= 400:                                        # show the API's own explanation
            raise RuntimeError(f"API error {r.status_code}: {r.text[:800]}")
        js = r.json()
        text = "".join(b.get("text", "") for b in js["content"])
        out = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
        return {o["id"]: o["label"] for o in out}, js.get("model", MODEL)
    raise RuntimeError("API kept failing; rerun later (progress is saved)")


def run(api_key, limit=None):
    sent = pd.read_csv(SENTENCES)
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["sent_id", "label", "model", "prompt_version", "temperature", "labeled_at"])
    todo = sent[~sent["sent_id"].isin(done["sent_id"])]
    if limit:
        todo = todo.head(limit)
    print(f"{len(done):,} already labeled; {len(todo):,} to go")
    for i in range(0, len(todo), BATCH):
        b = todo.iloc[i:i + BATCH]
        labels, model_used = call(b, api_key)
        rows = [{"sent_id": s, "label": labels.get(s), "model": model_used, "prompt_version": PROMPT_VERSION,
                 "temperature": "0" if USE_TEMPERATURE else "model default (not settable)",
                 "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")} for s in b["sent_id"]]
        bad = [r["sent_id"] for r in rows if r["label"] not in LABELS]
        rows = [r for r in rows if r["label"] in LABELS]          # missing/invalid ones are retried on the next run
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"  {min(i + BATCH, len(todo)):,}/{len(todo):,} labeled" + (f" ({len(bad)} invalid, will retry)" if bad else ""))
    final = pd.read_csv(OUT).drop_duplicates("sent_id", keep="last")
    final.to_csv(OUT, index=False)
    print(final["label"].value_counts().to_dict(), "| total", len(final), "of", len(sent))
    return final
