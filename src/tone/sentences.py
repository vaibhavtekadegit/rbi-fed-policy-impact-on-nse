"""
Rule-based sentence extraction for RBI and FOMC policy documents.

Reads the 142 HTML files in data/raw/events/, keeps only the policy text
(RBI: the press-release table; Fed: the #article block), removes boilerplate
(attendance lists, voting records, statutory text, signatures, footnotes) and
splits paragraphs into sentences with a splitter written for this project.

Output: data/interim/sentences.csv with columns
    sent_id, event_id, bank, event_type, para_no, sent_no, text, n_words

Run from the repository root:
    python src/tone/sentences.py
"""
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
EVENTS = ROOT / "data/raw/events"
OUT = ROOT / "data/interim/sentences.csv"

# abbreviations that end with a period but do not end a sentence
ABBREV = {"u.s", "u.k", "dr", "mr", "mrs", "ms", "shri", "smt", "prof", "no", "nos", "vs", "e.g", "i.e", "etc",
          "a.m", "p.m", "st", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
          "inc", "co", "corp", "ltd", "govt", "approx", "fig", "sec", "q1", "q2", "q3", "q4", "h1", "h2",
          "fy", "rs", "re", "viz", "cf", "ed", "p", "pp", "vol"}

# paragraphs that are procedural, not policy communication
DROP_PARA = re.compile("|".join([
    r"^the meeting was attended by", r"was attended by all the members", r"section 45z", r"reserve bank of india act, 1934",
    r"shall publish, on the fourteenth day", r"^press release", r"chief general manager", r"general manager",
    r"^\(?[a-z .]+\)?\s*(chief )?general manager", r"the minutes of the proceedings",
    r"^voting for (this|the monetary policy)", r"return to text", r"^last update",
    r"media inquiries", r"implementation note", r"^secretary'?s note", r"^attendance", r"^present:",
    r"^the next meeting of the (mpc|committee)", r"is scheduled (during|for)", r"^\d+\s*attended",
    r"^(alternate members|presidents of the federal reserve banks|manager, system open market account)",
    r"^by notation vote", r"^joshua gallin|^secretary$", r"^a joint meeting of the federal open market committee",
    r"was held in the offices of the board of governors", r"^the vote of each member", r"^the resolution adopted at the meeting",
    r"^the votes? of each member", r"^(staff|members|participants) (attending|present)",
]), re.I)
# staff and attendance lines in Fed minutes: "Name, Title, Federal Reserve Bank of X" or "..., Division of ..."
ATTENDEE = re.compile(r"(Federal Reserve Bank of|Division of|Board of Governors|Office of|Special Adviser|Senior Adviser|"
                      r"Associate Director|Deputy Director|Vice President|Section Chief|Group Manager|Principal Economist)", re.I)

POLICY_WORDS = re.compile(r"\b(inflation|rates?|growth|econom\w*|polic\w*|markets?|prices?|employment|labou?r|demand|"
                          r"outlook|liquidity|yields?|credit|output|spending|risks?|uncertain\w*)\b", re.I)
PERSON = re.compile(r"\b[A-Z][a-z]+(?: [A-Z]\.)+ [A-Z][a-zA-Z'-]+|\b[A-Z][a-z]+ [A-Z][a-z]+,")
REFERENCE = re.compile(r"(^ref\b|journal\b|quarterly review|working paper|\(\d{4}[a-z]?\)\.|\d+\(\d+\),\s*\d+|"
                       r"\bpp\.\s*\d|bulletin,|et al\.)", re.I)
TITLES = re.compile(r"\b(Secretary|Counsel|Economists?|Manager|Advis[eo]r|Director|Presidents?|Analyst|Members?|Officer|"
                    r"Assistant|Associate|Deputy|Chief|Governors?|Chair|Vice Chair|Counselor|Specialist|Lead)\b")
VERBS = re.compile(r"\b(is|are|was|were|be|been|being|has|have|had|will|would|could|should|may|might|can|must|"
                   r"\w+ed|\w+es|noted|saw|see|said|rose|fell|grew|led|held|remain\w*)\b")
LOGISTICS = re.compile(r"(next meeting of the committee|meeting adjourned|^notation vote|approved the minutes|"
                       r"unanimously approved|were informed of the board's approval)", re.I)

# sentence-level noise
DROP_SENT = re.compile("|".join([
    r"^(dr|shri|smt|professor|prof)\.? [a-z]", r"https?://", r"@", r"\bpdf\b",
]), re.I)


def doc_paragraphs(path):
    soup = BeautifulSoup(path.read_text(errors="ignore"), "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer"]):
        t.decompose()
    if path.name.startswith("RBI"):
        box = soup.select_one("table.tablebg tr.tablecontent1") or soup.select_one("table.tablebg") or soup
    else:
        box = soup.select_one("div#article") or soup
    blocks = box.find_all(["p", "li"])
    paras = []
    for b in blocks:
        if b.find(["p", "li"]):          # keep the innermost block only
            continue
        t = " ".join(b.get_text(" ").split())
        if t:
            paras.append(t)
    return paras


def clean_para(t):
    t = re.sub(r"^\(?[ivx\d]+[.)]\s+", "", t)            # leading paragraph numbers: "2. ", "(iii) "
    t = re.sub(r"\[\s*\d+\s*\]|\s\d+\s(?=[A-Z])", " ", t)   # footnote markers
    return " ".join(t.split())


def split_sentences(text):
    """Split on . ? ! followed by space and an upper-case letter or quote, protecting abbreviations and decimals."""
    out, start = [], 0
    for m in re.finditer(r"[.?!]['’”\")]*\s+(?=[A-Z“\"(‘'])", text):
        end = m.start() + 1
        prev = re.search(r"([A-Za-z.]+)\.$", text[:end])
        if prev and prev.group(1).lower().rstrip(".") in ABBREV:
            continue
        if re.search(r"\b[A-Z]\.$", text[:end]):          # initials, e.g. "Jerome H."
            continue
        seg = text[start:m.end()].strip()
        if seg:
            out.append(seg)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def keep_sentence(s):
    words = s.split()
    if re.match(r"^\d{1,2}\s+[A-Z“\"]", s):                              # footnotes
        return False
    if s[0].islower() and not POLICY_WORDS.search(s):                       # statutory list fragments
        return False
    if REFERENCE.search(s) or LOGISTICS.search(s):
        return False
    if len(PERSON.findall(s)) >= 2 and not POLICY_WORDS.search(s):         # attendance and name lists
        return False
    if TITLES.search(s) and PERSON.search(s) and not VERBS.search(re.sub(r"[A-Z]\w*", "", s)):   # "Name, Title, Division"
        return False
    if len(words) <= 25 and ATTENDEE.search(s) and not re.search(r"\b(inflation|rate|growth|economy|policy|market)\b", s, re.I):
        return False
    if len(words) < 6 or len(s) > 700:
        return False
    if sum(ch.isalpha() for ch in s) < 0.5 * len(s):
        return False
    if s.count(";") >= 3 and len(re.findall(r"\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+", s)) >= 2:   # name lists
        return False
    return not DROP_SENT.search(s)


def build():
    rows = []
    for path in sorted(EVENTS.rglob("*.html")):
        bank, etype, d = path.stem.split("_")
        event_id = path.stem
        para_no = 0
        for p in doc_paragraphs(path):
            if DROP_PARA.search(p):
                continue
            p = clean_para(p)
            sents = [s[0].upper() + s[1:] for s in split_sentences(p) if keep_sentence(s)]
            if not sents:
                continue
            para_no += 1
            for k, s in enumerate(sents, 1):
                rows.append({"event_id": event_id, "bank": bank, "event_type": etype, "para_no": para_no,
                             "sent_no": k, "text": s, "n_words": len(s.split())})
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(["event_id", "text"]).reset_index(drop=True)
    df.insert(0, "sent_id", [f"{e}_{i:04d}" for e, i in zip(df["event_id"], df.groupby("event_id").cumcount() + 1)])
    return df


if __name__ == "__main__":
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df):,} sentences from {df['event_id'].nunique()} documents -> {OUT.relative_to(ROOT)}")
    print(df.groupby(["bank", "event_type"]).agg(docs=("event_id", "nunique"), sentences=("text", "size"),
                                                  median_words=("n_words", "median")).to_string())
