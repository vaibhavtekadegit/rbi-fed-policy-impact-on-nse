"""
Gradio app: how might NIFTY 500 stocks react to an RBI or Fed announcement?

Tabs
    1. Try a scenario      describe an announcement; the app scores the tone of the text (optional) and
                           predicts, for each stock, the probability that its CAR[0,+1] is positive
    2. Past announcements  replay any of the 142 announcements and compare the prediction with what happened
    3. About the model     data, test results and limitations

Run locally from the repository root:  python app/app.py
"""
import gradio as gr
import pandas as pd

import model as m

# The models run on CPU. If the Space is on ZeroGPU hardware, ZeroGPU stops apps that declare no GPU function,
# so a small no-op function is registered when the "spaces" package is present (it is not needed elsewhere).
try:
    import spaces

    @spaces.GPU(duration=10)
    def _gpu_check():
        return "ok"
except ImportError:
    pass

TM = m.META["test_metrics"]
SHARE_UP = m.EVENTS.groupby("bank")["share_up"].mean()
REPO = "https://github.com/vaibhavtekadegit/rbi-fed-policy-impact-on-nse"

CAVEAT = (
    f"**Read with care.** On the 43 test announcements (from {m.META['first_test_event']} onward) this XGBoost model "
    f"had accuracy {TM['XGBoost']['accuracy']:.3f} and ROC-AUC {TM['XGBoost']['roc_auc']:.3f} "
    f"(95% CI {TM['XGBoost']['auc_ci'][0]:.3f} to {TM['XGBoost']['auc_ci'][1]:.3f}), against "
    f"{TM['Majority class']['accuracy']:.3f} for always guessing the most common outcome. "
    "The direction of a single stock over two days is close to unpredictable, so treat the numbers as a "
    "description of what the model learned, not as a forecast. This is a student research project, not investment advice."
)


# ---------------------------------------------------------------- tab 1
def run_scenario(bank, ev_type, fed_surprise, rbi_change, vix, nifty_pre5, text, tone_manual):
    tone, sent_table = m.score_tone(text) if text and text.strip() else (None, pd.DataFrame(columns=["sentence", "label"]))
    tone_used = tone if tone is not None else tone_manual
    if tone is not None:
        counts = sent_table["label"].value_counts()
        tone_msg = (f"Tone score from your text: **{tone:+.3f}** ({len(sent_table)} sentences: "
                    f"{counts.get('Hawkish', 0)} hawkish, {counts.get('Dovish', 0)} dovish, {counts.get('Neutral', 0)} neutral).")
    else:
        tone_msg = f"No text pasted, so tone {tone_manual:+.2f} is used (Advanced section)."

    ev = m.event_features(bank, ev_type, fed_surprise, rbi_change, vix, nifty_pre5, tone_used)
    pred = m.predict_stocks(ev)
    share = (pred["prob_up"] >= 0.5).mean()
    summary = (f"### Predicted share of stocks with a positive abnormal return: {share:.1%}\n"
               f"Mean predicted probability across {len(pred)} stocks: **{pred['prob_up'].mean():.3f}**. "
               f"For comparison, {SHARE_UP['FED']:.1%} of stocks had a positive abnormal return on an average Fed announcement "
               f"and {SHARE_UP['RBI']:.1%} on an average RBI announcement (2021-2026).\n\n{tone_msg}\n\n{CAVEAT}")
    cols = ["symbol", "company", "sector", "prob_up"]
    top = pred.nlargest(10, "prob_up")[cols].round(3)
    bottom = pred.nsmallest(10, "prob_up")[cols].round(3)
    return (summary, gr.update(value=m.sector_table(pred), visible=True), gr.update(value=top, visible=True),
            gr.update(value=bottom, visible=True), gr.update(value=sent_table, visible=tone is not None))


def toggle_inputs(bank, ev_type):
    return gr.update(visible=bank == "Fed"), gr.update(visible=bank == "RBI" and ev_type == "Decision")


# ---------------------------------------------------------------- tab 2
def label(row):
    return f"{row.event_id}  ({row.partition.lower()})"


EVENT_CHOICES = [(label(r), r.event_id) for r in m.EVENTS.sort_values("order", ascending=False).itertuples()]


def run_replay(event_id):
    e, pred, acc = m.replay_event(event_id)
    pred_share = (pred["prob_up"] >= 0.5).mean()
    actual_share = pred["up"].mean()
    part = "test (never seen in training)" if e["partition"] == "TEST" else "training (the model learned from it)"
    news = (f"Fed 2-year yield surprise {e['fed_surprise']:+.0f} bp" if e["is_fed"] == 1 and pd.notna(e["fed_surprise"])
            else f"RBI rate change {e['rbi_rate_change']:+.0f} bp" if e["bank"] == "RBI" and e["event_type"] == "DEC"
            else "minutes release (no rate news)")
    md = (f"### {event_id}\n"
          f"Partition: **{part}**. Day 0: {e['event_day0']}. {news}. Tone score {e['tone_score']:+.3f}. "
          f"India VIX {e['vix_m1']:.1f}.\n\n"
          f"| | Share of stocks up |\n|---|---|\n| Predicted | {pred_share:.1%} |\n| Actual | {actual_share:.1%} |\n\n"
          f"Stock-level accuracy for this announcement: **{acc:.1%}** ({len(pred)} stocks). "
          f"Actual CAAR[0,+1]: {e['caar']:+.3f} percentage points.")
    sec = (pred.groupby("sector").agg(stocks=("up", "size"), predicted_up=("prob_up", lambda x: (x >= 0.5).mean()),
                                      actual_up=("up", "mean")).astype({"predicted_up": float, "actual_up": float}).round(3).reset_index())
    return md, sec


# ---------------------------------------------------------------- tab 3
ABOUT = f"""
## About this app

**Project.** *Local or Global? How Reserve Bank of India and Federal Reserve Announcements Move Indian Stocks*,
QM640 Data Analytics Capstone, Walsh College (Vaibhav Tekade). Code, data and full results: [{REPO}]({REPO}).

**Data.** 142 announcements from October 2021 to September 2026 (31 RBI decisions, 31 RBI minutes, 40 Fed decisions,
40 Fed minutes) and daily NSE Bhavcopy prices for NIFTY 500 stocks: 61,957 stock-announcement observations.

**What is predicted.** For each stock, the probability that its cumulative abnormal return over the announcement day
and the next day (CAR[0,+1], market model on the NIFTY 50) is positive.

**Inputs used by the model.** Bank, decision or minutes, Fed surprise (day-0 change in the U.S. 2-year Treasury yield),
RBI policy-rate change, India VIX on the day before, NIFTY 50 return over the 5 days before, tone score, and for each
stock its beta, size (log traded value), residual volatility, abnormal return over the previous 5 days and sector.
Stock inputs in the scenario tab are computed from prices up to {m.META['last_price_date']}.

**Tone.** The text is split into sentences and each sentence is labeled hawkish, dovish or neutral by the author's
TF-IDF + linear SVM classifier (macro-F1 .785 on 200 sentences labeled by the author). Tone score = (hawkish - dovish) / all sentences.

**Test results (43 announcements after the training period).**

| Model | Test accuracy | Test ROC-AUC |
|---|---|---|
| Majority class | {TM['Majority class']['accuracy']:.3f} | 0.500 |
| Logistic regression | {TM['Logistic regression']['accuracy']:.3f} | {TM['Logistic regression']['roc_auc']:.3f} |
| XGBoost (used here) | {TM['XGBoost']['accuracy']:.3f} | {TM['XGBoost']['roc_auc']:.3f} |
| Neural network (MLP) | {TM['Neural network (MLP)']['accuracy']:.3f} | {TM['Neural network (MLP)']['roc_auc']:.3f} |

**Limitations.** No model reliably beat the simple baseline on the test period. Stock features are fixed at the last
price date, the RBI policy news is the raw rate change (not a market surprise), and results describe one policy cycle.
Not investment advice.
"""


# ---------------------------------------------------------------- layout
with gr.Blocks(title="RBI and Fed announcements: NIFTY 500 reaction") as demo:
    gr.Markdown("# RBI and Fed announcements: how might Indian stocks react?\n"
                "Describe a monetary policy announcement and see the model's view of which NIFTY 500 stocks are more "
                "likely to show a positive abnormal return over the announcement day and the next day.")
    with gr.Tab("Try a scenario"):
        with gr.Row():
            with gr.Column(scale=1):
                bank = gr.Radio(["Fed", "RBI"], value="Fed", label="Central bank")
                ev_type = gr.Radio(["Decision", "Minutes"], value="Decision", label="Announcement type")
                fed_s = gr.Slider(-30, 30, value=0, step=1, label="Fed surprise: day-0 change in U.S. 2-year yield (bp)")
                rbi_c = gr.Slider(-50, 50, value=0, step=5, label="RBI repo rate change (bp)", visible=False)
                text = gr.Textbox(lines=6, label="Policy text (optional): paste the statement or minutes to score its tone")
                with gr.Accordion("Advanced: market conditions and manual tone", open=False):
                    vix = gr.Number(value=round(m.META["vix_last"], 2), label="India VIX (previous close)")
                    npre = gr.Number(value=round(m.META["nifty_pre5_last"], 2), label="NIFTY 50 return over previous 5 days (%)")
                    tone_s = gr.Slider(-1, 1, value=0, step=0.05, label="Tone score if no text is pasted (hawkish > 0)")
                    gr.Markdown(f"Defaults are the values on {m.META['last_price_date']}, the last date in the data.")
                go = gr.Button("Predict", variant="primary")
            with gr.Column(scale=2):
                out_md = gr.Markdown("Choose the announcement on the left and press **Predict**.")
                out_sec = gr.Dataframe(label="By sector: mean probability and share of stocks predicted up", visible=False)
                with gr.Row():
                    out_top = gr.Dataframe(label="10 stocks most likely to rise", visible=False)
                    out_bot = gr.Dataframe(label="10 stocks least likely to rise", visible=False)
                out_sent = gr.Dataframe(label="Sentence tone labels", wrap=True, visible=False)
        bank.change(toggle_inputs, [bank, ev_type], [fed_s, rbi_c])
        ev_type.change(toggle_inputs, [bank, ev_type], [fed_s, rbi_c])
        go.click(run_scenario, [bank, ev_type, fed_s, rbi_c, vix, npre, text, tone_s],
                 [out_md, out_sec, out_top, out_bot, out_sent])
    with gr.Tab("Past announcements"):
        gr.Markdown("Pick one of the 142 announcements. Test announcements were never used for training, "
                    "so they show how the model does on new data.")
        ev_pick = gr.Dropdown(EVENT_CHOICES, value=EVENT_CHOICES[0][1], label="Announcement")
        rep_md = gr.Markdown()
        rep_sec = gr.Dataframe(label="By sector: predicted vs actual share of stocks up")
        ev_pick.change(run_replay, ev_pick, [rep_md, rep_sec])
        demo.load(run_replay, ev_pick, [rep_md, rep_sec])
    with gr.Tab("About the model"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
