# Tone Labeling Guideline (v1.1)

Used for both the **author's gold-standard labels** and the **LLM training labels**. Adapted from the hawkish–dovish annotation scheme of Shah, Paturi and Chava (2023) for FOMC text, extended to RBI Monetary Policy Committee (MPC) resolutions and minutes.

## Unit and task

- One **sentence** at a time, read on its own (do not look at neighbouring sentences or at the market reaction).
- Assign exactly one label: **Hawkish**, **Dovish** or **Neutral**.
- Judge the sentence's implication for the **announcing central bank's own policy**: RBI text → RBI policy; Fed text → Fed policy.

## Definitions

| Label | Meaning | Typical content |
|---|---|---|
| **Hawkish** | Points toward **tighter** policy (higher rates, less liquidity) or describes conditions that would justify tightening | rate hikes or preference for them; "withdrawal of accommodation"; inflation high, rising or persistent; upside risks to inflation; strong growth or tight labour market; upward revisions to inflation or growth projections; draining liquidity |
| **Dovish** | Points toward **easier** policy (lower rates, more liquidity) or describes conditions that would justify easing | rate cuts or preference for them; "accommodative" stance; inflation falling or benign; weak or slowing growth; rising unemployment; downside risks to growth or employment; downward revisions; injecting liquidity; "policy space to support growth" |
| **Neutral** | No clear direction for policy | procedural and administrative statements; mandate and target definitions; balanced statements ("risks are roughly in balance"); descriptions of financial markets or foreign economies without a stated implication for own policy; data without direction; uncertainty without direction |

## Decision rules

1. **Decisions and votes:** a hike, a vote for a hike, or a dissent preferring a higher rate → Hawkish; the reverse → Dovish; "keep unchanged" alone → Neutral, unless the sentence adds a direction (e.g. "unchanged and continuing withdrawal of accommodation" → Hawkish).
2. **RBI stance words:** "withdrawal of accommodation" → Hawkish; "accommodative" → Dovish; "neutral stance" alone → Neutral.
3. **Economic conditions:** label by what they imply for policy. Higher inflation or stronger activity → Hawkish; lower inflation or weaker activity → Dovish.
4. **Mixed sentences:** use the clause that carries the main point (usually the main clause or the clause after "but"/"however"). If both directions carry equal weight → Neutral.
5. **Other central banks and markets:** describing foreign policy moves or market prices (yields, equities, currencies) → Neutral, unless the sentence draws a conclusion for the announcing bank's own policy.
6. **Projections:** an upward revision to inflation or growth → Hawkish; downward → Dovish; a projection with no change mentioned → label by its level relative to target if stated, otherwise Neutral.
7. **Financial stability, regulation and operations** (SOMA, repo operations, directives to the Desk) → Neutral, unless they explicitly tighten or ease policy.
8. When genuinely unsure → Neutral, and note it in the *notes* column (gold set only).

### RBI-specific rules (added in v1.1 after the development check)

9. **Risks and outlook remarks without a policy call → Neutral.** RBI text often lists risks to growth or to the outlook (trade, tariffs, geopolitics, remittances, weather, global financial conditions) or describes supportive conditions for investment. If the sentence does not state or clearly imply a change in the repo rate, the stance or liquidity, label it **Neutral**, even when the risk is to growth.
10. **Vigilance and above-target inflation → Hawkish.** Sentences in which the MPC or a member commits to "remain vigilant", "keep a close vigil" or "remain watchful" on inflation, or that project or describe inflation **above the 4 per cent target**, are **Hawkish**. Projections **below** 4 per cent are **Dovish**; at 4 per cent, **Neutral**.
11. **Current growth data → Neutral.** Descriptions of the latest GDP, gross value added (GVA), sector or capacity-utilisation figures are **Neutral** unless the sentence draws an inflation or policy conclusion from them. A member's vote for the status quo that balances growth and inflation reasons is **Neutral**.

## Examples (real sentences from the corpus; excluded from the gold sample)

### RBI — Hawkish
- “Increase the policy repo rate under the liquidity adjustment facility (LAF) by 50 basis points to 4.90 per cent with immediate effect.”
- “The MPC also decided to remain focused on withdrawal of accommodation to ensure that inflation progressively aligns with the target, while supporting growth.”
- “I also vote for remaining focused on withdrawal of accommodation.”
- “Rising metals and energy prices, acute shortage of key industrial components and high logistics costs are adding to input cost pressures.”
- “Accordingly, I vote for keeping the policy repo rate unchanged and continuing with the focus on withdrawal of accommodation.”

### RBI — Dovish
- “After assessing the current and evolving macroeconomic situation, the MPC voted to reduce the policy repo rate by 50 basis points (bps) to 5.50 per cent with immediate effect.”
- “Monetary policy easing is warranted on both the inflation and growth fronts, independently.”
- “Thus, the growth-inflation balance, especially the benign inflation outlook on both headline and core, continues to provide the policy space to support the growth momentum.”
- “The MPC also decided to continue with the accommodative stance as long as necessary to revive and sustain growth on a durable basis and continue to mitigate the impact of COVID-19 on the economy, while ensuring that inflation remains within the target going forward.”
- “Headline CPI inflation declined to an all time low in October 2025.”

### RBI — Neutral
- “The MPC reviewed the surveys conducted by the Reserve Bank to gauge consumer confidence, households’ inflation expectations, corporate sector performance, credit conditions, the outlook for the industrial, services and infrastructure sectors, and the projections of professional forecasters.”
- “The MPC also reviewed in detail the staff’s macroeconomic projections, and alternative scenarios around various risks to the outlook.”
- “The minutes of the MPC’s meeting will be published on October 23, 2024.”
- “The US dollar strengthened after the upward revision of US growth numbers for the second quarter, and treasury yields hardened recently tracking changes in policy rate expectations.”
- “These decisions are in consonance with the objective of achieving the medium-term target for consumer price index (CPI) inflation of 4 per cent within a band of +/- 2 per cent, while supporting growth.”

### Fed — Hawkish
- “In support of these goals, the Committee decided to raise the target range for the federal funds rate to 4-3/4 to 5 percent.”
- “In support of these goals, the Committee decided to raise the target range for the federal funds rate to 1/4 to 1/2 percent and anticipates that ongoing increases in the target range will be appropriate.”
- “Members noted that there had been little change in the unemployment rate and solid growth in economic activity, but that inflation remained elevated relative to the Committee's 2 percent goal.”
- “Voting against the monetary policy action were Beth M. Hammack, Neel Kashkari, and Lorie K. Logan, who preferred to raise the target range for the federal funds rate by 1/4 percentage point at this meeting.”
- “Job gains have been robust in recent months, and the unemployment rate has remained low.”

### Fed — Dovish
- “In light of the progress on inflation and the balance of risks, the Committee decided to lower the target range for the federal funds rate by 1/2 percentage point to 4-3/4 to 5 percent.”
- “The Committee is attentive to the risks to both sides of its dual mandate and judges that downside risks to employment have risen.”
- “Some participants noted, however, that low- and moderate-income households were under increasing strains, with inflation eroding their real disposable income.”
- “Since earlier in the year, labor market conditions have generally eased, and the unemployment rate has moved up but remains low.”
- “One of these members preferred to lower the target range 1/2 percentage point, while the other member preferred to leave the target range unchanged.”

### Fed — Neutral
- “The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run.”
- “The Committee's assessments will take into account a wide range of information, including readings on labor market conditions, inflation pressures and inflation expectations, and financial and international developments.”
- “The shift in policy expectations contributed significantly to higher Treasury yields.”
- “Developments in Financial Markets and Open Market Operations The manager turned first to a review of developments in financial markets.”
- “Yields rose for an array of fixed-income securities, including investment-grade corporate bonds and senior commercial MBS tranches, consistent with movements in long-term Treasury yields.”

## Use in the project

- **Gold standard:** the author labels 300 sentences (150 RBI, 150 Fed; 100 development + 200 evaluation) **without seeing any LLM label**.
- **LLM labels:** the same definitions, rules and examples form the LLM prompt (temperature 0; model, version and prompt recorded).
- **Acceptance:** LLM labels are used for training only if Cohen's κ with the author's development labels is at least .60; otherwise the prompt or guideline is revised (new version number) and relabeled.
- **Version history:** v1.0 (Nov 13) used for all sentences; dev κ = .757 (Fed, accepted) and .483 (RBI, rejected). v1.1 adds rules 9–11 for RBI text only; RBI sentences are relabeled with v1.1 and Fed labels stay at v1.0.
- **Tone score** per announcement: (H − D) / (H + D + N).

Reference: Shah, A., Paturi, S., & Chava, S. (2023). Trillion dollar words: A new financial dataset, task & market analysis. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 6664–6679). https://doi.org/10.18653/v1/2023.acl-long.368
