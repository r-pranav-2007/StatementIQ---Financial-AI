# StatementIQ: Agentic AI-Based Personal Finance Analyzer

Upload a bank or UPI statement (CSV, Excel or PDF). An AI agent parses it, categorizes every transaction, finds recurring payments, flags unusual charges and answers questions in plain language. All numbers are computed in Python, and the language model only chooses tools and phrases answers.

## Problem

Bank statements are long lists of cryptic text such as `UPI/SWIGGY/813750214104/okaxis`. People lose track of where money goes, forget subscriptions and miss duplicate or unusual charges. Many finance apps need bank or SMS access and handle Indian UPI formats poorly.

## Features

- Reads CSV, XLSX and PDF statements in different bank and wallet layouts
- Cleans merchant names and assigns one of 13 categories (rules first, AI for unknown merchants)
- Detects recurring payments and subscriptions, with the next expected date
- Flags anomalies: duplicate charges, spending spikes, new subscriptions, large first-time transfers
- Answers questions such as "How much did I spend in August?" or "What can I cut?"
- "Needs review" tab where low-confidence categories can be corrected
- "Show tools used" panel that shows which tools the agent called

## Why it is agentic

The agent decides what to do. It picks which tool to call for each request (parse, categorize, detect recurring, find anomalies, answer a query), runs it, reads the result and continues for up to 6 steps. A number guard rejects any answer containing a number that no tool returned.

## Architecture

```
User uploads statement
        |
   Streamlit app (app.py)
        |
   StatementAgent (agent.py) <---> llm.py (Claude; single place to swap models)
        |
   +----+--------+--------------+--------------+-------------+
 parse_      categorize_    detect_        find_          answer_
 statement   transactions   recurring      anomalies      query
   +----+--------+--------------+--------------+-------------+
        |
 Session data (kept in memory) -> compact summaries -> final answer
```

Design rules:
1. Python computes every figure. The model never invents numbers.
2. The model sees compact summaries only, never the full transaction list.
3. Every tool works without an API key (offline mode).
4. All model calls go through `llm.py`.

## Folder structure

```
app.py                  Streamlit interface
agent.py                Agent loop and offline router
llm.py                  The only file that talks to a model
tools/                  parse_statement, categorize, recurring, anomalies, query
data/                   Fake statement generator, sample files, ground_truth.json
tests/                  One test per tool, plus evaluation.py
results.md / .json      Latest evaluation table
```

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Optional, for AI mode: create a `.env` file (never commit it):

```
ANTHROPIC_API_KEY=your_key_here
CLAUDE_MODEL=claude-haiku-4-5-20251001
```

## Run

```
python -m streamlit run app.py
```

Click "Use sample data" to try it. The "Use AI (Claude)" toggle in the sidebar is off by default; offline mode needs no key.

## Tests and evaluation

```
python tests/evaluation.py          # offline, free
python tests/evaluation.py --llm    # adds AI columns (about $0.07)
```

All data is synthetic with a known answer key (`data/ground_truth.json`).

## Results

| Section | Test | Offline (rules only) | With AI |
|---|---|---|---|
| Parsing | 6 formats (4 CSV, XLSX, PDF) | 77/77 each (100%) | - |
| Categorization | Bank CSV | 77/77 (100%) | 77/77 (100%) |
| Categorization | Wallet CSV | 77/77 (100%) | 77/77 (100%) |
| Categorization | Hard test (25 unseen) | 18/25 (72%) | 25/25 (100%) |
| Categorization | Holdout (25 unseen) | 9/25 (36%) | 23/25 (92%) |
| Recurring detection | Found vs planted | 8/8, 0 false positives | - |
| Anomaly detection | Caught vs planted | 4/4, 0 false alarms | - |
| Query | Correct answers | 20/20, 0 hallucinated numbers | 20/20, 0 hallucinated numbers |
| Agent | Tool selection | 6/6 | 6/6 |

AI run cost: 64 calls, about $0.07 (Claude Haiku 4.5).

How to read this: the holdout (merchants the rules had never seen) is the honest categorization number. Rules alone score 36%; rules plus the model score 92%. The other rows come from synthetic data we generated, so they show the logic works, not that it works on every real bank.

## Limitations

- Scanned or image PDFs are not supported (no OCR)
- Bank export formats vary; the parser was tested on six common layouts, not on every bank
- Rule keywords can be confidently wrong on ambiguous words (for example a "metro" water board labelled Transport)
- Anomaly rules are simple thresholds, not trained models
- Steady spending can look like a subscription, and price rises can hide one
- At least 3 months of history are needed to detect monthly recurring payments
- The number guard cannot catch a wrong date or a wrong link between facts when the numbers exist elsewhere in the results
- Test sets are small (25 items) and were written by the same tool that wrote the rules

## Privacy

- Processing happens in the session only; nothing is stored
- Offline mode sends nothing anywhere
- In AI mode, unknown merchant names and the agent's compact summaries are sent to the model provider. Use fake data for demos
- The agent is read-only and never moves money
- This is analysis, not financial advice

## Future scope

Budget goals and alerts, next-month expense prediction, multi-bank merging, WhatsApp or Telegram interface, Tamil and Hindi voice queries, learning from user corrections, verifying all low-specificity keyword matches with the model, and porting `llm.py` to IBM watsonx.ai (Granite).
