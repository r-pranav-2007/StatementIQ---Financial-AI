"""Checks answer_query on 20 questions. Expected answers come from pandas on ground_truth.json.

Usage:  python tests/test_query.py          (no-LLM mode)
        python tests/test_query.py --llm    (LLM for intent + phrasing; needs ANTHROPIC_API_KEY)

An answer is correct when every expected number and name appears in the answer text.
Hallucination check: every number in the answer must also appear in result["numbers"].
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Print as UTF-8 so a stray symbol (e.g. the rupee sign) cannot crash the Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:  # stdout replaced by an object without reconfigure()
    pass

sys.path.insert(0, str(ROOT))
import llm  # noqa: E402
from tools.parse_statement import parse_statement  # noqa: E402
from tools.categorize import categorize_transactions  # noqa: E402
from tools.query import answer_query  # noqa: E402

USE_LLM = "--llm" in sys.argv
if USE_LLM and not llm.has_api_key():
    print("WARNING: --llm given but ANTHROPIC_API_KEY is not set; running rules only.")

# ---------------------------------------------------------------------------
# Expected answers: pandas, straight from the ground truth (not from tools/query.py)
# ---------------------------------------------------------------------------
truth = json.load(open(ROOT / "data" / "ground_truth.json", encoding="utf-8"))
df = pd.DataFrame(truth["transactions"])
df["month"] = df["date"].str[:7]
deb = df[df["type"] == "debit"]
cred = df[df["type"] == "credit"]
latest = df["month"].max()


def spent(**filters) -> float:
    rows = deb
    for col, val in filters.items():
        rows = rows[rows[col] == val]
    return round(rows["amount"].sum(), 2)


merchant_rows = deb[(deb["category"] != "Transfers") & (deb["merchant"] != "ATM")]
merchant_totals = merchant_rows.groupby("merchant")["amount"].sum().sort_values(ascending=False)


def top(n: int) -> tuple[list, list]:
    t = merchant_totals.head(n)
    return [round(v, 2) for v in t.values], list(t.index)


def biggest(rows: pd.DataFrame) -> tuple[list, list]:
    row = rows.loc[rows["amount"].idxmax()]
    return [row["amount"]], [row["merchant"]]


def income_vs_exp(rows: pd.DataFrame) -> list:
    inc = rows[rows["type"] == "credit"]["amount"].sum()
    exp = rows[rows["type"] == "debit"]["amount"].sum()
    return [round(inc, 2), round(exp, 2), round(abs(inc - exp), 2)]


# Subscriptions = planted recurring items whose category is Subscriptions.
category_of = dict(zip(df["merchant"], df["category"]))
subs = [r for r in truth["recurring"] if category_of[r["merchant"]] == "Subscriptions"]
discretionary = (deb[deb["category"].isin(["Food Delivery", "Shopping", "Entertainment"])]
                 .groupby("category")["amount"].sum().sort_values(ascending=False).head(2))

top3_amounts, top3_names = top(3)
top5_amounts, top5_names = top(5)
big_all = biggest(deb)
big_aug = biggest(deb[deb["month"] == "2026-08"])

# (question, expected numbers, expected names)
CASES = [
    ("How much did I spend in August?", [spent(month="2026-08")], []),
    ("What was my total spending for July 2026?", [spent(month="2026-07")], []),
    ("How much money went on food delivery last month?",
     [spent(month=latest, category="Food Delivery")], []),
    ("How much do I spend on groceries?", [spent(category="Groceries")], []),
    ("How much did I spend on transport in August?",
     [spent(category="Transport", month="2026-08")], []),
    ("Did I spend more in August compared to July?",
     [spent(month="2026-08"), spent(month="2026-07")], []),
    ("Compare my September and August expenses.",
     [spent(month="2026-09"), spent(month="2026-08")], []),
    ("Who are my top 3 merchants?", top3_amounts, top3_names),
    ("Which five places did I spend the most money at?", top5_amounts, top5_names),
    ("How much have I paid Swiggy in total?", [spent(merchant="Swiggy")], []),
    ("What did Netflix cost me?", [spent(merchant="Netflix")], []),
    ("What's my average transaction size?", [round(deb["amount"].mean(), 2)], []),
    ("On average, how much was each Uber ride?",
     [round(deb[deb["merchant"] == "Uber"]["amount"].mean(), 2)], []),
    ("What was my biggest expense?", *big_all),
    ("What was the largest purchase I made in August?", *big_aug),
    ("How many transactions are there in total?", [len(df)], []),
    ("How many times did I order from Zomato?", [int((df["merchant"] == "Zomato").sum())], []),
    ("Compare my income and expenses.", income_vs_exp(df), []),
    ("How much did I save in July?", [income_vs_exp(df[df["month"] == "2026-07"])[2]], []),
    ("What can I cut to save money?",
     [round(sum(s["amount"] for s in subs), 2)] + [round(v, 2) for v in discretionary.values],
     [s["merchant"] for s in subs] + list(discretionary.index)),
]

# ---------------------------------------------------------------------------
# Checks (independent of tools/query.py)
# ---------------------------------------------------------------------------
NUM = re.compile(r"\d+(?:,\d+)*(?:\.\d+)?")


def nums_in(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in NUM.findall(text)]


def all_numbers(value) -> list[float]:
    """Every number inside the result's "numbers" dict (labels included)."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return nums_in(value)
    if isinstance(value, dict):
        return [n for v in value.values() for n in all_numbers(v)]
    if isinstance(value, list):
        return [n for v in value for n in all_numbers(v)]
    return []


def contains(values: list[float], x: float) -> bool:
    return any(abs(v - x) < 0.011 for v in values)


txns = categorize_transactions(parse_statement(str(ROOT / "data" / "sample_statement.csv")), use_llm=False)

correct = 0
hallucinated = 0
print(f"Mode: {'rules + LLM' if USE_LLM else 'no-LLM'}\n")
for i, (question, exp_nums, exp_names) in enumerate(CASES, 1):
    result = answer_query(question, txns, use_llm=USE_LLM)
    answer = result["answer"]
    in_answer = nums_in(answer)

    missing = [x for x in exp_nums if not contains(in_answer, x)]
    missing += [n for n in exp_names if n.lower() not in answer.lower()]
    ok = not missing
    correct += ok

    extra = [x for x in in_answer if not contains(all_numbers(result["numbers"]), x)]
    hallucinated += bool(extra)

    print(f"{i:>2}. [{'OK   ' if ok else 'WRONG'}] {question}")
    print(f"      {answer}")
    if missing:
        print(f"      expected but missing: {missing}")
    if extra:
        print(f"      HALLUCINATION - numbers not in 'numbers': {extra}")

print("\n" + "=" * 60)
print(f"Correct: {correct}/{len(CASES)}")
print(f"Hallucination check: {len(CASES) - hallucinated}/{len(CASES)} answers use only numbers "
      f"from 'numbers'" + ("" if hallucinated else "  -> PASS"))
if USE_LLM:
    u = llm.usage_summary()
    print(f"\nAPI usage: {u['calls']} calls, {u['input_tokens']} input + "
          f"{u['output_tokens']} output tokens, about ${u['estimated_cost_usd']}")
sys.exit(0 if correct == len(CASES) and not hallucinated else 1)
