"""Checks the analyze_spending facts (offline): per-month savings, comparison sentences, one-offs.

Expected values are recomputed here from the transactions, independently of tools/query.py.
"""
import json
import sys
from pathlib import Path

# Print as UTF-8 so a stray symbol (e.g. the rupee sign) cannot crash the Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.anomalies import _inr as inr  # noqa: E402
from tools.categorize import categorize_transactions  # noqa: E402
from tools.parse_statement import parse_statement  # noqa: E402
from tools.query import analysis_facts, analyze_spending, format_facts  # noqa: E402

checks = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append(ok)
    print(f"[{'OK   ' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail and not ok else ""))


txns = categorize_transactions(parse_statement(str(ROOT / "data" / "sample_statement.csv")), use_llm=False)
facts = analysis_facts(txns)
shown = format_facts(facts)            # what the model sees
text = json.dumps(shown, ensure_ascii=False)
debits = [t for t in txns if t["type"] == "debit"]
months = sorted({t["date"][:7] for t in txns})
n = len(months)


def month_total(m):
    return round(sum(t["amount"] for t in debits if t["date"][:7] == m), 2)


# 1. Savings are per month, computed as (period total / months) * percent, and clearly labelled.
for entry in shown["savings_potential"]:
    cat = entry["category"]
    period_total = sum(t["amount"] for t in debits if t["category"] == cat)
    check(f"{cat}: 10% saving is per month",
          entry["saving_if_cut_10pct_per_month"] == inr(round(period_total / n * 10 / 100)), str(entry))
    check(f"{cat}: period total is labelled", "total_over_whole_period" in entry and "total" not in entry)
check("no ambiguous quarter-total savings fields reach the model", "_percent_saves" not in text)

# 2. Comparison sentences name both months and match the arithmetic.
names = {m: ["", "January", "February", "March", "April", "May", "June", "July", "August", "September",
             "October", "November", "December"][int(m[5:])] for m in months}
expected = []
for a, b in zip(months, months[1:]):
    d = month_total(b) - month_total(a)
    expected.append(f"{names[b]} vs {names[a]}: {'up' if d >= 0 else 'down'} {inr(abs(d))} ({d / month_total(a) * 100:+.1f}%)")
d = month_total(months[-1]) - month_total(months[0])
if n > 2:
    expected.append(f"{names[months[-1]]} vs {names[months[0]]}: {'up' if d >= 0 else 'down'} {inr(abs(d))} "
                    f"({d / month_total(months[0]) * 100:+.1f}%)")
check("month comparison sentences", shown["month_comparisons"] == expected, f"{shown['month_comparisons']} != {expected}")
check("raw change_percent is no longer given", "change_percent" not in text)

# 3. One-offs: the planted spike and big transfer are listed, months show totals with and without.
one_offs = shown["one_off_transactions"]
check("one-offs list spike and big transfer", len(one_offs) >= 2 and all({"date", "merchant", "amount"} <= set(o) for o in one_offs), str(one_offs))
rows = [m for m in shown["monthly"] if "one_offs" in m]
check("month rows carry totals with and without one-offs",
      bool(rows) and all(m["total_without_one_offs"] != m["total"] for m in rows), str(rows))
raw_rows = [m for m in facts["monthly"] if "one_offs" in m]
check("without = with - one-offs", all(abs(m["total"] - m["one_offs_total"] - m["total_without_one_offs"]) < 0.01 for m in raw_rows))

# 5. Category ranking: strictly descending, matches an independent recomputation.
def check_ranking(label: str, rows: list[dict]) -> None:
    deb = [t for t in rows if t["type"] == "debit"]
    spent = sum(t["amount"] for t in deb)
    totals = {}
    for t in deb:
        totals[t["category"]] = totals.get(t["category"], 0) + t["amount"]
    expected = sorted(totals.items(), key=lambda kv: -kv[1])
    f = analysis_facts(rows)
    sentences = f["category_ranking"]
    want = [f"{i}. {c}: {inr(round(v, 2))} ({round(v / spent * 100, 1)}%)" + (" - saving" if c == "Investments" else "")
            for i, (c, v) in enumerate(expected, 1)]
    check(f"{label}: ranking matches recomputation", sentences == want, f"{sentences} != {want}")
    values = [v for _, v in expected]
    check(f"{label}: ranking strictly descending", all(a > b for a, b in zip(values, values[1:])))
    top_c, top_v = expected[0]
    check(f"{label}: largest_category is the first sentence",
          f["largest_category"]["category"] == top_c and sentences[0].startswith(f"1. {top_c}:")
          and f["largest_category"]["share_percent"] == round(top_v / spent * 100, 1))
    check(f"{label}: model sees the ranking sentences", format_facts(f)["category_ranking"] == want)


check_ranking("sample statement", txns)
check_ranking("fresh statement", categorize_transactions(
    parse_statement(str(ROOT / "data" / "fresh_test" / "fresh_statement.csv")), use_llm=False))

# 4. The Python fallback uses the same facts and stays compact.
answer = analyze_spending("What can I cut?", txns, use_llm=False)["answer"]
check("fallback answer is under 200 words", len(answer.split()) < 200, f"{len(answer.split())} words")
check("fallback mentions one-offs and per-month saving", "without them" in answer and "a month" in answer)

print("\nPASS" if all(checks) else "\nFAIL")
sys.exit(0 if all(checks) else 1)
