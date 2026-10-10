"""Amount filters, "list all" and unsupported filters in tools/query.py (offline, no API)."""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.anomalies import _inr as inr  # noqa: E402
from tools.categorize import categorize_transactions  # noqa: E402
from tools.parse_statement import parse_statement  # noqa: E402
from tools.query import answer_query, parse_amount_filter  # noqa: E402

txns = categorize_transactions(parse_statement(str(ROOT / "data" / "sample_statement.csv")), use_llm=False)
checks = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append(ok)
    print(f"[{'OK   ' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail and not ok else ""))


def ask(q: str) -> dict:
    return answer_query(q, txns, use_llm=False)


def expected(pred, kind=None):
    rows = [t for t in txns if pred(t["amount"]) and (kind is None or t["type"] == kind)]
    return len(rows), round(sum(t["amount"] for t in rows), 2), rows


# 1. Amount filters: only matching rows, counts and totals checked against the data.
n, total, rows = expected(lambda a: a > 5000, "debit")
for q in ["show me all payments that crossing above 5000", "payments over 5,000", "debits more than 5k",
          "payments greater than Rs 5000"]:
    r = ask(q)
    out = r["numbers"].get("payments_out", {})
    check(f"'{q}' -> {n} payments, {inr(total)}", out.get("count") == n and out.get("total") == total
          and "payments_in" not in r["numbers"], str(r["numbers"]))
    check(f"'{q}' -> every listed row is above 5000",
          all(t["amount"] > 5000 for t in r["supporting_transactions"]) and len(r["supporting_transactions"]) == min(n, 10))
check("no filtered row is missing from the top 10 (largest first)",
      [t["amount"] for t in ask("payments above 5000")["supporting_transactions"]] ==
      sorted((t["amount"] for t in rows), reverse=True)[:10])
n2, total2, _ = expected(lambda a: a < 100, "debit")
r = ask("payments under 100")
check("below / under works", r["numbers"]["payments_out"] == {"count": n2, "total": total2}, str(r["numbers"]))
n3, total3, _ = expected(lambda a: 1000 <= a <= 5000, "debit")
r = ask("payments between 1000 and 5000")
check("between works (inclusive)", r["numbers"]["payments_out"] == {"count": n3, "total": total3}, str(r["numbers"]))
r = ask("received more than 40,000")
n4, total4, _ = expected(lambda a: a > 40000, "credit")
check("received above 40,000 -> credits only", r["numbers"]["payments_in"] == {"count": n4, "total": total4}
      and "payments_out" not in r["numbers"], str(r["numbers"]))
for text, value in [("5000", 5000), ("5,000", 5000), ("5k", 5000), ("Rs 5,000", 5000), ("2 lakh", 200000)]:
    f = parse_amount_filter(f"payments above {text}")
    check(f"amount '{text}' is read as {value}", f and f["min"] == value, str(f))
check("'over 3 months' is not an amount", parse_amount_filter("spending over 3 months") is None)

# 2. List all: split by direction, never added together, 10 most recent.
for q in ["show me all that transactions", "list transactions", "show me all transactions"]:
    r = ask(q)
    n_out, t_out, _ = expected(lambda a: True, "debit")
    n_in, t_in, _ = expected(lambda a: True, "credit")
    nums = r["numbers"]
    check(f"'{q}' splits payments out and in",
          nums["payments_out"] == {"count": n_out, "total": t_out} and nums["payments_in"] == {"count": n_in, "total": t_in}, str(nums))
    check(f"'{q}' never shows debits + credits as one total", inr(round(t_out + t_in, 2)) not in r["answer"])
    recent = sorted(txns, key=lambda t: t["date"], reverse=True)[:10]
    check(f"'{q}' lists the 10 most recent", [m["date"] for m in nums["matches"]] == [t["date"] for t in recent])

# 3. Unsupported filters get an honest answer.
for q in ["show me payments on monday", "what did I spend yesterday", "payments between 5 and 10 June",
          "spending in the evening", "payments made abroad", "show payments via UPI"]:
    r = ask(q)
    check(f"'{q}' -> honest 'can't do that'", "can't filter by" in r["answer"] and r["numbers"] == {}, r["answer"])
check("supported month words are not blocked", "can't" not in ask("how much did I spend in August?")["answer"])

print("\nPASS" if all(checks) else "\nFAIL")
sys.exit(0 if all(checks) else 1)
