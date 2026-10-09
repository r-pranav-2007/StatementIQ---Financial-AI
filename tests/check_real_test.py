"""Scores StatementIQ on the extra realistic test statement.

Put these files in your project:
    data/real_test/real_test_statement.csv
    data/real_test/real_test_key.json
    tests/check_real_test.py          (this file)

Run (free, no API calls):   python tests/check_real_test.py
Run with the AI turned on:  python tests/check_real_test.py --llm     (costs a few cents)
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.parse_statement import parse_statement          # noqa: E402
from tools.categorize import categorize_transactions        # noqa: E402
from tools.recurring import detect_recurring                # noqa: E402
from tools.anomalies import find_anomalies                  # noqa: E402

use_llm = "--llm" in sys.argv
folder = ROOT / "data" / "real_test"
key = json.load(open(folder / "real_test_key.json", encoding="utf-8"))
txns = parse_statement(str(folder / "real_test_statement.csv"))

# 1. parsing
k = key["transactions"]
ok = sum(1 for g, t in zip(txns, k)
         if g["date"] == t["date"] and abs(g["amount"] - t["amount"]) < 0.01 and g["type"] == t["type"])
print(f"PARSING          {ok}/{len(k)} correct (rows read: {len(txns)})")

# 2. categorization
txns = categorize_transactions(txns, use_llm=use_llm)
right, wrong = 0, []
for g, t in zip(txns, k):
    if g["category"] == t["category"]:
        right += 1
    else:
        wrong.append((g["description"][:46], g["category"], g["confidence"], t["category"]))
mode = "rules + AI" if use_llm else "rules only"
print(f"CATEGORIZATION   {right}/{len(k)} correct ({100*right/len(k):.0f}%)  [{mode}]")
if wrong:
    print(f"  Wrong ({len(wrong)}):  description | got | confidence | expected")
    for d, got, conf, exp in wrong:
        print(f"    {d:<46} {got:<20} {conf:<5} {exp}")

# 3. recurring payments
rec = detect_recurring(txns)
found = {r["merchant"].lower() for r in rec}
expected = {m.lower() for m in key["recurring_expected"]}


def same(a: str, b: str) -> bool:
    """Names match if either one contains the other (case-insensitive)."""
    return a in b or b in a


hit = {e for e in expected if any(same(e, f) for f in found)}
extra = {f for f in found if not any(same(e, f) for e in expected)}
print(f"RECURRING        found {len(hit)}/{len(expected)} expected")
print("  missed:", sorted(expected - hit) or "none")
print("  extra (not in key):", sorted(extra) or "none")
print("  known hard cases (not counted):", ", ".join(key["recurring_hard_cases"]))

# 4. anomalies
flags = find_anomalies(txns, rec)
got = {(f["transaction"]["date"], f["transaction"]["merchant"].lower()) for f in flags}
want = {(a["date"], a["merchant"].lower()) for a in key["anomalies"]}
print(f"ANOMALIES        caught {len(got & want)}/{len(want)} planted, {len(got - want)} extra flag(s)")
print("  missed:", sorted(want - got) or "none")
for f in flags:
    t = f["transaction"]
    mark = "" if (t["date"], t["merchant"].lower()) in want else "   <-- not planted"
    print(f"  [{f['severity']:<6}] {f['reason']}{mark}")

if use_llm:
    try:
        import llm
        print("API usage:", llm.usage_summary())
    except Exception:
        pass
