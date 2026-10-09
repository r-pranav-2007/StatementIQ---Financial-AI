"""Scores StatementIQ on the FRESH statement (never seen during development).

Setup:
  data/fresh_test/fresh_statement.csv      (goes in the project)
  tests/check_fresh_test.py                (this file)
  fresh_key.json                           (keep OUTSIDE the project, e.g. D:\\private\\fresh_key.json)

Run once (free):        python tests/check_fresh_test.py --key D:\\private\\fresh_key.json
Run once with the AI:   python tests/check_fresh_test.py --key D:\\private\\fresh_key.json --llm
"""
import json, sys
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

args = sys.argv[1:]
use_llm = "--llm" in args
keypath = Path(args[args.index("--key") + 1]) if "--key" in args else ROOT / "data" / "fresh_test" / "fresh_key.json"
key = json.load(open(keypath, encoding="utf-8"))
txns = parse_statement(str(ROOT / "data" / "fresh_test" / "fresh_statement.csv"))
k = key["transactions"]

ok = sum(1 for g, t in zip(txns, k)
         if g["date"] == t["date"] and abs(g["amount"] - t["amount"]) < 0.01 and g["type"] == t["type"])
print(f"PARSING          {ok}/{len(k)} correct (rows read: {len(txns)})")

txns = categorize_transactions(txns, use_llm=use_llm)
right, wrong = 0, []
for g, t in zip(txns, k):
    if g["category"] == t["category"]:
        right += 1
    else:
        wrong.append((g["description"][:46], g["category"], g["confidence"], t["category"]))
print(f"CATEGORIZATION   {right}/{len(k)} correct ({100*right/len(k):.0f}%)  [{'rules + AI' if use_llm else 'rules only'}]")
if wrong:
    print(f"  Wrong ({len(wrong)}):  description | got | confidence | expected")
    for d, got, conf, exp in wrong:
        print(f"    {d:<46} {got:<20} {conf:<5} {exp}")

rec = detect_recurring(txns)
names = [r["merchant"].lower() for r in rec]
expected = [m.lower() for m in key["recurring_expected"]]
hit = [e for e in expected if any(e in n or n in e for n in names)]
extra = [n for n in names if not any(e in n or n in e for e in expected)]
print(f"RECURRING        found {len(hit)}/{len(expected)} expected")
print("  missed:", [e for e in expected if e not in hit] or "none")
print("  extra (not in key):", extra or "none")

flags = find_anomalies(txns, rec)
want = {(a["date"], round(a["amount"])) for a in key["anomalies"]}
got = {(f["transaction"]["date"], round(f["transaction"]["amount"])) for f in flags}
print(f"ANOMALIES        caught {len(got & want)}/{len(want)} planted, {len(got - want)} extra flag(s)")
print("  missed:", sorted(want - got) or "none")
for f in flags:
    t = f["transaction"]
    mark = "" if (t["date"], round(t["amount"])) in want else "   <-- not planted"
    print(f"  [{f['severity']:<6}] {f['reason']}{mark}")

if use_llm:
    try:
        import llm
        print("API usage:", llm.usage_summary())
    except Exception:
        pass
print("\nNOTE: this is a one-shot holdout. Report these numbers as they are; do not tune the code to this file.")
