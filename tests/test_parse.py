"""Checks parse_statement against ground_truth.json for every sample format."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.parse_statement import parse_statement, ParseError  # noqa: E402

truth = json.load(open(ROOT / "data" / "ground_truth.json", encoding="utf-8"))["transactions"]
files = sorted((ROOT / "data" / "formats").iterdir())

print(f"{'Format':<34}{'Rows':>10}{'Correct':>10}{'Accuracy':>10}")
print("-" * 64)
all_ok = True
for f in files:
    if f.suffix not in (".csv", ".xlsx", ".pdf"):
        continue
    try:
        got = parse_statement(str(f))
    except ParseError as e:
        print(f"{f.name:<34} FAILED: {e}")
        all_ok = False
        continue
    wallet = "wallet" in f.name
    correct = 0
    for g, t in zip(got, truth):
        ok = (g["date"] == t["date"] and abs(g["amount"] - t["amount"]) < 0.01 and g["type"] == t["type"])
        if not wallet:
            ok = ok and g["description"] == t["narration"]
        if ok:
            correct += 1
    n = len(truth)
    acc = 100 * correct / n if len(got) == n else 0
    all_ok &= (correct == n and len(got) == n)
    print(f"{f.name:<34}{len(got):>6}/{n:<3}{correct:>10}{acc:>9.0f}%")
print("-" * 64)
print("ALL FORMATS PASS" if all_ok else "SOME FORMATS FAILED")
