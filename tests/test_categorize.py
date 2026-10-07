"""Checks categorize_transactions against ground_truth.json (no API key needed)."""
import json
import sys
from collections import Counter
from pathlib import Path

# Print as UTF-8 so a stray symbol (e.g. the rupee sign) cannot crash the Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:  # stdout replaced by an object without reconfigure()
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.parse_statement import parse_statement  # noqa: E402
from tools.categorize import categorize_transactions, clean_merchant  # noqa: E402

TARGET = 95.0
truth = json.load(open(ROOT / "data" / "ground_truth.json", encoding="utf-8"))["transactions"]

# --- 1. Merchant cleaning examples from the spec ---------------------------
EXAMPLES = {
    "UPI/SWIGGY/813750214104/okaxis": "Swiggy",
    "UPI-SWIGGY-123@icici": "Swiggy",
    "PAYTM*SWIGGY 123": "Swiggy",
    "NETFLIX.COM MUMBAI 123": "Netflix",
    "UPI/ARUN KUMAR/123/okaxis": "Arun Kumar",
    "ATM WDL 123 CHENNAI": "ATM",
    "APPLE.COM/BILL 123": "iCloud",
    "Paid to Swiggy": "Swiggy",
}
print("Merchant cleaning examples")
print("-" * 64)
examples_ok = True
for raw, expected in EXAMPLES.items():
    got = clean_merchant(raw)
    mark = "OK  " if got == expected else "FAIL"
    examples_ok &= got == expected
    print(f"{mark} {raw:<34} -> {got}")
print()


# --- 2. Category accuracy per file -----------------------------------------
def check_file(filename: str) -> float:
    """Categorize one statement file and print the accuracy report. Returns accuracy %."""
    rows = categorize_transactions(parse_statement(str(ROOT / "data" / "formats" / filename)), use_llm=False)
    if len(rows) != len(truth):
        print(f"{filename}: row count {len(rows)} != {len(truth)} in ground truth")
        return 0.0

    total, correct = Counter(), Counter()
    wrong = []
    for got, exp in zip(rows, truth):
        total[exp["category"]] += 1
        if got["category"] == exp["category"]:
            correct[exp["category"]] += 1
        else:
            wrong.append((got, exp))

    acc = 100 * sum(correct.values()) / len(truth)
    print("=" * 64)
    print(f"{filename}:  {sum(correct.values())}/{len(truth)} correct  ->  {acc:.1f}%")
    print("=" * 64)
    print(f"{'Category':<22}{'Correct':>10}{'Accuracy':>10}")
    for cat in sorted(total):
        print(f"{cat:<22}{correct[cat]:>6}/{total[cat]:<3}{100 * correct[cat] / total[cat]:>9.0f}%")
    print(f"\nWrong ({len(wrong)}):" if wrong else "\nWrong: none")
    for got, exp in wrong:
        print(f"  {exp['date']}  {got['description'][:40]:<40}  got {got['category']!r} "
              f"({got['confidence']}), expected {exp['category']!r}")
    print()
    return acc


bank_acc = check_file("1_bank_separate_columns.csv")
wallet_acc = check_file("4_wallet_upi_app.csv")

passed = examples_ok and bank_acc >= TARGET
print(f"Bank CSV accuracy {bank_acc:.1f}% (target {TARGET:.0f}%), wallet {wallet_acc:.1f}%")
print("PASS" if passed else "FAIL")
sys.exit(0 if passed else 1)
