"""Holdout test: 25 transactions written AFTER the rules were frozen (no API key needed).

Merchants here are in neither tests/test_categorize_hard.py nor KNOWN_MERCHANTS.
Do not edit the rules to fix failures here - that would turn this into a second
development set. This number is the honest estimate for unseen statements.
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.categorize import categorize_transactions, _find_brand  # noqa: E402

# (description, type, amount, expected category)
CASES = [
    ("UPI/SARAVANA BHAVAN/731298456120/okaxis", "debit", 420, "Food Delivery"),
    ("POS SPAR HYPERMARKET 448812093321", "debit", 2310, "Groceries"),
    ("UPI/POTHYS/552901873345/okhdfc", "debit", 3499, "Shopping"),
    ("UPI/CHUMBAK DESIGN/661209874432/okicici", "debit", 1150, "Shopping"),
    ("BHARATPE*MANI TEA STALL 220198347761", "debit", 40, "Food Delivery"),
    ("UPI/QUICK RIDE/883310298745/okaxis", "debit", 95, "Transport"),
    ("NEFT-CHENNAI METRO WATER-551290387712", "debit", 850, "Bills and Utilities"),
    ("UPI/EXCITEL BROADBAND/447719283365/okhdfc", "debit", 708, "Bills and Utilities"),
    ("UPI/ABHIBUS/339821076654/okicici", "debit", 1240, "Transport"),
    ("UPI/SCALER ACADEMY/228190376654/okaxis", "debit", 12000, "Education"),
    ("Paid to Toppr", "debit", 2999, "Education"),
    ("UPI/MFINE/661290348871/okhdfc", "debit", 399, "Health"),
    ("UPI/VASAN EYE CARE/774410293356/okaxis", "debit", 650, "Health"),
    ("UPI/AGS CINEMAS/119028376645/okicici", "debit", 540, "Entertainment"),
    ("UPI/TIMEZONE GAMES/882910374456/okhdfc", "debit", 800, "Entertainment"),
    ("GOOGLE*CRUNCHYROLL 556109283374", "debit", 99, "Subscriptions"),
    ("NEFT-LAKSHMI PG ACCOMMODATION-442901837765", "debit", 8500, "Rent"),
    ("UPI/NIVA BUPA/990128374456/okaxis", "debit", 1450, "Bills and Utilities"),
    ("UPI/ANNAPOORNA PROVISION STORE/330192847765/okicici", "debit", 610, "Groceries"),
    ("UPI/LIBAS/771029384456/okhdfc", "debit", 1899, "Shopping"),
    ("UPI/WAKEFIT/228374910056/okaxis", "debit", 6999, "Shopping"),
    ("UPI/SENTHIL KUMAR/661029384471/okaxis", "debit", 1000, "Transfers"),
    ("Paid to Lakshmi Narayanan", "debit", 450, "Transfers"),
    ("Received from Neha Gupta", "credit", 600, "Transfers"),
    ("NEFT-INFOSYS LTD-SAL-883019283746", "credit", 62000, "Income"),
]

# Guard: no holdout merchant may already be a known brand in the rules.
leaks = [d for d, *_ in CASES if _find_brand(d)]
if leaks:
    print("WARNING: these holdout rows match KNOWN_MERCHANTS, holdout is not clean:")
    for d in leaks:
        print("  ", d)

txns = [{"date": "2026-10-01", "description": d, "type": ty, "amount": float(a), "balance": None}
        for d, ty, a, _ in CASES]
rows = categorize_transactions(txns, use_llm=False)

correct = 0
total_by_cat, correct_by_cat = Counter(), Counter()
wrong = []
for row, (_, _, _, expected) in zip(rows, CASES):
    total_by_cat[expected] += 1
    if row["category"] == expected:
        correct += 1
        correct_by_cat[expected] += 1
    else:
        wrong.append((row, expected))

acc = 100 * correct / len(CASES)
print("=" * 78)
print(f"HOLDOUT TEST (no-LLM mode):  {correct}/{len(CASES)} correct  ->  {acc:.1f}%")
print("=" * 78)
print(f"{'Category':<22}{'Correct':>10}{'Accuracy':>10}")
for cat in sorted(total_by_cat):
    print(f"{cat:<22}{correct_by_cat[cat]:>6}/{total_by_cat[cat]:<3}"
          f"{100 * correct_by_cat[cat] / total_by_cat[cat]:>9.0f}%")

print(f"\nWrong ({len(wrong)}):")
print(f"  {'Description':<44}{'Got':<15}{'Conf':>5}  Expected")
for row, expected in wrong:
    print(f"  {row['description'][:43]:<44}{row['category']:<15}{row['confidence']:>5}  {expected}")

confident_wrong = [r for r, _ in wrong if r["confidence"] >= 0.85]
print(f"\nWrong but flagged unsure (< 0.85, would go to the LLM): {len(wrong) - len(confident_wrong)}")
print(f"Wrong AND confident (>= 0.85, LLM never sees them):    {len(confident_wrong)}")
