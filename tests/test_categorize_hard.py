"""Honest stress test: 25 transactions the rules have never seen (no API key needed).

The rules were NOT tuned for these. A low score here shows what the LLM
fallback must handle; the confidence column shows whether the tool knew
it was unsure (< 0.95) or was confidently wrong (0.95).
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import llm  # noqa: E402
from tools.categorize import categorize_transactions  # noqa: E402

# (description, type, amount, expected category)
CASES = [
    ("UPI/RAJ TEXTILES/412398765432/okaxis", "debit", 1450, "Shopping"),
    ("UPI/CROMA RETAIL/558812349876/okhdfc", "debit", 2399, "Shopping"),
    ("UPI/RELIANCE FRESH/339912876540/okicici", "debit", 642, "Groceries"),
    ("DOMINOS PIZZA 728812349901", "debit", 549, "Food Delivery"),
    ("UPI/DECATHLON SPORTS/661234987123/okaxis", "debit", 1899, "Shopping"),
    ("POS INDIAN OIL PETROL 881234567812", "debit", 2000, "Transport"),
    ("UDEMY COURSE 449812763412", "debit", 455, "Education"),
    ("UPI/PRACTO/772398123456/okhdfc", "debit", 300, "Health"),
    ("UPI/PVR CINEMAS/118823456789/okicici", "debit", 760, "Entertainment"),
    ("POS BATA INDIA 223344556677", "debit", 1299, "Shopping"),
    ("Paid to Lenskart", "debit", 2500, "Shopping"),
    ("CLEARTRIP BOOKING 990012345678", "debit", 4520, "Transport"),
    ("UPI/JIO RECHARGE/554433221100/okaxis", "debit", 299, "Bills and Utilities"),
    ("NEFT-ABC ENGINEERING COLLEGE-FEES-776655443322", "debit", 45000, "Education"),
    ("TNEB ELECTRICITY BILL 667788990011", "debit", 1340, "Bills and Utilities"),
    ("UPI/SRI BALAJI MEDICALS/887766554433/okaxis", "debit", 230, "Health"),
    ("UPI/NILGIRIS SUPERMARKET/445566778899/okhdfc", "debit", 1120, "Groceries"),
    ("HP PETROL PUMP 334455667788", "debit", 1500, "Transport"),
    ("UPI/APOLLO HOSPITAL/221100998877/okicici", "debit", 800, "Health"),
    ("AMAZON PRIME VIDEO 556677889900", "debit", 299, "Subscriptions"),
    # Five real-looking person names (not in KNOWN_PEOPLE)
    ("UPI/RAHUL SHARMA/123987456321/okaxis", "debit", 500, "Transfers"),
    ("Paid to Meena Raghavan", "debit", 1200, "Transfers"),
    ("IMPS-ANAND P-998877665544", "debit", 2000, "Transfers"),
    ("UPI/FATIMA BEGUM/776655001122/okhdfc", "debit", 350, "Transfers"),
    ("Received from Karthik R", "credit", 700, "Transfers"),
]

txns = [{"date": "2026-10-01", "description": d, "type": ty, "amount": float(a), "balance": None}
        for d, ty, a, _ in CASES]
# Run with --llm to let the LLM check unsure rows (needs ANTHROPIC_API_KEY).
USE_LLM = "--llm" in sys.argv
MODE = "rules + LLM" if USE_LLM else "no-LLM mode"
if USE_LLM and not llm.has_api_key():
    print("WARNING: --llm given but ANTHROPIC_API_KEY is not set; results are rules only.")
rows = categorize_transactions(txns, use_llm=USE_LLM)

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
print(f"HARD TEST ({MODE}):  {correct}/{len(CASES)} correct  ->  {acc:.1f}%")
print("=" * 78)
print(f"{'Category':<22}{'Correct':>10}{'Accuracy':>10}")
for cat in sorted(total_by_cat):
    print(f"{cat:<22}{correct_by_cat[cat]:>6}/{total_by_cat[cat]:<3}"
          f"{100 * correct_by_cat[cat] / total_by_cat[cat]:>9.0f}%")

print(f"\nWrong ({len(wrong)}):")
print(f"  {'Description':<44}{'Got':<15}{'Conf':>5}  Expected")
for row, expected in wrong:
    print(f"  {row['description'][:43]:<44}{row['category']:<15}{row['confidence']:>5}  {expected}")

# Wrong answers at 0.95 are the dangerous ones: the LLM never sees them.
confident_wrong = [r for r, _ in wrong if r["confidence"] >= 0.95]
print(f"\nWrong but flagged unsure (< 0.95, would go to the LLM): {len(wrong) - len(confident_wrong)}")
print(f"Wrong AND confident (0.95, LLM never sees them):      {len(confident_wrong)}")
