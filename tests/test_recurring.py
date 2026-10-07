"""Checks detect_recurring against the "recurring" list in ground_truth.json (no-LLM mode)."""
import json
import sys
from pathlib import Path

# Print as UTF-8 so a stray symbol (e.g. the rupee sign) cannot crash the Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:  # stdout replaced by an object without reconfigure()
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.parse_statement import parse_statement  # noqa: E402
from tools.categorize import categorize_transactions  # noqa: E402
from tools.recurring import detect_recurring, summarize_subscriptions  # noqa: E402

truth = json.load(open(ROOT / "data" / "ground_truth.json", encoding="utf-8"))
planted = truth["recurring"]

# Expected subscription summary, computed from the ground truth itself.
category_of = {t["merchant"]: t["category"] for t in truth["transactions"]}
planted_subs = [p for p in planted if category_of[p["merchant"]] == "Subscriptions"]
expected_summary = {"count": len(planted_subs),
                    "monthly_total": round(sum(p["amount"] for p in planted_subs), 2)}

# Merchants with no recurring tag anywhere = random spending that must never be flagged.
recurring_names = {t["merchant"] for t in truth["transactions"] if t["tag"] == "recurring"}
random_merchants = {t["merchant"] for t in truth["transactions"]} - recurring_names


def matches(found: dict, plant: dict) -> bool:
    """Same merchant, amount within 1%, same frequency."""
    return (found["merchant"].lower() == plant["merchant"].lower()
            and abs(found["amount"] - plant["amount"]) <= 0.01 * plant["amount"]
            and found["frequency"] == plant["frequency"])


def check_file(path: Path) -> bool:
    """Run parse -> categorize -> detect on one file and print the report."""
    txns = categorize_transactions(parse_statement(str(path)), use_llm=False)
    found = detect_recurring(txns)

    hits = [p for p in planted if any(matches(f, p) for f in found)]
    missed = [p for p in planted if p not in hits]
    false_pos = [f for f in found if not any(matches(f, p) for p in planted)]
    random_flagged = [f for f in found if f["merchant"] in random_merchants]
    summary = summarize_subscriptions(found)

    print("=" * 96)
    print(f"{path.name}:  found {len(hits)}/{len(planted)} planted, "
          f"{len(false_pos)} false positive(s)")
    print("=" * 96)
    print(f"{'Merchant':<14}{'Amount':>10}  {'Freq':<9}{'Count':>5}  {'Last':<12}"
          f"{'Next':<12}{'Kind':<14}{'Monthly':>10}")
    for f in found:
        print(f"{f['merchant']:<14}{f['amount']:>10.2f}  {f['frequency']:<9}{f['count']:>5}  "
              f"{f['last_charged']:<12}{f['next_expected']:<12}{f['kind']:<14}{f['monthly_cost']:>10.2f}")

    print(f"\nMissed ({len(missed)}): " + (", ".join(p["merchant"] for p in missed) or "none"))
    print(f"False positives ({len(false_pos)}): "
          + (", ".join(f["merchant"] for f in false_pos) or "none"))
    print(f"Random spending flagged ({len(random_flagged)}): "
          + (", ".join(f["merchant"] for f in random_flagged) or "none"))
    summary_ok = summary == expected_summary
    print(f"Subscriptions: {summary}  expected {expected_summary}  "
          f"{'OK' if summary_ok else 'MISMATCH'}\n")
    return not missed and not false_pos and summary_ok


ok_bank = check_file(ROOT / "data" / "sample_statement.csv")
ok_wallet = check_file(ROOT / "data" / "formats" / "4_wallet_upi_app.csv")
print("PASS" if ok_bank and ok_wallet else "FAIL")
sys.exit(0 if ok_bank and ok_wallet else 1)
