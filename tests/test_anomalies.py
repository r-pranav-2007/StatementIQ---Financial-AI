"""Checks find_anomalies against the "anomalies" list in ground_truth.json (no-LLM mode)."""
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
from tools.recurring import detect_recurring  # noqa: E402
from tools.anomalies import find_anomalies  # noqa: E402

planted = json.load(open(ROOT / "data" / "ground_truth.json", encoding="utf-8"))["anomalies"]


def matches(flag: dict, plant: dict) -> bool:
    """Same type, date, merchant and amount."""
    t = flag["transaction"]
    return (flag["type"] == plant["type"] and t["date"] == plant["date"]
            and t["merchant"].lower() == plant["merchant"].lower()
            and abs(t["amount"] - plant["amount"]) < 0.01)


def check_file(path: Path) -> bool:
    """Run the full pipeline on one file and print the report."""
    txns = categorize_transactions(parse_statement(str(path)), use_llm=False)
    flags = find_anomalies(txns, detect_recurring(txns))

    caught = [p for p in planted if any(matches(f, p) for f in flags)]
    missed = [p for p in planted if p not in caught]
    false_alarms = [f for f in flags if not any(matches(f, p) for p in planted)]

    print("=" * 100)
    print(f"{path.name}:  caught {len(caught)}/{len(planted)} planted, {len(false_alarms)} false alarm(s)")
    print("=" * 100)
    for f in flags:
        mark = "FALSE ALARM" if f in false_alarms else "planted"
        print(f"  [{f['severity']:<6}] {f['type']:<17} {f['transaction']['date']}  ({mark})")
        print(f"           {f['reason']}")
    print(f"\nMissed ({len(missed)}): "
          + (", ".join(f"{p['type']} {p['merchant']} {p['date']}" for p in missed) or "none"))
    print(f"False alarms ({len(false_alarms)}): "
          + (", ".join(f"{f['type']} {f['transaction']['merchant']}" for f in false_alarms) or "none"))
    print()
    return not missed and not false_alarms


ok_bank = check_file(ROOT / "data" / "sample_statement.csv")
ok_wallet = check_file(ROOT / "data" / "formats" / "4_wallet_upi_app.csv")
print("PASS" if ok_bank and ok_wallet else "FAIL")
sys.exit(0 if ok_bank and ok_wallet else 1)
