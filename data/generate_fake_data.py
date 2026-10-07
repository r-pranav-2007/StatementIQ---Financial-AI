"""
StatementIQ - fake bank/UPI statement generator.

Creates:
  sample_statement.csv  -> what the user uploads (looks like a real bank export)
  ground_truth.json     -> the "answer key" used by tests/evaluation.py

Uses only the Python standard library. Run:  python generate_fake_data.py
Same seed = same data every time, so your accuracy numbers are reproducible.
"""
import csv
import json
import random
from datetime import date
from pathlib import Path

random.seed(42)
OUT_DIR = Path(__file__).parent

START_BALANCE = 30000.00
MONTHS = [(2026, 7), (2026, 8), (2026, 9)]  # Jul, Aug, Sep 2026

# ---------------------------------------------------------------------------
# Merchants: name -> (category, narration templates, (min, max) amount)
# Several templates per merchant = messy, realistic descriptions.
# ---------------------------------------------------------------------------
MERCHANTS = {
    "Swiggy":   ("Food Delivery", ["UPI/SWIGGY/{ref}/okaxis", "UPI-SWIGGY-{ref}@icici", "PAYTM*SWIGGY {ref}"], (150, 650)),
    "Zomato":   ("Food Delivery", ["UPI/ZOMATO/{ref}/okhdfc", "ZOMATO ONLINE ORDER {ref}"], (180, 700)),
    "Amazon":   ("Shopping", ["UPI/AMAZON PAY/{ref}/yesbank", "POS AMAZON SELLER SERVICES {ref}"], (200, 2500)),
    "Flipkart": ("Shopping", ["UPI/FLIPKART/{ref}/axisbank", "FLIPKART INTERNET PVT {ref}"], (300, 2200)),
    "Uber":     ("Transport", ["UPI/UBER INDIA/{ref}/axisbank", "UBER TRIP {ref}"], (90, 450)),
    "Ola":      ("Transport", ["UPI/OLA CABS/{ref}/okicici"], (80, 400)),
    "Rapido":   ("Transport", ["UPI/RAPIDO/{ref}/paytm", "RAPIDO BIKE {ref}"], (30, 180)),
    "BigBasket":("Groceries", ["UPI/BIGBASKET/{ref}/okhdfc", "BB DAILY {ref}"], (400, 1800)),
    "Sri Murugan Stores": ("Groceries", ["UPI/SRI MURUGAN STORES/{ref}/okicici"], (60, 500)),
    "Apollo Pharmacy": ("Health", ["UPI/APOLLO PHARMACY/{ref}/okaxis", "POS APOLLO PHARMACY {ref}"], (120, 900)),
    "BookMyShow": ("Entertainment", ["UPI/BOOKMYSHOW/{ref}/okhdfc"], (250, 800)),
}
PEOPLE = ["ARUN KUMAR", "PRIYA S", "KARTHIK R", "DIVYA M", "SURESH B"]

# (merchant, weight) used to pick random everyday spending
SPEND_WEIGHTS = [
    ("Swiggy", 5), ("Zomato", 3), ("Amazon", 2), ("Flipkart", 1), ("Uber", 3),
    ("Ola", 1), ("Rapido", 2), ("BigBasket", 2), ("Sri Murugan Stores", 3),
    ("Apollo Pharmacy", 1), ("BookMyShow", 1), ("__TRANSFER__", 3), ("__ATM__", 1),
]

# Fixed monthly items: merchant, category, day, amount, narration, is_subscription
RECURRING = [
    ("Netflix",   "Subscriptions",     3,  649.00, "NETFLIX.COM MUMBAI {ref}"),
    ("Spotify",   "Subscriptions",     8,  119.00, "UPI-SPOTIFY-AUTOPAY-{ref}"),
    ("Gym",       "Subscriptions",    10, 1500.00, "UPI/FITNESS FIRST GYM/{ref}/okaxis"),
    ("Hotstar",   "Subscriptions",    12,  299.00, "UPI/DISNEY HOTSTAR/{ref}/okhdfc"),
    ("Airtel",    "Bills and Utilities", 15, 399.00, "UPI/AIRTEL PREPAID/{ref}/airtel"),
    ("iCloud",    "Subscriptions",    20,   75.00, "APPLE.COM/BILL {ref}"),
    ("Rent",      "Rent",              5, 12000.00, "NEFT-RENT-HOUSE OWNER-{ref}"),
]
SALARY = ("Salary", "Income", 1, 45000.00, "NEFT-ACME TECH SOLUTIONS-SALARY-{ref}")


def ref() -> str:
    return "".join(random.choices("0123456789", k=12))


def make_row(d, narration, amount, kind, merchant, category, tag=None):
    return {"date": d, "narration": narration.format(ref=ref()), "amount": round(amount, 2),
            "type": kind, "merchant": merchant, "category": category, "tag": tag}


rows = []

for (y, m) in MONTHS:
    # salary + fixed monthly payments
    n, cat, day, amt, narr = SALARY
    rows.append(make_row(date(y, m, day), narr, amt, "credit", n, cat, "recurring"))
    for n, cat, day, amt, narr in RECURRING:
        rows.append(make_row(date(y, m, day), narr, amt, "debit", n, cat, "recurring"))

    # random everyday spending (16 per month)
    names = [w[0] for w in SPEND_WEIGHTS]
    weights = [w[1] for w in SPEND_WEIGHTS]
    for _ in range(16):
        pick = random.choices(names, weights)[0]
        d = date(y, m, random.randint(1, 28))
        if pick == "__TRANSFER__":
            who = random.choice(PEOPLE)
            amt = random.choice([200, 300, 500, 800, 1000, 1500])
            rows.append(make_row(d, f"UPI/{who}/{{ref}}/okaxis", amt, "debit", who.title(), "Transfers"))
        elif pick == "__ATM__":
            amt = random.choice([500, 1000, 2000, 3000])
            rows.append(make_row(d, "ATM WDL {ref} CHENNAI", amt, "debit", "ATM", "Other"))
        else:
            cat, templates, (lo, hi) = MERCHANTS[pick]
            amt = round(random.uniform(lo, hi), 0)
            rows.append(make_row(d, random.choice(templates), amt, "debit", pick, cat))

# ---------------------------------------------------------------------------
# Planted anomalies (the answer key the detector should find)
# ---------------------------------------------------------------------------
rows.append(make_row(date(2026, 8, 14), "UPI/SWIGGY/{ref}/okaxis", 389, "debit", "Swiggy", "Food Delivery", "dup"))
rows.append(make_row(date(2026, 8, 14), "UPI/SWIGGY/{ref}/okaxis", 389, "debit", "Swiggy", "Food Delivery", "dup"))
rows.append(make_row(date(2026, 8, 22), "POS AMAZON SELLER SERVICES {ref}", 18999, "debit", "Amazon", "Shopping", "spike"))
rows.append(make_row(date(2026, 9, 6), "GOOGLE*YOUTUBE PREMIUM {ref}", 129, "debit", "YouTube Premium", "Subscriptions", "new_sub"))
rows.append(make_row(date(2026, 9, 18), "UPI/VIKRAM T/{ref}/okicici", 25000, "debit", "Vikram T", "Transfers", "big_transfer"))

# sort by date (stable, so same-day order is preserved) and compute balances
rows.sort(key=lambda r: r["date"])
balance = START_BALANCE
for r in rows:
    balance += r["amount"] if r["type"] == "credit" else -r["amount"]
    r["balance"] = round(balance, 2)
assert min(r["balance"] for r in rows) >= 0, "Balance went negative - raise START_BALANCE"

# ---------------------------------------------------------------------------
# Write the CSV (looks like a typical Indian bank export)
# ---------------------------------------------------------------------------
with open(OUT_DIR / "sample_statement.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Date", "Narration", "Chq/Ref No", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"])
    for r in rows:
        ref_no = r["narration"].split("/")[2] if r["narration"].count("/") >= 2 else ""
        debit = f'{r["amount"]:.2f}' if r["type"] == "debit" else ""
        credit = f'{r["amount"]:.2f}' if r["type"] == "credit" else ""
        w.writerow([r["date"].strftime("%d/%m/%Y"), r["narration"], ref_no, debit, credit, f'{r["balance"]:.2f}'])

# ---------------------------------------------------------------------------
# Write the answer key
# ---------------------------------------------------------------------------
truth = {
    "start_balance": START_BALANCE,
    "total_rows": len(rows),
    "transactions": [
        {"date": r["date"].isoformat(), "narration": r["narration"], "amount": r["amount"],
         "type": r["type"], "merchant": r["merchant"], "category": r["category"], "tag": r["tag"]}
        for r in rows
    ],
    "recurring": [
        {"merchant": n, "amount": a, "frequency": "monthly"} for n, _, _, a, _ in RECURRING
    ] + [{"merchant": "Salary", "amount": SALARY[3], "frequency": "monthly"}],
    "anomalies": [
        {"type": "duplicate", "date": "2026-08-14", "merchant": "Swiggy", "amount": 389},
        {"type": "spike", "date": "2026-08-22", "merchant": "Amazon", "amount": 18999},
        {"type": "new_subscription", "date": "2026-09-06", "merchant": "YouTube Premium", "amount": 129},
        {"type": "big_transfer", "date": "2026-09-18", "merchant": "Vikram T", "amount": 25000},
    ],
}
with open(OUT_DIR / "ground_truth.json", "w", encoding="utf-8") as f:
    json.dump(truth, f, indent=2)

print(f"Generated {len(rows)} transactions")
print(f"  -> {OUT_DIR / 'sample_statement.csv'}")
print(f"  -> {OUT_DIR / 'ground_truth.json'}")
