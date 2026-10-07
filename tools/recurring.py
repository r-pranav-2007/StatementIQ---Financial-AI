"""
recurring.py - find subscriptions, bills and income that repeat on a schedule.

A group of transactions is "recurring" when:
  1. it has at least MIN_CHARGES charges,
  2. every gap between consecutive dates fits ONE frequency window
     (weekly ~7 days, monthly 27-33 days, yearly ~365 days), and
  3. all amounts are within 10% of each other.

Pure Python, no LLM. Random spending (Swiggy, Uber, Amazon...) fails rule 2 or 3.
"""
import calendar
from datetime import date, timedelta
from statistics import median

MIN_CHARGES = 3
AMOUNT_TOLERANCE = 0.10  # max amount may be at most 10% above min amount

# frequency -> (shortest gap, longest gap) in days, and charges per month
FREQUENCIES = {
    "weekly": (5, 9),
    "monthly": (27, 33),
    "yearly": (360, 370),
}
PER_MONTH = {"weekly": 52 / 12, "monthly": 1.0, "yearly": 1 / 12}

IGNORED_CATEGORIES = {"Transfers", "Other"}
IGNORED_MERCHANTS = {"ATM"}
INCOME_ROUNDING = 1000  # income is grouped by (category, amount rounded to Rs 1,000)


def _group_key(txn: dict) -> tuple:
    """Debits group by merchant; credits by category + rounded amount.

    The income key lets "Salary" and "Acme Tech Solutions" land in one group.
    """
    if txn["type"] == "credit":
        return ("credit", txn["category"], round(txn["amount"] / INCOME_ROUNDING))
    return ("debit", txn["merchant"])


def _frequency(gaps: list[int]) -> str | None:
    """Return the frequency every gap fits into, or None if the gaps are irregular."""
    for name, (low, high) in FREQUENCIES.items():
        if all(low <= g <= high for g in gaps):
            return name
    return None


def _kind(category: str, txn_type: str) -> str:
    """income for credits, subscription for Subscriptions, bill for everything else.

    Rent and Bills and Utilities are bills; any other fixed regular payment
    (e.g. monthly tuition) is also treated as a bill.
    """
    if txn_type == "credit":
        return "income"
    if category == "Subscriptions":
        return "subscription"
    return "bill"


def _add_one_month(d: date) -> date:
    """Same day next month, clamped to that month's last day (Jan 31 -> Feb 28/29)."""
    year, month = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, last_day))


def _most_common(values: list[str]) -> str:
    """Most frequent value (first one wins a tie)."""
    return max(values, key=values.count)


def detect_recurring(transactions: list[dict]) -> list[dict]:
    """Return one dict per recurring payment or income found in `transactions`."""
    groups: dict[tuple, list[dict]] = {}
    for t in transactions:
        if t["category"] in IGNORED_CATEGORIES or t["merchant"] in IGNORED_MERCHANTS:
            continue
        groups.setdefault(_group_key(t), []).append(t)

    results = []
    for txns in groups.values():
        if len(txns) < MIN_CHARGES:
            continue
        txns = sorted(txns, key=lambda t: t["date"])
        dates = [date.fromisoformat(t["date"]) for t in txns]
        amounts = [t["amount"] for t in txns]

        # Rule 2: regular gaps
        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
        frequency = _frequency(gaps)
        if frequency is None:
            continue

        # Rule 3: similar amounts
        if max(amounts) > min(amounts) * (1 + AMOUNT_TOLERANCE):
            continue

        typical_amount = round(median(amounts), 2)
        if frequency == "monthly":
            next_expected = _add_one_month(dates[-1])
        else:  # weekly / yearly: last date plus the typical gap in days
            next_expected = dates[-1] + timedelta(days=round(median(gaps)))
        results.append({
            "merchant": _most_common([t["merchant"] for t in txns]),
            "amount": typical_amount,
            "frequency": frequency,
            "count": len(txns),
            "last_charged": dates[-1].isoformat(),
            "next_expected": next_expected.isoformat(),
            "kind": _kind(txns[-1]["category"], txns[-1]["type"]),
            "monthly_cost": round(typical_amount * PER_MONTH[frequency], 2),
        })

    # Income first, then bills, then subscriptions; biggest first within each.
    order = {"income": 0, "bill": 1, "subscription": 2}
    results.sort(key=lambda r: (order[r["kind"]], -r["monthly_cost"]))
    return results


def summarize_subscriptions(recurring: list[dict]) -> dict:
    """Count subscription-type debits and add up their monthly cost."""
    subs = [r for r in recurring if r["kind"] == "subscription"]
    return {"count": len(subs), "monthly_total": round(sum(r["monthly_cost"] for r in subs), 2)}
