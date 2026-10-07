"""
anomalies.py - flag transactions that look unusual. Pure Python, no LLM.

Rules (debits only - credits such as salary are never flagged):
  1. duplicate        : same merchant + same amount on the same date, 2+ times   (high)
  2. spike            : > 3x the median debit of its category AND > Rs 2,000
                        (skips normal charges of stable recurring payments)     (medium/high)
  3. new_subscription : Subscriptions merchant seen only in the latest month     (low)
  4. big_transfer     : Transfers debit > 5x the median transfer, to a person
                        never paid before                                       (high)
If one transaction matches several rules it is reported once, with the most
serious type.
"""
from statistics import median

SPIKE_RATIO = 3.0           # spike if amount > 3x category median ...
SPIKE_MIN_AMOUNT = 2000.0   # ... and above Rs 2,000
SPIKE_HIGH_RATIO = 5.0      # 5x or more is a high-severity spike
BIG_TRANSFER_RATIO = 5.0    # transfer > 5x median transfer ...
BIG_TRANSFER_MIN = 5000     # ... and at least Rs 5,000
STABLE_TOLERANCE = 0.10     # a recurring charge within 10% of its usual amount is normal

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}
# Tie-break when two rules give the same severity: earlier = more serious.
TYPE_PRIORITY = ["duplicate", "big_transfer", "spike", "new_subscription"]


def _inr(amount: float) -> str:
    """Format as Indian rupees: 125000 -> 'Rs 1,25,000', 389.5 -> 'Rs 389.50'."""
    whole, frac = f"{amount:.2f}".split(".")
    if len(whole) > 3:
        head, groups = whole[:-3], []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        whole = ",".join(([head] if head else []) + groups) + "," + whole[-3:]
    # "Rs" rather than the rupee sign, so Windows consoles can print it.
    return "Rs " + whole + ("" if frac == "00" else "." + frac)


def _flag(txn: dict, kind: str, reason: str, severity: str) -> dict:
    return {"transaction": txn, "type": kind, "reason": reason, "severity": severity}


def _duplicates(debits: list[dict]) -> list[dict]:
    """Flag every repeat after the first of (date, merchant, amount)."""
    seen = {}
    flags = []
    for t in debits:
        key = (t["date"], t["merchant"], t["amount"])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 2:
            flags.append(_flag(t, "duplicate",
                               f"{t['merchant']} charged {_inr(t['amount'])} more than once on "
                               f"{t['date']} - possible duplicate payment.", "high"))
    return flags


def _is_normal_recurring(txn: dict, recurring: list[dict]) -> bool:
    """True if txn is a usual charge of a recurring payment (e.g. monthly rent)."""
    for r in recurring:
        if (r["merchant"] == txn["merchant"]
                and abs(txn["amount"] - r["amount"]) <= STABLE_TOLERANCE * r["amount"]):
            return True
    return False


def _spikes(debits: list[dict], recurring: list[dict]) -> list[dict]:
    """Debits far above the typical amount for their category."""
    by_category = {}
    for t in debits:
        by_category.setdefault(t["category"], []).append(t["amount"])
    medians = {c: median(a) for c, a in by_category.items()}

    flags = []
    for t in debits:
        typical = medians[t["category"]]
        ratio = t["amount"] / typical if typical else 0
        if ratio > SPIKE_RATIO and t["amount"] > SPIKE_MIN_AMOUNT and not _is_normal_recurring(t, recurring):
            severity = "high" if ratio >= SPIKE_HIGH_RATIO else "medium"
            flags.append(_flag(t, "spike",
                               f"{_inr(t['amount'])} at {t['merchant']} is {ratio:.1f}x your usual "
                               f"{t['category']} spend of {_inr(typical)}.", severity))
    return flags


def _new_subscriptions(debits: list[dict], all_months: set[str]) -> list[dict]:
    """Subscriptions merchants whose first charge is in the latest month of data."""
    if len(all_months) < 2:  # with one month of data everything would look "new"
        return []
    latest = max(all_months)
    first_month = {}
    first_txn = {}
    for t in sorted(debits, key=lambda t: t["date"]):
        if t["category"] == "Subscriptions" and t["merchant"] not in first_month:
            first_month[t["merchant"]] = t["date"][:7]
            first_txn[t["merchant"]] = t
    return [_flag(t, "new_subscription",
                  f"New subscription: {m} charged {_inr(t['amount'])} on {t['date']}, "
                  f"not seen in earlier months.", "low")
            for m, t in first_txn.items() if first_month[m] == latest]


def _big_transfers(debits: list[dict]) -> list[dict]:
    """Large transfers to someone never paid before."""
    transfers = sorted((t for t in debits if t["category"] == "Transfers"), key=lambda t: t["date"])
    if not transfers:
        return []
    typical = median(t["amount"] for t in transfers)
    paid_before = set()
    flags = []
    for t in transfers:
        if (t["amount"] > BIG_TRANSFER_RATIO * typical and t["amount"] >= BIG_TRANSFER_MIN
                and t["merchant"] not in paid_before):
            flags.append(_flag(t, "big_transfer",
                               f"{_inr(t['amount'])} sent to {t['merchant']}, a first-time payee - "
                               f"{t['amount'] / typical:.0f}x your usual transfer of {_inr(typical)}.",
                               "high"))
        paid_before.add(t["merchant"])
    return flags


def find_anomalies(transactions: list[dict], recurring: list[dict] | None = None) -> list[dict]:
    """Return one flag per unusual transaction, sorted by date."""
    recurring = recurring or []
    debits = [t for t in transactions if t["type"] == "debit"]  # never flag credits
    all_months = {t["date"][:7] for t in transactions}

    candidates = (_duplicates(debits) + _spikes(debits, recurring)
                  + _new_subscriptions(debits, all_months) + _big_transfers(debits))

    # Keep one flag per transaction: highest severity, then TYPE_PRIORITY.
    best = {}
    for f in candidates:
        key = id(f["transaction"])
        rank = (SEVERITY_RANK[f["severity"]], -TYPE_PRIORITY.index(f["type"]))
        if key not in best or rank > best[key][0]:
            best[key] = (rank, f)
    return sorted((f for _, f in best.values()), key=lambda f: f["transaction"]["date"])
