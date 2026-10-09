"""
query.py - answer plain-English questions about a statement, without hallucinated numbers.

Three steps:
  1. INTENT  : question -> spec {intent, category, merchant, month, month2, type, top_n}
               via llm.question_to_spec, or a keyword parser when there is no API key.
  2. COMPUTE : plain Python runs the spec on the transactions -> exact numbers.
  3. PHRASE  : llm.phrase_answer rewords ONLY those numbers (or a template without a key).
               An LLM answer containing any number not in "numbers" is thrown away and
               the template is used instead.

Definitions used everywhere:
  * spending / expenses = all debits (money out), including transfers and cash.
  * top merchants leave out Transfers (people) and ATM (cash), which are not merchants.
"""
import re
from datetime import date

import llm
from tools.anomalies import _inr as inr  # Indian rupee formatting: Rs 1,25,000
from tools.anomalies import find_anomalies
from tools.categorize import CATEGORIES, categorize_transactions
from tools.recurring import detect_recurring, summarize_subscriptions

DEFAULT_TOP_N = 3
DISCRETIONARY = ["Food Delivery", "Shopping", "Entertainment"]  # easy-to-cut categories
DISCRETIONARY_SHOWN = 2

# ---------------------------------------------------------------------------
# Step 1: keyword fallback parser (used when there is no API key)
# ---------------------------------------------------------------------------

MONTH_NUMBERS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
                 "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
MONTH_PATTERN = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b(?:\s*,?\s*(\d{4}))?", re.I)
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10}

# Everyday words -> category.
CATEGORY_WORDS = {
    "food delivery": "Food Delivery", "food": "Food Delivery", "takeaway": "Food Delivery",
    "eating out": "Food Delivery", "restaurants": "Food Delivery", "dining": "Food Delivery",
    "groceries": "Groceries", "grocery": "Groceries", "supermarket": "Groceries",
    "shopping": "Shopping", "clothes": "Shopping",
    "transport": "Transport", "travel": "Transport", "commute": "Transport", "cab": "Transport",
    "cabs": "Transport", "taxi": "Transport", "ride": "Transport", "rides": "Transport",
    "bills": "Bills and Utilities", "utilities": "Bills and Utilities", "phone": "Bills and Utilities",
    "mobile": "Bills and Utilities", "electricity": "Bills and Utilities", "internet": "Bills and Utilities",
    "subscriptions": "Subscriptions", "subscription": "Subscriptions", "streaming": "Subscriptions",
    "rent": "Rent", "housing": "Rent",
    "education": "Education", "courses": "Education", "fees": "Education", "tuition": "Education",
    "health": "Health", "medical": "Health", "medicine": "Health", "medicines": "Health",
    "pharmacy": "Health", "doctor": "Health",
    "entertainment": "Entertainment", "movies": "Entertainment", "movie": "Entertainment",
    "investments": "Investments", "investment": "Investments", "sip": "Investments",
    "mutual fund": "Investments", "mutual funds": "Investments", "stocks": "Investments",
    "transfers": "Transfers", "transfer": "Transfers", "friends": "Transfers", "family": "Transfers",
    "income": "Income", "salary": "Income", "earnings": "Income",
}

# Intent keyword patterns, checked in this order (first match wins).
INTENT_PATTERNS = [
    ("cut_suggestions", r"\b(cut|cut back|cut down|reduce|trim|spend less|save more|save money)\b"),
    ("income_vs_expenses", r"\b(income|earn|earned|earnings)\b.*\b(expenses?|spend|spent|spending)\b"
                           r"|\b(expenses?|spend|spent|spending)\b.*\b(income|earn|earned|earnings)\b"
                           r"|\b(save|saved|savings|net)\b"),
    ("compare_months", None),  # decided by "two months mentioned", see _parse_intent
    ("transaction_count", r"\bhow many\b|\bnumber of\b|\bcount\b"),
    ("average_transaction", r"\b(average|avg|mean|typical)\b"),
    ("top_merchants", r"\btop\b|\b(most|favou?rite)\b.*\b(merchants?|places?|shops?|stores?|brands?|where)\b"
                      r"|\b(merchants?|places?|shops?|stores?|brands?|where)\b.*\b(most|favou?rite)\b"),
    ("biggest_expense", r"\b(biggest|largest|highest|most expensive|costliest|priciest)\b"),
    ("category_breakdown", r"\b(breakdown|by category|per category|categories|split)\b"),
    ("monthly_trend", r"\b(each month|per month|month by month|every month|trend)\b"),
    ("total_income", r"\b(income|earn|earned|received|salary|credited)\b"),
    ("total_spend", r"\b(spend|spent|spending|pay|paid|cost|costs|expenses?|went on|go on|bill)\b"),
]


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.I) is not None


def _is_month_may(question: str, match: re.Match) -> bool:
    """'may' is the month only after in/for/during/of, or as 'May' right before a number/year."""
    after_preposition = re.search(r"\b(in|for|during|of)\s+$", question[:match.start()], re.I)
    capital_before_number = (match.group(1) == "May"
                             and re.match(r"\s*,?\s*\d", question[match.end(1):]) is not None)
    return bool(after_preposition or capital_before_number)


def _parse_months(question: str, months: list[str]) -> list[str]:
    """Months mentioned in the question as "YYYY-MM", in the order they appear."""
    found = []  # (position, "YYYY-MM")
    latest_year = months[-1][:4] if months else str(date.today().year)
    for m in MONTH_PATTERN.finditer(question):
        if m.group(1).lower() == "may" and not _is_month_may(question, m):
            continue  # "may" as in "I may have..." is not the month
        number = MONTH_NUMBERS[m.group(1).lower()[:3]]
        year = m.group(2)
        options = [ym for ym in months if int(ym[5:]) == number and (year is None or ym[:4] == year)]
        # Not in the data -> still keep it, so the answer says "no data" instead of "all months".
        found.append((m.start(), options[-1] if options else f"{year or latest_year}-{number:02d}"))
    rel = re.search(r"\b(last|this|latest|current|past) month\b", question, re.I)
    if rel and months:
        found.append((rel.start(), months[-1]))
    rel = re.search(r"\b(previous|prior) month\b|\bmonth before\b", question, re.I)
    if rel and len(months) >= 2:
        found.append((rel.start(), months[-2]))
    ordered = [ym for _, ym in sorted(found)]
    return list(dict.fromkeys(ordered))  # drop repeats, keep order


def _longest_word_match(question: str, words) -> str | None:
    """Longest phrase from `words` found in the question as whole words."""
    for w in sorted(words, key=len, reverse=True):
        if re.search(r"(?<![a-z0-9])" + re.escape(w.lower()) + r"(?![a-z0-9])", question.lower()):
            return w
    return None


def _parse_top_n(question: str) -> int | None:
    """First small number in the question ("top 3", "five places")."""
    for token in re.findall(r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b",
                            question, re.I):
        n = int(token) if token.isdigit() else NUMBER_WORDS[token.lower()]
        if 1 <= n <= 20:
            return n
    return None


def _parse_intent(question: str, n_months: int, has_filter: bool) -> str:
    for intent, pattern in INTENT_PATTERNS:
        if intent == "compare_months":
            if n_months >= 2:
                return intent
        elif _has(pattern, question):
            return intent
    return "total_spend" if has_filter else "unknown"


def parse_question(question: str, categories: list[str], merchants: list[str],
                   months: list[str]) -> dict:
    """Keyword-based fallback for llm.question_to_spec (same output format)."""
    found_months = _parse_months(question, months)
    word = _longest_word_match(question, CATEGORY_WORDS)
    category = CATEGORY_WORDS[word] if word and CATEGORY_WORDS[word] in categories else None
    merchant = _longest_word_match(question, merchants)
    intent = _parse_intent(question, len(found_months), bool(found_months or category or merchant))
    return {
        "intent": intent,
        "category": category,
        "merchant": merchant,
        "month": found_months[0] if found_months else None,
        "month2": found_months[1] if len(found_months) > 1 else None,
        "type": None,
        "top_n": _parse_top_n(question) if intent == "top_merchants" else None,
    }


# ---------------------------------------------------------------------------
# Step 2: compute (plain Python, exact numbers)
# ---------------------------------------------------------------------------

def _month_label(ym: str) -> str:
    """'2026-08' -> 'August 2026'."""
    return date(int(ym[:4]), int(ym[5:]), 1).strftime("%B %Y")


def _date_label(iso: str) -> str:
    """'2026-09-18' -> '18 Sep 2026'."""
    d = date.fromisoformat(iso)
    return f"{d.day} {d.strftime('%b %Y')}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _filter(txns: list[dict], month=None, category=None, merchant=None, type_=None) -> list[dict]:
    return [t for t in txns
            if (month is None or t["date"][:7] == month)
            and (category is None or t["category"] == category)
            and (merchant is None or t["merchant"] == merchant)
            and (type_ is None or t["type"] == type_)]


def _total(rows: list[dict]) -> float:
    return round(sum(t["amount"] for t in rows), 2)


def _scope(spec: dict, numbers: dict, use_month: bool = True) -> str:
    """Describe the filters in words (" on Groceries at BigBasket in July 2026")
    and record the labels in `numbers`."""
    text = ""
    if spec.get("category"):
        numbers["category"] = spec["category"]
        text += f" on {spec['category']}"
    if spec.get("merchant"):
        numbers["merchant"] = spec["merchant"]
        text += f" at {spec['merchant']}"
    if use_month and spec.get("month"):
        numbers["month"] = _month_label(spec["month"])
        text += f" in {numbers['month']}"
    return text


def _no_match(spec: dict) -> tuple[dict, list, str]:
    numbers = {"transactions": 0}
    scope = _scope(spec, numbers)
    return numbers, [], f"I found no matching transactions{scope}."


def _spend_rows(txns, spec, type_="debit"):
    return _filter(txns, spec.get("month"), spec.get("category"), spec.get("merchant"),
                   spec.get("type") or type_)


def _total_spend(txns, spec):
    rows = _spend_rows(txns, spec)
    if not rows:
        return _no_match(spec)
    numbers = {"total_spent": _total(rows), "transactions": len(rows)}
    scope = _scope(spec, numbers)
    return numbers, rows, (f"You spent {inr(numbers['total_spent'])}{scope} "
                           f"across {_plural(len(rows), 'transaction')}.")


def _total_income(txns, spec):
    rows = _spend_rows(txns, spec, type_="credit")
    if not rows:
        return _no_match(spec)
    numbers = {"total_received": _total(rows), "transactions": len(rows)}
    scope = _scope(spec, numbers)
    return numbers, rows, (f"You received {inr(numbers['total_received'])}{scope} "
                           f"across {_plural(len(rows), 'credit')}.")


def _compare_months(txns, spec, months):
    first, second = spec.get("month") or spec.get("month2"), spec.get("month2")
    if first == second:
        second = None
    if first and not second and first in months and months.index(first) > 0:
        second = months[months.index(first) - 1]  # "compare August" -> with July
    if not (first and second):
        return _total_spend(txns, spec)
    earlier, later = sorted([first, second])
    rows_e = _filter(txns, earlier, spec.get("category"), spec.get("merchant"), "debit")
    rows_l = _filter(txns, later, spec.get("category"), spec.get("merchant"), "debit")
    a, b = _total(rows_e), _total(rows_l)
    numbers = {"earlier_month": _month_label(earlier), "earlier_total": a,
               "later_month": _month_label(later), "later_total": b,
               "difference": round(abs(b - a), 2)}
    scope = _scope(spec, numbers, use_month=False)
    later_label, earlier_label = numbers["later_month"], numbers["earlier_month"]
    if a == b:
        intro = f"You spent the same{scope} in {later_label} and {earlier_label}."
    else:
        word = "more" if b > a else "less"
        intro = f"You spent {word}{scope} in {later_label} than in {earlier_label}."
    difference = f"difference: {inr(numbers['difference'])}"
    if a:
        numbers["percent_change"] = round(abs(b - a) / a * 100, 1)
        difference += f" ({numbers['percent_change']}%)"
    items = [f"{later_label}: {inr(b)}", f"{earlier_label}: {inr(a)}", difference]
    return numbers, rows_e + rows_l, intro, items, "; ".join(items) + "."


def _top_merchants(txns, spec):
    rows = [t for t in _filter(txns, spec.get("month"), spec.get("category"), None, "debit")
            if t["category"] != "Transfers" and t["merchant"] != "ATM"]
    if not rows:
        return _no_match(spec)
    totals = {}
    for t in rows:
        totals[t["merchant"]] = totals.get(t["merchant"], 0) + t["amount"]
    top = sorted(totals.items(), key=lambda kv: -kv[1])[:spec.get("top_n") or DEFAULT_TOP_N]
    numbers = {"top_n": len(top),
               "merchants": [{"merchant": m, "total": round(v, 2)} for m, v in top]}
    scope = _scope({**spec, "merchant": None}, numbers)
    items = [f"{m['merchant']} ({inr(m['total'])})" for m in numbers["merchants"]]
    names = {m for m, _ in top}
    return (numbers, [t for t in rows if t["merchant"] in names],
            f"Your top {len(top)} merchants{scope} are:", items, ", ".join(items) + ".")


def _average(txns, spec):
    rows = _spend_rows(txns, spec)
    if not rows:
        return _no_match(spec)
    total = _total(rows)
    numbers = {"average": round(total / len(rows), 2), "transactions": len(rows), "total_spent": total}
    scope = _scope(spec, numbers)
    return numbers, rows, (f"Your average spend per transaction{scope} is {inr(numbers['average'])}, "
                           f"over {_plural(len(rows), 'transaction')} totalling {inr(total)}.")


def _biggest(txns, spec):
    rows = _spend_rows(txns, spec)
    if not rows:
        return _no_match(spec)
    t = max(rows, key=lambda r: r["amount"])
    numbers = {"amount": t["amount"], "date": _date_label(t["date"])}
    scope = _scope(spec, numbers)
    numbers["merchant"], numbers["category"] = t["merchant"], t["category"]
    preposition = "to" if t["category"] == "Transfers" else "at"
    return numbers, [t], (f"Your biggest expense{scope} was {inr(t['amount'])} {preposition} "
                          f"{t['merchant']} ({t['category']}) on {numbers['date']}.")


def _count(txns, spec):
    rows = _filter(txns, spec.get("month"), spec.get("category"), spec.get("merchant"), spec.get("type"))
    numbers = {"transactions": len(rows)}
    scope = _scope(spec, numbers)
    return numbers, rows, f"There are {_plural(len(rows), 'transaction')}{scope}."


def _income_vs_expenses(txns, spec):
    rows = _filter(txns, spec.get("month"))
    if not rows:
        return _no_match({"month": spec.get("month")})
    income = _total([t for t in rows if t["type"] == "credit"])
    expenses = _total([t for t in rows if t["type"] == "debit"])
    net = round(income - expenses, 2)
    numbers = {"income": income, "expenses": expenses, "net": abs(net)}
    scope = _scope({"month": spec.get("month")}, numbers)
    start = f"In {numbers['month']}" if scope else "Overall"
    if net >= 0:
        text = f"{start} you earned {inr(income)} and spent {inr(expenses)}, so you saved {inr(abs(net))}"
        if income:
            numbers["savings_rate_percent"] = round(net / income * 100, 1)
            text += f" ({numbers['savings_rate_percent']}% of your income)"
    else:
        text = (f"{start} you earned {inr(income)} but spent {inr(expenses)}, "
                f"which is {inr(abs(net))} more than you earned")
    return numbers, rows, text + "."


def _cut_suggestions(txns, spec, months):
    subs = [r for r in detect_recurring(txns) if r["kind"] == "subscription"]
    summary = summarize_subscriptions(subs)
    rows = _filter(txns, spec.get("month"), None, None, "debit")
    by_cat = {}
    for t in rows:
        if t["category"] in DISCRETIONARY:
            by_cat[t["category"]] = by_cat.get(t["category"], 0) + t["amount"]
    top = sorted(by_cat.items(), key=lambda kv: -kv[1])[:DISCRETIONARY_SHOWN]

    period = (_month_label(spec["month"]) if spec.get("month")
              else f"{_month_label(months[0])} to {_month_label(months[-1])}" if months else "")
    numbers = {"subscription_count": summary["count"],
               "subscriptions_monthly_total": summary["monthly_total"],
               "subscriptions": [{"merchant": s["merchant"], "monthly_cost": s["monthly_cost"]} for s in subs],
               "period": period,
               "discretionary": [{"category": c, "total": round(v, 2)} for c, v in top]}
    sub_items = [f"{s['merchant']} ({inr(s['monthly_cost'])})" for s in subs]
    disc_items = [f"{d['category']} ({inr(d['total'])})" for d in numbers["discretionary"]]
    parts = []
    if subs:
        parts.append(f"Subscriptions, {inr(summary['monthly_total'])} a month in total: "
                     + ", ".join(sub_items) + ".")
    else:
        parts.append("No recurring subscriptions found.")
    if top:
        parts.append(f"Biggest discretionary spending over {period}: " + " and ".join(disc_items) + ".")
    invested = round(sum(t["amount"] for t in rows if t["category"] == "Investments"), 2)
    if invested:  # a saving, so shown for context but never offered as something to cut
        numbers["investments"] = {"total": invested}
        note = f"Investments ({inr(invested)}) count as saving, so they are not something to cut."
        parts.append(note)
        disc_items.append(note)
    supporting = [t for t in rows if t["category"] in dict(top) or t["merchant"] in
                  {s["merchant"] for s in subs}]
    intro = (f"You have {_plural(summary['count'], 'subscription')}, and these are the "
             f"easiest areas to cut back:")
    return numbers, supporting, intro, sub_items + disc_items, " ".join(parts)


def _category_breakdown(txns, spec):
    rows = _filter(txns, spec.get("month"), None, spec.get("merchant"), "debit")
    if not rows:
        return _no_match(spec)
    totals = {}
    for t in rows:
        totals[t["category"]] = totals.get(t["category"], 0) + t["amount"]
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    numbers = {"total_spent": _total(rows),
               "categories": [{"category": c, "total": round(v, 2)} for c, v in ranked]}
    scope = _scope({"month": spec.get("month")}, numbers)
    listing = ", ".join(f"{c['category']} {inr(c['total'])}" for c in numbers["categories"])
    return numbers, rows, f"You spent {inr(numbers['total_spent'])}{scope}: {listing}."


def _monthly_trend(txns, spec, months):
    rows = _filter(txns, None, spec.get("category"), spec.get("merchant"), "debit")
    if not rows:
        return _no_match(spec)
    numbers = {"months": [{"month": _month_label(m), "total": _total(_filter(rows, m))} for m in months]}
    scope = _scope({**spec, "month": None}, numbers)
    listing = ", ".join(f"{m['month']} {inr(m['total'])}" for m in numbers["months"])
    return numbers, rows, f"Your spending{scope} by month: {listing}."


def _recurring(txns, spec):
    found = detect_recurring(txns)
    numbers = {"recurring": [{"merchant": r["merchant"], "amount": r["amount"],
                              "frequency": r["frequency"]} for r in found]}
    if not found:
        return numbers, [], "I found no recurring payments."
    listing = ", ".join(f"{r['merchant']} {inr(r['amount'])} {r['frequency']}" for r in found)
    names = {r["merchant"] for r in found}
    return numbers, [t for t in txns if t["merchant"] in names], f"Your recurring payments: {listing}."


def _anomalies(txns, spec):
    flags = find_anomalies(txns, detect_recurring(txns))
    numbers = {"alerts": [f["reason"] for f in flags], "count": len(flags)}
    if not flags:
        return numbers, [], "Nothing unusual found."
    return numbers, [f["transaction"] for f in flags], \
        f"I found {_plural(len(flags), 'unusual transaction')}: " + " ".join(numbers["alerts"])


def _list_transactions(txns, spec):
    rows = _filter(txns, spec.get("month"), spec.get("category"), spec.get("merchant"), spec.get("type"))
    if not rows:
        return _no_match(spec)
    numbers = {"transactions": len(rows), "total": _total(rows)}
    scope = _scope(spec, numbers)
    return numbers, rows, f"I found {_plural(len(rows), 'transaction')}{scope} totalling {inr(numbers['total'])}."


UNKNOWN_ANSWER = ("Sorry, I can't answer that yet. Try asking about spending in a month, on a "
                  "category or at a merchant, comparing months, top merchants, averages, your "
                  "biggest expense, income vs expenses, or what you could cut.")


def _compute(spec: dict, txns: list[dict], months: list[str]) -> tuple[dict, list, str, list, str]:
    """Run the spec.

    Returns (numbers, supporting transactions, template answer, list items, list text).
    List-type handlers return (numbers, rows, intro, items, list_text); the others
    return (numbers, rows, template) and get no list.
    """
    result = _run_handler(spec, txns, months)
    if len(result) == 5:
        numbers, rows, intro, items, list_text = result
        return numbers, rows, f"{intro} {list_text}", items, list_text
    numbers, rows, template = result
    return numbers, rows, template, [], ""


def _run_handler(spec: dict, txns: list[dict], months: list[str]) -> tuple:
    """Call the handler for spec["intent"]."""
    intent = spec["intent"]
    if intent == "total_spend":
        return _total_spend(txns, spec)
    if intent == "total_income":
        return _total_income(txns, spec)
    if intent == "compare_months":
        return _compare_months(txns, spec, months)
    if intent == "top_merchants":
        return _top_merchants(txns, spec)
    if intent == "average_transaction":
        return _average(txns, spec)
    if intent == "biggest_expense":
        return _biggest(txns, spec)
    if intent == "transaction_count":
        return _count(txns, spec)
    if intent == "income_vs_expenses":
        return _income_vs_expenses(txns, spec)
    if intent == "cut_suggestions":
        return _cut_suggestions(txns, spec, months)
    if intent == "category_breakdown":
        return _category_breakdown(txns, spec)
    if intent == "monthly_trend":
        return _monthly_trend(txns, spec, months)
    if intent == "recurring":
        return _recurring(txns, spec)
    if intent == "anomalies":
        return _anomalies(txns, spec)
    if intent == "list_transactions":
        return _list_transactions(txns, spec)
    return {}, [], UNKNOWN_ANSWER


# ---------------------------------------------------------------------------
# Step 3: phrase + hallucination guard
# ---------------------------------------------------------------------------

NUMBER_PATTERN = re.compile(r"\d+(?:,\d+)*(?:\.\d+)?")


def _numbers_in_text(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in NUMBER_PATTERN.findall(text)]


def _allowed_numbers(value) -> list[float]:
    """Every number inside `numbers`, including digits inside label strings."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return _numbers_in_text(value)
    if isinstance(value, dict):
        return [n for v in value.values() for n in _allowed_numbers(v)]
    if isinstance(value, list):
        return [n for v in value for n in _allowed_numbers(v)]
    return []


def unsupported_numbers(answer: str, numbers: dict) -> list[float]:
    """Numbers in `answer` that do not appear anywhere in `numbers`."""
    allowed = _allowed_numbers(numbers)
    return [n for n in _numbers_in_text(answer) if not any(abs(n - a) < 0.005 for a in allowed)]


# Keys whose values are rupee amounts; these are pre-formatted before the LLM sees them.
AMOUNT_KEYS = {"total_spent", "total_received", "earlier_total", "later_total", "difference",
               "total", "average", "amount", "income", "expenses", "net",
               "subscriptions_monthly_total", "monthly_cost"}

# Added to the question for list-type intents: the LLM writes only the opening sentence.
INTRO_ONLY = ("\n\n(Write ONE short introductory sentence only. The full list with every "
              "amount is added right after your sentence, so do not list the items.)")


def _format_for_llm(value, key: str | None = None):
    """Copy of `numbers` with amounts as ready-made strings ("Rs 2,127"), so the model only copies."""
    if isinstance(value, dict):
        return {k: _format_for_llm(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_format_for_llm(v, key) for v in value]
    if key in AMOUNT_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
        return inr(value)
    return value


MAX_INTRO_CHARS = 200


def _is_one_sentence(text: str) -> bool:
    """A short single-line, single-sentence intro (a period inside 8.8 or 1,634.45 is fine)."""
    sentence_ends = re.findall(r"[.!?](?=\s|$)", text)
    return (0 < len(text) <= MAX_INTRO_CHARS and "\n" not in text
            and len(sentence_ends) <= 1 and "*" not in text)


def _phrase_with_llm(question: str, numbers: dict, items: list[str], list_text: str) -> str | None:
    """LLM-phrased answer, or None if it fails any check (caller then uses the template)."""
    formatted = _format_for_llm(numbers)
    if list_text:
        # List-type intent: LLM writes only the intro (it sees all numbers for context);
        # the full list is appended unchanged.
        intro = llm.phrase_answer(question + INTRO_ONLY, formatted).strip()
        if not _is_one_sentence(intro):
            return None
        candidate = f"{intro} {list_text}"
    else:
        candidate = llm.phrase_answer(question, formatted)

    # phrase_answer returns a plain "key: value" dump when the call fails - not an answer.
    dump = "\n".join(f"{k}: {v}" for k, v in formatted.items())
    if not candidate.strip() or dump in candidate:
        return None
    # Every number must exist in the raw or the formatted numbers.
    if unsupported_numbers(candidate, {"raw": numbers, "formatted": formatted}):
        return None
    # Every list item must survive.
    if any(item not in candidate for item in items):
        return None
    return candidate


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def answer_query(question: str, transactions: list[dict], use_llm: bool = True) -> dict:
    """Answer a question about the transactions.

    Returns {"answer": str, "numbers": dict, "supporting_transactions": list}.
    Every number in "answer" comes from "numbers", which is computed in Python.
    """
    txns = transactions
    if any("category" not in t for t in txns):
        txns = categorize_transactions(txns, use_llm=use_llm)

    months = sorted({t["date"][:7] for t in txns})
    merchants = sorted({t["merchant"] for t in txns})
    llm_on = use_llm and llm.has_api_key()

    # 1. INTENT
    spec = llm.question_to_spec(question, CATEGORIES, merchants, months) if llm_on else None
    if not spec or spec["intent"] == "unknown":
        spec = parse_question(question, CATEGORIES, merchants, months)

    # 2. COMPUTE
    numbers, supporting, template, items, list_text = _compute(spec, txns, months)

    # 3. PHRASE (LLM may only reword; unknown number or missing list item -> template)
    answer = template
    if llm_on and numbers:
        answer = _phrase_with_llm(question, numbers, items, list_text) or template
    return {"answer": answer, "numbers": numbers, "supporting_transactions": supporting}
