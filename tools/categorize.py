"""
categorize.py - give every transaction a clean merchant name and a category.

Three steps:
  1. Merchant cleaning : "UPI/SWIGGY/8137.../okaxis" -> "Swiggy"
  2. Keyword rules     : "Swiggy" -> "Food Delivery"   (confidence 0.95)
  3. LLM fallback      : unknown merchants and unsure person guesses go to
                         llm.classify_batch() in ONE batch. With no API key,
                         unknowns stay "Other" (0.3), person guesses "Transfers" (0.6)
"""
import re

CATEGORIES = [
    "Food Delivery", "Groceries", "Shopping", "Transport", "Bills and Utilities",
    "Subscriptions", "Rent", "Education", "Health", "Entertainment",
    "Transfers", "Income", "Other",
]

# Confidence scores for each way a category can be decided.
RULE_CONFIDENCE = 0.95      # keyword rule, or a payee in KNOWN_PEOPLE
PERSON_GUESS_CONFIDENCE = 0.60  # name looks like a person but isn't a known contact
CREDIT_CONFIDENCE = 0.70    # money received from a business, assumed Income
LLM_CONFIDENCE = 0.70       # category suggested by the LLM
UNKNOWN_CONFIDENCE = 0.30   # nobody knew, fell back to "Other"

# ---------------------------------------------------------------------------
# Step 1: merchant cleaning
# ---------------------------------------------------------------------------

# Known brands: if the keyword appears in the text, use the clean name.
# Checked top to bottom, so put more specific keywords first.
KNOWN_MERCHANTS = {
    "swiggy": "Swiggy",
    "zomato": "Zomato",
    "netflix": "Netflix",
    "spotify": "Spotify",
    "hotstar": "Hotstar",
    "youtube": "YouTube Premium",
    "apple.com": "iCloud",
    "icloud": "iCloud",
    "gym": "Gym",
    "bigbasket": "BigBasket",
    "bb daily": "BigBasket",
    "amazon": "Amazon",
    "flipkart": "Flipkart",
    "uber": "Uber",
    "ola": "Ola",
    "rapido": "Rapido",
    "airtel": "Airtel",
    "jio": "Jio",
    "bookmyshow": "BookMyShow",
    "atm": "ATM",
    "salary": "Salary",
    "rent": "Rent",
}

# Wallet apps write "Paid to X" / "Received from X".
WALLET_PATTERN = re.compile(r"^(paid to|sent to|received from|payment to|payment from)\s+(.+)$", re.I)

# Prefixes banks put in front of the real name.
PREFIX_PATTERN = re.compile(r"^(UPI|NEFT|IMPS|RTGS|POS|PAYTM|GOOGLE|ACH|NACH)\s*[/\-*]?\s*", re.I)


def _has_word(keyword: str, text: str) -> bool:
    """True if `keyword` appears in `text` as a whole word (case-insensitive)."""
    return re.search(r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])", text.lower()) is not None


def _extract_name(description: str) -> tuple[str, bool]:
    """Pull the payee/merchant part out of a raw description.

    Returns (name, is_payee). is_payee is True when the description clearly
    names who was paid (UPI or wallet style), which is how we spot
    person-to-person transfers later.
    """
    text = description.strip()

    # "Paid to Swiggy", "Received from ACME TECH SOLUTIONS"
    m = WALLET_PATTERN.match(text)
    if m:
        return m.group(2).strip(), True

    # "UPI/ARUN KUMAR/123/okaxis" -> "ARUN KUMAR"
    if text.upper().startswith("UPI/"):
        return text.split("/")[1].strip(), True

    # "UPI-SWIGGY-123@icici", "NEFT-RENT-HOUSE OWNER-329..." -> first real segment
    if re.match(r"^(UPI|NEFT|IMPS|RTGS)-", text, re.I):
        return text.split("-")[1].split("@")[0].strip(), True

    # "PAYTM*SWIGGY 123", "POS AMAZON SELLER 841..." -> drop prefix
    text = PREFIX_PATTERN.sub("", text)
    # Drop reference numbers (any word with 6+ digits) and "/BILL"-style suffixes
    text = re.sub(r"\S*\d{6,}\S*", "", text)
    text = text.split("/")[0]
    return " ".join(text.split()), False


def clean_merchant(description: str) -> str:
    """Turn a messy bank/UPI description into a readable merchant name."""
    name, _ = _extract_name(description)

    # First look for a known brand in the extracted name, then in the full text
    # (e.g. "NEFT-ACME TECH SOLUTIONS-SALARY-..." only says SALARY at the end).
    for text in (name, description):
        for keyword, clean in KNOWN_MERCHANTS.items():
            if _has_word(keyword, text):
                return clean

    # Unknown merchant: tidy it up. "SRI MURUGAN STORES" -> "Sri Murugan Stores"
    name = re.sub(r"\.com\b", "", name, flags=re.I)
    return name.title() if name.isupper() or name.islower() else name


# ---------------------------------------------------------------------------
# Step 2: keyword rules
# ---------------------------------------------------------------------------

# keyword -> category. Matched as whole words against the clean merchant name.
KEYWORD_RULES = {
    "swiggy": "Food Delivery", "zomato": "Food Delivery",
    "amazon": "Shopping", "flipkart": "Shopping",
    "uber": "Transport", "ola": "Transport", "rapido": "Transport",
    "bigbasket": "Groceries", "stores": "Groceries", "supermarket": "Groceries",
    "netflix": "Subscriptions", "spotify": "Subscriptions", "hotstar": "Subscriptions",
    "gym": "Subscriptions", "icloud": "Subscriptions", "youtube": "Subscriptions",
    "airtel": "Bills and Utilities", "jio": "Bills and Utilities",
    "rent": "Rent",
    "salary": "Income",
    "pharmacy": "Health",
    "bookmyshow": "Entertainment",
    "atm": "Other",
}

# Words that mean "this is a business, not a person".
BUSINESS_WORDS = {
    "stores", "store", "mart", "pharmacy", "hospital", "clinic", "cafe", "restaurant",
    "hotel", "pvt", "ltd", "limited", "solutions", "services", "technologies", "tech",
    "enterprises", "traders", "india", "online", "foods", "bank", "school", "college",
    "academy", "shop", "agency", "cabs", "motors", "fuels", "petroleum",
    "textiles", "retail", "medicals", "electronics", "bakery", "sweets", "jewellers",
    "fuel", "petrol", "mobiles",
}

# People the user has told us about (their friends/family). A payee in this
# list is a sure Transfer; any other person-like name is only a guess (0.6).
KNOWN_PEOPLE = {"arun kumar", "suresh b", "divya m", "priya s", "vikram t"}


def _looks_like_person(name: str) -> bool:
    """Guess whether a payee name is a person, e.g. "Arun Kumar" or "Suresh B"."""
    words = name.lower().split()
    return (
        1 <= len(words) <= 3
        and all(w.isalpha() for w in words)
        and not any(w in BUSINESS_WORDS for w in words)
    )


def _rule_category(txn: dict, merchant: str) -> tuple[str, float] | None:
    """Try the rules in order. Return (category, confidence) or None."""
    for keyword, category in KEYWORD_RULES.items():
        if _has_word(keyword, merchant):
            return category, RULE_CONFIDENCE

    payee, is_payee = _extract_name(txn["description"])
    if is_payee and _looks_like_person(payee):
        if payee.lower() in KNOWN_PEOPLE:
            return "Transfers", RULE_CONFIDENCE
        # Looks like a person, but could be a shop we don't know -> unsure.
        return "Transfers", PERSON_GUESS_CONFIDENCE

    # Money received from a company (no other rule matched) is most likely income.
    if txn.get("type") == "credit":
        return "Income", CREDIT_CONFIDENCE
    return None


# ---------------------------------------------------------------------------
# Step 3: LLM fallback
# ---------------------------------------------------------------------------

def _ask_llm(merchants: list[str]) -> dict[str, str]:
    """Send all unknown merchants to llm.py in one call. Empty dict if unavailable."""
    try:
        import llm  # lives at the project root
        if not llm.has_api_key():
            return {}
        return llm.classify_batch(merchants, CATEGORIES)
    except Exception:  # llm.py missing, no network, bad reply, ...
        return {}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def categorize_transactions(transactions: list[dict], use_llm: bool = True) -> list[dict]:
    """Add "merchant", "category" and "confidence" to every transaction.

    Returns new dicts; the input list is not modified.
    """
    results = []
    to_check = []         # unknown merchants + person guesses, for the LLM
    business_income = []  # credits from a business that matched no other rule

    for txn in transactions:
        out = dict(txn)
        out["merchant"] = clean_merchant(txn["description"])
        decided = _rule_category(txn, out["merchant"])
        if decided:
            out["category"], out["confidence"] = decided
        else:
            out["category"], out["confidence"] = "Other", UNKNOWN_CONFIDENCE
        if out["confidence"] in (UNKNOWN_CONFIDENCE, PERSON_GUESS_CONFIDENCE):
            to_check.append(out)
        if decided == ("Income", CREDIT_CONFIDENCE):
            business_income.append(out)
        results.append(out)

    _rename_monthly_salary(business_income)

    # One batched LLM call for all unsure merchants (each name sent once).
    # Without an API key nothing changes: unknowns stay "Other", person
    # guesses stay "Transfers" at 0.6.
    if use_llm and to_check:
        names = sorted({t["merchant"] for t in to_check})
        answers = _ask_llm(names)
        for t in to_check:
            category = answers.get(t["merchant"])
            # Only accept real categories; "Other" from the LLM changes nothing.
            if category in CATEGORIES and category != "Other":
                t["category"], t["confidence"] = category, LLM_CONFIDENCE

    return results


def _rename_monthly_salary(credits: list[dict]) -> None:
    """Rename a business credit to "Salary" if the same amount arrives in 2+ months.

    e.g. "Received from ACME TECH SOLUTIONS" Rs 45,000 on Jul 1, Aug 1, Sep 1.
    One-off credits keep the business name.
    """
    months = {}  # (merchant, amount) -> set of "YYYY-MM"
    for t in credits:
        months.setdefault((t["merchant"], t["amount"]), set()).add(t["date"][:7])
    for t in credits:
        if len(months[(t["merchant"], t["amount"])]) >= 2:
            t["merchant"] = "Salary"
