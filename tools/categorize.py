"""
categorize.py - give every transaction a clean merchant name and a category.

Three steps:
  1. Merchant cleaning : "UPI/SWIGGY/8137.../okaxis" -> "Swiggy"
  2. Rules             : known brand (KNOWN_MERCHANTS) or shop word (KEYWORD_RULES)
                         -> category, confidence 0.95 (0.8 for multi-business brands)
  3. LLM fallback      : everything below 0.85 confidence goes to
                         llm.classify_batch() in ONE batch. With no API key,
                         unknowns stay "Other" (0.3), person guesses "Transfers" (0.6)

Matching rule: keywords are tried LONGEST FIRST, so a specific keyword such as
"prime video" always wins over a generic one such as "amazon".
"""
import re

CATEGORIES = [
    "Food Delivery", "Groceries", "Shopping", "Transport", "Bills and Utilities",
    "Subscriptions", "Rent", "Education", "Health", "Entertainment",
    "Investments", "Transfers", "Income", "Other",
]

# Confidence scores for each way a category can be decided.
RULE_CONFIDENCE = 0.95          # brand / shop-word rule, or a payee in KNOWN_PEOPLE
AMBIGUOUS_CONFIDENCE = 0.80     # brand that runs several businesses (e.g. Amazon)
PERSON_GUESS_CONFIDENCE = 0.60  # name looks like a person but isn't a known contact
CREDIT_CONFIDENCE = 0.70        # money received from a business, assumed Income
LLM_CONFIDENCE = 0.70           # category suggested by the LLM
UNKNOWN_CONFIDENCE = 0.30       # nobody knew, fell back to "Other"
LLM_CHECK_BELOW = 0.85          # anything under this is sent to the LLM (if a key exists)

# ---------------------------------------------------------------------------
# Rule tables
# ---------------------------------------------------------------------------

# Known brands: keyword -> (clean merchant name, category).
# Order does not matter: longer keywords are always tried first.
KNOWN_MERCHANTS = {
    # Food delivery, restaurants and food chains
    "swiggy": ("Swiggy", "Food Delivery"), "zomato": ("Zomato", "Food Delivery"),
    "eatsure": ("EatSure", "Food Delivery"), "faasos": ("Faasos", "Food Delivery"),
    "box8": ("Box8", "Food Delivery"), "behrouz": ("Behrouz Biryani", "Food Delivery"),
    "kfc": ("KFC", "Food Delivery"), "mcdonalds": ("McDonald's", "Food Delivery"),
    "burger king": ("Burger King", "Food Delivery"), "subway": ("Subway", "Food Delivery"),
    "pizza hut": ("Pizza Hut", "Food Delivery"), "starbucks": ("Starbucks", "Food Delivery"),
    "chaayos": ("Chaayos", "Food Delivery"), "haldiram": ("Haldiram's", "Food Delivery"),
    "haldirams": ("Haldiram's", "Food Delivery"), "barbeque nation": ("Barbeque Nation", "Food Delivery"),
    "wow momo": ("Wow! Momo", "Food Delivery"), "taco bell": ("Taco Bell", "Food Delivery"),
    "dunkin": ("Dunkin'", "Food Delivery"), "third wave coffee": ("Third Wave Coffee", "Food Delivery"),
    # Groceries and quick commerce
    "bigbasket": ("BigBasket", "Groceries"), "bb daily": ("BigBasket", "Groceries"),
    "swiggy instamart": ("Swiggy Instamart", "Groceries"), "instamart": ("Swiggy Instamart", "Groceries"),
    "zepto": ("Zepto", "Groceries"), "blinkit": ("Blinkit", "Groceries"),
    "dmart": ("DMart", "Groceries"), "jiomart": ("JioMart", "Groceries"),
    "amazon fresh": ("Amazon Fresh", "Groceries"), "spencers": ("Spencer's", "Groceries"),
    "star bazaar": ("Star Bazaar", "Groceries"), "ratnadeep": ("Ratnadeep", "Groceries"),
    "natures basket": ("Nature's Basket", "Groceries"), "licious": ("Licious", "Groceries"),
    "freshtohome": ("FreshToHome", "Groceries"), "country delight": ("Country Delight", "Groceries"),
    "milkbasket": ("Milkbasket", "Groceries"), "dunzo": ("Dunzo", "Groceries"),
    # Shopping: online, fashion, electronics, jewellery, home
    "amazon": ("Amazon", "Shopping"), "flipkart": ("Flipkart", "Shopping"),
    "myntra": ("Myntra", "Shopping"), "ajio": ("AJIO", "Shopping"), "meesho": ("Meesho", "Shopping"),
    "nykaa": ("Nykaa", "Shopping"), "tata cliq": ("Tata CLiQ", "Shopping"),
    "snapdeal": ("Snapdeal", "Shopping"), "firstcry": ("FirstCry", "Shopping"),
    "pepperfry": ("Pepperfry", "Shopping"), "ikea": ("IKEA", "Shopping"),
    "urban ladder": ("Urban Ladder", "Shopping"), "vijay sales": ("Vijay Sales", "Shopping"),
    "poorvika": ("Poorvika", "Shopping"), "sangeetha mobiles": ("Sangeetha Mobiles", "Shopping"),
    "tanishq": ("Tanishq", "Shopping"), "caratlane": ("CaratLane", "Shopping"),
    "westside": ("Westside", "Shopping"), "zudio": ("Zudio", "Shopping"),
    "pantaloons": ("Pantaloons", "Shopping"), "max fashion": ("Max Fashion", "Shopping"),
    "shoppers stop": ("Shoppers Stop", "Shopping"), "h&m": ("H&M", "Shopping"),
    "uniqlo": ("Uniqlo", "Shopping"), "puma": ("Puma", "Shopping"), "nike": ("Nike", "Shopping"),
    "adidas": ("Adidas", "Shopping"), "samsung": ("Samsung", "Shopping"),
    "xiaomi": ("Xiaomi", "Shopping"), "apple": ("Apple", "Shopping"),
    # Transport and travel
    "uber": ("Uber", "Transport"), "ola": ("Ola", "Transport"), "rapido": ("Rapido", "Transport"),
    "namma yatri": ("Namma Yatri", "Transport"), "blusmart": ("BluSmart", "Transport"),
    "irctc": ("IRCTC", "Transport"), "redbus": ("redBus", "Transport"),
    "makemytrip": ("MakeMyTrip", "Transport"), "goibibo": ("Goibibo", "Transport"),
    "yatra": ("Yatra", "Transport"), "ixigo": ("ixigo", "Transport"),
    "easemytrip": ("EaseMyTrip", "Transport"), "indigo": ("IndiGo", "Transport"),
    "air india": ("Air India", "Transport"), "spicejet": ("SpiceJet", "Transport"),
    "akasa air": ("Akasa Air", "Transport"), "bpcl": ("Bharat Petroleum", "Transport"),
    "indian oil": ("Indian Oil", "Transport"), "indianoil": ("Indian Oil", "Transport"),
    "iocl": ("Indian Oil", "Transport"), "hpcl": ("HPCL", "Transport"),
    "hindustan petroleum": ("HPCL", "Transport"), "essar oil": ("Nayara Energy", "Transport"),
    "bharat petroleum": ("Bharat Petroleum", "Transport"), "shell": ("Shell", "Transport"),
    "nayara": ("Nayara Energy", "Transport"),
    # Bills: telecom, internet, DTH, electricity, gas, insurance
    "airtel": ("Airtel", "Bills and Utilities"), "jio": ("Jio", "Bills and Utilities"),
    "vodafone": ("Vi", "Bills and Utilities"), "bsnl": ("BSNL", "Bills and Utilities"),
    "act fibernet": ("ACT Fibernet", "Bills and Utilities"), "hathway": ("Hathway", "Bills and Utilities"),
    "tata play": ("Tata Play", "Bills and Utilities"), "dish tv": ("Dish TV", "Bills and Utilities"),
    "sun direct": ("Sun Direct", "Bills and Utilities"), "bescom": ("BESCOM", "Bills and Utilities"),
    "msedcl": ("MSEDCL", "Bills and Utilities"), "tata power": ("Tata Power", "Bills and Utilities"),
    "adani electricity": ("Adani Electricity", "Bills and Utilities"), "bses": ("BSES", "Bills and Utilities"),
    "torrent power": ("Torrent Power", "Bills and Utilities"),
    "mahanagar gas": ("Mahanagar Gas", "Bills and Utilities"), "indane": ("Indane Gas", "Bills and Utilities"),
    "bharat gas": ("Bharat Gas", "Bills and Utilities"), "lic": ("LIC", "Bills and Utilities"),
    "hdfc life": ("HDFC Life", "Bills and Utilities"), "icici lombard": ("ICICI Lombard", "Bills and Utilities"),
    "policybazaar": ("Policybazaar", "Bills and Utilities"), "star health": ("Star Health", "Bills and Utilities"),
    "acko": ("ACKO", "Bills and Utilities"), "cred": ("CRED", "Bills and Utilities"),
    # Subscriptions: streaming, cloud, apps, fitness
    "netflix": ("Netflix", "Subscriptions"), "spotify": ("Spotify", "Subscriptions"),
    "hotstar": ("Hotstar", "Subscriptions"), "disney hotstar": ("Hotstar", "Subscriptions"),
    "jiocinema": ("JioCinema", "Subscriptions"), "sonyliv": ("SonyLIV", "Subscriptions"),
    "zee5": ("ZEE5", "Subscriptions"), "prime video": ("Prime Video", "Subscriptions"),
    "amazon prime": ("Amazon Prime", "Subscriptions"),
    "youtube premium": ("YouTube Premium", "Subscriptions"), "youtube": ("YouTube Premium", "Subscriptions"),
    "google play": ("Google Play", "Subscriptions"), "google one": ("Google One", "Subscriptions"),
    "google": ("Google", "Subscriptions"),
    "apple.com/bill": ("iCloud", "Subscriptions"), "icloud": ("iCloud", "Subscriptions"),
    "apple music": ("Apple Music", "Subscriptions"), "audible": ("Audible", "Subscriptions"),
    "linkedin": ("LinkedIn Premium", "Subscriptions"), "openai": ("ChatGPT", "Subscriptions"),
    "chatgpt": ("ChatGPT", "Subscriptions"), "gaana": ("Gaana", "Subscriptions"),
    "cult fit": ("Cult.fit", "Subscriptions"), "cultfit": ("Cult.fit", "Subscriptions"),
    "gym": ("Gym", "Subscriptions"),
    # Education
    "coursera": ("Coursera", "Education"), "udemy": ("Udemy", "Education"), "byjus": ("BYJU'S", "Education"),
    "unacademy": ("Unacademy", "Education"), "vedantu": ("Vedantu", "Education"),
    "physics wallah": ("Physics Wallah", "Education"), "upgrad": ("upGrad", "Education"),
    "simplilearn": ("Simplilearn", "Education"), "great learning": ("Great Learning", "Education"),
    "skillshare": ("Skillshare", "Education"), "duolingo": ("Duolingo", "Education"),
    # Health
    "pharmeasy": ("PharmEasy", "Health"), "netmeds": ("Netmeds", "Health"),
    "1mg": ("Tata 1mg", "Health"), "medplus": ("MedPlus", "Health"),
    "wellness forever": ("Wellness Forever", "Health"), "fortis": ("Fortis", "Health"),
    "manipal hospital": ("Manipal Hospitals", "Health"), "narayana health": ("Narayana Health", "Health"),
    "thyrocare": ("Thyrocare", "Health"), "lal pathlabs": ("Dr Lal PathLabs", "Health"),
    "healthifyme": ("HealthifyMe", "Health"),
    # Entertainment
    "bookmyshow": ("BookMyShow", "Entertainment"), "inox": ("INOX", "Entertainment"),
    "cinepolis": ("Cinepolis", "Entertainment"), "steam": ("Steam", "Entertainment"),
    "playstation": ("PlayStation", "Entertainment"), "dream11": ("Dream11", "Entertainment"),
    "wonderla": ("Wonderla", "Entertainment"),
    # Payment apps (top-ups / transfers)
    "paytm": ("Paytm", "Transfers"), "phonepe": ("PhonePe", "Transfers"),
    "google pay": ("Google Pay", "Transfers"), "amazon pay": ("Amazon Pay", "Shopping"),
    # Investments: brokers, mutual funds, SIPs (also the ACH/NACH debits that fund them)
    "zerodha": ("Zerodha", "Investments"), "groww": ("Groww", "Investments"),
    "upstox": ("Upstox", "Investments"), "angel one": ("Angel One", "Investments"),
    "angel broking": ("Angel One", "Investments"), "kuvera": ("Kuvera", "Investments"),
    "indmoney": ("INDmoney", "Investments"), "smallcase": ("smallcase", "Investments"),
    "paytm money": ("Paytm Money", "Investments"), "iccl": ("Indian Clearing Corp", "Investments"),
    "indian clearing": ("Indian Clearing Corp", "Investments"),
    "mutual fund": ("Mutual Fund", "Investments"), "mutual funds": ("Mutual Fund", "Investments"),
    "sip": ("SIP", "Investments"), "bse": ("BSE", "Investments"), "nse": ("NSE", "Investments"),
    "nps": ("NPS", "Investments"), "ppf": ("PPF", "Investments"),
    # Rent, income, cash
    "rent": ("Rent", "Rent"), "nobroker": ("NoBroker", "Rent"),
    "salary": ("Salary", "Income"), "atm": ("ATM", "Other"),
}

# Brands that run several businesses: a match is only a good guess (0.8).
AMBIGUOUS_BRANDS = {"amazon", "google", "apple", "paytm", "phonepe", "flipkart", "amazon pay"}

# Generic shop words -> category, for merchants that are not known brands.
KEYWORD_RULES = {
    # Groceries
    "supermarket": "Groceries", "super market": "Groceries", "hypermarket": "Groceries",
    "mart": "Groceries", "stores": "Groceries", "kirana": "Groceries",
    "provisions": "Groceries", "vegetables": "Groceries", "fruits": "Groceries", "dairy": "Groceries",
    # Food
    "restaurant": "Food Delivery", "coffee": "Food Delivery", "pizza": "Food Delivery", "biryani": "Food Delivery",
    "bakery": "Food Delivery", "sweets": "Food Delivery", "cafe": "Food Delivery",
    "dhaba": "Food Delivery", "canteen": "Food Delivery", "food court": "Food Delivery",
    "tiffin": "Food Delivery",
    # Shopping
    "textiles": "Shopping", "garments": "Shopping", "fashion": "Shopping",
    "footwear": "Shopping", "electronics": "Shopping", "mobiles": "Shopping",
    "jewellers": "Shopping", "jewellery": "Shopping", "furniture": "Shopping",
    "opticals": "Shopping", "boutique": "Shopping", "silks": "Shopping",
    # Transport
    "petrol": "Transport", "diesel": "Transport", "fuel": "Transport", "fuels": "Transport",
    "filling station": "Transport", "petroleum": "Transport", "fastag": "Transport",
    "toll": "Transport", "parking": "Transport", "metro": "Transport",
    "travels": "Transport", "airlines": "Transport", "taxi": "Transport", "cabs": "Transport",
    # Bills and utilities
    "electricity": "Bills and Utilities", "broadband": "Bills and Utilities",
    "fibernet": "Bills and Utilities", "recharge": "Bills and Utilities",
    "postpaid": "Bills and Utilities", "prepaid": "Bills and Utilities",
    "dth": "Bills and Utilities", "water bill": "Bills and Utilities",
    "insurance": "Bills and Utilities", "maintenance": "Bills and Utilities",
    # Health
    "hospital": "Health", "hospitals": "Health", "clinic": "Health", "medical": "Health",
    "medicals": "Health", "pharmacy": "Health", "chemist": "Health", "chemists": "Health",
    "diagnostics": "Health", "dental": "Health", "healthcare": "Health",
    "eye care": "Health", "nursing home": "Health",
    # Education
    "college": "Education", "university": "Education", "school": "Education",
    "academy": "Education", "institute": "Education", "tuition": "Education",
    "coaching": "Education", "course": "Education", "courses": "Education", "fees": "Education",
    # Entertainment
    "cinemas": "Entertainment", "cinema": "Entertainment", "theatre": "Entertainment",
    "multiplex": "Entertainment", "gaming": "Entertainment", "amusement": "Entertainment",
    # Subscriptions
    "subscription": "Subscriptions", "membership": "Subscriptions", "fitness": "Subscriptions",
    # Rent and income
    "landlord": "Rent", "house owner": "Rent",
    "payroll": "Income", "stipend": "Income", "dividend": "Income",
}

# Words that mean "this is a business, not a person".
BUSINESS_WORDS = {
    "stores", "store", "mart", "pharmacy", "hospital", "clinic", "cafe", "restaurant",
    "hotel", "pvt", "ltd", "limited", "solutions", "services", "technologies", "tech",
    "enterprises", "traders", "india", "online", "foods", "bank", "school", "college",
    "academy", "shop", "agency", "cabs", "motors", "fuels", "petroleum",
    "textiles", "retail", "medicals", "electronics", "bakery", "sweets", "jewellers",
    "fuel", "petrol", "mobiles", "coffee",
}

# People the user has told us about (their friends/family). A payee in this
# list is a sure Transfer; any other person-like name is only a guess (0.6).
KNOWN_PEOPLE = {"arun kumar", "suresh b", "divya m", "priya s", "vikram t"}

# ---------------------------------------------------------------------------
# Step 1: merchant cleaning
# ---------------------------------------------------------------------------

# Wallet apps write "Paid to X" / "Received from X".
WALLET_PATTERN = re.compile(r"^(paid to|sent to|received from|payment to|payment from)\s+(.+)$", re.I)

# Prefixes banks put in front of the real name: "UPI/", "POS ", "PAYTM*", "GOOGLE*" ...
# "ACH D-" / "NACH C-" carry a debit/credit letter before the name.
PREFIX_PATTERN = re.compile(r"^((UPI|NEFT|IMPS|RTGS|POS|ACH|NACH)\s*[/\-]?\s*(?:[DC]\s*-\s*)?|[A-Z]+\*)", re.I)

# UPI narration with a reference first: UPI/<ref>/<note>/<vpa>/<bank>
UPI_WITH_REF = re.compile(r"^UPI/(\d{6,})/([^/]*)/([^/]*)(?:/([^/]*))?$", re.I)
# VPA handles that say nothing about who was paid (gateways, app names, bank suffixes).
GENERIC_HANDLES = {
    "upi", "pay", "payment", "payments", "paytm", "paytmqr", "qr", "merchant", "shop", "store",
    "stores", "gpay", "phonepe", "ybl", "ibl", "axl", "bharatpe", "razorpay", "payu", "billdesk",
    "cashfree", "vyapar", "okaxis", "oksbi", "okhdfc", "okicici",
}
CITIES = {
    "chennai", "mumbai", "delhi", "new delhi", "bangalore", "bengaluru", "hyderabad", "kolkata",
    "pune", "ahmedabad", "jaipur", "kochi", "coimbatore", "gurgaon", "gurugram", "noida",
    "lucknow", "indore", "chandigarh", "surat", "nagpur", "mysore", "madurai", "bhopal",
    "thiruvananthapuram", "visakhapatnam",
}
INTEREST_PATTERN = re.compile(r"\bint\.?\s?pd\b|\binterest\s+(?:paid|credit(?:ed)?)\b|\bcredit\s+interest\b", re.I)


def _parse_upi(description: str) -> dict | None:
    """Split "UPI/<ref>/<note>/<vpa>/<bank>" into the note and a name taken from the VPA handle.

    The numeric ref is never a name. The handle before "@" is split on . _ - and digits and
    title-cased; a phone-number or generic handle falls back to the note.
    """
    m = UPI_WITH_REF.match(description.strip())
    if not m:
        return None
    note, vpa = m.group(2).strip(), m.group(3).strip()
    words = [w for w in re.split(r"[._\-\d]+", vpa.split("@")[0]) if w]
    generic = not words or " ".join(words).lower() in GENERIC_HANDLES
    if generic:
        return {"note": note, "words": [], "name": note.title() if note.isupper() else note}
    return {"note": note, "words": words, "name": " ".join(words).title()}


def extract_note(description: str) -> str:
    """The free-text note a payer typed (UPI/<ref>/<note>/<vpa>/<bank>), or ""."""
    parsed = _parse_upi(description)
    return parsed["note"] if parsed else ""


def _has_word(keyword: str, text: str) -> bool:
    """True if `keyword` appears in `text` as a whole word (case-insensitive).

    A space in the keyword also matches no space, so "youtube premium" finds "YOUTUBEPREMIUM"
    and "act fibernet" finds the handle "actfibernet".
    """
    pattern = re.escape(keyword).replace(r"\ ", r"\s*")
    return re.search(r"(?<![a-z0-9])" + pattern + r"(?![a-z0-9])", text.lower()) is not None


def _longest_match(text: str, table: dict) -> str | None:
    """Return the longest keyword from `table` found in `text`, or None."""
    for keyword in sorted(table, key=len, reverse=True):
        if _has_word(keyword, text):
            return keyword
    return None


def _search_text(description: str) -> str:
    """Description without the bank prefix and the UPI handle, for keyword search.

    "UPI/ARUN KUMAR/123/paytm" must not match the brand "paytm" via its handle.
    """
    parsed = _parse_upi(description)
    if parsed:  # note + VPA handle words; never the ref number or the bank name
        return f"{parsed['note']} {' '.join(parsed['words'])}"
    text = description.strip()
    if text.upper().startswith("UPI"):
        text = re.sub(r"[/@][A-Za-z]+$", "", text)
    return PREFIX_PATTERN.sub("", text)


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

    # "UPI/<ref>/<note>/<vpa>/<bank>" -> name from the VPA handle (or the note)
    parsed = _parse_upi(text)
    if parsed:
        return parsed["name"], True

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
    words = text.split()
    while len(words) > 1 and words[-1].lower() in CITIES:  # trailing city name
        words.pop()
    return " ".join(words), False


def _find_brand(description: str) -> str | None:
    """Return the KNOWN_MERCHANTS keyword found in the description, or None."""
    return _longest_match(_search_text(description), KNOWN_MERCHANTS)


def clean_merchant(description: str) -> str:
    """Turn a messy bank/UPI description into a readable merchant name."""
    if INTEREST_PATTERN.search(description):
        return "Bank Interest"
    brand = _find_brand(description)
    if brand:
        return KNOWN_MERCHANTS[brand][0]

    # Unknown merchant: tidy it up. "SRI MURUGAN STORES" -> "Sri Murugan Stores"
    name, _ = _extract_name(description)
    name = re.sub(r"\.com\b", "", name, flags=re.I)
    return name.title() if name.isupper() or name.islower() else name


# ---------------------------------------------------------------------------
# Step 2: rules
# ---------------------------------------------------------------------------

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
    # 0) Interest paid by the bank
    if txn.get("type") == "credit" and INTEREST_PATTERN.search(txn["description"]):
        return "Income", RULE_CONFIDENCE

    # a) Known brand (beats the person rule: a brand name is never a person)
    brand = _find_brand(txn["description"])
    if brand:
        confidence = AMBIGUOUS_CONFIDENCE if brand in AMBIGUOUS_BRANDS else RULE_CONFIDENCE
        return KNOWN_MERCHANTS[brand][1], confidence

    # b) Generic shop word in the merchant name or description
    word = _longest_match(merchant + " " + _search_text(txn["description"]), KEYWORD_RULES)
    if word:
        return KEYWORD_RULES[word], RULE_CONFIDENCE

    # c) Person-to-person payment
    payee, is_payee = _extract_name(txn["description"])
    if is_payee and _looks_like_person(payee):
        if payee.lower() in KNOWN_PEOPLE:
            return "Transfers", RULE_CONFIDENCE
        # Looks like a person, but could be a shop we don't know -> unsure.
        return "Transfers", PERSON_GUESS_CONFIDENCE

    # d) Money received from a company (no other rule matched) is most likely income.
    if txn.get("type") == "credit":
        return "Income", CREDIT_CONFIDENCE
    return None


# ---------------------------------------------------------------------------
# Step 3: LLM fallback
# ---------------------------------------------------------------------------

def _ask_llm(merchants: list[str], notes: dict[str, str] | None = None) -> dict[str, str]:
    """Send all unsure merchants (cleaned names + notes) to llm.py in one call. {} if unavailable."""
    try:
        import llm  # lives at the project root
        if not llm.has_api_key():
            return {}
        return llm.classify_batch(merchants, CATEGORIES, notes)
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
    to_check = []         # anything below LLM_CHECK_BELOW, for the LLM
    business_income = []  # credits from a business that matched no other rule

    for txn in transactions:
        out = dict(txn)
        out["merchant"] = clean_merchant(txn["description"])
        out["note"] = extract_note(txn["description"])
        decided = _rule_category(txn, out["merchant"])
        if decided:
            out["category"], out["confidence"] = decided
        else:
            out["category"], out["confidence"] = "Other", UNKNOWN_CONFIDENCE
        if out["confidence"] < LLM_CHECK_BELOW:
            to_check.append(out)
        if decided == ("Income", CREDIT_CONFIDENCE):
            business_income.append(out)
        results.append(out)

    _rename_monthly_salary(business_income)

    # One batched LLM call for all unsure merchants (each name sent once).
    # Without an API key nothing changes: unknowns stay "Other", guesses keep
    # their rule category and confidence.
    if use_llm and to_check:
        names = sorted({t["merchant"] for t in to_check})
        notes = {}
        for t in to_check:  # up to 3 distinct notes per merchant
            seen = notes.setdefault(t["merchant"], [])
            if t.get("note") and t["note"] not in seen and len(seen) < 3:
                seen.append(t["note"])
        answers = _ask_llm(names, {m: "; ".join(v) for m, v in notes.items()})
        for t in to_check:
            category = answers.get(t["merchant"])
            # Only accept real categories; "Other" from the LLM changes nothing. A rule that was
            # already sure (confidence 0.85 or more) is never replaced.
            if category in CATEGORIES and category != "Other" and t["confidence"] < LLM_CHECK_BELOW:
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
