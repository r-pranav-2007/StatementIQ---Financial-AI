"""
llm.py - the ONLY place in StatementIQ that talks to a language model.

HOW TO SWAP MODELS OR PROVIDERS
  * Different Claude model: set CLAUDE_MODEL in .env (e.g. CLAUDE_MODEL=claude-sonnet-5-5).
    No code change needed.
  * Different provider (Gemini, OpenAI, a local model...): rewrite _client() and
    _call() below for that provider's SDK and change API_KEY_ENV. The public
    functions (has_api_key, classify_batch, question_to_spec, phrase_answer,
    run_agent_turn) must keep the same names, arguments and return types -
    then no other file in the project has to change.

SAFETY RULES
  * The LLM never produces numbers for the user. All figures are computed in
    plain Python; phrase_answer() may only reword numbers it is given.
  * Every public function catches its own errors and returns a safe default,
    so the app keeps working in no-LLM mode if the key is missing, the
    network is down, or the model replies with garbage.
"""
import json
import os
import re
import sys

from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY / CLAUDE_MODEL from a .env file in the project root (if present).
load_dotenv()

API_KEY_ENV = "ANTHROPIC_API_KEY"
MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
TIMEOUT_SECONDS = 30   # per request
MAX_RETRIES = 1        # the SDK retries once on network errors, 429 and 5xx

# Output-token limits per function (small on purpose: short JSON / short answers).
CLASSIFY_MAX_TOKENS = 300          # grows by CLASSIFY_TOKENS_PER_NAME for big batches
CLASSIFY_TOKENS_PER_NAME = 16
SPEC_MAX_TOKENS = 300
PHRASE_MAX_TOKENS = 400
AGENT_MAX_TOKENS = 800

# Price used by usage_summary() (US dollars per million tokens; Claude Haiku 4.5 rates).
INPUT_COST_PER_MILLION = 1.00
OUTPUT_COST_PER_MILLION = 5.00

# Running totals for every API call made in this process.
_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

_cached_client = None


# ---------------------------------------------------------------------------
# Internal helpers (provider-specific - change these to swap provider)
# ---------------------------------------------------------------------------

def _warn(where: str, error: Exception) -> None:
    """Print a short, clear error (never the key) and carry on in no-LLM mode."""
    print(f"[llm] {where} failed ({type(error).__name__}: {error}). "
          f"Falling back to no-LLM mode.", file=sys.stderr)


def _client():
    """Create the Anthropic client once and reuse it."""
    global _cached_client
    if _cached_client is None:
        import anthropic  # imported here so the app still runs if the SDK is missing
        _cached_client = anthropic.Anthropic(
            api_key=os.environ[API_KEY_ENV],
            timeout=TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
        )
    return _cached_client


def _call(system: str, messages: list[dict], max_tokens: int = 1024, tools: list | None = None):
    """One request to the model. Raises on failure (callers catch)."""
    kwargs = {"model": MODEL, "max_tokens": max_tokens, "system": system, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    response = _client().messages.create(**kwargs)
    _record_usage(response)
    return response


def _record_usage(response) -> None:
    """Add one response's token counts to the running totals."""
    _usage["calls"] += 1
    usage = getattr(response, "usage", None)
    if usage is not None:
        _usage["input_tokens"] += usage.input_tokens or 0
        _usage["output_tokens"] += usage.output_tokens or 0


def usage_summary() -> dict:
    """Totals of all API calls so far, with an estimated cost in US dollars."""
    cost = (_usage["input_tokens"] * INPUT_COST_PER_MILLION
            + _usage["output_tokens"] * OUTPUT_COST_PER_MILLION) / 1_000_000
    return {**_usage, "estimated_cost_usd": round(cost, 4)}


def _text_of(response) -> str:
    """Join the text blocks of a response."""
    return "".join(block.text for block in response.content if block.type == "text")


def _ask_json(system: str, user: str, max_tokens: int = 1024) -> dict:
    """Ask for a JSON object. Retries once if the reply is not valid JSON."""
    last_error = None
    for _ in range(2):  # first try + one retry
        reply = _text_of(_call(system, [{"role": "user", "content": user}], max_tokens))
        # Accept a bare object or one wrapped in ```json fences.
        match = re.search(r"\{.*\}", reply, re.S)
        try:
            data = json.loads(match.group(0) if match else reply)
            if isinstance(data, dict):
                return data
            last_error = ValueError("reply was JSON but not an object")
        except json.JSONDecodeError as e:
            last_error = e
    raise ValueError(f"model did not return valid JSON: {last_error}")


def _pick(value, allowed: list[str]) -> str | None:
    """Return the allowed item matching `value` (case-insensitive), else None."""
    if not isinstance(value, str):
        return None
    lookup = {a.lower(): a for a in allowed}
    return lookup.get(value.strip().lower())


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def has_api_key() -> bool:
    """Return True only if ANTHROPIC_API_KEY is set (in .env or the environment)."""
    return bool(os.environ.get(API_KEY_ENV))


def classify_batch(merchant_names: list[str], categories: list[str]) -> dict[str, str]:
    """Map each merchant name to one of `categories`, all in ONE request.

    Anything missing, invalid or failed comes back as "Other".
    """
    result = {name: "Other" for name in merchant_names}
    if not merchant_names or not has_api_key():
        return result

    system = (
        "You classify merchants from Indian bank and UPI statements into spending categories. "
        "Reply with JSON only: one object mapping each merchant name exactly as given to "
        "exactly one category from the allowed list. No explanation, no extra keys. "
        "If a name looks like a private person, use \"Transfers\". "
        "If you are not sure, use \"Other\"."
    )
    user = (f"Allowed categories: {json.dumps(categories)}\n"
            f"Merchants: {json.dumps(merchant_names)}")
    try:
        limit = max(CLASSIFY_MAX_TOKENS, 64 + CLASSIFY_TOKENS_PER_NAME * len(merchant_names))
        data = _ask_json(system, user, max_tokens=limit)
    except Exception as e:  # network, auth, rate limit, bad JSON, ...
        _warn("classify_batch", e)
        return result

    # Validate: only known names, only allowed categories.
    names_lookup = {n.lower(): n for n in merchant_names}
    for key, value in data.items():
        name = names_lookup.get(str(key).strip().lower())
        category = _pick(value, categories)
        if name and category:
            result[name] = category
    return result


# Kinds of questions question_to_spec() can turn into a query spec.
INTENTS = ["total_spend", "total_income", "category_breakdown", "top_merchants",
           "monthly_trend", "list_transactions", "recurring", "anomalies",
           "compare_months", "average_transaction", "biggest_expense", "transaction_count",
           "income_vs_expenses", "cut_suggestions", "unknown"]


def question_to_spec(question: str, categories: list[str], merchants: list[str],
                     months: list[str]) -> dict:
    """Turn a user question into a validated query spec (no numbers are produced).

    Returns {"intent", "category", "merchant", "month", "month2", "type", "top_n"};
    any field the model got wrong becomes None, and on failure intent is "unknown".
    """
    spec = {"intent": "unknown", "category": None, "merchant": None, "month": None,
            "month2": None, "type": None, "top_n": None}
    if not has_api_key():
        return spec

    system = (
        "You convert a personal-finance question into a query spec. Reply with JSON only, "
        "exactly these keys: intent, category, merchant, month, month2, type, top_n.\n"
        f"- intent: one of {json.dumps(INTENTS)}. Use compare_months for one month vs another, "
        "income_vs_expenses for savings or income vs spending, cut_suggestions for "
        "'what can I cut / where can I save'.\n"
        "- category: one of the allowed categories, or null\n"
        "- merchant: one of the allowed merchants, or null\n"
        "- month: one of the allowed months (YYYY-MM), or null for all months. "
        "'Last month' or 'this month' means the latest allowed month.\n"
        "- month2: the second month for compare_months, else null\n"
        "- type: \"debit\" or \"credit\" only if the question explicitly limits to money out "
        "or money in, else null\n"
        "- top_n: how many items for top_merchants (integer), else null\n"
        "Never answer the question and never include numbers you were not given."
    )
    user = (f"Allowed categories: {json.dumps(categories)}\n"
            f"Allowed merchants: {json.dumps(merchants)}\n"
            f"Allowed months: {json.dumps(months)}\n"
            f"Question: {question}")
    try:
        data = _ask_json(system, user, max_tokens=SPEC_MAX_TOKENS)
    except Exception as e:
        _warn("question_to_spec", e)
        return spec

    spec["intent"] = _pick(data.get("intent"), INTENTS) or "unknown"
    spec["category"] = _pick(data.get("category"), categories)
    spec["merchant"] = _pick(data.get("merchant"), merchants)
    spec["month"] = _pick(data.get("month"), months)
    spec["month2"] = _pick(data.get("month2"), months)
    spec["type"] = _pick(data.get("type"), ["debit", "credit"])
    top_n = data.get("top_n")
    valid_n = isinstance(top_n, int) and not isinstance(top_n, bool) and 1 <= top_n <= 20
    spec["top_n"] = top_n if valid_n else None
    return spec


def phrase_answer(question: str, numbers: dict) -> str:
    """Write a short plain-English answer using ONLY the given numbers.

    Without an LLM (or on failure) returns a simple "key: value" listing.
    """
    fallback = "\n".join(f"{key}: {value}" for key, value in numbers.items())
    if not has_api_key():
        return fallback

    system = (
        "You explain personal-finance results to the user in 1-3 short sentences. "
        "Use ONLY the numbers provided in the data, copied exactly. "
        "Never add, estimate, round differently, or calculate any figure, "
        "and never mention a number that is not in the data. "
        "Amounts are in Indian Rupees: write them as \"Rs\" followed by the number with "
        "Indian comma grouping, for example Rs 1,25,000 or Rs 389. "
        "Never use the rupee symbol."
    )
    user = f"Question: {question}\nData (JSON): {json.dumps(numbers, ensure_ascii=False)}"
    for attempt in range(2):  # first try + one retry
        try:
            reply = _call(system, [{"role": "user", "content": user}], max_tokens=PHRASE_MAX_TOKENS)
            text = _text_of(reply).strip()
            if text:
                return text
        except Exception as e:
            if attempt == 1:
                _warn("phrase_answer", e)
    return fallback


def run_agent_turn(messages: list[dict], tool_declarations: list[dict],
                   system: str = "You are StatementIQ, a personal finance assistant. "
                                 "Use the tools for every number; never invent figures.") -> dict:
    """One step of a tool-using agent.

    tool_declarations use Anthropic's format: {"name", "description", "input_schema"}.
    Returns:
      {"type": "tool_calls", "tool_calls": [{"id", "name", "input"}], "assistant_content": [...]}
      {"type": "text", "text": "...", "assistant_content": [...]}
      {"type": "error", "text": "<message>"}    if the LLM is unavailable
    Append assistant_content to `messages` as the assistant turn before sending tool results.
    """
    if not has_api_key():
        return {"type": "error", "text": "No ANTHROPIC_API_KEY set - agent is unavailable in no-LLM mode."}
    for attempt in range(2):  # first try + one retry
        try:
            response = _call(system, messages, max_tokens=AGENT_MAX_TOKENS, tools=tool_declarations)
            content = [block.model_dump() for block in response.content]
            if response.stop_reason == "tool_use":
                calls = [{"id": b.id, "name": b.name, "input": b.input}
                         for b in response.content if b.type == "tool_use"]
                return {"type": "tool_calls", "tool_calls": calls, "assistant_content": content}
            return {"type": "text", "text": _text_of(response), "assistant_content": content}
        except Exception as e:
            if attempt == 1:
                _warn("run_agent_turn", e)
                return {"type": "error", "text": f"LLM call failed: {type(e).__name__}"}
    return {"type": "error", "text": "LLM call failed"}
