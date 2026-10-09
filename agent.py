"""
agent.py - StatementIQ's chat agent.

The model decides WHICH tool to run; the tools (plain Python) compute every number.

  * Session state (transactions, recurring list, anomalies) lives in this object and is
    never sent to the model. Tool results sent back are COMPACT summaries only.
  * LLM mode  : llm.run_agent_turn picks tools, up to MAX_STEPS model calls per message.
  * Offline   : no API key (or the model call fails) -> keyword routing to the right tool.
  * Guard     : if the model's final answer contains a number no tool returned, it is
                replaced by a plain answer built from the tool results.
"""
import json
import re
import time

import llm
from tools.anomalies import _inr as inr  # "Rs 1,25,000"
from tools.anomalies import find_anomalies
from tools.categorize import categorize_transactions
from tools.parse_statement import parse_statement
from tools.query import analyze_spending, answer_query, format_facts, unsupported_numbers
from tools.recurring import detect_recurring, summarize_subscriptions

MAX_STEPS = 6   # model calls per user message
MAX_ROWS = 10   # max list items in any tool result sent to the model
HISTORY_TURNS = 4  # previous chat turns sent to the model so follow-up questions work

SYSTEM_PROMPT = """You are StatementIQ, a personal finance assistant for Indian bank and UPI statements.
You have tools; the statement data stays inside the tools and you only see summaries.

Rules:
- When the user uploads a new file, ALWAYS call parse_statement first, then categorize_transactions.
- Subscription or recurring-payment questions: call detect_recurring.
- Suspicious, unusual, fraud or duplicate-charge questions: call find_anomalies.
- Open-ended questions that need analysis or advice (what can I cut, how to save, where is my money
  going each month, how am I doing, a spending overview): call analyze_spending with the user's question.
- Everything else, i.e. one specific figure (a total, a month, a category, a merchant, a comparison,
  an average, the biggest expense, income vs expenses): call answer_query with the user's question.
  Never use answer_query for an overview or advice question.
- Never state a number that a tool did not return. Copy amounts exactly as the tools give them.
  Do not add, subtract or estimate amounts, give ranges, or guess savings yourself; if a new
  figure is needed, call answer_query.
- Only connect facts the tools stated; do not guess why something happened.
- Write amounts as "Rs" with Indian comma grouping, e.g. Rs 1,25,000. Never use the rupee symbol.
- If parsing fails, say so and ask the user to upload a CSV or Excel file instead.
- Keep answers short: 1-4 sentences, plain text, no markdown tables.
- After analyze_spending you get a facts object. Write a helpful, specific analysis in plain words,
  under about 150 words, with 3 to 5 concrete suggestions as short lines starting with "-". Quote
  numbers exactly as the facts give them, digit for digit with decimals and the % sign (write 39.4%,
  never 39%), and never calculate a new one: do not add numbers together (not even shares), no
  differences, averages, ranges or percentages of your own, and no counts like "three ideas".
  If you want to mention a total, use one the facts already give. Investments are saving, never something to cut. If the
  facts cannot support a suggestion, leave it out.
- Call a tool for every new question, even if an earlier answer seems to cover it."""

TOOLS = [
    {"name": "parse_statement",
     "description": "Read a bank/UPI statement file (CSV, Excel or PDF) into the session. "
                    "Returns a short summary: number of transactions, date range and totals.",
     "input_schema": {"type": "object",
                      "properties": {"file_path": {"type": "string",
                                                   "description": "Path of the uploaded file."}},
                      "required": ["file_path"]}},
    {"name": "categorize_transactions",
     "description": "Give every transaction in the session a clean merchant name and a category. "
                    "Run right after parse_statement. Returns spending per category.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "detect_recurring",
     "description": "Find subscriptions, bills and income that repeat on a schedule. Returns each "
                    "recurring item (merchant, amount, frequency, kind) and the monthly subscription total.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "find_anomalies",
     "description": "Find duplicate charges, unusual spikes, new subscriptions and big transfers to "
                    "new people. Returns each flag with a plain-English reason.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "analyze_spending",
     "description": "Get precomputed facts for an open-ended analysis: spending by category with "
                    "share, monthly totals with change, top merchants, recurring payments, alerts, "
                    "income vs spending and savings potential. Use for advice and overview questions.",
     "input_schema": {"type": "object",
                      "properties": {"question": {"type": "string",
                                                  "description": "The user's question, in their words."}},
                      "required": ["question"]}},
    {"name": "answer_query",
     "description": "Answer any other question about the statement (totals by month, category or "
                    "merchant, month comparisons, top merchants, averages, biggest expense, income vs "
                    "expenses, what to cut). Returns a checked answer and its numbers.",
     "input_schema": {"type": "object",
                      "properties": {"question": {"type": "string",
                                                  "description": "The user's question, in their words."}},
                      "required": ["question"]}},
]

# Offline keyword routing
UPLOAD_WORDS = r"\b(upload|uploaded|file|statement|load|import|parse|read)\b"
RECURRING_WORDS = r"\b(subscriptions?|recurring|repeat\w*|regular payments?|standing|auto-?pay|memberships?)\b"
# A message about money (when the model answered without any tool) -> answer_query instead.
FINANCE_WORDS = (r"\b(spen[dt]|spending|expenses?|costs?|paid|pay|paying|save|savings?|cut|budget|income|"
                 r"salary|earn\w*|transactions?|merchants?|categor\w+|subscriptions?|total|average|"
                 r"biggest|largest|most|compare|month|rent|bills?|food|groceries|shopping|transport|"
                 r"invest\w*|money|rs|rupees?|jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|jul\w*|aug\w*|"
                 r"sep\w*|oct\w*|nov\w*|dec\w*)\b|\d")
ASKS_FOR_UPLOAD = r"\b(upload|attach|share|provide)\b.{0,40}\b(statement|file|csv|pdf|excel)\b"
ANOMALY_WORDS = r"\b(suspicious|unusual|fraud\w*|duplicate\w*|double|anomal\w*|strange|weird|odd|scam)\b"


def _clean_content(blocks: list[dict]) -> list[dict]:
    """Keep only the fields the API accepts back (model_dump adds extra None fields)."""
    cleaned = []
    for b in blocks:
        if b.get("type") == "text" and b.get("text"):
            cleaned.append({"type": "text", "text": b["text"]})
        elif b.get("type") == "tool_use":
            cleaned.append({"type": "tool_use", "id": b["id"], "name": b["name"], "input": b["input"]})
    return cleaned


def _cap(items: list) -> list:
    return items[:MAX_ROWS]


class StatementAgent:
    """One chat session over one uploaded statement."""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.file_path = None
        self.transactions = []   # parsed rows
        self.categorized = []    # rows with merchant / category / confidence
        self.recurring = None
        self.anomalies = None
        self.messages = []       # conversation sent to the model (summaries only)
        self.tool_log = []       # every tool run, for the UI

    # ------------------------------------------------------------------
    # Tools (full data stays here; each returns a COMPACT summary)
    # ------------------------------------------------------------------

    def _tool_parse_statement(self, file_path: str) -> dict:
        try:
            rows = parse_statement(file_path)
        except Exception as e:  # ParseError, missing file, unsupported format...
            return {"ok": False, "error": str(e) or type(e).__name__}
        if not rows:
            return {"ok": False, "error": "no transactions found in the file"}
        self.file_path, self.transactions = file_path, rows
        self.categorized, self.recurring, self.anomalies = [], None, None
        debits = [t for t in rows if t["type"] == "debit"]
        credits = [t for t in rows if t["type"] == "credit"]
        return {"ok": True, "transactions": len(rows),
                "from": min(t["date"] for t in rows), "to": max(t["date"] for t in rows),
                "debits": len(debits), "credits": len(credits),
                "total_spent": inr(round(sum(t["amount"] for t in debits), 2)),
                "total_received": inr(round(sum(t["amount"] for t in credits), 2))}

    def _tool_categorize_transactions(self) -> dict:
        if not self.transactions:
            return {"ok": False, "error": "No statement loaded yet. Ask the user to upload a file."}
        self.categorized = categorize_transactions(self.transactions, use_llm=self._llm_on())
        self.recurring, self.anomalies = None, None
        totals, counts = {}, {}
        for t in self.categorized:
            if t["type"] == "debit":
                totals[t["category"]] = totals.get(t["category"], 0) + t["amount"]
                counts[t["category"]] = counts.get(t["category"], 0) + 1
        ranked = sorted(totals.items(), key=lambda kv: -kv[1])
        return {"ok": True, "transactions": len(self.categorized),
                "spending_by_category": _cap([{"category": c, "transactions": counts[c],
                                               "total": inr(round(v, 2))} for c, v in ranked]),
                "low_confidence": sum(1 for t in self.categorized if t["confidence"] < 0.85)}

    def _ensure_categorized(self) -> dict | None:
        """Categorize automatically if a file is parsed but not categorized yet."""
        if not self.transactions:
            return {"ok": False, "error": "No statement loaded yet. Ask the user to upload a file."}
        if not self.categorized:
            self._tool_categorize_transactions()
        return None

    def _tool_detect_recurring(self) -> dict:
        error = self._ensure_categorized()
        if error:
            return error
        self.recurring = detect_recurring(self.categorized)
        subs = summarize_subscriptions(self.recurring)
        return {"ok": True, "recurring_count": len(self.recurring),
                "subscriptions": {"count": subs["count"],
                                  "monthly_total": inr(subs["monthly_total"])},
                "items": _cap([{"merchant": r["merchant"], "amount": inr(r["amount"]),
                                "frequency": r["frequency"], "kind": r["kind"],
                                "next_expected": r["next_expected"]} for r in self.recurring])}

    def _tool_find_anomalies(self) -> dict:
        error = self._ensure_categorized()
        if error:
            return error
        if self.recurring is None:
            self.recurring = detect_recurring(self.categorized)
        self.anomalies = find_anomalies(self.categorized, self.recurring)
        return {"ok": True, "count": len(self.anomalies),
                "items": _cap([{"type": a["type"], "severity": a["severity"], "reason": a["reason"]}
                               for a in self.anomalies])}

    def _tool_answer_query(self, question: str) -> dict:
        error = self._ensure_categorized()
        if error:
            return error
        result = answer_query(question, self.categorized, use_llm=self._llm_on())
        numbers = {k: (_cap(v) if isinstance(v, list) else v) for k, v in result["numbers"].items()}
        return {"ok": True, "answer": result["answer"], "numbers": numbers}

    def _tool_analyze_spending(self, question: str) -> dict:
        error = self._ensure_categorized()
        if error:
            return error
        result = analyze_spending(question, self.categorized, use_llm=False)
        return {"ok": True, "facts": format_facts(result["numbers"]), "plain_answer": result["answer"]}

    def _run_tool(self, name: str, args: dict) -> dict:
        """Run one tool, time it and add it to tool_log."""
        handlers = {
            "parse_statement": lambda: self._tool_parse_statement(str(args.get("file_path", ""))),
            "categorize_transactions": self._tool_categorize_transactions,
            "detect_recurring": self._tool_detect_recurring,
            "find_anomalies": self._tool_find_anomalies,
            "answer_query": lambda: self._tool_answer_query(str(args.get("question", ""))),
            "analyze_spending": lambda: self._tool_analyze_spending(str(args.get("question", ""))),
        }
        start = time.perf_counter()
        if name not in handlers:
            result = {"ok": False, "error": f"unknown tool {name}"}
        else:
            try:
                result = handlers[name]()
            except Exception as e:  # a tool bug must not crash the chat
                result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.tool_log.append({"tool": name, "args": args, "summary": self._short(name, result),
                              "seconds": round(time.perf_counter() - start, 3)})
        return result

    @staticmethod
    def _short(name: str, r: dict) -> str:
        """One-line summary of a tool result for the UI log."""
        if not r.get("ok"):
            return f"error: {r.get('error')}"
        if name == "parse_statement":
            return f"{r['transactions']} transactions, {r['from']} to {r['to']}"
        if name == "categorize_transactions":
            return f"{len(r['spending_by_category'])} categories, {r['low_confidence']} low-confidence"
        if name == "detect_recurring":
            return (f"{r['recurring_count']} recurring, {r['subscriptions']['count']} subscriptions "
                    f"({r['subscriptions']['monthly_total']}/month)")
        if name == "find_anomalies":
            return f"{r['count']} flags"
        if name == "analyze_spending":
            f = r["facts"]
            return f"facts: {len(f['categories'])} categories, {len(f['monthly'])} months"
        return r["answer"][:80]

    # ------------------------------------------------------------------
    # Plain answers from tool results (offline mode and guard fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _plain_answer(name: str, r: dict) -> str:
        if not r.get("ok"):
            if name == "parse_statement":
                return (f"I couldn't read that file ({r['error']}). "
                        f"Please upload a CSV or Excel file instead.")
            return f"Sorry, that didn't work: {r['error']}"
        if name == "parse_statement":
            return f"Loaded {r['transactions']} transactions from {r['from']} to {r['to']}."
        if name == "categorize_transactions":
            top = r["spending_by_category"][:3]
            listing = ", ".join(f"{c['category']} ({c['total']})" for c in top)
            return f"Sorted {r['transactions']} transactions into categories. Biggest spending: {listing}."
        if name == "detect_recurring":
            subs = [i for i in r["items"] if i["kind"] == "subscription"]
            others = [i for i in r["items"] if i["kind"] != "subscription"]
            text = (f"You pay for {r['subscriptions']['count']} subscriptions, "
                    f"{r['subscriptions']['monthly_total']} a month: "
                    + ", ".join(f"{i['merchant']} ({i['amount']})" for i in subs) + ".")
            if others:
                text += " Other recurring items: " + ", ".join(
                    f"{i['merchant']} ({i['amount']} {i['frequency']}, {i['kind']})" for i in others) + "."
            return text
        if name == "analyze_spending":
            return r["plain_answer"]
        if name == "find_anomalies":
            if not r["count"]:
                return "Nothing unusual found."
            return f"I found {r['count']} unusual transactions: " + " ".join(i["reason"] for i in r["items"])
        return r["answer"]

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def _llm_on(self) -> bool:
        return self.use_llm and llm.has_api_key()

    def chat(self, message: str, file_path: str | None = None) -> dict:
        """Handle one user message. Returns {"answer", "mode", "tool_log", "guard"}."""
        log_start = len(self.tool_log)
        text = f"{message}\n[Uploaded file: {file_path}]" if file_path else message
        result = self._chat_llm(text, message) if self._llm_on() else None
        if result is None:  # no key, or the model call failed
            result = {"answer": self._chat_offline(message, file_path), "mode": "offline", "guard": ""}
            self.messages += [{"role": "user", "content": text},
                              {"role": "assistant", "content": result["answer"]}]
        result["tool_log"] = self.tool_log[log_start:]
        return result

    def _chat_offline(self, message: str, file_path: str | None) -> str:
        """Keyword routing: same tools, no model."""
        if file_path or (re.search(UPLOAD_WORDS, message, re.I) and not self.transactions):
            if not file_path:
                return "Please upload a CSV, Excel or PDF statement first."
            parsed = self._run_tool("parse_statement", {"file_path": file_path})
            if not parsed["ok"]:
                return self._plain_answer("parse_statement", parsed)
            cats = self._run_tool("categorize_transactions", {})
            return self._plain_answer("parse_statement", parsed) + " " + \
                self._plain_answer("categorize_transactions", cats)
        if not self.transactions:
            return "Please upload a CSV, Excel or PDF statement first."
        name, args = self._route(message)
        return self._plain_answer(name, self._run_tool(name, args))

    @staticmethod
    def _route(message: str) -> tuple[str, dict]:
        """Keyword routing to a tool (offline mode, and the fallback when the model skips tools)."""
        if re.search(RECURRING_WORDS, message, re.I):
            return "detect_recurring", {}
        if re.search(ANOMALY_WORDS, message, re.I):
            return "find_anomalies", {}
        return "answer_query", {"question": message}

    def _statement_facts(self) -> dict | None:
        """What is loaded: count, date range, total debits and credits (None if nothing is)."""
        if not self.transactions:
            return None
        debits = sum(t["amount"] for t in self.transactions if t["type"] == "debit")
        credits = sum(t["amount"] for t in self.transactions if t["type"] == "credit")
        return {"transactions": len(self.transactions),
                "from": min(t["date"] for t in self.transactions),
                "to": max(t["date"] for t in self.transactions),
                "total_debits": inr(round(debits, 2)), "total_credits": inr(round(credits, 2))}

    def _system_prompt(self) -> str:
        """The base prompt plus what is loaded right now, rebuilt on every model call."""
        facts = self._statement_facts()
        if not facts:
            return SYSTEM_PROMPT + "\n\nNo statement is loaded yet; ask the user to upload one."
        return (SYSTEM_PROMPT + f"\n\nLoaded statement: {facts['transactions']} transactions from "
                f"{facts['from']} to {facts['to']}, total debits {facts['total_debits']}, "
                f"total credits {facts['total_credits']}. A statement is already loaded. "
                "Never ask the user to upload a file. For any question about spending, call a tool.")

    def _chat_llm(self, text: str, message: str) -> dict | None:
        """Model-driven loop. Returns None if the model fails, so offline mode takes over."""
        # Last few turns (user text + final answer only) so follow-ups work; tool traffic of
        # this message lives only in `convo`.
        convo = self.messages[-2 * HISTORY_TURNS:] + [{"role": "user", "content": text}]
        results = []  # (tool name, result) this message, for the number guard

        for _ in range(MAX_STEPS):
            step = llm.run_agent_turn(convo, TOOLS, system=self._system_prompt())
            if step["type"] == "error":
                return None
            convo.append({"role": "assistant", "content": _clean_content(step["assistant_content"])})
            if step["type"] == "text":
                return self._finish(text, self._guarded(step["text"].strip(), results, message))

            tool_results = []
            for call in step["tool_calls"]:
                result = self._run_tool(call["name"], call["input"] or {})
                results.append((call["name"], result))
                tool_results.append({"type": "tool_result", "tool_use_id": call["id"],
                                     "content": json.dumps(result, ensure_ascii=False)})
            convo.append({"role": "user", "content": tool_results})

        # Out of steps: answer from the last tool result and close the turn.
        answer = self._plain_answer(*results[-1]) if results else "Sorry, I couldn't finish that."
        return self._finish(text, {"answer": answer, "mode": "llm", "guard": "max steps reached"})

    def _finish(self, text: str, result: dict) -> dict:
        """Remember the exchange (question + final answer) for the next turns."""
        self.messages += [{"role": "user", "content": text},
                          {"role": "assistant", "content": result["answer"]}]
        return result

    def _guarded(self, answer: str, results: list, message: str) -> dict:
        """Replace the model's answer if it states a number no tool returned."""
        if not results and self.transactions and (
                re.search(FINANCE_WORDS, message, re.I) or re.search(ASKS_FOR_UPLOAD, answer, re.I)):
            # A statement is loaded and the model skipped the tools: do not show its text.
            name, args = self._route(message)
            fallback = self._run_tool(name, args)
            return {"answer": self._plain_answer(name, fallback), "mode": "llm",
                    "guard": f"no tool called; answered with {name}", "rejected_answer": answer}
        allowed = {"tools": [r for _, r in results], "user": message, "statement": self._statement_facts()}
        checked = re.sub(r"(?m)^\s*\d+[.)]\s", "", answer)  # list numbering "1. " is not a figure
        bad = unsupported_numbers(checked, allowed) if results else []
        if not answer or bad:
            fallback = self._plain_answer(*results[-1]) if results else "Sorry, I couldn't answer that."
            return {"answer": fallback, "mode": "llm", "guard": f"replaced unsupported numbers {bad}",
                    "rejected_answer": answer}
        if not results and unsupported_numbers(answer, {"user": message, "statement": self._statement_facts()}):
            return {"answer": "Sorry, I need to look at your statement for that. Please ask again.",
                    "mode": "llm", "guard": "replaced unsupported numbers"}
        return {"answer": answer, "mode": "llm", "guard": ""}
