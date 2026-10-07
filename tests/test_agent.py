"""Checks that StatementAgent picks the right tool for 6 requests on sample_statement.csv.

Usage:  python tests/test_agent.py          (offline keyword routing)
        python tests/test_agent.py --llm    (model picks tools; needs ANTHROPIC_API_KEY)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Print as UTF-8 so a stray symbol (e.g. the rupee sign) cannot crash the Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:  # stdout replaced by an object without reconfigure()
    pass

sys.path.insert(0, str(ROOT))
import llm  # noqa: E402
from agent import StatementAgent  # noqa: E402

USE_LLM = "--llm" in sys.argv
if USE_LLM and not llm.has_api_key():
    print("WARNING: --llm given but ANTHROPIC_API_KEY is not set; running offline.")

FILE = str(ROOT / "data" / "sample_statement.csv")

# (message, attach the file?, expected FIRST tool)
REQUESTS = [
    ("Here's my bank statement, please upload this file.", True, "parse_statement"),
    ("What subscriptions am I paying for?", False, "detect_recurring"),
    ("Is there anything suspicious in my account?", False, "find_anomalies"),
    ("How much did I spend on food in August?", False, "answer_query"),
    ("Compare August and July for me.", False, "answer_query"),
    ("What can I cut to save some money?", False, "answer_query"),
]

agent = StatementAgent(use_llm=USE_LLM)
correct = 0
print(f"Mode requested: {'LLM' if USE_LLM else 'offline'}\n")

for i, (message, attach, expected) in enumerate(REQUESTS, 1):
    result = agent.chat(message, file_path=FILE if attach else None)
    tools_used = [entry["tool"] for entry in result["tool_log"]]
    first = tools_used[0] if tools_used else "(none)"
    ok = first == expected
    # Request 1 must also categorize right after parsing.
    if i == 1:
        ok = ok and tools_used[:2] == ["parse_statement", "categorize_transactions"]
    correct += ok

    print("=" * 90)
    print(f"{i}. [{'OK   ' if ok else 'WRONG'}] {message}")
    print(f"   expected first tool: {expected}   mode: {result['mode']}"
          + (f"   guard: {result['guard']}" if result["guard"] else ""))
    for entry in result["tool_log"]:
        args = f" {entry['args']}" if entry["args"] else ""
        args = re.sub(r"'[A-Za-z]:\\\\.*?([^\\\\]+)'", r"'...\\\1'", args)  # shorten file paths
        print(f"   - {entry['tool']}{args}  ({entry['seconds']}s)  -> {entry['summary']}")
    print(f"   ANSWER: {result['answer']}")

print("=" * 90)
print(f"Tool selection accuracy: {correct}/{len(REQUESTS)}")
if USE_LLM:
    u = llm.usage_summary()
    print(f"\nAPI usage: {u['calls']} calls, {u['input_tokens']} input + "
          f"{u['output_tokens']} output tokens, about ${u['estimated_cost_usd']}")
sys.exit(0 if correct == len(REQUESTS) else 1)
