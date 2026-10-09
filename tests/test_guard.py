"""Checks the number guard, the retry and the fallbacks of StatementAgent in LLM mode.

No API calls: the model is replaced by a scripted fake (llm.run_agent_turn) and llm._call is
rigged to fail if anything tries to reach the network.
"""
import sys
from pathlib import Path

# Print as UTF-8 so a stray symbol (e.g. the rupee sign) cannot crash the Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import llm  # noqa: E402
from agent import StatementAgent  # noqa: E402

NETWORK_CALLS = []


def _no_network(*args, **kwargs):
    NETWORK_CALLS.append(1)
    raise RuntimeError("network is off in tests")


llm._call = _no_network

checks = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append(ok)
    print(f"[{'OK   ' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail and not ok else ""))


def new_agent() -> StatementAgent:
    agent = StatementAgent(use_llm=False)  # tools run offline; we call _chat_llm directly
    agent.chat("load it", file_path=str(ROOT / "data" / "sample_statement.csv"))
    return agent


class FakeModel:
    """Plays a script of steps and records the messages it was sent."""

    def __init__(self, steps, question):
        self.steps, self.seen, self.question = list(steps), [], question

    def __call__(self, messages, tools, system="", temperature=0):
        self.seen.append([m for m in messages])
        kind, payload = self.steps.pop(0)
        if kind == "tool":
            return {"type": "tool_calls", "tool_calls": [{"id": "t1", "name": payload, "input": {"question": self.question}}],
                    "assistant_content": [{"type": "tool_use", "id": "t1", "name": payload, "input": {"question": self.question}}]}
        return {"type": "text", "text": payload, "assistant_content": [{"type": "text", "text": payload}]}


def run(agent, question, steps):
    fake = FakeModel(steps, question)
    llm.run_agent_turn = fake
    return agent._chat_llm(question, question), fake


agent = new_agent()
facts_number = agent._run_tool("analyze_spending", {"question": "x"})["facts"]["total_spent"]  # e.g. "Rs 1,20,949"

# 1. The guard still rejects a number no tool returned (judge level).
results = [("answer_query", agent._run_tool("answer_query", {"question": "How much did I spend in August?"}))]
real = results[0][1]["answer"]
verdict = agent._judge("You spent Rs 99,999 in August.", results, "How much did I spend in August?")
check("rejects an invented amount", verdict["status"] == "reject" and 99999.0 in verdict["bad"])
check("accepts the tool's own answer", agent._judge(real, results, "How much did I spend in August?")["status"] == "ok")
check("rejects an invented date", agent._judge(real + " That was on 19 June.", results, "q")["status"] == "reject")

# 2. Retry: bad first reply, good second reply -> retry used, second reply shown.
good = f"You spent {facts_number} in total. Review your subscriptions first."
result, fake = run(agent, "What can I cut to save money?",
                   [("tool", "analyze_spending"), ("text", "You spent Rs 99,999 overall."), ("text", good)])
retry_msg = fake.seen[-1][-1]["content"]
check("retry used once and second reply accepted",
      result["retried"] and result["answer"] == good and "rewrite accepted" in result["guard"], str(result))
check("retry message names the rejected number", isinstance(retry_msg, str) and "99,999" in retry_msg, str(retry_msg))
check("retry made exactly three model calls", len(fake.seen) == 3)

# 3. Two rejections on an analysis question -> rich Python analysis, no model text.
result, _ = run(agent, "What can I cut to save money?",
                [("tool", "analyze_spending"), ("text", "Rs 99,999."), ("text", "Rs 88,888.")])
check("rejected twice -> Python analysis", result["retried"] and "Ideas to save" in result["answer"]
      and "99,999" not in result["answer"] and "rejected twice" in result["guard"], str(result))

# 4. Two rejections on a narrow question -> plain answer_query result.
question = "How much did I spend in August?"
result, _ = run(agent, question, [("tool", "answer_query"), ("text", "Rs 99,999."), ("text", "Rs 88,888.")])
check("narrow question falls back to answer_query", result["answer"] == real and result["retried"], str(result))

# 5. A good first reply needs no retry and no intervention.
result, _ = run(agent, question, [("tool", "answer_query"), ("text", real)])
check("good reply: no retry, no intervention", not result["retried"] and result["guard"] == "", str(result))

# 6. Model skips the tools on a finance question -> its text is not shown.
result, _ = run(agent, question, [("text", "Please upload a statement first.")])
check("skipped tools -> answered by a tool", "no tool called" in result["guard"] and "upload" not in result["answer"].lower(), str(result))

check("no network calls were made", not NETWORK_CALLS)
print("\nPASS" if all(checks) else "\nFAIL")
sys.exit(0 if all(checks) else 1)
