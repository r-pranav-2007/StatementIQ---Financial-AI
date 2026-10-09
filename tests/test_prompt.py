"""Offline check that the analysis prompt in agent.py carries the wording rules."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import SYSTEM_PROMPT  # noqa: E402

text = re.sub(r"\s+", " ", SYSTEM_PROMPT)
RULES = {
    "bans overspent/overspending/wasteful": ['"overspent"', '"overspending"', '"wasteful"'],
    "neutral gap sentence": ["higher than income over this period", "neutrally"],
    "one-offs explain the gap, amounts from facts only": ["one_off_transactions", "explain it", "from the facts only"],
    "no one-offs: just state the gap": ["If there are no one_off_transactions, just state the gap neutrally"],
    "full month_comparisons sentences": ["quote the full sentence from month_comparisons", "May vs April: up Rs X"],
    "one per-month saving figure, no range": ["single per-month figure per category", "never a range"],
    "supportive tone, under 120 words": ["supportive and practical", "under 120 words"],
    "guard-related rules kept": ["Never use numbers from earlier turns", "quote category_ranking exactly"],
}
ok = True
for name, needles in RULES.items():
    missing = [n for n in needles if n not in text]
    ok &= not missing
    print(f"[{'OK   ' if not missing else 'FAIL'}] {name}" + (f"  -> missing {missing}" if missing else ""))
print("\nPASS" if ok else "\nFAIL")
sys.exit(0 if ok else 1)
