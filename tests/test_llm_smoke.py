"""Smoke test: ONE tiny real classify_batch call. Needs ANTHROPIC_API_KEY in .env."""
import sys
from pathlib import Path

# Print as UTF-8 so a stray symbol (e.g. the rupee sign) cannot crash the Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:  # stdout replaced by an object without reconfigure()
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import llm  # noqa: E402
from tools.categorize import CATEGORIES  # noqa: E402

if not llm.has_api_key():
    print("SKIPPED: ANTHROPIC_API_KEY is not set (no-LLM mode).")
    sys.exit(0)

names = ["Saravana Bhavan", "Toppr", "Wakefit"]
print(f"Model: {llm.MODEL}")
result = llm.classify_batch(names, CATEGORIES)
for name in names:
    print(f"  {name:<18} -> {result[name]}")

ok = all(result[n] in CATEGORIES for n in names)
print("PASS (valid categories returned)" if ok else "FAIL")
sys.exit(0 if ok else 1)
