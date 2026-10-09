"""One results table for the whole project.

Usage:  python tests/evaluation.py          (offline: no API calls, ever)
        python tests/evaluation.py --llm    (also fills the LLM column; needs ANTHROPIC_API_KEY)

It runs the existing test scripts (tests/test_*.py) as they are and reads their results,
so the numbers here are the numbers those tests produce. Saves results.md and results.json
in the project root.
"""
import contextlib
import io
import json
import sys
from datetime import date
from pathlib import Path

# Print as UTF-8 so a stray symbol (e.g. the rupee sign) cannot crash the Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import llm  # noqa: E402
from tools.categorize import categorize_transactions  # noqa: E402
from tools.parse_statement import ParseError, parse_statement  # noqa: E402
from tools.recurring import detect_recurring  # noqa: E402
from tools.anomalies import find_anomalies  # noqa: E402

WANT_LLM = "--llm" in sys.argv
USE_LLM = WANT_LLM and llm.has_api_key()
if WANT_LLM and not USE_LLM:
    print("WARNING: --llm given but no API key found; the LLM column stays empty.\n")

SAMPLE = ROOT / "data" / "sample_statement.csv"
WALLET = ROOT / "data" / "formats" / "4_wallet_upi_app.csv"


def run_test(name: str, llm_flag: bool) -> dict:
    """Execute tests/<name>.py unchanged and return its variables (output is discarded)."""
    path = ROOT / "tests" / f"{name}.py"
    ns = {"__file__": str(path), "__name__": "__evaluated__"}
    old_argv = sys.argv
    sys.argv = [str(path)] + (["--llm"] if llm_flag else [])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
            except SystemExit:  # the tests end with sys.exit(0 or 1)
                pass
    finally:
        sys.argv = old_argv
    return ns


def score(passed, total, note: str = "") -> dict:
    return {"passed": passed, "total": total, "note": note}


def show(cell: dict | None) -> str:
    if cell is None:
        return "-"
    text = f"{cell['passed']}/{cell['total']}"
    if cell["total"] and not cell["note"]:
        text += f" ({100 * cell['passed'] / cell['total']:.0f}%)"
    return f"{text}, {cell['note']}" if cell["note"] else text


def safe(fn):
    """Run one measurement; a crash becomes an error cell instead of killing the table."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return {"passed": 0, "total": 0, "note": f"error: {type(e).__name__}: {e}"}


# ----------------------------------------------------------------------
# Measurements (each reuses the test script's data and matching rules)
# ----------------------------------------------------------------------

def parsing(ns: dict) -> list[tuple[str, dict]]:
    truth, out = ns["truth"], []
    for f in ns["files"]:
        if f.suffix not in (".csv", ".xlsx", ".pdf"):
            continue

        def measure(f=f):
            try:
                got = parse_statement(str(f))
            except ParseError as e:
                return score(0, len(truth), f"failed: {e}")
            wallet = "wallet" in f.name
            correct = 0
            for g, t in zip(got, truth):
                ok = g["date"] == t["date"] and abs(g["amount"] - t["amount"]) < 0.01 and g["type"] == t["type"]
                if not wallet:
                    ok = ok and g["description"] == t["narration"]
                correct += ok
            return score(correct if len(got) == len(truth) else 0, len(truth))
        out.append((f.name, safe(measure)))
    return out


def categorization_file(ns: dict, filename: str, use_llm: bool) -> dict:
    truth = ns["truth"]
    rows = categorize_transactions(parse_statement(str(ROOT / "data" / "formats" / filename)), use_llm=use_llm)
    if len(rows) != len(truth):
        return score(0, len(truth), "row count mismatch")
    return score(sum(r["category"] == t["category"] for r, t in zip(rows, truth)), len(truth))


def case_set(name: str, llm_flag: bool) -> dict:
    ns = run_test(name, llm_flag)
    return score(ns["correct"], len(ns["CASES"]))


def recurring(ns: dict, path: Path) -> dict:
    planted, matches = ns["planted"], ns["matches"]
    txns = categorize_transactions(parse_statement(str(path)), use_llm=False)
    found = detect_recurring(txns)
    hits = sum(any(matches(f, p) for f in found) for p in planted)
    false_pos = sum(not any(matches(f, p) for p in planted) for f in found)
    return score(hits, len(planted), f"{false_pos} false positives")


def anomalies(ns: dict, path: Path) -> dict:
    planted, matches = ns["planted"], ns["matches"]
    txns = categorize_transactions(parse_statement(str(path)), use_llm=False)
    flags = find_anomalies(txns, detect_recurring(txns))
    caught = sum(any(matches(f, p) for f in flags) for p in planted)
    false_alarms = sum(not any(matches(f, p) for p in planted) for f in flags)
    return score(caught, len(planted), f"{false_alarms} false alarms")


def query(llm_flag: bool) -> dict:
    ns = run_test("test_query", llm_flag)
    return score(ns["correct"], len(ns["CASES"]), f"{ns['hallucinated']} hallucinated")


def agent(llm_flag: bool) -> dict:
    ns = run_test("test_agent", llm_flag)
    return score(ns["correct"], len(ns["REQUESTS"]))


# ----------------------------------------------------------------------
# Build the table: (section, name, offline cell, llm cell)
# ----------------------------------------------------------------------

def build() -> list[dict]:
    rows = []

    def add(section, name, offline, llm_cell=None):
        rows.append({"section": section, "test": name, "offline": offline, "llm": llm_cell})

    for name, cell in parsing(run_test("test_parse", False)):
        add("Parsing", name, cell)

    cat = run_test("test_categorize", False)
    for label, filename in (("Bank CSV", "1_bank_separate_columns.csv"), ("Wallet CSV", "4_wallet_upi_app.csv")):
        add("Categorization", label, safe(lambda f=filename: categorization_file(cat, f, False)),
            safe(lambda f=filename: categorization_file(cat, f, True)) if USE_LLM else None)
    for label, test in (("Hard test (25 unseen)", "test_categorize_hard"), ("Holdout (25 unseen)", "test_categorize_holdout")):
        add("Categorization", label, safe(lambda t=test: case_set(t, False)),
            safe(lambda t=test: case_set(t, True)) if USE_LLM else None)

    rec = run_test("test_recurring", False)
    ano = run_test("test_anomalies", False)
    for label, path in (("Bank CSV", SAMPLE), ("Wallet CSV", WALLET)):
        add("Recurring detection", f"{label}: found vs planted", safe(lambda p=path: recurring(rec, p)))
    for label, path in (("Bank CSV", SAMPLE), ("Wallet CSV", WALLET)):
        add("Anomaly detection", f"{label}: caught vs planted", safe(lambda p=path: anomalies(ano, p)))

    add("Query", "Correct answers + hallucinated numbers", safe(lambda: query(False)),
        safe(lambda: query(True)) if USE_LLM else None)
    add("Agent", "Tool selection", safe(lambda: agent(False)),
        safe(lambda: agent(True)) if USE_LLM else None)
    return rows


def markdown(rows: list[dict], usage: dict | None) -> str:
    lines = [f"# StatementIQ evaluation ({date.today().isoformat()})", "",
             "| Section | Test | Offline (rules only) | LLM |", "|---|---|---|---|"]
    for r in rows:
        llm_text = show(r["llm"]) if USE_LLM else "not run"
        lines.append(f"| {r['section']} | {r['test']} | {show(r['offline'])} | {llm_text} |")
    if usage:
        lines += ["", f"API usage: {usage['calls']} calls, {usage['input_tokens']} input + "
                      f"{usage['output_tokens']} output tokens, about ${usage['estimated_cost_usd']}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = build()
    usage = llm.usage_summary() if USE_LLM else None
    table = markdown(rows, usage)
    print(table)
    (ROOT / "results.md").write_text(table, encoding="utf-8")
    (ROOT / "results.json").write_text(json.dumps(
        {"date": date.today().isoformat(), "llm_run": USE_LLM, "results": rows, "api_usage": usage},
        indent=2), encoding="utf-8")
    print("Saved results.md and results.json")


if __name__ == "__main__":
    main()
