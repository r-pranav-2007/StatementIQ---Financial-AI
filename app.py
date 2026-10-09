"""
app.py - StatementIQ web app (Streamlit).

Run:  streamlit run app.py

All numbers come from the tools in tools/ (plain Python). The chat box uses StatementAgent;
with "Use AI (Claude)" OFF (the default) nothing is ever sent to an API.
"""
import copy
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import llm
from agent import StatementAgent
from tools.anomalies import _inr as inr  # "Rs 1,25,000"
from tools.anomalies import find_anomalies
from tools.categorize import CATEGORIES
from tools.recurring import detect_recurring, summarize_subscriptions

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_statement.csv"
REVIEW_BELOW = 0.85
SEVERITY_COLOR = {"high": "red", "medium": "orange", "low": "blue"}

st.set_page_config(page_title="StatementIQ", page_icon=":bar_chart:", layout="wide")


# ----------------------------------------------------------------------
# Session state and loading
# ----------------------------------------------------------------------

def _init_state() -> None:
    defaults = {"agent": None, "source": None, "recurring": [], "anomalies": [],
                "chat": [], "load_error": None, "tmp_dir": None, "original": []}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _refresh_analysis() -> None:
    """Recompute subscriptions and alerts from the (possibly corrected) categories."""
    agent = st.session_state.agent
    agent.recurring = detect_recurring(agent.categorized)
    agent.anomalies = find_anomalies(agent.categorized, agent.recurring)
    st.session_state.recurring = agent.recurring
    st.session_state.anomalies = agent.anomalies


def _load(path: str, source: str, use_ai: bool) -> None:
    """Parse and categorize a statement into a fresh agent. Never raises."""
    agent = StatementAgent(use_llm=use_ai)
    parsed = agent._run_tool("parse_statement", {"file_path": path})
    if not parsed["ok"]:
        st.session_state.load_error = parsed["error"]
        return
    categorized = agent._run_tool("categorize_transactions", {})
    if not categorized["ok"]:
        st.session_state.load_error = categorized["error"]
        return
    st.session_state.update(agent=agent, source=source, chat=[], load_error=None,
                            original=copy.deepcopy(agent.categorized))
    _refresh_analysis()


def _save_upload(upload) -> str:
    if st.session_state.tmp_dir is None:
        st.session_state.tmp_dir = tempfile.mkdtemp(prefix="statementiq_")
    suffix = Path(upload.name).suffix.lower()
    path = Path(st.session_state.tmp_dir) / f"upload{suffix}"
    path.write_bytes(upload.getbuffer())
    return str(path)


def _set_category(idx: int, widget_key: str) -> None:
    """Dropdown callback: fix one transaction's category."""
    agent = st.session_state.agent
    agent.categorized[idx]["category"] = st.session_state[widget_key]
    agent.categorized[idx]["confidence"] = 1.0  # confirmed by the user
    _refresh_analysis()


def _confirm(idx: int) -> None:
    """"Looks right" callback: keep the category and mark it confirmed."""
    st.session_state.agent.categorized[idx]["confidence"] = 1.0
    _refresh_analysis()


def _reset_corrections() -> None:
    """Restore the original categorization and forget the dropdown choices."""
    st.session_state.agent.categorized = copy.deepcopy(st.session_state.original)
    for key in [k for k in st.session_state if str(k).startswith("fix_")]:
        del st.session_state[key]
    _refresh_analysis()


def _frame() -> pd.DataFrame:
    df = pd.DataFrame(st.session_state.agent.categorized)
    df["idx"] = range(len(df))
    return df


# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------

def _sidebar() -> bool:
    with st.sidebar:
        st.header("Settings")
        has_key = llm.has_api_key()
        use_ai = st.toggle("Use AI (Claude)", value=False, disabled=not has_key, key="use_ai",
                           help="Lets Claude help with categories and chat answers.")
        if not has_key:
            st.caption("No API key found, so the app runs fully offline.")
        usage = llm.usage_summary()
        st.subheader("API usage")
        st.metric("Calls", usage["calls"])
        st.caption(f"Tokens: {usage['input_tokens']:,} in / {usage['output_tokens']:,} out")
        st.caption(f"Estimated cost: ${usage['estimated_cost_usd']:.4f}")
    return bool(use_ai and has_key)


def _upload_area(use_ai: bool) -> None:
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    upload = left.file_uploader("Statement file", type=["csv", "xlsx", "pdf"])
    use_sample = right.button("Use sample data", width="stretch")
    st.caption("Processed in this session only. Demo uses fake data.")

    if use_sample:
        with st.spinner("Reading sample statement..."):
            _load(str(SAMPLE_PATH), "sample", use_ai)
    elif upload is not None:
        source = f"{upload.name}:{upload.size}"
        if source != st.session_state.source and source != st.session_state.get("failed_source"):
            with st.spinner("Reading your statement..."):
                _load(_save_upload(upload), source, use_ai)
            st.session_state.failed_source = source if st.session_state.load_error else None

    if st.session_state.load_error:
        st.error(f"Sorry, I couldn't read that file. {st.session_state.load_error}\n\n"
                 "Try uploading the CSV or Excel version of your statement.")


def _overview(df: pd.DataFrame) -> None:
    debits = df[df["type"] == "debit"]
    credits = df[df["type"] == "credit"]
    by_cat = debits.groupby("category")["amount"].sum().sort_values(ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total spent", inr(round(debits["amount"].sum(), 2)))
    c2.metric("Total received", inr(round(credits["amount"].sum(), 2)))
    c3.metric("Transactions", f"{len(df):,}")
    c4.metric("Top category", by_cat.index[0] if len(by_cat) else "-",
              inr(round(by_cat.iloc[0], 2)) if len(by_cat) else None, delta_color="off")

    if debits.empty:
        st.info("No spending found in this statement.")
        return
    pie_col, bar_col = st.columns(2)
    pie = px.pie(by_cat.reset_index(), names="category", values="amount", hole=0.4,
                 title="Spending by category")
    pie.update_traces(hovertemplate="%{label}: Rs %{value:,.0f}<extra></extra>")
    pie_col.plotly_chart(pie, width="stretch")

    months = pd.to_datetime(debits["date"]).dt.to_period("M")
    monthly = debits.groupby(months)["amount"].sum().sort_index().reset_index()
    monthly["month"] = monthly["date"].dt.strftime("%b %Y")  # plain strings, e.g. "Jul 2026"
    bar = px.bar(monthly, x="month", y="amount", title="Spending per month",
                 labels={"month": "Month", "amount": "Spent (Rs)"})
    bar.update_xaxes(type="category", categoryorder="array", categoryarray=list(monthly["month"]))
    bar.update_traces(hovertemplate="%{x}: Rs %{y:,.0f}<extra></extra>")
    bar_col.plotly_chart(bar, width="stretch")


def _tab_subscriptions() -> None:
    recurring = st.session_state.recurring
    if not recurring:
        st.info("No recurring payments found.")
        return
    subs = summarize_subscriptions(recurring)
    st.markdown(f"**You pay {inr(subs['monthly_total'])} per month across "
                f"{subs['count']} subscriptions.**")
    st.dataframe(pd.DataFrame([{
        "Merchant": r["merchant"], "Kind": r["kind"], "Amount": inr(r["amount"]),
        "Frequency": r["frequency"], "Charges": r["count"], "Last charged": r["last_charged"],
        "Next expected": r["next_expected"], "Per month": inr(r["monthly_cost"]),
    } for r in recurring]), hide_index=True, width="stretch")


def _tab_alerts() -> None:
    anomalies = st.session_state.anomalies
    if not anomalies:
        st.success("Nothing unusual found.")
        return
    for a in anomalies:
        badge, text = st.columns([1, 7])
        badge.badge(a["severity"].upper(), color=SEVERITY_COLOR.get(a["severity"], "gray"))
        t = a["transaction"]
        text.markdown(f"{a['reason']}  \n:gray[{t['date']} · {t['merchant']} · {inr(t['amount'])}]")


def _tab_review(df: pd.DataFrame) -> None:
    changed = st.session_state.agent.categorized != st.session_state.original
    st.button("Reset corrections", on_click=_reset_corrections, disabled=not changed)
    low = df[df["confidence"] < REVIEW_BELOW]
    if low.empty:
        st.success("Every transaction was categorized with high confidence.")
        return
    st.caption(f"{len(low)} transactions the app is unsure about. Fix the category, "
               "or click Looks right to keep it.")
    for _, row in low.iterrows():
        info, pick, ok = st.columns([3, 2, 1], vertical_alignment="center")
        info.markdown(f"**{row['merchant']}** · {inr(row['amount'])} · {row['date']}  \n"
                      f":gray[{row['description'][:70]} · confidence {row['confidence']:.0%}]")
        key = f"fix_{row['idx']}"
        pick.selectbox("Category", CATEGORIES, index=CATEGORIES.index(row["category"]), key=key,
                       label_visibility="collapsed", on_change=_set_category,
                       args=(int(row["idx"]), key))
        ok.button("Looks right", key=f"ok_{row['idx']}", on_click=_confirm, args=(int(row["idx"]),))


def _tab_all(df: pd.DataFrame) -> None:
    search_col, cat_col = st.columns([2, 1])
    query = search_col.text_input("Search", placeholder="Search description or merchant")
    cats = cat_col.multiselect("Category", sorted(df["category"].unique()))
    view = df
    if query:
        mask = (view["description"].str.contains(query, case=False, regex=False)
                | view["merchant"].str.contains(query, case=False, regex=False))
        view = view[mask]
    if cats:
        view = view[view["category"].isin(cats)]
    table = view[["date", "merchant", "description", "amount", "type", "category", "confidence"]]
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Download CSV", table.to_csv(index=False).encode("utf-8"),
                       "transactions.csv", "text/csv")


def _render_tool_log(entry: dict) -> None:
    with st.expander("Show tools used"):
        if entry["tool_log"]:
            st.table(pd.DataFrame([{
                "Tool": t["tool"], "Arguments": str(t["args"]) if t["args"] else "-",
                "Summary": t["summary"], "Seconds": t["seconds"]} for t in entry["tool_log"]]))
        else:
            st.write("No tools were needed for this answer.")
        guard = entry.get("guard") or ""
        st.write(f"Number guard intervened: **yes** ({guard})" if guard
                 else "Number guard intervened: **no**")
        st.caption(f"Mode: {entry['mode']}")


def _chat() -> None:
    st.subheader("Ask about your statement")
    for entry in st.session_state.chat:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry["role"] == "assistant":
                _render_tool_log(entry)

    prompt = st.chat_input("e.g. How much did I spend on food in August?")
    if not prompt:
        return
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.agent.chat(prompt)
        st.markdown(result["answer"])
        entry = {"role": "assistant", "content": result["answer"], "mode": result["mode"],
                 "tool_log": result["tool_log"], "guard": result.get("guard", "")}
        _render_tool_log(entry)
    st.session_state.chat.append(entry)


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------

_init_state()
st.title("StatementIQ")
st.write("Upload a bank or UPI statement, get instant insights")
use_ai = _sidebar()
_upload_area(use_ai)

if st.session_state.agent is None:
    st.info("Upload a statement or click **Use sample data** to get started.")
else:
    st.session_state.agent.use_llm = use_ai
    df = _frame()
    _overview(df)
    tabs = st.tabs(["Recurring payments", "Alerts", "Needs review", "All transactions"])
    with tabs[0]:
        _tab_subscriptions()
    with tabs[1]:
        _tab_alerts()
    with tabs[2]:
        _tab_review(df)
    with tabs[3]:
        _tab_all(df)
    st.divider()
    _chat()
