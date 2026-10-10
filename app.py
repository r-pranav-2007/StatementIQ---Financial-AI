"""
app.py - StatementIQ web app (Streamlit).

Run:  streamlit run app.py

All numbers come from the tools in tools/ (plain Python). The chat box uses StatementAgent;
with "AI mode" OFF (the default) nothing is ever sent to an API.
Look and feel: styles.css (design tokens), ui.py (HTML pieces), .streamlit/config.toml (theme).
"""
import copy
import re
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

import llm
import ui
from agent import StatementAgent
from tools.anomalies import _inr as inr  # "Rs 1,25,000"
from tools.anomalies import find_anomalies
from tools.categorize import CATEGORIES
from tools.recurring import detect_recurring, summarize_subscriptions

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_statement.csv"
REVIEW_BELOW = 0.85
NAV = [("overview", "Overview", "dashboard"), ("recurring", "Recurring Payments", "event_repeat"),
       ("alerts", "Alerts", "notifications"), ("review", "Needs Review", "fact_check"),
       ("chat", "AI Chat", "chat")]
SUGGESTIONS = ["What can I cut to save money?", "Where is my money going every month?",
               "Which subscriptions am I paying for?", "Is there anything unusual?"]

st.set_page_config(page_title="StatementIQ", page_icon=":material/insights:", layout="wide",
                   initial_sidebar_state="expanded")


# ----------------------------------------------------------------------
# Session state and loading
# ----------------------------------------------------------------------

def _init_state() -> None:
    defaults = {"agent": None, "source": None, "recurring": [], "anomalies": [],
                "chat": [], "load_error": None, "tmp_dir": None, "original": [], "page": "overview"}
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
    st.session_state.update(agent=agent, source=source, chat=[], load_error=None, page="overview",
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


def _go(page: str) -> None:
    st.session_state.page = page


def _new_statement() -> None:
    st.session_state.update(agent=None, source=None, recurring=[], anomalies=[], chat=[],
                            load_error=None, original=[], page="overview")


def _clear_chat() -> None:
    st.session_state.chat = []
    st.session_state.agent.messages = []  # the model forgets the conversation too


def _ask(prompt: str) -> None:
    st.session_state.pending_prompt = prompt


def _type_categories() -> dict:
    """merchant -> category for the Type chip. Display only: a narration that names a mutual fund / SIP
    (MF, SIP, AMC) counts as Investments even when the categorizer left it as Other."""
    out = {}
    for t in st.session_state.agent.categorized:
        category = t["category"]
        if category != "Investments" and re.search(r"\b(mf|mutual fund|sip|amc)\b", t["description"], re.I):
            category = "Investments"
        out[t["merchant"]] = category
    return out


def _frame() -> pd.DataFrame:
    df = pd.DataFrame(st.session_state.agent.categorized)
    df["idx"] = range(len(df))
    return df


def _html(content: str) -> None:
    st.markdown(content, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Sidebar: brand, navigation, AI mode
# ----------------------------------------------------------------------

def _sidebar(loaded: bool) -> bool:
    with st.sidebar:
        _html(ui.brand())
        for page_id, label, icon_name in NAV:
            active = loaded and st.session_state.page == page_id
            st.button(label, key=f"{'navon' if active else 'nav'}_{page_id}", icon=f":material/{icon_name}:",
                      disabled=not loaded, on_click=_go, args=(page_id,), width="stretch")
        if loaded:
            st.button("New statement", key="new_statement", icon=":material/upload_file:",
                      on_click=_new_statement, width="stretch")
        else:
            _html('<div class="siq-nav-lock">Load a statement to unlock these pages.</div>')

        has_key = llm.has_api_key()
        with st.container(key="aibox"):
            title, toggle = st.columns([3, 1], vertical_alignment="center")
            title.markdown('<div class="siq-ai-title">AI mode</div>', unsafe_allow_html=True)
            use_ai = toggle.toggle("AI mode", value=False, disabled=not has_key, key="use_ai",
                                   label_visibility="collapsed")
            usage = llm.usage_summary()
            _html('<div class="siq-helper">Get deeper insights with Claude. '
                  + ("" if has_key else "No API key found, so the app runs fully offline. ")
                  + f"API calls: {usage['calls']} (about ${usage['estimated_cost_usd']:.4f}).</div>")
            _html('<div class="siq-helper">Processed in this session only. Demo uses fake data.</div>')
    return bool(use_ai and has_key)


# ----------------------------------------------------------------------
# Landing page (no statement loaded yet)
# ----------------------------------------------------------------------

def _landing(use_ai: bool) -> None:
    _html(ui.landing_hero())
    upload_col, sample_col = st.columns([2, 1])
    with upload_col, st.container(key="card_upload"):
        _html(f'<div class="siq-drop-icon">{ui.icon("cloud_upload")}</div>')
        _html('<div class="siq-drop-title">Drop your statement here</div>')
        _html('<div class="siq-helper">PDF, CSV or Excel &middot; Up to 200 MB</div>')
        upload = st.file_uploader("Statement file", type=["csv", "xlsx", "pdf"], label_visibility="collapsed")
        if upload is not None:  # Streamlit's own file row is hidden; this chip is the single clean state
            _html(ui.file_chip(upload.name, upload.size))
    with sample_col, st.container(key="card_sample"):
        _html('<div class="siq-drop-title left">Just exploring?</div>')
        _html('<div class="siq-helper">Try with a sample statement to see the insights you can get.</div>')
        use_sample = st.button("Explore sample data", icon=":material/bar_chart:", width="stretch")
    _html(ui.privacy_note())

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
        _html(ui.error_box(f"Sorry, I couldn't read that file. {st.session_state.load_error} "
                           "Try uploading the CSV or Excel version of your statement."))
    elif st.session_state.agent is not None:
        st.rerun()  # loaded: redraw with the navigation unlocked


# ----------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------

def _hero(left_html: str, widget=None) -> None:
    """The gradient hero band at the top of a page; `widget` draws a control on its right."""
    with st.container(key="hero"):
        if widget is None:
            _html(left_html)
        else:
            left, right = st.columns([5, 1.4], vertical_alignment="center")
            left.markdown(left_html, unsafe_allow_html=True)
            with right:
                widget()


def _range(df: pd.DataFrame) -> str:
    first, last = date.fromisoformat(df["date"].min()), date.fromisoformat(df["date"].max())
    fmt = "%d %b" if first.year == last.year else "%d %b %Y"
    return f"{first.strftime(fmt)} to {last.strftime(fmt)}"


def _signed(amount: float) -> str:
    return inr(amount) if amount >= 0 else f"-{inr(abs(amount))}"


def _page_overview(df: pd.DataFrame) -> None:
    debits, credits = df[df["type"] == "debit"], df[df["type"] == "credit"]
    spent, income = round(debits["amount"].sum(), 2), round(credits["amount"].sum(), 2)
    n_months = df["date"].str[:7].nunique() or 1
    recurring = [r for r in st.session_state.recurring if r["kind"] != "income"]
    recurring_total = round(sum(r["monthly_cost"] for r in recurring), 2)

    _hero(ui.overview_band(inr(spent), f"{len(df):,} transactions - {_range(df)}",
                           [("Income", inr(income)), ("Spent", inr(spent)), ("Net change", _signed(round(income - spent, 2)))]))
    _html(ui.metric_grid([
        ui.metric_card("Total spent", inr(spent), f"{len(debits):,} payments out", "payments", "violet"),
        ui.metric_card("Total received", inr(income), f"{len(credits):,} payments in", "download", "green"),
        ui.metric_card("Transactions", f"{len(df):,}", f"{round(len(df) / n_months):,} per month", "swap_horiz", "blue"),
        ui.metric_card("Recurring per month", inr(recurring_total), f"{len(recurring)} recurring payments", "event_repeat", "amber"),
    ]))

    by_cat = debits.groupby("category")["amount"].sum().sort_values(ascending=False)
    total = by_cat.sum() or 1
    items = [{"category": c, "total": round(v, 2), "share_percent": round(v / total * 100, 1),
              "kind": "saving" if c == "Investments" else "spending"} for c, v in by_cat.items()]
    months = debits.assign(month=debits["date"].str[:7]).groupby("month")["amount"].sum().sort_index()
    nothing = ui.empty_state("savings", "No spending found in this statement.")

    donut_col, months_col = st.columns(2)  # side by side; Streamlit stacks them on narrow screens
    with donut_col, st.container(key="card_donut"):
        _html(ui.card_title("Spending by category"))
        _html(ui.donut(items, inr(spent)) if items else nothing)
    with months_col, st.container(key="card_monthly"):
        _html(ui.card_title("Monthly spending"))
        _html(ui.monthly_columns([(m, round(v, 2)) for m, v in months.items()]) if len(months) else nothing)

    left, right = st.columns([1.15, 1])
    with left, st.container(key="card_category"):
        head, toggle = st.columns([3, 2], vertical_alignment="center")
        head.markdown(ui.card_title("Top categories"), unsafe_allow_html=True)
        mode = toggle.segmented_control("View", ["Amount", "Percentage"], default="Amount", key="cat_mode",
                                        label_visibility="collapsed") or "Amount"
        _html(ui.category_bars(items, mode) if items else nothing)
    with right:
        latest = sorted(st.session_state.anomalies, key=lambda a: a["transaction"]["date"], reverse=True)[:3]
        upcoming = sorted(recurring, key=lambda r: r["next_expected"])[:3]
        with st.container(key="card_latest_alerts"):
            head, link = st.columns([3, 1], vertical_alignment="center")
            head.markdown(ui.card_title("Latest alerts"), unsafe_allow_html=True)
            link.button("View all", key="link_alerts", on_click=_go, args=("alerts",), width="stretch")
            _html(ui.latest_alerts_list(latest))
        with st.container(key="card_next_payments"):
            head, link = st.columns([3, 1], vertical_alignment="center")
            head.markdown(ui.card_title("Next recurring payments"), unsafe_allow_html=True)
            link.button("View all", key="link_recurring", on_click=_go, args=("recurring",), width="stretch")
            _html(ui.next_payments_list(upcoming, _type_categories()))

    _all_transactions(df)


def _all_transactions(df: pd.DataFrame) -> None:
    with st.container(key="card_all"):
        _html(ui.card_title("All transactions"))
        search_col, cat_col = st.columns([2, 1])
        query = search_col.text_input("Search", placeholder="Search description or merchant",
                                      label_visibility="collapsed")
        cats = cat_col.multiselect("Category", sorted(df["category"].unique()), placeholder="All categories",
                                   label_visibility="collapsed")
        view = df
        if query:
            mask = (view["description"].str.contains(query, case=False, regex=False)
                    | view["merchant"].str.contains(query, case=False, regex=False))
            view = view[mask]
        if cats:
            view = view[view["category"].isin(cats)]
        table = view[["date", "merchant", "description", "amount", "type", "category", "confidence"]]
        _html(ui.transactions_table(table.head(ui.TABLE_LIMIT).to_dict("records"), len(table)))
        st.download_button("Download CSV", table.to_csv(index=False).encode("utf-8"),
                           "transactions.csv", "text/csv")


# ----------------------------------------------------------------------
# Recurring payments
# ----------------------------------------------------------------------

def _page_recurring() -> None:
    recurring = st.session_state.recurring
    outflow = [r for r in recurring if r["kind"] != "income"]
    n_in = len(recurring) - len(outflow)
    counts = f"{len(outflow)} payment{'' if len(outflow) == 1 else 's'}" + (f" + {n_in} income" if n_in else "")
    badge, badge_label = (f"{len(outflow)} + {n_in}", "payments + income") if n_in else (f"{len(outflow)}", "payments")
    _hero(ui.band("Recurring Payments", "Merchants and income that repeat on a schedule.", badge, badge_label, "Recurring"))
    if not recurring:
        _html(f'<div class="siq-card">{ui.empty_state("event_repeat", "No recurring payments found.")}</div>')
        return
    subs = summarize_subscriptions(recurring)
    note = (f"You pay {inr(subs['monthly_total'])} per month across "
            f"{subs['count']} subscription{'' if subs['count'] == 1 else 's'}.")
    if len(outflow) != len(recurring):
        note += " Recurring income is listed too."
    _html(f'<div class="siq-card">{ui.recurring_strip(counts, round(sum(r["monthly_cost"] for r in outflow), 2), note)}</div>')
    _html(f'<div class="siq-card">{ui.recurring_table(recurring, _type_categories())}</div>')


# ----------------------------------------------------------------------
# Alerts
# ----------------------------------------------------------------------

def _page_alerts() -> None:
    anomalies = st.session_state.anomalies
    labels = sorted({ui.ALERT_TYPES.get(a["type"], (a["type"], ""))[0] for a in anomalies})
    chosen = {"value": "All types"}

    def pick() -> None:
        chosen["value"] = st.selectbox("Alert type", ["All types"] + labels, label_visibility="collapsed")

    _hero(ui.band("Alerts & Insights", "Important events and unusual activity to keep an eye on.",
                  f"{len(anomalies)}", "alerts to check", "Alerts"), pick)
    shown = [a for a in anomalies
             if chosen["value"] == "All types" or ui.ALERT_TYPES.get(a["type"], (a["type"], ""))[0] == chosen["value"]]
    if not shown:
        _html(f'<div class="siq-card">{ui.empty_state("verified", "Nothing unusual found. Your spending looks normal.", "green")}</div>')
        return
    _html(ui.alert_summary(anomalies))
    _html(f'<div class="siq-stack">{"".join(ui.alert_card(a) for a in shown)}</div>')


# ----------------------------------------------------------------------
# Needs review
# ----------------------------------------------------------------------

def _page_review(df: pd.DataFrame) -> None:
    low = df[df["confidence"] < REVIEW_BELOW]
    changed = st.session_state.agent.categorized != st.session_state.original

    def reset() -> None:
        st.button("Reset corrections", on_click=_reset_corrections, disabled=not changed, width="stretch")

    _hero(ui.band("Needs Review", "Transactions that were harder to categorise. Check the category or pick the right one.",
                  f"{len(low)}", "to review", "Review"), reset)
    with st.container(key="card_review"):
        if low.empty:
            _html(ui.empty_state("task_alt", "Every transaction was categorised with high confidence.", "green"))
            return
        _html(ui.review_count(len(low)))
        widths = [1.1, 2.8, 1.1, 1.7, 2.1, 1.1]
        for col, text in zip(st.columns(widths), ui.review_header(
                ["Date", "Merchant / description", "Amount", "Suggested category", "Change to", "Action"])):
            col.markdown(text, unsafe_allow_html=True)
        for _, row in low.iterrows():
            c_date, c_desc, c_amt, c_chip, c_pick, c_ok = st.columns(widths, vertical_alignment="center")
            c_date.markdown(ui.review_cell(ui.fmt_date(row["date"])), unsafe_allow_html=True)
            c_desc.markdown(ui.review_cell(row["merchant"], row["description"]), unsafe_allow_html=True)
            c_amt.markdown(ui.review_cell(inr(row["amount"])), unsafe_allow_html=True)
            c_chip.markdown(ui.review_chip_cell(row["category"]), unsafe_allow_html=True)
            key = f"fix_{row['idx']}"
            c_pick.selectbox("Category", CATEGORIES, index=CATEGORIES.index(row["category"]), key=key,
                             label_visibility="collapsed", on_change=_set_category, args=(int(row["idx"]), key))
            c_ok.button("Looks right", key=f"ok_{row['idx']}", on_click=_confirm, args=(int(row["idx"]),),
                        width="stretch")


# ----------------------------------------------------------------------
# AI chat
# ----------------------------------------------------------------------

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
        if entry.get("retried"):
            st.write("Retry used: **yes** (the model was asked to rewrite its reply once)")
        st.caption(f"Mode: {entry['mode']}")


def _message(role: str):
    avatar = ":material/person:" if role == "user" else ":material/auto_awesome:"
    box = st.chat_message(role, avatar=avatar)
    if role == "user":  # marker so styles.css can right-align and tint user bubbles
        box.markdown('<span class="siq-user-msg"></span>', unsafe_allow_html=True)
    return box


def _page_chat() -> None:
    def clear() -> None:
        st.button("Clear chat", key="clear_chat", icon=":material/delete:", on_click=_clear_chat, width="stretch")

    _hero(ui.band("AI Financial Assistant", "Ask anything about your spending, savings or habits.",
                  f"{len(st.session_state.agent.categorized):,}", "transactions ready", "AI chat"), clear)

    prompt = st.chat_input("Ask your financial question...") or st.session_state.pop("pending_prompt", None)
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})

    area = st.container(key="chat_area")
    with area:
        if not st.session_state.chat:  # empty state: hero card + 2x2 suggestion cards
            _html(ui.chat_hero())
            for row in (SUGGESTIONS[:2], SUGGESTIONS[2:]):
                for col, text in zip(st.columns(2), row):
                    col.button(text, key=f"sugg_{SUGGESTIONS.index(text)}", on_click=_ask, args=(text,), width="stretch")
        for entry in st.session_state.chat:
            with _message(entry["role"]):
                st.markdown(entry["content"])
                if entry["role"] == "assistant":
                    _render_tool_log(entry)
        if prompt:
            with _message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.agent.chat(prompt)
                st.markdown(result["answer"])
                entry = {"role": "assistant", "content": result["answer"], "mode": result["mode"],
                         "tool_log": result["tool_log"], "guard": result.get("guard", ""),
                         "retried": result.get("retried", False)}
                _render_tool_log(entry)
            st.session_state.chat.append(entry)

    if st.session_state.chat:  # after the first message: compact suggestion chips above the input bar
        with st.container(key="chips"):
            for col, text in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
                col.button(text, key=f"chip_{SUGGESTIONS.index(text)}", on_click=_ask, args=(text,), width="stretch")


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------

_init_state()
ui.inject_css()
loaded = st.session_state.agent is not None
use_ai = _sidebar(loaded)

if not loaded:
    _landing(use_ai)
else:
    st.session_state.agent.use_llm = use_ai
    frame = _frame()
    page = st.session_state.page
    if page == "recurring":
        _page_recurring()
    elif page == "alerts":
        _page_alerts()
    elif page == "review":
        _page_review(frame)
    elif page == "chat":
        _page_chat()
    else:
        _page_overview(frame)
