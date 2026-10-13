"""HTML pieces for the StatementIQ interface. Colours, sizes and fonts live in styles.css only;
everything here uses CSS classes (or CSS variables), never literal colours."""
import math
import re
import zlib
from math import pi
from datetime import date
from html import escape
from pathlib import Path

import streamlit as st

from tools.anomalies import _inr as inr  # "Rs 1,25,000"

CSS_PATH = Path(__file__).parent / "styles.css"

CATEGORY_ICONS = {
    "Food Delivery": "restaurant", "Groceries": "shopping_basket", "Shopping": "shopping_bag",
    "Transport": "directions_car", "Bills and Utilities": "receipt_long", "Subscriptions": "subscriptions",
    "Rent": "home", "Education": "school", "Health": "favorite", "Entertainment": "movie",
    "Investments": "trending_up", "Transfers": "swap_horiz", "Income": "payments", "Other": "category",
}
# Each category has its own palette colour (names match the .tone-* classes in styles.css).
CATEGORY_TONES = {
    "Food Delivery": "orange", "Groceries": "green", "Shopping": "pink", "Transport": "blue",
    "Bills and Utilities": "amber", "Subscriptions": "violet", "Rent": "indigo", "Education": "teal",
    "Health": "rose", "Entertainment": "sky", "Investments": "emerald", "Transfers": "periwinkle",
    "Income": "green", "Other": "slate",
}
AVATAR_TONES = ["violet", "indigo", "blue", "sky", "teal", "green", "amber", "orange", "pink", "rose"]
ALERT_TYPES = {  # anomaly type -> (label, icon)
    "duplicate": ("Duplicate charge", "content_copy"),
    "spike": ("Spending spike", "trending_up"),
    "new_subscription": ("New subscription", "autorenew"),
    "big_transfer": ("Large first-time transfer", "swap_horiz"),
}
KIND_TONES = {"income": ("Income", "green"), "bill": ("Bill", "amber"), "subscription": ("Subscription", "violet")}
SEVERITY_TONES = {"high": ("High", "rose"), "medium": ("Medium", "amber"), "low": ("Low", "amber")}
CADENCE = {"monthly": "Monthly", "weekly": "Weekly", "yearly": "Yearly"}


def type_chip(kind: str, category: str | None = None) -> tuple[str, str]:
    """(label, tone) for the Type chip: ONE rule used by the Recurring table and the Overview list."""
    if category == "Investments":
        return "Investment", CATEGORY_TONES["Investments"]
    if kind == "income" or category == "Income":
        return "Income", "green"
    if kind == "subscription" or category == "Subscriptions":
        return "Subscription", "violet"
    return "Bill", "amber"


ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def friendly_dates(text: str) -> str:
    """Display only: turn 2026-05-17 into 17 May 2026 inside a sentence."""
    def swap(m: re.Match) -> str:
        try:
            return fmt_date(m.group(0))
        except ValueError:
            return m.group(0)
    return ISO_DATE.sub(swap, text)


def inject_css() -> None:
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def icon(name: str) -> str:
    return f'<span class="mi">{name}</span>'


def tile(icon_name: str, tone: str, size: str = "") -> str:
    """A soft tinted square with a darker icon, in one palette colour."""
    return f'<div class="siq-tile tone-{tone} {size}">{icon(icon_name)}</div>'


def chip(text: str, tone: str = "neutral") -> str:
    return f'<span class="siq-chip tone-{"slate" if tone == "neutral" else tone}">{escape(text)}</span>'


def fmt_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day} {d.strftime('%b %Y')}"


def month_name(ym: str) -> str:
    return date(int(ym[:4]), int(ym[5:7]), 1).strftime("%b %Y")


# ---------------------------------------------------------------- headings
def card_title(title: str) -> str:
    return f'<div class="siq-card-title">{escape(title)}</div>'


def brand() -> str:
    return (f'<div class="siq-brand"><div class="siq-logo">{icon("insights")}</div>'
            '<div class="siq-brand-text"><div class="siq-brand-name">StatementIQ</div>'
            '<div class="siq-brand-sub">Agentic finance analyzer</div></div></div>')


# ---------------------------------------------------------------- overview
def metric_card(label: str, value: str, sub: str, icon_name: str, tone: str) -> str:
    return (f'<div class="siq-metric tone-{tone}"><div class="siq-metric-top"><span>{escape(label)}</span>'
            f'{tile(icon_name, tone)}</div>'
            f'<div><div class="siq-metric-value">{escape(value)}</div>'
            f'<div class="siq-metric-sub">{escape(sub)}</div></div></div>')


def metric_grid(cards: list[str]) -> str:
    return f'<div class="siq-metrics">{"".join(cards)}</div>'


def band(title: str, subtitle: str, key_value: str = "", key_label: str = "", eyebrow: str = "") -> str:
    """Shorter hero band for the inner pages: title, one-line subtitle and one key number."""
    key = (f'<div class="siq-band-key"><b class="{"compact" if "+" in key_value else ""}">{escape(key_value)}</b>'
           f'<span>{escape(key_label)}</span></div>'
           if key_value else "")
    top = f'<div class="siq-band-eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    return (f'<div class="siq-band"><div>{top}<h1 class="siq-band-title">{escape(title)}</h1>'
            f'<p class="siq-band-sub">{escape(subtitle)}</p></div>{key}</div>')


def overview_band(total: str, subtitle: str, pills: list[tuple[str, str]]) -> str:
    """Overview hero: the big total spent, a one-line summary and glass stat pills."""
    glass = "".join(f'<div class="siq-glass"><span>{escape(l)}</span><b>{escape(v)}</b></div>' for l, v in pills)
    return (f'<div class="siq-band-eyebrow">Total spent</div><h1 class="siq-band-total">{escape(total)}</h1>'
            f'<p class="siq-band-sub">{escape(subtitle)}</p><div class="siq-glass-row">{glass}</div>')


def empty_state(icon_name: str, text: str, tone: str = "violet") -> str:
    return f'<div class="siq-empty">{tile(icon_name, tone, "large")}<b>{escape(text)}</b></div>'


def _q(text: str) -> str:
    """Escape for an HTML attribute / CSS attribute selector."""
    return escape(text, quote=True)


def donut(items: list[dict], total: str) -> str:
    """SVG donut (one circle per slice, in the category colours), total in the centre, legend beside it.

    Hover is CSS only: a popup per slice, the hovered slice grows, the others fade. Legend rows carry the
    same data-cat attribute, so hovering a legend row highlights its slice too. Every <circle> also has a
    <title> with the same text as the popup (native tooltip fallback). All numbers come from `items`."""
    size, centre, radius, ring = 220, 110, 82, 28
    circumference = 2 * pi * radius
    slices, tips, legend, rules, start = [], [], [], [], 0.0
    for i in items:
        cat, tone = i["category"], CATEGORY_TONES.get(i["category"], "slate")
        text = f"{cat} - {inr(i['total'])} - {i['share_percent']}%"
        length = max(i["share_percent"] / 100 * circumference - 1.0, 0.5)  # 1px gap between slices
        slices.append(
            f'<circle class="siq-slice tone-{tone}" data-cat="{_q(cat)}" cx="{centre}" cy="{centre}" r="{radius}" '
            f'fill="none" stroke-width="{ring}" stroke-dasharray="{length:.2f} {circumference - length:.2f}" '
            f'stroke-dashoffset="{-start / 100 * circumference:.2f}"><title>{escape(text)}</title></circle>')
        tips.append(f'<div class="siq-tip donut-tip" data-cat="{_q(cat)}">{escape(text)}</div>')
        legend.append(
            f'<div class="siq-legend-item tone-{tone}" data-cat="{_q(cat)}" title="{_q(text)}"><span class="siq-legend-dot"></span>'
            f'<span class="siq-legend-name">{escape(cat)}</span><span class="siq-legend-pct">{i["share_percent"]}%</span></div>')
        sel = f'[data-cat="{_q(cat)}"]'
        rules.append(f'.siq-donut-wrap:has({sel}:hover) .siq-slice:not({sel}){{opacity:.45}}'
                     f'.siq-donut-wrap:has({sel}:hover) .siq-slice{sel}{{stroke-width:{ring + 3}}}'
                     f'.siq-donut-wrap:has({sel}:hover) .siq-tip{sel}{{opacity:1}}'
                     f'.siq-donut-wrap:has({sel}:hover) .siq-legend-item{sel}{{background:var(--tone-tint)}}')
        start += i["share_percent"]
    svg = (f'<svg class="siq-svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img" '
           f'aria-label="Spending by category"><g transform="rotate(-90 {centre} {centre})">{"".join(slices)}</g></svg>')
    return (f'<style>{"".join(rules)}</style><div class="siq-donut-wrap"><div class="siq-donut">{svg}'
            f'<div class="siq-donut-hole"><span>Total spent</span><b>{escape(total)}</b></div>{"".join(tips)}</div>'
            f'<div class="siq-legend">{"".join(legend)}</div></div>')


def category_bars(items: list[dict], mode: str) -> str:
    """items: [{category, total, share_percent, kind}] sorted descending. Bar width = amount / largest amount
    (min 4%). Hovering a row pops up "Category - Rs amount - share% of total" at the end of the bar."""
    rows = []
    top = max((i["total"] for i in items), default=0) or 1
    for i in items:
        value = f"{i['share_percent']}%" if mode == "Percentage" else inr(i["total"])
        saving = chip("saving", "green") if i.get("kind") == "saving" else ""
        tone = CATEGORY_TONES.get(i["category"], "slate")
        width = max(4.0, i["total"] / top * 100)
        text = f"{i['category']} - {inr(i['total'])} - {i['share_percent']}% of total"
        tip_pos = "right:0" if width > 65 else f"left:{width:.1f}%;transform:translateX(-50%)"
        rows.append(
            f'<div class="siq-bar-row tone-{tone}" title="{_q(text)}">'
            f'{tile(CATEGORY_ICONS.get(i["category"], "category"), tone, "small")}'
            f'<div class="siq-bar-main"><div class="siq-bar-name">{escape(i["category"])} {saving}</div>'
            f'<div class="siq-bar-track"><div class="siq-bar-fill" style="width:{width:.1f}%"></div></div>'
            f'<div class="siq-tip bar-tip" style="{tip_pos}">{escape(text)}</div></div>'
            f'<div class="siq-bar-val">{value}</div></div>')
    return f'<div class="siq-bars">{"".join(rows)}</div>'


def _axis_label(v: float) -> str:
    if v == 0:
        return "Rs 0"
    return f"Rs {v / 100000:g}L" if v >= 100000 else f"Rs {v / 1000:g}K"


def _nice_max(v: float) -> float:
    magnitude = 10 ** math.floor(math.log10(v)) if v > 0 else 1
    for step in (1, 1.2, 2, 2.4, 3, 4, 5, 6, 8, 10):
        if v <= step * magnitude:
            return step * magnitude
    return 10 * magnitude


def monthly_columns(months: list[tuple[str, float]]) -> str:
    """months: [("2026-04", total)] -> a column chart with month names and a Rs axis.
    Hovering a bar pops up "Apr 2026: Rs 35,569 (18.2%)" (same totals that draw the bars)."""
    top = _nice_max(max((t for _, t in months), default=0))
    grand = sum(t for _, t in months) or 1
    axis = "".join(f"<span>{_axis_label(top * k / 4)}</span>" for k in (4, 3, 2, 1, 0))
    lines = "".join("<div></div>" for _ in range(5))
    cols = []
    for n, (m, t) in enumerate(months):
        pct = t / top * 100
        text = f"{month_name(m)}: {inr(t)} ({t / grand * 100:.1f}%)"
        edge = "edge-left" if n == 0 and len(months) > 1 else "edge-right" if n == len(months) - 1 and len(months) > 1 else ""
        inside = "inside" if pct > 70 else ""  # tall bars: the popup sits inside the bar so it stays in the card
        cols.append(
            f'<div class="siq-col {edge}" title="{_q(text)}">'
            f'<div class="siq-col-val" style="bottom:{pct:.1f}%">{inr(t)}</div>'
            f'<div class="siq-col-bar" style="height:{pct:.1f}%"></div>'
            f'<div class="siq-tip col-tip {inside}" style="bottom:{pct:.1f}%">{escape(text)}</div>'
            f'<div class="siq-col-label">{escape(month_name(m))}</div></div>')
    return (f'<div class="siq-chart"><div class="siq-axis">{axis}</div>'
            f'<div class="siq-plot"><div class="siq-grid-lines">{lines}</div>{"".join(cols)}</div></div>')


def _alert_parts(flag: dict) -> tuple[str, str, str, str, dict]:
    label, icon_name = ALERT_TYPES.get(flag["type"], (flag["type"].replace("_", " ").title(), "notifications"))
    sev_label, tone = SEVERITY_TONES.get(flag["severity"], (flag["severity"].title(), "amber"))
    return label, icon_name, sev_label, tone, flag["transaction"]


def latest_alerts_list(flags: list[dict]) -> str:
    """Compact rows: tile, type + merchant/date, severity badge, amount."""
    if not flags:
        return empty_state("verified", "No alerts for this statement.", "green")
    rows = []
    for f in flags:
        label, icon_name, sev_label, tone, t = _alert_parts(f)
        rows.append(f'<div class="siq-mini-row">{tile(icon_name, tone, "small")}'
                    f'<div class="siq-mini-main"><b>{escape(label)}</b>'
                    f'<span>{escape(t["merchant"])} · {fmt_date(t["date"])}</span></div>'
                    f'{chip(sev_label, tone)}<div class="siq-mini-amount">{inr(t["amount"])}</div></div>')
    return f'<div class="siq-mini">{"".join(rows)}</div>'


def next_payments_list(items: list[dict], categories: dict | None = None) -> str:
    """Compact rows for the next recurring payments: tile, merchant + date, amount."""
    if not items:
        return empty_state("event_repeat", "No recurring payments found.")
    rows = []
    for r in items:
        label, tone = type_chip(r["kind"], (categories or {}).get(r["merchant"]))
        rows.append(f'<div class="siq-mini-row">{tile("event_repeat", tone, "small")}'
                    f'<div class="siq-mini-main"><b>{escape(r["merchant"])}</b>'
                    f'<span>Expected {fmt_date(r["next_expected"])}</span></div>'
                    f'{chip(label, tone)}<div class="siq-mini-amount">{inr(r["amount"])}</div></div>')
    return f'<div class="siq-mini">{"".join(rows)}</div>'


# ---------------------------------------------------------------- recurring
def recurring_strip(title: str, per_month: float, note: str) -> str:
    return (f'<div class="siq-strip"><div class="siq-strip-main">{tile("event_repeat", "violet", "large")}'
            f'<div><div class="siq-strip-title">{escape(title)}</div>'
            f'<div class="siq-helper">{escape(note)}</div></div></div>'
            f'<div><div class="siq-strip-value">{inr(per_month)}</div>'
            f'<div class="siq-helper" style="text-align:right">going out per month</div></div></div>')


def avatar(name: str) -> str:
    """Initials in a circle; the colour is picked from the palette by a stable hash of the name."""
    tone = AVATAR_TONES[zlib.crc32(name.encode("utf-8")) % len(AVATAR_TONES)]
    words = [w for w in name.replace("-", " ").split() if w]
    initials = "".join(w[0] for w in words[:2]).upper() or "?"
    return (f'<div class="siq-merchant"><span class="siq-avatar tone-{tone}">{escape(initials)}</span>'
            f'<span>{escape(name)}</span></div>')


def recurring_table(rows: list[dict], categories: dict | None = None) -> str:
    """Amount cells carry a thin share bar: the item's monthly cost against the largest outgoing item."""
    top = max((r["monthly_cost"] for r in rows if r["kind"] != "income"), default=0) or 1
    body = []
    for n, r in enumerate(rows, 1):
        label, tone = type_chip(r["kind"], (categories or {}).get(r["merchant"]))
        bar = ("" if r["kind"] == "income" else
               f'<div class="siq-share tone-violet"><div style="width:{max(4.0, r["monthly_cost"] / top * 100):.1f}%"></div></div>')
        each = "" if r["frequency"] == "monthly" else (
            f' <span class="muted">({inr(r["amount"])} {escape(r["frequency"])})</span>')
        body.append(
            f'<tr><td class="muted">{n}</td><td>{avatar(r["merchant"])}</td>'
            f'<td>{inr(r["monthly_cost"])}{each}{bar}</td>'
            f'<td>{chip(CADENCE.get(r["frequency"], r["frequency"].title()), "teal")}</td><td>{chip(label, tone)}</td>'
            f'<td class="muted">{fmt_date(r["last_charged"])}</td><td class="muted">{fmt_date(r["next_expected"])}</td></tr>')
    return ('<div class="siq-table-wrap"><table class="siq-table"><thead><tr><th>#</th><th>Merchant</th>'
            '<th>Monthly amount</th><th>Cadence</th><th>Type</th><th>Last paid</th><th>Next expected</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


# ---------------------------------------------------------------- alerts
def alert_summary(flags: list[dict]) -> str:
    counts = {}
    for f in flags:
        label, tone = SEVERITY_TONES.get(f["severity"], (f["severity"].title(), "amber"))
        counts[label] = (counts.get(label, (0, tone))[0] + 1, tone)
    order = [l for l in ("High", "Medium", "Low") if l in counts]
    chips = "".join(chip(f"{counts[l][0]} {l.lower()}", counts[l][1]) for l in order)
    n = len(flags)
    return (f'<div class="siq-card siq-strip"><div class="siq-strip-main">{tile("notifications", "violet", "large")}'
            f'<div class="siq-strip-title">{n} alert{"" if n == 1 else "s"}</div></div>'
            f'<div class="siq-badges">{chips}</div></div>')


def alert_card(flag: dict) -> str:
    label, icon_name, sev_label, tone, t = _alert_parts(flag)
    return (f'<div class="siq-card siq-alert tone-{tone} {flag["severity"]}">{tile(icon_name, tone, "large round")}'
            f'<div>{chip(sev_label, tone)}</div><div class="siq-alert-type">{escape(label)}</div>'
            f'<div class="siq-alert-merchant">{escape(t["merchant"])}</div>'
            f'<div class="siq-alert-date">{fmt_date(t["date"])}</div>'
            f'<div class="siq-alert-amount">{inr(t["amount"])}</div>'
            f'<div class="siq-alert-reason">{escape(friendly_dates(flag["reason"]))}</div>'
            f'<div class="siq-chevron">{icon("chevron_right")}</div></div>')


# ---------------------------------------------------------------- review
def review_count(n: int) -> str:
    return f'<div class="siq-count">{n} transaction{"" if n == 1 else "s"} to review</div>'


def review_cell(main: str, small: str = "") -> str:
    extra = f'<small title="{escape(small)}">{escape(small)}</small>' if small else ""
    return f'<div class="siq-review-cell">{escape(main)}{extra}</div>'


def review_chip_cell(category: str) -> str:
    return f'<div class="siq-review-cell">{chip(category, CATEGORY_TONES.get(category, "slate"))}</div>'


def review_header(labels: list[str]) -> list[str]:
    return [f'<div class="siq-review-head">{escape(l)}</div>' for l in labels]


# ---------------------------------------------------------------- landing
def landing_hero() -> str:
    """First screen: big headline, a statement illustration, and the three steps."""
    rows = "".join(
        f'<div class="siq-paper-row{" on" if n == 3 else ""}"><i></i><i></i><i></i></div>' for n in range(6))
    art = ('<div class="siq-scene"><div class="siq-notebook"></div><div class="siq-paper">'
           '<div class="siq-paper-title">Bank Statement</div>'
           '<div class="siq-paper-head"><span>Date</span><span>Description</span><span>Amount</span></div>'
           f'{rows}</div><div class="siq-scene-chip">{chip("Income", "green")}</div></div>')
    steps = [("01", "description", "Upload statement", "Your bank or UPI statement (PDF, CSV or Excel)"),
             ("02", "bar_chart", "Review your finances", "We extract and categorise your transactions"),
             ("03", "chat_bubble_outline", "Ask better questions", "Get insights, alerts and personalised advice")]
    step_html = '<div class="siq-step-line"></div>'.join(
        f'<div class="siq-step2"><div class="siq-step-n">{n}</div><span class="siq-step-icon">{icon(ic)}</span>'
        f'<div><b>{t}</b><span>{d}</span></div></div>' for n, ic, t, d in steps)
    return ('<div class="siq-land"><div><h1>Understand<br>your money, <span class="siq-accent">clearly.</span></h1>'
            '<p>One clear view of your spending, recurring payments, and what deserves a closer look.</p></div>'
            f'{art}</div><div class="siq-steps2">{step_html}</div>')


def privacy_note() -> str:
    return (f'<div class="siq-privacy">{icon("lock")}<div><b>Your data stays private.</b>'
            '<span>Processed in this session only. Demo uses fake data.</span></div></div>')


def chat_hero() -> str:
    """Empty chat state: gradient orb, headline and one-line hint."""
    return ('<div class="siq-card siq-chat-hero"><div class="siq-orb">' + icon("auto_awesome") + '</div>'
            '<h2>Ask me anything about your statement</h2>'
            '<div class="siq-helper">Pick a question below or type your own. Every number comes from your statement.</div></div>')


def file_chip(name: str, size: int) -> str:
    """The chosen file: icon, name and size, in one clean chip."""
    shown = f"{size / 1024:.0f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"
    return (f'<div class="siq-file">{icon("description")}<b title="{escape(name)}">{escape(name)}</b>'
            f'<span>{shown}</span></div>')


TABLE_LIMIT = 200


def transactions_table(rows: list[dict], total: int) -> str:
    """Styled All-transactions table (scrolls, sticky header). rows are already filtered, at most TABLE_LIMIT."""
    if not rows:
        return empty_state("search_off", "No transactions match your search.")
    body = []
    for r in rows:
        credit = r["type"] == "credit"
        amount = f'<span class="{"pos" if credit else "neg"}">{"+" if credit else "-"}{inr(r["amount"])}</span>'
        tone = CATEGORY_TONES.get(r["category"], "slate")
        body.append(
            f'<tr><td class="muted nowrap">{fmt_date(r["date"])}</td><td class="nowrap">{escape(r["merchant"])}</td>'
            f'<td class="trunc muted" title="{escape(r["description"])}">{escape(r["description"])}</td>'
            f'<td class="num nowrap">{amount}</td><td>{chip(r["category"], tone)}</td>'
            f'<td class="num muted">{round(r["confidence"] * 100)}%</td></tr>')
    note = (f'<div class="siq-helper siq-table-note">Showing first {TABLE_LIMIT} of {total:,}</div>'
            if total > len(rows) else "")
    return ('<div class="siq-scroll"><table class="siq-table"><thead><tr><th>Date</th><th>Merchant</th><th>Description</th>'
            '<th class="num">Amount</th><th>Category</th><th class="num">Confidence</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>{note}')


def error_box(message: str) -> str:
    return f'<div class="siq-error">{escape(message)}</div>'
