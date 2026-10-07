"""
StatementIQ - generates the SAME transactions in different real-world layouts,
so the parser can be tested against every format users are likely to upload.

Reads ground_truth.json (run generate_fake_data.py first).
Writes to data/formats/:
  1_bank_separate_columns.csv   (HDFC-like: Withdrawal / Deposit columns)
  2_bank_with_header_junk.csv   (SBI-like: preamble lines, DD Mon YYYY dates, 1,234.00 amounts)
  3_bank_single_amount_drcr.csv (Kotak-like: one Amount column with (Dr)/(Cr))
  4_wallet_upi_app.csv          (PhonePe/Paytm-like: no balance, "Paid to ...", rupee sign)
  5_bank_statement.xlsx         (Excel export of layout 1)
  6_bank_statement.pdf          (PDF table of layout 1)
"""
import csv
import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "formats"
OUT.mkdir(exist_ok=True)

truth = json.load(open(HERE / "ground_truth.json", encoding="utf-8"))
rows = truth["transactions"]

# recompute running balance
bal = truth["start_balance"]
for r in rows:
    r["d"] = date.fromisoformat(r["date"])
    bal += r["amount"] if r["type"] == "credit" else -r["amount"]
    r["balance"] = round(bal, 2)


def indian(n: float) -> str:
    """12345.5 -> '12,345.50' (Indian grouping: 1,23,456.00)."""
    s = f"{n:.2f}"
    whole, dec = s.split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return f"{whole}.{dec}"


# 1. HDFC-like -----------------------------------------------------------
with open(OUT / "1_bank_separate_columns.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Date", "Narration", "Chq/Ref No", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"])
    for r in rows:
        parts = r["narration"].split("/")
        ref = parts[2] if len(parts) >= 3 else ""
        w.writerow([r["d"].strftime("%d/%m/%Y"), r["narration"], ref,
                    f'{r["amount"]:.2f}' if r["type"] == "debit" else "",
                    f'{r["amount"]:.2f}' if r["type"] == "credit" else "",
                    f'{r["balance"]:.2f}'])

# 2. SBI-like: junk lines above the real header, comma-grouped amounts ---------
with open(OUT / "2_bank_with_header_junk.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Account Name :", "DEMO USER"])
    w.writerow(["Address :", "12, SAMPLE STREET, CHENNAI"])
    w.writerow(["Account Number :", "XXXXXXXX4821"])
    w.writerow(["Statement From :", "01 Jul 2026 To 30 Sep 2026"])
    w.writerow([])
    w.writerow(["Txn Date", "Value Date", "Description", "Ref No./Cheque No.", "Debit", "Credit", "Balance"])
    for r in rows:
        ds = r["d"].strftime("%d %b %Y")
        w.writerow([ds, ds, r["narration"], "",
                    indian(r["amount"]) if r["type"] == "debit" else "",
                    indian(r["amount"]) if r["type"] == "credit" else "",
                    indian(r["balance"])])
    w.writerow([])
    w.writerow(["**** End of Statement ****"])

# 3. Kotak-like: single Amount column with (Dr)/(Cr) ---------------------------
with open(OUT / "3_bank_single_amount_drcr.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Date", "Narration", "Chq/Ref No", "Amount", "Balance"])
    for r in rows:
        tag = "Dr" if r["type"] == "debit" else "Cr"
        w.writerow([r["d"].strftime("%d-%m-%Y"), r["narration"], "",
                    f'{indian(r["amount"])}({tag})', f'{indian(r["balance"])}(Cr)'])

# 4. Wallet / UPI app export: no balance, friendly text, rupee sign -------------
with open(OUT / "4_wallet_upi_app.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Date", "Time", "Transaction Details", "Type", "Amount", "Status"])
    for i, r in enumerate(rows):
        verb = "Paid to" if r["type"] == "debit" else "Received from"
        who = r["merchant"] if r["merchant"] != "Salary" else "ACME TECH SOLUTIONS"
        w.writerow([r["d"].strftime("%b %d, %Y"), f"{(9 + i * 7) % 12 + 1:02d}:{(i * 13) % 60:02d} PM",
                    f"{verb} {who}", r["type"].upper(), f'₹{indian(r["amount"])}', "SUCCESS"])

# 5. Excel ---------------------------------------------------------------------
from openpyxl import Workbook
wb = Workbook()
ws = wb.active
ws.title = "Statement"
ws.append(["Date", "Narration", "Chq/Ref No", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"])
for r in rows:
    parts = r["narration"].split("/")
    ws.append([r["d"], r["narration"], parts[2] if len(parts) >= 3 else "",
               r["amount"] if r["type"] == "debit" else None,
               r["amount"] if r["type"] == "credit" else None, r["balance"]])
for row in ws.iter_rows(min_row=2, max_col=1):
    row[0].number_format = "DD/MM/YYYY"
wb.save(OUT / "5_bank_statement.xlsx")

# 6. PDF table -----------------------------------------------------------------
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate(str(OUT / "6_bank_statement.pdf"), pagesize=landscape(A4),
                        leftMargin=20, rightMargin=20, topMargin=20, bottomMargin=20)
styles = getSampleStyleSheet()
data = [["Date", "Narration", "Chq/Ref No", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"]]
small = styles["BodyText"]
small.fontSize = 7
for r in rows:
    parts = r["narration"].split("/")
    data.append([r["d"].strftime("%d/%m/%Y"), Paragraph(r["narration"], small),
                 parts[2] if len(parts) >= 3 else "",
                 indian(r["amount"]) if r["type"] == "debit" else "",
                 indian(r["amount"]) if r["type"] == "credit" else "",
                 indian(r["balance"])])
t = Table(data, repeatRows=1, colWidths=[70, 300, 100, 90, 90, 90])
t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                       ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                       ("FONTSIZE", (0, 0), (-1, -1), 7),
                       ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
doc.build([Paragraph("DEMO BANK - Account Statement (Sample)", styles["Title"]), Spacer(1, 6), t])

print("Generated format samples in", OUT)
for p in sorted(OUT.iterdir()):
    print(" -", p.name)
