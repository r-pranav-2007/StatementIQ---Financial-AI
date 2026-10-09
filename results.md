# StatementIQ evaluation (2026-10-09)

| Section | Test | Offline (rules only) | LLM |
|---|---|---|---|
| Parsing | 1_bank_separate_columns.csv | 77/77 (100%) | not run |
| Parsing | 2_bank_with_header_junk.csv | 77/77 (100%) | not run |
| Parsing | 3_bank_single_amount_drcr.csv | 77/77 (100%) | not run |
| Parsing | 4_wallet_upi_app.csv | 77/77 (100%) | not run |
| Parsing | 5_bank_statement.xlsx | 77/77 (100%) | not run |
| Parsing | 6_bank_statement.pdf | 77/77 (100%) | not run |
| Categorization | Bank CSV | 77/77 (100%) | not run |
| Categorization | Wallet CSV | 77/77 (100%) | not run |
| Categorization | Hard test (25 unseen) | 18/25 (72%) | not run |
| Categorization | Holdout (25 unseen) | 9/25 (36%) | not run |
| Recurring detection | Bank CSV: found vs planted | 8/8, 0 false positives | not run |
| Recurring detection | Wallet CSV: found vs planted | 8/8, 0 false positives | not run |
| Anomaly detection | Bank CSV: caught vs planted | 4/4, 0 false alarms | not run |
| Anomaly detection | Wallet CSV: caught vs planted | 4/4, 0 false alarms | not run |
| Query | Correct answers + hallucinated numbers | 20/20, 0 hallucinated | not run |
| Agent | Tool selection | 6/6 (100%) | not run |
