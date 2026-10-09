# StatementIQ evaluation (2026-10-09)

| Section | Test | Offline (rules only) | LLM |
|---|---|---|---|
| Parsing | 1_bank_separate_columns.csv | 77/77 (100%) | - |
| Parsing | 2_bank_with_header_junk.csv | 77/77 (100%) | - |
| Parsing | 3_bank_single_amount_drcr.csv | 77/77 (100%) | - |
| Parsing | 4_wallet_upi_app.csv | 77/77 (100%) | - |
| Parsing | 5_bank_statement.xlsx | 77/77 (100%) | - |
| Parsing | 6_bank_statement.pdf | 77/77 (100%) | - |
| Categorization | Bank CSV | 77/77 (100%) | 77/77 (100%) |
| Categorization | Wallet CSV | 77/77 (100%) | 77/77 (100%) |
| Categorization | Hard test (25 unseen) | 18/25 (72%) | 25/25 (100%) |
| Categorization | Holdout (25 unseen) | 9/25 (36%) | 23/25 (92%) |
| Recurring detection | Bank CSV: found vs planted | 8/8, 0 false positives | - |
| Recurring detection | Wallet CSV: found vs planted | 8/8, 0 false positives | - |
| Anomaly detection | Bank CSV: caught vs planted | 4/4, 0 false alarms | - |
| Anomaly detection | Wallet CSV: caught vs planted | 4/4, 0 false alarms | - |
| Query | Correct answers + hallucinated numbers | 20/20, 0 hallucinated | 20/20, 0 hallucinated |
| Agent | Tool selection | 6/6 (100%) | 6/6 (100%) |

API usage: 62 calls, 43880 input + 3972 output tokens, about $0.0637
