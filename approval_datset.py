"""
Loan Approval Recommendation — Prompt-Completion Dataset Generator
Produces: outputs/approval_dataset.jsonl

Output classes : Approve | Approve with Conditions | Reject
Completion     : Label only — no rationale, no extra text.

Label derivation logic (rule-based, deterministic)
───────────────────────────────────────────────────
REJECT
  · Default or write-off confirmed
  · Active delinquency > 60 DPD
  · RISK_TIER == CRITICAL
  · Bureau score < 600 (severe credit risk)

APPROVE
  · RISK_TIER == LOW  AND  Current DPD == 0  AND  Max DPD ≤ 30
  · OR bureau ≥ 750  AND  FOIR ≤ 0.40  AND  no active DPD

APPROVE WITH CONDITIONS
  · Everything else (moderate signals, recoverable risk)

Column selection
────────────────
IN  : AGE, OCCUPATION, MONTHLY_INCOME, DISPOSABLE_INCOME,
      LOAN_PRODUCT, LOAN_AMOUNT, LOAN_TENURE, EMI_AMOUNT,
      FOIR + FOIR_BAND, BUREAU_SCORE + BUREAU_BAND,
      CURRENT_DPD + CURR_DPD_LABEL, MAX_DPD + MAX_DPD_LABEL,
      CREDIT_UTIL + UTIL_LABEL, PAYMENT_BEHAVIOUR,
      DEFAULT_FLAG, WRITE_OFF_FLAG
DROP: IDs, STATE, GENDER (fair-lending), OUTSTANDING_BALANCE
      (captured by CREDIT_UTIL), COLLECTION_BUCKET (covered by DPD),
      LOAN_STATUS / RISK_TIER / ACCOUNT_HEALTH (output-side leakage),
      SANCTION_AMOUNT / SANCTION_RATIO (post-decision fields),
      REPAYMENT_STAGE / TENURE_STAGE, *_IMPUTED flags, raw dates.
"""

import pandas as pd
import numpy as np
import json
import os
from collections import Counter

df = pd.read_csv("Lending_Loan_Portfolio_1000_Featured.csv")
os.makedirs("outputs", exist_ok=True)


# ── LABEL DERIVATION ──────────────────────────────────────────────────────────

def derive_label(row) -> str:
    score  = row['BUREAU_SCORE'] if pd.notna(row['BUREAU_SCORE']) else None
    cdpd   = int(row['CURRENT_DPD'])
    mdpd   = int(row['MAX_DPD'])
    foir   = row['FOIR']
    tier   = row['RISK_TIER']

    # ── Hard REJECT ──
    if row['DEFAULT_FLAG'] or row['WRITE_OFF_FLAG']:
        return "Reject"
    if cdpd > 60:
        return "Reject"
    if tier == "CRITICAL":
        return "Reject"
    if score is not None and score < 600:
        return "Reject"

    # ── Clean APPROVE ──
    if tier == "LOW" and cdpd == 0 and mdpd <= 30:
        return "Approve"
    if score is not None and score >= 750 and foir <= 0.40 and cdpd == 0 and mdpd == 0:
        return "Approve"

    # ── Everything else ──
    return "Approve with Conditions"


df['APPROVAL_LABEL'] = df.apply(derive_label, axis=1)


# ── PROMPT BUILDER ────────────────────────────────────────────────────────────

def build_prompt(row) -> str:
    bureau_str = (
        f"{int(row['BUREAU_SCORE'])} ({row['BUREAU_BAND']})"
        if pd.notna(row['BUREAU_SCORE'])
        else f"Not Available ({row['BUREAU_BAND']})"
    )

    return f"""[TASK] Recommend a loan approval decision: Approve | Approve with Conditions | Reject

[APPLICANT]
Age / Occupation : {row['AGE']} yrs | {row['OCCUPATION']}
Monthly Income   : ₹{row['MONTHLY_INCOME']:,.0f} | Disposable ₹{row['DISPOSABLE_INCOME']:,.0f}

[LOAN REQUEST]
Product          : {row['LOAN_PRODUCT']}
Amount Requested : ₹{row['LOAN_AMOUNT']:,.0f}
Tenure           : {row['LOAN_TENURE']} months | Est. EMI ₹{row['EMI_AMOUNT']:,.0f}
FOIR             : {row['FOIR']:.2f} ({row['FOIR_BAND']})

[CREDIT PROFILE]
Bureau Score     : {bureau_str}
Payment History  : {row['PAYMENT_BEHAVIOUR']}
Current DPD      : {int(row['CURRENT_DPD'])} days ({row['CURR_DPD_LABEL']})
Max DPD          : {int(row['MAX_DPD'])} days ({row['MAX_DPD_LABEL']})
Credit Util.     : {row['CREDIT_UTIL']:.0%} ({row['UTIL_LABEL']})

[FLAGS]
Default          : {"Yes" if row['DEFAULT_FLAG'] else "No"} | Write-Off: {"Yes" if row['WRITE_OFF_FLAG'] else "No"}

[RECOMMENDATION]"""


# ── GENERATE JSONL ────────────────────────────────────────────────────────────

output_path = "outputs/approval_dataset.jsonl"
records = []

for _, row in df.iterrows():
    records.append({
        "prompt":     build_prompt(row),
        "completion": row['APPROVAL_LABEL'],   # bare label only
        "_meta": {
            "label":   row['APPROVAL_LABEL'],
            "product": row['LOAN_PRODUCT'],
            "tier":    row['RISK_TIER'],
        }
    })

with open(output_path, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✓ Generated {len(records)} records → {output_path}")

# ── Distribution check ────────────────────────────────────────────────────────
print("\n=== CLASS DISTRIBUTION ===")
dist = Counter(r['_meta']['label'] for r in records)
for label in ['Approve', 'Approve with Conditions', 'Reject']:
    n = dist.get(label, 0)
    pct = n / len(records) * 100
    bar = '█' * (n // 15)
    print(f"  {label:25s}: {n:4d} ({pct:5.1f}%)  {bar}")

# ── Sample outputs ────────────────────────────────────────────────────────────
print("\n=== SAMPLES ===")
for label in ['Approve', 'Approve with Conditions', 'Reject']:
    sample = next((r for r in records if r['_meta']['label'] == label), None)
    if sample:
        print(f"\n── {label.upper()} ──")
        print(sample['prompt'])
        print(f">>> {sample['completion']}")