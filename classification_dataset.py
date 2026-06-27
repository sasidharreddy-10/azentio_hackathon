"""
Credit Risk Classification — Prompt-Completion Dataset Generator
Produces: outputs/classification_dataset.jsonl

Output classes : LOW RISK | MEDIUM RISK | HIGH RISK | CRITICAL RISK
Completion style: Label + one-line rationale (helps model learn WHY, not just WHAT)

Column selection
────────────────
IN  : Demographics, bureau score+band, FOIR+band, DPD fields+labels,
      collection bucket, credit utilization+label, payment behaviour,
      disposable income, default/write-off flags, loan product.
OUT : RISK_TIER, ACCOUNT_HEALTH, PRIMARY_RISK  ← target; never in prompt.
DROP: Loan IDs / Customer IDs (identifiers, not signals), SANCTION_AMOUNT /
      EMI_AMOUNT / OUTSTANDING_BALANCE (captured via FOIR & UTIL ratios),
      LOAN_TENURE / INTEREST_RATE / DISBURSAL dates (low classification signal),
      REPAYMENT_STAGE / TENURE_STAGE (loan-age info, not risk signal),
      *_IMPUTED flags (pipeline artefacts).
"""

import pandas as pd
import numpy as np
import json
import os

df = pd.read_csv("Lending_Loan_Portfolio_1000_Featured.csv")
os.makedirs("outputs", exist_ok=True)


# ── PROMPT BUILDER ────────────────────────────────────────────────────────────

def build_prompt(row) -> str:
    bureau_str = (
        f"{int(row['BUREAU_SCORE'])} ({row['BUREAU_BAND']})"
        if pd.notna(row['BUREAU_SCORE'])
        else f"Not Available ({row['BUREAU_BAND']})"
    )

    return f"""[TASK] Classify the credit risk of the borrower below.
Output one of: LOW RISK | MEDIUM RISK | HIGH RISK | CRITICAL RISK

[BORROWER]
Age / Gender   : {row['AGE']} yrs | {row['GENDER']}
Occupation     : {row['OCCUPATION']}
Monthly Income : ₹{row['MONTHLY_INCOME']:,.0f} | Disposable ₹{row['DISPOSABLE_INCOME']:,.0f}

[CREDITWORTHINESS]
Bureau Score   : {bureau_str}
FOIR           : {row['FOIR']:.2f} ({row['FOIR_BAND']})
Credit Util.   : {row['CREDIT_UTIL']:.0%} ({row['UTIL_LABEL']})

[REPAYMENT HISTORY]
Current DPD    : {int(row['CURRENT_DPD'])} days ({row['CURR_DPD_LABEL']})
Max DPD        : {int(row['MAX_DPD'])} days ({row['MAX_DPD_LABEL']})
Bucket         : {row['COLLECTION_BUCKET']}
Payment Trend  : {row['PAYMENT_BEHAVIOUR']}

[FLAGS]
Loan Product   : {row['LOAN_PRODUCT']}
Default        : {"Yes" if row['DEFAULT_FLAG'] else "No"}
Write-Off      : {"Yes" if row['WRITE_OFF_FLAG'] else "No"}

[CLASSIFICATION]"""


# ── COMPLETION BUILDER ────────────────────────────────────────────────────────

def build_completion(row) -> str:
    """
    Label + one-line rationale.
    Rationale cites the 2-3 strongest signals that drove the classification.
    """
    r   = row
    tier = r['RISK_TIER']
    bureau = (
        f"{int(r['BUREAU_SCORE'])} ({r['BUREAU_BAND']})"
        if pd.notna(r['BUREAU_SCORE']) else f"N/A ({r['BUREAU_BAND']})"
    )

    def top_signals():
        """Pick the 2-3 most decisive signals for the rationale sentence."""
        signals = []

        # Bureau
        if pd.notna(r['BUREAU_SCORE']):
            if r['BUREAU_SCORE'] >= 750:
                signals.append(f"strong bureau score ({bureau})")
            elif r['BUREAU_SCORE'] >= 700:
                signals.append(f"good bureau score ({bureau})")
            elif r['BUREAU_SCORE'] >= 650:
                signals.append(f"moderate bureau score ({bureau})")
            else:
                signals.append(f"weak bureau score ({bureau})")
        else:
            signals.append("bureau score unavailable")

        # DPD — highest priority delinquency signal
        cdpd, mdpd = int(r['CURRENT_DPD']), int(r['MAX_DPD'])
        if cdpd > 90:
            signals.append(f"critical active delinquency ({cdpd} DPD)")
        elif cdpd > 60:
            signals.append(f"serious active delinquency ({cdpd} DPD)")
        elif cdpd > 30:
            signals.append(f"moderate active delinquency ({cdpd} DPD)")
        elif cdpd > 0:
            signals.append(f"slight active delay ({cdpd} DPD)")
        elif mdpd > 90:
            signals.append(f"historical default-level DPD ({mdpd} days max)")
        elif mdpd > 0:
            signals.append(f"past delinquency (max {mdpd} days, now recovered)")
        else:
            signals.append("no payment delays")

        # FOIR
        if r['FOIR'] > 0.6:
            signals.append(f"stressed FOIR {r['FOIR']:.2f}")
        elif r['FOIR'] > 0.4:
            signals.append(f"manageable FOIR {r['FOIR']:.2f}")
        else:
            signals.append(f"comfortable FOIR {r['FOIR']:.2f}")

        # Default / write-off override
        if r['DEFAULT_FLAG']:   signals.insert(0, "confirmed default")
        if r['WRITE_OFF_FLAG']: signals.insert(0, "write-off recorded")

        return signals[:3]   # top 3 only

    signals_str = "; ".join(top_signals())

    ACTION = {
        "LOW":      "Standard monitoring — eligible for cross-sell if vintage > 12 months.",
        "MEDIUM":   "Enhanced monitoring — escalate if DPD or FOIR worsens.",
        "HIGH":     "Immediate collection follow-up — evaluate restructuring.",
        "CRITICAL": "Initiate recovery proceedings — assess one-time settlement.",
    }

    label = f"{tier} RISK"
    rationale = f"Key signals: {signals_str}. {ACTION[tier]}"

    return f"{label}\n{rationale}"


# ── GENERATE JSONL ────────────────────────────────────────────────────────────

output_path = "outputs/classification_dataset.jsonl"
records = []

for _, row in df.iterrows():
    records.append({
        "prompt":     build_prompt(row),
        "completion": build_completion(row),
        "_meta": {
            "risk_tier":   row['RISK_TIER'],
            "product":     row['LOAN_PRODUCT'],
            "default":     bool(row['DEFAULT_FLAG']),
            "write_off":   bool(row['WRITE_OFF_FLAG']),
        }
    })

with open(output_path, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✓ Generated {len(records)} records → {output_path}")

# ── Distribution check ────────────────────────────────────────────────────────
print("\n=== CLASS DISTRIBUTION ===")
from collections import Counter
dist = Counter(r['_meta']['risk_tier'] for r in records)
for tier in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
    n = dist.get(tier, 0)
    bar = '█' * (n // 10)
    print(f"  {tier:8s} : {n:4d}  {bar}")

# ── Sample outputs ────────────────────────────────────────────────────────────
print("\n=== SAMPLES ===")
shown = set()
for tier in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
    sample = next((r for r in records if r['_meta']['risk_tier'] == tier), None)
    if sample:
        print(f"\n── {tier} RISK SAMPLE ──")
        print("PROMPT:\n" + sample['prompt'])
        print("COMPLETION:\n" + sample['completion'])