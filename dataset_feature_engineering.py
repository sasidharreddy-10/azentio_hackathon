import pandas as pd
import numpy as np
import json
import io
from datetime import datetime

df = pd.read_excel("Lending_Loan_Portfolio_1000_Raw.xlsx", sheet_name="Loan_Portfolio_Raw", engine="openpyxl")
# rest of the pipeline is identical

# ─── 2. CLEAN & NORMALIZE ────────────────────────────────────────────────────

import numpy as np
import pandas as pd

def clean(df):
    df = df.copy()

    TODAY = pd.Timestamp("2026-06-27")

    # -----------------------------
    # Helpers
    # -----------------------------
    def norm_text(s):
        return s.astype("string").str.strip()

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def clip_round(x, low, high):
        return float(np.clip(x, low, high))

    # -----------------------------
    # Standardize text fields
    # -----------------------------
    gender_map = {
        "male": "Male", "m": "Male",
        "female": "Female", "f": "Female"
    }

    df["GENDER"] = (
        norm_text(df["GENDER"])
        .str.lower()
        .map(gender_map)
        .fillna("Not Specified")
    )

    df["OCCUPATION"] = norm_text(df["OCCUPATION"]).fillna("Not Specified")
    df["STATE"] = norm_text(df["STATE"]).fillna("Not Specified")
    df["LOAN_PRODUCT"] = norm_text(df["LOAN_PRODUCT"]).fillna("Not Specified")
    df["COLLECTION_BUCKET"] = norm_text(df["COLLECTION_BUCKET"]).fillna("Current")
    df["LOAN_STATUS"] = norm_text(df["LOAN_STATUS"]).fillna("Active")

    # -----------------------------
    # Numeric fields
    # -----------------------------
    numeric_cols = [
        "AGE", "MONTHLY_INCOME", "LOAN_AMOUNT", "SANCTION_AMOUNT",
        "LOAN_TENURE", "INTEREST_RATE", "EMI_AMOUNT", "BUREAU_SCORE",
        "VINTAGE_MONTHS", "CURRENT_DPD", "MAX_DPD", "OUTSTANDING_BALANCE",
        "DEFAULT_FLAG", "WRITE_OFF_FLAG"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = to_num(df[c])

    # Flags as integers
    df["DEFAULT_FLAG"] = df["DEFAULT_FLAG"].fillna(0).astype(int)
    df["WRITE_OFF_FLAG"] = df["WRITE_OFF_FLAG"].fillna(0).astype(int)

    # Dates
    df["DISBURSAL_DATE"] = pd.to_datetime(df["DISBURSAL_DATE"], errors="coerce")
    df["LAST_PAYMENT_DATE"] = pd.to_datetime(df["LAST_PAYMENT_DATE"], errors="coerce")

    # -----------------------------
    # MONTHLY_INCOME imputation
    # Row-only FOIR assumptions
    # -----------------------------
    foir_by_occupation = {
        "salaried": 0.45,
        "government": 0.40,
        "professional": 0.50,
        "business": 0.55,
        "self employed": 0.60,
        "retired": 0.35,
        "student": 0.30,
        "not specified": 0.55
    }

    income_imputed = []

    def estimate_income(row):
        if pd.notna(row["MONTHLY_INCOME"]) and row["MONTHLY_INCOME"] > 0:
            income_imputed.append(False)
            return row["MONTHLY_INCOME"]

        # As requested: if default/writeoff is present, set income to 0
        if row["DEFAULT_FLAG"] == 1 or row["WRITE_OFF_FLAG"] == 1:
            income_imputed.append(True)
            return 0.0

        emi = row["EMI_AMOUNT"]
        if pd.isna(emi) or emi <= 0:
            income_imputed.append(True)
            return np.nan

        occ = str(row["OCCUPATION"]).strip().lower()
        ratio = foir_by_occupation.get(occ, 0.55)

        # Extra row-level guardrails:
        # keep FOIR in a sensible band for non-default loans
        ratio = float(np.clip(ratio, 0.30, 0.60))

        estimated = emi / ratio

        # Income should not be below EMI
        estimated = max(estimated, emi)

        # Round to nearest ₹100 for stability
        estimated = round(estimated / 100.0) * 100.0

        income_imputed.append(True)
        return estimated

    df["MONTHLY_INCOME"] = df.apply(estimate_income, axis=1)
    df["MONTHLY_INCOME_IMPUTED"] = income_imputed

    # -----------------------------
    # BUREAU_SCORE imputation
    # Row-only heuristic score
    # -----------------------------
    bucket_penalty = {
        "Current": 0,
        "1-30": 25,
        "31-60": 55,
        "61-90": 90,
        "90+": 130
    }

    status_penalty = {
        "Active": 0,
        "Current": 0,
        "Delinquent": 35,
        "Closed": -10,
        "Settled": -5,
        "Written Off": 140,
        "Write Off": 140
    }

    bureau_imputed = []

    def estimate_bureau_score(row):
        # Keep original if already present
        if pd.notna(row["BUREAU_SCORE"]):
            bureau_imputed.append(False)
            return row["BUREAU_SCORE"]

        bureau_imputed.append(True)

        score = 760.0  # fixed anchor, not a dataset statistic

        # Hard penalties for known adverse events
        if row["DEFAULT_FLAG"] == 1:
            score -= 120
        if row["WRITE_OFF_FLAG"] == 1:
            score -= 80

        # Delinquency signals
        current_dpd = max(float(row["CURRENT_DPD"]) if pd.notna(row["CURRENT_DPD"]) else 0.0, 0.0)
        max_dpd = max(float(row["MAX_DPD"]) if pd.notna(row["MAX_DPD"]) else 0.0, 0.0)

        score -= min(current_dpd, 180.0) * 1.20
        score -= min(max_dpd, 150.0) * 0.45

        # Collection bucket
        bucket = str(row["COLLECTION_BUCKET"]).strip()
        score -= bucket_penalty.get(bucket, 20)

        # Loan status
        status = str(row["LOAN_STATUS"]).strip()
        score -= status_penalty.get(status, 15)

        # Vintage helps slightly if the account is old and still performing
        vintage = max(float(row["VINTAGE_MONTHS"]) if pd.notna(row["VINTAGE_MONTHS"]) else 0.0, 0.0)
        score += min(vintage, 72.0) * 0.70

        # Higher interest rate usually signals risk
        ir = max(float(row["INTEREST_RATE"]) if pd.notna(row["INTEREST_RATE"]) else 0.0, 0.0)
        score -= max(ir - 12.0, 0.0) * 2.0

        # EMI burden using the row's own income estimate
        emi = float(row["EMI_AMOUNT"]) if pd.notna(row["EMI_AMOUNT"]) else 0.0
        income = float(row["MONTHLY_INCOME"]) if pd.notna(row["MONTHLY_INCOME"]) and row["MONTHLY_INCOME"] > 0 else 0.0
        if income > 0 and emi > 0:
            emi_ratio = emi / income
            # Penalty only when EMI becomes heavy
            score -= max(emi_ratio - 0.30, 0.0) * 220.0

        # Tiny age regularization, purely row-based
        age = float(row["AGE"]) if pd.notna(row["AGE"]) else 0.0
        if age > 0:
            score += max(0.0, 10.0 - abs(age - 40.0) / 4.0)

        score = clip_round(score, 300.0, 850.0)
        return score

    df["BUREAU_SCORE"] = df.apply(estimate_bureau_score, axis=1)
    df["BUREAU_SCORE_IMPUTED"] = bureau_imputed

    # -----------------------------
    # LAST_PAYMENT_DATE imputation
    # Row-only extrapolation from disbursal date, tenure, status, and DPD
    # -----------------------------
    bucket_month_shift = {
        "Current": 1,
        "1-30": 2,
        "31-60": 3,
        "61-90": 4,
        "90+": 5
    }

    last_payment_imputed = []

    def estimate_last_payment_date(row):
        if pd.notna(row["LAST_PAYMENT_DATE"]):
            last_payment_imputed.append(False)
            return row["LAST_PAYMENT_DATE"]

        disb = row["DISBURSAL_DATE"]
        if pd.isna(disb):
            last_payment_imputed.append(True)
            return pd.NaT

        tenure = int(max(row["LOAN_TENURE"], 0)) if pd.notna(row["LOAN_TENURE"]) else 0
        vintage = int(max(row["VINTAGE_MONTHS"], 0)) if pd.notna(row["VINTAGE_MONTHS"]) else 0

        bucket = str(row["COLLECTION_BUCKET"]).strip()
        status = str(row["LOAN_STATUS"]).strip()

        current_dpd = max(int(row["CURRENT_DPD"]) if pd.notna(row["CURRENT_DPD"]) else 0, 0)
        max_dpd = max(int(row["MAX_DPD"]) if pd.notna(row["MAX_DPD"]) else 0, 0)

        # Convert DPD into approximate month shift
        dpd_shift = int(np.ceil(max(current_dpd, max_dpd) / 30.0))
        bucket_shift = bucket_month_shift.get(bucket, 1)

        # Non-default loans: estimate the last payment as a recent installment date
        if row["DEFAULT_FLAG"] == 0 and row["WRITE_OFF_FLAG"] == 0:
            if status.lower() in {"closed", "settled"}:
                months_paid = tenure
            else:
                # recent installment before the current delinquency state
                months_paid = max(0, min(vintage, tenure) - max(1, bucket_shift))
        else:
            # Default/write-off: move the last payment earlier using the worst delinquency signal
            months_paid = max(0, min(vintage, tenure) - max(1, bucket_shift, dpd_shift))

        last_payment = disb + pd.DateOffset(months=int(months_paid))

        # Optional: keep it from becoming earlier than disbursal
        if last_payment < disb:
            last_payment = disb

        last_payment_imputed.append(True)
        return last_payment

    df["LAST_PAYMENT_DATE"] = df.apply(estimate_last_payment_date, axis=1)
    df["LAST_PAYMENT_DATE_IMPUTED"] = last_payment_imputed

    # -----------------------------
    # Extra safety flags
    # -----------------------------
    df["LAST_PMT_IS_FUTURE"] = df["LAST_PAYMENT_DATE"] > TODAY

    return df

df = clean(df)




# ─── 3. FEATURE ENGINEERING ──────────────────────────────────────────────────

def engineer(df):
    df = df.copy()

    # ------------------------------------------------------------------
    # Numerical Features
    # ------------------------------------------------------------------

    # Fixed Obligation to Income Ratio
    df['FOIR'] = (df['EMI_AMOUNT'] / df['MONTHLY_INCOME']).clip(0, 1.0).round(4)

    # Credit Utilization
    df['CREDIT_UTIL'] = (
        df['OUTSTANDING_BALANCE'] /
        df['SANCTION_AMOUNT'].replace(0, np.nan)
    ).clip(0, 1.0).round(4)

    # Sanction Ratio
    df['SANCTION_RATIO'] = (
        df['SANCTION_AMOUNT'] /
        df['LOAN_AMOUNT'].replace(0, np.nan)
    ).round(4)

    # Repayment Progress
    df['REPAYMENT_RATIO'] = (
        1 - (
            df['OUTSTANDING_BALANCE'] /
            df['SANCTION_AMOUNT'].replace(0, np.nan)
        )
    ).clip(0, 1).round(4)

    # Remaining Disposable Income
    df['DISPOSABLE_INCOME'] = (
        df['MONTHLY_INCOME'] -
        df['EMI_AMOUNT']
    ).clip(lower=0)

    # Tenure Completion
    df['TENURE_PROGRESS'] = (
        df['VINTAGE_MONTHS'] /
        df['LOAN_TENURE'].replace(0, np.nan)
    ).clip(0, 1).round(4)

    # Remaining Tenure
    df['MONTHS_LEFT'] = (
        df['LOAN_TENURE'] -
        df['VINTAGE_MONTHS']
    ).clip(lower=0)

    # Delinquency Flags
    df['IS_DELINQUENT'] = (df['CURRENT_DPD'] > 0).astype(int)

    df['HAD_PAST_DPD'] = (
        (df['CURRENT_DPD'] == 0) &
        (df['MAX_DPD'] > 0)
    ).astype(int)

    # ------------------------------------------------------------------
    # Label Functions
    # ------------------------------------------------------------------

    def bureau_band(score):
        if pd.isna(score):
            return "Not Available"
        if score >= 800:
            return "Excellent"
        if score >= 750:
            return "Very Good"
        if score >= 700:
            return "Good"
        if score >= 650:
            return "Moderate Risk"
        return "High Risk"

    def foir_band(foir):
        if foir <= 0.30:
            return "Comfortable"
        if foir <= 0.50:
            return "Manageable"
        if foir <= 0.60:
            return "Stretched"
        return "Financially Stressed"

    def dpd_label(dpd):
        if dpd <= 0:
            return "No Delinquency"
        if dpd <= 30:
            return "Slight Delay"
        if dpd <= 60:
            return "Moderate Delinquency"
        if dpd <= 90:
            return "Serious Delinquency"
        return "Potential Default"

    def util_label(util):
        if util <= 0.40:
            return "Low"
        if util <= 0.70:
            return "Moderate"
        if util <= 0.85:
            return "High"
        return "Very High"

    def repayment_stage(r):
        if pd.isna(r):
            return "Unknown"
        if r >= 0.90:
            return "Almost Fully Repaid"
        if r >= 0.70:
            return "Mostly Repaid"
        if r >= 0.40:
            return "Partially Repaid"
        return "Early Repayment Stage"

    def tenure_stage(p):
        if pd.isna(p):
            return "Unknown"
        if p <= 0.25:
            return "New Loan"
        if p <= 0.75:
            return "Mid Tenure"
        if p < 1.0:
            return "Near Completion"
        return "Completed"

    def payment_behaviour(row):

        if row['WRITE_OFF_FLAG'] == 1:
            return "Written Off"

        if row['DEFAULT_FLAG'] == 1:
            return "Defaulted"

        if row['CURRENT_DPD'] == 0 and row['MAX_DPD'] == 0:
            return "Always Paid On Time"

        if row['CURRENT_DPD'] == 0:
            return "Recovered From Past Delays"

        if row['CURRENT_DPD'] <= 30:
            return "Currently Delayed"

        return "Severely Delinquent"

    def sanction_label(r):
        if pd.isna(r):
            return "Unknown"
        if r >= 0.95:
            return "Fully Approved"
        if r >= 0.75:
            return "Mostly Approved"
        if r >= 0.50:
            return "Partially Approved"
        return "Significantly Reduced"

    def interest_band(rate):
        if rate < 8:
            return "Very Low"
        if rate < 12:
            return "Low"
        if rate < 16:
            return "Average"
        if rate < 20:
            return "High"
        return "Very High"

    def disposable_band(x):
        if x >= 50000:
            return "Very Comfortable"
        if x >= 25000:
            return "Comfortable"
        if x >= 10000:
            return "Limited"
        return "Financially Tight"

    # ------------------------------------------------------------------
    # Overall Risk Tier
    # ------------------------------------------------------------------

    def risk_tier(row):

        score = row['BUREAU_SCORE'] if pd.notna(row['BUREAU_SCORE']) else 650

        if row['WRITE_OFF_FLAG']:
            return "CRITICAL"

        if row['DEFAULT_FLAG']:
            return "CRITICAL"

        if row['CURRENT_DPD'] > 90:
            return "CRITICAL"

        if row['CURRENT_DPD'] > 60:
            return "HIGH"

        if row['CURRENT_DPD'] > 30:
            return "HIGH"

        if row['CURRENT_DPD'] > 0:
            return "MEDIUM"

        if row['MAX_DPD'] > 90:
            return "MEDIUM"

        if row['FOIR'] > 0.60:
            return "MEDIUM"

        if score < 650:
            return "MEDIUM"

        return "LOW"

    # ------------------------------------------------------------------
    # Loan Health Score (0-13)
    # ------------------------------------------------------------------

    def loan_health_score(row):

        s = 0

        if row['WRITE_OFF_FLAG']:
            s += 5

        if row['DEFAULT_FLAG']:
            s += 4

        if row['CURRENT_DPD'] > 0:
            s += 2

        if row['MAX_DPD'] > 90:
            s += 2

        if row['BUREAU_SCORE'] < 650:
            s += 2

        if row['FOIR'] > 0.60:
            s += 1

        if row['CREDIT_UTIL'] > 0.80:
            s += 1

        return s

    def account_health(score):

        if score == 0:
            return "Excellent"

        if score <= 2:
            return "Healthy"

        if score <= 4:
            return "Needs Monitoring"

        if score <= 6:
            return "High Risk"

        return "Critical"

    def primary_risk(row):

        if row['WRITE_OFF_FLAG']:
            return "Written Off"

        if row['DEFAULT_FLAG']:
            return "Defaulted"

        if row['CURRENT_DPD'] > 30:
            return "Current Delinquency"

        if row['FOIR'] > 0.60:
            return "High EMI Burden"

        if row['BUREAU_SCORE'] < 650:
            return "Weak Credit Profile"

        if row['CREDIT_UTIL'] > 0.85:
            return "High Credit Utilization"

        return "Healthy Account"

    # ------------------------------------------------------------------
    # Apply Labels
    # ------------------------------------------------------------------

    df['BUREAU_BAND'] = df['BUREAU_SCORE'].apply(bureau_band)

    df['FOIR_BAND'] = df['FOIR'].apply(foir_band)

    df['CURR_DPD_LABEL'] = df['CURRENT_DPD'].apply(dpd_label)

    df['MAX_DPD_LABEL'] = df['MAX_DPD'].apply(dpd_label)

    df['UTIL_LABEL'] = df['CREDIT_UTIL'].apply(util_label)

    df['REPAYMENT_STAGE'] = df['REPAYMENT_RATIO'].apply(repayment_stage)

    df['TENURE_STAGE'] = df['TENURE_PROGRESS'].apply(tenure_stage)

    df['PAYMENT_BEHAVIOUR'] = df.apply(payment_behaviour, axis=1)

    df['SANCTION_LABEL'] = df['SANCTION_RATIO'].apply(sanction_label)

    df['INTEREST_RATE_BAND'] = df['INTEREST_RATE'].apply(interest_band)

    df['DISPOSABLE_INCOME_BAND'] = df['DISPOSABLE_INCOME'].apply(disposable_band)

    df['RISK_TIER'] = df.apply(risk_tier, axis=1)

    df['LOAN_HEALTH_SCORE'] = df.apply(loan_health_score, axis=1)

    df['ACCOUNT_HEALTH'] = df['LOAN_HEALTH_SCORE'].apply(account_health)

    df['PRIMARY_RISK'] = df.apply(primary_risk, axis=1)

    return df



df = engineer(df)

df.to_csv("Lending_Loan_Portfolio_1000_Featured.csv", index=False)