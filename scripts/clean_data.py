"""
clean_data.py — Load raw Lending Club CSV, clean, and save to data/clean/.

Steps:
  1. Filter to fully-paid vs charged-off / defaulted loans
  2. Drop leakage & admin columns
  3. Cast dtypes, parse dates
  4. Cap numeric outliers (Winsorisation at 1st/99th percentile)
  5. Impute missing values
  6. Encode binary target
  7. Save clean parquet + csv
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.helper import DATA_RAW, DATA_CLEAN, save_csv

RAW_CSV   = os.path.join(DATA_RAW,   "lending_club.csv")
CLEAN_CSV = os.path.join(DATA_CLEAN, "lending_club_clean.csv")
CLEAN_PKL = os.path.join(DATA_CLEAN, "lending_club_clean.parquet")

# Target mapping — 1 = default/bad, 0 = good
TARGET_MAP = {
    "Fully Paid":               0,
    "Charged Off":              1,
    "Default":                  1,
    "Does not meet the credit policy. Status:Fully Paid":  0,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

# Columns with direct data leakage (post-origination outcomes)
LEAKAGE_COLS = [
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "out_prncp", "out_prncp_inv",
]

# Columns to drop (ids, free text, redundant)
DROP_COLS = [
    "id", "member_id", "url", "desc", "title",
    "zip_code",                # redundant with addr_state
    "policy_code",             # constant
    "pymnt_plan",              # constant
    "application_type",        # mostly individual
    "hardship_flag",
    "debt_settlement_flag",
]


def load_raw(path: str, nrows: int = None) -> pd.DataFrame:
    print(f"[clean] Reading raw CSV: {path}")
    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    print(f"[clean] Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def filter_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["loan_status"].isin(TARGET_MAP)].copy()
    df["target"] = df["loan_status"].map(TARGET_MAP)
    df.drop(columns=["loan_status"], inplace=True)
    print(f"[clean] After target filter: {len(df):,} rows  "
          f"(bad rate = {df['target'].mean():.2%})")
    return df


def drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in LEAKAGE_COLS + DROP_COLS if c in df.columns]
    df.drop(columns=to_drop, inplace=True)
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["issue_d", "earliest_cr_line", "last_credit_pull_d"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%b-%Y", errors="coerce")
    if "issue_d" in df.columns and "earliest_cr_line" in df.columns:
        df["credit_age_months"] = (
            (df["issue_d"] - df["earliest_cr_line"]) / np.timedelta64(1, "M")
        ).round(0)
    df.drop(columns=["issue_d", "earliest_cr_line", "last_credit_pull_d"],
            inplace=True, errors="ignore")
    return df


def parse_pct_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["int_rate", "revol_util"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("%", "").str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_term(df: pd.DataFrame) -> pd.DataFrame:
    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
    return df


def parse_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    if "emp_length" in df.columns:
        mapping = {
            "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
            "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
            "8 years": 8, "9 years": 9, "10+ years": 10,
        }
        df["emp_length"] = df["emp_length"].map(mapping)
    return df


def winsorise(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "target"]
    for col in num_cols:
        lo, hi = df[col].quantile([lower, upper])
        df[col] = df[col].clip(lo, hi)
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    return df


def drop_high_null(df: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    null_frac = df.isnull().mean()
    to_drop   = null_frac[null_frac > threshold].index.tolist()
    df.drop(columns=to_drop, inplace=True)
    print(f"[clean] Dropped {len(to_drop)} columns with >{threshold:.0%} missing")
    return df


def main(nrows: int = None):
    df = load_raw(RAW_CSV, nrows=nrows)
    df = filter_target(df)
    df = drop_cols(df)
    df = drop_high_null(df, threshold=0.50)
    df = parse_dates(df)
    df = parse_pct_cols(df)
    df = parse_term(df)
    df = parse_emp_length(df)
    df = winsorise(df)
    df = impute(df)

    print(f"[clean] Final shape: {df.shape}")

    df.to_parquet(CLEAN_PKL, index=False)
    print(f"[clean] Saved parquet → {CLEAN_PKL}")

    df.to_csv(CLEAN_CSV, index=False)
    print(f"[clean] Saved CSV    → {CLEAN_CSV}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", type=int, default=None,
                        help="Limit rows for quick testing")
    args = parser.parse_args()
    main(nrows=args.nrows)
