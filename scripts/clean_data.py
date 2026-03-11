import pandas as pd
import numpy as np
import os
from helper import (
    DATA_RAW_APP_TRAIN,
    DATA_CLEAN_PATH,
    data_audit
)

# Load
df = pd.read_csv(DATA_RAW_APP_TRAIN)

# Checking for missing or inconsistencies in dataset
data_audit(df)

DROP_COLS = [
    "SK_ID_CURR",
    "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL",
    "FLAG_DOCUMENT_2",  "FLAG_DOCUMENT_3",  "FLAG_DOCUMENT_4",
    "FLAG_DOCUMENT_5",  "FLAG_DOCUMENT_6",  "FLAG_DOCUMENT_7",
    "FLAG_DOCUMENT_8",  "FLAG_DOCUMENT_9",  "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16",
    "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19",
    "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21",
    "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",     "LIVE_CITY_NOT_WORK_CITY",
]

def drop_high_null(df, threshold=0.50):
    to_drop = [c for c in df.columns if df[c].isnull().mean() > threshold and c != "TARGET"]
    df.drop(columns=to_drop, inplace=True)
    print(f"[clean] Dropped {len(to_drop)} columns with >{threshold:.0%} missingness")
    return df

def drop_cols(df):
    to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=to_drop, inplace=True)
    print(f"[clean] Dropped {len(to_drop)} low-signal / ID columns")
    return df

def fix_anomalies(df):
    # DAYS_EMPLOYED: 365243 is a placeholder for unemployed/pensioners
    if "DAYS_EMPLOYED" in df.columns:
        n = (df["DAYS_EMPLOYED"] == 365243).sum()
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        print(f"[clean] DAYS_EMPLOYED: replaced {n:,} anomaly values with NaN")
    # CODE_GENDER: XNA is essentially missing
    if "CODE_GENDER" in df.columns:
        df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)
    return df

def encode_binary_flags(df):
    yn_cols = [c for c in df.columns if df[c].dropna().isin(["Y", "N"]).all()]
    for col in yn_cols:
        df[col] = df[col].map({"Y": 1, "N": 0})
    print(f"[clean] Encoded {len(yn_cols)} Y/N columns to 1/0")
    return df

def engineer_features(df):
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)
    if "DAYS_EMPLOYED" in df.columns:
        df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"] / 365.25).round(1)
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["CREDIT_TO_INCOME"] = (df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)).round(4)
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["ANNUITY_TO_INCOME"] = (df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)).round(4)
    if "AMT_CREDIT" in df.columns and "AMT_GOODS_PRICE" in df.columns:
        df["CREDIT_TO_GOODS"] = (df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)).round(4)
    if "AMT_INCOME_TOTAL" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
        df["INCOME_PER_PERSON"] = (df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)).round(2)
    if "YEARS_EMPLOYED" in df.columns and "AGE_YEARS" in df.columns:
        df["EMPLOYED_TO_AGE"] = (df["YEARS_EMPLOYED"] / df["AGE_YEARS"].replace(0, np.nan)).round(4)
    print("[clean] Engineered 7 derived features")
    return df

def winsorise(df, lower=0.01, upper=0.99):
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "TARGET"]
    for col in num_cols:
        lo, hi = df[col].quantile([lower, upper])
        df[col] = df[col].clip(lo, hi)
    return df

def impute(df):
    # AMT_REQ_CREDIT_BUREAU_* — missing means no bureau queries, not unknown
    bureau_cols = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    df[bureau_cols] = df[bureau_cols].fillna(0)

    # OWN_CAR_AGE — missing means no car
    if "OWN_CAR_AGE" in df.columns:
        df["OWN_CAR_AGE"] = df["OWN_CAR_AGE"].fillna(0)

    # Remaining numeric — median, categorical — 'Unknown'
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != "TARGET" and c not in bureau_cols and c != "OWN_CAR_AGE"]
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    print(f"[clean] Bureau query cols → 0 : {len(bureau_cols)}")
    print(f"[clean] Numeric cols → median  : {len(num_cols)}")
    print(f"[clean] Categorical cols → Unknown : {len(cat_cols)}")
    return df

def save_clean(df):
    os.makedirs(DATA_CLEAN_PATH, exist_ok=True)
    pkl_path = os.path.join(DATA_CLEAN_PATH, "application_train_clean.parquet")
    csv_path = os.path.join(DATA_CLEAN_PATH, "application_train_clean.csv")
    df.to_parquet(pkl_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"[clean] Saved → {pkl_path}")
    print(f"[clean] Saved → {csv_path}")

# Clean
df = drop_high_null(df)
df = drop_cols(df)
df = fix_anomalies(df)
df = encode_binary_flags(df)
df = engineer_features(df)
df = winsorise(df)
df = impute(df)

# Rename target for pipeline consistency
df.rename(columns={"TARGET": "target"}, inplace=True)

# Audit after cleaning
print("\n── Post-clean audit ──")
data_audit(df)

# Save
save_clean(df)