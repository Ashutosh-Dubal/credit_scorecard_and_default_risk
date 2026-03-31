import os
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW_APP_TRAIN = "data/raw/application_train.csv"
DATA_CLEAN_PATH = "data/clean"
VISUALS_EDA = "visuals/EDA"

# ── Audit ─────────────────────────────────────────────────────────────────────
def data_audit(df):
    print("\nData Shape:", df.shape)
    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nNumeric Summary:")
    print(df.describe())

    print("\nDuplicates:", df.duplicated().sum())

    for col in df.select_dtypes(include='object'):
        print(f"\nUnique values in {col}:", df[col].dropna().unique())

