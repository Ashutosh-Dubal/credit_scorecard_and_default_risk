import pandas as pd
import numpy as np
import os
from optbinning import BinningProcess
from helper import DATA_CLEAN_PATH, save_model

# ── Paths ─────────────────────────────────────────────────────────────────────
CLEAN_PKL   = os.path.join(DATA_CLEAN_PATH, "application_train_clean.parquet")
WOE_PKL     = os.path.join(DATA_CLEAN_PATH, "woe_encoded.parquet")
WOE_CSV     = os.path.join(DATA_CLEAN_PATH, "woe_encoded.csv")
IV_CSV      = os.path.join(DATA_CLEAN_PATH, "iv_summary.csv")

# IV threshold — features below this are dropped
IV_MIN      = 0.02
# Max bins per continuous feature
MAX_BINS    = 10


# ── 1. Load Clean Data ────────────────────────────────────────────────────────
def load_clean():
    print("[FE] Loading clean dataset ...")
    df = pd.read_parquet(CLEAN_PKL)
    print(f"[FE] Shape: {df.shape}")
    print(f"[FE] Bad rate: {df['target'].mean():.2%}")
    return df


# ── 2. Split Feature Types ─────────────────────────────────────────────────────
def split_feature_types(df, target_col="target"):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Remove target from numeric list
    num_cols = [c for c in num_cols if c != target_col]

    print(f"[FE] Numeric features   : {len(num_cols)}")
    print(f"[FE] Categorical features: {len(cat_cols)}")

    return num_cols, cat_cols


# ── 3. Build & Fit Binning Process ────────────────────────────────────────────
def fit_binning_process(df, num_cols, cat_cols, target_col="target"):
    print("\n[FE] Fitting BinningProcess ...")
    print("[FE] This may take a few minutes on 300K rows ...")

    all_cols = num_cols + cat_cols
    X = df[all_cols]
    y = df[target_col].values

    # Tell optbinning which columns are categorical
    categorical_variables = cat_cols if cat_cols else None

    bp = BinningProcess(
        variable_names=all_cols,
        categorical_variables=categorical_variables,
        max_n_bins=MAX_BINS,
        min_bin_size=0.05       # each bin must have at least 5% of data
    )

    bp.fit(X, y)
    print("[FE] BinningProcess fitted successfully")

    # Save the fitted process — needed later for scoring new applicants
    save_model(bp, "binning_process")

    return bp, all_cols


# ── 4. Extract IV Summary ──────────────────────────────────────────────────────
def extract_iv_summary(bp, all_cols):
    print("\n[FE] Extracting IV summary ...")

    rows = []
    for col in all_cols:
        try:
            summary = bp.get_binned_variable(col).binning_table.build()
            iv      = summary["IV"].sum()
        except Exception:
            iv = np.nan
        rows.append({"feature": col, "IV": round(iv, 4)})

    iv_df = (pd.DataFrame(rows)
               .sort_values("IV", ascending=False)
               .reset_index(drop=True))

    # Add predictive power label
    iv_df["predictive_power"] = iv_df["IV"].apply(label_iv)

    print(f"\n[FE] IV Summary (top 20):")
    print(iv_df.head(20).to_string(index=False))

    # Save for reference
    iv_df.to_csv(IV_CSV, index=False)
    print(f"\n[FE] IV summary saved → {IV_CSV}")

    return iv_df


def label_iv(iv):
    if iv < 0.02:  return "Useless — drop"
    if iv < 0.10:  return "Weak"
    if iv < 0.30:  return "Medium"
    if iv < 0.50:  return "Strong"
    return "Suspicious — check for leakage"


# ── 5. Filter Features by IV ──────────────────────────────────────────────────
def filter_by_iv(iv_df, threshold=IV_MIN):
    kept    = iv_df[iv_df["IV"] >= threshold]["feature"].tolist()
    dropped = iv_df[iv_df["IV"] <  threshold]["feature"].tolist()

    print(f"\n[FE] IV filter (threshold = {threshold})")
    print(f"[FE] Features kept   : {len(kept)}")
    print(f"[FE] Features dropped: {len(dropped)}")

    if dropped:
        print(f"[FE] Dropped features: {dropped}")

    return kept


# ── 6. WoE Transform ──────────────────────────────────────────────────────────
def woe_transform(bp, df, all_cols, kept_features, target_col="target"):
    print("\n[FE] Applying WoE transformation ...")

    # Transform must receive all columns bp was fitted on
    X = df[all_cols]
    X_woe_all = bp.transform(X, metric="woe", show_digits=4)
    X_woe_all.columns = all_cols

    # Select only the IV-filtered features
    X_woe = X_woe_all[kept_features].copy()

    # Add target back
    X_woe[target_col] = df[target_col].values

    print(f"[FE] WoE dataset shape: {X_woe.shape}")
    print(f"[FE] Bad rate preserved: {X_woe[target_col].mean():.2%}")

    return X_woe


# ── 7. Save WoE Dataset ───────────────────────────────────────────────────────
def save_woe_dataset(df_woe):
    df_woe.to_parquet(WOE_PKL, index=False)
    df_woe.to_csv(WOE_CSV, index=False)
    print(f"\n[FE] Saved parquet → {WOE_PKL}")
    print(f"[FE] Saved CSV     → {WOE_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load
    df = load_clean()

    # 2. Split feature types
    num_cols, cat_cols = split_feature_types(df)

    # 3. Fit binning process
    bp, all_cols = fit_binning_process(df, num_cols, cat_cols)

    # 4. Extract IV for all features
    iv_df = extract_iv_summary(bp, all_cols)

    # 5. Filter by IV
    kept_features = filter_by_iv(iv_df, threshold=IV_MIN)

    # 6. WoE transform on kept features only
    df_woe = woe_transform(bp, df, all_cols, kept_features)

    # 7. Save
    save_woe_dataset(df_woe)

    print("\n[FE] Feature engineering complete!")
    print(f"[FE] Final feature count : {len(kept_features)}")
    print(f"[FE] Output saved to     : {WOE_PKL}")