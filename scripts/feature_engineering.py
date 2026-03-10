"""
feature_engineering.py — Weight of Evidence (WoE) / Information Value (IV) pipeline.

Uses `optbinning` for optimal binning of continuous and categorical features.
Features with IV < IV_MIN are dropped as non-predictive.

Outputs:
  • data/clean/woe_encoded.parquet   — WoE-transformed feature matrix + target
  • data/clean/iv_summary.csv        — IV for every feature (sorted desc)
  • models/binning_process.joblib    — fitted BinningProcess (for scoring)
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.helper import DATA_CLEAN, MODELS_DIR, save_model

try:
    from optbinning import BinningProcess
    OPTBIN_OK = True
except ImportError:
    OPTBIN_OK = False
    print("[fe] WARNING: 'optbinning' not installed. Run: pip install optbinning")

CLEAN_PKL = os.path.join(DATA_CLEAN, "lending_club_clean.parquet")
WOE_PKL   = os.path.join(DATA_CLEAN, "woe_encoded.parquet")
IV_CSV    = os.path.join(DATA_CLEAN, "iv_summary.csv")

IV_MIN    = 0.02   # drop features with IV below this threshold
MAX_BINS  = 10     # max bins per continuous feature


def load_clean() -> pd.DataFrame:
    print(f"[fe] Loading {CLEAN_PKL}")
    return pd.read_parquet(CLEAN_PKL)


def split_feature_types(df: pd.DataFrame, target_col: str = "target"):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    return num_cols, cat_cols


def build_binning_process(num_cols, cat_cols, max_n_bins: int = MAX_BINS):
    """Return a configured (unfitted) optbinning BinningProcess."""
    all_cols = num_cols + cat_cols
    special_codes = {"Missing": -9999}

    variable_dtypes = {}
    for c in num_cols:
        variable_dtypes[c] = "numerical"
    for c in cat_cols:
        variable_dtypes[c] = "categorical"

    bp = BinningProcess(
        variable_names=all_cols,
        max_n_bins=max_n_bins,
        special_codes=special_codes,
    )
    return bp, all_cols


def compute_iv_summary(bp: BinningProcess, feature_names: list) -> pd.DataFrame:
    """Extract IV for each feature from a fitted BinningProcess."""
    rows = []
    for name in feature_names:
        try:
            binning_table = bp.get_binned_variable(name).binning_table
            iv = binning_table.build()["IV"].sum()
        except Exception:
            iv = np.nan
        rows.append({"feature": name, "IV": iv})
    df_iv = pd.DataFrame(rows).sort_values("IV", ascending=False).reset_index(drop=True)
    return df_iv


def filter_by_iv(df_iv: pd.DataFrame, min_iv: float = IV_MIN) -> list:
    selected = df_iv.loc[df_iv["IV"] >= min_iv, "feature"].tolist()
    dropped  = df_iv.loc[df_iv["IV"] < min_iv, "feature"].tolist()
    print(f"[fe] IV filter (>= {min_iv}): keeping {len(selected)}, dropping {len(dropped)}")
    return selected


def main():
    if not OPTBIN_OK:
        print("[fe] Cannot proceed without optbinning. Exiting.")
        return

    df = load_clean()
    y  = df["target"].values
    num_cols, cat_cols = split_feature_types(df)

    print(f"[fe] Numeric features : {len(num_cols)}")
    print(f"[fe] Categorical feats: {len(cat_cols)}")

    bp, all_cols = build_binning_process(num_cols, cat_cols)
    X = df[all_cols]

    print("[fe] Fitting BinningProcess …")
    bp.fit(X, y)

    # IV summary
    df_iv = compute_iv_summary(bp, all_cols)
    df_iv.to_csv(IV_CSV, index=False)
    print(f"[fe] IV summary saved → {IV_CSV}")
    print(df_iv.head(20).to_string(index=False))

    selected_features = filter_by_iv(df_iv, IV_MIN)

    # Transform to WoE
    print("[fe] Transforming to WoE …")
    X_woe = bp.transform(X[selected_features], metric="woe")
    X_woe.columns = selected_features
    X_woe["target"] = y

    X_woe.to_parquet(WOE_PKL, index=False)
    print(f"[fe] WoE dataset saved → {WOE_PKL}  shape={X_woe.shape}")

    save_model(bp, "binning_process")
    print("[fe] Feature engineering complete.")


if __name__ == "__main__":
    main()
