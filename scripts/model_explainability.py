import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shap
from sklearn.model_selection import train_test_split
from helper import DATA_CLEAN_PATH, load_model

# ── Paths ─────────────────────────────────────────────────────────────────────
WOE_PKL = os.path.join(DATA_CLEAN_PATH, "woe_encoded.parquet")
VIS_DIR = "visuals/model_evaluation"
os.makedirs(VIS_DIR, exist_ok=True)

# ── Same split as model_training.py ──────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── 1. Load Data and Model ────────────────────────────────────────────────────
def load_data_and_model():
    print("[ME] Loading data and XGBoost model ...")
    df = pd.read_parquet(WOE_PKL)

    X = df.drop(columns=["target"])
    y = df["target"]

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    xgb = load_model("xgboost_challenger")

    print(f"[ME] Test set: {X_test.shape[0]:,} rows x {X_test.shape[1]} features")
    return X_test, y_test, xgb


# ── 2. Compute SHAP Values ────────────────────────────────────────────────────
def compute_shap(xgb, X_test, sample_size=2000):
    print(f"\n[ME] Computing SHAP values on {sample_size} test samples ...")
    print("[ME] This may take a few minutes ...")

    # Sample for speed
    X_sample = X_test.sample(n=sample_size, random_state=RANDOM_STATE)

    explainer   = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_sample)

    print("[ME] SHAP values computed successfully")
    return shap_values, X_sample


# ── 3. Global Feature Importance (SHAP) ───────────────────────────────────────
def plot_shap_importance(shap_values, X_sample):
    print("\n[ME] Plotting global SHAP importance ...")

    mean_abs_shap = pd.DataFrame({
        "feature":   X_sample.columns,
        "importance": np.abs(shap_values).mean(axis=0)
    }).sort_values("importance", ascending=False)

    print("\n[ME] Top 10 features by mean absolute SHAP:")
    print(mean_abs_shap.head(10).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 6))
    top10 = mean_abs_shap.head(10)
    ax.barh(top10["feature"][::-1], top10["importance"][::-1],
            color="steelblue", edgecolor="white")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Global Feature Importance — XGBoost (SHAP)")
    fig.tight_layout()
    path = os.path.join(VIS_DIR, "05_shap_global_importance.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"[ME] Saved → {path}")

    return mean_abs_shap


# ── 4. SHAP Summary Plot (Beeswarm) ───────────────────────────────────────────
def plot_shap_summary(shap_values, X_sample):
    print("\n[ME] Plotting SHAP summary beeswarm ...")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="dot",
        show=False,
        max_display=15
    )
    plt.title("SHAP Summary — XGBoost Challenger")
    plt.tight_layout()
    path = os.path.join(VIS_DIR, "06_shap_summary_beeswarm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ME] Saved → {path}")


# ── 5. SHAP Dependence Plot ────────────────────────────────────────────────────
def plot_shap_dependence(shap_values, X_sample, feature="EXT_SOURCE_2"):
    print(f"\n[ME] Plotting SHAP dependence for {feature} ...")

    if feature not in X_sample.columns:
        print(f"[ME] {feature} not in dataset — skipping dependence plot")
        return

    feat_idx = list(X_sample.columns).index(feature)
    shap_feat = shap_values[:, feat_idx]
    feat_vals  = X_sample[feature].values

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(feat_vals, shap_feat,
                         c=shap_feat, cmap="coolwarm",
                         alpha=0.4, s=8)
    plt.colorbar(scatter, ax=ax, label="SHAP value")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"{feature} (WoE encoded)")
    ax.set_ylabel("SHAP value")
    ax.set_title(f"SHAP Dependence — {feature}")
    fig.tight_layout()
    path = os.path.join(VIS_DIR, f"07_shap_dependence_{feature}.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"[ME] Saved → {path}")


# ── 6. Individual Borrower Explanation ────────────────────────────────────────
def explain_individual(shap_values, X_sample, explainer, n_examples=3):
    print(f"\n[ME] Explaining {n_examples} individual borrowers ...")

    # Pick one high risk, one low risk, one borderline borrower
    proba = explainer.shap_values(X_sample)
    base  = explainer.expected_value

    # Rank by predicted risk
    total_shap = proba.sum(axis=1)
    high_risk_idx  = np.argmax(total_shap)
    low_risk_idx   = np.argmin(total_shap)
    mid_idx        = np.argsort(np.abs(total_shap))[len(total_shap)//2]

    for idx, label in [
        (high_risk_idx, "High Risk Borrower"),
        (low_risk_idx,  "Low Risk Borrower"),
        (mid_idx,       "Borderline Borrower"),
    ]:
        borrower_shap = proba[idx]
        borrower_feat = X_sample.iloc[idx]

        # Top contributing features for this borrower
        contrib = pd.DataFrame({
            "feature":    X_sample.columns,
            "shap_value": borrower_shap,
            "feat_value": borrower_feat.values
        }).reindex(
            pd.Series(borrower_shap).abs().sort_values(ascending=False).index
        ).head(5)

        print(f"\n[ME] {label}")
        print(f"     Base rate    : {base:.4f}")
        print(f"     Total SHAP   : {borrower_shap.sum():+.4f}")
        print(f"     Top 5 drivers:")
        print(contrib.to_string(index=False))

    print("\n[ME] Individual explanations complete")


# ── 7. SHAP vs IV Comparison ──────────────────────────────────────────────────
def compare_shap_vs_iv(mean_abs_shap):
    iv_path = os.path.join(DATA_CLEAN_PATH, "iv_summary.csv")
    if not os.path.exists(iv_path):
        print("[ME] IV summary not found — skipping comparison")
        return

    iv_df = pd.read_csv(iv_path)

    comparison = pd.merge(
        mean_abs_shap[["feature", "importance"]].rename(
            columns={"importance": "shap_importance"}
        ).head(15),
        iv_df[["feature", "IV"]],
        on="feature",
        how="inner"
    )

    # Rank both
    comparison["shap_rank"] = comparison["shap_importance"].rank(ascending=False).astype(int)
    comparison["iv_rank"]   = comparison["IV"].rank(ascending=False).astype(int)
    comparison["rank_diff"] = (comparison["shap_rank"] - comparison["iv_rank"]).abs()

    comparison = comparison.sort_values("shap_rank")

    print("\n[ME] ── SHAP vs IV Comparison ──────────────────────────")
    print(comparison[["feature", "shap_rank", "iv_rank", "rank_diff"]].to_string(index=False))
    print("─────────────────────────────────────────────────────────")

    high_disagreement = comparison[comparison["rank_diff"] > 5]
    if not high_disagreement.empty:
        print(f"\n[ME] Features with large rank disagreement (>5 positions):")
        print(high_disagreement[["feature", "shap_rank", "iv_rank"]].to_string(index=False))
    else:
        print("\n[ME] No major disagreements between SHAP and IV rankings")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load
    X_test, y_test, xgb = load_data_and_model()

    # 2. Compute SHAP
    shap_values, X_sample = compute_shap(xgb, X_test, sample_size=2000)

    # 3. Global importance
    mean_abs_shap = plot_shap_importance(shap_values, X_sample)

    # 4. Beeswarm summary
    plot_shap_summary(shap_values, X_sample)

    # 5. Dependence plots for top 2 features
    plot_shap_dependence(shap_values, X_sample, feature="EXT_SOURCE_2")
    plot_shap_dependence(shap_values, X_sample, feature="EXT_SOURCE_3")

    # 6. Individual explanations
    explainer = shap.TreeExplainer(xgb)
    explain_individual(shap_values, X_sample, explainer, n_examples=3)

    # 7. SHAP vs IV comparison
    compare_shap_vs_iv(mean_abs_shap)

    print("\n[ME] Model explainability complete!")
    print(f"[ME] Plots saved to: {VIS_DIR}/")