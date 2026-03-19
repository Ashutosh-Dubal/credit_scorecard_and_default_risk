import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from helper import DATA_CLEAN_PATH, VISUALS_EDA, data_audit
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
CLEAN_PKL   = os.path.join(DATA_CLEAN_PATH, "application_train_clean.parquet")
os.makedirs(VISUALS_EDA, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")


# ── 1. Target Distribution ────────────────────────────────────────────────────
def plot_target_distribution(df):
    counts = df["target"].value_counts().rename({0: "Non-Defaulter", 1: "Defaulter"})
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="white")
    ax.set_title("Target Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + 0.1, p.get_height() * 1.01))
    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS_EDA, "01_target_distribution.png"), dpi=150)
    plt.close()
    print("[EDA] Saved → 01_target_distribution.png")


# ── 2. Train/Test Split (shared across both models) ───────────────────────────
def prepare_data(df, target_col="target"):
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X, X_train, X_test, y_train, y_test


# ── 3. Random Forest + Permutation Importance ─────────────────────────────────
def get_rf_importance(X, X_train, X_test, y_train, y_test, top_n=10):
    print("\n[EDA] Running Random Forest + Permutation Importance ...")
    print("[EDA] This may take a few minutes ...")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    result = permutation_importance(
        rf, X_test, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    rf_df = pd.DataFrame({
        "feature":    X.columns,
        "importance": result.importances_mean,
        "std":        result.importances_std
    }).sort_values("importance", ascending=False).head(top_n)

    print(f"\n[EDA] RF Top {top_n} Features:")
    print(rf_df.to_string(index=False))
    return rf_df


# ── 4. XGBoost Feature Importance ─────────────────────────────────────────────
def get_xgb_importance(X, X_train, X_test, y_train, y_test, top_n=10):
    print("\n[EDA] Running XGBoost Feature Importance ...")

    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric="auc"
    )
    xgb.fit(X_train, y_train)

    xgb_df = pd.DataFrame({
        "feature":    X.columns,
        "importance": xgb.feature_importances_
    }).sort_values("importance", ascending=False).head(top_n)

    print(f"\n[EDA] XGBoost Top {top_n} Features:")
    print(xgb_df.to_string(index=False))
    return xgb_df


# ── 5. Compare RF vs XGBoost ──────────────────────────────────────────────────
def compare_models(rf_df, xgb_df, df, top_n=10):
    rf_top  = set(rf_df["feature"].tolist())
    xgb_top = set(xgb_df["feature"].tolist())

    agreed   = rf_top & xgb_top
    rf_only  = rf_top - xgb_top
    xgb_only = xgb_top - rf_top

    print("\n[EDA] ── Feature Agreement ──────────────────────────")
    print(f"  Agreed by both models : {agreed}")
    print(f"  RF only               : {rf_only}")
    print(f"  XGBoost only          : {xgb_only}")
    print(f"  Agreement rate        : {len(agreed)}/10 features")
    print("─────────────────────────────────────────────────────")

    # Assign ranks (1 = most important)
    rf_df = rf_df.copy()
    xgb_df = xgb_df.copy()

    rf_df["rank"] = range(1, len(rf_df) + 1)
    xgb_df["rank"] = range(1, len(xgb_df) + 1)

    # Merge on feature name
    comparison = pd.merge(
        rf_df[["feature", "importance", "rank"]].rename(
            columns={"importance": "rf_importance", "rank": "rf_rank"}
        ),
        xgb_df[["feature", "importance", "rank"]].rename(
            columns={"importance": "xgb_importance", "rank": "xgb_rank"}
        ),
        on="feature",
        how="outer"
    )

    # Fill NaN ranks with worst rank (top_n + 1) for features missing from one model
    comparison["rf_rank"] = comparison["rf_rank"].fillna(top_n + 1)
    comparison["xgb_rank"] = comparison["xgb_rank"].fillna(top_n + 1)

    # Average rank — lower is better
    comparison["avg_rank"] = (
            (comparison["rf_rank"] + comparison["xgb_rank"]) / 2
    ).round(1)

    # Consensus flag
    comparison["in_both"] = (comparison["rf_rank"] <= top_n) & (comparison["xgb_rank"] <= top_n)

    comparison = comparison.sort_values("avg_rank")

    print("\n[EDA] ── Rank-Based Feature Comparison ─────────────────")
    print(comparison.to_string(index=False))
    print("─────────────────────────────────────────────────────────")
    print(f"\n  Features agreed by both : {comparison['in_both'].sum()}")
    print(f"  Top 10 by avg rank       : {comparison['feature'].head(10).tolist()}")

    return agreed, rf_only, xgb_only, comparison


# ── 6. Plot RF vs XGBoost Side by Side ───────────────────────────────────────
def plot_importance_comparison(rf_df, xgb_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # RF plot
    axes[0].barh(
        rf_df["feature"][::-1],
        rf_df["importance"][::-1],
        xerr=rf_df["std"][::-1],
        color="steelblue", edgecolor="white"
    )
    axes[0].set_title("Random Forest — Permutation Importance")
    axes[0].set_xlabel("Mean Decrease in AUC when Shuffled")

    # XGBoost plot
    axes[1].barh(
        xgb_df["feature"][::-1],
        xgb_df["importance"][::-1],
        color="tomato", edgecolor="white"
    )
    axes[1].set_title("XGBoost — Feature Importance")
    axes[1].set_xlabel("Importance Score")

    fig.suptitle("RF vs XGBoost — Top 10 Feature Importance Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS_EDA, "02_rf_vs_xgb_importance.png"), dpi=150)
    plt.close()
    print("[EDA] Saved → 02_rf_vs_xgb_importance.png")


# ── 7. Plot Agreed Features Distribution by Target ───────────────────────────
def plot_agreed_distributions(df, agreed_features):
    agreed_features = list(agreed_features)
    n     = len(agreed_features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
    axes = axes.flatten()

    for i, feature in enumerate(agreed_features):
        for label, color in [(0, "steelblue"), (1, "tomato")]:
            axes[i].hist(
                df[df["target"] == label][feature].dropna(),
                bins=40, alpha=0.6, color=color,
                label="Non-Defaulter" if label == 0 else "Defaulter",
                density=True
            )
        axes[i].set_title(feature, fontsize=9)
        axes[i].set_ylabel("Density")
        if i == 0:
            axes[i].legend(fontsize=8)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Agreed Features — Distribution by Target", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS_EDA, "03_agreed_features_distribution.png"), dpi=150)
    plt.close()
    print("[EDA] Saved → 03_agreed_features_distribution.png")


# ── 8. Correlation Heatmap on Agreed Features ─────────────────────────────────
def plot_correlation_heatmap(df, agreed_features):
    cols = list(agreed_features) + ["target"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        linewidths=0.3, ax=ax
    )
    ax.set_title("Correlation Heatmap — Agreed Features + Target")
    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS_EDA, "04_correlation_heatmap.png"), dpi=150)
    plt.close()
    print("[EDA] Saved → 04_correlation_heatmap.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[EDA] Loading clean dataset ...")
    df = pd.read_parquet(CLEAN_PKL)

    # Audit
    data_audit(df)

    # Target distribution
    plot_target_distribution(df)

    # Prepare data — shared split for both models
    X, X_train, X_test, y_train, y_test = prepare_data(df)

    # Feature importance — both models
    rf_df  = get_rf_importance(X, X_train, X_test, y_train, y_test, top_n=10)
    xgb_df = get_xgb_importance(X, X_train, X_test, y_train, y_test, top_n=10)

    # Compare
    agreed, rf_only, xgb_only, comparison = compare_models(rf_df, xgb_df, df)

    # Plots
    plot_importance_comparison(rf_df, xgb_df)
    plot_agreed_distributions(df, agreed)
    plot_correlation_heatmap(df, agreed)

    print("\n[EDA] Complete! All plots saved to visuals/EDA/")