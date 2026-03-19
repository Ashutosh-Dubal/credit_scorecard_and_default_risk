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
def compare_models(rf_df, xgb_df, top_n=10):
    rf_df  = rf_df.copy()
    xgb_df = xgb_df.copy()

    rf_df["rf_rank"]   = range(1, len(rf_df)  + 1)
    xgb_df["xgb_rank"] = range(1, len(xgb_df) + 1)

    comparison = pd.merge(
        rf_df[["feature", "importance", "rf_rank"]].rename(
            columns={"importance": "rf_importance"}),
        xgb_df[["feature", "importance", "xgb_rank"]].rename(
            columns={"importance": "xgb_importance"}),
        on="feature", how="outer"
    )

    comparison["rf_rank"]  = comparison["rf_rank"].fillna(top_n + 1)
    comparison["xgb_rank"] = comparison["xgb_rank"].fillna(top_n + 1)
    comparison["avg_rank"] = (
        (comparison["rf_rank"] + comparison["xgb_rank"]) / 2
    ).round(1)
    comparison["in_both"] = (
        (comparison["rf_rank"]  <= top_n) &
        (comparison["xgb_rank"] <= top_n)
    )
    comparison = comparison.sort_values("avg_rank")

    # ── Only change: top 10 by avg rank ──
    top_10_features = comparison["feature"].head(top_n).tolist()

    print("\n[EDA] ── Rank-Based Top 10 ───────────────────────────────")
    print(comparison.head(top_n)[["feature", "rf_rank", "xgb_rank", "avg_rank", "in_both"]].to_string(index=False))
    print("─────────────────────────────────────────────────────────")

    return comparison, top_10_features


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


# ── 7. Box Plots by Target ────────────────────────────────────────────────────
def plot_boxplots(df, all_top_features):
    n     = len(all_top_features)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4))
    axes = axes.flatten()

    for i, feature in enumerate(all_top_features):
        df.boxplot(column=feature, by="target", ax=axes[i])
        axes[i].set_title(feature, fontsize=9)
        axes[i].set_xlabel("0 = Non-Defaulter   1 = Defaulter")
        axes[i].set_ylabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distribution by Target — Box Plots", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(VISUALS_EDA, "03_boxplots_by_target.png"), dpi=150)
    plt.close()
    print("[EDA] Saved → 03_boxplots_by_target.png")

# ── 8. Bad Rate by Decile ─────────────────────────────────────────────────────
def plot_bad_rate_by_decile(df, all_top_features):
    n     = len(all_top_features)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4))
    axes = axes.flatten()

    for i, feature in enumerate(all_top_features):
        temp = df[[feature, "target"]].copy()
        temp["decile"] = pd.qcut(temp[feature], q=10, duplicates="drop")

        bad_rate = temp.groupby("decile", observed=True)["target"].mean()
        count    = temp.groupby("decile", observed=True)["target"].count()

        ax1 = axes[i]
        ax2 = ax1.twinx()

        ax1.bar(range(len(bad_rate)), bad_rate.values,
                color="tomato", alpha=0.7)
        ax2.plot(range(len(count)), count.values,
                 color="steelblue", marker="o", linewidth=1.5)

        ax1.set_title(feature, fontsize=9)
        ax1.set_ylabel("Bad rate", color="tomato", fontsize=8)
        ax2.set_ylabel("Count",    color="steelblue", fontsize=8)
        ax1.set_xticks(range(len(bad_rate)))
        ax1.set_xticklabels(
            [f"D{j+1}" for j in range(len(bad_rate))],
            fontsize=7
        )

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Bad Rate by Decile — Top Features", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(VISUALS_EDA, "04_bad_rate_by_decile.png"), dpi=150)
    plt.close()
    print("[EDA] Saved → 04_bad_rate_by_decile.png")

# ── 9. Mean Feature Value by Target ──────────────────────────────────────────
def plot_mean_by_target(df, all_top_features):
    means = df.groupby("target")[all_top_features].mean().T
    means.columns = ["Non-Defaulter", "Defaulter"]

    # Normalise so features on different scales are comparable
    means_norm = means.div(means.abs().max(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(all_top_features))
    w = 0.35

    ax.bar(x - w/2, means_norm["Non-Defaulter"],
           w, label="Non-Defaulter", color="steelblue", alpha=0.85)
    ax.bar(x + w/2, means_norm["Defaulter"],
           w, label="Defaulter",     color="tomato",    alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_top_features, rotation=30, ha="right", fontsize=9)
    ax.set_title("Normalised Mean Feature Value — Defaulter vs Non-Defaulter")
    ax.set_ylabel("Normalised mean (relative scale)")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS_EDA, "05_mean_by_target.png"), dpi=150)
    plt.close()
    print("[EDA] Saved → 05_mean_by_target.png")

# ── 10. Cumulative Bad Rate Curve ─────────────────────────────────────────────
def plot_cumulative_bad_rate(df, all_top_features):
    n     = len(all_top_features)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4))
    axes = axes.flatten()

    total_bads = (df["target"] == 1).sum()

    for i, feature in enumerate(all_top_features):
        sorted_df = df[[feature, "target"]].sort_values(feature)
        cum_bad  = (sorted_df["target"] == 1).cumsum() / total_bads
        cum_pop  = np.arange(1, len(sorted_df) + 1) / len(sorted_df)

        axes[i].plot(cum_pop, cum_bad.values, color="tomato",    lw=2, label="Feature")
        axes[i].plot([0, 1],  [0, 1],         color="steelblue", lw=1,
                     linestyle="--", label="Random")
        axes[i].fill_between(cum_pop, cum_bad.values, cum_pop,
                             alpha=0.1, color="tomato")
        axes[i].set_title(feature, fontsize=9)
        axes[i].set_xlabel("% Population", fontsize=8)
        axes[i].set_ylabel("% Defaults Captured", fontsize=8)
        if i == 0:
            axes[i].legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Cumulative Default Capture — Top Features", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(VISUALS_EDA, "06_cumulative_bad_rate.png"), dpi=150)
    plt.close()
    print("[EDA] Saved → 06_cumulative_bad_rate.png")

# ── 11. Correlation Heatmap ─────────────────────────────────
def plot_correlation_heatmap(df, all_top_features):
    cols = all_top_features + ["target"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        linewidths=0.3, ax=ax
    )
    ax.set_title("Correlation Heatmap — Top Features + Target")
    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS_EDA, "07_correlation_heatmap.png"), dpi=150)
    plt.close()
    print("[EDA] Saved → 07_correlation_heatmap.png")


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
    agreed, features = compare_models(rf_df, xgb_df)

    # Plots
    plot_importance_comparison(rf_df, xgb_df)
    plot_boxplots(df, features)
    plot_bad_rate_by_decile(df, features)
    plot_mean_by_target(df, features)
    plot_cumulative_bad_rate(df, features)
    plot_correlation_heatmap(df, features)

    print("\n[EDA] Complete! All plots saved to visuals/EDA/")