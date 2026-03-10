"""
EDA.py — Exploratory Data Analysis on the cleaned Lending Club dataset.

Outputs (saved to visuals/EDA/):
  • target distribution bar chart
  • numeric feature distributions (histograms)
  • bad-rate by grade, sub-grade, home_ownership, purpose
  • correlation heatmap (numeric features)
  • missing-value heatmap
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.helper import DATA_CLEAN, VIS_EDA

CLEAN_PKL = os.path.join(DATA_CLEAN, "lending_club_clean.parquet")

sns.set_theme(style="whitegrid", palette="muted")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    print(f"[EDA] Loading {CLEAN_PKL}")
    return pd.read_parquet(CLEAN_PKL)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def savefig(fig, name: str):
    path = os.path.join(VIS_EDA, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved → {path}")


# ── Individual analyses ───────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame):
    counts = df["target"].value_counts().rename({0: "Good", 1: "Bad"})
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="white")
    ax.set_title("Target Distribution (Good vs Bad)")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + 0.1, p.get_height() * 1.01))
    savefig(fig, "01_target_distribution.png")


def plot_numeric_distributions(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "target"][:20]  # cap at 20

    n_cols = 4
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=40, color="steelblue", edgecolor="white")
        axes[i].set_title(col, fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions", y=1.01)
    fig.tight_layout()
    savefig(fig, "02_numeric_distributions.png")


def plot_badrate_by_cat(df: pd.DataFrame, cat_col: str, filename: str):
    if cat_col not in df.columns:
        return
    br = (df.groupby(cat_col)["target"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "bad_rate", "count": "n"})
            .sort_values("bad_rate", ascending=False))

    fig, ax1 = plt.subplots(figsize=(max(8, len(br) * 0.6), 5))
    ax2 = ax1.twinx()
    ax1.bar(br.index, br["bad_rate"], color="tomato", alpha=0.75, label="Bad Rate")
    ax2.plot(br.index, br["n"], color="steelblue", marker="o", label="Count")
    ax1.set_ylabel("Bad Rate", color="tomato")
    ax2.set_ylabel("Count", color="steelblue")
    ax1.set_title(f"Bad Rate by {cat_col}")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    savefig(fig, filename)


def plot_correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=[np.number]).drop(columns=["target"], errors="ignore")
    corr   = num_df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                annot=False, linewidths=0.3, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    fig.tight_layout()
    savefig(fig, "06_correlation_heatmap.png")


def plot_missing_values(df: pd.DataFrame):
    miss = df.isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        print("[EDA] No missing values found.")
        return
    fig, ax = plt.subplots(figsize=(8, max(4, len(miss) * 0.3)))
    miss.plot(kind="barh", ax=ax, color="coral")
    ax.set_xlabel("Missing Fraction")
    ax.set_title("Missing Value Rates")
    fig.tight_layout()
    savefig(fig, "07_missing_values.png")


def summary_stats(df: pd.DataFrame):
    print("\n[EDA] Dataset shape:", df.shape)
    print(f"[EDA] Bad rate:      {df['target'].mean():.2%}")
    print("\n[EDA] Dtypes overview:")
    print(df.dtypes.value_counts())
    print("\n[EDA] Top 10 numeric stats:")
    print(df.describe().T.head(10))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    summary_stats(df)
    plot_target_distribution(df)
    plot_numeric_distributions(df)
    plot_badrate_by_cat(df, "grade",        "03_badrate_by_grade.png")
    plot_badrate_by_cat(df, "home_ownership","04_badrate_by_home_ownership.png")
    plot_badrate_by_cat(df, "purpose",       "05_badrate_by_purpose.png")
    plot_correlation_heatmap(df)
    plot_missing_values(df)
    print("\n[EDA] All plots saved to visuals/EDA/")


if __name__ == "__main__":
    main()
