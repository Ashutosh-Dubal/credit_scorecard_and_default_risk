"""
helper.py — Shared utilities for metrics, plotting, and I/O.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW   = os.path.join(ROOT, "data", "raw")
DATA_CLEAN = os.path.join(ROOT, "data", "clean")
MODELS_DIR = os.path.join(ROOT, "models")
VIS_EDA    = os.path.join(ROOT, "visuals", "EDA")
VIS_SCORE  = os.path.join(ROOT, "visuals", "score_distribution")
VIS_EVAL   = os.path.join(ROOT, "visuals", "model_evaluation")

for _dir in [DATA_RAW, DATA_CLEAN, MODELS_DIR, VIS_EDA, VIS_SCORE, VIS_EVAL]:
    os.makedirs(_dir, exist_ok=True)


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_model(model, name: str) -> str:
    """Serialise a model to /models/<name>.joblib and return the path."""
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"[saved] {path}")
    return path


def load_model(name: str):
    """Load a serialised model from /models/<name>.joblib."""
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    return joblib.load(path)


def save_csv(df: pd.DataFrame, subdir: str, filename: str) -> str:
    """Save a DataFrame to data/<subdir>/<filename>.csv."""
    base = DATA_RAW if subdir == "raw" else DATA_CLEAN
    path = os.path.join(base, filename)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")
    return path


# ── Classification Metrics ────────────────────────────────────────────────────

def gini(y_true, y_prob) -> float:
    """Gini coefficient = 2 * AUC - 1."""
    return 2 * roc_auc_score(y_true, y_prob) - 1


def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov–Smirnov statistic (max separation of cumulative distributions)."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def print_metrics(y_true, y_prob, label: str = "Model") -> dict:
    auc   = roc_auc_score(y_true, y_prob)
    gini_ = gini(y_true, y_prob)
    ks    = ks_statistic(y_true, y_prob)
    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"  AUC  : {auc:.4f}")
    print(f"  Gini : {gini_:.4f}")
    print(f"  KS   : {ks:.4f}")
    print(f"{'─'*40}")
    return {"label": label, "AUC": auc, "Gini": gini_, "KS": ks}


# ── Plotting Helpers ───────────────────────────────────────────────────────────

def plot_roc(y_true, y_prob, label: str = "Model", save_path: str = None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[saved] {save_path}")
    return fig


def plot_ks(y_true, y_prob, label: str = "Model", save_path: str = None):
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
    n  = len(df)
    df["cum_bad"]  = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    df["cum_good"] = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    df["ks"] = df["cum_bad"] - df["cum_good"]
    idx_max  = df["ks"].idxmax()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["cum_bad"].values,  label="Cumulative Bad",  lw=2)
    ax.plot(df["cum_good"].values, label="Cumulative Good", lw=2)
    ax.axvline(df.index.get_loc(idx_max), color="red", linestyle="--",
               label=f"KS = {df['ks'].max():.3f}")
    ax.set_xlabel("Population (sorted by score desc)")
    ax.set_ylabel("Cumulative proportion")
    ax.set_title(f"KS Plot — {label}")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[saved] {save_path}")
    return fig


def plot_score_distribution(scores, y_true, save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, mask, color in [(0, y_true == 0, "steelblue"), (1, y_true == 1, "tomato")]:
        ax.hist(scores[mask], bins=40, alpha=0.6, color=color,
                label="Good (0)" if label == 0 else "Bad (1)", density=True)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Class")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[saved] {save_path}")
    return fig


# ── Population Stability Index ─────────────────────────────────────────────────

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
      PSI < 0.10  → no significant shift
      PSI 0.10–0.25 → moderate shift (monitor)
      PSI > 0.25  → major shift (investigate / retrain)
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints  = np.unique(breakpoints)

    def _bucket(arr):
        counts = np.histogram(arr, bins=breakpoints)[0]
        pct    = counts / len(arr)
        pct    = np.where(pct == 0, 1e-6, pct)
        return pct

    exp_pct = _bucket(expected)
    act_pct = _bucket(actual)
    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi_val)
