"""
champion_challenger.py — A/B comparison of the LR Scorecard vs XGBoost.

Outputs:
  • Console: head-to-head metric table + PSI
  • visuals/model_evaluation/champion_challenger_roc.png
  • visuals/model_evaluation/champion_challenger_metrics.png
  • visuals/model_evaluation/psi_report.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.helper import (
    DATA_CLEAN, VIS_EVAL,
    load_model, gini, ks_statistic, psi,
    print_metrics,
)

WOE_PKL      = os.path.join(DATA_CLEAN, "woe_encoded.parquet")
RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ── Data ──────────────────────────────────────────────────────────────────────

def load_test_set():
    df = pd.read_parquet(WOE_PKL)
    y  = df["target"].values
    X  = df.drop(columns=["target"])
    _, X_test, _, y_test = train_test_split(
        X.values, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE
    )
    return X_test, y_test, X.columns.tolist()


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_models(X_test, y_test):
    lr    = load_model("lr_scorecard")
    xgb_m = load_model("xgboost_challenger")

    lr_proba  = lr.predict_proba(X_test)[:, 1]
    xgb_proba = xgb_m.predict_proba(X_test)[:, 1]

    lr_metrics  = print_metrics(y_test, lr_proba,  "LR Scorecard (Champion)")
    xgb_metrics = print_metrics(y_test, xgb_proba, "XGBoost (Challenger)")

    return lr_proba, xgb_proba, lr_metrics, xgb_metrics


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_combined_roc(y_test, lr_proba, xgb_proba):
    from sklearn.metrics import roc_auc_score
    fig, ax = plt.subplots(figsize=(6, 5))
    for proba, label, color in [
        (lr_proba,  "LR Scorecard (Champion)",  "steelblue"),
        (xgb_proba, "XGBoost (Challenger)",      "tomato"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{label} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Champion vs Challenger — ROC Curves")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = os.path.join(VIS_EVAL, "champion_challenger_roc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[cc] Saved → {path}")


def plot_metric_bars(lr_m: dict, xgb_m: dict):
    metrics = ["AUC", "Gini", "KS"]
    lr_vals  = [lr_m[m]  for m in metrics]
    xgb_vals = [xgb_m[m] for m in metrics]

    x  = np.arange(len(metrics))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, lr_vals,  w, label="LR Scorecard", color="steelblue")
    ax.bar(x + w/2, xgb_vals, w, label="XGBoost",      color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Champion vs Challenger — Metrics")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(VIS_EVAL, "champion_challenger_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[cc] Saved → {path}")


# ── PSI Report ────────────────────────────────────────────────────────────────

def psi_report(lr_proba, xgb_proba):
    """Compare score distributions between champion and challenger via PSI."""
    psi_val = psi(lr_proba, xgb_proba, bins=10)
    label   = ("No significant shift" if psi_val < 0.10 else
                "Moderate shift — monitor" if psi_val < 0.25 else
                "Major shift — investigate")
    df_psi  = pd.DataFrame([{"comparison": "LR vs XGBoost scores", "PSI": round(psi_val, 4), "label": label}])
    path    = os.path.join(VIS_EVAL, "psi_report.csv")
    df_psi.to_csv(path, index=False)
    print(f"\n[cc] PSI (LR scores vs XGBoost scores): {psi_val:.4f}  →  {label}")
    print(f"[cc] PSI report saved → {path}")


# ── Recommendation ────────────────────────────────────────────────────────────

def recommend(lr_m: dict, xgb_m: dict):
    print("\n[cc] ══ Champion–Challenger Recommendation ══")
    auc_delta = xgb_m["AUC"] - lr_m["AUC"]
    ks_delta  = xgb_m["KS"]  - lr_m["KS"]
    print(f"  ΔAUC (Challenger - Champion): {auc_delta:+.4f}")
    print(f"  ΔKS  (Challenger - Champion): {ks_delta:+.4f}")
    if auc_delta > 0.01:
        print("  → XGBoost shows meaningful improvement. Consider promoting to Champion.")
    elif auc_delta > 0:
        print("  → Marginal XGBoost gain. Retain LR Scorecard for interpretability.")
    else:
        print("  → LR Scorecard matches or outperforms XGBoost. Retain as Champion.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_test, y_test, _ = load_test_set()
    lr_proba, xgb_proba, lr_m, xgb_m = score_models(X_test, y_test)
    plot_combined_roc(y_test, lr_proba, xgb_proba)
    plot_metric_bars(lr_m, xgb_m)
    psi_report(lr_proba, xgb_proba)
    recommend(lr_m, xgb_m)

    # Save comparison table
    summary = pd.DataFrame([lr_m, xgb_m])
    out_path = os.path.join(VIS_EVAL, "champion_challenger_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"\n[cc] Summary table → {out_path}")


if __name__ == "__main__":
    main()
