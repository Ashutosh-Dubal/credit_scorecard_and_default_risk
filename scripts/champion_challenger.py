import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from helper import DATA_CLEAN_PATH, load_model

# ── Paths ─────────────────────────────────────────────────────────────────────
WOE_PKL  = os.path.join(DATA_CLEAN_PATH, "woe_encoded.parquet")
VIS_DIR  = "visuals/model_evaluation"
os.makedirs(VIS_DIR, exist_ok=True)

# ── Same split parameters as model_training.py ────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

sns.set_theme(style="whitegrid", palette="muted")


# ── 1. Load Data & Models ─────────────────────────────────────────────────────
def load_data_and_models():
    """
    Loads the same test set used during training.
    Using identical random_state and test_size guarantees
    we evaluate on the exact same 61,503 rows — a different
    test set would make the comparison unfair.
    """
    print("[CC] Loading data and models ...")
    df = pd.read_parquet(WOE_PKL)

    X = df.drop(columns=["target"])
    y = df["target"]

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    lr  = load_model("lr_scorecard")
    xgb = load_model("xgboost_challenger")

    print(f"[CC] Test set: {X_test.shape[0]:,} rows  "
          f"Bad rate: {y_test.mean():.2%}")

    return X_test, y_test, lr, xgb


# ── 2. Score Both Models ───────────────────────────────────────────────────────
def score_models(X_test, y_test, lr, xgb):
    """
    Generate probability of default predictions from both models.
    predict_proba returns [prob_good, prob_bad] — we take [:, 1]
    which is the probability of being a defaulter (target=1).
    """
    print("\n[CC] Scoring both models ...")

    lr_proba  = lr.predict_proba(X_test)[:, 1]
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    return lr_proba, xgb_proba


# ── 3. Compute Metrics ─────────────────────────────────────────────────────────
def compute_metrics(y_test, lr_proba, xgb_proba):
    """
    Computes AUC, Gini and KS for both models.

    KS statistic — maximum separation between the cumulative
    distribution of good and bad borrowers when ranked by score.
    This tells you at what threshold the model best separates
    the two populations. Higher KS = better separation.
    """
    print("\n[CC] Computing metrics ...")

    def metrics(name, proba):
        auc  = roc_auc_score(y_test, proba)
        gini = 2 * auc - 1
        fpr, tpr, _ = roc_curve(y_test, proba)
        ks   = float(np.max(tpr - fpr))
        return {"Model": name, "AUC": round(auc, 4),
                "Gini": round(gini, 4), "KS": round(ks, 4)}

    lr_metrics  = metrics("LR Scorecard (Champion)", lr_proba)
    xgb_metrics = metrics("XGBoost (Challenger)",    xgb_proba)

    results = pd.DataFrame([lr_metrics, xgb_metrics])

    print("\n[CC] ══ Champion vs Challenger ════════════════════════")
    print(results.to_string(index=False))
    print("═══════════════════════════════════════════════════════")

    return results, lr_metrics, xgb_metrics


# ── 4. PSI (Population Stability Index) ───────────────────────────────────────
def compute_psi(lr_proba, xgb_proba, bins=10):
    """
    PSI compares the score distributions of champion vs challenger.
    It answers: are both models scoring the population similarly?
    A large PSI means the two models are behaving very differently
    which is useful to know before deciding which to deploy.

    PSI < 0.10  → distributions are similar
    PSI 0.10–0.25 → moderate difference
    PSI > 0.25  → significant difference — investigate
    """
    breakpoints = np.percentile(lr_proba, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    def bucket(arr):
        counts = np.histogram(arr, bins=breakpoints)[0]
        pct    = counts / len(arr)
        return np.where(pct == 0, 1e-6, pct)

    exp = bucket(lr_proba)
    act = bucket(xgb_proba)
    psi = float(np.sum((act - exp) * np.log(act / exp)))

    label = ("Similar distributions" if psi < 0.10 else
             "Moderate difference — review" if psi < 0.25 else
             "Significant difference — investigate")

    print(f"\n[CC] PSI (LR vs XGBoost scores): {psi:.4f} → {label}")
    return psi, label


# ── 5. Plot ROC Curves ────────────────────────────────────────────────────────
def plot_roc_curves(y_test, lr_proba, xgb_proba):
    fig, ax = plt.subplots(figsize=(7, 6))

    for proba, label, color in [
        (lr_proba,  "LR Scorecard (Champion)", "steelblue"),
        (xgb_proba, "XGBoost (Challenger)",     "tomato"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{label}  AUC={auc:.4f}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
    ax.fill_between([0, 1], [0, 1], alpha=0.03, color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Champion vs Challenger")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = os.path.join(VIS_DIR, "01_roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"[CC] Saved → {path}")


# ── 6. Plot KS Curves ─────────────────────────────────────────────────────────
def plot_ks_curves(y_test, lr_proba, xgb_proba):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, proba, label, color in [
        (axes[0], lr_proba,  "LR Scorecard (Champion)", "steelblue"),
        (axes[1], xgb_proba, "XGBoost (Challenger)",     "tomato"),
    ]:
        df = pd.DataFrame({"y": y_test.values, "p": proba})
        df = df.sort_values("p", ascending=False).reset_index(drop=True)

        cum_bad  = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
        cum_good = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
        ks_vals  = cum_bad - cum_good
        ks_idx   = ks_vals.idxmax()
        ks_val   = ks_vals.max()

        x = np.arange(len(df)) / len(df)
        ax.plot(x, cum_bad.values,  color="tomato",    lw=2, label="Cumulative Bad")
        ax.plot(x, cum_good.values, color="steelblue", lw=2, label="Cumulative Good")
        ax.axvline(x[ks_idx], color="black", linestyle="--", lw=1,
                   label=f"KS = {ks_val:.4f}")
        ax.fill_between(x, cum_bad.values, cum_good.values, alpha=0.08, color=color)
        ax.set_title(label)
        ax.set_xlabel("Population (ranked by score desc)")
        ax.set_ylabel("Cumulative proportion")
        ax.legend(fontsize=9)

    fig.suptitle("KS Curves — Champion vs Challenger", fontsize=12)
    fig.tight_layout()
    path = os.path.join(VIS_DIR, "02_ks_curves.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"[CC] Saved → {path}")


# ── 7. Plot Score Distributions ───────────────────────────────────────────────
def plot_score_distributions(y_test, lr_proba, xgb_proba):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, proba, label in [
        (axes[0], lr_proba,  "LR Scorecard (Champion)"),
        (axes[1], xgb_proba, "XGBoost (Challenger)"),
    ]:
        for target, color, name in [
            (0, "steelblue", "Non-Defaulter"),
            (1, "tomato",    "Defaulter"),
        ]:
            mask = y_test.values == target
            ax.hist(proba[mask], bins=40, alpha=0.6,
                    color=color, label=name, density=True)
        ax.set_title(label)
        ax.set_xlabel("Predicted probability of default")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.suptitle("Score Distributions — Champion vs Challenger", fontsize=12)
    fig.tight_layout()
    path = os.path.join(VIS_DIR, "03_score_distributions.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"[CC] Saved → {path}")


# ── 8. Plot Metric Comparison ─────────────────────────────────────────────────
def plot_metric_comparison(results):
    metrics = ["AUC", "Gini", "KS"]
    x = np.arange(len(metrics))
    w = 0.35

    lr_vals  = [results.loc[0, m] for m in metrics]
    xgb_vals = [results.loc[1, m] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, lr_vals,  w, label="LR Scorecard", color="steelblue", alpha=0.85)
    ax.bar(x + w/2, xgb_vals, w, label="XGBoost",      color="tomato",    alpha=0.85)

    for i, (lv, xv) in enumerate(zip(lr_vals, xgb_vals)):
        ax.text(i - w/2, lv  + 0.005, f"{lv:.3f}",  ha="center", fontsize=9)
        ax.text(i + w/2, xv  + 0.005, f"{xv:.3f}",  ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Metric Comparison — Champion vs Challenger")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(VIS_DIR, "04_metric_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"[CC] Saved → {path}")


# ── 9. Recommendation ─────────────────────────────────────────────────────────
def recommend(lr_metrics, xgb_metrics, psi):
    """
    Makes a deployment recommendation based on:
    - Performance gap between champion and challenger
    - PSI (are they scoring the population similarly?)
    - Industry standard: challenger needs meaningful improvement
      to justify replacing an interpretable champion
    """
    auc_delta  = xgb_metrics["AUC"]  - lr_metrics["AUC"]
    gini_delta = xgb_metrics["Gini"] - lr_metrics["Gini"]
    ks_delta   = xgb_metrics["KS"]   - lr_metrics["KS"]

    print("\n[CC] ── Recommendation ─────────────────────────────────")
    print(f"  ΔAUC  (Challenger - Champion): {auc_delta:+.4f}")
    print(f"  ΔGini (Challenger - Champion): {gini_delta:+.4f}")
    print(f"  ΔKS   (Challenger - Champion): {ks_delta:+.4f}")
    print(f"  PSI between models           : {psi:.4f}")

    if auc_delta > 0.02:
        rec = ("XGBoost shows meaningful improvement (ΔAUC > 0.02). "
               "Consider promoting to Champion if explainability "
               "requirements can be met via SHAP.")
    elif auc_delta > 0:
        rec = ("XGBoost shows marginal improvement. "
               "Retain LR Scorecard as Champion — the performance "
               "gain does not justify replacing an interpretable model.")
    else:
        rec = ("LR Scorecard matches or outperforms XGBoost. "
               "Retain LR Scorecard as Champion.")

    print(f"\n  Recommendation: {rec}")
    print("─────────────────────────────────────────────────────────")
    return rec


# ── 10. Save Summary ──────────────────────────────────────────────────────────
def save_summary(results, psi, psi_label, rec):
    results["PSI_vs_champion"] = [0, round(psi, 4)]
    results["PSI_label"]       = ["—", psi_label]
    results["recommendation"]  = [rec, "—"]
    path = os.path.join(VIS_DIR, "champion_challenger_summary.csv")
    results.to_csv(path, index=False)
    print(f"\n[CC] Summary saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load
    X_test, y_test, lr, xgb = load_data_and_models()

    # 2. Score
    lr_proba, xgb_proba = score_models(X_test, y_test, lr, xgb)

    # 3. Metrics
    results, lr_metrics, xgb_metrics = compute_metrics(
        y_test, lr_proba, xgb_proba
    )

    # 4. PSI
    psi, psi_label = compute_psi(lr_proba, xgb_proba)

    # 5. Plots
    plot_roc_curves(y_test, lr_proba, xgb_proba)
    plot_ks_curves(y_test, lr_proba, xgb_proba)
    plot_score_distributions(y_test, lr_proba, xgb_proba)
    plot_metric_comparison(results)

    # 6. Recommendation
    rec = recommend(lr_metrics, xgb_metrics, psi)

    # 7. Save
    save_summary(results, psi, psi_label, rec)

    print("\n[CC] Champion-Challenger analysis complete!")
    print(f"[CC] Plots saved to: {VIS_DIR}/")