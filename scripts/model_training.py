"""
model_training.py — Train Logistic Regression Scorecard (champion) + XGBoost (challenger).

Steps:
  1. Load WoE-encoded dataset
  2. Stratified 80/20 train-test split
  3. Fit Logistic Regression → convert to points-based scorecard
  4. Fit XGBoost with early stopping
  5. Evaluate both models (AUC, Gini, KS)
  6. Save artefacts + evaluation plots

Scorecard scaling:
  Score = Offset + Factor × log(odds)
  Factor = PDO / ln(2)      (PDO = points to double the odds)
  Offset = Base_score − Factor × ln(Base_odds)
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.helper import (
    DATA_CLEAN, VIS_EVAL, VIS_SCORE,
    save_model, print_metrics,
    plot_roc, plot_ks, plot_score_distribution,
)

WOE_PKL = os.path.join(DATA_CLEAN, "woe_encoded.parquet")

# ── Scorecard scaling parameters ──────────────────────────────────────────────
BASE_SCORE = 600    # score that corresponds to base odds
BASE_ODDS  = 50     # good-to-bad ratio at base score
PDO        = 20     # points to double the odds

RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_woe() -> tuple:
    df = pd.read_parquet(WOE_PKL)
    y  = df["target"].values
    X  = df.drop(columns=["target"])
    print(f"[train] WoE dataset: {X.shape[0]:,} rows × {X.shape[1]} features")
    return X, y


def scale_scorecard(model: LogisticRegression, feature_names: list, scaler: StandardScaler):
    """Return a DataFrame of feature-level scorecard points."""
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)

    coefs = model.coef_[0]
    intercept = model.intercept_[0]

    rows = []
    for name, coef in zip(feature_names, coefs):
        # Un-scale coefficient back to original WoE space
        coef_orig = coef / scaler.scale_[feature_names.index(name)]
        points = -factor * coef_orig
        rows.append({"feature": name, "coefficient": coef_orig, "points_per_WoE_unit": points})

    df_sc = pd.DataFrame(rows)
    df_sc["intercept_contribution"] = -factor * intercept / len(feature_names)
    return df_sc, factor, offset


# ── Model 1: Logistic Regression Scorecard ────────────────────────────────────

def train_logistic(X_train, y_train, X_test, y_test, feature_names):
    print("\n[train] ─── Logistic Regression Scorecard ───")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.1,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])
    pipe.fit(X_train, y_train)

    proba_train = pipe.predict_proba(X_train)[:, 1]
    proba_test  = pipe.predict_proba(X_test)[:, 1]

    metrics_train = print_metrics(y_train, proba_train, "LR Scorecard — Train")
    metrics_test  = print_metrics(y_test,  proba_test,  "LR Scorecard — Test")

    # Scorecard points
    scaler  = pipe.named_steps["scaler"]
    lr_model = pipe.named_steps["lr"]
    sc_df, factor, offset = scale_scorecard(lr_model, list(feature_names), scaler)
    print("\n[train] Scorecard points (top 10):")
    print(sc_df.head(10).to_string(index=False))

    # Convert probability to score  →  score = offset + factor * ln(p_good / p_bad)
    eps   = 1e-8
    score = offset + factor * np.log((1 - proba_test + eps) / (proba_test + eps))

    # Plots
    plot_roc(y_test, proba_test,  "LR Scorecard",
             save_path=os.path.join(VIS_EVAL, "lr_roc.png"))
    plot_ks(y_test,  proba_test,  "LR Scorecard",
            save_path=os.path.join(VIS_EVAL, "lr_ks.png"))
    plot_score_distribution(score, y_test,
                            save_path=os.path.join(VIS_SCORE, "lr_score_dist.png"))

    save_model(pipe, "lr_scorecard")
    return pipe, proba_test, score, metrics_test


# ── Model 2: XGBoost ──────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n[train] ─── XGBoost ───")

    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())

    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    proba_test = model.predict_proba(X_test)[:, 1]
    metrics_test = print_metrics(y_test, proba_test, "XGBoost — Test")

    # Plots
    plot_roc(y_test, proba_test,  "XGBoost",
             save_path=os.path.join(VIS_EVAL, "xgb_roc.png"))
    plot_ks(y_test,  proba_test,  "XGBoost",
            save_path=os.path.join(VIS_EVAL, "xgb_ks.png"))
    plot_score_distribution(proba_test, y_test,
                            save_path=os.path.join(VIS_SCORE, "xgb_score_dist.png"))

    save_model(model, "xgboost_challenger")
    return model, proba_test, metrics_test


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X, y = load_woe()
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE
    )
    print(f"[train] Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")
    print(f"[train] Bad rate — train: {y_train.mean():.2%}  test: {y_test.mean():.2%}")

    lr_pipe, lr_proba, lr_score, lr_metrics = train_logistic(
        X_train, y_train, X_test, y_test, feature_names
    )
    xgb_model, xgb_proba, xgb_metrics = train_xgboost(
        X_train, y_train, X_test, y_test
    )

    # Summary table
    summary = pd.DataFrame([lr_metrics, xgb_metrics])
    print("\n[train] ══ Final comparison ══")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(VIS_EVAL, "model_comparison.csv"), index=False)

    print("\n[train] Training complete. All artefacts saved.")


if __name__ == "__main__":
    main()
