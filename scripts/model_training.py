import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from helper import DATA_CLEAN_PATH, save_model, load_model

# ── Paths ─────────────────────────────────────────────────────────────────────
WOE_PKL = os.path.join(DATA_CLEAN_PATH, "woe_encoded.parquet")

# ── Scorecard scaling parameters ──────────────────────────────────────────────
BASE_SCORE = 600   # score assigned at base odds
BASE_ODDS  = 50    # good:bad ratio at base score
PDO        = 20    # points to double the odds

# ── Train/test split parameters ───────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── 1. Load WoE Encoded Data ──────────────────────────────────────────────────
def load_woe():
    print("[MT] Loading WoE encoded dataset ...")
    df = pd.read_parquet(WOE_PKL)
    print(f"[MT] Shape: {df.shape}")
    print(f"[MT] Bad rate: {df['target'].mean():.2%}")
    return df


# ── 2. Train/Test Split ───────────────────────────────────────────────────────
def split_data(df, target_col="target"):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"\n[MT] Train: {X_train.shape[0]:,} rows  "
          f"Bad rate: {y_train.mean():.2%}")
    print(f"[MT] Test : {X_test.shape[0]:,} rows  "
          f"Bad rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test


# ── 3. Logistic Regression Scorecard (Champion) ───────────────────────────────
def train_logistic(X_train, X_test, y_train, y_test):
    print("\n[MT] ── Training Logistic Regression Scorecard ──────────")

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=0.1,
        solver="lbfgs",
        random_state=RANDOM_STATE
    )
    lr.fit(X_train, y_train)

    # Probabilities
    train_proba = lr.predict_proba(X_train)[:, 1]
    test_proba  = lr.predict_proba(X_test)[:, 1]

    # Metrics
    train_auc = roc_auc_score(y_train, train_proba)
    test_auc  = roc_auc_score(y_test,  test_proba)
    train_gini = 2 * train_auc - 1
    test_gini  = 2 * test_auc  - 1

    print(f"[MT] Train AUC : {train_auc:.4f}  Gini: {train_gini:.4f}")
    print(f"[MT] Test  AUC : {test_auc:.4f}  Gini: {test_gini:.4f}")

    # Overfit check
    gini_gap = train_gini - test_gini
    if gini_gap > 0.05:
        print(f"[MT] WARNING: Gini gap of {gini_gap:.4f} suggests overfitting")
    else:
        print(f"[MT] Gini gap: {gini_gap:.4f} — no significant overfitting")

    # Scorecard scaling
    scorecard_df = build_scorecard(lr, X_train.columns.tolist())
    print("\n[MT] Scorecard points (top 10 features):")
    print(scorecard_df.head(10).to_string(index=False))

    # Convert probabilities to scores
    train_scores = proba_to_score(train_proba)
    test_scores  = proba_to_score(test_proba)

    print(f"\n[MT] Score distribution (test set):")
    print(f"     Mean  : {test_scores.mean():.1f}")
    print(f"     Std   : {test_scores.std():.1f}")
    print(f"     Min   : {test_scores.min():.1f}")
    print(f"     Max   : {test_scores.max():.1f}")

    save_model(lr, "lr_scorecard")

    return lr, test_proba, test_scores, {"model": "LR Scorecard",
                                          "AUC": test_auc,
                                          "Gini": test_gini}


# ── 4. Scorecard Scaling ──────────────────────────────────────────────────────
def build_scorecard(lr, feature_names):
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)

    rows = []
    for feature, coef in zip(feature_names, lr.coef_[0]):
        # Points contribution per unit of WoE for this feature
        points = -factor * coef
        rows.append({
            "feature":     feature,
            "coefficient": round(coef, 4),
            "points":      round(points, 2)
        })

    scorecard_df = (pd.DataFrame(rows)
                      .sort_values("points", ascending=False)
                      .reset_index(drop=True))
    return scorecard_df


def proba_to_score(proba, eps=1e-8):
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    odds   = (1 - proba + eps) / (proba + eps)
    scores = offset + factor * np.log(odds)
    return pd.Series(scores)


# ── 5. XGBoost Challenger ─────────────────────────────────────────────────────
def train_xgboost(X_train, X_test, y_train, y_test):
    print("\n[MT] ── Training XGBoost Challenger ──────────────────────")

    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())
    print(f"[MT] scale_pos_weight: {scale_pos}")

    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=RANDOM_STATE,
        verbosity=0
    )

    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    test_proba = xgb.predict_proba(X_test)[:, 1]
    test_auc   = roc_auc_score(y_test, test_proba)
    test_gini  = 2 * test_auc - 1

    train_proba = xgb.predict_proba(X_train)[:, 1]
    train_auc   = roc_auc_score(y_train, train_proba)
    train_gini  = 2 * train_auc - 1

    print(f"[MT] Train AUC : {train_auc:.4f}  Gini: {train_gini:.4f}")
    print(f"[MT] Test  AUC : {test_auc:.4f}  Gini: {test_gini:.4f}")

    gini_gap = train_gini - test_gini
    if gini_gap > 0.05:
        print(f"[MT] WARNING: Gini gap of {gini_gap:.4f} suggests overfitting")
    else:
        print(f"[MT] Gini gap: {gini_gap:.4f} — no significant overfitting")

    save_model(xgb, "xgboost_challenger")

    return xgb, test_proba, {"model": "XGBoost",
                              "AUC": test_auc,
                              "Gini": test_gini}

# ── 6. Summary Table ──────────────────────────────────────────────────────────
def print_summary(lr_metrics, xgb_metrics):
    summary = pd.DataFrame([lr_metrics, xgb_metrics])
    print("\n[MT] ══ Model Comparison Summary ═════════════════════════")
    print(summary.to_string(index=False))
    print("══════════════════════════════════════════════════════════")

    # Benchmark check
    for _, row in summary.iterrows():
        gini_ok = "✅" if row["Gini"] >= 0.45 else "❌"
        auc_ok  = "✅" if row["AUC"]  >= 0.75 else "❌"
        print(f"  {row['model']:<25} "
              f"Gini {gini_ok} {row['Gini']:.4f}   "
              f"AUC {auc_ok} {row['AUC']:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load
    df = load_woe()

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Train champion — Logistic Regression Scorecard
    lr, lr_proba, lr_scores, lr_metrics = train_logistic(
        X_train, X_test, y_train, y_test
    )

    # 4. Train challenger — XGBoost
    xgb, xgb_proba, xgb_metrics = train_xgboost(
        X_train, X_test, y_train, y_test
    )

    # 5. Summary
    print_summary(lr_metrics, xgb_metrics)

    print("\n[MT] Training complete!")
    print(f"[MT] Models saved to: models/")