# Credit Scorecard & Default Risk Model

A production-style credit risk pipeline built on **Lending Club** loan data. The project implements a traditional **logistic regression scorecard** (with Weight of Evidence / Information Value binning) alongside an **XGBoost challenger model**, with a full champion–challenger A/B comparison framework.

---

## Project Structure

```
credit-scorecard-default-risk/
├── data/
│   ├── raw/          # Lending Club CSV (original, not committed)
│   └── clean/        # Feature-engineered, WoE-encoded outputs
├── scripts/
│   ├── fetch_data.py            # Download raw data from source
│   ├── clean_data.py            # Missing values, type casting, target encoding
│   ├── EDA.py                   # Exploratory data analysis & visualisations
│   ├── feature_engineering.py   # WoE / IV binning
│   ├── model_training.py        # Logistic regression scorecard + XGBoost
│   ├── champion_challenger.py   # A/B model comparison framework
│   └── helper.py                # Shared utilities (metrics, plotting, I/O)
├── visuals/
│   ├── EDA/                     # Distribution plots, correlation heatmaps
│   ├── score_distribution/      # Score histograms by target class
│   └── model_evaluation/        # Gini, KS statistic, ROC/PR curves
├── models/                      # Serialised model artefacts (.pkl / .joblib)
├── LICENSE
├── README.md
└── .gitignore
```

---

## Methodology

### 1. Data
- **Source:** Lending Club public loan dataset
- **Target:** `loan_status` → binary flag (`1 = Default`, `0 = Fully Paid`)

### 2. Feature Engineering
- Missing-value imputation & outlier capping
- **Weight of Evidence (WoE)** encoding for categorical & binned continuous features
- **Information Value (IV)** for feature selection (IV < 0.02 → dropped)

### 3. Models
| Model | Role | Interpretability |
|---|---|---|
| Logistic Regression Scorecard | Champion | Full (points-based) |
| XGBoost | Challenger | Partial (SHAP) |

### 4. Evaluation Metrics
- **Gini coefficient** (= 2 × AUC − 1)
- **KS statistic** (max separation between cumulative good/bad distributions)
- **ROC & Precision-Recall curves**
- **Score distribution** by target class

### 5. Champion–Challenger Framework
- Stratified train/test split
- Population Stability Index (PSI) for score drift monitoring
- Side-by-side Gini / KS / AUC comparison table

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
scipy
joblib
optbinning      # WoE / IV binning
shap            # XGBoost explainability
```

---

## Results (example — update after training)

| Metric | Logistic Scorecard | XGBoost |
|---|---|---|
| AUC | — | — |
| Gini | — | — |
| KS | — | — |

---

## License
MIT — see [LICENSE](LICENSE).