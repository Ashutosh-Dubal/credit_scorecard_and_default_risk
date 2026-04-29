# Credit Scorecard & Default Risk Model

---

## 📚 Table of Contents  
1. [Dataset Description]()  
2. [Challenges & Learnings]()
3. [Key Insights & Analysis]() 
4. [Prediction & Evaluation]()
5. [Model Interpretation]()
6. [Tech Stack]()  
7. [Project Structure]()  
8. [Author]()
9. [License]()

---

```
TODO: clean up - 1. Challenges and Learning section - DONE
                 2. IV results
                 3. EDA  

FINAL_TODO: 1. double check project structure - DONE
            2. Add all links to Table of content 
            
```

---

## 📦 Dataset Description

**Source:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) — Kaggle Competition  
**Provider:** Home Credit Group  
**File Used:** `application_train.csv`

### Overview
Home Credit is a consumer finance provider that serves populations who have 
little to no traditional credit history. The dataset contains 307,511 loan 
applications with 122 features per applicant covering demographics, financial 
profile, employment history, and external credit bureau assessments.

The core question the dataset is designed to answer is:

> *Given what we know about a borrower at the time of application, 
> how likely are they to default on their loan?*

### Target Variable
`TARGET` — Binary default indicator:

| Value | Meaning | Count | Share |
|---|---|---|---|
| 0 | Loan repaid — Non-Defaulter | 282,686 | 91.9% |
| 1 | Payment difficulties — Defaulter | 24,825 | 8.1% |

The dataset is heavily imbalanced at roughly 1 defaulter for every 11 
non-defaulters. This makes raw accuracy a misleading success metric — a 
model that predicts "no default" for every applicant would be correct 92% 
of the time without learning anything meaningful. For this reason the 
primary evaluation metrics used in this project are Gini coefficient and 
KS statistic rather than accuracy.

### Features
The 122 features fall into 5 broad categories:

| Category | Examples | Count |
|---|---|---|
| Demographics | Age, gender, family status, education | ~10 |
| Financial profile | Income, credit amount, annuity, goods price | ~15 |
| Employment | Days employed, occupation type, organisation type | ~10 |
| External credit scores | EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3 | 3 |
| Credit bureau enquiries | AMT_REQ_CREDIT_BUREAU_* (hour/day/week/month/year) | ~10 |

### Top Predictive Features (identified during EDA)
Through Random Forest permutation importance and XGBoost feature importance 
combined with a rank-based comparison, the following 10 features were 
identified as the most influential predictors of default:

| Rank | Feature | Why it matters |
|---|---|---|
| 1 | `EXT_SOURCE_2` | External credit bureau score — strongest single predictor |
| 2 | `EXT_SOURCE_3` | External credit bureau score — confirms EXT_SOURCE_2 signal |
| 3 | `CREDIT_TO_GOODS` | Loan amount relative to goods price — measures over-borrowing |
| 4 | `FLAG_OWN_CAR` | Asset ownership — proxy for financial stability |
| 5 | `ANNUITY_TO_INCOME` | Monthly repayment burden relative to income |
| 6 | `AMT_INCOME_TOTAL` | Total applicant income |
| 7 | `DAYS_ID_PUBLISH` | How recently ID was issued — proxy for life stability |
| 8 | `DEF_30_CNT_SOCIAL_CIRCLE` | Defaults in applicant's social circle |
| 9 | `AMT_ANNUITY` | Monthly loan repayment amount |
| 10 | `DAYS_EMPLOYED` | Employment duration — shorter tenure = higher risk |

### Data Cleaning Summary
Before analysis the raw data was cleaned through the following steps:
- Dropped columns with more than 50% missing values
- Replaced `DAYS_EMPLOYED = 365243` anomaly (placeholder for unemployed applicants) with `NaN`
- Replaced `CODE_GENDER = XNA` with `NaN`
- Encoded all Y/N binary columns to 1/0
- Applied Winsorisation at the 1st and 99th percentile to cap outliers
- Imputed `AMT_REQ_CREDIT_BUREAU_*` missing values with `0` (no bureau activity)
- Imputed `OWN_CAR_AGE` missing values with `0` (no car)
- Imputed remaining numeric columns with median and categorical columns with `"Unknown"`

### Engineered Features
7 derived features were created to capture meaningful financial ratios:
- `AGE_YEARS` — applicant age in years
- `YEARS_EMPLOYED` — employment duration in years
- `CREDIT_TO_INCOME` — loan amount divided by income
- `ANNUITY_TO_INCOME` — monthly repayment divided by income
- `CREDIT_TO_GOODS` — loan amount divided by goods price
- `INCOME_PER_PERSON` — income divided by family members
- `EMPLOYED_TO_AGE` — employment length as fraction of age

---

## 🧠 Challenges & Learnings

1. **Feature selection with 122 variables**  
After cleaning the data and starting EDA it became clear that 122 variables 
was too many to analyse meaningfully. I needed a systematic way to identify 
the most predictive features before building any model. There are three broad 
categories of feature selection methods: Statistical, Machine Learning, and 
Credit Risk Specific. For this project I used Random Forest with permutation 
importance and XGBoost feature importance, both falling under ML methods. 
RFE was ruled out as it is too slow for 300K rows and does not interact well 
with WoE encoding. LASSO was ruled out because it cannot capture non-linear 
relationships. SHAP values are planned for Phase 2 as they are better suited 
for post-model explainability than pre-model feature selection.

2. **Comparing feature importance across different methods**  
The RF permutation importance and XGBoost importance scores are on completely 
different scales — you cannot compare 0.008 from RF with 0.111 from XGBoost 
directly. The solution was a rank-based comparison system that converts both 
sets of scores into positions (rank 1 to 10) and averages them. Features that 
both models agreed on rose to the top with high confidence. Features only one 
model found important received penalty ranks, naturally pushing them lower.

3. **Architecture mismatch on Apple Silicon**  
Running Python 3.9 under Rosetta (x86_64 emulation) caused numpy and XGBoost 
to fail due to incompatible binary architectures. The fix was rebuilding the 
virtual environment using the native ARM64 system Python at `/usr/bin/python3` 
and installing a native ARM64 version of libomp via a native ARM Homebrew 
installation at `/opt/homebrew`.

---

## 👁 Key Insights & Analysis

### Exploratory Data Analysis (EDA)

#### Graph 1 — Target Distribution

![Target Distribution](/visuals/EDA/01_target_distribution.png)

   The story here is simple but critical — our dataset is heavily imbalanced. Out of 307,511 borrowers, only 24,825 defaulted. That's roughly 1 in 12. 
   This tells you that the majority of people in this dataset are reliable borrowers, and your model has far fewer examples of the "bad" behaviour 
   it needs to detect. This is why accuracy alone is a misleading metric — a model that just says "everyone is fine" would be right 92% of the time 
   without learning anything useful.

#### Graph 2 — RF vs XGBoost Importance

![RF vs XGB](/visuals/EDA/02_rf_vs_xgb_importance.png)

   This is the casting call for our story. Both models unanimously agree that EXT_SOURCE_2 and EXT_SOURCE_3 are the two most important characters. 
   Everything else is a supporting cast. The gap between the external scores and the rest of the features is massive — look at how the RF bars drop  
   off a cliff after EXT_SOURCE_3. This tells you that whoever these external credit bureaus are, they've already done a lot of the risk assessment 
   work for us.

#### Graph 3 — Box Plots

![Boxplots](/visuals/EDA/03_boxplots_by_target.png)

   This is where the story starts getting personal. Look at EXT_SOURCE_2 and EXT_SOURCE_3 specifically. The box for non-defaulters (0) sits noticeably 
   higher than the box for defaulters (1) — meaning people who repaid their loans consistently had higher external credit scores. The separation is clear 
   and clean. Now look at DAYS_EMPLOYED — both boxes are negative because it's stored as days (negative numbers mean further in the past). The 
   defaulters box is shifted closer to zero meaning they had been employed for less time. Shorter employment history correlates with higher default risk. 
   DEF_30_CNT_SOCIAL_CIRCLE tells an interesting social story — it barely moves between defaulters and non-defaulters, suggesting that knowing someone 
   who defaulted in the last 30 days isn't as predictive as the other features.

#### Graph 4 — Bad Rate by Decile

![Bad Rate](/visuals/EDA/04_bad_rate_by_decile.png)

   This is the most important graph for our project. Let me walk you through the key characters:

   EXT_SOURCE_2 — read it left to right. D1 (lowest scores) has a bad rate of about 17-18%. By D10 (highest scores) it drops to about 2-3%. That is a 
   beautiful monotonic decline — as the score goes up, default risk goes down consistently across every single decile. This is exactly what a strong predictive 
   feature looks like.

   EXT_SOURCE_3 — same story, same clean decline. These two features are essentially telling you the same thing from two different credit bureaus.

   CREDIT_TO_GOODS — more erratic. The bad rate jumps around rather than declining smoothly. This tells you the relationship is noisier and less linear — 
   which is why WoE binning will be important to capture it properly.

   FLAG_OWN_CAR — only 2 bars because it's binary (you either own a car or you don't). Slight difference in bad rate between owners and non-owners but not dramatic.

   DEF_30_CNT_SOCIAL_CIRCLE — only 2 bars too, heavily skewed. Most people have zero defaults in their social circle. The tiny group with 1+ social 
   circle defaults has a noticeably higher bad rate. Sparse but meaningful.

   DAYS_EMPLOYED — the most interesting one here. The bad rate actually increases as you go from D1 to D10 — meaning longer employed people default less, 
   and shorter-employed or recently employed people default more. The spike at D5 is worth noting — could be an artefact of the data distribution.

#### Graph 5 — Normalised Mean by Target

![Normalised Mean](/visuals/EDA/05_mean_by_target.png)

   This is your quick summary card. For every feature it shows whether defaulters or non-defaulters have a higher average value. The key insights:

   EXT_SOURCE_2 and EXT_SOURCE_3 — non-defaulters (blue) are clearly taller, confirming higher scores = lower risk.

   DAYS_ID_PUBLISH and DAYS_EMPLOYED — both go negative because they're stored as negative days. The defaulters bar being less negative for DAYS_EMPLOYED means 
   defaulters had been employed for fewer days on average — shorter job tenure = higher risk.

   DEF_30_CNT_SOCIAL_CIRCLE and AMT_ANNUITY — defaulters have slightly higher values, meaning higher social circle defaults and higher loan repayment amounts 
   correlate with default.

#### Graph 6 — Cumulative Bad Rate

![Cumulative Bad Rate](/visuals/EDA/06_cumulative_bad_rate.png)

   This is the preview of your model's power before you've even built it. Think of it this way — imagine you lined up all 307K borrowers ranked from lowest to 
   highest EXT_SOURCE_2 score. If you stopped at the worst 20% of borrowers, how many of the total defaults would you have captured?

   For EXT_SOURCE_2 — the curve bows way above the diagonal. By the time you've looked at the riskiest 40% of borrowers you've already captured about 60% of 
   all defaults. That's a strong signal. The shaded area between the curve and the diagonal is essentially a visual preview of your Gini coefficient.

   For FLAG_OWN_CAR — the curve barely moves from the diagonal. Owning a car is almost random in terms of predicting default. This is telling you this feature 
   may not survive IV filtering.

   For DAYS_EMPLOYED — the curve goes below the diagonal at first then crosses back. That U-shape means it's predictive but in a non-linear way — the relationship 
   reverses at some point, which is exactly why WoE binning needs to handle it carefully.

#### Graph 7 — Correlation Heatmap

![Correlation Heatmap](/visuals/EDA/07_correlation_heatmap.png)

   Two things jump out. First ANNUITY_TO_INCOME and AMT_ANNUITY have a correlation of 0.51 with each other — that's moderate multicollinearity. They're both 
   measuring something about loan repayment burden relative to income, so they're telling a similar story. IV filtering will help decide which one to keep. 
   Second AMT_INCOME_TOTAL and AMT_ANNUITY also correlate at 0.48 — again related concepts. None of these are dangerously high (above 0.70 would be a problem) 
   but worth noting.

### IV Results

#### Summary

The top features confirmed exactly what EDA predicted:

-> EXT_SOURCE_3 (0.65) and EXT_SOURCE_2 (0.64) — as discussed, keep these despite the "Suspicious" label. They are genuine external bureau scores, not leakage.

-> DAYS_EMPLOYED (0.196) and YEARS_EMPLOYED (0.194) — notice these are essentially the same feature since we engineered YEARS_EMPLOYED from DAYS_EMPLOYED. They are telling the same story twice. This is something to address before modelling — we should keep only one of them.

-> AMT_GOODS_PRICE (0.183) — didn't appear in our EDA top 10 but IV says it's meaningful. This is exactly why we run IV on all features rather than just the EDA shortlist.

-> DAYS_BIRTH (0.174) and AGE_YEARS (0.173) — same problem as DAYS_EMPLOYED and YEARS_EMPLOYED. We engineered AGE_YEARS from DAYS_BIRTH so they're duplicates. Keep AGE_YEARS since it's more interpretable.

-> OCCUPATION_TYPE (0.159) and ORGANIZATION_TYPE (0.143) — categorical features that IV confirmed as meaningful. EDA didn't surface these because our feature importance only ran on numeric columns.

#### 17 featurtes dropped by IV

-> FLAG_OWN_CAR (dropped) — this confirms what the cumulative bad rate curve showed in EDA — the curve barely moved from the diagonal. IV agreed — not predictive enough.

-> ANNUITY_TO_INCOME (dropped) — this was in our EDA top 10 but IV says otherwise. This is a case where IV overrules EDA — the feature has weak predictive signal when binned properly.

-> CNT_CHILDREN, CNT_FAM_MEMBERS — both dropped, family size doesn't meaningfully predict default in this dataset.

-> All AMT_REQ_CREDIT_BUREAU_* columns (hour/day/week/month/quarter/year) — all dropped. Bureau enquiry frequency has very little predictive power here.

#### Data retention check

Prior to the start of the project I had set the data retention rate at 75%, starting from 54 features we're keeping 37. This gives us
37/54 = 68.5% this is lower but its close to 75%. However we initially started with 122 features which is 34/122 = 30%. The correct
interpretation for our benchmark is retention after IV filtering of cleaned features and the 17 dropped features are genuinly low signal 
so losing them makes the model better, not worse.

---

## 🎯 Prediction & Evaluation

### Modelling Approach

Two models were trained on the WoE-encoded dataset using a stratified 80/20 
train-test split — stratified to preserve the 8.07% bad rate in both sets.

**Champion — Logistic Regression Scorecard**  
The industry standard for credit risk. WoE encoding linearises all features 
so logistic regression works optimally. Coefficients are converted into 
interpretable score points using standard scorecard scaling:

Factor = PDO / ln(2) = 20 / 0.693 = 28.85
Offset = Base_score − Factor × ln(Base_odds) = 600 − 28.85 × ln(50) = 487
Score  = Offset + Factor × ln(odds of repayment)

- Base score: 600 | Base odds: 50:1 | PDO: 20
- `class_weight="balanced"` handles the 8:92 class imbalance
- `C=0.1` applies L2 regularisation to prevent overfitting

**Challenger — XGBoost**  
A gradient boosted tree model used as a performance benchmark. 
`scale_pos_weight=11` handles class imbalance. Early stopping at round 235 
prevented overfitting.

---

### Results

| Model | AUC | Gini | KS | Overfit Gap |
|---|---|---|---|---|
| LR Scorecard (Champion) | 0.7453 | 0.4905 | 0.3617 | -0.0054 ✅ |
| XGBoost (Challenger) | 0.7511 | 0.5021 | 0.3730 | 0.0430 ✅ |

**Score distribution (LR Scorecard on test set):**

| Stat | Value |
|---|---|
| Mean score | 497.2 |
| Std deviation | 26.6 |
| Min score | 394.8 |
| Max score | 593.2 |

---

### Champion vs Challenger

| Metric | Delta (XGBoost − LR) |
|---|---|
| ΔAUC | +0.0058 |
| ΔGini | +0.0116 |
| ΔKS | +0.0113 |
| PSI between models | 0.0131 → Similar distributions |

**Recommendation: Retain LR Scorecard as Champion.**  
XGBoost shows marginal improvement across all metrics but the performance 
gap (ΔAUC = 0.006) is well below the 0.02 threshold that would justify 
replacing an interpretable champion model. Both models score the population 
similarly (PSI = 0.013), confirming they capture the same underlying signal.

![ROC Curves](/visuals/model_evaluation/01_roc_curves.png)
![KS Curves](/visuals/model_evaluation/02_ks_curves.png)
![Score Distributions](/visuals/model_evaluation/03_score_distributions.png)
![Metric Comparison](/visuals/model_evaluation/04_metric_comparison.png)

---

## 🧩 Model Interpretation

### Logistic Regression Scorecard — Top Features by Points

Each feature contributes a fixed number of points to the total credit score. 
Higher points = lower default risk for that feature's WoE bin.

| Rank | Feature | Points |
|---|---|---|
| 1 | `EXT_SOURCE_3` | 24.49 |
| 2 | `NAME_CONTRACT_TYPE` | 23.70 |
| 3 | `EXT_SOURCE_2` | 21.59 |
| 4 | `AMT_GOODS_PRICE` | 18.85 |
| 5 | `CODE_GENDER` | 17.67 |
| 6 | `NAME_EDUCATION_TYPE` | 17.37 |
| 7 | `CREDIT_TO_GOODS` | 15.60 |
| 8 | `DEF_60_CNT_SOCIAL_CIRCLE` | 11.44 |
| 9 | `ORGANIZATION_TYPE` | 11.24 |
| 10 | `DEF_30_CNT_SOCIAL_CIRCLE` | 10.91 |

### How the Scorecard Works
A borrower's total credit score is the sum of points from every feature 
based on which WoE bin their value falls into. For example:

- A borrower with high `EXT_SOURCE_3` falls in a positive WoE bin → gains points
- A borrower with high `CREDIT_TO_GOODS` falls in a negative WoE bin → loses points
- The total determines their final score on a 394–593 scale

**Score interpretation:**
- Higher score → lower predicted probability of default → lower risk
- Lower score → higher predicted probability of default → higher risk

### Why Logistic Regression Over XGBoost?
| Consideration | LR Scorecard | XGBoost |
|---|---|---|
| Performance gap | Baseline | +0.006 AUC |
| Interpretability | Full — points per feature | Requires SHAP |
| Regulatory compliance | ✅ Directly explainable | ⚠️ Harder to audit |
| Deployment | Simple scoring table | Requires model file |

The marginal performance gain from XGBoost does not justify the loss of 
interpretability. In a regulated banking environment the LR Scorecard is 
the appropriate champion model.

### Planned — SHAP Explainability (Phase 2)
SHAP (SHapley Additive exPlanations) will be added in Phase 2 to explain 
individual XGBoost predictions. This will answer two key questions:
- Why was this specific borrower flagged as high risk?
- Is the model using features in a sensible, expected direction?

---

## 🔧 Tech Stack

**Language:** Python 3.9 (ARM64 native on Apple Silicon)

| Category | Libraries |
|---|---|
| Data processing | `pandas`, `numpy` |
| Machine learning | `scikit-learn`, `xgboost` |
| WoE / IV binning | `optbinning` |
| Visualisation | `matplotlib`, `seaborn` |
| Model serialisation | `joblib` |
| Data storage | `pyarrow` (parquet format) |
| Data acquisition | `kagglehub` |
| Version control | `git`, GitHub |

---

## Project Structure

```
credit_scorecard_and_default_risk/
├── data/
│   ├── raw/          # Raw CSV from Kaggle (not committed — run fetch_data.py)
│   └── clean/        # WoE-encoded parquet files (not committed — run pipeline)
├── scripts/
│   ├── fetch_data.py            # Downloads Home Credit dataset from Kaggle
│   ├── clean_data.py            # Cleaning, imputation, feature engineering
│   ├── EDA.py                   # Feature importance, decile plots, heatmaps
│   ├── feature_engineering.py   # WoE/IV binning and feature selection
│   ├── model_training.py        # LR Scorecard + XGBoost training
│   ├── champion_challenger.py   # A/B model comparison and recommendation
│   └── helper.py                # Shared utilities, paths, model I/O
├── visuals/
│   ├── EDA/                     # Feature importance, distribution, heatmap plots
│   ├── score_distribution/      # Score histogram by target class
│   └── model_evaluation/        # ROC, KS, metric comparison plots
├── models/                      # Saved model artifacts (not committed)
├── LICENSE
├── README.md
└── .gitignore
```

---

## 👨‍💻 Author

Ashutosh Dubal  
🔗 [GitHub Profile](https://github.com/Ashutosh-Dubal)

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).