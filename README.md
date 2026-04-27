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

TODO: clean up - 1. Challenges and Learning section
                 2. IV results
LAST_TODO: 1. double check project structure
           2. Add all links to Table of content 

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

-> After cleaning the data I had just started the EDA when I realized that there are way too many variables(looking at the data description).
   So I needed a way to reduce the number of variables, I started to work on top 10 features that would contribute to helping us predict weather
   or not someone would default on a loan or not. There are sevreal methods we can use for feature selection, 3 broad classifications are
   Statistical methods, ML methods and Credit Risk Specific. I used XGBoost, Random Forest and Permutaion importance all three of these fall 
   ML methods. In ML we have a couple more like Recursive feature elimination(RFE), LASSO regularisation and SHAP values In this project we will not 
   use RFE and LASSO since this is the wrong use case for RFE and LASSO will not be able to see non-linear relationships. SHAP values on the
   other hand will be more useful for after we build a model espeacially with XGBoost model. It would help us answer 2 very important questions
   Explain why individual borrowers were flagged as high risk and Validate that the model is using features in a sensible direction. Now the 

-> The 2 sets of variables we got at feature selection step can be put into one list together with the help of a ranking system since we cant
   directly compare the importance scroes we get from XGBoost method and Random Forest + permutaion importance method

---

## 👁 Key Insights & Analysis

### Exploratory Data Analysis (EDA)

#### Graph 1 — Target Distribution
   The story here is simple but critical — our dataset is heavily imbalanced. Out of 307,511 borrowers, only 24,825 defaulted. That's roughly 1 in 12. 
   This tells you that the majority of people in this dataset are reliable borrowers, and your model has far fewer examples of the "bad" behaviour 
   it needs to detect. This is why accuracy alone is a misleading metric — a model that just says "everyone is fine" would be right 92% of the time 
   without learning anything useful.

   ![Target Distribution](/visuals/EDA/01_target_distribution.png)

#### Graph 2 — RF vs XGBoost Importance
   This is the casting call for our story. Both models unanimously agree that EXT_SOURCE_2 and EXT_SOURCE_3 are the two most important characters. 
   Everything else is a supporting cast. The gap between the external scores and the rest of the features is massive — look at how the RF bars drop  
   off a cliff after EXT_SOURCE_3. This tells you that whoever these external credit bureaus are, they've already done a lot of the risk assessment 
   work for us.

   ![RF vs XGB](/visuals/EDA/02_rf_vs_xgb_importance.png)

#### Graph 3 — Box Plots
   This is where the story starts getting personal. Look at EXT_SOURCE_2 and EXT_SOURCE_3 specifically. The box for non-defaulters (0) sits noticeably 
   higher than the box for defaulters (1) — meaning people who repaid their loans consistently had higher external credit scores. The separation is clear 
   and clean. Now look at DAYS_EMPLOYED — both boxes are negative because it's stored as days (negative numbers mean further in the past). The 
   defaulters box is shifted closer to zero meaning they had been employed for less time. Shorter employment history correlates with higher default risk. 
   DEF_30_CNT_SOCIAL_CIRCLE tells an interesting social story — it barely moves between defaulters and non-defaulters, suggesting that knowing someone 
   who defaulted in the last 30 days isn't as predictive as the other features.

   ![Boxplots](/visuals/EDA/03_boxplots_by_target.png)

#### Graph 4 — Bad Rate by Decile
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

   ![Bad Rate](/visuals/EDA/04_bad_rate_by_decile.png)

#### Graph 5 — Normalised Mean by Target
   This is your quick summary card. For every feature it shows whether defaulters or non-defaulters have a higher average value. The key insights:

   EXT_SOURCE_2 and EXT_SOURCE_3 — non-defaulters (blue) are clearly taller, confirming higher scores = lower risk.

   DAYS_ID_PUBLISH and DAYS_EMPLOYED — both go negative because they're stored as negative days. The defaulters bar being less negative for DAYS_EMPLOYED means 
   defaulters had been employed for fewer days on average — shorter job tenure = higher risk.

   DEF_30_CNT_SOCIAL_CIRCLE and AMT_ANNUITY — defaulters have slightly higher values, meaning higher social circle defaults and higher loan repayment amounts 
   correlate with default.

   ![Normalised Mean](/visuals/EDA/05_mean_by_target.png)

#### Graph 6 — Cumulative Bad Rate
   This is the preview of your model's power before you've even built it. Think of it this way — imagine you lined up all 307K borrowers ranked from lowest to 
   highest EXT_SOURCE_2 score. If you stopped at the worst 20% of borrowers, how many of the total defaults would you have captured?

   For EXT_SOURCE_2 — the curve bows way above the diagonal. By the time you've looked at the riskiest 40% of borrowers you've already captured about 60% of 
   all defaults. That's a strong signal. The shaded area between the curve and the diagonal is essentially a visual preview of your Gini coefficient.

   For FLAG_OWN_CAR — the curve barely moves from the diagonal. Owning a car is almost random in terms of predicting default. This is telling you this feature 
   may not survive IV filtering.

   For DAYS_EMPLOYED — the curve goes below the diagonal at first then crosses back. That U-shape means it's predictive but in a non-linear way — the relationship 
   reverses at some point, which is exactly why WoE binning needs to handle it carefully.

   ![Cumulative Bad Rate](/visuals/EDA/06_cumulative_bad_rate.png)

#### Graph 7 — Correlation Heatmap
   Two things jump out. First ANNUITY_TO_INCOME and AMT_ANNUITY have a correlation of 0.51 with each other — that's moderate multicollinearity. They're both 
   measuring something about loan repayment burden relative to income, so they're telling a similar story. IV filtering will help decide which one to keep. 
   Second AMT_INCOME_TOTAL and AMT_ANNUITY also correlate at 0.48 — again related concepts. None of these are dangerously high (above 0.70 would be a problem) 
   but worth noting.

   ![Correlation Heatmap](/visuals/EDA/07_correlation_heatmap.png)

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

---

## 🧩 Model Interpretation

---

## 🔧 Tech Stack

This project is built with: Python 3.9

Data Processing & Analysis: pandas, numpy

Visualization: matplotlib, seaborn

Modeling (planned): scikit-learn (Random Forest, Logistic Regression, Gradient Boosting, etc.)

Project Organization: git & GitHub for version control

---

## Project Structure

```
credit-scorecard-default-risk/
├── data/
│   ├── raw/          # 
│   └── clean/        # 
├── scripts/
│   ├── fetch_data.py            # 
│   ├── clean_data.py            # 
│   ├── EDA.py                   # 
│   ├── feature_engineering.py   # 
│   ├── model_training.py        # 
│   ├── champion_challenger.py   # 
│   └── helper.py                # 
├── visuals/
│   ├── EDA/                     # 
│   ├── score_distribution/      #
│   └── model_evaluation/        # 
├── models/                      # 
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