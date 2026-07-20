# Customer Churn Prediction — Telecom Industry

Predicting which customers are likely to cancel their subscription, so the business can act **before** they leave — not after.

🔗 **Live Demo:** [https://customer-churn-prediction-m.streamlit.app/](https://customer-churn-prediction-m.streamlit.app/) — try the Quick Demo presets, or upload a CSV to batch-score a customer list.

---

## Business Problem

Customer acquisition costs 5–7x more than retention. This project builds an end-to-end machine learning system that:
- Flags at-risk customers early, before they churn
- Quantifies the revenue at stake
- Identifies *why* customers churn — not just who — so the business can act on root causes, not just symptoms

## Dataset

IBM Sample Telco Customer Churn dataset — 7,043 customers, 21 features covering demographics, account details, subscribed services, and billing. ~26.5% churn rate (moderately imbalanced).

## Approach

1. **EDA** — explored churn patterns across contract type, tenure, monthly charges, and internet service
2. **SQL Analysis** — replicated key business insights using SQL (aggregations, `CASE WHEN`, `HAVING`, window functions) against a SQLite database, to demonstrate query-based analysis alongside pandas
3. **Data Cleaning** — fixed data type issues (e.g. `TotalCharges` stored as text with hidden blanks), handled missing values, encoded categorical features
4. **Modeling** — trained and compared Logistic Regression, Random Forest, and XGBoost
5. **Validation** — used 5-fold stratified cross-validation to confirm model stability (not just a single lucky train/test split)
6. **Hyperparameter Tuning** — GridSearchCV over 27 parameter combinations to find the best-performing XGBoost configuration
7. **Evaluation** — prioritized Recall and ROC-AUC over raw Accuracy (missing a churner costs the business more than a false alarm)
8. **Interpretation** — feature importance to identify actionable churn drivers
9. **Deployment** — interactive Streamlit web app with single-customer prediction, batch CSV scoring, and an insights dashboard

## Results

**Model comparison (initial, default hyperparameters):**

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.74 | 0.50 | 0.78 | 0.61 | 0.842 |
| Random Forest | 0.76 | 0.53 | 0.78 | 0.63 | 0.844 |
| XGBoost | 0.75 | 0.52 | 0.79 | 0.63 | 0.839 |

**Cross-validation (5-fold, XGBoost):** Mean ROC-AUC = 0.843 (std = 0.011) — confirms the model performs consistently across different data splits, not just one favorable split.

**After hyperparameter tuning (GridSearchCV, `learning_rate=0.05, max_depth=3, n_estimators=200`):**

| Metric | Before Tuning | After Tuning |
|---|---|---|
| Recall (Churn) | 0.79 | **0.81** |
| ROC-AUC (test) | 0.839 | **0.844** |

**Final model:** Tuned XGBoost — catches ~81% of actual churners (Recall), which is the metric that matters most for a retention use case, where missing a churner is costlier than a false alarm.

## Key Drivers of Churn

1. **Contract type** — month-to-month customers churn at 42.7%, vs. 11.3% (one-year) and 2.8% (two-year)
2. **Tenure** — churn risk is highest in the first few months; the highest-risk segment (month-to-month + tenure ≤ 6 months) churns at 55.2%, more than double the overall average
3. **Monthly charges** — higher bills correlate with higher churn
4. **Payment method** — Electronic check users churn at 45.3%, the highest of any payment method
5. **Internet service (Fiber)** — fiber customers churn more than DSL, likely a price/quality signal
6. **Missing add-ons** (Online Security, Tech Support) — customers without these churn more

## Business Recommendations

- Incentivize longer-term contracts for month-to-month customers (discount for annual upgrade)
- Build a structured first-90-days onboarding/engagement program — this is where churn risk peaks
- Investigate the Electronic check → high churn correlation (payment friction? demographic proxy?)
- Review fiber pricing/service quality
- Promote Online Security & Tech Support as retention tools
- Deploy the model monthly, route highest-risk + highest-value customers to a retention team

## App Features

- **Single Prediction** — fill in a customer profile (or use a one-click Risky/Loyal example), get a churn probability gauge, key risk factors, and a downloadable report
- **Batch Prediction** — upload a CSV of customers, get churn scores for all of them, sorted by risk, with a downloadable results file
- **Model Insights** — model comparison table, EDA charts, and business insights, all inside the app

## Assumptions & Limitations

- Trained on a single snapshot of data (no time dimension) — in production, this would need periodic retraining as customer behavior shifts
- The 50% classification threshold is a default; in practice, the threshold should be tuned based on the actual cost of a retention offer vs. the value of a retained customer
- Precision (Churn) is ~50-53%, meaning roughly half of flagged customers would not have actually churned — acceptable for a low-cost retention outreach, but should be considered if retention actions are expensive
- Dataset is a public benchmark, not live production data — real deployment would require ongoing data pipeline integration and monitoring for model drift

## Files

- `churn_prediction.ipynb` — full analysis notebook (EDA → SQL → modeling → tuning → business insights)
- `churn_prediction.html` — same notebook, viewable in any browser without Jupyter
- `app.py` — Streamlit web app (single + batch prediction, insights dashboard)
- `churn_model.pkl` / `model_columns.pkl` — final tuned model and feature schema
- `churn_database.db` — SQLite database used for the SQL analysis section
- `outputs/` — all charts generated during analysis
- `data/telco_churn.csv` — source dataset

## Tech Stack

Python · pandas · scikit-learn · XGBoost · SQLite/SQL · Streamlit · Plotly · matplotlib · seaborn

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook (EDA, SQL, modeling, tuning)
jupyter nbconvert --to notebook --execute --inplace churn_prediction.ipynb

# Run the web app
streamlit run app.py
```

### Run with Docker (alternative)
```bash
docker build -t churn-predictor .
docker run -p 8501:8501 churn-predictor
```
Then open http://localhost:8501