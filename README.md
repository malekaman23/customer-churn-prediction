# Customer Churn Prediction

Predicting which telecom customers are likely to cancel their subscription, using machine learning — so businesses can act before customers leave.

## Problem
Customer retention is far cheaper than acquisition. This project builds a model that flags at-risk customers early, and identifies *why* they churn.

## Dataset
IBM Telco Customer Churn dataset — 7,043 customers, 21 features (demographics, contract, billing, services).

## Approach
1. Exploratory Data Analysis (EDA) — found key churn drivers
2. Data Cleaning & Feature Engineering — encoding, handling missing values
3. Model Training — Logistic Regression, Random Forest, XGBoost
4. Evaluation — prioritized Recall & ROC-AUC (catching churners matters more than false alarms)
5. Deployment — interactive Streamlit web app for live predictions

## Results

| Model | Recall (Churn) | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.78 | 0.842 |
| Random Forest | 0.78 | 0.844 |
| XGBoost | 0.79 | 0.839 |

## Key Churn Drivers
1. Contract type (month-to-month = highest risk)
2. Low tenure (new customers churn more)
3. High monthly charges
4. Fiber optic internet service

## Tech Stack
Python · pandas · scikit-learn · XGBoost · Streamlit

## How to Run
```
pip install -r requirements.txt
streamlit run app.py
```