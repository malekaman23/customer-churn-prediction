import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import time
from datetime import datetime

# ---------------- Page Config ----------------
st.set_page_config(page_title="Churn Predictor", layout="wide")

# ---------------- Load Model ----------------
model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# ---------------- Defaults / Presets ----------------
DEFAULTS = {
    "tenure": 12, "monthly_charges": 70.0, "total_charges": 840.0,
    "contract": "Month-to-month", "paperless_billing": "No",
    "payment_method": "Electronic check", "gender": "Male",
    "senior_citizen": "No", "partner": "No", "dependents": "No",
    "phone_service": "Yes", "multiple_lines": "No",
    "internet_service": "DSL", "online_security": "No",
    "online_backup": "No", "device_protection": "No", "tech_support": "No",
    "streaming_tv": "No", "streaming_movies": "No",
}

PRESETS = {
    "risky": {
        "tenure": 2, "monthly_charges": 95.0, "total_charges": 190.0,
        "contract": "Month-to-month", "paperless_billing": "Yes",
        "payment_method": "Electronic check", "gender": "Female",
        "senior_citizen": "No", "partner": "No", "dependents": "No",
        "phone_service": "Yes", "multiple_lines": "Yes",
        "internet_service": "Fiber optic", "online_security": "No",
        "online_backup": "No", "device_protection": "No", "tech_support": "No",
        "streaming_tv": "Yes", "streaming_movies": "Yes",
    },
    "loyal": {
        "tenure": 60, "monthly_charges": 45.0, "total_charges": 2700.0,
        "contract": "Two year", "paperless_billing": "No",
        "payment_method": "Bank transfer (automatic)", "gender": "Male",
        "senior_citizen": "No", "partner": "Yes", "dependents": "Yes",
        "phone_service": "Yes", "multiple_lines": "No",
        "internet_service": "DSL", "online_security": "Yes",
        "online_backup": "Yes", "device_protection": "Yes", "tech_support": "Yes",
        "streaming_tv": "No", "streaming_movies": "No",
    }
}

def apply_preset(name):
    for key, value in PRESETS[name].items():
        st.session_state[key] = value

def reset_form():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value

# ---------------- Header ----------------
st.title("Customer Churn Prediction")
st.markdown("Predict whether a telecom customer is likely to churn, based on their profile and service usage.")
st.caption("Tip: use the ⋮ menu (top-right) → Settings → Theme to switch between Light and Dark mode.")
st.divider()

# ---------------- Sidebar ----------------
st.sidebar.header("Quick Demo")
pc1, pc2 = st.sidebar.columns(2)
pc1.button("Risky Example", on_click=apply_preset, args=("risky",), use_container_width=True,
           help="Fill the form with a high-risk customer profile (new, month-to-month, fiber)")
pc2.button("Loyal Example", on_click=apply_preset, args=("loyal",), use_container_width=True,
           help="Fill the form with a low-risk, long-tenure customer profile")
st.sidebar.button("Reset Form", on_click=reset_form, use_container_width=True)
st.sidebar.divider()

st.sidebar.header("Customer Details")

with st.sidebar.expander("Account Info", expanded=True):
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12, key="tenure",
                        help="How many months has this customer been with the company?")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0, key="monthly_charges")
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0, key="total_charges")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract",
                             help="Month-to-month customers historically churn ~15x more than 2-year contract customers")
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], key="paperless_billing")
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key="payment_method")

with st.sidebar.expander("Demographics", expanded=False):
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior_citizen")
    partner = st.selectbox("Partner", ["No", "Yes"], key="partner")
    dependents = st.selectbox("Dependents", ["No", "Yes"], key="dependents")

with st.sidebar.expander("Services", expanded=False):
    phone_service = st.selectbox("Phone Service", ["No", "Yes"], key="phone_service")
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="multiple_lines")
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet_service")
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="online_security")
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="online_backup")
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key="device_protection")
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="tech_support")
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="streaming_tv")
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="streaming_movies")

predict_clicked = st.sidebar.button("Predict Churn", use_container_width=True, type="primary")

# ---------------- Shared preprocessing function ----------------
def build_feature_row(row):
    """row: dict-like with raw field values -> returns one-hot encoded dict matching model_columns"""
    return {
        'SeniorCitizen': 1 if row['senior_citizen'] == "Yes" else 0,
        'tenure': row['tenure'],
        'MonthlyCharges': row['monthly_charges'],
        'TotalCharges': row['total_charges'],
        'gender_Male': 1 if row['gender'] == "Male" else 0,
        'Partner_Yes': 1 if row['partner'] == "Yes" else 0,
        'Dependents_Yes': 1 if row['dependents'] == "Yes" else 0,
        'PhoneService_Yes': 1 if row['phone_service'] == "Yes" else 0,
        'MultipleLines_No phone service': 1 if row['multiple_lines'] == "No phone service" else 0,
        'MultipleLines_Yes': 1 if row['multiple_lines'] == "Yes" else 0,
        'InternetService_Fiber optic': 1 if row['internet_service'] == "Fiber optic" else 0,
        'InternetService_No': 1 if row['internet_service'] == "No" else 0,
        'OnlineSecurity_No internet service': 1 if row['online_security'] == "No internet service" else 0,
        'OnlineSecurity_Yes': 1 if row['online_security'] == "Yes" else 0,
        'OnlineBackup_No internet service': 1 if row['online_backup'] == "No internet service" else 0,
        'OnlineBackup_Yes': 1 if row['online_backup'] == "Yes" else 0,
        'DeviceProtection_No internet service': 1 if row['device_protection'] == "No internet service" else 0,
        'DeviceProtection_Yes': 1 if row['device_protection'] == "Yes" else 0,
        'TechSupport_No internet service': 1 if row['tech_support'] == "No internet service" else 0,
        'TechSupport_Yes': 1 if row['tech_support'] == "Yes" else 0,
        'StreamingTV_No internet service': 1 if row['streaming_tv'] == "No internet service" else 0,
        'StreamingTV_Yes': 1 if row['streaming_tv'] == "Yes" else 0,
        'StreamingMovies_No internet service': 1 if row['streaming_movies'] == "No internet service" else 0,
        'StreamingMovies_Yes': 1 if row['streaming_movies'] == "Yes" else 0,
        'Contract_One year': 1 if row['contract'] == "One year" else 0,
        'Contract_Two year': 1 if row['contract'] == "Two year" else 0,
        'PaperlessBilling_Yes': 1 if row['paperless_billing'] == "Yes" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if row['payment_method'] == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if row['payment_method'] == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if row['payment_method'] == "Mailed check" else 0,
    }

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["Prediction", "Batch Prediction", "Model Insights"])

# ================= TAB 1: SINGLE PREDICTION =================
with tab1:
    if predict_clicked:
        with st.spinner("Scoring customer..."):
            time.sleep(0.4)
            input_dict = build_feature_row({
                'senior_citizen': senior_citizen, 'tenure': tenure, 'monthly_charges': monthly_charges,
                'total_charges': total_charges, 'gender': gender, 'partner': partner, 'dependents': dependents,
                'phone_service': phone_service, 'multiple_lines': multiple_lines, 'internet_service': internet_service,
                'online_security': online_security, 'online_backup': online_backup,
                'device_protection': device_protection, 'tech_support': tech_support,
                'streaming_tv': streaming_tv, 'streaming_movies': streaming_movies,
                'contract': contract, 'paperless_billing': paperless_billing, 'payment_method': payment_method,
            })
            input_df = pd.DataFrame([input_dict]).reindex(columns=model_columns, fill_value=0)
            prediction = model.predict(input_df)[0]
            probability = float(model.predict_proba(input_df)[0][1])

        k1, k2, k3 = st.columns(3)
        k1.metric("Churn Probability", f"{probability*100:.1f}%")
        k2.metric("Tenure", f"{tenure} mo")
        k3.metric("Monthly Bill", f"${monthly_charges:.0f}")

        col1, col2 = st.columns([1, 1.4])

        with col1:
            gauge_color = "#E63946" if prediction == 1 else "#2E86AB"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={'suffix': "%", 'font': {'size': 40}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': gauge_color},
                    'steps': [
                        {'range': [0, 40], 'color': "#E8F4F8"},
                        {'range': [40, 70], 'color': "#FFE8A3"},
                        {'range': [70, 100], 'color': "#FFC1C1"},
                    ],
                    'threshold': {'line': {'color': "gray", 'width': 3}, 'thickness': 0.8, 'value': probability * 100}
                },
                title={'text': "Churn Probability"}
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            if prediction == 1:
                st.error("**High Churn Risk**")
                st.markdown("**Suggested action:** Prioritize this customer for a retention offer "
                             "(e.g. contract upgrade discount, loyalty pricing).")
            else:
                st.success("**Low Churn Risk**")
                st.markdown("**Suggested action:** No immediate action needed — monitor periodically.")
                st.balloons()

            report_text = f"""CUSTOMER CHURN PREDICTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

RESULT: {'HIGH CHURN RISK' if prediction == 1 else 'LOW CHURN RISK'}
Churn Probability: {probability*100:.1f}%

CUSTOMER PROFILE
- Tenure: {tenure} months
- Monthly Charges: ${monthly_charges:.2f}
- Total Charges: ${total_charges:.2f}
- Contract: {contract}
- Internet Service: {internet_service}
- Payment Method: {payment_method}

RECOMMENDED ACTION
{'Prioritize for retention offer (contract upgrade discount, loyalty pricing).' if prediction == 1 else 'No immediate action needed — monitor periodically.'}
"""
            st.download_button(
                label="Download Report", data=report_text,
                file_name=f"churn_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain", use_container_width=True
            )

        with col2:
            st.subheader("Key Risk Factors Present")
            risk_notes = []
            if contract == "Month-to-month":
                risk_notes.append("- Month-to-month contract — highest churn segment (~43% historical churn rate)")
            if tenure <= 6:
                risk_notes.append("- Low tenure (≤6 months) — new customers churn far more often")
            if internet_service == "Fiber optic":
                risk_notes.append("- Fiber optic service — historically higher churn than DSL")
            if monthly_charges > 80:
                risk_notes.append("- High monthly charges — price sensitivity increases churn risk")
            if online_security == "No" and internet_service != "No":
                risk_notes.append("- No Online Security add-on — customers without it churn more")

            if risk_notes:
                for note in risk_notes:
                    st.write(note)
            else:
                st.write("No major historical risk factors detected in this profile.")
    else:
        st.info("Fill in the customer details in the sidebar (or click a Quick Demo preset) and click **Predict Churn** to see the result.")

# ================= TAB 2: BATCH PREDICTION =================
with tab2:
    st.subheader("Score a Whole Customer List at Once")
    st.markdown("Upload a CSV with the same columns as the original Telco dataset (no `Churn` column needed) "
                "to get churn predictions for every customer in one go.")

    sample_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                   'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                   'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
    with st.expander("Show required column format"):
        st.code(", ".join(sample_cols))

    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            missing_cols = [c for c in sample_cols if c not in batch_df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                with st.spinner(f"Scoring {len(batch_df)} customers..."):
                    work_df = batch_df.copy()
                    work_df['TotalCharges'] = pd.to_numeric(work_df['TotalCharges'], errors='coerce').fillna(0)

                    encoded = pd.get_dummies(
                        work_df,
                        columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                                 'PaperlessBilling', 'PaymentMethod'],
                        drop_first=True
                    )
                    encoded = encoded.reindex(columns=model_columns, fill_value=0)

                    predictions = model.predict(encoded)
                    probabilities = model.predict_proba(encoded)[:, 1]

                    result_df = batch_df.copy()
                    result_df['Churn_Prediction'] = ["Yes" if p == 1 else "No" for p in predictions]
                    result_df['Churn_Probability'] = (probabilities * 100).round(1)
                    result_df = result_df.sort_values('Churn_Probability', ascending=False)

                st.success(f"Scored {len(result_df)} customers.")

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Customers", len(result_df))
                m2.metric("Predicted Churners", int((result_df['Churn_Prediction'] == "Yes").sum()))
                m3.metric("Avg. Churn Probability", f"{result_df['Churn_Probability'].mean():.1f}%")

                st.dataframe(result_df, use_container_width=True, hide_index=True)

                csv_out = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Scored CSV", data=csv_out,
                    file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv", use_container_width=True
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")

# ================= TAB 3: MODEL INSIGHTS =================
with tab3:
    st.subheader("Model Performance")
    perf_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Recall (Churn)": [0.78, 0.78, 0.79],
        "Precision (Churn)": [0.50, 0.53, 0.52],
        "ROC-AUC": [0.842, 0.844, 0.839]
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.subheader("What Drives Churn — EDA & Feature Importance")
    c1, c2 = st.columns(2)
    with c1:
        st.image("outputs/02_churn_by_contract.png", caption="Churn by Contract Type")
        st.image("outputs/04_churn_by_charges.png", caption="Churn by Monthly Charges")
    with c2:
        st.image("outputs/03_churn_by_tenure.png", caption="Churn by Tenure")
        st.image("outputs/09_feature_importance.png", caption="Top Features Driving Churn (XGBoost)")

    st.subheader("Key Business Insights")
    st.markdown("""
    - **Contract type** is the single biggest churn driver — month-to-month customers churn at ~43%, vs ~3% for two-year contracts.
    - **New customers (0-10 months tenure)** are at the highest risk — a structured onboarding program could reduce early churn.
    - **Higher monthly charges** correlate with higher churn — worth reviewing pricing for high-usage segments.
    - **Fiber optic customers** churn more than DSL customers, suggesting a possible price or service-quality gap.
    """)

st.divider()
st.caption("Built with Python, scikit-learn, XGBoost & Streamlit")