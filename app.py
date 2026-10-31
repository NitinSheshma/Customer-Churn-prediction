from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "churn_model.joblib"

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction")
st.markdown("Predict customer churn risk using the trained Random Forest model.")

if not MODEL_PATH.exists():
    st.error("The trained model is missing. Run `python churn_prediction.py --quick` first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

with st.sidebar:
    st.header("Customer details")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly = st.number_input("Monthly charges", min_value=0.0, max_value=500.0, value=70.0)
    total = st.number_input("Total charges", min_value=0.0, max_value=100000.0, value=float(monthly * tenure))


def tenure_group(months: int) -> str:
    if months <= 12:
        return "0-12m"
    if months <= 24:
        return "13-24m"
    if months <= 48:
        return "25-48m"
    if months <= 60:
        return "49-60m"
    return "60m+"


sample = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": contract,
    "PaperlessBilling": "Yes",
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
}])
sample["AvgMonthlyCharge"] = np.where(
    sample["tenure"] > 0, sample["TotalCharges"] / sample["tenure"], sample["MonthlyCharges"]
)
sample["TenureGroup"] = sample["tenure"].apply(tenure_group)

col1, col2 = st.columns([1, 2])
with col1:
    if st.button("Predict churn", type="primary", use_container_width=True):
        probability = float(model.predict_proba(sample)[0, 1])
        prediction = "Churn" if probability >= 0.5 else "Stay"
        st.metric("Churn probability", f"{probability:.1%}")
        if prediction == "Churn":
            st.error("This customer is likely to churn.")
        else:
            st.success("This customer is likely to stay.")

with col2:
    st.subheader("Model performance")
    comparison = BASE_DIR / "outputs" / "model_comparison.csv"
    if comparison.exists():
        st.dataframe(pd.read_csv(comparison), use_container_width=True, hide_index=True)

with st.expander("Analyze the complete CSV dataset"):
    uploaded = st.file_uploader("Upload the Telco Customer Churn CSV", type=["csv"])
    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
            if "customerID" in batch.columns:
                batch = batch.drop(columns="customerID")
            batch["TotalCharges"] = pd.to_numeric(batch["TotalCharges"], errors="coerce").fillna(0)
            batch["AvgMonthlyCharge"] = np.where(
                batch["tenure"] > 0,
                batch["TotalCharges"] / batch["tenure"].replace(0, np.nan),
                batch["MonthlyCharges"],
            )
            batch["TenureGroup"] = pd.cut(
                batch["tenure"], bins=[-1, 12, 24, 48, 60, np.inf],
                labels=["0-12m", "13-24m", "25-48m", "49-60m", "60m+"],
            ).astype(str)
            if "Churn" in batch.columns:
                batch = batch.drop(columns="Churn")
            probabilities = model.predict_proba(batch)[:, 1]
            results = batch.copy()
            results["ChurnProbability"] = probabilities
            results["Prediction"] = np.where(probabilities >= 0.5, "Churn", "Stay")
            st.dataframe(results.head(100), use_container_width=True)
            st.download_button(
                "Download predictions",
                results.to_csv(index=False).encode("utf-8"),
                "churn_predictions.csv",
                "text/csv",
            )
        except Exception as exc:
            st.error(f"Could not process the file: {exc}")
