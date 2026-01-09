
import pandas as pd
import streamlit as st
import os
import joblib


st.set_page_config(
    page_title="Loan Approval Checker",
    page_icon="🏦",
    layout="centered"
)

# Title
st.title("🏦 Loan Approval Checker")

MODEL_PATH = "loan_model.joblib"

# 2. Check if the model exists (as required in the project PDF)
if not os.path.exists(MODEL_PATH):
    st.warning("The system is initializing, please wait")
    st.error("The model file was not found. Please train the model and save it before running the app.")
    st.stop()


@st.cache_resource
def load_model(path: str):
    """Loads the trained model pipeline."""
    return joblib.load(path)

# Load the model
model = load_model(MODEL_PATH)

# ---- Input form (screenshot-friendly UI) ----
with st.form("loan_form"):
    st.subheader("Loan Application Details")

    applicant_income = st.number_input(
        "Applicant Income (monthly)",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income (monthly)",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    loan_amount = st.number_input(
        "Requested Loan Amount",
        min_value=0.0,
        value=150.0,
        step=10.0
    )

    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=1.0,
        value=360.0,
        step=12.0
    )

    credit_history = st.selectbox(
        "Credit History",
        options=[1, 0],
        index=0,
        format_func=lambda x: "Exists (1)" if x == 1 else "Does not exist (0)"
    )


    submitted = st.form_submit_button("Check Loan Eligibility")



    # IMPORTANT:
    # Column names must exactly match those used during model training
    X = pd.DataFrame([{
        "ApplicantIncome": float(applicant_income),
        "CoapplicantIncome": float(coapplicant_income),
        "LoanAmount": float(loan_amount),
        "Loan_Amount_Term": float(loan_term),
        "Credit_History": int(credit_history),

    }])

    # Prediction
    prediction = model.predict(X)

    if prediction[0] == 1:
        st.success("✅ Loan APPROVED!")
    else:
        st.error("❌ Loan REJECTED.")
