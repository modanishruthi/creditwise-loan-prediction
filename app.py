import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="CreditWise", page_icon="💳", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    body { background-color: #0f172a; }
    .stApp { background-color: #0f172a; color: white; }
    .stSelectbox label, .stNumberInput label { color: #94a3b8 !important; font-size: 14px; }
    
    /* Fix invisible input text */
    input { color: white !important; background-color: #1e293b !important; }
    .stNumberInput input { color: white !important; }
    .stSelectbox div[data-baseweb="select"] { background-color: #1e293b !important; color: white !important; }
    
    div[data-testid="stMarkdownContainer"] p { color: #94a3b8; }
    .stButton button {
        background-color: #2563eb;
        color: white;
        border: none;
        padding: 12px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
    }
    .stButton button:hover { background-color: #1d4ed8; }
    .risk-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 15px 20px;
        margin: 8px 0;
        border-left: 4px solid;
    }
    .risk-good { border-color: #22c55e; }
    .risk-bad { border-color: #ef4444; }
    .risk-mid { border-color: #f59e0b; }
    </style>
""", unsafe_allow_html=True)
# Load model
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("ohe.pkl")

# Header
st.markdown("<h1 style='text-align:center; color:#2563eb;'>💳 CreditWise</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8;'>AI-Powered Loan Approval Prediction</p>", unsafe_allow_html=True)
st.markdown("---")

# Input form
st.markdown("### 📋 Applicant Details")
col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=50000)
    coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0)
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
    existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=0)
    savings = st.number_input("Savings (₹)", min_value=0, value=20000)

with col2:
    collateral_value = st.number_input("Collateral Value (₹)", min_value=0, value=50000)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=100000)
    loan_term = st.number_input("Loan Term (months)", min_value=6, max_value=360, value=60)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    dti_ratio = st.number_input("DTI Ratio (0 to 1)", min_value=0.0, max_value=1.0, value=0.3)
    education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])

st.markdown("### 🏢 Employment & Loan Info")
col3, col4 = st.columns(2)

with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    employment = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])

with col4:
    loan_purpose = st.selectbox("Loan Purpose", ["Car", "Education", "Home", "Personal", "Business"])
    property_area = st.selectbox("Property Area", ["Semiurban", "Urban", "Rural"])
    employer_cat = st.selectbox("Employer Category", ["Government", "MNC", "Private", "Unemployed"])

st.markdown("---")

if st.button("🔍 Predict Loan Approval"):

    # Feature engineering
    dti_sq = dti_ratio ** 2
    credit_sq = credit_score ** 2

    # Education encoding
    edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    edu_encoded = edu_map[education]

    # OHE
    cat_df = pd.DataFrame([[employment, marital_status, loan_purpose, property_area, gender, employer_cat]],
                          columns=["Employment_Status", "Marital_Status", "Loan_Purpose",
                                   "Property_Area", "Gender", "Employer_Category"])
    cat_encoded = ohe.transform(cat_df)
    cat_df_encoded = pd.DataFrame(cat_encoded,
                                  columns=ohe.get_feature_names_out(
                                      ["Employment_Status", "Marital_Status", "Loan_Purpose",
                                       "Property_Area", "Gender", "Employer_Category"]))

    # Numerical
    num_df = pd.DataFrame([[applicant_income, coapplicant_income, age, dependents, existing_loans,
                            savings, collateral_value, loan_amount, loan_term, edu_encoded]],
                          columns=["Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
                                   "Existing_Loans", "Savings", "Collateral_Value", "Loan_Amount",
                                   "Loan_Term", "Education_Level"])

    # Combine
    final_df = pd.concat([num_df, cat_df_encoded], axis=1)
    final_df["DTI_Ratio_sq"] = dti_sq
    final_df["Credit_Score_sq"] = credit_sq

    # Predict
    final_scaled = scaler.transform(final_df)
    prediction = model.predict(final_scaled)[0]
    probability = model.predict_proba(final_scaled)[0][1]
    confidence = probability * 100

    # Result
    st.markdown("---")
    st.markdown("### 📊 Prediction Result")

    # Gauge chart
    color = "#22c55e" if prediction == 1 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"color": "white", "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": color},
            "bgcolor": "#1e293b",
            "steps": [
                {"range": [0, 40], "color": "#450a0a"},
                {"range": [40, 60], "color": "#422006"},
                {"range": [60, 100], "color": "#052e16"}
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": 60
            }
        },
        title={"text": "Approval Confidence", "font": {"color": "white"}}
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        font={"color": "white"},
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

    if prediction == 1:
        st.markdown("<h2 style='text-align:center; color:#22c55e;'>✅ Loan Approved!</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align:center; color:#ef4444;'>❌ Loan Rejected</h2>", unsafe_allow_html=True)

    # Risk factor analysis
    st.markdown("### 🔎 Risk Factor Analysis")

    # Credit Score
    if credit_score >= 700:
        st.markdown(f"<div class='risk-card risk-good'>✅ <b>Credit Score: {credit_score}</b> — Good credit history</div>", unsafe_allow_html=True)
    elif credit_score >= 600:
        st.markdown(f"<div class='risk-card risk-mid'>⚠️ <b>Credit Score: {credit_score}</b> — Average, may affect approval</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='risk-card risk-bad'>❌ <b>Credit Score: {credit_score}</b> — Low credit score is a red flag</div>", unsafe_allow_html=True)

    # DTI Ratio
    if dti_ratio <= 0.3:
        st.markdown(f"<div class='risk-card risk-good'>✅ <b>DTI Ratio: {dti_ratio}</b> — Healthy debt-to-income ratio</div>", unsafe_allow_html=True)
    elif dti_ratio <= 0.5:
        st.markdown(f"<div class='risk-card risk-mid'>⚠️ <b>DTI Ratio: {dti_ratio}</b> — Moderate, manageable debt</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='risk-card risk-bad'>❌ <b>DTI Ratio: {dti_ratio}</b> — High debt burden detected</div>", unsafe_allow_html=True)

    # Savings vs Loan
    if savings >= loan_amount * 0.2:
        st.markdown(f"<div class='risk-card risk-good'>✅ <b>Savings:</b> Adequate savings relative to loan amount</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='risk-card risk-bad'>❌ <b>Savings:</b> Low savings compared to loan amount</div>", unsafe_allow_html=True)

    # Existing loans
    if existing_loans == 0:
        st.markdown(f"<div class='risk-card risk-good'>✅ <b>Existing Loans:</b> No existing loans — low risk</div>", unsafe_allow_html=True)
    elif existing_loans <= 2:
        st.markdown(f"<div class='risk-card risk-mid'>⚠️ <b>Existing Loans: {existing_loans}</b> — Moderate liability</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='risk-card risk-bad'>❌ <b>Existing Loans: {existing_loans}</b> — High number of existing loans</div>", unsafe_allow_html=True)