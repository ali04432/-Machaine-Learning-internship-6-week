# ==============================================================================
# WEEK 5 - TASK 2: DEPLOY YOUR MODEL AS A LIVE WEB APP (STREAMLIT)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🤖",
    layout="centered"
)

# Title & Description
st.title("🤖 Customer Churn Prediction Web App")
st.write("A Machine Learning powered web dashboard to predict whether a customer will leave or stay.")
st.markdown("---")

# ------------------------------------------------------------------------------
# TRAIN DEMO MODEL (Cached for Performance)
# ------------------------------------------------------------------------------
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1000
    
    # Synthetic dataset
    age = np.random.randint(18, 70, size=n)
    tenure = np.random.randint(1, 10, size=n)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n)
    support_calls = np.random.randint(0, 8, size=n)
    
    # Simple logic for churn label
    churn = ((monthly_charges > 75) & (support_calls > 3) | (tenure < 2)).astype(int)
    
    X = pd.DataFrame({
        'Age': age,
        'Tenure_Years': tenure,
        'Monthly_Charges': monthly_charges,
        'Support_Calls': support_calls
    })
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, churn)
    return model

model = train_model()

# ------------------------------------------------------------------------------
# SIDEBAR / INPUT FORM
# ------------------------------------------------------------------------------
st.sidebar.header("📋 Enter Customer Details")

age_input = st.sidebar.slider("Age", min_value=18, max_value=80, value=35)
tenure_input = st.sidebar.slider("Tenure (Years)", min_value=1, max_value=10, value=3)
monthly_input = st.sidebar.slider("Monthly Charges ($)", min_value=20.0, max_value=150.0, value=65.0)
calls_input = st.sidebar.number_input("Customer Support Calls", min_value=0, max_value=10, value=2)

# Input DataFrame
input_data = pd.DataFrame({
    'Age': [age_input],
    'Tenure_Years': [tenure_input],
    'Monthly_Charges': [monthly_input],
    'Support_Calls': [calls_input]
})

# Display User Input
st.subheader("Customer Input Profile:")
st.dataframe(input_data)

# ------------------------------------------------------------------------------
# PREDICTION LOGIC
# ------------------------------------------------------------------------------
if st.button("🚀 Predict Customer Churn", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    st.markdown("---")
    st.subheader("Prediction Result:")
    
    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn! (Probability: {probabilities[1]*100:.1f}%)")
        st.write("Recommendation: Offer a discount or personalized support plan.")
    else:
        st.success(f"✅ Low Risk - Customer is likely to stay. (Confidence: {probabilities[0]*100:.1f}%)")