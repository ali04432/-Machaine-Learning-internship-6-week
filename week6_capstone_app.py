# ==============================================================================
# WEEK 6 - CAPSTONE: LIVE WEB APPLICATION DASHBOARD
# ==============================================================================

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Real-Estate House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# Load Saved Artifacts
@st.cache_resource
def load_pipeline():
    if os.path.exists('capstone_model.joblib'):
        return joblib.load('capstone_model.joblib')
    return None

artifacts = load_pipeline()

st.title("🏡 Capstone Project: Real-Estate Price Predictor")
st.write("An End-to-End Machine Learning System built with Gradient Boosting Regression.")
st.markdown("---")

if artifacts is None:
    st.error("⚠️ Model file 'capstone_model.joblib' not found! Please run 'python week6_capstone_train.py' first.")
else:
    model = artifacts['model']
    scaler = artifacts['scaler']
    imputer = artifacts['imputer']

    # Sidebar Inputs
    st.sidebar.header("⚙️ Enter Property Details")

    sqft_input = st.sidebar.slider("Square Feet Area", 500, 5000, 2000, step=50)
    bedrooms_input = st.sidebar.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5], index=2)
    bathrooms_input = st.sidebar.selectbox("Number of Bathrooms", [1, 2, 3, 4], index=1)
    age_input = st.sidebar.slider("House Age (Years)", 0, 40, 5)
    dist_input = st.sidebar.slider("Distance to City Center (km)", 1.0, 35.0, 8.0, step=0.5)

    # Input Summary Table
    input_df = pd.DataFrame({
        'Square_Feet': [sqft_input],
        'Bedrooms': [bedrooms_input],
        'Bathrooms': [bathrooms_input],
        'House_Age': [age_input],
        'Distance_City_Center': [dist_input]
    })

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Property Features Summary")
        st.dataframe(input_df, use_container_width=True)

    with col2:
        st.subheader("🔮 Estimated Price Output")
        
        if st.button("🚀 Calculate Estimated Market Value", use_container_width=True):
            # Preprocess inputs
            input_imputed = imputer.transform(input_df)
            input_scaled = scaler.transform(input_imputed)
            
            # Predict
            predicted_price = model.predict(input_scaled)[0]

            st.success(f"### 💵 Valuation: **${predicted_price:,.2f}**")
            
            # Insights
            st.info("""
            **Market Insights:**
            - Larger area and more bathrooms increase valuation significantly.
            - Higher distance from city center and older age lower market price.
            """)