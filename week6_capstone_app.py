import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Real-Estate Price Predictor Pro",
    page_icon="🏡",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #334155;
        text-align: center;
    }
    .badge {
        padding: 6px 12px;
        border-radius: 15px;
        font-weight: bold;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    try:
        return joblib.load('capstone_model.joblib')
    except Exception as e:
        return None

artifacts = load_artifacts()

# Header Section
st.title("🏡 Real-Estate Valuation & Investment Dashboard")
st.caption("AI-Powered Property Price Prediction with Financial Insights")
st.markdown("---")

# Sidebar - User Inputs
st.sidebar.header("⚙️ Property Parameters")

sqft = st.sidebar.slider("Square Feet Area", 500, 5000, 1500, step=50)
bedrooms = st.sidebar.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
bathrooms = st.sidebar.selectbox("Bathrooms", [1, 2, 3, 4, 5], index=1)
house_age = st.sidebar.slider("House Age (Years)", 0, 50, 5)
distance = st.sidebar.slider("Distance to City Center (km)", 0.5, 30.0, 5.0, step=0.5)

# Main Dashboard Layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📋 Input Summary")
    
    input_data = pd.DataFrame({
        'Square_Feet': [sqft],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'House_Age': [house_age],
        'Distance_City_Center': [distance]
    })
    
    st.dataframe(input_data, use_container_width=True)

    # Calculate Prediction
    if st.button("🔮 Calculate Market Valuation", type="primary", use_container_width=True):
        if artifacts is not None:
            model = artifacts['model']
            scaler = artifacts['scaler']
            imputer = artifacts['imputer']

            # Transform input
            input_imputed = imputer.transform(input_data)
            input_scaled = scaler.transform(input_imputed)
            
            # Predict
            predicted_price = float(model.predict(input_scaled)[0])
            st.session_state['predicted_price'] = predicted_price
        else:
            # Fallback estimation if joblib missing
            base_price = (sqft * 150) + (bedrooms * 10000) + (bathrooms * 8000) - (house_age * 1200) - (distance * 2500)
            st.session_state['predicted_price'] = max(base_price, 30000)

with col2:
    st.subheader("💰 Valuation Output")
    
    if 'predicted_price' in st.session_state:
        price = st.session_state['predicted_price']
        
        st.metric(label="Estimated Market Value", value=f"${price:,.2f}")
        
        # Category Badge
        if price < 150000:
            st.info("🏷️ Property Category: **Budget / Entry-Level**")
        elif price < 350000:
            st.success("🏷️ Property Category: **Mid-Tier Family Home**")
        else:
            st.warning("🏷️ Property Category: **Premium Luxury Estate**")

        st.markdown("---")
        
        # EMI / Home Loan Calculator Section
        st.subheader("🏦 Home Loan & EMI Estimator")
        interest_rate = st.slider("Interest Rate (%)", 3.0, 15.0, 7.5, step=0.25)
        tenure_years = st.selectbox("Loan Tenure (Years)", [10, 15, 20, 25, 30], index=2)
        
        # EMI Formula calculation
        r = (interest_rate / 100) / 12
        n = tenure_years * 12
        emi = (price * r * ((1 + r)**n)) / (((1 + r)**n) - 1)
        
        st.write(f"💵 **Monthly Installment (EMI):** `${emi:,.2f}`/month")
        st.write(f"📊 **Total Payable Amount:** `${emi * n:,.2f}`")

        # Report Download Feature
        report_text = f"""--- REAL ESTATE VALUATION REPORT ---
Square Feet: {sqft} sqft
Bedrooms: {bedrooms} | Bathrooms: {bathrooms}
House Age: {house_age} years
Distance to City Center: {distance} km

ESTIMATED VALUE: ${price:,.2f}
ESTIMATED MONTHLY EMI ({tenure_years} yrs @ {interest_rate}%): ${emi:,.2f}
------------------------------------"""
        
        st.download_button(
            label="📄 Download Valuation Summary",
            data=report_text,
            file_name=f"Valuation_Report_{sqft}sqft.txt",
            mime="text/plain"
        )
    else:
        st.info("👈 Left panel se input set karein aur **Calculate Market Valuation** par click karein.")