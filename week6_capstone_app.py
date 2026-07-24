import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page Configuration
st.set_page_config(
    page_title="Real Estate Price Prediction (Taiwan)",
    page_icon="🏡",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size:30px;
        font-weight:bold;
        color: #38bdf8;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# 1. Generate / Load Sample Dataset (Taiwan Real Estate Structure)
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 200
    mrt_dist = np.random.uniform(50, 3000, n)
    stores = np.random.randint(0, 11, n)
    lat = np.random.uniform(24.93, 25.02, n)
    long = np.random.uniform(121.49, 121.57, n)
    
    # House price equation with noise
    price = 50 - (mrt_dist * 0.008) + (stores * 1.5) + (lat * 2) + (long * 0.5) + np.random.normal(0, 3, n)
    price = np.maximum(price, 10.0) # avoid negative
    
    df = pd.DataFrame({
        'Distance_to_MRT': mrt_dist,
        'Convenience_Stores': stores,
        'Latitude': lat,
        'Longitude': long,
        'Price_Per_Unit_Area': price
    })
    return df

df = load_data()

# Train Linear Regression Model
X = df[['Distance_to_MRT', 'Convenience_Stores', 'Latitude', 'Longitude']]
y = df['Price_Per_Unit_Area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Sidebar Navigation & Inputs
st.sidebar.title("📌 Navigation")
app_mode = st.sidebar.radio("Choose Page", ["Price Predictor", "Dataset & Analytics"])

# Exchange Rate (1 NTD = ~0.031 USD)
NTD_TO_USD = 0.031 

# --- PAGE 1: PREDICTOR ---
if app_mode == "Price Predictor":
    st.markdown("<p class='main-header'>🏡 Real Estate Price Prediction App</p>", unsafe_allow_html=True)
    st.caption("Predict property unit area prices in Taiwan using Machine Learning (Linear Regression).")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("⚙️ Input Property Features")
        
        mrt_input = st.number_input("Distance to Nearest MRT Station (meters):", min_value=10.0, max_value=5000.0, value=300.0, step=10.0)
        stores_input = st.slider("Number of Convenience Stores Nearby:", 0, 15, 5)
        lat_input = st.number_input("Latitude:", min_value=24.0, max_value=26.0, value=25.033, format="%.4f")
        long_input = st.number_input("Longitude:", min_value=120.0, max_value=123.0, value=121.565, format="%.4f")
        
        currency = st.selectbox("Select Display Currency:", ["New Taiwan Dollars (NTD)", "US Dollars (USD)"])

        predict_btn = st.button("🔮 Predict Price", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 Prediction Result")
        
        if predict_btn:
            input_df = pd.DataFrame([[mrt_input, stores_input, lat_input, long_input]], 
                                    columns=['Distance_to_MRT', 'Convenience_Stores', 'Latitude', 'Longitude'])
            
            predicted_ntd = model.predict(input_df)[0]
            
            st.success("Valuation Complete!")
            
            if currency == "New Taiwan Dollars (NTD)":
                st.metric(label="Predicted House Price per Unit Area", value=f"{predicted_ntd:.2f} NTD")
            else:
                predicted_usd = predicted_ntd * NTD_TO_USD
                st.metric(label="Predicted House Price per Unit Area", value=f"${predicted_usd:.2f} USD")
                
            st.info(f"📍 **Location Coordinates:** ({lat_input}, {long_input})\n\n🚶 **MRT Proximity:** {mrt_input} meters")
        else:
            st.info("👈 Please enter property values on the left panel and click **Predict Price**.")

# --- PAGE 2: ANALYTICS & VISUALIZATIONS ---
else:
    st.markdown("<p class='main-header'>📈 Dataset Information & Visualizations</p>", unsafe_allow_html=True)
    st.write("Exploratory Data Analysis (EDA) of the Taiwan Real Estate Dataset.")
    st.markdown("---")

    # Data Table Preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Histograms (Feature Distributions)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['Price_Per_Unit_Area'], kde=True, color='skyblue', ax=ax)
        ax.set_title("Distribution of Unit Area Price")
        st.pyplot(fig)

    with col2:
        st.subheader("🔥 Correlation Matrix Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Feature Correlations")
        st.pyplot(fig)

    # Scatter Plots
    st.subheader("📉 Distance to MRT vs Price Scatter Plot")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.scatterplot(x=df['Distance_to_MRT'], y=df['Price_Per_Unit_Area'], hue=df['Convenience_Stores'], palette='viridis', ax=ax)
    ax.set_xlabel("Distance to MRT (meters)")
    ax.set_ylabel("Price Per Unit Area")
    st.pyplot(fig)