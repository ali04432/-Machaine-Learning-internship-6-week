# ==============================================================================
# WEEK 6 - CAPSTONE: END-TO-END MACHINE LEARNING PIPELINE
# PROJECT: REAL-ESTATE HOUSING PRICE PREDICTION SYSTEM
# ==============================================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def main():
    print("="*65)
    print("      CAPSTONE PROJECT: END-TO-END ML TRAINING PIPELINE      ")
    print("="*65)

    # --------------------------------------------------------------------------
    # STEP 1: RAW DATA CREATION & SIMULATION
    # --------------------------------------------------------------------------
    print("\n[STEP 1] Generating Raw Real-Estate Housing Dataset...")
    
    np.random.seed(42)
    n_samples = 2500

    # Features: Area (sqft), Bedrooms, Bathrooms, Age (Years), Distance to City Center (km)
    sqft = np.random.normal(1800, 500, n_samples).clip(500, 5000)
    bedrooms = np.random.randint(1, 6, size=n_samples)
    bathrooms = np.random.randint(1, 4, size=n_samples)
    age = np.random.randint(0, 40, size=n_samples)
    distance_km = np.random.uniform(1.0, 35.0, size=n_samples)

    # Price Calculation with Real-World Logic & Noise
    price = (
        sqft * 150 +
        bedrooms * 10000 +
        bathrooms * 15000 -
        age * 800 -
        distance_km * 2500 +
        np.random.normal(0, 15000, n_samples)
    )

    df = pd.DataFrame({
        'Square_Feet': sqft,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'House_Age': age,
        'Distance_City_Center': distance_km,
        'Price_USD': price
    })

    # Introducing Missing Values to simulate real messy data
    df.loc[df.sample(frac=0.03).index, 'Square_Feet'] = np.nan
    df.loc[df.sample(frac=0.03).index, 'House_Age'] = np.nan

    print(f"-> Dataset Generated: {df.shape[0]} Rows, {df.shape[1]} Columns")
    print(f"-> Missing Values Found: {df.isnull().sum().sum()}")

    # --------------------------------------------------------------------------
    # STEP 2: DATA CLEANING & PREPROCESSING
    # --------------------------------------------------------------------------
    print("\n[STEP 2] Preprocessing & Feature Scaling...")
    
    X = df.drop('Price_USD', axis=1)
    y = df['Price_USD']

    # Impute missing values with Median strategy
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Standardize Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42
    )

    # --------------------------------------------------------------------------
    # STEP 3: MODEL HYPERPARAMETER TUNING & TRAINING
    # --------------------------------------------------------------------------
    print("\n[STEP 3] Tuning & Training Gradient Boosting Regressor...")

    param_grid = {
        'n_estimators': [100, 150],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 6]
    }

    base_model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"-> Best Parameters Found: {grid_search.best_params_}")

    # --------------------------------------------------------------------------
    # STEP 4: EVALUATION & METRICS REPORT
    # --------------------------------------------------------------------------
    print("\n[STEP 4] Evaluating Model Performance...")

    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("\n" + "="*45)
    print("         CAPSTONE MODEL EVALUATION          ")
    print("="*45)
    print(f" R2 Score (Accuracy):        {r2 * 100:.2f}%")
    print(f" Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f" Root Mean Sq Error (RMSE): ${rmse:,.2f}")
    print("="*45)

    # --------------------------------------------------------------------------
    # STEP 5: SAVE MODEL PIPELINE FOR DEPLOYMENT
    # --------------------------------------------------------------------------
    print("\n[STEP 5] Saving Artifacts (Model, Scaler, Imputer)...")
    
    artifacts = {
        'model': best_model,
        'scaler': scaler,
        'imputer': imputer,
        'feature_names': list(X.columns)
    }

    joblib.dump(artifacts, 'capstone_model.joblib')
    print("-> Model pipeline successfully saved as 'capstone_model.joblib'!")

if __name__ == "__main__":
    main()