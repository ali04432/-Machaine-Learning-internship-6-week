import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Housing Dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Features and Target
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Predictions & Metrics
y_pred = reg_model.predict(X_test)

print(f"R2 Score (Accuracy Metric): {r2_score(y_test, y_pred):.4f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")