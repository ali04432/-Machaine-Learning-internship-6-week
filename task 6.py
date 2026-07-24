import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ---------------------------------------------------------
# Step 1: Simulate Real-World Customer Churn Dataset
# ---------------------------------------------------------
np.random.seed(42)
data_size = 1000

data = {
    'Tenure_Months': np.random.randint(1, 72, size=data_size),
    'Monthly_Charges': np.random.uniform(20.0, 120.0, size=data_size),
    'Total_Charges': np.random.uniform(100.0, 8000.0, size=data_size),
    'Contract_Type': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=data_size),
    'Tech_Support': np.random.choice(['Yes', 'No'], size=data_size),
    'Churn': np.random.choice([0, 1], size=data_size, p=[0.75, 0.25]) # 25% Churn Rate
}

df = pd.DataFrame(data)

# ---------------------------------------------------------
# Step 2: Data Preprocessing & Encoding
# ---------------------------------------------------------
# Encode Categorical Features
le_contract = LabelEncoder()
le_tech = LabelEncoder()

df['Contract_Type'] = le_contract.fit_transform(df['Contract_Type'])
df['Tech_Support'] = le_tech.fit_transform(df['Tech_Support'])

# Features & Target Separation
X = df.drop(columns=['Churn'])
y = df['Churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# Step 3: Model Building & Evaluation
# ---------------------------------------------------------
print("--- Training Customer Churn Prediction Model ---")
churn_model = GradientBoostingClassifier(random_state=42)
churn_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = churn_model.predict(X_test_scaled)
y_proba = churn_model.predict_proba(X_test_scaled)[:, 1]

# Business Output & Metrics
print("\n[Churn Prediction Results]")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

# ---------------------------------------------------------
# Step 4: Feature Importance Analysis
# ---------------------------------------------------------
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': churn_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nKey Drivers for Customer Churn:")
print(feature_importance)