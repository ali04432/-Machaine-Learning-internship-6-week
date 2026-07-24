# ==============================================================================
# WEEK 5 - TASK 1: HANDLING IMBALANCED & MESSY REAL-WORLD DATA
# ==============================================================================

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

# ------------------------------------------------------------------------------
# STEP 1: CREATE MESSY & HIGHLY IMBALANCED DATASET (e.g., Fraud Detection)
# ------------------------------------------------------------------------------
print("\n[STEP 1] Generating Imbalanced Real-World Dataset...")

# Creating an imbalanced dataset (90% Normal Transactions, 10% Fraud Cases)
X_raw, y_raw = make_classification(
    n_samples=4000,
    n_features=15,
    n_informative=10,
    weights=[0.90, 0.10],  # 90:10 Imbalance Ratio
    random_state=42
)

feature_names = [f"feature_{i+1}" for i in range(15)]
df = pd.DataFrame(X_raw, columns=feature_names)
df['is_fraud'] = y_raw

# Introducing Messy Data (Missing Values)
np.random.seed(42)
df.loc[df.sample(frac=0.07).index, 'feature_1'] = np.nan
df.loc[df.sample(frac=0.05).index, 'feature_3'] = np.nan

print(f"-> Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("-> Class Distribution in Target Column ('is_fraud'):")
print(df['is_fraud'].value_counts(normalize=True).map(lambda x: f"{x*100:.1f}%"))
print(f"-> Missing Values Found: {df.isnull().sum().sum()}")


# ------------------------------------------------------------------------------
# STEP 2: DATA CLEANING & PREPROCESSING
# ------------------------------------------------------------------------------
print("\n[STEP 2] Cleaning Missing Values & Scaling Features...")

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Impute Missing Values with Median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train/Test Split (Stratified to maintain class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)


# ------------------------------------------------------------------------------
# STEP 3: MODEL 1 - TRAIN WITHOUT HANDLING IMBALANCE
# ------------------------------------------------------------------------------
print("\n[STEP 3] Training Standard Random Forest (Ignoring Imbalance)...")

baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_model.fit(X_train, y_train)

y_pred_base = baseline_model.predict(X_test)
f1_base = f1_score(y_test, y_pred_base)
auc_base = roc_auc_score(y_test, baseline_model.predict_proba(X_test)[:, 1])


# ------------------------------------------------------------------------------
# STEP 4: MODEL 2 - TRAIN WITH CLASS WEIGHT BALANCING
# ------------------------------------------------------------------------------
print("\n[STEP 4] Training Balanced Random Forest (Handling Imbalance)...")

balanced_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Penalizes mistakes on minority class
    random_state=42
)
balanced_model.fit(X_train, y_train)

y_pred_bal = balanced_model.predict(X_test)
f1_bal = f1_score(y_test, y_pred_bal)
auc_bal = roc_auc_score(y_test, balanced_model.predict_proba(X_test)[:, 1])


# ------------------------------------------------------------------------------
# STEP 5: COMPARISON & REPORT
# ------------------------------------------------------------------------------
print("\n" + "="*55)
print("             IMBALANCE HANDLING COMPARISON             ")
print("="*55)
print(f"Standard Model  -> F1-Score: {f1_base:.4f} | ROC-AUC: {auc_base:.4f}")
print(f"Balanced Model  -> F1-Score: {f1_bal:.4f} | ROC-AUC: {auc_bal:.4f}")
print("="*55)

print("\n--- Detailed Classification Report (Balanced Model) ---")
print(classification_report(y_test, y_pred_bal, target_names=['Normal', 'Fraud']))