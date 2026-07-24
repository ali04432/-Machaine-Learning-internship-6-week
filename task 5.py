import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# ---------------------------------------------------------
# Step 1: Create Imbalanced Dataset (90% Majority, 10% Minority)
# ---------------------------------------------------------
print("--- Step 1: Generating Imbalanced Dataset ---")
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    weights=[0.9, 0.1], 
    random_state=42
)

# Split Dataset into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# Step 2: Train Baseline Model & Evaluate
# ---------------------------------------------------------
print("\n--- Step 2: Training Baseline Random Forest Model ---")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

# Comprehensive Metrics Evaluation
print("\n[Baseline Evaluation Metrics]")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")

# Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# Step 3: Hyperparameter Tuning via GridSearchCV
# ---------------------------------------------------------
print("\n--- Step 3: Hyperparameter Tuning with GridSearchCV ---")
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 10, None],
    'class_weight': ['balanced', None] # Address class imbalance
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='f1', # Optimizing for F1-score rather than simple accuracy
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate Tuned Model
y_pred_tuned = best_model.predict(X_test)
print(f"\nTuned F1-Score: {f1_score(y_test, y_pred_tuned):.4f}")