# ==============================================================================
# WEEK 4 - TASK 2: ENSEMBLE LEARNING - RANDOM FOREST VS GRADIENT BOOSTING
# ==============================================================================

import time
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ------------------------------------------------------------------------------
# STEP 1: DATASET CREATION
# ------------------------------------------------------------------------------
print("\n[1] Classification Dataset prepare ho raha hai...")

X, y = make_classification(
    n_samples=3000,
    n_features=25,
    n_informative=18,
    n_redundant=7,
    n_classes=2,
    random_state=42
)

# Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"-> Total Training Samples: {X_train.shape[0]}")
print(f"-> Total Testing Samples:  {X_test.shape[0]}")


# ------------------------------------------------------------------------------
# STEP 2: TRAIN RANDOM FOREST MODEL
# ------------------------------------------------------------------------------
print("\n[2] Training Random Forest Classifier...")

start_rf_time = time.time()
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_execution_time = time.time() - start_rf_time

# Predictions
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

print(f"-> Random Forest Training Completed in {rf_execution_time:.3f} seconds.")


# ------------------------------------------------------------------------------
# STEP 3: TRAIN GRADIENT BOOSTING MODEL (XGBoost Equivalent)
# ------------------------------------------------------------------------------
print("\n[3] Training Gradient Boosting Classifier...")

start_xgb_time = time.time()
gb_model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.08,
    max_depth=6,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_execution_time = time.time() - start_xgb_time

# Predictions
gb_preds = gb_model.predict(X_test)
gb_probs = gb_model.predict_proba(X_test)[:, 1]

print(f"-> Gradient Boosting Training Completed in {gb_execution_time:.3f} seconds.")


# ------------------------------------------------------------------------------
# STEP 4: DETAILED COMPARISON TABLE
# ------------------------------------------------------------------------------
print("\n[4] Performance Metrics Calculate ho rahe hain...")

comparison_data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC Score", "Training Time (s)"],
    "Random Forest": [
        f"{accuracy_score(y_test, rf_preds):.4f}",
        f"{precision_score(y_test, rf_preds):.4f}",
        f"{recall_score(y_test, rf_preds):.4f}",
        f"{f1_score(y_test, rf_preds):.4f}",
        f"{roc_auc_score(y_test, rf_probs):.4f}",
        f"{rf_execution_time:.3f}"
    ],
    "Gradient Boosting (XGB)": [
        f"{accuracy_score(y_test, gb_preds):.4f}",
        f"{precision_score(y_test, gb_preds):.4f}",
        f"{recall_score(y_test, gb_preds):.4f}",
        f"{f1_score(y_test, gb_preds):.4f}",
        f"{roc_auc_score(y_test, gb_probs):.4f}",
        f"{gb_execution_time:.3f}"
    ]
}

report_df = pd.DataFrame(comparison_data)

print("\n" + "="*55)
print("     RANDOM FOREST vs GRADIENT BOOSTING COMPARISON     ")
print("="*55)
print(report_df.to_string(index=False))
print("="*55)