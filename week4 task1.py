# ==============================================================================
# WEEK 4 - TASK 1: BUILD A PROPER ML PIPELINE WITH FEATURE ENGINEERING
# ==============================================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------------------------
# STEP 1: CREATE SAMPLE DATASET WITH MISSING VALUES
# ------------------------------------------------------------------------------
print("\n[STEP 1] Dataset create ho raha hai...")

np.random.seed(42)
n_samples = 1200

# Synthetic customer dataset
data = pd.DataFrame({
    'age': np.random.randint(18, 70, size=n_samples).astype(float),
    'annual_income': np.random.normal(55000, 15000, size=n_samples),
    'credit_score': np.random.randint(300, 850, size=n_samples).astype(float),
    'city': np.random.choice(['Lahore', 'Karachi', 'Islamabad', 'Peshawar'], size=n_samples),
    'loan_approved': np.random.choice([0, 1], size=n_samples, p=[0.55, 0.45])
})

# Missing values introduce kar rahe hain demonstration ke liye
data.loc[data.sample(frac=0.08).index, 'annual_income'] = np.nan
data.loc[data.sample(frac=0.05).index, 'credit_score'] = np.nan

print(f"-> Total Rows: {data.shape[0]}, Total Columns: {data.shape[1]}")
print("-> Missing Values Summary:")
print(data.isnull().sum())


# ------------------------------------------------------------------------------
# STEP 2: FEATURE ENGINEERING
# ------------------------------------------------------------------------------
print("\n[STEP 2] Feature Engineering ho rahi hai...")

# New Feature 1: Income-to-Age Ratio
data['income_to_age_ratio'] = data['annual_income'] / data['age']

# New Feature 2: High Credit Score Flag (Threshold >= 700)
data['is_high_credit'] = (data['credit_score'] >= 700).astype(int)

print("-> Naye features add ho gaye hain: 'income_to_age_ratio' aur 'is_high_credit'")


# ------------------------------------------------------------------------------
# STEP 3: SPLIT DATA INTO FEATURES (X) AND TARGET (y)
# ------------------------------------------------------------------------------
X = data.drop('loan_approved', axis=1)
y = data['loan_approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"-> Training Set Size: {X_train.shape[0]} rows")
print(f"-> Testing Set Size:  {X_test.shape[0]} rows")


# ------------------------------------------------------------------------------
# STEP 4: PREPROCESSING PIPELINES FOR NUMERIC & CATEGORICAL DATA
# ------------------------------------------------------------------------------
print("\n[STEP 3] Scikit-Learn Preprocessing Pipeline setup ho rahi hai...")

# Numeric columns identification
numeric_features = ['age', 'annual_income', 'credit_score', 'income_to_age_ratio', 'is_high_credit']

# Numeric Pipeline: Missing values median se fill hongi aur values scale hongi
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical columns identification
categorical_features = ['city']

# Categorical Pipeline: Missing values mode se fill hongi aur One-Hot Encoding hogi
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num_pipeline', numeric_transformer, numeric_features),
        ('cat_pipeline', categorical_transformer, categorical_features)
    ]
)


# ------------------------------------------------------------------------------
# STEP 5: FULL MACHINE LEARNING PIPELINE
# ------------------------------------------------------------------------------
# Combining Preprocessing + Model Classifier
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42))
])


# ------------------------------------------------------------------------------
# STEP 6: TRAIN & EVALUATE PIPELINE
# ------------------------------------------------------------------------------
print("\n[STEP 4] Model train ho raha hai...")
full_pipeline.fit(X_train, y_train)
print("-> Training Complete!")

# Make Predictions
y_pred = full_pipeline.predict(X_test)

# Evaluation Results
acc = accuracy_score(y_test, y_pred)
print("\n" + "="*50)
print("              EVALUATION RESULTS              ")
print("="*50)
print(f"Accuracy Score: {acc * 100:.2f}%\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("="*50)