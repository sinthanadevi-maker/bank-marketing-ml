# train.py — path-safe, robust version
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -----------------------------
#   Constants & Paths (relative to this file)
# -----------------------------
RANDOM_STATE = 42
REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "bank-additional-full.csv"
MODELS_DIR = REPO_ROOT / "models"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

print("=== Path Debug ===")
print("Script location     :", REPO_ROOT)
print("Data CSV            :", DATA_PATH)
print("Models directory    :", MODELS_DIR)
print("Artifacts directory :", ARTIFACTS_DIR)
print("===================\n")

# -----------------------------
#   Load Dataset (semicolon-separated)
# -----------------------------
df = pd.read_csv(DATA_PATH, sep=';')

# Drop duration (target leakage per dataset documentation)
if "duration" in df.columns:
    df = df.drop(columns=["duration"])

# Target to 0/1
df["y"] = (df["y"].astype(str).str.lower() == "yes").astype(int)

X = df.drop(columns=["y"])
y = df["y"].values

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save test split for Streamlit
test_df = X_test.copy()
test_df["y"] = y_test
test_split_path = ARTIFACTS_DIR / "test_split.csv"
print("Saving test split to:", test_split_path)
test_df.to_csv(test_split_path, index=False)

# -----------------------------
# Preprocessing
# -----------------------------

# -----------------------------
# Preprocessing (Robust Version)
# -----------------------------
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical columns:", cat_cols)
print("Numerical columns  :", num_cols)


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols)
    ],
    remainder="drop"
)

# -----------------------------
# Models Dictionary
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),

    # Tuned Decision Tree (reduces collapse to all "no")
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "kNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),

    "Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1
        # Optional: add scale_pos_weight later to boost recall of "yes"
    ),
}

results = []

# -----------------------------
# Train & Evaluate Models
# -----------------------------
for name, model in models.items():
    print(f"\n===== Training {name} =====\n")

    pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    # Save model to project/models
    model_path = MODELS_DIR / f"{name.replace(' ', '_').lower()}.joblib"
    print("Saving model to:", model_path)
    joblib.dump(pipeline, model_path,compress=3)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # AUC (needs probabilities)
    if hasattr(pipeline.named_steps['model'], "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_score)
    else:
        auc = float("nan")

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([name, acc, auc, prec, rec, f1, mcc])

    # Classification report (for your BITS Lab screenshot)
    print(classification_report(y_test, y_pred, target_names=["no", "yes"]))

# -----------------------------
# Save Metrics
# -----------------------------
metrics_df = pd.DataFrame(results, columns=["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"])
metrics_path = ARTIFACTS_DIR / "metrics.csv"
print("\nSaving metrics to:", metrics_path)
metrics_df.to_csv(metrics_path, index=False)

print("\nTraining complete.")
print(f"Models saved to    : {MODELS_DIR}")
print(f"Artifacts saved to : {ARTIFACTS_DIR}")

