# app.py
from pathlib import Path
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report, RocCurveDisplay
)

# ---------- Setup ----------
REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DEFAULT_TEST = ARTIFACTS_DIR / "test_split.csv"

st.set_page_config(page_title="Bank Marketing - ML Classifiers", layout="wide")
st.title("📞 UCI Bank Marketing — Term Deposit Subscription")
st.caption("Select a trained model, upload test CSV, and view metrics & plots.")

# ---------- Load trained models ----------
model_files = sorted(MODELS_DIR.glob("*.joblib"))
if not model_files:
    st.error("No models found in 'models/'. Please run train.py first.")
    st.stop()

pretty_names = [mf.stem.replace("_", " ").title() for mf in model_files]
name_to_path = dict(zip(pretty_names, model_files))

with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Select a model", pretty_names)
    st.markdown("---")
    st.write("**Note:** Dataset uses categorical & numeric features; "
             "training excluded **duration** to avoid target leakage.")

# Load chosen pipeline (prep + model)
pipe = joblib.load(name_to_path[model_choice])

# ---------- Data input ----------
st.subheader("1) Upload Test CSV (or use sample)")
st.write("- If your CSV contains column **`y`** (yes/no), the app will compute metrics.\n"
         "- Without `y`, the app will show predictions only.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

def read_csv_flexible(file_or_path):
    """Try semicolon first (UCI), then comma."""
    try:
        df = pd.read_csv(file_or_path, sep=';')
        # If it collapsed into 1 col, try comma
        if df.shape[1] == 1:
            if hasattr(file_or_path, "seek"): file_or_path.seek(0)
            df = pd.read_csv(file_or_path, sep=',')
    except Exception:
        if hasattr(file_or_path, "seek"): file_or_path.seek(0)
        df = pd.read_csv(file_or_path)
    return df

if uploaded is not None:
    test_df = read_csv_flexible(uploaded)
else:
    st.info(f"Using bundled sample: `{DEFAULT_TEST.name}`")
    test_df = read_csv_flexible(DEFAULT_TEST)

# ---------- Split features/target if present ----------

has_target = 'y' in test_df.columns
if has_target:
    y_col = test_df['y']

    # Map common representations to 0/1
    if pd.api.types.is_numeric_dtype(y_col):
        y_true = y_col.astype(int).values
    else:
        y_map = {
            'yes': 1, 'no': 0,
            'y': 1, 'n': 0,
            'true': 1, 'false': 0,
            '1': 1, '0': 0
        }
        y_true = y_col.astype(str).str.strip().str.lower().map(y_map).astype(int).values

    X_test = test_df.drop(columns=['y'])
else:
    X_test = test_df.copy()

# ---------- Predict ----------
if hasattr(pipe, "predict_proba"):
    proba = pipe.predict_proba(X_test)[:, 1]
else:
    proba = None
y_pred = pipe.predict(X_test)

# ---------- Show predictions ----------
st.subheader("2) Predictions (first 25 rows)")
pred_df = pd.DataFrame({
    "pred": np.where(y_pred == 1, "yes", "no"),
    "prob_yes": proba if proba is not None else np.nan
})
st.dataframe(pred_df.head(25))

# ---------- Metrics & plots ----------
st.subheader("3) Evaluation Metrics & Plots")
if has_target:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, proba) if proba is not None else np.nan

    metrics_df = pd.DataFrame([{
        "Model": model_choice,
        "Accuracy": acc, "AUC": auc, "Precision": prec,
        "Recall": rec, "F1": f1, "MCC": mcc
    }])
    st.dataframe(metrics_df.style.format({
        "Accuracy": "{:.4f}", "AUC": "{:.4f}", "Precision": "{:.4f}",
        "Recall": "{:.4f}", "F1": "{:.4f}", "MCC": "{:.4f}"
    }))

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['no','yes'], yticklabels=['no','yes'], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col2:
        st.markdown("**ROC Curve**")
        if proba is not None:
            fig2, ax2 = plt.subplots(figsize=(5,4))
            RocCurveDisplay.from_predictions(y_true, proba, name=model_choice, ax=ax2)
            st.pyplot(fig2)
        else:
            st.info("AUC/ROC not available for this model.")

    st.markdown("**Classification Report**")
    st.code(classification_report(y_true, y_pred, target_names=["no", "yes"]), language="text")
else:
    st.warning("No 'y' column found — metrics/plots are disabled. Add 'y' to evaluate.")

st.markdown("---")
st.caption("Tip: Ensure uploaded CSV column names match training schema. "
           "If you used the UCI raw file, remember it's semicolon-separated (`;`).")