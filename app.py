# app.py
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    classification_report, RocCurveDisplay
)

# ---------- Setup ----------
REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DEFAULT_TEST = ARTIFACTS_DIR / "test_split.csv"

st.set_page_config(page_title="Bank Marketing - ML Classifiers", layout="wide")
st.title("📞 UCI Bank Marketing — Term Deposit Subscription")
st.caption("Select a trained model and evaluate predictions.")

# ---------- Load models ----------
model_files = sorted(MODELS_DIR.glob("*.joblib"))
if not model_files:
    st.error("No models found in 'models/'. Please run train.py first.")
    st.stop()

pretty_names = [mf.stem.replace("_", " ").title() for mf in model_files]
name_to_path = dict(zip(pretty_names, model_files))

with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Select a model", pretty_names)

pipe = joblib.load(name_to_path[model_choice])

# ---------- Data input section ----------
st.subheader("1️⃣ Choose Data Source")

st.markdown("### **📁 Use Sample test_split.csv (Recommended)**")
use_sample = st.button("Use Sample test_split.csv")

st.markdown("---")

st.markdown("### 📂 Or Upload Your Own CSV")
uploaded = st.file_uploader(
    "Drag and drop your CSV file here",
    type=["csv"]
)

# ---------- Decide which dataset to use ----------
def read_csv_flexible(file_or_path):
    try:
        df = pd.read_csv(file_or_path, sep=';')
        if df.shape[1] == 1:
            if hasattr(file_or_path, "seek"): file_or_path.seek(0)
            df = pd.read_csv(file_or_path, sep=',')
    except Exception:
        if hasattr(file_or_path, "seek"): file_or_path.seek(0)
        df = pd.read_csv(file_or_path)
    return df

if use_sample:
    test_df = read_csv_flexible(DEFAULT_TEST)

elif uploaded is not None:
    test_df = read_csv_flexible(uploaded)

else:
    st.info("Please select a data source above to continue.")
    st.stop()


# ---------- Split target ----------
has_target = 'y' in test_df.columns

if has_target:
    y_col = test_df['y']
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
with st.spinner("Running model prediction..."):
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
    else:
        proba = None
    y_pred = pipe.predict(X_test)

# ---------- Predictions ----------
st.subheader("2️⃣ Predictions (first 25 rows)")

pred_df = pd.DataFrame({
    "Prediction": np.where(y_pred == 1, "yes", "no"),
    "Probability_yes": proba if proba is not None else np.nan
})

st.dataframe(pred_df.head(25))

# ---------- Metrics ----------
st.subheader("3️⃣ Evaluation Metrics")

if has_target:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, proba) if proba is not None else np.nan

    metrics_df = pd.DataFrame([{
        "Model": model_choice,
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc
    }])

    st.dataframe(
    metrics_df.style.format({
        "Accuracy": "{:.4f}",
        "AUC": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1": "{:.4f}",
        "MCC": "{:.4f}",
    })
)


    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=['no','yes'],
                    yticklabels=['no','yes'],
                    ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col2:
        if proba is not None:
            st.markdown("**ROC Curve**")
            fig2, ax2 = plt.subplots()
            RocCurveDisplay.from_predictions(y_true, proba, ax=ax2)
            st.pyplot(fig2)

    st.markdown("**Classification Report**")
    st.code(classification_report(y_true, y_pred,
                                  target_names=["no", "yes"]),
            language="text")
else:
    st.warning("No 'y' column found — metrics disabled.")
