
# (a) Problem Statement

Predict whether a client will **subscribe to a term deposit** after a marketing contact using supervised **binary classification**. Build and compare six ML models on a common train/test split and report Accuracy, AUC, Precision, Recall, F1, and MCC; then summarize per‑model observations and select a final model.

---

# (b) Dataset Description
- **Type:** Tabular dataset for binary classification.
- **Target column:** `y` (values: `yes` / `no`).
- **Positive class:** `yes`.
- **Size & features:** 41188 rows × 21 columns. Class distribution → `no`: 36548 (88.73%), `yes`: 4640 (11.27%).
  - **Numeric features (9, after dropping `duration` to avoid leakage):** age, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed
  - **Categorical features (10):** job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome
- **Train/test split:** Stratified **80/20**, `random_state=42`.
- **Preprocessing:** `OneHotEncoder(handle_unknown="ignore")` for categoricals and `StandardScaler` for numericals, applied via a `ColumnTransformer` inside the pipeline.
- **Leakage handling:** The `duration` column (call duration) is **excluded** during training to prevent target leakage.

---

# (c) Models Used & Comparison

The following six algorithms were trained with a shared preprocessing pipeline:
- **Logistic Regression** (`max_iter=2000`, `class_weight='balanced'`)
- **Decision Tree** (`max_depth=10`, `min_samples_split=20`, `min_samples_leaf=10`, `class_weight='balanced'`)
- **k‑Nearest Neighbors (kNN)** (`n_neighbors=15`, `weights='distance'`)
- **Naive Bayes** (GaussianNB)
- **Random Forest (Ensemble)** (`n_estimators=100`, `max_depth=None`, `random_state=42`, `n_jobs=-1`)
- **XGBoost (Ensemble)** (`n_estimators=400`, `learning_rate=0.05`, `max_depth=5`, `subsample=0.9`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `random_state=42`, `eval_metric='logloss'`)

## Comparison Table (All Metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8350 | 0.8009 | 0.3679 | 0.6466 | 0.4689 | 0.4011 |
| Decision Tree | 0.8416 | 0.7853 | 0.3770 | 0.6228 | 0.4697 | 0.3999 |
| kNN | 0.8950 | 0.7552 | 0.5721 | 0.2694 | 0.3663 | 0.3439 |
| Naive Bayes | 0.8049 | 0.7755 | 0.3172 | 0.6347 | 0.4230 | 0.3490 |
| Random Forest (Ensemble) | 0.8977 | 0.7816 | 0.5910 | 0.2974 | 0.3957 | 0.3709 |
| **XGBoost (Ensemble)** | **0.9024** | **0.8111** | **0.6582** | 0.2780 | 0.3909 | **0.3857** |

**Final Choice:** **XGBoost** — best **Accuracy (0.9024)** and highest **AUC (0.8111)** across the six models.

---

## Observations (Per‑Model)

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong **AUC (0.8009)** and **Recall (0.6466)** → good ranking ability and sensitivity; **low Precision (0.3679)** indicates many false positives—threshold tuning or class‑weighting may help. |
| Decision Tree | Similar F1 to Logistic Regression with slightly **lower AUC (0.7853)**; trees can overfit—pruning and regularization (already applied) improve generalization. |
| kNN | High **Accuracy (0.8950)** but **low Recall (0.2694)** → biased toward majority class; sensitive to scaling and neighborhood size. |
| Naive Bayes | Lower Accuracy overall but **Recall (0.6347)** is relatively strong; independence assumptions imperfect but useful when sensitivity is prioritized. |
| Random Forest (Ensemble) | Solid **Accuracy (0.8977)** and Precision, but **Recall (0.2974)** remains low; conservative positives—class weights or threshold adjustment can raise Recall. |
| XGBoost (Ensemble) | Best **Accuracy (0.9024)** and **AUC (0.8111)** with highest Precision; **Recall (0.2780)** still limited—consider `scale_pos_weight`, calibration, or threshold tuning to trade precision for recall. |
