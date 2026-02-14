
# (a) Problem Statement

Build and evaluate supervised **binary classification** models on the provided tabular dataset, compare their performance using standard metrics, and identify the best-performing model for this task.

---

# (b) Dataset Description [1 mark]
- **Type:** Tabular dataset for binary classification.
- **Target column:** *(Update here with the exact target name; e.g., `target`)*
- **Positive class:** *(Specify which label denotes the positive class; e.g., `1`)*
- **Size & features:** *(Add rows/columns count and brief feature summary if required by rubric)*
- **Notes:** Data was split into train/test and preprocessed using standard encodings/scaling as implemented in `train.py` (update if different). Address potential class imbalance during evaluation/thresholding.

---

# (c) Models Used & Comparison [6 marks]

The following six algorithms were trained and evaluated on the same train/test protocol:
- Logistic Regression
- Decision Tree
- k-Nearest Neighbors (kNN)
- Naive Bayes
- Random Forest (**Ensemble**)
- XGBoost (**Ensemble**)

## Comparison Table (All Metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8350 | 0.8009 | 0.3679 | 0.6466 | 0.4689 | 0.4011 |
| Decision Tree | 0.8416 | 0.7853 | 0.3770 | 0.6228 | 0.4697 | 0.3999 |
| kNN | 0.8950 | 0.7552 | 0.5721 | 0.2694 | 0.3663 | 0.3439 |
| Naive Bayes | 0.8049 | 0.7755 | 0.3172 | 0.6347 | 0.4230 | 0.3490 |
| Random Forest (Ensemble) | 0.8977 | 0.7816 | 0.5910 | 0.2974 | 0.3957 | 0.3709 |
| **XGBoost (Ensemble)** | **0.9024** | **0.8111** | **0.6582** | 0.2780 | 0.3909 | **0.3857** |

> **Final Choice:** **XGBoost**—best **Accuracy (0.9024)** and highest **AUC (0.8111)** among the six models, with competitive MCC.

---

## Observations (Per-Model) [3 marks]

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong **AUC (0.8009)** and **Recall (0.6466)** suggest good ranking ability and sensitivity; **low Precision (0.3679)** indicates many false positives—threshold tuning or class-weighting could help. |
| Decision Tree | Similar F1 to Logistic Regression with slightly **lower AUC (0.7853)**; trees can overfit—pruning or depth limits may improve generalization. |
| kNN | High **Accuracy (0.8950)** but **low Recall (0.2694)** implies bias toward majority class; performance is sensitive to feature scaling and `k`. |
| Naive Bayes | Lower Accuracy overall but **Recall (0.6347)** is relatively strong; conditional independence assumptions likely not fully met, yet the model captures signal useful for sensitivity-focused tasks. |
| Random Forest (Ensemble) | Solid **Accuracy (0.8977)** and Precision, but **Recall (0.2974)** remains low; indicates conservative positive predictions—class weights or threshold adjustment may raise Recall. |
| XGBoost (Ensemble) | Best **Accuracy (0.9024)** and **AUC (0.8111)** with highest Precision; **Recall (0.2780)** is still limited—optimize with scale_pos_weight, balanced subsampling, or decision threshold tuning. |


