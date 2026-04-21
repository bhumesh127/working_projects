# Confusion Matrix — Complete Guide with Real-Time Examples

# What is a Confusion Matrix?

# A Confusion Matrix is a table that shows how well your classification model performed by 
# comparing actual vs predicted values.

# Real-World Analogy

# Imagine a doctor diagnosing cancer:

# Said "Has Cancer" → Actually Has Cancer ✅ → Correct!
# Said "Has Cancer" → Actually No Cancer ❌ → False Alarm!
# Said "No Cancer" → Actually Has Cancer ❌ → Missed it! (Dangerous!)
# Said "No Cancer" → Actually No Cancer ✅ → Correct!

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Simulate cancer detection dataset
# 0 = No Cancer, 1 = Cancer
X, y = make_classification(
    n_samples     = 1000,
    n_features    = 15,
    n_informative = 10,
    weights       = [0.85, 0.15],   # 85% no cancer, 15% cancer (imbalanced!)
    random_state  = 42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost model
model = XGBClassifier(
    n_estimators  = 200,
    max_depth     = 5,
    learning_rate = 0.05,
    eval_metric   = 'logloss',
    verbosity     = 0
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print("=" * 55)
print("CONFUSION MATRIX — Cancer Detection")
print("=" * 55)
print(f"""
                  PREDICTED
              No Cancer  Has Cancer
            ┌──────────┬──────────┐
ACTUAL  No  │  TN={TN:>4}  │  FP={FP:>4}  │
        Can │──────────┼──────────│
        cer │  FN={FN:>4}  │  TP={TP:>4}  │
            └──────────┴──────────┘
""")

print(f"TP (Correctly caught cancer)      : {TP}")
print(f"TN (Correctly said no cancer)     : {TN}")
print(f"FP (False alarm — no cancer)      : {FP}  ← Unnecessary treatment")
print(f"FN (Missed cancer — DANGEROUS!)   : {FN}  ← Patient untreated ⚠️")

# ─────────────────────────────────────────────
# ALL METRICS
# ─────────────────────────────────────────────
accuracy    = (TP + TN) / (TP + TN + FP + FN)
precision   = TP / (TP + FP)
recall      = TP / (TP + FN)
specificity = TN / (TN + FP)
f1          = 2 * (precision * recall) / (precision + recall)
auc         = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 55)
print("PERFORMANCE METRICS")
print("=" * 55)
print(f"Accuracy    : {accuracy:.4f}   → {accuracy*100:.1f}% overall correct")
print(f"Precision   : {precision:.4f}   → Of predicted cancer, {precision*100:.1f}% were right")
print(f"Recall      : {recall:.4f}   → Caught {recall*100:.1f}% of all cancer cases")
print(f"Specificity : {specificity:.4f}   → Correctly ruled out {specificity*100:.1f}% of no-cancer")
print(f"F1-Score    : {f1:.4f}   → Balance of precision & recall")
print(f"ROC-AUC     : {auc:.4f}   → Overall discrimination ability")
# ```

# ### Output
# ```
# =====================================================
# CONFUSION MATRIX — Cancer Detection
# =====================================================

#               PREDICTED
#           No Cancer  Has Cancer
#         ┌──────────┬──────────┐
# ACTUAL  │  TN=163  │   FP=6   │
#         │──────────┼──────────│
#         │   FN=4   │  TP=27   │
#         └──────────┴──────────┘

# TP (Correctly caught cancer)      : 27
# TN (Correctly said no cancer)     : 163
# FP (False alarm — no cancer)      :  6  ← Unnecessary treatment
# FN (Missed cancer — DANGEROUS!)   :  4  ← Patient untreated ⚠️

# =====================================================
# PERFORMANCE METRICS
# =====================================================
# Accuracy    : 0.9500   → 95.0% overall correct
# Precision   : 0.8182   → Of predicted cancer, 81.8% were right
# Recall      : 0.8710   → Caught 87.1% of all cancer cases
# Specificity : 0.9645   → Correctly ruled out 96.5% of no-cancer
# F1-Score    : 0.8438   → Balance of precision & recall
# ROC-AUC     : 0.9712   → Overall discrimination ability