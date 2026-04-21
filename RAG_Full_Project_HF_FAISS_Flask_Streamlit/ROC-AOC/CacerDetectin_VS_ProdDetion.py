# ROC Curve & AUC, Precision-Recall Curve & SMOTE — Complete Guide

# PART 1: ROC Curve & AUC
# What is ROC Curve?

# ROC (Receiver Operating Characteristic) curve plots True Positive Rate vs False Positive Rate at every possible threshold.

# Real-World Analogy

# Imagine a security scanner at an airport:

# Very sensitive setting → Catches all threats BUT flags many innocent people (High TPR, High FPR)
# Very strict setting → Misses some threats BUT fewer false alarms (Low TPR, Low FPR)
# ROC curve shows ALL possible trade-offs between these two settings
# AUC tells you how good the scanner is overall


# AUC = 1.0  → Perfect model ✅
# AUC = 0.9  → Excellent
# AUC = 0.8  → Good
# AUC = 0.7  → Fair
# AUC = 0.5  → Random guessing (useless) ❌
# AUC < 0.5  → Worse than random (flip predictions!)

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, roc_auc_score,
                              precision_recall_curve,
                              average_precision_score,
                              confusion_matrix,
                              classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Simulate Cancer Detection Dataset ──
X, y = make_classification(
    n_samples      = 2000,
    n_features     = 20,
    n_informative  = 12,
    weights        = [0.85, 0.15],  # Imbalanced: 85% no cancer
    random_state   = 42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Train 3 Models ──
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=200, max_depth=5,
                                         learning_rate=0.05,
                                         eval_metric='logloss', verbosity=0)
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {
        'fpr': fpr, 'tpr': tpr,
        'thresholds': thresholds,
        'auc': auc, 'y_prob': y_prob
    }

# ── Print ROC Results ──
print("=" * 55)
print("ROC-AUC SCORES — Cancer Detection")
print("=" * 55)
for name, res in results.items():
    bar = "█" * int(res['auc'] * 30)
    print(f"{name:<25} AUC={res['auc']:.4f}  {bar}")

# ── Find Optimal Threshold (Youden's J) ──
print("\n" + "=" * 55)
print("OPTIMAL THRESHOLD (Youden's J Statistic)")
print("=" * 55)
print(f"{'Model':<25} {'Threshold':>10} {'TPR':>8} {'FPR':>8}")
print("-" * 55)

for name, res in results.items():
    fpr   = res['fpr']
    tpr   = res['tpr']
    thresh = res['thresholds']
    # Youden's J = TPR - FPR → maximize this
    j_scores   = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    opt_thresh  = thresh[optimal_idx]
    opt_tpr     = tpr[optimal_idx]
    opt_fpr     = fpr[optimal_idx]
    print(f"{name:<25} {opt_thresh:>10.4f} {opt_tpr:>8.4f} {opt_fpr:>8.4f}")
# ```

# ### Output
# ```
# =======================================================
# ROC-AUC SCORES — Cancer Detection
# =======================================================
# Logistic Regression       AUC=0.8923  ██████████████████████████
# Random Forest             AUC=0.9412  ████████████████████████████
# XGBoost                   AUC=0.9634  █████████████████████████████

# =======================================================
# OPTIMAL THRESHOLD (Youden's J Statistic)
# =======================================================
# Model                     Threshold      TPR      FPR
# -------------------------------------------------------
# Logistic Regression          0.3821   0.8710   0.1021
# Random Forest                0.4123   0.9032   0.0812
# XGBoost                      0.3956   0.9355   0.0623
# ```

# ---

# ### ROC Curve Interpretation
# ```
# TPR (True Positive Rate)  = Recall = Sensitivity
# FPR (False Positive Rate) = 1 - Specificity

#          TPR
#     1.0  |    .........XGBoost (AUC=0.96)
#          |   /  ......RF (AUC=0.94)
#          |  / ../  LR (AUC=0.89)
#          | /./
#          |/./
#     0.5  |/  ← Random Guess (diagonal line)
#          |
#     0.0  └─────────────── FPR
#          0.0    0.5    1.0

# → Curve closer to TOP-LEFT corner = Better model
# → AUC = Area under the curve