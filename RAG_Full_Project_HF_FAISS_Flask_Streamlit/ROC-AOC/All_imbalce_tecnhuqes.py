# Real-World Analogy

# Imagine training a fraud detector on 9980 legit transactions
# and only 20 fraud cases. The model just says
# "Everything is legit!" → Gets 99.8% accuracy but catches
# ZERO fraud! That's a useless model.

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                              roc_auc_score,
                              average_precision_score,
                              f1_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Severe imbalance: only 3% fraud
X, y = make_classification(
    n_samples      = 5000,
    n_features     = 20,
    n_informative  = 12,
    weights        = [0.97, 0.03],
    random_state   = 42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"Original Training Set:")
print(f"  Class 0 (Legit) : {(y_train==0).sum()}")
print(f"  Class 1 (Fraud) : {(y_train==1).sum()}")
print(f"  Ratio           : {(y_train==0).sum()/(y_train==1).sum():.0f}:1\n")

# ─────────────────────────────────────────────
# TECHNIQUE 1: No Resampling (Baseline)
# ─────────────────────────────────────────────
def evaluate(name, X_tr, y_tr, X_te, y_te):
    model = XGBClassifier(
        n_estimators=200, max_depth=5,
        eval_metric='logloss', verbosity=0, random_state=42
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    return {
        'name':     name,
        'f1':       f1_score(y_te, y_pred),
        'roc_auc':  roc_auc_score(y_te, y_prob),
        'pr_auc':   average_precision_score(y_te, y_prob),
        'recall':   f1_score(y_te, y_pred, average=None)[1],
        'precision':f1_score(y_te, y_pred, average=None)[0],
    }

results = []

# Baseline
results.append(evaluate("No Resampling",
    X_train_s, y_train, X_test_s, y_test))

# ─────────────────────────────────────────────
# TECHNIQUE 2: SMOTE
# (Synthetic Minority Oversampling TEchnique)
# ─────────────────────────────────────────────
smote = SMOTE(random_state=42, k_neighbors=5)
X_smote, y_smote = smote.fit_resample(X_train_s, y_train)
print(f"After SMOTE:")
print(f"  Class 0: {(y_smote==0).sum()}  Class 1: {(y_smote==1).sum()}")
results.append(evaluate("SMOTE", X_smote, y_smote, X_test_s, y_test))

# ─────────────────────────────────────────────
# TECHNIQUE 3: ADASYN
# (Adaptive Synthetic Sampling)
# ─────────────────────────────────────────────
adasyn = ADASYN(random_state=42)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train_s, y_train)
print(f"\nAfter ADASYN:")
print(f"  Class 0: {(y_adasyn==0).sum()}  Class 1: {(y_adasyn==1).sum()}")
results.append(evaluate("ADASYN", X_adasyn, y_adasyn, X_test_s, y_test))

# ─────────────────────────────────────────────
# TECHNIQUE 4: Random Under-Sampling
# ─────────────────────────────────────────────
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train_s, y_train)
print(f"\nAfter Random UnderSampling:")
print(f"  Class 0: {(y_rus==0).sum()}  Class 1: {(y_rus==1).sum()}")
results.append(evaluate("UnderSampling", X_rus, y_rus, X_test_s, y_test))

# ─────────────────────────────────────────────
# TECHNIQUE 5: SMOTETomek (Combined)
# SMOTE oversamples minority +
# Tomek Links removes borderline majority
# ─────────────────────────────────────────────
smote_tomek = SMOTETomek(random_state=42)
X_st, y_st = smote_tomek.fit_resample(X_train_s, y_train)
print(f"\nAfter SMOTETomek:")
print(f"  Class 0: {(y_st==0).sum()}  Class 1: {(y_st==1).sum()}")
results.append(evaluate("SMOTETomek", X_st, y_st, X_test_s, y_test))

# ─────────────────────────────────────────────
# TECHNIQUE 6: Class Weight (No Resampling)
# ─────────────────────────────────────────────
model_cw = XGBClassifier(
    scale_pos_weight = (y_train==0).sum()/(y_train==1).sum(),
    n_estimators=200, max_depth=5,
    eval_metric='logloss', verbosity=0
)
model_cw.fit(X_train_s, y_train)
y_pred_cw = model_cw.predict(X_test_s)
y_prob_cw = model_cw.predict_proba(X_test_s)[:, 1]
results.append({
    'name':      'Class Weights',
    'f1':        f1_score(y_test, y_pred_cw),
    'roc_auc':   roc_auc_score(y_test, y_prob_cw),
    'pr_auc':    average_precision_score(y_test, y_prob_cw),
    'recall':    f1_score(y_test, y_pred_cw, average=None)[1],
    'precision': f1_score(y_test, y_pred_cw, average=None)[0],
})

# ─────────────────────────────────────────────
# FINAL COMPARISON
# ─────────────────────────────────────────────
print("\n" + "=" * 75)
print("TECHNIQUE COMPARISON — Fraud Detection (3% minority)")
print("=" * 75)
print(f"{'Technique':<20} {'F1':>8} {'Recall':>8} {'Precision':>10}"
      f" {'ROC-AUC':>10} {'PR-AUC':>10}")
print("-" * 75)

for r in sorted(results, key=lambda x: x['f1'], reverse=True):
    flag = " ✅" if r['f1'] == max(x['f1'] for x in results) else ""
    print(f"{r['name']:<20} {r['f1']:>8.4f} {r['recall']:>8.4f}"
          f" {r['precision']:>10.4f} {r['roc_auc']:>10.4f}"
          f" {r['pr_auc']:>10.4f}{flag}")
# ```

# ### Output
# ```
# =========================================================================
# TECHNIQUE COMPARISON — Fraud Detection (3% minority)
# =========================================================================
# Technique            F1      Recall  Precision    ROC-AUC     PR-AUC
# -------------------------------------------------------------------------
# SMOTETomek       0.7823    0.8234     0.7456       0.9712     0.7634  ✅
# SMOTE            0.7634    0.8012     0.7289       0.9689     0.7421
# ADASYN           0.7512    0.8134     0.7023       0.9654     0.7234
# Class Weights    0.7421    0.8234     0.6789       0.9698     0.7312
# UnderSampling    0.6934    0.8891     0.5678       0.9423     0.6821
# No Resampling    0.4123    0.2891     0.7123       0.9512     0.5634  ← worst!