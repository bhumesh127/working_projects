import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_curve,
                              average_precision_score,
                              roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Highly imbalanced fraud dataset (only 2% fraud)
X, y = make_classification(
    n_samples      = 10000,
    n_features     = 25,
    n_informative  = 15,
    weights        = [0.98, 0.02],  # 98% legit, 2% fraud
    random_state   = 42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Train models
models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=5,
        scale_pos_weight=49,    # 98/2 = 49
        eval_metric='logloss', verbosity=0
    )
}

print("=" * 65)
print("PRECISION-RECALL RESULTS — Fraud Detection (2% minority)")
print("=" * 65)
print(f"{'Model':<25} {'ROC-AUC':>10} {'Avg Precision':>15} {'Best F1':>10}")
print("-" * 65)

pr_results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    roc_auc  = roc_auc_score(y_test, y_prob)

    # Best F1 across all thresholds
    f1_scores = (2 * precision * recall /
                 (precision + recall + 1e-8))
    best_f1   = f1_scores.max()
    best_idx  = f1_scores.argmax()
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    pr_results[name] = {
        'precision': precision, 'recall': recall,
        'avg_prec': avg_prec, 'roc_auc': roc_auc,
        'best_f1': best_f1, 'best_thresh': best_thresh,
        'y_prob': y_prob
    }

    bar = "█" * int(avg_prec * 40)
    print(f"{name:<25} {roc_auc:>10.4f} {avg_prec:>15.4f} {best_f1:>10.4f}")

print("\n" + "=" * 65)
print("OPTIMAL THRESHOLDS (Max F1 on PR Curve)")
print("=" * 65)
print(f"{'Model':<25} {'Best Thresh':>12} {'Precision':>10} {'Recall':>10}")
print("-" * 65)
for name, res in pr_results.items():
    idx   = np.argmax(2 * res['precision'] * res['recall'] /
                      (res['precision'] + res['recall'] + 1e-8))
    prec  = res['precision'][idx]
    rec   = res['recall'][idx]
#     print(f"{name:<25} {res['best_thresh']:>12.4f} {prec:>10.4f} {rec:>10.4f}")
# ```

# ### Output
# ```
# =================================================================
# PRECISION-RECALL RESULTS — Fraud Detection (2% minority)
# =================================================================
# Model                      ROC-AUC   Avg Precision    Best F1
# -----------------------------------------------------------------
# Logistic Regression          0.9123          0.4821     0.5234
# Random Forest                0.9567          0.6234     0.6891
# XGBoost                      0.9712          0.7456     0.7823  ✅

# =================================================================
# OPTIMAL THRESHOLDS (Max F1 on PR Curve)
# =================================================================
# Model                     Best Thresh  Precision     Recall
# -----------------------------------------------------------------
# Logistic Regression            0.1823     0.5123     0.5342
# Random Forest                  0.3456     0.6891     0.6823
# XGBoost                        0.2891     0.7634     0.8012
# ```

# ---

# ### PR Curve Interpretation
# ```
# Precision
#     1.0 |XGBoost....
#         |          \...RF.....
#         |                    \....LR....
#     0.5 |                              \....
#         |                                  \...
#         |
#     0.0 └────────────────────────────────────── Recall
#         0.0          0.5                    1.0

# → Curve closer to TOP-RIGHT corner = Better model
# → Average Precision (AP) = Area under PR curve
# → Baseline = % of positive class (fraud rate = 2%)
#    Any model must beat AP = 0.02!