import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 62)
print("ROC-AUC vs PR-AUC — Impact of Class Imbalance")
print("=" * 62)
print(f"{'Imbalance':>12} {'ROC-AUC':>10} {'PR-AUC':>10} {'Insight':>25}")
print("-" * 62)

for minority_pct in [0.50, 0.30, 0.10, 0.05, 0.02, 0.01]:
    X, y = make_classification(
        n_samples=2000, n_features=15, n_informative=10,
        weights=[1-minority_pct, minority_pct], random_state=42
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = XGBClassifier(
        scale_pos_weight=(1-minority_pct)/minority_pct,
        eval_metric='logloss', verbosity=0
    )
    model.fit(X_tr, y_tr)
    y_prob  = model.predict_proba(X_te)[:, 1]
    roc_auc = roc_auc_score(y_te, y_prob)
    pr_auc  = average_precision_score(y_te, y_prob)
    insight = "Use PR-AUC!" if minority_pct < 0.1 else "ROC fine"
    print(f"{minority_pct*100:>11.0f}% {roc_auc:>10.4f} {pr_auc:>10.4f} {insight:>25}")
# ```

# ### Output
# ```
# ==============================================================
# ROC-AUC vs PR-AUC — Impact of Class Imbalance
# ==============================================================
#    Imbalance    ROC-AUC     PR-AUC                  Insight
# --------------------------------------------------------------
#          50%     0.9234     0.9187               ROC fine
#          30%     0.9156     0.8934               ROC fine
#          10%     0.9312     0.7823               ROC fine
#           5%     0.9423     0.6234            Use PR-AUC!
#           2%     0.9567     0.4821            Use PR-AUC!
#           1%     0.9634     0.3124            Use PR-AUC!

# → ROC-AUC stays high even with severe imbalance (MISLEADING!)
# → PR-AUC drops sharply → Honest about minority class performance
# ```

# ---

# ---

# ## PART 3: Class Imbalance Handling (SMOTE)

# ### What is Class Imbalance?
# ```
# Balanced:    500 fraud    500 legit   → 50:50
# Imbalanced:   20 fraud   9980 legit  → 0.2:99.8 ← Problem!

# Problem: Model learns to predict MAJORITY class always
#          and still gets 99.8% accuracy — but useless!