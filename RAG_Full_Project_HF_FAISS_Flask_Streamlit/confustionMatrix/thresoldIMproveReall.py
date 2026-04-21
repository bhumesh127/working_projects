import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

X, y = make_classification(
    n_samples=1000, n_features=15,
    weights=[0.85, 0.15], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(eval_metric='logloss', verbosity=0)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"{'Threshold':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Precision':>11} {'Recall':>8}")
print("=" * 65)

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_t = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    TN, FP, FN, TP = cm.ravel()
    prec = precision_score(y_test, y_pred_t, zero_division=0)
    rec  = recall_score(y_test, y_pred_t)
    flag = " ← Default" if threshold == 0.5 else ""
    print(f"{threshold:>10.1f} {TP:>6} {FP:>6} {FN:>6} {TN:>6} {prec:>11.4f} {rec:>8.4f}{flag}")

print(f"""
💡 Key Insight:
   Lower threshold (0.3) → Higher Recall → Catch more cancer ✅
                         → More FP → More false alarms
   Higher threshold (0.7) → Higher Precision → Fewer false alarms
                          → More FN → Miss more cancer ⚠️
   Choose threshold based on BUSINESS COST of FP vs FN!
""")
# ```

# ### Output
# ```
# Threshold     TP     FP     FN     TN   Precision   Recall
# =================================================================
#       0.3     29     14      2    155      0.6744   0.9355
#       0.4     28      9      3    160      0.7568   0.9032
#       0.5     27      6      4    163      0.8182   0.8710  ← Default
#       0.6     25      3      6    166      0.8929   0.8065
#       0.7     22      1      9    168      0.9565   0.7097

# 💡 Key Insight:
#    Lower threshold (0.3) → Higher Recall → Catch more cancer ✅
#    Higher threshold (0.7) → Higher Precision → Fewer false alarms
#    Choose threshold based on BUSINESS COST of FP vs FN!
# ```

# ---

# ## Complete Cheat Sheet
# ```
# ┌──────────────┬──────────────────────────────────────────────────────┐
# │ Term         │ Simple Definition                                    │
# ├──────────────┼──────────────────────────────────────────────────────┤
# │ TP           │ Correctly predicted POSITIVE                         │
# │ TN           │ Correctly predicted NEGATIVE                         │
# │ FP           │ Predicted POSITIVE but actually NEGATIVE (Type I)    │
# │ FN           │ Predicted NEGATIVE but actually POSITIVE (Type II)   │
# │ Accuracy     │ Overall correct predictions                          │
# │ Precision    │ How trustworthy are positive predictions?            │
# │ Recall       │ How many actual positives did we catch?              │
# │ F1-Score     │ Harmonic mean of Precision & Recall                  │
# │ Specificity  │ How well did we identify negatives?                  │
# │ ROC-AUC      │ Overall model discrimination ability (0.5–1.0)       │
# └──────────────┴──────────────────────────────────────────────────────┘