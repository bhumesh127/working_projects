import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Highly imbalanced — only 2% fraud
X, y = make_classification(
    n_samples     = 5000,
    n_features    = 20,
    n_informative = 12,
    weights       = [0.98, 0.02],   # 98% legit, 2% fraud
    random_state  = 42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = XGBClassifier(
    scale_pos_weight = 49,   # Handle imbalance: 98/2 = 49
    n_estimators     = 200,
    max_depth        = 5,
    eval_metric      = 'logloss',
    verbosity        = 0
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print("=" * 60)
print("CONFUSION MATRIX — Fraud Detection")
print("=" * 60)
print(f"""
                   PREDICTED
               Legit      Fraud
            ┌──────────┬──────────┐
ACTUAL Legit│  TN={TN:>4}  │  FP={FP:>4}  │ ← False alerts (annoying)
       Fraud│  FN={FN:>4}  │  TP={TP:>4}  │ ← Missed fraud (costly!)
            └──────────┴──────────┘
""")

precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
recall      = TP / (TP + FN) if (TP + FN) > 0 else 0
f1          = 2*(precision*recall)/(precision+recall) if (precision+recall) > 0 else 0
accuracy    = (TP+TN)/(TP+TN+FP+FN)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}  → Less false alerts on legit transactions")
print(f"Recall    : {recall:.4f}  → Catching actual fraud cases")
print(f"F1-Score  : {f1:.4f}  → Balance (important for imbalanced data)")
print(f"\n💡 In fraud detection:")
print(f"   HIGH RECALL preferred → Better to flag legit as fraud")
print(f"   than to MISS actual fraud (costs company millions!)")