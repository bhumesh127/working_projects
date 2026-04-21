import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import optuna
import numpy as np
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = make_classification(n_samples=1000, n_features=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

results = {}

# ── GridSearchCV ──
start = time.time()
grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss', verbosity=0),
    {'max_depth': [3,5,7], 'n_estimators': [100,200], 'learning_rate': [0.01,0.1]},
    cv=3, scoring='roc_auc', n_jobs=-1
)
grid.fit(X_train, y_train)
results['GridSearch'] = {
    'auc':  roc_auc_score(y_test, grid.predict_proba(X_test)[:,1]),
    'time': round(time.time() - start, 2)
}

# ── RandomizedSearchCV ──
from scipy.stats import randint, uniform
start = time.time()
rand = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', verbosity=0),
    {'max_depth': randint(2,10), 'n_estimators': randint(50,300),
     'learning_rate': uniform(0.01, 0.29)},
    n_iter=20, cv=3, scoring='roc_auc', random_state=42, n_jobs=-1
)
rand.fit(X_train, y_train)
results['RandomSearch'] = {
    'auc':  roc_auc_score(y_test, rand.predict_proba(X_test)[:,1]),
    'time': round(time.time() - start, 2)
}

# ── Optuna ──
start = time.time()
def obj(trial):
    m = XGBClassifier(
        max_depth=trial.suggest_int('max_depth', 2, 10),
        n_estimators=trial.suggest_int('n_estimators', 50, 300),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
        eval_metric='logloss', verbosity=0
    )
    from sklearn.model_selection import cross_val_score
    return cross_val_score(m, X_train, y_train, cv=3, scoring='roc_auc').mean()

s = optuna.create_study(direction='maximize')
s.optimize(obj, n_trials=20)
best = XGBClassifier(**s.best_params, eval_metric='logloss', verbosity=0)
best.fit(X_train, y_train)
results['Optuna'] = {
    'auc':  roc_auc_score(y_test, best.predict_proba(X_test)[:,1]),
    'time': round(time.time() - start, 2)
}

# ── Print Comparison ──
print(f"\n{'Method':<15} {'ROC-AUC':>10} {'Time (s)':>10}")
print("=" * 38)
for method, res in results.items():
    print(f"{method:<15} {res['auc']:>10.4f} {res['time']:>10.2f}s")
