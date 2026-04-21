import optuna
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = make_classification(n_samples=1000, n_features=20,
                            n_informative=8, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Optuna tunes BOTH feature count (ANOVA k) AND XGBoost params!
def combined_objective(trial):
    k = trial.suggest_int('k_features', 3, 15)  # How many features to keep

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('anova',  SelectKBest(f_classif, k=k)),
        ('model',  XGBClassifier(
            max_depth        = trial.suggest_int('max_depth', 2, 8),
            n_estimators     = trial.suggest_int('n_estimators', 50, 300),
            learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample        = trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0),
            eval_metric      = 'logloss',
            verbosity        = 0,
        ))
    ])

    return cross_val_score(
        pipeline, X_train, y_train,
        cv=5, scoring='roc_auc', n_jobs=-1
    ).mean()


study = optuna.create_study(direction='maximize')
print("🚀 Running ANOVA + Optuna Combined Optimization...")
study.optimize(combined_objective, n_trials=40)

print(f"\n✅ Best ROC-AUC:  {study.best_value:.4f}")
print(f"✅ Best k (features to keep): {study.best_params['k_features']}")
print(f"✅ Best XGBoost Params:")
for k, v in study.best_params.items():
    if k != 'k_features':
        print(f"   {k:<25}: {v}")
