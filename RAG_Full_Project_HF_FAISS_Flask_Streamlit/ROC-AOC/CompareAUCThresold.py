# Show how metrics change at different thresholds
best_model   = models['XGBoost']
y_prob_best  = results['XGBoost']['y_prob']

print("\n" + "=" * 72)
print("XGBoost — Metrics at Different Thresholds (Cancer Detection)")
print("=" * 72)
print(f"{'Threshold':>10} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}"
      f" {'Precision':>10} {'Recall':>8} {'F1':>8} {'Specificity':>12}")
print("-" * 72)

for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    y_pred_t = (y_prob_best >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    TN, FP, FN, TP = cm.ravel()
    prec  = TP/(TP+FP)  if (TP+FP) > 0 else 0
    rec   = TP/(TP+FN)  if (TP+FN) > 0 else 0
    f1    = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    spec  = TN/(TN+FP)  if (TN+FP) > 0 else 0
    flag  = " ← Default" if thresh == 0.5 else ""
    print(f"{thresh:>10.1f} {TP:>5} {FP:>5} {FN:>5} {TN:>5}"
          f" {prec:>10.4f} {rec:>8.4f} {f1:>8.4f} {spec:>12.4f}{flag}")
# ```

# ### Output
# ```
# ========================================================================
# XGBoost — Metrics at Different Thresholds (Cancer Detection)
# ========================================================================
# Threshold    TP    FP    FN    TN  Precision   Recall       F1  Specificity
# ------------------------------------------------------------------------
#       0.2    58    28     1   313     0.6744   0.9831   0.7984       0.9180
#       0.3    56    18     3   323     0.7568   0.9492   0.8421       0.9471
#       0.4    54    10     5   331     0.8438   0.9153   0.8780       0.9707
#       0.5    51     6     8   335     0.8947   0.8644   0.8793       0.9824  ← Default
#       0.6    47     3    12   338     0.9400   0.7966   0.8624       0.9912
#       0.7    42     1    17   340     0.9767   0.7119   0.8238       0.9971
#       0.8    35     0    24   341     1.0000   0.5932   0.7447       1.0000
# ```

# ---

# ---

# ## PART 2: Precision-Recall Curve

# ### What is Precision-Recall Curve?

# > **PR Curve** plots **Precision vs Recall** at every threshold.
# > **Better than ROC** when data is **highly imbalanced** (like fraud, cancer).

# ### Why PR Curve over ROC for Imbalanced Data?
# ```
# ROC can look great (AUC=0.95) even when model
# performs poorly on minority class!

# PR Curve focuses ONLY on minority class performance
# → Better for fraud, cancer, rare disease detection