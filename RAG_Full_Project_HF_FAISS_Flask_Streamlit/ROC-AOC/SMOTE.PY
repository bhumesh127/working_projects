import numpy as np
from sklearn.neighbors import NearestNeighbors

# SMOTE Step-by-Step Explanation
print("""
HOW SMOTE WORKS:
═══════════════════════════════════════════════

Step 1: Find minority class samples
        Fraud cases: [A, B, C, D, E] (only 5 fraud)

Step 2: For each minority sample, find K nearest neighbors
        Sample A's neighbors: [B, C, D]

Step 3: Randomly pick a neighbor
        Picked: B

Step 4: Generate synthetic point BETWEEN A and B
        Synthetic = A + random(0,1) × (B - A)

        A ●────────────●Synthetic────────● B
          ↑                              ↑
        Real fraud                  Real fraud
        
Step 5: Add synthetic point to training data
        New fraud cases: [A, B, C, D, E, Synthetic1, Synthetic2...]

RESULT: Minority class grows without just copying!
        Model learns broader fraud patterns.
═══════════════════════════════════════════════
""")

# Manual SMOTE illustration
np.random.seed(42)
minority_samples = np.array([
    [2.1, 3.4],   # Fraud A
    [2.3, 3.1],   # Fraud B
    [1.9, 3.6],   # Fraud C
])

# Generate 3 synthetic samples between A and B
print("Original Fraud Samples:")
for i, s in enumerate(minority_samples):
    print(f"  Sample {chr(65+i)}: {s}")

synthetic_samples = []
for _ in range(3):
    idx1, idx2 = np.random.choice(len(minority_samples), 2, replace=False)
    alpha = np.random.random()
    synthetic = minority_samples[idx1] + alpha * (
        minority_samples[idx2] - minority_samples[idx1]
    )
    synthetic_samples.append(synthetic)

print("\nSMOTE Generated Synthetic Samples:")
for i, s in enumerate(synthetic_samples):
    print(f"  Synthetic {i+1}: {s.round(3)} ← interpolated between real frauds")

print("""
WHICH TECHNIQUE TO USE?
═══════════════════════════════════════════════════════

Imbalance Ratio    Recommended Strategy
───────────────────────────────────────────────────────
    2:1  –  5:1  → Class Weights (simplest, effective)

    5:1  – 10:1  → SMOTE (oversample minority)

   10:1  – 50:1  → SMOTETomek or SMOTEENN
                   (oversample + clean boundary)

   50:1  – 100:1 → ADASYN (adaptive, focuses on hard cases)
                   + Class Weights combined

  100:1+          → Anomaly Detection approach instead!
                   (IsolationForest, Autoencoder)

YOUR FRAUD CASE (98:2):
  ✅ Use: SMOTETomek + scale_pos_weight in XGBoost
  ✅ Evaluate with: PR-AUC (not ROC-AUC!)
  ✅ Optimize for: Recall (catch all fraud!)
  ✅ Threshold: Tune below 0.5 for higher recall
═══════════════════════════════════════════════════════
""")


## Complete Summary
# ```
# ┌─────────────────────┬────────────────────────────────────────────────┐
# │ Topic               │ Key Takeaway                                   │
# ├─────────────────────┼────────────────────────────────────────────────┤
# │ ROC Curve           │ TPR vs FPR at every threshold                  │
# │                     │ AUC closer to 1.0 = better model               │
# │                     │ Use Youden's J for optimal threshold            │
# ├─────────────────────┼────────────────────────────────────────────────┤
# │ PR Curve            │ Precision vs Recall at every threshold          │
# │                     │ Better than ROC for imbalanced data             │
# │                     │ Use Average Precision (AP) as summary score     │
# ├─────────────────────┼────────────────────────────────────────────────┤
# │ SMOTE               │ Creates synthetic minority samples              │
# │                     │ Interpolates between real minority points       │
# ├─────────────────────┼────────────────────────────────────────────────┤
# │ ADASYN              │ Adaptive SMOTE — focuses on hard-to-learn       │
# │                     │ boundary cases                                  │
# ├─────────────────────┼────────────────────────────────────────────────┤
# │ SMOTETomek          │ Best of both: oversample minority +             │
# │                     │ clean majority boundary → cleanest data ✅      │
# ├─────────────────────┼────────────────────────────────────────────────┤
# │ Class Weights       │ No resampling — penalize model for missing      │
# │                     │ minority class — simple & effective             │
# └─────────────────────┴────────────────────────────────────────────────┘