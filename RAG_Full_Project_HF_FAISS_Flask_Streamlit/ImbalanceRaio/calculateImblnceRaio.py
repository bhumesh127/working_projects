import numpy as np
import pandas as pd
from collections import Counter

# ── Healthcare Dataset ──
np.random.seed(42)
n = 1000

labels = np.array(
    [0]*850 +   # No Disease  (majority)
    [1]*100 +   # Pre-Diabetic
    [2]*50      # Diabetic    (minority)
)
np.random.shuffle(labels)

# Count classes
class_counts = Counter(labels)
print("=" * 50)
print("CLASS DISTRIBUTION")
print("=" * 50)
for cls, count in sorted(class_counts.items()):
    name = {0:"No Disease", 1:"Pre-Diabetic", 2:"Diabetic"}[cls]
    bar  = "█" * (count // 10)
    pct  = count / len(labels) * 100
    print(f"  Class {cls} ({name:<12}): {count:>4} ({pct:>5.1f}%)  {bar}")

# Binary Imbalance Ratio
majority = max(class_counts.values())
minority = min(class_counts.values())
ir = majority / minority

print(f"\n  Majority Class Count : {majority}")
print(f"  Minority Class Count : {minority}")
print(f"  Imbalance Ratio      : {ir:.1f}:1")
print(f"  Meaning              : For every 1 Diabetic,")
print(f"                         there are {ir:.0f} No-Disease patients")
# ```

# ### Output
# ```
# ==================================================
# CLASS DISTRIBUTION
# ==================================================
#   Class 0 (No Disease  ):  850 (85.0%)  █████████████████████████████████████████████████████████████████████████████████████
#   Class 1 (Pre-Diabetic):  100 (10.0%)  ██████████
#   Class 2 (Diabetic    ):   50  (5.0%)  █████

#   Majority Class Count : 850
#   Minority Class Count :  50
#   Imbalance Ratio      : 17.0:1
#   Meaning              : For every 1 Diabetic,
#                          there are 17 No-Disease patients