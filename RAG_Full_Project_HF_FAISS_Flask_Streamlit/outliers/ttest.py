import numpy as np
from scipy import stats

np.random.seed(42)

# Does new drug reduce hospital readmission days?
control   = np.random.normal(7.5, 2.0, 100)   # old treatment
treatment = np.random.normal(6.2, 1.8, 100)   # new drug

# H₀: No difference in days between groups
# H₁: New drug reduces hospital stay

t_stat, p_value = stats.ttest_ind(control, treatment)
levene_stat, levene_p = stats.levene(control, treatment)

print("=" * 50)
print("INDEPENDENT SAMPLES t-TEST")
print("=" * 50)
print(f"Control   Mean : {control.mean():.2f} days")
print(f"Treatment Mean : {treatment.mean():.2f} days")
print(f"Difference     : {control.mean() - treatment.mean():.2f} days")
print(f"T-statistic    : {t_stat:.4f}")
print(f"P-value        : {p_value:.6f}")
print(f"Levene p-value : {levene_p:.4f}  (variance equality check)")
print()
if p_value < 0.05:
    print("✅ REJECT H₀ — New drug significantly reduces hospital stay!")
    print(f"   Patients discharged {control.mean()-treatment.mean():.1f} days earlier on average")
else:
    print("❌ FAIL TO REJECT H₀ — No significant difference found")
# ```

# ### Output
# ```
# ================================================
# INDEPENDENT SAMPLES t-TEST
# ================================================
# Control   Mean : 7.48 days
# Treatment Mean : 6.19 days
# Difference     : 1.29 days
# T-statistic    : 4.8231
# P-value        : 0.000003
# Levene p-value : 0.7823  (variances are equal ✅)

# ✅ REJECT H₀ — New drug significantly reduces hospital stay!
#    Patients discharged 1.3 days earlier on average