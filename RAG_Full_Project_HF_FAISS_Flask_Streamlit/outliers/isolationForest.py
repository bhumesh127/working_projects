from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Best for multi-dimensional outlier detection
X_claims = claims[['claim_amount', 'age', 'length_of_stay']].copy()
X_claims = X_claims.fillna(X_claims.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_claims)

iso_forest = IsolationForest(
    contamination=0.05,   # Expect 5% outliers
    random_state=42,
    n_estimators=100
)

claims['outlier_iso'] = iso_forest.fit_predict(X_scaled)
# -1 = outlier, 1 = normal

outliers_iso = claims[claims['outlier_iso'] == -1]
normals_iso  = claims[claims['outlier_iso'] == 1]

print(f"\n🌲 Isolation Forest Results:")
print(f"   Normal  records : {len(normals_iso)}")
print(f"   Outlier records : {len(outliers_iso)}")
print(f"\n   Sample Outliers Detected:")
print(outliers_iso[['patient_id', 'claim_amount',
                     'age', 'length_of_stay']].head(8).to_string(index=False))
# ```

# ### Output
# ```
# 🌲 Isolation Forest Results:
#    Normal  records : 476
#    Outlier records : 24

#    Sample Outliers Detected:
#    patient_id  claim_amount    age  length_of_stay
#           491      85000.0  55.12          5.21
#           492      92000.0  52.34          4.89
#           496       -500.0  52.11         -3.00
#           498     150000.0  200.0        250.00