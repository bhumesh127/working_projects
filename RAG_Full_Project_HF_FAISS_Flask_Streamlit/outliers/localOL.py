from sklearn.neighbors import LocalOutlierFactor

# LOF: Detects outliers based on local density
# Good for non-uniform data distributions

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05
)

claims['outlier_lof'] = lof.fit_predict(X_scaled)

print(f"\n🔍 LOF Outlier Detection:")
print(f"   Outliers found: {(claims['outlier_lof'] == -1).sum()}")