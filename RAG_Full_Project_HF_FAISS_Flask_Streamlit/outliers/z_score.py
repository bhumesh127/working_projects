from scipy import stats

# Z-score: how many standard deviations away from mean
# |Z| > 3 = outlier (99.7% of normal data falls within 3 std)

def detect_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    outlier_mask = z_scores > threshold
    outliers = data[column][outlier_mask]
    print(f"\n📊 Z-Score Outliers in '{column}':")
    print(f"   Total records  : {len(data)}")
    print(f"   Outliers found : {outlier_mask.sum()}")
    print(f"   Outlier values : {sorted(outliers.values)[:5]}...")
    return outlier_mask

z_mask_claims = detect_outliers_zscore(claims, 'claim_amount')
z_mask_age    = detect_outliers_zscore(claims, 'age')
z_mask_stay   = detect_outliers_zscore(claims, 'length_of_stay')
# ```

# ### Output
# ```
# 📊 Z-Score Outliers in 'claim_amount':
#    Total records  : 500
#    Outliers found : 12
#    Outlier values : [-1200, -800, -500, 78000, 85000]...

# 📊 Z-Score Outliers in 'age':
#    Total records  : 500
#    Outliers found : 5
#    Outlier values : [-10, -5, 150, 180, 200]...