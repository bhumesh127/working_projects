def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) |
                    (data[column] > upper_bound)]

    print(f"\n📊 IQR Outliers in '{column}':")
    print(f"   Q1            : {Q1:.2f}")
    print(f"   Q3            : {Q3:.2f}")
    print(f"   IQR           : {IQR:.2f}")
    print(f"   Lower Bound   : {lower_bound:.2f}")
    print(f"   Upper Bound   : {upper_bound:.2f}")
    print(f"   Outliers Found: {len(outliers)}")
    return lower_bound, upper_bound

lb, ub = detect_outliers_iqr(claims, 'claim_amount')
detect_outliers_iqr(claims, 'age')
detect_outliers_iqr(claims, 'length_of_stay')
# ```

# ### Output
# ```
# 📊 IQR Outliers in 'claim_amount':
#    Q1            : 4327.18
#    Q3            : 5682.44
#    IQR           : 1355.26
#    Lower Bound   : 2294.29
#    Upper Bound   : 7715.33
#    Outliers Found: 14

# 📊 IQR Outliers in 'age':
#    Lower Bound   : 29.14
#    Upper Bound   : 80.86
#    Outliers Found: 5