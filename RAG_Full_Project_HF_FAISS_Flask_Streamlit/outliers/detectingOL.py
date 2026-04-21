import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Simulate healthcare claims data
n = 500
claims = pd.DataFrame({
    'patient_id':      range(1, n+1),
    'claim_amount':    np.concatenate([
                          np.random.normal(5000, 1000, 490),   # normal claims
                          [85000, 92000, 78000, 95000, 88000,  # outliers (fraud?)
                           -500, -1200, 150000, 200000, -800]  # negative + extreme
                       ]),
    'age':             np.concatenate([
                          np.random.normal(55, 12, 495),
                          [150, 180, -5, 200, -10]             # impossible ages
                       ]),
    'length_of_stay':  np.concatenate([
                          np.random.normal(5, 2, 490),
                          [120, 145, 0.001, 200, 180,          # extreme stays
                           -3, -5, 250, 300, -1]
                       ])
})

print("=" * 55)
print("RAW DATA STATISTICS")
print("=" * 55)
print(claims[['claim_amount', 'age', 'length_of_stay']].describe().round(2))