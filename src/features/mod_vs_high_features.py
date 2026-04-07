import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_complete_covariates_fixed_1229.csv')
df['SBS'] = df['SBS'].replace({'mod': 'moderate'})

# Filter to moderate and high only
df_binary = df[df['SBS'].isin(['moderate', 'high'])].copy()
print(f"Moderate: {(df_binary['SBS']=='moderate').sum()}, High: {(df_binary['SBS']=='high').sum()}")

TOP_30_PLUS_COORDS = [
    'dnbr', 'dndvi', 'dndbi', 'dbsi', 'nbr', 'bsi', 'ndvi', 'ndbi',
    'meanelev_32', 'wc_bio19', 'nirBand', 'wc_bio05', 'rdgh_6', 'blueBand',
    'minelev_4', 'greenBand', 'wc_bio06', 'swir2Band', 'pisrdif_2021-11-22',
    'pisrdif_2021-12-22', 'stddevelev_32', 'maxc_2', 'wc_bio12', 'wc_bio07',
    'dmndwi', 'wc_bio18', 'wc_bio17', 'wc_bio02', 'vd_5', 'planc_32',
    'latitude', 'longitude'  # added
]

available = [f for f in TOP_30_PLUS_COORDS if f in df_binary.columns]
X = df_binary[available].fillna(df_binary[available].median()).values.astype(np.float32)
X = np.where(np.isnan(X) | np.isinf(X), 0.0, X)
y = (df_binary['SBS'] == 'high').astype(int).values

rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X, y)

# Plot importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(len(available)), importances[indices])
plt.xticks(range(len(available)), [available[i] for i in indices], rotation=90, fontsize=8)
plt.title('Binary Moderate vs High — Feature Importance (with coordinates)')
plt.tight_layout()
plt.savefig('/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/results_mod_vs_high/binary_mod_high_importance.png', dpi=200)
plt.close()

print("\nTop 10 features for moderate/high separation:")
for i in range(10):
    print(f"  {i+1:2d}. {available[indices[i]]:30s} {importances[indices[i]]:.4f}")