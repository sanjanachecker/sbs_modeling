"""
Diagnostic: Why isn't 'high' severity being predicted?
======================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

print("="*80)
print("DIAGNOSING THE 'HIGH' CLASS PROBLEM")
print("="*80)

# Load the imputed dataset
df = pd.read_csv('results/imputed_dataset.csv')

print("\n1. OVERALL CLASS DISTRIBUTION")
print("-"*80)
class_dist = df['SBS'].value_counts()
total = len(df)
for cls, count in class_dist.items():
    pct = count / total * 100
    print(f"{cls:12s}: {count:5d} samples ({pct:5.1f}%)")

imbalance_ratio = class_dist.max() / class_dist.min()
print(f"\nImbalance ratio: {imbalance_ratio:.1f}:1")

print("\n2. DISTRIBUTION BY FIRE")
print("-"*80)
fires_with_high = df[df['SBS'] == 'high'].groupby('Fire_year').size()
print(f"Fires with 'high' severity: {len(fires_with_high)}")
print(f"Total fires: {df['Fire_year'].nunique()}")
print(f"\nTop fires by 'high' severity count:")
print(fires_with_high.sort_values(ascending=False).head(10))

print("\n3. POTENTIAL ISSUES")
print("-"*80)

# Issue 1: Too few high severity samples
high_count = (df['SBS'] == 'high').sum()
print(f"\n⚠️  Issue 1: Very few 'high' samples")
print(f"   Total 'high' samples: {high_count}")
print(f"   Percentage: {high_count/len(df)*100:.2f}%")
print(f"   Recommendation: Need at least 10-15% for good learning")

# Issue 2: High samples concentrated in few fires
fires_with_high_sorted = fires_with_high.sort_values(ascending=False)
top_3_fires = fires_with_high_sorted.head(3).sum()
print(f"\n⚠️  Issue 2: 'High' samples concentrated")
print(f"   Top 3 fires contain: {top_3_fires}/{high_count} high samples ({top_3_fires/high_count*100:.1f}%)")
print(f"   This means: If these fires end up in test set, model has little training data")

# Issue 3: Check if high samples are in train/val/test evenly
print("\n4. CHECKING DATA SPLIT")
print("-"*80)
print("If fires with lots of 'high' samples ended up in test set,")
print("the model never learned to predict them properly!")
print("\nFires with most 'high' samples:")
for fire, count in fires_with_high_sorted.head(5).items():
    total_samples = (df['Fire_year'] == fire).sum()
    pct = count / total_samples * 100
    print(f"  {fire:30s}: {count:3d} high / {total_samples:4d} total ({pct:5.1f}%)")

print("\n5. FEATURE SEPARABILITY TEST")
print("-"*80)
print("Checking if 'high' class is distinguishable from others...")

# Check top features for each class
top_features = ['dnbr', 'nbr', 'wc_bio06', 'greenBand', 'swir2Band']
available_features = [f for f in top_features if f in df.columns]

for feature in available_features[:3]:  # Just check top 3
    print(f"\n{feature}:")
    for cls in ['high', 'moderate', 'low', 'unburned']:
        if cls in df['SBS'].unique():
            values = df[df['SBS'] == cls][feature]
            if not values.isna().all():
                print(f"  {cls:12s}: mean={values.mean():7.2f}, std={values.std():7.2f}")

print("\n6. RECOMMENDED SOLUTIONS")
print("="*80)
print()
print("Solution 1: AGGRESSIVE UPSAMPLING FOR 'HIGH' CLASS")
print("-"*50)
print("Modify the upsampling logic to specifically target 'high' class:")
print("""
# In select_upsampled_points or create a new function:
# For each fire, sample MORE points where high severity exists

for fire in fires:
    high_count = (df_complete[(df_complete['Fire_year']==fire) & 
                              (df_complete['SBS']=='high')]).shape[0]
    
    if high_count > 0:
        # MUCH MORE aggressive for high class
        quota_high = int(5 * np.sqrt(high_count))  # 5x multiplier!
        
        # Sample from upsampled 'high' points
        high_upsampled = df_upsampled[(df_upsampled['Fire_year']==fire) & 
                                       (df_upsampled['SBS']=='high')]
        if len(high_upsampled) > 0:
            sample_high = high_upsampled.sample(min(quota_high, len(high_upsampled)))
""")

print("\nSolution 2: SYNTHETIC MINORITY OVERSAMPLING (SMOTE)")
print("-"*50)
print("Use SMOTE specifically for the 'high' class:")
print("""
from imblearn.over_sampling import SMOTE

# After creating X_train, y_train
smote = SMOTE(sampling_strategy={'high': int(0.25 * len(y_train))},
              random_state=RANDOM_STATE)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
""")

print("\nSolution 3: EXTREME CLASS WEIGHTS FOR 'HIGH'")
print("-"*50)
print("Give 'high' class MUCH higher weight:")
print("""
# In main(), when computing class weights:
class_counts = np.bincount(y_train)
class_weights = torch.FloatTensor(len(class_counts) / (len(class_counts) * class_counts))

# BOOST the high class weight specifically
high_class_idx = label_encoder.transform(['high'])[0]
class_weights[high_class_idx] *= 3.0  # Triple the weight for 'high'!

# Normalize
class_weights = class_weights / class_weights.sum() * len(class_weights)
""")

print("\nSolution 4: STRATIFIED SPLIT WITH EMPHASIS ON 'HIGH'")
print("-"*50)
print("Make sure fires with 'high' samples are in TRAINING set:")
print("""
# Identify fires with significant 'high' samples
fires_with_high = df[df['SBS'] == 'high'].groupby('Fire_year').size()
important_fires = fires_with_high[fires_with_high >= 5].index.tolist()

# Ensure these are in training set
train_fires = list(important_fires)  # Start with these
remaining_fires = [f for f in fires if f not in train_fires]
np.random.shuffle(remaining_fires)

# Add more to reach 70% split
n_needed = int(0.7 * len(fires)) - len(train_fires)
train_fires.extend(remaining_fires[:n_needed])
""")

print("\nSolution 5: TWO-STAGE CLASSIFICATION")
print("-"*50)
print("Train two models:")
print("  Stage 1: Burned vs Unburned")
print("  Stage 2: Low vs Moderate vs High (only for burned pixels)")
print("This isolates the 'high' prediction problem.")

print("\n" + "="*80)
print("PRIORITY: Try Solution 1 (aggressive upsampling) + Solution 3 (weights)")
print("="*80)