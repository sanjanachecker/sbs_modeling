"""
Feature Importance Analysis for Wildfire Burn Severity Prediction
==================================================================
Analyzes feature importance using multiple methods:
1. Random Forest feature importance
2. XGBoost feature importance (gain, weight, cover)
3. Permutation importance
4. SHAP values (if time permits)

Handles missing data by analyzing only complete cases for importance ranking.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, f1_score, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================

print("="*80)
print("FEATURE IMPORTANCE ANALYSIS FOR WILDFIRE BURN SEVERITY")
print("="*80)

# Load data
df = pd.read_csv('data/all_fires_complete_covariates_fixed_129.csv')
print(f"\nDataset shape: {df.shape}")
print(f"Total samples: {len(df)}")

# Identify metadata columns (not features)
metadata_cols = ['system:index', 'Fire_year', 'PointX', 'PointY', 'SBS', 'Source', 
                 'latitude', 'longitude', '.geo']
metadata_cols = [c for c in metadata_cols if c in df.columns]

# Get all feature columns
feature_cols = [c for c in df.columns if c not in metadata_cols]
print(f"Total feature columns: {len(feature_cols)}")

# ============================================================================
# 2. MISSING DATA ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("MISSING DATA ANALYSIS")
print("="*80)

# Check missing values per column
missing_counts = df[feature_cols].isnull().sum()
missing_pct = (missing_counts / len(df) * 100).round(2)

missing_df = pd.DataFrame({
    'column': missing_counts.index,
    'missing_count': missing_counts.values,
    'missing_pct': missing_pct.values
}).sort_values('missing_pct', ascending=False)

print("\nFeatures with missing data:")
features_with_missing = missing_df[missing_df['missing_count'] > 0]
if len(features_with_missing) > 0:
    print(features_with_missing.to_string())
else:
    print("No missing data in feature columns!")

# Identify features with NO missing data
complete_features = missing_df[missing_df['missing_count'] == 0]['column'].tolist()
print(f"\nFeatures with complete data: {len(complete_features)}")

# ============================================================================
# 3. CLASS DISTRIBUTION
# ============================================================================

print("\n" + "="*80)
print("CLASS DISTRIBUTION")
print("="*80)

# Handle class label inconsistencies
df['SBS'] = df['SBS'].replace({'mod': 'moderate'})
class_counts = df['SBS'].value_counts()
print("\nBurn Severity Class Distribution:")
for cls, count in class_counts.items():
    print(f"  {cls:12s}: {count:4d} ({count/len(df)*100:5.1f}%)")

# Distribution by fire
print("\nDistribution by Fire:")
fire_class_dist = df.groupby('Fire_year')['SBS'].value_counts().unstack(fill_value=0)
print(fire_class_dist)

# ============================================================================
# 4. PREPARE DATA FOR FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*80)
print("PREPARING DATA FOR FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# For feature importance, we'll use rows with complete data
# First, let's see how many complete rows we have
complete_rows_mask = df[feature_cols].notna().all(axis=1)
df_complete = df[complete_rows_mask].copy()
print(f"\nRows with complete feature data: {len(df_complete)} ({len(df_complete)/len(df)*100:.1f}%)")

# If we have very few complete rows, use features with complete data only
if len(df_complete) < 500:
    print("Too few complete rows. Using features with no missing values...")
    X = df[complete_features].values
    feature_names = complete_features
    y_raw = df['SBS'].values
else:
    X = df_complete[feature_cols].values
    feature_names = feature_cols
    y_raw = df_complete['SBS'].values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"\nClasses: {le.classes_}")
print(f"Feature matrix shape: {X.shape}")

# Handle any remaining NaN by checking
nan_cols = np.where(np.isnan(X).any(axis=0))[0]
if len(nan_cols) > 0:
    print(f"\nWarning: {len(nan_cols)} columns still have NaN values")
    # Remove these columns
    valid_cols = ~np.isnan(X).any(axis=0)
    X = X[:, valid_cols]
    feature_names = [f for i, f in enumerate(feature_names) if valid_cols[i]]
    print(f"After removing: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================================
# 5. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*80)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("="*80)

# Train Random Forest with class weights
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")
rf.fit(X_train, y_train)

# Evaluate
y_pred_rf = rf.predict(X_test)
print(f"\nRandom Forest Test Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Macro F1 Score: {f1_score(y_test, y_pred_rf, average='macro'):.4f}")

# Feature importance (MDI - Mean Decrease in Impurity)
rf_importance = pd.DataFrame({
    'feature': feature_names,
    'rf_importance': rf.feature_importances_
}).sort_values('rf_importance', ascending=False)

print("\nTop 30 Features by Random Forest Importance (MDI):")
print(rf_importance.head(30).to_string())

# ============================================================================
# 6. XGBOOST FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*80)
print("XGBOOST FEATURE IMPORTANCE")
print("="*80)

# Compute class weights
class_counts_arr = np.bincount(y_train)
class_weights = len(y_train) / (len(le.classes_) * class_counts_arr)
sample_weights = class_weights[y_train]

# Train XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

print("Training XGBoost...")
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluate
y_pred_xgb = xgb_model.predict(X_test)
print(f"\nXGBoost Test Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Macro F1 Score: {f1_score(y_test, y_pred_xgb, average='macro'):.4f}")

# Feature importance (gain)
xgb_importance = pd.DataFrame({
    'feature': feature_names,
    'xgb_importance': xgb_model.feature_importances_
}).sort_values('xgb_importance', ascending=False)

print("\nTop 30 Features by XGBoost Importance (Gain):")
print(xgb_importance.head(30).to_string())

# ============================================================================
# 7. PERMUTATION IMPORTANCE
# ============================================================================

print("\n" + "="*80)
print("PERMUTATION IMPORTANCE (using XGBoost)")
print("="*80)

print("Computing permutation importance (this may take a moment)...")
perm_importance = permutation_importance(
    xgb_model, X_test, y_test, 
    n_repeats=10, 
    random_state=42, 
    n_jobs=-1,
    scoring='f1_macro'
)

perm_importance_df = pd.DataFrame({
    'feature': feature_names,
    'perm_importance_mean': perm_importance.importances_mean,
    'perm_importance_std': perm_importance.importances_std
}).sort_values('perm_importance_mean', ascending=False)

print("\nTop 30 Features by Permutation Importance:")
print(perm_importance_df.head(30).to_string())

# ============================================================================
# 8. AGGREGATE RANKINGS
# ============================================================================

print("\n" + "="*80)
print("AGGREGATE FEATURE RANKINGS")
print("="*80)

# Merge all importance measures
importance_df = rf_importance.merge(xgb_importance, on='feature')
importance_df = importance_df.merge(perm_importance_df[['feature', 'perm_importance_mean']], on='feature')

# Create rankings (1 = best)
importance_df['rf_rank'] = importance_df['rf_importance'].rank(ascending=False)
importance_df['xgb_rank'] = importance_df['xgb_importance'].rank(ascending=False)
importance_df['perm_rank'] = importance_df['perm_importance_mean'].rank(ascending=False)

# Average rank
importance_df['avg_rank'] = (importance_df['rf_rank'] + importance_df['xgb_rank'] + importance_df['perm_rank']) / 3
importance_df = importance_df.sort_values('avg_rank')

# Normalize importance scores to [0, 1] for comparison
for col in ['rf_importance', 'xgb_importance', 'perm_importance_mean']:
    max_val = importance_df[col].max()
    if max_val > 0:
        importance_df[col + '_norm'] = importance_df[col] / max_val

# Combined score (average of normalized importance)
importance_df['combined_score'] = (
    importance_df['rf_importance_norm'] + 
    importance_df['xgb_importance_norm'] + 
    importance_df['perm_importance_mean_norm']
) / 3

importance_df = importance_df.sort_values('combined_score', ascending=False)

print("\nTop 50 Features by Combined Score:")
display_cols = ['feature', 'rf_importance', 'xgb_importance', 'perm_importance_mean', 
                'avg_rank', 'combined_score']
print(importance_df[display_cols].head(50).to_string())

# ============================================================================
# 9. FEATURE CATEGORIZATION
# ============================================================================

print("\n" + "="*80)
print("FEATURE IMPORTANCE BY CATEGORY")
print("="*80)

# Categorize features
def categorize_feature(name):
    name_lower = name.lower()
    
    # Spectral bands
    if name in ['blueBand', 'greenBand', 'redBand', 'nirBand', 'swir1Band', 'swir2Band']:
        return 'Spectral Bands'
    
    # Spectral indices
    if name in ['ndvi', 'ndbi', 'mndwi', 'nbr', 'bsi']:
        return 'Spectral Indices'
    
    # Differenced indices (pre/post fire)
    if name.startswith('d') and name[1:] in ['ndvi', 'ndbi', 'mndwi', 'nbr', 'bsi']:
        return 'Differenced Indices'
    
    # Climate (WorldClim)
    if name.startswith('wc_'):
        return 'Climate (WorldClim)'
    
    # Solar radiation
    if 'pisr' in name_lower:
        return 'Solar Radiation'
    
    # Slope/Aspect
    if 'aspct' in name_lower or 'slope' in name_lower or name == 'slopeRadians':
        return 'Slope/Aspect'
    
    # Elevation-related
    if 'elev' in name_lower:
        return 'Elevation'
    
    # Curvature
    if any(x in name_lower for x in ['crosc', 'longc', 'planc', 'profc', 'maxc', 'minc', 'tsc']):
        return 'Curvature'
    
    # Terrain indices
    if any(x in name_lower for x in ['tpi', 'tri', 'twi', 'spi', 'vrm', 'rdgh']):
        return 'Terrain Indices'
    
    # Geomorphology
    if any(x in name_lower for x in ['gmrph', 'morpfeat', 'genelev', 'dah', 'msp', 'sth']):
        return 'Geomorphology'
    
    # Hydrological
    if any(x in name_lower for x in ['swi', 'vd', 'vdcn', 'hn', 'hac', 'hbc', 'hs_', 'mbi', 'rlp', 'sl_', 'po_', 'no_']):
        return 'Hydrology'
    
    return 'Other Terrain'

importance_df['category'] = importance_df['feature'].apply(categorize_feature)

# Aggregate by category
category_importance = importance_df.groupby('category').agg({
    'combined_score': ['mean', 'max', 'count'],
    'avg_rank': 'mean'
}).round(4)
category_importance.columns = ['mean_score', 'max_score', 'feature_count', 'mean_rank']
category_importance = category_importance.sort_values('mean_score', ascending=False)

print("\nFeature Category Importance Summary:")
print(category_importance.to_string())

# Top features per category
print("\n\nTop 5 Features per Category:")
for cat in category_importance.index:
    cat_features = importance_df[importance_df['category'] == cat].head(5)
    print(f"\n{cat}:")
    for _, row in cat_features.iterrows():
        print(f"  {row['feature']:30s} | Score: {row['combined_score']:.4f} | Rank: {row['avg_rank']:.1f}")

# ============================================================================
# 10. SAVE RESULTS
# ============================================================================

# Save full importance results
output_path = 'results/feature_importance_results.csv'
importance_df.to_csv(output_path, index=False)
print(f"\n\nFull results saved to: {output_path}")

# Save top features list
top_features = importance_df.head(50)['feature'].tolist()
with open('results/top_50_features.txt', 'w') as f:
    for feat in top_features:
        f.write(f"{feat}\n")
print("Top 50 features saved to: results/top_50_features.txt")

# ============================================================================
# 11. VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Plot 1: Top 30 Features Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(18, 10))

top30 = importance_df.head(30)

# RF Importance
ax1 = axes[0]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top30)))
ax1.barh(range(len(top30)), top30['rf_importance'].values, color=colors)
ax1.set_yticks(range(len(top30)))
ax1.set_yticklabels(top30['feature'].values)
ax1.invert_yaxis()
ax1.set_xlabel('Importance')
ax1.set_title('Random Forest Importance', fontweight='bold')

# XGB Importance
ax2 = axes[1]
ax2.barh(range(len(top30)), top30['xgb_importance'].values, color=colors)
ax2.set_yticks(range(len(top30)))
ax2.set_yticklabels(top30['feature'].values)
ax2.invert_yaxis()
ax2.set_xlabel('Importance')
ax2.set_title('XGBoost Importance', fontweight='bold')

# Permutation Importance
ax3 = axes[2]
ax3.barh(range(len(top30)), top30['perm_importance_mean'].values, color=colors)
ax3.set_yticks(range(len(top30)))
ax3.set_yticklabels(top30['feature'].values)
ax3.invert_yaxis()
ax3.set_xlabel('Importance')
ax3.set_title('Permutation Importance', fontweight='bold')

plt.tight_layout()
plt.savefig('results/feature_importance_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: feature_importance_comparison.png")

# Plot 2: Category Importance
fig, ax = plt.subplots(figsize=(12, 6))
cat_order = category_importance.index.tolist()
colors = plt.cm.Set3(np.linspace(0, 1, len(cat_order)))

bars = ax.barh(cat_order, category_importance['mean_score'].values, color=colors)
ax.set_xlabel('Mean Combined Score', fontsize=12)
ax.set_title('Feature Importance by Category', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# Add count labels
for i, (bar, count) in enumerate(zip(bars, category_importance['feature_count'].values)):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
            f'n={int(count)}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('results/category_importance.png', dpi=150, bbox_inches='tight')
print("Saved: category_importance.png")

# Plot 3: Heatmap of top features importance
fig, ax = plt.subplots(figsize=(10, 14))
heatmap_data = importance_df.head(40)[['feature', 'rf_importance_norm', 
                                        'xgb_importance_norm', 
                                        'perm_importance_mean_norm']].set_index('feature')
heatmap_data.columns = ['RF', 'XGB', 'Perm']

sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
            cbar_kws={'label': 'Normalized Importance'})
ax.set_title('Feature Importance Comparison (Top 40)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/feature_importance_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: feature_importance_heatmap.png")

# ============================================================================
# 12. RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("RECOMMENDATIONS FOR FEATURE SELECTION")
print("="*80)

# Features with high importance (combined score > 0.3)
high_importance = importance_df[importance_df['combined_score'] > 0.3]
medium_importance = importance_df[(importance_df['combined_score'] > 0.1) & 
                                   (importance_df['combined_score'] <= 0.3)]

print(f"\nHigh importance features (score > 0.3): {len(high_importance)}")
print(f"Medium importance features (0.1 < score <= 0.3): {len(medium_importance)}")

# Check missing data in top features
print("\n\nMissing Data in Top 30 Features:")
top30_features = importance_df.head(30)['feature'].tolist()
for feat in top30_features:
    if feat in df.columns:
        missing = df[feat].isnull().sum()
        if missing > 0:
            print(f"  {feat:30s}: {missing:4d} missing ({missing/len(df)*100:.1f}%)")

# Suggested feature sets
print("\n\n" + "="*80)
print("SUGGESTED FEATURE SETS FOR MODELING")
print("="*80)

print("\n1. MINIMAL SET (Top 15 features):")
for i, feat in enumerate(importance_df.head(15)['feature'].tolist(), 1):
    print(f"   {i:2d}. {feat}")

print("\n2. RECOMMENDED SET (Top 30 features):")
for i, feat in enumerate(importance_df.head(30)['feature'].tolist(), 1):
    print(f"   {i:2d}. {feat}")

print("\n3. EXTENDED SET (Top 50 features):")
for i, feat in enumerate(importance_df.head(50)['feature'].tolist(), 1):
    print(f"   {i:2d}. {feat}")

# Features that need imputation
needs_imputation = []
for feat in importance_df.head(50)['feature'].tolist():
    if feat in df.columns:
        missing = df[feat].isnull().sum()
        if missing > 0:
            needs_imputation.append((feat, missing))

if needs_imputation:
    print("\n\nFEATURES IN TOP 50 THAT NEED IMPUTATION:")
    for feat, missing in needs_imputation:
        print(f"  {feat:30s}: {missing:4d} missing values")
else:
    print("\n\nNo missing values in top 50 features!")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)