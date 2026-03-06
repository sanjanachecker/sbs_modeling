"""
Burn Severity — 10-Fold Stratified CV Model Search
====================================================
Tests combinations of:
  Models: RF, XGBoost, ExtraTrees, GradientBoosting
  Features: Top 30, Top 50, Top 75, All
  Data: No upsampling, Old upsampled (moderate-matched)

Outputs a ranked table of all combos by mean Kappa.
Use the best combo in a separate LOFO + training script.
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION — UPDATE PATHS
# ============================================================================

MAIN_CSV = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_complete_covariates_fixed_1229.csv'
OLD_UPSAMPLED_CSV = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_upsampled_points_with_covariates_fixed.csv'
OUTPUT_DIR = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/results_model_search'

RANDOM_STATE = 42
N_FOLDS = 10

CLASS_NAMES = ['unburned', 'low', 'moderate', 'high']
LABEL_MAP = {'unburned': 0, 'low': 1, 'moderate': 2, 'high': 3}

TOP_30_FEATURES = [
    'dnbr', 'dndvi', 'dndbi', 'dbsi', 'nbr', 'bsi', 'ndvi', 'ndbi',
    'meanelev_32', 'wc_bio19', 'nirBand', 'wc_bio05', 'rdgh_6', 'blueBand',
    'minelev_4', 'greenBand', 'wc_bio06', 'swir2Band', 'pisrdif_2021-11-22',
    'pisrdif_2021-12-22', 'stddevelev_32', 'maxc_2', 'wc_bio12', 'wc_bio07',
    'dmndwi', 'wc_bio18', 'wc_bio17', 'wc_bio02', 'vd_5', 'planc_32'
]

EXCLUDE_COLS = {
    'SBS', 'Fire_year', 'fire', 'source', 'data_source', 'Source',
    'PointX', 'PointY', '.geo', 'system:index', 'label',
    'latitude', 'longitude', 'lat', 'lon', 'x', 'y'
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load main + upsampled CSVs, return combined dataset."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df_main = pd.read_csv(MAIN_CSV)
    df_main['SBS'] = df_main['SBS'].replace({'mod': 'moderate'})
    df_main = df_main[df_main['SBS'].isin(CLASS_NAMES)].copy()
    print(f"Main CSV: {len(df_main)} rows")

    df_up = pd.read_csv(OLD_UPSAMPLED_CSV)
    common_cols = list(set(df_main.columns) & set(df_up.columns))
    df = pd.concat([df_main[common_cols], df_up[common_cols]], ignore_index=True)
    df['SBS'] = df['SBS'].replace({'mod': 'moderate'})
    df = df[df['SBS'].isin(CLASS_NAMES)].copy()
    print(f"With upsampling: {len(df)} rows")

    for cls in CLASS_NAMES:
        count = (df['SBS'] == cls).sum()
        print(f"  {cls:12s}: {count:5d} ({count/len(df)*100:.1f}%)")

    return {'OldUp': df}


def get_top_n_features(df, n):
    """Rank features by RF importance, return top N."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    all_feats = sorted([c for c in numeric if c.lower() not in {x.lower() for x in EXCLUDE_COLS}])
    available = [f for f in all_feats if f in df.columns]

    X = df[available].fillna(df[available].median()).values.astype(np.float32)
    X = np.where(np.isnan(X) | np.isinf(X), 0.0, X)
    y = df['SBS'].values

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X, y)

    indices = np.argsort(rf.feature_importances_)[::-1][:n]
    return [available[i] for i in indices]


def get_all_features(df):
    """Get all numeric feature columns."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return sorted([c for c in numeric if c.lower() not in {x.lower() for x in EXCLUDE_COLS}])


def prepare_xy(df, features):
    """Extract X, y arrays from dataframe."""
    available = [f for f in features if f in df.columns]
    X = df[available].fillna(df[available].median()).values.astype(np.float32)
    X = np.where(np.isnan(X) | np.isinf(X), 0.0, X)
    y = df['SBS'].values
    return X, y, available


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def get_models():
    """Return dict of model name → model instance."""
    return {
        'RF_balanced': RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'RF_bal_sub': RandomForestClassifier(
            n_estimators=500, max_depth=20, min_samples_split=10,
            min_samples_leaf=2, class_weight='balanced_subsample',
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'RF_1000trees': RandomForestClassifier(
            n_estimators=1000, max_depth=25, min_samples_split=2,
            min_samples_leaf=2, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'ExtraTrees_tuned': ExtraTreesClassifier(
            n_estimators=1000, max_depth=25, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced_subsample',
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'GBM_default': GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, min_samples_split=5,
            random_state=RANDOM_STATE
        ),
        'GBM_deep': GradientBoostingClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, min_samples_split=10,
            random_state=RANDOM_STATE
        ),
        'XGB_default': xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            objective='multi:softprob', num_class=4,
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric='mlogloss'
        ),
        'XGB_deep': xgb.XGBClassifier(
            n_estimators=800, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
            objective='multi:softprob', num_class=4,
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric='mlogloss'
        ),
        'XGB_shallow': xgb.XGBClassifier(
            n_estimators=1000, max_depth=4, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            objective='multi:softprob', num_class=4,
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric='mlogloss'
        ),
        'XGB_lr01': xgb.XGBClassifier(
            n_estimators=1000, max_depth=6, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            objective='multi:softprob', num_class=4,
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric='mlogloss'
        ),
    }


# ============================================================================
# 10-FOLD CV RUNNER
# ============================================================================

def run_10fold_cv(X, y, model, model_name, n_folds=N_FOLDS):
    """Run stratified K-fold CV, return mean and std Kappa."""
    kappa_scorer = make_scorer(cohen_kappa_score)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    # For XGBoost, we need sample_weight via fit_params
    is_xgb = 'XGB' in model_name

    if is_xgb:
        # Manual CV loop for XGBoost with sample weights
        fold_kappas = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            y_train_num = np.array([LABEL_MAP[s] for s in y_train])
            y_val_num = np.array([LABEL_MAP[s] for s in y_val])

            sw = compute_sample_weight('balanced', y_train_num)
            model_clone = xgb.XGBClassifier(**model.get_params())
            model_clone.fit(X_train, y_train_num, sample_weight=sw)

            y_pred_num = model_clone.predict(X_val)
            y_pred_str = np.array([INV_LABEL_MAP[p] for p in y_pred_num])

            k = cohen_kappa_score(y_val, y_pred_str)
            fold_kappas.append(k)

        return np.array(fold_kappas)
    else:
        scores = cross_val_score(model, X, y, cv=skf, scoring=kappa_scorer, n_jobs=-1)
        return scores


# ============================================================================
# MAIN
# ============================================================================

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def main():
    print("=" * 70)
    print("10-FOLD STRATIFIED CV — MODEL SEARCH")
    print("=" * 70)

    # Load data
    datasets = load_data()

    # Determine feature sets from largest dataset
    print("\n" + "=" * 70)
    print("DETERMINING FEATURE SETS")
    print("=" * 70)
    df_largest = datasets['OldUp']
    top50 = get_top_n_features(df_largest, 50)
    top75 = get_top_n_features(df_largest, 75)
    all_feats = get_all_features(df_largest)
    print(f"  Top 30: {len(TOP_30_FEATURES)}")
    print(f"  Top 50: {len(top50)}")
    print(f"  Top 75: {len(top75)}")
    print(f"  All:    {len(all_feats)}")

    feature_sets = {
        'Top30': TOP_30_FEATURES,
        'Top50': top50,
        'Top75': top75,
        'All': all_feats,
    }

    models = get_models()

    # Run all combos
    print("\n" + "=" * 70)
    print(f"RUNNING {len(models)} MODELS × {len(feature_sets)} FEATURE SETS × {len(datasets)} DATASETS")
    print(f"= {len(models) * len(feature_sets) * len(datasets)} experiments, {N_FOLDS}-fold CV each")
    print("=" * 70)

    results = []
    total = len(models) * len(feature_sets) * len(datasets)
    count = 0

    for ds_name, df in datasets.items():
        for fs_name, fs in feature_sets.items():
            X, y, available = prepare_xy(df, fs)

            for model_name, model in models.items():
                count += 1
                label = f"{model_name} | {ds_name} | {fs_name}"

                try:
                    fold_scores = run_10fold_cv(X, y, model, model_name)
                    mean_k = fold_scores.mean()
                    std_k = fold_scores.std()

                    results.append({
                        'label': label,
                        'model': model_name,
                        'dataset': ds_name,
                        'features': fs_name,
                        'n_features': len(available),
                        'n_samples': len(y),
                        'mean_kappa': mean_k,
                        'std_kappa': std_k,
                        'min_kappa': fold_scores.min(),
                        'max_kappa': fold_scores.max(),
                        'fold_scores': fold_scores.tolist(),
                    })

                    marker = " ★" if mean_k >= 0.65 else " ✓" if mean_k >= 0.60 else ""
                    print(f"  [{count:3d}/{total}] {label:55s}  Kappa: {mean_k:.4f} ± {std_k:.4f}{marker}")

                except Exception as e:
                    print(f"  [{count:3d}/{total}] {label:55s}  ERROR: {e}")
                    results.append({
                        'label': label,
                        'model': model_name,
                        'dataset': ds_name,
                        'features': fs_name,
                        'n_features': len(available),
                        'n_samples': len(y),
                        'mean_kappa': 0.0,
                        'std_kappa': 0.0,
                        'min_kappa': 0.0,
                        'max_kappa': 0.0,
                        'fold_scores': [],
                    })

    # Sort results
    results_df = pd.DataFrame(results).sort_values('mean_kappa', ascending=False)

    # Print ranked table
    print("\n" + "=" * 110)
    print(f"{'Rank':>4} {'Model':<20} {'Data':<8} {'Feats':<8} {'Mean κ':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'N':>6}")
    print("=" * 110)
    for i, (_, row) in enumerate(results_df.head(30).iterrows()):
        marker = " ★" if row['mean_kappa'] >= 0.65 else " ✓" if row['mean_kappa'] >= 0.60 else ""
        print(f"{i+1:>4} {row['model']:<20} {row['dataset']:<8} {row['features']:<8} "
              f"{row['mean_kappa']:>8.4f} {row['std_kappa']:>8.4f} {row['min_kappa']:>8.4f} "
              f"{row['max_kappa']:>8.4f} {row['n_samples']:>6}{marker}")
    print("=" * 110)

    # Save full results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(f'{OUTPUT_DIR}/10fold_cv_all_results.csv', index=False)
    print(f"\nFull results saved: {OUTPUT_DIR}/10fold_cv_all_results.csv")

    # Plot top 20
    top20 = results_df.head(20)
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['#e74c3c' if k < 0.55 else '#f39c12' if k < 0.60 else '#2ecc71' if k < 0.65 else '#27ae60'
              for k in top20['mean_kappa']]

    y_pos = range(len(top20))
    bars = ax.barh(y_pos, top20['mean_kappa'].values, xerr=top20['std_kappa'].values,
                   color=colors, edgecolor='black', linewidth=0.5, capsize=3)
    ax.axvline(x=0.65, color='green', linestyle='--', linewidth=2, label='Target: 0.65')
    ax.axvline(x=0.60, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='0.60')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20['label'].values, fontsize=7)
    ax.set_xlabel("Cohen's Kappa (10-fold CV)", fontsize=12)
    ax.set_title("Top 20 Model Configurations — 10-Fold Stratified CV",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/top20_10fold_cv.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/top20_10fold_cv.png")

    # Print best config details
    best = results_df.iloc[0]
    print(f"\n{'★'*60}")
    print(f"  BEST CONFIG:")
    print(f"  Model:    {best['model']}")
    print(f"  Dataset:  {best['dataset']}")
    print(f"  Features: {best['features']} ({best['n_features']})")
    print(f"  10-fold Kappa: {best['mean_kappa']:.4f} ± {best['std_kappa']:.4f}")
    print(f"{'★'*60}")
    print(f"\nUse this config in your LOFO CV + final training script.")

    return results_df


if __name__ == '__main__':
    results_df = main()