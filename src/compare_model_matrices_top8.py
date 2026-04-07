"""
Burn Severity — Detailed Evaluation of Top Configs
====================================================
Takes the top 8 configs from 10-fold CV search.
For each: trains on 80% data, evaluates on 20% with:
  - Confusion matrix (counts + percentages)
  - Per-class precision, recall, F1
  - Moderate row breakdown (moderate→low vs moderate→high)
  - Overall Kappa and accuracy

Generates side-by-side comparison plots.
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    cohen_kappa_score, classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

MAIN_CSV = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_complete_covariates_fixed_1229.csv'
OLD_UPSAMPLED_CSV = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_upsampled_points_with_covariates_fixed.csv'
OUTPUT_DIR = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/results_top8_eval'

RANDOM_STATE = 42
TEST_SIZE = 0.2

CLASS_NAMES = ['unburned', 'low', 'moderate', 'high']
LABEL_MAP = {'unburned': 0, 'low': 1, 'moderate': 2, 'high': 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

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
    df_main = pd.read_csv(MAIN_CSV)
    df_main['SBS'] = df_main['SBS'].replace({'mod': 'moderate'})
    df_main = df_main[df_main['SBS'].isin(CLASS_NAMES)].copy()

    df_up = pd.read_csv(OLD_UPSAMPLED_CSV)
    common_cols = list(set(df_main.columns) & set(df_up.columns))
    df = pd.concat([df_main[common_cols], df_up[common_cols]], ignore_index=True)
    df['SBS'] = df['SBS'].replace({'mod': 'moderate'})
    df = df[df['SBS'].isin(CLASS_NAMES)].copy()

    print(f"Data: {len(df)} rows")
    return df


def get_top_n_features(df, n):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    all_feats = sorted([c for c in numeric if c.lower() not in {x.lower() for x in EXCLUDE_COLS}])
    available = [f for f in all_feats if f in df.columns]

    X = df[available].fillna(df[available].median()).values.astype(np.float32)
    X = np.where(np.isnan(X) | np.isinf(X), 0.0, X)
    y = df['SBS'].values

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, class_weight='balanced',
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    indices = np.argsort(rf.feature_importances_)[::-1][:n]
    return [available[i] for i in indices]


def prepare_xy(df, features):
    available = [f for f in features if f in df.columns]
    X = df[available].fillna(df[available].median()).values.astype(np.float32)
    X = np.where(np.isnan(X) | np.isinf(X), 0.0, X)
    y = df['SBS'].values
    return X, y, available


# ============================================================================
# TOP 8 CONFIGS
# ============================================================================

def get_top8_configs():
    """Top 8 from 10-fold CV search results."""
    return [
        {
            'name': 'ExtraTrees_tuned | Top50',
            'cv_kappa': 0.6182,
            'features': 'Top50',
            'model_fn': lambda: ExtraTreesClassifier(
                n_estimators=1000, max_depth=25, min_samples_split=5,
                min_samples_leaf=2, class_weight='balanced_subsample',
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            'is_xgb': False,
        },
        {
            'name': 'XGB_deep | Top30',
            'cv_kappa': 0.6176,
            'features': 'Top30',
            'model_fn': lambda: xgb.XGBClassifier(
                n_estimators=800, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                objective='multi:softprob', num_class=4,
                random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss'
            ),
            'is_xgb': True,
        },
        {
            'name': 'XGB_default | Top30',
            'cv_kappa': 0.6170,
            'features': 'Top30',
            'model_fn': lambda: xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                objective='multi:softprob', num_class=4,
                random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss'
            ),
            'is_xgb': True,
        },
        {
            'name': 'ExtraTrees | Top30',
            'cv_kappa': 0.6142,
            'features': 'Top30',
            'model_fn': lambda: ExtraTreesClassifier(
                n_estimators=500, max_depth=None, min_samples_split=2,
                min_samples_leaf=1, class_weight='balanced',
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            'is_xgb': False,
        },
        {
            'name': 'GBM_deep | Top50',
            'cv_kappa': 0.6138,
            'features': 'Top50',
            'model_fn': lambda: GradientBoostingClassifier(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, min_samples_split=10,
                random_state=RANDOM_STATE
            ),
            'is_xgb': False,
        },
        {
            'name': 'ExtraTrees_tuned | Top75',
            'cv_kappa': 0.6124,
            'features': 'Top75',
            'model_fn': lambda: ExtraTreesClassifier(
                n_estimators=1000, max_depth=25, min_samples_split=5,
                min_samples_leaf=2, class_weight='balanced_subsample',
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            'is_xgb': False,
        },
        {
            'name': 'ExtraTrees_tuned | Top30',
            'cv_kappa': 0.6118,
            'features': 'Top30',
            'model_fn': lambda: ExtraTreesClassifier(
                n_estimators=1000, max_depth=25, min_samples_split=5,
                min_samples_leaf=2, class_weight='balanced_subsample',
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            'is_xgb': False,
        },
        {
            'name': 'ExtraTrees | Top50',
            'cv_kappa': 0.6118,
            'features': 'Top50',
            'model_fn': lambda: ExtraTreesClassifier(
                n_estimators=500, max_depth=None, min_samples_split=2,
                min_samples_leaf=1, class_weight='balanced',
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            'is_xgb': False,
        },
    ]


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_config(config, X_train, X_test, y_train, y_test):
    """Train and evaluate a single config."""
    model = config['model_fn']()

    if config['is_xgb']:
        y_train_num = np.array([LABEL_MAP[s] for s in y_train])
        y_test_num = np.array([LABEL_MAP[s] for s in y_test])
        sw = compute_sample_weight('balanced', y_train_num)
        model.fit(X_train, y_train_num, sample_weight=sw)
        y_pred_num = model.predict(X_test)
        y_pred = np.array([INV_LABEL_MAP[p] for p in y_pred_num])
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    kappa = cohen_kappa_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=CLASS_NAMES, zero_division=0
    )

    return {
        'name': config['name'],
        'cv_kappa': config['cv_kappa'],
        'test_kappa': kappa,
        'test_accuracy': accuracy,
        'cm': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'y_test': y_test,
        'y_pred': y_pred,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrices(results, save_path):
    """Plot all 8 confusion matrices side by side."""
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    axes = axes.flatten()

    for i, r in enumerate(results):
        ax = axes[i]
        cm = r['cm']
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        annot = np.empty_like(cm, dtype=object)
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                annot[row, col] = f'{cm[row, col]}\n({cm_pct[row, col]:.1f}%)'

        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                    cbar=False)
        ax.set_title(f'{r["name"]}\nKappa: {r["test_kappa"]:.4f} (CV: {r["cv_kappa"]:.4f})',
                     fontweight='bold', fontsize=10)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('True', fontsize=9)

    # Hide unused axes
    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_class_metrics(results, save_path):
    """Plot per-class precision, recall, F1 for all configs."""
    n_configs = len(results)
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    metrics = [('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1-Score')]

    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / n_configs

    colors = plt.cm.tab10(np.linspace(0, 1, n_configs))

    for ax, (metric_key, metric_name) in zip(axes, metrics):
        for i, r in enumerate(results):
            offset = width * (i - n_configs / 2 + 0.5)
            values = r[metric_key]
            bars = ax.bar(x + offset, values, width, label=r['name'] if ax == axes[0] else '',
                         color=colors[i], edgecolor='black', linewidth=0.3)

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.2)

    axes[0].legend(bbox_to_anchor=(0, -0.25), loc='upper left', ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_moderate_breakdown(results, save_path):
    """Compare how each model handles the moderate class."""
    fig, ax = plt.subplots(figsize=(14, 7))

    names = [r['name'] for r in results]
    mod_idx = CLASS_NAMES.index('moderate')

    mod_correct = []
    mod_to_low = []
    mod_to_high = []
    mod_to_unb = []

    for r in results:
        cm = r['cm']
        mod_row = cm[mod_idx]
        total = mod_row.sum()
        mod_correct.append(mod_row[mod_idx] / total * 100)
        mod_to_low.append(mod_row[CLASS_NAMES.index('low')] / total * 100)
        mod_to_high.append(mod_row[CLASS_NAMES.index('high')] / total * 100)
        mod_to_unb.append(mod_row[CLASS_NAMES.index('unburned')] / total * 100)

    x = np.arange(len(names))
    width = 0.2

    ax.bar(x - 1.5*width, mod_correct, width, label='Correct (moderate)', color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.bar(x - 0.5*width, mod_to_low, width, label='→ low (bad)', color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax.bar(x + 0.5*width, mod_to_high, width, label='→ high (acceptable)', color='#f39c12', edgecolor='black', linewidth=0.5)
    ax.bar(x + 1.5*width, mod_to_unb, width, label='→ unburned (bad)', color='#9b59b6', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Model Configuration', fontsize=12)
    ax.set_ylabel('Percentage of Moderate Samples', fontsize=12)
    ax.set_title('Moderate Class Breakdown — Where Do Moderate Pixels Go?',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_kappa_comparison(results, save_path):
    """Bar chart of test Kappa vs CV Kappa for each config."""
    fig, ax = plt.subplots(figsize=(14, 6))

    names = [r['name'] for r in results]
    cv_kappas = [r['cv_kappa'] for r in results]
    test_kappas = [r['test_kappa'] for r in results]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, cv_kappas, width, label='10-fold CV Kappa', color='#3498db', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, test_kappas, width, label='Test Set Kappa', color='#2ecc71', edgecolor='black', linewidth=0.5)

    for i in range(len(names)):
        ax.text(x[i] - width/2, cv_kappas[i] + 0.005, f'{cv_kappas[i]:.3f}', ha='center', fontsize=7, fontweight='bold')
        ax.text(x[i] + width/2, test_kappas[i] + 0.005, f'{test_kappas[i]:.3f}', ha='center', fontsize=7, fontweight='bold')

    ax.axhline(y=0.65, color='green', linestyle='--', linewidth=2, label='Target: 0.65')
    ax.set_xlabel('Model Configuration', fontsize=12)
    ax.set_ylabel("Cohen's Kappa", fontsize=12)
    ax.set_title("CV Kappa vs Test Set Kappa", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim(0, max(max(cv_kappas), max(test_kappas)) * 1.15)
    ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("DETAILED EVALUATION — TOP 8 CONFIGS")
    print("=" * 70)

    df = load_data()

    # Get feature sets
    print("\nDetermining feature sets...")
    top50 = get_top_n_features(df, 50)
    top75 = get_top_n_features(df, 75)

    feature_sets = {
        'Top30': TOP_30_FEATURES,
        'Top50': top50,
        'Top75': top75,
    }

    configs = get_top8_configs()

    # Single train/test split (same for all configs for fair comparison)
    # Use Top50 for the split since it's the largest feature set used
    all_feats_available = [f for f in top75 if f in df.columns]
    X_full = df[all_feats_available].fillna(df[all_feats_available].median()).values.astype(np.float32)
    X_full = np.where(np.isnan(X_full) | np.isinf(X_full), 0.0, X_full)
    y_full = df['SBS'].values

    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_full
    )

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Evaluate each config
    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {config['name']}")

        # Get the right feature subset
        fs_name = config['features']
        fs = feature_sets[fs_name]
        available = [f for f in fs if f in all_feats_available]
        feat_indices = [all_feats_available.index(f) for f in available]

        X_train = X_train_full[:, feat_indices]
        X_test = X_test_full[:, feat_indices]

        r = evaluate_config(config, X_train, X_test, y_train, y_test)
        results.append(r)

        # Print summary
        print(f"  Kappa: {r['test_kappa']:.4f}  Acc: {r['test_accuracy']:.4f}")
        cm = r['cm']
        mod_idx = CLASS_NAMES.index('moderate')
        mod_row = cm[mod_idx]
        mod_total = mod_row.sum()
        print(f"  Moderate: {mod_row[mod_idx]/mod_total*100:.1f}% correct, "
              f"{mod_row[CLASS_NAMES.index('low')]/mod_total*100:.1f}% → low, "
              f"{mod_row[CLASS_NAMES.index('high')]/mod_total*100:.1f}% → high")
        print(f"  Per-class recall: " + 
              ' | '.join([f"{cls}: {r['recall'][j]:.2f}" for j, cls in enumerate(CLASS_NAMES)]))

    # Sort by test kappa
    results.sort(key=lambda r: r['test_kappa'], reverse=True)

    # Print comparison table
    print("\n" + "=" * 120)
    print(f"{'Rank':>4} {'Config':<30} {'CV κ':>7} {'Test κ':>7} {'Acc':>7} "
          f"{'Unb_R':>7} {'Low_R':>7} {'Mod_R':>7} {'High_R':>7} "
          f"{'Mod→L%':>7} {'Mod→H%':>7}")
    print("=" * 120)
    for i, r in enumerate(results):
        cm = r['cm']
        mod_idx = CLASS_NAMES.index('moderate')
        mod_row = cm[mod_idx]
        mod_total = mod_row.sum()
        mod_to_low = mod_row[CLASS_NAMES.index('low')] / mod_total * 100
        mod_to_high = mod_row[CLASS_NAMES.index('high')] / mod_total * 100

        print(f"{i+1:>4} {r['name']:<30} {r['cv_kappa']:>7.4f} {r['test_kappa']:>7.4f} "
              f"{r['test_accuracy']:>7.4f} "
              f"{r['recall'][0]:>7.2f} {r['recall'][1]:>7.2f} "
              f"{r['recall'][2]:>7.2f} {r['recall'][3]:>7.2f} "
              f"{mod_to_low:>6.1f}% {mod_to_high:>6.1f}%")
    print("=" * 120)

    # Plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_confusion_matrices(results, f'{OUTPUT_DIR}/top8_confusion_matrices.png')
    plot_per_class_metrics(results, f'{OUTPUT_DIR}/top8_per_class_metrics.png')
    plot_moderate_breakdown(results, f'{OUTPUT_DIR}/top8_moderate_breakdown.png')
    plot_kappa_comparison(results, f'{OUTPUT_DIR}/top8_kappa_comparison.png')

    # Save results
    summary_rows = []
    for r in results:
        cm = r['cm']
        mod_idx = CLASS_NAMES.index('moderate')
        mod_row = cm[mod_idx]
        mod_total = mod_row.sum()
        summary_rows.append({
            'config': r['name'],
            'cv_kappa': r['cv_kappa'],
            'test_kappa': r['test_kappa'],
            'test_accuracy': r['test_accuracy'],
            **{f'{cls}_precision': r['precision'][j] for j, cls in enumerate(CLASS_NAMES)},
            **{f'{cls}_recall': r['recall'][j] for j, cls in enumerate(CLASS_NAMES)},
            **{f'{cls}_f1': r['f1'][j] for j, cls in enumerate(CLASS_NAMES)},
            'moderate_to_low_pct': mod_row[CLASS_NAMES.index('low')] / mod_total * 100,
            'moderate_to_high_pct': mod_row[CLASS_NAMES.index('high')] / mod_total * 100,
            'moderate_correct_pct': mod_row[mod_idx] / mod_total * 100,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{OUTPUT_DIR}/top8_detailed_comparison.csv', index=False)
    print(f"\nDetailed results saved: {OUTPUT_DIR}/top8_detailed_comparison.csv")

    print(f"\n{'★'*60}")
    best = results[0]
    print(f"  BEST ON TEST SET: {best['name']}")
    print(f"  Test Kappa: {best['test_kappa']:.4f}")
    print(f"  CV Kappa: {best['cv_kappa']:.4f}")
    print(f"{'★'*60}")

    return results


if __name__ == '__main__':
    results = main()