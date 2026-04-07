"""
XGBoost Burn Severity — LOFO CV + Final Training + TF Export
=============================================================
Best config from 10-fold stratified CV model search:
  Model:    XGB_lr01
  Features: Top30 (hardcoded, from tuning script)
  Params:   n_estimators=1000, max_depth=6, learning_rate=0.01
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3
            balanced sample weights

Evaluation: Leave-One-Fire-Out cross-validation
Final model: Trained on ALL fires, exported as TF SavedModel for GEE
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import joblib
from datetime import datetime
from sklearn.metrics import (
    cohen_kappa_score, classification_report, confusion_matrix,
    accuracy_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION — UPDATE PATHS IF NEEDED
# ============================================================================

MAIN_CSV = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_complete_covariates_fixed_1229.csv'
OLD_UPSAMPLED_CSV = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_upsampled_points_with_covariates_fixed.csv'

OUTPUT_DIR = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/results_xgb_lr01_top30'
GEE_EXPORT_DIR = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/gee_models_xgb_lr01_top30'

RANDOM_STATE = 42

CLASS_NAMES = ['unburned', 'low', 'moderate', 'high']
LABEL_MAP = {'unburned': 0, 'low': 1, 'moderate': 2, 'high': 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Best config from model search
XGB_PARAMS = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'objective': 'multi:softprob',
    'num_class': 4,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
}

# Top 30 features — hardcoded from tuning script (same as model search)
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
# SMOTE CONFIGURATION
# ============================================================================
# Applied inside the LOFO loop (on training folds only) to avoid data leakage.
# Also applied before final model training on all data.
#
# SMOTE_STRATEGY: target count for the 'high' class after resampling.
#   - Current high count: ~402 (after combining main + upsampled CSVs)
#   - Set to 700-800 to boost without over-generating
#   - Try 'borderline' if regular SMOTE doesn't improve high recall
#
# Set SMOTE_STRATEGY to None to disable SMOTE entirely (baseline comparison).

SMOTE_TYPE = 'smote'          # 'smote' or 'borderline'
SMOTE_TARGET_HIGH = 700       # synthetic high samples to reach this count
SMOTE_K_NEIGHBORS = 5         # k neighbors for interpolation (default 5)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load and combine main + upsampled CSVs."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df_main = pd.read_csv(MAIN_CSV)
    df_main['SBS'] = df_main['SBS'].replace({'mod': 'moderate'})
    df_main = df_main[df_main['SBS'].isin(CLASS_NAMES)].copy()
    print(f"Main CSV: {len(df_main)} rows")

    df_up = pd.read_csv(OLD_UPSAMPLED_CSV)
    df_up['SBS'] = df_up['SBS'].replace({'mod': 'moderate'})
    df_up = df_up[df_up['SBS'].isin(CLASS_NAMES)].copy()
    print(f"Upsampled CSV: {len(df_up)} rows")

    common_cols = list(set(df_main.columns) & set(df_up.columns))
    df = pd.concat([df_main[common_cols], df_up[common_cols]], ignore_index=True)
    df['SBS'] = df['SBS'].replace({'mod': 'moderate'})
    df = df[df['SBS'].isin(CLASS_NAMES)].copy()

    print(f"Combined: {len(df)} rows")
    print(f"\nClass distribution:")
    for cls in CLASS_NAMES:
        count = (df['SBS'] == cls).sum()
        print(f"  {cls:12s}: {count:5d} ({count/len(df)*100:.1f}%)")

    fire_col = 'Fire_year' if 'Fire_year' in df.columns else 'fire'
    fires = df[fire_col].unique()
    print(f"\nFires: {len(fires)}")
    print(f"Fire column: {fire_col}")

    return df, fire_col


def prepare_xy(df, features):
    """Extract X, y arrays. Returns only features present in df."""
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features not found in df, skipping: {missing[:5]}{'...' if len(missing)>5 else ''}")
    X = df[available].fillna(df[available].median()).values.astype(np.float32)
    X = np.where(np.isnan(X) | np.isinf(X), 0.0, X)
    y = np.array([LABEL_MAP[s] for s in df['SBS'].values])
    return X, y, available


def apply_smote(X_train, y_train, smote_type=SMOTE_TYPE,
                target_high=SMOTE_TARGET_HIGH, k_neighbors=SMOTE_K_NEIGHBORS):
    """
    Apply SMOTE to the training fold only — never to test data.
    Only oversamples the 'high' class (label=3) to avoid distorting other classes.
    Returns resampled X_train, y_train and sample weights.
    """
    if target_high is None:
        return X_train, y_train, compute_sample_weight('balanced', y_train)

    current_high = (y_train == 3).sum()
    if current_high == 0:
        # No high samples in this fold's training set — skip SMOTE
        return X_train, y_train, compute_sample_weight('balanced', y_train)

    # Only oversample high if we'd actually be increasing it
    if current_high >= target_high:
        return X_train, y_train, compute_sample_weight('balanced', y_train)

    # k_neighbors can't exceed the number of high samples minus 1
    k = min(k_neighbors, current_high - 1)
    if k < 1:
        return X_train, y_train, compute_sample_weight('balanced', y_train)

    sampling_strategy = {3: target_high}  # only resample class 3 (high)

    if smote_type == 'borderline':
        sampler = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k,
            random_state=RANDOM_STATE
        )
    else:
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k,
            random_state=RANDOM_STATE
        )

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    sample_weights = compute_sample_weight('balanced', y_res)
    return X_res, y_res, sample_weights


# ============================================================================
# LEAVE-ONE-FIRE-OUT CROSS-VALIDATION
# ============================================================================

def leave_one_fire_out_cv(df, fire_col, features):
    """
    Leave-One-Fire-Out CV:
    Train on all fires except one, predict on the held-out fire.
    Fires with < MIN_SAMPLES are used for training only (not held-out).
    """
    MIN_SAMPLES = 1

    print("\n" + "=" * 70)
    print("LEAVE-ONE-FIRE-OUT CROSS-VALIDATION")
    print("=" * 70)

    X_all, y_all, available = prepare_xy(df, features)
    fires_all = df[fire_col].values
    y_all_str = df['SBS'].values

    unique_fires = sorted(df[fire_col].unique())
    fire_counts = df[fire_col].value_counts()
    eval_fires = [f for f in unique_fires if fire_counts.get(f, 0) >= MIN_SAMPLES]
    small_fires = [f for f in unique_fires if fire_counts.get(f, 0) < MIN_SAMPLES]

    print(f"Total fires: {len(unique_fires)}")
    print(f"Features used: {len(available)}")
    print(f"Total samples: {len(y_all)}")
    print(f"Fires for evaluation (>= {MIN_SAMPLES} samples): {len(eval_fires)}")
    if small_fires:
        print(f"Training-only fires (< {MIN_SAMPLES} samples): {len(small_fires)}")

    all_y_true = []
    all_y_pred = []
    fire_results = []

    for i, held_out_fire in enumerate(eval_fires):
        train_mask = fires_all != held_out_fire
        test_mask = fires_all == held_out_fire

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]

        # Apply SMOTE to training fold only — never touches test data
        X_train, y_train, sample_weights = apply_smote(X_train, y_train)

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred = model.predict(X_test)

        # Per-class recall
        recalls = recall_score(
            y_test, y_pred,
            labels=list(range(4)),
            average=None,
            zero_division=0
        )

        kappa = cohen_kappa_score(y_test, y_pred) if len(np.unique(y_test)) >= 2 else float('nan')
        acc = accuracy_score(y_test, y_pred)

        fire_results.append({
            'fire': held_out_fire,
            'kappa': kappa,
            'accuracy': acc,
            'n_samples': len(y_test),
            'n_classes': len(np.unique(y_test)),
            'recall_unburned': recalls[0],
            'recall_low': recalls[1],
            'recall_moderate': recalls[2],
            'recall_high': recalls[3],
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        kappa_str = f"Kappa: {kappa:.4f}" if not np.isnan(kappa) else "Kappa: N/A (1 class)"
        print(
            f"  [{i+1:2d}/{len(eval_fires)}] {str(held_out_fire):30s} "
            f"{kappa_str}  Acc: {acc:.4f}  "
            f"High R: {recalls[3]:.3f}  Mod R: {recalls[2]:.3f}  (n={len(y_test)})"
        )

    all_y_true = np.array(all_y_true)  # numeric: 0=unburned,1=low,2=moderate,3=high
    all_y_pred = np.array(all_y_pred)

    overall_kappa = cohen_kappa_score(all_y_true, all_y_pred)
    overall_acc = accuracy_score(all_y_true, all_y_pred)

    # Use numeric labels with explicit ordering — avoids alphabetical sort bug
    NUMERIC_LABELS = [0, 1, 2, 3]  # matches CLASS_NAMES order exactly
    cm = confusion_matrix(all_y_true, all_y_pred, labels=NUMERIC_LABELS)

    fire_df = pd.DataFrame(fire_results)
    valid_kappas = fire_df['kappa'].dropna()

    print("\n" + "=" * 70)
    print("LOFO CV RESULTS")
    print("=" * 70)
    print(f"\nOverall Cohen's Kappa:   {overall_kappa:.4f}")
    print(f"Overall Accuracy:        {overall_acc:.4f}")
    print(f"\nPer-fire Kappa stats (n={len(valid_kappas)} fires):")
    print(f"  Mean:   {valid_kappas.mean():.4f}")
    print(f"  Median: {valid_kappas.median():.4f}")
    print(f"  Std:    {valid_kappas.std():.4f}")
    print(f"  Min:    {valid_kappas.min():.4f}  ({fire_df.loc[valid_kappas.idxmin(), 'fire']})")
    print(f"  Max:    {valid_kappas.max():.4f}  ({fire_df.loc[valid_kappas.idxmax(), 'fire']})")
    print(f"\nMean per-class recall across fires:")
    for cls in CLASS_NAMES:
        col = f'recall_{cls}'
        print(f"  {cls:12s}: {fire_df[col].mean():.4f}")

    print(f"\nClassification Report (pooled across all LOFO folds):")
    print(classification_report(
        all_y_true, all_y_pred,
        labels=NUMERIC_LABELS,       # enforce 0,1,2,3 order
        target_names=CLASS_NAMES     # maps 0→unburned, 1→low, 2→moderate, 3→high
    ))

    # Moderate confusion breakdown — the key diagnostic
    mod_idx = CLASS_NAMES.index('moderate')  # == 2
    mod_row = cm[mod_idx]
    mod_total = mod_row.sum()
    print(f"Moderate class — where predictions go:")
    for j, cls in enumerate(CLASS_NAMES):
        print(f"  → {cls:12s}: {mod_row[j]:4d} ({mod_row[j]/mod_total*100:.1f}%)")

    # High confusion breakdown
    high_idx = CLASS_NAMES.index('high')  # == 3
    high_row = cm[high_idx]
    high_total = high_row.sum()
    print(f"\nHigh class — where predictions go:")
    for j, cls in enumerate(CLASS_NAMES):
        print(f"  → {cls:12s}: {high_row[j]:4d} ({high_row[j]/high_total*100:.1f}%)")

    return {
        'overall_kappa': overall_kappa,
        'overall_accuracy': overall_acc,
        'fire_results': fire_df,
        'y_true': all_y_true,   # numeric arrays — correctly ordered
        'y_pred': all_y_pred,
        'cm': cm,
        'features_used': available,
    }


# ============================================================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================================================

def train_final_model(df, features):
    """Train XGBoost on ALL data for deployment."""
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 70)

    X, y, available = prepare_xy(df, features)

    # Apply SMOTE to full training set before final model fit
    X, y, sample_weights = apply_smote(X, y)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, sample_weight=sample_weights)

    print(f"Trained on {len(y)} samples, {len(available)} features")
    print(f"Class distribution: { {INV_LABEL_MAP[c]: int((y==c).sum()) for c in range(4)} }")

    # Fit scaler for metadata (used in Colab normalization if needed)
    scaler = StandardScaler()
    scaler.fit(X)

    return model, scaler, available


# ============================================================================
# TF EXPORT
# ============================================================================

def export_model(model, scaler, features, cv_results, save_dir):
    """Export XGBoost as TF SavedModel with metadata."""
    import tensorflow as tf

    os.makedirs(save_dir, exist_ok=True)
    n_features = len(features)

    # Save raw XGBoost model
    joblib.dump(model, f'{save_dir}/xgb_lr01_top30.joblib')
    print(f"XGBoost model saved: {save_dir}/xgb_lr01_top30.joblib")

    # Numeric classes from XGBoost are already [0,1,2,3] = [unburned, low, moderate, high]
    reorder_idx = list(range(4))
    print(f"Model classes: {list(model.classes_)}")

    class XGBModule(tf.Module):
        def __init__(self, xgb_model, reorder):
            super().__init__()
            self.xgb_model = xgb_model
            self.reorder = reorder

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, n_features], dtype=tf.float32, name='covariate_input')
        ])
        def __call__(self, covariate_input):
            result = tf.py_function(
                func=self._predict,
                inp=[covariate_input],
                Tout=tf.float32
            )
            result.set_shape([None, 4])
            return result

        def _predict(self, inputs):
            proba = self.xgb_model.predict_proba(inputs.numpy()).astype(np.float32)
            proba = proba[:, self.reorder]
            return tf.constant(proba)

    module = XGBModule(model, reorder_idx)

    # Smoke test
    test_input = tf.constant(np.random.randn(2, n_features).astype(np.float32))
    test_out = module(test_input)
    print(f"TF wrapper test: (2, {n_features}) → {test_out.shape}")
    print(f"Row sums (should be ~1.0): {test_out.numpy().sum(axis=1)}")

    export_path = f'{save_dir}/xgb_lr01_top30_savedmodel'
    tf.saved_model.save(
        module, export_path,
        signatures={
            'serving_default': module.__call__.get_concrete_function(
                tf.TensorSpec(shape=[None, n_features], dtype=tf.float32, name='covariate_input')
            )
        }
    )
    print(f"SavedModel exported: {export_path}")

    # Model metadata (for Colab notebook)
    metadata = {
        'model_type': 'XGBClassifier_TF_Wrapped',
        'model_name': 'XGB_lr01_Top30',
        'feature_names': features,
        'n_features': n_features,
        'class_names': CLASS_NAMES,
        'label_map': LABEL_MAP,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'xgb_params': {k: str(v) for k, v in XGB_PARAMS.items()},
        'tuning_cv_kappa': 0.6101,
        'tuning_cv_high_recall': 0.6366,
        'tuning_cv_composite': 0.6181,
        'training_info': {
            'trained_on': 'all_fires_with_upsampling',
            'lofo_cv_kappa': float(cv_results['overall_kappa']),
            'lofo_cv_accuracy': float(cv_results['overall_accuracy']),
            'mean_per_fire_kappa': float(cv_results['fire_results']['kappa'].dropna().mean()),
            'n_fires': int(len(cv_results['fire_results'])),
            'n_samples': int(cv_results['fire_results']['n_samples'].sum()),
            'mean_high_recall': float(cv_results['fire_results']['recall_high'].mean()),
            'mean_moderate_recall': float(cv_results['fire_results']['recall_moderate'].mean()),
        },
        'export_date': datetime.now().isoformat(),
    }

    with open(f'{save_dir}/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {save_dir}/model_metadata.json")

    with open(f'{save_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved: {save_dir}/scaler.pkl")

    return export_path


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(cv_results, save_dir):
    """Generate diagnostic plots."""
    os.makedirs(save_dir, exist_ok=True)
    fire_df = cv_results['fire_results']
    valid = fire_df.dropna(subset=['kappa']).sort_values('kappa', ascending=True)

    # 1. Per-fire Kappa bar chart
    fig, ax = plt.subplots(figsize=(12, max(8, len(valid) * 0.35)))
    colors = [
        '#e74c3c' if k < 0.3 else
        '#f39c12' if k < 0.5 else
        '#2ecc71' if k < 0.65 else
        '#27ae60'
        for k in valid['kappa']
    ]
    ax.barh(valid['fire'].astype(str), valid['kappa'], color=colors,
            edgecolor='black', linewidth=0.5)
    ax.axvline(x=0.65, color='green', linestyle='--', linewidth=2, label='Target: 0.65')
    ax.axvline(x=valid['kappa'].mean(), color='blue', linestyle='--', linewidth=1.5,
               label=f"Mean: {valid['kappa'].mean():.3f}")
    ax.set_xlabel("Cohen's Kappa", fontsize=12)
    ax.set_title(
        f"Leave-One-Fire-Out CV — Per-Fire Kappa (XGB_lr01 Top30)\n"
        f"Overall Pooled Kappa: {cv_results['overall_kappa']:.4f}",
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/lofo_per_fire_kappa.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/lofo_per_fire_kappa.png")

    # 2. Confusion matrix (counts + %)
    cm = cv_results['cm']
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'

    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                cbar_kws={'label': 'Count'})
    # Rows/cols: 0=unburned, 1=low, 2=moderate, 3=high — matches CLASS_NAMES order
    ax.set_title(
        f'XGB_lr01 Top30 — LOFO CV Confusion Matrix\nPooled Kappa: {cv_results["overall_kappa"]:.4f}',
        fontweight='bold', fontsize=14
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/lofo_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/lofo_confusion_matrix.png")

    # 3. Per-fire Kappa distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid['kappa'], bins=15, color='#3498db', edgecolor='black', alpha=0.8)
    ax.axvline(x=valid['kappa'].mean(), color='red', linestyle='--', linewidth=2,
               label=f"Mean: {valid['kappa'].mean():.3f}")
    ax.axvline(x=valid['kappa'].median(), color='orange', linestyle='--', linewidth=2,
               label=f"Median: {valid['kappa'].median():.3f}")
    ax.axvline(x=0.65, color='green', linestyle='--', linewidth=2, label='Target: 0.65')
    ax.set_xlabel("Cohen's Kappa", fontsize=12)
    ax.set_ylabel("Number of Fires", fontsize=12)
    ax.set_title("Distribution of Per-Fire Kappa Scores (XGB_lr01 Top30)",
                 fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/lofo_kappa_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/lofo_kappa_distribution.png")

    # 4. Per-fire high recall scatter vs kappa (key diagnostic plot)
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        valid['kappa'], valid['recall_high'],
        c=valid['recall_moderate'], cmap='RdYlGn',
        s=80, edgecolors='black', linewidths=0.5, alpha=0.85,
        label='Fire (color = moderate recall)'
    )
    plt.colorbar(sc, ax=ax, label='Moderate Recall')
    ax.axvline(x=0.65, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Kappa target 0.65')
    ax.axhline(y=0.60, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='High recall target 0.60')
    ax.set_xlabel("Per-Fire Cohen's Kappa", fontsize=12)
    ax.set_ylabel("Per-Fire High Class Recall", fontsize=12)
    ax.set_title("Kappa vs High Recall per Fire\n(color = moderate recall)",
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/lofo_kappa_vs_high_recall.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/lofo_kappa_vs_high_recall.png")

    # 5. Save per-fire CSV
    fire_df.to_csv(f'{save_dir}/lofo_per_fire_results.csv', index=False)
    print(f"Saved: {save_dir}/lofo_per_fire_results.csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("XGB_lr01 Top30 — LOFO CV + FINAL TRAINING + TF EXPORT")
    print("=" * 70)
    print(f"XGB params: {XGB_PARAMS}")
    print(f"Features: Top30 ({len(TOP_30_FEATURES)} features)")
    print(f"SMOTE: type={SMOTE_TYPE}, target_high={SMOTE_TARGET_HIGH}, k={SMOTE_K_NEIGHBORS}")

    # Load data
    df, fire_col = load_data()

    # LOFO CV
    cv_results = leave_one_fire_out_cv(df, fire_col, TOP_30_FEATURES)

    # Plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_results(cv_results, OUTPUT_DIR)

    # Train final model on ALL data
    final_model, scaler, features_used = train_final_model(df, TOP_30_FEATURES)

    # Export
    os.makedirs(GEE_EXPORT_DIR, exist_ok=True)
    export_path = export_model(final_model, scaler, features_used, cv_results, GEE_EXPORT_DIR)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    fire_df = cv_results['fire_results']
    print(f"""
Model: XGB_lr01 (n_estimators=1000, max_depth=6, lr=0.01)
Features: Top30

LOFO Cross-Validation:
  Overall Pooled Kappa:    {cv_results['overall_kappa']:.4f}
  Overall Accuracy:        {cv_results['overall_accuracy']:.4f}
  Mean per-fire Kappa:     {fire_df['kappa'].dropna().mean():.4f}
  Mean high recall:        {fire_df['recall_high'].mean():.4f}
  Mean moderate recall:    {fire_df['recall_moderate'].mean():.4f}
  Fires evaluated:         {len(fire_df)}

Final Model:
  Trained on ALL {len(df)} samples ({len(df[fire_col].unique())} fires)
  Exported to: {export_path}

Next steps:
  1. Upload SavedModel to GCS:
     gsutil -m cp -r {GEE_EXPORT_DIR}/xgb_lr01_top30_savedmodel \\
       gs://ee2-sanjana-wildfire-ml/models/xgb_lr01_top30_final/

  2. Update Colab notebook:
     MODEL_PATH = 'gs://ee2-sanjana-wildfire-ml/models/xgb_lr01_top30_final'
     FEATURE_NAMES = (from {GEE_EXPORT_DIR}/model_metadata.json)

  3. GEE: export Top30 features for all fires (same band order as FEATURE_NAMES)
  4. Run Colab inference pipeline for wall-to-wall maps
""")

    return cv_results, final_model


if __name__ == '__main__':
    cv_results, model = main()