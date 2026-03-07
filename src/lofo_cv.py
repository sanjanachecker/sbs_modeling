"""
ExtraTrees Burn Severity — Train on All Fires + LOFO CV + GEE Export
=====================================================================
Best config from model search (10-fold CV Kappa = 0.6118, Test Kappa = 0.6053):

  ExtraTreesClassifier(
      n_estimators=500, max_depth=None, min_samples_split=2,
      min_samples_leaf=1, class_weight='balanced'
  )
  Features: Top 50 (RF importance ranked)
  Data: Old upsampled (moderate-matched, 3204 samples)

Pipeline:
  1. LOFO CV for honest evaluation
  2. Train final model on ALL data
  3. Export as TF SavedModel for GEE prediction pipeline
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
    accuracy_score, precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

MAIN_CSV = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_complete_covariates_fixed_1229.csv'
OLD_UPSAMPLED_CSV = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_upsampled_points_with_covariates_fixed.csv'

OUTPUT_DIR = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/results_et_final'
GEE_EXPORT_DIR = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/gee_models_et_final'

RANDOM_STATE = 42

CLASS_NAMES = ['unburned', 'low', 'moderate', 'high']
STAGE1_CLASS_NAMES = ['unburned', 'low', 'burned_strong']
STAGE2_CLASS_NAMES = ['moderate', 'high']

EXCLUDE_COLS = {
    'SBS', 'Fire_year', 'fire', 'source', 'data_source', 'Source',
    'PointX', 'PointY', '.geo', 'system:index', 'label',
    'latitude', 'longitude', 'lat', 'lon', 'x', 'y'
}


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
    print(f"Main CSV: {len(df_main)} rows")

    df_up = pd.read_csv(OLD_UPSAMPLED_CSV)
    print(f"Upsampled CSV: {len(df_up)} rows")

    common_cols = list(set(df_main.columns) & set(df_up.columns))
    df = pd.concat([df_main[common_cols], df_up[common_cols]], ignore_index=True)
    df['SBS'] = df['SBS'].replace({'mod': 'moderate'})
    df = df[df['SBS'].isin(CLASS_NAMES)].copy()
    # ------------------------------------------------------------------
    # Feature engineering: rdNBR
    # rdNBR = dNBR / sqrt(|pre-fire NBR|)
    # Here we use existing 'nbr' as the denominator proxy already in the data.
    # Small epsilon prevents divide-by-zero / unstable values.
    # ------------------------------------------------------------------
    if 'dnbr' in df.columns and 'nbr' in df.columns:
        df['rdnbr'] = df['dnbr'] / np.sqrt(np.abs(df['nbr']) + 1e-6)
        df['rdnbr'] = df['rdnbr'].replace([np.inf, -np.inf], np.nan)

        # ------------------------------------------------------------------
    # Feature engineering: interaction features
    # These help the model separate moderate vs high severity
    # by combining burn change with vegetation / SWIR response.
    # ------------------------------------------------------------------
    interaction_specs = [
        ('dnbr', 'ndvi', 'dnbr_x_ndvi'),
        ('dnbr', 'swir2Band', 'dnbr_x_swir2'),
        ('dnbr', 'swir1Band', 'dnbr_x_swir1'),
        ('dndvi', 'swir2Band', 'dndvi_x_swir2'),
        ('rdnbr', 'swir2Band', 'rdnbr_x_swir2'),
    ]

    for a, b, new_col in interaction_specs:
        if a in df.columns and b in df.columns:
            df[new_col] = df[a] * df[b]
            df[new_col] = df[new_col].replace([np.inf, -np.inf], np.nan)

    print(f"Combined: {len(df)} rows")
    print(f"\nClass distribution:")
    for cls in CLASS_NAMES:
        count = (df['SBS'] == cls).sum()
        print(f"  {cls:12s}: {count:5d} ({count/len(df)*100:.1f}%)")

    fire_col = 'Fire_year' if 'Fire_year' in df.columns else 'fire'
    print(f"\nFires: {df[fire_col].nunique()}")
    print(f"Fire column: {fire_col}")

    return df, fire_col


def get_top50_features(df):
    """Get top 50 features using a preliminary RF."""
    print("\n" + "=" * 70)
    print("DETERMINING TOP 50 FEATURES")
    print("=" * 70)

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    all_features = sorted([c for c in numeric if c.lower() not in {x.lower() for x in EXCLUDE_COLS}])
    available = [f for f in all_features if f in df.columns]

    X = df[available].fillna(df[available].median()).values.astype(np.float32)
    X = np.where(np.isnan(X) | np.isinf(X), 0.0, X)
    y = df['SBS'].values

    prelim_rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    prelim_rf.fit(X, y)

    importances = prelim_rf.feature_importances_
    indices = np.argsort(importances)[::-1][:50]
    top50 = [available[i] for i in indices]

    print(f"Top 50 features determined")
    print(f"Top 50: {top50}")
    return top50


def prepare_xy(df, features):
    """Extract X, y arrays."""
    available = [f for f in features if f in df.columns]
    X = df[available].fillna(df[available].median()).values.astype(np.float32)
    X = np.where(np.isnan(X) | np.isinf(X), 0.0, X)
    y = df['SBS'].values
    return X, y, available

def make_stage1_labels(y):
    """Map 4-class labels into stage-1 3-class labels."""
    y_stage1 = np.array(y, dtype=object).copy()
    y_stage1[np.isin(y_stage1, ['moderate', 'high'])] = 'burned_strong'
    return y_stage1

def predict_two_stage(stage1_model, stage2_model, X):
    """
    Return:
      y_pred_final: final 4-class predictions
      final_proba:  Nx4 probabilities in CLASS_NAMES order
    """
    n = X.shape[0]

    # ------------------------------------------------------------
    # Stage 1 probabilities: ['burned_strong', 'low', 'unburned'] or similar
    # Reorder into STAGE1_CLASS_NAMES = ['unburned', 'low', 'burned_strong']
    # ------------------------------------------------------------
    s1_raw = stage1_model.predict_proba(X)
    s1_classes = list(stage1_model.classes_)
    s1_reorder = [s1_classes.index(c) for c in STAGE1_CLASS_NAMES]
    s1_proba = s1_raw[:, s1_reorder]

    # Default final probability matrix: [unburned, low, moderate, high]
    final_proba = np.zeros((n, 4), dtype=np.float32)

    # Put stage-1 unburned and low directly
    final_proba[:, 0] = s1_proba[:, STAGE1_CLASS_NAMES.index('unburned')]
    final_proba[:, 1] = s1_proba[:, STAGE1_CLASS_NAMES.index('low')]

    burned_idx = STAGE1_CLASS_NAMES.index('burned_strong')
    burned_mask = s1_proba[:, burned_idx] > 0

    # ------------------------------------------------------------
    # Stage 2 probabilities only for burned_strong branch
    # Reorder into STAGE2_CLASS_NAMES = ['moderate', 'high']
    # Then multiply by P(burned_strong)
    # ------------------------------------------------------------
    if burned_mask.any():
        X_burned = X[burned_mask]
        s2_raw = stage2_model.predict_proba(X_burned)
        s2_classes = list(stage2_model.classes_)
        s2_reorder = [s2_classes.index(c) for c in STAGE2_CLASS_NAMES]
        s2_proba = s2_raw[:, s2_reorder]

        p_burned = s1_proba[burned_mask, burned_idx]
        final_proba[burned_mask, 2] = p_burned * s2_proba[:, STAGE2_CLASS_NAMES.index('moderate')]
        final_proba[burned_mask, 3] = p_burned * s2_proba[:, STAGE2_CLASS_NAMES.index('high')]

    # Numerical safety
    row_sums = final_proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    final_proba = final_proba / row_sums

    y_pred_final = np.array(CLASS_NAMES)[np.argmax(final_proba, axis=1)]
    return y_pred_final, final_proba

# def create_model():
#     """Create the best ExtraTrees model with boosted high class weight."""
#     # 'balanced' weights based on class counts (3204 total):
#     #   unburned (1110): ~0.72
#     #   low (665):       ~1.20
#     #   moderate (1027): ~0.78
#     #   high (402):      ~1.99
#     #
#     # We further boost high to 3.0 to fix the high→moderate misclassification
#     return ExtraTreesClassifier(
#         n_estimators=500,
#         max_depth=None,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         class_weight={
#             'unburned': 0.72,
#             'low': 1.20,
#             'moderate': 0.78,
#             'high': 2,
#         },
#         random_state=RANDOM_STATE,
#         n_jobs=-1
#     )

def create_stage1_model():
    """Stage 1: unburned vs low vs burned_strong."""
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def create_stage2_model():
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight={
            'moderate': 1.0,
            'high': 2.5
        },
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

# ============================================================================
# LEAVE-ONE-FIRE-OUT CROSS-VALIDATION
# ============================================================================

def leave_one_fire_out_cv(df, fire_col, features):
    """LOFO CV with ExtraTrees."""
    MIN_SAMPLES = 5

    print("\n" + "=" * 70)
    print("LEAVE-ONE-FIRE-OUT CROSS-VALIDATION")
    print("=" * 70)

    available = [f for f in features if f in df.columns]
    X_all = df[available].fillna(df[available].median()).values.astype(np.float32)
    X_all = np.where(np.isnan(X_all) | np.isinf(X_all), 0.0, X_all)
    y_all = df['SBS'].values
    fires_all = df[fire_col].values

    unique_fires = sorted(df[fire_col].unique())
    fire_counts = df[fire_col].value_counts()
    eval_fires = [f for f in unique_fires if fire_counts.get(f, 0) >= MIN_SAMPLES]
    small_fires = [f for f in unique_fires if fire_counts.get(f, 0) < MIN_SAMPLES]

    print(f"Total fires: {len(unique_fires)}")
    print(f"Features: {len(available)}")
    print(f"Total samples: {len(y_all)}")
    print(f"Fires evaluated (>= {MIN_SAMPLES} samples): {len(eval_fires)}")
    print(f"Fires training-only (< {MIN_SAMPLES} samples): {len(small_fires)}")

    all_y_true = []
    all_y_pred = []
    fire_results = []

    for i, held_out_fire in enumerate(eval_fires):
        train_mask = fires_all != held_out_fire
        test_mask = fires_all == held_out_fire

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]

        # -----------------------------
        # Stage 1: unburned / low / burned_strong
        # -----------------------------
        y_train_stage1 = make_stage1_labels(y_train)

        stage1_model = create_stage1_model()
        stage1_model.fit(X_train, y_train_stage1)

        # -----------------------------
        # Stage 2: moderate / high only
        # -----------------------------
        mh_mask_train = np.isin(y_train, ['moderate', 'high'])
        X_train_stage2 = X_train[mh_mask_train]
        y_train_stage2 = y_train[mh_mask_train]

        stage2_model = create_stage2_model()
        stage2_model.fit(X_train_stage2, y_train_stage2)

        # -----------------------------
        # Two-stage prediction
        # -----------------------------
                # -----------------------------
        # Two-stage prediction
        # -----------------------------
        y_pred, final_proba = predict_two_stage(stage1_model, stage2_model, X_test)
        

        if len(np.unique(y_test)) >= 2:
            kappa = cohen_kappa_score(y_test, y_pred)
        else:
            kappa = float('nan')
        acc = accuracy_score(y_test, y_pred)

        fire_results.append({
            'fire': held_out_fire,
            'kappa': kappa,
            'accuracy': acc,
            'n_samples': len(y_test),
            'n_classes': len(np.unique(y_test)),
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        status = f"Kappa: {kappa:.4f}" if not np.isnan(kappa) else "Kappa: N/A (1 class)"
        print(f"  [{i+1}/{len(eval_fires)}] {held_out_fire:30s} {status}  Acc: {acc:.4f}  (n={len(y_test)})")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    overall_kappa = cohen_kappa_score(all_y_true, all_y_pred)
    overall_acc = accuracy_score(all_y_true, all_y_pred)
    cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASS_NAMES)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_y_true, all_y_pred, labels=CLASS_NAMES, zero_division=0
    )

    fire_df = pd.DataFrame(fire_results)
    valid_kappas = fire_df['kappa'].dropna()

    print("\n" + "=" * 70)
    print("LOFO CV RESULTS")
    print("=" * 70)
    print(f"\nFires evaluated: {len(fire_df)} / {len(unique_fires)} total")
    print(f"Samples evaluated: {len(all_y_true)} / {len(y_all)} total")
    print(f"\nOverall Cohen's Kappa:  {overall_kappa:.4f}")
    print(f"Overall Accuracy:       {overall_acc:.4f}")
    print(f"\nPer-fire Kappa stats:")
    print(f"  Mean:   {valid_kappas.mean():.4f}")
    print(f"  Median: {valid_kappas.median():.4f}")
    print(f"  Std:    {valid_kappas.std():.4f}")
    print(f"  Min:    {valid_kappas.min():.4f} ({fire_df.loc[valid_kappas.idxmin(), 'fire']})")
    print(f"  Max:    {valid_kappas.max():.4f} ({fire_df.loc[valid_kappas.idxmax(), 'fire']})")
    print(f"\nClassification Report:")
    print(classification_report(
    all_y_true,
    all_y_pred,
    labels=CLASS_NAMES,
    target_names=CLASS_NAMES))

    print(f"Per-class metrics:")
    for j, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:12s}  P: {precision[j]:.3f}  R: {recall[j]:.3f}  F1: {f1[j]:.3f}  (n={support[j]})")

    mod_idx = CLASS_NAMES.index('moderate')
    mod_row = cm[mod_idx]
    mod_total = mod_row.sum()
    print(f"\nModerate row breakdown:")
    for j, cls in enumerate(CLASS_NAMES):
        print(f"  → {cls:12s}: {mod_row[j]:4d} ({mod_row[j]/mod_total*100:.1f}%)")

    return {
        'overall_kappa': overall_kappa,
        'overall_accuracy': overall_acc,
        'fire_results': fire_df,
        'y_true': all_y_true,
        'y_pred': all_y_pred,
        'cm': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# ============================================================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================================================

def train_final_model(df, features):
    """Train 2-stage ExtraTrees on ALL data for deployment."""
    print("\n" + "=" * 70)
    print("TRAINING FINAL 2-STAGE MODEL ON ALL DATA")
    print("=" * 70)

    X, y, available = prepare_xy(df, features)

    # -----------------------------
    # Stage 1
    # -----------------------------
    y_stage1 = make_stage1_labels(y)
    stage1_model = create_stage1_model()
    stage1_model.fit(X, y_stage1)

    # -----------------------------
    # Stage 2
    # -----------------------------
    mh_mask = np.isin(y, ['moderate', 'high'])
    X_stage2 = X[mh_mask]
    y_stage2 = y[mh_mask]

    stage2_model = create_stage2_model()
    stage2_model.fit(X_stage2, y_stage2)

    print(f"Trained on {len(y)} samples, {len(available)} features")
    for cls in CLASS_NAMES:
        count = (y == cls).sum()
        print(f"  {cls:12s}: {count}")

    scaler = StandardScaler()
    scaler.fit(X)

    return {
        'stage1_model': stage1_model,
        'stage2_model': stage2_model,
        'scaler': scaler,
        'features': available
    }

# ============================================================================
# EXPORT
# ============================================================================

def export_model(final_bundle, cv_results, save_dir):
    """Export 2-stage model as joblib + TF SavedModel + metadata."""
    import tensorflow as tf

    os.makedirs(save_dir, exist_ok=True)

    stage1_model = final_bundle['stage1_model']
    stage2_model = final_bundle['stage2_model']
    scaler = final_bundle['scaler']
    features = final_bundle['features']
    n_features = len(features)

    # ------------------------------------------------------------
    # Save sklearn models
    # ------------------------------------------------------------
    joblib.dump(stage1_model, f'{save_dir}/et_stage1.joblib')
    joblib.dump(stage2_model, f'{save_dir}/et_stage2.joblib')
    print(f"Stage 1 model saved: {save_dir}/et_stage1.joblib")
    print(f"Stage 2 model saved: {save_dir}/et_stage2.joblib")

    # Reorder helpers
    s1_model_classes = list(stage1_model.classes_)
    s1_reorder = [s1_model_classes.index(c) for c in STAGE1_CLASS_NAMES]

    s2_model_classes = list(stage2_model.classes_)
    s2_reorder = [s2_model_classes.index(c) for c in STAGE2_CLASS_NAMES]

    print(f"Stage 1 classes: {s1_model_classes}")
    print(f"Stage 2 classes: {s2_model_classes}")

    class TwoStageETModule(tf.Module):
        def __init__(self, stage1_model, stage2_model, s1_reorder, s2_reorder):
            super().__init__()
            self.stage1_model = stage1_model
            self.stage2_model = stage2_model
            self.s1_reorder = s1_reorder
            self.s2_reorder = s2_reorder

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, n_features], dtype=tf.float32, name='covariate_input')
        ])
        def __call__(self, covariate_input):
            result = tf.py_function(func=self._predict, inp=[covariate_input], Tout=tf.float32)
            result.set_shape([None, 4])
            return result

        def _predict(self, inputs):
            X = inputs.numpy()

            # Stage 1
            s1_raw = self.stage1_model.predict_proba(X).astype(np.float32)
            s1_proba = s1_raw[:, self.s1_reorder]  # ['unburned', 'low', 'burned_strong']

            n = X.shape[0]
            out = np.zeros((n, 4), dtype=np.float32)

            out[:, 0] = s1_proba[:, STAGE1_CLASS_NAMES.index('unburned')]
            out[:, 1] = s1_proba[:, STAGE1_CLASS_NAMES.index('low')]

            burned_idx = STAGE1_CLASS_NAMES.index('burned_strong')
            burned_mask = s1_proba[:, burned_idx] > 0

            if burned_mask.any():
                X_burned = X[burned_mask]
                s2_raw = self.stage2_model.predict_proba(X_burned).astype(np.float32)
                s2_proba = s2_raw[:, self.s2_reorder]  # ['moderate', 'high']

                p_burned = s1_proba[burned_mask, burned_idx]
                out[burned_mask, 2] = p_burned * s2_proba[:, STAGE2_CLASS_NAMES.index('moderate')]
                out[burned_mask, 3] = p_burned * s2_proba[:, STAGE2_CLASS_NAMES.index('high')]

            row_sums = out.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            out = out / row_sums
            return tf.constant(out, dtype=tf.float32)

    module = TwoStageETModule(stage1_model, stage2_model, s1_reorder, s2_reorder)

    test_input = tf.constant(np.random.randn(2, n_features).astype(np.float32))
    test_out = module(test_input)
    print(f"TF wrapper test: (2, {n_features}) → {test_out.shape}")
    print(f"Row sums (should be ~1.0): {test_out.numpy().sum(axis=1)}")

    export_path = f'{save_dir}/et_burn_severity_savedmodel'
    tf.saved_model.save(
        module, export_path,
        signatures={
            'serving_default': module.__call__.get_concrete_function(
                tf.TensorSpec(shape=[None, n_features], dtype=tf.float32, name='covariate_input')
            )
        }
    )
    print(f"SavedModel: {export_path}")

    metadata = {
        'model_type': 'ExtraTreesClassifier_TwoStage_TF_Wrapped',
        'feature_names': features,
        'n_features': n_features,
        'class_names': CLASS_NAMES,
        'stage1_class_names': STAGE1_CLASS_NAMES,
        'stage2_class_names': STAGE2_CLASS_NAMES,
        'stage1_model_class_order': s1_model_classes,
        'stage2_model_class_order': s2_model_classes,
        'stage1_reorder_indices': s1_reorder,
        'stage2_reorder_indices': s2_reorder,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'training_info': {
            'trained_on': 'all_fires',
            'lofo_cv_kappa': float(cv_results['overall_kappa']),
            'lofo_cv_accuracy': float(cv_results['overall_accuracy']),
            'mean_per_fire_kappa': float(cv_results['fire_results']['kappa'].dropna().mean()),
            'n_fires': int(len(cv_results['fire_results'])),
            'n_samples': int(cv_results['fire_results']['n_samples'].sum()),
        },
        'export_date': datetime.now().isoformat(),
    }

    with open(f'{save_dir}/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(f'{save_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Metadata + scaler saved")
    return export_path

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(cv_results, save_dir):
    """Generate all plots."""
    os.makedirs(save_dir, exist_ok=True)
    fire_df = cv_results['fire_results']

    # 1. Per-fire Kappa bar chart
    valid = fire_df.dropna(subset=['kappa']).sort_values('kappa', ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(8, len(valid) * 0.3)))
    colors = ['#e74c3c' if k < 0.3 else '#f39c12' if k < 0.5 else '#2ecc71' if k < 0.65 else '#27ae60'
              for k in valid['kappa']]
    ax.barh(valid['fire'], valid['kappa'], color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0.65, color='green', linestyle='--', linewidth=2, label='Target: 0.65')
    ax.axvline(x=valid['kappa'].mean(), color='blue', linestyle='--', linewidth=1.5,
               label=f"Mean: {valid['kappa'].mean():.3f}")
    ax.set_xlabel("Cohen's Kappa", fontsize=12)
    ax.set_title(f"LOFO CV — Per-Fire Kappa (ExtraTrees | Top50)\nOverall: {cv_results['overall_kappa']:.4f}",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/lofo_per_fire_kappa.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/lofo_per_fire_kappa.png")

    # 2. Confusion matrix
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
    ax.set_title(f'ExtraTrees LOFO CV — Kappa: {cv_results["overall_kappa"]:.4f}',
                 fontweight='bold', fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/lofo_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/lofo_confusion_matrix.png")

    # 3. Kappa distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid['kappa'], bins=15, color='#3498db', edgecolor='black', alpha=0.8)
    ax.axvline(x=valid['kappa'].mean(), color='red', linestyle='--', linewidth=2,
               label=f"Mean: {valid['kappa'].mean():.3f}")
    ax.axvline(x=valid['kappa'].median(), color='orange', linestyle='--', linewidth=2,
               label=f"Median: {valid['kappa'].median():.3f}")
    ax.axvline(x=0.65, color='green', linestyle='--', linewidth=2, label='Target: 0.65')
    ax.set_xlabel("Cohen's Kappa", fontsize=12)
    ax.set_ylabel("Number of Fires", fontsize=12)
    ax.set_title("Distribution of Per-Fire Kappa Scores (ExtraTrees)", fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/lofo_kappa_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/lofo_kappa_distribution.png")

    # 4. Per-fire results CSV
    fire_df.to_csv(f'{save_dir}/lofo_per_fire_results.csv', index=False)
    print(f"Saved: {save_dir}/lofo_per_fire_results.csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("ExtraTrees BURN SEVERITY — ALL FIRES + LOFO CV")
    print("=" * 70)

    df, fire_col = load_data()
    top50 = get_top50_features(df)

    # LOFO CV
    cv_results = leave_one_fire_out_cv(df, fire_col, top50)

    # Plots
    plot_results(cv_results, OUTPUT_DIR)

    # Train final model on ALL data
    final_bundle = train_final_model(df, top50)

    # Export
    export_path = export_model(final_bundle, cv_results, GEE_EXPORT_DIR)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Model: 2-Stage ExtraTrees
  Stage 1: unburned / low / burned_strong
  Stage 2: moderate / high
Features: Top 50 (RF importance ranked)

LOFO Cross-Validation:
  Overall Kappa:       {cv_results['overall_kappa']:.4f}
  Overall Accuracy:    {cv_results['overall_accuracy']:.4f}
  Mean per-fire Kappa: {cv_results['fire_results']['kappa'].dropna().mean():.4f}
  Fires evaluated:     {len(cv_results['fire_results'])}

Final Model:
  Trained on ALL {len(df)} samples
  Features: {len(final_bundle['features'])}
  Exported to: {export_path}

To deploy:
  1. gsutil -m cp -r {GEE_EXPORT_DIR}/et_burn_severity_savedmodel \\
       gs://ee2-sanjana-wildfire-ml/models/et_burn_severity_final/

  2. Update Colab notebook:
     - MODEL_PATH = 'gs://ee2-sanjana-wildfire-ml/models/et_burn_severity_final'
     - Update SCALER_MEAN and SCALER_SCALE from {GEE_EXPORT_DIR}/model_metadata.json
     - Update FEATURE_NAMES from model_metadata.json

  3. Run GEE 50-feature export for all fires (if not done already)
  4. Run Colab notebook to generate maps
""")

    return cv_results, final_bundle


if __name__ == '__main__':
    cv_results, model = main()