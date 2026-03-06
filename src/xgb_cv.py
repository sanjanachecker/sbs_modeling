"""
XGBoost Burn Severity — Train on All Fires + Fire-Level LOO CV
================================================================
Best config from tuning: XGB OldUp + Top50
  - n_estimators=500, max_depth=6, learning_rate=0.05
  - subsample=0.8, colsample_bytree=0.8, min_child_weight=3
  - balanced sample weights
  - Top 50 features (RF importance ranked)

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
    accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

MAIN_CSV = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_complete_covariates_fixed_1229.csv'
OLD_UPSAMPLED_CSV = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_upsampled_points_with_covariates_fixed.csv'

OUTPUT_DIR = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/results_xgb_final'
GEE_EXPORT_DIR = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/gee_models_xgb_final'

RANDOM_STATE = 42

CLASS_NAMES = ['unburned', 'low', 'moderate', 'high']
LABEL_MAP = {'unburned': 0, 'low': 1, 'moderate': 2, 'high': 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Best XGBoost hyperparameters from tuning
XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'objective': 'multi:softprob',
    'num_class': 4,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
}

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

    # Combine on common columns
    common_cols = list(set(df_main.columns) & set(df_up.columns))
    df = pd.concat([df_main[common_cols], df_up[common_cols]], ignore_index=True)
    df['SBS'] = df['SBS'].replace({'mod': 'moderate'})
    df = df[df['SBS'].isin(CLASS_NAMES)].copy()

    print(f"Combined: {len(df)} rows")
    print(f"\nClass distribution:")
    for cls in CLASS_NAMES:
        count = (df['SBS'] == cls).sum()
        print(f"  {cls:12s}: {count:5d} ({count/len(df)*100:.1f}%)")

    # Identify fire column
    fire_col = 'Fire_year' if 'Fire_year' in df.columns else 'fire'
    fires = df[fire_col].unique()
    print(f"\nFires: {len(fires)}")
    print(f"Fire column: {fire_col}")

    return df, fire_col


def get_top50_features(df):
    """Get top 50 features using a preliminary RF (same as tuning script)."""
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
    print(f"Top 5: {top50[:5]}")
    return top50


# ============================================================================
# LEAVE-ONE-FIRE-OUT CROSS-VALIDATION
# ============================================================================

def leave_one_fire_out_cv(df, fire_col, features):
    """
    Leave-One-Fire-Out CV:
    For each fire with enough samples, train on all other fires, predict on the held-out fire.
    Small fires (< MIN_SAMPLES) are still used for TRAINING but not as held-out test fires.
    """
    MIN_SAMPLES = 1  # Minimum samples to use a fire as a held-out test fold

    print("\n" + "=" * 70)
    print("LEAVE-ONE-FIRE-OUT CROSS-VALIDATION")
    print("=" * 70)

    available = [f for f in features if f in df.columns]
    X_all = df[available].fillna(df[available].median()).values.astype(np.float32)
    X_all = np.where(np.isnan(X_all) | np.isinf(X_all), 0.0, X_all)
    y_all_str = df['SBS'].values
    y_all = np.array([LABEL_MAP[s] for s in y_all_str])
    fires_all = df[fire_col].values

    unique_fires = sorted(df[fire_col].unique())
    print(f"Total fires: {len(unique_fires)}")
    print(f"Features: {len(available)}")
    print(f"Total samples: {len(y_all)}")

    # Count samples per fire and identify which are large enough for evaluation
    fire_counts = df[fire_col].value_counts()
    eval_fires = [f for f in unique_fires if fire_counts.get(f, 0) >= MIN_SAMPLES]
    small_fires = [f for f in unique_fires if fire_counts.get(f, 0) < MIN_SAMPLES]
    print(f"Fires with >= {MIN_SAMPLES} samples (used for evaluation): {len(eval_fires)}")
    print(f"Fires with < {MIN_SAMPLES} samples (training only): {len(small_fires)}")

    all_y_true = []
    all_y_pred = []
    fire_results = []

    for i, held_out_fire in enumerate(eval_fires):
        train_mask = fires_all != held_out_fire
        test_mask = fires_all == held_out_fire

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]

        # Balanced sample weights
        sample_weights = compute_sample_weight('balanced', y_train)

        # Train XGBoost
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics for this fire
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

    # Overall metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    overall_kappa = cohen_kappa_score(all_y_true, all_y_pred)
    overall_acc = accuracy_score(all_y_true, all_y_pred)

    # Convert to string labels for classification report
    y_true_str = np.array([INV_LABEL_MAP[y] for y in all_y_true])
    y_pred_str = np.array([INV_LABEL_MAP[y] for y in all_y_pred])

    cm = confusion_matrix(y_true_str, y_pred_str, labels=CLASS_NAMES)

    fire_df = pd.DataFrame(fire_results)
    valid_kappas = fire_df['kappa'].dropna()

    print("\n" + "=" * 70)
    print("LOFO CV RESULTS")
    print("=" * 70)
    print(f"\nMinimum samples threshold: {MIN_SAMPLES}")
    print(f"Fires evaluated: {len(fire_df)} / {len(unique_fires)} total")
    print(f"Fires excluded (< {MIN_SAMPLES} samples, used for training only): {len(small_fires)}")
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
    print(classification_report(y_true_str, y_pred_str, target_names=CLASS_NAMES))

    # Moderate row
    mod_idx = CLASS_NAMES.index('moderate')
    mod_row = cm[mod_idx]
    mod_total = mod_row.sum()
    print(f"Moderate row breakdown:")
    for j, cls in enumerate(CLASS_NAMES):
        print(f"  → {cls:12s}: {mod_row[j]:4d} ({mod_row[j]/mod_total*100:.1f}%)")

    return {
        'overall_kappa': overall_kappa,
        'overall_accuracy': overall_acc,
        'fire_results': fire_df,
        'y_true': y_true_str,
        'y_pred': y_pred_str,
        'cm': cm,
    }


# ============================================================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================================================

def train_final_model(df, features):
    """Train XGBoost on ALL data for deployment."""
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 70)

    available = [f for f in features if f in df.columns]

    X = df[available].fillna(df[available].median()).values.astype(np.float32)
    X = np.where(np.isnan(X) | np.isinf(X), 0.0, X)
    y = np.array([LABEL_MAP[s] for s in df['SBS'].values])

    sample_weights = compute_sample_weight('balanced', y)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, sample_weight=sample_weights)

    print(f"Trained on {len(y)} samples, {len(available)} features")
    print(f"Class distribution: { {INV_LABEL_MAP[c]: int((y==c).sum()) for c in range(4)} }")

    # Scaler for normalization in prediction pipeline
    scaler = StandardScaler()
    scaler.fit(X)

    return model, scaler, available


# ============================================================================
# EXPORT
# ============================================================================

def export_model(model, scaler, features, cv_results, save_dir):
    """Export as joblib + TF SavedModel + metadata."""
    import tensorflow as tf

    os.makedirs(save_dir, exist_ok=True)
    n_features = len(features)

    # Save XGBoost model
    joblib.dump(model, f'{save_dir}/xgb_burn_severity.joblib')
    print(f"XGBoost model saved: {save_dir}/xgb_burn_severity.joblib")

    # TF wrapper
    # XGBoost classes_ are [0, 1, 2, 3] matching LABEL_MAP order
    # which matches CLASS_NAMES = [unburned, low, moderate, high]
    model_classes = list(model.classes_)
    reorder_idx = list(range(4))  # Already in correct order for numeric labels
    print(f"Model classes: {model_classes}")

    class XGBModule(tf.Module):
        def __init__(self, xgb_model, reorder):
            super().__init__()
            self.xgb_model = xgb_model
            self.reorder = reorder

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, n_features], dtype=tf.float32, name='covariate_input')
        ])
        def __call__(self, covariate_input):
            result = tf.py_function(func=self._predict, inp=[covariate_input], Tout=tf.float32)
            result.set_shape([None, 4])
            return result

        def _predict(self, inputs):
            proba = self.xgb_model.predict_proba(inputs.numpy()).astype(np.float32)
            proba = proba[:, self.reorder]
            return tf.constant(proba)

    module = XGBModule(model, reorder_idx)

    # Test
    test_input = tf.constant(np.random.randn(2, n_features).astype(np.float32))
    test_out = module(test_input)
    print(f"TF wrapper test: (2, {n_features}) → {test_out.shape}")
    print(f"Row sums: {test_out.numpy().sum(axis=1)}")

    export_path = f'{save_dir}/xgb_burn_severity_savedmodel'
    tf.saved_model.save(
        module, export_path,
        signatures={
            'serving_default': module.__call__.get_concrete_function(
                tf.TensorSpec(shape=[None, n_features], dtype=tf.float32, name='covariate_input')
            )
        }
    )
    print(f"SavedModel: {export_path}")

    # Metadata
    metadata = {
        'model_type': 'XGBClassifier_TF_Wrapped',
        'feature_names': features,
        'n_features': n_features,
        'class_names': CLASS_NAMES,
        'label_map': LABEL_MAP,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'xgb_params': {k: str(v) for k, v in XGB_PARAMS.items()},
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

    print(f"Metadata + scaler saved")
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
    ax.set_title(f"Leave-One-Fire-Out CV — Per-Fire Kappa\nOverall: {cv_results['overall_kappa']:.4f}",
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
    ax.set_title(f'XGBoost LOFO CV — Kappa: {cv_results["overall_kappa"]:.4f}',
                 fontweight='bold', fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/lofo_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/lofo_confusion_matrix.png")

    # 3. Kappa distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid['kappa'], bins=15, color='#3498db', edgecolor='black', alpha=0.8)
    ax.axvline(x=valid['kappa'].mean(), color='red', linestyle='--', linewidth=2,
               label=f"Mean: {valid['kappa'].mean():.3f}")
    ax.axvline(x=valid['kappa'].median(), color='orange', linestyle='--', linewidth=2,
               label=f"Median: {valid['kappa'].median():.3f}")
    ax.axvline(x=0.65, color='green', linestyle='--', linewidth=2, label='Target: 0.65')
    ax.set_xlabel("Cohen's Kappa", fontsize=12)
    ax.set_ylabel("Number of Fires", fontsize=12)
    ax.set_title("Distribution of Per-Fire Kappa Scores", fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/lofo_kappa_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/lofo_kappa_distribution.png")

    # 4. Save per-fire results CSV
    fire_df.to_csv(f'{save_dir}/lofo_per_fire_results.csv', index=False)
    print(f"Saved: {save_dir}/lofo_per_fire_results.csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("XGBoost BURN SEVERITY — ALL FIRES + LOFO CV")
    print("=" * 70)

    # Load data
    df, fire_col = load_data()

    # Get top 50 features
    top50 = get_top50_features(df)

    # Leave-One-Fire-Out CV
    cv_results = leave_one_fire_out_cv(df, fire_col, top50)

    # Plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_results(cv_results, OUTPUT_DIR)

    # Train final model on ALL data
    final_model, scaler, features = train_final_model(df, top50)

    # Export
    os.makedirs(GEE_EXPORT_DIR, exist_ok=True)
    export_path = export_model(final_model, scaler, features, cv_results, GEE_EXPORT_DIR)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
LOFO Cross-Validation:
  Overall Kappa:      {cv_results['overall_kappa']:.4f}
  Overall Accuracy:   {cv_results['overall_accuracy']:.4f}
  Mean per-fire Kappa: {cv_results['fire_results']['kappa'].dropna().mean():.4f}
  Fires evaluated:    {len(cv_results['fire_results'])}

Final Model:
  Trained on ALL {len(df)} samples across all fires
  Features: {len(features)}
  Exported to: {export_path}

To deploy:
  1. gsutil -m cp -r {GEE_EXPORT_DIR}/xgb_burn_severity_savedmodel \\
       gs://ee2-sanjana-wildfire-ml/models/xgb_burn_severity_final/

  2. Update Colab notebook:
     - MODEL_PATH = 'gs://ee2-sanjana-wildfire-ml/models/xgb_burn_severity_final'
     - Update SCALER_MEAN and SCALER_SCALE from {GEE_EXPORT_DIR}/model_metadata.json

  3. Run GEE 50-feature export for all fires
  4. Run Colab notebook to generate maps
""")

    return cv_results, final_model


if __name__ == '__main__':
    cv_results, model = main()