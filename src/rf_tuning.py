"""
RF Burn Severity — Hyperparameter Tuning + TF Export
=====================================================
1. Tests top 30 vs all features
2. GridSearch over RF hyperparameters
3. Wraps best RF in TensorFlow SavedModel for GEE pipeline
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    cohen_kappa_score, classification_report, confusion_matrix,
    make_scorer, accuracy_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

MAIN_CSV = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_complete_covariates_fixed_1229.csv'
UPSAMPLED_CSV = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_upsampled_points_with_covariates_fixed123.csv'
OUTPUT_DIR = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/results_rf_tuned'
GEE_EXPORT_DIR = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/gee_models_rf'

RANDOM_STATE = 42
TEST_SIZE = 0.2

TOP_30_FEATURES = [
    'dnbr', 'dndvi', 'dndbi', 'dbsi', 'nbr', 'bsi', 'ndvi', 'ndbi',
    'meanelev_32', 'wc_bio19', 'nirBand', 'wc_bio05', 'rdgh_6', 'blueBand',
    'minelev_4', 'greenBand', 'wc_bio06', 'swir2Band', 'pisrdif_2021-11-22',
    'pisrdif_2021-12-22', 'stddevelev_32', 'maxc_2', 'wc_bio12', 'wc_bio07',
    'dmndwi', 'wc_bio18', 'wc_bio17', 'wc_bio02', 'vd_5', 'planc_32'
]

# Columns to exclude when using "all features"
EXCLUDE_COLS = {
    'SBS', 'Fire_year', 'fire', 'source', 'data_source', 'Source',
    'PointX', 'PointY', '.geo', 'system:index', 'label',
    'latitude', 'longitude', 'lat', 'lon', 'x', 'y'
}

CLASS_NAMES = ['unburned', 'low', 'moderate', 'high']


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load and combine CSVs."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    df_main = pd.read_csv(MAIN_CSV)
    df_main['SBS'] = df_main['SBS'].replace({'mod': 'moderate'})
    print(f"Main CSV: {len(df_main)} rows")

    df_up = pd.read_csv(UPSAMPLED_CSV)
    print(f"Upsampled CSV: {len(df_up)} rows")

    # Find common columns
    common_cols = list(set(df_main.columns) & set(df_up.columns))
    df_main_sub = df_main[common_cols].copy()
    df_up_sub = df_up[common_cols].copy()

    df_main_sub['data_source'] = 'original'
    df_up_sub['data_source'] = 'upsampled'

    df = pd.concat([df_main_sub, df_up_sub], ignore_index=True)
    df = df.dropna(subset=['SBS'])
    df = df[df['SBS'].isin(['unburned', 'low', 'moderate', 'high'])]

    print(f"Combined: {len(df)} rows")
    print("\nClass distribution:")
    for cls in CLASS_NAMES:
        count = (df['SBS'] == cls).sum()
        print(f"  {cls:12s}: {count:5d} ({count/len(df)*100:.1f}%)")

    return df


def get_all_numeric_features(df):
    """Get all numeric feature columns (excluding metadata)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c.lower() not in {x.lower() for x in EXCLUDE_COLS}]
    return sorted(feature_cols)


# ============================================================================
# TUNING
# ============================================================================

def run_experiment(df, features, label, param_grid=None):
    """Run a single experiment with given features and optional grid search."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {label}")
    print(f"{'='*60}")
    print(f"Features: {len(features)}")

    # Prepare data
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"Missing {len(missing)} features: {missing[:5]}...")

    X = df[available].fillna(df[available].median()).values.astype(np.float32)
    y = df['SBS'].values

    # Handle NaN/Inf
    nan_mask = np.isnan(X) | np.isinf(X)
    if nan_mask.any():
        print(f"Replacing {nan_mask.sum()} NaN/Inf with 0")
        X = np.where(nan_mask, 0.0, X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Grid search or single model
    kappa_scorer = make_scorer(cohen_kappa_score)

    if param_grid:
        print(f"\nRunning GridSearchCV...")
        base_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        grid = GridSearchCV(
            base_rf, param_grid, cv=5, scoring=kappa_scorer,
            verbose=1, n_jobs=-1, refit=True
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        print(f"\nBest params: {grid.best_params_}")
        print(f"Best CV Kappa: {grid.best_score_:.4f}")

        # Show top 5 results
        results_df = pd.DataFrame(grid.cv_results_)
        results_df = results_df.sort_values('rank_test_score')
        print("\nTop 5 configurations:")
        for _, row in results_df.head(5).iterrows():
            print(f"  Kappa={row['mean_test_score']:.4f} (+/-{row['std_test_score']:.4f}) — {row['params']}")
    else:
        best_model = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1
        )
        best_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    kappa = cohen_kappa_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n--- TEST SET RESULTS ---")
    print(f"Cohen's Kappa:    {kappa:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)

    # Moderate row analysis
    mod_idx = CLASS_NAMES.index('moderate')
    mod_row = cm[mod_idx]
    mod_total = mod_row.sum()
    print(f"Moderate row breakdown:")
    for j, cls in enumerate(CLASS_NAMES):
        print(f"  → {cls:12s}: {mod_row[j]:3d} ({mod_row[j]/mod_total*100:.1f}%)")

    return {
        'label': label,
        'model': best_model,
        'features': available,
        'kappa': kappa,
        'accuracy': accuracy,
        'cm': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'best_params': grid.best_params_ if param_grid else None
    }


# ============================================================================
# TF WRAPPER FOR GEE DEPLOYMENT
# ============================================================================

def export_rf_as_savedmodel(model, features, scaler_mean, scaler_scale, export_dir):
    """
    Wrap a scikit-learn RF in a TensorFlow SavedModel for the GEE pipeline.
    The model expects normalized inputs (same as MLP pipeline).
    """
    import tensorflow as tf

    n_features = len(features)
    n_classes = len(CLASS_NAMES)

    # Get RF predictions as a numpy function
    # We need to "bake" the model into a TF graph via tf.py_function or
    # by converting the RF to a lookup

    # Approach: extract all trees and build a TF voting model
    # Simpler approach: use a tf.function with numpy interop

    # Actually simplest: create a Keras model that replicates RF predictions
    # by storing the RF predict_proba output for the training data range

    # MOST PRACTICAL: create a TF SavedModel that wraps the sklearn predict
    # This works because the Colab notebook calls serve_fn() in Python anyway

    class RFModule(tf.Module):
        def __init__(self, rf_model, feature_names):
            super().__init__()
            self.rf_model = rf_model
            self.feature_names = feature_names
            self.class_names = CLASS_NAMES

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, n_features], dtype=tf.float32, name='covariate_input')
        ])
        def __call__(self, covariate_input):
            # This will be traced but the actual prediction happens via py_function
            result = tf.py_function(
                func=self._predict,
                inp=[covariate_input],
                Tout=tf.float32
            )
            result.set_shape([None, n_classes])
            return result

        def _predict(self, inputs):
            """Run RF prediction on numpy data."""
            x_np = inputs.numpy()
            proba = self.rf_model.predict_proba(x_np).astype(np.float32)
            return tf.constant(proba)

    module = RFModule(model, features)

    # Test
    test_input = tf.constant(np.random.randn(2, n_features).astype(np.float32))
    test_output = module(test_input)
    print(f"TF wrapper test: input {test_input.shape} → output {test_output.shape}")

    # Save
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, 'rf_burn_severity')

    tf.saved_model.save(
        module,
        export_path,
        signatures={
            'serving_default': module.__call__.get_concrete_function(
                tf.TensorSpec(shape=[None, n_features], dtype=tf.float32, name='covariate_input')
            )
        }
    )
    print(f"SavedModel exported to: {export_path}")
    return export_path


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison(results_list, save_dir):
    """Compare all experiments."""
    os.makedirs(save_dir, exist_ok=True)

    # Kappa comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [r['label'] for r in results_list]
    kappas = [r['kappa'] for r in results_list]
    colors = ['#e74c3c' if k < 0.5 else '#f39c12' if k < 0.65 else '#2ecc71' for k in kappas]

    bars = ax.bar(labels, kappas, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0.65, color='green', linestyle='--', linewidth=2, label='Target: 0.65')
    for bar, k in zip(bars, kappas):
        ax.annotate(f'{k:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')

    ax.set_ylabel("Cohen's Kappa", fontsize=12)
    ax.set_title("RF Tuning — Kappa Comparison", fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, max(kappas) * 1.25)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rf_tuning_kappa_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/rf_tuning_kappa_comparison.png")

    # Confusion matrices for all experiments
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 6))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results_list):
        cm = r['cm']
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'

        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f'{r["label"]}\nKappa: {r["kappa"]:.4f}', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/rf_tuning_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/rf_tuning_confusion_matrices.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("RF BURN SEVERITY — HYPERPARAMETER TUNING")
    print("=" * 80)

    df = load_data()
    all_features = get_all_numeric_features(df)
    print(f"\nAll numeric features available: {len(all_features)}")
    print(f"Top 30 features: {len(TOP_30_FEATURES)}")

    results_list = []

    # ------------------------------------------------------------------
    # Experiment 1: Baseline (top 30, current params)
    # ------------------------------------------------------------------
    r1 = run_experiment(df, TOP_30_FEATURES, "Top30 Baseline")
    results_list.append(r1)

    # ------------------------------------------------------------------
    # Experiment 2: All features, default params
    # ------------------------------------------------------------------
    r2 = run_experiment(df, all_features, "All Features")
    results_list.append(r2)

#     # ------------------------------------------------------------------
#     # Experiment 3: Top 30 + GridSearch
#     # ------------------------------------------------------------------
#     param_grid_30 = {
#         'n_estimators': [300, 500, 800],
#         'max_depth': [15, 25, None],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4],
#         'class_weight': ['balanced', 'balanced_subsample'],
#     }
#     r3 = run_experiment(df, TOP_30_FEATURES, "Top30 GridSearch", param_grid=param_grid_30)
#     results_list.append(r3)

#     # ------------------------------------------------------------------
#     # Experiment 4: All features + GridSearch
#     # ------------------------------------------------------------------
#     param_grid_all = {
#         'n_estimators': [300, 500, 800],
#         'max_depth': [15, 25, None],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4],
#         'class_weight': ['balanced', 'balanced_subsample'],
#     }
#     r4 = run_experiment(df, all_features, "AllFeat GridSearch", param_grid=param_grid_all)
#     results_list.append(r4)

#     # ------------------------------------------------------------------
#     # Find best
#     # ------------------------------------------------------------------
#     best = max(results_list, key=lambda r: r['kappa'])
#     print("\n" + "=" * 60)
#     print(f"BEST MODEL: {best['label']}")
#     print(f"Kappa: {best['kappa']:.4f}")
#     if best['best_params']:
#         print(f"Params: {best['best_params']}")
#     print(f"Features: {len(best['features'])}")
#     print("=" * 60)

#     # ------------------------------------------------------------------
#     # Visualizations
#     # ------------------------------------------------------------------
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     plot_comparison(results_list, OUTPUT_DIR)

#     # ------------------------------------------------------------------
#     # Save best RF model
#     # ------------------------------------------------------------------
#     model_path = os.path.join(OUTPUT_DIR, 'best_rf_model.joblib')
#     joblib.dump(best['model'], model_path)
#     print(f"\nBest RF saved: {model_path}")

#     # ------------------------------------------------------------------
#     # Export as TF SavedModel for GEE pipeline
#     # ------------------------------------------------------------------
#     print("\n" + "=" * 60)
#     print("EXPORTING AS TENSORFLOW SAVEDMODEL")
#     print("=" * 60)

#     # Compute scaler params (for the Colab notebook)
#     X_all = df[best['features']].fillna(df[best['features']].median()).values.astype(np.float32)
#     X_all = np.where(np.isnan(X_all) | np.isinf(X_all), 0.0, X_all)
#     scaler = StandardScaler()
#     scaler.fit(X_all)

#     os.makedirs(GEE_EXPORT_DIR, exist_ok=True)
#     export_path = export_rf_as_savedmodel(
#         best['model'], best['features'],
#         scaler.mean_, scaler.scale_,
#         GEE_EXPORT_DIR
#     )

#     # Save metadata
#     metadata = {
#         'model_type': 'RandomForest_TF_Wrapped',
#         'feature_names': best['features'],
#         'n_features': len(best['features']),
#         'class_names': CLASS_NAMES,
#         'input_shape': [len(best['features'])],
#         'output_shape': [4],
#         'scaler_mean': scaler.mean_.tolist(),
#         'scaler_scale': scaler.scale_.tolist(),
#         'training_info': {
#             'test_kappa': float(best['kappa']),
#             'test_accuracy': float(best['accuracy']),
#             'best_params': best['best_params'],
#             'n_features': len(best['features']),
#             'experiment': best['label'],
#         },
#         'export_date': datetime.now().isoformat(),
#     }

#     meta_path = os.path.join(GEE_EXPORT_DIR, 'model_metadata.json')
#     with open(meta_path, 'w') as f:
#         json.dump(metadata, f, indent=2)
#     print(f"Metadata: {meta_path}")

#     scaler_path = os.path.join(GEE_EXPORT_DIR, 'scaler.pkl')
#     with open(scaler_path, 'wb') as f:
#         pickle.dump(scaler, f)
#     print(f"Scaler: {scaler_path}")

#     # ------------------------------------------------------------------
#     # Summary
#     # ------------------------------------------------------------------
#     print("\n" + "=" * 60)
#     print("DONE — NEXT STEPS")
#     print("=" * 60)
#     print(f"""
# Best model: {best['label']} (Kappa: {best['kappa']:.4f})

# To deploy to GEE:
#   1. Upload {GEE_EXPORT_DIR}/rf_burn_severity/ to
#      gs://ee2-sanjana-wildfire-ml/models/rf_burn_severity/

#   2. Update the Colab notebook:
#      - MODEL_PATH = 'gs://ee2-sanjana-wildfire-ml/models/rf_burn_severity'
#      - Update SCALER_MEAN and SCALER_SCALE from {meta_path}
#      - Update FEATURE_NAMES if using all features
#      - Update N_FEATURES

#   3. If using all features, also update the GEE export script
#      to export all covariates (not just 30)

#   4. Re-run the Colab notebook
# """)

#     return results_list, best
    return results_list, None


if __name__ == '__main__':
    results_list, best = main()