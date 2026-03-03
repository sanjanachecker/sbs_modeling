"""
MLP Wildfire Burn Severity Classification (v2)
================================================
Improvements over v1:
1. Loads from CSVs instead of TFRecords (simpler for MLP-only)
2. Downsamples unburned to match high class count
3. Asymmetric cost matrix: penalizes moderate→low more than moderate→high
4. Focused on MLP only (removed CNN and Hybrid)
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import pickle

# ============================================================================
# CONFIGURATION
# ============================================================================

# Top 30 features from RF baseline
TOP_30_FEATURES = [
    'dnbr', 'dndvi', 'dndbi', 'dbsi', 'nbr', 'bsi', 'ndvi', 'ndbi',
    'meanelev_32', 'wc_bio19', 'nirBand', 'wc_bio05', 'rdgh_6', 'blueBand',
    'minelev_4', 'greenBand', 'wc_bio06', 'swir2Band', 'pisrdif_2021-11-22',
    'pisrdif_2021-12-22', 'stddevelev_32', 'maxc_2', 'wc_bio12', 'wc_bio07',
    'dmndwi', 'wc_bio18', 'wc_bio17', 'wc_bio02', 'vd_5', 'planc_32'
]

# Class mapping
CLASS_MAP = {'unburned': 0, 'low': 1, 'moderate': 2, 'mod': 2, 'high': 3}
CLASS_NAMES = ['unburned', 'low', 'moderate', 'high']
N_CLASSES = 4

# Training config
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15

# Set to True to enable asymmetric loss (EXPERIMENTAL — currently causes collapse)
USE_ASYMMETRIC_LOSS = False

# Paths — UPDATE THESE
MAIN_CSV = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_complete_covariates_fixed_1229.csv'
UPSAMPLED_CSV = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_upsampled_points_with_covariates_fixed.csv'
OUTPUT_DIR = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/results_v2'
GEE_EXPORT_DIR = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/gee_models_v2'


# ============================================================================
# DATA LOADING & BALANCING
# ============================================================================

def load_and_balance_data(main_csv, upsampled_csv, features, class_map):
    """
    Load CSVs, combine, and downsample unburned to match high class count.
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # Load both CSVs
    df_main = pd.read_csv(main_csv)
    print(f"Main CSV: {len(df_main)} rows")

    df_upsampled = pd.read_csv(upsampled_csv)
    print(f"Upsampled CSV: {len(df_upsampled)} rows")

    df = pd.concat([df_main, df_upsampled], ignore_index=True)
    print(f"Combined: {len(df)} rows")

    # Map labels
    df['label'] = df['SBS'].str.lower().map(class_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # Show original distribution
    print("\nOriginal class distribution:")
    for name, idx in sorted(set((v, k) for k, v in class_map.items() if k != 'mod'), key=lambda x: x[0]):
        count = (df['label'] == name).sum()
        print(f"  {idx:12s}: {count:5d}")

    # =====================================================================
    # FIX 1: Downsample unburned PER-FIRE to match high class count
    # =====================================================================
    fire_col = 'Fire_year' if 'Fire_year' in df.columns else 'fire'
    print(f"\nDownsampling unburned per-fire to match high class count per-fire")
    print(f"Fire column: {fire_col}")

    df_unburned = df[df['label'] == 0]
    df_others = df[df['label'] != 0]

    unburned_sampled_parts = []
    total_before = 0
    total_after = 0

    for fire in df[fire_col].unique():
        # Get unburned for this fire
        fire_unburned = df_unburned[df_unburned[fire_col] == fire]
        # Get high count for this fire
        fire_high_count = ((df[fire_col] == fire) & (df['label'] == 3)).sum()

        total_before += len(fire_unburned)

        if fire_high_count == 0 or len(fire_unburned) == 0:
            # No high samples for this fire — keep original unburned only (not synthetic)
            if 'source' in fire_unburned.columns:
                keep = fire_unburned[fire_unburned['source'] != 'upsampled_buffer']
            else:
                keep = fire_unburned
            unburned_sampled_parts.append(keep)
            total_after += len(keep)
            continue

        if len(fire_unburned) <= fire_high_count:
            # Already fewer unburned than high — keep all
            unburned_sampled_parts.append(fire_unburned)
            total_after += len(fire_unburned)
        else:
            # Downsample: prefer original points, then fill with synthetic
            if 'source' in fire_unburned.columns:
                originals = fire_unburned[fire_unburned['source'] != 'upsampled_buffer']
                synthetics = fire_unburned[fire_unburned['source'] == 'upsampled_buffer']

                if len(originals) >= fire_high_count:
                    sampled = originals.sample(n=fire_high_count, random_state=42)
                else:
                    n_synth_needed = fire_high_count - len(originals)
                    sampled = pd.concat([
                        originals,
                        synthetics.sample(n=min(n_synth_needed, len(synthetics)), random_state=42)
                    ])
            else:
                sampled = fire_unburned.sample(n=fire_high_count, random_state=42)

            unburned_sampled_parts.append(sampled)
            total_after += len(sampled)

    print(f"Unburned: {total_before} → {total_after} (per-fire matched to high)")

    df_unburned_sampled = pd.concat(unburned_sampled_parts, ignore_index=True)
    df_balanced = pd.concat([df_unburned_sampled, df_others], ignore_index=True)

    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nBalanced class distribution:")
    for cls_name in CLASS_NAMES:
        cls_idx = class_map[cls_name]
        count = (df_balanced['label'] == cls_idx).sum()
        pct = count / len(df_balanced) * 100
        print(f"  {cls_name:12s}: {count:5d} ({pct:.1f}%)")

    # Extract features
    available = [f for f in features if f in df_balanced.columns]
    missing = [f for f in features if f not in df_balanced.columns]
    if missing:
        print(f"\nWARNING: Missing features: {missing}")

    X = df_balanced[available].values.astype(np.float32)
    y = df_balanced['label'].values

    # Handle NaN/Inf
    nan_mask = np.isnan(X) | np.isinf(X)
    if nan_mask.any():
        print(f"Replacing {nan_mask.sum()} NaN/Inf values with 0")
        X = np.where(nan_mask, 0.0, X)

    print(f"\nFinal: X={X.shape}, y={y.shape}")
    return X, y, available, df_balanced


# ============================================================================
# ASYMMETRIC LOSS FUNCTION
# ============================================================================

def create_asymmetric_loss(cost_matrix):
    """
    Custom loss that penalizes certain misclassifications more heavily.

    cost_matrix[i][j] = penalty for predicting class j when true class is i.
    Diagonal should be 0 (correct predictions).
    Higher values = stronger penalty for that specific error.
    """
    cost_matrix_tf = tf.constant(cost_matrix, dtype=tf.float32)

    def asymmetric_categorical_crossentropy(y_true, y_pred):
        # y_true is sparse (integer labels)
        y_true_int = tf.cast(y_true, tf.int32)
        y_true_onehot = tf.one_hot(y_true_int, depth=cost_matrix_tf.shape[0])

        # Get per-sample cost weights based on true class
        # cost_weights[i] = row i of cost matrix = penalties for each predicted class
        cost_weights = tf.gather(cost_matrix_tf, y_true_int)

        # Clip predictions for numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Weighted cross-entropy
        # Standard CE: -sum(y_true * log(y_pred))
        # Asymmetric: -sum((y_true + cost_weights * (1 - y_true)) * log(y_pred))
        #
        # For correct class: weight = 1
        # For wrong classes: weight = cost_matrix[true_class][pred_class]
        weights = y_true_onehot + cost_weights * (1.0 - y_true_onehot)

        loss = -tf.reduce_sum(weights * tf.math.log(y_pred), axis=-1)
        return tf.reduce_mean(loss)

    return asymmetric_categorical_crossentropy


# ============================================================================
# MODEL
# ============================================================================

def create_mlp_model(input_dim=30, num_classes=4):
    """MLP for tabular covariate classification — same as original v1"""
    inputs = tf.keras.Input(shape=(input_dim,), name='covariate_input')

    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='MLP_Only')
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_mlp(model, X_train, y_train, X_val, y_val, class_weights, cost_matrix):
    """Train MLP with optional asymmetric loss and class weights."""

    if USE_ASYMMETRIC_LOSS:
        loss_fn = create_asymmetric_loss(cost_matrix)
        print("Using: asymmetric categorical crossentropy")
    else:
        loss_fn = 'sparse_categorical_crossentropy'
        print("Using: standard sparse categorical crossentropy")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=['accuracy']
    )

    print("\n===== TRAINING MLP v2 =====")
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


# ============================================================================
# EVALUATION & VISUALIZATION
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate and print metrics."""
    print("\n===== EVALUATION =====")

    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    kappa = cohen_kappa_score(y_test, y_pred)
    accuracy = (y_test == y_pred).mean()

    print(f"Cohen's Kappa:    {kappa:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)

    return {
        'kappa': kappa,
        'accuracy': accuracy,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(results, save_path):
    """Plot confusion matrix with counts and percentages."""
    cm = results['confusion_matrix']
    kappa = results['kappa']
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'

    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                cbar_kws={'label': 'Count'})
    ax.set_title(f'MLP v2 (Test Set) — Kappa: {kappa:.4f}', fontweight='bold', fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_history(history, save_path):
    """Plot loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title('Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
    axes[1].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_fire_kappa(X_test, y_test, y_pred, test_fires, save_path):
    """Per-fire kappa breakdown."""
    fire_kappas = []
    for fire in sorted(np.unique(test_fires)):
        mask = test_fires == fire
        if mask.sum() < 10:
            continue
        y_f = y_test[mask]
        p_f = y_pred[mask]
        if len(np.unique(y_f)) < 2:
            continue
        k = cohen_kappa_score(y_f, p_f)
        fire_kappas.append({'fire': fire, 'kappa': k, 'n': mask.sum()})

    if not fire_kappas:
        print("Not enough per-fire data to plot")
        return

    fire_df = pd.DataFrame(fire_kappas).sort_values('kappa', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(fire_df) * 0.35)))
    colors = ['#e74c3c' if k < 0.4 else '#f39c12' if k < 0.6 else '#2ecc71'
              for k in fire_df['kappa']]
    ax.barh(fire_df['fire'], fire_df['kappa'], color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(x=fire_df['kappa'].mean(), color='black', linestyle='--', label=f"Mean: {fire_df['kappa'].mean():.3f}")
    ax.set_xlabel("Cohen's Kappa", fontsize=12)
    ax.set_title('Per-Fire Kappa Scores (Test Set)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("MLP BURN SEVERITY v2 — Balanced + Asymmetric Loss")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load & balance data
    # ------------------------------------------------------------------
    X, y, available_features, df_balanced_for_split = load_and_balance_data(
        MAIN_CSV, UPSAMPLED_CSV, TOP_30_FEATURES, CLASS_MAP
    )

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # Class weights (balanced)
    # ------------------------------------------------------------------
    class_weights_array = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(class_weights_array))
    print(f"\nClass weights: {class_weights}")

    # ------------------------------------------------------------------
    # FIX 2: Asymmetric cost matrix (OPTIONAL)
    # ------------------------------------------------------------------
    # cost_matrix[true][pred] = extra penalty
    # Key idea: moderate→low is BAD (underestimating severity)
    #           moderate→high is less bad (overestimating is safer)
    #
    # MILD version — only slightly nudges moderate away from low
    # Rows = true class, Cols = predicted class
    #            unburned  low  moderate  high
    cost_matrix = np.array([
        #         unb   low   mod   high
        [0.0,   1.0,  1.0,  1.0],   # true=unburned
        [1.0,   0.0,  1.0,  1.0],   # true=low
        [1.0,   1.5,  0.0,  0.8],   # true=moderate: low pred 1.5x penalty, high only 0.8x
        [1.0,   1.0,  1.0,  0.0],   # true=high
    ], dtype=np.float32)

    if USE_ASYMMETRIC_LOSS:
        print("\nAsymmetric cost matrix ENABLED:")
        print(f"  {'':12s} {'unburned':>10s} {'low':>10s} {'moderate':>10s} {'high':>10s}")
        for i, name in enumerate(CLASS_NAMES):
            row = '  '.join([f'{cost_matrix[i, j]:>10.1f}' for j in range(N_CLASSES)])
            print(f"  {name:12s} {row}")
        print("  Key: moderate→low = 1.5 (mild penalty), moderate→high = 0.8 (slight reward)")
    else:
        print("\nUsing STANDARD cross-entropy loss (asymmetric loss disabled)")
        print("Set USE_ASYMMETRIC_LOSS = True to enable after confirming baseline works")

    # ------------------------------------------------------------------
    # Train/val/test split
    # ------------------------------------------------------------------
    print("\n===== SPLITTING DATA =====")

    # Get fire names for per-fire analysis
    fire_col = 'Fire_year' if 'Fire_year' in df_balanced_for_split.columns else 'fire'
    fire_names = df_balanced_for_split[fire_col].values if fire_col in df_balanced_for_split.columns else np.array(['unknown'] * len(y))

    X_temp, X_test, y_temp, y_test, fires_temp, fires_test = train_test_split(
        X_norm, y, fire_names, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model = create_mlp_model(input_dim=X_norm.shape[1], num_classes=N_CLASSES)
    model, history = train_mlp(model, X_train, y_train, X_val, y_val,
                                class_weights, cost_matrix)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    results = evaluate_model(model, X_test, y_test)

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_confusion_matrix(results, f'{OUTPUT_DIR}/mlp_v2_confusion_matrix.png')
    plot_training_history(history, f'{OUTPUT_DIR}/mlp_v2_training_history.png')
    plot_per_fire_kappa(X_test, y_test, results['y_pred'], fires_test,
                        f'{OUTPUT_DIR}/mlp_v2_per_fire_kappa.png')

    # ------------------------------------------------------------------
    # Export model for GEE prediction pipeline
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPORTING MODEL FOR GEE")
    print("=" * 60)

    os.makedirs(GEE_EXPORT_DIR, exist_ok=True)

    mlp_export_path = f'{GEE_EXPORT_DIR}/mlp_burn_severity'
    try:
        mlp_model_export = model
        mlp_model_export.export(mlp_export_path)
        print(f"SavedModel exported to: {mlp_export_path}")
    except Exception as e:
        print(f"export() failed ({e}), using tf.saved_model.save...")
        tf.saved_model.save(model, mlp_export_path)
        print(f"SavedModel exported to: {mlp_export_path}")

    # Save metadata
    metadata = {
        'model_type': 'MLP_v2',
        'feature_names': available_features,
        'n_features': len(available_features),
        'class_names': CLASS_NAMES,
        'class_mapping': {k: v for k, v in CLASS_MAP.items() if k != 'mod'},
        'input_shape': [len(available_features)],
        'output_shape': [N_CLASSES],
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'cost_matrix': cost_matrix.tolist(),
        'training_info': {
            'n_train': len(y_train),
            'n_val': len(y_val),
            'n_test': len(y_test),
            'test_kappa': float(results['kappa']),
            'test_accuracy': float(results['accuracy']),
            'unburned_downsampled_to': int((y == 0).sum()),
            'batch_size': BATCH_SIZE,
            'epochs_trained': len(history.history['loss']),
            'learning_rate': LEARNING_RATE,
        },
        'export_date': datetime.now().isoformat(),
    }

    with open(f'{GEE_EXPORT_DIR}/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(f'{GEE_EXPORT_DIR}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Metadata saved to: {GEE_EXPORT_DIR}/model_metadata.json")
    print(f"Scaler saved to: {GEE_EXPORT_DIR}/scaler.pkl")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Cohen's Kappa:    {results['kappa']:.4f}")
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"\nConfusion matrix moderate row:")
    cm = results['confusion_matrix']
    mod_row = cm[2]
    mod_total = mod_row.sum()
    print(f"  → unburned: {mod_row[0]} ({mod_row[0]/mod_total*100:.1f}%)")
    print(f"  → low:      {mod_row[1]} ({mod_row[1]/mod_total*100:.1f}%)  ← should decrease")
    print(f"  → moderate: {mod_row[2]} ({mod_row[2]/mod_total*100:.1f}%)  ← should increase")
    print(f"  → high:     {mod_row[3]} ({mod_row[3]/mod_total*100:.1f}%)  ← OK if increases")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nTo re-run the GEE prediction pipeline with this new model:")
    print(f"  1. Upload {mlp_export_path}/ to gs://ee2-sanjana-wildfire-ml/models/mlp_burn_severity_v2/")
    print(f"  2. Update MODEL_PATH in the Colab notebook")
    print(f"  3. Update SCALER_MEAN and SCALER_SCALE from {GEE_EXPORT_DIR}/model_metadata.json")
    print(f"  4. Re-run the Colab notebook to generate new prediction maps")

    return results, model


if __name__ == '__main__':
    results, model = main()