"""
Multi-Modal Wildfire Burn Severity Classification
==================================================
Trains multiple model architectures:
1. CNN only (patches)
2. MLP only (tabular covariates)
3. Hybrid CNN+MLP (fusion model)

Compares all approaches to Random Forest baseline
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Patch configuration
PATCH_SIZE = 33
N_BANDS = 12  # 6 pre + 6 post bands (SR_B2 through SR_B7)

# Top 30 features from your RF baseline
TOP_30_FEATURES = [
    'dnbr', 'dndvi', 'dndbi', 'dbsi', 'nbr', 'bsi', 'ndvi', 'ndbi',
    'meanelev_32', 'wc_bio19', 'nirBand', 'wc_bio05', 'rdgh_6', 'blueBand',
    'minelev_4', 'greenBand', 'wc_bio06', 'swir2Band', 'pisrdif_2021-11-22',
    'pisrdif_2021-12-22', 'stddevelev_32', 'maxc_2', 'wc_bio12', 'wc_bio07',
    'dmndwi', 'wc_bio18', 'wc_bio17', 'wc_bio02', 'vd_5', 'planc_32'
]

# Band names in TFRecord (6 bands per timepoint, no SR_B1)
BAND_NAMES = [
    'pre_SR_B2', 'pre_SR_B3', 'pre_SR_B4', 'pre_SR_B5', 'pre_SR_B6', 'pre_SR_B7',
    'post_SR_B2', 'post_SR_B3', 'post_SR_B4', 'post_SR_B5', 'post_SR_B6', 'post_SR_B7'
]

# Class mapping
CLASS_MAP = {
    'unburned': 0,
    'low': 1,
    'moderate': 2,
    'high': 3
}

# Training config
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15

# ============================================================================
# DATA LOADING
# ============================================================================

def parse_tfrecord(example_proto, include_patches=True, include_covariates=True):
    """
    Parse TFRecord with both patches and covariates
    """
    # Base features (always needed)
    feature_description = {
        'SBS': tf.io.FixedLenFeature([], tf.string),
        'source': tf.io.FixedLenFeature([], tf.string),
        'Fire_year': tf.io.FixedLenFeature([], tf.string),
    }
    
    # Add patch bands if needed
    if include_patches:
        for band_name in BAND_NAMES:
            feature_description[band_name] = tf.io.FixedLenFeature([PATCH_SIZE * PATCH_SIZE], tf.float32)
    
    # Add covariates if needed
    if include_covariates:
        for feat in TOP_30_FEATURES:
            feature_description[feat] = tf.io.FixedLenFeature([], tf.float32)
    
    # Parse
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Build outputs
    outputs = {}
    
    # Patches
    if include_patches:
        bands = []
        for band_name in BAND_NAMES:
            band = tf.reshape(parsed[band_name], [PATCH_SIZE, PATCH_SIZE])
            bands.append(band)
        patch = tf.stack(bands, axis=-1)  # [33, 33, 12]
        outputs['patch'] = patch
    
    # Covariates
    if include_covariates:
        covariates = []
        for feat in TOP_30_FEATURES:
            # Handle missing features
            if feat in parsed:
                covariates.append(parsed[feat])
            else:
                covariates.append(0.0)  # Default value
        covariates_tensor = tf.stack(covariates)  # [30]
        outputs['covariates'] = covariates_tensor
    
    # Label
    label_str = parsed['SBS']
    
    # Metadata
    outputs['fire'] = parsed['Fire_year']
    outputs['source'] = parsed['source']
    
    return outputs, label_str


def load_tfrecords(tfrecord_dir, pattern='patches_*.tfrecord', 
                   include_patches=True, include_covariates=True,
                   limit=None):
    """
    Load all TFRecords and extract data
    """
    tfrecord_paths = sorted(Path(tfrecord_dir).glob(pattern))
    
    if limit:
        tfrecord_paths = tfrecord_paths[:limit]
    
    print(f"\n===== LOADING TFRECORDS =====")
    print(f"Found {len(tfrecord_paths)} TFRecord files")
    
    all_patches = []
    all_covariates = []
    all_labels = []
    all_metadata = []
    
    for path in tfrecord_paths:
        print(f"  Loading: {path.name}")
        dataset = tf.data.TFRecordDataset(str(path))
        
        for raw_record in dataset:
            try:
                outputs, label_str = parse_tfrecord(
                    raw_record, 
                    include_patches=include_patches,
                    include_covariates=include_covariates
                )
                
                # Convert label
                label_str_py = label_str.numpy().decode('utf-8')
                
                # ⭐ FIX: Valley fire has 'mod' instead of 'moderate'
                if label_str_py == 'mod':
                    label_str_py = 'moderate'
                
                if label_str_py not in CLASS_MAP:
                    print(f"    Warning: Unknown label '{label_str_py}', skipping")
                    continue
                label = CLASS_MAP[label_str_py]
                
                # Store data
                if include_patches:
                    all_patches.append(outputs['patch'].numpy())
                if include_covariates:
                    all_covariates.append(outputs['covariates'].numpy())
                
                all_labels.append(label)
                all_metadata.append({
                    'fire': outputs['fire'].numpy().decode('utf-8'),
                    'source': outputs['source'].numpy().decode('utf-8'),
                    'label_str': label_str_py
                })
                
            except Exception as e:
                print(f"    Warning: Failed to parse record - {e}")
                continue
    
    # Convert to arrays
    results = {
        'labels': np.array(all_labels),
        'metadata': pd.DataFrame(all_metadata)
    }
    
    if include_patches:
        results['patches'] = np.array(all_patches)
    if include_covariates:
        results['covariates'] = np.array(all_covariates)
    
    print(f"\n===== LOADED DATA =====")
    print(f"Total samples: {len(all_labels)}")
    if include_patches:
        print(f"Patches shape: {results['patches'].shape}")
    if include_covariates:
        print(f"Covariates shape: {results['covariates'].shape}")
    
    print(f"\nClass distribution:")
    for cls_name, cls_idx in CLASS_MAP.items():
        count = (results['labels'] == cls_idx).sum()
        pct = count / len(all_labels) * 100
        print(f"  {cls_name:12s}: {count:5d} ({pct:5.1f}%)")
    
    return results


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

def create_cnn_model(input_shape=(PATCH_SIZE, PATCH_SIZE, N_BANDS), num_classes=4):
    """
    CNN for patch-based classification
    Input: 33x33x12 patches (6 bands × 2 timepoints)
    """
    inputs = tf.keras.Input(shape=input_shape, name='patch_input')
    
    # Conv blocks
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # Dense layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='CNN_Only')
    return model


def create_mlp_model(input_dim=30, num_classes=4):
    """
    MLP for tabular covariate classification
    """
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


def create_hybrid_model(patch_shape=(PATCH_SIZE, PATCH_SIZE, N_BANDS), 
                       covariate_dim=30, num_classes=4):
    """
    Hybrid CNN+MLP model with early fusion
    CNN input: 33x33x12 patches (6 bands × 2 timepoints)
    MLP input: 30 tabular covariates
    """
    # Patch input (CNN branch)
    patch_input = tf.keras.Input(shape=patch_shape, name='patch_input')
    
    # CNN branch
    x1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(patch_input)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
    x1 = tf.keras.layers.Dropout(0.25)(x1)
    
    x1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
    x1 = tf.keras.layers.Dropout(0.25)(x1)
    
    x1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
    x1 = tf.keras.layers.Dropout(0.25)(x1)
    
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dense(128, activation='relu')(x1)
    x1 = tf.keras.layers.Dropout(0.4)(x1)
    
    # Covariate input (MLP branch)
    covariate_input = tf.keras.Input(shape=(covariate_dim,), name='covariate_input')
    
    x2 = tf.keras.layers.Dense(64, activation='relu')(covariate_input)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Dropout(0.3)(x2)
    
    x2 = tf.keras.layers.Dense(32, activation='relu')(x2)
    x2 = tf.keras.layers.Dropout(0.3)(x2)
    
    # Fusion
    merged = tf.keras.layers.Concatenate()([x1, x2])
    
    x = tf.keras.layers.Dense(128, activation='relu')(merged)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = tf.keras.Model(
        inputs=[patch_input, covariate_input],
        outputs=outputs,
        name='Hybrid_CNN_MLP'
    )
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, class_weights, model_name):
    """
    Train model with early stopping
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n===== TRAINING {model_name} =====")
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
# VISUALIZATION (Enhanced)
# ============================================================================

def plot_training_history(history_dict, save_path):
    """
    Plot training and validation loss/accuracy for all models
    """
    n_models = len(history_dict)
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
    
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (model_name, history) in enumerate(history_dict.items()):
        # Loss
        axes[0, i].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, i].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, i].set_title(f'{model_name} - Loss', fontsize=12, fontweight='bold')
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].legend()
        axes[0, i].grid(alpha=0.3)
        
        # Accuracy
        axes[1, i].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
        axes[1, i].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
        axes[1, i].set_title(f'{model_name} - Accuracy', fontsize=12, fontweight='bold')
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Accuracy')
        axes[1, i].legend()
        axes[1, i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training history: {save_path}")


def plot_metrics_comparison(results_dict, class_names, save_path):
    """
    Compare precision, recall, F1 across all models and classes
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = list(results_dict.keys())
    x = np.arange(len(class_names))
    width = 0.25
    
    # Extract metrics for each model
    metrics_data = {}
    for model_name, results in results_dict.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(class_names)), zero_division=0
        )
        accuracy = (y_true == y_pred).mean()
        
        metrics_data[model_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }
    
    # Plot 1: Precision by class
    ax = axes[0, 0]
    for i, model_name in enumerate(models):
        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.bar(x + offset, metrics_data[model_name]['precision'], 
                     width, label=model_name, alpha=0.8)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision by Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Plot 2: Recall by class
    ax = axes[0, 1]
    for i, model_name in enumerate(models):
        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.bar(x + offset, metrics_data[model_name]['recall'], 
                     width, label=model_name, alpha=0.8)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Recall by Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Plot 3: F1-Score by class
    ax = axes[1, 0]
    for i, model_name in enumerate(models):
        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.bar(x + offset, metrics_data[model_name]['f1'], 
                     width, label=model_name, alpha=0.8)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score by Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Plot 4: Overall accuracy comparison
    ax = axes[1, 1]
    accuracies = [metrics_data[m]['accuracy'] for m in models]
    kappas = [results_dict[m]['kappa'] for m in models]
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x_pos + width/2, kappas, width, label="Cohen's Kappa", alpha=0.8, color='#e74c3c')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics comparison: {save_path}")


def plot_model_comparison(results_dict, save_path):
    """
    Compare multiple models (Kappa only - simplified)
    """
    models = list(results_dict.keys())
    kappas = [results_dict[m]['kappa'] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax.bar(models, kappas, color=colors[:len(models)], edgecolor='black', linewidth=1.5)
    
    for bar, kappa in zip(bars, kappas):
        height = bar.get_height()
        ax.annotate(f'{kappa:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
    
    ax.set_ylabel("Cohen's Kappa", fontsize=12)
    ax.set_title("Model Comparison - Burn Severity Classification", fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(kappas) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison: {save_path}")


def plot_confusion_matrices(results_dict, class_names, save_dir):
    """
    Plot confusion matrix for each model
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, results) in zip(axes, results_dict.items()):
        cm = results['confusion_matrix']
        
        # Normalize for percentages
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both count and percentage
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax,
                   cbar_kws={'label': 'Count'})
        ax.set_title(f'{model_name}\nKappa: {results["kappa"]:.4f}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
    
    plt.tight_layout()
    save_path = f'{save_dir}/confusion_matrices_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrices: {save_path}")

def evaluate_model(model, X_test, y_test, model_name, class_names):
    """
    Evaluate model and return metrics
    """
    print(f"\n===== EVALUATING {model_name} =====")
    
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    kappa = cohen_kappa_score(y_test, y_pred)
    
    print(f"\nCohen's Kappa: {kappa:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'kappa': kappa,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_model_comparison(results_dict, save_path):
    """
    Compare multiple models
    """
    models = list(results_dict.keys())
    kappas = [results_dict[m]['kappa'] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax.bar(models, kappas, color=colors[:len(models)], edgecolor='black', linewidth=1.5)
    
    for bar, kappa in zip(bars, kappas):
        height = bar.get_height()
        ax.annotate(f'{kappa:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
    
    ax.set_ylabel("Cohen's Kappa", fontsize=12)
    ax.set_title("Model Comparison - Burn Severity Classification", fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(kappas) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison: {save_path}")


def plot_confusion_matrices(results_dict, class_names, save_dir):
    """
    Plot confusion matrix for each model
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, results) in zip(axes, results_dict.items()):
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'{model_name}\nKappa: {results["kappa"]:.4f}', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    save_path = f'{save_dir}/confusion_matrices_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrices: {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("MULTI-MODAL WILDFIRE BURN SEVERITY CLASSIFICATION")
    print("=" * 80)
    
    # Load data
    tfrecord_dir = '/Users/sanjanachecker/csc/fire_patches2'
    
    data = load_tfrecords(
        tfrecord_dir,
        include_patches=True,
        include_covariates=True,
        limit=None
    )
    
    X_patches = data['patches']
    X_covariates = data['covariates']
    y = data['labels']
    
    # Normalize patches
    print("\n===== NORMALIZING DATA =====")
    patch_mean = X_patches.mean(axis=(0, 1, 2), keepdims=True)
    patch_std = X_patches.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    X_patches_norm = (X_patches - patch_mean) / patch_std
    
    # Normalize covariates
    scaler = StandardScaler()
    X_covariates_norm = scaler.fit_transform(X_covariates)
    
    # Compute class weights
    class_weights_array = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(class_weights_array))
    print(f"Class weights: {class_weights}")
    
    # Train/val/test split
    print("\n===== SPLITTING DATA =====")
    X_temp_patches, X_test_patches, X_temp_cov, X_test_cov, y_temp, y_test = train_test_split(
        X_patches_norm, X_covariates_norm, y,
        test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_patches, X_val_patches, X_train_cov, X_val_cov, y_train, y_val = train_test_split(
        X_temp_patches, X_temp_cov, y_temp,
        test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # Train models
    results = {}
    class_names = ['unburned', 'low', 'moderate', 'high']
    
    # 1. CNN Only
    print("\n" + "=" * 80)
    print("MODEL 1: CNN (Patches Only)")
    print("=" * 80)
    cnn_model = create_cnn_model()
    cnn_model, cnn_history = train_model(
        cnn_model, X_train_patches, y_train, X_val_patches, y_val,
        class_weights, "CNN"
    )
    results['CNN'] = evaluate_model(cnn_model, X_test_patches, y_test, "CNN", class_names)
    
    # 2. MLP Only
    print("\n" + "=" * 80)
    print("MODEL 2: MLP (Covariates Only)")
    print("=" * 80)
    mlp_model = create_mlp_model(input_dim=X_covariates_norm.shape[1])
    mlp_model, mlp_history = train_model(
        mlp_model, X_train_cov, y_train, X_val_cov, y_val,
        class_weights, "MLP"
    )
    results['MLP'] = evaluate_model(mlp_model, X_test_cov, y_test, "MLP", class_names)
    
    # 3. Hybrid
    print("\n" + "=" * 80)
    print("MODEL 3: Hybrid CNN+MLP")
    print("=" * 80)
    hybrid_model = create_hybrid_model(covariate_dim=X_covariates_norm.shape[1])
    hybrid_model, hybrid_history = train_model(
        hybrid_model,
        [X_train_patches, X_train_cov], y_train,
        [X_val_patches, X_val_cov], y_val,
        class_weights, "Hybrid"
    )
    results['Hybrid'] = evaluate_model(
        hybrid_model, [X_test_patches, X_test_cov], y_test, "Hybrid", class_names
    )
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # Create output directory
    output_dir = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all visualizations
    plot_model_comparison(results, f'{output_dir}/multimodal_model_comparison_kappa.png')
    plot_confusion_matrices(results, class_names, output_dir)
    plot_metrics_comparison(results, class_names, f'{output_dir}/multimodal_metrics_comparison.png')
    plot_training_history(
        {'CNN': cnn_history, 'MLP': mlp_history, 'Hybrid': hybrid_history},
        f'{output_dir}/training_history.png'
    )
    
    # Save models (.keras format for local use)
    print("\n" + "=" * 80)
    print("SAVING MODELS (.keras)")
    print("=" * 80)
    
    cnn_model.save(f'{output_dir}/multimodal_cnn_model.keras')
    mlp_model.save(f'{output_dir}/multimodal_mlp_model.keras')
    hybrid_model.save(f'{output_dir}/multimodal_hybrid_model.keras')
    print("✅ Keras models saved")
    
    # ========================================================================
    # EXPORT MLP FOR GEE DEPLOYMENT
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPORTING MLP MODEL FOR GEE DEPLOYMENT")
    print("=" * 80)
    
    # Create GEE export directory
    gee_export_dir = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/gee_models'
    os.makedirs(gee_export_dir, exist_ok=True)
    
    # Export MLP as SavedModel (TensorFlow serving format)
    mlp_export_path = f'{gee_export_dir}/mlp_burn_severity'
    try:
        mlp_model.export(mlp_export_path)
        print(f"✅ MLP SavedModel exported to: {mlp_export_path}")
    except Exception as e:
        print(f"⚠️  Error exporting SavedModel: {e}")
        print("   Attempting alternative export method...")
        tf.saved_model.save(mlp_model, mlp_export_path)
        print(f"✅ MLP SavedModel exported (alternative method)")
    
    # Save model metadata for GEE
    model_metadata = {
        'model_type': 'MLP',
        'feature_names': TOP_30_FEATURES,
        'n_features': len(TOP_30_FEATURES),
        'class_names': class_names,
        'class_mapping': CLASS_MAP,
        'input_shape': [30],
        'output_shape': [4],
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'training_info': {
            'n_train_samples': len(y_train),
            'n_val_samples': len(y_val),
            'n_test_samples': len(y_test),
            'test_kappa': float(results['MLP']['kappa']),
            'batch_size': BATCH_SIZE,
            'epochs_trained': len(mlp_history.history['loss']),
            'learning_rate': LEARNING_RATE,
        },
        'export_date': datetime.now().isoformat(),
    }
    
    metadata_path = f'{gee_export_dir}/model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    print(f"✅ Model metadata saved: {metadata_path}")
    
    # Save scaler as pickle for easy loading
    import pickle
    scaler_path = f'{gee_export_dir}/scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved: {scaler_path}")
    
    print("\n" + "=" * 80)
    print("GEE EXPORT COMPLETE")
    print("=" * 80)
    print(f"\nFiles created:")
    print(f"  1. SavedModel: {mlp_export_path}/")
    print(f"  2. Metadata:   {metadata_path}")
    print(f"  3. Scaler:     {scaler_path}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Cohen's Kappa: {result['kappa']:.4f}")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    
    return results, {'CNN': cnn_model, 'MLP': mlp_model, 'Hybrid': hybrid_model}


if __name__ == '__main__':
    results, models = main()