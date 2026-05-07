"""
Multi-Modal Wildfire Burn Severity Classification — 10-Fold Cross Validation
=============================================================================
Trains and evaluates three model architectures using stratified 10-fold CV:
  1. CNN only        (33×33 spatial patches)
  2. MLP only        (30 tabular covariates)
  3. Hybrid CNN+MLP  (fusion of both)

Produces:
  - Per-fold Kappa scores for each model
  - Mean ± std summary statistics
  - Polished comparison figures (confusion matrices, metrics bar charts, box plots)
  - A clean printed summary table at the end
"""

import os
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    cohen_kappa_score, precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

# ── Matplotlib style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linestyle':    '--',
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})

# ── Palette ──────────────────────────────────────────────────────────────────
PALETTE = {
    'CNN':    '#E63946',   # vivid red
    'MLP':    '#457B9D',   # steel blue
    'Hybrid': '#2A9D8F',   # teal
    'bg':     '#F8F9FA',
    'dark':   '#1D3557',
}
CLASS_COLORS = ['#4CAF50', '#8BC34A', '#FF9800', '#F44336']   # unburned→high

# ============================================================================
# CONFIGURATION
# ============================================================================

PATCH_SIZE   = 33
# N_BANDS      = 12   # 6 pre + 6 post surface-reflectance bands
N_BANDS = 16
N_FOLDS      = 10
BATCH_SIZE   = 32
EPOCHS       = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15

TOP_30_FEATURES = [
    'dnbr', 'dndvi', 'dndbi', 'dbsi', 'nbr', 'bsi', 'ndvi', 'ndbi',
    'meanelev_32', 'wc_bio19', 'nirBand', 'wc_bio05', 'rdgh_6', 'blueBand',
    'minelev_4', 'greenBand', 'wc_bio06', 'swir2Band', 'pisrdif_2021-11-22',
    'pisrdif_2021-12-22', 'stddevelev_32', 'maxc_2', 'wc_bio12', 'wc_bio07',
    'dmndwi', 'wc_bio18', 'wc_bio17', 'wc_bio02', 'vd_5', 'planc_32'
]

# BAND_NAMES = [
#     'pre_SR_B2',  'pre_SR_B3',  'pre_SR_B4',  'pre_SR_B5',  'pre_SR_B6',  'pre_SR_B7',
#     'post_SR_B2', 'post_SR_B3', 'post_SR_B4', 'post_SR_B5', 'post_SR_B6', 'post_SR_B7',
# ]

BAND_NAMES = [
    'pre_SR_B2',  'pre_SR_B3',  'pre_SR_B4',  'pre_SR_B5',  'pre_SR_B6',  'pre_SR_B7',
    'post_SR_B2', 'post_SR_B3', 'post_SR_B4', 'post_SR_B5', 'post_SR_B6', 'post_SR_B7',
    'patch_dnbr', 'patch_dndvi', 'patch_nbr', 'patch_ndvi',  # ← new
]

CLASS_MAP   = {'unburned': 0, 'low': 1, 'moderate': 2, 'high': 3}
CLASS_NAMES = ['Unburned', 'Low', 'Moderate', 'High']

# ============================================================================
# DATA LOADING
# ============================================================================

def parse_tfrecord(example_proto):
    feature_description = {
        'SBS':       tf.io.FixedLenFeature([], tf.string),
        'source':    tf.io.FixedLenFeature([], tf.string),
        'Fire_year': tf.io.FixedLenFeature([], tf.string),
    }
    for band in BAND_NAMES:
        feature_description[band] = tf.io.FixedLenFeature([PATCH_SIZE * PATCH_SIZE], tf.float32)
    for feat in TOP_30_FEATURES:
        feature_description[feat] = tf.io.FixedLenFeature([], tf.float32)

    parsed = tf.io.parse_single_example(example_proto, feature_description)

    bands = [tf.reshape(parsed[b], [PATCH_SIZE, PATCH_SIZE]) for b in BAND_NAMES]
    patch = tf.stack(bands, axis=-1)

    covariates = tf.stack([parsed[f] for f in TOP_30_FEATURES])

    return patch, covariates, parsed['SBS'], parsed['Fire_year'], parsed['source']


def load_tfrecords(tfrecord_dir, pattern='patches_cov_*.tfrecord', limit=None):
    paths = sorted(Path(tfrecord_dir).glob(pattern))
    if limit:
        paths = paths[:limit]

    print(f"\n{'='*60}")
    print(f"  LOADING DATA")
    print(f"{'='*60}")
    print(f"  Found {len(paths)} TFRecord file(s)")

    patches, covariates, labels, metadata = [], [], [], []

    for path in paths:
        print(f"  → {path.name}")
        for raw in tf.data.TFRecordDataset(str(path)):
            try:
                patch, cov, label_t, fire_t, src_t = parse_tfrecord(raw)
                label_str = label_t.numpy().decode('utf-8')
                if label_str == 'mod':
                    label_str = 'moderate'
                if label_str not in CLASS_MAP:
                    continue
                patches.append(patch.numpy())
                covariates.append(cov.numpy())
                labels.append(CLASS_MAP[label_str])
                metadata.append({
                    'fire':   fire_t.numpy().decode('utf-8'),
                    'source': src_t.numpy().decode('utf-8'),
                    'label':  label_str,
                })
            except Exception as e:
                print(f"    Warning: {e}")

    labels = np.array(labels)
    print(f"\n  Total samples : {len(labels):,}")
    print(f"  Class distribution:")
    for name, idx in CLASS_MAP.items():
        n = (labels == idx).sum()
        print(f"    {name:12s}: {n:5,}  ({n/len(labels)*100:.1f}%)")

    return np.array(patches), np.array(covariates), labels, pd.DataFrame(metadata)


# ============================================================================
# MODEL FACTORIES  (rebuilt fresh each fold)
# ============================================================================

def build_cnn():
    inp = tf.keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, N_BANDS), name='patch')
    x = inp
    for filters in [32, 64, 128]:
        x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(4, activation='softmax')(x)
    return tf.keras.Model(inp, out, name='CNN')


def build_mlp():
    inp = tf.keras.Input(shape=(30,), name='covariates')
    x = tf.keras.layers.Dense(128, activation='relu')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(4, activation='softmax')(x)
    return tf.keras.Model(inp, out, name='MLP')


def build_hybrid():
    patch_inp = tf.keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, N_BANDS), name='patch')
    cov_inp   = tf.keras.Input(shape=(30,), name='covariates')

    x1 = patch_inp
    for filters in [32, 64, 128]:
        x1 = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.MaxPooling2D(2)(x1)
        x1 = tf.keras.layers.Dropout(0.25)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dense(128, activation='relu')(x1)
    x1 = tf.keras.layers.Dropout(0.4)(x1)

    x2 = tf.keras.layers.Dense(64, activation='relu')(cov_inp)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Dropout(0.3)(x2)
    x2 = tf.keras.layers.Dense(32, activation='relu')(x2)
    x2 = tf.keras.layers.Dropout(0.3)(x2)

    merged = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Dense(128, activation='relu')(merged)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(4, activation='softmax')(x)
    return tf.keras.Model([patch_inp, cov_inp], out, name='Hybrid_CNN_MLP')


BUILDERS = {'CNN': build_cnn, 'MLP': build_mlp, 'Hybrid': build_hybrid}


# ============================================================================
# TRAINING HELPER
# ============================================================================

def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5,
            min_lr=1e-6, verbose=0),
    ]


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )


def get_inputs(name, patches, covariates):
    if name == 'CNN':    return patches
    if name == 'MLP':    return covariates
    if name == 'Hybrid': return [patches, covariates]


# ============================================================================
# 10-FOLD CV
# ============================================================================

def run_cv(X_patches, X_cov, y):
    """
    Runs stratified 10-fold CV for CNN, MLP, and Hybrid.
    Returns a dict of per-fold results for each model.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = list(skf.split(X_patches, y))

    # Storage: {model_name: {kappa, y_true, y_pred, cm_sum}}
    all_results = {name: {
        'kappas':    [],
        'accuracies': [],
        'y_true_all': [],
        'y_pred_all': [],
    } for name in BUILDERS}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'─'*60}")
        print(f"  FOLD {fold_idx+1}/{N_FOLDS}")
        print(f"{'─'*60}")

        # Split raw
        Xp_tr, Xp_val = X_patches[train_idx], X_patches[val_idx]
        Xc_tr, Xc_val = X_cov[train_idx],     X_cov[val_idx]
        y_tr,  y_val  = y[train_idx],          y[val_idx]

        # Normalize patches per fold (fit on train)
        p_mean = Xp_tr.mean(axis=(0,1,2), keepdims=True)
        p_std  = Xp_tr.std(axis=(0,1,2),  keepdims=True) + 1e-8
        Xp_tr_n  = (Xp_tr  - p_mean) / p_std
        Xp_val_n = (Xp_val - p_mean) / p_std

        # Normalize covariates per fold
        scaler = StandardScaler()
        Xc_tr_n  = scaler.fit_transform(Xc_tr)
        Xc_val_n = scaler.transform(Xc_val)

        # Class weights
        cw_arr = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
        cw = dict(enumerate(cw_arr))

        for name, builder in BUILDERS.items():
            print(f"    Training {name}...")
            tf.keras.backend.clear_session()
            model = builder()
            compile_model(model)

            X_tr  = get_inputs(name, Xp_tr_n,  Xc_tr_n)
            X_val = get_inputs(name, Xp_val_n, Xc_val_n)

            model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                class_weight=cw,
                callbacks=get_callbacks(),
                verbose=1,
            )

            y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
            kappa  = cohen_kappa_score(y_val, y_pred)
            acc    = (y_val == y_pred).mean()

            all_results[name]['kappas'].append(kappa)
            all_results[name]['accuracies'].append(acc)
            all_results[name]['y_true_all'].extend(y_val.tolist())
            all_results[name]['y_pred_all'].extend(y_pred.tolist())

            print(f"      Kappa = {kappa:.4f}  |  Accuracy = {acc:.4f}")

    # Convert lists → arrays
    for name in all_results:
        all_results[name]['kappas']     = np.array(all_results[name]['kappas'])
        all_results[name]['accuracies'] = np.array(all_results[name]['accuracies'])
        all_results[name]['y_true_all'] = np.array(all_results[name]['y_true_all'])
        all_results[name]['y_pred_all'] = np.array(all_results[name]['y_pred_all'])

    return all_results


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def print_summary(all_results):
    sep = '=' * 72
    print(f"\n{sep}")
    print(f"  10-FOLD CROSS-VALIDATION SUMMARY")
    print(sep)

    header = f"  {'Model':<12}  {'Mean κ':>8}  {'Std κ':>7}  {'Min κ':>7}  {'Max κ':>7}  {'Mean Acc':>9}  {'Std Acc':>8}"
    print(header)
    print(f"  {'─'*68}")

    for name in ['CNN', 'MLP', 'Hybrid']:
        r = all_results[name]
        k = r['kappas']
        a = r['accuracies']
        print(f"  {name:<12}  {k.mean():>8.4f}  {k.std():>7.4f}  {k.min():>7.4f}  {k.max():>7.4f}  {a.mean():>9.4f}  {a.std():>8.4f}")

    print(f"\n  Per-class F1 (aggregated across all folds):")
    print(f"  {'Model':<12}  {'Unburned':>9}  {'Low':>7}  {'Moderate':>9}  {'High':>7}  {'Macro F1':>9}")
    print(f"  {'─'*68}")

    for name in ['CNN', 'MLP', 'Hybrid']:
        r = all_results[name]
        _, _, f1, _ = precision_recall_fscore_support(
            r['y_true_all'], r['y_pred_all'], labels=[0,1,2,3], zero_division=0)
        macro_f1 = f1.mean()
        print(f"  {name:<12}  {f1[0]:>9.4f}  {f1[1]:>7.4f}  {f1[2]:>9.4f}  {f1[3]:>7.4f}  {macro_f1:>9.4f}")

    print(sep)

    # Per-fold detail
    print(f"\n  Per-fold Kappa breakdown:")
    header2 = f"  {'Fold':<6}" + "".join(f"  {n:>8}" for n in ['CNN','MLP','Hybrid'])
    print(header2)
    print(f"  {'─'*36}")
    for i in range(N_FOLDS):
        row = f"  {i+1:<6}"
        for name in ['CNN', 'MLP', 'Hybrid']:
            row += f"  {all_results[name]['kappas'][i]:>8.4f}"
        print(row)
    print(f"  {'─'*36}")
    mean_row = f"  {'Mean':<6}"
    for name in ['CNN', 'MLP', 'Hybrid']:
        mean_row += f"  {all_results[name]['kappas'].mean():>8.4f}"
    print(mean_row)
    print(sep)


# ============================================================================
# VISUALIZATIONS
# ============================================================================

# ── helper ───────────────────────────────────────────────────────────────────
def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


# ── 1. Box-plot + scatter of per-fold Kappa ──────────────────────────────────
def plot_kappa_boxplot(all_results, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])

    names  = ['CNN', 'MLP', 'Hybrid']
    colors = [PALETTE[n] for n in names]
    data   = [all_results[n]['kappas'] for n in names]

    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    medianprops=dict(color='white', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markersize=5, alpha=0.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    # Overlay individual fold dots with jitter
    rng = np.random.default_rng(0)
    for i, (d, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(d))
        ax.scatter(i + jitter, d, color=color, edgecolors='white',
                   linewidths=0.8, s=60, zorder=5, alpha=0.9)

    # Mean labels
    for i, (d, color) in enumerate(zip(data, colors), start=1):
        ax.text(i, d.mean(), f'μ={d.mean():.3f}',
                ha='center', va='bottom', fontsize=9.5,
                fontweight='bold', color=PALETTE['dark'],
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.5))

    ax.set_xticks([1,2,3])
    ax.set_xticklabels(names, fontsize=13, fontweight='bold')
    ax.set_ylabel("Cohen's Kappa  (per fold)", fontsize=12)
    ax.set_title("10-Fold CV Kappa — Model Comparison", fontsize=14,
                 fontweight='bold', color=PALETTE['dark'], pad=14)
    ax.set_xlim(0.4, 3.6)

    # Interpretation bands
    for yval, label, alpha in [(0.6, 'Substantial', 0.07), (0.4, 'Moderate', 0.07)]:
        ax.axhline(yval, color='#555', linestyle=':', linewidth=1.2, alpha=0.7)
        ax.text(3.62, yval, label, va='center', fontsize=8, color='#555', alpha=0.9)

    save(fig, f"{out_dir}/01_kappa_boxplot.png")


# ── 2. Per-fold line chart ────────────────────────────────────────────────────
def plot_kappa_per_fold(all_results, out_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])

    names = ['CNN', 'MLP', 'Hybrid']
    folds = np.arange(1, N_FOLDS+1)

    for name in names:
        k = all_results[name]['kappas']
        ax.plot(folds, k, marker='o', linewidth=2.2, markersize=7,
                label=f"{name}  (μ={k.mean():.3f})", color=PALETTE[name])
        ax.fill_between(folds, k - k.std(), k + k.std(),
                        alpha=0.10, color=PALETTE[name])

    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("Cohen's Kappa", fontsize=12)
    ax.set_title("Per-Fold Kappa — 10-Fold Stratified CV", fontsize=14,
                 fontweight='bold', color=PALETTE['dark'])
    ax.set_xticks(folds)
    ax.legend(fontsize=10, framealpha=0.85)

    save(fig, f"{out_dir}/02_kappa_per_fold.png")


# ── 3. Aggregated confusion matrices (side-by-side) ─────────────────────────
def plot_confusion_matrices(all_results, out_dir):
    names = ['CNN', 'MLP', 'Hybrid']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor(PALETTE['bg'])
    fig.suptitle("Aggregated Confusion Matrices  (all 10 folds)",
                 fontsize=15, fontweight='bold', color=PALETTE['dark'], y=1.01)

    for ax, name in zip(axes, names):
        ax.set_facecolor(PALETTE['bg'])
        r  = all_results[name]
        cm = confusion_matrix(r['y_true_all'], r['y_pred_all'], labels=[0,1,2,3])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        annot = np.empty_like(cm, dtype=object)
        for i in range(4):
            for j in range(4):
                annot[i,j] = f"{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)"

        kappa_mean = r['kappas'].mean()
        kappa_std  = r['kappas'].std()

        sns.heatmap(cm_pct, annot=annot, fmt='', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, linewidths=0.5, linecolor='#ddd',
                    cbar_kws={'label': '% of true class', 'shrink': 0.8},
                    vmin=0, vmax=100)

        ax.set_title(f"{name}\nκ = {kappa_mean:.3f} ± {kappa_std:.3f}",
                     fontsize=12, fontweight='bold', color=PALETTE[name])
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)

    plt.tight_layout()
    save(fig, f"{out_dir}/03_confusion_matrices.png")


# ── 4. Per-class F1 grouped bar ───────────────────────────────────────────────
def plot_per_class_metrics(all_results, out_dir):
    names = ['CNN', 'MLP', 'Hybrid']

    metrics = {}
    for name in names:
        r = all_results[name]
        p, rc, f1, _ = precision_recall_fscore_support(
            r['y_true_all'], r['y_pred_all'], labels=[0,1,2,3], zero_division=0)
        metrics[name] = {'precision': p, 'recall': rc, 'f1': f1}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    fig.patch.set_facecolor(PALETTE['bg'])
    fig.suptitle("Per-Class Metrics  (aggregated across all folds)",
                 fontsize=14, fontweight='bold', color=PALETTE['dark'])

    x = np.arange(4)
    w = 0.25

    for ax, metric_key, title in zip(
        axes,
        ['precision', 'recall', 'f1'],
        ['Precision', 'Recall', 'F1-Score']
    ):
        ax.set_facecolor(PALETTE['bg'])
        for i, name in enumerate(names):
            vals = metrics[name][metric_key]
            bars = ax.bar(x + (i-1)*w, vals, w, label=name,
                          color=PALETTE[name], alpha=0.85, edgecolor='white', linewidth=0.6)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold', color=PALETTE['dark'])
        ax.set_ylim(0, 1.12)
        if ax == axes[0]:
            ax.set_ylabel("Score", fontsize=11)
        ax.legend(fontsize=9, framealpha=0.7)

    plt.tight_layout()
    save(fig, f"{out_dir}/04_per_class_metrics.png")


# ── 5. Summary bar: mean Kappa + Accuracy ────────────────────────────────────
def plot_summary_bar(all_results, out_dir):
    names = ['CNN', 'MLP', 'Hybrid']
    mean_kappas = [all_results[n]['kappas'].mean()     for n in names]
    std_kappas  = [all_results[n]['kappas'].std()      for n in names]
    mean_accs   = [all_results[n]['accuracies'].mean() for n in names]
    std_accs    = [all_results[n]['accuracies'].std()  for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(PALETTE['bg'])
    fig.suptitle("10-Fold CV Summary — Mean ± Std",
                 fontsize=14, fontweight='bold', color=PALETTE['dark'])

    for ax, vals, stds, ylabel, title in zip(
        axes,
        [mean_kappas, mean_accs],
        [std_kappas,  std_accs],
        ["Cohen's Kappa", "Accuracy"],
        ["Mean Cohen's Kappa (10-fold)", "Mean Accuracy (10-fold)"]
    ):
        ax.set_facecolor(PALETTE['bg'])
        colors = [PALETTE[n] for n in names]
        bars = ax.bar(names, vals, yerr=stds, capsize=7,
                      color=colors, alpha=0.85, edgecolor='white',
                      linewidth=1, error_kw=dict(elinewidth=2, ecolor='#333'))

        for bar, val, std in zip(bars, vals, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + std + 0.008,
                    f'{val:.3f}\n±{std:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color=PALETTE['dark'])

        ax.set_ylim(0, min(1.0, max(vals) + max(stds) + 0.18))
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', color=PALETTE['dark'])
        ax.set_xticklabels(names, fontsize=12)

    plt.tight_layout()
    save(fig, f"{out_dir}/05_summary_bar.png")


# ── 6. Kappa distribution violin ─────────────────────────────────────────────
def plot_kappa_violin(all_results, out_dir):
    names = ['CNN', 'MLP', 'Hybrid']
    data  = [all_results[n]['kappas'] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])

    parts = ax.violinplot(data, positions=[1,2,3], widths=0.55,
                          showmeans=True, showmedians=False)

    for i, (pc, name) in enumerate(zip(parts['bodies'], names)):
        pc.set_facecolor(PALETTE[name])
        pc.set_alpha(0.75)
        pc.set_edgecolor(PALETTE['dark'])

    for part in ['cmeans', 'cbars', 'cmins', 'cmaxes']:
        if part in parts:
            parts[part].set_color(PALETTE['dark'])
            parts[part].set_linewidth(1.8)

    rng = np.random.default_rng(1)
    for i, (d, name) in enumerate(zip(data, names), start=1):
        jitter = rng.uniform(-0.06, 0.06, size=len(d))
        ax.scatter(i + jitter, d, color=PALETTE[name],
                   edgecolors='white', linewidths=0.8, s=65, zorder=5)

    ax.set_xticks([1,2,3])
    ax.set_xticklabels(names, fontsize=13, fontweight='bold')
    ax.set_ylabel("Cohen's Kappa", fontsize=12)
    ax.set_title("Kappa Distribution Across 10 Folds", fontsize=14,
                 fontweight='bold', color=PALETTE['dark'])

    save(fig, f"{out_dir}/06_kappa_violin.png")


# ============================================================================
# SAVE RESULTS JSON
# ============================================================================

def save_results_json(all_results, out_dir):
    out = {}
    for name in all_results:
        r = all_results[name]
        p, rc, f1, sup = precision_recall_fscore_support(
            r['y_true_all'], r['y_pred_all'], labels=[0,1,2,3], zero_division=0)
        out[name] = {
            'kappa_per_fold':    r['kappas'].tolist(),
            'kappa_mean':        float(r['kappas'].mean()),
            'kappa_std':         float(r['kappas'].std()),
            'kappa_min':         float(r['kappas'].min()),
            'kappa_max':         float(r['kappas'].max()),
            'accuracy_per_fold': r['accuracies'].tolist(),
            'accuracy_mean':     float(r['accuracies'].mean()),
            'accuracy_std':      float(r['accuracies'].std()),
            'per_class': {
                CLASS_NAMES[i]: {
                    'precision': float(p[i]),
                    'recall':    float(rc[i]),
                    'f1':        float(f1[i]),
                    'support':   int(sup[i]),
                }
                for i in range(4)
            },
            'macro_f1': float(f1.mean()),
        }

    path = f"{out_dir}/cv_results.json"
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"  Saved → {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # ── paths ────────────────────────────────────────────────────────────────
    TFRECORD_DIR = '/Users/sanjanachecker/csc/fire_patches5'
    OUT_DIR      = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/results_cv10'
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MULTI-MODAL BURN SEVERITY  —  10-Fold CV")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # ── load ─────────────────────────────────────────────────────────────────
    X_patches, X_cov, y, meta = load_tfrecords(TFRECORD_DIR)

    # ── cross-validation ─────────────────────────────────────────────────────
    all_results = run_cv(X_patches, X_cov, y)

    # ── summary to console ───────────────────────────────────────────────────
    print_summary(all_results)

    # ── figures ──────────────────────────────────────────────────────────────
    print(f"\n  Generating figures → {OUT_DIR}/")
    plot_kappa_boxplot(all_results, OUT_DIR)
    plot_kappa_per_fold(all_results, OUT_DIR)
    plot_confusion_matrices(all_results, OUT_DIR)
    plot_per_class_metrics(all_results, OUT_DIR)
    plot_summary_bar(all_results, OUT_DIR)
    plot_kappa_violin(all_results, OUT_DIR)

    # ── JSON dump ────────────────────────────────────────────────────────────
    save_results_json(all_results, OUT_DIR)

    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  All outputs in: {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()