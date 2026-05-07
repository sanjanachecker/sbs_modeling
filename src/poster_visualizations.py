"""
Poster Visualizations — XGB lr01 Top30 Model
=============================================
Generates two publication-quality plots for the poster:
  1. Feature Importance Heatmap (Top 30 features, warm palette: YlOrRd)
  2. LOFO CV Confusion Matrix (warm palette: YlOrRd)

Both use a warm color theme consistent with the fire/burn severity poster.

Color palette:
  - Feature importance heatmap: matplotlib 'YlOrRd'
    (low importance = pale yellow, high importance = deep crimson red)
  - Confusion matrix: matplotlib 'YlOrRd'
    (low count/off-diagonal = pale yellow, high count/diagonal = deep red)

Run:
    python src/poster_visualizations.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH     = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/gee_models_xgb_lr01_top30/xgb_lr01_top30.joblib'
LOFO_CSV_PATH  = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/results_xgb_lr01_top30/lofo_per_fire_results.csv'
OUTPUT_DIR     = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/results_xgb_lr01_top30'

CLASS_NAMES = ['unburned', 'low', 'moderate', 'high']
LABEL_MAP   = {'unburned': 0, 'low': 1, 'moderate': 2, 'high': 3}

# Exact TOP_30 order (matches model training order = importance index order)
TOP_30_FEATURES = [
    'dnbr', 'dndvi', 'dndbi', 'dbsi', 'nbr', 'bsi', 'ndvi', 'ndbi',
    'meanelev_32', 'wc_bio19', 'nirBand', 'wc_bio05', 'rdgh_6', 'blueBand',
    'minelev_4', 'greenBand', 'wc_bio06', 'swir2Band', 'pisrdif_2021-11-22',
    'pisrdif_2021-12-22', 'stddevelev_32', 'maxc_2', 'wc_bio12', 'wc_bio07',
    'dmndwi', 'wc_bio18', 'wc_bio17', 'wc_bio02', 'vd_5', 'planc_32'
]

# Prettier display names for poster labels
FEATURE_DISPLAY_NAMES = {
    'dnbr':                  'dNBR',
    'dndvi':                 'dNDVI',
    'dndbi':                 'dNDBI',
    'dbsi':                  'dBSI',
    'nbr':                   'NBR',
    'bsi':                   'BSI',
    'ndvi':                  'NDVI',
    'ndbi':                  'NDBI',
    'meanelev_32':           'Mean Elev (32-cell)',
    'wc_bio19':              'WorldClim BIO19',
    'nirBand':               'NIR Band',
    'wc_bio05':              'WorldClim BIO05',
    'rdgh_6':                'Ridge Height (order 6)',
    'blueBand':              'Blue Band',
    'minelev_4':             'Min Elev (4-cell)',
    'greenBand':             'Green Band',
    'wc_bio06':              'WorldClim BIO06',
    'swir2Band':             'SWIR2 Band',
    'pisrdif_2021-11-22':    'PISR Diffuse (Nov 22)',
    'pisrdif_2021-12-22':    'PISR Diffuse (Dec 22)',
    'stddevelev_32':         'Std Dev Elev (32-cell)',
    'maxc_2':                'Max Curvature (2-cell)',
    'wc_bio12':              'WorldClim BIO12',
    'wc_bio07':              'WorldClim BIO07',
    'dmndwi':                'dMNDWI',
    'wc_bio18':              'WorldClim BIO18',
    'wc_bio17':              'WorldClim BIO17',
    'wc_bio02':              'WorldClim BIO02',
    'vd_5':                  'Valley Depth (5-cell)',
    'planc_32':              'Plan Curvature (32-cell)',
}

# ── Warm colour theme ──────────────────────────────────────────────────────
CMAP_FEAT  = 'YlOrRd'   # feature importance heatmap  (fire theme: yellow → deep red)
CMAP_CM    = 'YlGn'     # confusion matrix            (agreement theme: yellow → forest green)
# ──────────────────────────────────────────────────────────────────────────


# ============================================================================
# 1. FEATURE IMPORTANCE HEATMAP
# ============================================================================

def plot_feature_importance_heatmap(model, save_path, top_n=30):
    """
    Heatmap of XGBoost 'gain' feature importances for the Top-N model features.
    One column (Gain), one row per feature, sorted descending.
    Warm YlOrRd palette: low importance = pale yellow, high = deep crimson.
    """
    # --- extract importances, sort, slice to top_n ---
    importances = model.feature_importances_   # shape: (30,)
    feat_imp = pd.DataFrame({
        'feature': TOP_30_FEATURES,
        'Gain':    importances,
    }).sort_values('Gain', ascending=False).head(top_n).reset_index(drop=True)

    # Apply display names
    feat_imp['label'] = feat_imp['feature'].map(
        lambda f: FEATURE_DISPLAY_NAMES.get(f, f)
    )

    # Build heatmap matrix: shape (top_n, 1)
    heat_data = feat_imp[['Gain']].set_index(feat_imp['label'])

    fig_height = max(4, top_n * 0.55)   # ~0.55in per row, min 4in
    fig, ax = plt.subplots(figsize=(5, fig_height))
    fig.patch.set_facecolor('#FEFEFE')

    sns.heatmap(
        heat_data,
        ax=ax,
        cmap=CMAP_FEAT,
        annot=True,
        fmt='.4f',
        linewidths=0.4,
        linecolor='#e0d0c0',
        cbar_kws={
            'label': 'Feature Importance (Gain)',
            'shrink': 0.6,
            'pad': 0.02,
        },
        annot_kws={'size': 8, 'weight': 'bold'},
        vmin=0,
        vmax=feat_imp['Gain'].max(),
    )

    ax.set_title(
        'Top 30 Feature Importances\nXGBoost',
        fontsize=13, fontweight='bold', pad=12,
        color='#3d1c02',
    )
    ax.set_xlabel('Importance Metric', fontsize=10, color='#3d1c02')
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=9, colors='#3d1c02')
    ax.tick_params(axis='x', labelsize=9, colors='#3d1c02')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # Style colorbar text
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Feature Importance (Gain)', size=8, color='#3d1c02')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#FEFEFE')
    plt.close()
    print(f"[✓] Feature importance heatmap saved: {save_path}")


# ============================================================================
# 2. CONFUSION MATRIX (warm palette)
# ============================================================================

def plot_confusion_matrix_warm(cm, overall_kappa, save_path):
    """
    Confusion matrix with warm YlOrRd palette.
    Annotated with counts + row-percentage.
    Low count = pale yellow, high count = deep red.
    """
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Build annotation strings
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'

    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.patch.set_facecolor('#FEFEFE')

    sns.heatmap(
        cm,
        annot=annot,
        fmt='',
        cmap=CMAP_CM,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        linewidths=0.5,
        linecolor='#e0d0c0',
        cbar_kws={'label': 'Count', 'shrink': 0.8},
        annot_kws={'size': 11, 'weight': 'bold'},
    )

    ax.set_title(
        f'XGB Top 30 Predictors – LOFO CV Confusion Matrix\n'
        f'Pooled Kappa: {overall_kappa:.4f}',
        fontsize=13, fontweight='bold', pad=12,
        color='#3d1c02',
    )
    ax.set_xlabel('Predicted', fontsize=12, color='#3d1c02', labelpad=8)
    ax.set_ylabel('True', fontsize=12, color='#3d1c02', labelpad=8)
    ax.tick_params(axis='both', labelsize=11, colors='#3d1c02')

    # Style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('Count', size=10, color='#3d1c02')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#FEFEFE')
    plt.close()
    print(f"[✓] Confusion matrix saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"  Model classes: {model.classes_}")
    print(f"  n_features: {model.n_features_in_}")

    # ── 1. Feature importance heatmap ─────────────────────────────────────
    fi_save = os.path.join(OUTPUT_DIR, 'poster_feature_importance_heatmap.png')
    plot_feature_importance_heatmap(model, fi_save, top_n=10)

    # ── 2. Confusion matrix ───────────────────────────────────────────────
    # The pooled LOFO CM values are fixed from the previous run (Kappa=0.4399).
    # These match exactly what was used in the poster.
    # Values from lofo_cv results (rows=true, cols=pred; order: unburned,low,mod,high):
    lofo_kappa = 0.4399
    cm = np.array([
        [ 980, 106,  20,   4],   # unburned
        [ 269, 222, 150,  24],   # low
        [  99, 189, 533, 206],   # moderate
        [  19,  23, 171, 189],   # high
    ])

    cm_save = os.path.join(OUTPUT_DIR, 'poster_confusion_matrix_green.png')
    plot_confusion_matrix_warm(cm, lofo_kappa, cm_save)

    print("\n✅ Done! Both poster plots saved to:", OUTPUT_DIR)
    print("   • poster_feature_importance_heatmap.png")
    print("   • poster_confusion_matrix_green.png")


if __name__ == '__main__':
    main()
