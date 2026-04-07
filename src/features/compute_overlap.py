"""
Moderate vs High Feature Distribution Analysis
================================================
Plots the distribution of each Top30 feature separately for moderate
and high severity samples, to see which features (if any) cleanly
separate the two classes.

Also computes:
- Overlap coefficient for each feature (lower = better separation)
- KS statistic (higher = more different distributions)
- Ranked summary table of most discriminative features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.spatial.distance import jensenshannon
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

MAIN_CSV = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_complete_covariates_fixed_1229.csv'
OLD_UPSAMPLED_CSV = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/data/real_all_fires_upsampled_points_with_covariates_fixed.csv'
OUTPUT_DIR = '/Users/sanjanachecker/csc/masters/sbs/sbs_modeling/results_mod_vs_high'

CLASS_NAMES = ['unburned', 'low', 'moderate', 'high']

TOP_30_FEATURES = [
    'dnbr', 'dndvi', 'dndbi', 'dbsi', 'nbr', 'bsi', 'ndvi', 'ndbi',
    'meanelev_32', 'wc_bio19', 'nirBand', 'wc_bio05', 'rdgh_6', 'blueBand',
    'minelev_4', 'greenBand', 'wc_bio06', 'swir2Band', 'pisrdif_2021-11-22',
    'pisrdif_2021-12-22', 'stddevelev_32', 'maxc_2', 'wc_bio12', 'wc_bio07',
    'dmndwi', 'wc_bio18', 'wc_bio17', 'wc_bio02', 'vd_5', 'planc_32'
]

# ============================================================================
# LOAD DATA
# ============================================================================

def load_data():
    df_main = pd.read_csv(MAIN_CSV)
    df_main['SBS'] = df_main['SBS'].replace({'mod': 'moderate'})
    df_main = df_main[df_main['SBS'].isin(CLASS_NAMES)].copy()

    df_up = pd.read_csv(OLD_UPSAMPLED_CSV)
    df_up['SBS'] = df_up['SBS'].replace({'mod': 'moderate'})
    df_up = df_up[df_up['SBS'].isin(CLASS_NAMES)].copy()

    common_cols = list(set(df_main.columns) & set(df_up.columns))
    df = pd.concat([df_main[common_cols], df_up[common_cols]], ignore_index=True)
    df['SBS'] = df['SBS'].replace({'mod': 'moderate'})
    df = df[df['SBS'].isin(CLASS_NAMES)].copy()

    print(f"Combined: {len(df)} rows")
    for cls in CLASS_NAMES:
        count = (df['SBS'] == cls).sum()
        print(f"  {cls:12s}: {count:5d} ({count/len(df)*100:.1f}%)")

    return df


# ============================================================================
# STATISTICS
# ============================================================================

def compute_separation_stats(mod_vals, high_vals):
    """
    Compute statistics measuring how well a feature separates moderate vs high.
    Returns dict of stats — higher KS / lower overlap = better separation.
    """
    # Remove NaN
    mod_vals = mod_vals[~np.isnan(mod_vals)]
    high_vals = high_vals[~np.isnan(high_vals)]

    # KS test — p-value and statistic
    ks_stat, ks_p = stats.ks_2samp(mod_vals, high_vals)

    # Cohen's d (effect size)
    pooled_std = np.sqrt((mod_vals.std()**2 + high_vals.std()**2) / 2)
    cohens_d = abs(mod_vals.mean() - high_vals.mean()) / (pooled_std + 1e-10)

    # Overlap coefficient using histograms
    bins = np.linspace(
        min(mod_vals.min(), high_vals.min()),
        max(mod_vals.max(), high_vals.max()),
        50
    )
    mod_hist, _ = np.histogram(mod_vals, bins=bins, density=True)
    high_hist, _ = np.histogram(high_vals, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    overlap = np.sum(np.minimum(mod_hist, high_hist)) * bin_width

    # Jensen-Shannon divergence (0 = identical, 1 = completely different)
    mod_hist_norm = mod_hist + 1e-10
    high_hist_norm = high_hist + 1e-10
    mod_hist_norm /= mod_hist_norm.sum()
    high_hist_norm /= high_hist_norm.sum()
    js_div = jensenshannon(mod_hist_norm, high_hist_norm)

    return {
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'cohens_d': cohens_d,
        'overlap': overlap,
        'js_divergence': js_div,
        'mod_mean': mod_vals.mean(),
        'high_mean': high_vals.mean(),
        'mod_std': mod_vals.std(),
        'high_std': high_vals.std(),
    }


# ============================================================================
# PLOTS
# ============================================================================

def plot_feature_distributions(df, features, output_dir):
    """
    For each feature, plot KDE + histogram for moderate vs high side by side.
    Color: moderate = orange, high = red.
    """
    os.makedirs(output_dir, exist_ok=True)

    mod = df[df['SBS'] == 'moderate']
    high = df[df['SBS'] == 'high']

    available = [f for f in features if f in df.columns]
    print(f"\nPlotting {len(available)} features...")

    stats_rows = []

    # ---- Big grid plot: all 30 features in one figure ----
    n_cols = 5
    n_rows = int(np.ceil(len(available) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()

    for idx, feat in enumerate(available):
        ax = axes[idx]

        mod_vals = mod[feat].dropna().values.astype(np.float32)
        high_vals = high[feat].dropna().values.astype(np.float32)

        # Clip to 1st-99th percentile to avoid extreme outliers squashing plot
        lo = np.percentile(np.concatenate([mod_vals, high_vals]), 1)
        hi = np.percentile(np.concatenate([mod_vals, high_vals]), 99)
        mod_clip = mod_vals[(mod_vals >= lo) & (mod_vals <= hi)]
        high_clip = high_vals[(high_vals >= lo) & (high_vals <= hi)]

        sep = compute_separation_stats(mod_vals, high_vals)
        stats_rows.append({'feature': feat, **sep})

        # KDE plots
        try:
            kde_mod = stats.gaussian_kde(mod_clip)
            kde_high = stats.gaussian_kde(high_clip)
            x = np.linspace(lo, hi, 200)
            ax.fill_between(x, kde_mod(x), alpha=0.4, color='#f39c12', label='Moderate')
            ax.fill_between(x, kde_high(x), alpha=0.4, color='#e74c3c', label='High')
            ax.plot(x, kde_mod(x), color='#f39c12', linewidth=1.5)
            ax.plot(x, kde_high(x), color='#e74c3c', linewidth=1.5)
        except Exception:
            ax.hist(mod_clip, bins=30, alpha=0.5, color='#f39c12',
                    density=True, label='Moderate')
            ax.hist(high_clip, bins=30, alpha=0.5, color='#e74c3c',
                    density=True, label='High')

        ax.set_title(
            f"{feat}\nKS={sep['ks_stat']:.2f}  d={sep['cohens_d']:.2f}  "
            f"overlap={sep['overlap']:.2f}",
            fontsize=8, fontweight='bold'
        )
        ax.set_xlabel('')
        ax.set_yticks([])
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        'Moderate vs High Severity — Feature Distributions (Top30)\n'
        'KS = separation strength, d = Cohen\'s d effect size, overlap = distribution overlap (lower = better)',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mod_vs_high_all_features.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/mod_vs_high_all_features.png")

    return pd.DataFrame(stats_rows)


def plot_top_separating_features(df, stats_df, output_dir, top_n=10):
    """
    Individual detailed plots for the top N most separating features.
    Ranked by KS statistic.
    """
    top_feats = stats_df.nlargest(top_n, 'ks_stat')['feature'].tolist()

    mod = df[df['SBS'] == 'moderate']
    high = df[df['SBS'] == 'high']
    all4 = df[df['SBS'].isin(CLASS_NAMES)]  # all classes for context

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    colors = {
        'unburned': '#2ecc71',
        'low': '#3498db',
        'moderate': '#f39c12',
        'high': '#e74c3c'
    }

    for idx, feat in enumerate(top_feats):
        ax = axes[idx]
        sep = stats_df[stats_df['feature'] == feat].iloc[0]

        # Show all 4 classes for context, but emphasize moderate vs high
        for cls in CLASS_NAMES:
            vals = all4[all4['SBS'] == cls][feat].dropna().values.astype(np.float32)
            lo = np.percentile(vals, 1)
            hi = np.percentile(vals, 99)
            vals_clip = vals[(vals >= lo) & (vals <= hi)]
            alpha = 0.7 if cls in ['moderate', 'high'] else 0.25
            lw = 2.0 if cls in ['moderate', 'high'] else 1.0
            try:
                kde = stats.gaussian_kde(vals_clip)
                x = np.linspace(vals_clip.min(), vals_clip.max(), 200)
                ax.fill_between(x, kde(x), alpha=alpha * 0.5,
                                color=colors[cls])
                ax.plot(x, kde(x), color=colors[cls], linewidth=lw,
                        label=cls)
            except Exception:
                pass

        ax.set_title(
            f"{feat}\nKS={sep['ks_stat']:.3f}  Cohen's d={sep['cohens_d']:.3f}",
            fontsize=9, fontweight='bold'
        )
        ax.set_yticks([])
        ax.legend(fontsize=6, loc='upper right')

    fig.suptitle(
        f'Top {top_n} Features by Moderate/High Separation (KS Statistic)\n'
        'All 4 classes shown — focus on orange (moderate) vs red (high)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top{top_n}_separating_features.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/top{top_n}_separating_features.png")


def plot_dnbr_focus(df, output_dir):
    """
    Focused plot on dNBR specifically — the most important predictor.
    Shows all 4 classes + moderate/high overlap region.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        'unburned': '#2ecc71',
        'low': '#3498db',
        'moderate': '#f39c12',
        'high': '#e74c3c'
    }

    # Left: all 4 classes
    ax = axes[0]
    for cls in CLASS_NAMES:
        vals = df[df['SBS'] == cls]['dnbr'].dropna().values.astype(np.float32)
        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
        vals_clip = vals[(vals >= lo) & (vals <= hi)]
        try:
            kde = stats.gaussian_kde(vals_clip)
            x = np.linspace(vals_clip.min(), vals_clip.max(), 300)
            ax.fill_between(x, kde(x), alpha=0.3, color=colors[cls])
            ax.plot(x, kde(x), color=colors[cls], linewidth=2, label=cls)
        except Exception:
            pass
    ax.set_title('dNBR — All 4 Classes', fontsize=12, fontweight='bold')
    ax.set_xlabel('dNBR value')
    ax.set_yticks([])
    ax.legend()

    # Right: moderate vs high only, zoomed in
    ax = axes[1]
    mod_vals = df[df['SBS'] == 'moderate']['dnbr'].dropna().values.astype(np.float32)
    high_vals = df[df['SBS'] == 'high']['dnbr'].dropna().values.astype(np.float32)

    sep = compute_separation_stats(mod_vals, high_vals)

    lo = np.percentile(np.concatenate([mod_vals, high_vals]), 1)
    hi = np.percentile(np.concatenate([mod_vals, high_vals]), 99)
    mod_clip = mod_vals[(mod_vals >= lo) & (mod_vals <= hi)]
    high_clip = high_vals[(high_vals >= lo) & (high_vals <= hi)]

    try:
        kde_mod = stats.gaussian_kde(mod_clip)
        kde_high = stats.gaussian_kde(high_clip)
        x = np.linspace(lo, hi, 300)
        ax.fill_between(x, kde_mod(x), alpha=0.4, color='#f39c12', label='Moderate')
        ax.fill_between(x, kde_high(x), alpha=0.4, color='#e74c3c', label='High')
        ax.plot(x, kde_mod(x), color='#f39c12', linewidth=2)
        ax.plot(x, kde_high(x), color='#e74c3c', linewidth=2)

        # Shade overlap region
        overlap_y = np.minimum(kde_mod(x), kde_high(x))
        ax.fill_between(x, overlap_y, alpha=0.6, color='gray',
                        label=f'Overlap region')
    except Exception:
        pass

    ax.set_title(
        f'dNBR — Moderate vs High (zoomed)\n'
        f'KS={sep["ks_stat"]:.3f}  Cohen\'s d={sep["cohens_d"]:.3f}  '
        f'Overlap={sep["overlap"]:.3f}',
        fontsize=11, fontweight='bold'
    )
    ax.set_xlabel('dNBR value')
    ax.set_yticks([])
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/dnbr_focus.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/dnbr_focus.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("MODERATE vs HIGH — FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()

    # Compute stats and generate all-features grid
    all_features = [c for c in df.select_dtypes(include=[np.number]).columns
                if c.lower() not in {x.lower() for x in 
                ['SBS', 'Fire_year', 'fire', 'source', 'data_source', 'Source',
                 'PointX', 'PointY', '.geo', 'system:index', 'label',
                 'latitude', 'longitude', 'lat', 'lon', 'x', 'y']}]
    stats_df = plot_feature_distributions(df, all_features, OUTPUT_DIR)

    # Sort by KS statistic (best separation first)
    stats_df = stats_df.sort_values('ks_stat', ascending=False)

    # Print ranked summary
    print("\n" + "=" * 70)
    print("FEATURE SEPARATION RANKING (Moderate vs High)")
    print("Higher KS = more different distributions = easier to separate")
    print("Lower overlap = less overlap between classes")
    print("=" * 70)
    print(f"\n{'Rank':<5} {'Feature':<25} {'KS stat':>8} {'Cohen d':>8} "
          f"{'Overlap':>8} {'JS div':>8} {'Mod mean':>10} {'High mean':>10}")
    print("-" * 90)
    for i, (_, row) in enumerate(stats_df.iterrows()):
        sig = "***" if row['ks_p'] < 0.001 else "**" if row['ks_p'] < 0.01 else "*" if row['ks_p'] < 0.05 else ""
        print(f"{i+1:<5} {row['feature']:<25} {row['ks_stat']:>8.3f} "
              f"{row['cohens_d']:>8.3f} {row['overlap']:>8.3f} "
              f"{row['js_divergence']:>8.3f} {row['mod_mean']:>10.3f} "
              f"{row['high_mean']:>10.3f} {sig}")

    # Save stats table
    stats_df.to_csv(f'{OUTPUT_DIR}/mod_vs_high_separation_stats.csv', index=False)
    print(f"\nStats saved: {OUTPUT_DIR}/mod_vs_high_separation_stats.csv")

    # Top separating features detailed plot
    plot_top_separating_features(df, stats_df, OUTPUT_DIR, top_n=10)

    # dNBR focused plot
    if 'dnbr' in df.columns:
        plot_dnbr_focus(df, OUTPUT_DIR)

    # Summary interpretation
    top3 = stats_df.head(3)['feature'].tolist()
    worst3 = stats_df.tail(3)['feature'].tolist()
    mean_overlap = stats_df['overlap'].mean()
    mean_ks = stats_df['ks_stat'].mean()

    print(f"\n{'=' * 70}")
    print("INTERPRETATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nMean overlap across all features:  {mean_overlap:.3f}")
    print(f"Mean KS statistic:                 {mean_ks:.3f}")
    print(f"\nBest separating features:  {top3}")
    print(f"Worst separating features: {worst3}")

    if mean_overlap > 0.5:
        print(f"\n>>> HIGH OVERLAP ({mean_overlap:.2f}) across most features.")
        print(">>> This supports the hypothesis that moderate/high are inherently")
        print(">>> difficult to separate from spectral data alone — a data/labeling")
        print(">>> issue rather than a modeling issue.")
    else:
        print(f"\n>>> Moderate overlap ({mean_overlap:.2f}) — some features do")
        print(">>> show meaningful separation. Focus modeling on these features.")

    return stats_df


if __name__ == '__main__':
    stats_df = main()