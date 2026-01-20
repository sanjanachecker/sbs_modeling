"""
TFRecord Dataset Verification and Summary
==========================================
Analyzes all exported TFRecords to verify structure and provide statistics
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def verify_tfrecord_structure(tfrecord_path):
    """Check if TFRecord has correct structure"""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        features = example.features.feature
        
        # Check for required fields
        required = ['SBS', 'Fire_year', 'source']
        patches = ['pre_SR_B4', 'post_SR_B4']  # Sample patch bands
        covariates = ['dnbr', 'ndvi', 'nbr']  # Sample covariates
        
        has_required = all(f in features for f in required)
        has_patches = all(f in features for f in patches)
        has_covariates = all(f in features for f in covariates)
        
        return {
            'has_required': has_required,
            'has_patches': has_patches,
            'has_covariates': has_covariates,
            'total_features': len(features),
            'feature_names': list(features.keys())
        }


def analyze_tfrecord(tfrecord_path):
    """Analyze single TFRecord file"""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    sbs_list = []
    source_list = []
    fire_list = []
    
    count = 0
    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        sbs = example.features.feature['SBS'].bytes_list.value[0].decode('utf-8')
        source = example.features.feature['source'].bytes_list.value[0].decode('utf-8')
        fire = example.features.feature['Fire_year'].bytes_list.value[0].decode('utf-8')
        
        sbs_list.append(sbs)
        source_list.append(source)
        fire_list.append(fire)
        count += 1
    
    return {
        'file': tfrecord_path.name,
        'total_samples': count,
        'sbs_dist': Counter(sbs_list),
        'source_dist': Counter(source_list),
        'fires': set(fire_list)
    }


def analyze_all_tfrecords(tfrecord_dir, pattern='patches_*.tfrecord'):
    """Analyze all TFRecords in directory"""
    tfrecord_paths = sorted(Path(tfrecord_dir).glob(pattern))
    
    print("=" * 80)
    print("TFRECORD DATASET ANALYSIS")
    print("=" * 80)
    print(f"\nDirectory: {tfrecord_dir}")
    print(f"Pattern: {pattern}")
    print(f"Found {len(tfrecord_paths)} TFRecord files\n")
    
    # Verify first file structure
    if tfrecord_paths:
        print("=" * 80)
        print("STRUCTURE VERIFICATION (First File)")
        print("=" * 80)
        structure = verify_tfrecord_structure(tfrecord_paths[0])
        print(f"File: {tfrecord_paths[0].name}")
        print(f"  Required fields: {'✅' if structure['has_required'] else '❌'}")
        print(f"  Patches: {'✅' if structure['has_patches'] else '❌'}")
        print(f"  Covariates: {'✅' if structure['has_covariates'] else '❌'}")
        print(f"  Total features: {structure['total_features']}")
    
    # Analyze all files
    print("\n" + "=" * 80)
    print("PER-FILE ANALYSIS")
    print("=" * 80)
    
    all_results = []
    total_samples = 0
    all_sbs = Counter()
    all_sources = Counter()
    all_fires = set()
    
    for path in tfrecord_paths:
        print(f"\nAnalyzing: {path.name}")
        result = analyze_tfrecord(path)
        all_results.append(result)
        
        total_samples += result['total_samples']
        all_sbs.update(result['sbs_dist'])
        all_sources.update(result['source_dist'])
        all_fires.update(result['fires'])
        
        print(f"  Samples: {result['total_samples']}")
        print(f"  SBS: {dict(result['sbs_dist'])}")
        print(f"  Sources: {dict(result['source_dist'])}")
    
    # Summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"\nTotal files: {len(tfrecord_paths)}")
    print(f"Total samples: {total_samples}")
    print(f"Total unique fires: {len(all_fires)}")
    
    print(f"\nOverall SBS distribution:")
    for sbs, count in sorted(all_sbs.items()):
        pct = count / total_samples * 100
        print(f"  {sbs:12s}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\nOverall source distribution:")
    for source, count in sorted(all_sources.items()):
        pct = count / total_samples * 100
        print(f"  {source:20s}: {count:5d} ({pct:5.1f}%)")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([
        {
            'fire': r['file'].replace('patches_', '').replace('.tfrecord', ''),
            'total': r['total_samples'],
            'high': r['sbs_dist'].get('high', 0),
            'low': r['sbs_dist'].get('low', 0),
            'moderate': r['sbs_dist'].get('moderate', 0),
            'unburned': r['sbs_dist'].get('unburned', 0),
            'original': r['source_dist'].get('original', 0),
            'upsampled': r['source_dist'].get('upsampled_buffer', 0)
        }
        for r in all_results
    ])
    
    # Save summary
    output_path = Path(tfrecord_dir) / 'dataset_summary.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"\nSaved summary to: {output_path}")
    
    # Visualizations
    create_visualizations(summary_df, all_sbs, all_sources, tfrecord_dir)
    
    return summary_df, all_sbs, all_sources


def create_visualizations(summary_df, all_sbs, all_sources, output_dir):
    """Create visualization charts"""
    
    # 1. Overall class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sbs_counts = pd.Series(dict(all_sbs))
    sbs_counts = sbs_counts.reindex(['unburned', 'low', 'moderate', 'high'])
    colors_sbs = ['#2ecc71', '#f39c12', '#3498db', '#e74c3c']
    
    axes[0].bar(sbs_counts.index, sbs_counts.values, color=colors_sbs, edgecolor='black')
    axes[0].set_title('Overall Burn Severity Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Samples')
    for i, (idx, val) in enumerate(sbs_counts.items()):
        axes[0].text(i, val, f'{val}\n({val/sbs_counts.sum()*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
    
    source_counts = pd.Series(dict(all_sources))
    axes[1].bar(source_counts.index, source_counts.values, color=['#3498db', '#e74c3c'], 
                edgecolor='black')
    axes[1].set_title('Data Source Distribution', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Number of Samples')
    for i, (idx, val) in enumerate(source_counts.items()):
        axes[1].text(i, val, f'{val}\n({val/source_counts.sum()*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/dataset_overview.png")
    
    # 2. Per-fire breakdown
    if len(summary_df) > 5:
        fig, ax = plt.subplots(figsize=(max(12, len(summary_df) * 0.4), 6))
        
        x = np.arange(len(summary_df))
        width = 0.2
        
        ax.bar(x - 1.5*width, summary_df['high'], width, label='High', color='#e74c3c')
        ax.bar(x - 0.5*width, summary_df['moderate'], width, label='Moderate', color='#3498db')
        ax.bar(x + 0.5*width, summary_df['low'], width, label='Low', color='#f39c12')
        ax.bar(x + 1.5*width, summary_df['unburned'], width, label='Unburned', color='#2ecc71')
        
        ax.set_xlabel('Fire')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Per-Fire Class Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df['fire'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/per_fire_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/per_fire_distribution.png")


if __name__ == '__main__':
    # Adjust this path to your TFRecord directory
    tfrecord_dir = '/Users/sanjanachecker/Downloads/fire_patches2'
    
    summary_df, sbs_dist, source_dist = analyze_all_tfrecords(tfrecord_dir)
    
    print("\n" + "=" * 80)
    print("Top 10 largest fires by sample count:")
    print("=" * 80)
    print(summary_df.nlargest(10, 'total')[['fire', 'total', 'high', 'moderate', 'low', 'unburned']])