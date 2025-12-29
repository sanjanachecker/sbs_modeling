"""
Random Forest Hyperparameter Tuning for Burn Severity Classification
=====================================================================
Systematic grid search with cross-validation to find optimal parameters.
Results are logged to CSV for documentation and reproducibility.

Key hyperparameters to tune:
- n_estimators: More trees = more stable but slower
- max_depth: Controls overfitting (your test vs CV gap suggests too deep)
- min_samples_split: Minimum samples to split a node
- min_samples_leaf: Minimum samples at leaf node
- max_features: Features considered at each split
- class_weight: Handle imbalanced classes

Date: December 2024
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score, make_scorer, classification_report
import joblib
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

ORIGINAL_DATA_PATH = 'data/all_fires_complete_covariates_fixed_129.csv'
UPSAMPLED_DATA_PATH = 'data/all_fires_upsampled_points_with_covariates_fixed_129.csv'

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '/home/claude'
RESULTS_DIR = os.path.join(BASE_DIR, 'tuning_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TOP_30_FEATURES = [
    'dnbr', 'dndvi', 'dndbi', 'dbsi', 'nbr', 'bsi', 'ndvi', 'ndbi',
    'meanelev_32', 'wc_bio19', 'nirBand', 'wc_bio05', 'rdgh_6', 'blueBand',
    'minelev_4', 'greenBand', 'wc_bio06', 'swir2Band', 'pisrdif_2021-11-22',
    'pisrdif_2021-12-22', 'stddevelev_32', 'maxc_2', 'wc_bio12', 'wc_bio07',
    'dmndwi', 'wc_bio18', 'wc_bio17', 'wc_bio02', 'vd_5', 'planc_32'
]

# ============================================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================================

# Based on your current results (overfitting indicated by test >> CV kappa),
# we focus on regularization parameters

PARAM_GRID_FULL = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],  # Try shallower trees
    'min_samples_split': [2, 5, 10, 20],  # Higher = more regularization
    'min_samples_leaf': [1, 2, 4, 8],     # Higher = more regularization
    'max_features': ['sqrt', 'log2', 0.3, 0.5],  # Limit features per split
    'class_weight': ['balanced', 'balanced_subsample']
}

# Smaller grid for quick testing
PARAM_GRID_QUICK = {
    'n_estimators': [200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt', 0.5],
    'class_weight': ['balanced']
}

# ============================================================================
# DATA LOADING (same as training script)
# ============================================================================

def load_and_prepare_data():
    """Load and combine datasets."""
    # Load original
    df_orig = pd.read_csv(ORIGINAL_DATA_PATH)
    df_orig['SBS'] = df_orig['SBS'].replace({'mod': 'moderate'})
    
    # Load upsampled
    df_ups = pd.read_csv(UPSAMPLED_DATA_PATH)
    
    # Find common columns
    exclude_cols = {'PointX', 'PointY', 'Source', 'source', '.geo', 'system:index'}
    common_cols = (set(df_orig.columns) & set(df_ups.columns)) - exclude_cols
    required_cols = ['SBS', 'Fire_year'] + [c for c in common_cols if c not in ['SBS', 'Fire_year']]
    required_cols = [c for c in required_cols if c in df_orig.columns and c in df_ups.columns]
    
    # Combine
    df_combined = pd.concat([
        df_orig[required_cols],
        df_ups[required_cols]
    ], ignore_index=True)
    
    # Prepare features
    available_features = [f for f in TOP_30_FEATURES if f in df_combined.columns]
    df_clean = df_combined.dropna(subset=['SBS'])
    X = df_clean[available_features].fillna(df_clean[available_features].median())
    y = df_clean['SBS']
    
    # Encode
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le, available_features


# ============================================================================
# TUNING FUNCTIONS
# ============================================================================

def run_grid_search(X_train, y_train, param_grid, cv=5, n_jobs=-1):
    """Run GridSearchCV with Kappa scoring."""
    kappa_scorer = make_scorer(cohen_kappa_score)
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=kappa_scorer,
        cv=cv,
        n_jobs=n_jobs,
        verbose=2,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search


def run_randomized_search(X_train, y_train, param_distributions, n_iter=50, cv=5, n_jobs=-1):
    """Run RandomizedSearchCV for faster exploration."""
    kappa_scorer = make_scorer(cohen_kappa_score)
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=kappa_scorer,
        cv=cv,
        n_jobs=n_jobs,
        verbose=2,
        return_train_score=True,
        random_state=RANDOM_STATE
    )
    
    random_search.fit(X_train, y_train)
    return random_search


def evaluate_best_model(model, X_test, y_test, le):
    """Evaluate the best model from search."""
    y_pred = model.predict(X_test)
    
    kappa = cohen_kappa_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    
    return kappa, report


def save_tuning_results(search_results, X_test, y_test, le, filename_prefix):
    """Save comprehensive tuning results to CSV and JSON."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Save all CV results to CSV
    cv_results_df = pd.DataFrame(search_results.cv_results_)
    cv_results_df = cv_results_df.sort_values('rank_test_score')
    
    # Add readable columns
    cv_results_df['train_cv_gap'] = cv_results_df['mean_train_score'] - cv_results_df['mean_test_score']
    
    csv_path = os.path.join(RESULTS_DIR, f'{filename_prefix}_all_results_{timestamp}.csv')
    cv_results_df.to_csv(csv_path, index=False)
    print(f"\n    All results saved: {csv_path}")
    
    # 2. Save top 10 configurations
    top_10 = cv_results_df.head(10)[['params', 'mean_test_score', 'std_test_score', 
                                      'mean_train_score', 'train_cv_gap', 'rank_test_score']]
    top_10_path = os.path.join(RESULTS_DIR, f'{filename_prefix}_top10_{timestamp}.csv')
    top_10.to_csv(top_10_path, index=False)
    print(f"    Top 10 saved: {top_10_path}")
    
    # 3. Evaluate best model on test set
    test_kappa, test_report = evaluate_best_model(search_results.best_estimator_, X_test, y_test, le)
    
    # 4. Save summary JSON
    summary = {
        'timestamp': timestamp,
        'best_params': search_results.best_params_,
        'best_cv_kappa': float(search_results.best_score_),
        'test_kappa': float(test_kappa),
        'cv_test_gap': float(search_results.best_score_ - test_kappa),
        'train_cv_gap': float(cv_results_df.iloc[0]['train_cv_gap']),
        'classification_report': test_report,
        'n_configurations_tested': len(cv_results_df),
        'cv_folds': CV_FOLDS
    }
    
    json_path = os.path.join(RESULTS_DIR, f'{filename_prefix}_summary_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"    Summary saved: {json_path}")
    
    return summary, cv_results_df


def print_tuning_summary(summary, baseline_cv_kappa=0.4183, baseline_test_kappa=0.6142):
    """Print formatted tuning summary with comparison to baseline."""
    print("\n" + "="*80)
    print("TUNING RESULTS SUMMARY")
    print("="*80)
    
    print("\nBest Hyperparameters:")
    for param, value in summary['best_params'].items():
        print(f"    {param}: {value}")
    
    print(f"\nPerformance:")
    print(f"    Best CV Kappa:    {summary['best_cv_kappa']:.4f}  (baseline: {baseline_cv_kappa:.4f}, Δ: {summary['best_cv_kappa']-baseline_cv_kappa:+.4f})")
    print(f"    Test Kappa:       {summary['test_kappa']:.4f}  (baseline: {baseline_test_kappa:.4f}, Δ: {summary['test_kappa']-baseline_test_kappa:+.4f})")
    print(f"    Train-CV Gap:     {summary['train_cv_gap']:.4f}  (lower = less overfitting)")
    
    print(f"\nPer-Class Performance:")
    for cls in ['high', 'low', 'moderate', 'unburned']:
        if cls in summary['classification_report']:
            metrics = summary['classification_report'][cls]
            print(f"    {cls:12s}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}")
    
    print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(search_type='quick'):
    """
    Run hyperparameter tuning.
    
    Parameters:
    -----------
    search_type : str
        'quick' - Small grid, fast (~5 min)
        'full'  - Full grid search (~30+ min)
        'random' - Randomized search, good balance (~15 min)
    """
    print("="*80)
    print(f"RANDOM FOREST HYPERPARAMETER TUNING ({search_type.upper()})")
    print("="*80)
    
    # Load data
    print("\n[1] Loading data...")
    X, y, le, features = load_and_prepare_data()
    print(f"    Samples: {len(X)}, Features: {len(features)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Run search
    print(f"\n[2] Running {search_type} search...")
    
    if search_type == 'quick':
        param_grid = PARAM_GRID_QUICK
        n_configs = np.prod([len(v) for v in param_grid.values()])
        print(f"    Testing {n_configs} configurations with {CV_FOLDS}-fold CV")
        search_results = run_grid_search(X_train, y_train, param_grid, cv=CV_FOLDS)
        prefix = 'grid_quick'
        
    elif search_type == 'full':
        param_grid = PARAM_GRID_FULL
        n_configs = np.prod([len(v) for v in param_grid.values()])
        print(f"    Testing {n_configs} configurations with {CV_FOLDS}-fold CV")
        print(f"    This will take a while...")
        search_results = run_grid_search(X_train, y_train, param_grid, cv=CV_FOLDS)
        prefix = 'grid_full'
        
    elif search_type == 'random':
        print(f"    Testing 50 random configurations with {CV_FOLDS}-fold CV")
        search_results = run_randomized_search(X_train, y_train, PARAM_GRID_FULL, n_iter=50, cv=CV_FOLDS)
        prefix = 'random'
    
    else:
        raise ValueError(f"Unknown search_type: {search_type}")
    
    # Save results
    print("\n[3] Saving results...")
    summary, all_results = save_tuning_results(search_results, X_test, y_test, le, prefix)
    
    # Print summary
    print_tuning_summary(summary)
    
    # Save best model
    print("\n[4] Saving best model...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(RESULTS_DIR, f'best_rf_model_{prefix}_{timestamp}.joblib')
    joblib.dump(search_results.best_estimator_, model_path)
    print(f"    Model saved: {model_path}")
    
    return search_results, summary, all_results


if __name__ == "__main__":
    # Start with quick search to verify everything works
    # Then run 'full' or 'random' for thorough tuning
    
    import sys
    search_type = sys.argv[1] if len(sys.argv) > 1 else 'quick'
    
    search_results, summary, all_results = main(search_type=search_type)