# Documentation: `02_rf_tuning.py`

**Script location:** `src/thesis_scripts/02_rf_tuning.py`  
**Purpose:** Systematically search across model families, feature set sizes, and data configurations using 10-fold stratified cross-validation to identify the best model for burn severity classification.  
**Preceded by:** `01_feature_importance.py` (produces `TOP_30_FEATURES` hardcoded here)  
**Followed by:** `03_compare_model_matrices_top8.py` (deep-dives the top 8 configs), then `04_xgb_cvlofo.py` (LOFO evaluation and deployment of the winner)  
**Outputs:** `02_results_model_search/`

---

## Overview

After `01_feature_importance.py` identifies which covariates are predictive, a second question remains: *which model architecture and feature set size should be used?* Rather than committing to a single algorithm (e.g., XGBoost only), this script runs an exhaustive search across:

- **11 model configurations** spanning 4 algorithm families (Random Forest, ExtraTrees, Gradient Boosting, XGBoost)
- **4 feature set sizes** (Top 30, Top 50, Top 75, All features)
- **1 dataset** (original + upsampled combined)

That gives **~44 distinct experiments**, each evaluated with 10-fold stratified CV. The output is a ranked table sorted by a composite score that balances overall accuracy (κ) against detection of the rarest and most important class (high severity recall).

---

## Configuration (lines 32–59)

```python
MAIN_CSV      = '.../real_all_fires_complete_covariates_fixed_1229.csv'
UPSAMPLED_CSV = '.../real_all_fires_upsampled_points_with_covariates_fixed.csv'
OUTPUT_DIR    = '.../results_model_search'

KAPPA_WEIGHT      = 0.7
HIGH_RECALL_WEIGHT = 0.3
```

### Composite scoring weights: 70% κ + 30% high recall

The central methodological choice in this script is the composite ranking metric:

```
composite_score = 0.7 × mean_kappa + 0.3 × mean_high_recall
```

**Why not rank by κ alone?**  
Cohen's κ is a measure of overall agreement across all classes. In a multi-class imbalanced setting, a model can achieve high κ by being excellent at the dominant classes (unburned, low) while entirely ignoring the rare but scientifically critical `high` severity class. For wildfire management, failing to detect high-severity pixels is the most consequential error — it affects post-fire watershed assessments, replanting decisions, and erosion risk estimates.

**Why 70/30 split?**  
The 70/30 split treats κ as the primary criterion (the model must be generally accurate) while giving meaningful secondary weight to high-class recall (it cannot completely ignore high severity). A 50/50 split would over-penalize models with slightly lower κ, and a 90/10 split would make high recall nearly irrelevant.

**`TOP_30_FEATURES` hardcoded:**  
Rather than re-running feature importance each time, the top 30 features from `01_feature_importance.py` are hardcoded. This ensures the feature set is stable and reproducible across all downstream scripts. Features ranked 31–50 and 51–75 are computed dynamically from a preliminary RF at run time.

---

## Stage 1 — Data Loading (lines 66–88)

```python
df_main = pd.read_csv(MAIN_CSV)
df_up   = pd.read_csv(UPSAMPLED_CSV)
common_cols = list(set(df_main.columns) & set(df_up.columns))
df = pd.concat([df_main[common_cols], df_up[common_cols]], ignore_index=True)
return {'Upsampled': df}
```

**Why combine original + upsampled?**  
Burn severity classes are severely imbalanced. In the natural data, unburned and low severity pixels dominate most fire perimeters. Running CV on the raw imbalanced data would produce models that achieve misleadingly high accuracy by simply predicting the majority class. The upsampled data adds synthetic points for moderate and high severity, helping all models in the search space learn to distinguish the full range of severity levels.

**Why return a dict `{'Upsampled': df}`?**  
The script was originally designed to test multiple dataset configurations (e.g., `{'NoUp': df_orig, 'Upsampled': df_combined}`). Returning a dict makes it trivial to extend the search to additional dataset variants without restructuring the main loop. The current version tests only the upsampled combination, but the architecture allows for easy expansion.

**`common_cols` intersection:**  
The two CSVs may have slightly different columns depending on when each was exported from GEE. The intersection ensures every row in the combined dataset has a value for every column — no NaN contamination from columns present in one source but not the other.

---

## Stage 2 — Feature Set Determination (lines 91–114)

```python
def get_top_n_features(df, n):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X, y)
    indices = np.argsort(rf.feature_importances_)[::-1][:n]
    return [available[i] for i in indices]

top50 = get_top_n_features(df_largest, 50)
top75 = get_top_n_features(df_largest, 75)
all_feats = get_all_features(df_largest)
```

**Why re-run RF importance here for Top 50 and Top 75?**  
`01_feature_importance.py` used a multi-method consensus approach (RF MDI + XGB gain + permutation) to produce the Top 30 features, which are hardcoded. For the larger feature sets (Top 50, Top 75), a simpler single-RF importance ranking is used because:
1. A full three-method analysis for every feature set would add significant runtime
2. The Top 50 and Top 75 sets are secondary experiments — the Top 30 is the primary set
3. RF MDI is sufficient for quickly identifying a broader set of potentially useful features

**`class_weight='balanced'`:** Applied to the preliminary RF to ensure minority classes (high, moderate) influence the importance ranking — otherwise features important for those classes would be underweighted.

**`max_depth=None`:** Fully grown trees are used for the preliminary RF because the goal is feature ranking, not generalization. Deep trees have higher variance but also higher sensitivity to feature relationships, which is desirable for importance estimation.

**`EXCLUDE_COLS`:** A comprehensive set of metadata and identifier columns is excluded from the feature universe. This prevents the model from learning spurious patterns from spatial coordinates, source identifiers, or fire labels — any of which would cause data leakage.

---

## Stage 3 — Model Definitions (lines 130–196)

```python
def get_models():
    return {
        'RF_balanced':     RandomForestClassifier(n_estimators=500, class_weight='balanced', ...),
        'RF_bal_sub':      RandomForestClassifier(n_estimators=500, max_depth=20, class_weight='balanced_subsample', ...),
        'RF_1000trees':    RandomForestClassifier(n_estimators=1000, max_depth=25, ...),
        'ExtraTrees':      ExtraTreesClassifier(n_estimators=500, class_weight='balanced', ...),
        'ExtraTrees_tuned':ExtraTreesClassifier(n_estimators=1000, max_depth=25, class_weight='balanced_subsample', ...),
        'GBM_default':     GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, ...),
        'GBM_deep':        GradientBoostingClassifier(n_estimators=500, max_depth=8, learning_rate=0.05, ...),
        'XGB_default':     xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, ...),
        'XGB_deep':        xgb.XGBClassifier(n_estimators=800, max_depth=8, learning_rate=0.05, ...),
        'XGB_shallow':     xgb.XGBClassifier(n_estimators=1000, max_depth=4, learning_rate=0.01, ...),
        'XGB_lr01':        xgb.XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.01, ...),
    }
```

### Why these four algorithm families?

| Family | Mechanism | Included variants |
|--------|-----------|------------------|
| **Random Forest** | Parallel ensemble of decision trees, bagging + feature randomness | `RF_balanced`, `RF_bal_sub`, `RF_1000trees` |
| **ExtraTrees** | Like RF but splits are chosen at random thresholds (not optimal), even faster and more regularized | `ExtraTrees`, `ExtraTrees_tuned` |
| **Gradient Boosting (sklearn)** | Sequential ensemble, each tree corrects residuals of the previous | `GBM_default`, `GBM_deep` |
| **XGBoost** | Optimized gradient boosting with regularization, column/row subsampling | `XGB_default`, `XGB_deep`, `XGB_shallow`, `XGB_lr01` |

Testing multiple families avoids the risk of algorithm selection bias — perhaps RF overfits to spatial autocorrelation in this dataset, or GBM is more sensitive to the class imbalance. The sweep lets the data answer the question.

### Why multiple XGB variants?

XGBoost's performance is sensitive to the learning rate / n_estimators tradeoff. Four variants explore this space:

| Config | Learning Rate | Trees | Depth | Design intent |
|--------|--------------|-------|-------|---------------|
| `XGB_default` | 0.05 | 500 | 6 | Baseline |
| `XGB_deep` | 0.05 | 800 | 8 | More expressive, risk of overfitting |
| `XGB_shallow` | 0.01 | 1000 | 4 | Slow learning, low depth — high regularization |
| **`XGB_lr01`** | **0.01** | **1000** | **6** | **Slow learning, standard depth — winner** |

`XGB_lr01` won because the slow learning rate (`lr=0.01`) with more trees gives the model more opportunities to correct errors gradually, reducing overfitting, while `max_depth=6` is expressive enough to capture interaction effects.

### `class_weight='balanced'` vs. `'balanced_subsample'`

| Setting | How it works | When preferred |
|---------|-------------|----------------|
| `'balanced'` | Weights computed once on the full dataset, same weights for every tree | Stable, consistent across trees |
| `'balanced_subsample'` | Weights recomputed on each tree's bootstrap sample | Better for highly imbalanced datasets; each tree sees a different class distribution |

`balanced_subsample` is generally more robust but adds variance. Both are tested to see which works better for this specific dataset.

---

## Stage 4 — 10-Fold Stratified CV Runner (lines 203–248)

```python
def run_10fold_cv(X, y, model, model_name, n_folds=10):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(X, y):
        if is_xgb:
            y_train_num = np.array([LABEL_MAP[s] for s in y_train])
            sw = compute_sample_weight('balanced', y_train_num)
            model_clone.fit(X_train, y_train_num, sample_weight=sw)
        else:
            model_clone.fit(X_train, y_train)

        kappa = cohen_kappa_score(y_val, y_pred)
        high_recall = recall_score(...)[CLASS_NAMES.index('high')]
```

### Why stratified k-fold?

`StratifiedKFold` ensures each fold contains approximately the same class proportions as the full dataset. Without stratification, a random split could place nearly all `high` severity samples in a single fold, making 9 out of 10 folds unable to meaningfully train or evaluate on that class.

**`shuffle=True, random_state=42`:** Shuffling before folding prevents any ordering bias in the data (e.g., if samples are sorted by fire and fire order correlates with severity patterns). The fixed random state ensures every model is evaluated on the identical 10 fold assignments — comparisons are apples-to-apples.

### Why `model_clone`?

```python
model_clone = xgb.XGBClassifier(**model.get_params())
model_clone = model.__class__(**model.get_params())
```

The model defined in `get_models()` is a template, not the fitted instance. A fresh clone is created at each fold to ensure no state leaks between folds. Without cloning, fitting on fold 1 would alter the model's internal state, corrupting fold 2's training.

### XGB requires numeric labels + manual sample weights

```python
y_train_num = np.array([LABEL_MAP[s] for s in y_train])
sw = compute_sample_weight('balanced', y_train_num)
model_clone.fit(X_train, y_train_num, sample_weight=sw)
```

XGBoost's multi-class objective requires integer labels (`0, 1, 2, 3`), not strings. sklearn models accept string labels directly. This branching (via `is_xgb = 'XGB' in model_name`) handles the different APIs without duplicating the fold loop.

`compute_sample_weight('balanced', y_train_num)` recomputes class-balanced weights at each fold. Since each fold's training set has a slightly different class distribution, weights must be recomputed per fold — not once globally.

### Per-fold metrics: only κ and high recall

```python
kappa = cohen_kappa_score(y_val, y_pred)
high_recall = recall_score(..., labels=CLASS_NAMES, average=None)[CLASS_NAMES.index('high')]
```

Only two metrics are collected per fold: κ (the primary summary metric) and high-class recall (the secondary metric driving the composite score). Collecting the full classification report for all ~440 fold evaluations would produce an enormous output with little additional value for ranking purposes.

---

## Stage 5 — Main Experiment Loop (lines 291–363)

```python
for ds_name, df in datasets.items():       # 1 dataset variant
    for fs_name, fs in feature_sets.items():   # 4 feature sets
        X, y, available = prepare_xy(df, fs)
        for model_name, model in models.items():  # 11 models
            cv_metrics = run_10fold_cv(X, y, model, model_name)
            combined_score = 0.7 * mean_k + 0.3 * mean_high_recall
            results.append({...})
```

**Progress markers:**
```python
marker = " ★" if combined_score >= 0.60 else " ✓" if combined_score >= 0.55 else ""
```
During the ~45-minute run, stars and checkmarks printed to stdout give live feedback on which configs are competitive, without waiting for the full sorted table at the end.

**Exception handling per experiment:**  
Each experiment is wrapped in `try/except`. If one model fails (e.g., due to a convergence issue or memory error), it logs `combined_score = 0.0` and continues to the next configuration. This prevents a single failing configuration from crashing the entire multi-hour sweep.

---

## Stage 6 — Results and Ranking (lines 366–436)

```python
results_df = pd.DataFrame(results).sort_values(
    ['combined_score', 'mean_kappa', 'mean_high_recall'],
    ascending=False
)
results_df.to_csv(f'{OUTPUT_DIR}/10fold_cv_all_results.csv', index=False)
results_df.sort_values('mean_high_recall').to_csv(
    f'{OUTPUT_DIR}/10fold_cv_sorted_by_high_recall.csv', index=False)
```

**Multi-key sort:** Primary sort by `combined_score`, then `mean_kappa`, then `mean_high_recall`. This means ties in composite score (rare but possible) are broken by overall κ first.

**Two CSV outputs:**
- `10fold_cv_all_results.csv` — sorted by composite score; this is the table used to select configs for Stage 3
- `10fold_cv_sorted_by_high_recall.csv` — sorted by high recall alone; useful for checking whether any config achieves exceptional high recall even at the cost of overall accuracy

**Bar chart of top 20:**

```python
colors = ['#e74c3c' if s < 0.50 else '#f39c12' if s < 0.55 else
          '#2ecc71' if s < 0.60 else '#27ae60' for s in top20['combined_score']]
ax.axvline(x=0.60, color='green', linestyle='--', label='Strong composite')
```

Color-coded by composite score tier, with a 0.60 threshold marking configurations considered "strong." The chart is saved to `02_results_model_search/top20_10fold_cv.png`.

---

## Results: What the Sweep Found

The winning configuration — used in all subsequent scripts — was:

**`XGB_lr01 | Upsampled | Top30`**

| Metric | Value |
|--------|-------|
| Mean 10-fold κ | 0.6101 ± 0.024 |
| Mean high recall | 0.6366 |
| **Composite score** | **0.6181** |

### Why XGB_lr01 | Top30 beat competitors

| Config | Mean κ | High Recall | Composite | Reason it lost |
|--------|--------|-------------|-----------|----------------|
| XGB_lr01 \| Top30 | 0.610 | **0.637** | **0.618** | — winner — |
| XGB_default \| Top30 | 0.617 | 0.602 | 0.612 | Higher κ but lower high recall; composite lower |
| ExtraTrees_tuned \| Top30 | 0.612 | 0.612 | 0.612 | Virtually tied, but XGB_lr01 slightly better composite |
| XGB_shallow \| Top30 | 0.582 | 0.656 | 0.605 | Best high recall but κ too low overall |
| ExtraTrees_tuned \| Top50 | 0.618 | 0.570 | 0.604 | Best raw κ of any config, but high recall too low |

`XGB_lr01` uniquely combines a strong κ with the highest high recall of any non-shallow config. The slow learning rate (`lr=0.01`) with 1000 trees provides more regularized, stable predictions that generalize better to held-out folds, while `max_depth=6` is expressive enough to capture the interaction between fire behavior indicators and terrain/climate features.

---

## Key Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| Composite score = 0.7κ + 0.3 high recall | Pure κ optimization ignores the rare but critical high-severity class |
| 10-fold stratified CV | Stratification ensures minority classes appear in every fold; k=10 balances variance and compute cost |
| Shuffle + fixed random state | Removes ordering bias; ensures identical fold assignments across all model comparisons |
| Model cloning per fold | Prevents internal state leakage between folds |
| XGB requires manual `sample_weight` | XGBoost API does not support `class_weight` parameter |
| Dynamic Top 50/75 via preliminary RF | Avoids re-running the expensive three-method consensus analysis; RF MDI is sufficient for secondary feature sets |
| Exception handling per experiment | Prevents one failing config from aborting the full multi-hour sweep |
| Two sorted CSV outputs | Composite-ranked (primary use) + high-recall-ranked (secondary analysis of high-severity sensitivity) |

---

## Data Flow

```
data/real_all_fires_complete_covariates_fixed_1229.csv  ─┐
data/real_all_fires_upsampled_points_with_covariates_*.csv ─┘
         │
         ├──► [Preliminary RF] ──► Top 50 features
         │                   └──► Top 75 features
         │
         └──► [44 experiments: 11 models × 4 feature sets]
                   │
                   └──► 10-fold stratified CV per experiment
                              │
                              ├──► 02_results_model_search/10fold_cv_all_results.csv
                              ├──► 02_results_model_search/10fold_cv_sorted_by_high_recall.csv
                              └──► 02_results_model_search/top20_10fold_cv.png
                                             │
                                             ▼
                                  Top 8 configs → 03_compare_model_matrices_top8.py
                                  Winner (XGB_lr01 | Top30) → 04_xgb_cvlofo.py
```

---

## Dependencies

| Library | Usage |
|---------|-------|
| `scikit-learn` | `RandomForestClassifier`, `ExtraTreesClassifier`, `GradientBoostingClassifier`, `StratifiedKFold`, `cohen_kappa_score`, `recall_score`, `compute_sample_weight` |
| `xgboost` | `XGBClassifier` |
| `pandas` / `numpy` | Data handling, result aggregation |
| `matplotlib` | Top-20 bar chart visualization |

---

*Documentation generated for thesis pipeline — this is the second of four stages: `01_feature_importance.py` → **`02_rf_tuning.py`** → `03_compare_model_matrices_top8.py` → `04_xgb_cvlofo.py`.*
