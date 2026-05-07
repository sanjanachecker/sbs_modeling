# Documentation: `01_feature_importance.py`

**Script location:** `src/thesis_scripts/01_feature_importance.py`  
**Purpose:** Determine which covariates are most predictive of wildfire burn severity, and produce a ranked feature list that drives all downstream modeling scripts.  
**Output consumed by:** `02_rf_tuning.py`, `03_compare_model_matrices_top8.py`, `04_xgb_cvlofo.py` (all use the hardcoded `TOP_30_FEATURES` list derived from this analysis)

---

## Overview

Before training any final model, we need to answer: *which of the ~100+ candidate covariates actually matter for predicting burn severity, and which are redundant or noisy?* Using all features blindly would increase model complexity, slow inference in Google Earth Engine, and risk overfitting. This script addresses that problem by running three independent feature importance methods and aggregating their rankings into a single consensus score.

---

## Pipeline Stages

### Stage 1 — Data Loading (Section 1, lines 35–47)

```python
df = pd.read_csv('data/all_fires_complete_covariates_fixed_129.csv')
```

**What it loads:** The original (non-upsampled) dataset of point samples across all fires, with each row representing one field-validated burn severity point. This is the `_129` version, meaning it was fixed/cleaned through iteration 129.

**Why use the original (not upsampled) data here?**  
Feature importance analysis should reflect the natural class distribution of the real world. Upsampling artificially inflates minority classes (moderate, high), which would bias importance scores toward features that separate those inflated classes — potentially misleading. The original imbalanced dataset gives a more honest signal about which features are genuinely discriminative.

**Metadata exclusion:**  
Columns like `Fire_year`, `PointX/Y`, `Source`, and `.geo` are explicitly excluded. These are identifiers, not predictive covariates — including them would let the model cheat by memorizing which fire a point belongs to rather than learning generalizable burn severity signals.

---

### Stage 2 — Missing Data Analysis (Section 2, lines 57–76)

```python
missing_counts = df[feature_cols].isnull().sum()
complete_rows_mask = df[feature_cols].notna().all(axis=1)
```

**What it does:** Counts missing values per column and identifies which rows have complete data across all features.

**Why this matters:** Some covariates (e.g., certain terrain derivatives computed at specific window sizes) may not be available for every sample location — for example, if a point is near the edge of a DEM tile. Running importance methods on incomplete data would either silently drop rows or produce NaN-contaminated importance scores. The script handles this with a threshold: if fewer than 500 complete rows exist, it falls back to using only features with zero missing values.

**Design choice:** The 500-row threshold is a pragmatic heuristic — below that, there aren't enough samples to fit a reliable Random Forest and XGBoost, so restricting to complete features is safer than imputing.

---

### Stage 3 — Class Distribution Check (Section 3, lines 86–96)

```python
df['SBS'] = df['SBS'].replace({'mod': 'moderate'})
```

**What it does:** Normalizes a label inconsistency (`'mod'` → `'moderate'`) present in some GEE exports, then prints class counts and the per-fire breakdown.

**Why check per-fire distribution:** Burn severity is not uniformly distributed across fires. Some fires are predominantly low-severity (e.g., prescribed burns), while others are dominated by high-severity patches. Understanding this before importance analysis confirms whether the dataset is sufficiently representative across the full severity spectrum.

---

### Stage 4 — Data Preparation (Section 4, lines 108–150)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

**Stratified split (`stratify=y`):** The 80/20 split is stratified, meaning each class appears in its natural proportion in both train and test sets. Without stratification, random splitting could put almost no `high` samples in the test set (since `high` is the rarest class), making evaluation unreliable.

**Why `random_state=42`:** Reproducibility. Every run of this script produces identical splits, so results are stable and comparable.

**StandardScaler:** Features are standardized (zero mean, unit variance) for use by XGBoost and permutation importance. Note: Random Forest does *not* require scaling (it's invariant to monotonic feature transformations), but scaling is applied to the shared feature matrix for consistency. Tree-based methods ignore it; it only affects XGBoost's numerical optimization slightly.

**Remaining NaN removal:** After the complete-rows filter, a second NaN check removes any columns still containing NaN (e.g., features that are NaN for only a subset of the complete rows). This is a safety net rather than the primary strategy.

---

### Stage 5 — Random Forest Feature Importance: MDI (Section 5, lines 160–186)

```python
rf = RandomForestClassifier(
    n_estimators=200, max_depth=15,
    min_samples_split=5, min_samples_leaf=2,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf_importance = pd.DataFrame({'feature': feature_names,
                               'rf_importance': rf.feature_importances_})
```

**Method: Mean Decrease in Impurity (MDI)**  
Each time a tree node splits on a feature, Gini impurity drops. RF accumulates this drop across all trees and all nodes, then normalizes to sum to 1.0. Features that produce large, consistent impurity reductions are considered important.

**Why Random Forest MDI?**
- Fast to compute (no additional fitting beyond the model itself)
- Captures non-linear interactions naturally (trees can split on any feature combinations)
- Well-established baseline for feature importance in ecological and remote sensing studies

**Known limitation — MDI bias:** MDI systematically overestimates the importance of high-cardinality features (features with many unique values), because more distinct values means more potential split points, creating more opportunities to reduce impurity by chance. In this dataset, continuous covariates like WorldClim variables or elevation statistics are all high-cardinality, so this bias applies uniformly and is somewhat neutralized by also using two other methods.

**`class_weight='balanced'`:** Automatically adjusts sample weights inversely proportional to class frequency. Since `high` severity is the rarest class, it gets upweighted — otherwise the RF would focus disproportionately on unburned and low severity, and features relevant to high-severity detection would be undervalued.

**Why `max_depth=15`:** A soft constraint to reduce overfitting in the importance estimation RF itself. Deep trees memorize training data, which inflates the importance of features that happen to split noise. This is not the final deployed model — it just needs to be representative enough to produce reliable importance estimates.

---

### Stage 6 — XGBoost Feature Importance: Gain (Section 6, lines 196–228)

```python
class_weights = len(y_train) / (len(le.classes_) * class_counts_arr)
sample_weights = class_weights[y_train]

xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
xgb_importance = pd.DataFrame({'feature': feature_names,
                                'xgb_importance': xgb_model.feature_importances_})
```

**Method: XGBoost Gain**  
XGBoost's `feature_importances_` attribute reports *gain* — the average improvement in the loss function (multiclass log-loss here) from all splits on that feature, weighted by the number of samples affected by each split. Unlike MDI, gain is loss-function-aware and accounts for sample counts per split.

**Why XGBoost gain as a second method?**
- Provides an independent estimate uncorrelated with RF MDI
- Gain-based importance is less susceptible to the high-cardinality bias of MDI, because it directly measures predictive improvement rather than impurity reduction
- XGBoost builds trees sequentially (boosting), so features that correct residual errors from earlier trees get credited — this captures a different aspect of importance than RF's parallel ensemble

**Manual class weighting (`class_weights = len(y_train) / (len(le.classes_) * class_counts_arr)`):**  
XGBoost does not accept `class_weight='balanced'` as a parameter; instead, per-sample weights must be passed via `sample_weight`. This formula implements exactly the same balanced weighting as sklearn's `class_weight='balanced'`: each sample's weight is inversely proportional to its class frequency.

**`colsample_bytree=0.8`:** At each tree, only 80% of features are considered. This introduces randomness that prevents any single feature from dominating every tree, leading to more distributed importance scores — better for identifying a broad set of useful features.

---

### Stage 7 — Permutation Importance (Section 7, lines 239–254)

```python
perm_importance = permutation_importance(
    xgb_model, X_test, y_test,
    n_repeats=10, random_state=42, n_jobs=-1,
    scoring='f1_macro'
)
```

**Method:** For each feature, its values are randomly shuffled in the test set, the model predicts on the shuffled data, and the drop in macro-F1 score is measured. This is repeated 10 times (`n_repeats=10`) and averaged. A large performance drop means the feature is genuinely important.

**Why permutation importance as the third method?**  
This is the most model-agnostic and theoretically sound measure of the three:
- It does not depend on the internal structure of trees (no MDI bias)
- It directly measures the contribution of each feature to actual **held-out test performance**, not training behavior
- It detects features that are *redundant* — if two features carry the same information, removing one won't hurt much, so both get lower permutation importance even if both show up high in MDI

**Why `scoring='f1_macro'`?**  
Macro-F1 gives equal weight to all four classes regardless of class frequency. Since `high` severity is rare but scientifically critical, using accuracy (which is dominated by the majority class) would undervalue features important for detecting high severity. Macro-F1 ensures importance scores reflect performance across all severity levels equally.

**`n_repeats=10`:** Each feature is shuffled 10 times independently. The mean across repeats is used as the importance score and the standard deviation captures stability. A single shuffle could be lucky or unlucky; 10 repeats gives a stable estimate.

**Applied to the XGBoost model (not RF):** Permutation importance can be computed on any fitted model. Using the XGBoost model means it measures importance *through the lens of the better-performing model*, which is more relevant to the final modeling decision.

---

### Stage 8 — Aggregate Rankings (Section 8, lines 265–290)

```python
importance_df['rf_rank'] = importance_df['rf_importance'].rank(ascending=False)
importance_df['xgb_rank'] = importance_df['xgb_importance'].rank(ascending=False)
importance_df['perm_rank'] = importance_df['perm_importance_mean'].rank(ascending=False)
importance_df['avg_rank'] = (rf_rank + xgb_rank + perm_rank) / 3

# Normalize each method to [0,1], then average
importance_df['combined_score'] = (rf_norm + xgb_norm + perm_norm) / 3
```

**Why aggregate three methods?**  
No single importance measure is definitive. Each has different theoretical assumptions and known failure modes:

| Method | Strength | Weakness |
|--------|----------|----------|
| RF MDI | Fast, captures interactions | High-cardinality bias, can overfit |
| XGB Gain | Loss-aware, boosting perspective | Sensitive to correlated features |
| Permutation | Test-set performance, model-agnostic | Slow, unstable with correlated features |

Features that consistently rank highly across all three methods are almost certainly genuinely informative — their importance is robust to the choice of method. Features that rank highly in one method but not others may be artifacts of that method's assumptions.

**Two complementary aggregations:**
1. **Average rank** — treats each method's ordinal ranking equally (good for detecting consensus)
2. **Combined score** — normalizes raw importance scores to [0,1] then averages (good for understanding relative magnitude)

The final sort uses `combined_score` as the primary criterion, which captures both rank and magnitude.

---

### Stage 9 — Feature Categorization (Section 9, lines 306–374)

```python
def categorize_feature(name):
    if name.startswith('wc_'): return 'Climate (WorldClim)'
    if 'elev' in name_lower:   return 'Elevation'
    if 'pisr' in name_lower:   return 'Solar Radiation'
    ...
```

**What it does:** Maps each feature name to a semantic category based on naming conventions. Categories include: Spectral Bands, Spectral Indices, Differenced Indices (pre/post fire), Climate (WorldClim), Solar Radiation, Elevation, Curvature, Terrain Indices, Geomorphology, Hydrology.

**Why categorize?**  
Individual feature rankings are useful for selecting a final feature set, but category-level aggregation reveals which *types* of covariates drive burn severity. For the thesis, this answers the interpretive question: *Is burn severity primarily driven by fire behavior (differenced spectral indices like dNBR), pre-fire vegetation state (NDVI, NBR), climate (WorldClim), or terrain?* Category importances are more defensible in a scientific discussion than raw per-feature rankings.

**The naming convention pattern-matching approach:** Feature names follow GEE export conventions (`wc_bio05`, `pisrdif_2021-11-22`, `rdgh_6`, `meanelev_32`, etc.), so string-based rules are reliable. This avoids needing a separate lookup table.

---

### Stage 10 — Save Results (Section 10, lines 381–390)

```python
importance_df.to_csv('results/feature_importance_results.csv', index=False)
top_features = importance_df.head(50)['feature'].tolist()
with open('results/top_50_features.txt', 'w') as f: ...
```

**Outputs:**
- `results/feature_importance_results.csv` — full table with all three raw importance scores, normalized scores, ranks, and combined score for every feature
- `results/top_50_features.txt` — ordered list of the top 50 features by combined score

The **Top 30** from this ranking are hardcoded into `02_rf_tuning.py`, `03_compare_model_matrices_top8.py`, and `04_xgb_cvlofo.py` as `TOP_30_FEATURES`. The cutoff at 30 was chosen because the combined-score curve has an elbow around rank 30–35 (diminishing returns beyond that), and 30 features is a practical size for GEE inference without excessive compute cost.

---

### Stage 11 — Visualizations (Section 11, lines 401–468)

Three plots are generated:

| Plot | File | What it shows |
|------|------|---------------|
| Side-by-side bar charts | `feature_importance_comparison.png` | Top 30 features ranked by each of the three methods — lets you visually compare agreement/disagreement |
| Category bar chart | `category_importance.png` | Mean combined score per feature category with feature counts — answers "what type of feature matters most?" |
| Heatmap | `feature_importance_heatmap.png` | Normalized scores for top 40 features across all three methods — highlights where methods agree (all three high) vs. disagree |

---

### Stage 12 — Recommendations (Section 12, lines 479–525)

**Three suggested feature sets are printed:**

| Set | Size | Use case |
|-----|------|----------|
| Minimal | Top 15 | Fast inference, ablation studies |
| **Recommended** | **Top 30** | **Used in all downstream scripts** |
| Extended | Top 50 | Sensitivity analysis, alternative model runs |

The 30-feature set is the one that propagates through the entire pipeline. It balances predictive power (near-plateau region of the importance curve) with practical constraints (GEE can export and process 30-band images efficiently).

---

## Key Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| Use original (non-upsampled) data | Avoid biasing importance toward features that separate artificially inflated classes |
| Three independent importance methods | No single method is definitive; consensus across methods reduces artifacts |
| `class_weight='balanced'` / manual sample weights | Prevent majority-class (unburned) dominance in importance estimation |
| `scoring='f1_macro'` for permutation | Equal credit to all severity classes, including rare high-severity |
| Combined score = normalized average | Robust to scale differences across methods; captures both rank and magnitude |
| Top 30 cutoff | Elbow in combined-score curve; practical for GEE inference |

---

## Data Flow

```
data/all_fires_complete_covariates_fixed_129.csv
        │
        ▼
  [Missing data analysis]
        │
        ▼
  [Stratified 80/20 split]
        │
        ├──► RF (MDI)             ─┐
        ├──► XGBoost (Gain)        ├──► Aggregate rankings ──► TOP_30_FEATURES
        └──► Permutation (F1 drop) ─┘           │
                                                 ▼
                                  results/feature_importance_results.csv
                                  results/top_50_features.txt
                                  results/feature_importance_comparison.png
                                  results/category_importance.png
                                  results/feature_importance_heatmap.png
```

---

## Dependencies

| Library | Version requirement | Usage |
|---------|-------------------|-------|
| `scikit-learn` | ≥ 1.0 | RF, permutation importance, train_test_split, StandardScaler, LabelEncoder |
| `xgboost` | ≥ 1.6 | XGBClassifier, gain importance |
| `pandas` | ≥ 1.3 | Data loading, DataFrame manipulation |
| `numpy` | ≥ 1.21 | Array operations, NaN handling |
| `matplotlib` / `seaborn` | — | Visualization |

---

*Documentation generated for thesis pipeline — script is the first of four stages feeding into `02_rf_tuning.py`.*
