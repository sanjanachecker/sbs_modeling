# Documentation: `04_xgb_cvlofo.py`

**Script location:** `src/thesis_scripts/04_xgb_cvlofo.py`  
**Purpose:** Evaluate the chosen XGBoost configuration using Leave-One-Fire-Out (LOFO) cross-validation, train the final model on all available data, and export it as a TensorFlow SavedModel for Google Earth Engine (GEE) inference.  
**Preceded by:** `02_rf_tuning.py` (10-fold CV model search that identified `XGB_lr01 | Top30` as the best configuration)  
**Outputs consumed by:** GEE Colab inference notebook (wall-to-wall burn severity maps)

---

## Overview

After the 10-fold CV sweep in `02_rf_tuning.py` identified `XGB_lr01` with the Top 30 features as the best configuration, this script performs three tasks in sequence:

1. **Rigorous evaluation** — Leave-One-Fire-Out CV, which is the most ecologically appropriate validation strategy for fire-level data
2. **Final model training** — fits on the complete dataset (all fires) for maximum coverage before GEE deployment
3. **Export** — packages the model as a TF SavedModel so GEE's `ee.Model.fromAiPlatformPredictor` can call it at pixel scale

---

## Configuration (lines 34–78)

```python
MAIN_CSV    = '.../real_all_fires_complete_covariates_fixed_1229.csv'
UPSAMPLED_CSV = '.../real_all_fires_upsampled_points_with_covariates_fixed.csv'
OUTPUT_DIR  = '.../results_xgb_lr01_top30'
GEE_EXPORT_DIR = '.../gee_models_xgb_lr01_top30'

XGB_PARAMS = {
    'n_estimators': 1000, 'max_depth': 6, 'learning_rate': 0.01,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
    'objective': 'multi:softprob', 'num_class': 4,
    'eval_metric': 'mlogloss'
}

TOP_30_FEATURES = ['dnbr', 'dndvi', 'dndbi', ...]  # from 01_feature_importance.py
```

**Why these XGB hyperparameters?** These come directly from the model search sweep in `02_rf_tuning.py`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators=1000` | 1000 trees | More trees → more stable predictions; low `learning_rate` requires more trees to converge |
| `learning_rate=0.01` | 0.01 | Slow learning rate reduces overfitting; XGB shrinks each tree's contribution, forcing subsequent trees to correct smaller residuals |
| `max_depth=6` | 6 | Moderate depth captures interaction effects without memorizing individual samples |
| `subsample=0.8` | 0.8 | Each tree is trained on a random 80% of samples (stochastic gradient boosting), adding regularizing randomness |
| `colsample_bytree=0.8` | 0.8 | Each tree considers only 80% of features, preventing any single feature from dominating every tree |
| `min_child_weight=3` | 3 | A node is only split if at least 3 samples fall into it; prevents splits on tiny subsets that would only fit noise |
| `objective='multi:softprob'` | — | Returns class probabilities (not hard labels) — allows soft threshold decisions and probability maps in GEE |
| `eval_metric='mlogloss'` | — | Multiclass log-loss; penalizes confident wrong predictions more than uncertain ones |

**Two output directories:** The split between `OUTPUT_DIR` (CV diagnostic plots and per-fire CSV) and `GEE_EXPORT_DIR` (the model artifacts) is deliberate — evaluation results are kept separate from deployment artifacts to avoid confusion about what is "the model" vs. "the evaluation."

---

## Stage 1 — Data Loading (lines 85–117)

```python
df_main = pd.read_csv(MAIN_CSV)
df_up   = pd.read_csv(UPSAMPLED_CSV)
common_cols = list(set(df_main.columns) & set(df_up.columns))
df = pd.concat([df_main[common_cols], df_up[common_cols]], ignore_index=True)
```

**Two data sources are combined:**

| Source | Role |
|--------|------|
| `real_all_fires_complete_covariates_fixed_1229.csv` | Original field-validated points from all fires |
| `real_all_fires_upsampled_points_with_covariates_fixed.csv` | Synthetically upsampled points targeting moderate/high class balance |

**Why combine rather than use original only?**  
Burn severity datasets are inherently imbalanced — unburned and low-severity pixels dominate most fire perimeters. Without any correction, XGBoost would optimize toward the majority class. The upsampled dataset adds synthetic points for underrepresented severity classes, improving the model's ability to detect moderate and high severity without discarding real observations.

**`common_cols` intersection:**  
The two CSVs may have slightly different column sets (e.g., if the upsampled version was exported from GEE at a different time). Taking the intersection of columns ensures no feature appears in one dataset but not the other, which would cause NaN contamination in the combined matrix.

**`'mod'` → `'moderate'` normalization:**  
GEE label exports sometimes use `'mod'` as a shorthand. This is standardized to `'moderate'` immediately after loading to prevent silent label-mismatch bugs where samples labeled `'mod'` would be excluded from the four valid classes.

**Fire column detection:**
```python
fire_col = 'Fire_year' if 'Fire_year' in df.columns else 'fire'
```
Handles the naming inconsistency between older and newer GEE exports gracefully without hard-coding.

---

## Stage 2 — Leave-One-Fire-Out Cross-Validation (lines 136–275)

```python
def leave_one_fire_out_cv(df, fire_col, features):
    for held_out_fire in eval_fires:
        train_mask = fires_all != held_out_fire
        test_mask  = fires_all == held_out_fire
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = model.predict(X_test)
```

### Why LOFO instead of k-fold?

Standard k-fold CV splits data randomly, which means training and test sets often share samples from the same fire. Because nearby pixels within a fire share climate, terrain, and burn conditions, including train and test points from the same fire inflates apparent performance — the model has seen very similar samples during training.

LOFO treats each fire as the unit of generalization: train on all fires except one, predict on the held-out fire. This asks: *"Can this model predict burn severity in a fire it has never seen during training?"* — which is exactly the question that matters for operational deployment to new fires.

### Minimum samples threshold

```python
MIN_SAMPLES = 1
eval_fires = [f for f in unique_fires if fire_counts.get(f, 0) >= MIN_SAMPLES]
```

`MIN_SAMPLES = 1` means every fire with at least one sample is used as a held-out fold. In practice, 2 fires end up excluded from κ computation because they have only 1 class in their test set (Cohen's κ is undefined for single-class prediction). These fires still participate as training data in every other fold.

### Balanced sample weights

```python
sample_weights = compute_sample_weight('balanced', y_train)
```

Applied at every LOFO fold, independently. Each fold's training set has a slightly different class distribution depending on which fire is held out (e.g., holding out a fire with many high-severity points temporarily reduces the high class in training). Recomputing balanced weights per fold ensures the model doesn't shift its class emphasis based on which fire is missing.

**Why `compute_sample_weight` instead of `class_weight`?**  
XGBoost's `XGBClassifier` does not support a `class_weight` parameter. Balanced weights must be passed explicitly as a per-sample array via `sample_weight`. The formula is: `weight_i = n_samples / (n_classes × n_samples_in_class_i)`.

### Per-fire metrics

```python
recalls = recall_score(y_test, y_pred, labels=list(range(4)),
                       average=None, zero_division=0)
kappa = cohen_kappa_score(y_test, y_pred) if len(np.unique(y_test)) >= 2 else float('nan')
```

For each held-out fire, the script records:
- **Cohen's κ** — agreement beyond chance, the primary metric
- **Accuracy** — for comparison, but less meaningful given class imbalance
- **Per-class recall** — separately for unburned, low, moderate, high

**`zero_division=0`:** If a fire has no high-severity samples, the high recall is set to 0.0 rather than raising an error or returning NaN. This is important for the per-fire mean high recall statistic — see the analysis note below.

> [!NOTE]
> 41 of 52 fires have `recall_high = 0.0` because they have zero high-severity samples in their test set. The **mean per-fire high recall of 0.128** is therefore a structural artifact, not a reflection of model performance on high-severity pixels. Use the **pooled recall of 0.470** (from the classification report) as the thesis headline number.

### Pooled vs. per-fire κ

Two κ values are computed:

| Metric | How computed | Value | Use |
|--------|-------------|-------|-----|
| **Pooled κ** | Single κ on all LOFO predictions concatenated | **0.4399** | Headline number — thesis |
| Mean per-fire κ | Average of 50 valid per-fire κ scores | 0.4708 | Secondary — shows fire-level consistency |

The pooled κ is lower because large fires (e.g., `dixie_2021` with 1,010 samples) dominate the pooled calculation. `dixie_2021` has κ = 0.423, which pulls the pooled value below the unweighted mean.

### Confusion matrix ordering

```python
NUMERIC_LABELS = [0, 1, 2, 3]  # unburned, low, moderate, high
cm = confusion_matrix(all_y_true, all_y_pred, labels=NUMERIC_LABELS)
```

Explicit `labels=` argument forces the matrix into the expected order. Without this, sklearn's `confusion_matrix` sorts classes alphabetically — which would put `high` first, `low` second, `moderate` third, `unburned` fourth, breaking the severity-order interpretation.

---

## Stage 3 — Final Model Training (lines 282–301)

```python
def train_final_model(df, features):
    X, y, available = prepare_xy(df, features)
    sample_weights = compute_sample_weight('balanced', y)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, sample_weight=sample_weights)
    scaler = StandardScaler()
    scaler.fit(X)
    return model, scaler, available
```

**Why train on ALL data after LOFO?**  
LOFO CV is purely for evaluation — it tells us how well the model generalizes. The final deployed model should use every available sample, because more training data generally improves generalization to new fires. Withholding any fire for a final held-out test would reduce the training set unnecessarily.

**The LOFO results apply directly to this final model** because the hyperparameters and feature set are fixed. LOFO tells us the expected performance of a model trained on N−1 fires, which is a conservative lower bound for a model trained on all N fires.

**StandardScaler fitted here:**  
The scaler is fitted on the full training matrix and its parameters (`mean_`, `scale_`) are saved to `model_metadata.json` and `scaler.pkl`. This is critical for GEE inference — when the model runs on satellite imagery pixels, those pixels must be normalized using *exactly the same mean and standard deviation* as the training data. Any mismatch would corrupt predictions.

> [!IMPORTANT]
> XGBoost tree models are technically invariant to feature scaling (splits are threshold-based, not distance-based). The scaler is saved for the GEE inference pipeline which applies normalization as a preprocessing step, maintaining consistency with how the Colab notebook preprocesses input data.

---

## Stage 4 — TF SavedModel Export (lines 308–400)

```python
class XGBModule(tf.Module):
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, n_features], dtype=tf.float32,
                      name='covariate_input')
    ])
    def __call__(self, covariate_input):
        result = tf.py_function(func=self._predict,
                                inp=[covariate_input], Tout=tf.float32)
        result.set_shape([None, 4])
        return result

    def _predict(self, inputs):
        proba = self.xgb_model.predict_proba(inputs.numpy()).astype(np.float32)
        return tf.constant(proba)

tf.saved_model.save(module, export_path, signatures={...})
```

### Why TF SavedModel format?

Google Earth Engine's hosted inference (`ee.Model.fromAiPlatformPredictor`) requires models in TF SavedModel format. XGBoost natively exports `.json` or `.ubj` files that GEE cannot use. The TF wrapper bridges this gap: it wraps the XGBoost model in a `tf.Module`, exposing a `@tf.function`-decorated `__call__` method that GEE can call at pixel scale.

### Why `tf.py_function`?

XGBoost's `predict_proba` is a Python/C++ function — it cannot be compiled into a TensorFlow computation graph directly. `tf.py_function` tells TF to execute the enclosed Python callable eagerly (bypassing graph compilation) and pass the result back into the TF graph as a tensor. This is the standard pattern for wrapping non-TF models for GEE.

The trade-off is that `tf.py_function` ops cannot be run in parallel on accelerators, but for a 30-feature tabular model this is not a bottleneck.

### `result.set_shape([None, 4])`

`tf.py_function` returns a tensor of unknown shape (TF cannot inspect the output shape of an arbitrary Python function at graph-build time). `set_shape` manually declares the shape as `[batch_size, 4]`, enabling downstream TF ops and GEE's serving infrastructure to know they will receive a 4-class probability vector per pixel.

### `reorder_idx = list(range(4))`

XGBoost's internal class order follows the numeric label map (`{unburned: 0, low: 1, moderate: 2, high: 3}`). The `reorder_idx` variable was designed to handle cases where XGBoost's `classes_` attribute returns classes in a different order (e.g., alphabetical). Since labels are integers here, no reordering is needed — but the structure exists as a safety mechanism.

### Serving signature

```python
signatures={
    'serving_default': module.__call__.get_concrete_function(
        tf.TensorSpec(shape=[None, n_features], dtype=tf.float32,
                      name='covariate_input')
    )
}
```

The `serving_default` key is the standard GEE-required signature name. The input tensor is named `covariate_input` — this name must match exactly in the GEE Colab notebook's API call configuration.

### Model metadata JSON

```python
metadata = {
    'feature_names': features,      # exact order for GEE band stacking
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'xgb_params': {...},
    'tuning_cv_kappa': 0.6101,      # from 02_rf_tuning.py sweep
    'training_info': {
        'lofo_cv_kappa': float(cv_results['overall_kappa']),
        'mean_high_recall': ...,
        ...
    }
}
```

`feature_names` is the most critical field — it defines the exact order of the 30 features. In GEE, multi-band images must have bands stacked in the same order as the training feature matrix. If even one band is out of order, all predictions are corrupted. The Colab notebook reads this JSON and constructs the GEE band selector accordingly.

---

## Stage 5 — Visualization (lines 407–505)

Four diagnostic plots are generated and saved to `results_xgb_lr01_top30/`:

### Plot 1: Per-fire Kappa bar chart (`lofo_per_fire_kappa.png`)

```python
colors = ['#e74c3c' if k < 0.3 else '#f39c12' if k < 0.5 else
          '#2ecc71' if k < 0.65 else '#27ae60' for k in valid['kappa']]
ax.barh(valid['fire'].astype(str), valid['kappa'], color=colors)
ax.axvline(x=0.65, color='green', linestyle='--', label='Target: 0.65')
```

Color-coded by κ tier:
- **Red** (κ < 0.30) — poor agreement, essentially no skill beyond chance
- **Orange** (0.30–0.50) — fair agreement, some skill
- **Light green** (0.50–0.65) — moderate–good agreement
- **Dark green** (≥ 0.65) — substantial agreement (Landis & Koch 1977 threshold)

The 0.65 target line reflects the thesis's primary performance goal. Fires to the right of this line are those where the model performs "substantially" well.

### Plot 2: Pooled confusion matrix (`lofo_confusion_matrix.png`)

```python
cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
annot[i, j] = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'
```

Each cell shows raw counts and the percentage of that true class (row-normalized). Row normalization reveals recall per class — the diagonal percentages are per-class recall. This is more interpretable than column normalization (which would show precision) because recall is the primary concern: we want to know how often each true severity class is correctly identified.

The color scale uses `'YlGn'` (yellow-green) which is perceptually uniform and appropriate for sequential data — darker cells are more correct (higher count).

### Plot 3: Kappa distribution histogram (`lofo_kappa_distribution.png`)

```python
ax.hist(valid['kappa'], bins=15, color='#3498db', edgecolor='black', alpha=0.8)
ax.axvline(x=valid['kappa'].mean(), color='red', ...)
ax.axvline(x=valid['kappa'].median(), color='orange', ...)
ax.axvline(x=0.65, color='green', ...)
```

Shows the spread of per-fire performance. Key diagnostics:
- **Mean vs. median gap**: A large gap indicates the distribution is skewed (likely by a few very poorly-predicted fires)
- **Fires left of 0.20**: Structurally difficult fires (very few samples, only 1–2 severity classes present)
- **Fires right of 0.65**: The "substantial agreement" tier — strong generalization

### Plot 4: Kappa vs. High Recall scatter (`lofo_kappa_vs_high_recall.png`)

```python
sc = ax.scatter(valid['kappa'], valid['recall_high'],
                c=valid['recall_moderate'], cmap='RdYlGn', ...)
plt.colorbar(sc, ax=ax, label='Moderate Recall')
ax.axvline(x=0.65, ...)
ax.axhline(y=0.60, ...)
```

This is the key diagnostic for the thesis's primary tension: **overall accuracy vs. high-severity detection**. A model can achieve high κ while completely missing high-severity pixels (which are rare). This plot reveals:
- **Top-right quadrant** (κ ≥ 0.65 AND high recall ≥ 0.60): the target zone — both good overall and good high detection
- **Bottom-right** (high κ, low high recall): accurate overall but misses high severity
- **Color (moderate recall)**: shows whether good high recall comes at the cost of moderate class performance

---

## Stage 6 — Output Files

| File | Location | Description |
|------|----------|-------------|
| `lofo_per_fire_kappa.png` | `results_xgb_lr01_top30/` | Per-fire κ bar chart |
| `lofo_confusion_matrix.png` | `results_xgb_lr01_top30/` | Pooled LOFO confusion matrix |
| `lofo_kappa_distribution.png` | `results_xgb_lr01_top30/` | Histogram of per-fire κ |
| `lofo_kappa_vs_high_recall.png` | `results_xgb_lr01_top30/` | Kappa vs. high recall scatter |
| `lofo_per_fire_results.csv` | `results_xgb_lr01_top30/` | Per-fire κ, accuracy, per-class recall |
| `xgb_lr01_top30.joblib` | `gee_models_xgb_lr01_top30/` | Serialized XGBoost model |
| `xgb_lr01_top30_savedmodel/` | `gee_models_xgb_lr01_top30/` | TF SavedModel directory for GEE |
| `model_metadata.json` | `gee_models_xgb_lr01_top30/` | Feature names, params, scaler, CV metrics |
| `scaler.pkl` | `gee_models_xgb_lr01_top30/` | Fitted StandardScaler for inference |

---

## Key Numerical Results

| Metric | Value |
|--------|-------|
| Pooled LOFO κ | **0.4399** |
| Pooled accuracy | 0.6005 |
| Mean per-fire κ | 0.4708 |
| Median per-fire κ | 0.4628 |
| Fires with κ ≥ 0.65 | 10/50 (20%) |
| Unburned recall | 0.883 |
| Low recall | 0.334 |
| Moderate recall | 0.519 |
| High recall (pooled) | **0.470** |
| 10-fold CV composite (from sweep) | 0.618 |

---

## Key Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| LOFO instead of k-fold | Avoids spatial autocorrelation within fires; tests generalization to unseen fires |
| Balanced weights per fold | Each fold's training class distribution changes when a fire is held out; recomputing weights maintains consistent emphasis on minority classes |
| Train final model on ALL data | LOFO is for evaluation only; all data should be used for the deployed model |
| TF SavedModel wrapper | GEE's hosted inference requires TF format; `tf.py_function` bridges XGBoost to TF graph |
| `serving_default` signature | Required by GEE's `ee.Model.fromAiPlatformPredictor` API |
| Feature names in metadata JSON | GEE band stacking must match training feature order exactly |
| Pooled κ as headline metric | Weighted by sample count; representative of performance on the full pixel distribution rather than per-fire averages |

---

## Data Flow

```
data/real_all_fires_complete_covariates_fixed_1229.csv  ─┐
data/real_all_fires_upsampled_points_with_covariates_*.csv ─┘
         │
         ├──► [LOFO CV — 52 folds]
         │         │
         │         ├──► results_xgb_lr01_top30/lofo_per_fire_results.csv
         │         ├──► results_xgb_lr01_top30/lofo_per_fire_kappa.png
         │         ├──► results_xgb_lr01_top30/lofo_confusion_matrix.png
         │         ├──► results_xgb_lr01_top30/lofo_kappa_distribution.png
         │         └──► results_xgb_lr01_top30/lofo_kappa_vs_high_recall.png
         │
         └──► [Final training on ALL data]
                   │
                   ├──► gee_models_xgb_lr01_top30/xgb_lr01_top30.joblib
                   ├──► gee_models_xgb_lr01_top30/xgb_lr01_top30_savedmodel/
                   ├──► gee_models_xgb_lr01_top30/model_metadata.json
                   └──► gee_models_xgb_lr01_top30/scaler.pkl
                                  │
                                  ▼
                         GEE Colab notebook
                         (wall-to-wall burn severity maps)
```

---

## Dependencies

| Library | Usage |
|---------|-------|
| `xgboost` | `XGBClassifier` — the final model |
| `scikit-learn` | `compute_sample_weight`, `cohen_kappa_score`, `confusion_matrix`, `StandardScaler` |
| `tensorflow` | `tf.Module`, `tf.function`, `tf.py_function`, `tf.saved_model.save` |
| `joblib` | Serializing XGBoost model to `.joblib` |
| `pickle` | Serializing the scaler to `.pkl` |
| `pandas` / `numpy` | Data handling, metric arrays |
| `matplotlib` / `seaborn` | Diagnostic visualizations |

---

*Documentation generated for thesis pipeline — this is the fourth and final script in the sequence: `01_feature_importance.py` → `02_rf_tuning.py` → `03_compare_model_matrices_top8.py` → `04_xgb_cvlofo.py`.*
