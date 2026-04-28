# Neural Encoding

## Overview

For each **(subject, condition)** pair, we fit a **voxel-wise linear encoding model** that maps VLM embeddings to single-trial fMRI betas. The model is evaluated using Pearson correlation between predicted and measured responses on held-out stimuli.

Implemented in `cross_modal_neural_encoding/neural_encoding.py`.

---

## Input Alignment

Before fitting, embeddings and betas are aligned trial-by-trial:

1. Parse `events.tsv` files (in session → run order) to build a dataframe of:
   `(beta_index, coco_id, modality, run_label)` for every non-blank trial.
2. Load the corresponding VLM embeddings indexed by `coco_id`.
3. Separate trials into the four conditions by crossing `{embedding_modality} × {response_modality}`.

For cross-modal conditions (e.g., image→text), the **brain response** comes from text-presentation runs while the **embedding** comes from the vision encoder for the matching image stimulus.

---

## Dimensionality Reduction

Before regression, VLM embeddings (dimension D ≈ 1024–4096) are reduced with **PCA**:

- **Components:** 50 (default)
- **Fitting:** PCA is fit on the training split only; the same projection is applied to the test split.
- **Centering only:** Feature variance is not rescaled after PCA (centering only), because PCA already orders components by variance.

This reduces the feature-to-trial ratio and regularises the regression problem.

---

## Encoding Model

### Algorithm: Fractional Ridge Regression

Ridge regression minimises:

```text
min_B  ||Y - X B||² + α ||B||²
```

where Y is (n_trials × n_voxels), X is (n_trials × 50 PCA features), and α is the ridge penalty.

**Fractional ridge** (Rokem & Kay, 2020) reparametrises α as a fraction of the maximum-eigenvalue regularisation:

```text
α = frac × λ_max
```

This makes the regularisation scale-invariant and places the search grid in [0, 1] with interpretable endpoints (0 = no shrinkage, 1 = maximal shrinkage). A grid of `frac ∈ {0.1, 0.25, 0.5, 0.75, 1.0}` is searched per voxel.

### Per-Voxel Optimisation

Each voxel independently selects its best `frac` via the inner cross-validation loop. This respects the heterogeneity of signal-to-noise across brain regions.

---

## Cross-Validation Design

A **nested group cross-validation** scheme is used to:

- Prevent stimulus leakage (repeated stimuli must stay in the same fold).
- Select regularisation hyperparameters without inflating test-set performance.

### Outer Loop (model evaluation)

- **Splitter:** `GroupKFold(n_splits=5)`, groups = stimulus IDs
- **Purpose:** Unbiased estimate of encoding accuracy
- **Alternative (single holdout):** `GroupShuffleSplit(test_size=0.2)` when `n_outer_folds=1`

### Inner Loop (hyperparameter selection)

- **Splitter:** `GroupKFold(n_splits=5)` on the training portion of the outer fold
- **Purpose:** Select best `frac` per voxel without touching the outer test set

### Test-Set Evaluation

After fitting on the outer training set with the best per-voxel `frac`:

1. Average repeated-stimulus responses within the test set (reducing noise).
2. Predict test-stimulus responses with the fitted model.
3. Compute **Pearson r** between predicted and measured responses, per voxel.

The averaging step mirrors the noise ceiling computation (which also averages repeats) and aligns the evaluation metric with the reliability baseline.

---

## Permutation Test

An optional permutation test is run to validate that encoding accuracy exceeds chance:

1. Shuffle the mapping between stimulus IDs and embeddings (breaking the true correspondence).
2. Run the full encoding pipeline on the shuffled data.
3. Repeat `n_permutations` times (default 100), parallelised across CPUs.
4. Compute a **p-value**: `p = (#{null ≥ observed} + 1) / (n_permutations + 1)`

The null distribution reflects the expected accuracy under no true brain–model alignment.

---

## Outputs

For each **(subject, condition)**:

| File | Shape | Description |
| --- | --- | --- |
| `per_voxel_r.npy` | (n_voxels,) | Pearson r per voxel |
| `noise_ceiling.npy` | (n_voxels,) | NC in correlation units per voxel |
| `best_frac_per_voxel.npy` | (n_voxels,) | Selected regularisation fraction |
| `null_mean_r.npy` | (n_permutations,) | Null distribution (if run) |

Aggregated across subjects:

| File | Description |
| --- | --- |
| `summary.csv` | Per-subject, per-condition mean r and mean normalised r |
| `aggregated.csv` | Group-level mean ± std per condition |
| `noise_ceiling_subject_stats.csv` | Cross-subject NC consistency check |

---

## Voxel Selection

Before encoding, an optional voxel selection step retains the **top K% of voxels** ranked by noise ceiling, separately for each modality:

- **Default:** top 20% per modality
- **Purpose:** Focuses the analysis on the most reliable voxels and reduces multiple comparisons
- **Implementation:** The noise ceiling is computed on the full brain mask; the K% threshold is applied per modality before fitting encoding models
