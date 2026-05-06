# Evaluation & Statistics

## Primary Metric: Pearson r

Encoding performance is measured as the **Pearson correlation coefficient** between predicted and measured fMRI responses across held-out test stimuli, computed **per voxel**. For reporting, voxel-wise r values are averaged across the selected voxel pool (top 20% by noise ceiling), yielding one scalar per (subject, condition) cell.

---

## Noise Ceiling

The noise ceiling (NC) quantifies the maximum theoretically achievable encoding accuracy given the signal-to-noise characteristics of the fMRI data. It accounts for measurement noise and sets a benchmark against which encoding model performance can be interpreted.

### NCSNR Framework (Allen et al., 2022)

For each voxel, the noise ceiling signal-to-noise ratio is:

```text
NCSNR = σ_signal / σ_noise
```

where:

- **σ_noise** = standard deviation of a voxel's response across repeated presentations of the same stimulus, averaged over all stimuli
- **σ_signal** = estimated standard deviation of the true (noiseless) signal = sqrt(max(0, σ_total² − σ_noise²))

### Converting NCSNR to Noise Ceiling Percentage

```text
NC% = 100 × NCSNR² / (NCSNR² + 1/n_averages)
```

where `n_averages = 6` (number of stimulus repetitions). This formulation accounts for the benefit of averaging — a model evaluated on averaged responses can achieve higher accuracy than on single trials.

### Converting to Correlation Units

```text
r_NC = sqrt(NC% / 100)
```

This converts the NC percentage to the same scale as Pearson r, enabling direct comparison.

### Modality-Specific Noise Ceiling

Noise ceilings are computed **separately** for image-evoked and text-evoked fMRI responses, using only the trials from the corresponding modality. This ensures the NC accurately reflects the signal reliability for each condition.

---

## Noise-Ceiling-Normalised Accuracy

The primary reported metric is:

```text
r_norm = r / r_NC
```

A value of 1.0 means the encoding model explains as much variance as is theoretically explainable given the data's SNR. This normalisation is essential for comparing performance across brain regions with different intrinsic reliability levels.

---

## Group-Level Statistics

### Within-Subject Summary

For each subject and condition, the mean (and optionally standard deviation) of per-voxel r and r_norm are computed over the selected voxel pool, yielding `summary.csv`.

### Group-Level Aggregation

Across subjects, the group-level estimate is the mean ± standard deviation of the per-subject scalar values, stored in `aggregated.csv`. N = 8 subjects.

### Significance Testing

Two approaches are available:

**Approach A — Averaged permutation p-values:**
Subject-level p-values from permutation tests (see [Neural Encoding](04_neural_encoding.md)) are averaged across subjects. This reflects the probability of observing the mean encoding accuracy under the null hypothesis of no brain–model correspondence.

**Approach B — Sign-flip test on subject values:**
A one-sample sign-flip permutation test (H₀: group mean = 0) is applied to the vector of per-subject mean-r values:

1. Generate 10,000 random sign-flip patterns (each subject's value is randomly negated or kept).
2. Compute the group mean for each permuted set.
3. p-value = proportion of permuted means ≥ observed mean.

This non-parametric test is robust with N = 8 subjects where normality cannot be assumed.

### Significance Thresholds

| Stars | p-value |
| --- | --- |
| *** | p < 0.001 |
| ** | p < 0.01 |
| * | p < 0.05 |
| ns | p ≥ 0.05 |

---

## Cross-Modal Comparison

The key statistical comparison is between within-modality and cross-modal conditions. Because all four conditions are measured in the same subjects, a **paired comparison** (e.g., paired t-test or Wilcoxon signed-rank) is used to test:

- `image→image` vs. `image→text` (does cross-modal transfer reduce performance?)
- `text→text` vs. `text→image`
- `image→text` vs. `text→image` (asymmetry between cross-modal directions)

Multiple comparisons across conditions are corrected with Bonferroni or FDR correction.
