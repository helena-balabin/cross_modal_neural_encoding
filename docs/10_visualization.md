# Visualization

## Noise Ceiling Surface Maps

**Script:** `cross_modal_neural_encoding/visualization/visualize_noise_ceiling.py`

**Purpose:** Show the spatial distribution of signal reliability across the cortical surface, separately for image-evoked and text-evoked responses.

### Processing Steps

1. Load GLMsingle betas and parse stimulus order from DESIGNINFO.
2. Normalize betas per run (z-score within run, per voxel).
3. Compute modality-specific noise ceiling (NCSNR → NC%) for text and image trials separately.
4. Load subject-native FreeSurfer surfaces from fMRIPrep:
   - **Pial surface:** used for volume-to-surface projection (sampling voxel values at the grey-matter surface)
   - **Inflated surface:** used for visualization (unfolds sulci for visibility)
   - **Sulcal depth map:** used as a background grayscale texture
5. Project volumetric NC maps to the pial surface via `nilearn.surface.vol_to_surf`.
6. Plot the pial-sampled values on the inflated mesh for display.
7. Apply percentile thresholds from config (`percentiles`, default [10, 20, 30]).

### Figure Layout

A 4-row × N-column subplot grid (where N = number of thresholds):

| Row | Hemisphere | Modality |
| --- | --- | --- |
| 1 | Left | Text NC |
| 2 | Right | Text NC |
| 3 | Left | Image NC |
| 4 | Right | Image NC |

- **Colormap:** Blues for text, Reds for image
- **View:** Lateral
- **Output:** `reports/figures/noise_ceiling/sub-{id}_noise_ceiling_modality_overlay_modality-overlay.png`

---

## Encoding Results Bar Charts

**Script:** `cross_modal_neural_encoding/visualization/visualize_encoding_results.py`

**Purpose:** Summarise group-level encoding performance across the four conditions with statistical annotations.

### Data Input

- `aggregated.csv`: Multi-index **column** header with `(metric, statistic)` pairs; condition is the index
- Optional: `summary.csv` for per-subject individual bars

### Group-Level Plot

- One bar per condition (text→text, image→image, image→text, text→image)
- Bar height = group mean of per-subject mean r (or r_norm)
- Error bars = standard deviation across subjects
- Significance stars from either group-level sign-flip (if `use_group_level_significance=true`) or `p_value_col` in `aggregated.csv`

### Per-Subject Panel (optional)

- Grouped bar chart: one group per condition, one bar per subject
- Subject-specific colours
- Same significance annotations as group-level plot

### Metrics Available

| Metric | Description |
| --- | --- |
| `mean_r` | Raw Pearson r averaged over selected voxels |
| `mean_normalized_r` | r / r_NC averaged over selected voxels |

### Style

- Matplotlib with configured font defaults and `font_scale` multiplier
- Output: `reports/figures/encoding_results.png`

---

## Output Directory Structure

```text
reports/
├── figures/
│   ├── noise_ceiling/
│   │   └── sub-{id}_noise_ceiling.png   (per subject)
│   └── encoding_results.png             (group summary)
```
