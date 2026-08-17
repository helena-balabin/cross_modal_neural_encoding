# Visualization

## Color scheme

Color encodes **modality** consistently across the figures: **text = blue**
(`#7EAEDB`), **image/vision = red** (`#E88989`), and **VLMs / cross-modal = purple**
(`#A988C8`) — visually the blue + red mix. **Green** (`#84C895`) is reserved for the
fMRI/brain space in the overview schematic. Consequently the model-family bar
palettes carry modality meaning (`VLM_MODEL_PALETTE` = purple, `VISION_MODEL_PALETTE`
= red, `TEXT_MODEL_PALETTE` = blue in `visualize_encoding_results.py`), the
predict-modality heatmaps are colored by input modality (Text→Vision blue,
Vision→Text red), and the difference map uses a colorblind-safe amber↔purple
diverging scale.

## Noise Ceiling Surface Maps

**Script:** `cross_modal_neural_encoding/visualization/visualize_noise_ceiling.py`

**Purpose:** Show **how much** of the cortex the most reliable voxels cover, separately for image-evoked and text-evoked responses, at several selection levels. By default the maps are warped to MNI so all subjects are directly comparable.

The overlay is a **binary selection mask in one flat colour per modality** (text `#7EAEDB`, image `#E88989`), not a noise ceiling heatmap. That is deliberate: the figure's question is how the covered area grows with the level, and a value ramp answers a different one. A ramp draws the low-noise-ceiling vertices pale — and those are exactly the ones a wider level adds — so the levels render as near-identical however much they actually differ. Flat colour makes area the only variable.

**Run:** `sbatch scripts/visualize_noise_ceiling.sh` (loads `gcc/12.3 ants/2.6.5`; the `ants` PyPI package in the venv is *not* ANTsPy). Hydra overrides pass straight through, e.g. `sbatch scripts/visualize_noise_ceiling.sh space=native subject=sub-03`.

### Processing Steps

1. Load GLMsingle betas and parse stimulus order from DESIGNINFO.
2. Normalize betas per run (z-score within run, per voxel).
3. Compute modality-specific noise ceiling (NCSNR → NC%) for text and image trials separately.
4. If `space: mni`, warp to `MNI152NLin2009cAsym` with the fMRIPrep
   `from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5` composite transform
   (`antsApplyTransforms`), onto the subject's `space-MNI152NLin2009cAsym_boldref.nii.gz` grid —
   fMRIPrep writes the identical template grid for every subject, so the results stay
   voxel-aligned across subjects.
5. Load the display surfaces: fsaverage for `space: mni`, the subject's own fMRIPrep FreeSurfer
   surfaces for `space: native`. In both cases the **pial** surface samples the volume (nilearn's
   default 3 mm ball), the **inflated** surface is drawn, and **sulcal depth** is the background.
6. Project via `nilearn.surface.vol_to_surf` and threshold — see below.

### What Is Shown, and Why It Is Not 20% of the Cortex

The figure marks the **voxels the encoding models are fit on**: `select_top_nc_voxels`, the same
helper `build_fmri_cache` uses to build `voxel_keep`. Cutoffs are computed in native volume space,
before warping, and the warp uses `NearestNeighbor` so the edge of each selection survives
resampling.

At the 20% level this colours roughly **13% of grey matter**, not 20%:

| step | value |
| --- | --- |
| top 20% of **positive-NC** in-brain voxels (only ~57% of the brain mask has NC > 0) | 11.5% of in-brain voxels |
| those voxels are mildly enriched in grey matter (GM prob 0.488 vs 0.407 in-brain) | ~13% of grey matter |

A percentile taken over surface *vertices* instead would colour exactly 20% by construction,
regardless of what the volume contains — a different quantity, and not the models' voxel set.

Because nearest-neighbour warping preserves voxel values, every level is just a threshold on one
warped volume per modality: two warps per subject rather than two per level.

**Coverage is monotone by construction.** A wider level has a lower cutoff, so its voxel set
*contains* the narrower one. A superset can only raise the fraction of each vertex's sampled
neighbourhood that falls inside the selection, so the drawn vertices are nested too. Nothing
filters them afterwards — the display threshold (0.5 on a 0/1 map) only separates selected from
unselected.

**Coverage sensitivity.** A vertex is drawn when at least `selection_coverage` of its sampled
neighbourhood falls inside the selection. This dominates the apparent extent — for sub-03 text at
the 20% level, the same 13% of grey matter renders as:

| `selection_coverage` | ~0 | 0.25 (default) | 0.5 | 0.75 |
| --- | --- | --- | --- | --- |
| vertices coloured | 64% | 19.5% | 8.5% | 3.1% |

0.25 mildly overstates the true extent; 0.5 clearly understates it.

### Figure Layout

Two rows per selection level in `percents` (default `[20, 40, 60]`), so six rows, by
hemisphere × view columns. Levels are separated by a rule and labelled down the left edge. A level
of `100` is allowed and means every positive-NC voxel, labelled `All (NC > 0)`, not literally
every voxel.

| | | L lateral | L medial | R lateral | R medial |
| --- | --- | --- | --- | --- | --- |
| **Top 20%** | Text / Image | ✓ | ✓ | ✓ | ✓ |
| **Top 40%** | Text / Image | ✓ | ✓ | ✓ | ✓ |
| **Top 60%** | Text / Image | ✓ | ✓ | ✓ | ✓ |

Observed vertex coverage, sub-02 left hemisphere text: 19.2% / 39.0% / 60.1%. At `100` it reaches
~80–86%.

- **Colour:** one flat colour per modality (`MODALITY_COLORS`), not a value ramp — see above
- **Views:** configurable (`views`, default lateral + medial)
- **Sizing:** `font_scale` multiplies every font size; `row_spacing` is the gridspec `hspace` and
  must be **negative** to close the gaps, since 3-D axes carry large internal padding
- **Output:** `reports/figures/noise_ceiling/sub-{id}_noise_ceiling_space-MNI152NLin2009cAsym_voxelsel_modality-overlay.png`
  (the `space-…` part is dropped for `space: native`)

`antsApplyTransforms` is a CLI, so each warp stages its input and output in a temporary directory
that is discarded once the data is read back — no intermediate NIfTIs are left behind. If the
warped volumes are ever wanted for a group analysis, `warp_volume_to_mni` is the one place to
change.

### Caveat

fsaverage is MNI305-derived, so the surface rendering inherits a few-millimetre MNI305↔MNI152
offset. The warped volumes themselves are in true MNI152NLin2009cAsym and can be used directly
for any group analysis.

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
- Pairwise significance brackets between the four conditions, from a two-sided Wilcoxon signed-rank test (`scipy.stats.wilcoxon`) on the per-model means paired by model, BH-FDR corrected across the six condition pairs (distinct from the per-bar vs.-chance stars)

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

## Layer-Depth Curves

**Script:** `cross_modal_neural_encoding/visualization/visualize_layer_sweep.py`

**Purpose:** Show how encoding performance depends on **which layer** the embeddings come from. Every other figure uses one hard-coded middle layer per encoder (`floor(n_layers / 2)`); this one re-runs the same analysis at every layer of the best-performing model (Qwen3.5-9B-Base) so the depth dependence, and the position of the peak relative to that middle layer, are visible.

**Data input:** the sweep from `scripts/neural_encoding_layer_sweep.sh` — one `<modality>_layer_<idx>/` directory per layer, each holding a normal `<model>/summary.csv` plus the per-subject `null_mean_r.npy` files. The layer is read from the `embed_layer` / `vision_layer` / `text_layer` columns that the encoding pipeline writes into `summary.csv`, not from the directory name.

**Run:** `sbatch --export=ALL,JOBID=<array job id> scripts/visualize_layer_sweep.sh`, which also copies the CSVs and permutation nulls off `$SCRATCH` first.

### Layout: 2×2, One Condition Per Panel

Rows are the **embedding modality** (which sets the x-axis), columns the **fMRI modality** predicted. The diagonal is within-modality, the off-diagonal cross-modal:

| x-axis | → image fMRI | → text fMRI |
| --- | --- | --- |
| vision, transformer block (27, 0–26) | `image_to_image` | `image_to_text` |
| text, hidden state (33, 0–32) | `text_to_image` | `text_to_text` |

Each condition reads exactly one encoder, and the two encoders have different depths, so there is no single layer axis. The indices are **not** interchangeable either: a vision index is the output of block *i* (forward hook); a text index is `hidden_states[i]`, so index 0 is the input embedding and index *i* is the output of block *i−1*. Each row's x-axis label says which it is, so the label cannot be collapsed onto the bottom row.

**Every panel scales its own y-axis** (`y_limits: null`). The conditions differ by an order of magnitude — `image_to_text` peaks near 0.010 while `image_to_image` reaches 0.088 — and on a shared axis the weak conditions collapse into flat lines at zero, hiding real layer structure. The cost is that panel heights are not comparable by eye, so every panel keeps its own tick labels and the caption states it. Setting `y_limits` applies one range to all four panels when direct comparison is wanted.

### Encoding

- **Colour = fMRI modality predicted**: image red (`#D96F6F`), text blue (`#6F9FC9`) — the palette convention above, carried by the panel title as well as the curve. Green stays reserved for brain space, which is why the four conditions are not given the four `PALETTE` colours here.
- **Linestyle = within-modality (solid) vs cross-modal (dashed)**, redundant with the diagonal/off-diagonal position but kept so a panel lifted out of the grid still says which it is.
- **Shaded band = ±SEM** across subjects (SD with `ddof=1` / √n), matching the bar figures' error definition.
- **Marker fill = significance**: filled where q < `alpha`, hollow otherwise. Same statistic as the bar charts' stars — `combined_perm_group_pvalue`, the Stelzer-style combination of per-subject permutation nulls, BH-FDR corrected across every layer × condition cell in the figure — just re-encoded, since 60 asterisks along a curve are unreadable.
- **Dotted vertical rule** at the middle layer used in the main analysis (`main_analysis_layers`, vision 13 / text 16).

### Output

- `reports/figures/layer_sweep/<model>_layer_sweep.png`
- `reports/figures/layer_sweep/layer_sweep_summary.csv` — per-layer group mean, SEM, `mean_r`, p and q. The peak-layer numbers quoted in the text come from here, not from reading the curve.

---

## Output Directory Structure

```text
reports/
├── figures/
│   ├── noise_ceiling/
│   │   └── sub-{id}_..._voxelsel_modality-overlay.png  (per subject, MNI or native)
│   ├── layer_sweep/
│   │   ├── {model}_layer_sweep.png                     (encoding vs. layer depth)
│   │   └── layer_sweep_summary.csv                     (per-layer mean, SEM, p, q)
│   └── encoding_results.png                            (group summary)
```
