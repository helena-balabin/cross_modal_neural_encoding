# Configuration

All pipelines are configured via **Hydra 1.3+** YAML files under `configs/`. No parameters are hardcoded in the source files.

---

## Neural Encoding Config

**File:** `configs/modeling/neural_encoding.yaml`

| Parameter | Default | Description |
| --- | --- | --- |
| `subjects` | sub-02…sub-09 | List of subject IDs |
| `sessions` | 1, 2, 3 | Sessions per subject |
| `n_runs` | 12 | Runs per session |
| `model` | Qwen3.5-0.8B | VLM model label |
| `vision_layer` | 6 | Vision encoder layer to use |
| `text_layer` | 12 | Language model layer to use |
| `pca_components` | 50 | PCA dimensionality reduction |
| `voxel_selection_pct` | 20 | Top % voxels retained by NC |
| `n_outer_folds` | 5 | Outer CV folds |
| `n_inner_folds` | 5 | Inner CV folds (hyperparameter selection) |
| `frac_grid` | [0.1, 0.25, 0.5, 0.75, 1.0] | Fractional ridge search grid |
| `n_permutations` | 100 | Permutation test shuffles |
| `glmsingle_dir` | `comp_fmri_study_2025/glmsingle` | Path to GLMsingle outputs |
| `fmriprep_dir` | `comp_fmri_study_2025/fmriprep` | Path to fMRIPrep outputs |
| `bids_dir` | `comp_fmri_study_2025/bids` | Path to BIDS root |
| `embeddings_dir` | `models/embeddings` | Path to extracted embeddings |
| `design_mapping_file` | `data/design_matrix_mapping.csv` | Condition → modality mapping |

---

## Feature Extraction Config

**File:** `configs/modeling/extract_embeddings.yaml`

| Parameter | Default | Description |
| --- | --- | --- |
| `model` | Qwen3.5-0.8B | HuggingFace model identifier |
| `metadata_csv` | `data/metadata.csv` | Image/caption metadata |
| `image_dir` | `data/coco_images` | COCO image directory |
| `text_col` | `text` | Caption column name |
| `image_col` | `filepath` | Image filepath column name |
| `device` | `cuda` | Compute device |
| `dtype` | `bfloat16` | Model precision |
| `pooling` | `mean` | Pooling strategy |
| `vision_layers` | [6] | Vision layers to extract |
| `text_layers` | [12] | Text layers to extract |
| `batch_size` | 8 | Text tokenisation batch size |
| `max_length` | 512 | Max token length |

---

## Visualization Configs

**Noise ceiling:** `configs/visualization/visualize_noise_ceiling.yaml`

| Parameter | Default | Description |
| --- | --- | --- |
| `subjects` | sub-02…sub-09 | Subjects to visualize |
| `nc_percentiles` | [10, 20, 30] | Threshold percentiles for surface maps |
| `nc_num_averages` | 6 | Stimulus repetitions (for NC formula) |
| `output_dir` | `reports/figures/noise_ceiling` | Figure output path |

**Encoding results:** `configs/visualization/visualize_encoding_results.yaml`

| Parameter | Default | Description |
| --- | --- | --- |
| `metric` | `mean_normalized_r` | Metric to plot |
| `significance_n_permutations` | 10000 | Sign-flip test iterations |
| `font_scale` | 1.2 | Seaborn font scale |
| `figure_size` | [10, 6] | Figure dimensions in inches |

---

## Execution

```bash
# Extract embeddings
python -m cross_modal_neural_encoding.extract_embeddings

# Run neural encoding (all subjects and conditions)
python -m cross_modal_neural_encoding.neural_encoding

# Visualize noise ceiling
python -m cross_modal_neural_encoding.visualize_noise_ceiling

# Visualize encoding results
python -m cross_modal_neural_encoding.visualize_encoding_results
```

Override any parameter at runtime with Hydra syntax:

```bash
python -m cross_modal_neural_encoding.neural_encoding \
  model=Qwen2.5-VL-7B \
  pca_components=100 \
  n_permutations=500
```

---

## Reproducibility

- **Package manager:** `uv` with a locked `uv.lock` file — exact dependency versions are pinned.
- **Python version:** 3.13
- **Random seeds:** Set for PCA and ridge regression where applicable.
- **Config logging:** Hydra writes a `.hydra/` directory alongside each run output, recording the full resolved config for every execution.
