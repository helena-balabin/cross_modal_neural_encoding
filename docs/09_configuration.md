# Configuration

All pipelines are configured via **Hydra 1.3+** YAML files under `configs/`. No parameters are hardcoded in the source files.

---

## Neural Encoding Config

**File:** `configs/modeling/neural_encoding.yaml`

| Parameter | Default | Description |
| --- | --- | --- |
| `subjects` | sub-07, sub-03, sub-04, sub-05, sub-06, sub-02, sub-08, sub-09 | List of subject IDs |
| `sessions` | [1, 2, 3] | Sessions per subject |
| `runs_per_session` | 12 | Runs per session |
| `task` | comp | Task name for BIDS events |
| `bids_root` | `/project/def-afyshe-ab/hbalabin/comp_fmri_study_2025/bids` | BIDS root |
| `glmsingle_root` | `/project/def-afyshe-ab/hbalabin/comp_fmri_study_2025/glmsingle` | GLMsingle outputs root |
| `fmriprep_dir` | `/project/def-afyshe-ab/hbalabin/comp_fmri_study_2025/fmriprep` | fMRIPrep outputs root |
| `design_matrix_mapping_file` | `/project/def-afyshe-ab/hbalabin/comp_fmri_study_2025/coco/coco_252_stimuli_local/design_matrix_mapping.csv` | Condition → modality mapping |
| `embeddings_dir` | `/scratch/hbalabin/coco` | Path to extracted embeddings |
| `output_dir` | `outputs/neural_encoding` | Output directory |
| `model` | Qwen/Qwen3.5-0.8B | VLM model label |
| `vision_layer` | 6 | Vision encoder layer to use |
| `text_layer` | 12 | Language model layer to use |
| `n_pca_components` | 50 | PCA dimensionality reduction |
| `nc_top_percent` | 20 | Top % voxels retained by NC |
| `nc_num_averages` | 6 | Stimulus repetitions (for NC formula) |
| `test_size` | 0.2 | Holdout fraction if `n_outer_folds = 1` |
| `n_inner_folds` | 5 | Inner CV folds (hyperparameter selection) |
| `n_outer_folds` | 5 | Outer CV folds |
| `frac_grid` | [0.1, 0.25, 0.5, 0.75, 1.0] | Fractional ridge search grid |
| `n_permutations` | 100 | Permutation test shuffles |
| `n_jobs_permutations` | 0 | Parallel workers (0 = auto) |
| `permutation_cpu_reserve` | 8 | Reserved CPUs when auto-scaling |
| `modality_column` | modality | Events column with trial modality |
| `cocoid_column` | cocoid | Events column with COCO ID |
| `conditions` | text_to_text, image_to_image, image_to_text, text_to_image | Encoding condition mappings |

---

## Feature Extraction Config

**File:** `configs/modeling/extract_embeddings.yaml`

| Parameter | Default | Description |
| --- | --- | --- |
| `models` | [Qwen/Qwen3.5-0.8B] | HuggingFace model identifiers |
| `metadata_path` | `data/coco_metadata/all_metadata_merged.csv` | Image/caption metadata |
| `image_dir` | `data/coco_images` | COCO image directory |
| `image_filename_column` | `filepath` | Image filename column |
| `coco_id_column` | `cocoid_x` | COCO ID column |
| `text_column` | `text` | Caption column name |
| `output_dir` | `models/embeddings` | Output directory |
| `cache_dir` | `cache` | Model cache directory |
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
| `glmsingle_dir` | `/project/def-afyshe-ab/hbalabin/comp_fmri_study_2025/glmsingle` | GLMsingle outputs root |
| `bids_dir` | `/project/def-afyshe-ab/hbalabin/comp_fmri_study_2025/bids` | BIDS root |
| `fmriprep_dir` | `/project/def-afyshe-ab/hbalabin/comp_fmri_study_2025/fmriprep` | fMRIPrep outputs root |
| `design_matrix_mapping_file` | `/project/def-afyshe-ab/hbalabin/comp_fmri_study_2025/coco/coco_252_stimuli_local/design_matrix_mapping.csv` | Condition → modality mapping |
| `subject` | null | Subject to visualize |
| `percentiles` | [10, 20, 30] | Threshold percentiles for surface maps |
| `nc_num_averages` | 6 | Stimulus repetitions (for NC formula) |
| `hemispheres` | [left, right] | Hemispheres to visualize |
| `output_dir` | null | Figure output path |
| `cmap` | hot | Colormap for surfaces |
| `log_level` | INFO | Logging level |

**Encoding results:** `configs/visualization/visualize_encoding_results.yaml`

| Parameter | Default | Description |
| --- | --- | --- |
| `aggregated_csv` | `""` | Path to aggregated CSV (required if `run_dir` is null) |
| `run_dir` | null | Run directory containing model subfolders |
| `summary_csv` | null | Per-subject summary CSV (optional) |
| `metric` | `mean_normalized_r` | Metric to plot |
| `p_value_col` | `p_value_mean_r` | Column name for p-values |
| `alpha` | 0.05 | Significance threshold |
| `use_group_level_significance` | true | Compute group-level p-values via sign-flip |
| `group_sig_permutations` | 10000 | Sign-flip iterations |
| `group_sig_random_state` | 42 | Sign-flip RNG seed |
| `output_path` | null | Figure output path |
| `show_subject_panel` | true | Include per-subject panel |
| `font_scale` | 1.2 | Font scaling factor |
| `compress_normalized_axis` | false | Use symlog for normalized metrics |
| `normalized_axis_linthresh` | 0.08 | Symlog linear threshold |
| `figsize` | [10, 3.6] | Figure dimensions in inches |

---

## Execution

```bash
# Extract embeddings
python -m cross_modal_neural_encoding.modeling.extract_embeddings

# Run neural encoding (all subjects and conditions)
python -m cross_modal_neural_encoding.modeling.neural_encoding

# Visualize noise ceiling
python -m cross_modal_neural_encoding.visualization.visualize_noise_ceiling

# Visualize encoding results
python -m cross_modal_neural_encoding.visualization.visualize_encoding_results
```

Override any parameter at runtime with Hydra syntax:

```bash
python -m cross_modal_neural_encoding.modeling.neural_encoding \
  model="Qwen/Qwen3.5-2B" \
  n_pca_components=100 \
  n_permutations=500
```

---

## Reproducibility

- **Package manager:** `uv` with a locked `uv.lock` file — exact dependency versions are pinned.
- **Python version:** 3.13
- **Random seeds:** Set for PCA and ridge regression where applicable.
- **Config logging:** Hydra writes a `.hydra/` directory alongside each run output, recording the full resolved config for every execution.
