# Predicting Embeddings Across Modalities

## Purpose

This module evaluates how well **text encoder embeddings** can predict **vision encoder embeddings**, and vice versa, across all candidate models. It uses the 252 fMRI stimulus pairs and reports **5-fold cross-validated Pearson correlation**.

Two directions are evaluated:

- **Text → Vision**: predict image encoder representations from text encoder representations
- **Vision → Text**: predict text encoder representations from image encoder representations

---

## Scripts

- **Modeling:** `cross_modal_neural_encoding/modeling/predict_modalities.py`
- **Visualization:** `cross_modal_neural_encoding/visualization/visualize_predict_modalities.py`

---

## Configuration

### Modeling

**File:** `configs/modeling/predict_modalities.yaml`

Key options:

- `embeddings_dir`: root directory with extracted embeddings
- `design_matrix_mapping_file`: restrict to the 252 fMRI stimuli (optional)
- `text_models`, `vision_models`: explicit model lists (empty = auto-discover)
- `text_layer`, `vision_layer`: layer indices to use
- `n_splits`: number of CV folds (default 5)
- `regressor`: `ridge`, `mlp_linear`, or `mlp_relu` (default)
- `ridge_alpha`: ridge regularization strength
- `mlp.*`: skip-connected MLP settings (hidden size, early stopping, device)

### Visualization

**File:** `configs/visualization/visualize_predict_modalities.yaml`

Key options:

- `results_csv`: path to the CSV saved by the modeling script
- `output_dir`: directory for heatmap images
- `text_model_order`, `vision_model_order`: optional row/column ordering

---

## Outputs

The modeling script saves a CSV:

```text
outputs/predict_modalities/predict_modalities_results.csv
```

The visualization script saves two heatmaps:

```text
reports/figures/predict_modalities/text_to_vision_heatmap.png
reports/figures/predict_modalities/vision_to_text_heatmap.png
```
