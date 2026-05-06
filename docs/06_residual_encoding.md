# Residual Neural Encoding

## Overview

Residual neural encoding (§3.3) provides a causal test of whether the
compositional structural properties identified by [structural probing](structural_analysis.md)
are what carry cross-modal brain alignment.  The intuition: if structure is the
substrate of cross-modal correspondence, then VLM embeddings stripped of their
structural component should **retain within-modality alignment** but **lose
cross-modal alignment**.

Implemented in `cross_modal_neural_encoding/modeling/residual_encoding.py`.

```bash
python -m cross_modal_neural_encoding.modeling.residual_encoding
```

Config: `configs/modeling/residual_encoding.yaml`

---

## Method

### Structural feature vector

For each stimulus we construct **s** ∈ ℝ³:

| Dimension | Text stimuli | Image stimuli |
|---|---|---|
| node count | `amr_n_nodes` | `coco_a_nodes` |
| edge count | `amr_n_edges` | `coco_a_edges` |
| graph depth | `amr_graph_depth` | `coco_a_graph_depth` |

The same three properties are used as probing targets (§3.2), so the
residualisation removes exactly the information that is linearly recoverable
from the structural targets.

### Residualisation (Equation 1)

On the **training split** of each outer CV fold, we fit a ridge regression
**W\*** from **s** to each dimension of the PCA-reduced VLM embedding **e**:

$$\hat{\mathbf{e}} = \mathbf{W}^* \mathbf{s}, \quad
\mathbf{W}^* = \arg\min_{\mathbf{W}}
\|\mathbf{e}_\text{train} - \mathbf{W}\mathbf{s}_\text{train}\|^2_F
+ \lambda \|\mathbf{W}\|^2_F$$

The structure-residualised embedding **ẽ** = **e** − **ê** retains only the
components of **e** that cannot be linearly explained by **s**.  W\* is applied
to the test split using the training-split fit only — no test information leaks
into the residualisation step.

### Full pipeline re-run

The full [encoding pipeline](04_neural_encoding.md) is re-run with **ẽ** in
place of **e** across all four conditions of the 2×2 design (text→text,
image→image, image→text, text→image).

### Permuted-s control

To rule out that accuracy changes are caused by generic variance reduction
rather than removal of structurally meaningful information, an additional
control run uses a randomly permuted **s** (stimuli shuffled before fitting
**W\***).  Since permuted **s** carries no structural signal, this measures the
baseline accuracy change attributable to the residualisation procedure alone.

---

## Hypothesis

| Condition | Predicted effect |
|---|---|
| Text → text (within) | Minimal drop — modality-specific content survives |
| Image → image (within) | Minimal drop |
| Text → image (cross) | **Substantial drop** — structure carries cross-modal alignment |
| Image → text (cross) | **Substantial drop** |
| All conditions, permuted control | No systematic drop |

A selective drop in cross-modal but not within-modality accuracy is the
expected signature that compositional structure is the principal carrier of
cross-modal brain alignment.

---

## Outputs

Results are written to `outputs/residual_encoding/<model>/`.  The directory
layout mirrors the [standard encoding outputs](04_neural_encoding.md#outputs).
Condition names for the residual runs use the original condition labels
(e.g. `text_to_image`); permuted-control conditions are prefixed `permuted_`
(e.g. `permuted_text_to_image`).

---

## Visualisation

```bash
python -m cross_modal_neural_encoding.visualization.visualize_residual_encoding \
    standard_summary=outputs/neural_encoding/<model>/summary.csv \
    residual_summary=outputs/residual_encoding/<model>/summary.csv
```

Produces two figures:

| Figure | Description |
|---|---|
| `residual_comparison.png` | Grouped bars: standard / residualised / permuted per condition |
| `ablation_delta.png` | Δ(residualised − standard) per condition, highlighting the asymmetric drop |
