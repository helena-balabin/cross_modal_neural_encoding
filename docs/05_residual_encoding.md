# Residual Neural Encoding

## Overview

Cross-modal encoding is asymmetric: text embeddings predict image-evoked fMRI
reasonably well, but image embeddings predict text-evoked fMRI poorly. The
embedding-to-embedding analysis ([Predict Modalities](10_predict_modalities.md))
shows a matching asymmetry — vision is more recoverable from text than text is
from vision. Residual neural encoding asks whether these are the same phenomenon
by removing the **cross-modally shared** component of an embedding (the part
linearly predictable from the other modality) and re-running the encoding model:

> Does removing modality-shared information hurt encoding performance, and if so,
> does it hurt the **cross-modal** prediction more than the **within-modality**
> prediction?

If the shared component is what carries cross-modal brain alignment, removing it
should hurt cross-modal encoding selectively, while within-modality encoding —
which can still draw on modality-private information — is comparatively spared.
This is an ablation-based necessity test on the representations, not an
intervention on the brain.

Implemented in `cross_modal_neural_encoding/modeling/residual_encoding.py`.

```bash
python -m cross_modal_neural_encoding.modeling.residual_encoding
```

Config: `configs/modeling/residual_encoding.yaml`

---

## Method

### Cross-modal residual feature

Each stimulus has both a text embedding and an image embedding (PCA-reduced, the
same reduction used in the standard encoding pipeline). For each condition, the
embedding being encoded is residualized against the **other** modality's
embedding for the same stimulus — exactly the linear text→vision / vision→text
mapping used in [Predict Modalities](10_predict_modalities.md):

| Embedding being encoded | Residual feature **r** | Removed component |
| --- | --- | --- |
| text | image embedding | image-derivable (shared) part of text |
| vision (image) | text embedding | text-derivable (shared) part of image |

### Residualization

On the **training split** of each outer CV fold, we fit a ridge regression
**W\*** from the residual feature **r** (the paired other-modality embedding) to
each dimension of the embedding **e**:

$$\hat{\mathbf{e}} = \mathbf{W}^* \mathbf{r}, \quad
\mathbf{W}^* = \arg\min_{\mathbf{W}}
\|\mathbf{e}_\text{train} - \mathbf{W}\mathbf{r}_\text{train}\|^2_F
+ \lambda \|\mathbf{W}\|^2_F$$

The residualized embedding **ẽ** = **e** − **ê** retains only the components of
**e** that **cannot** be linearly predicted from the other modality — i.e., the
modality-private information. W\* is applied to the test split using the
training-split fit only — no test information leaks into the residualization step.

### Residualization side: embedding vs fMRI

The `residual_side` config flag selects *which* side of the encoding is
residualized against the other modality's embedding **r**:

| `residual_side` | Residualized quantity | Interpretation |
| --- | --- | --- |
| `embedding` (default) | **ẽ** = **e** − W\*·**r** | remove the part of the *embedding* predictable from the other modality |
| `fmri` | **Ỹ** = **Y** − V\*·**r** | remove the part of the *fMRI* predictable from the other modality, e.g. *image fMRI − image fMRI predicted from text embeddings* |

The mechanics are identical — a ridge regression fit on the training split only
(`Ridge(alpha=residual_alpha)`), applied to both train and test — only the
regression *target* changes (the embedding **e** vs the fMRI **Y**). For the
fMRI-side variant V\* is fit from **r** to the multi-voxel fMRI, and the residual
**Ỹ** replaces **Y** as the encoding target while the embedding **e** is encoded
unchanged. Both variants run across all four conditions of the 2×2 design.

> **Noise-ceiling caveat.** The noise ceiling is always computed on the original
> betas. For the fMRI-side variant, `mean_normalized_r` therefore normalises the
> encoding of *residualized* Y by the *original* noise ceiling — the same
> convention as the embedding-side variant, kept for comparability.

The fMRI-side variant is run via a thin override config that only changes
`residual_side` and `output_dir`:

```bash
python -m cross_modal_neural_encoding.modeling.residual_encoding \
    --config-name residual_encoding_fmri
```

### Full pipeline re-run

The full [encoding pipeline](04_neural_encoding.md) is re-run with **ẽ** in
place of **e** across all four conditions of the 2×2 design (text→text,
image→image, image→text, text→image).

### Permuted control

As a control, an additional run permutes the residual feature **r** across stimuli
before fitting **W\***. A stimulus-misaligned **r** carries no cross-modal signal,
so **W\*** collapses toward the mean and little stimulus-specific variance is
removed. This isolates accuracy changes caused by the residualization *procedure*
itself (fitting and subtracting a regressor at all) from those caused by removing
genuine cross-modal information. Note it does **not** match the *amount* of
variance removed by the real residualization — that confound is handled instead by
the within-row design in the hypothesis below.

---

## Hypothesis

Residualization is determined by the **embedding** modality alone (text loses its
vision-predictable part; vision loses its text-predictable part), so the two
conditions that share an embedding modality undergo the *identical* ablation and
differ only in their fMRI target. The test is therefore a within-row comparison of
the encoding drop on the cross-modal vs the within-modality target:

| Embedding | Within-modality target | Cross-modal target | Predicted contrast |
| --- | --- | --- | --- |
| text | text → text | text → image | larger drop for text → image |
| vision | image → image | image → text | larger drop for image → text |

Because the same residualized embedding feeds both conditions in a row, the
*amount* of information removed is held constant by construction — a selectively
larger drop on the cross-modal target cannot be explained by "too much variance
was removed," only by the removed shared component mattering specifically for
cross-modal prediction. The reported quantity is the difference of the two drops
within each row.

Two caveats:

- **Floor effect.** `image → text` is already near chance, so its drop is bounded;
  the informative signal is expected mainly in the text row (`text → image`).
- **Permuted control.** As above, it controls for artifacts of the residualization
  procedure, not for the magnitude of variance removed — the within-row design is
  what controls for magnitude.

---

## Outputs

Results are written to `outputs/residual_encoding/<model>/` (embedding-side) or
`outputs/residual_encoding_fmri/<model>/` (fMRI-side). The directory layout
and file set mirror the [standard encoding outputs](04_neural_encoding.md#outputs)
(`per_voxel_r.npy`, `noise_ceiling.npy`, `voxel_keep.npy`,
`best_frac_per_voxel.npy`, NIfTIs, plus top-level `summary.csv` / `aggregated.csv`),
so the two analyses are directly comparable. Condition names for the residual runs
use the original condition labels (e.g. `text_to_image`); permuted-control
conditions are prefixed `permuted_` (e.g. `permuted_text_to_image`). Both
`residual_side` variants share the same labels and schema, differing only in the
output directory.

---

## Visualisation

```bash
python -m cross_modal_neural_encoding.visualization.visualize_residual_encoding \
    standard_summary=outputs/neural_encoding/<model>/summary.csv \
    residual_summary=outputs/residual_encoding/<model>/summary.csv
```

For the fMRI-side variant, point it at the fMRI-side outputs:

```bash
python -m cross_modal_neural_encoding.visualization.visualize_residual_encoding \
    --config-name visualize_residual_encoding_fmri
```

Run this after the main neural-encoding visualisation; it reads the standard and
residual `summary.csv` files and produces two figures:

| Figure | Description |
| --- | --- |
| `residual_encoding_bars.png` | Residualized encoding accuracy in the **same** per-model bar-plot format as the main neural-encoding figure (reuses `plot_encoding_results`), so the two are directly comparable |
| `ablation_delta.png` | One bar per condition, the residualization effect as a **percentage** of the original encoding performance (`(residualized − standard) / standard × 100`, using the group-mean original as the denominator), so a drop is a downward (negative) bar; conditions are grouped by embedding modality (within then cross) |

Both figures also carry **pairwise significance brackets** between the four conditions, from a two-sided Wilcoxon signed-rank test (`scipy.stats.wilcoxon`) on the per-model means paired by model, BH-FDR corrected across the six condition pairs (one bracket level per pair). These are distinct from the per-bar `*`/`ns` stars, which test each bar against chance.
