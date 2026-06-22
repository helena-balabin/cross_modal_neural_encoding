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
embedding being encoded is residualised against the **other** modality's
embedding for the same stimulus — exactly the linear text→vision / vision→text
mapping used in [Predict Modalities](10_predict_modalities.md):

| Embedding being encoded | Residual feature **r** | Removed component |
| --- | --- | --- |
| text | image embedding | image-derivable (shared) part of text |
| vision (image) | text embedding | text-derivable (shared) part of image |

### Residualisation

On the **training split** of each outer CV fold, we fit a ridge regression
**W\*** from the residual feature **r** (the paired other-modality embedding) to
each dimension of the embedding **e**:

$$\hat{\mathbf{e}} = \mathbf{W}^* \mathbf{r}, \quad
\mathbf{W}^* = \arg\min_{\mathbf{W}}
\|\mathbf{e}_\text{train} - \mathbf{W}\mathbf{r}_\text{train}\|^2_F
+ \lambda \|\mathbf{W}\|^2_F$$

The residualised embedding **ẽ** = **e** − **ê** retains only the components of
**e** that **cannot** be linearly predicted from the other modality — i.e., the
modality-private information. W\* is applied to the test split using the
training-split fit only — no test information leaks into the residualisation step.

### Full pipeline re-run

The full [encoding pipeline](04_neural_encoding.md) is re-run with **ẽ** in
place of **e** across all four conditions of the 2×2 design (text→text,
image→image, image→text, text→image).

### Permuted control

As a control, an additional run permutes the residual feature **r** across stimuli
before fitting **W\***. A stimulus-misaligned **r** carries no cross-modal signal,
so **W\*** collapses toward the mean and little stimulus-specific variance is
removed. This isolates accuracy changes caused by the residualisation *procedure*
itself (fitting and subtracting a regressor at all) from those caused by removing
genuine cross-modal information. Note it does **not** match the *amount* of
variance removed by the real residualisation — that confound is handled instead by
the within-row design in the hypothesis below.

---

## Hypothesis

Residualisation is determined by the **embedding** modality alone (text loses its
vision-predictable part; vision loses its text-predictable part), so the two
conditions that share an embedding modality undergo the *identical* ablation and
differ only in their fMRI target. The test is therefore a within-row comparison of
the encoding drop on the cross-modal vs the within-modality target:

| Embedding | Within-modality target | Cross-modal target | Predicted contrast |
| --- | --- | --- | --- |
| text | text → text | text → image | larger drop for text → image |
| vision | image → image | image → text | larger drop for image → text |

Because the same residualised embedding feeds both conditions in a row, the
*amount* of information removed is held constant by construction — a selectively
larger drop on the cross-modal target cannot be explained by "too much variance
was removed," only by the removed shared component mattering specifically for
cross-modal prediction. The reported quantity is the difference of the two drops
within each row.

Two caveats:

- **Floor effect.** `image → text` is already near chance, so its drop is bounded;
  the informative signal is expected mainly in the text row (`text → image`).
- **Permuted control.** As above, it controls for artifacts of the residualisation
  procedure, not for the magnitude of variance removed — the within-row design is
  what controls for magnitude.

---

## Outputs

Results are written to `outputs/residual_encoding/<model>/`. The directory layout
and file set mirror the [standard encoding outputs](04_neural_encoding.md#outputs)
(`per_voxel_r.npy`, `noise_ceiling.npy`, `voxel_keep.npy`,
`best_frac_per_voxel.npy`, NIfTIs, plus top-level `summary.csv` / `aggregated.csv`),
so the two analyses are directly comparable. Condition names for the residual runs
use the original condition labels (e.g. `text_to_image`); permuted-control
conditions are prefixed `permuted_` (e.g. `permuted_text_to_image`).

---

## Visualisation

```bash
python -m cross_modal_neural_encoding.visualization.visualize_residual_encoding \
    standard_summary=outputs/neural_encoding/<model>/summary.csv \
    residual_summary=outputs/residual_encoding/<model>/summary.csv
```

Produces two figures:

| Figure | Description |
| --- | --- |
| `residual_comparison.png` | Grouped bars: standard / residualised / permuted per condition |
| `ablation_delta.png` | Δ(residualised − standard) per condition, highlighting the asymmetric drop |
