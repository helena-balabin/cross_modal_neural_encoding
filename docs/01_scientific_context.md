# Scientific Context

## Research Question

Does the joint visual-linguistic representational space of a Vision-Language Model (VLM) predict neural responses in a **cross-modal** manner — i.e., can image embeddings predict brain responses to verbal descriptions of the same scenes, and vice versa?

This question probes whether VLMs and the human brain share a **modality-invariant** representational structure for concept-level content, or whether their representations remain modality-specific even when the underlying semantic content is matched.

---

## Experimental Logic

The study employs a **2 × 2 factorial encoding design**:

- **Embedding modality** (rows): whether the predictor features come from the vision encoder or the text encoder of the VLM.
- **Brain response modality** (columns): whether the fMRI responses being predicted were evoked by viewing an image or reading the corresponding caption.

| Condition | Embedding source | fMRI response | Interpretation |
| --- | --- | --- | --- |
| **image→image** | Vision encoder | Image-evoked | Within-modality baseline |
| **text→text** | Text encoder | Text-evoked | Within-modality baseline |
| **image→text** | Vision encoder | Text-evoked | Cross-modal: image features → verbal brain |
| **text→image** | Text encoder | Image-evoked | Cross-modal: verbal features → visual brain |

Within-modality conditions (image→image, text→text) establish the ceiling performance achievable when embedding and stimulus modalities match. Cross-modal conditions (image→text, text→image) reveal how much of that performance is retained when the modalities are swapped — the signature of a truly shared representation.

---

## Hypotheses

**H1 (within-modality performance):** Both image→image and text→text encoding will yield above-chance prediction accuracy across sensory and association cortices.

**H2 (cross-modal transfer):** Significant cross-modal encoding (image→text, text→image) will be found in higher-order cortical areas (e.g., lateral occipital cortex, ventral temporal cortex, language-related regions), reflecting semantic content shared between modalities.

**H3 (asymmetry):** Cross-modal performance will be lower than the corresponding within-modality baseline, and the degree of reduction will differ between cross-modal directions, reflecting the asymmetric integration capabilities of vision-dominant vs. language-dominant neural systems.

---

## Stimuli Rationale

The Microsoft COCO dataset provides **matched image–caption pairs**: each image is accompanied by multiple human-written descriptions of the same scene. This pairing is essential because it grounds the cross-modal comparison in a controlled semantic equivalence — the same conceptual content appears in both modalities, differing only in sensory format.

Using 252 unique stimuli (each repeated ~6 times) provides both statistical power for noise ceiling estimation and enough trials to fit and cross-validate encoding models reliably.

---

## Related Work & Positioning

This work builds on the neural encoding framework (Huth et al., 2016; Allen et al., 2022) and recent cross-modal alignment studies using language models (Scotti et al., 2023; Ozcelik & VanRullen, 2023). The novelty is the **systematic 2×2 design** that isolates cross-modal transfer at the level of VLM internal representations, rather than comparing separate unimodal models.

The use of **noise-ceiling normalisation** (as in the Natural Scenes Dataset benchmark) ensures that performance differences reflect genuine representational structure rather than differences in data reliability across conditions.
