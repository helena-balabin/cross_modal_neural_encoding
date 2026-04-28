# Cross-Modal Neural Encoding

## Overview

This project investigates whether Vision-Language Model (VLM) representations exhibit **cross-modal neural alignment** — i.e., whether embeddings derived from one sensory modality (images) can predict brain responses to the corresponding stimuli presented in a different modality (text), and vice versa.

fMRI responses are measured while human participants process COCO images and their corresponding captions. Single-trial BOLD responses are then regressed against embeddings extracted from a VLM to quantify how well the model's internal representations predict brain activity across and within modalities.

---

## Documentation Map

| Document | Content |
| --- | --- |
| [Scientific Context](01_scientific_context.md) | Research question, hypotheses, experimental design |
| [Datasets](02_datasets.md) | fMRI acquisition, stimuli, BIDS structure |
| [Feature Extraction](03_feature_extraction.md) | VLM architecture, embedding extraction, pooling |
| [Neural Encoding](04_neural_encoding.md) | Ridge regression, cross-validation, permutation tests |
| [Evaluation & Statistics](05_evaluation_and_statistics.md) | Noise ceiling, metrics, significance testing |
| [Visualization](06_visualization.md) | Surface projection, result figures |
| [Configuration](07_configuration.md) | Hydra parameters, reproducibility |

---

## Quick-Start Pipeline

```text
1. Extract VLM embeddings      → extract_embeddings.py
2. Run neural encoding          → neural_encoding.py
3. Visualize noise ceiling      → visualize_noise_ceiling.py
4. Visualize encoding results   → visualize_encoding_results.py
```

All steps are configured via YAML files under `configs/` and launched through the `Makefile`.

---

## Key Result Structure

The pipeline produces one scalar metric per **(subject × condition)** cell, where conditions are the four entries in the 2×2 design:

| Embedding modality / Brain response | Image-evoked | Text-evoked |
| --- | --- | --- |
| **Vision embeddings** | image→image | image→text |
| **Text embeddings** | text→image | text→text |

Within-modality conditions serve as upper bounds; cross-modal conditions test whether the VLM's shared representational space generalizes across modalities in a brain-predictive sense.
