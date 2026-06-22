# Cross-Modal Neural Encoding

[![CCDS Project template](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

Investigates whether Vision-Language Model (VLM) representations exhibit **cross-modal neural alignment**: whether embeddings from one sensory modality (images or text) can predict fMRI brain responses to stimuli presented in the other modality.

Full documentation: `docs/` (built with mkdocs).

## Project Organisation

```text
├── configs/                    <- Hydra YAML configuration files
│   ├── modeling/               <- extract_embeddings, neural_encoding, predict_modalities,
│   │                              residual_encoding
│   └── visualization/          <- matching visualisation configs
│
├── cross_modal_neural_encoding/
│   ├── config.py               <- Path constants (PROJ_ROOT, DATA_DIR, etc.)
│   ├── utils.py                <- Shared utilities (noise ceiling, beta loading, etc.)
│   ├── modeling/
│   │   ├── datasets.py         <- PyTorch Dataset for VG-COCO stimuli
│   │   ├── extract_embeddings.py   <- VLM hidden-state extraction via hooks
│   │   ├── neural_encoding.py      <- Ridge regression encoding model
│   │   ├── predict_modalities.py   <- Cross-modal embedding prediction (MLP / ridge)
│   │   └── residual_encoding.py    <- Cross-modal residual ablation
│   └── visualization/
│       ├── visualize_encoding_results.py
│       ├── visualize_noise_ceiling.py
│       ├── visualize_predict_modalities.py
│       └── visualize_residual_encoding.py
│
├── data/
│   ├── coco_metadata/          <- COCO stimulus metadata CSV
│   ├── external/               <- Third-party data
│   ├── interim/                <- Intermediate outputs
│   ├── processed/              <- Final analysis-ready data
│   └── raw/                    <- Original immutable data
│
├── docs/                       <- mkdocs documentation
├── models/                     <- Saved model checkpoints
├── outputs/                    <- Pipeline results (per-analysis subdirs)
├── reports/figures/            <- Generated figures
│
├── pyproject.toml              <- Package metadata and tool configuration (ruff)
└── uv.lock                     <- Locked dependency versions
```

## Pipeline

```text
1. Extract VLM embeddings      →  cross_modal_neural_encoding/modeling/extract_embeddings.py
2. Run neural encoding          →  cross_modal_neural_encoding/modeling/neural_encoding.py
3. Predict modalities           →  cross_modal_neural_encoding/modeling/predict_modalities.py
4. Visualise results            →  cross_modal_neural_encoding/visualization/
```

Each step is configured via its corresponding YAML file in `configs/`.

## Setup

```bash
uv sync
source .venv/bin/activate
```

--------
