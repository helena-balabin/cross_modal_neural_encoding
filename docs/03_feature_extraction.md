# Feature Extraction

## Model

All experiments use **Qwen2.5-VL-3B** (configurable via Hydra). This is a transformer-based Vision-Language Model that processes both images and text through separate but jointly trained encoders, enabling extraction of both vision and language representations from a single model.

### Architecture Summary

| Component | Depth | Hidden dim |
| --- | --- | --- |
| Vision encoder | 12 transformer layers (0–11) | Model-dependent |
| Language model (text) | 24 transformer layers (0–23) | Model-dependent |

Both encoders are accessed via HuggingFace `AutoModel` / `AutoProcessor`.

---

## Embedding Extraction Pipeline

Implemented in `cross_modal_neural_encoding/extract_embeddings.py`.

### Vision Embeddings

1. Load unique COCO images from disk (deduplicated by `cocoid_x`).
2. Preprocess images with the VLM processor (resize, normalise, patch tokenisation).
3. Register a **forward hook** on the target vision-encoder layer.
4. Run a forward pass; the hook captures the hidden-state tensor at that layer.
5. **Pool** across the spatial (patch) dimension using **mean pooling** → one vector per image.
6. Broadcast back to the full CSV row ordering (re-indexing duplicate COCO IDs).
7. Save as `vision_embeddings/layer_{n:03d}.npy` alongside `coco_ids.npy`.

**Default layer:** Vision layer 6 (mid-network).

### Text Embeddings

1. Tokenise all captions with the model tokeniser (`max_length=512`, batched at 8).
2. Run forward pass; extract the language-model hidden states at the target layer.
3. Apply **mean pooling** over non-padding tokens → one vector per caption.
4. Save as `text_embeddings/layer_{n:03d}.npy`.

**Default layer:** Language-model layer 12 (mid-network).

### Output Layout

```text
models/embeddings/{model_label}/
├── vision_embeddings/
│   ├── coco_ids.npy        # (N,) COCO IDs in row order
│   └── layer_006.npy       # (N, D_vision) float32
└── text_embeddings/
    ├── coco_ids.npy        # (N,) COCO IDs in row order
    └── layer_012.npy       # (N, D_text) float32
```

---

## Design Choices

### Layer Selection

Mid-network layers are extracted rather than the final layer because:

- Final-layer representations in large VLMs are tuned for next-token prediction and may over-specialise.
- Mid-network layers tend to capture richer perceptual and semantic features that correlate better with cortical representations (as established in prior encoding model literature).
- Layer selection is a hyperparameter that can be swept; the Hydra config exposes it explicitly.

### Mean Pooling

Mean pooling over spatial patches (vision) or token positions (text) produces a fixed-dimensional summary vector without requiring positional alignment between stimuli. Alternative strategies (CLS token, last token) are supported but mean pooling is the default.

### Compute & Precision

- **Device:** CUDA (GPU)
- **Dtype:** `bfloat16` (reduces memory footprint without meaningful precision loss for encoding)
- **Batch size:** 8 (text tokenisation); images processed individually due to variable sizes

---

## Supported Model Families

The extraction code includes model-specific introspection helpers for:

- Qwen2-VL, Qwen2.5-VL, Qwen3.5-VL
- LLaVA (with CLIP or SigLIP vision tower)
- BLIP-2 / InstructBLIP
- CLIP / OpenCLIP (vision-only)

Each family requires a different hook-attachment strategy due to varying internal attribute names.
