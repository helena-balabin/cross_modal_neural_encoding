# Models

## Vision-Language Models

We evaluate a range of vision-language models spanning different architectures, training objectives, and parameter counts to assess the generality of our findings.

### Qwen3.5-VL

- **Sizes**: 1B, 2B, 4B, and 9B parameters
- **Architecture**: Vision-language transformer with separate vision and text encoders
- **Layer selection**: Vision encoder layer 6, text encoder layer 12 (by default)
- **Source**: Hugging Face (`Qwen/Qwen3.5-0.8B-Base`, `Qwen/Qwen3.5-2B-Base`, `Qwen/Qwen3.5-4B-Base`, `Qwen/Qwen3.5-9B-Base`)

### InternVL3.5

- **Sizes**: 1B, 2B, 4B, and 8B parameters
- **Architecture**: Vision-language model with joint training on image-text pairs
- **Layer selection**: Vision encoder layer 6, text encoder layer 12 (by default)
- **Source**: Hugging Face (`OpenGVLab/InternVL3_5-1B-HF`, `OpenGVLab/InternVL3_5-2B-HF`, `OpenGVLab/InternVL3_5-4B-HF`, `OpenGVLab/InternVL3_5-8B-HF`)

### OpenCLIP

- **Sizes**: Base and Large variants
- **Architecture**: Contrastive language-image pretraining model
- **Layer selection**: Vision transformer layer 6, text encoder layer 12 (by default)
- **Source**: Hugging Face (`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`)

## Unimodal Models

To dissociate effects of joint vision-language training from those of unimodal pretraining, we additionally evaluate unimodal models.

### Vision Encoders

#### DINOv2

- **Sizes**: Base and Large scales
- **Architecture**: Self-supervised vision transformer trained on image data
- **Layer selection**: Transformer block 6 (by default)
- **Source**: Hugging Face (`facebook/dinov2-giant`, `facebook/dinov2-large`)

#### iJEPA

- **Sizes**: Base and Large scales
- **Architecture**: Vision encoder trained with latent embedding Jacobian prediction
- **Layer selection**: Transformer block 6 (by default)
- **Source**: Hugging Face (`facebook/ijepa_vitg16_22k`, `facebook/ijepa_vith14_22k`)

### Language Encoders

#### Pythia

- **Sizes**: 1.4B and 6.9B parameters
- **Architecture**: Causal language model trained on text data
- **Layer selection**: Transformer block 12 (by default)
- **Source**: Hugging Face (`EleutherAI/pythia-1.4b`, `EleutherAI/pythia-6.9b`)

#### OPT

- **Sizes**: 1.3B and 2.7B parameters
- **Architecture**: Open Pre-trained Transformers language model
- **Layer selection**: Transformer block 12 (by default)
- **Source**: Hugging Face (`facebook/opt-1.3b`, `facebook/opt-2.7b`)

## Implementation Details

### Layer Selection

For all models, embeddings are extracted from mid-network layers:

- **Vision models**: Layer 6 (out of 12-24 total layers)
- **Language models**: Layer 12 (out of 24-40 total layers)

This selection balances early sensory processing with later semantic abstraction. The specific layer can be configured in the analysis parameters.

### Embedding Extraction

Embeddings are extracted using the `extract_embeddings.py` script, which:

1. Loads the pre-trained model from Hugging Face
2. Processes images through the vision encoder and text through the language encoder
3. Extracts hidden states from the specified layer
4. Applies mean pooling across spatial patches (vision) or non-padding tokens (text)
5. Saves the resulting fixed-dimensional vectors

The extraction process handles various model architectures through a modular design that can be extended to new models.

### Model Robustness

The implementation includes robust handling of different model architectures:

- **Vision encoders**: The `_get_vision_layers` function in `extract_embeddings.py` supports multiple VLM families (Qwen, LLaVA, BLIP-2, etc.)
- **Language encoders**: The `_get_language_model` function handles different LLM backbones
- **Processor loading**: The `_load_processor` function gracefully handles cases where AutoProcessor doesn't provide both image and text processors

This ensures compatibility across the diverse set of models evaluated in the study.
