"""Extract vision and text embeddings from Vision-Language Models (VLMs).

Processes image-text pairs from a metadata CSV and saves per-layer hidden
states from:

* **vision encoder** – the visual backbone (e.g. ViT transformer blocks)
* **text encoder** – the language-model backbone (decoder layers)

Images are loaded from a COCO image directory using the ``filepath``
column in the metadata CSV (e.g. ``COCO_train2014_000000546154.jpg``).

Embeddings are pooled (mean or last-token) and saved as NumPy ``.npy``
files that are row-aligned with the source metadata CSV.

Usage
-----
    python -m cross_modal_neural_encoding.modeling.extract_embeddings

Hydra config: ``configs/extract_embeddings.yaml``
"""

from __future__ import annotations

import gc
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from cross_modal_neural_encoding.config import PROJ_ROOT


# ---------------------------------------------------------------------------
# Forward-hook helper for capturing intermediate hidden states
# ---------------------------------------------------------------------------


class HiddenStateHooks:
    """Register forward hooks on a list of modules and collect their outputs."""

    def __init__(self) -> None:
        self._states: list[torch.Tensor] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    # -- callback -----------------------------------------------------------
    def _hook_fn(
        self,
        module: torch.nn.Module,
        input: tuple,
        output: torch.Tensor | tuple,
    ) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        self._states.append(tensor.detach().cpu())

    # -- public API ---------------------------------------------------------
    def register(self, modules: list[torch.nn.Module]) -> None:
        for m in modules:
            self._handles.append(m.register_forward_hook(self._hook_fn))

    def reset(self) -> None:
        self._states.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @property
    def states(self) -> list[torch.Tensor]:
        return self._states


# ---------------------------------------------------------------------------
# Model-introspection helpers  (extend these for new VLM families)
# ---------------------------------------------------------------------------


def _get_vision_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Return the ordered list of vision-encoder transformer blocks."""
    # Qwen2-VL / Qwen2.5-VL (top-level .visual)
    if hasattr(model, "visual") and hasattr(model.visual, "blocks"):
        return list(model.visual.blocks)  # type: ignore[attr-defined]
    # Qwen3.5-VL (nested under .model.visual)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "visual") and hasattr(inner.visual, "blocks"):
        return list(inner.visual.blocks)  # type: ignore[attr-defined]
    # LLaVA-style (vision_tower wrapping a CLIP / SigLIP ViT)
    if hasattr(model, "vision_tower"):
        tower = model.vision_tower
        enc = getattr(getattr(tower, "vision_model", None), "encoder", None)
        if enc is not None and hasattr(enc, "layers"):
            return list(enc.layers)
    # BLIP-2 / InstructBLIP
    enc = getattr(getattr(model, "vision_model", None), "encoder", None)
    if enc is not None and hasattr(enc, "layers"):
        return list(enc.layers)
    raise NotImplementedError(
        f"Cannot locate vision-encoder layers for {type(model).__name__}. "
        "Please extend _get_vision_layers()."
    )


def _forward_vision(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    **kwargs,
) -> None:
    """Run the vision-encoder forward pass (hooks capture hidden states)."""
    if hasattr(model, "visual"):  # Qwen2-VL / Qwen2.5-VL family
        model.visual(pixel_values, grid_thw=kwargs.get("image_grid_thw"))  # type: ignore[attr-defined]
    elif hasattr(getattr(model, "model", None), "visual"):  # Qwen3.5-VL (nested)
        model.model.visual(pixel_values, grid_thw=kwargs.get("image_grid_thw"))  # type: ignore[attr-defined]
    elif hasattr(model, "vision_tower"):  # LLaVA family
        model.vision_tower(pixel_values)  # type: ignore[attr-defined]
    elif hasattr(model, "vision_model"):  # BLIP-2 family
        model.vision_model(pixel_values)  # type: ignore[attr-defined]
    else:
        raise NotImplementedError(
            f"Unsupported vision encoder for {type(model).__name__}. "
            "Please extend _forward_vision()."
        )


def _get_language_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the language-model backbone (transformer decoder body)."""
    # Most VLMs store the LLM as .model with .layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model  # type: ignore[attr-defined]
    # Qwen3.5-VL – LLM is at model.model.language_model (with .layers)
    inner = getattr(model, "model", None)
    lm_inner = getattr(inner, "language_model", None)
    if lm_inner is not None and hasattr(lm_inner, "layers"):
        return lm_inner  # type: ignore[return-value]
    # BLIP-2 / InstructBLIP
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):  # type: ignore[attr-defined]
            return lm.model  # type: ignore[attr-defined]
        return lm  # type: ignore[return-value]
    raise NotImplementedError(
        f"Cannot locate language model for {type(model).__name__}. "
        "Please extend _get_language_model()."
    )


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def _pool(
    hidden: torch.Tensor,
    strategy: str,
    mask: torch.Tensor | None = None,
) -> np.ndarray:
    """Pool *(batch, seq, dim)* → *(batch, dim)* and convert to NumPy.

    Parameters
    ----------
    hidden : torch.Tensor
        Hidden states – shape ``(B, S, D)`` or ``(S, D)``.
    strategy : str
        ``"mean"`` (masked mean) or ``"last"`` (last non-pad token).
    mask : torch.Tensor | None
        Attention mask of shape ``(B, S)`` – used to ignore padding.
    """
    if hidden.ndim == 2:
        hidden = hidden.unsqueeze(0)
    if strategy == "mean":
        if mask is not None:
            m = mask.unsqueeze(-1).float().to(hidden.device)
            out = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            out = hidden.mean(1)
    elif strategy == "last":
        if mask is not None:
            lengths = mask.sum(1) - 1
            out = hidden[torch.arange(hidden.size(0)), lengths]
        else:
            out = hidden[:, -1]
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy!r}")
    return out.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Extraction routines
# ---------------------------------------------------------------------------


def extract_vision_embeddings(
    model: torch.nn.Module,
    processor: AutoProcessor,
    images: list[Image.Image],
    *,
    device: torch.device,
    dtype: torch.dtype,
    pooling: str,
    layer_indices: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Extract per-layer vision-encoder embeddings for a list of PIL images.

    Returns
    -------
    dict
        ``{layer_index: np.ndarray}`` where each array has shape
        ``(n_images, hidden_dim)``.
    """
    all_layers = _get_vision_layers(model)
    sel_idx = layer_indices if layer_indices is not None else list(range(len(all_layers)))
    sel_layers = [all_layers[i] for i in sel_idx]

    hooks = HiddenStateHooks()
    hooks.register(sel_layers)
    results: dict[int, list[np.ndarray]] = {i: [] for i in sel_idx}

    image_processor = getattr(processor, "image_processor", processor)

    for img in tqdm(images, desc="Vision embeddings"):
        inputs = image_processor(images=[img], return_tensors="pt")  # type: ignore[call-arg]
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
        extra: dict = {}
        if "image_grid_thw" in inputs:
            extra["image_grid_thw"] = inputs["image_grid_thw"].to(device)

        hooks.reset()
        with torch.no_grad():
            _forward_vision(model, pixel_values, **extra)

        for pos, idx in enumerate(sel_idx):
            results[idx].append(_pool(hooks.states[pos], pooling))

    hooks.remove()
    return {idx: np.concatenate(v) for idx, v in results.items()}


def extract_text_embeddings(
    model: torch.nn.Module,
    processor: AutoProcessor,
    texts: list[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
    pooling: str,
    layer_indices: list[int] | None = None,
    batch_size: int = 8,
    max_length: int = 512,
) -> dict[int, np.ndarray]:
    """Extract per-layer language-model embeddings for a list of texts.

    Returns
    -------
    dict
        ``{layer_index: np.ndarray}`` where each array has shape
        ``(n_texts, hidden_dim)``.
    """
    lm = _get_language_model(model)
    tokenizer = getattr(processor, "tokenizer", processor)

    results: dict[int, list[np.ndarray]] | None = None

    for start in tqdm(range(0, len(texts), batch_size), desc="Text embeddings"):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(  # type: ignore[call-arg]
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            out = lm(**tokens, output_hidden_states=True)

        hs = out.hidden_states  # tuple (n_layers + 1,) of (B, S, D)
        if results is None:
            sel = layer_indices if layer_indices is not None else list(range(len(hs)))
            results = {i: [] for i in sel}

        mask = tokens.get("attention_mask")
        for idx in results:
            results[idx].append(_pool(hs[idx], pooling, mask))

    assert results is not None, "No texts to process"
    return {idx: np.concatenate(v) for idx, v in results.items()}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _save_embeddings(
    embs: dict[int, np.ndarray],
    output_dir: Path,
    modality: str,
    coco_ids: np.ndarray,
) -> None:
    """Save per-layer embeddings as ``.npy`` files with a COCO-ID key file."""
    d = output_dir / modality
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "coco_ids.npy", coco_ids)
    logger.debug(f"  {modality}/coco_ids.npy  len={len(coco_ids)}")
    for idx, arr in embs.items():
        np.save(d / f"layer_{idx:03d}.npy", arr)
        logger.debug(f"  {modality}/layer_{idx:03d}.npy  shape={arr.shape}")
    logger.info(f"Saved {len(embs)} layer(s) + coco_ids → {d}")


_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# ---------------------------------------------------------------------------
# Hydra entry-point
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="extract_embeddings",
)
def main(cfg: DictConfig) -> None:
    """Extract VLM embeddings for all image-text pairs in the metadata CSV.

    Hydra config fields
    -------------------
    models : list[str]
        HuggingFace model identifiers.
    metadata_path : str
        CSV with image-text pairs (relative to project root or absolute).
    image_dir : str
        Directory containing COCO images referenced by ``image_filename_column``.
    image_filename_column : str
        CSV column with the image filename (default ``"filepath"``).
    coco_id_column : str
        CSV column with the COCO image ID, saved alongside embeddings as a
        row key (default ``"cocoid_x"``).
    text_column : str
        CSV column with caption text.
    output_dir : str
        Where to store the extracted embeddings.
    cache_dir : str
        Local cache directory for HuggingFace model / processor downloads.
    device / dtype : str
        Compute device and floating-point precision.
    pooling : str
        ``"mean"`` or ``"last"`` – how to pool token-level hidden states.
    batch_size / max_length : int
        Batching and tokenizer settings for text extraction.
    layers : list[int] | null
        Layer indices to extract (``null`` = all).
    """
    # -- resolve paths (relative to project root) --------------------------
    metadata_path = Path(cfg.metadata_path)
    if not metadata_path.is_absolute():
        metadata_path = PROJ_ROOT / metadata_path
    output_root = Path(cfg.output_dir)
    if not output_root.is_absolute():
        output_root = PROJ_ROOT / output_root

    cache_dir: str | None = cfg.get("cache_dir", None)
    device = torch.device(cfg.device)
    dtype = _DTYPES[cfg.dtype]
    pooling: str = cfg.pooling
    batch_size: int = cfg.get("batch_size", 8)
    max_length: int = cfg.get("max_length", 512)
    layer_indices: list[int] | None = (
        list(cfg.layers) if cfg.get("layers") is not None else None
    )

    # -- load metadata ------------------------------------------------------
    logger.info(f"Loading metadata from {metadata_path}")
    image_filename_col: str = cfg.get("image_filename_column", "filepath")
    coco_id_col: str = cfg.get("coco_id_column", "cocoid_x")
    usecols = [cfg.text_column, image_filename_col, coco_id_col]
    df = pd.read_csv(metadata_path, usecols=usecols)
    texts: list[str] = df[cfg.text_column].tolist()
    coco_ids: np.ndarray = df[coco_id_col].values  # type: ignore[assignment]

    # -- load images from the COCO image directory -------------------------
    image_dir = Path(cfg.image_dir)
    if not image_dir.is_absolute():
        image_dir = PROJ_ROOT / image_dir
    logger.info(f"Loading images from {image_dir} (filename column: {image_filename_col!r})")

    filenames: list[str] = df[image_filename_col].tolist()

    # Deduplicate images so each unique image is processed only once
    seen: dict[str, int] = {}
    unique_images: list[Image.Image] = []
    for fname in tqdm(filenames, desc="Loading images"):
        if fname not in seen:
            img_path = image_dir / fname
            img = Image.open(img_path).convert("RGB")
            seen[fname] = len(unique_images)
            unique_images.append(img)

    broadcast = np.array([seen[f] for f in filenames])

    logger.info(
        f"  {len(df)} rows  ·  {len(unique_images)} unique images  ·  "
        f"{len(texts)} texts"
    )

    # -- iterate over models ------------------------------------------------
    model_names: list[str] = (
        [cfg.models] if isinstance(cfg.models, str) else list(cfg.models)
    )

    for model_name in model_names:
        label = model_name.replace("/", "--")
        model_dir = output_root / label

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Model : {model_name}")
        logger.info(f"Output: {model_dir}")

        logger.info("Loading model & processor …")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        model = (
            AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            .to(device)  # type: ignore[method-assign]
            .eval()
        )

        # ---- vision embeddings (unique images, then broadcast) ------------
        logger.info("Extracting vision-encoder embeddings …")
        vis_embs = extract_vision_embeddings(
            model,
            processor,
            unique_images,
            device=device,
            dtype=dtype,
            pooling=pooling,
            layer_indices=layer_indices,
        )
        vis_embs = {k: v[broadcast] for k, v in vis_embs.items()}
        _save_embeddings(vis_embs, model_dir, "vision_embeddings", coco_ids)

        # ---- text embeddings (one per row) --------------------------------
        logger.info("Extracting text-encoder embeddings …")
        txt_embs = extract_text_embeddings(
            model,
            processor,
            texts,
            device=device,
            dtype=dtype,
            pooling=pooling,
            layer_indices=layer_indices,
            batch_size=batch_size,
            max_length=max_length,
        )
        _save_embeddings(txt_embs, model_dir, "text_embeddings", coco_ids)

        # ---- cleanup ------------------------------------------------------
        del model, processor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.success(f"Done: {model_name}")

    logger.success("All models processed!")


if __name__ == "__main__":
    main()
