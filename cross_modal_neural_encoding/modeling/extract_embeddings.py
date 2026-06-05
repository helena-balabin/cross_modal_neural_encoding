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

Hydra config: ``configs/modeling/extract_embeddings.yaml``
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

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


@dataclass
class _ProcessorBundle:
    """Simple container for separately loaded image processor + tokenizer."""

    image_processor: object
    tokenizer: object


# ---------------------------------------------------------------------------
# Model-introspection helpers
# ---------------------------------------------------------------------------


def _get_vision_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Return the ordered list of vision-encoder transformer blocks."""
    enc = getattr(model, "encoder", None)
    layers = getattr(enc, "layer", None)
    if layers is not None:
        return list(layers)
    vt = getattr(getattr(model, "model", None), "vision_tower", None)
    if vt is not None:
        enc = getattr(vt, "encoder", None)
        layers = getattr(enc, "layer", None)
        if layers is not None:
            return list(layers)
    if hasattr(model, "visual") and hasattr(model.visual, "blocks"):
        return list(model.visual.blocks)  # type: ignore[attr-defined]
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "visual") and hasattr(inner.visual, "blocks"):
        return list(inner.visual.blocks)  # type: ignore[attr-defined]
    if hasattr(model, "vision_tower"):
        tower = model.vision_tower
        enc = getattr(getattr(tower, "vision_model", None), "encoder", None)
        if enc is not None and hasattr(enc, "layers"):
            return list(enc.layers)
    enc = getattr(getattr(model, "vision_model", None), "encoder", None)
    if enc is not None and hasattr(enc, "layers"):
        return list(enc.layers)
    vm_inner = getattr(getattr(model, "vision_model", None), "vision_model", None)
    vm_enc = getattr(vm_inner, "encoder", None)
    if vm_enc is not None and hasattr(vm_enc, "layers"):
        return list(vm_enc.layers)
    if hasattr(model, "blocks"):
        return list(model.blocks)  # type: ignore[attr-defined]
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
    enc = getattr(model, "encoder", None)
    if enc is not None and hasattr(enc, "layer"):
        model(pixel_values)
        return
    vt = getattr(getattr(model, "model", None), "vision_tower", None)
    if vt is not None:
        vt(pixel_values)
        return
    if hasattr(model, "visual"):
        model.visual(pixel_values, grid_thw=kwargs.get("image_grid_thw"))  # type: ignore[attr-defined]
    elif hasattr(getattr(model, "model", None), "visual"):
        model.model.visual(pixel_values, grid_thw=kwargs.get("image_grid_thw"))  # type: ignore[attr-defined]
    elif hasattr(model, "vision_tower"):
        model.vision_tower(pixel_values)  # type: ignore[attr-defined]
    elif hasattr(model, "vision_model"):
        model.vision_model(pixel_values)  # type: ignore[attr-defined]
    elif hasattr(model, "blocks"):
        model(pixel_values)  # type: ignore[operator]
    else:
        raise NotImplementedError(
            f"Unsupported vision encoder for {type(model).__name__}. "
            "Please extend _forward_vision()."
        )


def _get_language_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the language-model backbone (transformer decoder body)."""
    inner = getattr(model, "model", None)
    lm_inner = getattr(inner, "language_model", None)
    if lm_inner is not None and hasattr(lm_inner, "layers"):
        return lm_inner  # type: ignore[return-value]
    dec = getattr(model, "decoder", None)
    if dec is not None and hasattr(dec, "layers"):
        return dec  # type: ignore[return-value]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model  # type: ignore[attr-defined]
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):  # type: ignore[attr-defined]
            return lm.model  # type: ignore[attr-defined]
        return lm  # type: ignore[return-value]
    tm = getattr(model, "text_model", None)
    if tm is not None and hasattr(tm, "encoder") and hasattr(tm.encoder, "layers"):
        return tm  # type: ignore[return-value]
    dec = getattr(getattr(model, "model", None), "decoder", None)
    if dec is not None and hasattr(dec, "layers"):
        return dec  # type: ignore[return-value]
    if hasattr(model, "layers"):
        return model  # type: ignore[return-value]
    raise NotImplementedError(
        f"Cannot locate language model for {type(model).__name__}. "
        "Please extend _get_language_model()."
    )


def _load_processor(
    model_name: str,
    cache_dir: str | None,
    model_type: str = "vlm",
) -> object:
    """Load a compatible processor bundle for the requested model type."""
    mode = model_type.lower()
    if mode not in {"vlm", "vision_only", "language_only"}:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    def _try_load(name: str) -> object | None:
        if mode == "language_only":
            try:
                return AutoTokenizer.from_pretrained(
                    name,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    use_fast=False,
                )
            except (OSError, ValueError, EnvironmentError) as exc:
                logger.warning(
                    "AutoTokenizer loading failed for "
                    f"{name}: {type(exc).__name__}: {exc}."
                )
                return None

        if mode == "vision_only":
            try:
                return AutoImageProcessor.from_pretrained(
                    name,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )
            except (OSError, ValueError, EnvironmentError) as exc:
                logger.warning(
                    "AutoImageProcessor loading failed for "
                    f"{name}: {type(exc).__name__}: {exc}. "
                    "Retrying AutoProcessor."  # some vision models only define AutoProcessor
                )
            try:
                proc = AutoProcessor.from_pretrained(
                    name,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )
                if hasattr(proc, "image_processor"):
                    return proc
                logger.warning(
                    "AutoProcessor for "
                    f"{name} does not expose image_processor."
                )
            except (OSError, ValueError, EnvironmentError) as exc:
                logger.warning(
                    "AutoProcessor loading failed for "
                    f"{name}: {type(exc).__name__}: {exc}."
                )
            return None

        # VLM: need both image processor and tokenizer.
        try:
            proc = AutoProcessor.from_pretrained(
                name,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            if hasattr(proc, "image_processor") and hasattr(proc, "tokenizer"):
                return proc
            logger.warning(
                "AutoProcessor for "
                f"{name} does not expose both image_processor and tokenizer. "
                "Falling back to AutoImageProcessor + AutoTokenizer."
            )
        except (OSError, ValueError, EnvironmentError) as exc:
            logger.warning(
                "AutoProcessor loading failed for "
                f"{name}: {type(exc).__name__}: {exc}. "
                "Falling back to AutoImageProcessor + AutoTokenizer."
            )

        try:
            image_processor = AutoImageProcessor.from_pretrained(
                name,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                name,
                trust_remote_code=True,
                cache_dir=cache_dir,
                use_fast=False,
            )
            return _ProcessorBundle(image_processor=image_processor, tokenizer=tokenizer)
        except (OSError, ValueError, EnvironmentError) as exc:
            logger.warning(
                "AutoImageProcessor/AutoTokenizer loading failed for "
                f"{name}: {type(exc).__name__}: {exc}."
            )
            return None

    proc = _try_load(model_name)
    if proc is not None:
        return proc

    raise OSError(
        "Failed to load a compatible processor for "
        f"{model_name}."
    )


def _load_model(
    model_name: str,
    dtype: torch.dtype,
    cache_dir: str | None,
) -> torch.nn.Module:
    """Load VLM model with fallback across common HF APIs.

    Tries image-text-to-text auto-class first, then generic AutoModel.
    """
    try:
        return AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            weights_only=False,
            cache_dir=cache_dir,
        )
    except (OSError, ValueError, EnvironmentError) as exc:
        logger.warning(
            "AutoModelForImageTextToText loading failed for "
            f"{model_name}: {type(exc).__name__}: {exc}. "
            "Falling back to AutoModel."
        )

    return AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        weights_only=False,
        cache_dir=cache_dir,
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
    processor: object,
    images: list[Image.Image] | DataLoader,
    *,
    device: torch.device,
    dtype: torch.dtype,
    pooling: str,
    layer_indices: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Extract per-layer vision-encoder embeddings for a list of PIL images.

    ``images`` may be a plain ``list[PIL.Image]`` or a ``DataLoader`` whose
    batches are ``(image_list, text_list)`` tuples (images are used, texts
    are ignored).

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

    if isinstance(images, DataLoader):
        image_iter = (img for imgs, _ in tqdm(images, desc="Vision embeddings") for img in imgs)
    else:
        image_iter = tqdm(images, desc="Vision embeddings")  # type: ignore[assignment]

    for img in image_iter:
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
    processor: object,
    texts: list[str] | DataLoader,
    *,
    device: torch.device,
    pooling: str,
    layer_indices: list[int] | None = None,
    batch_size: int = 8,
    max_length: int = 512,
) -> dict[int, np.ndarray]:
    """Extract per-layer language-model embeddings for a list of texts.

    ``texts`` may be a plain ``list[str]`` or a ``DataLoader`` whose batches
    are ``(image_list, text_list)`` tuples (texts are used, images are
    ignored).

    Returns
    -------
    dict
        ``{layer_index: np.ndarray}`` where each array has shape
        ``(n_texts, hidden_dim)``.
    """
    lm = _get_language_model(model)
    tokenizer = getattr(processor, "tokenizer", processor)

    results: dict[int, list[np.ndarray]] | None = None

    if isinstance(texts, DataLoader):
        batches_iter = (text_batch for _imgs, text_batch in tqdm(texts, desc="Text embeddings"))
    else:
        batches_iter = (
            texts[i : i + batch_size]
            for i in tqdm(range(0, len(texts), batch_size), desc="Text embeddings")
        )

    for batch in batches_iter:
        tokens = tokenizer(  # type: ignore[call-arg]
            list(batch),
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

    if results is None:
        raise RuntimeError("No texts to process")
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
# Stimulus loading helper
# ---------------------------------------------------------------------------


def _load_stimuli(
    cfg: DictConfig,
    metadata_path: Path,
    image_filename_col: str,
    coco_id_col: str,
) -> tuple[list[str], np.ndarray, list[Image.Image], np.ndarray]:
    """Load texts, COCO IDs, unique images, and a broadcast index from config.

    Returns
    -------
    texts : list[str]
    coco_ids : (n,) array of COCO IDs
    unique_images : list of deduplicated PIL images
    broadcast : (n,) index mapping each row to its entry in unique_images
    """
    dataset_name = cfg.get("dataset_hf_identifier", None)
    if dataset_name:
        dataset_split = cfg.get("dataset_split", "train")
        dataset_cache_dir = cfg.get("dataset_cache_dir", None)
        logger.info(
            f"Loading metadata from dataset {dataset_name} (split={dataset_split})"
        )
        from datasets import load_dataset as _load_dataset
        dataset = _load_dataset(dataset_name, split=dataset_split, cache_dir=dataset_cache_dir)
        df = dataset.to_pandas()
        missing_cols = [
            c for c in (cfg.text_column, image_filename_col, coco_id_col) if c not in df.columns
        ]
        if missing_cols:
            raise KeyError(f"Missing columns in dataset: {missing_cols}")
        if cfg.get("drop_empty_text", True):
            df[cfg.text_column] = df[cfg.text_column].astype(str)
            df = df[df[cfg.text_column].str.strip() != ""].reset_index(drop=True)
    else:
        logger.info(f"Loading metadata from {metadata_path}")
        usecols = [cfg.text_column, image_filename_col, coco_id_col]
        df = pd.read_csv(metadata_path, usecols=usecols)

    texts: list[str] = df[cfg.text_column].tolist()
    coco_ids: np.ndarray = df[coco_id_col].values  # type: ignore[assignment]

    image_dir = Path(cfg.image_dir)
    if not image_dir.is_absolute():
        image_dir = PROJ_ROOT / image_dir
    logger.info(f"Loading images from {image_dir} (filename column: {image_filename_col!r})")

    filenames: list[str] = df[image_filename_col].tolist()
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
        f"  {len(df)} rows  ·  {len(unique_images)} unique images  ·  {len(texts)} texts"
    )
    return texts, coco_ids, unique_images, broadcast


# ---------------------------------------------------------------------------
# Hydra entry-point
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="../../configs/modeling",
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
    vision_layers : list[int] | null
        Vision-encoder layer indices to extract (``null`` = all).
    text_layers : list[int] | null
        Text-encoder layer indices to extract (``null`` = all).
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
    vision_layer_indices: list[int] | None = (
        list(cfg.vision_layers) if cfg.get("vision_layers") is not None else None
    )
    text_layer_indices: list[int] | None = (
        list(cfg.text_layers) if cfg.get("text_layers") is not None else None
    )

    # -- load metadata + images ---------------------------------------------
    image_filename_col: str = cfg.get("image_filename_column", "filepath")
    coco_id_col: str = cfg.get("coco_id_column", "cocoid_x")
    texts, coco_ids, unique_images, broadcast = _load_stimuli(
        cfg, metadata_path, image_filename_col, coco_id_col
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

        # model_type controls which encoders to extract:
        #   "vlm"           → both vision and text (default)
        #   "vision_only"   → vision only  (DINOv2, LeJEPA, …)
        #   "language_only" → text only    (Pythia, OPT, …)
        model_type: str = cfg.get("model_type", "vlm").lower()

        logger.info("Loading model & processor …")
        processor = _load_processor(
            model_name,
            cache_dir,
            model_type=model_type,
        )
        model = _load_model(model_name, dtype, cache_dir).to(device).eval()

        # ---- vision embeddings (unique images, then broadcast) ------------
        if model_type in ("vlm", "vision_only"):
            logger.info("Extracting vision-encoder embeddings …")
            vis_embs = extract_vision_embeddings(
                model,
                processor,
                unique_images,
                device=device,
                dtype=dtype,
                pooling=pooling,
                layer_indices=vision_layer_indices,
            )
            vis_embs = {k: v[broadcast] for k, v in vis_embs.items()}
            _save_embeddings(vis_embs, model_dir, "vision_embeddings", coco_ids)
        else:
            logger.info("Skipping vision embeddings (model_type='language_only').")

        # ---- text embeddings (one per row) --------------------------------
        if model_type in ("vlm", "language_only"):
            logger.info("Extracting text-encoder embeddings …")
            txt_embs = extract_text_embeddings(
                model,
                processor,
                texts,
                device=device,
                pooling=pooling,
                layer_indices=text_layer_indices,
                batch_size=batch_size,
                max_length=max_length,
            )
            _save_embeddings(txt_embs, model_dir, "text_embeddings", coco_ids)
        else:
            logger.info("Skipping text embeddings (model_type='vision_only').")

        # ---- cleanup ------------------------------------------------------
        del model, processor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.success(f"Done: {model_name}")

    logger.success("All models processed!")


if __name__ == "__main__":
    main()
