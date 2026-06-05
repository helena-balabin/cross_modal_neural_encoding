"""
Implements structural probing framework to analyze compositional structure in VLM embeddings.

Probes for graph-based structural properties of paired text-image stimuli using ridge regression
on frozen encoder embeddings. The framework uses precomputed AMR graphs for text and
action graphs for images to extract structural targets.

Note
----
Embeddings are **pre-projection** encoder hidden states saved by the extraction pipeline
(`extract_embeddings.py`). This matches the representation used in neural encoding.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV as HimalayaRidgeCV
import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from PIL import Image
from sklearn.metrics import r2_score
import torch

from cross_modal_neural_encoding.utils import get_graph_metric

from .extract_embeddings import (
    _DTYPES,
    _get_language_model,
    _get_vision_layers,
    _load_model,
    _load_processor,
    extract_text_embeddings,
    extract_vision_embeddings,
)

PROJ_ROOT = Path(__file__).parent.parent.parent


def _list_available_layers(emb_dir: Path) -> list[int]:
    if not emb_dir.exists():
        return []
    layers: list[int] = []
    for p in emb_dir.glob("layer_*.npy"):
        try:
            layers.append(int(p.stem.split("_")[-1]))
        except (ValueError, IndexError):
            continue
    return sorted(set(layers))


def _pick_middle_layer(layers: list[int]) -> int | None:
    if not layers:
        return None
    return layers[len(layers) // 2]


def _auto_select_layer_from_model(
    model: torch.nn.Module,
    modality: str,
) -> int | None:
    modality = modality.lower()
    if modality == "vision":
        try:
            layers = _get_vision_layers(model)
        except AttributeError as exc:
            logger.warning(f"Failed to inspect vision layers for auto selection: {exc}")
            return None
        return _pick_middle_layer(list(range(len(layers))))
    if modality == "text":
        try:
            lm = _get_language_model(model)
        except AttributeError as exc:
            logger.warning(f"Failed to inspect text layers for auto selection: {exc}")
            return None

        if hasattr(lm, "layers"):
            n_layers = len(lm.layers)  # type: ignore[arg-type]
        elif hasattr(lm, "encoder") and hasattr(lm.encoder, "layers"):
            n_layers = len(lm.encoder.layers)  # type: ignore[arg-type]
        elif hasattr(lm, "layer"):
            n_layers = len(lm.layer)  # type: ignore[arg-type]
        else:
            logger.warning("Could not determine text layer count for auto selection.")
            return None
        return _pick_middle_layer(list(range(n_layers)))
    raise ValueError(f"Unknown modality: {modality}")


def _load_embedding_metadata_df(
    cfg: DictConfig,
    df: pd.DataFrame,
    coco_id_col: str,
) -> pd.DataFrame:
    text_col = cfg.get("text_column", "text")
    image_col = cfg.get("image_filename_column", "filepath")
    if text_col not in df.columns or image_col not in df.columns:
        raise KeyError(
            "Graph dataset must include text and image columns for on-the-fly "
            f"embeddings. Missing: {text_col if text_col not in df.columns else ''} "
            f"{image_col if image_col not in df.columns else ''}"
        )

    meta = df[[coco_id_col, text_col, image_col]].copy()

    return meta.rename(
        columns={
            coco_id_col: "coco_id",
            text_col: "text",
            image_col: "filepath",
        }
    )


def _deduplicate_images(
    image_dir: Path,
    filenames: list[str],
) -> tuple[list[Image.Image], np.ndarray]:
    seen: dict[str, int] = {}
    unique_images: list[Image.Image] = []
    for fname in filenames:
        if fname not in seen:
            img_path = image_dir / fname
            img = Image.open(img_path).convert("RGB")
            seen[fname] = len(unique_images)
            unique_images.append(img)
    broadcast = np.array([seen[f] for f in filenames])
    return unique_images, broadcast


def load_graph_dataset(
    dataset_name: str,
    split: str = "train",
) -> pd.DataFrame:
    """Load the graph-annotated dataset used for structural probing."""
    from datasets import load_dataset

    logger.info(f"Loading graph dataset {dataset_name}, split {split}")
    dataset = load_dataset(dataset_name, split=split)
    df = cast(pd.DataFrame, dataset.to_pandas())
    logger.info(f"Loaded {len(df)} samples")
    return df


def _extract_graph_labels(
    df: pd.DataFrame,
    graph_col: str,
    target: str,
) -> np.ndarray:
    if target == "num_nodes":
        return np.asarray(df[graph_col].apply(lambda g: get_graph_metric(g, "num_nodes")).values)
    if target == "num_edges":
        return np.asarray(df[graph_col].apply(lambda g: get_graph_metric(g, "num_edges")).values)
    if target == "depth":
        return np.asarray(df[graph_col].apply(lambda g: get_graph_metric(g, "depth")).values)
    raise ValueError(f"Unknown target: {target}")


def _random_train_val_split(
    n_samples: int,
    train_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    split_idx = int(n_samples * train_ratio)
    return indices[:split_idx], indices[split_idx:]


def run_graph_probing(
    embeddings: np.ndarray,
    embed_coco_ids: np.ndarray,
    df: pd.DataFrame,
    coco_id_col: str,
    graph_columns: list[str],
    targets: list[str],
    n_outer_folds: int = 5,
    inner_cv_folds: int = 5,
    train_ratio: float = 0.8,
    alpha_grid: np.ndarray = np.logspace(-3, 3, 7),
    random_seed: int = 0,
    verbose: bool = True,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    results: dict[str, float] = {}
    fold_scores: dict[str, np.ndarray] = {}

    embed_id_to_idx = {int(cid): i for i, cid in enumerate(embed_coco_ids)}
    df_ids = df[coco_id_col].astype(int).to_numpy()
    common_mask = np.array([int(cid) in embed_id_to_idx for cid in df_ids])
    if not np.any(common_mask):
        raise ValueError("No overlapping COCO IDs between embeddings and dataset")

    df_common = df.loc[common_mask].reset_index(drop=True)

    df_common_ids = df_common[coco_id_col].astype(int).to_numpy()
    X = np.array([embeddings[embed_id_to_idx[int(cid)]] for cid in df_common_ids])

    for graph_col in graph_columns:
        if graph_col not in df_common.columns:
            logger.warning(f"Graph column {graph_col} not found; skipping.")
            continue

        for target in targets:
            labels = _extract_graph_labels(df_common, graph_col, target)

            if target in {"num_nodes", "num_edges", "depth"}:
                valid_mask = labels > 0
            else:
                valid_mask = np.ones(len(labels), dtype=bool)

            y = labels[valid_mask].astype(np.float32)
            X_valid = X[valid_mask]

            if len(y) == 0:
                logger.warning(f"No valid examples for {graph_col}:{target}; skipping.")
                continue

            fold_r2: list[float] = []
            for fold in range(n_outer_folds):
                seed = random_seed + fold * 1000
                train_idx, val_idx = _random_train_val_split(len(y), train_ratio, seed)

                X_train, X_val = X_valid[train_idx], X_valid[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = HimalayaRidgeCV(
                    alphas=alpha_grid,
                    cv=inner_cv_folds,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                fold_r2.append(float(r2_score(y_val, y_pred)))

            key = f"{graph_col}_{target}"
            results[key] = float(np.mean(fold_r2))
            fold_scores[key] = np.array(fold_r2, dtype=float)

            if verbose:
                logger.info(f"{key}: R² = {results[key]:.4f}")

    return results, fold_scores


@hydra.main(
    version_base=None,
    config_path="../../configs/modeling",
    config_name="structural_probing",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for structural probing analysis.

    Configuration
    ---------------
    model : str
        HuggingFace model identifier.
    embeddings_dir : str
        Directory containing extracted embeddings.
    layer : int
        Layer index to extract embeddings from.
    output_dir : str
        Directory to save results.
    n_inner_folds : int
        Number of inner CV folds for hyperparameter selection.
    """
    backend = cfg.get("himalaya_backend", "numpy")
    set_backend(backend)
    logger.info(f"Himalaya backend set to {backend}")

    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJ_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_label = cfg.model.replace("/", "--")
    layer_cfg = cfg.get("layer", None)
    vision_layer_cfg = cfg.get("vision_layer", None)
    text_layer_cfg = cfg.get("text_layer", None)

    dataset_name = cfg.get(
        "dataset_hf_identifier",
        cfg.get("dataset_name", "helena-balabin/vg_coco_graphs_merged"),
    )
    split = cfg.get("dataset_split", cfg.get("split", "train"))
    df = load_graph_dataset(dataset_name=dataset_name, split=split)

    coco_id_col = cfg.get("coco_id_column", "cocoid")
    if coco_id_col not in df.columns:
        for fallback in ("cocoid_x", "coco_id"):
            if fallback in df.columns:
                coco_id_col = fallback
                break
    if coco_id_col not in df.columns:
        raise KeyError(f"COCO ID column '{coco_id_col}' not found in dataset columns")

    text_graph_columns = cfg.get(
        "text_graph_columns",
        ["amr_graphs", "dependency_graphs"],
    )
    image_graph_columns = cfg.get(
        "image_graph_columns",
        ["image_graphs", "action_image_graphs", "spatial_image_graphs"],
    )
    graph_targets = cfg.get("graph_targets", ["num_nodes", "num_edges", "depth"])

    n_outer_folds = cfg.get("n_outer_folds", 5)
    inner_cv_folds = cfg.get("inner_cv_folds", 5)
    train_ratio = cfg.get("train_ratio", 0.8)
    random_seed = cfg.get("random_seed", 0)
    alpha_grid = np.array(cfg.get("alpha_grid", np.logspace(-3, 3, 7)))

    # Resolve embedding layers
    vision_layer: int | None = None
    text_layer: int | None = None

    # Load and probe embeddings (on-the-fly)
    vision_results: dict[str, float] = {}
    vision_folds: dict[str, np.ndarray] = {}
    text_results: dict[str, float] = {}
    text_folds: dict[str, np.ndarray] = {}

    cache_dir: str | None = cfg.get("cache_dir", None)
    device = torch.device(cfg.get("device", "cpu"))
    dtype = _DTYPES[cfg.get("dtype", "float32")]
    pooling: str = cfg.get("pooling", "mean")
    batch_size: int = cfg.get("batch_size", 8)
    max_length: int = cfg.get("max_length", 512)
    model_type: str = cfg.get("model_type", "vlm").lower()

    logger.info("Loading model & processor for on-the-fly embeddings …")
    processor = _load_processor(cfg.model, cache_dir, model_type=model_type)
    model = _load_model(cfg.model, dtype, cache_dir).to(device).eval()

    if vision_layer_cfg is not None or text_layer_cfg is not None:
        vision_layer = int(vision_layer_cfg) if vision_layer_cfg is not None else None
        text_layer = int(text_layer_cfg) if text_layer_cfg is not None else None
    elif layer_cfg is None or (isinstance(layer_cfg, str) and layer_cfg.lower() == "auto"):
        vision_layer = _auto_select_layer_from_model(model, "vision")
        text_layer = _auto_select_layer_from_model(model, "text")
    else:
        vision_layer = int(layer_cfg)
        text_layer = int(layer_cfg)

    meta_df = _load_embedding_metadata_df(cfg, df, coco_id_col)

    if model_type in ("vlm", "vision_only") and vision_layer is not None:
        image_dir = Path(cfg.image_dir)
        if not image_dir.is_absolute():
            image_dir = PROJ_ROOT / image_dir

        vision_meta = meta_df.dropna(subset=["filepath"]).reset_index(drop=True)
        if vision_meta.empty:
            logger.warning("No valid image filepaths found for vision embeddings.")
        else:
            filenames = vision_meta["filepath"].astype(str).tolist()
            coco_ids = vision_meta["coco_id"].astype(int).to_numpy()
            unique_images, broadcast = _deduplicate_images(image_dir, filenames)
            logger.info("Extracting vision embeddings on the fly …")
            vis_embs = extract_vision_embeddings(
                model,
                processor,
                unique_images,
                device=device,
                dtype=dtype,
                pooling=pooling,
                layer_indices=[vision_layer],
            )[vision_layer]
            vision_pack = (coco_ids, vis_embs[broadcast])
            logger.info("\nProbing vision embeddings")
            vision_coco_ids, vision_embs = vision_pack
            vision_results, vision_folds = run_graph_probing(
                embeddings=vision_embs,
                embed_coco_ids=vision_coco_ids,
                df=df,
                coco_id_col=coco_id_col,
                graph_columns=image_graph_columns,
                targets=graph_targets,
                n_outer_folds=n_outer_folds,
                inner_cv_folds=inner_cv_folds,
                train_ratio=train_ratio,
                alpha_grid=alpha_grid,
                random_seed=random_seed,
            )

    if model_type in ("vlm", "language_only") and text_layer is not None:
        text_meta = meta_df.dropna(subset=["text"]).reset_index(drop=True)
        text_meta["text"] = text_meta["text"].astype(str)
        text_meta = text_meta[text_meta["text"].str.strip() != ""].reset_index(drop=True)
        if text_meta.empty:
            logger.warning("No valid text entries found for text embeddings.")
        else:
            texts = text_meta["text"].astype(str).tolist()
            coco_ids = text_meta["coco_id"].astype(int).to_numpy()
            logger.info("Extracting text embeddings on the fly …")
            text_embs = extract_text_embeddings(
                model,
                processor,
                texts,
                device=device,
                pooling=pooling,
                layer_indices=[text_layer],
                batch_size=batch_size,
                max_length=max_length,
            )[text_layer]
            logger.info("\nProbing text embeddings")
            text_results, text_folds = run_graph_probing(
                embeddings=text_embs,
                embed_coco_ids=coco_ids,
                df=df,
                coco_id_col=coco_id_col,
                graph_columns=text_graph_columns,
                targets=graph_targets,
                n_outer_folds=n_outer_folds,
                inner_cv_folds=inner_cv_folds,
                train_ratio=train_ratio,
                alpha_grid=alpha_grid,
                random_seed=random_seed,
            )

    if not vision_results and not text_results:
        raise FileNotFoundError(
            "No embeddings found for either vision or text. "
            "Check embeddings_dir, model label, and layer."
        )

    # Combine results
    all_results: dict[str, float | str | int] = {
        "vision_layer": vision_layer if vision_layer is not None else "",
        "text_layer": text_layer if text_layer is not None else "",
    }
    for k, v in vision_results.items():
        all_results[f"vision_{k}"] = v
    for k, v in text_results.items():
        all_results[f"text_{k}"] = v

    # Save fold scores for significance testing
    fold_records: list[dict[str, float | str | int]] = []
    for target, arr in vision_folds.items():
        for fold_idx, score in enumerate(arr):
            fold_records.append(
                {
                    "target": f"vision_{target}",
                    "fold": int(fold_idx),
                    "r2": float(score),
                }
            )
    for target, arr in text_folds.items():
        for fold_idx, score in enumerate(arr):
            fold_records.append(
                {
                    "target": f"text_{target}",
                    "fold": int(fold_idx),
                    "r2": float(score),
                }
            )

    # Save results
    results_df = pd.DataFrame([all_results])
    results_path = output_dir / model_label / "structural_probing_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    if fold_records:
        folds_df = pd.DataFrame(fold_records)
        folds_path = output_dir / model_label / "structural_probing_folds.csv"
        folds_df.to_csv(folds_path, index=False)
        logger.info(f"Saved fold scores to {folds_path}")

    logger.success(f"Structural probing complete! Results saved to {results_path}")


if __name__ == "__main__":
    main()
