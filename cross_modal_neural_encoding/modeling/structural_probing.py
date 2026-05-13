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

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from omegaconf import DictConfig
from typing import cast

from .neural_encoding import load_embeddings

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


def _auto_select_layer_for_modality(
    embeddings_dir: Path,
    model_label: str,
    modality: str,
) -> int | None:
    emb_dir = embeddings_dir / model_label / f"{modality}_embeddings"
    layers = _list_available_layers(emb_dir)
    if not layers:
        logger.warning(
            f"No {modality} layers found for {model_label} in {emb_dir}."
        )
        return None
    layer = _pick_middle_layer(layers)
    if layer is None:
        return None
    logger.info(
        f"Auto-selected {modality} layer {layer} for {model_label}"
    )
    return layer


def _load_embeddings_if_available(
    embeddings_dir: Path,
    model_label: str,
    embed_modality: str,
    layer: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load embeddings if present; return None when missing.

    This enables structural probing on unimodal models where only one
    of {vision,text} embeddings is available.
    """
    d = embeddings_dir / model_label / f"{embed_modality}_embeddings"
    coco_path = d / "coco_ids.npy"
    layer_path = d / f"layer_{layer:03d}.npy"
    if not d.exists() or not coco_path.exists() or not layer_path.exists():
        logger.warning(
            f"Skipping {embed_modality} embeddings: missing {d} or layer file."
        )
        return None
    return load_embeddings(embeddings_dir, model_label, embed_modality, layer)


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


def _get_graph_metric(graph_dict: object, key: str) -> int | float:
    if not isinstance(graph_dict, dict):
        return 0
    return graph_dict.get(key, 0)


def _extract_graph_labels(
    df: pd.DataFrame,
    graph_col: str,
    target: str,
) -> np.ndarray:
    if target == "num_nodes":
        return np.asarray(
            df[graph_col].apply(lambda g: _get_graph_metric(g, "num_nodes")).values
        )
    if target == "num_edges":
        return np.asarray(
            df[graph_col].apply(lambda g: _get_graph_metric(g, "num_edges")).values
        )
    if target == "depth":
        return np.asarray(
            df[graph_col].apply(lambda g: _get_graph_metric(g, "depth")).values
        )
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
                logger.warning(
                    f"No valid examples for {graph_col}:{target}; skipping."
                )
                continue

            fold_r2: list[float] = []
            for fold in range(n_outer_folds):
                seed = random_seed + fold * 1000
                train_idx, val_idx = _random_train_val_split(
                    len(y), train_ratio, seed
                )

                X_train, X_val = X_valid[train_idx], X_valid[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = RidgeCV(
                    alphas=alpha_grid,
                    cv=KFold(n_splits=inner_cv_folds, shuffle=False),
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
    raise ValueError(f"Unknown target: {target}")




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
    # Resolve paths
    embeddings_dir = Path(cfg.embeddings_dir)
    if not embeddings_dir.is_absolute():
        embeddings_dir = PROJ_ROOT / embeddings_dir

    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJ_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_label = cfg.model.replace("/", "--")
    layer_cfg = cfg.get("layer", None)
    if layer_cfg is None or (isinstance(layer_cfg, str) and layer_cfg.lower() == "auto"):
        vision_layer = _auto_select_layer_for_modality(
            embeddings_dir, model_label, "vision"
        )
        text_layer = _auto_select_layer_for_modality(
            embeddings_dir, model_label, "text"
        )
    else:
        vision_layer = int(layer_cfg)
        text_layer = int(layer_cfg)
    n_inner_folds = cfg.n_inner_folds

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
        raise KeyError(
            f"COCO ID column '{coco_id_col}' not found in dataset columns"
        )

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

    # Load and probe vision embeddings (if available)
    vision_results: dict[str, float] = {}
    vision_folds: dict[str, np.ndarray] = {}
    vision_pack = None
    if vision_layer is not None:
        vision_pack = _load_embeddings_if_available(
            embeddings_dir, model_label, "vision", vision_layer
        )
    if vision_pack is not None:
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

    # Load and probe text embeddings (if available)
    text_results: dict[str, float] = {}
    text_folds: dict[str, np.ndarray] = {}
    text_pack = None
    if text_layer is not None:
        text_pack = _load_embeddings_if_available(
            embeddings_dir, model_label, "text", text_layer
        )
    if text_pack is not None:
        logger.info("\nProbing text embeddings")
        text_coco_ids, text_embs = text_pack
        text_results, text_folds = run_graph_probing(
            embeddings=text_embs,
            embed_coco_ids=text_coco_ids,
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
