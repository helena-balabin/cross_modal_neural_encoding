"""Predict embeddings across modalities (text ↔ vision).

This script fits linear models to predict image-encoder embeddings from
text-encoder embeddings and vice versa, using 5-fold cross-validation
and Pearson correlation as the evaluation metric.

Usage
-----
    python -m cross_modal_neural_encoding.modeling.predict_modalities

Hydra config: ``configs/modeling/predict_modalities.yaml``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import hydra
import numpy as np
import pandas as pd
import torch
from torch import nn
from loguru import logger
from omegaconf import DictConfig
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from cross_modal_neural_encoding.config import PROJ_ROOT


@dataclass
class EmbeddingBundle:
    coco_ids: np.ndarray
    embeddings: np.ndarray
    layer: int
    model_dir: Path


def _normalize_model_label(model_name: str) -> str:
    return model_name.replace("/", "--") if "/" in model_name else model_name


def _discover_model_dirs(embeddings_dir: Path) -> dict[str, Path]:
    if not embeddings_dir.exists():
        return {}

    model_dirs: dict[str, Path] = {}
    for path in embeddings_dir.rglob("*"):
        if not path.is_dir():
            continue
        if path.name not in {"text_embeddings", "vision_embeddings"}:
            continue
        parent = path.parent
        if parent.name not in model_dirs:
            model_dirs[parent.name] = parent

    return dict(sorted(model_dirs.items()))


def _list_layers(modality_dir: Path) -> list[int]:
    if not modality_dir.exists():
        return []
    layers: list[int] = []
    for p in modality_dir.glob("layer_*.npy"):
        try:
            layers.append(int(p.stem.split("_")[-1]))
        except (ValueError, IndexError):
            continue
    return sorted(set(layers))


def _pick_middle_layer(layers: list[int]) -> int | None:
    if not layers:
        return None
    return layers[len(layers) // 2]


def _parse_layer(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in {"", "none", "null", "auto"}:
            return None
        return int(value)
    if isinstance(value, (int, np.integer)):
        if int(value) < 0:
            return None
        return int(value)
    return None


def _load_embeddings(
    model_dir: Path,
    model_label: str,
    modality: str,
    layer: int | None,
) -> EmbeddingBundle | None:
    modality_dir = model_dir / f"{modality}_embeddings"
    if not modality_dir.exists():
        return None

    available_layers = _list_layers(modality_dir)
    if not available_layers:
        return None

    if layer is None:
        layer = _pick_middle_layer(available_layers)
    elif layer not in available_layers:
        fallback = _pick_middle_layer(available_layers)
        logger.info(
            f"Requested layer {layer} not available for {model_label}/{modality}; "
            f"using {fallback} instead."
        )
        layer = fallback

    if layer is None:
        return None

    layer_file = modality_dir / f"layer_{layer:03d}.npy"
    coco_file = modality_dir / "coco_ids.npy"
    if not layer_file.exists() or not coco_file.exists():
        return None

    coco_ids = np.load(coco_file).astype(int)
    embeddings = np.load(layer_file)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings for {model_label}/{modality}, "
            f"got shape {embeddings.shape}."
        )

    return EmbeddingBundle(
        coco_ids=coco_ids,
        embeddings=embeddings,
        layer=layer,
        model_dir=model_dir,
    )


def _aggregate_by_coco_id(bundle: EmbeddingBundle) -> EmbeddingBundle:
    coco_ids = bundle.coco_ids
    embeddings = bundle.embeddings
    unique_ids, inverse = np.unique(coco_ids, return_inverse=True)
    sums = np.zeros((len(unique_ids), embeddings.shape[1]), dtype=np.float64)
    counts = np.zeros(len(unique_ids), dtype=np.int64)
    for row_idx, group_idx in enumerate(inverse):
        sums[group_idx] += embeddings[row_idx]
        counts[group_idx] += 1
    counts[counts == 0] = 1
    means = sums / counts[:, None]
    return EmbeddingBundle(
        coco_ids=unique_ids,
        embeddings=means,
        layer=bundle.layer,
        model_dir=bundle.model_dir,
    )


def _filter_to_coco_ids(
    bundle: EmbeddingBundle,
    desired_ids: Iterable[int] | None,
) -> EmbeddingBundle:
    if desired_ids is None:
        return bundle
    desired_ids = list(dict.fromkeys(int(x) for x in desired_ids))
    lookup = {int(cid): i for i, cid in enumerate(bundle.coco_ids)}
    available_ids = [cid for cid in desired_ids if cid in lookup]
    indices = [lookup[cid] for cid in available_ids]
    return EmbeddingBundle(
        coco_ids=np.asarray(available_ids, dtype=int),
        embeddings=bundle.embeddings[indices],
        layer=bundle.layer,
        model_dir=bundle.model_dir,
    )


def _load_design_matrix_coco_ids(mapping_file: Path) -> list[int] | None:
    if not mapping_file.exists():
        logger.warning(
            f"Design-matrix mapping file not found: {mapping_file}. "
            "Using all available coco IDs."
        )
        return None

    df = pd.read_csv(mapping_file)
    coco_col = None
    for candidate in ("coco_id", "cocoid"):
        if candidate in df.columns:
            coco_col = candidate
            break
    if coco_col is None:
        raise KeyError(
            "Design mapping must contain a 'coco_id' or 'cocoid' column."
        )

    coco_ids: list[int] = []
    for raw in df[coco_col].astype(str):
        if "_" in raw:
            stem = raw.rsplit("_", 1)[0]
        else:
            stem = raw
        try:
            coco_ids.append(int(stem))
        except ValueError:
            continue

    coco_ids = sorted(set(coco_ids))
    if not coco_ids:
        logger.warning(
            "No valid coco IDs parsed from design mapping file. "
            "Using all available coco IDs."
        )
        return None
    return coco_ids


def _pearson_r_columns(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = y_true - y_true.mean(axis=0, keepdims=True)
    y_pred = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (y_true * y_pred).sum(axis=0)
    den = np.sqrt((y_true**2).sum(axis=0) * (y_pred**2).sum(axis=0))
    den[den == 0] = np.nan
    return num / den


class SkipMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_size: int,
        bias: bool,
        use_relu: bool,
    ) -> None:
        super().__init__()
        self.use_relu = use_relu
        self.fc1 = nn.Linear(input_dim, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc4 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out = nn.Linear(hidden_size, output_dim, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.fc1(x)
        if self.use_relu:
            h1 = self.relu(h1)
        h2 = self.fc2(h1) + h1
        if self.use_relu:
            h2 = self.relu(h2)
        h3 = self.fc3(h2) + h2
        if self.use_relu:
            h3 = self.relu(h3)
        h4 = self.fc4(h3) + h3
        if self.use_relu:
            h4 = self.relu(h4)
        return self.out(h4)


def _evaluate_cv(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_splits: int,
    ridge_alpha: float,
    random_state: int,
    standardize: bool,
    regressor: str,
    mlp_config: dict,
) -> tuple[float, float, list[float]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        logger.info(
            f"  Fold {fold_idx}/{n_splits}: train={len(train_idx)} test={len(test_idx)}"
        )
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        if standardize:
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            X_train = x_scaler.fit_transform(X_train)
            X_test = x_scaler.transform(X_test)
            Y_train = y_scaler.fit_transform(Y_train)
            Y_test_scaled = y_scaler.transform(Y_test)
        else:
            Y_test_scaled = Y_test

        if regressor == "ridge":
            model = Ridge(alpha=ridge_alpha)
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)

            if standardize:
                Y_pred = y_scaler.inverse_transform(Y_pred)
                Y_test_eval = y_scaler.inverse_transform(Y_test_scaled)
            else:
                Y_test_eval = Y_test

            r = _pearson_r_columns(Y_test_eval, Y_pred)
        else:
            if not standardize:
                logger.warning(
                    "MLP regressors are sensitive to scaling; "
                    "consider standardize=true."
                )

            val_ratio = float(mlp_config.get("val_ratio", 0.1))
            val_ratio = min(max(val_ratio, 0.05), 0.3)
            rng = np.random.default_rng(random_state)
            idx = np.arange(len(X_train))
            rng.shuffle(idx)
            split = int(len(idx) * (1.0 - val_ratio))
            train_idx, val_idx = idx[:split], idx[split:]

            X_tr = torch.tensor(X_train[train_idx], dtype=torch.float32)
            Y_tr = torch.tensor(Y_train[train_idx], dtype=torch.float32)
            X_val = torch.tensor(X_train[val_idx], dtype=torch.float32)
            Y_val = torch.tensor(Y_train[val_idx], dtype=torch.float32)

            device = str(mlp_config.get("device", "cuda")).lower()
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            device_t = torch.device(device)

            model = SkipMLP(
                X_train.shape[1],
                Y_train.shape[1],
                hidden_size=int(mlp_config.get("hidden_size", 256)),
                bias=bool(mlp_config.get("bias", True)),
                use_relu=regressor == "mlp_relu",
            ).to(device_t)

            lr = float(mlp_config.get("learning_rate", 1e-3))
            weight_decay = float(mlp_config.get("weight_decay", 0.0))
            batch_size = int(mlp_config.get("batch_size", 128))
            max_epochs = int(mlp_config.get("max_epochs", 256))
            patience = int(mlp_config.get("patience", 8))
            log_interval = int(mlp_config.get("log_interval", 25))

            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            loss_fn = nn.MSELoss()

            best_val = float("inf")
            best_state: dict[str, torch.Tensor] | None = None
            epochs_no_improve = 0

            model.train()
            for _ in range(max_epochs):
                epoch = _ + 1
                perm = torch.randperm(X_tr.size(0))
                for start in range(0, X_tr.size(0), batch_size):
                    idx_batch = perm[start : start + batch_size]
                    xb = X_tr[idx_batch].to(device_t)
                    yb = Y_tr[idx_batch].to(device_t)
                    optimizer.zero_grad(set_to_none=True)
                    preds = model(xb)
                    loss = loss_fn(preds, yb)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val.to(device_t))
                    val_loss = loss_fn(val_pred, Y_val.to(device_t)).item()
                model.train()

                if log_interval > 0 and (epoch == 1 or epoch % log_interval == 0):
                    logger.info(
                        f"    Epoch {epoch}/{max_epochs} val_loss={val_loss:.6f}"
                    )

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        logger.info(
                            f"    Early stop at epoch {epoch} (best val_loss={best_val:.6f})"
                        )
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            model.eval()
            with torch.no_grad():
                X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device_t)
                Y_pred = model(X_test_t).cpu().numpy()

            if standardize:
                Y_pred = y_scaler.inverse_transform(Y_pred)
                Y_test_eval = y_scaler.inverse_transform(Y_test_scaled)
            else:
                Y_test_eval = Y_test

            r = _pearson_r_columns(Y_test_eval, Y_pred)

        fold_scores.append(float(np.nanmean(r)))

    mean_r = float(np.nanmean(fold_scores))
    std_r = float(np.nanstd(fold_scores))
    return mean_r, std_r, fold_scores


def _resolve_model_list(
    models: Iterable[str] | None,
    model_dirs: dict[str, Path],
    modality: str,
    layer: int | None,
) -> dict[str, EmbeddingBundle]:
    if models:
        requested_labels = [_normalize_model_label(m) for m in models]
        model_labels = [lbl for lbl in requested_labels if lbl in model_dirs]
    else:
        model_labels = list(model_dirs.keys())

    resolved: dict[str, EmbeddingBundle] = {}
    for label in model_labels:
        bundle = _load_embeddings(model_dirs[label], label, modality, layer)
        if bundle is None:
            logger.warning(
                f"Skipping {label} for {modality}: missing embeddings or layer."
            )
            continue
        resolved[label] = bundle
        logger.info(
            f"Loaded {modality} embeddings for {label} "
            f"(layer {bundle.layer}, n={len(bundle.coco_ids)})"
        )
    return resolved


@hydra.main(
    version_base=None,
    config_path="../../configs/modeling",
    config_name="predict_modalities",
)
def main(cfg: DictConfig) -> None:
    embeddings_dir = Path(cfg.embeddings_dir)
    if not embeddings_dir.is_absolute():
        embeddings_dir = PROJ_ROOT / embeddings_dir

    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJ_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    design_mapping_value = str(cfg.get("design_matrix_mapping_file", "")).strip()
    if design_mapping_value:
        design_mapping = Path(design_mapping_value)
        if not design_mapping.is_absolute():
            design_mapping = PROJ_ROOT / design_mapping
        desired_ids = _load_design_matrix_coco_ids(design_mapping)
    else:
        desired_ids = None

    text_layer = _parse_layer(cfg.get("text_layer"))
    vision_layer = _parse_layer(cfg.get("vision_layer"))
    logger.info(f"Scanning embeddings under {embeddings_dir}")
    model_dirs = _discover_model_dirs(embeddings_dir)
    if not model_dirs:
        raise ValueError(
            f"No embedding model directories found under {embeddings_dir}."
        )
    logger.info(f"Discovered {len(model_dirs)} model directories")

    text_cache = _resolve_model_list(
        cfg.get("text_models"),
        model_dirs,
        "text",
        text_layer,
    )
    vision_cache = _resolve_model_list(
        cfg.get("vision_models"),
        model_dirs,
        "vision",
        vision_layer,
    )

    if not text_cache or not vision_cache:
        raise ValueError("No valid text or vision models found to evaluate.")

    logger.info(
        f"Text models: {len(text_cache)} | Vision models: {len(vision_cache)}"
    )

    for label, bundle in list(text_cache.items()):
        bundle = _aggregate_by_coco_id(bundle)
        bundle = _filter_to_coco_ids(bundle, desired_ids)
        text_cache[label] = bundle
        logger.info(
            f"Text {label}: {len(bundle.coco_ids)} stimuli after filtering"
        )

    for label, bundle in list(vision_cache.items()):
        bundle = _aggregate_by_coco_id(bundle)
        bundle = _filter_to_coco_ids(bundle, desired_ids)
        vision_cache[label] = bundle
        logger.info(
            f"Vision {label}: {len(bundle.coco_ids)} stimuli after filtering"
        )

    n_splits = int(cfg.get("n_splits", 5))
    regressor = str(cfg.get("regressor", "mlp_relu")).lower()
    ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
    random_state = int(cfg.get("random_state", 42))
    standardize = bool(cfg.get("standardize", True))
    min_samples = int(cfg.get("min_samples", n_splits))
    mlp_config = dict(cfg.get("mlp", {}))

    logger.info(
        f"Regressor: {regressor} | standardize={standardize}"
    )

    records: list[dict] = []
    total_pairs = len(text_cache) * len(vision_cache)
    pair_idx = 0

    for text_label, text_bundle in text_cache.items():
        for vision_label, vision_bundle in vision_cache.items():
            pair_idx += 1
            logger.info(
                f"[{pair_idx}/{total_pairs}] {text_label} ↔ {vision_label}"
            )
            common_ids = np.intersect1d(
                text_bundle.coco_ids, vision_bundle.coco_ids
            )
            if common_ids.size < min_samples:
                logger.warning(
                    f"Skipping {text_label} -> {vision_label}: "
                    f"only {common_ids.size} common stimuli."
                )
                continue

            text_lookup = {int(cid): i for i, cid in enumerate(text_bundle.coco_ids)}
            vision_lookup = {int(cid): i for i, cid in enumerate(vision_bundle.coco_ids)}
            common_ids = np.asarray(sorted(common_ids), dtype=int)

            X_text = text_bundle.embeddings[
                [text_lookup[int(cid)] for cid in common_ids]
            ]
            Y_vision = vision_bundle.embeddings[
                [vision_lookup[int(cid)] for cid in common_ids]
            ]

            mean_r, std_r, fold_scores = _evaluate_cv(
                X_text,
                Y_vision,
                n_splits=n_splits,
                ridge_alpha=ridge_alpha,
                random_state=random_state,
                standardize=standardize,
                regressor=regressor,
                mlp_config=mlp_config,
            )
            logger.info(
                f"  text→vision mean r = {mean_r:.4f} (std {std_r:.4f})"
            )

            record = {
                "direction": "text_to_vision",
                "input_model": text_label,
                "output_model": vision_label,
                "input_modality": "text",
                "output_modality": "vision",
                "input_layer": text_bundle.layer,
                "output_layer": vision_bundle.layer,
                "n_samples": len(common_ids),
                "n_features_in": X_text.shape[1],
                "n_features_out": Y_vision.shape[1],
                "mean_r": mean_r,
                "std_r": std_r,
                "regressor": regressor,
            }
            for i, score in enumerate(fold_scores):
                record[f"fold_{i}"] = score
            records.append(record)

            mean_r, std_r, fold_scores = _evaluate_cv(
                Y_vision,
                X_text,
                n_splits=n_splits,
                ridge_alpha=ridge_alpha,
                random_state=random_state,
                standardize=standardize,
                regressor=regressor,
                mlp_config=mlp_config,
            )
            logger.info(
                f"  vision→text mean r = {mean_r:.4f} (std {std_r:.4f})"
            )

            record = {
                "direction": "vision_to_text",
                "input_model": vision_label,
                "output_model": text_label,
                "input_modality": "vision",
                "output_modality": "text",
                "input_layer": vision_bundle.layer,
                "output_layer": text_bundle.layer,
                "n_samples": len(common_ids),
                "n_features_in": Y_vision.shape[1],
                "n_features_out": X_text.shape[1],
                "mean_r": mean_r,
                "std_r": std_r,
                "regressor": regressor,
            }
            for i, score in enumerate(fold_scores):
                record[f"fold_{i}"] = score
            records.append(record)

    if not records:
        raise RuntimeError("No results produced. Check embeddings and config.")

    results_df = pd.DataFrame.from_records(records)
    results_path = output_dir / "predict_modalities_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.success(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
