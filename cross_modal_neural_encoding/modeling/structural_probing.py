"""
Implements structural probing framework to analyze compositional structure in VLM embeddings.

Probes for graph-based structural properties of paired text-image stimuli using ridge regression
on frozen encoder embeddings. The framework uses precomputed AMR graphs for text and
action graphs for images to extract structural targets.
"""

from __future__ import annotations

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from omegaconf import DictConfig

from .neural_encoding import load_embeddings

PROJ_ROOT = Path(__file__).parent.parent.parent

def load_structural_targets(
    dataset_name: str = "helena-balabin/coco_a_preprocessed_all",
    split: str = "train"
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load structural target properties from the COCO-A preprocessed dataset.

    Extracts node count, edge count, and graph depth for both AMR and action graphs.

    Returns
    -------
    coco_ids : (n_stimuli,) int array
        COCO IDs corresponding to each stimulus.
    targets : dict[str, np.ndarray]
        Dictionary containing the structural targets:
        - 'amr_n_nodes': node count for AMR graphs
        - 'amr_n_edges': edge count for AMR graphs
        - 'amr_graph_depth': depth for AMR graphs
        - 'coco_a_nodes': node count for action graphs
        - 'coco_a_edges': edge count for action graphs
        - 'coco_a_graph_depth': depth for action graphs
    """
    from datasets import load_dataset

    logger.info(f"Loading structural targets from dataset {dataset_name}, split {split}")
    dataset = load_dataset(dataset_name, split=split)

    # Extract COCO IDs and structural properties
    coco_ids = np.array([int(cid) for cid in dataset["cocoid_x"]])

    targets = {
        "amr_n_nodes": np.array(dataset["amr_n_nodes"]),
        "amr_n_edges": np.array(dataset["amr_n_edges"]),
        "amr_graph_depth": np.array(dataset["amr_graph_depth"]),
        "coco_a_nodes": np.array(dataset["coco_a_nodes"]),
        "coco_a_edges": np.array(dataset["coco_a_edges"]),
        "coco_a_graph_depth": np.array(dataset["coco_a_graph_depth"])
    }

    logger.info(f"Loaded structural targets for {len(coco_ids)} stimuli")
    return coco_ids, targets


def run_structural_probing(
    embeddings: np.ndarray,
    embed_coco_ids: np.ndarray,
    targets: dict[str, np.ndarray],
    target_coco_ids: np.ndarray,
    target_names: list[str],
    n_inner_folds: int = 5,
    alpha_grid: np.ndarray = np.logspace(-3, 3, 7),
    verbose: bool = True
) -> dict[str, float]:
    """Run structural probing analysis to predict graph properties from embeddings.

    Parameters
    ----------
    embeddings : (n_stimuli, n_features)
        VLM embeddings to be probed.
    embed_coco_ids : (n_stimuli,) int
        COCO IDs corresponding to the embeddings.
    targets : dict[str, np.ndarray]
        Dictionary containing the structural targets.
    target_coco_ids : (n_targets,) int
        COCO IDs corresponding to the targets.
    target_names : list[str]
        Names of the targets to probe for (keys in targets dict).
    n_inner_folds : int
        Number of inner CV folds for hyperparameter selection.
    alpha_grid : np.ndarray
        Grid of regularization strengths to search over.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    results : dict[str, float]
        R² scores for each target.
    """
    results = {}

    # Align embeddings with targets by COCO ID
    embed_id_to_idx = {int(cid): i for i, cid in enumerate(embed_coco_ids)}
    common_ids = [cid for cid in target_coco_ids if int(cid) in embed_id_to_idx]

    if len(common_ids) == 0:
        raise ValueError("No overlapping COCO IDs between embeddings and targets")

    X_list = []
    y_lists = {name: [] for name in target_names}

    for cid in common_ids:
        embed_idx = embed_id_to_idx[int(cid)]
        X_list.append(embeddings[embed_idx])

        target_idx = np.where(target_coco_ids == cid)[0][0]
        for name in target_names:
            y_lists[name].append(targets[name][target_idx])

    X = np.array(X_list)

    # Convert all targets to numpy arrays
    y_data = {name: np.array(y_lists[name]) for name in target_names}

    # Nested cross-validation for each target
    for target_name in target_names:
        y = y_data[target_name]

        # Skip if all values are the same (variance is zero)
        if np.var(y) < 1e-10:
            results[target_name] = 0.0
            if verbose:
                logger.warning(f"Target {target_name} has zero variance, setting R² to 0")
            continue

        # Group by stimulus ID for proper CV
        groups = np.array(common_ids)

        # Inner CV for hyperparameter selection
        inner_cv = GroupKFold(n_splits=n_inner_folds)
        best_alphas = []

        for train_idx, val_idx in inner_cv.split(X, groups=groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Search for best alpha
            best_score = -np.inf
            best_alpha = alpha_grid[0]

            for alpha in alpha_grid:
                ridge = Ridge(alpha=alpha, fit_intercept=True)
                ridge.fit(X_train, y_train)
                score = ridge.score(X_val, y_val)

                if score > best_score:
                    best_score = score
                    best_alpha = alpha

            best_alphas.append(best_alpha)

        # Use mean best alpha for final evaluation
        final_alpha = np.mean(best_alphas)

        # Evaluate on full dataset (standard practice for probing)
        final_ridge = Ridge(alpha=final_alpha, fit_intercept=True)
        final_ridge.fit(X, y)
        r2_score = final_ridge.score(X, y)

        results[target_name] = r2_score

        if verbose:
            logger.info(
                f"{target_name}: R² = {r2_score:.4f}, "
                f"alpha = {final_alpha:.3e}"
            )

    return results


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
    layer = cfg.layer
    n_inner_folds = cfg.n_inner_folds

    # Load structural targets
    target_coco_ids, targets = load_structural_targets()

    # Define the targets to probe for
    amr_targets = ["amr_n_nodes", "amr_n_edges", "amr_graph_depth"]
    action_targets = ["coco_a_nodes", "coco_a_edges", "coco_a_graph_depth"]

    # Load and probe vision embeddings
    logger.info("\nProbing vision embeddings")
    vision_coco_ids, vision_embs = load_embeddings(
        embeddings_dir, model_label, "vision", layer
    )

    vision_results = run_structural_probing(
        vision_embs,
        vision_coco_ids,
        targets,
        target_coco_ids,
        action_targets,
        n_inner_folds=n_inner_folds
    )

    # Load and probe text embeddings
    logger.info("\nProbing text embeddings")
    text_coco_ids, text_embs = load_embeddings(
        embeddings_dir, model_label, "text", layer
    )

    text_results = run_structural_probing(
        text_embs,
        text_coco_ids,
        targets,
        target_coco_ids,
        amr_targets,
        n_inner_folds=n_inner_folds
    )

    # Combine results
    all_results = {}
    for k, v in vision_results.items():
        all_results[f"vision_{k}"] = v
    for k, v in text_results.items():
        all_results[f"text_{k}"] = v

    # Save results
    results_df = pd.DataFrame([all_results])
    results_path = output_dir / model_label / "structural_probing_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    logger.success(f"Structural probing complete! Results saved to {results_path}")


if __name__ == "__main__":
    main()
