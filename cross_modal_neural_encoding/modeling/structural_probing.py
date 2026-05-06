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
    n_outer_folds: int = 5,
    n_inner_folds: int = 5,
    alpha_grid: np.ndarray = np.logspace(-3, 3, 7),
    verbose: bool = True
) -> dict[str, float]:
    """Run structural probing analysis to predict graph properties from embeddings.

    Uses nested 5-fold cross-validation: the outer loop evaluates R² on held-out
    folds (averaged across folds), and the inner loop selects the ridge alpha on
    the outer training portion. This matches the procedure in §3.2 of the paper.

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
    n_outer_folds : int
        Number of outer CV folds for R² evaluation (averaged across folds).
    n_inner_folds : int
        Number of inner CV folds for hyperparameter selection.
    alpha_grid : np.ndarray
        Grid of regularization strengths to search over.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    results : dict[str, float]
        R² scores for each target, averaged across outer CV folds.
    """
    results = {}

    # Align embeddings with targets by COCO ID
    embed_id_to_idx = {int(cid): i for i, cid in enumerate(embed_coco_ids)}
    common_ids = np.array([cid for cid in target_coco_ids if int(cid) in embed_id_to_idx])

    if len(common_ids) == 0:
        raise ValueError("No overlapping COCO IDs between embeddings and targets")

    X_list = []
    y_lists = {name: [] for name in target_names}

    target_id_to_idx = {int(cid): i for i, cid in enumerate(target_coco_ids)}
    for cid in common_ids:
        X_list.append(embeddings[embed_id_to_idx[int(cid)]])
        for name in target_names:
            y_lists[name].append(targets[name][target_id_to_idx[int(cid)]])

    X = np.array(X_list)
    y_data = {name: np.array(y_lists[name]) for name in target_names}
    groups = common_ids  # one group per unique stimulus

    actual_outer = min(n_outer_folds, len(np.unique(groups)))
    outer_cv = GroupKFold(n_splits=actual_outer)

    for target_name in target_names:
        y = y_data[target_name]

        if np.var(y) < 1e-10:
            results[target_name] = 0.0
            if verbose:
                logger.warning(f"Target {target_name} has zero variance, setting R² to 0")
            continue

        fold_r2: list[float] = []

        for outer_train_idx, outer_test_idx in outer_cv.split(X, groups=groups):
            X_train, X_test = X[outer_train_idx], X[outer_test_idx]
            y_train, y_test = y[outer_train_idx], y[outer_test_idx]
            groups_train = groups[outer_train_idx]

            # Inner CV: select best alpha on the outer training portion
            actual_inner = min(n_inner_folds, len(np.unique(groups_train)))
            inner_cv = GroupKFold(n_splits=actual_inner)

            alpha_scores = np.zeros(len(alpha_grid))
            n_inner_splits = 0
            for in_train_idx, in_val_idx in inner_cv.split(X_train, groups=groups_train):
                X_in, X_val = X_train[in_train_idx], X_train[in_val_idx]
                y_in, y_val = y_train[in_train_idx], y_train[in_val_idx]
                for ai, alpha in enumerate(alpha_grid):
                    ridge = Ridge(alpha=alpha, fit_intercept=True)
                    ridge.fit(X_in, y_in)
                    alpha_scores[ai] += ridge.score(X_val, y_val)
                n_inner_splits += 1

            best_alpha = alpha_grid[np.argmax(alpha_scores / max(n_inner_splits, 1))]

            # Evaluate on outer test fold
            final_ridge = Ridge(alpha=best_alpha, fit_intercept=True)
            final_ridge.fit(X_train, y_train)
            fold_r2.append(float(final_ridge.score(X_test, y_test)))

        r2_score = float(np.mean(fold_r2))
        results[target_name] = r2_score

        if verbose:
            logger.info(
                f"{target_name}: R² = {r2_score:.4f} "
                f"(mean across {actual_outer} outer folds)"
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

    n_outer_folds = cfg.get("n_outer_folds", 5)

    vision_results = run_structural_probing(
        vision_embs,
        vision_coco_ids,
        targets,
        target_coco_ids,
        action_targets,
        n_outer_folds=n_outer_folds,
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
        n_outer_folds=n_outer_folds,
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
