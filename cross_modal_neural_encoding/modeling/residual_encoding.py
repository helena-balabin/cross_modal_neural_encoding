"""Residual neural encoding: structure-targeted ablation of VLM embeddings.

Implements §3.3 of the paper.  For each encoding condition in the 2×2
cross-modal design, a ridge regression W* is fit on the training split to
predict each embedding dimension from the structural feature vector
**s** ∈ ℝ³ (node count, edge count, graph depth).  The residualised
embedding **ẽ** = **e** − W***s** is then used in the full encoding
pipeline in place of **e**.

A selective drop in cross-modal but not within-modality encoding accuracy
after residualisation constitutes evidence that compositional structure is
the principal carrier of cross-modal brain alignment.

Modality-specific structural features (§3.3):
    - text embeddings → AMR graph properties  (amr_n_nodes / edges / depth)
    - vision embeddings → action graph properties (coco_a_nodes / edges / depth)

An optional permuted-s control is also supported: **s** is randomly
permuted across stimuli before fitting W*, providing an empirical baseline
for the magnitude of accuracy change attributable to the residualisation
procedure itself.

Usage
-----
    python -m cross_modal_neural_encoding.modeling.residual_encoding

Hydra config: ``configs/modeling/residual_encoding.yaml``
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from tqdm import tqdm

from cross_modal_neural_encoding.config import PROJ_ROOT
from cross_modal_neural_encoding.modeling.neural_encoding import (
    align_single_trials,
    build_structural_feature_vectors,
    build_events_from_stimorder,
    load_designinfo_stimulus_ids_and_num_runs,
    load_embeddings,
    load_fmri,
    load_condition_to_cocoid_modality,
    run_encoding,
)
from cross_modal_neural_encoding.modeling.structural_probing import load_structural_targets
from cross_modal_neural_encoding.utils import (
    normalize_betas_per_run,
    compute_nc_by_modality,
    load_design_matrix_mapping,
    load_brain_mask,
    load_brain_mask_img,
    save_voxelwise_nifti,
)


@hydra.main(
    version_base=None,
    config_path="../../configs/modeling",
    config_name="residual_encoding",
)
def main(cfg: DictConfig) -> None:
    """Run residual neural encoding analysis across subjects and conditions."""

    # -- resolve paths -------------------------------------------------------
    glmsingle_root = Path(cfg.glmsingle_root)
    embeddings_dir = Path(cfg.embeddings_dir)
    if not embeddings_dir.is_absolute():
        embeddings_dir = PROJ_ROOT / embeddings_dir
    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJ_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_label: str = cfg.model.replace("/", "--")
    vision_layer: int = cfg.vision_layer
    text_layer: int = cfg.text_layer
    layer_for_modality = {"vision": vision_layer, "text": text_layer}
    n_pca: int = cfg.n_pca_components
    n_inner_folds: int = cfg.n_inner_folds
    n_outer_folds: int = int(cfg.get("n_outer_folds", 1))
    test_size: float = cfg.test_size
    nc_top_percent: float = float(cfg.get("nc_top_percent", 0.0))
    nc_num_averages: float = float(cfg.get("nc_num_averages", 6))
    design_matrix_mapping_file = Path(cfg.get("design_matrix_mapping_file", ""))
    frac_grid: np.ndarray = np.asarray(
        cfg.get("frac_grid", np.arange(0.1, 1.1, 0.1)),
        dtype="float32",
    )
    residual_alpha: float = float(cfg.get("residual_alpha", 1.0))
    run_permuted_control: bool = bool(cfg.get("run_permuted_control", True))
    conditions: dict = OmegaConf.to_container(cfg.conditions, resolve=True)  # type: ignore[assignment]

    # -- load structural targets (full COCO-A, aligned by COCO ID) -----------
    logger.info("Loading structural targets …")
    target_coco_ids, struct_targets = load_structural_targets(
        dataset_name=cfg.get(
            "dataset_name", "helena-balabin/coco_a_preprocessed_all"
        ),
        split=cfg.get("dataset_split", "train"),
    )

    # -- load & PCA embeddings (shared across subjects) ----------------------
    logger.info("Loading VLM embeddings …")
    embed_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for cond_cfg in conditions.values():
        emod = cond_cfg["embed_modality"]
        if emod in embed_data:
            continue
        lyr = layer_for_modality[emod]
        coco_ids, raw_embs = load_embeddings(embeddings_dir, model_label, emod, lyr)
        n_comp = min(n_pca, raw_embs.shape[0], raw_embs.shape[1])
        pca = PCA(n_components=n_comp)
        reduced = pca.fit_transform(raw_embs)
        logger.info(
            f"  PCA {emod}: {raw_embs.shape[1]}d → {n_comp}d "
            f"({pca.explained_variance_ratio_.sum():.1%} variance retained)"
        )
        embed_data[emod] = (coco_ids, reduced)

    # -- per-subject encoding ------------------------------------------------
    subjects: list[str] = list(cfg.subjects)
    summary_rows: list[dict] = []

    if not design_matrix_mapping_file.exists():
        raise FileNotFoundError(
            "design_matrix_mapping_file is required. "
            f"Missing: {design_matrix_mapping_file}"
        )
    condition_to_coco = load_condition_to_cocoid_modality(design_matrix_mapping_file)
    modality_map = load_design_matrix_mapping(design_matrix_mapping_file)

    for subject in tqdm(subjects, desc="Subjects"):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Subject: {subject}")

        betas_full = load_fmri(glmsingle_root, subject)
        fmriprep_dir = Path(cfg.get("fmriprep_dir", ""))
        brain_mask = load_brain_mask(fmriprep_dir, subject)
        brain_mask_img = load_brain_mask_img(fmriprep_dir, subject)

        stimulus_ids, num_runs = load_designinfo_stimulus_ids_and_num_runs(
            glmsingle_root, subject, n_trials=betas_full.shape[1]
        )
        betas_full_trials = normalize_betas_per_run(betas_full.T, num_runs=num_runs)
        betas = betas_full_trials.T[brain_mask, :]

        nc_by_modality_pct = compute_nc_by_modality(
            betas_full_trials, stimulus_ids, modality_map, num_averages=nc_num_averages
        )
        nc_corr_by_modality_full = {
            m: np.sqrt(np.clip(v, 0, None) / 100.0)
            for m, v in nc_by_modality_pct.items()
        }

        events_df = build_events_from_stimorder(
            stimulus_ids, condition_to_coco, subject
        )

        # Build per-modality fMRI cache (same logic as neural_encoding.py)
        fmri_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for fmri_mod in {c["fmri_modality"] for c in conditions.values()}:
            mod_df = events_df[events_df["modality"] == fmri_mod]
            trial_indices = np.asarray(mod_df["beta_index"].values, dtype=int)
            trial_coco_ids = np.asarray(mod_df["cocoid"].values, dtype=int)
            trial_betas = betas[:, trial_indices].T

            nc_ceiling = nc_corr_by_modality_full[fmri_mod][brain_mask]
            n_in_brain = brain_mask.sum()
            if nc_top_percent > 0:
                valid_nc = np.isfinite(nc_ceiling) & (nc_ceiling > 0)
                if valid_nc.any():
                    cutoff = np.nanpercentile(
                        nc_ceiling[valid_nc], 100.0 - nc_top_percent
                    )
                    voxel_keep = valid_nc & (nc_ceiling >= cutoff)
                else:
                    voxel_keep = np.isfinite(nc_ceiling)
                nc_ceiling = nc_ceiling[voxel_keep]
            else:
                voxel_keep = np.ones(n_in_brain, dtype=bool)

            trial_betas = trial_betas[:, voxel_keep]
            fmri_cache[fmri_mod] = (trial_coco_ids, trial_betas, nc_ceiling, voxel_keep)

        # Run each condition with residualised embeddings
        for cond_name, cond_cfg in tqdm(
            conditions.items(), desc="    Conditions", leave=False
        ):
            emod = cond_cfg["embed_modality"]
            fmod = cond_cfg["fmri_modality"]

            embed_ids, embed_feats = embed_data[emod]
            trial_cids, trial_betas, noise_ceiling_r, voxel_keep = fmri_cache[fmod]

            X, Y, groups = align_single_trials(
                embed_ids, embed_feats, trial_cids, trial_betas
            )

            # Build structural feature vectors s ∈ ℝ³ for each trial,
            # aligned to the study's 252 COCO IDs only.
            s = build_structural_feature_vectors(
                groups, struct_targets, target_coco_ids, emod
            )
            valid_s = np.all(np.isfinite(s), axis=1)
            if not valid_s.all():
                logger.warning(
                    f"    {valid_s.sum()}/{len(valid_s)} trials have structural "
                    "features — dropping the rest."
                )
            X, Y, groups, s = X[valid_s], Y[valid_s], groups[valid_s], s[valid_s]

            n_unique = len(np.unique(groups))
            if n_unique < max(n_outer_folds, n_inner_folds + 1):
                logger.warning(
                    f"    Too few unique stimuli ({n_unique}) — skipping {cond_name}."
                )
                continue

            # Variants: real s, and optionally permuted s (control)
            variants: list[tuple[str, np.ndarray]] = [(cond_name, s)]
            if run_permuted_control:
                s_perm = s[np.random.default_rng(42).permutation(len(s))]
                variants.append((f"permuted_{cond_name}", s_perm))

            for label, s_input in variants:
                logger.info(
                    f"  {'[permuted] ' if 'permuted' in label else ''}"
                    f"{emod} embed → {fmod} fMRI (residualised)"
                )
                result = run_encoding(
                    X, Y,
                    frac_grid=frac_grid,
                    groups=groups,
                    test_size=test_size,
                    n_inner_folds=n_inner_folds,
                    n_outer_folds=n_outer_folds,
                    noise_ceiling=noise_ceiling_r,
                    average_test_by_group=True,
                    structural_features=s_input,
                    residual_alpha=residual_alpha,
                )
                if "mean_normalized_r" in result:
                    logger.info(
                        f"    mean r/NC = {result['mean_normalized_r']:.4f}, "
                        f"mean NC = {result['mean_noise_ceiling_r']:.4f}"
                    )

                cond_dir = output_dir / model_label / subject / label
                cond_dir.mkdir(parents=True, exist_ok=True)
                np.save(cond_dir / "per_voxel_r.npy", result["per_voxel_r"])
                np.save(cond_dir / "noise_ceiling.npy", noise_ceiling_r)
                np.save(cond_dir / "voxel_keep.npy", voxel_keep)
                np.save(
                    cond_dir / "best_frac_per_voxel.npy",
                    result["best_frac_per_voxel"],
                )
                save_voxelwise_nifti(
                    result["per_voxel_r"], voxel_keep, brain_mask,
                    brain_mask_img, cond_dir / "per_voxel_r.nii.gz",
                )
                save_voxelwise_nifti(
                    noise_ceiling_r, voxel_keep, brain_mask,
                    brain_mask_img, cond_dir / "noise_ceiling.nii.gz",
                )

                summary_rows.append(
                    {
                        "subject": subject,
                        "condition": label,
                        "embed_modality": emod,
                        "fmri_modality": fmod,
                        "n_trials": X.shape[0],
                        "n_unique_stimuli": n_unique,
                        "n_train_stimuli": result.get("n_train_stimuli", np.nan),
                        "n_test_stimuli": result.get("n_test_stimuli", np.nan),
                        "n_outer_folds": result.get("n_outer_folds", np.nan),
                        "n_voxels": Y.shape[1],
                        "mean_r": result["mean_r"],
                        "mean_noise_ceiling_r": result.get(
                            "mean_noise_ceiling_r", np.nan
                        ),
                        "max_noise_ceiling_r": result.get(
                            "max_noise_ceiling_r", np.nan
                        ),
                        "mean_normalized_r": result.get("mean_normalized_r", np.nan),
                        "p_value_mean_r": np.nan,
                    }
                )

    # -- aggregate across subjects -------------------------------------------
    logger.info(f"\n{'=' * 60}")
    logger.info("Aggregating results …")

    summary_df = pd.DataFrame(summary_rows)
    results_dir = output_dir / model_label
    results_dir.mkdir(parents=True, exist_ok=True)

    agg_cols = [
        "mean_r", "mean_noise_ceiling_r", "max_noise_ceiling_r", "mean_normalized_r"
    ]
    agg = (
        summary_df.groupby("condition")[agg_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    logger.info(f"\n{agg.to_string()}")

    summary_path = results_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Per-subject summary → {summary_path}")

    agg_path = results_dir / "aggregated.csv"
    agg.to_csv(agg_path)
    logger.info(f"Aggregated results → {agg_path}")

    logger.success("Residual neural encoding analysis complete!")


if __name__ == "__main__":
    main()
