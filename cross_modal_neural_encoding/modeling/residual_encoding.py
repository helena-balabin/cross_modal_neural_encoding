"""Residual neural encoding: cross-modal ablation via the other modality.

For each encoding condition in the 2×2 cross-modal design, a ridge regression
is fit on the training split to predict the *target of residualization* from
the *other* modality's embedding **r** (the linear text→vision / vision→text
mapping), and the residual replaces that target in the full encoding pipeline.
``residual_side`` selects which side is residualized:

    - ``"embedding"`` (default): **ẽ** = **e** − W***r** removes the part of the
      embedding predictable from the other modality's embedding.
          * text embeddings   → remove the part predictable from vision embeddings
          * vision embeddings → remove the part predictable from text embeddings
    - ``"fmri"``: **Ỹ** = **Y** − V***r** removes the part of the fMRI response
      predictable from the other modality's embedding, e.g. *image fMRI −
      image fMRI predicted from text embeddings*.

In both cases what remains is modality-*private* information.  A selective drop
in cross-modal but not within-modality encoding accuracy after residualization
constitutes evidence that cross-modally shared representational content is
the principal carrier of cross-modal brain alignment.

An optional permuted control is also supported: the residual features **r**
are randomly permuted across stimuli before fitting W*, providing an
empirical baseline for the magnitude of accuracy change attributable to the
residualization procedure itself.

Usage
-----
    python -m cross_modal_neural_encoding.modeling.residual_encoding

Hydra config: ``configs/modeling/residual_encoding.yaml``
"""

from __future__ import annotations

from pathlib import Path

import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

from cross_modal_neural_encoding.config import PROJ_ROOT
from cross_modal_neural_encoding.modeling.neural_encoding import (
    align_single_trials,
    build_events_from_stimorder,
    load_condition_to_cocoid_modality,
    load_designinfo_stimulus_ids_and_num_runs,
    load_embeddings,
    load_fmri,
    run_encoding,
)
from cross_modal_neural_encoding.utils import (
    build_fmri_cache,
    compute_nc_by_modality,
    load_brain_mask,
    load_brain_mask_img,
    load_design_matrix_mapping,
    normalize_betas_per_run,
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
    residual_side: str = str(cfg.get("residual_side", "embedding"))
    if residual_side not in {"embedding", "fmri"}:
        raise ValueError(
            f"residual_side must be 'embedding' or 'fmri', got {residual_side!r}"
        )
    run_permuted_control: bool = bool(cfg.get("run_permuted_control", True))
    conditions: dict = OmegaConf.to_container(cfg.conditions, resolve=True)  # type: ignore[assignment]

    # -- load & PCA embeddings (shared across subjects) ----------------------
    # Both modalities are always loaded: each condition residualizes its
    # embedding against the *other* modality's embedding.
    logger.info("Loading VLM embeddings …")
    embed_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for emod in ("text", "vision"):
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
            f"design_matrix_mapping_file is required. Missing: {design_matrix_mapping_file}"
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
            m: np.sqrt(np.clip(v, 0, None) / 100.0) for m, v in nc_by_modality_pct.items()
        }

        events_df = build_events_from_stimorder(stimulus_ids, condition_to_coco, subject)

        # Build per-modality fMRI cache (shared helper)
        fmri_cache = build_fmri_cache(
            events_df,
            betas=betas,
            brain_mask=brain_mask,
            nc_corr_by_modality_full=nc_corr_by_modality_full,
            conditions=conditions,
            nc_top_percent=nc_top_percent,
        )

        # Run each condition with residualized embeddings
        for cond_name, cond_cfg in tqdm(conditions.items(), desc="    Conditions", leave=False):
            emod = cond_cfg["embed_modality"]
            fmod = cond_cfg["fmri_modality"]

            other_mod = "vision" if emod == "text" else "text"
            embed_ids, embed_feats = embed_data[emod]
            other_ids, other_feats = embed_data[other_mod]
            trial_cids, trial_betas, noise_ceiling_r, voxel_keep = fmri_cache[fmod]

            X, Y, groups = align_single_trials(embed_ids, embed_feats, trial_cids, trial_betas)

            # Build the residual-feature matrix r for each trial: the *other*
            # modality's PCA embedding for the same stimulus (COCO ID).  W* is
            # fit per training split inside run_encoding to remove the linearly
            # predictable (cross-modally shared) component from the embedding.
            other_lookup = {int(cid): i for i, cid in enumerate(other_ids)}
            valid_r = np.array([int(cid) in other_lookup for cid in groups])
            if not valid_r.all():
                logger.warning(
                    f"    {valid_r.sum()}/{len(valid_r)} trials have a paired "
                    f"{other_mod} embedding — dropping the rest."
                )
            X, Y, groups = X[valid_r], Y[valid_r], groups[valid_r]
            r = other_feats[[other_lookup[int(cid)] for cid in groups]]

            n_unique = len(np.unique(groups))
            if n_unique < max(n_outer_folds, n_inner_folds + 1):
                logger.warning(f"    Too few unique stimuli ({n_unique}) — skipping {cond_name}.")
                continue

            # Variants: real r, and optionally permuted r (control)
            variants: list[tuple[str, np.ndarray]] = [(cond_name, r)]
            if run_permuted_control:
                r_perm = r[np.random.default_rng(42).permutation(len(r))]
                variants.append((f"permuted_{cond_name}", r_perm))

            for label, r_input in variants:
                logger.info(
                    f"  {'[permuted] ' if 'permuted' in label else ''}"
                    f"{emod} embed → {fmod} fMRI "
                    f"({residual_side}-side residualized)"
                )
                result = run_encoding(
                    X,
                    Y,
                    frac_grid=frac_grid,
                    groups=groups,
                    test_size=test_size,
                    n_inner_folds=n_inner_folds,
                    n_outer_folds=n_outer_folds,
                    noise_ceiling=noise_ceiling_r,
                    average_test_by_group=True,
                    residual_features=r_input,
                    residual_alpha=residual_alpha,
                    residual_side=residual_side,
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
                    result["per_voxel_r"],
                    voxel_keep,
                    brain_mask,
                    brain_mask_img,
                    cond_dir / "per_voxel_r.nii.gz",
                )
                save_voxelwise_nifti(
                    noise_ceiling_r,
                    voxel_keep,
                    brain_mask,
                    brain_mask_img,
                    cond_dir / "noise_ceiling.nii.gz",
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
                        "mean_noise_ceiling_r": result.get("mean_noise_ceiling_r", np.nan),
                        "max_noise_ceiling_r": result.get("max_noise_ceiling_r", np.nan),
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

    agg_cols = ["mean_r", "mean_noise_ceiling_r", "max_noise_ceiling_r", "mean_normalized_r"]
    agg = summary_df.groupby("condition")[agg_cols].agg(["mean", "std"]).round(4)
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
