"""Visualize voxel-selection masks on inflated brain surfaces.

Supports two modes controlled by ``mode`` in the Hydra config:

* **r2** – threshold the GLMsingle Type-A on/off R² map at the top-k %
    of surface vertices (e.g. top 5 %, 10 %, 15 %).
* **ev** – compute per-voxel explainable variance (EV) from repeated
    stimulus presentations (all modalities combined), following the VEM
    framework (Dupré la Tour et al., 2025; Sahani & Linden, 2002), and
    threshold at the top-k % of surface vertices (e.g. top 5 %, 10 %, 15 %).

Both modes produce figures in the same style: one row per threshold,
four columns (L-lateral, L-medial, R-lateral, R-medial), "hot" cmap.
Thresholds are always top-k percentages of positive surface vertices.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from loguru import logger
from nilearn import datasets, image, surface
from nilearn.plotting import plot_surf_stat_map
from omegaconf import DictConfig
from scipy.stats import zscore as _scipy_zscore

from cross_modal_neural_encoding.config import FIGURES_DIR


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════


def load_typea_r2(
    glmsingle_path: Path, subject: str
) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load Type-A R² and return both the 3-D array and a NIfTI image.

    If ``R2_map.nii.gz`` exists it is used directly (correct affine);
    otherwise a NIfTI is constructed with a 2-mm MNI fallback affine.
    """
    sub_dir = glmsingle_path / subject
    typea_file = sub_dir / "TYPEA_ONOFF.npy"
    if not typea_file.exists():
        raise FileNotFoundError(f"TYPEA_ONOFF.npy not found: {typea_file}")

    r2_3d: np.ndarray = np.load(typea_file, allow_pickle=True).item()["onoffR2"]
    logger.info(
        f"  R² shape: {r2_3d.shape}, "
        f"range [{np.nanmin(r2_3d):.4f}, {np.nanmax(r2_3d):.4f}]"
    )

    r2_nifti_path = sub_dir / "R2_map.nii.gz"
    if r2_nifti_path.exists():
        r2_img = image.load_img(r2_nifti_path)
    else:
        logger.warning("R2_map.nii.gz not found – using 2-mm MNI fallback affine.")
        r2_img = nib.Nifti1Image(
            r2_3d.astype(np.float32), np.diag([-2.0, 2.0, 2.0, 1.0])
        )

    return r2_3d, r2_img


def _get_affine(glmsingle_path: Path, subject: str) -> np.ndarray:
    """Return the NIfTI affine for *subject* (from R2_map.nii.gz)."""
    r2_nifti = glmsingle_path / subject / "R2_map.nii.gz"
    if r2_nifti.exists():
        return nib.load(r2_nifti).affine  # type: ignore[attr-defined]
    logger.warning("R2_map.nii.gz not found – using 2-mm MNI fallback affine.")
    return np.diag([-2.0, 2.0, 2.0, 1.0])


def load_betas(glmsingle_path: Path, subject: str) -> np.ndarray:
    """Load Type-D denoised betas → (X, Y, Z, n_trials)."""
    typed_file = glmsingle_path / subject / "TYPED_FITHRF_GLMDENOISE_RR.npy"
    if not typed_file.exists():
        raise FileNotFoundError(f"TYPED file not found: {typed_file}")
    betas_4d: np.ndarray = np.load(typed_file, allow_pickle=True).item()["betasmd"]
    logger.info(f"  Betas shape: {betas_4d.shape}")
    return betas_4d


def load_events(
    bids_root: Path,
    subject: str,
    *,
    sessions: list[int],
    runs_per_session: int,
    task: str,
    modality_column: str = "modality",
    cocoid_column: str = "cocoid",
) -> pd.DataFrame:
    """Parse BIDS events files → DataFrame(beta_index, cocoid, modality)."""
    records: list[dict] = []
    beta_idx = 0

    for ses in sessions:
        for run in range(1, runs_per_session + 1):
            fname = f"{subject}_ses-{ses:02d}_task-{task}_run-{run:02d}_events.tsv"
            p = bids_root / subject / f"ses-{ses:02d}" / "func" / fname
            if not p.exists():
                logger.warning(f"Missing events file: {p}")
                continue

            df_run = pd.read_csv(p, sep="\t").sort_values("onset")
            for _, row in df_run.iterrows():
                mod = str(row[modality_column]).strip().lower()
                cid = row[cocoid_column]
                if mod in ("blank", "nan", "n/a", ""):
                    continue
                if pd.isna(cid) or str(cid).strip().lower() == "n/a":
                    continue
                records.append(
                    {"beta_index": beta_idx, "cocoid": int(float(cid)), "modality": mod}
                )
                beta_idx += 1

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# Explainable variance (all modalities combined)
# ═══════════════════════════════════════════════════════════════════════════


def _compute_ev_single_modality(
    betas_flat: np.ndarray,
    events_df: pd.DataFrame,
    modality_filter: str,
) -> np.ndarray:
    """Compute per-voxel EV for one stimulus modality.

    Parameters
    ----------
    betas_flat : (n_voxels, n_all_trials)
    events_df : table with ``beta_index``, ``cocoid``, ``modality``.
    modality_filter : ``"image"`` or ``"text"``.

    Returns
    -------
    ev : (n_voxels,) – explainable variance (not clipped).
    """
    n_voxels = betas_flat.shape[0]

    mod_df = events_df[events_df["modality"] == modality_filter]
    grouped = mod_df.groupby("cocoid")
    group_sizes = grouped.size()
    n_repeats = int(group_sizes.mode().iloc[0])

    valid_cids = group_sizes[group_sizes == n_repeats].index
    valid_df = mod_df[mod_df["cocoid"].isin(valid_cids)]
    valid_grouped = valid_df.groupby("cocoid")
    n_stimuli = len(valid_grouped)

    logger.info(
        f"  EV ({modality_filter}): {n_stimuli} stimuli × "
        f"{n_repeats} repeats, {n_voxels} voxels"
    )

    data = np.zeros((n_repeats, n_stimuli, n_voxels), dtype=np.float64)
    for i, (_, group) in enumerate(valid_grouped):
        indices = group["beta_index"].values.astype(int)
        data[:, i, :] = betas_flat[:, indices].T  # type: ignore[index]

    # Z-score across stimuli within each repeat
    data = np.asarray(_scipy_zscore(data, axis=1, nan_policy="omit"))
    data = np.nan_to_num(data, nan=0.0)

    mean_var = data.var(axis=1, dtype=np.float64, ddof=1).mean(axis=0)
    var_mean = data.mean(axis=0).var(axis=0, dtype=np.float64, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ev = np.where(mean_var > 0, var_mean / mean_var, 0.0)

    # Bias correction (Sahani & Linden, 2002)
    ev = ev - (1 - ev) / (n_repeats - 1)
    return ev


def compute_ev_3d(
    betas_4d: np.ndarray,
    events_df: pd.DataFrame,
) -> np.ndarray:
    """Compute per-voxel EV separately for each modality, then take the max.

    Image and text trials evoke different BOLD responses, so EV must
    be computed within-modality (6 image repeats and 6 text repeats
    separately).  The two maps are then combined by taking the
    element-wise maximum, giving each voxel the benefit of the doubt
    (i.e. if it is reliable for *either* modality, it passes).

    Returns
    -------
    ev_3d : (X, Y, Z) – explainable variance (clipped ≥ 0).
    """
    vol_shape = betas_4d.shape[:3]
    betas_flat = betas_4d.reshape(-1, betas_4d.shape[-1])

    modalities = sorted(events_df["modality"].unique())
    logger.info(f"  Computing EV per modality: {modalities}")

    ev_maps = []
    for mod in modalities:
        ev = _compute_ev_single_modality(betas_flat, events_df, mod)
        n_pos = int((ev > 0).sum())
        median_ev = float(np.median(ev[ev > 0])) if n_pos > 0 else 0.0
        logger.info(f"  EV ({mod}): {n_pos} voxels > 0, median = {median_ev:.4f}")
        ev_maps.append(ev)

    # Combine: element-wise max across modalities
    ev_combined = np.maximum.reduce(ev_maps)
    ev_combined = np.clip(ev_combined, 0.0, None)

    n_pos = int((ev_combined > 0).sum())
    median_ev = float(np.median(ev_combined[ev_combined > 0])) if n_pos > 0 else 0.0
    logger.info(f"  EV (max of {modalities}): {n_pos} voxels > 0, median = {median_ev:.4f}")

    return ev_combined.reshape(vol_shape)


# ═══════════════════════════════════════════════════════════════════════════
# Unified surface visualisation
# ═══════════════════════════════════════════════════════════════════════════


def visualize_surface(
    stat_img: nib.Nifti1Image,
    stat_3d: np.ndarray,
    thresholds: list[float],
    subject_id: str,
    output_path: Path,
    *,
    mode: str,
) -> None:
    """Create inflated-surface plots with one row per threshold.

    Parameters
    ----------
    stat_img : NIfTI image of the statistic map (R² or EV).
    stat_3d : raw 3-D array of the same map.
    thresholds : top-k percentages (e.g. [5, 10, 15]).
        Both modes use the same interpretation: keep the top *thr* %
        of positive surface vertices.
    subject_id : label for titles.
    output_path : where to save the figure.
    mode : ``"r2"`` or ``"ev"``.
    """
    fsaverage = datasets.fetch_surf_fsaverage()
    views = ["lateral", "medial"]
    hemis = ["left", "right"]
    n_rows = len(thresholds)

    # Project the full volume to each surface once
    surf_textures: dict[str, np.ndarray] = {}
    for hemi in hemis:
        pial_mesh = fsaverage[f"pial_{hemi}"]
        surf_textures[hemi] = surface.vol_to_surf(stat_img, pial_mesh)

    # Compute vmax from actual surface values (not volume) to avoid
    # nilearn colorbar crash when surface max < display threshold
    all_surf_vals = np.concatenate(list(surf_textures.values()))
    pos_surf_vals = all_surf_vals[all_surf_vals > 0]
    vmax = float(np.nanpercentile(pos_surf_vals, 95)) if len(pos_surf_vals) > 0 else 1.0
    vmax = max(vmax, 1e-4)  # must be strictly > threshold=1e-6

    fig, axes = plt.subplots(
        n_rows,
        len(views) * len(hemis),
        figsize=(6 * len(views) * len(hemis), 5 * n_rows),
        subplot_kw={"projection": "3d"},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Pre-compute positive surface values once (both hemispheres)
    all_surf = np.concatenate([surf_textures[h] for h in hemis])
    brain_surf = all_surf[all_surf > 0]

    stat_label = "R²" if mode == "r2" else "EV"

    for row, thr in enumerate(thresholds):
        # Both modes: keep the top thr% of positive surface vertices
        cutoff = float(np.nanpercentile(brain_surf, 100 - thr))
        n_vertices = int(np.sum(brain_surf >= cutoff))
        row_label = f"top {thr}% ({stat_label}≥{cutoff:.3f})"
        logger.info(
            f"  Top {thr}%: {n_vertices} surface vertices "
            f"({stat_label} ≥ {cutoff:.4f})"
        )

        col = 0
        for hemi in hemis:
            surf_mesh = fsaverage[f"infl_{hemi}"]
            texture = surf_textures[hemi].copy()
            texture[texture < cutoff] = 0.0

            for view in views:
                plot_surf_stat_map(
                    surf_mesh,
                    stat_map=texture,
                    hemi=hemi,
                    view=view,
                    title=f"{subject_id}  {row_label}  {hemi[0].upper()}-{view}",
                    colorbar=True,
                    cmap="hot",
                    vmax=vmax,
                    threshold=1e-6,  # just hide exact zeros
                    bg_map=fsaverage[f"sulc_{hemi}"],
                    axes=axes[row, col],
                )
                col += 1

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.success(f"Saved surface visualization → {output_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Subject-level drivers
# ═══════════════════════════════════════════════════════════════════════════


def process_subject_r2(
    glmsingle_path: Path,
    subject: str,
    thresholds: list[float],
    output_dir: Path,
) -> None:
    """R² mode: load Type-A R² and produce threshold surface plots."""
    logger.info(f"Processing {subject} (R² mode)")
    r2_3d, r2_img = load_typea_r2(glmsingle_path, subject)
    thr_label = "_".join(str(t) for t in thresholds)
    out = output_dir / f"{subject}_r2_mask_top_{thr_label}pct.png"
    visualize_surface(r2_img, r2_3d, thresholds, subject, out, mode="r2")


def process_subject_ev(
    glmsingle_path: Path,
    bids_root: Path,
    subject: str,
    thresholds: list[float],
    output_dir: Path,
    *,
    sessions: list[int],
    runs_per_session: int,
    task: str,
) -> None:
    """EV mode: compute combined EV and produce threshold surface plots."""
    logger.info(f"Processing {subject} (EV mode)")

    betas_4d = load_betas(glmsingle_path, subject)
    affine = _get_affine(glmsingle_path, subject)
    events_df = load_events(
        bids_root,
        subject,
        sessions=sessions,
        runs_per_session=runs_per_session,
        task=task,
    )

    ev_3d = compute_ev_3d(betas_4d, events_df)
    ev_img = nib.Nifti1Image(ev_3d.astype(np.float32), affine)
    thr_label = "_".join(str(t) for t in thresholds)
    out = output_dir / f"{subject}_ev_mask_{thr_label}.png"
    visualize_surface(ev_img, ev_3d, thresholds, subject, out, mode="ev")


# ═══════════════════════════════════════════════════════════════════════════
# Hydra entry-point
# ═══════════════════════════════════════════════════════════════════════════


@hydra.main(
    version_base=None,
    config_path="../../configs/visualization",
    config_name="visualize_glmsingle_mask",
)
def main(cfg: DictConfig) -> None:
    """Visualize R² or EV masks on inflated brain surfaces.

    Hydra config fields
    -------------------
    mode : str             – ``"r2"`` or ``"ev"``
    glmsingle_path : str   (required) – parent dir with sub-* GLMsingle outputs
    thresholds : list      – top-k percentages (e.g. [5, 10, 15]);
                             same interpretation for both modes.
    bids_root : str        – BIDS root (required for ev mode)
    sessions, runs_per_session, task – experiment parameters (ev mode)
    output_dir : str|null  – output directory (default: reports/figures/)
    """
    mode: str = cfg.get("mode", "r2")
    glmsingle_path = Path(cfg.glmsingle_path)
    if not glmsingle_path.exists():
        raise FileNotFoundError(f"GLMSingle path not found: {glmsingle_path}")

    thresholds: list[float] = list(cfg.thresholds)
    output_dir = Path(cfg.output_dir) if cfg.get("output_dir") else FIGURES_DIR

    # Discover subjects
    subject_dirs = sorted(
        d
        for d in glmsingle_path.iterdir()
        if d.is_dir()
        and d.name.startswith("sub-")
        and (d / "TYPEA_ONOFF.npy").exists()
    )
    if not subject_dirs:
        raise FileNotFoundError(
            f"No sub-*/TYPEA_ONOFF.npy under {glmsingle_path}."
        )

    logger.info(f"Mode: {mode}")
    logger.info(f"Found {len(subject_dirs)} subject(s): {[d.name for d in subject_dirs]}")
    logger.info(f"Thresholds: {thresholds}")
    logger.info(f"Output directory: {output_dir}")

    for subject_dir in subject_dirs:
        subject = subject_dir.name
        if mode == "r2":
            process_subject_r2(glmsingle_path, subject, thresholds, output_dir)
        elif mode == "ev":
            bids_root = Path(cfg.bids_root)
            if not bids_root.exists():
                raise FileNotFoundError(f"BIDS root not found: {bids_root}")
            process_subject_ev(
                glmsingle_path,
                bids_root,
                subject,
                thresholds,
                output_dir,
                sessions=list(cfg.sessions),
                runs_per_session=cfg.runs_per_session,
                task=cfg.task,
            )
        else:
            raise ValueError(f"Unknown mode: {mode!r} (expected 'r2' or 'ev').")

    logger.success(f"All {len(subject_dirs)} subject(s) done!")


if __name__ == "__main__":
    main()
