"""Visualize subject masks on inflated surfaces based on GLMSingle TYPEA R2."""

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from loguru import logger
from nilearn import datasets, image, surface
from nilearn.plotting import plot_surf_stat_map
from omegaconf import DictConfig

from cross_modal_neural_encoding.config import FIGURES_DIR


def load_typea_r2(glmsingle_path: Path) -> np.ndarray:
    """Load R2 values from GLMSingle TYPEA_ONOFF.npy.

    Parameters
    ----------
    glmsingle_path : Path
        Path to the GLMSingle output directory.

    Returns
    -------
    np.ndarray
        R2 values from the TYPEA model.
    """
    typea_file = glmsingle_path / "TYPEA_ONOFF.npy"
    if not typea_file.exists():
        raise FileNotFoundError(
            f"TYPEA_ONOFF.npy not found at {typea_file}. "
            "Please check the GLMSingle output path."
        )

    logger.info(f"Loading TYPEA model from {typea_file}")
    typea_data = np.load(typea_file, allow_pickle=True).item()
    r2_values: np.ndarray = typea_data["onoffR2"]

    logger.info(f"R2 shape: {r2_values.shape}")
    logger.info(f"R2 range: [{np.nanmin(r2_values):.4f}, {np.nanmax(r2_values):.4f}]")
    return r2_values


def load_r2_nifti(glmsingle_path: Path, r2_values: np.ndarray) -> nib.Nifti1Image:
    """Return a NIfTI image of the R2 map.

    Loads ``R2_map.nii.gz`` if it exists; otherwise builds one from the
    array (assuming 2-mm MNI affine).
    """
    r2_nifti = glmsingle_path / "R2_map.nii.gz"
    if r2_nifti.exists():
        logger.info(f"Loading R2 NIfTI from {r2_nifti}")
        return image.load_img(r2_nifti)

    logger.warning(
        "R2_map.nii.gz not found – constructing NIfTI from array with 2-mm MNI affine."
    )
    affine = np.diag([-2, 2, 2, 1])  # 2-mm MNI
    return nib.Nifti1Image(r2_values.astype(np.float32), affine)


def create_mask_from_r2(r2_values: np.ndarray, top_pct: float) -> tuple[np.ndarray, float]:
    """Create a binary mask keeping the top *top_pct* % of voxels by R2.

    Parameters
    ----------
    r2_values : np.ndarray
        R2 map (any shape).
    top_pct : float
        Percentage of voxels to keep (e.g. 10 = top 10 %).

    Returns
    -------
    mask : np.ndarray
        Binary mask (1 for selected voxels).
    cutoff : float
        The R2 value at the percentile boundary.
    """
    cutoff = float(np.nanpercentile(r2_values, 100 - top_pct))
    mask = (r2_values >= cutoff).astype(np.float32)
    n_voxels = int(np.sum(mask))
    logger.info(
        f"Top {top_pct}%: {n_voxels} voxels (R2 >= {cutoff:.4f})"
    )
    return mask, cutoff


def visualize_surface(
    r2_img: nib.Nifti1Image,
    top_percentages: list[float],
    subject_id: str,
    output_path: Path,
) -> None:
    """Create inflated-surface plots of masked R2 for each top-% selection.

    For every percentage the figure shows four views (lateral / medial × L / R)
    on the *fsaverage* inflated surface.

    Parameters
    ----------
    r2_img : Nifti1Image
        Volumetric R2 map.
    top_percentages : list[float]
        Top-k percentages to visualise (e.g. [5, 10, 15]).
    subject_id : str
        Label used in titles.
    output_path : Path
        Where to save the combined figure.
    """
    fsaverage = datasets.fetch_surf_fsaverage()
    r2_data = np.asarray(r2_img.dataobj, dtype=np.float32)
    vmax = float(np.nanpercentile(r2_data[r2_data > 0], 95))

    views = ["lateral", "medial"]
    hemis = ["left", "right"]
    n_rows = len(top_percentages)

    # Layout: rows = percentages, cols = 4 (L-lateral, L-medial, R-lateral, R-medial)
    fig, axes = plt.subplots(
        n_rows,
        len(views) * len(hemis),
        figsize=(6 * len(views) * len(hemis), 5 * n_rows),
        subplot_kw={"projection": "3d"},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # keep 2-D

    # Project the full (unmasked) R2 volume to each surface once
    surf_textures: dict[str, np.ndarray] = {}
    for hemi in hemis:
        pial_mesh = fsaverage[f"pial_{hemi}"]
        surf_textures[hemi] = surface.vol_to_surf(r2_img, pial_mesh)

    for row, pct in enumerate(top_percentages):
        _, cutoff = create_mask_from_r2(r2_data, pct)

        col = 0
        for hemi in hemis:
            surf_mesh = fsaverage[f"infl_{hemi}"]
            # Threshold on the surface: zero out vertices below the cutoff
            texture = surf_textures[hemi].copy()
            texture[texture < cutoff] = 0.0

            for view in views:
                plot_surf_stat_map(
                    surf_mesh,
                    stat_map=texture,
                    hemi=hemi,
                    view=view,
                    title=(
                        f"{subject_id}  top {pct}% (R²≥{cutoff:.3f})  "
                        f"{hemi[0].upper()}-{view}"
                    ),
                    colorbar=True,
                    cmap="hot",
                    vmax=vmax,
                    threshold=0.01,  # hide near-zero
                    bg_map=fsaverage[f"sulc_{hemi}"],
                    axes=axes[row, col],
                )
                col += 1

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.success(f"Saved surface visualization to {output_path}")
    plt.close(fig)


def process_subject(
    subject_dir: Path,
    top_percentages: list[float],
    output_dir: Path,
) -> None:
    """Load data and produce surface visualisation for a single subject."""
    subject_id = subject_dir.name
    pct_label = "_".join(str(p) for p in top_percentages)
    output_path = output_dir / f"{subject_id}_typea_mask_top_{pct_label}pct.png"

    logger.info(f"Processing {subject_id}")

    r2_values = load_typea_r2(subject_dir)
    r2_img = load_r2_nifti(subject_dir, r2_values)

    visualize_surface(
        r2_img=r2_img,
        top_percentages=top_percentages,
        subject_id=subject_id,
        output_path=output_path,
    )
    logger.success(f"Finished {subject_id} → {output_path}")


@hydra.main(
    version_base=None,
    config_path="../configs/visualization",
    config_name="visualize_glmsingle_mask",
)
def main(cfg: DictConfig) -> None:
    """Visualize GLMSingle masks on the inflated surface.

    Hydra config fields
    -------------------
    glmsingle_path : str   (required) – parent dir containing sub-* folders
    top_percentages : list[float]  (default [5, 10, 15]) – top-k % of voxels
    output_dir : str | null
    """
    glmsingle_path = Path(cfg.glmsingle_path)
    if not glmsingle_path.exists():
        raise FileNotFoundError(f"GLMSingle path not found: {glmsingle_path}")

    top_percentages: list[float] = list(cfg.top_percentages)
    output_dir = Path(cfg.output_dir) if cfg.get("output_dir") else FIGURES_DIR

    # Discover subject directories (sub-*) that contain TYPEA_ONOFF.npy
    subject_dirs = sorted(
        d for d in glmsingle_path.iterdir()
        if d.is_dir() and d.name.startswith("sub-") and (d / "TYPEA_ONOFF.npy").exists()
    )

    if not subject_dirs:
        raise FileNotFoundError(
            f"No sub-*/TYPEA_ONOFF.npy found under {glmsingle_path}. "
            "Make sure the path points to the parent GLMSingle directory."
        )

    logger.info(f"Found {len(subject_dirs)} subject(s): {[d.name for d in subject_dirs]}")
    logger.info(f"Top percentages: {top_percentages}")
    logger.info(f"Output directory: {output_dir}")

    for subject_dir in subject_dirs:
        process_subject(subject_dir, top_percentages, output_dir)

    logger.success(f"All {len(subject_dirs)} subject(s) done!")


if __name__ == "__main__":
    main()
