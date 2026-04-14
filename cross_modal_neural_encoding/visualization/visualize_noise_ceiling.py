"""Generate surface plots with vertices colored by noise ceiling percentiles.

Creates visualizations of brain surface voxels with top 5%, 10%, and 15% noise
ceiling. Loads GLMsingle betas, normalizes within runs, computes noise ceiling
signal-to-noise ratio, and generates corresponding surface plots.

Usage::

    python -m cross_modal_neural_encoding.visualization.visualize_noise_ceiling
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import hydra
import nibabel as nib
import numpy as np
from loguru import logger
from nilearn import surface, plotting
from omegaconf import DictConfig

from cross_modal_neural_encoding.config import FIGURES_DIR
from cross_modal_neural_encoding.utils import (
    compute_nc,
    compute_nc_by_modality,
    compute_ncsnr,
    load_design_matrix_mapping,
    normalize_betas_per_run,
    _find_subdir,
    get_affine,
    load_brain_mask,
)


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading and Processing
# ═══════════════════════════════════════════════════════════════════════════


def load_glmsingle_betas(
    glmsingle_dir: Path, subject: str
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Load GLMsingle betas and stimulus IDs. Removed unused run_ids to simplify."""
    subject_dir = glmsingle_dir / subject
    betas_file = subject_dir / "TYPED_FITHRF_GLMDENOISE_RR.npy"

    logger.info(f"Loading betas from {betas_file}")
    betas_obj = np.load(betas_file, allow_pickle=True).item()
    betas_vol = betas_obj["betasmd"]

    # Reshape to (num_conditions, num_voxels)
    x, y, z, num_conditions = betas_vol.shape
    betas = betas_vol.reshape(-1, num_conditions).T
    spatial_dims = (x, y, z)

    logger.info(f"Loaded betas with shape: {betas.shape}")

    # Load stimulus IDs from DESIGNINFO
    designinfo_file = subject_dir / "DESIGNINFO.npy"
    stimulus_ids = np.arange(betas.shape[0], dtype=int)
    if designinfo_file.exists():
        try:
            designinfo = np.load(designinfo_file, allow_pickle=True).item()
            stimulus_ids = np.array(designinfo.get("stimorder", stimulus_ids), dtype=int)
        except Exception:
            pass

    return betas, stimulus_ids, spatial_dims


def load_all_runs(
    glmsingle_dir: Path, subject: str
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Load GLMsingle betas and normalize within runs.

    Parameters
    ----------
    glmsingle_dir : Path
        Path to the glmsingle output directory
    subject : str
        Subject identifier

    Returns
    -------
    tuple
        (all_betas, all_stimulus_ids, spatial_dims) with betas normalized
        per run.
    """
    logger.info(f"Processing {subject}")
    betas, stimulus_ids, spatial_dims = load_glmsingle_betas(glmsingle_dir, subject)

    # Get number of runs from DESIGNINFO
    subject_dir = glmsingle_dir / subject
    designinfo_file = subject_dir / "DESIGNINFO.npy"
    if designinfo_file.exists():
        designinfo = np.load(designinfo_file, allow_pickle=True).item()
        num_runs = len(designinfo.get("design", []))
    else:
        num_runs = 36  # Default based on typical GLMsingle outputs

    # Use shared normalization function
    betas = normalize_betas_per_run(betas, num_runs=num_runs)

    # Flatten stimulus_ids to ensure it's 1D (matching betas first dimension)
    stimulus_ids = np.asarray(stimulus_ids).flatten()

    return betas, stimulus_ids, spatial_dims


# ═══════════════════════════════════════════════════════════════════════════
# Surface Extraction and Plotting
# ═══════════════════════════════════════════════════════════════════════════


def load_native_surfaces(fmriprep_dir: Path, subject: str) -> dict:
    """Load subject-native FreeSurfer surfaces and sulcal depth from fMRIPrep outputs.
    
    Loads both pial (for sampling) and inflated (for display) surfaces."""
    subject_dir = fmriprep_dir / subject
    anat_dirs = _find_subdir(subject_dir, "anat")
    
    if not anat_dirs:
        raise FileNotFoundError(f"No anatomical directory found for {subject}")
    
    anat_dir = anat_dirs[0]
    logger.info(f"Loading native surfaces from {anat_dir}")
    
    surfaces = {}
    for hemisphere, hemi_abbrev in [("left", "L"), ("right", "R")]:
        pial_file = list(anat_dir.glob(f"*hemi-{hemi_abbrev}_pial.surf.gii"))
        inflated_file = list(anat_dir.glob(f"*hemi-{hemi_abbrev}_inflated.surf.gii"))
        sulc_file = list(anat_dir.glob(f"*hemi-{hemi_abbrev}_sulc.shape.gii"))
        
        if not pial_file or not inflated_file or not sulc_file:
            raise FileNotFoundError(f"Missing surface files for {hemisphere} hemisphere")
        
        surfaces[f"{hemisphere}_pial"] = str(pial_file[0])
        surfaces[f"{hemisphere}_inflated"] = str(inflated_file[0])
        surfaces[f"{hemisphere}_sulc"] = str(sulc_file[0])
        logger.info(f"Found {hemisphere} pial, inflated, and sulcal depth")
    
    return surfaces


def project_to_surface_native(
    volume_data: np.ndarray,
    hemisphere: str,
    affine: np.ndarray,
    native_surfaces: dict,
) -> np.ndarray:
    """
    Project volume data to subject's native surface.

    CRITICAL: Use pial surface for sampling (data extraction from voxels),
    NOT inflated. The inflated surface has different vertex coordinates
    and won't align with the voxel volume.

    Parameters
    ----------
    volume_data : np.ndarray
        3D volume of data to project (e.g., NC values)
    hemisphere : str
        Hemisphere identifier ('left' or 'right')
    affine : np.ndarray
        Affine matrix from volume space
    native_surfaces : dict
        Dictionary with pial, inflated, and sulc surface paths

    Returns
    -------
    np.ndarray
        1D array of projected data (one value per vertex)
    """
    hemi = "left" if "left" in hemisphere.lower() else "right"
    # CRITICAL FIX: Sample using PIAL surface (aligns with voxels)
    pial_surface = native_surfaces[f"{hemi}_pial"]
    
    logger.info(f"Projecting {hemi} hemisphere data using pial surface for sampling...")
    
    # Create Nifti image with proper affine
    img = nib.Nifti1Image(volume_data, affine=affine)
    
    # Use pial surface to extract data from voxels
    surface_data = surface.vol_to_surf(
        img,
        pial_surface,
        kind="auto",  # auto-detect mesh type from GIFTI
    )
    
    logger.info(f"Projected to {len(surface_data)} vertices")
    return surface_data



def _plot_modality_row(
    axes_row: np.ndarray, surface_data: dict, hemisphere: str, modality: str, 
    percentiles: list[int], fsaverage_meshes: dict, sulc_data: dict, cmap: str
) -> None:
    """Helper: Plot one modality row across all percentiles into provided 3D axes."""
    hemi = "left" if "left" in hemisphere.lower() else "right"
    for col, percentile in enumerate(percentiles):
        data = surface_data[hemisphere].copy()
        threshold = np.nanpercentile(data, 100 - percentile)
        masked = data.copy()
        masked[~(data >= threshold)] = np.nan
        
        plotting.plot_surf_stat_map(
            surf_mesh=fsaverage_meshes[hemisphere],
            stat_map=masked,
            bg_map=sulc_data[hemisphere],
            vmin=threshold, vmax=np.nanmax(data),
            cmap=cmap,
            hemi=hemi,
            view="lateral",
            colorbar=False,
            axes=axes_row[col],
        )


def plot_surface_modality_overlay(
    nc_vol_text: np.ndarray,
    nc_vol_image: np.ndarray,
    affine: np.ndarray,
    percentiles: list[int],
    subject: str,
    native_surfaces: dict,
    output_path: Optional[Path] = None,
):
    """
    Plot text and image noise ceiling on a single figure with multiple subplots.

    Creates a single figure with 4 rows (text-left, text-right, image-left, image-right)
    and N columns (one per percentile threshold).

    Parameters
    ----------
    nc_vol_text : np.ndarray
        3D noise ceiling volume for text stimuli
    nc_vol_image : np.ndarray
        3D noise ceiling volume for image stimuli
    affine : np.ndarray
        Affine transformation matrix for the volume
    percentiles : list[int]
        List of percentiles to visualize (e.g., [10, 20, 30])
    subject : str
        Subject identifier for title
    native_surfaces : dict
        Dictionary of native surface file paths from load_native_surfaces()
    output_path : Optional[Path]
        Path to save figure
    """
    import matplotlib.pyplot as plt
    import nibabel as nib
    
    logger.info("Using native subject surfaces for projection")
    # Project both modalities to native surfaces
    logger.info("Projecting text noise ceiling to native surface...")
    nc_surface_text_left = project_to_surface_native(nc_vol_text, "left", affine, native_surfaces)
    nc_surface_text_right = project_to_surface_native(nc_vol_text, "right", affine, native_surfaces)
    
    logger.info("Projecting image noise ceiling to native surface...")
    nc_surface_image_left = project_to_surface_native(nc_vol_image, "left", affine, native_surfaces)
    nc_surface_image_right = project_to_surface_native(nc_vol_image, "right", affine, native_surfaces)
    
    # Load native sulcal depth for background
    logger.info("Loading native surface geometry...")
    sulc_data = {}
    for hemi in ["left", "right"]:
        logger.debug(f"Loading sulcal depth for {hemi}...")
        sulc_path = native_surfaces[f"{hemi}_sulc"]
        sulc_gii = nib.load(sulc_path)
        # Access GIFTI data safely - darrays is a list of GiftiDataArray objects
        sulc_arrays = getattr(sulc_gii, 'darrays', [])
        sulc_data[hemi] = np.asarray(sulc_arrays[0].data)
        logger.debug(f"  {hemi} sulc shape: {sulc_data[hemi].shape}")
    
    # Load native surface meshes for plotting (use INFLATED for display)
    fsaverage_meshes = {}
    for hemi in ["left", "right"]:
        logger.debug(f"Loading inflated surface mesh for display ({hemi})...")
        # Use INFLATED surface for visualization (prettier, shows folds)
        surf_path = native_surfaces[f"{hemi}_inflated"]
        surf_gii = nib.load(surf_path)
        surf_arrays = getattr(surf_gii, 'darrays', [])
        vertices = np.asarray(surf_arrays[0].data)
        faces = np.asarray(surf_arrays[1].data)
        fsaverage_meshes[hemi] = (vertices, faces)
        logger.debug(f"  {hemi} inflated mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    logger.info("Surface geometry loaded ✓")
    
    surface_text = {"left": nc_surface_text_left, "right": nc_surface_text_right}
    surface_image = {"left": nc_surface_image_left, "right": nc_surface_image_right}
    
    # Create single figure with 4 rows x len(percentiles) columns
    # Rows: Text-Left, Text-Right, Image-Left, Image-Right
    logger.info(f"Creating figure with shape (4, {len(percentiles)})...")
    fig, axes = plt.subplots(
        4, len(percentiles),
        figsize=(4 * len(percentiles), 16),
        subplot_kw={'projection': '3d'}
    )
    if len(percentiles) == 1:
        axes = axes.reshape(4, 1)
    
    logger.info("Populating subplots...")
    for row_offset, (modality, surface_data, cmap_color) in enumerate([
        ("Text", surface_text, "Blues"),
        ("Image", surface_image, "Reds")
    ]):
        for hemi_idx, hemisphere in enumerate(["left", "right"]):
            row_base = row_offset * 2 + hemi_idx
            _plot_modality_row(axes[row_base], surface_data, hemisphere, modality, percentiles, fsaverage_meshes, sulc_data, cmap_color)
    
    fig.suptitle(f"{subject} - Text vs Image Noise Ceiling by Threshold", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    
    if output_path:
        logger.info(f"Saving figure to {output_path}...")
        output_fig = output_path.parent / f"{output_path.stem}_modality-overlay.png"
        fig.savefig(output_fig, dpi=150, bbox_inches="tight")
        logger.success(f"Saved modality overlay plot to {output_fig}")


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════


@hydra.main(version_base="1.3", config_path="../../configs/visualization", config_name="visualize_noise_ceiling")
def main(cfg: DictConfig) -> None:
    """Generate noise ceiling surface plots for all subjects in subject-native T1w space."""
    logger.info(f"Configuration: {cfg}")

    glmsingle_dir = Path(cfg.glmsingle_dir)
    subject_filter = cfg.get("subject", None)  # Optional filter for specific subject
    fmriprep_dir = Path(cfg.get("fmriprep_dir", ""))
    percentiles = cfg.get("percentiles", [5, 10, 15])
    output_dir_cfg = cfg.get("output_dir")
    output_dir = Path(output_dir_cfg) if output_dir_cfg is not None else FIGURES_DIR / "noise_ceiling"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all subjects in glmsingle_dir
    subject_dirs = sorted([d for d in glmsingle_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    
    if subject_filter:
        # Filter to specific subject if provided
        subject_dirs = [d for d in subject_dirs if d.name == subject_filter]
    
    if not subject_dirs:
        logger.error(f"No subjects found in {glmsingle_dir}")
        return
    
    logger.info(f"Processing {len(subject_dirs)} subjects: {[d.name for d in subject_dirs]}")

    nc_num_averages = float(cfg.get("nc_num_averages", 6))

    for subject_dir in subject_dirs:
        subject = subject_dir.name
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {subject}")
        logger.info(f"{'='*60}")

        logger.info(f"Loading betas for {subject}...")
        betas, stimulus_ids, spatial_dims = load_all_runs(glmsingle_dir, subject)

        # Load brain mask from fMRIPrep (for affine + masking)
        fmriprep_dir = Path(cfg.get("fmriprep_dir", ""))
        
        # Load affine: prefer fMRIPrep for native space, fall back to GLMsingle
        affine = get_affine(fmriprep_dir, subject)
        logger.info(f"Affine:\n{affine}")

        logger.info(f"Betas shape: {betas.shape}")
        logger.info(f"Unique stimuli: {len(np.unique(stimulus_ids))}")

        # Load brain mask from fMRIPrep (space-aware - native T1w only)
        brain_mask = load_brain_mask(fmriprep_dir, subject)
        logger.info(f"Loaded brain mask with {int(np.asarray(brain_mask).sum())} voxels")  # type: ignore

        # Compute noise ceiling
        logger.info("Computing noise ceiling SNR...")
        ncsnr = compute_ncsnr(betas, stimulus_ids)
        logger.info(f"NCSNR range: [{np.nanmin(ncsnr):.3f}, {np.nanmax(ncsnr):.3f}]")

        # Convert to noise ceiling percentage
        logger.info("Converting to noise ceiling percentage...")
        nc = compute_nc(ncsnr, num_averages=nc_num_averages)
        logger.info(f"Noise ceiling range: [{np.nanmin(nc):.1f}, {np.nanmax(nc):.1f}]%")

        # Apply brain mask: set out-of-brain voxels to NaN
        nc_1d = nc.copy()
        brain_mask_1d = brain_mask.reshape(-1)
        nc_1d[~brain_mask_1d] = np.nan
        logger.info(f"Applied brain mask. In-brain noise ceiling range: [{np.nanmin(nc_1d):.1f}, {np.nanmax(nc_1d):.1f}]%")

        # Create visualization - Native space only
        logger.info("Creating surface plots in subject-native T1w space...")

        # Reshape noise ceiling back to 3D volume for surface projection
        x, y, z = spatial_dims

        # Load native surfaces for this subject
        native_surfaces = load_native_surfaces(fmriprep_dir, subject)
        logger.info(f"Loaded native surfaces for {subject}")

        # ===== Per-Modality Analysis =====
        # Load design matrix mapping to separate text and image stimuli
        design_mapping_file = Path(cfg.get("design_matrix_mapping_file", ""))
        logger.info(f"\n{'='*60}")
        logger.info("Computing per-modality noise ceiling...")
        logger.info(f"{'='*60}")

        # Load modality mapping
        modality_map = load_design_matrix_mapping(design_mapping_file)
        
        # Compute NC for each modality
        nc_by_modality = compute_nc_by_modality(
            betas,
            stimulus_ids,
            modality_map,
            num_averages=nc_num_averages,
        )
        
        # Apply brain mask to each modality
        for modality in ['text', 'image']:
            nc_modality = nc_by_modality[modality].copy()
            nc_modality[~brain_mask_1d] = np.nan
            
            logger.info(
                f"{modality.capitalize()} NC range: "
                f"[{np.nanmin(nc_modality):.1f}, {np.nanmax(nc_modality):.1f}]%"
            )
            
            nc_by_modality[modality] = nc_modality
        
        # Reshape to 3D volumes
        nc_vol_text = nc_by_modality['text'].reshape(x, y, z)
        nc_vol_image = nc_by_modality['image'].reshape(x, y, z)
        
        # Create overlay plot (native surfaces only)
        logger.info("Creating modality overlay surface plots...")
        output_fig_overlay = output_dir / f"{subject}_noise_ceiling_modality_overlay.png"
        
        logger.info("Projecting modality data to native surfaces...")
        plot_surface_modality_overlay(
            nc_vol_text=nc_vol_text,
            nc_vol_image=nc_vol_image,
            affine=affine,
            percentiles=percentiles,
            subject=subject,
            output_path=output_fig_overlay,
            native_surfaces=native_surfaces,
        )
        
        logger.success(
            f"Modality overlay visualization complete for {subject}. "
            f"Saved to {output_fig_overlay.parent}"
        )


if __name__ == "__main__":
    main()
