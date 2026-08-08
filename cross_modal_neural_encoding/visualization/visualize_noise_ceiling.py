"""Render the voxels the encoding models are fit on, per modality, on the cortex.

Loads GLMsingle betas, normalizes within runs, computes the modality-specific noise
ceiling and keeps the top ``percent`` of it via :func:`select_top_nc_voxels` — the
same rule ``build_fmri_cache`` applies for ``voxel_keep``, so the figure marks
exactly the voxels that enter the encoding models. Keep ``percent`` equal to
``nc_top_percent`` in ``configs/modeling/neural_encoding.yaml``.

Note that "top 20%" is 20% of the voxels with a *positive* noise ceiling, which are
only ~57% of the brain mask; the retained set is ~11.5% of in-brain voxels and
~13% of grey matter, so the rendered cortex is well under 20%.

Two target spaces are supported via ``space``:

``mni``
    Warp the noise ceiling volumes to ``MNI152NLin2009cAsym`` with the fMRIPrep
    ``from-T1w_to-MNI...`` transform (``antsApplyTransforms``) and render them on
    fsaverage, so all subjects are directly comparable.  Requires ANTs::

        module load gcc/12.3 ants/2.6.5

``native``
    Render on the subject's own FreeSurfer surfaces (not comparable across
    subjects).

Usage::

    python -m cross_modal_neural_encoding.visualization.visualize_noise_ceiling
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import cast

import hydra
from loguru import logger
from matplotlib.colors import ListedColormap
import nibabel as nib
from nilearn import datasets, plotting, surface
import numpy as np
from omegaconf import DictConfig

from cross_modal_neural_encoding.config import FIGURES_DIR
from cross_modal_neural_encoding.utils import (
    _find_subdir,
    compute_nc,
    compute_nc_by_modality,
    compute_ncsnr,
    get_affine,
    load_brain_mask,
    load_design_matrix_mapping,
    normalize_betas_per_run,
    select_top_nc_voxels,
)

MNI_TEMPLATE_SPACE = "MNI152NLin2009cAsym"

# One flat colour per modality, from the project palette (see docs/09_visualization.md).
# Flat rather than a value ramp on purpose: the figure's question is how much cortex
# each selection level covers, and a ramp answers a different one. Shading the low
# noise ceilings pale hides exactly the vertices a wider level adds, so the levels
# read as identical even when they differ threefold.
MODALITY_COLORS = {"text": "#7EAEDB", "image": "#E88989"}

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
        except (OSError, ValueError):
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


def require_ants() -> str:
    """Return the ``antsApplyTransforms`` executable or explain how to get it."""
    exe = shutil.which("antsApplyTransforms")
    if exe is None:
        raise RuntimeError(
            "antsApplyTransforms not found on PATH. On Vulcan run "
            "`module load gcc/12.3 ants/2.6.5` before this script "
            "(the `ants` PyPI package in .venv is unrelated to ANTsPy)."
        )
    return exe


def find_t1w_to_mni_transform(fmriprep_dir: Path, subject: str) -> Path:
    """Locate the fMRIPrep composite T1w → MNI warp for one subject."""
    anat_dirs = _find_subdir(fmriprep_dir / subject, "anat")
    pattern = f"*from-T1w_to-{MNI_TEMPLATE_SPACE}_mode-image_xfm.h5"
    matches = [f for d in anat_dirs for f in sorted(d.glob(pattern))]
    if not matches:
        raise FileNotFoundError(f"No T1w→{MNI_TEMPLATE_SPACE} transform for {subject}")
    return matches[0]


def find_mni_reference(fmriprep_dir: Path, subject: str) -> Path:
    """Return the MNI BOLD reference defining the output grid.

    fMRIPrep resamples every subject onto the same template grid at the BOLD
    resolution, so the subjects stay voxel-aligned with one another.
    """
    func_dirs = _find_subdir(fmriprep_dir / subject, "func")
    pattern = f"*space-{MNI_TEMPLATE_SPACE}_boldref.nii.gz"
    matches = [f for d in func_dirs for f in sorted(d.glob(pattern))]
    if not matches:
        raise FileNotFoundError(f"No MNI boldref for {subject}; cannot define output grid")
    return matches[0]


def warp_volume_to_mni(
    volume: np.ndarray,
    affine: np.ndarray,
    *,
    fmriprep_dir: Path,
    subject: str,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a native T1w-space volume into MNI; return (data, MNI affine).

    NaNs mark the voxels outside the retained selection and would spread through
    the interpolation, so they travel as zeros and are restored afterwards.

    Nearest-neighbour interpolation keeps the sharp edge between retained and
    discarded voxels: linear interpolation blurs it into a ramp down to zero,
    which then paints a halo of sub-threshold values around every blob.

    ``antsApplyTransforms`` is a CLI and can only read and write files, so the
    input and output volumes are staged in a temporary directory and discarded
    once the data has been read back.
    """
    exe = require_ants()
    transform = find_t1w_to_mni_transform(fmriprep_dir, subject)
    reference = find_mni_reference(fmriprep_dir, subject)

    with tempfile.TemporaryDirectory(prefix="nc_warp_") as tmp:
        src = Path(tmp) / f"{subject}_{name}_space-T1w.nii.gz"
        dst = Path(tmp) / f"{subject}_{name}_space-{MNI_TEMPLATE_SPACE}.nii.gz"
        nib.save(  # type: ignore[attr-defined]
            nib.Nifti1Image(np.nan_to_num(volume).astype(np.float32), affine),  # type: ignore[attr-defined]
            str(src),
        )

        cmd = [
            exe,
            "--dimensionality", "3",
            "--input", str(src),
            "--reference-image", str(reference),
            "--transform", str(transform),
            "--interpolation", "NearestNeighbor",
            "--output", str(dst),
            "--float", "1",
        ]  # fmt: skip
        logger.info(f"Warping {name} to {MNI_TEMPLATE_SPACE} (ref {reference.name})")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"antsApplyTransforms failed for {src.name}:\n{result.stderr}")

        # nib.load is annotated as returning the generic FileBasedImage, which
        # exposes neither get_fdata nor affine; this is always a NIfTI.
        warped_img = cast("nib.Nifti1Image", nib.load(str(dst)))  # type: ignore[attr-defined]
        warped = np.asarray(warped_img.get_fdata())
        warped_affine = np.asarray(warped_img.affine)

    warped[warped <= 0] = np.nan
    return warped, warped_affine


def load_fsaverage_surfaces(mesh: str) -> dict:
    """fsaverage surfaces, keyed like :func:`load_native_surfaces`.

    Used for the MNI figures so every subject is drawn on the same cortex.
    """
    logger.info(f"Loading fsaverage surfaces ({mesh})")
    fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)
    surfaces = {}
    for hemisphere, suffix in [("left", "left"), ("right", "right")]:
        surfaces[f"{hemisphere}_pial"] = fsaverage[f"pial_{suffix}"]
        surfaces[f"{hemisphere}_inflated"] = fsaverage[f"infl_{suffix}"]
        surfaces[f"{hemisphere}_sulc"] = fsaverage[f"sulc_{suffix}"]
    return surfaces


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
    Project volume data to the sampling surface.

    CRITICAL: Use pial surface for sampling (data extraction from voxels),
    NOT inflated. The inflated surface has different vertex coordinates
    and won't align with the voxel volume.

    Uses nilearn's default ball sampling (3 mm radius around each pial vertex).
    Restricting to the white–pial ribbon (``inner_mesh``) is the stricter choice
    but yields a sparser map at every threshold, so the original behaviour is
    kept here.

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

    surface_data = surface.vol_to_surf(
        img,
        pial_surface,
        kind="auto",  # auto-detect mesh type from GIFTI
    )

    logger.info(f"Projected to {len(surface_data)} vertices")
    return surface_data


def project_selection_levels(
    volume_data: np.ndarray,
    cutoffs: dict[float, float],
    hemisphere: str,
    affine: np.ndarray,
    surfaces: dict,
    coverage: float = 0.25,
) -> dict[float, np.ndarray]:
    """Project one volume at several selection levels as flat 0/1 masks.

    Each level is the 0/1 selection mask of its cutoff, sampled onto the surface.
    Sampling averages the voxels near each vertex, so what comes back is the
    *fraction* of the vertex's neighbourhood inside the selection; a vertex covered
    less than ``coverage`` is dropped. Without that cut, a vertex merely touching
    the edge of a blob would still be drawn, painting a halo around every cluster.

    Coverage grows monotonically with the level, by construction. A wider level has
    a lower cutoff, so its voxel set contains the narrower one; a superset can only
    raise each vertex's covered fraction, so the drawn vertices are nested too.

    ``coverage`` decides how much of the cortex is drawn, and the answer is very
    sensitive to it. For sub-03 text at the top-20% level, where 13% of grey matter
    is selected in the volume, ball sampling renders 64% of vertices at
    any-overlap, 19.5% at 0.25, 8.5% at 0.5 and 3.1% at 0.75. The 0.25 default
    therefore mildly overstates the true extent, where 0.5 clearly understates it.

    Returns
    -------
    dict[float, np.ndarray]
        Per level, 1.0 at the vertices inside that selection and 0.0 elsewhere.
    """
    projected = {}
    for percent, cutoff in cutoffs.items():
        mask_volume = np.where(np.isfinite(volume_data) & (volume_data >= cutoff), 1.0, 0.0)
        covered = project_to_surface_native(mask_volume, hemisphere, affine, surfaces)
        selected = np.nan_to_num(covered) >= coverage
        logger.info(
            f"  {hemisphere} top {percent:g}%: {int(selected.sum())}/{len(selected)} vertices "
            f"({100 * selected.mean():.1f}%) inside the selection"
        )
        projected[percent] = selected.astype(np.float64)
    return projected


def nc_volume_and_cutoffs(
    nc_1d: np.ndarray,
    brain_mask_1d: np.ndarray,
    percents: list[float],
    spatial_dims: tuple[int, int, int],
) -> tuple[np.ndarray, dict[float, float]]:
    """Build the in-brain noise ceiling volume and the cutoff for each level.

    Cutoffs come from :func:`select_top_nc_voxels`, the same rule the encoding
    pipeline applies when it builds ``voxel_keep``, so the levels mark exactly the
    voxels the models are fit on. They are computed in volume space, before any
    warping, so the retained sets are defined on the data as acquired.

    Only the cutoffs differ between levels, so a single volume carrying every
    positive-NC voxel is returned and each level is a threshold on it. This holds
    through a nearest-neighbour warp, which relabels voxels without changing their
    values, so the volume can be warped once and thresholded afterwards.

    Note that a level retains well under ``percent`` of the brain: the percentile
    runs over voxels with a positive noise ceiling, which are only ~57% of the
    brain mask, so "top 20%" keeps ~11.5% of in-brain voxels (~13% of grey matter).
    "Top 100%" is therefore every positive-NC voxel, not literally every voxel.

    Returns
    -------
    tuple
        (3-D volume, {percent: NC cutoff}) — NaN outside the brain and wherever
        the noise ceiling is not positive.
    """
    volume = np.full(nc_1d.shape, np.nan, dtype=np.float32)
    in_brain = np.flatnonzero(brain_mask_1d)
    in_brain_nc = nc_1d[in_brain]

    cutoffs: dict[float, float] = {}
    for percent in percents:
        keep = select_top_nc_voxels(in_brain_nc, percent)
        kept_values = in_brain_nc[keep]
        cutoffs[percent] = float(np.nanmin(kept_values)) if kept_values.size else 0.0
        logger.info(
            f"  top {percent:g}%: {int(keep.sum())}/{len(in_brain)} in-brain voxels "
            f"(NC >= {cutoffs[percent]:.1f}%)"
        )

    # The widest level defines which voxels the volume needs to carry at all.
    widest = select_top_nc_voxels(in_brain_nc, max(percents))
    volume[in_brain[widest]] = in_brain_nc[widest]
    return volume.reshape(spatial_dims), cutoffs


def plot_surface_modality_overlay(
    nc_vol_text: np.ndarray,
    nc_vol_image: np.ndarray,
    affine: np.ndarray,
    percents: list[float],
    subject: str,
    native_surfaces: dict,
    cutoffs: dict[str, dict[float, float]],
    coverage: float = 0.25,
    space: str = "native",
    views: list[str] | None = None,
    font_scale: float = 1.0,
    row_spacing: float = -0.12,
    output_path: Path | None = None,
):
    """
    Plot the voxels the encoding models were fit on, at several selection levels.

    The grid has two rows per level — text then image — and hemisphere × view
    columns, so three levels give six rows. Levels are separated by a rule and
    labelled down the left-hand side.

    ``nc_vol_*`` carry every positive-NC voxel, in the target space; each level is
    a threshold on them (see :func:`nc_volume_and_cutoffs`).

    Each panel is drawn in one flat colour per modality — the map is binary, not a
    noise ceiling ramp — so the only thing that changes between levels is how much
    cortex is painted, which is what the figure is for.

    At the 20% level this covers ~13% of grey matter, not 20%: the analysis takes
    20% of the *positive-NC* voxels, and those are only ~57% of the brain mask.

    Parameters
    ----------
    nc_vol_text : np.ndarray
        3D noise ceiling volume for text stimuli, in the target space
    nc_vol_image : np.ndarray
        Same for image stimuli.
    affine : np.ndarray
        Affine transformation matrix for the volume
    percents : list[float]
        Selection levels, one group of two rows each, in the order given
    subject : str
        Subject identifier for title
    native_surfaces : dict
        Surface file paths from load_native_surfaces() or load_fsaverage_surfaces()
    cutoffs : dict[str, dict[float, float]]
        Per modality ("text"/"image"), the noise ceiling cutoff per level, from
        :func:`nc_volume_and_cutoffs`
    coverage : float
        Minimum fraction of a vertex's neighbourhood that must be inside the
        selection for it to be drawn
    space : str
        "native" or "mni", for the title only; the volumes must already be in it.
    views : list[str] | None
        Surface views per hemisphere (default ``["lateral", "medial"]``)
    font_scale : float
        Multiplier on every font size in the figure
    row_spacing : float
        ``hspace`` between rows. 3-D axes carry a lot of internal padding, so a
        negative value is needed to bring the surfaces close together.
    output_path : Path | None
        Path to save figure
    """
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    import nibabel as nib

    views = list(views) if views else ["lateral", "medial"]

    logger.info(f"Projecting the voxel selections to {space} surfaces")
    projected: dict[tuple[str, str], dict[float, np.ndarray]] = {}
    for modality, volume in [("text", nc_vol_text), ("image", nc_vol_image)]:
        for hemi in ["left", "right"]:
            projected[(modality, hemi)] = project_selection_levels(
                volume, cutoffs[modality], hemi, affine, native_surfaces, coverage=coverage
            )

    # Load sulcal depth for background
    logger.info("Loading surface geometry...")
    sulc_data = {}
    for hemi in ["left", "right"]:
        logger.debug(f"Loading sulcal depth for {hemi}...")
        sulc_path = native_surfaces[f"{hemi}_sulc"]
        sulc_gii = nib.load(sulc_path)
        # Access GIFTI data safely - darrays is a list of GiftiDataArray objects
        sulc_arrays = getattr(sulc_gii, "darrays", [])
        sulc_data[hemi] = np.asarray(sulc_arrays[0].data)
        logger.debug(f"  {hemi} sulc shape: {sulc_data[hemi].shape}")

    # Load native surface meshes for plotting (use INFLATED for display)
    fsaverage_meshes = {}
    for hemi in ["left", "right"]:
        logger.debug(f"Loading inflated surface mesh for display ({hemi})...")
        # Use INFLATED surface for visualization (prettier, shows folds)
        surf_path = native_surfaces[f"{hemi}_inflated"]
        surf_gii = nib.load(surf_path)
        surf_arrays = getattr(surf_gii, "darrays", [])
        vertices = np.asarray(surf_arrays[0].data)
        faces = np.asarray(surf_arrays[1].data)
        fsaverage_meshes[hemi] = (vertices, faces)
        logger.debug(
            f"  {hemi} inflated mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces"
        )
    logger.info("Surface geometry loaded ✓")

    # Rows: (level, modality) pairs. Columns: hemisphere × view.
    columns = [(hemi, view) for hemi in ["left", "right"] for view in views]
    rows = [(percent, modality) for percent in percents for modality in ["text", "image"]]
    cmaps = {m: ListedColormap([color]) for m, color in MODALITY_COLORS.items()}

    logger.info(f"Creating figure with shape ({len(rows)}, {len(columns)})...")
    fig, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(4 * len(columns), 2.9 * len(rows)),
        subplot_kw={"projection": "3d"},
        squeeze=False,
        gridspec_kw={"hspace": row_spacing, "wspace": 0.0},
    )

    logger.info("Populating subplots...")
    for row, (percent, modality) in enumerate(rows):
        for col, (hemisphere, view) in enumerate(columns):
            plotting.plot_surf_stat_map(
                surf_mesh=fsaverage_meshes[hemisphere],
                stat_map=projected[(modality, hemisphere)][percent],
                bg_map=sulc_data[hemisphere],
                # The map is 0/1 and the colormap is a single colour, so the scale
                # only has to bracket it; `threshold` is what hides the zeros.
                vmin=0.0,
                vmax=1.0,
                threshold=0.5,
                # nilearn takes a Colormap here, but its unannotated `cmap="…"`
                # default makes type checkers infer `str`.
                cmap=cast("str", cmaps[modality]),
                hemi=hemisphere,
                view=view,
                colorbar=False,
                axes=axes[row, col],
            )
            # `row_spacing` is negative, so neighbouring axes overlap. Their opaque
            # backgrounds would then paint over the ventral edge of the row above,
            # which reads as the brains being clipped.
            axes[row, col].patch.set_alpha(0.0)
            if row == 0:
                axes[row, col].set_title(
                    f"{hemisphere[0].upper()} {view}", fontsize=13 * font_scale, pad=0
                )
        axes[row, 0].text2D(
            -0.04, 0.5, modality.capitalize(), transform=axes[row, 0].transAxes,
            rotation=90, va="center", ha="center", fontsize=13 * font_scale,
        )  # fmt: skip

    space_label = MNI_TEMPLATE_SPACE if space == "mni" else "native T1w"
    fig.suptitle(
        f"{subject} - most reliable voxels (encoding selection) per modality ({space_label})",
        fontsize=17 * font_scale,
        fontweight="bold",
        y=0.995,
    )
    # `top` leaves the column headers clear of the title; the 3-D axes already
    # carry enough internal padding that the other margins can be tight.
    fig.subplots_adjust(left=0.09, right=0.99, top=0.925, bottom=0.01)

    _annotate_level_groups(fig, axes, percents, font_scale, Line2D)

    if output_path:
        logger.info(f"Saving figure to {output_path}...")
        output_fig = output_path.parent / f"{output_path.stem}_modality-overlay.png"
        fig.savefig(output_fig, dpi=150, bbox_inches="tight")
        logger.success(f"Saved modality overlay plot to {output_fig}")


def _annotate_level_groups(fig, axes, percents: list[float], font_scale: float, line_cls) -> None:
    """Label each selection level down the left edge and rule between the groups.

    Positions come from the axes bounding boxes, so this must run after the final
    ``subplots_adjust``.
    """
    for group, percent in enumerate(percents):
        top = axes[2 * group, 0].get_position()
        bottom = axes[2 * group + 1, 0].get_position()
        label = "All (NC > 0)" if percent >= 100 else f"Top {percent:g}%"

        fig.text(
            0.028,
            (top.y1 + bottom.y0) / 2,
            label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=17 * font_scale,
            fontweight="bold",
        )

        # Rule above every group but the first, midway into the gap left by the
        # preceding group's last row.
        if group:
            previous = axes[2 * group - 1, 0].get_position()
            y = (previous.y0 + top.y1) / 2
            fig.add_artist(line_cls([0.02, 0.995], [y, y], color="0.75", linewidth=1.2, zorder=10))


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════


@hydra.main(
    version_base="1.3",
    config_path="../../configs/visualization",
    config_name="visualize_noise_ceiling",
)
def main(cfg: DictConfig) -> None:
    """Generate noise ceiling surface plots for all subjects, in MNI or native space."""
    logger.info(f"Configuration: {cfg}")

    glmsingle_dir = Path(cfg.glmsingle_dir)
    subject_filter = cfg.get("subject", None)  # Optional filter for specific subject
    fmriprep_dir = Path(cfg.get("fmriprep_dir", ""))
    percents = [float(p) for p in cfg.get("percents", [20, 60, 100])]
    if not percents:
        raise ValueError("`percents` must list at least one selection level")
    space = str(cfg.get("space", "mni")).lower()
    if space not in {"mni", "native"}:
        raise ValueError(f"space must be 'mni' or 'native', got {space!r}")
    views = list(cfg.get("views", ["lateral", "medial"]))
    coverage = float(cfg.get("selection_coverage", 0.25))
    font_scale = float(cfg.get("font_scale", 1.0))
    row_spacing = float(cfg.get("row_spacing", -0.12))
    output_dir_cfg = cfg.get("output_dir")
    output_dir = (
        Path(output_dir_cfg) if output_dir_cfg is not None else FIGURES_DIR / "noise_ceiling"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fail before the expensive beta loading if ANTs is missing.
    if space == "mni":
        require_ants()

    # Find all subjects in glmsingle_dir
    subject_dirs = sorted(
        [d for d in glmsingle_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    )

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
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {subject}")
        logger.info(f"{'=' * 60}")

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
        logger.info(
            f"Applied brain mask. In-brain noise ceiling range: [{np.nanmin(nc_1d):.1f}, {np.nanmax(nc_1d):.1f}]%"
        )

        logger.info(f"Creating surface plots in {space} space...")

        # Load the surfaces the maps are displayed on: the subject's own cortex
        # for native space, the shared fsaverage cortex for MNI.
        if space == "mni":
            surfaces = load_fsaverage_surfaces(str(cfg.get("fsaverage_mesh", "fsaverage6")))
        else:
            surfaces = load_native_surfaces(fmriprep_dir, subject)
            logger.info(f"Loaded native surfaces for {subject}")

        # ===== Per-Modality Analysis =====
        # Load design matrix mapping to separate text and image stimuli
        design_mapping_file = Path(cfg.get("design_matrix_mapping_file", ""))
        logger.info(f"\n{'=' * 60}")
        logger.info("Computing per-modality noise ceiling...")
        logger.info(f"{'=' * 60}")

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
        for modality in ["text", "image"]:
            nc_modality = nc_by_modality[modality].copy()
            nc_modality[~brain_mask_1d] = np.nan

            logger.info(
                f"{modality.capitalize()} NC range: "
                f"[{np.nanmin(nc_modality):.1f}, {np.nanmax(nc_modality):.1f}]%"
            )

            nc_by_modality[modality] = nc_modality

        suffix = "" if space == "native" else f"_space-{MNI_TEMPLATE_SPACE}"

        # Cutoffs are percentiles of the native data, so they are computed before
        # warping. The warp is nearest-neighbour and so preserves voxel values,
        # which lets one volume per modality carry every level.
        levels = ", ".join(f"{p:g}%" for p in percents)
        logger.info(f"Selecting the most reliable voxels per modality at {levels}...")
        plot_volumes, cutoffs = {}, {}
        for modality in ["text", "image"]:
            plot_volumes[modality], cutoffs[modality] = nc_volume_and_cutoffs(
                nc_by_modality[modality], brain_mask_1d, percents, spatial_dims
            )

        plot_affine = affine
        if space == "mni":
            for modality in ["text", "image"]:
                plot_volumes[modality], plot_affine = warp_volume_to_mni(
                    plot_volumes[modality],
                    affine,
                    fmriprep_dir=fmriprep_dir,
                    subject=subject,
                    name=f"nc_{modality}",
                )

        plot_surface_modality_overlay(
            nc_vol_text=plot_volumes["text"],
            nc_vol_image=plot_volumes["image"],
            affine=plot_affine,
            percents=percents,
            subject=subject,
            output_path=output_dir / f"{subject}_noise_ceiling{suffix}_voxelsel.png",
            native_surfaces=surfaces,
            cutoffs=cutoffs,
            coverage=coverage,
            space=space,
            views=views,
            font_scale=font_scale,
            row_spacing=row_spacing,
        )

        logger.success(f"Surface visualization complete for {subject} → {output_dir}")


if __name__ == "__main__":
    main()
