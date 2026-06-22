"""Shared noise ceiling computation and beta normalization utilities.

Provides functions for computing noise ceiling estimates and normalizing
fMRI betas according to the VEM framework, used by both neural encoding
and visualization modules.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import nibabel as nib
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# Visualization Helpers
# ═══════════════════════════════════════════════════════════════════════════


CONDITION_LABELS: dict[str, str] = {
    "image_to_image": "Image embeddings\n→ Image fMRI",
    "image_to_text": "Image embeddings\n→ Text fMRI",
    "text_to_image": "Text embeddings\n→ Image fMRI",
    "text_to_text": "Text embeddings\n→ Text fMRI",
}


def configure_plot_fonts(
    *,
    font_family: str = "sans-serif",
    sans_serif: list[str] | None = None,
) -> None:
    """Configure matplotlib font defaults for plots."""
    from matplotlib import rcParams

    if sans_serif is None:
        sans_serif = [
            "Lato",
            "Lato Thin",
            "Carlito",
            "DejaVu Sans",
            "Arial",
        ]

    rcParams["font.family"] = font_family
    rcParams["font.sans-serif"] = sans_serif


def short_model_label(model_label: str) -> str:
    """Drop the redundant ``vendor--`` / ``vendor/`` prefix for display.

    E.g. ``OpenGVLab--InternVL3_5-1B-HF`` -> ``InternVL3_5-1B-HF``.
    """
    label = model_label
    for sep in ("--", "/"):
        if sep in label:
            label = label.rsplit(sep, 1)[-1]
    return label


def significance_label(p: float, alpha: float = 0.05) -> str:
    """Return a significance annotation string for *p*."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < alpha:
        return "*"
    return "ns"


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Returns q-values (adjusted p-values) with the same shape as *pvalues*.
    ``NaN`` entries are excluded from the ranking (family size) and returned
    as ``NaN``, so missing/inapplicable tests do not inflate the correction.
    """
    pvals = np.asarray(pvalues, dtype=float)
    flat = pvals.ravel()
    finite_mask = np.isfinite(flat)
    q_flat = np.full(flat.shape, np.nan, dtype=float)

    p = flat[finite_mask]
    m = p.size
    if m == 0:
        return q_flat.reshape(pvals.shape)

    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    # Enforce monotonicity (step-up): q_(i) = min over j>=i.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    q_finite = np.empty(m, dtype=float)
    q_finite[order] = adjusted
    q_flat[finite_mask] = q_finite
    return q_flat.reshape(pvals.shape)


def signflip_pvalue_greater(
    values: np.ndarray,
    *,
    n_permutations: int = 10000,
    random_state: int = 42,
) -> float:
    """One-sample sign-flip permutation p-value for mean(values) > 0."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = vals.size
    if n == 0:
        return float("nan")

    observed = float(np.mean(vals))

    if n <= 16:
        all_signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * n, indexing="ij")).reshape(n, -1).T
        null_stats = np.mean(all_signs * vals[None, :], axis=1)
        p = (np.sum(null_stats >= observed) + 1) / (null_stats.size + 1)
        return float(p)

    rng = np.random.default_rng(random_state)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n), replace=True)
    null_stats = np.mean(signs * vals[None, :], axis=1)
    p = (np.sum(null_stats >= observed) + 1) / (n_permutations + 1)
    return float(p)


# ═══════════════════════════════════════════════════════════════════════════
# Noise Ceiling - NCSNR Framework (for single-trial data)
# ═══════════════════════════════════════════════════════════════════════════


def compute_ncsnr(
    betas: np.ndarray,
    stimulus_ids: np.ndarray,
) -> np.ndarray:
    """Compute noise ceiling signal-to-noise ratio (NCSNR).

    Used for single-trial fMRI data to estimate the noise ceiling.
    Follows the framework in visualize_noise_ceiling where betas are
    normalized within runs and then noise ceiling is computed per voxel.

    Parameters
    ----------
    betas : np.ndarray
        Array of betas with shape (num_betas, num_voxels).
    stimulus_ids : np.ndarray
        Array specifying the stimulus ID for each beta, shape (num_betas).

    Returns
    -------
    np.ndarray
        Noise ceiling SNR per voxel with shape (num_voxels).
    """
    unique_ids = np.unique(stimulus_ids)

    betas_var = []
    for i in unique_ids:
        stimulus_betas = betas[stimulus_ids == i]
        betas_var.append(stimulus_betas.var(axis=0, ddof=1))
    betas_var_mean = np.nanmean(np.stack(betas_var), axis=0)

    std_noise = np.sqrt(betas_var_mean)

    std_signal = 1.0 - betas_var_mean
    std_signal[std_signal < 0.0] = 0.0
    std_signal = np.sqrt(std_signal)
    with np.errstate(divide="ignore", invalid="ignore"):
        ncsnr = std_signal / std_noise
    return ncsnr


def compute_nc(ncsnr: np.ndarray, num_averages: int | float = 1) -> np.ndarray:
    """Convert noise ceiling SNR to actual noise ceiling estimate.

    Parameters
    ----------
    ncsnr : np.ndarray
        Noise ceiling SNR values with shape (num_voxels).
    num_averages : int | float, optional
        Number of repetitions averaged together (default 1). Can be an
        effective (non-integer) value when repetition counts vary.

    Returns
    -------
    np.ndarray
        Noise ceiling (0-100 scale) per voxel with shape (num_voxels).
    """
    ncsnr_squared = ncsnr**2
    with np.errstate(divide="ignore", invalid="ignore"):
        nc = 100.0 * ncsnr_squared / (ncsnr_squared + (1.0 / num_averages))
    return nc


def compute_nc_by_modality(
    betas: np.ndarray,
    stimulus_ids: np.ndarray,
    modality_map: dict[int, str],
    num_averages: int | float = 6,
) -> dict[str, np.ndarray]:
    """Compute noise ceiling separately for each modality (text vs image).

    Parameters
    ----------
    betas : np.ndarray
        Array of betas with shape (num_betas, num_voxels).
    stimulus_ids : np.ndarray
        Array specifying stimulus condition for each beta, shape (num_betas).
    modality_map : dict
        Mapping from condition index (int) to modality (str: 'text' or 'image').
    num_averages : int | float, optional
        Number of repetitions assumed to be averaged in the target response
        when converting NCSNR to NC (default 6).

    Returns
    -------
    dict
        Dictionary with keys 'text' and 'image', each containing NC array
        (num_voxels,) with noise ceiling values.
    """
    nc_by_modality = {}

    # Ensure stimulus_ids is a 1D integer array
    stimulus_ids = np.asarray(stimulus_ids, dtype=np.int64).flatten()

    for modality in ["text", "image"]:
        # Find indices of trials for this modality by mapping each stimulus ID
        trial_modalities = []
        for stim_id in stimulus_ids:
            stim_id_int = int(stim_id)
            trial_modalities.append(modality_map.get(stim_id_int, "unknown"))
        trial_modalities = np.array(trial_modalities)
        modality_mask = trial_modalities == modality

        # Get betas and stimulus IDs for this modality
        modality_betas = betas[modality_mask]
        modality_stim_ids = stimulus_ids[modality_mask].astype(np.int64)

        logger.info(
            f"{modality.capitalize()}: {len(modality_betas)} trials, "
            f"{len(np.unique(modality_stim_ids))} unique stimuli"
        )

        # Compute noise ceiling for this modality
        ncsnr = compute_ncsnr(modality_betas, modality_stim_ids)
        nc = compute_nc(ncsnr, num_averages=num_averages)
        nc_by_modality[modality] = nc

    return nc_by_modality


# ═══════════════════════════════════════════════════════════════════════════
# Beta Normalization
# ═══════════════════════════════════════════════════════════════════════════


def normalize_betas_per_run(
    betas: np.ndarray,
    events_df: pd.DataFrame | None = None,
    num_runs: int | None = None,
) -> np.ndarray:
    """Normalize betas within each run (z-score across all trials per run).

    Handles two input formats:
    - With events_df: Uses run_label column to identify run boundaries (flexible).
    - With num_runs: Assumes uniform structure (divides conditions uniformly by num_runs).

    For each run, every voxel's response is independently centred and scaled
    across the trials in that run. This removes each voxel's run-specific mean
    and variance, preventing run-level offsets from inflating between-run variance.

    Parameters
    ----------
    betas : np.ndarray
        Betas with shape (n_voxels, n_trials) if events_df provided,
        or (num_conditions, num_voxels) if num_runs provided.
    events_df : pd.DataFrame, optional
        Must contain ``beta_index`` and ``run_label`` columns.
        Used when data has variable trials per run (neural encoding case).
    num_runs : int, optional
        Number of runs (assumes uniform structure: num_conditions / num_runs).
        Used when betas are flattened (visualization case).

    Returns
    -------
    np.ndarray
        Normalized/z-scored copy with same shape as input.
    """
    betas_z = betas.copy().astype(np.float64)

    if events_df is not None:
        # Use events_df to identify run boundaries (flexible per-trial approach)
        for _, group in events_df.groupby("run_label"):
            idx = np.asarray(group["beta_index"].values, dtype=int)
            run_data = betas_z[:, idx]  # (n_voxels, n_trials_in_run)
            mu = run_data.mean(axis=1, keepdims=True)
            sd = run_data.std(axis=1, keepdims=True, ddof=0)
            sd[sd == 0] = 1.0  # avoid division by zero for constant voxels
            betas_z[:, idx] = (run_data - mu) / sd

        n_runs = events_df["run_label"].nunique()
        logger.info(f"  Normalized betas within {n_runs} runs")

    elif num_runs is not None:
        # Assume uniform structure: reshape and normalize per run
        num_conditions = betas_z.shape[0]
        num_voxels = betas_z.shape[1]
        conditions_per_run = num_conditions // num_runs

        logger.info(
            f"Reshaping {num_conditions} conditions into {num_runs} runs × "
            f"{conditions_per_run} conditions/run"
        )

        # Reshape to (num_runs, conditions_per_run, num_voxels)
        betas_runs = betas_z.reshape(num_runs, conditions_per_run, num_voxels)

        # Normalize within each run
        for run_idx in range(num_runs):
            run_betas = betas_runs[run_idx]
            mean = np.mean(run_betas, axis=0)
            std = np.std(run_betas, axis=0)
            std[std == 0] = 1.0  # Avoid division by zero
            betas_runs[run_idx] = (run_betas - mean) / std

        # Reshape back to original shape
        betas_z = betas_runs.reshape(num_conditions, num_voxels)

    else:
        raise ValueError("Either events_df or num_runs must be provided")

    return betas_z


# ═══════════════════════════════════════════════════════════════════════════
# Design Matrix and Modality Mapping
# ═══════════════════════════════════════════════════════════════════════════


def load_design_matrix_mapping(design_mapping_file: Path) -> dict[int, str]:
    """Load design matrix mapping CSV and create condition index to modality mapping.

    Parameters
    ----------
    design_mapping_file : Path
        Path to design_matrix_mapping.csv file with columns: design_matrix_idx, coco_id
        where coco_id format is "{id}_text" or "{id}_image".

    Returns
    -------
    dict
        Mapping from condition index (int) to modality (str: 'text' or 'image').
    """
    df = pd.read_csv(design_mapping_file)
    modality_map = {}

    for _, row in df.iterrows():
        idx = int(row["design_matrix_idx"])
        coco_id = str(row["coco_id"])

        if coco_id.endswith("_text"):
            modality_map[idx] = "text"
        else:
            modality_map[idx] = "image"

    logger.info(f"Loaded mapping for {len(modality_map)} conditions")
    text_count = sum(1 for m in modality_map.values() if m == "text")
    image_count = sum(1 for m in modality_map.values() if m == "image")
    logger.info(f"  Text conditions: {text_count}, Image conditions: {image_count}")

    return modality_map


# ═══════════════════════════════════════════════════════════════════════════
# fMRI Trial Cache Helpers
# ═══════════════════════════════════════════════════════════════════════════


def build_fmri_cache(
    events_df: pd.DataFrame,
    *,
    betas: np.ndarray,
    brain_mask: np.ndarray,
    nc_corr_by_modality_full: dict[str, np.ndarray],
    conditions: dict,
    nc_top_percent: float = 0.0,
    log: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Build per-modality fMRI cache with optional NC top-percentile filtering."""
    fmri_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for fmri_mod in {c["fmri_modality"] for c in conditions.values()}:
        mod_df = events_df[events_df["modality"] == fmri_mod]

        trial_indices = np.asarray(mod_df["beta_index"].values, dtype=int)
        trial_coco_ids = np.asarray(mod_df["cocoid"].values, dtype=int)
        trial_betas = betas[:, trial_indices].T  # (n_trials, n_voxels)

        if fmri_mod not in nc_corr_by_modality_full:
            raise KeyError(
                f"Modality '{fmri_mod}' missing in NC computation. "
                "Check design_matrix_mapping_file."
            )

        nc_ceiling = nc_corr_by_modality_full[fmri_mod][brain_mask]

        n_in_brain = brain_mask.sum()
        if nc_top_percent > 0:
            valid_nc = np.isfinite(nc_ceiling) & (nc_ceiling > 0)
            if valid_nc.any():
                cutoff = np.nanpercentile(nc_ceiling[valid_nc], 100.0 - nc_top_percent)
                voxel_keep = valid_nc & (nc_ceiling >= cutoff)
            else:
                voxel_keep = np.isfinite(nc_ceiling)
            nc_ceiling = nc_ceiling[voxel_keep]
        else:
            voxel_keep = np.ones(n_in_brain, dtype=bool)

        trial_betas = trial_betas[:, voxel_keep]
        fmri_cache[fmri_mod] = (trial_coco_ids, trial_betas, nc_ceiling, voxel_keep)

        if log:
            logger.info(
                f"  fMRI {fmri_mod}: {len(trial_coco_ids)} trials "
                f"({len(np.unique(trial_coco_ids))} stimuli, "
                f" NC) × {trial_betas.shape[1]} voxels"
            )

    return fmri_cache


# ═══════════════════════════════════════════════════════════════════════════
# fMRIPrep Space and Mask Handling (Subject-Native T1w)
# ═══════════════════════════════════════════════════════════════════════════


def _find_subdir(subject_dir: Path, subdir: str) -> list[Path]:
    """Find session or subject-level subdirectories.

    Looks for session-specific subdirectory (ses-*/subdir), then falls back
    to subject-level subdirectory (subdir) if not found.

    Parameters
    ----------
    subject_dir : Path
        Root subject directory.
    subdir : str
        Name of subdirectory to find (e.g., "func", "anat").

    Returns
    -------
    list[Path]
        Sorted list of matching subdirectories.
    """
    dirs = sorted(subject_dir.glob(f"ses-*/{subdir}"))
    if not dirs and (subject_dir / subdir).exists():
        dirs = [subject_dir / subdir]
    return dirs


def get_affine(fmriprep_dir: Path, subject: str) -> np.ndarray:
    """Get affine from fMRIPrep native T1w space BOLD.

    CRITICAL: Explicitly loads T1w (subject-native) space BOLD to guarantee
    native anatomy. fMRIPrep outputs both space-T1w (native) and
    space-MNI152NLin2009cAsym files - must filter for T1w only.

    Parameters
    ----------
    fmriprep_dir : Path
        Path to fMRIPrep output directory.
    subject : str
        Subject identifier (e.g., "sub-001").

    Returns
    -------
    np.ndarray
        4×4 affine transformation matrix for native T1w space.
    """
    subject_dir = fmriprep_dir / subject
    func_dirs = _find_subdir(subject_dir, "func")
    if not func_dirs:
        raise FileNotFoundError(f"No func directory found for {subject}")

    # Get T1w BOLD files from first func directory
    bold_files = list(func_dirs[0].glob("*space-T1w*desc-preproc_bold.nii.gz"))
    if not bold_files:
        raise FileNotFoundError(
            f"No T1w space BOLD files found for {subject}. Check fMRIPrep outputs."
        )

    # Load affine from first T1w BOLD file
    affine = np.asarray(nib.load(str(bold_files[0])).affine)  # type: ignore
    logger.info(f"Loaded affine from fMRIPrep T1w (native) space: {bold_files[0].name}")
    return affine


def load_brain_mask_img(fmriprep_dir: Path, subject: str) -> "nib.Nifti1Image":
    """Return the fMRIPrep T1w brain-mask as a nibabel image (affine + shape).

    Use this when you need the 3-D geometry to reconstruct a volumetric NIfTI
    from a flat per-voxel array.  For the plain boolean mask use
    :func:`load_brain_mask`.
    """
    subject_dir = fmriprep_dir / subject
    func_dirs = _find_subdir(subject_dir, "func")
    if not func_dirs:
        raise FileNotFoundError(f"No func directory found for {subject}")

    mask_files = [f for d in func_dirs for f in d.glob("*space-T1w*_desc-brain_mask.nii.gz")]
    if not mask_files:
        raise FileNotFoundError(f"No T1w space brain mask found for {subject}.")
    return nib.load(mask_files[0])  # type: ignore[return-value]


def save_voxelwise_nifti(
    values: np.ndarray,
    voxel_keep: np.ndarray,
    brain_mask: np.ndarray,
    ref_img: "nib.Nifti1Image",
    out_path: Path,
) -> None:
    """Save a per-voxel array as a NIfTI volume in native T1w space.

    Parameters
    ----------
    values : (n_selected_voxels,)
        Scalar values (e.g. Pearson r) for the *selected* in-brain voxels.
    voxel_keep : (n_in_brain_voxels,) bool
        Mask indicating which in-brain voxels are included in *values*.
        ``values[i]`` corresponds to the ``i``-th True position in voxel_keep.
    brain_mask : (n_flat_voxels,) bool
        Full flattened brain mask mapping in-brain voxels to the 3-D volume.
    ref_img : nib.Nifti1Image
        Reference image that provides the 3-D shape and affine.
    out_path : Path
        Destination ``.nii.gz`` path.
    """
    shape_3d = ref_img.shape[:3]
    volume = np.full(int(np.prod(shape_3d)), np.nan, dtype=np.float32)

    # Map selected values into the full flattened space
    in_brain_indices = np.flatnonzero(brain_mask)
    selected_indices = in_brain_indices[voxel_keep]
    volume[selected_indices] = values.astype(np.float32)

    img = nib.Nifti1Image(  # type: ignore[attr-defined]
        volume.reshape(shape_3d), affine=ref_img.affine, header=ref_img.header
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))  # type: ignore[attr-defined]
    logger.debug(f"Saved NIfTI map → {out_path}")


def load_brain_mask(fmriprep_dir: Path, subject: str) -> np.ndarray:
    """Load brain mask from fMRIPrep outputs in subject-native T1w space.

    Loads the brain mask and returns as 1D flattened array for voxel
    operations.

    Parameters
    ----------
    fmriprep_dir : Path
        Path to fMRIPrep output directory.
    subject : str
        Subject identifier (e.g., "sub-001").

    Returns
    -------
    np.ndarray
        Boolean mask array, shape (n_voxels,), where True indicates in-brain.
    """
    subject_dir = fmriprep_dir / subject
    func_dirs = _find_subdir(subject_dir, "func")
    if not func_dirs:
        raise FileNotFoundError(f"No func directory found for {subject}")

    # Find T1w space brain mask file
    mask_files = [f for d in func_dirs for f in d.glob("*space-T1w*_desc-brain_mask.nii.gz")]
    if not mask_files:
        raise FileNotFoundError(
            f"No T1w space brain mask found for {subject}. Check fMRIPrep outputs."
        )

    # Load mask and flatten
    mask_img = nib.load(mask_files[0])
    mask_data = np.asarray(mask_img.get_fdata(), dtype=bool)  # type: ignore
    mask_1d = mask_data.reshape(-1)

    logger.info(
        f"Brain mask: {int(np.asarray(mask_1d).sum())} in-brain voxels out of {len(mask_1d)} total"
    )
    return mask_1d
