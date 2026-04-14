"""Shared noise ceiling computation and beta normalization utilities.

Provides functions for computing noise ceiling estimates and normalizing
fMRI betas according to the VEM framework, used by both neural encoding
and visualization modules.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from loguru import logger


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
            f"No T1w space BOLD files found for {subject}. "
            f"Check fMRIPrep outputs."
        )

    # Load affine from first T1w BOLD file
    affine = np.asarray(nib.load(str(bold_files[0])).affine)  # type: ignore
    logger.info(
        f"Loaded affine from fMRIPrep T1w (native) space: {bold_files[0].name}"
    )
    return affine


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
    mask_files = [
        f for d in func_dirs
        for f in d.glob("*space-T1w*_desc-brain_mask.nii.gz")
    ]
    if not mask_files:
        raise FileNotFoundError(
            f"No T1w space brain mask found for {subject}. "
            f"Check fMRIPrep outputs."
        )

    # Load mask and flatten
    mask_img = nib.load(mask_files[0])
    mask_data = np.asarray(mask_img.get_fdata(), dtype=bool)  # type: ignore
    mask_1d = mask_data.reshape(-1)

    logger.info(
        f"Brain mask: {int(np.asarray(mask_1d).sum())} in-brain voxels "
        f"out of {len(mask_1d)} total"
    )
    return mask_1d
