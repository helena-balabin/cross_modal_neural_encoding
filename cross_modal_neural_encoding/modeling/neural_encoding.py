"""Neural encoding: predict fMRI responses from VLM embeddings.

Fits ridge-regression encoding models to predict single-trial fMRI
responses (GLMsingle Type-D betas) from Vision-Language Model embeddings,
analyzing data in subject-native T1w space with brain masking.

Four encoding conditions (configurable):

  1. **text_embed  → text_fmri**   (within-modality)
  2. **image_embed → image_fmri**  (within-modality)
  3. **image_embed → text_fmri**   (cross-modal)
  4. **text_embed  → image_fmri**  (cross-modal)

Workflow
--------
1. Load VLM embeddings (vision + text) and PCA-reduce them.
2. For each subject (subject-native T1w space):

    a. Load GLMsingle Type-D betas and apply subject brain mask.
    b. Normalize betas within run.
    c. Compute modality-specific noise ceilings using the same method
       as the visualization pipeline (DESIGNINFO + design mapping).
    d. Parse BIDS events to map trial index → COCO ID + modality.
    e. Build single-trial samples for each fMRI modality.
    f. For each encoding condition, align trials with embeddings by COCO ID.
    g. Fit fractional ridge with nested CV:
       - outer CV: GroupKFold (`n_outer_folds`) or single GroupShuffleSplit
       - inner CV: GroupKFold for voxelwise frac selection.
    h. Evaluate with per-voxel Pearson *r* on held-out data
       (optionally averaging repeats in the test set).
    i. Normalize *r* by per-voxel noise ceiling in correlation units.

3. Aggregate results across subjects.

Configuration
--------------
- fmriprep_dir: Required. Path to fMRIPrep outputs for brain mask loading.
- Analyses assume fMRIPrep preprocessing with FreeSurfer surface reconstruction.
- All voxel-wise operations performed in subject-native T1w anatomical space.

Usage
-----
    python -m cross_modal_neural_encoding.modeling.neural_encoding

Hydra config: ``configs/neural_encoding.yaml``
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from fracridge import FracRidgeRegressor
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from cross_modal_neural_encoding.config import PROJ_ROOT
from cross_modal_neural_encoding.utils import (
    normalize_betas_per_run,
    compute_nc_by_modality,
    load_design_matrix_mapping,
    load_brain_mask,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_events(
    bids_root: Path,
    subject: str,
    *,
    sessions: list[int],
    runs_per_session: int,
    task: str,
    modality_column: str,
    cocoid_column: str,
) -> pd.DataFrame:
    """Parse BIDS events files and return a table of non-blank trials.

    Iterates over sessions and runs in canonical order so that the
    sequential ``beta_index`` matches the GLMsingle ``betasmd`` last
    dimension.

    Returns
    -------
    DataFrame with columns ``beta_index``, ``cocoid`` (int), ``modality``,
    ``run_label``.
    """
    records: list[dict] = []
    beta_idx = 0
    run_counter = 0  # global run index across all sessions

    for ses in sessions:
        for run in range(1, runs_per_session + 1):
            fname = (
                f"{subject}_ses-{ses:02d}_task-{task}_run-{run:02d}_events.tsv"
            )
            events_path = (
                bids_root / subject / f"ses-{ses:02d}" / "func" / fname
            )
            if not events_path.exists():
                logger.warning(f"Missing events file: {events_path}")
                continue

            df_run = pd.read_csv(events_path, sep="\t").sort_values("onset")

            for _, row in df_run.iterrows():
                mod = str(row[modality_column]).strip().lower()
                cid = row[cocoid_column]

                invalid = False
                if mod in ("blank", "nan", "n/a", ""):
                    invalid = True
                if pd.isna(cid) or str(cid).strip().lower() == "n/a":
                    invalid = True

                if not invalid:
                    records.append(
                        {
                            "beta_index": beta_idx,
                            "cocoid": int(float(cid)),
                            "modality": mod,
                            "run_label": run_counter,
                        }
                    )
                    # GLMsingle betas are defined for retained task trials
                    # (blank/invalid events are not represented in betas).
                    beta_idx += 1

            run_counter += 1

    events_df = pd.DataFrame(records)
    n_img = (events_df["modality"] == "image").sum()
    n_txt = (events_df["modality"] == "text").sum()
    logger.info(
        f"  {subject}: {len(events_df)} trials "
        f"({n_img} image, {n_txt} text, "
        f"{events_df['cocoid'].nunique()} unique stimuli)"
    )
    return events_df


def load_designinfo_stimulus_ids_and_num_runs(
    glmsingle_root: Path,
    subject: str,
    n_trials: int,
) -> tuple[np.ndarray, int]:
    """Load DESIGNINFO stimulus IDs and run count for visualization-matched NC.

    Returns
    -------
    stimulus_ids : (n_trials,) design-matrix condition index per trial.
    num_runs : run count inferred from DESIGNINFO (fallback 36).
    """
    designinfo_file = glmsingle_root / subject / "DESIGNINFO.npy"
    stimulus_ids = np.arange(n_trials, dtype=int)
    num_runs = 36

    if not designinfo_file.exists():
        logger.warning(
            f"  {subject}: DESIGNINFO.npy not found, using fallback "
            f"stimulus_ids=arange(n_trials), num_runs={num_runs}."
        )
        return stimulus_ids, num_runs

    designinfo = np.load(designinfo_file, allow_pickle=True).item()
    stimulus_ids = np.asarray(
        designinfo.get("stimorder", stimulus_ids), dtype=int
    ).flatten()

    if stimulus_ids.shape[0] != n_trials:
        raise ValueError(
            f"{subject}: DESIGNINFO stimorder length {stimulus_ids.shape[0]} "
            f"does not match betas trials {n_trials}."
        )

    design_list = designinfo.get("design", [])
    if len(design_list) > 0:
        num_runs = int(len(design_list))

    return stimulus_ids, num_runs


def load_condition_to_cocoid_modality(
    design_mapping_file: Path,
) -> dict[int, tuple[int, str]]:
    """Load condition-index → (COCO ID, modality) mapping.

    Expects CSV columns ``design_matrix_idx`` and ``coco_id`` where
    ``coco_id`` has the format ``{id}_text`` or ``{id}_image``.
    """
    df = pd.read_csv(design_mapping_file)
    mapping: dict[int, tuple[int, str]] = {}

    for _, row in df.iterrows():
        cond_idx = int(row["design_matrix_idx"])
        coco_raw = str(row["coco_id"]).strip()
        coco_stem, modality = coco_raw.rsplit("_", 1)
        mapping[cond_idx] = (int(coco_stem), modality.lower())

    return mapping


def build_events_from_stimorder(
    stimulus_ids: np.ndarray,
    condition_to_coco: dict[int, tuple[int, str]],
    subject: str,
) -> pd.DataFrame:
    """Build beta-index mapping directly from DESIGNINFO stimorder.

    This aligns exactly with GLMsingle beta ordering:
    ``stimorder[beta_index] = condition_index``.
    """
    records: list[dict] = []
    n_missing = 0

    for beta_i, cond_idx in enumerate(stimulus_ids):
        cond_idx = int(cond_idx)
        if cond_idx not in condition_to_coco:
            n_missing += 1
            continue
        cocoid, modality = condition_to_coco[cond_idx]
        records.append(
            {
                "beta_index": beta_i,
                "cocoid": int(cocoid),
                "modality": modality,
            }
        )

    events_df = pd.DataFrame(records)
    n_img = (events_df["modality"] == "image").sum()
    n_txt = (events_df["modality"] == "text").sum()
    logger.info(
        f"  {subject}: {len(events_df)} trials from stimorder "
        f"({n_img} image, {n_txt} text, "
        f"{events_df['cocoid'].nunique()} unique stimuli, "
        f"{n_missing} unmapped conditions)"
    )
    return events_df


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference with NaN-safe handling."""
    if a.shape != b.shape:
        return float("nan")
    diff = np.abs(a - b)
    return float(np.nanmean(diff))


def load_fmri(
    glmsingle_root: Path,
    subject: str,
) -> np.ndarray:
    """Load Type-D betas for *subject*.

    Returns
    -------
    betas : (n_voxels, n_trials) – flattened denoised betas.
    """
    sub_dir = glmsingle_root / subject

    typed = np.load(
        sub_dir / "TYPED_FITHRF_GLMDENOISE_RR.npy", allow_pickle=True
    ).item()
    betas_4d: np.ndarray = typed["betasmd"]  # (X, Y, Z, n_trials)
    betas = betas_4d.reshape(-1, betas_4d.shape[-1])  # (n_vox, n_trials)

    logger.info(
        f"  {subject}: betas {betas_4d.shape} → "
        f"({betas.shape[0]}, {betas.shape[1]})"
    )
    return betas


def load_embeddings(
    embeddings_dir: Path,
    model_label: str,
    embed_modality: str,
    layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load VLM embeddings and their COCO-ID keys.

    Parameters
    ----------
    embed_modality : ``"vision"`` or ``"text"`` (directory prefix).

    Returns
    -------
    coco_ids : 1-D int array.
    embeddings : 2-D array (n_stimuli, hidden_dim).
    """
    d = embeddings_dir / model_label / f"{embed_modality}_embeddings"
    coco_ids = np.load(d / "coco_ids.npy").astype(int)
    embs = np.load(d / f"layer_{layer:03d}.npy")
    logger.info(f"  {embed_modality} embeddings: {embs.shape}")
    return coco_ids, embs


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------


def average_betas_by_stimulus(
    betas: np.ndarray,
    events_df: pd.DataFrame,
    mask: np.ndarray,
    modality_filter: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Average masked betas across repetitions for one fMRI modality.

    Returns
    -------
    coco_ids : (n_unique,)
    avg_betas : (n_unique, n_masked_voxels)
    """
    mod_df = events_df[events_df["modality"] == modality_filter]
    masked_betas = betas[mask]  # (n_masked_voxels, n_trials)

    grouped = mod_df.groupby("cocoid")["beta_index"].apply(list)
    coco_ids = np.array(grouped.index, dtype=int)
    avg_list = [masked_betas[:, idxs].mean(axis=1) for idxs in grouped.values]
    avg_betas = np.stack(avg_list, axis=0)  # (n_stimuli, n_masked_voxels)
    return coco_ids, avg_betas


def get_single_trial_data(
    betas: np.ndarray,
    events_df: pd.DataFrame,
    mask: np.ndarray,
    modality_filter: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-trial masked betas for one fMRI modality.

    Unlike ``average_betas_by_stimulus`` this keeps every single trial
    as a separate row, so repeated presentations yield more samples.

    Returns
    -------
    coco_ids : (n_trials,) – COCO ID for each trial.
    trial_betas : (n_trials, n_masked_voxels)
    """
    mod_df = events_df[events_df["modality"] == modality_filter]
    masked_betas = betas[mask]  # (n_masked_voxels, n_all_trials)

    indices = np.asarray(mod_df["beta_index"]).astype(int)
    coco_ids = np.asarray(mod_df["cocoid"]).astype(int)
    trial_betas = masked_betas[:, indices].T  # (n_trials, n_masked_voxels)
    return coco_ids, trial_betas


def align_single_trials(
    embed_ids: np.ndarray,
    embeddings: np.ndarray,
    trial_coco_ids: np.ndarray,
    trial_betas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Look up embeddings for each trial by COCO ID.

    Each trial is matched to its stimulus embedding.  Repeated
    presentations of the same stimulus share the same feature vector
    but have independent fMRI responses.

    Returns
    -------
    X : (n_trials, n_features) – embeddings (repeated per trial).
    Y : (n_trials, n_voxels) – single-trial betas.
    groups : (n_trials,) – COCO ID per trial (for ``GroupKFold``).
    """
    embed_lookup = {int(cid): i for i, cid in enumerate(embed_ids)}

    valid = np.array([int(cid) in embed_lookup for cid in trial_coco_ids])
    if valid.sum() == 0:
        raise ValueError("No overlapping COCO IDs between embeddings and fMRI.")

    trial_coco_ids = trial_coco_ids[valid]
    trial_betas = trial_betas[valid]

    embed_indices = np.array([embed_lookup[int(cid)] for cid in trial_coco_ids])
    X = embeddings[embed_indices]

    return X, trial_betas, trial_coco_ids


# ---------------------------------------------------------------------------
# Encoding model
# ---------------------------------------------------------------------------


def _pearson_r_columnwise(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Vectorised Pearson *r* per column (voxel)."""
    Yt = Y_true - Y_true.mean(0)
    Yp = Y_pred - Y_pred.mean(0)
    num = (Yt * Yp).sum(0)
    den = np.sqrt((Yt**2).sum(0) * (Yp**2).sum(0))
    den[den == 0] = 1.0
    return num / den


def _pearson_r_fracwise(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Pearson *r* per (frac, voxel).

    Parameters
    ----------
    Y_true : (n_samples, n_voxels)
    Y_pred : (n_samples, n_fracs, n_voxels)

    Returns
    -------
    (n_fracs, n_voxels)
    """
    Yt = Y_true - Y_true.mean(axis=0, keepdims=True)  # (n, v)
    Yp = Y_pred - Y_pred.mean(axis=0, keepdims=True)  # (n, f, v)

    num = np.einsum("nv,nfv->fv", Yt, Yp)
    den_t = np.sqrt((Yt**2).sum(axis=0, keepdims=True))  # (1, v)
    den_p = np.sqrt((Yp**2).sum(axis=0))  # (f, v)
    den = den_t * den_p
    den[den == 0] = 1.0
    return num / den


def run_encoding(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    frac_grid: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    n_inner_folds: int = 5,
    n_outer_folds: int = 1,
    noise_ceiling: np.ndarray | None = None,
    random_state: int = 42,
    average_test_by_group: bool = True,
    verbose: bool = True,
) -> dict:
    """Ridge encoding with nested group-aware CV.

     1. **Outer split** – if ``n_outer_folds > 1``, uses
         ``GroupKFold(n_splits=n_outer_folds)`` for proper outer CV over
         stimulus groups; otherwise uses a single ``GroupShuffleSplit`` holdout.
    2. **Inner CV for frac** – ``GroupKFold`` on the training stimuli
        is used to select a best frac **per voxel** from ``frac_grid``.
    3. **Feature centering** – ``StandardScaler(with_mean=True,
        with_std=False)`` centres X (no variance-scaling).
    4. **Y centering** – Y is centred on the training-set mean per
        voxel and the same shift is applied to the test set.
     5. **Evaluation** – by default, training uses single-trial samples, while
         held-out test trials are averaged per stimulus (group) before computing
         per-voxel Pearson *r*.

    Parameters
    ----------
    X : (n_samples, n_features) – PCA'd embeddings.
    Y : (n_samples, n_voxels) – fMRI responses (single-trial or averaged).
    frac_grid : 1-D array of candidate fractional ridge values in (0, 1].
    groups : (n_samples,) stimulus (COCO ID) per trial.
    test_size : fraction of *stimuli* held out for testing.
    n_inner_folds : number of inner CV folds for frac selection.
    n_outer_folds : number of outer CV folds. Set to 1 for a single
        train/test split controlled by ``test_size``.
    noise_ceiling : optional (n_voxels,) per-voxel noise ceiling
        in correlation units.
    random_state : random seed for the train/test split.
    average_test_by_group : if True, average held-out test responses per
        stimulus before scoring.

    Returns
    -------
    dict with ``per_voxel_r``, ``mean_r``,
    ``best_frac`` (mean across voxels), and (if *noise_ceiling* given)
    ``normalized_per_voxel_r``, ``mean_normalized_r``,
    ``mean_noise_ceiling_r``.
    """
    X = X.astype("float32")
    Y = Y.astype("float32")

    frac_grid = np.asarray(frac_grid, dtype=np.float64)
    n_fracs = len(frac_grid)
    n_voxels = Y.shape[1]

    if n_outer_folds > 1:
        outer_splits = list(
            GroupKFold(n_splits=int(n_outer_folds)).split(X, groups=groups)
        )
    else:
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        outer_splits = [next(gss.split(X, groups=groups))]
    n_outer_effective = len(outer_splits)

    def _fit_and_score_split(
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]

        n_train_stim = len(np.unique(groups_train))
        n_test_stim = len(np.unique(groups_test))

        # ---- 2. Centre Y on training set ----------------------------------
        Y_mean = Y_train.mean(axis=0, keepdims=True)
        Y_train = Y_train - Y_mean
        Y_test = Y_test - Y_mean

        # ---- 3. Inner CV for frac selection -------------------------------
        actual_inner = min(int(n_inner_folds), n_train_stim)
        inner_cv = GroupKFold(n_splits=actual_inner)
        inner_splits = list(inner_cv.split(X_train, groups=groups_train))

        # ---- 4. Scale X + inner selection --------------------------------
        scaler = StandardScaler(with_mean=True, with_std=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        cv_scores = np.zeros((n_fracs, n_voxels), dtype=np.float64)
        for tr_idx, val_idx in inner_splits:
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            Y_tr, Y_val = Y_train[tr_idx], Y_train[val_idx]

            inner_model = FracRidgeRegressor(
                fracs=frac_grid,
                fit_intercept=False,
            )
            inner_model.fit(X_tr, Y_tr)
            Y_val_pred_all = inner_model.predict(X_val)  # (n_val, f, v)
            if Y_val_pred_all.ndim == 2:
                Y_val_pred_all = Y_val_pred_all[:, np.newaxis, :]
            cv_scores += _pearson_r_fracwise(Y_val, Y_val_pred_all)

        cv_scores /= len(inner_splits)
        best_frac_idx = np.argmax(cv_scores, axis=0)  # (n_voxels,)
        best_frac_vox = frac_grid[best_frac_idx]  # (n_voxels,)

        # ---- 5. Final fit + test -----------------------------------------
        final_model = FracRidgeRegressor(
            fracs=frac_grid,
            fit_intercept=False,
        )
        final_model.fit(X_train, Y_train)

        if average_test_by_group:
            test_unique = np.unique(groups_test)
            # Embeddings are repeated per stimulus; one row per group is enough.
            first_idx = [np.flatnonzero(groups_test == g)[0] for g in test_unique]
            X_test_eval = X_test[first_idx]
            Y_test_eval = np.stack(
                [Y_test[groups_test == g].mean(axis=0) for g in test_unique],
                axis=0,
            )
            Y_eval_pred_all = final_model.predict(X_test_eval)
            if Y_eval_pred_all.ndim == 2:
                Y_eval_pred_all = Y_eval_pred_all[:, np.newaxis, :]
            Y_eval_pred = Y_eval_pred_all[:, best_frac_idx, np.arange(n_voxels)]
            per_voxel_r_split = _pearson_r_columnwise(Y_test_eval, Y_eval_pred)
        else:
            Y_test_pred_all = final_model.predict(X_test)  # (n_test, f, v)
            if Y_test_pred_all.ndim == 2:
                Y_test_pred_all = Y_test_pred_all[:, np.newaxis, :]
            Y_pred = Y_test_pred_all[:, best_frac_idx, np.arange(n_voxels)]
            per_voxel_r_split = _pearson_r_columnwise(Y_test, Y_pred)

        return per_voxel_r_split, best_frac_vox, n_train_stim, n_test_stim

    fold_r_list: list[np.ndarray] = []
    fold_best_frac_list: list[np.ndarray] = []
    n_train_list: list[int] = []
    n_test_list: list[int] = []

    for outer_i, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        if verbose:
            n_train_stimuli_fold = len(np.unique(groups[train_idx]))
            n_test_stimuli_fold = len(np.unique(groups[test_idx]))
            if n_outer_effective > 1:
                logger.info(
                    f"      Outer fold {outer_i}/{n_outer_effective}: "
                    f"{len(train_idx)} train samples ({n_train_stimuli_fold} stimuli) / "
                    f"{len(test_idx)} test samples ({n_test_stimuli_fold} stimuli)"
                )
            else:
                logger.info(
                    f"      Split: {len(train_idx)} train samples "
                    f"({n_train_stimuli_fold} stimuli) / "
                    f"{len(test_idx)} test samples ({n_test_stimuli_fold} stimuli)"
                )

        fold_r, fold_best_frac, n_train_stim, n_test_stim = _fit_and_score_split(
            train_idx, test_idx
        )
        fold_r_list.append(fold_r)
        fold_best_frac_list.append(fold_best_frac)
        n_train_list.append(n_train_stim)
        n_test_list.append(n_test_stim)

    per_voxel_r = np.nanmean(np.stack(fold_r_list, axis=0), axis=0)
    best_frac_per_voxel = np.nanmean(
        np.stack(fold_best_frac_list, axis=0), axis=0
    )
    frac_val = float(np.nanmean(best_frac_per_voxel))
    if verbose:
        logger.info(f"      Mean best frac = {frac_val:.3f}")

    n_train_stimuli = int(np.round(np.mean(n_train_list)))
    n_test_stimuli = int(np.round(np.mean(n_test_list)))

    result: dict = {
        "per_voxel_r": per_voxel_r,
        "mean_r": float(np.nanmean(per_voxel_r)),
        "best_frac": frac_val,
        "best_frac_per_voxel": best_frac_per_voxel,
        "n_train_stimuli": n_train_stimuli,
        "n_test_stimuli": n_test_stimuli,
        "n_outer_folds": n_outer_effective,
    }

    # ---- 6. Noise-ceiling normalisation -----------------------------------
    if noise_ceiling is not None:
        # ``noise_ceiling`` is already in correlation units.
        nc_r = np.clip(noise_ceiling, 0, None)
        valid = nc_r > 0
        normalized = np.full_like(per_voxel_r, np.nan)
        normalized[valid] = per_voxel_r[valid] / nc_r[valid]
        result["normalized_per_voxel_r"] = normalized
        result["mean_normalized_r"] = float(np.nanmean(normalized[valid]))
        mean_nc = float(np.mean(nc_r[valid]))
        max_nc = float(np.max(nc_r[valid]))
        result["mean_noise_ceiling_r"] = mean_nc
        result["max_noise_ceiling_r"] = max_nc
        # Backward-compatible aliases.
        result["mean_ev"] = mean_nc
        result["max_ev"] = max_nc

    return result


def run_permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    frac_grid: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    n_inner_folds: int = 5,
    n_outer_folds: int = 1,
    noise_ceiling: np.ndarray | None = None,
    n_permutations: int = 100,
    random_state: int = 42,
    real_result: dict,
) -> dict:
    """Permutation test for the encoding model.

    Shuffles the stimulus-to-embedding mapping (rows of *X*) to break
    the true correspondence between embeddings and fMRI responses,
    re-runs the full encoding pipeline, and builds a null distribution
    of mean *r*.

    The p-value is computed as ``(#{null ≥ real} + 1) / (n_perm + 1)``
    to avoid *p* = 0 and to correct for the
    finite number of permutations.

    Parameters
    ----------
    X, Y, frac_grid, groups, test_size, n_inner_folds, noise_ceiling
        Identical to ``run_encoding``.
    n_permutations : int
        Number of random shuffles (default 100).
    random_state : int
        Seed for the permutation RNG.
    real_result : dict
        Result dict returned by the real (un-shuffled) ``run_encoding``
        call; used to place the observed *r* in the null distribution.

    Returns
    -------
    dict with keys ``null_mean_r``, ``p_value_mean_r``.
    """
    rng = np.random.default_rng(random_state)
    null_mean_r = np.zeros(n_permutations, dtype=np.float64)

    for i in tqdm(
        range(n_permutations), desc="        Permutations", leave=False
    ):
        # Shuffle embeddings across stimuli, breaking the X <-> Y mapping
        perm_idx = rng.permutation(X.shape[0])
        X_perm = X[perm_idx]

        perm_res = run_encoding(
            X_perm,
            Y,
            frac_grid=frac_grid,
            groups=groups,
            test_size=test_size,
            n_inner_folds=n_inner_folds,
            n_outer_folds=n_outer_folds,
            noise_ceiling=noise_ceiling,
            random_state=random_state,  # same split for fair comparison
            verbose=False,
        )
        null_mean_r[i] = perm_res["mean_r"]

    # p-values (Phipson & Smyth, 2010)
    real_mean = real_result["mean_r"]
    p_mean = float(
        (np.sum(null_mean_r >= real_mean) + 1) / (n_permutations + 1)
    )

    logger.info(
        f"      Null mean r: {null_mean_r.mean():.4f} ± "
        f"{null_mean_r.std():.4f}  "
        f"(real = {real_mean:.4f}, p = {p_mean:.4f})"
    )

    return {
        "null_mean_r": null_mean_r,
        "p_value_mean_r": p_mean,
    }


# ---------------------------------------------------------------------------
# Hydra entry-point
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="../../configs/modeling",
    config_name="neural_encoding",
)
def main(cfg: DictConfig) -> None:
    """Run neural encoding analysis across subjects and conditions."""

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
    design_matrix_mapping_file = Path(cfg.get("design_matrix_mapping_file", ""))
    frac_grid: np.ndarray = np.asarray(
        cfg.get("frac_grid", np.arange(0.1, 1.1, 0.1)),
        dtype="float32",
    )
    n_permutations: int = cfg.get("n_permutations", 0)
    sessions: list[int] = list(cfg.sessions)
    runs_per_session: int = cfg.runs_per_session
    conditions: dict = OmegaConf.to_container(cfg.conditions, resolve=True)  # type: ignore[assignment]

    # -- load & PCA embeddings (shared across subjects) ----------------------
    logger.info("Loading VLM embeddings …")
    embed_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for cond_name, cond_cfg in conditions.items():
        emod = cond_cfg["embed_modality"]
        if emod in embed_data:
            continue
        lyr = layer_for_modality[emod]
        coco_ids, raw_embs = load_embeddings(
            embeddings_dir, model_label, emod, lyr
        )
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
    nc_sanity_rows: list[dict] = []
    previous_nc_by_modality: dict[str, np.ndarray] = {}

    if not design_matrix_mapping_file.exists():
        raise FileNotFoundError(
            "design_matrix_mapping_file is required for noise-ceiling "
            "computation consistency with visualization. "
            f"Missing: {design_matrix_mapping_file}"
        )
    condition_to_coco = load_condition_to_cocoid_modality(
        design_matrix_mapping_file
    )
    modality_map = load_design_matrix_mapping(design_matrix_mapping_file)

    for subject in tqdm(subjects, desc="Subjects"):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Subject: {subject} (subject-native T1w space)")

        # Load fMRI (full voxel set)
        betas_full = load_fmri(glmsingle_root, subject)

        # Load brain mask from fMRIPrep (subject-native T1w space)
        fmriprep_dir = Path(cfg.get("fmriprep_dir", ""))
        brain_mask = load_brain_mask(fmriprep_dir, subject)

        # Keep only in-brain voxels for encoding regression.
        n_total_voxels = betas_full.shape[0]
        betas = betas_full[brain_mask, :]
        logger.info(
            f"  Applied brain mask: kept {betas.shape[0]}/{n_total_voxels} "
            f"in-brain voxels"
        )

        # Match visualization normalization: normalize full betas per run
        # using DESIGNINFO run structure.
        stimulus_ids, num_runs = load_designinfo_stimulus_ids_and_num_runs(
            glmsingle_root, subject, n_trials=betas_full.shape[1]
        )
        betas_full_trials = normalize_betas_per_run(
            betas_full.T, num_runs=num_runs
        )  # (n_trials, n_voxels)
        betas = betas_full_trials.T[brain_mask, :]  # (n_masked_voxels, n_trials)

        # Match visualization NC: compute by modality from DESIGNINFO IDs +
        # design_matrix_mapping, then convert % NC to correlation scale.
        nc_by_modality_pct = compute_nc_by_modality(
            betas_full_trials,
            stimulus_ids,
            modality_map,
        )
        nc_corr_by_modality_full = {
            m: np.sqrt(np.clip(v, 0, None) / 100.0)
            for m, v in nc_by_modality_pct.items()
        }

        for m, nc_full in nc_corr_by_modality_full.items():
            nc_sanity_rows.append(
                {
                    "subject": subject,
                    "modality": m,
                    "mean_nc_corr": float(np.nanmean(nc_full)),
                    "std_nc_corr": float(np.nanstd(nc_full)),
                    "max_nc_corr": float(np.nanmax(nc_full)),
                }
            )
            if m in previous_nc_by_modality:
                mad = _mean_abs_diff(nc_full, previous_nc_by_modality[m])
                if nc_full.shape == previous_nc_by_modality[m].shape and np.allclose(
                    nc_full,
                    previous_nc_by_modality[m],
                    equal_nan=True,
                ):
                    logger.warning(
                        f"  Sanity check ({m}): NC is IDENTICAL to previous "
                        f"subject (unexpected)."
                    )
                else:
                    logger.info(
                        f"  Sanity check ({m}): differs from previous subject "
                        f"(mean |ΔNC| = {mad:.5f})."
                    )
            previous_nc_by_modality[m] = nc_full.copy()

        # Build beta_index -> (COCO ID, modality) mapping directly from
        # DESIGNINFO stimorder (GLMsingle beta order).
        events_df = build_events_from_stimorder(
            stimulus_ids,
            condition_to_coco,
            subject,
        )
        if not events_df.empty:
            max_beta_index = int(events_df["beta_index"].max())
            if max_beta_index >= betas.shape[1]:
                raise ValueError(
                    f"{subject}: events beta_index out of range "
                    f"(max={max_beta_index}, betas trials={betas.shape[1]}). "
                    "Check event filtering / GLMsingle trial definition."
                )

        # Prepare per-modality single-trial fMRI data + modality-specific NC.
        fmri_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
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

            # NC arrays from full voxel space -> in-brain masked voxel space
            nc_ceiling = nc_corr_by_modality_full[fmri_mod][brain_mask]

            # Optional: keep only top X% voxels by modality-specific noise ceiling
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
                voxel_keep = slice(None)

            trial_betas = trial_betas[:, voxel_keep]

            fmri_cache[fmri_mod] = (trial_coco_ids, trial_betas, nc_ceiling)
            logger.info(
                f"  fMRI {fmri_mod}: {len(trial_coco_ids)} trials "
                f"({len(np.unique(trial_coco_ids))} stimuli, "
                f" NC) × {trial_betas.shape[1]} voxels"
            )

        # Run each encoding condition
        for cond_name, cond_cfg in tqdm(
            conditions.items(), desc="    Conditions", leave=False
        ):
            emod = cond_cfg["embed_modality"]
            fmod = cond_cfg["fmri_modality"]
            logger.info(f"  Condition: {cond_name} ({emod} embed → {fmod} fMRI)")

            embed_ids, embed_feats = embed_data[emod]
            trial_cids, trial_betas, noise_ceiling_r = fmri_cache[fmod]

            X, Y, groups = align_single_trials(
                embed_ids, embed_feats, trial_cids, trial_betas
            )
            n_unique = len(np.unique(groups))
            logger.info(
                f"    Aligned: {X.shape[0]} trials ({n_unique} stimuli), "
                f"{X.shape[1]} features, {Y.shape[1]} voxels"
            )

            if n_outer_folds > 1:
                min_required = max(n_outer_folds, n_inner_folds + 1)
                requirement = (
                    f"n_outer_folds={n_outer_folds} outer CV "
                    f"+ {n_inner_folds}-fold inner CV"
                )
            else:
                min_required = max(n_inner_folds + 1, int(1 / test_size) + 1)
                requirement = (
                    f"test_size={test_size} holdout "
                    f"+ {n_inner_folds}-fold inner CV"
                )
            if n_unique < min_required:
                logger.warning(
                    f"    Too few unique stimuli ({n_unique}) for "
                    f"{requirement} – skipping."
                )
                continue

            result = run_encoding(
                X, Y, frac_grid=frac_grid, groups=groups,
                test_size=test_size,
                n_inner_folds=n_inner_folds,
                n_outer_folds=n_outer_folds,
                noise_ceiling=noise_ceiling_r,
                average_test_by_group=True,
            )
            logger.info(
                f"    mean r = {result['mean_r']:.4f}, "
                f"mean best frac = {result['best_frac']:.3f}, "
                f"outer folds = {result.get('n_outer_folds', 1)}"
            )
            if "mean_normalized_r" in result:
                logger.info(
                    f"    noise-ceiling-corrected: "
                    f"mean r/NC = {result['mean_normalized_r']:.4f}, "
                    f"mean NC = {result['mean_noise_ceiling_r']:.4f}"
                )

            # -- Permutation test -------------------------------------------
            perm_result: dict | None = None
            if n_permutations > 0:
                logger.info(
                    f"    Permutation test ({n_permutations} shuffles) …"
                )
                perm_result = run_permutation_test(
                    X,
                    Y,
                    frac_grid=frac_grid,
                    groups=groups,
                    test_size=test_size,
                    n_inner_folds=n_inner_folds,
                    n_outer_folds=n_outer_folds,
                    noise_ceiling=noise_ceiling_r,
                    n_permutations=n_permutations,
                    real_result=result,
                )

            # Save per-voxel correlation map
            cond_dir = output_dir / model_label / subject / cond_name
            cond_dir.mkdir(parents=True, exist_ok=True)
            np.save(cond_dir / "per_voxel_r.npy", result["per_voxel_r"])
            np.save(cond_dir / "noise_ceiling.npy", noise_ceiling_r)
            np.save(
                cond_dir / "best_frac_per_voxel.npy",
                result["best_frac_per_voxel"],
            )
            if perm_result is not None:
                np.save(
                    cond_dir / "null_mean_r.npy",
                    perm_result["null_mean_r"],
                )

            summary_rows.append(
                {
                    "subject": subject,
                    "condition": cond_name,
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
                    "mean_normalized_r": result.get(
                        "mean_normalized_r", np.nan
                    ),
                    "p_value_mean_r": (
                        perm_result["p_value_mean_r"]
                        if perm_result is not None
                        else np.nan
                    ),
                }
            )

    # -- aggregate across subjects -------------------------------------------
    logger.info(f"\n{'=' * 60}")
    logger.info("Aggregating results across subjects …")

    summary_df = pd.DataFrame(summary_rows)
    results_dir = output_dir / model_label
    results_dir.mkdir(parents=True, exist_ok=True)

    if nc_sanity_rows:
        nc_sanity_df = pd.DataFrame(nc_sanity_rows)
        nc_sanity_csv = results_dir / "noise_ceiling_subject_stats.csv"
        nc_sanity_df.to_csv(nc_sanity_csv, index=False)
        logger.info(f"Noise-ceiling sanity stats → {nc_sanity_csv}")

        for modality in sorted(nc_sanity_df["modality"].unique()):
            vals_arr = np.asarray(
                nc_sanity_df.loc[
                    nc_sanity_df["modality"] == modality, "mean_nc_corr"
                ],
                dtype=float,
            )
            vals_arr = np.round(vals_arr[np.isfinite(vals_arr)], 6)

    agg_cols = [
        "mean_r",
        "mean_noise_ceiling_r",
        "max_noise_ceiling_r",
        "mean_normalized_r",
    ]
    if "p_value_mean_r" in summary_df.columns:
        agg_cols += ["p_value_mean_r"]
    agg = (
        summary_df.groupby("condition")[agg_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    logger.info(f"\n{agg.to_string()}")

    # Save
    summary_path = results_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Per-subject summary → {summary_path}")

    agg_path = results_dir / "aggregated.csv"
    agg.to_csv(agg_path)
    logger.info(f"Aggregated results → {agg_path}")

    logger.success("Neural encoding analysis complete!")


if __name__ == "__main__":
    main()
