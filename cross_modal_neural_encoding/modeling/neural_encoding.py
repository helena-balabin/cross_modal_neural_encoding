"""Neural encoding: predict fMRI responses from VLM embeddings.

Fits ridge-regression encoding models to predict single-trial fMRI
responses (GLMsingle Type-D betas) from Vision-Language Model embeddings
within a Type-A R² voxel mask.

Four encoding conditions (configurable):

  1. **text_embed  → text_fmri**   (within-modality)
  2. **image_embed → image_fmri**  (within-modality)
  3. **image_embed → text_fmri**   (cross-modal)
  4. **text_embed  → image_fmri**  (cross-modal)

Workflow
-------
1. Load VLM embeddings (vision + text) and PCA-reduce them.
2. For each subject:

    a. Load GLMsingle Type-D betas (denoised single-trial fMRI).
    b. Z-score betas within each run (across trials, per voxel)
        before concatenating, to remove run-level mean/scale
        differences per voxel.
    c. Compute per-voxel **explainable variance** (EV) from repeated
        presentations, following the VEM framework (Dupré la Tour
        et al., 2025; Sahani & Linden, 2002).  Mask voxels by the
        top-k % of positive-EV voxels (e.g. top 20 %).
    d. Parse BIDS events files to map trial indices → COCO IDs + modality.
    e. For each encoding condition, average masked betas across
        repetitions of each stimulus, then align the averaged
        responses with embeddings by COCO ID (one sample per
        stimulus).
    f. Split stimuli into train and test sets
        (``GroupShuffleSplit``).  Fit ridge regression on the train
        set with per-voxel alpha selection via inner CV
        (``GroupKFold``), following Gallant lab best practices.
        Evaluate with per-voxel Pearson *r* on the held-out test set.
    g. Normalise *r* by the noise ceiling (√EV per voxel) to obtain
        the fraction of explainable signal captured by the model.

3. Aggregate results across subjects.

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
from himalaya.ridge import RidgeCV
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from cross_modal_neural_encoding.config import PROJ_ROOT


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

                # Skip blanks and invalid entries
                if mod in ("blank", "nan", "n/a", ""):
                    continue
                if pd.isna(cid) or str(cid).strip().lower() == "n/a":
                    continue

                records.append(
                    {
                        "beta_index": beta_idx,
                        "cocoid": int(float(cid)),
                        "modality": mod,
                        "run_label": run_counter,
                    }
                )
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


def load_fmri(
    glmsingle_root: Path,
    subject: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load Type-D betas and Type-A on-off R² for *subject*.

    Returns
    -------
    betas : (n_voxels, n_trials) – flattened denoised betas.
    r2_map : (n_voxels,) – flattened Type-A R² map.
    """
    sub_dir = glmsingle_root / subject

    typed = np.load(
        sub_dir / "TYPED_FITHRF_GLMDENOISE_RR.npy", allow_pickle=True
    ).item()
    betas_4d: np.ndarray = typed["betasmd"]  # (X, Y, Z, n_trials)
    betas = betas_4d.reshape(-1, betas_4d.shape[-1])  # (n_vox, n_trials)

    typea = np.load(
        sub_dir / "TYPEA_ONOFF.npy", allow_pickle=True
    ).item()
    r2_3d: np.ndarray = typea["onoffR2"]  # (X, Y, Z)
    r2_map = r2_3d.reshape(-1)

    logger.info(
        f"  {subject}: betas {betas_4d.shape} → "
        f"({betas.shape[0]}, {betas.shape[1]}), "
        f"R² ∈ [{r2_map.min():.3f}, {r2_map.max():.3f}]"
    )
    return betas, r2_map


def zscore_betas_per_run(
    betas: np.ndarray,
    events_df: pd.DataFrame,
) -> np.ndarray:
    """Z-score betas within each run (across trials, per voxel).

    For each run, every voxel's response is independently centred and
    scaled across the trials in that run.  This removes each voxel's
    run-specific mean and variance, preventing run-level offsets from
    inflating between-run variance.

    Parameters
    ----------
    betas : (n_voxels, n_trials)
    events_df : must contain ``beta_index`` and ``run_label`` columns.

    Returns
    -------
    betas_z : (n_voxels, n_trials) – z-scored copy.
    """
    betas_z = betas.copy().astype(np.float64)

    for run_label, group in events_df.groupby("run_label"):
        idx = np.asarray(group["beta_index"].values, dtype=int)
        run_data = betas_z[:, idx]  # (n_voxels, n_trials_in_run)
        mu = run_data.mean(axis=1, keepdims=True)
        sd = run_data.std(axis=1, keepdims=True, ddof=0)
        sd[sd == 0] = 1.0  # avoid division by zero for constant voxels
        betas_z[:, idx] = (run_data - mu) / sd

    n_runs = events_df["run_label"].nunique()
    logger.info(f"  Z-scored betas within {n_runs} runs")
    return betas_z


def create_mask(r2_map: np.ndarray, percentile: float) -> np.ndarray:
    """Boolean mask selecting the top voxels by Type-A R².

    Parameters
    ----------
    r2_map : 1-D R² values.
    percentile : Threshold percentile among R² > 0 voxels
                (e.g. 85 → top 15 %).
    """
    positive = r2_map > 0
    if positive.sum() == 0:
        raise ValueError("No voxels with R² > 0 – check GLMsingle outputs.")
    threshold = np.percentile(r2_map[positive], percentile)
    mask = r2_map >= threshold
    logger.info(
        f"  Mask: {mask.sum()} voxels "
        f"(top {100 - percentile:.0f}% of {positive.sum()} responsive, "
        f"R² ≥ {threshold:.4f})"
    )
    return mask


def compute_explainable_variance(
    betas: np.ndarray,
    events_df: pd.DataFrame,
    modality_filter: str,
) -> np.ndarray:
    """Compute per-voxel explainable variance from repeated presentations.

    Follows the VEM framework (Dupré la Tour et al., 2025; Sahani &
    Linden, 2002; Hsu et al., 2004).  Explainable variance (EV)
    quantifies the fraction of variance across stimuli that is
    consistent across repetitions.  It serves as the noise ceiling:
    max achievable R² (for Pearson *r*, the ceiling is √EV).

    Betas are expected to have been z-scored per run (across trials,
    per voxel) before calling this function.  No additional z-scoring
    is applied here so that the variance structure is preserved for
    the VEM ratio.

    Parameters
    ----------
    betas : (n_voxels, n_all_trials) – per-run z-scored single-trial betas.
    events_df : table mapping beta_index → cocoid × modality.
    modality_filter : ``"image"`` or ``"text"``.

    Returns
    -------
    ev : (n_voxels,) – explainable variance per voxel.  Values ≤ 0
        indicate voxels without consistent stimulus-driven signal.
    """
    mod_df = events_df[events_df["modality"] == modality_filter]
    grouped = mod_df.groupby("cocoid")

    # Determine the modal number of repetitions
    group_sizes = grouped.size()
    n_repeats = int(group_sizes.mode().iloc[0])

    # Warn if many stimuli don't match the modal repeat count
    n_excluded = int((group_sizes != n_repeats).sum())
    if n_excluded > 0:
        logger.warning(
            f"  EV ({modality_filter}): dropping {n_excluded} stimuli "
            f"with != {n_repeats} repeats (possibly missing runs)"
        )
    if n_repeats < 2:
        raise ValueError(
            f"Need ≥ 2 repeats to compute EV, got {n_repeats} "
            f"for modality '{modality_filter}'. Check events files."
        )

    # Keep only stimuli with exactly n_repeats presentations
    valid_cids = group_sizes[group_sizes == n_repeats].index
    valid_df = mod_df[mod_df["cocoid"].isin(valid_cids)]
    valid_grouped = valid_df.groupby("cocoid")

    n_stimuli = len(valid_grouped)
    n_voxels = betas.shape[0]

    logger.info(
        f"  EV ({modality_filter}): {n_stimuli} stimuli × "
        f"{n_repeats} repeats, {n_voxels} voxels"
    )

    # Build (n_repeats, n_stimuli, n_voxels) array
    data = np.zeros((n_repeats, n_stimuli, n_voxels), dtype=np.float64)
    for i, (_, group) in enumerate(valid_grouped):
        indices = group["beta_index"].values.astype(int)
        data[:, i, :] = betas[:, indices].T  # type: ignore[index]

    # EV = var(mean across repeats) / mean(var across stimuli per repeat)
    mean_var = data.var(axis=1, dtype=np.float64, ddof=1).mean(axis=0)
    var_mean = data.mean(axis=0).var(axis=0, dtype=np.float64, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ev = np.where(mean_var > 0, var_mean / mean_var, 0.0)

    # Bias correction (Sahani & Linden, 2002)
    ev = ev - (1 - ev) / (n_repeats - 1)

    n_pos = int((ev > 0).sum())
    if n_pos > 0:
        logger.info(
            f"  EV ({modality_filter}): {n_pos} voxels with EV > 0, "
            f"median EV = {np.median(ev[ev > 0]):.4f}"
        )
    else:
        logger.warning(f"  EV ({modality_filter}): no voxels with EV > 0!")

    return ev


def create_ev_mask(
    ev: np.ndarray,
    top_percentage: float = 20.0,
) -> np.ndarray:
    """Boolean mask selecting the top-k % of positive-EV voxels.

    Parameters
    ----------
    ev : 1-D explainable variance per voxel.
    top_percentage : percentage of positive-EV voxels to keep
        (e.g. 20.0 → top 20 %).
    """
    positive = ev > 0
    n_positive = int(positive.sum())
    if n_positive == 0:
        raise ValueError("No voxels with EV > 0 — check data quality.")
    cutoff = np.percentile(ev[positive], 100 - top_percentage)
    mask = ev >= cutoff
    logger.info(
        f"  EV mask: {mask.sum()} voxels "
        f"(top {top_percentage:.0f}% of {n_positive} positive-EV voxels, "
        f"EV ≥ {cutoff:.4f})"
    )
    return mask


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


def run_encoding(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    alphas: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    n_inner_folds: int = 5,
    noise_ceiling: np.ndarray | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """Gallant-style ridge encoding: train/test split + inner CV.

    Following the Gallant lab best practices
    (``gallantlab/voxelwise_tutorials``):

    1. **Train / test split** – ``GroupShuffleSplit`` holds out a
       fraction of *stimuli* (all repetitions stay together).
    2. **Inner CV for alpha** – ``GroupKFold`` on the training stimuli
        is passed to himalaya's ``RidgeCV(cv=...)``, so the inner
        validation never sees the same stimulus as inner training.
        Per-target alpha selection (``local_alpha=True``, the solver
        default) finds the optimal regularisation per voxel.
    3. **Feature centering** – ``StandardScaler(with_mean=True,
        with_std=False)`` centres X (no variance-scaling).
    4. **Y centering** – Y is centred on the training-set mean per
        voxel and the same shift is applied to the test set.
    5. **Evaluation** – per-voxel Pearson *r* on the held-out test.

    Parameters
    ----------
    X : (n_samples, n_features) – PCA'd embeddings.
    Y : (n_samples, n_voxels) – fMRI responses.
    alphas : 1-D array of regularisation candidates.
    groups : (n_samples,) stimulus (COCO ID) per trial.
    test_size : fraction of *stimuli* held out for testing.
    n_inner_folds : number of inner CV folds for alpha selection.
    noise_ceiling : optional (n_voxels,) per-voxel EV.
    random_state : random seed for the train/test split.

    Returns
    -------
    dict with ``per_voxel_r``, ``mean_r``, ``median_r``,
    ``best_alpha``, and (if *noise_ceiling* given)
    ``normalized_per_voxel_r``, ``mean_normalized_r``,
    ``median_normalized_r``, ``mean_ev``.
    """
    X = X.astype("float32")
    Y = Y.astype("float32")

    # ---- 1. Train / test split by stimulus identity -----------------------
    gss = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(gss.split(X, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    groups_train = groups[train_idx]

    n_train_stimuli = len(np.unique(groups_train))
    n_test_stimuli = len(np.unique(groups[test_idx]))
    if verbose:
        logger.info(
            f"      Split: {len(train_idx)} train trials "
            f"({n_train_stimuli} stimuli) / "
            f"{len(test_idx)} test trials ({n_test_stimuli} stimuli)"
        )

    # ---- 2. Centre Y on training set --------------------------------------
    Y_mean = Y_train.mean(axis=0, keepdims=True)
    Y_train = Y_train - Y_mean
    Y_test = Y_test - Y_mean

    # ---- 3. Build group-aware inner CV splits for alpha selection ----------
    actual_inner = min(n_inner_folds, n_train_stimuli)
    inner_cv = GroupKFold(n_splits=actual_inner)
    inner_splits = list(inner_cv.split(X_train, groups=groups_train))

    # ---- 4. Fit pipeline --------------------------------------------------
    pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        RidgeCV(alphas=alphas, cv=inner_splits),  # type: ignore[call-arg]
    )
    pipeline.fit(X_train, Y_train)

    best_alphas = pipeline[-1].best_alphas_
    alpha_val = float(np.median(best_alphas))
    if verbose:
        logger.info(f"      Median best α = {alpha_val:.2g}")

    # ---- 5. Predict & evaluate on held-out test ---------------------------
    Y_pred = pipeline.predict(X_test)
    per_voxel_r = _pearson_r_columnwise(Y_test, Y_pred)  # type: ignore[assignment]

    result: dict = {
        "per_voxel_r": per_voxel_r,
        "mean_r": float(np.nanmean(per_voxel_r)),
        "median_r": float(np.nanmedian(per_voxel_r)),
        "best_alpha": alpha_val,
        "n_train_stimuli": n_train_stimuli,
        "n_test_stimuli": n_test_stimuli,
    }

    # ---- 6. Noise-ceiling normalisation (VEM framework) -------------------
    if noise_ceiling is not None:
        nc_r = np.sqrt(np.clip(noise_ceiling, 0, None))  # √EV
        valid = nc_r > 0
        normalized = np.full_like(per_voxel_r, np.nan)
        normalized[valid] = per_voxel_r[valid] / nc_r[valid]
        result["normalized_per_voxel_r"] = normalized
        result["mean_normalized_r"] = float(np.nanmean(normalized[valid]))
        result["median_normalized_r"] = float(
            np.nanmedian(normalized[valid])
        )
        result["mean_ev"] = float(np.mean(noise_ceiling[valid]))
        result["median_ev"] = float(np.median(noise_ceiling[valid]))
        result["max_ev"] = float(np.max(noise_ceiling[valid]))

    return result


def run_permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    alphas: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    n_inner_folds: int = 5,
    noise_ceiling: np.ndarray | None = None,
    n_permutations: int = 100,
    random_state: int = 42,
    real_result: dict,
) -> dict:
    """Permutation test for the encoding model.

    Shuffles the stimulus-to-embedding mapping (rows of *X*) to break
    the true correspondence between embeddings and fMRI responses,
    re-runs the full encoding pipeline, and builds a null distribution
    of mean / median *r*.

    The p-value is computed as ``(#{null ≥ real} + 1) / (n_perm + 1)``
    (Phipson & Smyth, 2010) to avoid *p* = 0 and to correct for the
    finite number of permutations.

    Parameters
    ----------
    X, Y, alphas, groups, test_size, n_inner_folds, noise_ceiling
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
    dict with keys ``null_mean_r``, ``null_median_r``,
    ``p_value_mean_r``, ``p_value_median_r``.
    """
    rng = np.random.default_rng(random_state)
    null_mean_r = np.zeros(n_permutations, dtype=np.float64)
    null_median_r = np.zeros(n_permutations, dtype=np.float64)

    for i in tqdm(
        range(n_permutations), desc="        Permutations", leave=False
    ):
        # Shuffle embeddings across stimuli, breaking the X <-> Y mapping
        perm_idx = rng.permutation(X.shape[0])
        X_perm = X[perm_idx]

        perm_res = run_encoding(
            X_perm,
            Y,
            alphas=alphas,
            groups=groups,
            test_size=test_size,
            n_inner_folds=n_inner_folds,
            noise_ceiling=noise_ceiling,
            random_state=random_state,  # same split for fair comparison
            verbose=False,
        )
        null_mean_r[i] = perm_res["mean_r"]
        null_median_r[i] = perm_res["median_r"]

    # p-values (Phipson & Smyth, 2010)
    real_mean = real_result["mean_r"]
    real_median = real_result["median_r"]
    p_mean = float(
        (np.sum(null_mean_r >= real_mean) + 1) / (n_permutations + 1)
    )
    p_median = float(
        (np.sum(null_median_r >= real_median) + 1) / (n_permutations + 1)
    )

    logger.info(
        f"      Null mean r: {null_mean_r.mean():.4f} ± "
        f"{null_mean_r.std():.4f}  "
        f"(real = {real_mean:.4f}, p = {p_mean:.4f})"
    )

    return {
        "null_mean_r": null_mean_r,
        "null_median_r": null_median_r,
        "p_value_mean_r": p_mean,
        "p_value_median_r": p_median,
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
    bids_root = Path(cfg.bids_root)
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
    ev_top_percentage: float = cfg.ev_top_percentage
    n_pca: int = cfg.n_pca_components
    n_inner_folds: int = cfg.n_inner_folds
    test_size: float = cfg.test_size
    # Gallant-lab standard: wide logarithmic alpha grid
    alphas: np.ndarray = np.logspace(
        cfg.alpha_log_min, cfg.alpha_log_max, cfg.n_alphas
    ).astype("float32")
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

    for subject in tqdm(subjects, desc="Subjects"):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Subject: {subject}")

        # Load fMRI
        betas, r2_map = load_fmri(glmsingle_root, subject)

        # Parse events
        events_df = load_events(
            bids_root,
            subject,
            sessions=sessions,
            runs_per_session=runs_per_session,
            task=cfg.task,
            modality_column=cfg.modality_column,
            cocoid_column=cfg.cocoid_column,
        )

        # Sanity-check trial counts
        n_betas = betas.shape[1]
        n_events = len(events_df)
        if n_events != n_betas:
            logger.warning(
                f"  Trial count mismatch: {n_events} events vs "
                f"{n_betas} betas. Using first {min(n_events, n_betas)}."
            )

        # Z-score betas within each run before any further analysis
        betas = zscore_betas_per_run(betas, events_df)

        # Compute per-modality explainable variance & create EV masks
        fmri_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for fmri_mod in {c["fmri_modality"] for c in conditions.values()}:
            ev = compute_explainable_variance(betas, events_df, fmri_mod)
            mask = create_ev_mask(ev, ev_top_percentage)

            stim_cids, avg_betas = average_betas_by_stimulus(
                betas, events_df, mask, fmri_mod
            )
            ev_masked = ev[mask]  # noise ceiling for masked voxels
            fmri_cache[fmri_mod] = (stim_cids, avg_betas, ev_masked)
            logger.info(
                f"  fMRI {fmri_mod}: {len(stim_cids)} stimuli "
                f"(averaged across reps) × {avg_betas.shape[1]} voxels"
            )

        # Run each encoding condition
        for cond_name, cond_cfg in tqdm(
            conditions.items(), desc="    Conditions", leave=False
        ):
            emod = cond_cfg["embed_modality"]
            fmod = cond_cfg["fmri_modality"]
            logger.info(f"  Condition: {cond_name} ({emod} embed → {fmod} fMRI)")

            embed_ids, embed_feats = embed_data[emod]
            stim_cids, avg_betas, ev_masked = fmri_cache[fmod]

            X, Y, groups = align_single_trials(
                embed_ids, embed_feats, stim_cids, avg_betas
            )
            n_unique = len(np.unique(groups))
            logger.info(
                f"    Aligned: {X.shape[0]} stimuli, "
                f"{X.shape[1]} features, {Y.shape[1]} voxels"
            )

            min_required = max(n_inner_folds + 1, int(1 / test_size) + 1)
            if n_unique < min_required:
                logger.warning(
                    f"    Too few unique stimuli ({n_unique}) for "
                    f"test_size={test_size} + {n_inner_folds}-fold "
                    f"inner CV – skipping."
                )
                continue

            result = run_encoding(
                X, Y, alphas=alphas, groups=groups,
                test_size=test_size,
                n_inner_folds=n_inner_folds,
                noise_ceiling=ev_masked,
            )
            logger.info(
                f"    mean r = {result['mean_r']:.4f}, "
                f"median r = {result['median_r']:.4f}, "
                f"best α = {result['best_alpha']:.2g}"
            )
            if "mean_normalized_r" in result:
                logger.info(
                    f"    noise-ceiling-corrected: "
                    f"mean r/√EV = {result['mean_normalized_r']:.4f}, "
                    f"median r/√EV = {result['median_normalized_r']:.4f}, "
                    f"mean EV = {result['mean_ev']:.4f}"
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
                    alphas=alphas,
                    groups=groups,
                    test_size=test_size,
                    n_inner_folds=n_inner_folds,
                    noise_ceiling=ev_masked,
                    n_permutations=n_permutations,
                    real_result=result,
                )

            # Save per-voxel correlation map
            cond_dir = output_dir / model_label / subject / cond_name
            cond_dir.mkdir(parents=True, exist_ok=True)
            np.save(cond_dir / "per_voxel_r.npy", result["per_voxel_r"])
            np.save(cond_dir / "noise_ceiling.npy", ev_masked)
            if perm_result is not None:
                np.save(
                    cond_dir / "null_mean_r.npy",
                    perm_result["null_mean_r"],
                )
                np.save(
                    cond_dir / "null_median_r.npy",
                    perm_result["null_median_r"],
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
                    "n_voxels": Y.shape[1],
                    "mean_r": result["mean_r"],
                    "median_r": result["median_r"],
                    "mean_ev": result.get("mean_ev", np.nan),
                    "median_ev": result.get("median_ev", np.nan),
                    "max_ev": result.get("max_ev", np.nan),
                    "mean_normalized_r": result.get(
                        "mean_normalized_r", np.nan
                    ),
                    "median_normalized_r": result.get(
                        "median_normalized_r", np.nan
                    ),
                    "p_value_mean_r": (
                        perm_result["p_value_mean_r"]
                        if perm_result is not None
                        else np.nan
                    ),
                    "p_value_median_r": (
                        perm_result["p_value_median_r"]
                        if perm_result is not None
                        else np.nan
                    ),
                }
            )

    # -- aggregate across subjects -------------------------------------------
    logger.info(f"\n{'=' * 60}")
    logger.info("Aggregating results across subjects …")

    summary_df = pd.DataFrame(summary_rows)
    agg_cols = ["mean_r", "median_r", "mean_ev", "median_ev", "max_ev", "mean_normalized_r", "median_normalized_r"]
    if "p_value_mean_r" in summary_df.columns:
        agg_cols += ["p_value_mean_r", "p_value_median_r"]
    agg = (
        summary_df.groupby("condition")[agg_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    logger.info(f"\n{agg.to_string()}")

    # Save
    results_dir = output_dir / model_label
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Per-subject summary → {summary_path}")

    agg_path = results_dir / "aggregated.csv"
    agg.to_csv(agg_path)
    logger.info(f"Aggregated results  → {agg_path}")

    logger.success("Neural encoding analysis complete!")


if __name__ == "__main__":
    main()
