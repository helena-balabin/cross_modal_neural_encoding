"""Visualize residual neural encoding results across all models.

Run this once after the residual encoding array has produced
``outputs/residual_encoding/<model>/{summary,aggregated}.csv`` for every model.
It produces two combined figures (all models in one figure each):

    - ``residual_encoding_bars`` — residualized encoding accuracy as grouped bars
      (one bar per model per condition), the *same* model-comparison figure the
      main neural-encoding pipeline produces (reuses ``plot_grouped_model_means``).
    - ``ablation_delta`` — grouped bars of the residualization effect expressed as
      a *percentage* of the original (standard) encoding performance, per condition
      for every model, so a drop after residualization is a downward (negative) bar.
      The expected signature is a larger negative % drop for the cross-modal
      condition than for the matching within-modality condition.

Usage::

    python -m cross_modal_neural_encoding.visualization.visualize_residual_encoding \\
        residual_root=outputs/residual_encoding \\
        encoding_root=outputs/neural_encoding \\
        output_dir=reports/figures/residual_encoding
"""

from __future__ import annotations

from pathlib import Path

import hydra
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import pandas as pd

from cross_modal_neural_encoding.config import FIGURES_DIR, PROJ_ROOT
from cross_modal_neural_encoding.utils import (
    CONDITION_LABELS,
    benjamini_hochberg,
    configure_plot_fonts,
    short_model_label,
    signflip_pvalue_two_sided,
    significance_label,
)
from cross_modal_neural_encoding.visualization.visualize_encoding_results import (
    VLM_MODEL_PALETTE,
    _collect_model_dirs,
    _model_category_rank,
    _legend_ncol,
    annotate_pairwise_brackets,
    load_aggregated,
    load_summary,
    pairwise_condition_signrank,
    plot_grouped_model_means,
)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

configure_plot_fonts()

METRIC = "mean_normalized_r"

# Delta plot condition order: group by embedding modality, within-then-cross,
# so each pair of bars is the within-vs-cross contrast for one embedding.
DELTA_CONDITION_ORDER = ["text_to_text", "text_to_image", "image_to_image", "image_to_text"]

# Title prefix per residualization side (American spelling).
RESIDUAL_SIDE_TITLES = {
    "embedding": "Embedding-based residualization",
    "fmri": "fMRI-based residualization",
}


def _residual_condition_labels(residual_side: str) -> dict[str, str]:
    """Condition x-tick labels with the word 'residual' on the residualized side.

    Embedding-based residualization qualifies the embedding (``"Residual image
    embeddings → Image fMRI"``); fMRI-based residualization qualifies the fMRI
    (``"Image embeddings → Residual image fMRI"``).
    """
    mods = {"image": "Image", "text": "Text"}
    labels: dict[str, str] = {}
    for emb in ("image", "text"):
        for fmri in ("image", "text"):
            if residual_side == "fmri":
                emb_part = f"{mods[emb]} embeddings"
                fmri_part = f"Residual {fmri} fMRI"
            else:  # embedding-based
                emb_part = f"Residual {emb} embeddings"
                fmri_part = f"{mods[fmri]} fMRI"
            labels[f"{emb}_to_{fmri}"] = f"{emb_part}\n→ {fmri_part}"
    return labels


# ═══════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════


def _find_standard_summary(encoding_root: Path, model_label: str) -> Path | None:
    """Locate the standard neural-encoding summary.csv for a model."""
    for pattern in (f"*/{model_label}/summary.csv", f"*/*/{model_label}/summary.csv"):
        matches = sorted(encoding_root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _drop_permuted_aggregated(df: pd.DataFrame) -> pd.DataFrame:
    """Drop permuted-control rows (index like ``permuted_text_to_image``)."""
    return df[~df.index.astype(str).str.startswith("permuted")]


def _drop_permuted_summary(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Drop permuted-control rows from a per-subject summary frame."""
    if df is None:
        return None
    return df[~df["condition"].astype(str).str.startswith("permuted")].reset_index(drop=True)


def _paired_pct_change(
    standard_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    condition: str,
    metric: str,
) -> tuple[float, float]:
    """Mean and SEM of the residualization effect as a % of original performance.

    The per-subject paired change (residualized − standard) is expressed as a
    percentage of the *group-mean* original (standard) performance for the
    condition, so a drop after residualization is a negative percentage.  The
    group mean is used as the denominator (rather than each subject's own value)
    because the noise-ceiling-normalised metric is near zero for the cross-modal
    conditions, where per-subject ratios would explode.  Conditions whose
    original performance is not clearly positive are returned as NaN, since a
    percent decrease relative to a ~zero baseline is not interpretable.
    """
    std_vals = standard_df.loc[standard_df["condition"] == condition, metric].to_numpy(dtype=float)
    res_vals = residual_df.loc[residual_df["condition"] == condition, metric].to_numpy(dtype=float)
    n = min(len(std_vals), len(res_vals))
    if n == 0:
        return np.nan, 0.0
    std_vals = std_vals[:n]
    res_vals = res_vals[:n]
    d = res_vals - std_vals
    finite = np.isfinite(d) & np.isfinite(std_vals)
    d = d[finite]
    base = std_vals[finite]
    if not len(d):
        return np.nan, 0.0
    denom = float(np.mean(base))
    if not np.isfinite(denom) or denom <= 1e-3:
        return np.nan, 0.0
    pct = d / denom * 100.0
    mean = float(np.mean(pct))
    sem = float(np.std(pct) / np.sqrt(len(pct))) if len(pct) > 1 else 0.0
    return mean, sem


def _paired_diffs(
    standard_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    condition: str,
    metric: str,
) -> np.ndarray:
    """Per-subject paired differences (residualized − standard) for a condition.

    Used to test whether a bar's change differs from zero. Subjects are paired by
    row order (the same convention as ``_paired_pct_change``).
    """
    std_vals = standard_df.loc[standard_df["condition"] == condition, metric].to_numpy(dtype=float)
    res_vals = residual_df.loc[residual_df["condition"] == condition, metric].to_numpy(dtype=float)
    n = min(len(std_vals), len(res_vals))
    if n == 0:
        return np.array([])
    d = res_vals[:n] - std_vals[:n]
    return d[np.isfinite(d)]


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════


def plot_combined_delta(
    model_results: list[dict],
    *,
    metric: str = METRIC,
    output_path: Path,
    font_scale: float = 1.3,
    title: str = "Residualization effect across models",
    condition_labels: dict[str, str] | None = None,
) -> None:
    """One figure: grouped bars of the % change (residualized vs. standard) per condition.

    x-axis = conditions (within/cross grouped per embedding modality); bars within
    each condition = models. Each bar is the residualization effect expressed as a
    percentage of the original encoding performance, so a drop after residualization
    is a downward (negative) bar.
    """
    items = [
        it
        for it in model_results
        if it.get("summary_df") is not None and it.get("standard_summary_df") is not None
    ]
    if not items:
        logger.warning("No models with both residual and standard summaries; skipping delta plot.")
        return
    items = sorted(items, key=lambda it: _model_category_rank(it["model_label"]))

    present: set[str] = set()
    for it in items:
        present |= set(it["summary_df"]["condition"].unique()) & set(
            it["standard_summary_df"]["condition"].unique()
        )
    conditions = [c for c in DELTA_CONDITION_ORDER if c in present]
    conditions += [c for c in CONDITION_LABELS if c in present and c not in conditions]

    n_models = len(items)
    n_conditions = len(conditions)
    values = np.full((n_models, n_conditions), np.nan)
    sems = np.full((n_models, n_conditions), np.nan)
    pvals = np.full((n_models, n_conditions), np.nan)
    for i, it in enumerate(items):
        for j, cond in enumerate(conditions):
            mean, sem = _paired_pct_change(it["standard_summary_df"], it["summary_df"], cond, metric)
            values[i, j] = mean
            sems[i, j] = sem
            d = _paired_diffs(it["standard_summary_df"], it["summary_df"], cond, metric)
            if len(d) >= 2:
                pvals[i, j] = signflip_pvalue_two_sided(d)
    # Correct the per-bar "different from zero" tests across the whole figure.
    pvals = benjamini_hochberg(pvals)

    model_labels = [it["model_label"] for it in items]
    fig_w = max(8.0, 3.0 + 0.22 * n_models * max(n_conditions, 1))
    # Taller than the bars to fit one pairwise-bracket level per condition pair.
    fig, ax = plt.subplots(figsize=(fig_w, 6.4))

    x = np.arange(n_conditions)
    total_width = 0.82
    bar_w = min(0.16, total_width / max(n_models, 1))
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_w
    colors = [VLM_MODEL_PALETTE[i % len(VLM_MODEL_PALETTE)] for i in range(n_models)]

    for i, label in enumerate(model_labels):
        ax.bar(
            x + offsets[i],
            values[i],
            width=bar_w * 0.95,
            yerr=sems[i],
            color=colors[i],
            edgecolor="#4A4A4A",
            linewidth=0.5,
            alpha=0.9,
            error_kw={"linewidth": 0.7},
            capsize=2,
            label=short_model_label(label),
            zorder=3,
        )

    # Per-bar test: is each bar's % change different from zero? (two-sided
    # sign-flip over the 8 subjects, BH-corrected above). Only significant bars
    # are starred — with this many bars, an "ns" on every one would be noise.
    span = max(float(np.nanmax(values + np.nan_to_num(sems)) - np.nanmin(values - np.nan_to_num(sems))), 1e-6)
    star_off = 0.012 * span
    lowest = float(np.nanmin(values - np.nan_to_num(sems)))
    for i in range(n_models):
        for j in range(n_conditions):
            val = values[i, j]
            if not np.isfinite(val) or not np.isfinite(pvals[i, j]):
                continue
            sig = significance_label(float(pvals[i, j]))
            if sig in ("", "ns"):
                continue
            sem = sems[i, j] if np.isfinite(sems[i, j]) else 0.0
            if val >= 0:
                y, va = val + sem + star_off, "bottom"
            else:
                y, va = val - sem - star_off, "top"
                lowest = min(lowest, y - star_off)
            ax.text(
                x[j] + offsets[i], y, sig, ha="center", va=va,
                fontsize=6.5 * font_scale, color="darkred", zorder=5,
            )
    ax.set_ylim(bottom=min(ax.get_ylim()[0], lowest))

    label_map = condition_labels if condition_labels is not None else CONDITION_LABELS
    ax.axhline(0, color="black", linewidth=0.6, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(c, c) for c in conditions], fontsize=10 * font_scale)
    ax.set_ylabel(
        "Change in encoding performance (%)\n(residualized vs. standard)", fontsize=11 * font_scale
    )
    ax.set_title(title, fontsize=14 * font_scale, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Pairwise Wilcoxon signed-rank tests between the condition groups (per-model
    # % change, paired by model), drawn as nested significance brackets above 0.
    pair_q = pairwise_condition_signrank(values)
    annotate_pairwise_brackets(
        ax,
        x_positions=x,
        values=values,
        pair_qvalues=pair_q,
        alpha=0.05,
        font_scale=font_scale,
    )

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=_legend_ncol(
            [short_model_label(m) for m in model_labels], fig_w, 8 * font_scale, 5
        ),
        fontsize=8 * font_scale,
        frameon=False,
    )
    note = ax.text(
        0.01,
        0.02,
        "Stars on bars: change ≠ 0 (sign-flip over subjects)   ·   "
        "Brackets: pairwise signed-rank between conditions   ·   BH-FDR corrected",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8 * font_scale,
        color="#333333",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", bbox_extra_artists=(leg, note))
    plt.close(fig)
    logger.success(f"Combined ablation delta plot → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


@hydra.main(
    version_base=None,
    config_path="../../configs/visualization",
    config_name="visualize_residual_encoding",
)
def main(cfg: DictConfig) -> None:
    residual_root = Path(cfg.get("residual_root", "outputs/residual_encoding"))
    if not residual_root.is_absolute():
        residual_root = PROJ_ROOT / residual_root
    encoding_root = Path(cfg.get("encoding_root", "outputs/neural_encoding"))
    if not encoding_root.is_absolute():
        encoding_root = PROJ_ROOT / encoding_root
    output_dir = Path(cfg.get("output_dir", str(FIGURES_DIR / "residual_encoding")))
    if not output_dir.is_absolute():
        output_dir = PROJ_ROOT / output_dir
    metric: str = cfg.get("metric", METRIC)

    # Which side was residualized — drives the figure titles and the residual
    # x-tick labels. Inferred from the residual_root name when not set explicitly.
    residual_side = str(cfg.get("residual_side", "")).lower()
    if residual_side not in {"embedding", "fmri"}:
        residual_side = "fmri" if "fmri" in residual_root.name.lower() else "embedding"
    title_prefix = RESIDUAL_SIDE_TITLES[residual_side]
    residual_labels = _residual_condition_labels(residual_side)
    logger.info(f"Residualization side: {residual_side}")

    if not residual_root.exists():
        raise FileNotFoundError(f"residual_root does not exist: {residual_root}")

    # Confirm the active font actually resolves to Lato (matplotlib falls back
    # silently otherwise).
    from matplotlib.font_manager import FontProperties, findfont

    lato_path = findfont(FontProperties(family="Lato"))
    logger.info(f"'Lato' resolves to: {lato_path}")
    if "lato" not in lato_path.lower():
        logger.warning("Lato not resolved — matplotlib is falling back to another font.")

    entries = _collect_model_dirs(residual_root)
    if not entries:
        raise FileNotFoundError(
            f"No residual model folders with aggregated.csv found under {residual_root}"
        )

    model_results: list[dict] = []
    for entry in entries:
        model_dir = entry["path"]
        model_label = entry["model_label"]
        summary_path = model_dir / "summary.csv"
        std_path = _find_standard_summary(encoding_root, model_label)
        if std_path is None:
            logger.warning(
                f"No standard summary for {model_label} under {encoding_root}; "
                "it will appear in the bars figure but not the delta plot."
            )
        # Drop the permuted-control conditions; they are a baseline, not their own
        # bars in the accuracy / delta figures.
        model_results.append(
            {
                "aggregated_df": _drop_permuted_aggregated(load_aggregated(model_dir / "aggregated.csv")),
                "summary_df": _drop_permuted_summary(
                    load_summary(summary_path) if summary_path.exists() else None
                ),
                "standard_summary_df": load_summary(std_path) if std_path else None,
                "model_label": model_label,
                "model_dir": None,  # no permutation nulls → sign-flip significance
            }
        )

    logger.info(f"Loaded {len(model_results)} residual model(s) from {residual_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Grouped model-comparison bars (one bar per model per condition) — the
    #    same figure the main neural-encoding pipeline uses to compare models.
    #    Match the width of the delta plot below. The delta uses tight_layout
    #    (axes fills ~0.88 of the figure) while the grouped bars do not (~0.78),
    #    so widen the bars figure by ~1.13× the delta's width to land at the same
    #    rendered width.
    delta_w = max(8.0, 3.0 + 0.22 * len(model_results) * 4)
    plot_grouped_model_means(
        model_results,
        metric=metric,
        alpha=0.05,
        font_scale=1.3,
        compress_normalized_axis=False,
        normalized_axis_linthresh=0.08,
        output_path=output_dir / "residual_encoding_bars.png",
        figsize=(1.13 * delta_w, 6.4),
        y_limits=None,
        title=f"{title_prefix}\nResidual encoding accuracy (group means)",
        condition_labels=residual_labels,
    )

    # 2) One combined delta figure across all models.
    plot_combined_delta(
        model_results,
        metric=metric,
        output_path=output_dir / "ablation_delta.png",
        title=f"{title_prefix}\nResidualization effect across models",
        condition_labels=residual_labels,
    )


if __name__ == "__main__":
    main()
