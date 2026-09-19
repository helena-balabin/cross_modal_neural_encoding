"""Robustness of the cross-modal prediction asymmetry to the ridge alpha.

Compares the fixed-alpha ridge run (``regressor=ridge``, alpha=1.0) with the
nested-CV run (``regressor=ridge_cv``), where alpha is selected independently
for every direction, encoder pair and outer fold. Both runs share the outer CV
folds, so every (direction, input, output) cell pairs up one-to-one.

Outputs (in ``output_dir``):

1) ``alpha_heatmap_<direction>.png``: log10 of the median selected alpha per
   cell, to show whether the grid edges were reached.
2) ``asymmetry_fixed_vs_tuned.png``: per encoder pair, the direction gap
   Δ = r(Text → Vision) − r(Vision → Text) under fixed vs tuned alpha.
3) ``ridge_alpha_robustness_summary.csv``: per-direction means and the
   asymmetry statistics (also logged).

Usage
-----
    python -m cross_modal_neural_encoding.visualization.visualize_ridge_alpha_robustness

Hydra config: ``configs/visualization/visualize_ridge_alpha_robustness.yaml``
"""

from __future__ import annotations

from pathlib import Path

import hydra
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from cross_modal_neural_encoding.config import FIGURES_DIR, PROJ_ROOT
from cross_modal_neural_encoding.utils import configure_plot_fonts
from cross_modal_neural_encoding.visualization.visualize_predict_modalities import (
    _load_results_file,
    _make_colormap,
    _plot_heatmap,
    _pretty_label,
    _resolve_output_path,
)

configure_plot_fonts()

KEYS = ["direction", "input_model", "output_model"]


def _merge_runs(df_fixed: pd.DataFrame, df_tuned: pd.DataFrame) -> pd.DataFrame:
    """One row per (direction, input, output) with fixed and tuned mean r."""
    merged = df_fixed[KEYS + ["mean_r"]].merge(
        df_tuned[KEYS + ["mean_r", "alpha_median"]
                 + [c for c in df_tuned.columns if c.startswith("alpha_fold_")]],
        on=KEYS,
        suffixes=("_fixed", "_tuned"),
        validate="one_to_one",
    )
    n_missing = len(df_fixed) - len(merged)
    if n_missing:
        logger.warning(f"{n_missing} fixed-alpha cells have no tuned counterpart")
    return merged


def _direction_gaps(merged: pd.DataFrame) -> pd.DataFrame:
    """Pair each Text → Vision cell (t, v) with its mirror Vision → Text (v, t)."""
    tv = merged[merged["direction"] == "text_to_vision"].rename(
        columns={"input_model": "text_model", "output_model": "vision_model"}
    )
    vt = merged[merged["direction"] == "vision_to_text"].rename(
        columns={"input_model": "vision_model", "output_model": "text_model"}
    )
    paired = tv.merge(vt, on=["text_model", "vision_model"], suffixes=("_tv", "_vt"))
    paired["gap_fixed"] = paired["mean_r_fixed_tv"] - paired["mean_r_fixed_vt"]
    paired["gap_tuned"] = paired["mean_r_tuned_tv"] - paired["mean_r_tuned_vt"]
    return paired


def _summarize(merged: pd.DataFrame, paired: pd.DataFrame, alphas: list[float]) -> pd.DataFrame:
    alpha_cols = [c for c in merged.columns if c.startswith("alpha_fold_")]
    rows = []
    for direction, grp in merged.groupby("direction"):
        fold_alphas = grp[alpha_cols].to_numpy(dtype=float).ravel()
        at_edge = np.isin(fold_alphas, [min(alphas), max(alphas)]) if alphas else []
        rows.append(
            {
                "statistic": f"{direction}: mean r fixed / tuned / tuned-fixed",
                "value": f"{grp['mean_r_fixed'].mean():.4f} / "
                f"{grp['mean_r_tuned'].mean():.4f} / "
                f"{(grp['mean_r_tuned'] - grp['mean_r_fixed']).mean():+.4f}",
            }
        )
        rows.append(
            {
                "statistic": f"{direction}: max |tuned-fixed|",
                "value": f"{(grp['mean_r_tuned'] - grp['mean_r_fixed']).abs().max():.4f}",
            }
        )
        rows.append(
            {
                "statistic": f"{direction}: selected alpha median [min, max]",
                "value": f"{np.median(fold_alphas):g} "
                f"[{fold_alphas.min():g}, {fold_alphas.max():g}]",
            }
        )
        if len(at_edge):
            rows.append(
                {
                    "statistic": f"{direction}: folds at grid edge",
                    "value": f"{int(np.sum(at_edge))}/{len(fold_alphas)}",
                }
            )

    for kind in ("fixed", "tuned"):
        gap = paired[f"gap_{kind}"].to_numpy()
        p = float(wilcoxon(paired[f"mean_r_{kind}_tv"], paired[f"mean_r_{kind}_vt"])[1])  # type: ignore
        rows.append(
            {
                "statistic": f"{kind}: mean gap r(T→V) − r(V→T), pairs with T→V higher, "
                "paired Wilcoxon p",
                "value": f"{gap.mean():+.4f}, {int(np.sum(gap > 0))}/{len(gap)}, {p:.2g}",
            }
        )

    same_sign = np.sign(paired["gap_fixed"]) == np.sign(paired["gap_tuned"])
    rho = float(spearmanr(paired["gap_fixed"], paired["gap_tuned"])[0])  # type: ignore
    rows.append(
        {
            "statistic": "gap sign agreement fixed vs tuned",
            "value": f"{int(same_sign.sum())}/{len(paired)}",
        }
    )
    rows.append({"statistic": "gap Spearman rho fixed vs tuned", "value": f"{rho:.3f}"})
    return pd.DataFrame(rows)


def _plot_alpha_heatmaps(
    merged: pd.DataFrame,
    cfg: DictConfig,
    output_dir: Path,
    alphas: list[float],
    font_scale: float,
    figsize: tuple[float, float],
) -> None:
    text_order = list(cfg.get("text_model_order") or [])
    vision_order = list(cfg.get("vision_model_order") or [])
    cmap = _make_colormap(["#F4F1FA", "#A988C8", "#4B2A7B"])
    log_alphas = np.log10(alphas) if alphas else None
    vmin = float(log_alphas.min()) if log_alphas is not None else None
    vmax = float(log_alphas.max()) if log_alphas is not None else None

    for direction, rows_order, cols_order, xlabel, ylabel, title in (
        (
            "text_to_vision",
            text_order,
            vision_order,
            "Vision encoder (target)",
            "Text encoder (input)",
            "Selected ridge α: Text → Vision",
        ),
        (
            "vision_to_text",
            vision_order,
            text_order,
            "Text encoder (target)",
            "Vision encoder (input)",
            "Selected ridge α: Vision → Text",
        ),
    ):
        grp = merged[merged["direction"] == direction]
        if grp.empty:
            continue
        pivot = grp.pivot(index="input_model", columns="output_model", values="alpha_median")
        if rows_order:
            pivot = pivot.reindex(rows_order)
        if cols_order:
            pivot = pivot.reindex(columns=cols_order)
        _plot_heatmap(
            np.log10(pivot.to_numpy(dtype=float)),
            [_pretty_label(x) for x in pivot.index],
            [_pretty_label(x) for x in pivot.columns],
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            cmap=cmap,
            output_path=_resolve_output_path(output_dir, f"alpha_heatmap_{direction}.png"),
            vmin=vmin,
            vmax=vmax,
            annotate=bool(cfg.get("annotate", True)),
            font_scale=font_scale,
            figsize=figsize,
            cbar_label="log₁₀ α (median over folds)",
            value_fmt=".1f",
        )


def _plot_gap_scatter(paired: pd.DataFrame, output_path: Path, font_scale: float) -> None:
    fig, ax = plt.subplots(figsize=(4.5 * font_scale, 4.5 * font_scale))
    x = paired["gap_fixed"].to_numpy()
    y = paired["gap_tuned"].to_numpy()
    lim = float(np.nanmax(np.abs(np.concatenate([x, y])))) * 1.08

    ax.axhline(0, color="#9AA5B1", lw=0.8, zorder=0)
    ax.axvline(0, color="#9AA5B1", lw=0.8, zorder=0)
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="#52606D", lw=1.0, zorder=1, label="y = x")
    ax.scatter(x, y, s=28, color="#7048A4", alpha=0.75, edgecolor="white", lw=0.4, zorder=2)

    rho = float(spearmanr(x, y)[0])  # type: ignore
    agree = int(np.sum(np.sign(x) == np.sign(y)))
    ax.set_title(
        f"Direction gap, fixed vs tuned α\nSpearman ρ = {rho:.3f}; "
        f"same sign in {agree}/{len(x)} pairs",
        fontsize=13 * font_scale,
    )
    ax.set_xlabel("r(T→V) − r(V→T), fixed α = 1", fontsize=12 * font_scale)
    ax.set_ylabel("r(T→V) − r(V→T), tuned α", fontsize=12 * font_scale)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=11 * font_scale)
    ax.legend(frameon=False, fontsize=11 * font_scale, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved scatter → {output_path}")


@hydra.main(
    version_base=None,
    config_path="../../configs/visualization",
    config_name="visualize_ridge_alpha_robustness",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.get("output_dir") or (FIGURES_DIR / "predict_modalities"))
    if not output_dir.is_absolute():
        output_dir = PROJ_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    font_scale = float(cfg.get("font_scale", 1.0))
    figsize = tuple(cfg.get("figsize", [11, 9]))
    alphas = [float(a) for a in (cfg.get("alpha_grid") or [])]

    df_fixed = _load_results_file(Path(cfg.fixed_results_csv))
    df_tuned = _load_results_file(Path(cfg.tuned_results_csv))
    merged = _merge_runs(df_fixed, df_tuned)
    paired = _direction_gaps(merged)

    summary = _summarize(merged, paired, alphas)
    summary_path = output_dir / "ridge_alpha_robustness_summary.csv"
    summary.to_csv(summary_path, index=False)
    for _, row in summary.iterrows():
        logger.info(f"{row['statistic']}: {row['value']}")
    logger.success(f"Saved summary → {summary_path}")

    _plot_alpha_heatmaps(merged, cfg, output_dir, alphas, font_scale, figsize)
    _plot_gap_scatter(
        paired, _resolve_output_path(output_dir, "asymmetry_fixed_vs_tuned.png"), font_scale
    )


if __name__ == "__main__":
    main()
