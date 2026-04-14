"""Visualize neural-encoding results from aggregated/summary CSV outputs.

Reads the aggregated results produced by the neural-encoding pipeline,
plots a bar chart for the selected metric across conditions, annotates
statistical significance (``ns`` for non-significant), and can draw
noise-ceiling references. Optionally, creates a per-subject grouped-bar
panel with per-bar significance labels.

Usage::

    python -m cross_modal_neural_encoding.visualization.visualize_encoding_results \
        aggregated_csv=/path/to/aggregated.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib import rcParams
from loguru import logger
from omegaconf import DictConfig

from cross_modal_neural_encoding.config import FIGURES_DIR, PROJ_ROOT

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

CONDITION_LABELS: dict[str, str] = {
    "image_to_image": "Image → Image",
    "image_to_text": "Image → Text",
    "text_to_image": "Text → Image",
    "text_to_text": "Text → Text",
}

PALETTE = ["#A8C8E8", "#F4A8A8", "#A8D8B0", "#F7D08A"]
GROUP_BAR_COLOR = "#9AA5B5"
SUBJECT_PALETTE = [
    "#7EAEDB",  # darker pastel blue   — sub-02
    "#E88989",  # darker pastel red    — sub-03
    "#84C895",  # darker pastel green  — sub-04
    "#EFBF63",  # darker pastel amber  — sub-05
    "#AF95DD",  # darker pastel purple — sub-06
    "#E9AF84",  # darker pastel orange — sub-07
    "#E39DC1",  # darker pastel pink   — sub-08
    "#93C8C8",  # darker pastel teal   — sub-09
]

# Typography: prefer Lato family when available on the cluster.
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [
    "Lato",
    "Lato Thin",
    "Carlito",
    "DejaVu Sans",
    "Arial",
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


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


def load_aggregated(path: str | Path) -> pd.DataFrame:
    """Load a multi-header aggregated CSV produced by the encoding pipeline."""
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    # The first data row may be the index-name label ("condition") – drop it.
    if "condition" in df.index:
        df = df.drop("condition")
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_summary(path: str | Path) -> pd.DataFrame:
    """Load per-subject summary CSV produced by the encoding pipeline."""
    df = pd.read_csv(path)
    if "condition" not in df.columns:
        raise ValueError("Summary CSV must contain a 'condition' column.")
    return df


def _infer_summary_path_from_aggregated(aggregated_path: Path) -> Path | None:
    """Infer sibling summary.csv path from aggregated.csv path if present."""
    candidate = aggregated_path.parent / "summary.csv"
    return candidate if candidate.exists() else None


def _condition_order(df: pd.DataFrame) -> list[str]:
    """Return plotting order with known conditions first, then unknown."""
    known = [c for c in CONDITION_LABELS if c in df.index]
    unknown = [c for c in df.index if c not in known]
    return known + unknown


def _safe_stat(df: pd.DataFrame, col: str, stat: str) -> np.ndarray | None:
    """Safely extract multi-index column values as float array."""
    key = (col, stat)
    if key not in df.columns:
        return None
    return np.asarray(df[key], dtype=float)


def _signflip_pvalue_greater(
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

    # Exact sign-flip test when feasible.
    if n <= 16:
        all_signs = np.array(
            np.meshgrid(*[[-1.0, 1.0]] * n, indexing="ij")
        ).reshape(n, -1).T
        null_stats = np.mean(all_signs * vals[None, :], axis=1)
        p = (np.sum(null_stats >= observed) + 1) / (null_stats.size + 1)
        return float(p)

    # Monte-Carlo approximation for larger n.
    rng = np.random.default_rng(random_state)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n), replace=True)
    null_stats = np.mean(signs * vals[None, :], axis=1)
    p = (np.sum(null_stats >= observed) + 1) / (n_permutations + 1)
    return float(p)


def _group_level_pvalues_from_summary(
    summary_df: pd.DataFrame,
    *,
    conditions: list[str],
    metric: str,
    n_permutations: int = 10000,
    random_state: int = 42,
) -> np.ndarray:
    """Compute one p-value per condition from subject-level metric values."""
    pvals: list[float] = []
    for cond in conditions:
        vals = np.asarray(
            summary_df.loc[summary_df["condition"] == cond, metric],
            dtype=float,
        )
        pvals.append(
            _signflip_pvalue_greater(
                vals,
                n_permutations=n_permutations,
                random_state=random_state,
            )
        )
    return np.asarray(pvals, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════


def _plot_model_row(
    ax: Axes,
    ax_subject: Axes | None,
    df: pd.DataFrame,
    *,
    model_label: str,
    metric: str,
    p_value_col: str,
    alpha: float,
    summary_df: pd.DataFrame | None,
    show_subject_panel: bool,
    use_group_level_significance: bool,
    group_sig_permutations: int,
    group_sig_random_state: int,
    font_scale: float,
    compress_normalized_axis: bool,
    normalized_axis_linthresh: float,
) -> None:
    """Plot one model row (aggregate panel + optional per-subject panel)."""
    conditions = _condition_order(df)
    df = df.loc[conditions]

    values = df[(metric, "mean")].values.astype(float)

    # Standard deviations (may be all-NaN when only one subject)
    stds = df[(metric, "std")].values.astype(float)
    has_error = not np.all(np.isnan(stds))

    # p-values for significance annotations
    p_values = _safe_stat(df, p_value_col, "mean")
    if p_values is None:
        p_values = np.full(len(conditions), np.nan, dtype=float)

    if (
        use_group_level_significance
        and summary_df is not None
        and {"condition", metric}.issubset(summary_df.columns)
    ):
        p_values = _group_level_pvalues_from_summary(
            summary_df,
            conditions=conditions,
            metric=metric,
            n_permutations=group_sig_permutations,
            random_state=group_sig_random_state,
        )

    is_normalized_metric = "normalized" in metric.lower()

    x = np.arange(len(conditions))
    labels = [CONDITION_LABELS.get(c, c) for c in conditions]

    ax.bar(
        x,
        values,  # type: ignore
        yerr=stds if has_error else None,
        capsize=4,
        color=GROUP_BAR_COLOR,
        edgecolor="black",
        linewidth=0.8,
        width=0.6,
        zorder=3,
    )

    # Only show the normalized reference line.
    if is_normalized_metric:
        ax.axhline(
            y=1.0,
            color="dimgray",
            linestyle="--",
            linewidth=1.4,
            label="Noise-ceiling-normalized reference (1.0)",
            zorder=2,
        )

    # ── Significance annotations ──────────────────────────────────────────
    y_ref = np.nanmax(np.abs(values)) if len(values) else 1.0
    if is_normalized_metric:
        y_ref = max(y_ref, 1.0)
    y_pad = max(y_ref * 0.04, 0.004)
    for i, (val, p) in enumerate(zip(values, p_values)):
        sig = significance_label(p, alpha)
        if not sig:
            continue
        bar_top = val + (stds[i] if has_error and not np.isnan(stds[i]) else 0)
        y_pos = max(bar_top, 0) + y_pad
        ax.text(
            i,
            y_pos,
            sig,
            ha="center",
            va="bottom",
            fontsize=12 * font_scale,
            fontweight="bold",
            color="black" if sig == "ns" else "darkred",
        )

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12 * font_scale)  # type: ignore
    ax.tick_params(axis="y", labelsize=12 * font_scale)
    if is_normalized_metric:
        ax.set_ylabel(
            "Normalized performance\n(r / NC)",
            fontsize=11.5 * font_scale,
        )
    else:
        ax.set_ylabel("Pearson correlation", fontsize=11.5 * font_scale)
    ax.set_title(
        f"{model_label} • Group results",
        fontsize=15 * font_scale,
        fontweight="bold",
    )
    handles, labels_legend = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels_legend, loc="upper right", fontsize=11 * font_scale)
    ax.axhline(y=0, color="black", linewidth=0.5, zorder=1)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Dynamic y-limits for consistent readability.
    stds_arr = np.asarray(stds, dtype=float)
    stds_filled = np.where(np.isnan(stds_arr), 0.0, stds_arr)
    lower = np.nanmin(values - stds_filled)
    upper = np.nanmax(values + stds_filled)
    if is_normalized_metric:
        upper = max(upper, 1.0)
    span = max(upper - lower, 0.03)
    y_min = min(-0.01, lower - 0.12 * span)
    y_max = upper + 0.16 * span

    if is_normalized_metric and compress_normalized_axis:
        y_min = min(y_min, -0.06)
        y_max = max(1.08, upper + 0.08 * span)
        ax.set_yscale(
            "symlog",
            linthresh=max(normalized_axis_linthresh, 1e-3),
            linscale=1.0,
            base=10,
        )

    ax.set_ylim(y_min, y_max)

    # Optional subject-level panel.
    include_subject_panel = (
        show_subject_panel
        and ax_subject is not None
        and summary_df is not None
        and {"subject", "condition", metric}.issubset(summary_df.columns)
    )
    if include_subject_panel and ax_subject is not None:
        assert summary_df is not None
        subject_table = (
            summary_df[summary_df["condition"].isin(conditions)]
            .pivot_table(index="subject", columns="condition", values=metric, aggfunc="mean")
            .reindex(columns=conditions)
            .sort_index()
        )

        if p_value_col in summary_df.columns:
            pval_table = (
                summary_df[summary_df["condition"].isin(conditions)]
                .pivot_table(index="subject", columns="condition", values=p_value_col, aggfunc="mean")
                .reindex(index=subject_table.index, columns=conditions)
            )
        else:
            pval_table = pd.DataFrame(
                np.nan,
                index=subject_table.index,
                columns=conditions,
            )

        subjects = list(subject_table.index)
        n_subj = len(subjects)
        if n_subj > 0:
            total_width = 0.82
            bar_w = min(0.16, total_width / n_subj)
            offsets = (np.arange(n_subj) - (n_subj - 1) / 2.0) * bar_w
            subject_colors = [
                SUBJECT_PALETTE[i % len(SUBJECT_PALETTE)] for i in range(n_subj)
            ]

            for s_idx, subject in enumerate(subjects):
                heights = np.asarray(subject_table.loc[subject].values, dtype=float)
                p_sub = np.asarray(pval_table.loc[subject].values, dtype=float)
                xpos = x + offsets[s_idx]
                ax_subject.bar(
                    xpos,
                    heights,
                    width=bar_w * 0.95,
                    color=subject_colors[s_idx],
                    edgecolor="#4A4A4A",
                    linewidth=0.5,
                    alpha=0.85,
                    zorder=3,
                    label=subject,
                )

                # Per-sub-bar significance labels.
                for j, (xj, hj, pj) in enumerate(zip(xpos, heights, p_sub)):
                    if not np.isfinite(hj):
                        continue
                    sig = significance_label(float(pj), alpha)
                    if not sig:
                        continue
                    y_sig = hj + (0.015 * (y_max - y_min) if hj >= 0 else -0.015 * (y_max - y_min))
                    va = "bottom" if hj >= 0 else "top"
                    ax_subject.text(
                        xj,
                        y_sig,
                        sig,
                        ha="center",
                        va=va,
                        fontsize=8 * font_scale,
                        rotation=90,
                        color="black" if sig == "ns" else "darkred",
                        zorder=5,
                    )

        if is_normalized_metric:
            ax_subject.axhline(
                y=1.0,
                color="dimgray",
                linestyle="--",
                linewidth=1.2,
                zorder=1,
            )

        ax_subject.set_xticks(x)
        ax_subject.set_xticklabels(labels, fontsize=12 * font_scale)
        ax_subject.tick_params(axis="y", labelsize=12 * font_scale)
        ax_subject.set_title(
            f"{model_label} • Per-subject grouped (n={subject_table.shape[0]})",
            fontsize=15 * font_scale,
            fontweight="bold",
        )
        ax_subject.axhline(y=0, color="black", linewidth=0.5, zorder=1)
        ax_subject.grid(axis="y", alpha=0.3, zorder=0)
        ncol = 2 if max(1, subject_table.shape[0]) > 6 else 1
        ax_subject.legend(loc="upper right", fontsize=10 * font_scale, ncol=ncol)

        if is_normalized_metric and compress_normalized_axis:
            ax_subject.set_yscale(
                "symlog",
                linthresh=max(normalized_axis_linthresh, 1e-3),
                linscale=1.0,
                base=10,
            )

        ax_subject.set_ylim(y_min, y_max)
    elif ax_subject is not None:
        ax_subject.axis("off")


def plot_encoding_results(
    model_results: list[dict[str, Any]],
    *,
    metric: str = "mean_r",
    p_value_col: str = "p_value_mean_r",
    alpha: float = 0.05,
    show_subject_panel: bool = True,
    use_group_level_significance: bool = True,
    group_sig_permutations: int = 10000,
    group_sig_random_state: int = 42,
    font_scale: float = 1.0,
    compress_normalized_axis: bool = True,
    normalized_axis_linthresh: float = 0.08,
    output_path: Path | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> None:
    """Create one row per model with group and optional per-subject panels."""
    n_rows = len(model_results)
    if n_rows == 0:
        raise ValueError("No model results to plot.")

    n_cols = 2 if show_subject_panel else 1
    base_w, base_h = figsize
    fig_w = base_w * (1.9 if n_cols == 2 else 1.0)
    fig_h = base_h * n_rows

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        sharey="row",
        gridspec_kw={"width_ratios": [1.0, 1.35]} if n_cols == 2 else None,
    )

    for row_i, item in enumerate(model_results):
        row_ax = axes[row_i, 0]
        row_ax_subject = axes[row_i, 1] if n_cols == 2 else None
        _plot_model_row(
            row_ax,
            row_ax_subject,
            item["aggregated_df"],
            model_label=item["model_label"],
            metric=metric,
            p_value_col=p_value_col,
            alpha=alpha,
            summary_df=item.get("summary_df"),
            show_subject_panel=show_subject_panel,
            use_group_level_significance=use_group_level_significance,
            group_sig_permutations=group_sig_permutations,
            group_sig_random_state=group_sig_random_state,
            font_scale=font_scale,
            compress_normalized_axis=compress_normalized_axis,
            normalized_axis_linthresh=normalized_axis_linthresh,
        )

    fig.tight_layout(pad=1.0, h_pad=2.0, w_pad=1.0)
    if n_rows > 1:
        fig.subplots_adjust(hspace=0.35)

    # ── Save ──────────────────────────────────────────────────────────────
    if output_path is None:
        output_path = FIGURES_DIR / "encoding_results.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.success(f"Figure saved to {output_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Hydra entry-point
# ═══════════════════════════════════════════════════════════════════════════


@hydra.main(
    config_path="../../configs/visualization",
    config_name="visualize_encoding_results",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Load encoding outputs and produce figure(s)."""
    model_results: list[dict[str, Any]] = []

    run_dir_cfg = cfg.get("run_dir", None)
    if run_dir_cfg:
        run_dir = Path(run_dir_cfg)
        if not run_dir.is_absolute():
            run_dir = PROJ_ROOT / run_dir
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

        logger.info(f"Scanning model subfolders in {run_dir}")
        for child in sorted(run_dir.iterdir()):
            if not child.is_dir():
                continue
            agg_path = child / "aggregated.csv"
            if not agg_path.exists():
                continue
            try:
                agg_df = load_aggregated(agg_path)
            except Exception as exc:
                logger.warning(f"Skipping {child.name}: failed to load aggregated.csv ({exc})")
                continue

            summary_path = child / "summary.csv"
            summary_df = load_summary(summary_path) if summary_path.exists() else None
            model_results.append(
                {
                    "model_label": child.name,
                    "aggregated_df": agg_df,
                    "summary_df": summary_df,
                }
            )

        if not model_results:
            raise FileNotFoundError(
                f"No model subfolders with aggregated.csv found in: {run_dir}"
            )
    else:
        path = Path(cfg.aggregated_csv)
        if not path.exists():
            raise FileNotFoundError(f"Aggregated CSV not found: {path}")

        logger.info(f"Loading aggregated results from {path}")
        df = load_aggregated(path)
        logger.info(f"Conditions: {df.index.tolist()}")

        summary_df: pd.DataFrame | None = None
        summary_cfg = cfg.get("summary_csv", None)
        if summary_cfg:
            summary_path = Path(summary_cfg)
            if summary_path.exists():
                logger.info(f"Loading per-subject summary from {summary_path}")
                summary_df = load_summary(summary_path)
            else:
                logger.warning(f"Configured summary_csv does not exist: {summary_path}")
        else:
            inferred = _infer_summary_path_from_aggregated(path)
            if inferred is not None:
                logger.info(f"Using inferred per-subject summary: {inferred}")
                summary_df = load_summary(inferred)

        model_results.append(
            {
                "model_label": path.parent.name,
                "aggregated_df": df,
                "summary_df": summary_df,
            }
        )

    output_path = Path(cfg.output_path) if cfg.output_path else None

    plot_encoding_results(
        model_results,
        metric=cfg.metric,
        p_value_col=cfg.p_value_col,
        alpha=cfg.alpha,
        show_subject_panel=bool(cfg.get("show_subject_panel", True)),
        use_group_level_significance=bool(cfg.get("use_group_level_significance", True)),
        group_sig_permutations=int(cfg.get("group_sig_permutations", 10000)),
        group_sig_random_state=int(cfg.get("group_sig_random_state", 42)),
        font_scale=float(cfg.get("font_scale", 1.15)),
        compress_normalized_axis=bool(cfg.get("compress_normalized_axis", False)),
        normalized_axis_linthresh=float(cfg.get("normalized_axis_linthresh", 0.08)),
        output_path=output_path,
        figsize=tuple(cfg.figsize),
    )


if __name__ == "__main__":
    main()
