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
from collections import Counter
from typing import Any, Iterable

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from loguru import logger
from omegaconf import DictConfig

from cross_modal_neural_encoding.config import FIGURES_DIR, PROJ_ROOT
from cross_modal_neural_encoding.utils import (
    CONDITION_LABELS,
    configure_plot_fonts,
    significance_label,
    signflip_pvalue_greater,
)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

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
VLM_MODEL_PALETTE = [
    "#F5A3A3",  # soft red
    "#E88989",  # dusty red
    "#D96F6F",  # muted crimson
    "#F2B6A0",  # warm peach
    "#F7B7B2",  # salmon
    "#E39DC1",  # pink-rose
]
VISION_MODEL_PALETTE = [
    "#C3A1D6",  # warm lavender
    "#A988C8",  # warm soft purple
    "#906EBA",  # muted warm violet
    "#D0B4E6",  # light warm lilac
    "#B39AD8",  # warm pale purple
    "#9B7CCF",  # warm medium purple
]
TEXT_MODEL_PALETTE = [
    "#F2D07B",  # ochre
    "#E8C05A",  # warm yellow
    "#D9AE3F",  # goldenrod
    "#F4D58D",  # pale ochre
    "#EFBF63",  # amber
    "#F7D08A",  # soft yellow
]
COLD_SUBJECT_PALETTE = [
    "#A8C8E8",  # cold pastel blue
    "#93C8C8",  # cold pastel teal
    "#B7DDF2",  # cold pastel sky
    "#9ECEDC",  # cold pastel cyan
    "#B0D9E8",  # cold pastel light blue
    "#8FBFD4",  # cold pastel steel
    "#C4E1F5",  # cold pastel ice
    "#A3D2E2",  # cold pastel aqua
]

configure_plot_fonts()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _model_category_rank(model_label: str) -> tuple[int, str]:
    """Sort key: VLM first, then vision-only, then text-only."""
    label = model_label.lower()
    if any(tag in label for tag in ("clip", "vlm", "qwen", "intern")):
        return (0, model_label)
    if any(tag in label for tag in ("dinov2", "ijepa", "dino")):
        return (1, model_label)
    return (2, model_label)


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


def _collect_model_dirs(run_dir: Path) -> list[dict[str, Any]]:
    """Collect model directories from a run directory.

    Priority:
    1) Direct model subfolders containing aggregated.csv (legacy behavior).
    2) If none found, search run_array_* subfolders and collect their models.
    """
    model_entries: list[dict[str, Any]] = []

    direct_models = [
        child
        for child in sorted(run_dir.iterdir())
        if child.is_dir() and (child / "aggregated.csv").exists()
    ]
    if direct_models:
        for child in direct_models:
            model_entries.append(
                {
                    "path": child,
                    "model_label": child.name,
                    "run_array": None,
                }
            )
        return model_entries

    run_arrays = [
        child
        for child in sorted(run_dir.iterdir())
        if child.is_dir() and child.name.startswith("run_array_")
    ]
    for run_array in run_arrays:
        for child in sorted(run_array.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "aggregated.csv").exists():
                continue
            model_entries.append(
                {
                    "path": child,
                    "model_label": child.name,
                    "run_array": run_array.name,
                }
            )

    return model_entries


def _condition_order(df: pd.DataFrame) -> list[str]:
    """Return plotting order with known conditions first, then unknown."""
    known = [c for c in CONDITION_LABELS if c in df.index]
    unknown = [c for c in df.index if c not in known]
    return known + unknown


def _condition_order_from_index(index: Iterable[str]) -> list[str]:
    """Return plotting order for a generic iterable of conditions."""
    index_list = list(index)
    known = [c for c in CONDITION_LABELS if c in index_list]
    unknown = [c for c in index_list if c not in known]
    return known + unknown


def _safe_stat(df: pd.DataFrame, col: str, stat: str) -> np.ndarray | None:
    """Safely extract multi-index column values as float array."""
    key = (col, stat)
    if key not in df.columns:
        return None
    return np.asarray(df[key], dtype=float)


def _compute_plot_ylims(
    values: np.ndarray,
    stds: np.ndarray,
    *,
    is_normalized_metric: bool,
    compress_normalized_axis: bool,
    normalized_axis_linthresh: float,
    force_normalized_reference: bool = True,
) -> tuple[float, float]:
    """Compute shared y-limits for encoding plots."""
    stds_filled = np.where(np.isnan(stds), 0.0, stds)
    lower = np.nanmin(values - stds_filled)
    upper = np.nanmax(values + stds_filled)
    if is_normalized_metric and force_normalized_reference:
        upper = max(upper, 1.0)
    span = max(upper - lower, 0.03)
    y_min = min(-0.01, lower - 0.08 * span)
    y_max = upper + 0.10 * span

    if is_normalized_metric and compress_normalized_axis:
        y_min = min(y_min, -0.05)
        y_max = max(1.02, upper + 0.05 * span)

    return y_min, y_max


def _shared_y_limits(
    model_results: list[dict[str, Any]],
    *,
    metric: str,
    compress_normalized_axis: bool,
    normalized_axis_linthresh: float,
    force_normalized_reference: bool = True,
) -> tuple[float, float] | None:
    """Compute shared y-limits across all models for the chosen metric."""
    values_list: list[np.ndarray] = []
    stds_list: list[np.ndarray] = []
    for item in model_results:
        df = item["aggregated_df"]
        try:
            values = np.asarray(df[(metric, "mean")], dtype=float)
            stds = np.asarray(df[(metric, "std")], dtype=float)
        except KeyError:
            continue
        values_list.append(values)
        stds_list.append(stds)

    if not values_list:
        return None

    values_all = np.concatenate(values_list)
    stds_all = np.concatenate(stds_list)
    is_normalized_metric = "normalized" in metric.lower()
    return _compute_plot_ylims(
        values_all,
        stds_all,
        is_normalized_metric=is_normalized_metric,
        compress_normalized_axis=compress_normalized_axis,
        normalized_axis_linthresh=normalized_axis_linthresh,
        force_normalized_reference=force_normalized_reference,
    )


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
            signflip_pvalue_greater(
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
    y_limits: tuple[float, float] | None,
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
    ax.set_xticklabels(
        labels,
        fontsize=10.5 * font_scale,
        rotation=20,
        ha="right",
    )  # type: ignore
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
    ax.margins(x=0.03)

    values_arr = np.asarray(values, dtype=float)
    stds_arr = np.asarray(stds, dtype=float)
    if y_limits is None:
        y_min, y_max = _compute_plot_ylims(
            values_arr,
            stds_arr,
            is_normalized_metric=is_normalized_metric,
            compress_normalized_axis=compress_normalized_axis,
            normalized_axis_linthresh=normalized_axis_linthresh,
            force_normalized_reference=True,
        )
    else:
        y_min, y_max = y_limits

    if is_normalized_metric and compress_normalized_axis:
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

        if is_normalized_metric and y_max >= 1.0:
            ax_subject.axhline(
                y=1.0,
                color="dimgray",
                linestyle="--",
                linewidth=1.2,
                zorder=1,
            )

        ax_subject.set_xticks(x)
        ax_subject.set_xticklabels(
            labels,
            fontsize=10.5 * font_scale,
            rotation=20,
            ha="right",
        )
        ax_subject.tick_params(axis="y", labelsize=12 * font_scale)
        ax_subject.set_title(
            f"{model_label}, per-subject grouped (n={subject_table.shape[0]})",
            fontsize=15 * font_scale,
            fontweight="bold",
        )
        ax_subject.axhline(y=0, color="black", linewidth=0.5, zorder=1)
        ax_subject.grid(axis="y", alpha=0.3, zorder=0)
        ax_subject.margins(x=0.03)
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
    y_limits: tuple[float, float] | None = None,
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
            y_limits=y_limits,
        )

    fig.tight_layout(pad=0.6, h_pad=1.2, w_pad=0.8)
    if n_rows > 1:
        fig.subplots_adjust(hspace=0.25)

    # ── Save ──────────────────────────────────────────────────────────────
    if output_path is None:
        output_path = FIGURES_DIR / "encoding_results.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.success(f"Figure saved to {output_path}")
    plt.close(fig)


def plot_grouped_model_means(
    model_results: list[dict[str, Any]],
    *,
    metric: str,
    alpha: float,
    font_scale: float,
    compress_normalized_axis: bool,
    normalized_axis_linthresh: float,
    output_path: Path | None,
    figsize: tuple[float, float],
    y_limits: tuple[float, float] | None,
) -> None:
    """Plot grouped bars comparing model means across conditions.

    Models are allowed to have different condition sets; missing values
    are shown as empty slots (no bar).
    """
    if len(model_results) < 2:
        logger.warning("Grouped model plot requires at least two models.")
        return

    model_results = sorted(
        model_results,
        key=lambda item: _model_category_rank(item.get("model_label", "")),
    )

    all_conditions: set[str] = set()
    for item in model_results:
        df = item["aggregated_df"]
        all_conditions.update(df.index)

    if not all_conditions:
        logger.warning("No conditions found across models to plot.")
        return

    conditions = _condition_order_from_index(all_conditions)
    labels = [CONDITION_LABELS.get(c, c) for c in conditions]

    model_labels = [item["model_label"] for item in model_results]
    n_models = len(model_labels)
    n_conditions = len(conditions)

    values = np.full((n_models, n_conditions), np.nan, dtype=float)
    for i, item in enumerate(model_results):
        df = item["aggregated_df"]
        for j, cond in enumerate(conditions):
            if cond not in df.index:
                continue
            values[i, j] = float(df.loc[cond, (metric, "mean")])

    pvals = np.full((n_models, n_conditions), np.nan, dtype=float)
    for i, item in enumerate(model_results):
        summary_df = item.get("summary_df")
        if summary_df is None or metric not in summary_df.columns:
            continue
        for j, cond in enumerate(conditions):
            if cond not in summary_df["condition"].unique():
                continue
            vals = np.asarray(
                summary_df.loc[summary_df["condition"] == cond, metric],
                dtype=float,
            )
            pvals[i, j] = signflip_pvalue_greater(vals)

    is_normalized_metric = "normalized" in metric.lower()

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_conditions)
    total_width = 0.82
    bar_w = min(0.16, total_width / max(n_models, 1))
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_w
    category_palettes = {
        0: VLM_MODEL_PALETTE,
        1: VISION_MODEL_PALETTE,
        2: TEXT_MODEL_PALETTE,
    }
    category_counts = {0: 0, 1: 0, 2: 0}
    colors: list[str] = []
    for label in model_labels:
        category = _model_category_rank(label)[0]
        palette = category_palettes.get(category, VLM_MODEL_PALETTE)
        idx = category_counts.get(category, 0) % len(palette)
        colors.append(palette[idx])
        category_counts[category] = category_counts.get(category, 0) + 1

    if y_limits is None:
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            logger.warning("No finite values available for grouped plot.")
            return
        stds = np.zeros_like(finite_values)
        y_min, y_max = _compute_plot_ylims(
            finite_values,
            stds,
            is_normalized_metric=is_normalized_metric,
            compress_normalized_axis=compress_normalized_axis,
            normalized_axis_linthresh=normalized_axis_linthresh,
        )
    else:
        y_min, y_max = y_limits

    y_span = y_max - y_min
    y_na_base = min(max(0.0, y_min), y_max)
    y_na = y_na_base + (0.02 * y_span if y_span > 0 else 0.0)

    for i, label in enumerate(model_labels):
        ax.bar(
            x + offsets[i],
            values[i],
            width=bar_w * 0.95,
            color=colors[i],
            edgecolor="#4A4A4A",
            linewidth=0.5,
            alpha=0.9,
            label=label,
            zorder=3,
        )

        # Mark missing values with "na" for conditions that don't apply.
        for j, xj in enumerate(x + offsets[i]):
            if np.isfinite(values[i, j]):
                continue
            ax.text(
                xj,
                y_na,
                "na",
                ha="center",
                va="bottom",
                fontsize=8.0 * font_scale,
                color="#5C5C5C",
                zorder=4,
            )

        # Significance annotations per model-condition (if available).
        for j, (xj, hj) in enumerate(zip(x + offsets[i], values[i])):
            if not np.isfinite(hj):
                continue
            pval = pvals[i, j]
            if not np.isfinite(pval):
                continue
            sig = significance_label(float(pval), alpha)
            if not sig:
                continue
            y_sig = hj + (0.015 * (y_max - y_min) if hj >= 0 else -0.015 * (y_max - y_min))
            va = "bottom" if hj >= 0 else "top"
            ax.text(
                xj,
                y_sig,
                sig,
                ha="center",
                va=va,
                fontsize=8.5 * font_scale,
                color="black" if sig == "ns" else "darkred",
                zorder=5,
            )

    if is_normalized_metric:
        ax.axhline(
            y=1.0,
            color="dimgray",
            linestyle="--",
            linewidth=1.2,
            zorder=1,
            label="Noise-ceiling-normalized reference (1.0)",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        fontsize=10.5 * font_scale,
        rotation=20,
        ha="right",
    )
    ax.tick_params(axis="y", labelsize=12 * font_scale)
    if is_normalized_metric:
        ax.set_ylabel("Normalized performance\n(r / NC)", fontsize=11.5 * font_scale)
    else:
        ax.set_ylabel("Pearson correlation", fontsize=11.5 * font_scale)
    ax.set_title(
        "Model Comparison Based on Group Means",
        fontsize=15 * font_scale,
        fontweight="bold",
    )
    ax.axhline(y=0, color="black", linewidth=0.5, zorder=1)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper right", fontsize=8.5 * font_scale, ncol=3)
    ax.margins(x=0.03)

    if is_normalized_metric and compress_normalized_axis:
        ax.set_yscale(
            "symlog",
            linthresh=max(normalized_axis_linthresh, 1e-3),
            linscale=1.0,
            base=10,
        )

    ax.set_ylim(y_min, y_max)
    fig.tight_layout(pad=0.6)

    if output_path is None:
        output_path = FIGURES_DIR / "encoding_results_models_grouped.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.success(f"Grouped-model figure saved to {output_path}")
    plt.close(fig)


def plot_subject_mean_across_models(
    model_results: list[dict[str, Any]],
    *,
    metric: str,
    p_value_col: str,
    alpha: float,
    font_scale: float,
    compress_normalized_axis: bool,
    normalized_axis_linthresh: float,
    output_path: Path | None,
    figsize: tuple[float, float],
    y_limits: tuple[float, float] | None,
) -> None:
    """Plot per-subject means aggregated across models (supplementary)."""
    dfs: list[pd.DataFrame] = []
    for item in model_results:
        summary_df = item.get("summary_df")
        if summary_df is None or metric not in summary_df.columns:
            continue
        keep_cols = ["subject", "condition", metric]
        if p_value_col in summary_df.columns:
            keep_cols.append(p_value_col)
        df = summary_df[keep_cols].copy()
        df["model_label"] = item["model_label"]
        dfs.append(df)

    if not dfs:
        logger.warning("No per-subject summaries available to aggregate across models.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    conditions = _condition_order_from_index(combined["condition"].unique())
    subject_table = (
        combined.pivot_table(index="subject", columns="condition", values=metric, aggfunc="mean")
        .reindex(columns=conditions)
        .sort_index()
    )
    if p_value_col in combined.columns:
        subject_pvals = (
            combined.pivot_table(
                index="subject",
                columns="condition",
                values=p_value_col,
                aggfunc="mean",
            )
            .reindex(index=subject_table.index, columns=conditions)
        )
    else:
        subject_pvals = pd.DataFrame(
            np.nan,
            index=subject_table.index,
            columns=conditions,
        )

    subjects = list(subject_table.index)
    n_subj = len(subjects)
    if n_subj == 0:
        logger.warning("No subjects found for aggregated per-subject plot.")
        return

    is_normalized_metric = "normalized" in metric.lower()
    if y_limits is None:
        vals = np.asarray(subject_table.values, dtype=float)
        stds = np.zeros_like(vals)
        y_min, y_max = _compute_plot_ylims(
            vals.ravel(),
            stds.ravel(),
            is_normalized_metric=is_normalized_metric,
            compress_normalized_axis=compress_normalized_axis,
            normalized_axis_linthresh=normalized_axis_linthresh,
        )
    else:
        y_min, y_max = y_limits

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(conditions))
    total_width = 0.82
    bar_w = min(0.16, total_width / max(n_subj, 1))
    offsets = (np.arange(n_subj) - (n_subj - 1) / 2.0) * bar_w
    subject_colors = [COLD_SUBJECT_PALETTE[i % len(COLD_SUBJECT_PALETTE)] for i in range(n_subj)]

    for s_idx, subject in enumerate(subjects):
        heights = np.asarray(subject_table.loc[subject].values, dtype=float)
        p_sub = np.asarray(subject_pvals.loc[subject].values, dtype=float)
        ax.bar(
            x + offsets[s_idx],
            heights,
            width=bar_w * 0.95,
            color=subject_colors[s_idx],
            edgecolor="#4A4A4A",
            linewidth=0.5,
            alpha=0.85,
            zorder=3,
            label=subject,
        )

        for j, (xj, hj, pj) in enumerate(zip(x + offsets[s_idx], heights, p_sub)):
            if not np.isfinite(hj):
                continue
            sig = significance_label(float(pj), alpha)
            if not sig:
                continue
            y_sig = hj + (0.015 * (y_max - y_min) if hj >= 0 else -0.015 * (y_max - y_min))
            va = "bottom" if hj >= 0 else "top"
            ax.text(
                xj,
                y_sig,
                sig,
                ha="center",
                va=va,
                fontsize=8.5 * font_scale,
                color="black" if sig == "ns" else "darkred",
                zorder=5,
            )

    if is_normalized_metric:
        ax.axhline(
            y=1.0,
            color="dimgray",
            linestyle="--",
            linewidth=1.2,
            zorder=1,
            label="Noise-ceiling-normalized reference (1.0)",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS.get(c, c) for c in conditions],
        fontsize=10.5 * font_scale,
        rotation=20,
        ha="right",
    )
    ax.tick_params(axis="y", labelsize=12 * font_scale)
    if is_normalized_metric:
        ax.set_ylabel("Normalized performance\n(r / NC)", fontsize=11.5 * font_scale)
    else:
        ax.set_ylabel("Pearson correlation", fontsize=11.5 * font_scale)
    ax.set_title(
        f"Per-subject mean across models (n={n_subj})",
        fontsize=15 * font_scale,
        fontweight="bold",
    )
    ax.axhline(y=0, color="black", linewidth=0.5, zorder=1)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ncol = 2 if n_subj > 6 else 1
    ax.legend(loc="upper right", fontsize=10 * font_scale, ncol=ncol)
    ax.margins(x=0.03)

    if is_normalized_metric and compress_normalized_axis:
        ax.set_yscale(
            "symlog",
            linthresh=max(normalized_axis_linthresh, 1e-3),
            linscale=1.0,
            base=10,
        )

    ax.set_ylim(y_min, y_max)
    fig.tight_layout(pad=0.6)

    if output_path is None:
        output_path = FIGURES_DIR / "encoding_results_subjects_mean.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.success(f"Per-subject mean figure saved to {output_path}")
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

        model_entries = _collect_model_dirs(run_dir)
        if not model_entries:
            raise FileNotFoundError(
                f"No model subfolders with aggregated.csv found in: {run_dir}"
            )

        run_arrays_present = any(entry["run_array"] is not None for entry in model_entries)
        if run_arrays_present:
            logger.info(
                f"Scanning run_array_* subfolders in {run_dir} ({len(model_entries)} model folders)"
            )
        else:
            logger.info(f"Scanning model subfolders in {run_dir} ({len(model_entries)} models)")

        name_counts = Counter(entry["model_label"] for entry in model_entries)

        for entry in model_entries:
            child = entry["path"]
            agg_path = child / "aggregated.csv"
            try:
                agg_df = load_aggregated(agg_path)
            except Exception as exc:
                logger.warning(
                    f"Skipping {child.name}: failed to load aggregated.csv ({exc})"
                )
                continue

            summary_path = child / "summary.csv"
            summary_df = load_summary(summary_path) if summary_path.exists() else None

            model_label = entry["model_label"]
            run_array = entry["run_array"]
            if name_counts.get(model_label, 0) > 1 and run_array is not None:
                model_label = f"{model_label} ({run_array})"

            model_results.append(
                {
                    "model_label": model_label,
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
    font_scale = float(cfg.get("font_scale", 1.15))
    compress_normalized_axis = bool(cfg.get("compress_normalized_axis", False))
    normalized_axis_linthresh = float(cfg.get("normalized_axis_linthresh", 0.08))
    share_y_limits = bool(cfg.get("share_y_limits", True))
    shared_y_limits = (
        _shared_y_limits(
            model_results,
            metric=cfg.metric,
            compress_normalized_axis=compress_normalized_axis,
            normalized_axis_linthresh=normalized_axis_linthresh,
        )
        if share_y_limits
        else None
    )

    if bool(cfg.get("plot_per_model_rows", True)):
        plot_encoding_results(
            model_results,
            metric=cfg.metric,
            p_value_col=cfg.p_value_col,
            alpha=cfg.alpha,
            show_subject_panel=bool(cfg.get("show_subject_panel", True)),
            use_group_level_significance=bool(cfg.get("use_group_level_significance", True)),
            group_sig_permutations=int(cfg.get("group_sig_permutations", 10000)),
            group_sig_random_state=int(cfg.get("group_sig_random_state", 42)),
            font_scale=font_scale,
            compress_normalized_axis=compress_normalized_axis,
            normalized_axis_linthresh=normalized_axis_linthresh,
            output_path=output_path,
            figsize=tuple(cfg.figsize),
            y_limits=shared_y_limits,
        )

    if bool(cfg.get("plot_grouped_models", True)):
        grouped_output = (
            Path(cfg.grouped_output_path)
            if cfg.get("grouped_output_path", None)
            else None
        )
        plot_grouped_model_means(
            model_results,
            metric=cfg.metric,
            alpha=cfg.alpha,
            font_scale=font_scale,
            compress_normalized_axis=compress_normalized_axis,
            normalized_axis_linthresh=normalized_axis_linthresh,
            output_path=grouped_output,
            figsize=tuple(cfg.get("grouped_figsize", cfg.figsize)),
            y_limits=shared_y_limits,
        )

    if bool(cfg.get("plot_subject_mean_across_models", True)):
        subjects_output = (
            Path(cfg.subjects_mean_output_path)
            if cfg.get("subjects_mean_output_path", None)
            else None
        )
        plot_subject_mean_across_models(
            model_results,
            metric=cfg.metric,
            p_value_col=cfg.p_value_col,
            alpha=cfg.alpha,
            font_scale=font_scale,
            compress_normalized_axis=compress_normalized_axis,
            normalized_axis_linthresh=normalized_axis_linthresh,
            output_path=subjects_output,
            figsize=tuple(cfg.get("subjects_mean_figsize", cfg.figsize)),
            y_limits=shared_y_limits,
        )


if __name__ == "__main__":
    main()
