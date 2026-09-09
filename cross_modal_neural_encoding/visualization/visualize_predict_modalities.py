"""Visualize cross-modal prediction results as heatmaps.

Creates two heatmaps:
1) Text → Vision (rows: text encoders, columns: vision encoders)
2) Vision → Text (rows: vision encoders, columns: text encoders)

Can optionally compare two results files and plot the difference heatmaps.

Usage
-----
    python -m cross_modal_neural_encoding.visualization.visualize_predict_modalities

Hydra config: ``configs/visualization/visualize_predict_modalities.yaml``
"""

from __future__ import annotations

from pathlib import Path
import textwrap
from typing import cast

import hydra
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from scipy.stats import wilcoxon

from cross_modal_neural_encoding.config import FIGURES_DIR, PROJ_ROOT
from cross_modal_neural_encoding.utils import (
    configure_plot_fonts,
    significance_label,
)
from cross_modal_neural_encoding.visualization.visualize_encoding_results import (
    MATRIX_FAMILY_DISPLAY,
    model_family,
    model_size,
)

configure_plot_fonts()


def _pretty_label(label: str) -> str:
    """``"Qwen--Qwen3.5-2B-Base"`` -> ``"Qwen3.5-VL 2B"``.

    Same family and scale naming as the encoding figures, so a reader moving
    between them sees one set of model names.
    """
    family = model_family(label)
    return f"{MATRIX_FAMILY_DISPLAY.get(family, family)} {model_size(label)[1]}".strip()


def _make_colormap(colors: list[str]) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("custom", colors)


def _resolve_output_path(output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def _plot_heatmap(
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: LinearSegmentedColormap,
    output_path: Path,
    vmin: float | None,
    vmax: float | None,
    annotate: bool,
    font_scale: float,
    figsize: tuple[float, float],
    comparison_note: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=13 * font_scale)
    ax.set_yticklabels(row_labels, fontsize=13 * font_scale)
    mean_val = float(np.nanmean(data))
    subtitle = f"mean = {mean_val:.3f}"
    if comparison_note:
        subtitle += f"    |    {comparison_note}"
    # Wrap each title line so the title never spills past the figure width
    # (rather than letting bbox_inches="tight" widen the whole canvas).
    title_font = 14.5 * font_scale
    max_chars = max(20, int(figsize[0] * 72 / (0.55 * title_font)))
    title_lines = textwrap.fill(title, width=max_chars)
    subtitle_lines = textwrap.fill(subtitle, width=max_chars)
    ax.set_title(f"{title_lines}\n{subtitle_lines}", fontsize=title_font, pad=10)
    ax.set_xlabel(xlabel, fontsize=14 * font_scale, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=14 * font_scale, labelpad=8)

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9.5 * font_scale,
                        color="#1f2d3d",
                    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=13 * font_scale)
    cbar.set_label("Mean Pearson r", fontsize=14 * font_scale)

    fig.tight_layout()
    # bbox_inches="tight" so the inline comparison note in the title is never
    # clipped at the figure edge.
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved heatmap → {output_path}")


def _load_results_file(file_path: Path) -> pd.DataFrame:
    if not file_path.is_absolute():
        file_path = PROJ_ROOT / file_path
    if not file_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {file_path}")
    df = pd.read_csv(file_path)
    if "direction" not in df.columns:
        raise ValueError(f"Results CSV must contain a 'direction' column: {file_path}")
    return df


def _create_pivot_matrices(
    df: pd.DataFrame,
    text_order: list[str],
    vision_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_tv = df[df["direction"] == "text_to_vision"].copy()
    df_vt = df[df["direction"] == "vision_to_text"].copy()

    pivot_tv = None
    pivot_vt = None

    if not df_tv.empty:
        pivot_tv = df_tv.pivot(index="input_model", columns="output_model", values="mean_r")
        if text_order:
            pivot_tv = pivot_tv.reindex(text_order)
        if vision_order:
            pivot_tv = pivot_tv.reindex(columns=vision_order)

    if not df_vt.empty:
        pivot_vt = df_vt.pivot(index="input_model", columns="output_model", values="mean_r")
        if vision_order:
            pivot_vt = pivot_vt.reindex(vision_order)
        if text_order:
            pivot_vt = pivot_vt.reindex(columns=text_order)

    return pivot_tv, pivot_vt  # type: ignore[return-value]


def _direction_comparison_note(
    pivot_tv: pd.DataFrame | None,
    pivot_vt: pd.DataFrame | None,
) -> str | None:
    """Paired Wilcoxon signed-rank between the two prediction directions.

    Each (text encoder, vision encoder) combination is paired: the Text → Vision
    cell ``pivot_tv[t, v]`` against its mirror Vision → Text cell
    ``pivot_vt[v, t]``. Returns a short annotation string (p-value, stars and
    which direction is higher), or ``None`` when the directions cannot be aligned
    or the test is undefined.
    """
    if pivot_tv is None or pivot_vt is None:
        return None
    # Put Vision → Text on the same (text-row, vision-column) grid as Text → Vision
    # so the two arrays are paired cell-for-cell by encoder combination.
    vt_aligned = pivot_vt.T.reindex(index=pivot_tv.index, columns=pivot_tv.columns)
    a = pivot_tv.to_numpy(dtype=float).ravel()
    b = vt_aligned.to_numpy(dtype=float).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2 or np.allclose(a, b):
        return None
    try:
        result = wilcoxon(a, b)
    except ValueError:
        return None
    p = float(cast(float, result[1]))
    sig = significance_label(p)
    return f"Text → Vision vs Vision → Text (paired Wilcoxon): p = {p:.2g} {sig}"


def _plot_single_results(
    df: pd.DataFrame,
    output_dir: Path,
    cfg: DictConfig,
    font_scale: float,
    annotate: bool,
    figsize: tuple[float, float],
) -> None:
    text_order = cfg.get("text_model_order") or []
    vision_order = cfg.get("vision_model_order") or []
    vmin = cfg.get("vmin")
    vmax = cfg.get("vmax")
    # Linear (ridge) vs nonlinear (MLP), inferred from the results' regressor column.
    regressor = str(df["regressor"].iloc[0]) if "regressor" in df.columns and len(df) else ""
    setting = "Linear" if regressor == "ridge" else "Nonlinear"

    # Color each panel by its input modality (text = blue, image = red), matching
    # the modality scheme used across the paper's figures.
    pastel_blue = _make_colormap(["#EEF4FB", "#A8C8EA", "#5B93D1", "#2A6BB5"])
    pastel_red = _make_colormap(["#FCF0EF", "#F0B2AD", "#DD7A73", "#BE3F37"])

    pivot_tv, pivot_vt = _create_pivot_matrices(df, text_order, vision_order)

    # Same paired test annotated on both directions' heatmaps.
    comparison_note = _direction_comparison_note(pivot_tv, pivot_vt)

    if pivot_tv is not None:
        row_labels = [_pretty_label(x) for x in pivot_tv.index.tolist()]
        col_labels = [_pretty_label(x) for x in pivot_tv.columns.tolist()]
        output_path = _resolve_output_path(
            output_dir, cfg.get("text_to_vision_output") or "text_to_vision_heatmap.png"
        )
        _plot_heatmap(
            pivot_tv.values,
            row_labels,
            col_labels,
            title=f"{setting}: Text → Vision prediction (mean Pearson r)",
            xlabel="Output (vision encoders)",
            ylabel="Input (text encoders)",
            cmap=pastel_blue,
            output_path=output_path,
            vmin=vmin,
            vmax=vmax,
            annotate=annotate,
            font_scale=font_scale,
            figsize=figsize,
            comparison_note=comparison_note,
        )
    else:
        logger.warning("No text_to_vision rows found in results.")

    if pivot_vt is not None:
        row_labels = [_pretty_label(x) for x in pivot_vt.index.tolist()]
        col_labels = [_pretty_label(x) for x in pivot_vt.columns.tolist()]
        output_path = _resolve_output_path(
            output_dir, cfg.get("vision_to_text_output") or "vision_to_text_heatmap.png"
        )
        _plot_heatmap(
            pivot_vt.values,
            row_labels,
            col_labels,
            title=f"{setting}: Vision → Text prediction (mean Pearson r)",
            xlabel="Output (text encoders)",
            ylabel="Input (vision encoders)",
            cmap=pastel_red,
            output_path=output_path,
            vmin=vmin,
            vmax=vmax,
            annotate=annotate,
            font_scale=font_scale,
            figsize=figsize,
            comparison_note=comparison_note,
        )
    else:
        logger.warning("No vision_to_text rows found in results.")


def _symmetric_limits(
    data: np.ndarray, half_range: float | None, robust_percentile: float = 98.0
) -> tuple[float, float]:
    """Color bounds centered on 0 so the diverging midpoint (white) maps to 0.

    ``half_range`` sets the symmetric span explicitly. ``None`` derives it from
    the *robust_percentile* of |data| rather than its maximum: these differences
    have a long tail (median 0.016, p95 0.045, max 0.225), and scaling to the
    max leaves the ramp almost unused for the ~95% of cells that matter. A few
    extreme cells clip instead, which the caller reports in the subtitle.
    """
    if half_range is not None:
        lim = float(half_range)
    else:
        lim = float(np.nanpercentile(np.abs(data), robust_percentile))
    if not np.isfinite(lim) or lim <= 0.0:
        lim = 1e-6
    return -lim, lim


def _plot_difference_results(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    label1: str,
    label2: str,
    output_dir: Path,
    cfg: DictConfig,
    font_scale: float,
    annotate: bool,
    figsize: tuple[float, float],
) -> None:
    text_order = cfg.get("text_model_order") or []
    vision_order = cfg.get("vision_model_order") or []
    diff_vmax = cfg.get("diff_vmax")  # optional symmetric half-range cap (None = auto)

    # Colorblind-safe diverging map (amber ↔ purple) for the nonlinear − linear
    # difference; avoids the red↔green pair and the categorical modality hues.
    diverging_cmap = _make_colormap(
        ["#B86D0E", "#E8A24C", "#F6F6F6", "#A988C8", "#7048A4"]
    )

    pivot1_tv, pivot1_vt = _create_pivot_matrices(df1, text_order, vision_order)
    pivot2_tv, pivot2_vt = _create_pivot_matrices(df2, text_order, vision_order)

    have_tv = pivot1_tv is not None and pivot2_tv is not None
    have_vt = pivot1_vt is not None and pivot2_vt is not None
    diff_tv = pivot2_tv.values - pivot1_tv.values if have_tv else None
    diff_vt = pivot2_vt.values - pivot1_vt.values if have_vt else None

    # One symmetric scale across both directions: computed per panel, the two
    # diff heatmaps would use different colour scales and could not be compared
    # with each other, which is the whole point of showing them side by side.
    present = [d for d in (diff_tv, diff_vt) if d is not None]
    if not present:
        logger.warning("Cannot compute either difference (missing data in one or both files).")
        return
    all_diffs = np.concatenate([d.ravel() for d in present])
    vmin_diff, vmax_diff = _symmetric_limits(all_diffs, diff_vmax)
    n_clipped = int(np.sum(np.abs(all_diffs) > vmax_diff))
    clip_note = (
        f"colour scale ±{vmax_diff:.3f}; {n_clipped}/{all_diffs.size} cells clipped"
        if n_clipped
        else None
    )

    if diff_tv is not None:
        row_labels = [_pretty_label(x) for x in pivot2_tv.index.tolist()]
        col_labels = [_pretty_label(x) for x in pivot2_tv.columns.tolist()]
        output_path = _resolve_output_path(
            output_dir,
            cfg.get("text_to_vision_diff_output")
            or f"text_to_vision_diff_{label2}_minus_{label1}.png",
        )
        _plot_heatmap(
            diff_tv,
            row_labels,
            col_labels,
            title=f"Text → Vision: {label2} minus {label1}",
            comparison_note=clip_note,
            xlabel="Output (vision encoders)",
            ylabel="Input (text encoders)",
            cmap=diverging_cmap,
            output_path=output_path,
            vmin=vmin_diff,
            vmax=vmax_diff,
            annotate=annotate,
            font_scale=font_scale,
            figsize=figsize,
        )
    else:
        logger.warning(
            "Cannot compute text_to_vision difference (missing data in one or both files)."
        )

    if diff_vt is not None:
        row_labels = [_pretty_label(x) for x in pivot2_vt.index.tolist()]
        col_labels = [_pretty_label(x) for x in pivot2_vt.columns.tolist()]
        output_path = _resolve_output_path(
            output_dir,
            cfg.get("vision_to_text_diff_output")
            or f"vision_to_text_diff_{label2}_minus_{label1}.png",
        )
        _plot_heatmap(
            diff_vt,
            row_labels,
            col_labels,
            title=f"Vision → Text: {label2} minus {label1}",
            comparison_note=clip_note,
            xlabel="Output (text encoders)",
            ylabel="Input (vision encoders)",
            cmap=diverging_cmap,
            output_path=output_path,
            vmin=vmin_diff,
            vmax=vmax_diff,
            annotate=annotate,
            font_scale=font_scale,
            figsize=figsize,
        )
    else:
        logger.warning(
            "Cannot compute vision_to_text difference (missing data in one or both files)."
        )


@hydra.main(
    version_base=None,
    config_path="../../configs/visualization",
    config_name="visualize_predict_modalities",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.get("output_dir") or (FIGURES_DIR / "predict_modalities"))
    if not output_dir.is_absolute():
        output_dir = PROJ_ROOT / output_dir

    font_scale = float(cfg.get("font_scale", 1.0))
    annotate = bool(cfg.get("annotate", False))
    figsize = tuple(cfg.get("figsize", [10, 8]))

    results_csv_1 = cfg.get("results_csv")
    results_csv_2 = cfg.get("results_csv_2")

    if results_csv_2:
        logger.info("Comparing two results files (linear vs nonlinear or similar)")
        df1 = _load_results_file(Path(results_csv_1))
        df2 = _load_results_file(Path(results_csv_2))

        label1 = str(cfg.get("label_1", "linear"))
        label2 = str(cfg.get("label_2", "nonlinear"))

        _plot_difference_results(
            df1, df2, label1, label2, output_dir, cfg, font_scale, annotate, figsize
        )
    else:
        logger.info("Plotting single results file")
        df = _load_results_file(Path(results_csv_1))
        _plot_single_results(df, output_dir, cfg, font_scale, annotate, figsize)


if __name__ == "__main__":
    main()
