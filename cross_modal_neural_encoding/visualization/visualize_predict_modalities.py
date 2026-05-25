"""Visualize cross-modal prediction results as heatmaps.

Creates two heatmaps:
1) Text → Vision (rows: text encoders, columns: vision encoders)
2) Vision → Text (rows: vision encoders, columns: text encoders)

Usage
-----
    python -m cross_modal_neural_encoding.visualization.visualize_predict_modalities

Hydra config: ``configs/visualization/visualize_predict_modalities.yaml``
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import DictConfig

from cross_modal_neural_encoding.config import FIGURES_DIR, PROJ_ROOT
from cross_modal_neural_encoding.utils import configure_plot_fonts

configure_plot_fonts()


def _pretty_label(label: str) -> str:
    return label.replace("--", "/")


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
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=10 * font_scale)
    ax.set_yticklabels(row_labels, fontsize=10 * font_scale)
    ax.set_title(title, fontsize=13 * font_scale, pad=10)
    ax.set_xlabel(xlabel, fontsize=11 * font_scale, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11 * font_scale, labelpad=8)

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
                        fontsize=7 * font_scale,
                        color="#1f2d3d",
                    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10 * font_scale)
    cbar.set_label("Mean Pearson r", fontsize=11 * font_scale)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.success(f"Saved heatmap → {output_path}")


@hydra.main(
    version_base=None,
    config_path="../../configs/visualization",
    config_name="visualize_predict_modalities",
)
def main(cfg: DictConfig) -> None:
    results_path = Path(cfg.results_csv)
    if not results_path.is_absolute():
        results_path = PROJ_ROOT / results_path

    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_path}")

    output_dir = Path(cfg.get("output_dir") or (FIGURES_DIR / "predict_modalities"))
    if not output_dir.is_absolute():
        output_dir = PROJ_ROOT / output_dir

    df = pd.read_csv(results_path)
    if "direction" not in df.columns:
        raise ValueError("Results CSV must contain a 'direction' column.")

    font_scale = float(cfg.get("font_scale", 1.0))
    annotate = bool(cfg.get("annotate", False))
    vmin = cfg.get("vmin")
    vmax = cfg.get("vmax")
    figsize = tuple(cfg.get("figsize", [10, 8]))

    text_order = cfg.get("text_model_order") or []
    vision_order = cfg.get("vision_model_order") or []

    pastel_blue = _make_colormap(["#F2F6FB", "#BBD4F0", "#6FA8DC"])
    pastel_green = _make_colormap(["#F1FAF4", "#BFE3C7", "#6FCF97"])

    # Text -> Vision heatmap
    df_tv = df[df["direction"] == "text_to_vision"].copy()
    if df_tv.empty:
        logger.warning("No text_to_vision rows found in results.")
    else:
        pivot_tv = df_tv.pivot(
            index="input_model", columns="output_model", values="mean_r"
        )
        if text_order:
            pivot_tv = pivot_tv.reindex(text_order)
        if vision_order:
            pivot_tv = pivot_tv.reindex(columns=vision_order)

        row_labels = [_pretty_label(x) for x in pivot_tv.index.tolist()]
        col_labels = [_pretty_label(x) for x in pivot_tv.columns.tolist()]
        output_path = _resolve_output_path(
            output_dir, cfg.get("text_to_vision_output") or "text_to_vision_heatmap.png"
        )
        _plot_heatmap(
            pivot_tv.values,
            row_labels,
            col_labels,
            title="Text → Vision prediction (mean Pearson r)",
            xlabel="Output (vision encoders)",
            ylabel="Input (text encoders)",
            cmap=pastel_blue,
            output_path=output_path,
            vmin=vmin,
            vmax=vmax,
            annotate=annotate,
            font_scale=font_scale,
            figsize=figsize,
        )

    # Vision -> Text heatmap
    df_vt = df[df["direction"] == "vision_to_text"].copy()
    if df_vt.empty:
        logger.warning("No vision_to_text rows found in results.")
    else:
        pivot_vt = df_vt.pivot(
            index="input_model", columns="output_model", values="mean_r"
        )
        if vision_order:
            pivot_vt = pivot_vt.reindex(vision_order)
        if text_order:
            pivot_vt = pivot_vt.reindex(columns=text_order)

        row_labels = [_pretty_label(x) for x in pivot_vt.index.tolist()]
        col_labels = [_pretty_label(x) for x in pivot_vt.columns.tolist()]
        output_path = _resolve_output_path(
            output_dir, cfg.get("vision_to_text_output") or "vision_to_text_heatmap.png"
        )
        _plot_heatmap(
            pivot_vt.values,
            row_labels,
            col_labels,
            title="Vision → Text prediction (mean Pearson r)",
            xlabel="Output (text encoders)",
            ylabel="Input (vision encoders)",
            cmap=pastel_green,
            output_path=output_path,
            vmin=vmin,
            vmax=vmax,
            annotate=annotate,
            font_scale=font_scale,
            figsize=figsize,
        )


if __name__ == "__main__":
    main()
