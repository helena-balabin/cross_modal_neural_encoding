"""Visualise neural-encoding results from an aggregated CSV as a bar chart.

Reads the aggregated results produced by the neural-encoding pipeline,
plots a bar chart for the selected metric across conditions, annotates
statistical significance (``ns`` for non-significant), and draws the
noise ceiling (``max_ev``) as a horizontal reference line.

Usage::

    python -m cross_modal_neural_encoding.visualization.visualize_encoding_results \
        aggregated_csv=/path/to/aggregated.csv
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from cross_modal_neural_encoding.config import FIGURES_DIR

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

CONDITION_LABELS: dict[str, str] = {
    "image_to_image": "Image → Image",
    "image_to_text": "Image → Text",
    "text_to_image": "Text → Image",
    "text_to_text": "Text → Text",
}

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


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


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════


def plot_encoding_results(
    df: pd.DataFrame,
    *,
    metric: str = "mean_r",
    p_value_col: str = "p_value_mean_r",
    alpha: float = 0.05,
    output_path: Path | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> None:
    """Create a bar chart with significance annotations and a noise-ceiling line.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``MultiIndex`` on the columns (metric, stat) where
        *stat* ∈ {``mean``, ``std``}.
    metric : str
        Column name for the bar heights (e.g. ``"mean_r"``).
    p_value_col : str
        Column name that holds the p-values used for significance markers.
    alpha : float
        Significance threshold.
    output_path : Path | None
        Where to save the figure. ``None`` → ``reports/figures/encoding_results.png``.
    figsize : tuple[float, float]
        Figure size in inches ``(width, height)``.
    """
    conditions = df.index.tolist()
    values = df[(metric, "mean")].values.astype(float)

    # Standard deviations (may be all-NaN when only one subject)
    stds = df[(metric, "std")].values.astype(float)
    has_error = not np.all(np.isnan(stds))

    # p-values for significance annotations
    p_values = df[(p_value_col, "mean")].values.astype(float)

    # Noise ceiling – average max_ev across conditions
    max_ev_values = df[("max_ev", "mean")].values.astype(float)
    max_ev = float(np.nanmean(max_ev_values))  # type: ignore

    # ── Draw ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(conditions))
    labels = [CONDITION_LABELS.get(c, c) for c in conditions]
    colors = (PALETTE * ((len(conditions) // len(PALETTE)) + 1))[: len(conditions)]

    bars = ax.bar(
        x,
        values,  # type: ignore
        yerr=stds if has_error else None,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.6,
        zorder=3,
    )

    # Noise-ceiling horizontal line
    ax.axhline(
        y=max_ev,
        color="dimgray",
        linestyle="--",
        linewidth=1.5,
        label=f"Noise ceiling (max EV = {max_ev:.4f})",
        zorder=2,
    )

    # ── Significance annotations ──────────────────────────────────────────
    y_pad = (max(np.nanmax(np.abs(values)), max_ev) - 0) * 0.04
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
            fontsize=11,
            fontweight="bold",
            color="black" if sig == "ns" else "darkred",
        )

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)  # type: ignore
    ax.set_ylabel("Pearson Correlation", fontsize=12)
    ax.set_title("Neural Encoding Results", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.axhline(y=0, color="black", linewidth=0.5, zorder=1)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()

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
    """Load aggregated CSV and produce the bar-chart figure."""
    path = Path(cfg.aggregated_csv)
    if not path.exists():
        raise FileNotFoundError(f"Aggregated CSV not found: {path}")

    logger.info(f"Loading aggregated results from {path}")
    df = load_aggregated(path)
    logger.info(f"Conditions: {df.index.tolist()}")

    output_path = Path(cfg.output_path) if cfg.output_path else None

    plot_encoding_results(
        df,
        metric=cfg.metric,
        p_value_col=cfg.p_value_col,
        alpha=cfg.alpha,
        output_path=output_path,
        figsize=tuple(cfg.figsize),
    )


if __name__ == "__main__":
    main()
