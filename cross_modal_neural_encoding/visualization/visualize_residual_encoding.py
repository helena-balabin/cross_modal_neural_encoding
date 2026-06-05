"""Visualize residual neural encoding results (§3.3).

Produces a grouped bar chart comparing:
    - Standard (unablated) encoding accuracy
    - Structure-residualised encoding accuracy
    - Permuted-s control (optional)

for each of the four 2×2 conditions.  The expected signature of compositional
structure as the carrier of cross-modal alignment is a **selective drop** in
cross-modal conditions after residualisation, while within-modality conditions
remain largely unchanged.

Usage::

    python -m cross_modal_neural_encoding.visualization.visualize_residual_encoding \\
        standard_summary=/path/to/neural_encoding/.../summary.csv \\
        residual_summary=/path/to/residual_encoding/.../summary.csv
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
    configure_plot_fonts,
    signflip_pvalue_greater,
    significance_label,
)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Colours for the three bar groups
COLOR_STANDARD = "#7EAEDB"  # blue — unablated baseline
COLOR_RESIDUAL = "#E88989"  # red  — structure residualised
COLOR_PERMUTED = "#EFBF63"  # amber — permuted-s control

configure_plot_fonts()

METRIC = "mean_normalized_r"


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════


def _load_per_subject(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _condition_means(
    df: pd.DataFrame,
    conditions: list[str],
    metric: str = METRIC,
) -> dict[str, np.ndarray]:
    """Return per-subject metric values for each condition."""
    out: dict[str, np.ndarray] = {}
    for cond in conditions:
        vals = df.loc[df["condition"] == cond, metric].values.astype(float)
        out[cond] = np.array(vals)
    return out


def _pvalue(vals: np.ndarray) -> float:
    return signflip_pvalue_greater(vals[np.isfinite(vals)])


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════


def plot_residual_comparison(
    standard_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    *,
    metric: str = METRIC,
    output_path: Path,
    show_permuted: bool = True,
) -> None:
    """Create grouped bar chart: unablated vs. residualised (vs. permuted).

    Each group of bars corresponds to one encoding condition.  The drop from
    the standard bar to the residual bar (and the absence of a similar drop in
    the permuted control) is the key diagnostic.
    """
    base_conditions = list(CONDITION_LABELS.keys())
    # Keep only conditions present in both dataframes
    std_conds = set(standard_df["condition"].unique())
    res_conds = set(residual_df["condition"].unique())
    conditions = [c for c in base_conditions if c in std_conds and c in res_conds]
    permuted_conditions = [f"permuted_{c}" for c in conditions]
    has_permuted = show_permuted and all(pc in res_conds for pc in permuted_conditions)

    n_conds = len(conditions)
    n_bars = 3 if has_permuted else 2
    bar_w = 0.22
    group_w = n_bars * bar_w + 0.08
    x = np.arange(n_conds) * group_w

    fig, ax = plt.subplots(figsize=(2.6 * n_conds, 4.0))

    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_w

    bar_specs = [(COLOR_STANDARD, "Standard"), (COLOR_RESIDUAL, "Residualised")]
    if has_permuted:
        bar_specs.append((COLOR_PERMUTED, "Permuted control"))

    for bi, (color, label) in enumerate(bar_specs):
        means, sems, pvals = [], [], []
        for ci, cond in enumerate(conditions):
            if bi == 0:
                vals = _condition_means(standard_df, [cond], metric)[cond]
            elif bi == 1:
                vals = _condition_means(residual_df, [cond], metric)[cond]
            else:
                vals = _condition_means(residual_df, [f"permuted_{cond}"], metric)[
                    f"permuted_{cond}"
                ]

            finite = vals[np.isfinite(vals)]
            means.append(float(np.mean(finite)) if len(finite) else np.nan)
            sems.append(float(np.std(finite) / np.sqrt(len(finite))) if len(finite) > 1 else 0.0)
            pvals.append(_pvalue(vals))

        means_arr = np.array(means)
        sems_arr = np.array(sems)
        bars = ax.bar(
            x + offsets[bi],
            means_arr,
            width=bar_w,
            color=color,
            label=label,
            yerr=sems_arr,
            capsize=3,
            error_kw={"linewidth": 0.8},
            zorder=3,
        )

        # Significance annotations above each bar
        for xi, (bar, p) in enumerate(zip(bars, pvals)):
            sig = significance_label(p)
            if sig:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    (bar.get_height() + sems_arr[xi]) * 1.02,
                    sig,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="black",
                )

    ax.axhline(0, color="black", linewidth=0.6, zorder=2)
    ax.axhline(
        1.0, color="grey", linewidth=0.7, linestyle="--", zorder=1, label="NC reference (1.0)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS.get(c, c) for c in conditions],
        fontsize=9,
    )
    ax.set_ylabel("Noise-ceiling-normalised r", fontsize=10)
    ax.set_title("Structure residualisation: encoding accuracy comparison", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Residual encoding plot → {output_path}")


def plot_ablation_delta(
    standard_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    *,
    metric: str = METRIC,
    output_path: Path,
    show_permuted: bool = True,
) -> None:
    """Plot the change in encoding accuracy after residualisation (Δr/NC).

    Positive values mean the residualised model performs *better* (unexpected);
    negative values mean ablation reduced performance.  The key pattern:
    cross-modal conditions should show a larger negative Δ than within-modal.
    """
    base_conditions = list(CONDITION_LABELS.keys())
    std_conds = set(standard_df["condition"].unique())
    res_conds = set(residual_df["condition"].unique())
    conditions = [c for c in base_conditions if c in std_conds and c in res_conds]
    permuted_conditions = [f"permuted_{c}" for c in conditions]
    has_permuted = show_permuted and all(pc in res_conds for pc in permuted_conditions)

    n_conds = len(conditions)
    n_bars = 2 if has_permuted else 1
    bar_w = 0.28
    group_w = n_bars * bar_w + 0.10
    x = np.arange(n_conds) * group_w
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_w

    fig, ax = plt.subplots(figsize=(2.4 * n_conds, 4.0))

    for bi, (color, desc, cond_list) in enumerate(
        [
            (COLOR_RESIDUAL, "Residualised − Standard", conditions),
            *(
                [(COLOR_PERMUTED, "Permuted − Standard", permuted_conditions)]
                if has_permuted
                else []
            ),
        ]
    ):
        deltas, sems = [], []
        for cond, res_cond in zip(conditions, cond_list):
            std_vals = _condition_means(standard_df, [cond], metric)[cond]
            res_vals = _condition_means(residual_df, [res_cond], metric)[res_cond]
            # Pair by subject order
            n = min(len(std_vals), len(res_vals))
            d = res_vals[:n] - std_vals[:n]
            finite = d[np.isfinite(d)]
            deltas.append(float(np.mean(finite)) if len(finite) else np.nan)
            sems.append(float(np.std(finite) / np.sqrt(len(finite))) if len(finite) > 1 else 0.0)

        deltas_arr = np.array(deltas)
        sems_arr = np.array(sems)
        ax.bar(
            x + offsets[bi],
            deltas_arr,
            width=bar_w,
            color=color,
            label=desc,
            yerr=sems_arr,
            capsize=3,
            error_kw={"linewidth": 0.8},
            zorder=3,
        )

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS.get(c, c) for c in conditions],
        fontsize=9,
    )
    ax.set_ylabel("Δ noise-ceiling-normalised r", fontsize=10)
    ax.set_title("Ablation effect: residualised minus standard encoding accuracy", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Ablation delta plot → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


@hydra.main(
    version_base=None,
    config_path="../../configs/visualization",
    config_name="visualize_residual_encoding",
)
def main(cfg: DictConfig) -> None:
    standard_path = Path(cfg.standard_summary)
    residual_path = Path(cfg.residual_summary)
    output_dir = Path(cfg.get("output_dir", str(FIGURES_DIR / "residual_encoding")))
    if not output_dir.is_absolute():
        output_dir = PROJ_ROOT / output_dir
    metric: str = cfg.get("metric", METRIC)
    show_permuted: bool = bool(cfg.get("show_permuted", True))

    if not standard_path.exists():
        raise FileNotFoundError(f"standard_summary not found: {standard_path}")
    if not residual_path.exists():
        raise FileNotFoundError(f"residual_summary not found: {residual_path}")

    standard_df = _load_per_subject(standard_path)
    residual_df = _load_per_subject(residual_path)

    model_label = residual_path.parent.name
    out_prefix = output_dir / model_label

    plot_residual_comparison(
        standard_df,
        residual_df,
        metric=metric,
        output_path=out_prefix / "residual_comparison.png",
        show_permuted=show_permuted,
    )
    plot_ablation_delta(
        standard_df,
        residual_df,
        metric=metric,
        output_path=out_prefix / "ablation_delta.png",
        show_permuted=show_permuted,
    )


if __name__ == "__main__":
    main()
