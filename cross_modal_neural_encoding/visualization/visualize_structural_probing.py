"""Visualize structural probing results from structural probing analysis.

Reads the structural probing results produced by the structural probing pipeline,
plots bar charts comparing R² scores across different models and structural
properties, and generates heatmaps showing relationships between properties.

Usage::

    python -m cross_modal_neural_encoding.visualization.visualize_structural_probing \
        results_dir=/path/to/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from cross_modal_neural_encoding.config import FIGURES_DIR, PROJ_ROOT
from cross_modal_neural_encoding.utils import (
    configure_plot_fonts,
    significance_label,
    signflip_pvalue_greater,
)
from cross_modal_neural_encoding.visualization.visualize_encoding_results import (
    TEXT_MODEL_PALETTE,
    VISION_MODEL_PALETTE,
    VLM_MODEL_PALETTE,
    _model_category_rank,
)

configure_plot_fonts()


def load_structural_probing_results(
    results_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load structural probing results from CSV files in the results directory.

    Parameters
    ----------
    results_dir : Path
        Directory containing the results CSV files

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping model names to their results DataFrames
    """
    results: dict[str, pd.DataFrame] = {}
    folds: dict[str, pd.DataFrame] = {}

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find all structural probing results CSV files (recursive)
    csv_files = list(results_dir.rglob("structural_probing_results.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No results CSV files found in: {results_dir}")

    for csv_file in csv_files:
        # Extract model name from the directory path
        model_name = csv_file.parent.name

        try:
            df = pd.read_csv(csv_file)
            results[model_name] = df
            logger.info(f"Loaded results for model {model_name} from {csv_file}")
        except Exception as e:
            logger.warning(f"Failed to load {csv_file}: {e}")
            continue

        fold_file = csv_file.parent / "structural_probing_folds.csv"
        if fold_file.exists():
            try:
                folds[model_name] = pd.read_csv(fold_file)
                logger.info(
                    f"Loaded fold scores for model {model_name} from {fold_file}"
                )
            except Exception as e:
                logger.warning(f"Failed to load {fold_file}: {e}")

    if not results:
        raise ValueError("No valid results files could be loaded")

    return results, folds


def _collect_property_targets(prefix: str) -> dict[str, str]:
    """Map property name to full column key for a given prefix."""
    return {
        "Nodes": f"{prefix}_n_nodes",
        "Edges": f"{prefix}_n_edges",
        "Depth": f"{prefix}_graph_depth",
    }


def _extract_model_scores(
    results: dict[str, pd.DataFrame],
    target_map: dict[str, str],
) -> tuple[list[str], np.ndarray, list[str]]:
    """Return model names, score matrix (models x groups), and group labels."""
    model_names = list(results.keys())
    group_labels = list(target_map.keys()) + ["Average"]
    scores = np.full((len(model_names), len(group_labels)), np.nan, dtype=float)

    for i, model in enumerate(model_names):
        df = results[model]
        for j, prop in enumerate(target_map.keys()):
            col = target_map[prop]
            if col in df.columns:
                scores[i, j] = float(df.iloc[0][col])
        # Average across available properties
        scores[i, -1] = np.nanmean(scores[i, : len(target_map)])

    return model_names, scores, group_labels


def _extract_fold_scores(
    folds: dict[str, pd.DataFrame],
    target_map: dict[str, str],
) -> dict[str, dict[str, np.ndarray]]:
    """Return fold scores per model and property for significance testing."""
    fold_scores: dict[str, dict[str, np.ndarray]] = {}
    for model, df in folds.items():
        model_scores: dict[str, np.ndarray] = {}
        for prop, col in target_map.items():
            if "target" not in df.columns or "r2" not in df.columns:
                continue
            vals = df.loc[df["target"] == col, "r2"].to_numpy(dtype=float)
            if vals.size:
                model_scores[prop] = vals
        # Average across properties per fold (if available)
        if model_scores:
            min_len = min((len(v) for v in model_scores.values()), default=0)
            if min_len > 0:
                stacked = np.vstack([v[:min_len] for v in model_scores.values()])
                model_scores["Average"] = np.nanmean(stacked, axis=0)
        fold_scores[model] = model_scores
    return fold_scores


def create_structural_probing_plots(
    results: dict[str, pd.DataFrame],
    folds: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Create visualization plots for structural probing results.

    Creates two types of plots:
    1. Bar plots comparing R² scores across models and properties
    2. Heatmaps showing relationships between different structural properties

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Dictionary of results DataFrames by model name
    output_dir : Path
        Directory to save the plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        logger.warning("No structural probing results provided.")
        return

    logger.info("Creating grouped bar plots...")

    # Prepare target maps
    text_targets = _collect_property_targets("text_amr")
    vision_targets = _collect_property_targets("vision_coco_a")

    # Extract data
    model_names, text_scores, text_groups = _extract_model_scores(
        results, text_targets
    )
    _, vision_scores, vision_groups = _extract_model_scores(
        results, vision_targets
    )

    # Fold-based significance (optional)
    text_fold_scores = _extract_fold_scores(folds, text_targets)
    vision_fold_scores = _extract_fold_scores(folds, vision_targets)

    n_models = len(model_names)
    category_palettes = {
        0: VLM_MODEL_PALETTE,
        1: VISION_MODEL_PALETTE,
        2: TEXT_MODEL_PALETTE,
    }
    category_counts = {0: 0, 1: 0, 2: 0}
    colors: list[str] = []
    for label in model_names:
        category = _model_category_rank(label)[0]
        palette = category_palettes.get(category, VLM_MODEL_PALETTE)
        idx = category_counts.get(category, 0) % len(palette)
        colors.append(palette[idx])
        category_counts[category] = category_counts.get(category, 0) + 1
    bar_w = min(0.16, 0.82 / max(n_models, 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    panels = [
        (axes[0], "AMR graphs (text)", text_scores, text_groups, text_fold_scores),
        (
            axes[1],
            "COCO-A graphs (vision)",
            vision_scores,
            vision_groups,
            vision_fold_scores,
        ),
    ]

    for ax, title, scores, groups, fold_scores in panels:
        x = np.arange(len(groups))
        offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_w

        if not np.isfinite(scores).any():
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.set_axis_off()
            continue

        for i, model in enumerate(model_names):
            ax.bar(
                x + offsets[i],
                scores[i],
                width=bar_w * 0.95,
                color=colors[i],
                edgecolor="#4A4A4A",
                linewidth=0.5,
                alpha=0.9,
                label=model if ax is axes[0] else None,
                zorder=3,
            )

            # Significance annotations if fold scores available
            model_fold_scores = fold_scores.get(model, {})
            for j, group in enumerate(groups):
                if group not in model_fold_scores:
                    continue
                pval = signflip_pvalue_greater(model_fold_scores[group])
                sig = significance_label(pval)
                if not sig:
                    continue
                val = scores[i, j]
                if not np.isfinite(val):
                    continue
                y_pad = 0.02 * (np.nanmax(scores) - np.nanmin(scores) + 1e-6)
                ax.text(
                    x[j] + offsets[i],
                    val + y_pad,
                    sig,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="black" if sig == "ns" else "darkred",
                    zorder=4,
                )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=0)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.axhline(y=0, color="black", linewidth=0.5, zorder=1)

    axes[0].set_ylabel("R² Score")
    axes[0].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "structural_probing_grouped.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"Saved grouped structural probing plot to {output_path}")

    # Create correlation heatmaps if we have multiple models
    if len(results) > 1:
        logger.info("Creating correlation heatmaps...")

        # Create a combined DataFrame for all models
        all_data = []
        for model_name, df in results.items():
            # Extract the first row (assuming single row per model)
            if len(df) > 0:
                row_data = df.iloc[0].to_dict()
                row_data['model'] = model_name
                all_data.append(row_data)

        if all_data:
            combined_df = pd.DataFrame(all_data)

            # Separate vision and text features for correlation analysis
            vision_cols = [col for col in combined_df.columns if col.startswith('vision_') and col != 'model']
            text_cols = [col for col in combined_df.columns if col.startswith('text_') and col != 'model']

            # Create vision properties correlation heatmap
            if len(vision_cols) > 1:
                vision_data = combined_df[vision_cols]
                fig, ax = plt.subplots(figsize=(8, 6))

                # Create heatmap
                im = ax.imshow(vision_data.corr(), cmap='coolwarm', vmin=-1, vmax=1)

                # Set ticks and labels
                ax.set_xticks(np.arange(len(vision_cols)))
                ax.set_yticks(np.arange(len(vision_cols)))
                ax.set_xticklabels([col.replace('vision_', '').replace('_', ' ').title() for col in vision_cols], rotation=45)
                ax.set_yticklabels([col.replace('vision_', '').replace('_', ' ').title() for col in vision_cols])

                # Add colorbar
                plt.colorbar(im, ax=ax)

                # Add text annotations
                for i in range(len(vision_cols)):
                    for j in range(len(vision_cols)):
                        text = ax.text(j, i, f"{vision_data.corr().iloc[i, j]:.2f}",
                                        ha="center", va="center", color="black")

                ax.set_title('Correlation Between Vision Structural Properties')
                plt.tight_layout()
                output_path = output_dir / "structural_probing_vision_corr.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.success(f"Saved vision correlation heatmap to {output_path}")

            # Create text properties correlation heatmap
            if len(text_cols) > 1:
                text_data = combined_df[text_cols]
                fig, ax = plt.subplots(figsize=(8, 6))

                # Create heatmap
                im = ax.imshow(text_data.corr(), cmap='coolwarm', vmin=-1, vmax=1)

                # Set ticks and labels
                ax.set_xticks(np.arange(len(text_cols)))
                ax.set_yticks(np.arange(len(text_cols)))
                ax.set_xticklabels([col.replace('text_', '').replace('_', ' ').title() for col in text_cols], rotation=45)
                ax.set_yticklabels([col.replace('text_', '').replace('_', ' ').title() for col in text_cols])

                # Add colorbar
                plt.colorbar(im, ax=ax)

                # Add text annotations
                for i in range(len(text_cols)):
                    for j in range(len(text_cols)):
                        text = ax.text(j, i, f"{text_data.corr().iloc[i, j]:.2f}",
                                        ha="center", va="center", color="black")

                ax.set_title('Correlation Between Text Structural Properties')
                plt.tight_layout()
                output_path = output_dir / "structural_probing_text_corr.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.success(f"Saved text correlation heatmap to {output_path}")


def main(results_dir: str | Path, output_dir: str | Path | None = None) -> None:
    """Main function to run the structural probing visualization pipeline.

    Parameters
    ----------
    results_dir : str or Path
        Path to the directory containing structural probing results
    output_dir : str or Path, optional
        Path to save the visualizations. If not provided, uses the default
        figures directory.
    """
    results_dir = Path(results_dir)

    # Set default output directory if not provided
    if output_dir is None:
        output_dir = FIGURES_DIR / "structural_probing"
    else:
        output_dir = Path(output_dir)

    logger.info(f"Loading structural probing results from {results_dir}")

    # Load results
    results, folds = load_structural_probing_results(results_dir)

    logger.info(f"Creating visualizations and saving to {output_dir}")

    # Create plots
    create_structural_probing_plots(results, folds, output_dir)

    logger.success("Structural probing visualization complete!")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize structural probing results")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=PROJ_ROOT / "outputs/modeling/structural_probing",
        help="Directory containing the structural probing results"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save the visualizations (default: figures/structural_probing)"
    )

    args = parser.parse_args()

    main(args.results_dir, args.output_dir)