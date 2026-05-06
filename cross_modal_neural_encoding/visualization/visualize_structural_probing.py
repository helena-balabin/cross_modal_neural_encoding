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
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from cross_modal_neural_encoding.config import FIGURES_DIR, PROJ_ROOT


def load_structural_probing_results(results_dir: Path) -> dict[str, pd.DataFrame]:
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
    results = {}

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find all CSV files in the results directory
    csv_files = list(results_dir.glob("*_results.csv"))

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

    if not results:
        raise ValueError("No valid results files could be loaded")

    return results


def create_structural_probing_plots(results: dict[str, pd.DataFrame],
                                    output_dir: Path) -> None:
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

    # Extract all target names from the results
    all_targets = set()
    for df in results.values():
        all_targets.update(df.columns)

    # Separate vision and text targets
    vision_targets = [t for t in all_targets if t.startswith('vision_')]
    text_targets = [t for t in all_targets if t.startswith('text_')]

    # Sort targets for consistent ordering
    vision_targets.sort()
    text_targets.sort()

    # Create bar plots
    logger.info("Creating bar plots...")

    # Plot for vision targets
    if vision_targets:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Prepare data
        model_names = list(results.keys())
        n_models = len(model_names)
        n_targets = len(vision_targets)

        # Create grouped bar plot
        x = np.arange(n_models)
        width = 0.8 / n_targets  # Width of each bar

        # Colors for different targets
        colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, n_targets))

        # Plot each target as a group of bars
        for i, target in enumerate(vision_targets):
            scores = []
            for model in model_names:
                if target in results[model].columns:
                    # Take the first (and usually only) row
                    score = results[model].iloc[0][target]
                else:
                    score = np.nan
                scores.append(score)

            # Plot bars for this target
            ax.bar(x + (i - n_targets/2 + 0.5) * width,
                    scores,
                    width,
                    label=target.replace('vision_', '').replace('_', ' ').title(),
                    color=colors[i],
                    alpha=0.8)

        ax.set_xlabel('Model')
        ax.set_ylabel('R² Score')
        ax.set_title('Vision Embedding Structural Probing Results')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        output_path = output_dir / "structural_probing_vision.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.success(f"Saved vision probing plot to {output_path}")

    # Plot for text targets
    if text_targets:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Prepare data
        model_names = list(results.keys())
        n_models = len(model_names)
        n_targets = len(text_targets)

        # Create grouped bar plot
        x = np.arange(n_models)
        width = 0.8 / n_targets  # Width of each bar

        # Colors for different targets
        colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, n_targets))

        # Plot each target as a group of bars
        for i, target in enumerate(text_targets):
            scores = []
            for model in model_names:
                if target in results[model].columns:
                    # Take the first (and usually only) row
                    score = results[model].iloc[0][target]
                else:
                    score = np.nan
                scores.append(score)

            # Plot bars for this target
            ax.bar(x + (i - n_targets/2 + 0.5) * width,
                    scores,
                    width,
                    label=target.replace('text_', '').replace('_', ' ').title(),
                    color=colors[i],
                    alpha=0.8)

        ax.set_xlabel('Model')
        ax.set_ylabel('R² Score')
        ax.set_title('Text Embedding Structural Probing Results')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        output_path = output_dir / "structural_probing_text.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.success(f"Saved text probing plot to {output_path}")

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
    results = load_structural_probing_results(results_dir)

    logger.info(f"Creating visualizations and saving to {output_dir}")

    # Create plots
    create_structural_probing_plots(results, output_dir)

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