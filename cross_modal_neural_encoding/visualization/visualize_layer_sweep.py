"""Plot encoding performance as a function of encoder layer depth.

Companion to :mod:`visualize_encoding_results`, which shows one bar per model at
a single hard-coded middle layer. This module consumes the sweep produced by
``scripts/neural_encoding_layer_sweep.sh`` — the same encoding analysis re-run at
every layer of one model — and draws the depth curves.

One panel per condition, in a 2x2 grid whose **rows are the embedding modality**
(which is what sets the x-axis) and whose **columns are the fMRI modality** being
predicted, so the diagonal is within-modality and the off-diagonal cross-modal:

======================  =====================  =====================
x-axis                  → image fMRI           → text fMRI
======================  =====================  =====================
vision (blocks)         ``image_to_image``     ``image_to_text``
text (hidden states)    ``text_to_image``      ``text_to_text``
======================  =====================  =====================

The two rows' x-axes are *not* interchangeable: a vision index is the output of
transformer block *i*, whereas a text index is ``hidden_states[i]``, so index 0
is the input embedding and index *i* is the output of block *i-1*.

Every panel scales its own y-axis, because the conditions differ by an order of
magnitude and a shared axis flattens the weak ones into featureless lines.

Lines are keyed to the palette semantics used across the paper: colour is the
fMRI modality being predicted (image = red, text = blue) and linestyle separates
within-modality (solid) from cross-modal (dashed) prediction.

Usage::

    python -m cross_modal_neural_encoding.visualization.visualize_layer_sweep \
        run_dir=outputs/neural_encoding_layer_sweep/run_array_123456
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

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
    signflip_pvalue_greater,
)
from cross_modal_neural_encoding.visualization.visualize_encoding_results import (
    combined_perm_group_pvalue,
    load_summary,
)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Colour encodes the fMRI modality being predicted, matching the convention in
# docs/09_visualization.md (text = blue, image = red). Green is deliberately
# unused here — it is reserved for brain space in the overview schematic.
FMRI_MODALITY_COLORS: dict[str, str] = {
    "image": "#D96F6F",  # muted crimson, as in VISION_MODEL_PALETTE
    "text": "#6F9FC9",  # muted blue, as in TEXT_MODEL_PALETTE
}

# One panel per embedding modality: (title, x-axis label). The two labels differ
# because the layer indices mean different things — see the module docstring.
PANEL_SPECS: tuple[tuple[str, str, str], ...] = (
    ("vision", "Vision embeddings", "Transformer block"),
    ("text", "Text embeddings", "Hidden state (0 = input embeddings)"),
)

# ``<embed_modality>_layer_<idx>`` directories written by the sweep script.
LAYER_DIR_PATTERN = re.compile(r"^(vision|text)_layer_(\d+)$")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def is_within_modality(embed_modality: str, fmri_modality: str) -> bool:
    """Whether embeddings and fMRI come from the same stimulus modality.

    ``vision`` embeddings are extracted from the images and ``text`` embeddings
    from the captions, so the naming differs on the fMRI side (``image`` vs
    ``text``) from the embedding side (``vision`` vs ``text``).
    """
    return (embed_modality, fmri_modality) in {("vision", "image"), ("text", "text")}


def _layer_dirs(run_dir: Path) -> list[tuple[str, int, Path]]:
    """Find ``<modality>_layer_<idx>`` directories under *run_dir*.

    Falls back to searching ``run_array_*`` subfolders, mirroring
    :func:`visualize_encoding_results._collect_model_dirs`, so a sweep root
    holding several array runs also works.
    """

    def scan(parent: Path) -> list[tuple[str, int, Path]]:
        found: list[tuple[str, int, Path]] = []
        for child in sorted(parent.iterdir()):
            if not child.is_dir():
                continue
            match = LAYER_DIR_PATTERN.match(child.name)
            if match:
                found.append((match.group(1), int(match.group(2)), child))
        return found

    direct = scan(run_dir)
    if direct:
        return direct

    nested: list[tuple[str, int, Path]] = []
    for child in sorted(run_dir.iterdir()):
        if child.is_dir() and child.name.startswith("run_array_"):
            nested.extend(scan(child))
    return nested


def _resolve_model_label(layer_dirs: list[tuple[str, int, Path]], model_label: str | None) -> str:
    """Return the model subfolder name, auto-detecting when not configured."""
    if model_label:
        return model_label

    candidates = {
        child.name
        for _, _, layer_dir in layer_dirs
        for child in layer_dir.iterdir()
        if child.is_dir() and (child / "summary.csv").exists()
    }
    if not candidates:
        raise FileNotFoundError("No model subfolder with a summary.csv found in the layer dirs.")
    if len(candidates) > 1:
        raise ValueError(
            "Multiple models found in the sweep "
            f"({', '.join(sorted(candidates))}); set model_label to pick one."
        )
    return candidates.pop()


def _layer_from_rows(rows: pd.DataFrame, embed_modality: str, dir_layer: int) -> int:
    """Layer index for a condition, preferring the columns in ``summary.csv``.

    ``embed_layer`` is written by the encoding pipeline and is the layer the
    condition actually used; ``<modality>_layer`` is the older fallback. Both are
    absent from CSVs written before that change, in which case the directory name
    is the only source.
    """
    for column in ("embed_layer", f"{embed_modality}_layer"):
        if column in rows.columns:
            value = rows[column].iloc[0]
            if pd.notna(value):
                return int(value)
    return dir_layer


# ═══════════════════════════════════════════════════════════════════════════
# Collection
# ═══════════════════════════════════════════════════════════════════════════


def collect_layer_results(
    run_dir: Path,
    *,
    model_label: str | None,
    metric: str,
    group_sig_permutations: int,
    group_sig_random_state: int,
    group_sig_correction: str,
) -> tuple[pd.DataFrame, str]:
    """Build the per-(layer, condition) table behind the curves.

    Each row aggregates one condition at one layer across subjects: the group
    mean of *metric*, its SEM, and a group-level p-value. Significance reuses
    :func:`combined_perm_group_pvalue` — the Stelzer-style combination of the
    per-subject permutation nulls — so the stars mean exactly what they mean in
    the main figure, falling back to a sign-flip test where the ``null_mean_r``
    files were not copied back.
    """
    layer_dirs = _layer_dirs(run_dir)
    if not layer_dirs:
        raise FileNotFoundError(
            f"No <modality>_layer_<idx> directories found under {run_dir}. "
            "Did scripts/neural_encoding_layer_sweep.sh finish?"
        )
    resolved_label = _resolve_model_label(layer_dirs, model_label)
    logger.info(f"Found {len(layer_dirs)} layer directories for model {resolved_label}")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()

    for dir_modality, dir_layer, layer_dir in layer_dirs:
        model_dir = layer_dir / resolved_label
        summary_path = model_dir / "summary.csv"
        if not summary_path.exists():
            logger.warning(f"Skipping {layer_dir.name}: no summary.csv for {resolved_label}")
            continue

        summary_df = load_summary(summary_path)
        if metric not in summary_df.columns:
            raise ValueError(f"{summary_path} has no '{metric}' column.")

        for condition in summary_df["condition"].unique():
            cond_rows = summary_df[summary_df["condition"] == condition]
            embed_modality = str(cond_rows["embed_modality"].iloc[0])

            # A task sweeping vision layers still carries a placeholder text
            # layer. Its text conditions are normally deleted by the Hydra
            # overrides, but if a run kept them they would be identical copies
            # of the placeholder layer repeated across every vision directory —
            # so drop anything whose encoder this directory was not sweeping.
            if embed_modality != dir_modality:
                continue

            layer = _layer_from_rows(cond_rows, embed_modality, dir_layer)
            key = (embed_modality, layer, str(condition))
            if key in seen:
                logger.warning(
                    f"Duplicate result for {key}; keeping the first and skipping {layer_dir.name}"
                )
                continue
            seen.add(key)

            values = np.asarray(cond_rows[metric], dtype=float)
            values = values[np.isfinite(values)]
            n_subjects = values.size
            if n_subjects == 0:
                logger.warning(f"No finite '{metric}' values for {key}; skipping")
                continue

            p_value = combined_perm_group_pvalue(
                model_dir,
                summary_df,
                str(condition),
                n_group_draws=group_sig_permutations,
                random_state=group_sig_random_state,
            )
            if not np.isfinite(p_value):
                p_value = signflip_pvalue_greater(
                    values,
                    n_permutations=group_sig_permutations,
                    random_state=group_sig_random_state,
                )

            rows.append(
                {
                    "embed_modality": embed_modality,
                    "fmri_modality": str(cond_rows["fmri_modality"].iloc[0]),
                    "layer": layer,
                    "condition": str(condition),
                    "n_subjects": n_subjects,
                    metric: float(np.mean(values)),
                    # ddof=1: these 8 subjects are a sample, not the population.
                    "sem": (
                        float(np.std(values, ddof=1) / np.sqrt(n_subjects))
                        if n_subjects > 1
                        else np.nan
                    ),
                    "mean_r": float(np.nanmean(np.asarray(cond_rows["mean_r"], dtype=float))),
                    "p_value": float(p_value),
                }
            )

    if not rows:
        raise ValueError(f"No usable results collected from {run_dir}.")

    results = pd.DataFrame(rows)
    # One correction family: every layer x condition cell drawn in the figure.
    if group_sig_correction == "fdr_bh":
        results["q_value"] = benjamini_hochberg(np.asarray(results["p_value"], dtype=float))
    else:
        results["q_value"] = results["p_value"]

    results = results.sort_values(["embed_modality", "condition", "layer"]).reset_index(drop=True)
    return results, resolved_label


def _log_peaks(results: pd.DataFrame, metric: str, main_analysis_layers: dict[str, int]) -> None:
    """Log the peak layer per condition against the published middle layer."""
    for condition, group in results.groupby("condition"):
        best = group.loc[group[metric].idxmax()]
        modality = str(best["embed_modality"])
        middle = main_analysis_layers.get(modality)
        middle_rows = group[group["layer"] == middle] if middle is not None else group.iloc[0:0]
        middle_note = (
            f", main-analysis layer {middle}: {float(middle_rows[metric].iloc[0]):.4f}"
            if len(middle_rows)
            else ""
        )
        logger.info(
            f"{condition}: peak at {modality} layer {int(best['layer'])} "  # type: ignore[index]
            f"({metric} = {float(best[metric]):.4f}){middle_note}"  # type: ignore[index]
        )


# ═══════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════


def _layer_ticks(layers: np.ndarray) -> np.ndarray:
    """Tick positions that stay legible when an encoder has ~33 layers."""
    low, high = int(layers.min()), int(layers.max())
    span = high - low + 1
    step = 1 if span <= 12 else 2 if span <= 20 else 4
    ticks = np.arange(low, high + 1, step)
    # Always show the deepest layer; it anchors the right edge of the curve.
    if ticks[-1] != high:
        ticks = np.append(ticks, high)
    return ticks


def plot_layer_curves(
    results: pd.DataFrame,
    *,
    metric: str,
    model_label: str,
    alpha: float,
    main_analysis_layers: dict[str, int],
    y_limits: tuple[float, float] | None,
    figsize: tuple[float, float],
    font_scale: float,
    output_path: Path,
) -> None:
    """Draw the 2x2 layer-depth figure (one condition per panel) and save it.

    Rows are the embedding modality, which is what sets the x-axis; columns are
    the fMRI modality being predicted. The diagonal is therefore within-modality
    and the off-diagonal is cross-modal.

    Each panel scales its own y-axis. The four conditions differ by an order of
    magnitude — ``image_to_text`` hugs zero while ``image_to_image`` reaches
    ~0.09 — so one shared axis flattens the weak conditions into featureless
    lines and hides whatever layer structure they have. The cost is that heights
    are no longer comparable across panels, which is why each keeps its own tick
    labels and the caption says so.
    """
    row_specs = [spec for spec in PANEL_SPECS if (results["embed_modality"] == spec[0]).any()]
    if not row_specs:
        raise ValueError("No panel matches the collected embedding modalities.")
    # Column = fMRI modality, ordered as in CONDITION_LABELS so the layout lines
    # up with the bar figures.
    ordered_conditions = [c for c in CONDITION_LABELS if c in set(results["condition"])]
    cond_to_fmri = dict(zip(results["condition"], results["fmri_modality"]))
    col_modalities = list(dict.fromkeys(cond_to_fmri[c] for c in ordered_conditions))

    fig, axes = plt.subplots(len(row_specs), len(col_modalities), figsize=figsize, squeeze=False)

    is_normalized = "normalized" in metric.lower()
    y_label = "Normalized performance\n(r / NC)" if is_normalized else "Pearson correlation"

    for row, (modality, _row_title, x_label) in enumerate(row_specs):
        panel_df = results[results["embed_modality"] == modality]

        # Reference layer, skipped when a half-finished sweep has not reached it.
        observed_layers = np.asarray(panel_df["layer"], dtype=int)
        middle = main_analysis_layers.get(modality)
        if middle is not None and not (observed_layers.min() <= middle <= observed_layers.max()):
            logger.warning(
                f"{modality}: main-analysis layer {middle} outside the swept range "
                f"{observed_layers.min()}-{observed_layers.max()}; omitting the reference line"
            )
            middle = None

        for col, fmri_modality in enumerate(col_modalities):
            ax = axes[row][col]
            cond_df = panel_df[panel_df["fmri_modality"] == fmri_modality].sort_values("layer")
            if cond_df.empty:
                ax.set_visible(False)
                continue

            condition = str(cond_df["condition"].iloc[0])
            layers = np.asarray(cond_df["layer"], dtype=float)
            values = np.asarray(cond_df[metric], dtype=float)
            sems = np.nan_to_num(np.asarray(cond_df["sem"], dtype=float))
            qvals = np.asarray(cond_df["q_value"], dtype=float)

            color = FMRI_MODALITY_COLORS.get(fmri_modality, "#888888")
            within = is_within_modality(modality, fmri_modality)

            ax.fill_between(layers, values - sems, values + sems, color=color, alpha=0.18, lw=0)
            ax.plot(
                layers,
                values,
                color=color,
                linestyle="-" if within else "--",
                linewidth=2.0,
                zorder=3,
            )
            # Marker fill carries significance: an asterisk at every layer would
            # be unreadable, and the curve already occupies the vertical space.
            significant = np.isfinite(qvals) & (qvals < alpha)
            ax.plot(
                layers[significant],
                values[significant],
                linestyle="none",
                marker="o",
                markersize=4.5,
                color=color,
                zorder=4,
            )
            ax.plot(
                layers[~significant],
                values[~significant],
                linestyle="none",
                marker="o",
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.2,
                zorder=4,
            )

            if middle is not None:
                ax.axvline(float(middle), color="#8A8A8A", linestyle=":", linewidth=1.4, zorder=1)
                ax.annotate(
                    f"layer {middle}",
                    xy=(float(middle), 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(4, -3),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    fontsize=8.0 * font_scale,
                    color="#6A6A6A",
                )

            ax.axhline(0.0, color="#BBBBBB", linewidth=0.8, zorder=0)
            ax.set_xticks(_layer_ticks(np.asarray(cond_df["layer"], dtype=int)))
            ax.tick_params(axis="both", labelsize=9.5 * font_scale)
            # Each row has its own x meaning (blocks vs hidden states), so the
            # label cannot be collapsed onto the bottom row only.
            ax.set_xlabel(x_label, fontsize=9.5 * font_scale)
            # Title carries the condition, so no legend is needed per panel.
            ax.set_title(
                CONDITION_LABELS.get(condition, condition).replace("\n", " "),
                fontsize=11 * font_scale,
                fontweight="bold",
                color=color,
            )
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(y_label, fontsize=9.5 * font_scale)

            if y_limits is not None:
                ax.set_ylim(*y_limits)
            else:
                upper = float(np.nanmax(values + sems))
                lower = float(np.nanmin(values - sems))
                span = max(upper - lower, 0.005)
                ax.set_ylim(min(0.0, lower - 0.12 * span), upper + 0.20 * span)

    correction_note = (
        f"Filled markers: q < {alpha:g} (BH-FDR across all layers × conditions)."
        "  Dotted line: layer used in the main analysis."
        "  Note the independent y-axes."
    )
    fig.suptitle(
        f"Encoding performance by layer — {short_model_label(model_label)}",
        fontsize=13 * font_scale,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.text(0.5, 0.006, correction_note, ha="center", fontsize=8.0 * font_scale, color="#555555")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Layer sweep figure → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


@hydra.main(
    version_base=None,
    config_path="../../configs/visualization",
    config_name="visualize_layer_sweep",
)
def main(cfg: DictConfig) -> None:
    """Collect a layer sweep and render its depth curves."""
    configure_plot_fonts()

    run_dir = Path(cfg.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJ_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    metric: str = cfg.metric
    main_analysis_layers: dict[str, int] = {
        str(k): int(v) for k, v in dict(cfg.main_analysis_layers).items()
    }

    results, model_label = collect_layer_results(
        run_dir,
        model_label=cfg.get("model_label") or None,
        metric=metric,
        group_sig_permutations=int(cfg.group_sig_permutations),
        group_sig_random_state=int(cfg.group_sig_random_state),
        group_sig_correction=str(cfg.group_sig_correction),
    )
    _log_peaks(results, metric, main_analysis_layers)

    output_path = (
        Path(cfg.output_path)
        if cfg.get("output_path")
        else (
            FIGURES_DIR
            / "layer_sweep"
            / f"{short_model_label(model_label).lower()}_layer_sweep.png"
        )
    )
    if not output_path.is_absolute():
        output_path = PROJ_ROOT / output_path

    csv_path = (
        Path(cfg.summary_output_path)
        if cfg.get("summary_output_path")
        else output_path.with_name("layer_sweep_summary.csv")
    )
    if not csv_path.is_absolute():
        csv_path = PROJ_ROOT / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False)
    logger.info(f"Per-layer summary → {csv_path}")

    y_limits = tuple(cfg.y_limits) if cfg.get("y_limits") else None

    plot_layer_curves(
        results,
        metric=metric,
        model_label=model_label,
        alpha=float(cfg.alpha),
        main_analysis_layers=main_analysis_layers,
        y_limits=y_limits,  # type: ignore[arg-type]
        figsize=tuple(cfg.figsize),  # type: ignore[arg-type]
        font_scale=float(cfg.font_scale),
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
