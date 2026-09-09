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

from collections import Counter
from pathlib import Path
import re
from typing import Any, Iterable, cast
import warnings

import hydra
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from scipy.stats import wilcoxon

from cross_modal_neural_encoding.config import FIGURES_DIR, PROJ_ROOT
from cross_modal_neural_encoding.utils import (
    CONDITION_LABELS,
    benjamini_hochberg,
    configure_plot_fonts,
    short_model_label,
    signflip_pvalue_greater,
    significance_label,
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
# Color encodes modality across the paper: text = blue, image = red, and VLMs
# (cross-modal) = purple — visually the blue+red mix. Each model family therefore
# takes the hue of its modality (green is reserved for the fMRI/brain space in
# the overview schematic).
VLM_MODEL_PALETTE = [
    "#C3A1D6",  # lavender
    "#A988C8",  # soft purple
    "#906EBA",  # muted violet
    "#D0B4E6",  # light lilac
    "#B39AD8",  # pale purple
    "#9B7CCF",  # medium purple
]
VISION_MODEL_PALETTE = [
    "#F5A3A3",  # soft red
    "#E88989",  # dusty red
    "#D96F6F",  # muted crimson
    "#F2B6A0",  # warm peach
    "#F7B7B2",  # salmon
    "#E39DC1",  # pink-rose
]
TEXT_MODEL_PALETTE = [
    "#9CC4DD",  # soft blue
    "#7EAEDB",  # pastel blue
    "#6F9FC9",  # muted blue
    "#B7D9EC",  # light sky blue
    "#8FBFD4",  # steel blue
    "#A3D2E2",  # pale aqua
]
# Neutral body for the condition violins — the overlaid per-model points carry
# the (category) colour coding, so the violin itself stays quiet.
VIOLIN_BODY_COLOR = "#C7CFDB"
# One representative colour per model category, matching the category palettes
# used for the grouped bars (VLM = purple, vision-only = red, text-only = blue).
CATEGORY_POINT_COLORS = {0: "#906EBA", 1: "#D96F6F", 2: "#6F9FC9"}
CATEGORY_LABELS = {0: "Vision–language", 1: "Vision-only", 2: "Text-only"}
# Model families: size variants of the same checkpoint collapse into one family.
# Matched as substrings of the short (vendor-stripped) model label, lower-cased.
MODEL_FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("internvl", "InternVL3.5"),
    ("qwen", "Qwen3.5"),
    ("clip", "CLIP"),
    ("dinov2", "DINOv2"),
    ("ijepa", "I-JEPA"),
    ("pythia", "Pythia"),
    ("opt-", "OPT"),
)
# ── Condition-matrix palette ────────────────────────────────────────────────
# Hue encodes model family, lightness encodes model size within that family.
# Every family gets its own hue, stepped within the modality colours used across
# the paper: the three vision-language families take three shades of purple
# (cool → warm: InternVL blue-lavender, Qwen mauve, CLIP rose), the vision-only
# baselines two warm shades (DINOv2 red, I-JEPA amber) and the text-only
# baselines two blues (Pythia sky, OPT deeper).
#
# Checked with the data-viz palette validator (light surface #fcfcfb). Every
# ramp passes the ordinal checks — monotone lightness, adjacent ΔL >= 0.06, one
# hue. Two checks are knowingly not met, both as a direct consequence of the
# pastel brief:
#   - light-end contrast is 1.4–1.8:1 rather than >= 2:1, which the 0.5 pt dark
#     bar outline compensates for;
#   - the closest family pairs sit below the ΔE 15 normal-vision floor:
#     InternVL/Qwen at 4.5, DINOv2/I-JEPA at 5.1 and Pythia/OPT at 6.7, so
#     colour alone cannot separate any of those three pairs.
# Both are safe here only because colour is *redundant* in this figure: every
# family block is named directly on the axis and every bar carries its size
# label, so identity is never colour-alone.
MATRIX_FAMILY_RAMPS: dict[str, list[str]] = {
    "InternVL3.5": ["#C7C4EC", "#A8A4DC", "#8A85C7", "#6F6AAE"],  # blue-lavender
    "Qwen3.5": ["#CFB4D8", "#B896C6", "#A278AF", "#8C5D97"],  # mauve
    "CLIP": ["#DEB4CA", "#CE9DB6", "#BD86A2", "#A96D8B"],  # rose
    "DINOv2": ["#F4BCB6", "#E89A93", "#D9776F", "#C4564D"],  # red
    "I-JEPA": ["#F6C4A6", "#EDA274", "#DC8250", "#C56634"],  # burnt orange
    "Pythia": ["#C0D8F0", "#9CBCE4", "#789FD2", "#567FBA"],  # sky blue
    "OPT": ["#B3D2EE", "#7FA8DC", "#5480C4", "#3A5FA0"],  # deeper blue
}
# Fallbacks for families without an explicit ramp above, by model category.
MATRIX_VISION_RAMP = MATRIX_FAMILY_RAMPS["DINOv2"]
MATRIX_TEXT_RAMP = MATRIX_FAMILY_RAMPS["Pythia"]
MATRIX_CATEGORY_RAMPS = {
    0: MATRIX_FAMILY_RAMPS["Qwen3.5"],
    1: MATRIX_VISION_RAMP,
    2: MATRIX_TEXT_RAMP,
}
# Draw order of the family blocks inside each condition panel.
MATRIX_FAMILY_ORDER = (
    "InternVL3.5",
    "Qwen3.5",
    "CLIP",
    "DINOv2",
    "I-JEPA",
    "Pythia",
    "OPT",
)
# Axis labels for the family blocks, spelled as the paper's model section does.
# The dict keys are the internal family names from MODEL_FAMILY_PATTERNS.
MATRIX_FAMILY_DISPLAY = {"Qwen3.5": "Qwen3.5-VL", "CLIP": "OpenCLIP"}
# Blank x-units left between family blocks inside a condition panel.
BLOCK_GAP = 1.2
# Gutters between the 2x2 panels, as fractions of panel size. Kept tight: they
# only have to hold the pairwise connectors, and every point saved goes to the
# panels, where the model labels have to stay legible in print.
MATRIX_WSPACE = 0.21
MATRIX_HSPACE = 0.30

# Named size tokens for checkpoints that do not spell out a parameter count.
# Checked before the "<n>B" regex because CLIP labels carry a training-set size
# ("laion2B") that the regex would otherwise read as the model size.
MATRIX_SIZE_TOKENS: tuple[tuple[str, float, str], ...] = (
    ("vit-g", 3.0, "G"),
    ("vitg", 3.0, "G"),
    ("giant", 3.0, "G"),
    ("vit-h", 2.0, "H"),
    ("vith", 2.0, "H"),
    ("huge", 2.0, "H"),
    ("vit-l", 1.0, "L"),
    ("vitl", 1.0, "L"),
    ("large", 1.0, "L"),
)

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


def _legend_ncol(
    labels: list[str], fig_width_in: float, font_pt: float, desired_max: int
) -> int:
    """Largest column count whose (widest-label) columns fit within the figure.

    Conservative on purpose — it sizes every column to the longest label so a
    horizontal legend never ends up wider than the figure, even if that means
    wrapping onto more rows.
    """
    longest = max((len(str(s)) for s in labels), default=1)
    # ~0.52 em per character for the sans-serif labels, plus ~3 chars for the
    # legend handle and inter-column padding.
    col_in = (longest + 3) * 0.52 * font_pt / 72.0
    fit = int(fig_width_in / max(col_in, 1e-6))
    return max(1, min(desired_max, fit, len(labels)))


def _model_category_rank(model_label: str) -> tuple[int, str]:
    """Sort key: VLM first, then vision-only, then text-only."""
    label = model_label.lower()
    if any(tag in label for tag in ("clip", "vlm", "qwen", "intern")):
        return (0, model_label)
    if any(tag in label for tag in ("dinov2", "ijepa", "dino")):
        return (1, model_label)
    return (2, model_label)


def model_family(model_label: str) -> str:
    """Family a model belongs to, collapsing size variants into one name.

    E.g. ``OpenGVLab--InternVL3_5-1B-HF`` and ``…-8B-HF`` both map to
    ``InternVL3.5``. Checkpoints that match no known pattern stay their own
    family — merging them on a guessed name pattern would silently drop models.
    """
    short = short_model_label(model_label)
    lowered = short.lower()
    for pattern, family in MODEL_FAMILY_PATTERNS:
        if pattern in lowered:
            return family
    return short


def model_size(model_label: str) -> tuple[float, str]:
    """Model size as ``(sort_rank, short display label)``.

    The rank orders checkpoints within a family so the colour ramp runs
    small→large; the display label is what goes under the bar (``"2B"``,
    ``"L"``, …). Unrecognised checkpoints rank first with an empty label rather
    than raising — an unlabelled bar is recoverable, a crash is not.
    """
    short = short_model_label(model_label).lower()
    for token, rank, display in MATRIX_SIZE_TOKENS:
        if token in short:
            return rank, display
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", short)
    if match:
        return float(match.group(1)), f"{match.group(1)}B"
    return 0.0, ""


def _matrix_model_order(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort panel members into family blocks, small→large inside each block."""

    def key(item: dict[str, Any]) -> tuple[int, float, str]:
        label = item["model_label"]
        family = model_family(label)
        try:
            family_rank = MATRIX_FAMILY_ORDER.index(family)
        except ValueError:
            family_rank = len(MATRIX_FAMILY_ORDER)
        return (family_rank, model_size(label)[0], label)

    return sorted(items, key=key)


def _matrix_color(family: str, category: int, position: int, family_size: int) -> str:
    """Ramp step for the *position*-th (small→large) member of a family.

    Families with fewer members than the ramp has steps spread across the whole
    ramp, so a two-model family still gets maximum lightness contrast rather than
    two neighbouring steps.
    """
    ramp = MATRIX_FAMILY_RAMPS.get(family) or MATRIX_CATEGORY_RAMPS.get(
        category, MATRIX_VISION_RAMP
    )
    if family_size <= 1:
        return ramp[len(ramp) // 2]
    step = round(position * (len(ramp) - 1) / (family_size - 1))
    return ramp[int(min(step, len(ramp) - 1))]


def select_best_per_family(
    model_results: list[dict[str, Any]],
    *,
    metric: str,
) -> list[dict[str, Any]]:
    """Keep only the best-scoring model of each family.

    "Best" is the highest mean of *metric* across the conditions that model was
    run on (family members share a condition set, so the average is comparable
    within a family). Models without a usable value are dropped. The surviving
    models keep their input order.
    """
    best: dict[str, tuple[float, int]] = {}
    for idx, item in enumerate(model_results):
        df = item["aggregated_df"]
        try:
            column = np.asarray(df[(metric, "mean")], dtype=float)
        except KeyError:
            logger.warning(f"Skipping {item['model_label']}: no '{metric}' column.")
            continue
        if not np.any(np.isfinite(column)):
            continue
        score = float(np.nanmean(column))
        family = model_family(item["model_label"])
        current = best.get(family)
        if current is None or score > current[0]:
            best[family] = (score, idx)

    keep = {idx for _, idx in best.values()}
    for family, (score, idx) in sorted(best.items()):
        logger.info(
            f"Best in family {family}: {model_results[idx]['model_label']} "
            f"(mean {metric} = {score:.4f})"
        )
    return [item for idx, item in enumerate(model_results) if idx in keep]


def _model_condition_matrix(
    model_results: list[dict[str, Any]],
    conditions: list[str],
    metric: str,
) -> np.ndarray:
    """``(n_models, n_conditions)`` matrix of per-model group means.

    Conditions a model was not run on stay ``NaN``.
    """
    values = np.full((len(model_results), len(conditions)), np.nan, dtype=float)
    for i, item in enumerate(model_results):
        df = item["aggregated_df"]
        for j, cond in enumerate(conditions):
            if cond not in df.index:
                continue
            values[i, j] = float(df.loc[cond, (metric, "mean")])
    return values


def _legend_y_below_xticklabels(fig, ax, *, fallback: float, nudge: float) -> float:
    """Axes-fraction y just below the *rendered* x-tick labels.

    Keeps a legend clear of multi-line/rotated condition labels however short the
    figure gets; falls back to a fixed offset when no renderer is available.
    """
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()  # type: ignore
        inv_axes = ax.transAxes.inverted()
        label_bottom = min(
            (
                inv_axes.transform((0.0, lbl.get_window_extent(renderer=renderer).y0))[1]
                for lbl in ax.get_xticklabels()
            ),
            default=fallback,
        )
    except Exception:  # pragma: no cover - renderer unavailable
        return fallback
    return label_bottom + nudge


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
    force_normalized_reference: bool = False,
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
    force_normalized_reference: bool = False,
) -> tuple[float, float] | None:
    """Compute shared y-limits across all models for the chosen metric.

    Scans both the aggregated model means (±std) and the per-subject values
    so neither the grouped nor the per-subject figures clip any bars.
    """
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

        summary_df = item.get("summary_df")
        if summary_df is not None and metric in summary_df.columns:
            sub_vals = np.asarray(summary_df[metric], dtype=float)
            values_list.append(sub_vals)
            stds_list.append(np.zeros_like(sub_vals))

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


def combined_perm_group_pvalue(
    model_dir: str | Path,
    summary_df: pd.DataFrame,
    condition: str,
    *,
    n_group_draws: int = 10000,
    random_state: int = 42,
) -> float:
    """Group p-value combining subject-level permutation null distributions.

    For each subject the encoding pipeline saved an empirical null
    distribution of ``mean_r`` (``null_mean_r.npy``), obtained by shuffling
    the stimulus-to-embedding mapping. The group-level null is built by
    repeatedly drawing one value from each subject's null distribution and
    averaging across subjects (Stelzer et al., 2013, NeuroImage). The
    observed group statistic is the mean of the subjects' real ``mean_r``.

    Significance is always assessed on the raw ``mean_r`` (the quantity the
    permutation null was built for); the displayed metric may differ, but
    "above chance" is a property of the raw correlation.

    Returns ``nan`` when no per-subject null files are available, so the
    caller can fall back to another test.
    """
    model_dir = Path(model_dir)
    rows = summary_df[summary_df["condition"] == condition]
    observed: list[float] = []
    nulls: list[np.ndarray] = []
    for _, row in rows.iterrows():
        subject = str(row["subject"])
        mean_r = float(row.get("mean_r", np.nan))
        null_path = model_dir / subject / condition / "null_mean_r.npy"
        if not np.isfinite(mean_r) or not null_path.exists():
            continue
        null_vals = np.asarray(np.load(null_path), dtype=float)
        null_vals = null_vals[np.isfinite(null_vals)]
        if null_vals.size == 0:
            continue
        observed.append(mean_r)
        nulls.append(null_vals)

    if not nulls:
        return float("nan")

    observed_stat = float(np.mean(observed))
    rng = np.random.default_rng(random_state)
    # Each group-null draw = average of one random value per subject.
    group_null = np.zeros(n_group_draws, dtype=float)
    for null_vals in nulls:
        idx = rng.integers(0, null_vals.size, size=n_group_draws)
        group_null += null_vals[idx]
    group_null /= len(nulls)

    return float((np.sum(group_null >= observed_stat) + 1) / (n_group_draws + 1))


def _group_level_pvalues(
    summary_df: pd.DataFrame,
    *,
    conditions: list[str],
    metric: str,
    model_dir: str | Path | None,
    n_permutations: int = 10000,
    random_state: int = 42,
) -> np.ndarray:
    """One group p-value per condition.

    Prefers the combined subject-level permutation test (uses the saved
    ``null_mean_r.npy`` files). Falls back to the one-sample sign-flip test
    on the per-subject *metric* values when no null files are available.
    """
    pvals: list[float] = []
    for cond in conditions:
        p = float("nan")
        if model_dir is not None:
            p = combined_perm_group_pvalue(
                model_dir,
                summary_df,
                cond,
                n_group_draws=n_permutations,
                random_state=random_state,
            )
        if not np.isfinite(p):
            vals = np.asarray(
                summary_df.loc[summary_df["condition"] == cond, metric],
                dtype=float,
            )
            p = signflip_pvalue_greater(
                vals,
                n_permutations=n_permutations,
                random_state=random_state,
            )
        pvals.append(p)
    return np.asarray(pvals, dtype=float)


def _group_stats_matrices(
    model_results: list[dict[str, Any]],
    conditions: list[str],
    *,
    metric: str,
    group_sig_permutations: int,
    group_sig_random_state: int,
    group_sig_correction: str,
    show_error_bars: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(values, qvalues, sems)`` matrices of shape ``(n_models, n_conditions)``.

    Shared by every figure that draws one bar per model and condition, so they
    all correct over the same family of tests and cannot drift into showing
    different stars for the same numbers. Conditions a model was not run on stay
    ``NaN`` throughout. ``sems`` is all-``NaN`` unless *show_error_bars*.
    """
    n_models, n_conditions = len(model_results), len(conditions)
    values = _model_condition_matrix(model_results, conditions, metric)

    pvals = np.full((n_models, n_conditions), np.nan, dtype=float)
    for i, item in enumerate(model_results):
        summary_df = item.get("summary_df")
        if summary_df is None or "condition" not in summary_df.columns:
            continue
        present = set(summary_df["condition"].unique())
        model_conditions = [c for c in conditions if c in present]
        cond_pvals = _group_level_pvalues(
            summary_df,
            conditions=model_conditions,
            metric=metric,
            model_dir=item.get("model_dir"),
            n_permutations=group_sig_permutations,
            random_state=group_sig_random_state,
        )
        for cond, p in zip(model_conditions, cond_pvals):
            pvals[i, conditions.index(cond)] = p

    # Correct for multiple comparisons across all model x condition tests
    # shown in this figure (NaN/"na" cells are excluded from the family).
    if group_sig_correction == "fdr_bh":
        pvals = benjamini_hochberg(pvals)

    # Per-bar error bars: SEM across subjects of each model's condition mean,
    # from the aggregated across-subject SD and the per-subject subject count
    # (SEM = SD / sqrt(n)).
    sems = np.full((n_models, n_conditions), np.nan, dtype=float)
    if show_error_bars:
        for i, item in enumerate(model_results):
            df = item["aggregated_df"]
            sdf = item.get("summary_df")
            has_summary = (
                sdf is not None and "condition" in sdf.columns and metric in sdf.columns
            )
            for j, cond in enumerate(conditions):
                if cond not in df.index:
                    continue
                try:
                    sd = float(df.loc[cond, (metric, "std")])
                except (KeyError, ValueError):
                    continue
                n = 0
                if has_summary:
                    col = sdf.loc[sdf["condition"] == cond, metric].to_numpy(dtype=float)  # type: ignore
                    n = int(np.isfinite(col).sum())
                if n >= 2 and np.isfinite(sd):
                    sems[i, j] = sd / np.sqrt(n)

    return values, pvals, sems


def pairwise_condition_signrank(
    values: np.ndarray,
    *,
    correction: str = "fdr_bh",
) -> dict[tuple[int, int], float]:
    """Pairwise Wilcoxon signed-rank tests between conditions.

    ``values`` is the ``(n_models, n_conditions)`` matrix of per-model means that
    are actually drawn as bars. The same model contributes to every condition, so
    the conditions are *paired* by model: for each pair ``j < k`` the test runs on
    the per-model differences over the models with a finite value in both
    conditions (two-sided ``scipy.stats.wilcoxon``). Returns ``{(j, k): q}`` for
    every pair, BH-FDR corrected across the family when ``correction == "fdr_bh"``.
    Pairs with fewer than two paired models, or with no non-zero difference
    (``wilcoxon`` is undefined), are returned as ``NaN``.

    The exact signed-rank distribution is used (``method="exact"``) so the p-value
    depends only on the signs/ranks and not on whether scipy would otherwise fall
    back to the normal approximation when the absolute differences contain ties.
    """
    n_conditions = values.shape[1]
    pairs = [(j, k) for j in range(n_conditions) for k in range(j + 1, n_conditions)]
    raw = np.full(len(pairs), np.nan, dtype=float)
    for idx, (j, k) in enumerate(pairs):
        paired = np.isfinite(values[:, j]) & np.isfinite(values[:, k])
        a = values[paired, j]
        b = values[paired, k]
        if len(a) < 2 or np.allclose(a, b):
            continue
        try:
            with warnings.catch_warnings():
                # We deliberately force the exact distribution; ignore scipy's
                # note that ties in the absolute differences make it approximate.
                warnings.simplefilter("ignore")
                result = wilcoxon(a, b, method="exact")
        except ValueError:
            # e.g. every paired difference is zero after dropping ties.
            continue
        # scipy's stubs type the result as a bare tuple, so pull the p-value by
        # index and cast it to keep the type checker happy.
        raw[idx] = float(cast(float, result[1]))
    q = benjamini_hochberg(raw) if correction == "fdr_bh" else raw
    return {pair: q[idx] for idx, pair in enumerate(pairs)}


def annotate_pairwise_brackets(
    ax,
    *,
    x_positions: np.ndarray,
    values: np.ndarray,
    pair_qvalues: dict[tuple[int, int], float],
    alpha: float,
    font_scale: float,
) -> None:
    """Draw significance brackets for pairwise condition comparisons.

    Each pair gets its own level so every bracket reads as a single comparison
    (shared levels would let adjacent brackets' ticks merge into one apparent
    bracket). Shorter spans sit on lower levels, giving a nested-pyramid look with
    the longest span on top. Significant pairs get stars, the rest an ``ns`` label;
    pairs whose test could not be run (``NaN``) are skipped. The y-limit is
    extended so the brackets fit.
    """
    pairs = [p for p, q in pair_qvalues.items() if np.isfinite(q)]
    if not pairs:
        return
    # One bracket per level; shorter spans below longer ones, left-to-right.
    pairs.sort(key=lambda p: (p[1] - p[0], p[0]))
    pair_level = {pair: lvl for lvl, pair in enumerate(pairs)}

    y0, y1 = ax.get_ylim()
    span = y1 - y0
    # Reference top: highest bar extent — bars rise to their max value, or to 0
    # when every bar is negative (as in the % delta plot).
    top = max(float(np.nanmax(values)), 0.0)
    base = top + 0.06 * span
    step = 0.055 * span
    tick = 0.012 * span

    highest = base
    for (j, k), lvl in pair_level.items():
        y = base + lvl * step
        highest = max(highest, y)
        x_j, x_k = float(x_positions[j]), float(x_positions[k])
        ax.plot(
            [x_j, x_j, x_k, x_k],
            [y - tick, y, y, y - tick],
            color="#333333",
            linewidth=0.8,
            zorder=6,
        )
        sig = significance_label(float(pair_qvalues[(j, k)]), alpha)
        ax.text(
            (x_j + x_k) / 2.0,
            y + 0.005 * span,
            sig,
            ha="center",
            va="bottom",
            fontsize=9.5 * font_scale,
            color="#333333" if sig == "ns" else "darkred",
            fontweight="normal" if sig == "ns" else "bold",
            zorder=6,
        )
    ax.set_ylim(y0, max(y1, highest + 0.06 * span))


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
    model_dir: str | Path | None,
    show_subject_panel: bool,
    use_group_level_significance: bool,
    group_sig_permutations: int,
    group_sig_random_state: int,
    group_sig_correction: str,
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
        p_values = _group_level_pvalues(
            summary_df,
            conditions=conditions,
            metric=metric,
            model_dir=model_dir,
            n_permutations=group_sig_permutations,
            random_state=group_sig_random_state,
        )

    # Correct for multiple comparisons across this model's conditions.
    if group_sig_correction == "fdr_bh":
        p_values = benjamini_hochberg(p_values)

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
            force_normalized_reference=False,
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
                .pivot_table(
                    index="subject", columns="condition", values=p_value_col, aggfunc="mean"
                )
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
            subject_colors = [SUBJECT_PALETTE[i % len(SUBJECT_PALETTE)] for i in range(n_subj)]

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
        ax_subject.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=10 * font_scale,
            ncol=1,
            frameon=False,
        )

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
    group_sig_correction: str = "fdr_bh",
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
            model_dir=item.get("model_dir"),
            show_subject_panel=show_subject_panel,
            use_group_level_significance=use_group_level_significance,
            group_sig_permutations=group_sig_permutations,
            group_sig_random_state=group_sig_random_state,
            group_sig_correction=group_sig_correction,
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
    group_sig_permutations: int = 10000,
    group_sig_random_state: int = 42,
    group_sig_correction: str = "fdr_bh",
    title: str = "Model Comparison Based on Group Means",
    condition_labels: dict[str, str] | None = None,
    significance_note: bool = False,
    show_error_bars: bool = False,
) -> None:
    """Plot grouped bars comparing model means across conditions.

    Models are allowed to have different condition sets; missing values
    are shown as empty slots (no bar). Significance stars come from the
    combined subject-level permutation test (saved ``null_mean_r.npy``
    files), falling back to the sign-flip test when nulls are unavailable.

    When ``significance_note`` is set, the framed star-threshold legend is
    replaced by a compact one-line note inside the axes (matching the
    ``plot_combined_delta`` ablation-delta figure), leaving just the model
    legend as a single row below the plot.
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
    label_map = condition_labels if condition_labels is not None else CONDITION_LABELS
    labels = [label_map.get(c, c) for c in conditions]

    model_labels = [item["model_label"] for item in model_results]
    n_models = len(model_labels)
    n_conditions = len(conditions)

    values, pvals, sems = _group_stats_matrices(
        model_results,
        conditions,
        metric=metric,
        group_sig_permutations=group_sig_permutations,
        group_sig_random_state=group_sig_random_state,
        group_sig_correction=group_sig_correction,
        show_error_bars=show_error_bars,
    )

    is_normalized_metric = "normalized" in metric.lower()

    # Adaptive width so individual bars stay readable as the model count
    # grows; never narrower than the configured width. With many models per
    # group the bars get skinny and their significance stars collide, so scale
    # the width generously with the total bar count. The legend goes below in
    # balanced columns (more of them as the figure widens), capped so it never
    # grows wider than the figure.
    base_w, base_h = figsize
    fig_w = max(base_w, 3.0 + 0.22 * n_models * n_conditions)
    # The legend, significance key and per-bar stars carry the actual content of
    # the figure, so give them a larger size than the (less informative) y-axis
    # ticks. Compute the legend column count at the rendered size so a wider font
    # never overflows the figure width.
    legend_fs = 10.5 * font_scale
    legend_labels = [short_model_label(m) for m in model_labels]
    n_legend_cols = _legend_ncol(legend_labels, fig_w, legend_fs, 6)
    fig, ax = plt.subplots(figsize=(fig_w, base_h))
    x = np.arange(n_conditions)
    total_width = 0.94
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
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]
        if finite_values.size == 0:
            logger.warning("No finite values available for grouped plot.")
            return
        # Let the error bars widen the limits so whiskers are never clipped.
        stds = (
            np.where(np.isfinite(sems), sems, 0.0)[finite_mask]
            if show_error_bars
            else np.zeros_like(finite_values)
        )
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
            yerr=sems[i] if show_error_bars else None,
            color=colors[i],
            edgecolor="#4A4A4A",
            linewidth=0.5,
            alpha=0.9,
            error_kw={"linewidth": 0.7},
            capsize=2,
            label=short_model_label(label),
            zorder=3,
        )

        # Mark missing values with "na" for conditions that don't apply.
        for j, xj in enumerate(x + offsets[i]):
            if np.isfinite(values[i, j]):
                continue
            ax.text(
                xj,
                y_na,
                "n\na",
                ha="center",
                va="bottom",
                linespacing=0.6,
                fontsize=9.5 * font_scale,
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
            # Clear the error-bar whisker (if any) before adding the star offset.
            err = sems[i, j] if show_error_bars and np.isfinite(sems[i, j]) else 0.0
            pad = 0.015 * (y_max - y_min)
            y_sig = hj + (err + pad if hj >= 0 else -(err + pad))
            va = "bottom" if hj >= 0 else "top"
            # Stack the significance characters vertically with newlines (rather
            # than rotating the string 90°) so each glyph stays horizontally
            # centred on the bar — a rotated "***" lands off-centre because the
            # asterisk sits high in its character cell.
            ax.text(
                xj,
                y_sig,
                "\n".join(sig),
                ha="center",
                va=va,
                linespacing=0.6,
                fontsize=10.0 * font_scale,
                color="black" if sig == "ns" else "darkred",
                zorder=5,
            )

    ax.set_xticks(x)
    # In note mode, match the ablation-delta figure's horizontal x labels (which
    # keep the figure flat); otherwise rotate to fit many narrow condition groups.
    # The horizontal labels are wide, so use a slightly smaller size than the
    # rotated variant to keep adjacent condition labels from touching.
    ax.set_xticklabels(
        labels,
        fontsize=(9 if significance_note else 12.5) * font_scale,
        rotation=0 if significance_note else 32,
        ha="center" if significance_note else "right",
    )
    ax.tick_params(axis="y", labelsize=12 * font_scale)
    if is_normalized_metric:
        ax.set_ylabel("Normalized performance\n(r / NC)", fontsize=10 * font_scale)
    else:
        ax.set_ylabel("Pearson correlation", fontsize=10 * font_scale)
    ax.set_title(
        title,
        fontsize=15 * font_scale,
        fontweight="bold",
    )
    ax.axhline(y=0, color="black", linewidth=0.5, zorder=1)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Anchor the legends a fixed gap below the *actual* rendered x-tick labels
    # (rotated, two lines) so they never overlap the condition labels, however
    # short the figure gets. Fall back to a fixed offset if no renderer is ready.
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()  # type: ignore
        inv_axes = ax.transAxes.inverted()
        label_bottom = min(
            (
                inv_axes.transform((0.0, lbl.get_window_extent(renderer=renderer).y0))[1]
                for lbl in ax.get_xticklabels()
            ),
            default=-0.20,
        )
        # Horizontal labels (note mode) sit close under the axis, so drop the
        # legend a bit *below* their bottom edge; rotated labels extend far down
        # and instead need the small positive nudge back toward the axis.
        model_legend_y = label_bottom + (-0.05 if significance_note else 0.06)
    except Exception:  # pragma: no cover - renderer unavailable
        renderer = None
        inv_axes = None
        model_legend_y = -0.20

    model_legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, model_legend_y),
        fontsize=legend_fs,
        ncol=n_legend_cols,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    ax.add_artist(model_legend)

    if significance_note:
        # Compact one-line significance note inside the axes, matching the
        # ablation-delta figure (plot_combined_delta) instead of the framed
        # threshold box. Per-bar stars test each bar against chance; the brackets
        # test pairs of conditions (signed-rank). Carve extra headroom at the
        # bottom first so the note sits in a clear strip below the lowest bars and
        # their "ns" labels (as it does in the delta figure).
        y_min = y_min - 0.16 * (y_max - y_min)
        note_text = (
            "Stars on bars: vs. chance   ·   "
            "Brackets: pairwise signed-rank between conditions"
        )
        if group_sig_correction == "fdr_bh":
            note_text += "   ·   BH-FDR corrected"
        ax.text(
            0.01,
            0.02,
            note_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8 * font_scale,
            color="#333333",
            zorder=6,
        )
        sig_legend = None
    else:
        # Second legend explaining the significance stars. The same star scheme is
        # used for two tests, so spell out which is which: per-bar stars test each
        # bar against chance; the brackets test pairs of conditions (signed-rank).
        stat_sym = "q" if group_sig_correction == "fdr_bh" else "p"
        sig_entries = [
            ("***", f"{stat_sym} < 0.001"),
            ("**", f"{stat_sym} < 0.01"),
            ("*", f"{stat_sym} < {alpha:g}"),
            ("ns", "not significant"),
            ("", "bars: vs. chance"),
            ("", "brackets: pairwise (signed-rank)"),
        ]
        sig_handles = [Line2D([], [], linestyle="none") for _ in sig_entries]
        sig_labels = [(f"{stars}  {desc}" if stars else desc) for stars, desc in sig_entries]
        sig_title = "Significance"
        if group_sig_correction == "fdr_bh":
            sig_title += "\n(BH-FDR corrected)"
        # Place the significance key as a horizontal row below the model legend
        # (which sits below the axes), capped so neither legend exceeds the figure
        # width.
        # Stack the significance key just below the model legend, measured from its
        # rendered extent so the gap is constant regardless of legend row count.
        if renderer is not None and inv_axes is not None:
            fig.canvas.draw()
            leg_ext = model_legend.get_window_extent(renderer=renderer)
            sig_y = inv_axes.transform((0.0, leg_ext.y0))[1] - 0.03
        else:
            n_legend_rows = int(np.ceil(n_models / max(n_legend_cols, 1)))
            sig_y = model_legend_y - 0.08 * n_legend_rows - 0.05
        sig_ncol = _legend_ncol(sig_labels, fig_w, legend_fs, len(sig_entries))
        sig_legend = ax.legend(
            sig_handles,
            sig_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, sig_y),
            ncol=sig_ncol,
            fontsize=legend_fs,
            title=sig_title,
            title_fontsize=legend_fs,
            frameon=True,
            framealpha=0.9,
            handlelength=0,
            handletextpad=0,
            columnspacing=1.5,
            borderpad=0.6,
            labelspacing=0.3,
        )
    ax.margins(x=0.03)

    if is_normalized_metric and compress_normalized_axis:
        ax.set_yscale(
            "symlog",
            linthresh=max(normalized_axis_linthresh, 1e-3),
            linscale=1.0,
            base=10,
        )

    ax.set_ylim(y_min, y_max)

    # Pairwise Wilcoxon signed-rank tests between the condition groups (per-model
    # means, paired by model), drawn as nested significance brackets above the
    # bars. Bracket offsets are linear, so skip them on a compressed (symlog) axis.
    if not (is_normalized_metric and compress_normalized_axis):
        pair_q = pairwise_condition_signrank(values, correction=group_sig_correction)
        annotate_pairwise_brackets(
            ax,
            x_positions=x,
            values=values,
            pair_qvalues=pair_q,
            alpha=alpha,
            font_scale=font_scale,
        )

    if output_path is None:
        output_path = FIGURES_DIR / "encoding_results_models_grouped.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extra_artists = (model_legend,) if sig_legend is None else (model_legend, sig_legend)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=extra_artists,
    )
    logger.success(f"Grouped-model figure saved to {output_path}")
    plt.close(fig)


def _matrix_panel_members(
    model_results: list[dict[str, Any]],
    values: np.ndarray,
    condition_index: int,
) -> list[dict[str, Any]]:
    """Models with a finite value in one condition, in family/size draw order.

    Each entry keeps ``row`` — the model's index into the figure-wide statistics
    matrices — so a panel never has to re-derive its own statistics.
    """
    members = [
        {"row": i, "model_label": item["model_label"]}
        for i, item in enumerate(model_results)
        if np.isfinite(values[i, condition_index])
    ]
    return _matrix_model_order(members)


def _fit_fontsize(requested_pt: float, available_pt: float, n_chars: int) -> float:
    """Largest of *requested_pt* and a size whose *n_chars* fit *available_pt*.

    The per-bar and per-block labels sit in slots whose width is fixed by the
    model count, so a font size chosen as a constant either collides in the
    dense panels or is needlessly small in the sparse ones. ``0.62`` em per
    character is a conservative average for the sans-serif face used here.
    """
    if n_chars <= 0:
        return requested_pt
    return max(1.0, min(requested_pt, available_pt / (0.62 * n_chars)))


def _draw_matrix_panel(
    ax: Axes,
    members: list[dict[str, Any]],
    *,
    values: np.ndarray,
    qvalues: np.ndarray,
    sems: np.ndarray,
    condition_index: int,
    alpha: float,
    font_scale: float,
    show_error_bars: bool,
    labels_on_top: bool,
    fig_size_in: tuple[float, float],
    block_gap: float = BLOCK_GAP,
) -> float:
    """Draw one condition panel: family blocks of bars with direct size labels.

    Bars are grouped into family blocks separated by *block_gap*, coloured by
    category hue and size lightness, and labelled directly on the axis (size
    under each bar, family under each block) so the figure needs no legend.

    Returns the height in points that the label tiers take up above the axes
    (zero when they are drawn below), so the caller can clear them.
    """
    families = [model_family(m["model_label"]) for m in members]

    # Lay the bars out in family blocks, leaving a gap between blocks so family
    # membership is pre-attentive rather than something to look up.
    positions: list[float] = []
    cursor = 0.0
    for idx, family in enumerate(families):
        if idx > 0 and family != families[idx - 1]:
            cursor += block_gap
        positions.append(cursor)
        cursor += 1.0

    family_counts = Counter(families)
    seen: Counter[str] = Counter()
    colors: list[str] = []
    for member, family in zip(members, families):
        category = _model_category_rank(member["model_label"])[0]
        colors.append(
            _matrix_color(family, category, seen[family], family_counts[family])
        )
        seen[family] += 1

    # Width of one bar slot in points, so the label sizes below can be fitted to
    # the space that actually exists rather than assumed.
    x_span = (positions[-1] + 0.9) - (positions[0] - 0.9)
    axes_pt = ax.get_position().width * fig_size_in[0] * 72.0
    unit_pt = axes_pt / x_span

    heights = np.array([values[m["row"], condition_index] for m in members], dtype=float)
    errors = (
        np.array([sems[m["row"], condition_index] for m in members], dtype=float)
        if show_error_bars
        else np.full(len(members), np.nan)
    )
    errors = np.where(np.isfinite(errors), errors, 0.0)

    ax.bar(
        positions,
        heights,
        width=0.82,
        yerr=errors if show_error_bars else None,
        color=colors,
        edgecolor="#4A4A4A",
        linewidth=0.5,
        error_kw={"linewidth": 0.8, "ecolor": "#333333"},
        capsize=2,
        zorder=3,
    )

    # Per-bar significance vs. chance. Drawn horizontally where a bar is wide
    # enough for "***" at full size, and stacked vertically where it is not —
    # a horizontal string that overflows its bar collides with its neighbours.
    star_pt = 12.0 * font_scale
    y_min, y_max = ax.get_ylim()
    pad = 0.018 * (y_max - y_min)
    for pos, height, err, member in zip(positions, heights, errors, members):
        q = qvalues[member["row"], condition_index]
        if not np.isfinite(q):
            continue
        sig = significance_label(float(q), alpha)
        if not sig:
            continue
        top = height + err if height >= 0 else height - err
        ax.text(
            pos,
            top + (pad if height >= 0 else -pad),
            "\n".join(sig) if 0.62 * star_pt * len(sig) > unit_pt else sig,
            ha="center",
            va="bottom" if height >= 0 else "top",
            linespacing=0.62,
            fontsize=star_pt,
            color="black" if sig == "ns" else "darkred",
            zorder=5,
        )

    ax.set_xlim(positions[0] - 0.9, positions[-1] + 0.9)
    ax.set_xticks(positions)
    size_labels = [model_size(m["model_label"])[1] for m in members]
    size_chars = max(len(t) for t in size_labels)
    size_pt = 12.0 * font_scale
    rotate_sizes = 0.62 * size_pt * size_chars > unit_pt
    ax.set_xticklabels(size_labels, fontsize=size_pt, rotation=90 if rotate_sizes else 0)
    ax.tick_params(axis="x", length=0, pad=3)
    if labels_on_top:
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(axis="x", top=False, labeltop=True, bottom=False, labelbottom=False)

    # Family names as a second axis tier, outside the panel on the same side as
    # the size labels (above for the top row, below for the bottom row) so the
    # gutter between the panels stays free for the pairwise connectors.
    panel_pt = ax.get_position().height * fig_size_in[1] * 72.0
    size_tier_pt = 0.62 * size_pt * size_chars if rotate_sizes else size_pt
    gap_frac = (size_tier_pt * 1.15 + 8.0) / panel_pt
    family_y = 1.0 + gap_frac if labels_on_top else -gap_frac
    names, counts = _run_lengths(families)
    # One size for every block in the panel, set by the tightest block, so the
    # family tier does not look ragged.
    family_pt = min(
        _fit_fontsize(
            12.5 * font_scale,
            count * unit_pt + block_gap * unit_pt,
            len(MATRIX_FAMILY_DISPLAY.get(name, name)),
        )
        for name, count in zip(names, counts)
    )
    start = 0
    for family, count in zip(names, counts):
        display = MATRIX_FAMILY_DISPLAY.get(family, family)
        center = float(np.mean(positions[start : start + count]))
        ax.text(
            center,
            family_y,
            display,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom" if labels_on_top else "top",
            fontsize=family_pt,
            color="#333333",
        )
        start += count

    ax.axhline(y=0, color="black", linewidth=0.5, zorder=1)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Points of vertical space the size + family tiers occupy above the axes,
    # for the caller to pad the column header past.
    return gap_frac * panel_pt + family_pt + 6.0 if labels_on_top else 0.0


def _run_lengths(items: list[str]) -> tuple[list[str], list[int]]:
    """Consecutive runs in *items* as ``(values, lengths)``."""
    names: list[str] = []
    counts: list[int] = []
    for item in items:
        if names and names[-1] == item:
            counts[-1] += 1
        else:
            names.append(item)
            counts.append(1)
    return names, counts


def _connector_label(q: float, alpha: float) -> tuple[str, str]:
    """``(text, colour)`` for a pairwise-comparison connector."""
    sig = significance_label(float(q), alpha) if np.isfinite(q) else ""
    return sig, "#333333" if sig in ("", "ns") else "darkred"


def _draw_matrix_connectors(
    fig,
    axes: np.ndarray,
    *,
    pair_qvalues: dict[tuple[int, int], float],
    panel_of_condition: dict[int, tuple[int, int]],
    alpha: float,
    font_scale: float,
) -> None:
    """Draw the six pairwise-condition comparisons in the gutters between panels.

    The 2x2 layout turns the six comparisons into geometry: the two *row*
    comparisons run horizontally between the panels of a row, the two *column*
    comparisons run vertically between the rows, and the two *diagonal*
    comparisons cross in the central box. The cross-modal asymmetry the results
    section is about is one of the diagonals, so it reads as a single line rather
    than as one bracket among six stacked above the bars.
    """
    condition_of_panel = {panel: cond for cond, panel in panel_of_condition.items()}
    boxes = {
        (r, c): axes[r][c].get_position() for r in range(2) for c in range(2)
    }
    # The gutter between the four panels, in figure coordinates.
    gx0, gx1 = boxes[(0, 0)].x1, boxes[(0, 1)].x0
    gy0, gy1 = boxes[(1, 0)].y1, boxes[(0, 0)].y0
    fs = 14.0 * font_scale

    def pair_q(panel_a: tuple[int, int], panel_b: tuple[int, int]) -> float:
        j, k = sorted((condition_of_panel[panel_a], condition_of_panel[panel_b]))
        return pair_qvalues.get((j, k), np.nan)

    def line(xs, ys, **kwargs) -> None:
        fig.add_artist(
            Line2D(xs, ys, transform=fig.transFigure, color="#666666", linewidth=0.9, **kwargs)
        )

    # Pull the connector ends back from the panels so both end ticks sit in clear
    # space; drawn flush against a spine they read as a plain line with no ends.
    inset_x = 0.16 * (gx1 - gx0)
    inset_y = 0.16 * (gy1 - gy0)
    tick_x = 0.009  # half-length of the end ticks, figure fraction
    tick_y = 0.016

    # Row comparisons: horizontal, at the vertical midpoint of each row.
    for r in range(2):
        q = pair_q((r, 0), (r, 1))
        text, color = _connector_label(q, alpha)
        if not text:
            continue
        y = (boxes[(r, 0)].y0 + boxes[(r, 0)].y1) / 2.0
        x_left, x_right = gx0 + inset_x, gx1 - inset_x
        line([x_left, x_right], [y, y])
        line([x_left, x_left], [y - tick_y, y + tick_y])
        line([x_right, x_right], [y - tick_y, y + tick_y])
        fig.text(
            (gx0 + gx1) / 2.0, y + tick_y + 0.006, text,
            ha="center", va="bottom", fontsize=fs, color=color,
        )

    # Column comparisons: vertical, at the horizontal midpoint of each column.
    for c in range(2):
        q = pair_q((0, c), (1, c))
        text, color = _connector_label(q, alpha)
        if not text:
            continue
        x = (boxes[(0, c)].x0 + boxes[(0, c)].x1) / 2.0
        y_bottom, y_top = gy0 + inset_y, gy1 - inset_y
        line([x, x], [y_bottom, y_top])
        line([x - tick_x, x + tick_x], [y_bottom, y_bottom])
        line([x - tick_x, x + tick_x], [y_top, y_top])
        fig.text(
            x + tick_x + 0.005, (gy0 + gy1) / 2.0, text,
            ha="left", va="center", fontsize=fs, color=color,
        )

    # Diagonal comparisons: corner to corner of the central box. Each label sits
    # a quarter of the way along its own line, where only that line passes, so
    # the two diagonals stay individually readable where they cross.
    diagonals = (
        (((0, 0), (1, 1)), (gx0, gy1), (gx1, gy0), "right"),
        (((0, 1), (1, 0)), (gx1, gy1), (gx0, gy0), "left"),
    )
    for panels, (x_start, y_start), (x_end, y_end), ha in diagonals:
        q = pair_q(*panels)
        text, color = _connector_label(q, alpha)
        if not text:
            continue
        dx, dy = x_end - x_start, y_end - y_start
        x_start, y_start = x_start + 0.14 * dx, y_start + 0.14 * dy
        x_end, y_end = x_end - 0.14 * dx, y_end - 0.14 * dy
        line([x_start, x_end], [y_start, y_end], linestyle=(0, (4, 3)))
        for x_cap, y_cap in ((x_start, y_start), (x_end, y_end)):
            line([x_cap - tick_x, x_cap + tick_x], [y_cap, y_cap])
        t = 0.16
        fig.text(
            x_start + t * (x_end - x_start) + (-0.005 if ha == "right" else 0.005),
            y_start + t * (y_end - y_start) + 0.006,
            text,
            ha=ha, va="bottom", fontsize=fs, color=color,
        )


def _wrap_row_label(text: str, max_chars: int = 12) -> str:
    """Break a row label onto two balanced lines when it is long.

    The row label is rotated, so its length is bounded by the *panel height*.
    "Residual image embeddings" set on one line is taller than the panel it
    labels; split near the middle it fits at full size.
    """
    if len(text) <= max_chars or " " not in text:
        return text
    words = text.split()
    split = min(
        range(1, len(words)),
        key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))),
    )
    return " ".join(words[:split]) + "\n" + " ".join(words[split:])


def _matrix_colour_note(categories: set[int]) -> str:
    """Caption clause naming the colour families, for the categories drawn."""
    words = [
        word
        for rank, word in (
            (0, "purples: vision–language"),
            (1, "reds: vision-only"),
            (2, "blues: text-only"),
        )
        if rank in categories
    ]
    return f" ({'; '.join(words)})" if words else ""


def _matrix_scale_note(families: set[str]) -> str:
    """Caption clause defining the ViT-scale letters, for the families drawn.

    Only the parameter-count families are named by size in the paper, so the
    letters need spelling out — but only for the families this figure contains.
    """
    parts: list[str] = ["parameter count in billions (B)"]
    if families & {"CLIP", "DINOv2"}:
        parts.append("L = ViT-L/14")  # noqa: E501
    if families & {"CLIP", "I-JEPA"}:
        parts.append("H = ViT-H/14")
    if {"DINOv2", "I-JEPA"} <= families:
        parts.append("G = ViT-g/14 for DINOv2 and ViT-g/16 for I-JEPA")
    elif "DINOv2" in families:
        parts.append("G = ViT-g/14")
    elif "I-JEPA" in families:
        parts.append("G = ViT-g/16")
    return "Bar labels give model scale: " + ", ".join(parts) + "."


def plot_condition_matrix(
    model_results: list[dict[str, Any]],
    *,
    metric: str,
    alpha: float,
    font_scale: float,
    output_path: Path | None,
    figsize: tuple[float, float],
    y_limits: tuple[float, float] | None,
    group_sig_permutations: int = 10000,
    group_sig_random_state: int = 42,
    group_sig_correction: str = "fdr_bh",
    title: str = "Cross-modal neural encoding",
    show_error_bars: bool = True,
    row_labels: tuple[str, str] = ("Image embeddings", "Text embeddings"),
    col_labels: tuple[str, str] = ("→ Image fMRI", "→ Text fMRI"),
) -> None:
    """Plot the four encoding conditions as a 2x2 embedding x fMRI matrix.

    Rows are the embedding modality, columns the fMRI modality, so the diagonal
    is within-modality encoding and the off-diagonal is cross-modal. All four
    panels share one y-axis, which is what makes the panels comparable by eye.

    Two structural facts do the decluttering relative to the single-axis grouped
    figure. First, the vision-only baselines have no text embeddings and the
    text-only baselines no image embeddings, so each row simply shows the models
    that apply and the "na" placeholders disappear. Second, the six pairwise
    condition comparisons become the connectors between panels instead of six
    brackets stacked above the bars, which frees the vertical range those
    brackets used to reserve.
    """
    conditions = _condition_order_from_index(
        {c for item in model_results for c in item["aggregated_df"].index}
    )
    row_modalities = ("image", "text")  # embedding modality
    col_modalities = ("image", "text")  # fMRI modality
    panel_of_condition: dict[int, tuple[int, int]] = {}
    for r, row_mod in enumerate(row_modalities):
        for c, col_mod in enumerate(col_modalities):
            key = f"{row_mod}_to_{col_mod}"
            if key not in conditions:
                logger.warning(f"Condition matrix needs '{key}'; skipping figure.")
                return
            panel_of_condition[conditions.index(key)] = (r, c)

    values, qvalues, sems = _group_stats_matrices(
        model_results,
        conditions,
        metric=metric,
        group_sig_permutations=group_sig_permutations,
        group_sig_random_state=group_sig_random_state,
        group_sig_correction=group_sig_correction,
        show_error_bars=show_error_bars,
    )
    pair_qvalues = pairwise_condition_signrank(values, correction=group_sig_correction)

    if y_limits is None:
        finite = np.isfinite(values)
        if not np.any(finite):
            logger.warning("No finite values available for the condition matrix.")
            return
        spread = np.where(np.isfinite(sems), sems, 0.0)[finite]
        y_limits = _compute_plot_ylims(
            values[finite],
            spread,
            is_normalized_metric="normalized" in metric.lower(),
            compress_normalized_axis=False,
            normalized_axis_linthresh=0.08,
        )

    families_drawn = {model_family(item["model_label"]) for item in model_results}
    scale_note = _matrix_scale_note(families_drawn)
    colour_note = _matrix_colour_note(
        {_model_category_rank(item["model_label"])[0] for item in model_results}
    )
    missing_note = (
        ""
        if np.all(np.isfinite(values))
        else "  Each row shows only the models that have embeddings of that modality."
    )
    footnote = (
        "Bars: per-model group mean ± SEM across subjects.  Colour = model family"
        + colour_note
        + ", lightness = model size within family.\n"
        + scale_note
        + "\nStars on bars: encoding vs. chance.  Connectors between panels: pairwise "
        "condition comparisons (Wilcoxon signed-rank across models).\n"
        "All q-values BH-FDR corrected — *** q<0.001, ** q<0.01, * q<0.05, "
        "ns not significant." + missing_note
    )
    footnote_pt = 8.5 * font_scale

    row_labels = (_wrap_row_label(row_labels[0]), _wrap_row_label(row_labels[1]))
    longest_row_line = max(len(line) for label in row_labels for line in label.split("\n"))
    row_lines = max(label.count("\n") + 1 for label in row_labels)
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    # The title block and the row labels both eat into the panel area, and both
    # grow with the text they are given, so size their margins from that text.
    # Above the top row sit, in order: the size labels, the family names, the
    # column header and the suptitle. Reserve the worst case for all four —
    # _fit_fontsize only ever shrinks the label tiers from these ceilings, so an
    # upper bound here can waste a little space but can never collide.
    size_chars = max(
        len(model_size(item["model_label"])[1]) for item in model_results
    )
    top_stack_pt = (
        1.15 * 0.62 * (12.0 * font_scale) * size_chars + 8.0  # size-label tier
        + 12.5 * font_scale + 6.0  # family tier
        + 16.5 * font_scale + 8.0  # column header
        + 18.5 * font_scale * 1.25 * (title.count("\n") + 1)  # suptitle
        + 16.0
    )
    top_margin = 1.0 - top_stack_pt / (figsize[1] * 72.0)
    bottom_stack_pt = (
        1.15 * 0.62 * (12.0 * font_scale) * size_chars + 8.0  # size-label tier
        + 12.5 * font_scale + 6.0  # family tier
        + footnote_pt * 1.5 * (footnote.count("\n") + 1)  # footnote block
        + 16.0
    )
    bottom_margin = bottom_stack_pt / (figsize[1] * 72.0)

    # Left of the panels sit the shared metric label, the (rotated) row label
    # and the y tick labels. The row label's size depends on the panel height,
    # which is fixed by the margins above, so it can be resolved here and reused
    # when the label is actually set.
    panel_pt_est = (
        (top_margin - bottom_margin) / (2.0 + MATRIX_HSPACE) * figsize[1] * 72.0
    )
    row_label_pt = _fit_fontsize(17.0 * font_scale, panel_pt_est, longest_row_line)
    tick_pt = 13.5 * font_scale
    left_margin = (
        tick_pt * 1.5  # shared metric label, rotated
        + row_label_pt * 1.25 * row_lines  # row label, rotated
        + 0.62 * tick_pt * 5  # widest y tick label ("-0.05")
        + 34.0
    ) / (figsize[0] * 72.0)
    fig.subplots_adjust(
        left=left_margin, right=0.985, top=top_margin, bottom=bottom_margin,
        wspace=MATRIX_WSPACE, hspace=MATRIX_HSPACE,
    )
    panel_members = {
        condition_index: _matrix_panel_members(model_results, values, condition_index)
        for condition_index in panel_of_condition
    }

    # Where a bar is too narrow for a horizontal "***" the stars stack, and the
    # stack needs clear space above the tallest bar or it runs into the family
    # labels. Convert that requirement from points into data units: to leave a
    # fraction f of the panel height clear, the span has to grow by f*span/(1-f).
    panel_box = axes[0][0].get_position()
    panel_pt = panel_box.height * figsize[1] * 72.0
    panel_w_pt = panel_box.width * figsize[0] * 72.0

    def _unit_pt(members: list[dict[str, Any]]) -> float:
        """Width of one bar slot in a panel holding *members*, in points."""
        n_blocks = len({model_family(m["model_label"]) for m in members})
        return panel_w_pt / (len(members) + BLOCK_GAP * (n_blocks - 1) + 1.8)

    unit_pt = min(_unit_pt(members) for members in panel_members.values())
    star_pt = 12.0 * font_scale
    stacked = 0.62 * star_pt * 3 > unit_pt
    star_pt_h = (3 * star_pt * 0.62 if stacked else star_pt) + 8.0
    headroom = star_pt_h / panel_pt
    if 0.0 < headroom < 0.5:
        span = y_limits[1] - y_limits[0]
        grow = headroom * span / (1.0 - headroom)
        # Bars below zero carry their stars underneath, so they need the same
        # clearance at the bottom of the panel as the upward bars do at the top.
        low = y_limits[0] - (grow if np.nanmin(values) < 0 else 0.0)
        y_limits = (low, y_limits[1] + grow)

    for ax in axes.flat:
        ax.set_ylim(*y_limits)

    header_pad = 0.0
    for condition_index, (r, c) in panel_of_condition.items():
        clearance = _draw_matrix_panel(
            axes[r][c],
            panel_members[condition_index],
            values=values,
            qvalues=qvalues,
            sems=sems,
            condition_index=condition_index,
            alpha=alpha,
            font_scale=font_scale,
            show_error_bars=show_error_bars,
            labels_on_top=(r == 0),
            fig_size_in=figsize,
        )
        header_pad = max(header_pad, clearance)

    # Column headers name the fMRI modality, row labels the embedding modality,
    # so each panel is read as the intersection of the two.
    for c in range(len(col_modalities)):
        axes[0][c].set_title(
            col_labels[c],
            fontsize=16.5 * font_scale,
            fontweight="bold",
            pad=header_pad + 8.0,
        )
    for r in range(len(row_modalities)):
        axes[r][0].set_ylabel(
            row_labels[r],
            fontsize=row_label_pt,
            fontweight="bold",
            labelpad=16,
        )
        axes[r][0].tick_params(axis="y", labelsize=13.5 * font_scale)
        axes[r][0].yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5]))
        axes[r][0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    metric_label = (
        "Normalized performance (r / noise ceiling)"
        if "normalized" in metric.lower()
        else "Pearson correlation"
    )
    fig.supylabel(metric_label, fontsize=13.5 * font_scale, x=0.007)
    fig.suptitle(title, fontsize=18.5 * font_scale, fontweight="bold", y=0.985)

    # Positions are only final once the layout is fixed, and the connectors are
    # placed in figure coordinates from the panel boxes.
    fig.canvas.draw()
    _draw_matrix_connectors(
        fig,
        axes,
        pair_qvalues=pair_qvalues,
        panel_of_condition=panel_of_condition,
        alpha=alpha,
        font_scale=font_scale,
    )

    fig.text(
        0.5,
        0.012,
        footnote,
        ha="center",
        va="bottom",
        fontsize=footnote_pt,
        linespacing=1.5,
        color="#4A4A4A",
    )

    if output_path is None:
        output_path = FIGURES_DIR / "encoding_results_condition_matrix.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.success(f"Saved condition-matrix figure to {output_path}")


def plot_condition_violins(
    model_results: list[dict[str, Any]],
    *,
    metric: str,
    alpha: float,
    font_scale: float,
    output_path: Path | None,
    figsize: tuple[float, float],
    y_limits: tuple[float, float] | None = None,
    group_sig_correction: str = "fdr_bh",
    title: str = "Condition Comparison Across All Models",
    condition_labels: dict[str, str] | None = None,
) -> None:
    """Collapse all models into one violin per condition.

    Each violin is the distribution, over models, of the per-model group mean —
    exactly the quantity the grouped bar figure draws as one bar — so the two
    figures show the same numbers at different granularity. Every model is
    overlaid as a jittered point coloured by its category, and the conditions are
    compared with the same paired Wilcoxon signed-rank tests as the grouped
    figure (models are the pairing unit), drawn as brackets above the violins.

    Models are not run on every condition (text-only models have no image
    conditions and vice versa), so the violins do not all summarise the same set
    of models; the per-condition model count is printed under each x label.
    """
    if len(model_results) < 2:
        logger.warning("Condition violin plot requires at least two models.")
        return

    model_results = sorted(
        model_results,
        key=lambda item: _model_category_rank(item.get("model_label", "")),
    )

    all_conditions: set[str] = set()
    for item in model_results:
        all_conditions.update(item["aggregated_df"].index)
    if not all_conditions:
        logger.warning("No conditions found across models to plot.")
        return

    conditions = _condition_order_from_index(all_conditions)
    label_map = condition_labels if condition_labels is not None else CONDITION_LABELS
    values = _model_condition_matrix(model_results, conditions, metric)
    if not np.any(np.isfinite(values)):
        logger.warning("No finite values available for the condition violin plot.")
        return

    categories = [_model_category_rank(item["model_label"])[0] for item in model_results]
    is_normalized_metric = "normalized" in metric.lower()
    n_conditions = len(conditions)
    x = np.arange(n_conditions)
    datasets = [values[np.isfinite(values[:, j]), j] for j in range(n_conditions)]
    labels = [
        f"{label_map.get(c, c)}\n(n = {d.size} models)" for c, d in zip(conditions, datasets)
    ]

    fig, ax = plt.subplots(figsize=figsize)

    # ── Violin bodies ─────────────────────────────────────────────────────
    # A KDE needs at least two distinct points; conditions with fewer models
    # are shown by their overlaid points alone.
    violin_idx = [j for j, d in enumerate(datasets) if d.size >= 2 and np.ptp(d) > 0]
    tops = np.array([np.nanmax(d) if d.size else np.nan for d in datasets], dtype=float)
    bottoms = np.array([np.nanmin(d) if d.size else np.nan for d in datasets], dtype=float)
    if violin_idx:
        parts = ax.violinplot(
            [datasets[j] for j in violin_idx],
            positions=x[violin_idx],
            widths=0.72,
            showextrema=False,
            showmedians=False,
        )
        for body, j in zip(parts["bodies"], violin_idx):  # type: ignore[arg-type]
            body.set_facecolor(VIOLIN_BODY_COLOR)
            body.set_edgecolor("#4A4A4A")
            body.set_linewidth(0.8)
            body.set_alpha(0.75)
            # The KDE tails, not the data range, set how far the drawing extends.
            vertices = body.get_paths()[0].vertices[:, 1]
            tops[j] = max(tops[j], float(np.max(vertices)))
            bottoms[j] = min(bottoms[j], float(np.min(vertices)))

    # ── Per-model points ──────────────────────────────────────────────────
    # Deterministic jitter so the figure is reproducible across regenerations.
    rng = np.random.default_rng(0)
    seen_categories: set[int] = set()
    for i, category in enumerate(categories):
        finite = np.isfinite(values[i])
        if not np.any(finite):
            continue
        jitter = rng.uniform(-0.12, 0.12, size=int(finite.sum()))
        label = (
            CATEGORY_LABELS.get(category, "Other") if category not in seen_categories else None
        )
        seen_categories.add(category)
        ax.scatter(
            x[finite] + jitter,
            values[i, finite],
            s=26,
            color=CATEGORY_POINT_COLORS.get(category, GROUP_BAR_COLOR),
            edgecolor="#3A3A3A",
            linewidth=0.4,
            alpha=0.9,
            zorder=4,
            label=label,
        )

    # ── Medians ───────────────────────────────────────────────────────────
    for j, d in enumerate(datasets):
        if d.size == 0:
            continue
        ax.hlines(
            float(np.median(d)),
            x[j] - 0.22,
            x[j] + 0.22,
            color="#2B2B2B",
            linewidth=1.8,
            zorder=5,
        )

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9 * font_scale, rotation=0, ha="center")
    ax.tick_params(axis="y", labelsize=12 * font_scale)
    if is_normalized_metric:
        ax.set_ylabel("Normalized performance\n(r / NC)", fontsize=10 * font_scale)
    else:
        ax.set_ylabel("Pearson correlation", fontsize=10 * font_scale)
    ax.set_title(title, fontsize=15 * font_scale, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.5, zorder=1)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.margins(x=0.03)

    if y_limits is None:
        finite_tops = tops[np.isfinite(tops)]
        finite_bottoms = bottoms[np.isfinite(bottoms)]
        lower = float(np.min(finite_bottoms))
        upper = float(np.max(finite_tops))
        span = max(upper - lower, 0.03)
        y_min, y_max = min(-0.01, lower - 0.06 * span), upper + 0.08 * span
    else:
        y_min, y_max = y_limits
    # Headroom for the note strip under the lowest violin.
    y_min -= 0.14 * (y_max - y_min)
    ax.set_ylim(y_min, y_max)

    note_text = "One point = one model   ·   line = median   ·   brackets: pairwise signed-rank"
    if group_sig_correction == "fdr_bh":
        note_text += " (BH-FDR)"
    ax.text(
        0.01,
        0.02,
        note_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7 * font_scale,
        color="#333333",
        zorder=6,
    )

    legend_fs = 10.5 * font_scale
    handles, legend_labels = ax.get_legend_handles_labels()
    category_legend = None
    if handles:
        category_legend = ax.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                _legend_y_below_xticklabels(fig, ax, fallback=-0.20, nudge=-0.05),
            ),
            fontsize=legend_fs,
            ncol=_legend_ncol(legend_labels, figsize[0], legend_fs, len(legend_labels)),
            frameon=False,
            columnspacing=1.4,
            handletextpad=0.5,
        )

    # Same test (and same pairing unit) as the grouped bar figure, so the
    # brackets carry over unchanged.
    pair_q = pairwise_condition_signrank(values, correction=group_sig_correction)
    annotate_pairwise_brackets(
        ax,
        x_positions=x,
        values=tops,
        pair_qvalues=pair_q,
        alpha=alpha,
        font_scale=font_scale,
    )

    if output_path is None:
        output_path = FIGURES_DIR / "encoding_results_condition_violins.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=() if category_legend is None else (category_legend,),
    )
    logger.success(f"Condition-violin figure saved to {output_path}")
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
        subject_pvals = combined.pivot_table(
            index="subject",
            columns="condition",
            values=p_value_col,
            aggfunc="mean",
        ).reindex(index=subject_table.index, columns=conditions)
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
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=10 * font_scale,
        ncol=1,
        frameon=False,
    )
    ax.margins(x=0.03)

    if is_normalized_metric and compress_normalized_axis:
        ax.set_yscale(
            "symlog",
            linthresh=max(normalized_axis_linthresh, 1e-3),
            linscale=1.0,
            base=10,
        )
    else:
        # This panel is short and its tick labels are large, so matplotlib's
        # default locator thins the axis down to two ticks (0.0 and 0.2), which
        # makes individual subject bars hard to read off. Ask for a denser set of
        # round-numbered ticks. Skipped under symlog, which needs its own locator.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))

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
            raise FileNotFoundError(f"No model subfolders with aggregated.csv found in: {run_dir}")

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
            except (OSError, pd.errors.ParserError) as exc:
                logger.warning(f"Skipping {child.name}: failed to load aggregated.csv ({exc})")
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
                    "model_dir": child,
                }
            )

        if not model_results:
            raise FileNotFoundError(f"No model subfolders with aggregated.csv found in: {run_dir}")
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
                "model_dir": path.parent,
            }
        )

    output_path = Path(cfg.output_path) if cfg.output_path else None
    font_scale = float(cfg.get("font_scale", 1.15))
    group_sig_correction = str(cfg.get("group_sig_correction", "fdr_bh"))
    compress_normalized_axis = bool(cfg.get("compress_normalized_axis", False))
    normalized_axis_linthresh = float(cfg.get("normalized_axis_linthresh", 0.08))
    share_y_limits = bool(cfg.get("share_y_limits", True))
    y_limits_cfg = cfg.get("y_limits", None)
    if y_limits_cfg is not None:
        shared_y_limits = (float(y_limits_cfg[0]), float(y_limits_cfg[1]))
    elif share_y_limits:
        shared_y_limits = _shared_y_limits(
            model_results,
            metric=cfg.metric,
            compress_normalized_axis=compress_normalized_axis,
            normalized_axis_linthresh=normalized_axis_linthresh,
        )
    else:
        shared_y_limits = None

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
            group_sig_correction=group_sig_correction,
            font_scale=font_scale,
            compress_normalized_axis=compress_normalized_axis,
            normalized_axis_linthresh=normalized_axis_linthresh,
            output_path=output_path,
            figsize=tuple(cfg.figsize),
            y_limits=shared_y_limits,
        )

    if bool(cfg.get("plot_grouped_models", True)):
        grouped_output = (
            Path(cfg.grouped_output_path) if cfg.get("grouped_output_path", None) else None
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
            group_sig_permutations=int(cfg.get("group_sig_permutations", 10000)),
            group_sig_random_state=int(cfg.get("group_sig_random_state", 42)),
            group_sig_correction=group_sig_correction,
            show_error_bars=bool(cfg.get("show_error_bars", True)),
        )

    if bool(cfg.get("plot_best_per_family", True)):
        best_results = select_best_per_family(model_results, metric=cfg.metric)
        best_output = (
            Path(cfg.best_per_family_output_path)
            if cfg.get("best_per_family_output_path", None)
            else FIGURES_DIR / "encoding_results_best_per_family_grouped.png"
        )
        plot_grouped_model_means(
            best_results,
            metric=cfg.metric,
            alpha=cfg.alpha,
            font_scale=font_scale,
            compress_normalized_axis=compress_normalized_axis,
            normalized_axis_linthresh=normalized_axis_linthresh,
            output_path=best_output,
            figsize=tuple(
                cfg.get("best_per_family_figsize", cfg.get("grouped_figsize", cfg.figsize))
            ),
            y_limits=shared_y_limits,
            group_sig_permutations=int(cfg.get("group_sig_permutations", 10000)),
            group_sig_random_state=int(cfg.get("group_sig_random_state", 42)),
            group_sig_correction=group_sig_correction,
            show_error_bars=bool(cfg.get("show_error_bars", True)),
            title="Best Model per Family Based on Group Means",
        )

    if bool(cfg.get("plot_condition_matrix", True)):
        matrix_y_cfg = cfg.get("condition_matrix_y_limits", None)
        plot_condition_matrix(
            model_results,
            metric=cfg.metric,
            alpha=cfg.alpha,
            font_scale=font_scale,
            output_path=(
                Path(cfg.condition_matrix_output_path)
                if cfg.get("condition_matrix_output_path", None)
                else None
            ),
            figsize=tuple(cfg.get("condition_matrix_figsize", [17.0, 9.2])),
            # Independent of the shared y_limits: those reserve headroom for the
            # pairwise brackets the other figures stack above their bars, which
            # this layout moves out of the axes entirely. All four panels still
            # share one scale, which is what makes them comparable.
            y_limits=(
                (float(matrix_y_cfg[0]), float(matrix_y_cfg[1]))
                if matrix_y_cfg is not None
                else None
            ),
            group_sig_permutations=int(cfg.get("group_sig_permutations", 10000)),
            group_sig_random_state=int(cfg.get("group_sig_random_state", 42)),
            group_sig_correction=group_sig_correction,
            show_error_bars=bool(cfg.get("show_error_bars", True)),
            title=str(cfg.get("condition_matrix_title", "Cross-modal neural encoding")),
        )

    if bool(cfg.get("plot_condition_violins", True)):
        violin_y_limits_cfg = cfg.get("violin_y_limits", None)
        plot_condition_violins(
            model_results,
            metric=cfg.metric,
            alpha=cfg.alpha,
            font_scale=font_scale,
            output_path=(
                Path(cfg.violin_output_path) if cfg.get("violin_output_path", None) else None
            ),
            figsize=tuple(cfg.get("violin_figsize", cfg.figsize)),
            y_limits=(
                (float(violin_y_limits_cfg[0]), float(violin_y_limits_cfg[1]))
                if violin_y_limits_cfg is not None
                else None
            ),
            group_sig_correction=group_sig_correction,
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
