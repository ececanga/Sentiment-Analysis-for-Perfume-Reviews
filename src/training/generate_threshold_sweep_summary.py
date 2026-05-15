# Date Created: 2026-05-06 15:46:46+03:00
# Date Last Modified: 2026-05-15

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.training.generate_final_benchmark_figures import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_NAMES,
    MODEL_ORDER,
    PROJECT_ROOT,
    _rounded_axis_ceiling,
    configure_plot_style,
)

DEFAULT_SUMMARY_ROOT = PROJECT_ROOT / "outputs" / "model_benchmark_threshold_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed-threshold comparison tables and plots from benchmark output roots."
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=DEFAULT_SUMMARY_ROOT,
        help="Output root for the threshold sweep summary artifacts.",
    )
    parser.add_argument(
        "--threshold-root",
        type=Path,
        action="append",
        required=True,
        help="Benchmark output root for one fixed-threshold run. Repeat for each threshold root.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional subset of official model ids to include in the summary.",
    )
    return parser.parse_args()


def format_threshold_value(threshold: float) -> str:
    return f"{threshold:.2f}".rstrip("0").rstrip(".")


def threshold_to_tag(threshold: float) -> str:
    threshold = float(threshold)
    return f"t{int(round(threshold * 10)):02d}"


def normalize_models(models: Sequence[str] | None) -> List[str]:
    ordered = list(MODEL_ORDER)
    if models is None:
        return ordered

    requested = [str(model_id) for model_id in models]
    invalid = sorted(set(requested) - set(ordered))
    if invalid:
        raise ValueError(f"Unknown model ids: {', '.join(invalid)}")

    deduped: List[str] = []
    for model_id in requested:
        if model_id not in deduped:
            deduped.append(model_id)
    return [model_id for model_id in ordered if model_id in deduped]


def _load_completed_rows(table_path: Path) -> pd.DataFrame:
    if not table_path.exists():
        raise FileNotFoundError(f"Missing comparison table: {table_path}")

    df = pd.read_csv(table_path)
    required_columns = {"model_id", "threshold", "macro_f1", "macro_roc_auc", "macro_pr_auc", "row_complete"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {table_path}: {', '.join(missing)}")

    out = df.loc[df["model_id"].isin(MODEL_ORDER)].copy()
    out["row_complete"] = out["row_complete"].astype(str).str.lower().isin({"true", "1"})
    out = out.loc[out["row_complete"]].copy()
    for metric_col in ["threshold", "macro_f1", "macro_roc_auc", "macro_pr_auc"]:
        out[metric_col] = pd.to_numeric(out[metric_col], errors="coerce")
    return out


def _resolve_summary_models(
    completed_by_root: Sequence[pd.DataFrame],
    requested_models: Sequence[str] | None,
) -> List[str]:
    if requested_models is not None:
        return normalize_models(requested_models)

    if not completed_by_root:
        raise ValueError("No threshold roots were provided.")

    model_sets = [set(df["model_id"].tolist()) for df in completed_by_root]
    available = set.intersection(*model_sets) if model_sets else set()
    ordered = [model_id for model_id in MODEL_ORDER if model_id in available]
    if not ordered:
        raise ValueError("No complete official model rows were found across the provided threshold roots.")
    return ordered


def build_long_form_summary(
    *,
    threshold_roots: Sequence[Path],
    models: Sequence[str] | None,
) -> tuple[pd.DataFrame, List[str], List[float]]:
    completed_by_root = [_load_completed_rows(root / "tables" / "comparison_table.csv") for root in threshold_roots]
    ordered_models = _resolve_summary_models(completed_by_root, models)

    rows: List[Dict[str, Any]] = []
    threshold_order: List[float] = []

    for root, completed_df in zip(threshold_roots, completed_by_root):
        filtered = completed_df.loc[completed_df["model_id"].isin(ordered_models)].copy()
        if filtered.empty:
            raise ValueError(f"No matching completed rows found in {root / 'tables' / 'comparison_table.csv'}")

        threshold_values = filtered["threshold"].dropna().unique().tolist()
        if len(threshold_values) != 1:
            raise ValueError(f"Expected exactly one threshold value in {root / 'tables' / 'comparison_table.csv'}")

        threshold = float(threshold_values[0])
        threshold_order.append(threshold)
        expected_models = set(ordered_models)
        actual_models = set(filtered["model_id"].tolist())
        missing_models = [model_id for model_id in ordered_models if model_id not in actual_models]
        if missing_models:
            raise ValueError(
                f"Threshold root {root} is missing complete rows for: {', '.join(missing_models)}"
            )

        filtered["sort_key"] = filtered["model_id"].map({model_id: idx for idx, model_id in enumerate(MODEL_ORDER)})
        filtered = filtered.sort_values("sort_key")

        for _, row in filtered.iterrows():
            model_id = str(row["model_id"])
            metrics_path = root / "metrics" / f"{model_id}.json"
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics JSON for {model_id}: {metrics_path}")
            rows.append(
                {
                    "model_id": model_id,
                    "model_name": MODEL_NAMES[model_id],
                    "threshold": threshold,
                    "threshold_tag": threshold_to_tag(threshold),
                    "out_root": str(root),
                    "macro_f1": float(row["macro_f1"]),
                    "macro_roc_auc": float(row["macro_roc_auc"]),
                    "macro_pr_auc": float(row["macro_pr_auc"]),
                    "comparison_table_path": str(root / "tables" / "comparison_table.csv"),
                    "metrics_path": str(metrics_path),
                }
            )

    if len(set(threshold_order)) != len(threshold_order):
        raise ValueError("Duplicate threshold values were provided in the threshold roots.")

    long_df = pd.DataFrame(rows)
    threshold_rank = {value: idx for idx, value in enumerate(threshold_order)}
    model_rank = {model_id: idx for idx, model_id in enumerate(MODEL_ORDER)}
    long_df["threshold_rank"] = long_df["threshold"].map(threshold_rank)
    long_df["model_rank"] = long_df["model_id"].map(model_rank)
    long_df = long_df.sort_values(["model_rank", "threshold_rank"]).reset_index(drop=True)
    long_df = long_df.drop(columns=["threshold_rank", "model_rank"])
    return long_df, ordered_models, threshold_order


def build_wide_summary(long_df: pd.DataFrame, *, ordered_models: Sequence[str], threshold_order: Sequence[float]) -> pd.DataFrame:
    pivot = long_df.pivot(index="model_id", columns="threshold_tag", values="macro_f1")
    rename_map = {threshold_to_tag(threshold): f"macro_f1_{threshold_to_tag(threshold)}" for threshold in threshold_order}
    pivot = pivot.rename(columns=rename_map)

    records: List[Dict[str, Any]] = []
    for model_id in ordered_models:
        row = {"model_id": model_id, "model_name": MODEL_NAMES[model_id]}
        best_threshold = None
        best_macro_f1 = float("-inf")
        for threshold in threshold_order:
            tag = threshold_to_tag(threshold)
            column_name = f"macro_f1_{tag}"
            value = float(pivot.loc[model_id, column_name]) if model_id in pivot.index else float("nan")
            row[column_name] = value
            if value > best_macro_f1:
                best_macro_f1 = value
                best_threshold = threshold
        row["best_threshold"] = float(best_threshold) if best_threshold is not None else float("nan")
        row["best_macro_f1"] = float(best_macro_f1) if best_threshold is not None else float("nan")
        records.append(row)

    return pd.DataFrame(records)


def save_macro_f1_line_plot(
    long_df: pd.DataFrame,
    *,
    ordered_models: Sequence[str],
    threshold_order: Sequence[float],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    y_max = _rounded_axis_ceiling(float(long_df["macro_f1"].max()))

    for model_id in ordered_models:
        model_df = long_df.loc[long_df["model_id"] == model_id].copy()
        model_df["threshold_rank"] = model_df["threshold"].map({value: idx for idx, value in enumerate(threshold_order)})
        model_df = model_df.sort_values("threshold_rank")
        x_values = model_df["threshold"].to_numpy()
        y_values = model_df["macro_f1"].to_numpy()
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.0,
            markersize=6,
            color=MODEL_COLORS[model_id],
            label=MODEL_NAMES[model_id],
        )
        for x, y in zip(x_values, y_values):
            ax.text(x, y + y_max * 0.015, f"{y:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Fixed Baseline Threshold")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Fixed-Baseline Macro-F1 by Threshold")
    ax.set_xticks(list(threshold_order))
    ax.set_xticklabels([format_threshold_value(value) for value in threshold_order])
    ax.set_ylim(0.0, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_grouped_macro_f1_plot(
    wide_df: pd.DataFrame,
    *,
    threshold_order: Sequence[float],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 6.1))
    x = np.arange(len(wide_df))
    width = min(0.72 / max(len(threshold_order), 1), 0.20)
    offsets = np.linspace(-width * (len(threshold_order) - 1) / 2, width * (len(threshold_order) - 1) / 2, len(threshold_order))

    y_columns = [f"macro_f1_{threshold_to_tag(threshold)}" for threshold in threshold_order]
    y_max = _rounded_axis_ceiling(float(wide_df[y_columns].to_numpy().max()))
    palette = plt.cm.Set2(np.linspace(0.15, 0.85, len(threshold_order)))

    for offset, threshold, color in zip(offsets, threshold_order, palette):
        column_name = f"macro_f1_{threshold_to_tag(threshold)}"
        bars = ax.bar(
            x + offset,
            wide_df[column_name].to_numpy(),
            width=width,
            label=f"Threshold {format_threshold_value(threshold)}",
            color=color,
            edgecolor="#2B2B2B",
            linewidth=0.6,
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (y_max * 0.015),
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[model_id] for model_id in wide_df["model_id"].tolist()])
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Fixed-Baseline Macro-F1 Across Thresholds")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=min(4, len(threshold_order)), loc="upper center", bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_threshold_comparison_table_image(
    wide_df: pd.DataFrame,
    *,
    threshold_order: Sequence[float],
    path: Path,
) -> None:
    display_columns = [f"Macro-F1 @{format_threshold_value(threshold)}" for threshold in threshold_order]
    display_df = pd.DataFrame({"Model": wide_df["model_name"]})
    for threshold, column_name in zip(threshold_order, display_columns):
        display_df[column_name] = wide_df[f"macro_f1_{threshold_to_tag(threshold)}"].map(lambda value: f"{float(value):.3f}")
    display_df["Best Threshold"] = wide_df["best_threshold"].map(format_threshold_value)
    display_df["Best Macro-F1"] = wide_df["best_macro_f1"].map(lambda value: f"{float(value):.3f}")

    fig_height = max(2.4, 1.25 + 0.58 * len(display_df))
    fig, ax = plt.subplots(figsize=(12.6, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values.tolist(),
        colLabels=display_df.columns.tolist(),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    header_color = "#E9EEF6"
    highlight_color = "#DFF3E4"
    border_color = "#B8C2CC"
    metric_columns = {display_columns[idx]: idx + 1 for idx in range(len(display_columns))}

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(border_color)
        if row_idx == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FFFFFF")

    for data_row_idx, (_, row) in enumerate(wide_df.iterrows(), start=1):
        best_threshold = float(row["best_threshold"])
        best_column_label = f"Macro-F1 @{format_threshold_value(best_threshold)}"
        if best_column_label in metric_columns:
            table[(data_row_idx, metric_columns[best_column_label])].set_facecolor(highlight_color)
        table[(data_row_idx, len(display_df.columns) - 1)].set_facecolor(highlight_color)

    ax.set_title("Fixed-Baseline Threshold Comparison", fontsize=14, pad=18)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_threshold_sweep_summary(
    *,
    summary_root: Path,
    threshold_roots: Sequence[Path],
    models: Sequence[str] | None = None,
) -> Dict[str, str]:
    if not threshold_roots:
        raise ValueError("At least one threshold root is required.")

    configure_plot_style()

    tables_dir = summary_root / "tables"
    figures_dir = summary_root / "figures"
    for path in [tables_dir, figures_dir]:
        path.mkdir(parents=True, exist_ok=True)

    long_df, ordered_models, threshold_order = build_long_form_summary(
        threshold_roots=threshold_roots,
        models=models,
    )
    wide_df = build_wide_summary(long_df, ordered_models=ordered_models, threshold_order=threshold_order)
    best_df = wide_df.loc[:, ["model_id", "model_name", "best_threshold", "best_macro_f1"]].copy()

    long_path = tables_dir / "fixed_threshold_comparison_long.csv"
    wide_path = tables_dir / "fixed_threshold_macro_f1_wide.csv"
    thesis_path = tables_dir / "final_model_threshold_comparison_for_thesis.csv"
    best_path = tables_dir / "best_fixed_threshold_per_model.csv"
    line_plot_path = figures_dir / "macro_f1_vs_threshold_by_model.png"
    grouped_plot_path = figures_dir / "grouped_macro_f1_by_threshold.png"
    table_plot_path = figures_dir / "final_model_threshold_comparison_table.png"

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    wide_df.to_csv(thesis_path, index=False)
    best_df.to_csv(best_path, index=False)

    save_macro_f1_line_plot(long_df, ordered_models=ordered_models, threshold_order=threshold_order, path=line_plot_path)
    save_grouped_macro_f1_plot(wide_df, threshold_order=threshold_order, path=grouped_plot_path)
    save_threshold_comparison_table_image(wide_df, threshold_order=threshold_order, path=table_plot_path)

    return {
        "long_table_path": str(long_path),
        "wide_table_path": str(wide_path),
        "thesis_table_path": str(thesis_path),
        "best_threshold_path": str(best_path),
        "line_plot_path": str(line_plot_path),
        "grouped_plot_path": str(grouped_plot_path),
        "table_plot_path": str(table_plot_path),
    }


def main() -> None:
    args = parse_args()
    outputs = generate_threshold_sweep_summary(
        summary_root=args.summary_root,
        threshold_roots=args.threshold_root,
        models=args.models,
    )
    print("Fixed-threshold summary complete")
    for label, path in outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
