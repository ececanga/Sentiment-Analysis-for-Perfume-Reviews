from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_TABLE = PROJECT_ROOT / "outputs" / "model_benchmark" / "tables" / "comparison_table.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_benchmark"

MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "bert",
    "bilstm_attention",
]

MODEL_LABELS = {
    "logistic_regression": "Logistic\nRegression",
    "random_forest": "Random\nForest",
    "xgboost": "XGBoost",
    "bert": "BERT",
    "bilstm_attention": "BiLSTM\n+ Attention",
}

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "bert": "BERT",
    "bilstm_attention": "BiLSTM + Attention",
}

MODEL_COLORS = {
    "logistic_regression": "#32527B",
    "random_forest": "#5A8F7B",
    "xgboost": "#D17B49",
    "bert": "#7C5C8E",
    "bilstm_attention": "#B24C63",
}

METRIC_SPECS = [
    ("macro_f1", "Macro-F1", "all_models_macro_f1_comparison.png"),
    ("macro_roc_auc", "Macro ROC-AUC", "all_models_macro_roc_auc_comparison.png"),
    ("macro_pr_auc", "Macro PR-AUC", "all_models_macro_pr_auc_comparison.png"),
]

GROUPED_METRIC_COLORS = {
    "macro_f1": "#2E5EAA",
    "macro_roc_auc": "#2F8C69",
    "macro_pr_auc": "#C97A2B",
}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final thesis comparison figures from the frozen benchmark comparison table."
    )
    parser.add_argument(
        "--source-table",
        type=Path,
        default=DEFAULT_SOURCE_TABLE,
        help="Path to outputs/model_benchmark/tables/comparison_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for outputs/model_benchmark",
    )
    return parser.parse_args()


def load_final_model_comparison(source_table: Path) -> pd.DataFrame:
    df = pd.read_csv(source_table)
    required_columns = {"model_id", "macro_f1", "macro_roc_auc", "macro_pr_auc", "row_complete"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in {source_table}: {', '.join(missing_columns)}")

    df = df.loc[df["model_id"].isin(MODEL_ORDER)].copy()
    if df["model_id"].duplicated().any():
        duplicate_ids = ", ".join(sorted(df.loc[df["model_id"].duplicated(), "model_id"].unique()))
        raise ValueError(f"Duplicate model rows found in {source_table}: {duplicate_ids}")

    missing_model_ids = [model_id for model_id in MODEL_ORDER if model_id not in set(df["model_id"])]
    if missing_model_ids:
        raise ValueError(
            "The comparison table is missing final benchmark rows for: "
            + ", ".join(missing_model_ids)
        )

    for metric in ["macro_f1", "macro_roc_auc", "macro_pr_auc"]:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        if df[metric].isna().any():
            missing_metric_ids = ", ".join(df.loc[df[metric].isna(), "model_id"].tolist())
            raise ValueError(f"Metric {metric} is missing for: {missing_metric_ids}")

    row_complete = df["row_complete"].astype(str).str.lower()
    incomplete_model_ids = df.loc[~row_complete.isin({"true", "1"}), "model_id"].tolist()
    if incomplete_model_ids:
        raise ValueError(
            "Final comparison figures require complete benchmark rows. Incomplete rows: "
            + ", ".join(incomplete_model_ids)
        )

    df["sort_key"] = df["model_id"].map({model_id: idx for idx, model_id in enumerate(MODEL_ORDER)})
    df["model_name"] = df["model_id"].map(MODEL_NAMES)
    df["plot_label"] = df["model_id"].map(MODEL_LABELS)
    df = df.sort_values("sort_key").reset_index(drop=True)
    return df


def _rounded_axis_ceiling(max_value: float) -> float:
    margin = 0.06 if max_value < 0.35 else 0.08
    ceiling = np.ceil((max_value + margin) / 0.05) * 0.05
    return float(min(1.0, max(0.20, ceiling)))


def _annotate_bars(ax: plt.Axes, bars, *, y_max: float, fontsize: int) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (y_max * 0.015),
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def save_single_metric_figure(df: pd.DataFrame, *, metric: str, metric_label: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(df))
    y_values = df[metric].to_numpy()
    colors = [MODEL_COLORS[model_id] for model_id in df["model_id"]]
    y_max = _rounded_axis_ceiling(float(y_values.max()))

    bars = ax.bar(x, y_values, color=colors, width=0.68, edgecolor="#2B2B2B", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(df["plot_label"])
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Benchmark Comparison Across All Models: {metric_label}")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    _annotate_bars(ax, bars, y_max=y_max, fontsize=9)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_grouped_metric_figure(df: pd.DataFrame, *, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.1))
    x = np.arange(len(df))
    width = 0.22
    y_max = _rounded_axis_ceiling(float(df[[metric for metric, _, _ in METRIC_SPECS]].to_numpy().max()))

    for offset, (metric, metric_label, _) in zip([-width, 0.0, width], METRIC_SPECS):
        bars = ax.bar(
            x + offset,
            df[metric].to_numpy(),
            width=width,
            label=metric_label,
            color=GROUPED_METRIC_COLORS[metric],
            edgecolor="#2B2B2B",
            linewidth=0.6,
        )
        _annotate_bars(ax, bars, y_max=y_max, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(df["plot_label"])
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Benchmark Comparison Across All Models: Macro Metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def export_final_table(df: pd.DataFrame, *, path: Path) -> None:
    export_df = df.loc[:, ["model_id", "model_name", "macro_f1", "macro_roc_auc", "macro_pr_auc"]].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    configure_plot_style()

    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"

    comparison_df = load_final_model_comparison(args.source_table)

    export_final_table(
        comparison_df,
        path=tables_dir / "final_model_comparison_for_thesis.csv",
    )

    for metric, metric_label, filename in METRIC_SPECS:
        save_single_metric_figure(
            comparison_df,
            metric=metric,
            metric_label=metric_label,
            path=figures_dir / filename,
        )

    save_grouped_metric_figure(
        comparison_df,
        path=figures_dir / "all_models_grouped_metric_comparison.png",
    )


if __name__ == "__main__":
    main()
