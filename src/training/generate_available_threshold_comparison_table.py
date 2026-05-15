# Date Created: 2026-05-06 15:46:46+03:00
# Date Last Modified: 2026-05-15

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.training.generate_final_benchmark_figures import MODEL_NAMES, PROJECT_ROOT, configure_plot_style

MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "bert",
    "bilstm_attention",
]

DISPLAY_COLUMNS = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "BERT",
    "BiLSTM + Attention",
]

MODEL_TO_DISPLAY = {model_id: display for model_id, display in zip(MODEL_ORDER, DISPLAY_COLUMNS)}
THRESHOLD_ORDER = [0.3, 0.5, 0.7]
METRIC_KEYS = [
    ("macro_f1", "Macro-F1"),
    ("macro_roc_auc", "Macro ROC-AUC"),
    ("macro_pr_auc", "Macro PR-AUC"),
]

DEFAULT_BASELINE_ROOT = PROJECT_ROOT / "outputs" / "model_benchmark"
DEFAULT_THRESHOLD07_ROOT = PROJECT_ROOT / "outputs" / "model_benchmark_t07"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "model_benchmark_threshold_sweep"
FROZEN_05_COMPARISON_TABLE_PATH = "outputs/model_benchmark/tables/comparison_table.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a combined threshold comparison table from existing benchmark outputs only."
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_BASELINE_ROOT,
        help="Complete benchmark root used for 0.3 metrics and 0.5 derivation from saved predictions.",
    )
    parser.add_argument(
        "--threshold07-root",
        type=Path,
        default=DEFAULT_THRESHOLD07_ROOT,
        help="Benchmark root used for available 0.7 metrics.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for the combined CSV and PNG.",
    )
    parser.add_argument(
        "--frozen-05-ref",
        default=None,
        help="Optional git ref containing the original official 0.5 benchmark comparison table.",
    )
    return parser.parse_args()


def _load_comparison_table(root: Path) -> pd.DataFrame:
    path = root / "tables" / "comparison_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison table: {path}")
    df = pd.read_csv(path)
    if "model_id" not in df.columns:
        raise ValueError(f"Invalid comparison table: {path}")
    return df


def discover_frozen_05_ref() -> str:
    proc = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", FROZEN_05_COMPARISON_TABLE_PATH],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    refs = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not refs:
        raise RuntimeError(f"Could not discover frozen 0.5 source ref for {FROZEN_05_COMPARISON_TABLE_PATH}")
    return refs[0]


def load_git_table(*, git_ref: str, repo_relative_path: str) -> pd.DataFrame:
    proc = subprocess.run(
        ["git", "show", f"{git_ref}:{repo_relative_path}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    from io import StringIO

    return pd.read_csv(StringIO(proc.stdout))


def build_combined_table(
    *,
    baseline_root: Path,
    threshold07_root: Path,
    frozen_05_ref: str | None = None,
) -> tuple[pd.DataFrame, List[str]]:
    comparison_03 = _load_comparison_table(baseline_root)
    comparison_07 = _load_comparison_table(threshold07_root)
    frozen_ref = frozen_05_ref or discover_frozen_05_ref()
    comparison_05 = load_git_table(git_ref=frozen_ref, repo_relative_path=FROZEN_05_COMPARISON_TABLE_PATH)

    read_files = [
        str(baseline_root / "tables" / "comparison_table.csv"),
        str(threshold07_root / "tables" / "comparison_table.csv"),
        f"{frozen_ref}:{FROZEN_05_COMPARISON_TABLE_PATH}",
    ]

    values: Dict[tuple[float, str, str], str] = {}

    for model_id in MODEL_ORDER:
        row03 = comparison_03.loc[comparison_03["model_id"] == model_id]
        if row03.empty:
            raise ValueError(f"Missing 0.3 row for {model_id} in {baseline_root / 'tables' / 'comparison_table.csv'}")
        row03 = row03.iloc[0]
        values[(0.3, model_id, "macro_f1")] = f"{float(row03['macro_f1']):.6f}"
        values[(0.3, model_id, "macro_roc_auc")] = f"{float(row03['macro_roc_auc']):.6f}"
        values[(0.3, model_id, "macro_pr_auc")] = f"{float(row03['macro_pr_auc']):.6f}"

        row05 = comparison_05.loc[comparison_05["model_id"] == model_id]
        if row05.empty:
            raise ValueError(f"Missing 0.5 row for {model_id} in frozen source {frozen_ref}:{FROZEN_05_COMPARISON_TABLE_PATH}")
        row05 = row05.iloc[0]
        values[(0.5, model_id, "macro_f1")] = f"{float(row05['macro_f1']):.6f}"
        values[(0.5, model_id, "macro_roc_auc")] = f"{float(row05['macro_roc_auc']):.6f}"
        values[(0.5, model_id, "macro_pr_auc")] = f"{float(row05['macro_pr_auc']):.6f}"

        row07 = comparison_07.loc[comparison_07["model_id"] == model_id]
        if row07.empty or str(row07.iloc[0]["row_complete"]).lower() not in {"true", "1"}:
            values[(0.7, model_id, "macro_f1")] = ""
            values[(0.7, model_id, "macro_roc_auc")] = ""
            values[(0.7, model_id, "macro_pr_auc")] = ""
            continue
        row07 = row07.iloc[0]
        values[(0.7, model_id, "macro_f1")] = f"{float(row07['macro_f1']):.6f}"
        values[(0.7, model_id, "macro_roc_auc")] = f"{float(row07['macro_roc_auc']):.6f}"
        values[(0.7, model_id, "macro_pr_auc")] = f"{float(row07['macro_pr_auc']):.6f}"

    records = []
    for model_id in MODEL_ORDER:
        row = {"Model": MODEL_TO_DISPLAY[model_id]}
        for metric_key, metric_label in METRIC_KEYS:
            for threshold in THRESHOLD_ORDER:
                column_name = f"{format_threshold(threshold)} {metric_label}"
                row[column_name] = values[(threshold, model_id, metric_key)]
        records.append(row)

    deduped_read_files: List[str] = []
    for path in read_files:
        if path not in deduped_read_files:
            deduped_read_files.append(path)
    return pd.DataFrame(records), deduped_read_files


def format_threshold(threshold: float) -> str:
    return f"{threshold:.1f}"


def render_table_png(df: pd.DataFrame, *, path: Path) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(18.8, 4.8))
    ax.axis("off")

    table = ax.table(
        cellText=df.values.tolist(),
        colLabels=df.columns.tolist(),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1.0, 1.65)

    header_color = "#E9EEF6"
    border_color = "#B8C2CC"
    row_fill = "#FFFFFF"
    threshold_a = "#FFFFFF"
    threshold_b = "#F7F9FC"
    missing_fill = "#F2F2F2"

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(border_color)
        if row_idx == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold")
            continue
        if col_idx == 0:
            cell.set_facecolor(row_fill)
        else:
            group_idx = (col_idx - 1) % 3
            block_color = threshold_a if group_idx % 2 == 0 else threshold_b
            cell.set_facecolor(block_color)
        if col_idx > 0 and str(df.iloc[row_idx - 1, col_idx]) == "":
            cell.set_facecolor(missing_fill)

    for row_idx in range(1, len(df) + 1):
        model_cell = table[(row_idx, 0)]
        model_cell.set_text_props(weight="bold", ha="left")

    ax.set_title("Benchmark Comparison Across Thresholds: Available Results", fontsize=14, pad=18)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_available_threshold_comparison_table(
    *,
    baseline_root: Path,
    threshold07_root: Path,
    output_root: Path,
    frozen_05_ref: str | None = None,
) -> Dict[str, str]:
    df, read_files = build_combined_table(
        baseline_root=baseline_root,
        threshold07_root=threshold07_root,
        frozen_05_ref=frozen_05_ref,
    )

    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "final_model_threshold_comparison_metric_grouped_03_05_07.csv"
    png_path = figures_dir / "final_model_threshold_comparison_metric_grouped_03_05_07.png"
    df.to_csv(csv_path, index=False)
    render_table_png(df, path=png_path)

    return {
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "read_files": read_files,
    }


def main() -> None:
    args = parse_args()
    result = generate_available_threshold_comparison_table(
        baseline_root=args.baseline_root,
        threshold07_root=args.threshold07_root,
        output_root=args.output_root,
        frozen_05_ref=args.frozen_05_ref,
    )
    print("Combined threshold comparison table complete")
    print(f"- CSV: {result['csv_path']}")
    print(f"- PNG: {result['png_path']}")
    print("- Read files:")
    for path in result["read_files"]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
