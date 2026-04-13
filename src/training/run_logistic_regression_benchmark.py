"""Run the official Logistic Regression benchmark baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.model_benchmark_runner import BenchmarkModelSpec, run_classical_model_benchmark
from src.utils.paths import PROJECT_ROOT

MODEL_SPEC = BenchmarkModelSpec(
    model_id="logistic_regression",
    model_family="Logistic Regression",
    build_model=lambda seed, _y_train: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    ),
    model_params={
        "imputer_strategy": "median",
        "scaler": "standard",
        "classifier": {
            "type": "LogisticRegression",
            "max_iter": 2000,
            "class_weight": "balanced",
            "solver": "liblinear",
        },
    },
    package_names=(),
)


def run_logistic_regression_benchmark(
    *,
    input_csv: Path,
    out_root: Path,
    config_path: Path,
) -> Dict[str, Any]:
    return run_classical_model_benchmark(
        input_csv=input_csv,
        out_root=out_root,
        config_path=config_path,
        model_spec=MODEL_SPEC,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "evaluation_config.yaml",
        help="Frozen benchmark config path.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis_out" / "gemini_unified_merged.csv",
        help="Benchmark dataset CSV path.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "model_benchmark",
        help="Benchmark output root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_logistic_regression_benchmark(
        input_csv=args.input_csv,
        out_root=args.out_root,
        config_path=args.config,
    )
    metrics = result["macro_metrics"]
    print("Logistic Regression benchmark complete")
    print(f"- Fold manifest: {result['fold_manifest_path']}")
    print(f"- OOF predictions: {result['predictions_path']}")
    print(f"- Metrics JSON: {result['metrics_path']}")
    print(f"- Per-label CSV: {result['per_label_path']}")
    print(f"- Comparison table: {result['comparison_table_path']}")
    print(f"- Classical summary table: {result['classical_summary_table_path']}")
    print(f"- Classical summary figure: {result['classical_summary_figure_path']}")
    print(
        f"- Main metrics: macro_f1={metrics['macro_f1']:.4f}, "
        f"macro_roc_auc={metrics['macro_roc_auc']:.4f}, macro_pr_auc={metrics['macro_pr_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
