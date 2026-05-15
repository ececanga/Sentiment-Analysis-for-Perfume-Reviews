# Date Created: 2026-04-13 23:02:44+03:00
# Date Last Modified: 2026-05-15

"""Run the official XGBoost benchmark baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.model_benchmark_runner import BenchmarkModelSpec, run_classical_model_benchmark
from src.utils.paths import PROJECT_ROOT


def build_xgboost_model(seed: int, y_train: np.ndarray) -> Pipeline:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for the official XGBoost benchmark. "
            "Install it in the project environment before running this script."
        ) from exc

    pos = int(np.sum(y_train))
    neg = int(len(y_train) - pos)
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    min_child_weight=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=seed,
                    n_jobs=1,
                    tree_method="hist",
                    scale_pos_weight=scale_pos_weight,
                    verbosity=0,
                ),
            ),
        ]
    )


MODEL_SPEC = BenchmarkModelSpec(
    model_id="xgboost",
    model_family="XGBoost",
    build_model=build_xgboost_model,
    model_params={
        "imputer_strategy": "median",
        "classifier": {
            "type": "XGBClassifier",
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "min_child_weight": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": 1,
            "tree_method": "hist",
            "scale_pos_weight": "labelwise_neg_to_pos_ratio",
        },
    },
    package_names=("xgboost",),
)


def run_xgboost_benchmark(
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
    result = run_xgboost_benchmark(
        input_csv=args.input_csv,
        out_root=args.out_root,
        config_path=args.config,
    )
    metrics = result["macro_metrics"]
    print("XGBoost benchmark complete")
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
    tuning = result.get("threshold_tuning", {})
    if tuning.get("enabled", False):
        for strategy, strategy_report in tuning["strategies"].items():
            print(
                f"- Tuned {strategy} Macro-F1={strategy_report['tuned_macro_f1']:.4f} "
                f"(baseline {tuning['baseline_threshold']:.2f}={tuning['baseline_macro_f1']:.4f})"
            )


if __name__ == "__main__":
    main()
