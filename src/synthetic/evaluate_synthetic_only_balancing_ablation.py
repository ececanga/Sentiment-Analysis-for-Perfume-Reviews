# Date Created: 2026-06-22
# Date Last Modified: 2026-06-22

"""Compare unbalanced vs balanced synthetic-only full-corpus models."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.synthetic.evaluate_review_level_augmentation import (
    DEFAULT_TARGETS,
    MODEL_SPECS,
    coerce_targets,
    evaluate_grouped_oof,
    load_synthetic_frame,
    select_feature_columns,
    stringify_path,
)
from src.training.model_benchmark_runner import BenchmarkModelSpec
from src.utils.paths import PROJECT_ROOT


DEFAULT_FEATURES = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_review_level_tests"
    / "emotion_rich_v2_6reviews"
    / "supervised_eval_gemini_browser"
    / "review_level_features_labeled_g25pro.csv"
)
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "synthetic_only_balancing_ablation_gemini_browser_6reviews_threshold05"
DEFAULT_PERMUTATION_SEEDS = [101, 202, 303, 404, 505]


def build_unbalanced_logistic(seed: int, _y_train: np.ndarray) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=None,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )


def build_unbalanced_xgboost(seed: int, _y_train: np.ndarray) -> Pipeline:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required for this ablation.") from exc

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
                    scale_pos_weight=1.0,
                    verbosity=0,
                ),
            ),
        ]
    )


def model_variants() -> list[dict[str, Any]]:
    return [
        {
            "model_id": "logistic_regression_unbalanced",
            "model_family": "Logistic Regression",
            "balance_setting": "unbalanced",
            "balance_detail": "class_weight=None",
            "spec": BenchmarkModelSpec(
                model_id="logistic_regression_unbalanced",
                model_family="Logistic Regression",
                build_model=build_unbalanced_logistic,
                model_params={
                    "classifier": {
                        "type": "LogisticRegression",
                        "max_iter": 2000,
                        "class_weight": None,
                        "solver": "liblinear",
                    }
                },
                package_names=(),
            ),
        },
        {
            "model_id": "logistic_regression_balanced",
            "model_family": "Logistic Regression",
            "balance_setting": "balanced",
            "balance_detail": 'class_weight="balanced"',
            "spec": MODEL_SPECS["logistic_regression"],
        },
        {
            "model_id": "xgboost_unbalanced",
            "model_family": "XGBoost",
            "balance_setting": "unbalanced",
            "balance_detail": "scale_pos_weight=1.0",
            "spec": BenchmarkModelSpec(
                model_id="xgboost_unbalanced",
                model_family="XGBoost",
                build_model=build_unbalanced_xgboost,
                model_params={
                    "classifier": {
                        "type": "XGBClassifier",
                        "scale_pos_weight": 1.0,
                    }
                },
                package_names=("xgboost",),
            ),
        },
        {
            "model_id": "xgboost_balanced",
            "model_family": "XGBoost",
            "balance_setting": "balanced",
            "balance_detail": "scale_pos_weight=labelwise_neg_to_pos_ratio",
            "spec": MODEL_SPECS["xgboost"],
        },
    ]


def validate_input(df: pd.DataFrame, target_cols: list[str]) -> dict[str, Any]:
    per_perfume_counts = df.groupby(df["perfume_id"].astype(str)).size()
    review_indices = sorted(int(value) for value in df["review_index"].unique().tolist())
    label_missing = int(df[target_cols].isna().sum().sum())
    validation = {
        "rows": int(len(df)),
        "unique_perfumes": int(df["perfume_id"].nunique()),
        "review_indices": review_indices,
        "per_perfume_counts": sorted(int(value) for value in per_perfume_counts.unique().tolist()),
        "unique_synthetic_review_ids": int(df["synthetic_review_id"].nunique()),
        "missing_label_values": label_missing,
    }
    if validation["rows"] != 4704:
        raise ValueError(f"Expected 4704 rows, found {validation['rows']}.")
    if validation["unique_perfumes"] != 784:
        raise ValueError(f"Expected 784 perfumes, found {validation['unique_perfumes']}.")
    if validation["review_indices"] != [1, 2, 3, 4, 5, 6]:
        raise ValueError(f"Expected review indices 1..6, found {validation['review_indices']}.")
    if validation["per_perfume_counts"] != [6]:
        raise ValueError(f"Expected exactly 6 rows per perfume, found {validation['per_perfume_counts']}.")
    if validation["unique_synthetic_review_ids"] != validation["rows"]:
        raise ValueError("Synthetic review IDs are not unique.")
    if validation["missing_label_values"] != 0:
        raise ValueError(f"Found {validation['missing_label_values']} missing label values.")
    return validation


def label_distribution(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    rows = []
    for target in target_cols:
        positives = int(df[target].sum())
        negatives = int(len(df) - positives)
        rows.append(
            {
                "label": target,
                "rows": int(len(df)),
                "positive_count": positives,
                "negative_count": negatives,
                "positive_rate": float(positives / len(df)),
                "neg_to_pos_ratio": float(negatives / positives) if positives > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def reorder_by_perfume_seed(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    ordered_perfumes = df["perfume_id"].astype(str).drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    shuffled_perfumes = [str(perfume_id) for perfume_id in rng.permutation(ordered_perfumes)]
    order_lookup = {perfume_id: index for index, perfume_id in enumerate(shuffled_perfumes)}
    out = df.copy()
    out["_permutation_position"] = out["perfume_id"].astype(str).map(order_lookup).astype(int)
    return (
        out.sort_values(["_permutation_position", "review_index", "synthetic_review_id"])
        .drop(columns=["_permutation_position"])
        .reset_index(drop=True)
    )


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "model_id",
        "model_family",
        "balance_setting",
        "balance_detail",
        "threshold",
        "synthetic_reviews",
        "synthetic_perfumes",
        "feature_count",
    ]
    metric_cols = [
        "macro_f1",
        "macro_roc_auc",
        "macro_pr_auc",
        "micro_f1",
        "hamming_loss",
        "label_cardinality",
        "predicted_label_cardinality",
    ]
    aggregations: dict[str, Any] = {"runs": ("permutation_seed", "nunique")}
    for metric in metric_cols:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
        aggregations[f"{metric}_min"] = (metric, "min")
        aggregations[f"{metric}_max"] = (metric, "max")
    summary = runs.groupby(group_cols, as_index=False).agg(**aggregations)
    std_cols = [column for column in summary.columns if column.endswith("_std")]
    summary[std_cols] = summary[std_cols].fillna(0.0)
    return summary.sort_values(["model_family", "balance_setting"]).reset_index(drop=True)


def aggregate_per_label(per_label: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "model_id",
        "model_family",
        "balance_setting",
        "balance_detail",
        "label",
        "support_pos",
        "support_neg",
        "positive_rate",
        "neg_to_pos_ratio",
    ]
    metric_cols = ["f1", "roc_auc", "pr_auc", "predicted_positive_rate", "mean_probability"]
    aggregations: dict[str, Any] = {"runs": ("permutation_seed", "nunique")}
    for metric in metric_cols:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
        aggregations[f"{metric}_min"] = (metric, "min")
        aggregations[f"{metric}_max"] = (metric, "max")
    summary = per_label.groupby(group_cols, as_index=False).agg(**aggregations)
    std_cols = [column for column in summary.columns if column.endswith("_std")]
    summary[std_cols] = summary[std_cols].fillna(0.0)
    return summary.sort_values(["model_family", "label", "balance_setting"]).reset_index(drop=True)


def add_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    metric_cols = [
        "macro_f1_mean",
        "macro_roc_auc_mean",
        "macro_pr_auc_mean",
        "micro_f1_mean",
        "hamming_loss_mean",
        "predicted_label_cardinality_mean",
    ]
    for metric in metric_cols:
        out[f"delta_{metric}_vs_unbalanced"] = 0.0

    for model_family in out["model_family"].unique():
        family_mask = out["model_family"].eq(model_family)
        baseline = out[family_mask & out["balance_setting"].eq("unbalanced")]
        if len(baseline) != 1:
            raise ValueError(f"Expected one unbalanced baseline row for {model_family}.")
        base_row = baseline.iloc[0]
        for idx in out[family_mask].index:
            for metric in metric_cols:
                out.loc[idx, f"delta_{metric}_vs_unbalanced"] = float(out.loc[idx, metric] - base_row[metric])
    return out


def add_per_label_deltas(per_label: pd.DataFrame) -> pd.DataFrame:
    out = per_label.copy()
    for metric in ["f1_mean", "roc_auc_mean", "pr_auc_mean", "predicted_positive_rate_mean", "mean_probability_mean"]:
        out[f"delta_{metric}_vs_unbalanced"] = 0.0

    for model_family in out["model_family"].unique():
        for label in out["label"].unique():
            baseline = out[
                out["model_family"].eq(model_family)
                & out["label"].eq(label)
                & out["balance_setting"].eq("unbalanced")
            ]
            if len(baseline) != 1:
                raise ValueError(f"Expected one unbalanced baseline for {model_family}/{label}.")
            base_row = baseline.iloc[0]
            mask = out["model_family"].eq(model_family) & out["label"].eq(label)
            for metric in ["f1_mean", "roc_auc_mean", "pr_auc_mean", "predicted_positive_rate_mean", "mean_probability_mean"]:
                out.loc[mask, f"delta_{metric}_vs_unbalanced"] = out.loc[mask, metric].astype(float) - float(base_row[metric])
    return out


def write_markdown_report(
    path: Path,
    *,
    summary: pd.DataFrame,
    per_label: pd.DataFrame,
    labels: pd.DataFrame,
    validation: dict[str, Any],
    threshold: float,
) -> None:
    summary_cols = [
        "model_family",
        "balance_setting",
        "balance_detail",
        "runs",
        "macro_f1_mean",
        "macro_f1_std",
        "delta_macro_f1_mean_vs_unbalanced",
        "macro_roc_auc_mean",
        "macro_roc_auc_std",
        "delta_macro_roc_auc_mean_vs_unbalanced",
        "macro_pr_auc_mean",
        "macro_pr_auc_std",
        "delta_macro_pr_auc_mean_vs_unbalanced",
        "predicted_label_cardinality_mean",
    ]
    label_cols = [
        "model_family",
        "balance_setting",
        "label",
        "support_pos",
        "positive_rate",
        "f1_mean",
        "f1_std",
        "delta_f1_mean_vs_unbalanced",
        "pr_auc_mean",
        "pr_auc_std",
        "delta_pr_auc_mean_vs_unbalanced",
        "predicted_positive_rate_mean",
    ]
    distribution_cols = ["label", "positive_count", "negative_count", "positive_rate", "neg_to_pos_ratio"]

    lines = [
        "# Synthetic-Only 784x6 Balancing Ablation",
        "",
        "This compares deliberately unbalanced models against the balanced settings used by the current LR/XGBoost sweeps.",
        "",
        f"- Rows: {validation['rows']}",
        f"- Unique perfumes: {validation['unique_perfumes']}",
        f"- Review indices: {', '.join(str(value) for value in validation['review_indices'])}",
        f"- Rows per perfume: {', '.join(str(value) for value in validation['per_perfume_counts'])}",
        f"- Threshold: {threshold:g}",
        "- Split: GroupKFold with 5 folds, grouped by perfume_id",
        "- Repeated over five random perfume-order seeds so the full-corpus fold assignment is not tied to one row order",
        "",
        "## Metric Summary",
        "",
        "| " + " | ".join(summary_cols) + " |",
        "| " + " | ".join("---" for _ in summary_cols) + " |",
    ]
    for _, row in summary[summary_cols].iterrows():
        values = []
        for col in summary_cols:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Per-Label Metrics", "", "| " + " | ".join(label_cols) + " |", "| " + " | ".join("---" for _ in label_cols) + " |"])
    for _, row in per_label[label_cols].sort_values(["model_family", "label", "balance_setting"]).iterrows():
        values = []
        for col in label_cols:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Label Distribution", "", "| " + " | ".join(distribution_cols) + " |", "| " + " | ".join("---" for _ in distribution_cols) + " |"])
    for _, row in labels[distribution_cols].iterrows():
        values = []
        for col in distribution_cols:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The earlier threshold-0.5 synthetic-only sweep already used balanced LR and balanced XGBoost.",
            "- This ablation isolates whether that balancing helps compared with deliberately unbalanced versions.",
            "- Balancing should be judged by macro F1 and rare-label F1/PR-AUC, not only by aggregate ROC-AUC.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic-only balanced vs unbalanced full-corpus ablation.")
    parser.add_argument("--synthetic-labeled-features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--permutation-seeds", type=int, nargs="+", default=DEFAULT_PERMUTATION_SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    target_cols = DEFAULT_TARGETS
    synthetic_df = coerce_targets(load_synthetic_frame(args.synthetic_labeled_features, target_cols), target_cols)
    validation = validate_input(synthetic_df, target_cols)
    feature_cols = select_feature_columns(synthetic_df)
    if not feature_cols:
        raise ValueError("No numeric synthetic feature columns found.")

    keep_cols = [
        "synthetic_review_id",
        "perfume_id",
        "Perfume",
        "review_index",
        "source_type",
        "row_level",
        *feature_cols,
        *target_cols,
    ]
    base_eval_df = synthetic_df[keep_cols].copy()
    labels = label_distribution(base_eval_df, target_cols)

    run_rows: list[dict[str, Any]] = []
    per_label_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for permutation_seed in args.permutation_seeds:
        eval_df = reorder_by_perfume_seed(base_eval_df, permutation_seed)
        for variant in model_variants():
            predictions, per_label, metrics = evaluate_grouped_oof(
                eval_df,
                model_spec=variant["spec"],
                feature_cols=feature_cols,
                target_cols=target_cols,
                group_col="perfume_id",
                threshold=args.threshold,
                n_splits=args.n_splits,
                seed=args.model_seed,
            )
            predictions = predictions.copy()
            predictions.insert(0, "model_id", variant["model_id"])
            predictions.insert(1, "model_family", variant["model_family"])
            predictions.insert(2, "balance_setting", variant["balance_setting"])
            predictions.insert(3, "permutation_seed", permutation_seed)
            prediction_frames.append(predictions)

            per_label = per_label.copy()
            per_label.insert(0, "model_id", variant["model_id"])
            per_label.insert(1, "model_family", variant["model_family"])
            per_label.insert(2, "balance_setting", variant["balance_setting"])
            per_label.insert(3, "balance_detail", variant["balance_detail"])
            per_label.insert(4, "permutation_seed", permutation_seed)
            per_label = per_label.merge(labels[["label", "neg_to_pos_ratio"]], on="label", how="left")
            per_label_frames.append(per_label)

            run_rows.append(
                {
                    "model_id": variant["model_id"],
                    "model_family": variant["model_family"],
                    "balance_setting": variant["balance_setting"],
                    "balance_detail": variant["balance_detail"],
                    "threshold": args.threshold,
                    "permutation_seed": permutation_seed,
                    "synthetic_reviews": int(len(eval_df)),
                    "synthetic_perfumes": int(eval_df["perfume_id"].nunique()),
                    "feature_count": int(len(feature_cols)),
                    "macro_f1": float(metrics["macro_f1"]),
                    "macro_roc_auc": float(metrics["macro_roc_auc"]),
                    "macro_pr_auc": float(metrics["macro_pr_auc"]),
                    "micro_f1": float(metrics["micro_f1"]),
                    "hamming_loss": float(metrics["hamming_loss"]),
                    "label_cardinality": float(metrics["label_cardinality"]),
                    "predicted_label_cardinality": float(metrics["predicted_label_cardinality"]),
                }
            )
            print(
                f"seed={permutation_seed} {variant['model_id']}: macro_f1={metrics['macro_f1']:.4f} "
                f"macro_roc_auc={metrics['macro_roc_auc']:.4f} macro_pr_auc={metrics['macro_pr_auc']:.4f}"
            )

    runs = pd.DataFrame(run_rows)
    summary = add_deltas(aggregate_runs(runs))
    per_label_runs = pd.concat(per_label_frames, ignore_index=True)
    per_label = add_per_label_deltas(aggregate_per_label(per_label_runs))
    predictions = pd.concat(prediction_frames, ignore_index=True)

    runs_path = args.out_root / "synthetic_only_balancing_ablation_runs.csv"
    summary_path = args.out_root / "synthetic_only_balancing_ablation_summary.csv"
    per_label_runs_path = args.out_root / "synthetic_only_balancing_ablation_per_label_runs.csv"
    per_label_path = args.out_root / "synthetic_only_balancing_ablation_per_label.csv"
    predictions_path = args.out_root / "synthetic_only_balancing_ablation_oof_predictions.csv"
    labels_path = args.out_root / "synthetic_only_balancing_ablation_label_distribution.csv"
    report_md_path = args.out_root / "synthetic_only_balancing_ablation_summary.md"
    report_json_path = args.out_root / "synthetic_only_balancing_ablation_report.json"

    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    per_label_runs.to_csv(per_label_runs_path, index=False)
    per_label.to_csv(per_label_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    labels.to_csv(labels_path, index=False)
    write_markdown_report(
        report_md_path,
        summary=summary,
        per_label=per_label,
        labels=labels,
        validation=validation,
        threshold=args.threshold,
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"synthetic_labeled_features": stringify_path(args.synthetic_labeled_features)},
        "validation": validation,
        "threshold": args.threshold,
        "n_splits": args.n_splits,
        "model_seed": args.model_seed,
        "permutation_seeds": args.permutation_seeds,
        "target_columns": target_cols,
        "feature_count": len(feature_cols),
        "outputs": {
            "runs_csv": stringify_path(runs_path),
            "summary_csv": stringify_path(summary_path),
            "per_label_runs_csv": stringify_path(per_label_runs_path),
            "per_label_csv": stringify_path(per_label_path),
            "predictions_csv": stringify_path(predictions_path),
            "label_distribution_csv": stringify_path(labels_path),
            "summary_md": stringify_path(report_md_path),
            "report_json": stringify_path(report_json_path),
        },
        "note": (
            "The existing synthetic-only LR/XGBoost sweeps already use balanced model settings. "
            "This ablation compares those balanced settings with deliberately unbalanced versions on the full 784x6 corpus."
        ),
    }
    report_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote report: {report_md_path}")


if __name__ == "__main__":
    main()
