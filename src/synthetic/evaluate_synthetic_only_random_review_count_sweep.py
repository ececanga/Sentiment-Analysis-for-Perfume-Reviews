# Date Created: 2026-06-16
# Date Last Modified: 2026-06-16

"""Evaluate synthetic-only performance across random perfume-order subsets."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, roc_auc_score

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.synthetic.evaluate_review_level_augmentation import (
    DEFAULT_TARGETS,
    MODEL_SPECS,
    coerce_targets,
    evaluate_grouped_oof,
    finite_mean,
    load_synthetic_frame,
    metric_or_nan,
    select_feature_columns,
    stringify_path,
)
from src.utils.paths import PROJECT_ROOT

DEFAULT_REVIEW_COUNTS = [99, 201, 300, 501, 1002, 2001, 2352]
DEFAULT_PERMUTATION_SEEDS = [101, 202, 303, 404, 505]


def infer_or_validate_reviews_per_perfume(synthetic_df: pd.DataFrame, reviews_per_perfume: int | None) -> int:
    per_perfume_counts = synthetic_df.groupby(synthetic_df["perfume_id"].astype(str)).size()
    observed_counts = sorted(int(value) for value in per_perfume_counts.unique().tolist())
    if reviews_per_perfume is None:
        if len(observed_counts) != 1:
            raise ValueError(f"Could not infer one reviews-per-perfume value. Observed counts: {observed_counts}")
        reviews_per_perfume = observed_counts[0]
    if reviews_per_perfume <= 0:
        raise ValueError("reviews_per_perfume must be positive.")
    bad_counts = per_perfume_counts[per_perfume_counts.ne(reviews_per_perfume)]
    if not bad_counts.empty:
        preview = ", ".join(f"{perfume_id}={count}" for perfume_id, count in bad_counts.head(10).items())
        raise ValueError(f"Expected exactly {reviews_per_perfume} reviews per perfume. Bad counts: {preview}")
    return reviews_per_perfume


def shuffled_perfume_order(synthetic_df: pd.DataFrame, seed: int) -> list[str]:
    ordered_perfumes = synthetic_df["perfume_id"].astype(str).drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    return [str(perfume_id) for perfume_id in rng.permutation(ordered_perfumes)]


def select_random_complete_perfume_blocks(
    synthetic_df: pd.DataFrame,
    *,
    shuffled_perfumes: list[str],
    review_count: int,
    reviews_per_perfume: int,
) -> pd.DataFrame:
    if review_count <= 0:
        raise ValueError("Synthetic-only review counts must be positive.")
    if review_count % reviews_per_perfume != 0:
        raise ValueError(
            f"{review_count} is not divisible by {reviews_per_perfume}; "
            f"expected whole {reviews_per_perfume}-review perfume blocks."
        )
    if review_count > len(synthetic_df):
        raise ValueError(f"{review_count} exceeds available synthetic rows ({len(synthetic_df)}).")

    perfume_count = review_count // reviews_per_perfume
    selected_perfumes = set(shuffled_perfumes[:perfume_count])
    order_lookup = {perfume_id: index for index, perfume_id in enumerate(shuffled_perfumes)}
    subset = synthetic_df[synthetic_df["perfume_id"].astype(str).isin(selected_perfumes)].copy()
    subset["_permutation_position"] = subset["perfume_id"].astype(str).map(order_lookup).astype(int)
    subset = subset.sort_values(["_permutation_position", "review_index", "synthetic_review_id"]).drop(
        columns=["_permutation_position"]
    ).reset_index(drop=True)

    if len(subset) != review_count:
        raise ValueError(f"Expected {review_count} rows, found {len(subset)}.")
    per_perfume_counts = subset.groupby(subset["perfume_id"].astype(str)).size()
    if per_perfume_counts.min() != reviews_per_perfume or per_perfume_counts.max() != reviews_per_perfume:
        raise ValueError(f"Selected subset does not have exactly {reviews_per_perfume} synthetic reviews per perfume.")
    return subset


def score_predictions_at_threshold(
    eval_df: pd.DataFrame,
    predictions: pd.DataFrame,
    per_label_template: pd.DataFrame,
    *,
    target_cols: list[str],
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    training_status = per_label_template.set_index("label")["training_status"].to_dict()
    per_label_rows: list[dict[str, Any]] = []
    pred_columns: list[np.ndarray] = []

    for target in target_cols:
        y_true = eval_df[target].astype(int).to_numpy()
        y_prob = predictions[f"prob_{target}"].astype(float).to_numpy()
        y_pred = (y_prob >= threshold).astype(int)
        pred_columns.append(y_pred)
        per_label_rows.append(
            {
                "label": target,
                "support_pos": int(y_true.sum()),
                "support_neg": int(len(y_true) - y_true.sum()),
                "positive_rate": float(y_true.mean()),
                "predicted_positive_count": int(y_pred.sum()),
                "predicted_positive_rate": float(y_pred.mean()),
                "mean_probability": float(np.nanmean(y_prob)),
                "f1": metric_or_nan(f1_score, y_true, y_pred),
                "roc_auc": metric_or_nan(roc_auc_score, y_true, y_prob),
                "pr_auc": metric_or_nan(average_precision_score, y_true, y_prob),
                "training_status": training_status.get(target, ""),
            }
        )

    per_label = pd.DataFrame(per_label_rows)
    y_true_matrix = eval_df[target_cols].astype(int).to_numpy()
    y_pred_matrix = np.column_stack(pred_columns)
    metrics = {
        "macro_f1": finite_mean(per_label["f1"].astype(float).tolist()),
        "macro_roc_auc": finite_mean(per_label["roc_auc"].astype(float).tolist()),
        "macro_pr_auc": finite_mean(per_label["pr_auc"].astype(float).tolist()),
        "micro_f1": metric_or_nan(f1_score, y_true_matrix.ravel(), y_pred_matrix.ravel()),
        "hamming_loss": metric_or_nan(hamming_loss, y_true_matrix, y_pred_matrix),
        "label_cardinality": float(y_true_matrix.sum(axis=1).mean()),
        "predicted_label_cardinality": float(y_pred_matrix.sum(axis=1).mean()),
    }
    return per_label, metrics


def aggregate_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_id", "model_family", "threshold", "synthetic_reviews", "synthetic_perfumes"]
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

    summary = run_df.groupby(group_cols, as_index=False).agg(**aggregations)
    std_cols = [column for column in summary.columns if column.endswith("_std")]
    summary[std_cols] = summary[std_cols].fillna(0.0)
    summary = summary.sort_values(["model_family", "threshold", "synthetic_reviews"]).reset_index(drop=True)

    for metric in ["macro_f1", "macro_roc_auc", "macro_pr_auc"]:
        mean_col = f"{metric}_mean"
        delta_col = f"delta_{metric}_mean_vs_previous_count"
        summary[delta_col] = summary.groupby(["model_family", "threshold"])[mean_col].diff().fillna(0.0)
    return summary


def write_markdown_report(
    path: Path,
    summary: pd.DataFrame,
    *,
    seeds: list[int],
    reviews_per_perfume: int,
    thresholds: list[float],
) -> None:
    metric_cols = [
        "model_family",
        "threshold",
        "synthetic_reviews",
        "synthetic_perfumes",
        "runs",
        "macro_f1_mean",
        "macro_f1_std",
        "macro_roc_auc_mean",
        "macro_roc_auc_std",
        "macro_pr_auc_mean",
        "macro_pr_auc_std",
    ]
    figure_lines = [
        "- `figures/synthetic_only_random_review_count_sweep_metrics.svg`",
        "- `figures/synthetic_only_random_review_count_sweep_metrics.png`",
        "- `figures/synthetic_only_random_review_count_sweep_table.svg`",
        "- `figures/synthetic_only_random_review_count_sweep_table.png`",
    ]
    if len(thresholds) > 1:
        figure_lines = [
            "- `figures/synthetic_only_random_review_count_threshold_sweep_f1.svg`",
            "- `figures/synthetic_only_random_review_count_threshold_sweep_f1.png`",
            "- `figures/synthetic_only_random_review_count_threshold_sweep_table.svg`",
            "- `figures/synthetic_only_random_review_count_threshold_sweep_table.png`",
        ]

    lines = [
        "# Randomized Synthetic-Only Review Count Sweep",
        "",
        (
            "Synthetic-only grouped cross-validation with random perfume-order subsets. "
            f"Each count is a complete {reviews_per_perfume}-review perfume block."
        ),
        "",
        f"Permutation seeds: {', '.join(str(seed) for seed in seeds)}",
        f"Thresholds: {', '.join(f'{threshold:g}' for threshold in thresholds)}",
        "",
        "Figures:",
        "",
        *figure_lines,
        "",
        "| " + " | ".join(metric_cols) + " |",
        "| " + " | ".join("---" for _ in metric_cols) + " |",
    ]
    for _, row in summary.iterrows():
        values: list[str] = []
        for col in metric_cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run randomized synthetic-only review-count sweep.")
    parser.add_argument(
        "--synthetic-labeled-features",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "synthetic_review_level_tests"
        / "emotion_rich_v2_3reviews"
        / "supervised_eval_gemini_browser"
        / "review_level_features_labeled_g25pro.csv",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "synthetic_only_random_review_count_sweep_gemini_browser",
    )
    parser.add_argument("--review-counts", type=int, nargs="+", default=DEFAULT_REVIEW_COUNTS)
    parser.add_argument(
        "--reviews-per-perfume",
        type=int,
        default=None,
        help="Expected synthetic reviews per perfume. Defaults to the uniform count inferred from the input file.",
    )
    parser.add_argument("--permutation-seeds", type=int, nargs="+", default=DEFAULT_PERMUTATION_SEEDS)
    parser.add_argument("--model-id", choices=sorted(MODEL_SPECS), nargs="+", default=sorted(MODEL_SPECS))
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of thresholds to score from the same out-of-fold probabilities.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--model-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    target_cols = DEFAULT_TARGETS
    synthetic_df = coerce_targets(load_synthetic_frame(args.synthetic_labeled_features, target_cols), target_cols)
    reviews_per_perfume = infer_or_validate_reviews_per_perfume(synthetic_df, args.reviews_per_perfume)
    feature_cols = select_feature_columns(synthetic_df)
    if not feature_cols:
        raise ValueError("No numeric synthetic feature columns found.")
    threshold_values = args.thresholds if args.thresholds is not None else [args.threshold]
    threshold_values = [float(threshold) for threshold in threshold_values]
    invalid_thresholds = [threshold for threshold in threshold_values if threshold < 0.0 or threshold > 1.0]
    if invalid_thresholds:
        raise ValueError(f"Thresholds must be between 0 and 1. Invalid values: {invalid_thresholds}")

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
    run_rows: list[dict[str, Any]] = []
    per_label_frames: list[pd.DataFrame] = []

    for permutation_seed in args.permutation_seeds:
        perfume_order = shuffled_perfume_order(synthetic_df, permutation_seed)
        for review_count in args.review_counts:
            subset = select_random_complete_perfume_blocks(
                synthetic_df,
                shuffled_perfumes=perfume_order,
                review_count=review_count,
                reviews_per_perfume=reviews_per_perfume,
            )
            eval_df = subset[keep_cols].copy()
            for model_id in args.model_id:
                model_spec = MODEL_SPECS[model_id]
                predictions, per_label_template, _ = evaluate_grouped_oof(
                    eval_df,
                    model_spec=model_spec,
                    feature_cols=feature_cols,
                    target_cols=target_cols,
                    group_col="perfume_id",
                    threshold=threshold_values[0],
                    n_splits=args.n_splits,
                    seed=args.model_seed,
                )

                for threshold in threshold_values:
                    per_label, metrics = score_predictions_at_threshold(
                        eval_df,
                        predictions,
                        per_label_template,
                        target_cols=target_cols,
                        threshold=threshold,
                    )

                    per_label = per_label.copy()
                    per_label.insert(0, "model_id", model_id)
                    per_label.insert(1, "model_family", model_spec.model_family)
                    per_label.insert(2, "threshold", threshold)
                    per_label.insert(3, "permutation_seed", permutation_seed)
                    per_label.insert(4, "synthetic_reviews", review_count)
                    per_label.insert(5, "synthetic_perfumes", review_count // reviews_per_perfume)
                    per_label_frames.append(per_label)

                    row = {
                        "model_id": model_id,
                        "model_family": model_spec.model_family,
                        "threshold": threshold,
                        "permutation_seed": permutation_seed,
                        "synthetic_reviews": review_count,
                        "synthetic_perfumes": review_count // reviews_per_perfume,
                        "train_rows": int(len(eval_df)),
                        "eval_rows": int(len(eval_df)),
                        "macro_f1": float(metrics["macro_f1"]),
                        "macro_roc_auc": float(metrics["macro_roc_auc"]),
                        "macro_pr_auc": float(metrics["macro_pr_auc"]),
                        "micro_f1": float(metrics["micro_f1"]),
                        "hamming_loss": float(metrics["hamming_loss"]),
                        "label_cardinality": float(metrics["label_cardinality"]),
                        "predicted_label_cardinality": float(metrics["predicted_label_cardinality"]),
                        "experiment": "synthetic_only_random_perfume_order_grouped_cv",
                    }
                    run_rows.append(row)
                    print(
                        f"seed={permutation_seed} {model_spec.model_family} threshold={threshold:g} "
                        f"synthetic_reviews={review_count}: "
                        f"macro_f1={row['macro_f1']:.4f} "
                        f"macro_roc_auc={row['macro_roc_auc']:.4f} "
                        f"macro_pr_auc={row['macro_pr_auc']:.4f}"
                    )

    runs = pd.DataFrame(run_rows)
    summary = aggregate_runs(runs)
    per_label = pd.concat(per_label_frames, ignore_index=True)

    runs_path = args.out_root / "synthetic_only_random_review_count_sweep_runs.csv"
    summary_path = args.out_root / "synthetic_only_random_review_count_sweep_summary.csv"
    per_label_path = args.out_root / "synthetic_only_random_review_count_sweep_per_label.csv"
    report_md_path = args.out_root / "synthetic_only_random_review_count_sweep_summary.md"
    report_json_path = args.out_root / "synthetic_only_random_review_count_sweep_report.json"
    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    per_label.to_csv(per_label_path, index=False)
    write_markdown_report(
        report_md_path,
        summary,
        seeds=args.permutation_seeds,
        reviews_per_perfume=reviews_per_perfume,
        thresholds=threshold_values,
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "synthetic_labeled_features": stringify_path(args.synthetic_labeled_features),
        },
        "review_counts": args.review_counts,
        "reviews_per_perfume": reviews_per_perfume,
        "permutation_seeds": args.permutation_seeds,
        "models": args.model_id,
        "target_columns": target_cols,
        "threshold": args.threshold,
        "thresholds": threshold_values,
        "n_splits": args.n_splits,
        "model_seed": args.model_seed,
        "feature_count": len(feature_cols),
        "outputs": {
            "runs_csv": stringify_path(runs_path),
            "summary_csv": stringify_path(summary_path),
            "per_label_csv": stringify_path(per_label_path),
            "summary_md": stringify_path(report_md_path),
            "report_json": stringify_path(report_json_path),
            "metrics_plot_svg": stringify_path(args.out_root / "figures" / "synthetic_only_random_review_count_sweep_metrics.svg"),
            "metrics_plot_png": stringify_path(args.out_root / "figures" / "synthetic_only_random_review_count_sweep_metrics.png"),
            "metrics_table_svg": stringify_path(args.out_root / "figures" / "synthetic_only_random_review_count_sweep_table.svg"),
            "metrics_table_png": stringify_path(args.out_root / "figures" / "synthetic_only_random_review_count_sweep_table.png"),
        },
        "note": (
            "Randomized synthetic-only learning-curve pilot. Each permutation shuffles perfumes, "
            f"then selects complete {reviews_per_perfume}-review perfume blocks for each count. Cross-validation remains "
            "grouped by perfume_id. The maximum review-count condition uses all perfumes, so random ordering "
            "does not change its subset composition."
        ),
    }
    report_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote runs: {runs_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote report: {report_md_path}")


if __name__ == "__main__":
    main()
