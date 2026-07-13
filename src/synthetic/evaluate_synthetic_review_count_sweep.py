# Date Created: 2026-06-16
# Date Last Modified: 2026-07-13

"""Evaluate held-out YouTube performance across synthetic review counts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.synthetic.evaluate_review_level_augmentation import (
    DEFAULT_TARGETS,
    MODEL_SPECS,
    coerce_targets,
    evaluate_fixed_og_folds,
    load_og_frame,
    load_synthetic_frame,
    select_feature_columns,
    stringify_path,
)
from src.training.source_grouped_folds import load_and_validate_outer_fold_manifest, sha256_file
from src.utils.paths import PROJECT_ROOT

DEFAULT_REVIEW_COUNTS = [99, 201, 300, 501, 1002, 2001, 2352]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def metric_from_baseline(report: dict[str, Any], key: str) -> float:
    value = report.get(key)
    if value is None:
        return float("nan")
    return float(value)


def select_complete_perfume_blocks(synthetic_df: pd.DataFrame, review_count: int) -> pd.DataFrame:
    if review_count < 0:
        raise ValueError("Synthetic review counts must be non-negative.")
    if review_count % 3 != 0:
        raise ValueError(f"{review_count} is not divisible by 3; expected whole 3-review perfume blocks.")
    if review_count > len(synthetic_df):
        raise ValueError(f"{review_count} exceeds available synthetic rows ({len(synthetic_df)}).")

    perfume_count = review_count // 3
    ordered_perfumes = synthetic_df["perfume_id"].astype(str).drop_duplicates().tolist()
    selected_perfumes = set(ordered_perfumes[:perfume_count])
    subset = synthetic_df[synthetic_df["perfume_id"].astype(str).isin(selected_perfumes)].copy()

    if len(subset) != review_count:
        raise ValueError(
            f"Expected {review_count} rows from {perfume_count} perfumes, found {len(subset)}. "
            "The synthetic input may not contain exactly 3 reviews per selected perfume."
        )
    per_perfume_counts = subset.groupby("perfume_id").size()
    if per_perfume_counts.min() != 3 or per_perfume_counts.max() != 3:
        raise ValueError("Selected synthetic subset does not have exactly 3 reviews per perfume.")
    return subset


def baseline_row(model_id: str, og_metrics_dir: Path) -> dict[str, Any]:
    model_spec = MODEL_SPECS[model_id]
    report = load_json(og_metrics_dir / f"{model_id}.json")
    return {
        "model_id": model_id,
        "model_family": model_spec.model_family,
        "synthetic_reviews": 0,
        "synthetic_perfumes": 0,
        "train_rows": int(report.get("n_rows", 784)),
        "eval_rows": int(report.get("n_rows", 784)),
        "macro_f1": metric_from_baseline(report, "macro_f1"),
        "macro_roc_auc": metric_from_baseline(report, "macro_roc_auc"),
        "macro_pr_auc": metric_from_baseline(report, "macro_pr_auc"),
        "micro_f1": float(report.get("secondary_metrics", {}).get("micro_f1", np.nan)),
        "hamming_loss": float(report.get("secondary_metrics", {}).get("hamming_loss", np.nan)),
        "experiment": "youtube_only_reference",
    }


def write_markdown_report(path: Path, summary: pd.DataFrame) -> None:
    metric_cols = [
        "model_family",
        "synthetic_reviews",
        "synthetic_perfumes",
        "macro_f1",
        "macro_roc_auc",
        "macro_pr_auc",
        "delta_f1_vs_youtube",
        "delta_roc_auc_vs_youtube",
        "delta_pr_auc_vs_youtube",
    ]
    lines = [
        "# Synthetic Review Count Sweep",
        "",
        "Held-out YouTube evaluation with fixed perfume-level folds. Synthetic rows are added as complete 3-review perfume blocks.",
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
    parser = argparse.ArgumentParser(description="Run synthetic review-count augmentation sweep.")
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
        "--og-features",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gemini_browser_og784" / "analysis_out" / "gemini_browser_og784_feature_labels.csv",
    )
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gemini_browser_og784" / "model_benchmark" / "tables" / "fold_assignments.csv",
    )
    parser.add_argument(
        "--og-metrics-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gemini_browser_og784" / "model_benchmark" / "metrics",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "synthetic_review_count_sweep_gemini_browser",
    )
    parser.add_argument("--review-counts", type=int, nargs="+", default=DEFAULT_REVIEW_COUNTS)
    parser.add_argument("--model-id", choices=sorted(MODEL_SPECS), nargs="+", default=sorted(MODEL_SPECS))
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-folds",
        type=int,
        default=5,
        help="Required number of source-grouped outer folds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "per_label").mkdir(exist_ok=True)
    (args.out_root / "predictions").mkdir(exist_ok=True)

    target_cols = DEFAULT_TARGETS
    synthetic_df = coerce_targets(load_synthetic_frame(args.synthetic_labeled_features, target_cols), target_cols)
    og_df = coerce_targets(load_og_frame(args.og_features, target_cols), target_cols)
    fold_df = load_and_validate_outer_fold_manifest(
        args.fold_assignments,
        expected_ids=og_df["perfume_id"].astype(str).tolist(),
        n_splits=args.expected_folds,
    )
    feature_cols = select_feature_columns(og_df, synthetic_df)
    if not feature_cols:
        raise ValueError("No common numeric feature columns found between OG and synthetic frames.")

    keep_cols = ["synthetic_review_id", "perfume_id", "Perfume", "review_index", "source_type", "row_level", *feature_cols, *target_cols]
    og_eval_df = og_df[keep_cols].copy()

    rows: list[dict[str, Any]] = []
    for model_id in args.model_id:
        model_spec = MODEL_SPECS[model_id]
        base = baseline_row(model_id, args.og_metrics_dir)
        rows.append(base)
        baseline_f1 = float(base["macro_f1"])
        baseline_roc = float(base["macro_roc_auc"])
        baseline_pr = float(base["macro_pr_auc"])

        previous_f1 = baseline_f1
        previous_roc = baseline_roc
        previous_pr = baseline_pr

        for review_count in args.review_counts:
            subset = select_complete_perfume_blocks(synthetic_df, review_count)
            combined = pd.concat([og_eval_df, subset[keep_cols]], ignore_index=True)
            predictions, per_label, metrics = evaluate_fixed_og_folds(
                combined,
                og_eval_df,
                model_spec=model_spec,
                fold_df=fold_df,
                feature_cols=feature_cols,
                target_cols=target_cols,
                threshold=args.threshold,
                seed=args.seed,
            )
            per_label_path = args.out_root / "per_label" / f"{model_id}_synthetic_reviews_{review_count}_per_label.csv"
            predictions_path = (
                args.out_root / "predictions" / f"{model_id}_synthetic_reviews_{review_count}_oof.csv"
            )
            per_label.to_csv(per_label_path, index=False)
            predictions.to_csv(predictions_path, index=False)

            row = {
                "model_id": model_id,
                "model_family": model_spec.model_family,
                "synthetic_reviews": review_count,
                "synthetic_perfumes": review_count // 3,
                "train_rows": int(len(combined)),
                "eval_rows": int(len(og_eval_df)),
                "macro_f1": float(metrics["macro_f1"]),
                "macro_roc_auc": float(metrics["macro_roc_auc"]),
                "macro_pr_auc": float(metrics["macro_pr_auc"]),
                "micro_f1": float(metrics["micro_f1"]),
                "hamming_loss": float(metrics["hamming_loss"]),
                "delta_f1_vs_youtube": float(metrics["macro_f1"] - baseline_f1),
                "delta_roc_auc_vs_youtube": float(metrics["macro_roc_auc"] - baseline_roc),
                "delta_pr_auc_vs_youtube": float(metrics["macro_pr_auc"] - baseline_pr),
                "delta_f1_vs_previous_count": float(metrics["macro_f1"] - previous_f1),
                "delta_roc_auc_vs_previous_count": float(metrics["macro_roc_auc"] - previous_roc),
                "delta_pr_auc_vs_previous_count": float(metrics["macro_pr_auc"] - previous_pr),
                "per_label_csv": stringify_path(per_label_path),
                "predictions_csv": stringify_path(predictions_path),
                "experiment": "youtube_eval_with_synthetic_count",
            }
            rows.append(row)
            previous_f1 = row["macro_f1"]
            previous_roc = row["macro_roc_auc"]
            previous_pr = row["macro_pr_auc"]
            print(
                f"{model_spec.model_family} synthetic_reviews={review_count}: "
                f"macro_f1={row['macro_f1']:.4f} "
                f"macro_roc_auc={row['macro_roc_auc']:.4f} "
                f"macro_pr_auc={row['macro_pr_auc']:.4f}"
            )

    summary = pd.DataFrame(rows)
    for metric in ["f1", "roc_auc", "pr_auc"]:
        column = f"delta_{metric}_vs_youtube"
        if column not in summary:
            summary[column] = np.nan
    baseline_mask = summary["synthetic_reviews"].eq(0)
    summary.loc[baseline_mask, ["delta_f1_vs_youtube", "delta_roc_auc_vs_youtube", "delta_pr_auc_vs_youtube"]] = 0.0
    summary.loc[
        baseline_mask,
        ["delta_f1_vs_previous_count", "delta_roc_auc_vs_previous_count", "delta_pr_auc_vs_previous_count"],
    ] = 0.0

    summary_path = args.out_root / "synthetic_review_count_sweep_summary.csv"
    report_md_path = args.out_root / "synthetic_review_count_sweep_summary.md"
    report_json_path = args.out_root / "synthetic_review_count_sweep_report.json"
    summary.to_csv(summary_path, index=False)
    write_markdown_report(report_md_path, summary)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "synthetic_labeled_features": stringify_path(args.synthetic_labeled_features),
            "og_features": stringify_path(args.og_features),
            "fold_assignments": stringify_path(args.fold_assignments),
            "og_metrics_dir": stringify_path(args.og_metrics_dir),
        },
        "review_counts": args.review_counts,
        "models": args.model_id,
        "target_columns": target_cols,
        "threshold": args.threshold,
        "seed": args.seed,
        "split_provenance": {
            "design": "source_component_groupkfold",
            "manifest_sha256": sha256_file(args.fold_assignments),
            "expected_folds": int(args.expected_folds),
            "source_group_count": int(fold_df["source_group_id"].nunique()),
            "fold_sizes": {
                str(int(fold_id)): int(count)
                for fold_id, count in fold_df.groupby("fold_id").size().items()
            },
        },
        "feature_count": len(feature_cols),
        "outputs": {
            "summary_csv": stringify_path(summary_path),
            "summary_md": stringify_path(report_md_path),
            "report_json": stringify_path(report_json_path),
            "per_label_dir": stringify_path(args.out_root / "per_label"),
            "predictions_dir": stringify_path(args.out_root / "predictions"),
        },
        "note": (
            "Each synthetic review count is selected as complete 3-review perfume blocks. "
            "Metrics are evaluated only on held-out YouTube perfume-level rows using fixed "
            "source-component-grouped folds. Synthetic rows belonging to held-out perfume IDs are excluded from training."
        ),
    }
    report_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote report: {report_md_path}")


if __name__ == "__main__":
    main()
