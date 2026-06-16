# Date Created: 2026-06-16
# Date Last Modified: 2026-06-16

"""Evaluate synthetic-only model performance across synthetic review counts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

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
from src.utils.paths import PROJECT_ROOT

DEFAULT_REVIEW_COUNTS = [99, 201, 300, 501, 1002, 2001, 2352]


def select_complete_perfume_blocks(synthetic_df: pd.DataFrame, review_count: int) -> pd.DataFrame:
    if review_count <= 0:
        raise ValueError("Synthetic-only review counts must be positive.")
    if review_count % 3 != 0:
        raise ValueError(f"{review_count} is not divisible by 3; expected whole 3-review perfume blocks.")
    if review_count > len(synthetic_df):
        raise ValueError(f"{review_count} exceeds available synthetic rows ({len(synthetic_df)}).")

    perfume_count = review_count // 3
    ordered_perfumes = synthetic_df["perfume_id"].astype(str).drop_duplicates().tolist()
    selected_perfumes = set(ordered_perfumes[:perfume_count])
    subset = synthetic_df[synthetic_df["perfume_id"].astype(str).isin(selected_perfumes)].copy()

    if len(subset) != review_count:
        raise ValueError(f"Expected {review_count} rows, found {len(subset)}.")
    per_perfume_counts = subset.groupby("perfume_id").size()
    if per_perfume_counts.min() != 3 or per_perfume_counts.max() != 3:
        raise ValueError("Selected subset does not have exactly 3 synthetic reviews per perfume.")
    return subset


def write_markdown_report(path: Path, summary: pd.DataFrame) -> None:
    metric_cols = [
        "model_family",
        "synthetic_reviews",
        "synthetic_perfumes",
        "macro_f1",
        "macro_roc_auc",
        "macro_pr_auc",
        "delta_f1_vs_previous_count",
        "delta_roc_auc_vs_previous_count",
        "delta_pr_auc_vs_previous_count",
    ]
    lines = [
        "# Synthetic-Only Review Count Sweep",
        "",
        "Synthetic-only grouped cross-validation. Each count is a complete 3-review perfume block; folds are grouped by perfume.",
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
    parser = argparse.ArgumentParser(description="Run synthetic-only review-count sweep.")
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
        default=PROJECT_ROOT / "outputs" / "synthetic_only_review_count_sweep_gemini_browser",
    )
    parser.add_argument("--review-counts", type=int, nargs="+", default=DEFAULT_REVIEW_COUNTS)
    parser.add_argument("--model-id", choices=sorted(MODEL_SPECS), nargs="+", default=sorted(MODEL_SPECS))
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "per_label").mkdir(exist_ok=True)

    target_cols = DEFAULT_TARGETS
    synthetic_df = coerce_targets(load_synthetic_frame(args.synthetic_labeled_features, target_cols), target_cols)
    feature_cols = select_feature_columns(synthetic_df)
    if not feature_cols:
        raise ValueError("No numeric synthetic feature columns found.")

    keep_cols = ["synthetic_review_id", "perfume_id", "Perfume", "review_index", "source_type", "row_level", *feature_cols, *target_cols]
    rows: list[dict[str, Any]] = []

    for model_id in args.model_id:
        model_spec = MODEL_SPECS[model_id]
        previous_f1: float | None = None
        previous_roc: float | None = None
        previous_pr: float | None = None

        for review_count in args.review_counts:
            subset = select_complete_perfume_blocks(synthetic_df, review_count)
            eval_df = subset[keep_cols].copy()
            predictions, per_label, metrics = evaluate_grouped_oof(
                eval_df,
                model_spec=model_spec,
                feature_cols=feature_cols,
                target_cols=target_cols,
                group_col="perfume_id",
                threshold=args.threshold,
                n_splits=args.n_splits,
                seed=args.seed,
            )
            per_label_path = args.out_root / "per_label" / f"{model_id}_synthetic_reviews_{review_count}_per_label.csv"
            oof_path = args.out_root / "per_label" / f"{model_id}_synthetic_reviews_{review_count}_oof.csv"
            per_label.to_csv(per_label_path, index=False)
            predictions.to_csv(oof_path, index=False)

            macro_f1 = float(metrics["macro_f1"])
            macro_roc = float(metrics["macro_roc_auc"])
            macro_pr = float(metrics["macro_pr_auc"])
            row = {
                "model_id": model_id,
                "model_family": model_spec.model_family,
                "synthetic_reviews": review_count,
                "synthetic_perfumes": review_count // 3,
                "train_rows": int(len(eval_df)),
                "eval_rows": int(len(eval_df)),
                "macro_f1": macro_f1,
                "macro_roc_auc": macro_roc,
                "macro_pr_auc": macro_pr,
                "micro_f1": float(metrics["micro_f1"]),
                "hamming_loss": float(metrics["hamming_loss"]),
                "label_cardinality": float(metrics["label_cardinality"]),
                "predicted_label_cardinality": float(metrics["predicted_label_cardinality"]),
                "delta_f1_vs_previous_count": 0.0 if previous_f1 is None else macro_f1 - previous_f1,
                "delta_roc_auc_vs_previous_count": 0.0 if previous_roc is None else macro_roc - previous_roc,
                "delta_pr_auc_vs_previous_count": 0.0 if previous_pr is None else macro_pr - previous_pr,
                "per_label_csv": stringify_path(per_label_path),
                "oof_csv": stringify_path(oof_path),
                "experiment": "synthetic_only_grouped_cv",
            }
            rows.append(row)
            previous_f1 = macro_f1
            previous_roc = macro_roc
            previous_pr = macro_pr
            print(
                f"{model_spec.model_family} synthetic_reviews={review_count}: "
                f"macro_f1={macro_f1:.4f} macro_roc_auc={macro_roc:.4f} macro_pr_auc={macro_pr:.4f}"
            )

    summary = pd.DataFrame(rows)
    summary_path = args.out_root / "synthetic_only_review_count_sweep_summary.csv"
    report_md_path = args.out_root / "synthetic_only_review_count_sweep_summary.md"
    report_json_path = args.out_root / "synthetic_only_review_count_sweep_report.json"
    summary.to_csv(summary_path, index=False)
    write_markdown_report(report_md_path, summary)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "synthetic_labeled_features": stringify_path(args.synthetic_labeled_features),
        },
        "review_counts": args.review_counts,
        "models": args.model_id,
        "target_columns": target_cols,
        "threshold": args.threshold,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "feature_count": len(feature_cols),
        "outputs": {
            "summary_csv": stringify_path(summary_path),
            "summary_md": stringify_path(report_md_path),
            "report_json": stringify_path(report_json_path),
            "per_label_dir": stringify_path(args.out_root / "per_label"),
        },
        "note": (
            "Synthetic-only learning curve. Each count is selected as complete 3-review perfume blocks. "
            "Cross-validation is grouped by perfume_id."
        ),
    }
    report_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote report: {report_md_path}")


if __name__ == "__main__":
    main()
