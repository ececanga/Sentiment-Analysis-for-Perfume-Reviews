# Date Created: 2026-06-15
# Date Last Modified: 2026-06-15

"""Evaluate OG-only, synthetic-only, and OG+synthetic classical experiments."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, roc_auc_score

from src.labeling.strict_schema import EMOTIONS
from src.training.run_logistic_regression_benchmark import MODEL_SPEC as LOGISTIC_MODEL_SPEC
from src.training.run_xgboost_benchmark import MODEL_SPEC as XGBOOST_MODEL_SPEC
from src.utils.paths import PROJECT_ROOT

FEATURE_PREFIXES = ("NRC_", "VAD_", "Hartmann_", "Savani_", "GoEmotions_", "Cardiff_")
DEFAULT_TARGETS = ["joy", "sadness", "fear", "trust", "surprise", "anticipation", "disgust"]
MODEL_SPECS = {
    "logistic_regression": LOGISTIC_MODEL_SPEC,
    "xgboost": XGBOOST_MODEL_SPEC,
}


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def metric_or_nan(fn: Any, *args: Any) -> float:
    try:
        return float(fn(*args))
    except Exception:
        return float("nan")


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return float(np.mean(finite)) if finite else float("nan")


def positive_class_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    class_list = list(clf.classes_)
    positive_idx = class_list.index(1) if 1 in class_list else len(class_list) - 1
    return proba[:, positive_idx]


def select_feature_columns(*frames: pd.DataFrame) -> list[str]:
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    return [
        column
        for column in frames[0].columns
        if column in common
        and column.startswith(FEATURE_PREFIXES)
        and all(pd.api.types.is_numeric_dtype(frame[column]) for frame in frames)
    ]


def load_synthetic_frame(path: Path, target_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"synthetic_review_id", "perfume_id", "Perfume", "review_index", *target_cols}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Synthetic labeled feature file missing columns: {missing}")
    df = df.copy()
    df["source_type"] = "synthetic_review_v2"
    df["row_level"] = "review"
    return df


def load_og_frame(path: Path, target_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"perfume_id", "Perfume", *target_cols}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"OG feature file missing columns: {missing}")
    df = df.copy()
    df["synthetic_review_id"] = "og-perfume-" + df["perfume_id"].astype(str)
    df["review_index"] = 0
    df["source_type"] = "og_perfume_level"
    df["row_level"] = "perfume"
    return df


def coerce_targets(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for target in target_cols:
        out[target] = pd.to_numeric(out[target], errors="raise").astype(int)
    return out


def summarize_labels(df: pd.DataFrame, target_cols: list[str], source_name: str) -> pd.DataFrame:
    rows = []
    for target in target_cols:
        rows.append(
            {
                "source": source_name,
                "label": target,
                "rows": int(len(df)),
                "positive_count": int(df[target].sum()),
                "positive_rate": float(df[target].mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_fixed_og_folds(
    train_df: pd.DataFrame,
    og_df: pd.DataFrame,
    *,
    model_spec: Any,
    fold_df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    threshold: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fold_lookup = fold_df.set_index("perfume_id")["fold_id"].astype(int).to_dict()
    if set(og_df["perfume_id"].astype(str)) - set(fold_lookup):
        missing = sorted(set(og_df["perfume_id"].astype(str)) - set(fold_lookup))[:10]
        raise ValueError(f"Missing OG fold assignments for perfume IDs: {missing}")

    predictions = og_df[["synthetic_review_id", "perfume_id", "Perfume", "review_index", "source_type"]].copy()
    for target in target_cols:
        predictions[f"prob_{target}"] = np.nan
        predictions[f"pred_{target}"] = 0
    predictions["fold_id"] = predictions["perfume_id"].astype(str).map(fold_lookup).astype(int)

    fold_status: dict[str, list[str]] = {target: [] for target in target_cols}
    for fold_id in sorted(fold_df["fold_id"].astype(int).unique()):
        test_mask = og_df["perfume_id"].astype(str).map(fold_lookup).astype(int).eq(fold_id)
        test_ids = set(og_df.loc[test_mask, "perfume_id"].astype(str))
        train_mask = ~train_df["perfume_id"].astype(str).isin(test_ids)
        X_train = train_df.loc[train_mask, feature_cols]
        X_test = og_df.loc[test_mask, feature_cols]
        for target in target_cols:
            y_train = train_df.loc[train_mask, target].astype(int)
            if y_train.nunique() < 2:
                fallback = float(y_train.mean())
                predictions.loc[test_mask, f"prob_{target}"] = fallback
                predictions.loc[test_mask, f"pred_{target}"] = int(fallback >= threshold)
                fold_status[target].append(f"fold_{fold_id}:single_class_train_fallback")
                continue
            model = model_spec.build_model(seed, y_train.to_numpy())
            model.fit(X_train, y_train)
            probs = positive_class_probability(model, X_test)
            predictions.loc[test_mask, f"prob_{target}"] = probs
            predictions.loc[test_mask, f"pred_{target}"] = (probs >= threshold).astype(int)
            fold_status[target].append(f"fold_{fold_id}:trained")

    per_label_rows: list[dict[str, Any]] = []
    for target in target_cols:
        y_true = og_df[target].astype(int).to_numpy()
        y_prob = predictions[f"prob_{target}"].astype(float).to_numpy()
        y_pred = predictions[f"pred_{target}"].astype(int).to_numpy()
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
                "training_status": ";".join(fold_status[target]),
            }
        )
    per_label = pd.DataFrame(per_label_rows)
    y_true_matrix = og_df[target_cols].astype(int).to_numpy()
    y_pred_matrix = predictions[[f"pred_{target}" for target in target_cols]].astype(int).to_numpy()
    metrics = {
        "macro_f1": finite_mean(per_label["f1"].astype(float).tolist()),
        "macro_roc_auc": finite_mean(per_label["roc_auc"].astype(float).tolist()),
        "macro_pr_auc": finite_mean(per_label["pr_auc"].astype(float).tolist()),
        "micro_f1": metric_or_nan(f1_score, y_true_matrix.ravel(), y_pred_matrix.ravel()),
        "hamming_loss": metric_or_nan(hamming_loss, y_true_matrix, y_pred_matrix),
        "label_cardinality": float(y_true_matrix.sum(axis=1).mean()),
        "predicted_label_cardinality": float(y_pred_matrix.sum(axis=1).mean()),
    }
    return predictions, per_label, metrics


def evaluate_grouped_oof(
    df: pd.DataFrame,
    *,
    model_spec: Any,
    feature_cols: list[str],
    target_cols: list[str],
    group_col: str,
    threshold: float,
    n_splits: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from sklearn.model_selection import GroupKFold

    groups = df[group_col].astype(str)
    splitter = GroupKFold(n_splits=n_splits)

    predictions = df[["synthetic_review_id", "perfume_id", "Perfume", "review_index", "source_type"]].copy()
    for target in target_cols:
        predictions[f"prob_{target}"] = np.nan
        predictions[f"pred_{target}"] = 0
    predictions["fold_id"] = 0

    fold_status: dict[str, list[str]] = {target: [] for target in target_cols}
    X = df[feature_cols]
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, groups=groups), start=1):
        predictions.loc[test_idx, "fold_id"] = fold_id
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        for target in target_cols:
            y_train = df[target].iloc[train_idx].astype(int)
            if y_train.nunique() < 2:
                fallback = float(y_train.mean())
                predictions.loc[test_idx, f"prob_{target}"] = fallback
                predictions.loc[test_idx, f"pred_{target}"] = int(fallback >= threshold)
                fold_status[target].append(f"fold_{fold_id}:single_class_train_fallback")
                continue
            model = model_spec.build_model(seed, y_train.to_numpy())
            model.fit(X_train, y_train)
            probs = positive_class_probability(model, X_test)
            predictions.loc[test_idx, f"prob_{target}"] = probs
            predictions.loc[test_idx, f"pred_{target}"] = (probs >= threshold).astype(int)
            fold_status[target].append(f"fold_{fold_id}:trained")

    per_label_rows: list[dict[str, Any]] = []
    for target in target_cols:
        y_true = df[target].astype(int).to_numpy()
        y_prob = predictions[f"prob_{target}"].astype(float).to_numpy()
        y_pred = predictions[f"pred_{target}"].astype(int).to_numpy()
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
                "training_status": ";".join(fold_status[target]),
            }
        )
    per_label = pd.DataFrame(per_label_rows)
    y_true_matrix = df[target_cols].astype(int).to_numpy()
    y_pred_matrix = predictions[[f"pred_{target}" for target in target_cols]].astype(int).to_numpy()
    metrics = {
        "macro_f1": finite_mean(per_label["f1"].astype(float).tolist()),
        "macro_roc_auc": finite_mean(per_label["roc_auc"].astype(float).tolist()),
        "macro_pr_auc": finite_mean(per_label["pr_auc"].astype(float).tolist()),
        "micro_f1": metric_or_nan(f1_score, y_true_matrix.ravel(), y_pred_matrix.ravel()),
        "hamming_loss": metric_or_nan(hamming_loss, y_true_matrix, y_pred_matrix),
        "label_cardinality": float(y_true_matrix.sum(axis=1).mean()),
        "predicted_label_cardinality": float(y_pred_matrix.sum(axis=1).mean()),
    }
    return predictions, per_label, metrics


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_report(path: Path, report: dict[str, Any], comparison: pd.DataFrame, label_distribution: pd.DataFrame) -> None:
    lines = [
        "# Full Synthetic Review-Level Augmentation Comparison",
        "",
        f"- Model: {report['model_family']} (`{report['model_id']}`)",
        "",
        "## Metric Summary",
        "",
    ]
    metric_cols = ["experiment", "eval_rows", "train_rows", "macro_f1", "macro_roc_auc", "macro_pr_auc", "micro_f1", "hamming_loss"]
    lines.append("| " + " | ".join(metric_cols) + " |")
    lines.append("| " + " | ".join("---" for _ in metric_cols) + " |")
    for _, row in comparison.iterrows():
        values = []
        for col in metric_cols:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Label Distribution", ""])
    lines.append("| source | label | rows | positive_count | positive_rate |")
    lines.append("| --- | --- | --- | --- | --- |")
    for _, row in label_distribution.iterrows():
        lines.append(
            f"| {row['source']} | {row['label']} | {int(row['rows'])} | "
            f"{int(row['positive_count'])} | {float(row['positive_rate']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            f"- `og_only_reference` is the existing 784 perfume-level {report['model_family']} result.",
            "- `synthetic_only_review_level` is 2352 synthetic review rows, grouped by perfume.",
            "- `og_plus_synthetic_all_rows_oof` evaluates the mixed row set, so synthetic rows dominate the metric 3:1.",
            "- `og_plus_synthetic_eval_og_rows` is the thesis-relevant augmentation check: train with OG+synthetic rows, but evaluate only held-out OG perfume rows using the frozen fold assignment.",
            "- This still mixes row granularities: OG rows are perfume-level, synthetic rows are review-level.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate review-level synthetic augmentation against OG classical results.")
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
        "--synthetic-metrics",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "synthetic_review_level_tests"
        / "emotion_rich_v2_3reviews"
        / "supervised_eval_gemini_browser"
        / "logistic_regression_review_level_metrics.json",
    )
    parser.add_argument(
        "--og-features",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gemini_judge" / "judge_g25pro_v1" / "analysis_out" / "gemini_unified_merged.csv",
    )
    parser.add_argument(
        "--og-metrics",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gemini_judge" / "judge_g25pro_v1" / "model_benchmark" / "metrics" / "logistic_regression.json",
    )
    parser.add_argument("--model-id", choices=sorted(MODEL_SPECS), default="logistic_regression")
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gemini_judge" / "judge_g25pro_v1" / "model_benchmark" / "tables" / "fold_assignments.csv",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "review_level_synthetic_augmentation_gemini_browser",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    target_cols = DEFAULT_TARGETS
    model_spec = MODEL_SPECS[args.model_id]

    synthetic_df = coerce_targets(load_synthetic_frame(args.synthetic_labeled_features, target_cols), target_cols)
    og_df = coerce_targets(load_og_frame(args.og_features, target_cols), target_cols)
    feature_cols = select_feature_columns(og_df, synthetic_df)
    if not feature_cols:
        raise ValueError("No common numeric feature columns found between OG and synthetic frames.")

    keep_cols = ["synthetic_review_id", "perfume_id", "Perfume", "review_index", "source_type", "row_level", *feature_cols, *target_cols]
    combined = pd.concat([og_df[keep_cols], synthetic_df[keep_cols]], ignore_index=True)
    combined_path = args.out_root / "og_plus_synthetic_labeled_features.csv"
    combined.to_csv(combined_path, index=False)

    synthetic_predictions, synthetic_per_label, synthetic_metrics = evaluate_grouped_oof(
        synthetic_df[keep_cols],
        model_spec=model_spec,
        feature_cols=feature_cols,
        target_cols=target_cols,
        group_col="perfume_id",
        threshold=args.threshold,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    synthetic_predictions.to_csv(args.out_root / "synthetic_only_review_level_oof.csv", index=False)
    synthetic_per_label.to_csv(args.out_root / "synthetic_only_review_level_per_label.csv", index=False)

    combined_predictions, combined_per_label, combined_metrics = evaluate_grouped_oof(
        combined,
        model_spec=model_spec,
        feature_cols=feature_cols,
        target_cols=target_cols,
        group_col="perfume_id",
        threshold=args.threshold,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    combined_predictions.to_csv(args.out_root / "og_plus_synthetic_all_rows_oof.csv", index=False)
    combined_per_label.to_csv(args.out_root / "og_plus_synthetic_all_rows_per_label.csv", index=False)

    fold_df = pd.read_csv(args.fold_assignments)
    aug_predictions, aug_per_label, aug_metrics = evaluate_fixed_og_folds(
        combined,
        og_df[keep_cols],
        model_spec=model_spec,
        fold_df=fold_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        threshold=args.threshold,
        seed=args.seed,
    )
    aug_predictions.to_csv(args.out_root / "og_plus_synthetic_eval_og_rows_oof.csv", index=False)
    aug_per_label.to_csv(args.out_root / "og_plus_synthetic_eval_og_rows_per_label.csv", index=False)

    og_metrics = load_json(args.og_metrics)
    rows = [
        {
            "model_id": args.model_id,
            "model_family": model_spec.model_family,
            "experiment": "og_only_reference",
            "train_rows": int(og_metrics.get("n_rows", len(og_df))),
            "eval_rows": int(og_metrics.get("n_rows", len(og_df))),
            "macro_f1": float(og_metrics.get("macro_f1", np.nan)),
            "macro_roc_auc": float(og_metrics.get("macro_roc_auc", np.nan)),
            "macro_pr_auc": float(og_metrics.get("macro_pr_auc", np.nan)),
            "micro_f1": float(og_metrics.get("secondary_metrics", {}).get("micro_f1", np.nan)),
            "hamming_loss": float(og_metrics.get("secondary_metrics", {}).get("hamming_loss", np.nan)),
        },
        {
            "model_id": args.model_id,
            "model_family": model_spec.model_family,
            "experiment": "synthetic_only_review_level",
            "train_rows": int(len(synthetic_df)),
            "eval_rows": int(len(synthetic_df)),
            **synthetic_metrics,
        },
        {
            "model_id": args.model_id,
            "model_family": model_spec.model_family,
            "experiment": "og_plus_synthetic_all_rows_oof",
            "train_rows": int(len(combined)),
            "eval_rows": int(len(combined)),
            **combined_metrics,
        },
        {
            "model_id": args.model_id,
            "model_family": model_spec.model_family,
            "experiment": "og_plus_synthetic_eval_og_rows",
            "train_rows": int(len(combined)),
            "eval_rows": int(len(og_df)),
            **aug_metrics,
        },
    ]
    comparison = pd.DataFrame(rows)
    comparison_path = args.out_root / "metric_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    label_distribution = pd.concat(
        [
            summarize_labels(og_df, target_cols, "og_784_perfume_level"),
            summarize_labels(synthetic_df, target_cols, "synthetic_v2_review_level"),
            summarize_labels(combined, target_cols, "og_plus_synthetic_rows"),
        ],
        ignore_index=True,
    )
    label_distribution_path = args.out_root / "label_distribution.csv"
    label_distribution.to_csv(label_distribution_path, index=False)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "model_family": model_spec.model_family,
        "inputs": {
            "synthetic_labeled_features": stringify_path(args.synthetic_labeled_features),
            "synthetic_metrics": stringify_path(args.synthetic_metrics),
            "og_features": stringify_path(args.og_features),
            "og_metrics": stringify_path(args.og_metrics),
            "fold_assignments": stringify_path(args.fold_assignments),
        },
        "rows": {
            "og": int(len(og_df)),
            "synthetic": int(len(synthetic_df)),
            "combined": int(len(combined)),
        },
        "unique_perfumes": {
            "og": int(og_df["perfume_id"].nunique()),
            "synthetic": int(synthetic_df["perfume_id"].nunique()),
            "combined": int(combined["perfume_id"].nunique()),
        },
        "feature_count": int(len(feature_cols)),
        "target_columns": target_cols,
        "threshold": args.threshold,
        "metrics": rows,
        "outputs": {
            "combined_features": stringify_path(combined_path),
            "metric_comparison": stringify_path(comparison_path),
            "label_distribution": stringify_path(label_distribution_path),
            "synthetic_oof": stringify_path(args.out_root / "synthetic_only_review_level_oof.csv"),
            "synthetic_per_label": stringify_path(args.out_root / "synthetic_only_review_level_per_label.csv"),
            "combined_oof": stringify_path(args.out_root / "og_plus_synthetic_all_rows_oof.csv"),
            "combined_per_label": stringify_path(args.out_root / "og_plus_synthetic_all_rows_per_label.csv"),
            "augmented_og_oof": stringify_path(args.out_root / "og_plus_synthetic_eval_og_rows_oof.csv"),
            "augmented_og_per_label": stringify_path(args.out_root / "og_plus_synthetic_eval_og_rows_per_label.csv"),
            "report_md": stringify_path(args.out_root / "comparison_report.md"),
            "report_json": stringify_path(args.out_root / "comparison_report.json"),
        },
        "note": (
            "OG rows are perfume-level while synthetic rows are review-level. "
            "The OG-eval augmentation result is the most relevant check for whether synthetic rows help the real-review benchmark."
        ),
    }
    (args.out_root / "comparison_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_report(args.out_root / "comparison_report.md", report, comparison, label_distribution)

    print(f"Wrote combined features: {combined_path}")
    print(f"Wrote comparison: {comparison_path}")
    print(f"Wrote report: {args.out_root / 'comparison_report.md'}")
    for row in rows:
        print(
            f"{row['experiment']}: macro_f1={row['macro_f1']:.4f} "
            f"macro_roc_auc={row['macro_roc_auc']:.4f} macro_pr_auc={row['macro_pr_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
