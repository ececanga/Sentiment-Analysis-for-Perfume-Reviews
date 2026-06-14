# Date Created: 2026-06-12
# Date Last Modified: 2026-06-12

"""Evaluate labeled review-level synthetic pilot with the classical LR pipeline."""

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
from sklearn.model_selection import GroupKFold

from src.labeling.strict_schema import EMOTIONS, VAD_DIMS
from src.training.run_logistic_regression_benchmark import MODEL_SPEC as LOGISTIC_MODEL_SPEC
from src.utils.paths import PROJECT_ROOT

FEATURE_PREFIXES = ("NRC_", "VAD_", "Hartmann_", "Savani_", "GoEmotions_", "Cardiff_")


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column.startswith(FEATURE_PREFIXES) and pd.api.types.is_numeric_dtype(df[column])
    ]


def metric_or_nan(fn, *args: Any) -> float:
    try:
        return float(fn(*args))
    except Exception:
        return float("nan")


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return float(np.mean(finite)) if finite else float("nan")


def load_labeled_feature_frame(features_csv: Path, labels_csv: Path) -> pd.DataFrame:
    features = pd.read_csv(features_csv)
    labels = pd.read_csv(labels_csv)
    if "synthetic_review_id" not in features.columns or "synthetic_review_id" not in labels.columns:
        raise ValueError("Both feature and label files must include synthetic_review_id.")
    label_cols = [column for column in EMOTIONS + VAD_DIMS if column in labels.columns]
    merged = features.merge(
        labels[["synthetic_review_id", *label_cols]],
        on="synthetic_review_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(features) or len(merged) != len(labels):
        raise ValueError(f"Feature/label row mismatch after merge: features={len(features)} labels={len(labels)} merged={len(merged)}")
    for emotion in EMOTIONS:
        if emotion in merged.columns:
            merged[emotion] = pd.to_numeric(merged[emotion], errors="raise").astype(int)
    for dim in VAD_DIMS:
        if dim in merged.columns:
            merged[dim] = pd.to_numeric(merged[dim], errors="raise")
    return merged


def evaluate_logistic(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_cols: list[str],
    group_col: str,
    threshold: float,
    n_splits: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    groups = df[group_col].astype(str)
    splitter = GroupKFold(n_splits=n_splits)

    prob_cols = [f"prob_{target}" for target in target_cols]
    pred_cols = [f"pred_{target}" for target in target_cols]
    prediction_df = df[["synthetic_review_id", "perfume_id", "Perfume", "review_index"]].copy()
    for column in prob_cols:
        prediction_df[column] = np.nan
    for column in pred_cols:
        prediction_df[column] = 0
    prediction_df["fold_id"] = 0

    fold_status: dict[str, list[str]] = {target: [] for target in target_cols}
    X = df[feature_cols]

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, groups=groups), start=1):
        prediction_df.loc[test_idx, "fold_id"] = fold_id
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        for target in target_cols:
            y_train = df[target].iloc[train_idx].astype(int)
            if y_train.nunique() < 2:
                fallback = float(y_train.mean())
                prediction_df.loc[test_idx, f"prob_{target}"] = fallback
                prediction_df.loc[test_idx, f"pred_{target}"] = int(fallback >= threshold)
                fold_status[target].append(f"fold_{fold_id}:single_class_train_fallback")
                continue
            model = LOGISTIC_MODEL_SPEC.build_model(seed, y_train.to_numpy())
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)
            class_list = list(model.named_steps["clf"].classes_) if hasattr(model, "named_steps") else list(model.classes_)
            positive_idx = class_list.index(1) if 1 in class_list else len(class_list) - 1
            probs = proba[:, positive_idx]
            prediction_df.loc[test_idx, f"prob_{target}"] = probs
            prediction_df.loc[test_idx, f"pred_{target}"] = (probs >= threshold).astype(int)
            fold_status[target].append(f"fold_{fold_id}:trained")

    per_label_rows: list[dict[str, Any]] = []
    for target in target_cols:
        y_true = df[target].astype(int).to_numpy()
        y_prob = prediction_df[f"prob_{target}"].astype(float).to_numpy()
        y_pred = prediction_df[f"pred_{target}"].astype(int).to_numpy()
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
    y_pred_matrix = prediction_df[pred_cols].astype(int).to_numpy()
    macro = {
        "macro_f1": finite_mean(per_label["f1"].astype(float).tolist()),
        "macro_roc_auc": finite_mean(per_label["roc_auc"].astype(float).tolist()),
        "macro_pr_auc": finite_mean(per_label["pr_auc"].astype(float).tolist()),
        "micro_f1": metric_or_nan(f1_score, y_true_matrix.ravel(), y_pred_matrix.ravel()),
        "hamming_loss": metric_or_nan(hamming_loss, y_true_matrix, y_pred_matrix),
        "label_cardinality": float(y_true_matrix.sum(axis=1).mean()),
        "predicted_label_cardinality": float(y_pred_matrix.sum(axis=1).mean()),
    }
    return prediction_df, per_label, macro


def write_markdown_report(path: Path, report: dict[str, Any], per_label: pd.DataFrame) -> None:
    lines = [
        "# Review-Level Synthetic V2 Pilot Evaluation",
        "",
        f"- Rows: {report['n_rows']}",
        f"- Perfumes: {report['unique_perfumes']}",
        f"- Model: {report['model']}",
        f"- Split: {report['split']}",
        f"- Threshold: {report['threshold']}",
        "",
        "## Metrics",
        "",
        f"- Macro F1: {report['metrics']['macro_f1']:.4f}",
        f"- Macro ROC AUC: {report['metrics']['macro_roc_auc']:.4f}",
        f"- Macro PR AUC: {report['metrics']['macro_pr_auc']:.4f}",
        f"- Micro F1: {report['metrics']['micro_f1']:.4f}",
        f"- Hamming loss: {report['metrics']['hamming_loss']:.4f}",
        "",
        "## Per Label",
        "",
    ]
    columns = ["label", "support_pos", "positive_rate", "predicted_positive_rate", "f1", "roc_auc", "pr_auc"]
    table = per_label[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for _, row in table.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        body.append("| " + " | ".join(values) + " |")
    lines.extend([header, sep, *body, ""])
    lines.append(
        "Note: this is a 100-review pilot with Gemini 2.5 Pro pseudo-labels. "
        "The split is grouped by perfume, so results are useful as a sanity check but not a final thesis benchmark."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate labeled v2 review-level pilot.")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "synthetic_review_level_tests"
        / "emotion_rich_v2_pilot100"
        / "features"
        / "review_level_features_unlabeled.csv",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=PROJECT_ROOT
        / "labels"
        / "gemini_review_level"
        / "gemini_review_labels_emotion_rich_v2_pilot100_g25pro_v1_pipeline.csv",
    )
    parser.add_argument(
        "--og-metrics-json",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "gemini_judge"
        / "judge_g25pro_v1"
        / "model_benchmark"
        / "metrics"
        / "logistic_regression.json",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "synthetic_review_level_tests"
        / "emotion_rich_v2_pilot100"
        / "supervised_eval_g25pro",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    labeled_df = load_labeled_feature_frame(args.features_csv, args.labels_csv)
    feature_cols = select_feature_columns(labeled_df)
    if not feature_cols:
        raise ValueError("No numeric pipeline feature columns found.")

    target_cols = [
        emotion
        for emotion in EMOTIONS
        if emotion in labeled_df.columns and labeled_df[emotion].nunique() == 2
    ]
    excluded_targets = {
        emotion: "single_class_in_labeled_v2_pilot"
        for emotion in EMOTIONS
        if emotion in labeled_df.columns and labeled_df[emotion].nunique() < 2
    }
    if not target_cols:
        raise ValueError("No binary emotion target has both positive and negative examples.")

    labeled_path = args.out_root / "review_level_features_labeled_g25pro.csv"
    predictions_path = args.out_root / "logistic_regression_review_level_oof.csv"
    per_label_path = args.out_root / "logistic_regression_review_level_per_label.csv"
    metrics_path = args.out_root / "logistic_regression_review_level_metrics.json"
    report_path = args.out_root / "logistic_regression_review_level_report.md"
    labeled_df.to_csv(labeled_path, index=False)

    predictions, per_label, macro = evaluate_logistic(
        labeled_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        group_col="perfume_id",
        threshold=args.threshold,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    predictions.to_csv(predictions_path, index=False)
    per_label.to_csv(per_label_path, index=False)

    og_metrics: dict[str, Any] = {}
    if args.og_metrics_json.exists():
        og_raw = json.loads(args.og_metrics_json.read_text(encoding="utf-8"))
        og_metrics = {
            "path": stringify_path(args.og_metrics_json),
            "n_rows": og_raw.get("n_rows"),
            "macro_f1": og_raw.get("macro_f1"),
            "macro_roc_auc": og_raw.get("macro_roc_auc"),
            "macro_pr_auc": og_raw.get("macro_pr_auc"),
            "micro_f1": og_raw.get("secondary_metrics", {}).get("micro_f1"),
            "threshold": og_raw.get("threshold"),
            "target_columns": og_raw.get("target_columns"),
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "features_csv": stringify_path(args.features_csv),
        "labels_csv": stringify_path(args.labels_csv),
        "n_rows": int(len(labeled_df)),
        "unique_perfumes": int(labeled_df["perfume_id"].nunique()),
        "model": "logistic_regression",
        "split": f"GroupKFold(n_splits={args.n_splits}, group=perfume_id)",
        "threshold": args.threshold,
        "feature_count": int(len(feature_cols)),
        "target_columns": target_cols,
        "excluded_target_columns": excluded_targets,
        "metrics": macro,
        "og_logistic_regression_reference": og_metrics,
        "outputs": {
            "labeled_features": stringify_path(labeled_path),
            "predictions": stringify_path(predictions_path),
            "per_label": stringify_path(per_label_path),
            "metrics": stringify_path(metrics_path),
            "report": stringify_path(report_path),
        },
    }
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(report_path, report, per_label)

    print(f"Wrote labeled features: {labeled_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(
        "Review-level LR: "
        f"macro_f1={macro['macro_f1']:.4f} "
        f"macro_roc_auc={macro['macro_roc_auc']:.4f} "
        f"macro_pr_auc={macro['macro_pr_auc']:.4f}"
    )
    print(f"Targets: {target_cols}; excluded: {excluded_targets}")


if __name__ == "__main__":
    main()
