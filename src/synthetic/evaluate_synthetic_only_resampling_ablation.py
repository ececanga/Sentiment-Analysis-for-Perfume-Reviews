# Date Created: 2026-06-22
# Date Last Modified: 2026-06-22

"""Compare data-level resampling methods for a synthetic-only labeled corpus."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imblearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.synthetic.evaluate_review_level_augmentation import (
    DEFAULT_TARGETS,
    MODEL_SPECS,
    coerce_targets,
    finite_mean,
    load_synthetic_frame,
    metric_or_nan,
    positive_class_probability,
    select_feature_columns,
    stringify_path,
)
from src.synthetic.evaluate_synthetic_only_balancing_ablation import (
    DEFAULT_FEATURES,
    DEFAULT_PERMUTATION_SEEDS,
    build_unbalanced_logistic,
    build_unbalanced_xgboost,
    label_distribution,
    reorder_by_perfume_seed,
)
from src.training.model_benchmark_runner import BenchmarkModelSpec
from src.utils.paths import PROJECT_ROOT


DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "synthetic_only_resampling_ablation_gemini_browser_6reviews_threshold05"


def validate_resampling_input(
    df: pd.DataFrame,
    target_cols: list[str],
    *,
    expected_unique_perfumes: int = 784,
    expected_reviews_per_perfume: int | None = None,
) -> dict[str, Any]:
    per_perfume_counts = df.groupby(df["perfume_id"].astype(str)).size()
    unique_counts = sorted(int(value) for value in per_perfume_counts.unique().tolist())
    if len(unique_counts) != 1:
        raise ValueError(f"Expected a constant row count per perfume, found {unique_counts}.")

    inferred_reviews_per_perfume = unique_counts[0]
    reviews_per_perfume = expected_reviews_per_perfume or inferred_reviews_per_perfume
    expected_review_indices = list(range(1, reviews_per_perfume + 1))
    review_indices = sorted(int(value) for value in df["review_index"].unique().tolist())
    label_missing = int(df[target_cols].isna().sum().sum())
    validation = {
        "rows": int(len(df)),
        "unique_perfumes": int(df["perfume_id"].nunique()),
        "reviews_per_perfume": int(reviews_per_perfume),
        "review_indices": review_indices,
        "per_perfume_counts": unique_counts,
        "unique_synthetic_review_ids": int(df["synthetic_review_id"].nunique()),
        "missing_label_values": label_missing,
    }
    expected_rows = expected_unique_perfumes * reviews_per_perfume
    if validation["rows"] != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {validation['rows']}.")
    if validation["unique_perfumes"] != expected_unique_perfumes:
        raise ValueError(f"Expected {expected_unique_perfumes} perfumes, found {validation['unique_perfumes']}.")
    if validation["review_indices"] != expected_review_indices:
        raise ValueError(f"Expected review indices {expected_review_indices}, found {validation['review_indices']}.")
    if validation["per_perfume_counts"] != [reviews_per_perfume]:
        raise ValueError(
            f"Expected exactly {reviews_per_perfume} rows per perfume, found {validation['per_perfume_counts']}."
        )
    if validation["unique_synthetic_review_ids"] != validation["rows"]:
        raise ValueError("Synthetic review IDs are not unique.")
    if validation["missing_label_values"] != 0:
        raise ValueError(f"Found {validation['missing_label_values']} missing label values.")
    return validation


def model_variants() -> list[dict[str, Any]]:
    return [
        {
            "model_id": "logistic_regression_unbalanced",
            "model_family": "Logistic Regression",
            "strategy": "unbalanced",
            "strategy_detail": "no class weighting or resampling",
            "resampling_method": "none",
            "spec": BenchmarkModelSpec(
                model_id="logistic_regression_unbalanced",
                model_family="Logistic Regression",
                build_model=build_unbalanced_logistic,
                model_params={},
                package_names=(),
            ),
        },
        {
            "model_id": "logistic_regression_class_weight",
            "model_family": "Logistic Regression",
            "strategy": "class_weight",
            "strategy_detail": 'class_weight="balanced"',
            "resampling_method": "none",
            "spec": MODEL_SPECS["logistic_regression"],
        },
        {
            "model_id": "logistic_regression_oversampling",
            "model_family": "Logistic Regression",
            "strategy": "oversampling",
            "strategy_detail": "randomly duplicate minority-class training rows per label/fold",
            "resampling_method": "oversampling",
            "spec": BenchmarkModelSpec(
                model_id="logistic_regression_oversampling",
                model_family="Logistic Regression",
                build_model=build_unbalanced_logistic,
                model_params={},
                package_names=(),
            ),
        },
        {
            "model_id": "logistic_regression_undersampling",
            "model_family": "Logistic Regression",
            "strategy": "undersampling",
            "strategy_detail": "randomly downsample majority-class training rows per label/fold",
            "resampling_method": "undersampling",
            "spec": BenchmarkModelSpec(
                model_id="logistic_regression_undersampling",
                model_family="Logistic Regression",
                build_model=build_unbalanced_logistic,
                model_params={},
                package_names=(),
            ),
        },
        {
            "model_id": "logistic_regression_smote",
            "model_family": "Logistic Regression",
            "strategy": "smote",
            "strategy_detail": "imbalanced-learn SMOTE per label/fold",
            "resampling_method": "smote",
            "spec": BenchmarkModelSpec(
                model_id="logistic_regression_smote",
                model_family="Logistic Regression",
                build_model=build_unbalanced_logistic,
                model_params={},
                package_names=(),
            ),
        },
        {
            "model_id": "xgboost_unbalanced",
            "model_family": "XGBoost",
            "strategy": "unbalanced",
            "strategy_detail": "scale_pos_weight=1.0 and no resampling",
            "resampling_method": "none",
            "spec": BenchmarkModelSpec(
                model_id="xgboost_unbalanced",
                model_family="XGBoost",
                build_model=build_unbalanced_xgboost,
                model_params={},
                package_names=("xgboost",),
            ),
        },
        {
            "model_id": "xgboost_scale_pos_weight",
            "model_family": "XGBoost",
            "strategy": "scale_pos_weight",
            "strategy_detail": "scale_pos_weight=labelwise_neg_to_pos_ratio",
            "resampling_method": "none",
            "spec": MODEL_SPECS["xgboost"],
        },
        {
            "model_id": "xgboost_oversampling",
            "model_family": "XGBoost",
            "strategy": "oversampling",
            "strategy_detail": "randomly duplicate minority-class training rows per label/fold",
            "resampling_method": "oversampling",
            "spec": BenchmarkModelSpec(
                model_id="xgboost_oversampling",
                model_family="XGBoost",
                build_model=build_unbalanced_xgboost,
                model_params={},
                package_names=("xgboost",),
            ),
        },
        {
            "model_id": "xgboost_undersampling",
            "model_family": "XGBoost",
            "strategy": "undersampling",
            "strategy_detail": "randomly downsample majority-class training rows per label/fold",
            "resampling_method": "undersampling",
            "spec": BenchmarkModelSpec(
                model_id="xgboost_undersampling",
                model_family="XGBoost",
                build_model=build_unbalanced_xgboost,
                model_params={},
                package_names=("xgboost",),
            ),
        },
        {
            "model_id": "xgboost_smote",
            "model_family": "XGBoost",
            "strategy": "smote",
            "strategy_detail": "imbalanced-learn SMOTE per label/fold",
            "resampling_method": "smote",
            "spec": BenchmarkModelSpec(
                model_id="xgboost_smote",
                model_family="XGBoost",
                build_model=build_unbalanced_xgboost,
                model_params={},
                package_names=("xgboost",),
            ),
        },
    ]


def resample_random_oversampling(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.Series, str]:
    y_arr = y.astype(int).to_numpy()
    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) < 2 or counts.min() == counts.max():
        return X.copy(), y.copy(), "unchanged"
    minority = int(classes[np.argmin(counts)])
    add_count = int(counts.max() - counts.min())
    sampler = RandomOverSampler(random_state=int(rng.integers(0, np.iinfo(np.int32).max)))
    X_res, y_res = sampler.fit_resample(X, y.astype(int))
    return (
        pd.DataFrame(X_res, columns=X.columns).reset_index(drop=True),
        pd.Series(y_res, name=y.name).astype(int).reset_index(drop=True),
        f"imblearn_random_oversampled_class_{minority}_by_{add_count}",
    )


def resample_random_undersampling(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.Series, str]:
    y_arr = y.astype(int).to_numpy()
    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) < 2 or counts.min() == counts.max():
        return X.copy(), y.copy(), "unchanged"
    sampler = RandomUnderSampler(random_state=int(rng.integers(0, np.iinfo(np.int32).max)))
    X_res, y_res = sampler.fit_resample(X, y.astype(int))
    return (
        pd.DataFrame(X_res, columns=X.columns).reset_index(drop=True),
        pd.Series(y_res, name=y.name).astype(int).reset_index(drop=True),
        "imblearn_random_downsampled_majority",
    )


def resample_smote(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    rng: np.random.Generator,
    k_neighbors: int = 5,
) -> tuple[pd.DataFrame, pd.Series, str]:
    y_arr = y.astype(int).to_numpy()
    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) < 2 or counts.min() == counts.max():
        return X.copy(), y.copy(), "unchanged"

    minority = int(classes[np.argmin(counts)])
    minority_count = int(counts.min())
    add_count = int(counts.max() - counts.min())
    if minority_count < 2:
        return resample_random_oversampling(X, y, rng=rng)

    sampler = SMOTE(
        random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
        k_neighbors=min(k_neighbors, minority_count - 1),
    )
    X_res, y_res = sampler.fit_resample(X, y.astype(int))
    return (
        pd.DataFrame(X_res, columns=X.columns).reset_index(drop=True),
        pd.Series(y_res, name=y.name).astype(int).reset_index(drop=True),
        f"imblearn_smote_class_{minority}_by_{add_count}",
    )


def resample_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    method: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, str]:
    rng = np.random.default_rng(seed)
    if method == "none":
        return X.copy(), y.copy().astype(int), "none"
    if method == "oversampling":
        return resample_random_oversampling(X, y, rng=rng)
    if method == "undersampling":
        return resample_random_undersampling(X, y, rng=rng)
    if method == "smote":
        return resample_smote(X, y, rng=rng)
    raise ValueError(f"Unknown resampling method: {method}")


def evaluate_grouped_oof_with_resampling(
    df: pd.DataFrame,
    *,
    model_spec: BenchmarkModelSpec,
    feature_cols: list[str],
    target_cols: list[str],
    group_col: str,
    threshold: float,
    n_splits: int,
    seed: int,
    resampling_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
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
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx]
        for target_index, target in enumerate(target_cols):
            y_train = df[target].iloc[train_idx].astype(int).reset_index(drop=True)
            if y_train.nunique() < 2:
                fallback = float(y_train.mean())
                predictions.loc[test_idx, f"prob_{target}"] = fallback
                predictions.loc[test_idx, f"pred_{target}"] = int(fallback >= threshold)
                fold_status[target].append(f"fold_{fold_id}:single_class_train_fallback")
                continue

            resample_seed = seed + fold_id * 1009 + target_index * 9173
            X_fit, y_fit, resample_status = resample_training_data(
                X_train,
                y_train,
                method=resampling_method,
                seed=resample_seed,
            )
            model = model_spec.build_model(seed, y_fit.to_numpy())
            model.fit(X_fit, y_fit)
            probs = positive_class_probability(model, X_test)
            predictions.loc[test_idx, f"prob_{target}"] = probs
            predictions.loc[test_idx, f"pred_{target}"] = (probs >= threshold).astype(int)
            fold_status[target].append(f"fold_{fold_id}:trained:{resample_status}:n_train_{len(y_fit)}")

    per_label_rows: list[dict[str, Any]] = []
    pred_columns: list[np.ndarray] = []
    for target in target_cols:
        y_true = df[target].astype(int).to_numpy()
        y_prob = predictions[f"prob_{target}"].astype(float).to_numpy()
        y_pred = predictions[f"pred_{target}"].astype(int).to_numpy()
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
                "training_status": ";".join(fold_status[target]),
            }
        )

    per_label = pd.DataFrame(per_label_rows)
    y_true_matrix = df[target_cols].astype(int).to_numpy()
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
    return predictions, per_label, metrics


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "model_id",
        "model_family",
        "strategy",
        "strategy_detail",
        "resampling_method",
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
    return summary.sort_values(["model_family", "macro_f1_mean"], ascending=[True, False]).reset_index(drop=True)


def aggregate_per_label(per_label: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "model_id",
        "model_family",
        "strategy",
        "strategy_detail",
        "resampling_method",
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
    return summary.sort_values(["model_family", "label", "strategy"]).reset_index(drop=True)


def add_deltas_vs_unbalanced(summary: pd.DataFrame) -> pd.DataFrame:
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
        baseline = out[out["model_family"].eq(model_family) & out["strategy"].eq("unbalanced")]
        if len(baseline) != 1:
            raise ValueError(f"Expected one unbalanced baseline for {model_family}.")
        base_row = baseline.iloc[0]
        mask = out["model_family"].eq(model_family)
        for metric in metric_cols:
            out.loc[mask, f"delta_{metric}_vs_unbalanced"] = out.loc[mask, metric].astype(float) - float(base_row[metric])
    return out


def add_per_label_deltas_vs_unbalanced(per_label: pd.DataFrame) -> pd.DataFrame:
    out = per_label.copy()
    metric_cols = ["f1_mean", "roc_auc_mean", "pr_auc_mean", "predicted_positive_rate_mean", "mean_probability_mean"]
    for metric in metric_cols:
        out[f"delta_{metric}_vs_unbalanced"] = 0.0
    for model_family in out["model_family"].unique():
        for label in out["label"].unique():
            baseline = out[
                out["model_family"].eq(model_family)
                & out["label"].eq(label)
                & out["strategy"].eq("unbalanced")
            ]
            if len(baseline) != 1:
                raise ValueError(f"Expected one unbalanced baseline for {model_family}/{label}.")
            base_row = baseline.iloc[0]
            mask = out["model_family"].eq(model_family) & out["label"].eq(label)
            for metric in metric_cols:
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
    corpus_label = f"{validation['unique_perfumes']}x{validation['reviews_per_perfume']}"
    summary_cols = [
        "model_family",
        "strategy",
        "resampling_method",
        "runs",
        "macro_f1_mean",
        "macro_f1_std",
        "delta_macro_f1_mean_vs_unbalanced",
        "macro_roc_auc_mean",
        "macro_pr_auc_mean",
        "predicted_label_cardinality_mean",
    ]
    lines = [
        f"# Synthetic-Only {corpus_label} Data-Level Resampling Ablation",
        "",
        "This compares imbalanced-learn random oversampling, random undersampling, and SMOTE against unbalanced and model-weighted baselines.",
        "",
        f"- Rows: {validation['rows']}",
        f"- Unique perfumes: {validation['unique_perfumes']}",
        f"- Reviews per perfume: {validation['reviews_per_perfume']}",
        f"- Review indices: {', '.join(str(value) for value in validation['review_indices'])}",
        f"- Threshold: {threshold:g}",
        f"- imbalanced-learn version: {imblearn.__version__}",
        "- Split: GroupKFold with 5 folds, grouped by perfume_id",
        "- Resampling is done inside each training fold separately, for each binary label classifier.",
        "- Test folds are never resampled.",
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

    label_cols = [
        "model_family",
        "strategy",
        "label",
        "positive_rate",
        "f1_mean",
        "delta_f1_mean_vs_unbalanced",
        "pr_auc_mean",
        "delta_pr_auc_mean_vs_unbalanced",
        "predicted_positive_rate_mean",
    ]
    lines.extend(["", "## Per-Label Metrics", "", "| " + " | ".join(label_cols) + " |", "| " + " | ".join("---" for _ in label_cols) + " |"])
    for _, row in per_label[label_cols].sort_values(["model_family", "label", "strategy"]).iterrows():
        values = []
        for col in label_cols:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Label Distribution", "", "| label | positive_count | positive_rate | neg_to_pos_ratio |", "| --- | --- | --- | --- |"])
    for _, row in labels.iterrows():
        lines.append(
            f"| {row['label']} | {int(row['positive_count'])} | "
            f"{float(row['positive_rate']):.4f} | {float(row['neg_to_pos_ratio']):.4f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic-only data-level resampling ablation.")
    parser.add_argument("--synthetic-labeled-features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--permutation-seeds", type=int, nargs="+", default=DEFAULT_PERMUTATION_SEEDS)
    parser.add_argument("--expected-unique-perfumes", type=int, default=784)
    parser.add_argument("--expected-reviews-per-perfume", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    target_cols = DEFAULT_TARGETS
    synthetic_df = coerce_targets(load_synthetic_frame(args.synthetic_labeled_features, target_cols), target_cols)
    validation = validate_resampling_input(
        synthetic_df,
        target_cols,
        expected_unique_perfumes=args.expected_unique_perfumes,
        expected_reviews_per_perfume=args.expected_reviews_per_perfume,
    )
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

    for permutation_seed in args.permutation_seeds:
        eval_df = reorder_by_perfume_seed(base_eval_df, permutation_seed)
        for variant in model_variants():
            _, per_label, metrics = evaluate_grouped_oof_with_resampling(
                eval_df,
                model_spec=variant["spec"],
                feature_cols=feature_cols,
                target_cols=target_cols,
                group_col="perfume_id",
                threshold=args.threshold,
                n_splits=args.n_splits,
                seed=args.model_seed,
                resampling_method=variant["resampling_method"],
            )
            per_label = per_label.copy()
            per_label.insert(0, "model_id", variant["model_id"])
            per_label.insert(1, "model_family", variant["model_family"])
            per_label.insert(2, "strategy", variant["strategy"])
            per_label.insert(3, "strategy_detail", variant["strategy_detail"])
            per_label.insert(4, "resampling_method", variant["resampling_method"])
            per_label.insert(5, "permutation_seed", permutation_seed)
            per_label = per_label.merge(labels[["label", "neg_to_pos_ratio"]], on="label", how="left")
            per_label_frames.append(per_label)

            run_rows.append(
                {
                    "model_id": variant["model_id"],
                    "model_family": variant["model_family"],
                    "strategy": variant["strategy"],
                    "strategy_detail": variant["strategy_detail"],
                    "resampling_method": variant["resampling_method"],
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
                f"seed={permutation_seed} {variant['model_id']}: "
                f"macro_f1={metrics['macro_f1']:.4f} "
                f"macro_roc_auc={metrics['macro_roc_auc']:.4f} "
                f"macro_pr_auc={metrics['macro_pr_auc']:.4f}"
            )

    runs = pd.DataFrame(run_rows)
    summary = add_deltas_vs_unbalanced(aggregate_runs(runs))
    per_label_runs = pd.concat(per_label_frames, ignore_index=True)
    per_label = add_per_label_deltas_vs_unbalanced(aggregate_per_label(per_label_runs))

    runs_path = args.out_root / "synthetic_only_resampling_ablation_runs.csv"
    summary_path = args.out_root / "synthetic_only_resampling_ablation_summary.csv"
    per_label_runs_path = args.out_root / "synthetic_only_resampling_ablation_per_label_runs.csv"
    per_label_path = args.out_root / "synthetic_only_resampling_ablation_per_label.csv"
    labels_path = args.out_root / "synthetic_only_resampling_ablation_label_distribution.csv"
    report_md_path = args.out_root / "synthetic_only_resampling_ablation_summary.md"
    report_json_path = args.out_root / "synthetic_only_resampling_ablation_report.json"

    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    per_label_runs.to_csv(per_label_runs_path, index=False)
    per_label.to_csv(per_label_path, index=False)
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
        "imbalanced_learn_version": imblearn.__version__,
        "target_columns": target_cols,
        "feature_count": len(feature_cols),
        "outputs": {
            "runs_csv": stringify_path(runs_path),
            "summary_csv": stringify_path(summary_path),
            "per_label_runs_csv": stringify_path(per_label_runs_path),
            "per_label_csv": stringify_path(per_label_path),
            "label_distribution_csv": stringify_path(labels_path),
            "summary_md": stringify_path(report_md_path),
            "report_json": stringify_path(report_json_path),
        },
        "note": (
            "Data-level resampling is applied only to each training fold and each binary label classifier. "
            "Held-out test folds are never resampled."
        ),
    }
    report_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote report: {report_md_path}")


if __name__ == "__main__":
    main()
