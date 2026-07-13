# Date Created: 2026-06-23
# Date Last Modified: 2026-07-13

"""Compare balancing methods for OG YouTube plus synthetic review training."""

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
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, roc_auc_score

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.synthetic.evaluate_review_level_augmentation import (
    DEFAULT_TARGETS,
    coerce_targets,
    finite_mean,
    load_og_frame,
    load_synthetic_frame,
    metric_or_nan,
    positive_class_probability,
    select_feature_columns,
    stringify_path,
    summarize_labels,
)
from src.synthetic.evaluate_synthetic_only_resampling_ablation import (
    balance_training_frame,
    model_variants,
    resample_training_data,
)
from src.training.model_benchmark_runner import BenchmarkModelSpec
from src.training.source_grouped_folds import load_and_validate_outer_fold_manifest, sha256_file
from src.utils.paths import PROJECT_ROOT


DEFAULT_OG_FEATURES = (
    PROJECT_ROOT / "outputs" / "gemini_browser_og784" / "analysis_out" / "gemini_browser_og784_feature_labels.csv"
)
DEFAULT_FOLD_ASSIGNMENTS = (
    PROJECT_ROOT / "outputs" / "gemini_browser_og784" / "model_benchmark" / "tables" / "fold_assignments.csv"
)
DEFAULT_SYNTHETIC_3 = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_review_level_tests"
    / "emotion_rich_v2_3reviews"
    / "supervised_eval_gemini_browser"
    / "review_level_features_labeled_g25pro.csv"
)
DEFAULT_SYNTHETIC_6 = (
    PROJECT_ROOT
    / "outputs"
    / "synthetic_review_level_tests"
    / "emotion_rich_v2_6reviews"
    / "supervised_eval_gemini_browser"
    / "review_level_features_labeled_g25pro.csv"
)
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "og_synthetic_balancing_ablation_gemini_browser_thresholds"


def validate_synthetic_frame(df: pd.DataFrame, *, reviews_per_perfume: int, target_cols: list[str]) -> dict[str, Any]:
    per_perfume_counts = sorted(int(value) for value in df.groupby(df["perfume_id"].astype(str)).size().unique())
    review_indices = sorted(int(value) for value in df["review_index"].unique().tolist())
    validation = {
        "rows": int(len(df)),
        "unique_perfumes": int(df["perfume_id"].nunique()),
        "reviews_per_perfume": int(reviews_per_perfume),
        "review_indices": review_indices,
        "per_perfume_counts": per_perfume_counts,
        "unique_synthetic_review_ids": int(df["synthetic_review_id"].nunique()),
        "missing_label_values": int(df[target_cols].isna().sum().sum()),
    }
    expected_rows = 784 * reviews_per_perfume
    if validation["rows"] != expected_rows:
        raise ValueError(f"Expected {expected_rows} synthetic rows, found {validation['rows']}.")
    if validation["unique_perfumes"] != 784:
        raise ValueError(f"Expected 784 synthetic perfumes, found {validation['unique_perfumes']}.")
    if validation["review_indices"] != list(range(1, reviews_per_perfume + 1)):
        raise ValueError(f"Unexpected synthetic review indices: {validation['review_indices']}.")
    if validation["per_perfume_counts"] != [reviews_per_perfume]:
        raise ValueError(f"Unexpected synthetic rows per perfume: {validation['per_perfume_counts']}.")
    if validation["unique_synthetic_review_ids"] != validation["rows"]:
        raise ValueError("Synthetic review IDs are not unique.")
    if validation["missing_label_values"] != 0:
        raise ValueError(f"Found {validation['missing_label_values']} missing synthetic label values.")
    return validation


def validate_og_frame(df: pd.DataFrame, fold_df: pd.DataFrame, target_cols: list[str]) -> dict[str, Any]:
    fold_lookup = set(fold_df["perfume_id"].astype(str))
    og_ids = set(df["perfume_id"].astype(str))
    validation = {
        "rows": int(len(df)),
        "unique_perfumes": int(df["perfume_id"].nunique()),
        "review_indices": sorted(int(value) for value in df["review_index"].unique().tolist()),
        "unique_synthetic_review_ids": int(df["synthetic_review_id"].nunique()),
        "missing_label_values": int(df[target_cols].isna().sum().sum()),
        "fold_rows": int(len(fold_df)),
        "fold_unique_perfumes": int(fold_df["perfume_id"].nunique()),
        "fold_ids": sorted(int(value) for value in fold_df["fold_id"].unique().tolist()),
    }
    if validation["rows"] != 784 or validation["unique_perfumes"] != 784:
        raise ValueError(f"Expected 784 OG rows/perfumes, found {validation['rows']}/{validation['unique_perfumes']}.")
    if validation["review_indices"] != [0]:
        raise ValueError(f"Expected OG review index 0, found {validation['review_indices']}.")
    if validation["unique_synthetic_review_ids"] != validation["rows"]:
        raise ValueError("OG synthetic_review_id surrogate values are not unique.")
    if validation["missing_label_values"] != 0:
        raise ValueError(f"Found {validation['missing_label_values']} missing OG label values.")
    if validation["fold_rows"] != 784 or validation["fold_unique_perfumes"] != 784:
        raise ValueError("Expected 784 OG fold assignments.")
    if validation["fold_ids"] != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected fold IDs 1..5, found {validation['fold_ids']}.")
    missing_folds = sorted(og_ids - fold_lookup)
    if missing_folds:
        raise ValueError(f"Missing fold assignments for OG perfume IDs: {missing_folds[:10]}.")
    return validation


def build_data_setups(
    og_df: pd.DataFrame,
    synthetic_3: pd.DataFrame,
    synthetic_6: pd.DataFrame,
    keep_cols: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "data_setup": "og_only",
            "data_label": "OG YouTube only",
            "synthetic_reviews_per_perfume": 0,
            "synthetic_rows": 0,
            "train_df": og_df[keep_cols].copy(),
        },
        {
            "data_setup": "og_plus_synthetic_3reviews",
            "data_label": "OG YouTube + synthetic 784x3",
            "synthetic_reviews_per_perfume": 3,
            "synthetic_rows": int(len(synthetic_3)),
            "train_df": pd.concat([og_df[keep_cols], synthetic_3[keep_cols]], ignore_index=True),
        },
        {
            "data_setup": "og_plus_synthetic_6reviews",
            "data_label": "OG YouTube + synthetic 784x6",
            "synthetic_reviews_per_perfume": 6,
            "synthetic_rows": int(len(synthetic_6)),
            "train_df": pd.concat([og_df[keep_cols], synthetic_6[keep_cols]], ignore_index=True),
        },
    ]


def evaluate_fixed_folds_with_resampling(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    model_spec: BenchmarkModelSpec,
    fold_df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    threshold: float,
    seed: int,
    resampling_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fold_lookup = fold_df.set_index("perfume_id")["fold_id"].astype(int).to_dict()
    missing_eval = sorted(set(eval_df["perfume_id"].astype(str)) - set(fold_lookup))
    if missing_eval:
        raise ValueError(f"Missing fold assignments for eval perfume IDs: {missing_eval[:10]}.")

    predictions = eval_df[["synthetic_review_id", "perfume_id", "Perfume", "review_index", "source_type"]].copy()
    for target in target_cols:
        predictions[f"prob_{target}"] = np.nan
        predictions[f"pred_{target}"] = 0
    predictions["fold_id"] = predictions["perfume_id"].astype(str).map(fold_lookup).astype(int)

    fold_status: dict[str, list[str]] = {target: [] for target in target_cols}
    for fold_id in sorted(fold_df["fold_id"].astype(int).unique()):
        test_ids = set(fold_df.loc[fold_df["fold_id"].astype(int).eq(fold_id), "perfume_id"].astype(str))
        train_mask = ~train_df["perfume_id"].astype(str).isin(test_ids)
        eval_mask = eval_df["perfume_id"].astype(str).isin(test_ids)
        fold_train = train_df.loc[train_mask].reset_index(drop=True)
        if resampling_method == "label_hybrid":
            balanced_train, balance_report = balance_training_frame(
                fold_train,
                target_cols,
                seed=seed + fold_id * 1009,
            )
            X_train = balanced_train[feature_cols].reset_index(drop=True)
            y_source = balanced_train
            fold_resample_status = (
                f"label_hybrid_train_rows_{len(fold_train)}_to_{len(balanced_train)}"
                f"_sd_{float(balance_report['after_label_sd']):.2f}"
            )
        else:
            X_train = fold_train[feature_cols].reset_index(drop=True)
            y_source = fold_train
            fold_resample_status = ""
        X_eval = eval_df.loc[eval_mask, feature_cols]
        for target_index, target in enumerate(target_cols):
            y_train = y_source[target].astype(int).reset_index(drop=True)
            if y_train.nunique() < 2:
                fallback = float(y_train.mean())
                predictions.loc[eval_mask, f"prob_{target}"] = fallback
                predictions.loc[eval_mask, f"pred_{target}"] = int(fallback >= threshold)
                fold_status[target].append(f"fold_{fold_id}:single_class_train_fallback")
                continue

            resample_seed = seed + fold_id * 1009 + target_index * 9173
            if resampling_method == "label_hybrid":
                X_fit = X_train.copy()
                y_fit = y_train.copy()
                resample_status = fold_resample_status
            else:
                X_fit, y_fit, resample_status = resample_training_data(
                    X_train,
                    y_train,
                    method=resampling_method,
                    seed=resample_seed,
                )
            model = model_spec.build_model(seed, y_fit.to_numpy())
            model.fit(X_fit, y_fit)
            probs = positive_class_probability(model, X_eval)
            predictions.loc[eval_mask, f"prob_{target}"] = probs
            predictions.loc[eval_mask, f"pred_{target}"] = (probs >= threshold).astype(int)
            fold_status[target].append(f"fold_{fold_id}:trained:{resample_status}:n_train_{len(y_fit)}")

    per_label_rows: list[dict[str, Any]] = []
    pred_columns: list[np.ndarray] = []
    for target in target_cols:
        y_true = eval_df[target].astype(int).to_numpy()
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
    return predictions, per_label, metrics


def metrics_row(
    *,
    eval_scope: str,
    setup: dict[str, Any],
    variant: dict[str, Any],
    threshold: float,
    feature_count: int,
    eval_rows: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "eval_scope": eval_scope,
        "data_setup": setup["data_setup"],
        "data_label": setup["data_label"],
        "threshold": float(threshold),
        "model_id": variant["model_id"],
        "model_family": variant["model_family"],
        "strategy": variant["strategy"],
        "strategy_detail": variant["strategy_detail"],
        "resampling_method": variant["resampling_method"],
        "train_rows": int(len(setup["train_df"])),
        "og_train_rows": 784,
        "synthetic_rows": int(setup["synthetic_rows"]),
        "synthetic_reviews_per_perfume": int(setup["synthetic_reviews_per_perfume"]),
        "eval_rows": int(eval_rows),
        "feature_count": int(feature_count),
        "macro_f1": float(metrics["macro_f1"]),
        "macro_roc_auc": float(metrics["macro_roc_auc"]),
        "macro_pr_auc": float(metrics["macro_pr_auc"]),
        "micro_f1": float(metrics["micro_f1"]),
        "hamming_loss": float(metrics["hamming_loss"]),
        "label_cardinality": float(metrics["label_cardinality"]),
        "predicted_label_cardinality": float(metrics["predicted_label_cardinality"]),
    }


def per_label_with_context(
    per_label: pd.DataFrame,
    *,
    eval_scope: str,
    setup: dict[str, Any],
    variant: dict[str, Any],
    threshold: float,
) -> pd.DataFrame:
    out = per_label.copy()
    out.insert(0, "eval_scope", eval_scope)
    out.insert(1, "data_setup", setup["data_setup"])
    out.insert(2, "data_label", setup["data_label"])
    out.insert(3, "threshold", float(threshold))
    out.insert(4, "model_id", variant["model_id"])
    out.insert(5, "model_family", variant["model_family"])
    out.insert(6, "strategy", variant["strategy"])
    out.insert(7, "strategy_detail", variant["strategy_detail"])
    out.insert(8, "resampling_method", variant["resampling_method"])
    return out


def predictions_with_context(
    predictions: pd.DataFrame,
    *,
    eval_scope: str,
    setup: dict[str, Any],
    variant: dict[str, Any],
    threshold: float,
) -> pd.DataFrame:
    """Attach experiment identity to one out-of-fold prediction frame."""

    out = predictions.copy()
    out.insert(0, "eval_scope", eval_scope)
    out.insert(1, "data_setup", setup["data_setup"])
    out.insert(2, "threshold", float(threshold))
    out.insert(3, "model_id", variant["model_id"])
    out.insert(4, "strategy", variant["strategy"])
    return out


def summarize_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    summary = runs.copy()
    metric_cols = [
        "macro_f1",
        "macro_roc_auc",
        "macro_pr_auc",
        "micro_f1",
        "hamming_loss",
        "predicted_label_cardinality",
    ]
    for metric in metric_cols:
        summary[f"delta_{metric}_vs_unbalanced"] = 0.0
    group_cols = ["eval_scope", "data_setup", "threshold", "model_family"]
    for _, group in summary.groupby(group_cols, sort=False):
        baseline = group[group["strategy"].eq("unbalanced")]
        if len(baseline) != 1:
            raise ValueError(f"Expected one unbalanced baseline for {group.iloc[0][group_cols].to_dict()}.")
        base = baseline.iloc[0]
        mask = (
            summary["eval_scope"].eq(base["eval_scope"])
            & summary["data_setup"].eq(base["data_setup"])
            & summary["threshold"].eq(base["threshold"])
            & summary["model_family"].eq(base["model_family"])
        )
        for metric in metric_cols:
            summary.loc[mask, f"delta_{metric}_vs_unbalanced"] = (
                summary.loc[mask, metric].astype(float) - float(base[metric])
            )
    return summary.sort_values(
        ["eval_scope", "threshold", "data_setup", "model_family", "macro_f1"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)


def balanced_winners(summary: pd.DataFrame, *, eval_scope: str) -> pd.DataFrame:
    balanced = summary[summary["eval_scope"].eq(eval_scope) & ~summary["strategy"].eq("unbalanced")].copy()
    balanced = balanced.sort_values(
        ["threshold", "data_setup", "model_family", "macro_f1"],
        ascending=[True, True, True, False],
    )
    return balanced.groupby(["threshold", "data_setup", "model_family"], as_index=False).head(1).reset_index(drop=True)


def select_variants(model_ids: list[str], strategies: list[str]) -> list[dict[str, Any]]:
    """Return only explicitly requested model/strategy combinations."""

    available = model_variants()
    valid_models = sorted({str(item["model_family"]).lower().replace(" ", "_") for item in available})
    valid_strategies = sorted({str(item["strategy"]) for item in available})
    invalid_models = sorted(set(model_ids) - set(valid_models))
    invalid_strategies = sorted(set(strategies) - set(valid_strategies))
    if invalid_models:
        raise ValueError(f"Unknown model IDs: {invalid_models}; expected one of {valid_models}.")
    if invalid_strategies:
        raise ValueError(f"Unknown strategies: {invalid_strategies}; expected one of {valid_strategies}.")
    selected = [
        item
        for item in available
        if str(item["model_family"]).lower().replace(" ", "_") in model_ids
        and str(item["strategy"]) in strategies
    ]
    if not selected:
        raise ValueError("The model/strategy filters selected no experiment variants.")
    return selected


def write_markdown_report(
    path: Path,
    *,
    summary: pd.DataFrame,
    winners: pd.DataFrame,
    label_distribution: pd.DataFrame,
    validation: dict[str, Any],
) -> None:
    winner_cols = [
        "threshold",
        "data_label",
        "model_family",
        "strategy",
        "macro_f1",
        "macro_roc_auc",
        "macro_pr_auc",
        "delta_macro_f1_vs_unbalanced",
        "predicted_label_cardinality",
    ]
    main_cols = [
        "threshold",
        "data_label",
        "model_family",
        "strategy",
        "macro_f1",
        "macro_roc_auc",
        "macro_pr_auc",
        "delta_macro_f1_vs_unbalanced",
        "predicted_label_cardinality",
    ]
    lines = [
        "# OG YouTube + Synthetic Balancing Ablation",
        "",
        "Main metric: held-out OG YouTube rows using the frozen OG fold assignments.",
        "",
        f"- OG rows: {validation['og']['rows']}",
        f"- Synthetic 784x3 rows: {validation['synthetic_3']['rows']}",
        f"- Synthetic 784x6 rows: {validation['synthetic_6']['rows']}",
        f"- imbalanced-learn version: {imblearn.__version__}",
        "",
        "## Balanced-Technique Winners On Held-Out OG Rows",
        "",
        "| " + " | ".join(winner_cols) + " |",
        "| " + " | ".join("---" for _ in winner_cols) + " |",
    ]
    for _, row in winners[winner_cols].iterrows():
        values = []
        for col in winner_cols:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Held-Out OG Row Metrics",
            "",
            "`unbalanced` is included only as a baseline reference.",
            "",
            "| " + " | ".join(main_cols) + " |",
            "| " + " | ".join("---" for _ in main_cols) + " |",
        ]
    )
    main_summary = summary[summary["eval_scope"].eq("heldout_og_rows")]
    for _, row in main_summary[main_cols].iterrows():
        values = []
        for col in main_cols:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Label Distribution",
            "",
            "| source | label | rows | positive_count | positive_rate |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in label_distribution.iterrows():
        lines.append(
            f"| {row['source']} | {row['label']} | {int(row['rows'])} | "
            f"{int(row['positive_count'])} | {float(row['positive_rate']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Secondary all-row metrics are saved separately because synthetic rows dominate mixed row counts.",
            "- Resampling is applied only inside each training fold and separately per binary emotion label.",
            "- Evaluation folds exclude every row from the held-out perfume IDs, including synthetic rows for those perfumes.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OG YouTube + synthetic balancing ablation.")
    parser.add_argument("--og-features", type=Path, default=DEFAULT_OG_FEATURES)
    parser.add_argument("--synthetic-3-features", type=Path, default=DEFAULT_SYNTHETIC_3)
    parser.add_argument("--synthetic-6-features", type=Path, default=DEFAULT_SYNTHETIC_6)
    parser.add_argument("--fold-assignments", type=Path, default=DEFAULT_FOLD_ASSIGNMENTS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-ids",
        nargs="+",
        default=["logistic_regression", "xgboost"],
        choices=["logistic_regression", "xgboost"],
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["unbalanced", "scale_pos_weight", "oversampling", "undersampling", "hybrid", "smote"],
    )
    parser.add_argument(
        "--eval-scopes",
        nargs="+",
        default=["heldout_og_rows", "all_rows_diagnostic"],
        choices=["heldout_og_rows", "all_rows_diagnostic"],
    )
    parser.add_argument("--expected-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    target_cols = DEFAULT_TARGETS
    og_df = coerce_targets(load_og_frame(args.og_features, target_cols), target_cols)
    synthetic_3 = coerce_targets(load_synthetic_frame(args.synthetic_3_features, target_cols), target_cols)
    synthetic_6 = coerce_targets(load_synthetic_frame(args.synthetic_6_features, target_cols), target_cols)
    fold_df = load_and_validate_outer_fold_manifest(
        args.fold_assignments,
        expected_ids=og_df["perfume_id"].astype(str).tolist(),
        n_splits=args.expected_folds,
    )
    variants = select_variants(args.model_ids, args.strategies)

    validation = {
        "og": validate_og_frame(og_df, fold_df, target_cols),
        "synthetic_3": validate_synthetic_frame(synthetic_3, reviews_per_perfume=3, target_cols=target_cols),
        "synthetic_6": validate_synthetic_frame(synthetic_6, reviews_per_perfume=6, target_cols=target_cols),
    }
    feature_cols = select_feature_columns(og_df, synthetic_3, synthetic_6)
    if not feature_cols:
        raise ValueError("No common numeric feature columns found between OG and synthetic frames.")

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
    setups = build_data_setups(og_df, synthetic_3, synthetic_6, keep_cols)

    label_distribution = pd.concat(
        [
            summarize_labels(og_df, target_cols, "og_youtube_rows"),
            summarize_labels(synthetic_3, target_cols, "synthetic_784x3_rows"),
            summarize_labels(synthetic_6, target_cols, "synthetic_784x6_rows"),
            *[
                summarize_labels(setup["train_df"], target_cols, setup["data_setup"])
                for setup in setups
                if setup["data_setup"] != "og_only"
            ],
        ],
        ignore_index=True,
    )

    run_rows: list[dict[str, Any]] = []
    per_label_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for threshold in args.thresholds:
        for setup in setups:
            for variant in variants:
                scope_metrics: dict[str, dict[str, Any]] = {}
                if "heldout_og_rows" in args.eval_scopes:
                    og_predictions, og_per_label, og_metrics = evaluate_fixed_folds_with_resampling(
                        setup["train_df"],
                        og_df[keep_cols],
                        model_spec=variant["spec"],
                        fold_df=fold_df,
                        feature_cols=feature_cols,
                        target_cols=target_cols,
                        threshold=threshold,
                        seed=args.seed,
                        resampling_method=variant["resampling_method"],
                    )
                    scope_metrics["heldout_og_rows"] = og_metrics
                    run_rows.append(
                        metrics_row(
                            eval_scope="heldout_og_rows",
                            setup=setup,
                            variant=variant,
                            threshold=threshold,
                            feature_count=len(feature_cols),
                            eval_rows=len(og_df),
                            metrics=og_metrics,
                        )
                    )
                    per_label_frames.append(
                        per_label_with_context(
                            og_per_label,
                            eval_scope="heldout_og_rows",
                            setup=setup,
                            variant=variant,
                            threshold=threshold,
                        )
                    )
                    prediction_frames.append(
                        predictions_with_context(
                            og_predictions,
                            eval_scope="heldout_og_rows",
                            setup=setup,
                            variant=variant,
                            threshold=threshold,
                        )
                    )

                if "all_rows_diagnostic" in args.eval_scopes:
                    all_predictions, all_per_label, all_metrics = evaluate_fixed_folds_with_resampling(
                        setup["train_df"],
                        setup["train_df"],
                        model_spec=variant["spec"],
                        fold_df=fold_df,
                        feature_cols=feature_cols,
                        target_cols=target_cols,
                        threshold=threshold,
                        seed=args.seed,
                        resampling_method=variant["resampling_method"],
                    )
                    scope_metrics["all_rows_diagnostic"] = all_metrics
                    run_rows.append(
                        metrics_row(
                            eval_scope="all_rows_diagnostic",
                            setup=setup,
                            variant=variant,
                            threshold=threshold,
                            feature_count=len(feature_cols),
                            eval_rows=len(setup["train_df"]),
                            metrics=all_metrics,
                        )
                    )
                    per_label_frames.append(
                        per_label_with_context(
                            all_per_label,
                            eval_scope="all_rows_diagnostic",
                            setup=setup,
                            variant=variant,
                            threshold=threshold,
                        )
                    )
                    prediction_frames.append(
                        predictions_with_context(
                            all_predictions,
                            eval_scope="all_rows_diagnostic",
                            setup=setup,
                            variant=variant,
                            threshold=threshold,
                        )
                    )

                metric_text = " ".join(
                    f"{scope}={metrics['macro_f1']:.4f}" for scope, metrics in scope_metrics.items()
                )
                print(f"threshold={threshold:g} {setup['data_setup']} {variant['model_id']}: {metric_text}")

    runs = pd.DataFrame(run_rows)
    per_label = pd.concat(per_label_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_metrics(runs)
    winners = (
        balanced_winners(summary, eval_scope="heldout_og_rows")
        if "heldout_og_rows" in args.eval_scopes
        else pd.DataFrame(columns=summary.columns)
    )

    runs_path = args.out_root / "og_synthetic_balancing_ablation_runs.csv"
    summary_path = args.out_root / "og_synthetic_balancing_ablation_summary.csv"
    winners_path = args.out_root / "og_synthetic_balancing_ablation_balanced_winners.csv"
    per_label_path = args.out_root / "og_synthetic_balancing_ablation_per_label.csv"
    predictions_path = args.out_root / "og_synthetic_balancing_ablation_predictions.csv"
    labels_path = args.out_root / "og_synthetic_balancing_ablation_label_distribution.csv"
    report_md_path = args.out_root / "og_synthetic_balancing_ablation_summary.md"
    report_json_path = args.out_root / "og_synthetic_balancing_ablation_report.json"

    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    winners.to_csv(winners_path, index=False)
    per_label.to_csv(per_label_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    label_distribution.to_csv(labels_path, index=False)
    write_markdown_report(
        report_md_path,
        summary=summary,
        winners=winners,
        label_distribution=label_distribution,
        validation=validation,
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "og_features": stringify_path(args.og_features),
            "synthetic_3_features": stringify_path(args.synthetic_3_features),
            "synthetic_6_features": stringify_path(args.synthetic_6_features),
            "fold_assignments": stringify_path(args.fold_assignments),
        },
        "thresholds": [float(value) for value in args.thresholds],
        "model_ids": args.model_ids,
        "strategies": args.strategies,
        "eval_scopes": args.eval_scopes,
        "seed": args.seed,
        "target_columns": target_cols,
        "feature_count": len(feature_cols),
        "imbalanced_learn_version": imblearn.__version__,
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
        "validation": validation,
        "outputs": {
            "runs_csv": stringify_path(runs_path),
            "summary_csv": stringify_path(summary_path),
            "balanced_winners_csv": stringify_path(winners_path),
            "per_label_csv": stringify_path(per_label_path),
            "predictions_csv": stringify_path(predictions_path),
            "label_distribution_csv": stringify_path(labels_path),
            "summary_md": stringify_path(report_md_path),
            "report_json": stringify_path(report_json_path),
        },
        "note": (
            "Primary evaluation scope is heldout_og_rows. all_rows_diagnostic is secondary because synthetic rows "
            "dominate mixed row counts. Resampling is fold-local; the hybrid strategy balances the multilabel "
            "training frame, while SMOTE is fitted separately for each binary label. The source-grouped manifest "
            "is validated before training."
        ),
    }
    report_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote report: {report_md_path}")


if __name__ == "__main__":
    main()
