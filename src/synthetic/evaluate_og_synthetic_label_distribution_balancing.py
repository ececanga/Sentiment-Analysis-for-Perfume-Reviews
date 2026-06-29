# Date Created: 2026-06-24
# Date Last Modified: 2026-06-24

"""Balance multi-label emotion counts across labels and evaluate YouTube folds."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.synthetic.evaluate_og_synthetic_balancing_ablation import (
    DEFAULT_FOLD_ASSIGNMENTS,
    DEFAULT_OG_FEATURES,
    DEFAULT_SYNTHETIC_3,
    DEFAULT_SYNTHETIC_6,
    build_data_setups,
    validate_og_frame,
    validate_synthetic_frame,
)
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
)
from src.synthetic.evaluate_synthetic_only_balancing_ablation import (
    build_unbalanced_logistic,
    build_unbalanced_xgboost,
)
from src.training.model_benchmark_runner import BenchmarkModelSpec
from src.utils.paths import PROJECT_ROOT


DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "og_synthetic_label_distribution_balancing_threshold03"


def model_variants() -> list[dict[str, Any]]:
    return [
        {
            "model_id": "logistic_regression",
            "model_family": "Logistic Regression",
            "spec": BenchmarkModelSpec(
                model_id="logistic_regression",
                model_family="Logistic Regression",
                build_model=build_unbalanced_logistic,
                model_params={},
                package_names=(),
            ),
        },
        {
            "model_id": "xgboost",
            "model_family": "XGBoost",
            "spec": BenchmarkModelSpec(
                model_id="xgboost",
                model_family="XGBoost",
                build_model=build_unbalanced_xgboost,
                model_params={},
                package_names=("xgboost",),
            ),
        },
    ]


def balance_methods() -> list[dict[str, str]]:
    return [
        {
            "method": "unbalanced",
            "method_label": "baseline",
            "method_detail": "original fold training rows",
        },
        {
            "method": "label_oversampling",
            "method_label": "label oversampling",
            "method_detail": "duplicate rows with under-target labels toward fold median label count",
        },
        {
            "method": "label_undersampling",
            "method_label": "label undersampling",
            "method_detail": "remove rows with over-target labels toward fold median label count",
        },
        {
            "method": "label_hybrid",
            "method_label": "label hybrid",
            "method_detail": "undersample over-target labels, then oversample under-target labels",
        },
        {
            "method": "label_smote",
            "method_label": "label SMOTE-style",
            "method_detail": "interpolate feature rows for under-target labels toward fold median label count",
        },
    ]


def label_counts(df: pd.DataFrame, target_cols: list[str]) -> pd.Series:
    return df[target_cols].sum().astype(int)


def balance_target(counts: pd.Series) -> int:
    return max(1, int(round(float(np.median(counts.to_numpy(dtype=float))))))


def balance_stats(df: pd.DataFrame, target_cols: list[str], *, prefix: str) -> dict[str, Any]:
    counts = label_counts(df, target_cols)
    min_count = int(counts.min())
    max_count = int(counts.max())
    return {
        f"{prefix}_rows": int(len(df)),
        f"{prefix}_label_min": min_count,
        f"{prefix}_label_max": max_count,
        f"{prefix}_label_median": float(np.median(counts.to_numpy(dtype=float))),
        f"{prefix}_label_sd": float(counts.std(ddof=0)),
        f"{prefix}_label_max_min_ratio": float(max_count / min_count) if min_count else float("inf"),
        **{f"{prefix}_count_{label}": int(counts[label]) for label in target_cols},
    }


def metric_or_nan_with_kwargs(fn: Any, *args: Any, **kwargs: Any) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def row_gain(row: pd.Series, deficits: pd.Series, excesses: pd.Series, target_cols: list[str]) -> float:
    labels = row[target_cols].astype(int)
    rare_gain = float((labels * deficits.clip(lower=0)).sum())
    common_penalty = float((labels * excesses.clip(lower=0)).sum())
    return rare_gain - 0.25 * common_penalty


def oversample_to_target(
    df: pd.DataFrame,
    target_cols: list[str],
    *,
    target: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    base = df.reset_index(drop=True).copy()
    counts = label_counts(base, target_cols)
    max_additions = max(0, target * len(target_cols) * 4)
    additions: list[pd.Series] = []
    for _ in range(max_additions):
        deficits = target - counts
        if not (deficits > 0).any():
            break
        excesses = counts - target
        needed = deficits[deficits > 0].sort_values(ascending=False)
        label = str(needed.index[0])
        candidates = base[base[label].astype(int).eq(1)]
        if candidates.empty:
            break
        sample = candidates.sample(
            n=min(100, len(candidates)),
            random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        scored = sample.apply(row_gain, axis=1, deficits=deficits, excesses=excesses, target_cols=target_cols)
        chosen = sample.loc[scored.idxmax()].copy()
        additions.append(chosen)
        counts = counts + chosen[target_cols].astype(int)
    if not additions:
        return base
    return pd.concat([base, pd.DataFrame(additions)], ignore_index=True)


def undersample_to_target(df: pd.DataFrame, target_cols: list[str], *, target: int) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    counts = label_counts(out, target_cols)
    excesses = counts - target
    if not (excesses > 0).any():
        return out
    label_matrix = out[target_cols].astype(int)
    scores = label_matrix.apply(lambda row: float((row * excesses.clip(lower=0)).sum()), axis=1)
    candidate_indices = scores.sort_values(ascending=False).index.tolist()
    keep = np.ones(len(out), dtype=bool)
    for idx in candidate_indices:
        if scores.loc[idx] <= 0:
            break
        row_labels = label_matrix.loc[idx]
        positive_labels = row_labels.index[row_labels.eq(1)]
        if len(positive_labels) == 0:
            keep[idx] = False
            continue
        if ((counts[positive_labels] - 1) >= target).all():
            keep[idx] = False
            counts[positive_labels] = counts[positive_labels] - 1
        if not ((counts - target) > 0).any():
            break
    return out.loc[keep].reset_index(drop=True)


def smote_style_to_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    *,
    target: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    base = df.reset_index(drop=True).copy()
    counts = label_counts(base, target_cols)
    synthetic_rows: list[pd.Series] = []
    max_additions = max(0, target * len(target_cols) * 4)
    for addition_idx in range(max_additions):
        deficits = target - counts
        if not (deficits > 0).any():
            break
        label = str(deficits[deficits > 0].sort_values(ascending=False).index[0])
        candidates = base[base[label].astype(int).eq(1)].reset_index(drop=True)
        if len(candidates) < 2:
            return oversample_to_target(base, target_cols, target=target, rng=rng)

        seed_idx = int(rng.integers(0, len(candidates)))
        seed = candidates.iloc[seed_idx]
        neighbor_pool = candidates.drop(index=seed_idx).reset_index(drop=True)
        neighbor = neighbor_pool.iloc[int(rng.integers(0, len(neighbor_pool)))]
        alpha = float(rng.random())

        new_row = seed.copy()
        seed_features = seed[feature_cols].astype(float)
        neighbor_features = neighbor[feature_cols].astype(float)
        new_row[feature_cols] = seed_features + alpha * (neighbor_features - seed_features)
        seed_labels = seed[target_cols].astype(int)
        neighbor_labels = neighbor[target_cols].astype(int)
        new_labels = (seed_labels & neighbor_labels).astype(int)
        new_labels[label] = 1
        new_row[target_cols] = new_labels
        new_row["synthetic_review_id"] = f"label-smote-{addition_idx:06d}"
        new_row["perfume_id"] = f"label-smote-{addition_idx:06d}"
        new_row["Perfume"] = "label balance smote"
        new_row["review_index"] = -1
        new_row["source_type"] = "label_balance_smote"
        new_row["row_level"] = "review"
        synthetic_rows.append(new_row)
        counts = counts + new_row[target_cols].astype(int)
    if not synthetic_rows:
        return base
    return pd.concat([base, pd.DataFrame(synthetic_rows)], ignore_index=True)


def balance_training_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    *,
    method: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    before = df.reset_index(drop=True).copy()
    counts = label_counts(before, target_cols)
    target = balance_target(counts)
    rng = np.random.default_rng(seed)
    if method == "unbalanced":
        after = before
    elif method == "label_oversampling":
        after = oversample_to_target(before, target_cols, target=target, rng=rng)
    elif method == "label_undersampling":
        after = undersample_to_target(before, target_cols, target=target)
    elif method == "label_hybrid":
        after = oversample_to_target(
            undersample_to_target(before, target_cols, target=target),
            target_cols,
            target=target,
            rng=rng,
        )
    elif method == "label_smote":
        after = smote_style_to_target(before, feature_cols, target_cols, target=target, rng=rng)
    else:
        raise ValueError(f"Unknown label balancing method: {method}")

    report = {
        "label_balance_target": int(target),
        **balance_stats(before, target_cols, prefix="before"),
        **balance_stats(after, target_cols, prefix="after"),
    }
    return after, report


def evaluate_fixed_folds_with_label_balancing(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    model_spec: BenchmarkModelSpec,
    fold_df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    threshold: float,
    seed: int,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    fold_lookup = fold_df.set_index("perfume_id")["fold_id"].astype(int).to_dict()
    predictions = eval_df[["synthetic_review_id", "perfume_id", "Perfume", "review_index", "source_type"]].copy()
    for target_col in target_cols:
        predictions[f"prob_{target_col}"] = np.nan
        predictions[f"pred_{target_col}"] = 0
    predictions["fold_id"] = predictions["perfume_id"].astype(str).map(fold_lookup).astype(int)

    fold_reports: list[dict[str, Any]] = []
    fold_status: dict[str, list[str]] = {target: [] for target in target_cols}
    for fold_id in sorted(fold_df["fold_id"].astype(int).unique()):
        test_ids = set(fold_df.loc[fold_df["fold_id"].astype(int).eq(fold_id), "perfume_id"].astype(str))
        train_mask = ~train_df["perfume_id"].astype(str).isin(test_ids)
        eval_mask = eval_df["perfume_id"].astype(str).isin(test_ids)
        fold_train = train_df.loc[train_mask].reset_index(drop=True)
        balanced_train, balance_report = balance_training_frame(
            fold_train,
            feature_cols,
            target_cols,
            method=method,
            seed=seed + fold_id * 1009,
        )
        balance_report["fold_id"] = int(fold_id)
        fold_reports.append(balance_report)
        X_train = balanced_train[feature_cols]
        X_eval = eval_df.loc[eval_mask, feature_cols]
        for target_col in target_cols:
            y_train = balanced_train[target_col].astype(int)
            if y_train.nunique() < 2:
                fallback = float(y_train.mean())
                predictions.loc[eval_mask, f"prob_{target_col}"] = fallback
                predictions.loc[eval_mask, f"pred_{target_col}"] = int(fallback >= threshold)
                fold_status[target_col].append(f"fold_{fold_id}:single_class_train_fallback")
                continue
            model = model_spec.build_model(seed, y_train.to_numpy())
            model.fit(X_train, y_train)
            probs = positive_class_probability(model, X_eval)
            predictions.loc[eval_mask, f"prob_{target_col}"] = probs
            predictions.loc[eval_mask, f"pred_{target_col}"] = (probs >= threshold).astype(int)
            fold_status[target_col].append(f"fold_{fold_id}:trained:{method}:n_train_{len(y_train)}")

    per_label_rows: list[dict[str, Any]] = []
    pred_columns: list[np.ndarray] = []
    for target_col in target_cols:
        y_true = eval_df[target_col].astype(int).to_numpy()
        y_prob = predictions[f"prob_{target_col}"].astype(float).to_numpy()
        y_pred = predictions[f"pred_{target_col}"].astype(int).to_numpy()
        pred_columns.append(y_pred)
        per_label_rows.append(
            {
                "label": target_col,
                "support_pos": int(y_true.sum()),
                "support_neg": int(len(y_true) - y_true.sum()),
                "positive_rate": float(y_true.mean()),
                "predicted_positive_count": int(y_pred.sum()),
                "predicted_positive_rate": float(y_pred.mean()),
                "precision": metric_or_nan_with_kwargs(precision_score, y_true, y_pred, zero_division=0),
                "recall": metric_or_nan_with_kwargs(recall_score, y_true, y_pred, zero_division=0),
                "f1": metric_or_nan(f1_score, y_true, y_pred),
                "roc_auc": metric_or_nan(roc_auc_score, y_true, y_prob),
                "pr_auc": metric_or_nan(average_precision_score, y_true, y_prob),
                "training_status": ";".join(fold_status[target_col]),
            }
        )

    per_label = pd.DataFrame(per_label_rows)
    y_true_matrix = eval_df[target_cols].astype(int).to_numpy()
    y_pred_matrix = np.column_stack(pred_columns)
    metrics = {
        "macro_f1": finite_mean(per_label["f1"].astype(float).tolist()),
        "macro_precision": finite_mean(per_label["precision"].astype(float).tolist()),
        "macro_recall": finite_mean(per_label["recall"].astype(float).tolist()),
        "macro_roc_auc": finite_mean(per_label["roc_auc"].astype(float).tolist()),
        "macro_pr_auc": finite_mean(per_label["pr_auc"].astype(float).tolist()),
        "micro_f1": metric_or_nan(f1_score, y_true_matrix.ravel(), y_pred_matrix.ravel()),
        "hamming_loss": metric_or_nan(hamming_loss, y_true_matrix, y_pred_matrix),
        "label_cardinality": float(y_true_matrix.sum(axis=1).mean()),
        "predicted_label_cardinality": float(y_pred_matrix.sum(axis=1).mean()),
    }
    return predictions, per_label, metrics, pd.DataFrame(fold_reports)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    summary = runs.copy()
    metric_cols = ["macro_f1", "macro_precision", "macro_recall", "macro_roc_auc", "macro_pr_auc"]
    for metric in metric_cols:
        summary[f"delta_{metric}_vs_unbalanced"] = 0.0
    group_cols = ["data_setup", "threshold", "model_family"]
    for _, group in summary.groupby(group_cols, sort=False):
        baseline = group[group["method"].eq("unbalanced")]
        if len(baseline) != 1:
            raise ValueError(f"Expected one unbalanced baseline for {group.iloc[0][group_cols].to_dict()}.")
        base = baseline.iloc[0]
        mask = (
            summary["data_setup"].eq(base["data_setup"])
            & summary["threshold"].eq(base["threshold"])
            & summary["model_family"].eq(base["model_family"])
        )
        for metric in metric_cols:
            summary.loc[mask, f"delta_{metric}_vs_unbalanced"] = (
                summary.loc[mask, metric].astype(float) - float(base[metric])
            )
    return summary.sort_values(["threshold", "data_setup", "model_family", "macro_f1"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fold-local multi-label distribution balancing.")
    parser.add_argument("--og-features", type=Path, default=DEFAULT_OG_FEATURES)
    parser.add_argument("--synthetic-3-features", type=Path, default=DEFAULT_SYNTHETIC_3)
    parser.add_argument("--synthetic-6-features", type=Path, default=DEFAULT_SYNTHETIC_6)
    parser.add_argument("--fold-assignments", type=Path, default=DEFAULT_FOLD_ASSIGNMENTS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    target_cols = DEFAULT_TARGETS
    og_df = coerce_targets(load_og_frame(args.og_features, target_cols), target_cols)
    synthetic_3 = coerce_targets(load_synthetic_frame(args.synthetic_3_features, target_cols), target_cols)
    synthetic_6 = coerce_targets(load_synthetic_frame(args.synthetic_6_features, target_cols), target_cols)
    fold_df = pd.read_csv(args.fold_assignments)
    fold_df["perfume_id"] = fold_df["perfume_id"].astype(str)
    validation = {
        "og": validate_og_frame(og_df, fold_df, target_cols),
        "synthetic_3": validate_synthetic_frame(synthetic_3, reviews_per_perfume=3, target_cols=target_cols),
        "synthetic_6": validate_synthetic_frame(synthetic_6, reviews_per_perfume=6, target_cols=target_cols),
    }

    feature_cols = select_feature_columns(og_df, synthetic_3, synthetic_6)
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

    run_rows: list[dict[str, Any]] = []
    per_label_frames: list[pd.DataFrame] = []
    balance_frames: list[pd.DataFrame] = []
    for setup in setups:
        for model_variant in model_variants():
            for method in balance_methods():
                _, per_label, metrics, balance_report = evaluate_fixed_folds_with_label_balancing(
                    setup["train_df"],
                    og_df[keep_cols],
                    model_spec=model_variant["spec"],
                    fold_df=fold_df,
                    feature_cols=feature_cols,
                    target_cols=target_cols,
                    threshold=args.threshold,
                    seed=args.seed,
                    method=method["method"],
                )
                row = {
                    "data_setup": setup["data_setup"],
                    "data_label": setup["data_label"],
                    "threshold": float(args.threshold),
                    "model_id": model_variant["model_id"],
                    "model_family": model_variant["model_family"],
                    **method,
                    "train_rows_original": int(len(setup["train_df"])),
                    "og_train_rows": 784,
                    "synthetic_rows": int(setup["synthetic_rows"]),
                    "synthetic_reviews_per_perfume": int(setup["synthetic_reviews_per_perfume"]),
                    "eval_rows": int(len(og_df)),
                    "feature_count": int(len(feature_cols)),
                    **metrics,
                }
                run_rows.append(row)
                context = {key: row[key] for key in ["data_setup", "data_label", "threshold", "model_id", "model_family", "method", "method_label"]}
                per_label_frames.append(per_label.assign(**context))
                balance_frames.append(balance_report.assign(**context))
                print(
                    f"{setup['data_setup']} {model_variant['model_id']} {method['method']}: "
                    f"macro_f1={metrics['macro_f1']:.4f} pred_labels={metrics['predicted_label_cardinality']:.2f}"
                )

    runs = pd.DataFrame(run_rows)
    summary = summarize_runs(runs)
    per_label_out = pd.concat(per_label_frames, ignore_index=True)
    balance_out = pd.concat(balance_frames, ignore_index=True)

    runs_path = args.out_root / "label_distribution_balancing_runs.csv"
    summary_path = args.out_root / "label_distribution_balancing_summary.csv"
    per_label_path = args.out_root / "label_distribution_balancing_per_label.csv"
    balance_path = args.out_root / "label_distribution_balancing_fold_balance.csv"
    report_path = args.out_root / "label_distribution_balancing_report.json"
    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    per_label_out.to_csv(per_label_path, index=False)
    balance_out.to_csv(balance_path, index=False)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "og_features": stringify_path(args.og_features),
            "synthetic_3_features": stringify_path(args.synthetic_3_features),
            "synthetic_6_features": stringify_path(args.synthetic_6_features),
            "fold_assignments": stringify_path(args.fold_assignments),
        },
        "threshold": float(args.threshold),
        "seed": int(args.seed),
        "target_columns": target_cols,
        "feature_count": len(feature_cols),
        "balancing_note": "Emotion-label counts are balanced fold-locally across labels toward the training-fold median count. YouTube test folds are unchanged.",
        "validation": validation,
        "outputs": {
            "runs_csv": stringify_path(runs_path),
            "summary_csv": stringify_path(summary_path),
            "per_label_csv": stringify_path(per_label_path),
            "fold_balance_csv": stringify_path(balance_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
