# Date Created: 2026-07-13
# Date Last Modified: 2026-07-13

"""Validate the repaired XGBoost review-count, threshold, and balancing runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.paths import PROJECT_ROOT

LABELS = ["joy", "sadness", "fear", "trust", "surprise", "anticipation", "disgust"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def macro_metrics(frame: pd.DataFrame, labels: pd.DataFrame) -> dict[str, float]:
    aligned = frame[["perfume_id"]].merge(labels, on="perfume_id", how="left", validate="one_to_one")
    if aligned[LABELS].isna().any().any():
        raise ValueError("Prediction frame contains perfume IDs without target labels.")
    f1_values: list[float] = []
    roc_values: list[float] = []
    pr_values: list[float] = []
    for label in LABELS:
        y_true = aligned[label].astype(int).to_numpy()
        y_pred = frame[f"pred_{label}"].astype(int).to_numpy()
        y_prob = frame[f"prob_{label}"].astype(float).to_numpy()
        f1_values.append(float(f1_score(y_true, y_pred)))
        roc_values.append(float(roc_auc_score(y_true, y_prob)))
        pr_values.append(float(average_precision_score(y_true, y_prob)))
    return {
        "macro_f1": float(np.mean(f1_values)),
        "macro_roc_auc": float(np.mean(roc_values)),
        "macro_pr_auc": float(np.mean(pr_values)),
    }


def assert_close(actual: float, expected: float, *, context: str, tolerance: float = 1e-9) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0.0):
        raise ValueError(f"{context}: recomputed {actual} != recorded {expected}.")


def validate_predictions(
    frame: pd.DataFrame,
    *,
    manifest: pd.DataFrame,
    threshold: float,
    context: str,
) -> dict[str, Any]:
    if len(frame) != 784 or frame["perfume_id"].nunique() != 784:
        raise ValueError(f"{context}: expected 784 unique held-out perfume predictions.")
    if frame[[f"prob_{label}" for label in LABELS]].isna().any().any():
        raise ValueError(f"{context}: missing probabilities.")
    expected_folds = manifest.set_index("perfume_id")["fold_id"].astype(int)
    observed_folds = frame.set_index("perfume_id")["fold_id"].astype(int)
    if not observed_folds.sort_index().equals(expected_folds.sort_index()):
        raise ValueError(f"{context}: prediction folds do not match the repaired manifest.")
    for label in LABELS:
        expected_pred = (frame[f"prob_{label}"].astype(float) >= threshold).astype(int)
        if not expected_pred.equals(frame[f"pred_{label}"].astype(int)):
            raise ValueError(f"{context}: pred_{label} does not match threshold {threshold}.")
    return {
        "rows": int(len(frame)),
        "unique_perfumes": int(frame["perfume_id"].nunique()),
        "fold_sizes": {
            str(int(fold_id)): int(count) for fold_id, count in frame.groupby("fold_id").size().items()
        },
    }


def validate_review_count(
    root: Path,
    *,
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    summary = pd.read_csv(root / "synthetic_review_count_sweep_summary.csv")
    augmented = summary.loc[summary["synthetic_reviews"].astype(int) > 0].copy()
    checks: list[dict[str, Any]] = []
    for row in augmented.to_dict("records"):
        predictions = pd.read_csv(PROJECT_ROOT / str(row["predictions_csv"]))
        context = f"review_count={int(row['synthetic_reviews'])}"
        prediction_check = validate_predictions(
            predictions,
            manifest=manifest,
            threshold=float(row.get("threshold", 0.3) or 0.3),
            context=context,
        )
        recomputed = macro_metrics(predictions, labels)
        for metric, value in recomputed.items():
            assert_close(value, float(row[metric]), context=f"{context} {metric}")
        checks.append({"synthetic_reviews": int(row["synthetic_reviews"]), **prediction_check, **recomputed})
    if len(checks) != 7:
        raise ValueError(f"Expected seven augmented review-count conditions, found {len(checks)}.")
    return {"condition_count": len(checks), "conditions": checks}


def validate_threshold_sweep(
    root: Path,
    *,
    manifest: pd.DataFrame,
) -> dict[str, Any]:
    summary = pd.read_csv(root / "tables" / "fixed_threshold_comparison_long.csv")
    expected_thresholds = [0.2, 0.3, 0.5, 0.7]
    if sorted(summary["threshold"].astype(float).tolist()) != expected_thresholds:
        raise ValueError("Threshold sweep does not contain exactly 0.2, 0.3, 0.5, and 0.7.")
    reference_probabilities: pd.DataFrame | None = None
    checks: list[dict[str, Any]] = []
    for row in summary.sort_values("threshold").to_dict("records"):
        threshold = float(row["threshold"])
        predictions = pd.read_csv(PROJECT_ROOT / str(row["out_root"]) / "predictions" / "xgboost_oof.csv")
        validate_predictions(predictions, manifest=manifest, threshold=threshold, context=f"threshold={threshold}")
        true_columns = {f"true_{label}": label for label in LABELS}
        labels = predictions[["perfume_id", *true_columns]].rename(columns=true_columns)
        recomputed = macro_metrics(predictions, labels)
        for metric, value in recomputed.items():
            assert_close(value, float(row[metric]), context=f"threshold={threshold} {metric}", tolerance=1e-6)
        probabilities = predictions[["perfume_id", *[f"prob_{label}" for label in LABELS]]].sort_values(
            "perfume_id"
        ).reset_index(drop=True)
        if reference_probabilities is None:
            reference_probabilities = probabilities
        elif not probabilities.equals(reference_probabilities):
            raise ValueError("XGBoost ranking probabilities changed across fixed-threshold conditions.")
        checks.append({"threshold": threshold, **recomputed})
    return {"condition_count": len(checks), "probabilities_identical_across_thresholds": True, "conditions": checks}


def validate_balancing(
    root: Path,
    *,
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    summary = pd.read_csv(root / "og_synthetic_balancing_ablation_summary.csv")
    predictions = pd.read_csv(root / "og_synthetic_balancing_ablation_predictions.csv")
    group_columns = ["eval_scope", "data_setup", "threshold", "model_id", "strategy"]
    checks: list[dict[str, Any]] = []
    for key, frame in predictions.groupby(group_columns, sort=True):
        identity = dict(zip(group_columns, key))
        threshold = float(identity["threshold"])
        context = " ".join(f"{name}={value}" for name, value in identity.items())
        validate_predictions(frame.reset_index(drop=True), manifest=manifest, threshold=threshold, context=context)
        matched = summary.copy()
        for name, value in identity.items():
            matched = matched.loc[matched[name].eq(value)]
        if len(matched) != 1:
            raise ValueError(f"{context}: expected one matching summary row, found {len(matched)}.")
        recomputed = macro_metrics(frame.reset_index(drop=True), labels)
        row = matched.iloc[0]
        for metric, value in recomputed.items():
            assert_close(value, float(row[metric]), context=f"{context} {metric}")
        checks.append({**identity, **recomputed})
    if len(checks) != 9:
        raise ValueError(f"Expected nine balancing conditions, found {len(checks)}.")
    return {"condition_count": len(checks), "conditions": checks}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "xgboost_source_grouped_experiments_v1",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "gemini_browser_og784_source_grouped_v1"
        / "model_benchmark"
        / "tables"
        / "fold_assignments.csv",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "gemini_browser_og784"
        / "analysis_out"
        / "gemini_browser_og784_feature_labels.csv",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    if manifest.groupby("source_group_id")["fold_id"].nunique().max() != 1:
        raise ValueError("A source group crosses repaired outer folds.")
    labels = pd.read_csv(args.labels)[["perfume_id", *LABELS]].copy()
    report = {
        "status": "passed",
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "source_group_count": int(manifest["source_group_id"].nunique()),
        "cross_fold_source_group_count": 0,
        "review_count": validate_review_count(
            args.experiment_root / "review_count_sweep", manifest=manifest, labels=labels
        ),
        "threshold_sweep": validate_threshold_sweep(
            args.experiment_root / "threshold_sweep", manifest=manifest
        ),
        "balancing": validate_balancing(args.experiment_root / "balancing", manifest=manifest, labels=labels),
    }
    output_path = args.output_json or args.experiment_root / "validation_report.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validation passed: {output_path}")
    print("- Review-count conditions: 7")
    print("- Fixed-threshold conditions: 4")
    print("- Balancing conditions: 9")


if __name__ == "__main__":
    main()
