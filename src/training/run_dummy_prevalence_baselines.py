# Date Created: 2026-07-13

"""Generate deterministic dummy and training-prevalence OOF diagnostics.

The diagnostic is intentionally separate from the five-model benchmark.  It
uses the already frozen outer-fold manifest and never fits on held-out labels.
For every held-out fold, the prevalence baseline is the label prevalence in
the complementary training folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.training.source_grouped_folds import OFFICIAL_LABELS, SOURCE_GROUP_COLUMN
from src.utils.paths import PROJECT_ROOT


FIXED_THRESHOLD = 0.3
BASELINE_ORDER = ("all_negative", "all_positive", "train_fold_prevalence")
BASELINE_DESCRIPTIONS = {
    "all_negative": "Assign probability 0.0 to every label and perfume.",
    "all_positive": "Assign probability 1.0 to every label and perfume.",
    "train_fold_prevalence": (
        "For each held-out fold and label, assign the positive prevalence calculated only "
        "from the complementary training folds."
    ),
}

DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "outputs/gemini_browser_og784/analysis_out/gemini_browser_og784_feature_labels.csv"
)
DEFAULT_FOLD_MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/tables/fold_assignments.csv"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/dummy_prevalence_baselines"
)

ARTIFACT_FILENAMES = {
    "oof_predictions_csv": "dummy_prevalence_oof_predictions.csv",
    "summary_csv": "dummy_prevalence_summary.csv",
    "per_label_csv": "dummy_prevalence_per_label.csv",
    "fold_diagnostics_csv": "dummy_prevalence_fold_diagnostics.csv",
    "report_json": "dummy_prevalence_report.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    # AP has no positive retrieval target when positive support is zero.
    if int(np.sum(y_true)) == 0:
        return None
    return float(average_precision_score(y_true, y_prob))


def _mean_defined(values: Iterable[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def _label_metrics(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    positive_support = int(y_true.sum())
    n_rows = int(len(y_true))
    return {
        "n_rows": n_rows,
        "positive_support": positive_support,
        "negative_support": int(n_rows - positive_support),
        "prevalence": float(positive_support / n_rows) if n_rows else None,
        "predicted_positive_count": int(y_pred.sum()),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_prob),
        "average_precision": _safe_average_precision(y_true, y_prob),
    }


def _validate_inputs(
    *,
    feature_label_csv: Path,
    fold_manifest_csv: Path,
) -> tuple[pd.DataFrame, List[int]]:
    if not feature_label_csv.is_file():
        raise FileNotFoundError(f"Feature-label CSV not found: {feature_label_csv}")
    if not fold_manifest_csv.is_file():
        raise FileNotFoundError(f"Fold manifest CSV not found: {fold_manifest_csv}")

    feature_df = pd.read_csv(feature_label_csv, keep_default_na=False)
    fold_df = pd.read_csv(fold_manifest_csv, keep_default_na=False)
    missing_targets = sorted(set(OFFICIAL_LABELS) - set(feature_df.columns))
    if missing_targets:
        raise ValueError(f"Feature-label CSV lacks official labels: {missing_targets}")
    if "perfume_id" not in feature_df.columns:
        raise ValueError("Feature-label CSV lacks perfume_id")
    if not {"perfume_id", "fold_id"}.issubset(fold_df.columns):
        raise ValueError("Fold manifest must contain perfume_id and fold_id")

    feature_df = feature_df.copy()
    fold_df = fold_df.copy()
    for frame, context in ((feature_df, "Feature-label CSV"), (fold_df, "Fold manifest")):
        frame["perfume_id"] = frame["perfume_id"].astype(str).str.strip()
        if (frame["perfume_id"] == "").any() or frame["perfume_id"].duplicated().any():
            raise ValueError(f"{context} must contain unique, non-empty perfume_id values")
    if set(feature_df["perfume_id"]) != set(fold_df["perfume_id"]):
        raise ValueError("Feature-label and fold-manifest perfume universes differ")

    fold_numeric = pd.to_numeric(fold_df["fold_id"], errors="raise")
    if (fold_numeric % 1 != 0).any():
        raise ValueError("Fold manifest contains non-integer fold IDs")
    fold_df["fold_id"] = fold_numeric.astype(int)
    fold_ids = sorted(int(value) for value in fold_df["fold_id"].unique())
    if len(fold_ids) < 2 or fold_ids != list(range(1, len(fold_ids) + 1)):
        raise ValueError("Fold manifest must contain contiguous fold IDs starting at 1")

    if SOURCE_GROUP_COLUMN in fold_df.columns:
        fold_df[SOURCE_GROUP_COLUMN] = fold_df[SOURCE_GROUP_COLUMN].astype(str).str.strip()
        if (fold_df[SOURCE_GROUP_COLUMN] == "").any():
            raise ValueError("Fold manifest contains empty source_group_id values")
        if int(fold_df.groupby(SOURCE_GROUP_COLUMN)["fold_id"].nunique().max()) != 1:
            raise ValueError("A source group crosses outer folds")

    for label in OFFICIAL_LABELS:
        values = pd.to_numeric(feature_df[label], errors="raise")
        if not values.isin([0, 1]).all():
            raise ValueError(f"Target column must contain only binary 0/1 values: {label}")
        feature_df[label] = values.astype(int)

    fold_columns = ["perfume_id", "fold_id"]
    if SOURCE_GROUP_COLUMN in fold_df.columns:
        fold_columns.append(SOURCE_GROUP_COLUMN)
    merged = feature_df.merge(
        fold_df[fold_columns],
        on="perfume_id",
        how="inner",
        validate="one_to_one",
    )
    return merged.sort_values("perfume_id").reset_index(drop=True), fold_ids


def compute_dummy_prevalence_diagnostics(
    *,
    feature_label_csv: Path,
    fold_manifest_csv: Path,
    threshold: float = FIXED_THRESHOLD,
) -> Dict[str, Any]:
    """Compute deterministic pooled OOF baseline predictions and diagnostics."""

    if float(threshold) != FIXED_THRESHOLD:
        raise ValueError(f"This diagnostic is frozen at threshold {FIXED_THRESHOLD}")
    data, fold_ids = _validate_inputs(
        feature_label_csv=Path(feature_label_csv),
        fold_manifest_csv=Path(fold_manifest_csv),
    )
    target_columns = list(OFFICIAL_LABELS)
    y_true = data[target_columns].to_numpy(dtype=int)
    n_rows, n_labels = y_true.shape
    probabilities = {
        baseline: np.full((n_rows, n_labels), np.nan, dtype=float)
        for baseline in BASELINE_ORDER
    }
    fold_diagnostics: List[Dict[str, Any]] = []

    for fold_id in fold_ids:
        test_mask = data["fold_id"].to_numpy(dtype=int) == fold_id
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            raise ValueError(f"Fold {fold_id} has an empty train or held-out partition")
        train_y = y_true[train_mask]
        test_y = y_true[test_mask]
        train_prevalence = train_y.mean(axis=0)
        fold_probabilities = {
            "all_negative": np.zeros(n_labels, dtype=float),
            "all_positive": np.ones(n_labels, dtype=float),
            "train_fold_prevalence": train_prevalence,
        }
        for baseline in BASELINE_ORDER:
            fold_prob = fold_probabilities[baseline]
            probabilities[baseline][test_mask] = fold_prob
            for label_index, label in enumerate(target_columns):
                metrics = _label_metrics(
                    y_true=test_y[:, label_index],
                    y_prob=np.full(int(test_mask.sum()), fold_prob[label_index], dtype=float),
                    threshold=threshold,
                )
                train_positive = int(train_y[:, label_index].sum())
                fold_diagnostics.append(
                    {
                        "baseline": baseline,
                        "fold_id": int(fold_id),
                        "label": label,
                        "train_rows": int(train_mask.sum()),
                        "train_positive_support": train_positive,
                        "train_negative_support": int(train_mask.sum() - train_positive),
                        "train_prevalence": float(train_prevalence[label_index]),
                        "test_rows": int(test_mask.sum()),
                        "test_positive_support": metrics["positive_support"],
                        "test_negative_support": metrics["negative_support"],
                        "test_prevalence": metrics["prevalence"],
                        "prediction_probability": float(fold_prob[label_index]),
                        "test_predicted_positive_count": metrics["predicted_positive_count"],
                        "test_f1": metrics["f1"],
                        "test_roc_auc": metrics["roc_auc"],
                        "test_average_precision": metrics["average_precision"],
                    }
                )

    if any(np.isnan(matrix).any() for matrix in probabilities.values()):
        raise RuntimeError("At least one OOF baseline probability was not assigned")

    per_label_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    oof_rows: List[Dict[str, Any]] = []
    display_column = "Perfume" if "Perfume" in data.columns else None

    for baseline in BASELINE_ORDER:
        matrix = probabilities[baseline]
        baseline_label_metrics: List[Dict[str, Any]] = []
        for label_index, label in enumerate(target_columns):
            metrics = _label_metrics(
                y_true=y_true[:, label_index],
                y_prob=matrix[:, label_index],
                threshold=threshold,
            )
            row = {"baseline": baseline, "label": label, **metrics}
            per_label_rows.append(row)
            baseline_label_metrics.append(row)

        summary_rows.append(
            {
                "baseline": baseline,
                "n_rows": int(n_rows),
                "n_labels": int(n_labels),
                "threshold": float(threshold),
                "macro_f1": float(np.mean([row["f1"] for row in baseline_label_metrics])),
                "macro_roc_auc": _mean_defined(
                    row["roc_auc"] for row in baseline_label_metrics
                ),
                "macro_average_precision": _mean_defined(
                    row["average_precision"] for row in baseline_label_metrics
                ),
                "roc_auc_defined_label_count": int(
                    sum(row["roc_auc"] is not None for row in baseline_label_metrics)
                ),
                "average_precision_defined_label_count": int(
                    sum(row["average_precision"] is not None for row in baseline_label_metrics)
                ),
            }
        )

        predictions = (matrix >= threshold).astype(int)
        for row_index, data_row in data.iterrows():
            for label_index, label in enumerate(target_columns):
                oof_row: Dict[str, Any] = {
                    "baseline": baseline,
                    "perfume_id": str(data_row["perfume_id"]),
                    "fold_id": int(data_row["fold_id"]),
                    "label": label,
                    "y_true": int(y_true[row_index, label_index]),
                    "y_probability": float(matrix[row_index, label_index]),
                    "y_pred": int(predictions[row_index, label_index]),
                }
                if display_column is not None:
                    oof_row[display_column] = str(data_row[display_column])
                if SOURCE_GROUP_COLUMN in data.columns:
                    oof_row[SOURCE_GROUP_COLUMN] = str(data_row[SOURCE_GROUP_COLUMN])
                oof_rows.append(oof_row)

    return {
        "threshold": float(threshold),
        "target_columns": target_columns,
        "fold_ids": fold_ids,
        "n_rows": int(n_rows),
        "summary": summary_rows,
        "per_label": per_label_rows,
        "fold_diagnostics": fold_diagnostics,
        "oof_predictions": oof_rows,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    frame = pd.DataFrame(list(rows))
    frame.to_csv(path, index=False, lineterminator="\n", na_rep="", float_format="%.12g")


def run_dummy_prevalence_baselines(
    *,
    feature_label_csv: Path,
    fold_manifest_csv: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Compute and write the complete deterministic baseline artifact set."""

    feature_label_csv = Path(feature_label_csv)
    fold_manifest_csv = Path(fold_manifest_csv)
    out_dir = Path(out_dir)
    artifact_paths = {key: out_dir / filename for key, filename in ARTIFACT_FILENAMES.items()}
    existing = [path for path in artifact_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite baseline artifact: {existing[0]}")

    result = compute_dummy_prevalence_diagnostics(
        feature_label_csv=feature_label_csv,
        fold_manifest_csv=fold_manifest_csv,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(artifact_paths["oof_predictions_csv"], result["oof_predictions"])
    _write_csv(artifact_paths["summary_csv"], result["summary"])
    _write_csv(artifact_paths["per_label_csv"], result["per_label"])
    _write_csv(artifact_paths["fold_diagnostics_csv"], result["fold_diagnostics"])

    csv_hashes = {
        key: sha256_file(path)
        for key, path in artifact_paths.items()
        if key.endswith("_csv")
    }
    report = {
        "schema_version": 1,
        "diagnostic_id": "dummy_prevalence_baselines_source_grouped_v1",
        "deterministic": True,
        "evaluation_unit": "perfume_id",
        "prediction_design": "pooled_out_of_fold",
        "threshold": FIXED_THRESHOLD,
        "metric_names": {
            "classification": "F1",
            "ranking": "ROC-AUC",
            "retrieval": "average precision",
        },
        "undefined_metric_policy": {
            "roc_auc": "null when held data contain fewer than two target classes",
            "average_precision": "null when positive support is zero",
            "macro_aggregation": "mean over labels for which the metric is defined",
        },
        "inputs": {
            "feature_label_csv": str(feature_label_csv),
            "feature_label_csv_sha256": sha256_file(feature_label_csv),
            "fold_manifest_csv": str(fold_manifest_csv),
            "fold_manifest_csv_sha256": sha256_file(fold_manifest_csv),
        },
        "target_columns": result["target_columns"],
        "fold_ids": result["fold_ids"],
        "n_rows": result["n_rows"],
        "baselines": [
            {"baseline": baseline, "description": BASELINE_DESCRIPTIONS[baseline]}
            for baseline in BASELINE_ORDER
        ],
        "summary": result["summary"],
        "per_label": result["per_label"],
        "fold_diagnostics": result["fold_diagnostics"],
        "artifacts": {
            key: {"filename": path.name, "sha256": csv_hashes.get(key)}
            for key, path in artifact_paths.items()
            if key != "report_json"
        },
    }
    artifact_paths["report_json"].write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "paths": artifact_paths,
        "summary": result["summary"],
        "report": report,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dummy and train-fold prevalence OOF baseline diagnostics.",
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--fold-manifest-csv", type=Path, default=DEFAULT_FOLD_MANIFEST_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace this module's existing output artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_dummy_prevalence_baselines(
            feature_label_csv=args.input_csv,
            fold_manifest_csv=args.fold_manifest_csv,
            out_dir=args.out_dir,
            overwrite=args.overwrite,
        )
    except Exception as exc:  # noqa: BLE001 - CLI reporting boundary
        print(f"Baseline diagnostic failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("Dummy/prevalence baseline diagnostic complete")
    for key, path in result["paths"].items():
        print(f"- {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
