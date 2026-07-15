# Date Created: 2026-07-15

"""Generate the complete Phase 3 rule-based source-grouped baseline set.

This supplemental diagnostic leaves the accepted five-model benchmark and its
original three-baseline receipt untouched.  Every score is generated out of
fold from the frozen source-grouped manifest.  Random baselines are seeded and
therefore exactly reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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

from src.training.source_grouped_folds import OFFICIAL_LABELS, SOURCE_GROUP_COLUMN
from src.utils.paths import PROJECT_ROOT


FIXED_THRESHOLD = 0.3
DEFAULT_SEED = 42
BASELINE_ORDER = (
    "all_negative",
    "all_positive",
    "always_common_labels",
    "train_fold_prevalence",
    "uniform_random",
    "stratified_random",
)
COMMON_LABELS = ("joy", "trust", "anticipation")
BASELINE_DESCRIPTIONS = {
    "all_negative": "Assign probability 0.0 to every label and perfume.",
    "all_positive": "Assign probability 1.0 to every label and perfume.",
    "always_common_labels": (
        "Assign probability 1.0 to joy, trust, and anticipation and 0.0 to the "
        "four rarer labels."
    ),
    "train_fold_prevalence": (
        "Assign each held-out row the label prevalence calculated only from its "
        "complementary outer-training folds."
    ),
    "uniform_random": (
        "Assign independent deterministic-seed Uniform(0,1) scores to every held-out "
        "perfume and label."
    ),
    "stratified_random": (
        "Draw independent deterministic-seed binary scores using label prevalence "
        "estimated only from the complementary outer-training folds."
    ),
}

DEFAULT_FEATURE_LABEL_CSV = (
    PROJECT_ROOT
    / "outputs/gemini_browser_og784/analysis_out/gemini_browser_og784_feature_labels.csv"
)
DEFAULT_FOLD_MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/tables/fold_assignments.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/source_grouped_extended_baselines_v1"

ARTIFACT_FILENAMES = {
    "summary": "extended_baseline_summary.csv",
    "per_label": "extended_baseline_per_label.csv",
    "fold_diagnostics": "extended_baseline_fold_diagnostics.csv",
    "oof_predictions": "extended_baseline_oof_predictions.csv",
    "report": "REPORT.json",
    "checksums": "SHA256SUMS",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_roc_auc(truth: np.ndarray, probability: np.ndarray) -> float | None:
    if np.unique(truth).size < 2:
        return None
    return float(roc_auc_score(truth, probability))


def _safe_average_precision(
    truth: np.ndarray, probability: np.ndarray
) -> float | None:
    if int(truth.sum()) == 0:
        return None
    return float(average_precision_score(truth, probability))


def _mean_defined(values: Iterable[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def _validate_inputs(
    feature_label_csv: Path, fold_manifest_csv: Path
) -> tuple[pd.DataFrame, list[int]]:
    feature = pd.read_csv(feature_label_csv, keep_default_na=False)
    folds = pd.read_csv(fold_manifest_csv, keep_default_na=False)
    required_feature = {"perfume_id", *OFFICIAL_LABELS}
    required_fold = {"perfume_id", "fold_id", SOURCE_GROUP_COLUMN}
    if missing := required_feature - set(feature.columns):
        raise ValueError(f"Feature-label CSV lacks columns: {sorted(missing)}")
    if missing := required_fold - set(folds.columns):
        raise ValueError(f"Fold manifest lacks columns: {sorted(missing)}")

    for frame, context in ((feature, "Feature-label CSV"), (folds, "Fold manifest")):
        frame["perfume_id"] = frame["perfume_id"].astype(str).str.strip()
        if (frame["perfume_id"] == "").any() or frame["perfume_id"].duplicated().any():
            raise ValueError(f"{context} requires unique, non-empty perfume_id values")
    if set(feature["perfume_id"]) != set(folds["perfume_id"]):
        raise ValueError("Feature-label and fold-manifest perfume universes differ")

    fold_numeric = pd.to_numeric(folds["fold_id"], errors="raise")
    if (fold_numeric % 1 != 0).any():
        raise ValueError("Fold manifest contains non-integer fold IDs")
    folds["fold_id"] = fold_numeric.astype(int)
    fold_ids = sorted(int(value) for value in folds["fold_id"].unique())
    if fold_ids != list(range(1, len(fold_ids) + 1)) or len(fold_ids) < 2:
        raise ValueError("Fold IDs must be contiguous and start at 1")

    folds[SOURCE_GROUP_COLUMN] = folds[SOURCE_GROUP_COLUMN].astype(str).str.strip()
    if (folds[SOURCE_GROUP_COLUMN] == "").any():
        raise ValueError("Fold manifest contains an empty source_group_id")
    if int(folds.groupby(SOURCE_GROUP_COLUMN)["fold_id"].nunique().max()) != 1:
        raise ValueError("A source group crosses outer folds")

    for label in OFFICIAL_LABELS:
        numeric = pd.to_numeric(feature[label], errors="raise")
        if not numeric.isin([0, 1]).all():
            raise ValueError(f"Target column is not binary: {label}")
        feature[label] = numeric.astype(int)

    fold_columns = ["perfume_id", "fold_id", SOURCE_GROUP_COLUMN]
    if "source_group_size" in folds.columns:
        fold_columns.append("source_group_size")
    merged = feature.merge(
        folds[fold_columns], on="perfume_id", how="inner", validate="one_to_one"
    )
    return merged.sort_values("perfume_id").reset_index(drop=True), fold_ids


def _label_metrics(
    truth: np.ndarray, probability: np.ndarray, *, threshold: float
) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(int)
    tp = int(((truth == 1) & (prediction == 1)).sum())
    fp = int(((truth == 0) & (prediction == 1)).sum())
    tn = int(((truth == 0) & (prediction == 0)).sum())
    fn = int(((truth == 1) & (prediction == 0)).sum())
    return {
        "n_rows": int(len(truth)),
        "positive_support": int(truth.sum()),
        "negative_support": int((truth == 0).sum()),
        "predicted_positive_count": int(prediction.sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "roc_auc": _safe_roc_auc(truth, probability),
        "average_precision": _safe_average_precision(truth, probability),
    }


def compute_extended_baselines(
    *,
    feature_label_csv: Path,
    fold_manifest_csv: Path,
    seed: int = DEFAULT_SEED,
    threshold: float = FIXED_THRESHOLD,
) -> dict[str, Any]:
    if float(threshold) != FIXED_THRESHOLD:
        raise ValueError(f"This diagnostic is frozen at threshold {FIXED_THRESHOLD}")
    data, fold_ids = _validate_inputs(feature_label_csv, fold_manifest_csv)
    labels = list(OFFICIAL_LABELS)
    truth = data[labels].to_numpy(dtype=int)
    n_rows, n_labels = truth.shape
    probability_by_baseline = {
        baseline: np.full((n_rows, n_labels), np.nan, dtype=float)
        for baseline in BASELINE_ORDER
    }
    common_scores = np.array(
        [1.0 if label in COMMON_LABELS else 0.0 for label in labels], dtype=float
    )
    fold_rows: list[dict[str, Any]] = []

    for fold_id in fold_ids:
        test_mask = data["fold_id"].to_numpy(dtype=int) == fold_id
        train_mask = ~test_mask
        train_truth = truth[train_mask]
        test_truth = truth[test_mask]
        train_prevalence = train_truth.mean(axis=0)
        n_test = int(test_mask.sum())
        uniform_rng = np.random.default_rng(np.random.SeedSequence([seed, fold_id, 1]))
        stratified_rng = np.random.default_rng(np.random.SeedSequence([seed, fold_id, 2]))
        fold_probabilities = {
            "all_negative": np.zeros((n_test, n_labels), dtype=float),
            "all_positive": np.ones((n_test, n_labels), dtype=float),
            "always_common_labels": np.tile(common_scores, (n_test, 1)),
            "train_fold_prevalence": np.tile(train_prevalence, (n_test, 1)),
            "uniform_random": uniform_rng.random((n_test, n_labels)),
            "stratified_random": stratified_rng.binomial(
                1, np.tile(train_prevalence, (n_test, 1))
            ).astype(float),
        }
        for baseline in BASELINE_ORDER:
            fold_matrix = fold_probabilities[baseline]
            probability_by_baseline[baseline][test_mask] = fold_matrix
            for label_index, label in enumerate(labels):
                metrics = _label_metrics(
                    test_truth[:, label_index],
                    fold_matrix[:, label_index],
                    threshold=threshold,
                )
                fold_rows.append(
                    {
                        "baseline": baseline,
                        "fold_id": int(fold_id),
                        "label": label,
                        "train_rows": int(train_mask.sum()),
                        "train_positive_support": int(train_truth[:, label_index].sum()),
                        "train_prevalence": float(train_prevalence[label_index]),
                        "test_rows": n_test,
                        "score_mean": float(fold_matrix[:, label_index].mean()),
                        "score_min": float(fold_matrix[:, label_index].min()),
                        "score_max": float(fold_matrix[:, label_index].max()),
                        **metrics,
                    }
                )

    if any(np.isnan(matrix).any() for matrix in probability_by_baseline.values()):
        raise RuntimeError("At least one baseline OOF score was not assigned")

    summary_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    for baseline in BASELINE_ORDER:
        probability = probability_by_baseline[baseline]
        prediction = (probability >= threshold).astype(int)
        label_rows = []
        for label_index, label in enumerate(labels):
            metrics = _label_metrics(
                truth[:, label_index], probability[:, label_index], threshold=threshold
            )
            row = {"baseline": baseline, "label": label, **metrics}
            per_label_rows.append(row)
            label_rows.append(row)
        summary_rows.append(
            {
                "baseline": baseline,
                "n_rows": n_rows,
                "n_labels": n_labels,
                "seed": int(seed),
                "threshold": float(threshold),
                "macro_f1": float(np.mean([row["f1"] for row in label_rows])),
                "micro_f1": float(f1_score(truth.ravel(), prediction.ravel(), zero_division=0)),
                "hamming_loss": float(hamming_loss(truth, prediction)),
                "macro_roc_auc": _mean_defined(row["roc_auc"] for row in label_rows),
                "macro_average_precision": _mean_defined(
                    row["average_precision"] for row in label_rows
                ),
                "mean_predicted_labels_per_perfume": float(prediction.sum(axis=1).mean()),
            }
        )
        for row_index, data_row in data.iterrows():
            for label_index, label in enumerate(labels):
                oof_rows.append(
                    {
                        "baseline": baseline,
                        "perfume_id": str(data_row["perfume_id"]),
                        "fold_id": int(data_row["fold_id"]),
                        SOURCE_GROUP_COLUMN: str(data_row[SOURCE_GROUP_COLUMN]),
                        "label": label,
                        "y_true": int(truth[row_index, label_index]),
                        "y_probability": float(probability[row_index, label_index]),
                        "y_pred": int(prediction[row_index, label_index]),
                    }
                )

    return {
        "summary": summary_rows,
        "per_label": per_label_rows,
        "fold_diagnostics": fold_rows,
        "oof_predictions": oof_rows,
        "n_rows": n_rows,
        "fold_ids": fold_ids,
        "labels": labels,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    pd.DataFrame(list(rows)).to_csv(
        path, index=False, lineterminator="\n", na_rep="", float_format="%.12g"
    )


def run_extended_baselines(
    *,
    feature_label_csv: Path,
    fold_manifest_csv: Path,
    out_dir: Path,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    feature_label_csv = Path(feature_label_csv)
    fold_manifest_csv = Path(fold_manifest_csv)
    out_dir = Path(out_dir)
    paths = {key: out_dir / name for key, name in ARTIFACT_FILENAMES.items()}
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite baseline artifact: {existing[0]}")

    result = compute_extended_baselines(
        feature_label_csv=feature_label_csv,
        fold_manifest_csv=fold_manifest_csv,
        seed=seed,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ("summary", "per_label", "fold_diagnostics", "oof_predictions"):
        _write_csv(paths[key], result[key])

    report = {
        "schema_version": 1,
        "status": "passed",
        "diagnostic_id": "source_grouped_extended_baselines_v1",
        "accepted_benchmark_modified": False,
        "evaluation_unit": "perfume_id",
        "split_design": "frozen five-fold exact-source-grouped OOF",
        "seed": int(seed),
        "threshold": FIXED_THRESHOLD,
        "baselines": [
            {"baseline": baseline, "description": BASELINE_DESCRIPTIONS[baseline]}
            for baseline in BASELINE_ORDER
        ],
        "inputs": {
            "feature_label_csv": str(feature_label_csv),
            "feature_label_csv_sha256": sha256_file(feature_label_csv),
            "fold_manifest_csv": str(fold_manifest_csv),
            "fold_manifest_csv_sha256": sha256_file(fold_manifest_csv),
        },
        "summary": result["summary"],
        "artifacts": {
            paths[key].name: {"sha256": sha256_file(paths[key])}
            for key in ("summary", "per_label", "fold_diagnostics", "oof_predictions")
        },
    }
    paths["report"].write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    checksum_keys = ("summary", "per_label", "fold_diagnostics", "oof_predictions", "report")
    paths["checksums"].write_text(
        "".join(f"{sha256_file(paths[key])}  {paths[key].name}\n" for key in checksum_keys),
        encoding="utf-8",
    )
    return {"paths": paths, "report": report, "summary": result["summary"]}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-label-csv", type=Path, default=DEFAULT_FEATURE_LABEL_CSV)
    parser.add_argument("--fold-manifest-csv", type=Path, default=DEFAULT_FOLD_MANIFEST_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_extended_baselines(
        feature_label_csv=args.feature_label_csv,
        fold_manifest_csv=args.fold_manifest_csv,
        out_dir=args.out_dir,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print("Extended source-grouped baselines complete")
    for key, path in result["paths"].items():
        print(f"- {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
