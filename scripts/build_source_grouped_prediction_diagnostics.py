"""Build per-label confusion and predicted-positive diagnostics from OOF files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.paths import PROJECT_ROOT


LABELS = ["joy", "sadness", "fear", "trust", "surprise", "anticipation", "disgust"]
MODEL_INPUTS = {
    "logistic_regression_weighted": (
        "Logistic Regression (class-weighted)",
        PROJECT_ROOT
        / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/predictions/logistic_regression_oof.csv",
    ),
    "logistic_regression_unweighted": (
        "Logistic Regression (unweighted sensitivity)",
        PROJECT_ROOT
        / "outputs/logistic_regression_unweighted_source_grouped_v1/model_benchmark/predictions/logistic_regression_oof.csv",
    ),
    "tfidf_logistic_regression": (
        "TF-IDF + Logistic Regression (text baseline)",
        PROJECT_ROOT
        / "outputs/tfidf_logistic_regression_source_grouped_v1/tfidf_logistic_regression_oof.csv",
    ),
    "random_forest": (
        "Random Forest",
        PROJECT_ROOT
        / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/predictions/random_forest_oof.csv",
    ),
    "xgboost": (
        "XGBoost",
        PROJECT_ROOT
        / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/predictions/xgboost_oof.csv",
    ),
    "bert": (
        "BERT",
        PROJECT_ROOT
        / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/predictions/bert_oof.csv",
    ),
    "bilstm_attention": (
        "BiLSTM + Attention",
        PROJECT_ROOT
        / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/predictions/bilstm_attention_oof.csv",
    ),
}
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/tables/fold_assignments.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/source_grouped_prediction_diagnostics_v1"


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def calculate_model_diagnostics(
    frame: pd.DataFrame,
    *,
    model_id: str,
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_tp = total_fp = total_tn = total_fn = 0
    predicted_cardinality = np.zeros(len(frame), dtype=int)

    for label in LABELS:
        true = frame[f"true_{label}"].to_numpy(dtype=int)
        pred = frame[f"pred_{label}"].to_numpy(dtype=int)
        if not set(np.unique(true)).issubset({0, 1}) or not set(np.unique(pred)).issubset({0, 1}):
            raise ValueError(f"{model_id}/{label}: truth and predictions must be binary")

        tp = int(np.sum((true == 1) & (pred == 1)))
        fp = int(np.sum((true == 0) & (pred == 1)))
        tn = int(np.sum((true == 0) & (pred == 0)))
        fn = int(np.sum((true == 1) & (pred == 0)))
        actual_positive = int(true.sum())
        predicted_positive = int(pred.sum())
        if tp + fn != actual_positive or tp + fp != predicted_positive:
            raise AssertionError(f"{model_id}/{label}: inconsistent confusion counts")

        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        f1 = safe_ratio(2 * tp, 2 * tp + fp + fn)
        rows.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "label": label,
                "n_perfumes": int(len(frame)),
                "actual_positive_count": actual_positive,
                "actual_positive_rate": safe_ratio(actual_positive, len(frame)),
                "predicted_positive_count": predicted_positive,
                "predicted_positive_rate": safe_ratio(predicted_positive, len(frame)),
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "specificity": safe_ratio(tn, tn + fp),
                "f1": f1,
            }
        )
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
        predicted_cardinality += pred

    summary = {
        "model_id": model_id,
        "model_name": model_name,
        "n_perfumes": int(len(frame)),
        "binary_decisions": int(len(frame) * len(LABELS)),
        "total_true_positive": total_tp,
        "total_false_positive": total_fp,
        "total_true_negative": total_tn,
        "total_false_negative": total_fn,
        "total_predicted_positive": int(total_tp + total_fp),
        "mean_predicted_labels_per_perfume": float(predicted_cardinality.mean()),
        "micro_precision": safe_ratio(total_tp, total_tp + total_fp),
        "micro_recall": safe_ratio(total_tp, total_tp + total_fn),
        "micro_f1": safe_ratio(2 * total_tp, 2 * total_tp + total_fp + total_fn),
    }
    return rows, summary


def validate_against_manifest(frame: pd.DataFrame, manifest: pd.DataFrame, *, model_id: str) -> pd.DataFrame:
    if len(frame) != 784 or frame["perfume_id"].nunique() != 784:
        raise ValueError(f"{model_id}: expected exactly 784 unique OOF perfume rows")
    merged = frame.merge(
        manifest[["perfume_id", "fold_id", "source_group_id"]],
        on="perfume_id",
        how="left",
        suffixes=("", "_manifest"),
        validate="one_to_one",
    )
    if merged[["fold_id_manifest", "source_group_id_manifest"]].isna().any().any():
        raise ValueError(f"{model_id}: OOF perfumes do not match the repaired manifest")
    if not (merged["fold_id"].astype(int) == merged["fold_id_manifest"].astype(int)).all():
        raise ValueError(f"{model_id}: fold IDs differ from the repaired manifest")
    if not (merged["source_group_id"] == merged["source_group_id_manifest"]).all():
        raise ValueError(f"{model_id}: source-group IDs differ from the repaired manifest")
    if merged.groupby("source_group_id")["fold_id"].nunique().max() != 1:
        raise ValueError(f"{model_id}: a source group crosses folds")
    return merged


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_to_markdown(frame: pd.DataFrame, *, float_digits: int | None = None) -> str:
    def render(value: Any) -> str:
        if float_digits is not None and isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        return str(value)

    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_report(detail: pd.DataFrame, summary: pd.DataFrame, wide: pd.DataFrame, path: Path) -> None:
    summary_view = summary[
        [
            "model_name",
            "total_predicted_positive",
            "mean_predicted_labels_per_perfume",
            "total_true_positive",
            "total_false_positive",
            "total_false_negative",
            "micro_f1",
        ]
    ]
    lines = [
        "# Source-Grouped OOF Confusion and Prediction Diagnostics",
        "",
        "All counts use the fixed threshold-0.3 OOF predictions for the same 784 perfumes and repaired source-grouped folds. The five accepted models are supplemented by a post-hoc unweighted-LR sensitivity row and a simple TF-IDF text-baseline row.",
        "",
        "## Predicted-positive counts by emotion",
        "",
        dataframe_to_markdown(wide),
        "",
        "## Overall prediction behavior",
        "",
        dataframe_to_markdown(summary_view, float_digits=4),
        "",
        "The detailed CSV contains TP, FP, TN, FN, precision, recall, specificity, and F1 for every model-emotion pair.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    detail_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    reference_truth: pd.DataFrame | None = None

    for model_id, (model_name, path) in MODEL_INPUTS.items():
        frame = pd.read_csv(path)
        validate_against_manifest(frame, manifest, model_id=model_id)
        for label in LABELS:
            expected = (frame[f"prob_{label}"].to_numpy(dtype=float) >= 0.3).astype(int)
            stored = frame[f"pred_{label}"].to_numpy(dtype=int)
            if not np.array_equal(expected, stored):
                raise ValueError(f"{model_id}/{label}: stored predictions do not match threshold 0.3")
        truth = frame[["perfume_id", *[f"true_{label}" for label in LABELS]]].sort_values(
            "perfume_id"
        ).reset_index(drop=True)
        if reference_truth is None:
            reference_truth = truth
        elif not truth.equals(reference_truth):
            raise ValueError(f"{model_id}: OOF truth labels differ from the other models")
        rows, summary = calculate_model_diagnostics(
            frame,
            model_id=model_id,
            model_name=model_name,
        )
        detail_rows.extend(rows)
        summaries.append(summary)
        input_hashes[model_id] = sha256_file(path)

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summaries)
    actual = detail.loc[detail["model_id"] == "logistic_regression_weighted", [
        "label", "actual_positive_count"
    ]].copy()
    predicted = detail.pivot(
        index="label",
        columns="model_id",
        values="predicted_positive_count",
    ).reset_index()
    wide = actual.merge(predicted, on="label", validate="one_to_one")
    wide = wide[["label", "actual_positive_count", *MODEL_INPUTS.keys()]]

    detail_path = output_root / "per_label_confusion_and_prediction_counts.csv"
    summary_path = output_root / "model_prediction_summary.csv"
    wide_path = output_root / "predicted_positive_counts_wide.csv"
    report_path = output_root / "REPORT.md"
    validation_path = output_root / "validation_report.json"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    wide.to_csv(wide_path, index=False)
    write_report(detail, summary, wide, report_path)
    validation_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "manifest_path": str(args.manifest),
                "manifest_sha256": sha256_file(args.manifest),
                "model_count": len(MODEL_INPUTS),
                "label_count": len(LABELS),
                "detail_rows": int(len(detail)),
                "perfumes_per_model": 784,
                "fixed_threshold": 0.3,
                "stored_predictions_match_probabilities": True,
                "shared_truth_verified": True,
                "shared_fold_assignments_verified": True,
                "cross_fold_source_groups": 0,
                "input_oof_sha256": input_hashes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    checksum_path = output_root / "SHA256SUMS"
    artifact_paths = sorted(
        path for path in output_root.iterdir() if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  ./{path.name}\n" for path in artifact_paths),
        encoding="utf-8",
    )
    print("Source-grouped prediction diagnostics complete")
    print(f"- Detailed counts: {detail_path}")
    print(f"- Predicted-positive table: {wide_path}")
    print(f"- Model summary: {summary_path}")
    print(f"- Validation: {validation_path}")


if __name__ == "__main__":
    main()
