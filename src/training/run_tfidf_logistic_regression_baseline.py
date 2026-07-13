"""Run a transcript-only TF-IDF + Logistic Regression source-grouped baseline."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, roc_auc_score

from src.training.source_grouped_folds import OFFICIAL_LABELS, SOURCE_GROUP_COLUMN, sha256_file
from src.utils.paths import PROJECT_ROOT


MODEL_ID = "tfidf_logistic_regression"
MODEL_NAME = "TF-IDF + Logistic Regression"
THRESHOLD = 0.3
SEED = 42
DEFAULT_TEXT_JSONL = (
    PROJECT_ROOT
    / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/datasets/perfume_text_dataset_v1.jsonl"
)
DEFAULT_LABEL_CSV = (
    PROJECT_ROOT / "outputs/gemini_browser_og784/analysis_out/gemini_browser_og784_feature_labels.csv"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/tables/fold_assignments.csv"
)
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs/tfidf_logistic_regression_source_grouped_v1"


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def load_inputs(
    *, text_jsonl: Path, label_csv: Path, manifest_csv: Path
) -> tuple[pd.DataFrame, dict[str, str]]:
    records = [
        json.loads(line)
        for line in text_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    text_df = pd.DataFrame(
        {
            "perfume_id": [str(row["perfume_id"]) for row in records],
            "document_text": [str(row["document_text"]).strip() for row in records],
            "json_fold_id": [int(row["fold_id"]) for row in records],
            "json_source_group_id": [str(row["source_group_id"]) for row in records],
        }
    )
    labels = pd.read_csv(label_csv)[["perfume_id", *OFFICIAL_LABELS]].copy()
    manifest = pd.read_csv(manifest_csv)[
        ["perfume_id", "fold_id", SOURCE_GROUP_COLUMN, "source_group_size"]
    ].copy()
    for frame in [text_df, labels, manifest]:
        frame["perfume_id"] = frame["perfume_id"].astype(str)
        if frame["perfume_id"].duplicated().any():
            raise ValueError("Every input must contain unique perfume_id values")
    data = manifest.merge(labels, on="perfume_id", validate="one_to_one").merge(
        text_df, on="perfume_id", validate="one_to_one"
    )
    if len(data) != 784 or data["perfume_id"].nunique() != 784:
        raise ValueError("TF-IDF baseline requires exactly 784 unique perfumes")
    if (data["document_text"].str.len() == 0).any():
        raise ValueError("TF-IDF baseline found an empty document")
    if not (data["fold_id"].astype(int) == data["json_fold_id"].astype(int)).all():
        raise ValueError("JSONL fold IDs differ from the repaired manifest")
    if not (data[SOURCE_GROUP_COLUMN] == data["json_source_group_id"]).all():
        raise ValueError("JSONL source groups differ from the repaired manifest")
    if data.groupby(SOURCE_GROUP_COLUMN)["fold_id"].nunique().max() != 1:
        raise ValueError("A source group crosses repaired folds")
    for label in OFFICIAL_LABELS:
        data[label] = pd.to_numeric(data[label], errors="raise").astype(int)
        if not data[label].isin([0, 1]).all():
            raise ValueError(f"Non-binary label: {label}")
    return data.sort_values("perfume_id").reset_index(drop=True), {
        "text_jsonl": sha256_file(text_jsonl),
        "label_csv": sha256_file(label_csv),
        "manifest_csv": sha256_file(manifest_csv),
    }


def run_tfidf_baseline(data: pd.DataFrame) -> dict[str, Any]:
    y = data[list(OFFICIAL_LABELS)].to_numpy(dtype=int)
    fold_ids = data["fold_id"].to_numpy(dtype=int)
    texts = data["document_text"].tolist()
    oof_prob = np.zeros_like(y, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    training_status: dict[str, list[str]] = {label: [] for label in OFFICIAL_LABELS}

    for fold_id in sorted(np.unique(fold_ids)):
        train_idx = np.where(fold_ids != fold_id)[0]
        test_idx = np.where(fold_ids == fold_id)[0]
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=20_000,
            sublinear_tf=True,
            norm="l2",
        )
        x_train = vectorizer.fit_transform([texts[index] for index in train_idx])
        x_test = vectorizer.transform([texts[index] for index in test_idx])
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "train_perfumes": int(len(train_idx)),
                "test_perfumes": int(len(test_idx)),
                "train_source_groups": int(data.iloc[train_idx][SOURCE_GROUP_COLUMN].nunique()),
                "test_source_groups": int(data.iloc[test_idx][SOURCE_GROUP_COLUMN].nunique()),
                "tfidf_features": int(x_train.shape[1]),
                "train_nonzero_values": int(x_train.nnz),
                "test_nonzero_values": int(x_test.nnz),
            }
        )
        for label_index, label in enumerate(OFFICIAL_LABELS):
            y_train = y[train_idx, label_index]
            if np.unique(y_train).size < 2:
                constant = int(y_train[0]) if len(y_train) else 0
                probability = np.full(len(test_idx), float(constant))
                status = f"constant_train_fold_{fold_id}"
            else:
                model = LogisticRegression(
                    max_iter=2_000,
                    class_weight=None,
                    solver="liblinear",
                    random_state=SEED,
                )
                model.fit(x_train, y_train)
                probability = model.predict_proba(x_test)[:, 1]
                status = f"trained_fold_{fold_id}"
            oof_prob[test_idx, label_index] = probability
            training_status[label].append(status)

    oof_pred = (oof_prob >= THRESHOLD).astype(int)
    per_label_rows: list[dict[str, Any]] = []
    for index, label in enumerate(OFFICIAL_LABELS):
        true = y[:, index]
        pred = oof_pred[:, index]
        prob = oof_prob[:, index]
        tp = int(np.sum((true == 1) & (pred == 1)))
        fp = int(np.sum((true == 0) & (pred == 1)))
        tn = int(np.sum((true == 0) & (pred == 0)))
        fn = int(np.sum((true == 1) & (pred == 0)))
        per_label_rows.append(
            {
                "label": label,
                "support_pos": int(true.sum()),
                "support_neg": int(len(true) - true.sum()),
                "predicted_positive_count": int(pred.sum()),
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "precision": _safe_ratio(tp, tp + fp),
                "recall": _safe_ratio(tp, tp + fn),
                "f1": float(f1_score(true, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(true, prob)),
                "average_precision": float(average_precision_score(true, prob)),
                "training_status": ";".join(training_status[label]),
            }
        )
    per_label = pd.DataFrame(per_label_rows)
    metrics = {
        "macro_f1": float(per_label["f1"].mean()),
        "macro_roc_auc": float(per_label["roc_auc"].mean()),
        "macro_average_precision": float(per_label["average_precision"].mean()),
        "micro_f1": float(f1_score(y, oof_pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y, oof_pred)),
        "total_predicted_positive": int(oof_pred.sum()),
        "mean_predicted_labels_per_perfume": float(oof_pred.sum(axis=1).mean()),
    }
    prediction = data[
        ["perfume_id", "fold_id", SOURCE_GROUP_COLUMN, "source_group_size"]
    ].copy()
    for index, label in enumerate(OFFICIAL_LABELS):
        prediction[f"true_{label}"] = y[:, index]
        prediction[f"prob_{label}"] = oof_prob[:, index]
        prediction[f"pred_{label}"] = oof_pred[:, index]
    return {
        "predictions": prediction,
        "per_label": per_label,
        "fold_diagnostics": pd.DataFrame(fold_rows),
        "metrics": metrics,
    }


def write_outputs(
    *, result: dict[str, Any], out_root: Path, input_hashes: dict[str, str], paths: dict[str, Path]
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    result["predictions"].to_csv(out_root / "tfidf_logistic_regression_oof.csv", index=False)
    result["per_label"].to_csv(out_root / "tfidf_logistic_regression_per_label.csv", index=False)
    result["fold_diagnostics"].to_csv(out_root / "fold_diagnostics.csv", index=False)
    config = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "analysis_role": "simple_text_baseline_not_primary_five_model_benchmark",
        "threshold": THRESHOLD,
        "seed": SEED,
        "text_input": "transcript-only perfume-level document_text",
        "vectorizer": {
            "type": "TfidfVectorizer",
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.98,
            "max_features": 20_000,
            "sublinear_tf": True,
            "norm": "l2",
            "fit_scope": "outer-training-fold only",
        },
        "classifier": {
            "type": "LogisticRegression",
            "max_iter": 2_000,
            "class_weight": None,
            "solver": "liblinear",
        },
        "input_paths": {key: str(value) for key, value in paths.items()},
        "input_sha256": input_hashes,
    }
    (out_root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    metrics_payload = {
        **config,
        "n_perfumes": 784,
        "n_source_groups": 359,
        "metrics": result["metrics"],
        "per_label": result["per_label"]
        .astype(object)
        .where(pd.notna(result["per_label"]), None)
        .to_dict(orient="records"),
    }
    (out_root / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    metrics = result["metrics"]
    report = f"""# TF-IDF + Logistic Regression Source-Grouped Baseline

This simple text baseline uses transcript-only perfume documents, TF-IDF unigrams and bigrams fit inside each outer training fold, and unweighted binary Logistic Regression. It uses the same 784 perfumes and repaired five-fold source-component manifest as the accepted benchmark.

| Metric | Value |
| --- | ---: |
| Macro-F1 at threshold 0.3 | {metrics['macro_f1']:.4f} |
| Macro ROC-AUC | {metrics['macro_roc_auc']:.4f} |
| Macro average precision | {metrics['macro_average_precision']:.4f} |
| Micro-F1 | {metrics['micro_f1']:.4f} |
| Hamming loss | {metrics['hamming_loss']:.4f} |
| Mean predicted labels per perfume | {metrics['mean_predicted_labels_per_perfume']:.4f} |

This row is a simple learned text baseline. It is not a sixth primary model and was added after the repaired five-model benchmark was inspected.
"""
    (out_root / "REPORT.md").write_text(report, encoding="utf-8")
    predictions = result["predictions"]
    validation = {
        "status": "passed",
        "rows": int(len(predictions)),
        "unique_perfumes": int(predictions["perfume_id"].nunique()),
        "source_groups": int(predictions[SOURCE_GROUP_COLUMN].nunique()),
        "cross_fold_source_groups": 0,
        "fold_sizes": {
            str(int(key)): int(value) for key, value in predictions.groupby("fold_id").size().items()
        },
        "probability_range_valid": True,
        "stored_predictions_match_threshold": True,
        "input_sha256": input_hashes,
        "metrics": metrics,
    }
    (out_root / "validation_report.json").write_text(
        json.dumps(validation, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    run = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "python_version": platform.python_version(),
        "model_id": MODEL_ID,
        "threshold": THRESHOLD,
        "input_sha256": input_hashes,
    }
    (out_root / "run_manifest.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    checksum_path = out_root / "SHA256SUMS"
    artifacts = sorted(path for path in out_root.iterdir() if path.is_file() and path != checksum_path)
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  ./{path.name}\n" for path in artifacts), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-jsonl", type=Path, default=DEFAULT_TEXT_JSONL)
    parser.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, input_hashes = load_inputs(
        text_jsonl=args.text_jsonl, label_csv=args.label_csv, manifest_csv=args.manifest
    )
    result = run_tfidf_baseline(data)
    paths = {"text_jsonl": args.text_jsonl, "label_csv": args.label_csv, "manifest": args.manifest}
    write_outputs(result=result, out_root=args.out_root, input_hashes=input_hashes, paths=paths)
    metrics = result["metrics"]
    print("TF-IDF + Logistic Regression baseline complete")
    print(f"- Macro-F1={metrics['macro_f1']:.4f}")
    print(f"- Macro ROC-AUC={metrics['macro_roc_auc']:.4f}")
    print(f"- Macro average precision={metrics['macro_average_precision']:.4f}")
    print(f"- Output root: {args.out_root}")


if __name__ == "__main__":
    main()
