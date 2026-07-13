"""Build a fair fixed-threshold table from frozen source-grouped OOF probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/codex-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, roc_auc_score

from scripts.build_source_grouped_prediction_diagnostics import (
    LABELS,
    MODEL_INPUTS,
    DEFAULT_MANIFEST,
    dataframe_to_markdown,
    validate_against_manifest,
)
from src.utils.paths import PROJECT_ROOT


THRESHOLDS = [0.2, 0.3, 0.5, 0.7]
OFFICIAL_THRESHOLD = 0.3
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/source_grouped_all_model_thresholds_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_threshold_rows(
    frame: pd.DataFrame,
    *,
    model_id: str,
    model_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    y_true = frame[[f"true_{label}" for label in LABELS]].to_numpy(dtype=int)
    y_prob = frame[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=float)
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError(f"{model_id}: probabilities must be finite and in [0, 1]")

    ranking = {
        "model_id": model_id,
        "model_name": model_name,
        "macro_roc_auc": float(
            np.mean([roc_auc_score(y_true[:, index], y_prob[:, index]) for index in range(len(LABELS))])
        ),
        "macro_average_precision": float(
            np.mean(
                [average_precision_score(y_true[:, index], y_prob[:, index]) for index in range(len(LABELS))]
            )
        ),
        "ranking_metrics_threshold_independent": True,
    }
    threshold_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        y_pred = (y_prob >= threshold).astype(int)
        per_label_f1 = [
            float(f1_score(y_true[:, index], y_pred[:, index], zero_division=0))
            for index in range(len(LABELS))
        ]
        threshold_rows.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "threshold": threshold,
                "threshold_role": "official" if threshold == OFFICIAL_THRESHOLD else "diagnostic",
                "macro_f1": float(np.mean(per_label_f1)),
                "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
                "hamming_loss": float(hamming_loss(y_true, y_pred)),
                "total_predicted_positive": int(y_pred.sum()),
                "mean_predicted_labels_per_perfume": float(y_pred.sum(axis=1).mean()),
            }
        )
        for index, label in enumerate(LABELS):
            true = y_true[:, index]
            pred = y_pred[:, index]
            tp = int(np.sum((true == 1) & (pred == 1)))
            fp = int(np.sum((true == 0) & (pred == 1)))
            tn = int(np.sum((true == 0) & (pred == 0)))
            fn = int(np.sum((true == 1) & (pred == 0)))
            per_label_rows.append(
                {
                    "model_id": model_id,
                    "model_name": model_name,
                    "threshold": threshold,
                    "threshold_role": "official" if threshold == OFFICIAL_THRESHOLD else "diagnostic",
                    "label": label,
                    "f1": per_label_f1[index],
                    "actual_positive_count": int(true.sum()),
                    "predicted_positive_count": int(pred.sum()),
                    "true_positive": tp,
                    "false_positive": fp,
                    "true_negative": tn,
                    "false_negative": fn,
                }
            )
    return threshold_rows, per_label_rows, ranking


def build_best_table(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, frame in long.groupby("model_id", sort=False):
        best_value = float(frame["macro_f1"].max())
        candidates = frame.loc[np.isclose(frame["macro_f1"], best_value)].copy()
        candidates["distance_from_official"] = (
            candidates["threshold"] - OFFICIAL_THRESHOLD
        ).abs()
        best = candidates.sort_values(["distance_from_official", "threshold"]).iloc[0]
        official = frame.loc[np.isclose(frame["threshold"], OFFICIAL_THRESHOLD)].iloc[0]
        rows.append(
            {
                "model_id": model_id,
                "model_name": best["model_name"],
                "best_diagnostic_threshold": float(best["threshold"]),
                "best_macro_f1": float(best["macro_f1"]),
                "official_threshold": OFFICIAL_THRESHOLD,
                "official_macro_f1": float(official["macro_f1"]),
                "best_minus_official_macro_f1": float(best["macro_f1"] - official["macro_f1"]),
                "selection_note": "diagnostic_only; official reporting remains threshold 0.3",
            }
        )
    return pd.DataFrame(rows)


def save_table_figure(wide: pd.DataFrame, path: Path) -> None:
    display = wide.copy()
    display.columns = ["Model", "0.2", "0.3 (official)", "0.5", "0.7"]
    cell_text = [
        [row.iloc[0], *[f"{float(value):.4f}" for value in row.iloc[1:]]]
        for _, row in display.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.38, 0.12, 0.16, 0.12, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.65)
    for column in range(len(display.columns)):
        table[(0, column)].set_facecolor("#1f4e78")
        table[(0, column)].set_text_props(color="white", weight="bold")
    for row_index in range(1, len(display) + 1):
        table[(row_index, 2)].set_facecolor("#dbeafe")
        values = [float(value) for value in display.iloc[row_index - 1, 1:]]
        best_column = int(np.argmax(values)) + 1
        table[(row_index, best_column)].set_text_props(weight="bold")
    ax.set_title(
        "Macro-F1 from identical frozen OOF probabilities at fixed thresholds",
        fontsize=14,
        weight="bold",
        pad=18,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    *,
    wide: pd.DataFrame,
    ranking: pd.DataFrame,
    best: pd.DataFrame,
    path: Path,
) -> None:
    display_wide = wide.copy()
    for column in display_wide.columns[1:]:
        display_wide[column] = display_wide[column].map(lambda value: f"{float(value):.4f}")
    display_ranking = ranking[["model_name", "macro_roc_auc", "macro_average_precision"]].copy()
    display_best = best[
        [
            "model_name",
            "best_diagnostic_threshold",
            "best_macro_f1",
            "official_macro_f1",
            "best_minus_official_macro_f1",
        ]
    ].copy()
    lines = [
        "# Fair All-Model Fixed-Threshold Comparison",
        "",
        "The same saved OOF probability matrix for each model is thresholded at 0.2, 0.3, 0.5, and 0.7. Models are not retrained between thresholds. Threshold 0.3 remains the prespecified official reporting threshold; the other columns are sensitivity diagnostics.",
        "",
        "## Macro-F1 by fixed threshold",
        "",
        dataframe_to_markdown(display_wide),
        "",
        "## Threshold-independent ranking metrics",
        "",
        "ROC-AUC and average precision are reported once because changing a cutoff does not change probability rankings.",
        "",
        dataframe_to_markdown(display_ranking, float_digits=4),
        "",
        "## Best tested threshold as a diagnostic",
        "",
        dataframe_to_markdown(display_best, float_digits=4),
        "",
        "The unweighted LR is a post-hoc sensitivity model and TF-IDF + LR is a simple text baseline; neither is an additional primary benchmark row.",
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
    long_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    reference_truth: pd.DataFrame | None = None

    for model_id, (model_name, input_path) in MODEL_INPUTS.items():
        frame = pd.read_csv(input_path)
        validate_against_manifest(frame, manifest, model_id=model_id)
        truth = frame[["perfume_id", *[f"true_{label}" for label in LABELS]]].sort_values(
            "perfume_id"
        ).reset_index(drop=True)
        if reference_truth is None:
            reference_truth = truth
        elif not truth.equals(reference_truth):
            raise ValueError(f"{model_id}: OOF truth labels differ from other models")
        rows, label_rows, ranking = calculate_threshold_rows(
            frame,
            model_id=model_id,
            model_name=model_name,
        )
        official_predictions = np.column_stack(
            [(frame[f"prob_{label}"].to_numpy(dtype=float) >= OFFICIAL_THRESHOLD).astype(int) for label in LABELS]
        )
        stored_predictions = frame[[f"pred_{label}" for label in LABELS]].to_numpy(dtype=int)
        if not np.array_equal(official_predictions, stored_predictions):
            raise ValueError(f"{model_id}: stored OOF predictions do not match threshold 0.3")
        long_rows.extend(rows)
        per_label_rows.extend(label_rows)
        ranking_rows.append(ranking)
        input_hashes[model_id] = sha256_file(input_path)

    long = pd.DataFrame(long_rows)
    per_label = pd.DataFrame(per_label_rows)
    ranking = pd.DataFrame(ranking_rows)
    wide = long.pivot(index=["model_id", "model_name"], columns="threshold", values="macro_f1").reset_index()
    wide = wide[["model_id", "model_name", *THRESHOLDS]]
    best = build_best_table(long)
    thesis_wide = wide.drop(columns="model_id")

    long_path = output_root / "fixed_threshold_metrics_long.csv"
    per_label_path = output_root / "per_label_fixed_threshold_metrics.csv"
    wide_path = output_root / "macro_f1_by_fixed_threshold.csv"
    ranking_path = output_root / "threshold_independent_ranking_metrics.csv"
    best_path = output_root / "best_diagnostic_threshold_per_model.csv"
    report_path = output_root / "REPORT.md"
    figure_path = output_root / "macro_f1_fixed_threshold_table.png"
    validation_path = output_root / "validation_report.json"
    long.to_csv(long_path, index=False)
    per_label.to_csv(per_label_path, index=False)
    thesis_wide.to_csv(wide_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    best.to_csv(best_path, index=False)
    write_report(wide=thesis_wide, ranking=ranking, best=best, path=report_path)
    save_table_figure(thesis_wide, figure_path)
    validation_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "model_count": len(MODEL_INPUTS),
                "primary_model_count": 5,
                "sensitivity_model_count": 1,
                "simple_text_baseline_count": 1,
                "thresholds": THRESHOLDS,
                "official_threshold": OFFICIAL_THRESHOLD,
                "perfumes_per_model": 784,
                "long_table_rows": int(len(long)),
                "per_label_table_rows": int(len(per_label)),
                "shared_truth_verified": True,
                "shared_fold_assignments_verified": True,
                "stored_threshold_0_3_predictions_verified": True,
                "probability_matrix_reused_across_thresholds": True,
                "ranking_metrics_reported_once_per_model": True,
                "manifest_sha256": sha256_file(args.manifest),
                "input_oof_sha256": input_hashes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    checksum_path = output_root / "SHA256SUMS"
    artifacts = sorted(path for path in output_root.iterdir() if path.is_file() and path != checksum_path)
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  ./{path.name}\n" for path in artifacts),
        encoding="utf-8",
    )
    print("All-model fixed-threshold table complete")
    print(f"- Macro-F1 table: {wide_path}")
    print(f"- Ranking metrics: {ranking_path}")
    print(f"- Report: {report_path}")
    print(f"- Figure: {figure_path}")
    print(f"- Validation: {validation_path}")


if __name__ == "__main__":
    main()
