# Date Created: 2026-07-13
# Date Last Modified: 2026-07-13

"""Generate source-group-aware benchmark diagnostics from frozen OOF outputs.

This module is intentionally independent of the benchmark orchestrator.  It
consumes frozen feature labels, outer/inner manifests, and the five model OOF
files, then writes deterministic support, fold-metric, component-weighted, and
paired component-bootstrap artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


MODEL_IDS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "bert",
    "bilstm_attention",
)
OFFICIAL_LABELS = (
    "joy",
    "sadness",
    "fear",
    "trust",
    "surprise",
    "anticipation",
    "disgust",
)
SOURCE_GROUP_COLUMN = "source_group_id"
OUTPUT_FILENAMES = {
    "inner_fold_balance": "inner_fold_balance.csv",
    "outer_fold_per_label_metrics": "outer_fold_per_label_metrics.csv",
    "component_weighted_model_metrics": "component_weighted_model_metrics.csv",
    "paired_component_bootstrap": "paired_component_bootstrap_macro_f1.csv",
    "report": "source_grouped_diagnostics_report.json",
}
COMPONENT_WEIGHT_METHOD = (
    "row_weight=1/source_group_perfume_count; each_source_group_total_weight=1"
)
BOOTSTRAP_METHOD = (
    "paired source-group cluster bootstrap; sample n_source_components with replacement; "
    "each sampled component has total weight 1; macro-F1 averages all seven labels"
)


@dataclass(frozen=True)
class ValidatedInputs:
    features: pd.DataFrame
    outer: pd.DataFrame
    inner: pd.DataFrame
    oof_by_model: Mapping[str, pd.DataFrame]
    oof_paths: Mapping[str, Path]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of exact file bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalize_ids(series: pd.Series, *, context: str) -> pd.Series:
    values = series.astype(str).str.strip()
    if (values == "").any():
        raise ValueError(f"{context} contains an empty perfume_id")
    return values


def _integer_series(series: pd.Series, *, context: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    if (numeric % 1 != 0).any():
        raise ValueError(f"{context} contains a non-integer value")
    return numeric.astype(int)


def _binary_series(series: pd.Series, *, context: str) -> pd.Series:
    numeric = _integer_series(series, context=context)
    if not set(numeric.unique()).issubset({0, 1}):
        raise ValueError(f"{context} must contain only 0/1 values")
    return numeric


def _read_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"perfume_id", *OFFICIAL_LABELS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature-label CSV lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["perfume_id"] = _normalize_ids(frame["perfume_id"], context="Feature-label CSV")
    if frame["perfume_id"].duplicated().any():
        raise ValueError("Feature-label CSV contains duplicate perfume_id values")
    for label in OFFICIAL_LABELS:
        frame[label] = _binary_series(frame[label], context=f"Feature-label {label}")
    return frame.sort_values("perfume_id").reset_index(drop=True)


def _read_outer(path: Path, *, feature_ids: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"perfume_id", "fold_id", SOURCE_GROUP_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Outer manifest lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["perfume_id"] = _normalize_ids(frame["perfume_id"], context="Outer manifest")
    frame[SOURCE_GROUP_COLUMN] = frame[SOURCE_GROUP_COLUMN].astype(str).str.strip()
    if (frame[SOURCE_GROUP_COLUMN] == "").any():
        raise ValueError("Outer manifest contains an empty source_group_id")
    if frame["perfume_id"].duplicated().any():
        raise ValueError("Outer manifest contains duplicate perfume_id values")
    frame["fold_id"] = _integer_series(frame["fold_id"], context="Outer manifest fold_id")
    if set(frame["fold_id"].unique()) != {1, 2, 3, 4, 5}:
        raise ValueError("Outer manifest fold IDs must be exactly 1..5")
    if set(frame["perfume_id"]) != set(feature_ids):
        raise ValueError("Outer manifest perfume universe differs from feature labels")
    if int(frame.groupby(SOURCE_GROUP_COLUMN)["fold_id"].nunique().max()) != 1:
        raise ValueError("A source group crosses outer folds")

    actual_sizes = frame.groupby(SOURCE_GROUP_COLUMN)["perfume_id"].transform("size").astype(int)
    if "source_group_size" in frame.columns:
        declared_sizes = _integer_series(
            frame["source_group_size"], context="Outer manifest source_group_size"
        )
        if not declared_sizes.equals(actual_sizes):
            raise ValueError("Outer manifest source_group_size values are inconsistent")
    frame["source_group_size"] = actual_sizes
    return frame.sort_values("perfume_id").reset_index(drop=True)


def _read_inner(path: Path, *, outer: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"outer_fold_id", "perfume_id", "inner_role", SOURCE_GROUP_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Inner manifest lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["perfume_id"] = _normalize_ids(frame["perfume_id"], context="Inner manifest")
    frame[SOURCE_GROUP_COLUMN] = frame[SOURCE_GROUP_COLUMN].astype(str).str.strip()
    frame["outer_fold_id"] = _integer_series(
        frame["outer_fold_id"], context="Inner manifest outer_fold_id"
    )
    frame["inner_role"] = frame["inner_role"].astype(str).str.strip().str.lower()
    if frame.duplicated(["outer_fold_id", "perfume_id"]).any():
        raise ValueError("Inner manifest duplicates an outer-fold/perfume pair")
    if set(frame["outer_fold_id"].unique()) != {1, 2, 3, 4, 5}:
        raise ValueError("Inner manifest outer fold IDs must be exactly 1..5")
    if not set(frame["inner_role"].unique()).issubset({"train", "val"}):
        raise ValueError("Inner manifest roles must be train or val")

    outer_by_id = outer.set_index("perfume_id")
    expected_groups = frame["perfume_id"].map(outer_by_id[SOURCE_GROUP_COLUMN])
    if expected_groups.isna().any():
        raise ValueError("Inner manifest contains an unknown perfume_id")
    if not (frame[SOURCE_GROUP_COLUMN].astype(str) == expected_groups.astype(str)).all():
        raise ValueError("Inner manifest source groups differ from the outer manifest")

    for outer_fold_id in range(1, 6):
        observed = frame.loc[frame["outer_fold_id"] == outer_fold_id]
        expected_ids = set(outer.loc[outer["fold_id"] != outer_fold_id, "perfume_id"])
        if set(observed["perfume_id"]) != expected_ids:
            raise ValueError(f"Inner universe is incorrect for outer fold {outer_fold_id}")
        if set(observed["inner_role"]) != {"train", "val"}:
            raise ValueError(f"Outer fold {outer_fold_id} lacks an inner train or val role")
        if int(observed.groupby(SOURCE_GROUP_COLUMN)["inner_role"].nunique().max()) != 1:
            raise ValueError(f"A source group crosses inner roles for outer fold {outer_fold_id}")

    role_order = pd.Categorical(frame["inner_role"], categories=["train", "val"], ordered=True)
    frame = frame.assign(_role_order=role_order)
    return frame.sort_values(["outer_fold_id", "_role_order", "perfume_id"]).drop(
        columns="_role_order"
    ).reset_index(drop=True)


def _read_oof(
    path: Path,
    *,
    model_id: str,
    features: pd.DataFrame,
    outer: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"perfume_id", "fold_id", SOURCE_GROUP_COLUMN}
    for label in OFFICIAL_LABELS:
        required.update({f"true_{label}", f"prob_{label}", f"pred_{label}"})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{model_id} OOF lacks columns: {sorted(missing)}")

    frame = frame.copy()
    frame["perfume_id"] = _normalize_ids(frame["perfume_id"], context=f"{model_id} OOF")
    if frame["perfume_id"].duplicated().any():
        raise ValueError(f"{model_id} OOF duplicates perfume_id")
    frame["fold_id"] = _integer_series(frame["fold_id"], context=f"{model_id} OOF fold_id")
    frame[SOURCE_GROUP_COLUMN] = frame[SOURCE_GROUP_COLUMN].astype(str).str.strip()

    canonical_ids = features["perfume_id"].tolist()
    if set(frame["perfume_id"]) != set(canonical_ids):
        raise ValueError(f"{model_id} OOF perfume universe differs from feature labels")
    frame = frame.set_index("perfume_id").loc[canonical_ids].reset_index()
    outer_by_id = outer.set_index("perfume_id")
    expected_fold = frame["perfume_id"].map(outer_by_id["fold_id"])
    expected_group = frame["perfume_id"].map(outer_by_id[SOURCE_GROUP_COLUMN]).astype(str)
    if not (frame["fold_id"].to_numpy() == expected_fold.to_numpy()).all():
        raise ValueError(f"{model_id} OOF fold IDs differ from the outer manifest")
    if not (frame[SOURCE_GROUP_COLUMN].astype(str) == expected_group).all():
        raise ValueError(f"{model_id} OOF source groups differ from the outer manifest")

    truth_by_id = features.set_index("perfume_id")
    for label in OFFICIAL_LABELS:
        truth = _binary_series(frame[f"true_{label}"], context=f"{model_id} true_{label}")
        expected_truth = frame["perfume_id"].map(truth_by_id[label]).astype(int)
        if not (truth.to_numpy() == expected_truth.to_numpy()).all():
            raise ValueError(f"{model_id} OOF truths differ from frozen labels for {label}")
        probability = pd.to_numeric(frame[f"prob_{label}"], errors="raise").astype(float)
        if not np.isfinite(probability.to_numpy()).all():
            raise ValueError(f"{model_id} OOF has non-finite probabilities for {label}")
        if ((probability < 0.0) | (probability > 1.0)).any():
            raise ValueError(f"{model_id} OOF probabilities fall outside [0, 1] for {label}")
        prediction = _binary_series(frame[f"pred_{label}"], context=f"{model_id} pred_{label}")
        expected_prediction = (probability.to_numpy() >= threshold).astype(int)
        if not (prediction.to_numpy() == expected_prediction).all():
            raise ValueError(
                f"{model_id} OOF pred_{label} is inconsistent with threshold {threshold}"
            )
        frame[f"true_{label}"] = truth
        frame[f"prob_{label}"] = probability
        frame[f"pred_{label}"] = prediction
    return frame


def load_and_validate_inputs(
    *,
    feature_label_csv: Path,
    outer_manifest_csv: Path,
    inner_manifest_csv: Path,
    benchmark_root: Path,
    threshold: float,
) -> ValidatedInputs:
    """Load all canonical inputs and enforce cross-artifact identity equality."""

    required_paths = [feature_label_csv, outer_manifest_csv, inner_manifest_csv]
    oof_paths = {
        model_id: benchmark_root / "predictions" / f"{model_id}_oof.csv"
        for model_id in MODEL_IDS
    }
    required_paths.extend(oof_paths.values())
    missing = [path for path in required_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing diagnostics input: {missing[0]}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    features = _read_features(feature_label_csv)
    outer = _read_outer(outer_manifest_csv, feature_ids=features["perfume_id"].tolist())
    inner = _read_inner(inner_manifest_csv, outer=outer)
    oof_by_model = {
        model_id: _read_oof(
            path,
            model_id=model_id,
            features=features,
            outer=outer,
            threshold=threshold,
        )
        for model_id, path in oof_paths.items()
    }
    return ValidatedInputs(
        features=features,
        outer=outer,
        inner=inner,
        oof_by_model=oof_by_model,
        oof_paths=oof_paths,
    )


def build_inner_fold_balance(inputs: ValidatedInputs) -> pd.DataFrame:
    """Return inner train/validation row, component, and label support counts."""

    labels_by_id = inputs.features.set_index("perfume_id")[list(OFFICIAL_LABELS)]
    rows: list[Dict[str, Any]] = []
    for outer_fold_id in range(1, 6):
        for role in ("train", "val"):
            subset = inputs.inner.loc[
                (inputs.inner["outer_fold_id"] == outer_fold_id)
                & (inputs.inner["inner_role"] == role)
            ].copy()
            support = labels_by_id.loc[subset["perfume_id"].tolist()]
            row: Dict[str, Any] = {
                "outer_fold_id": outer_fold_id,
                "inner_role": role,
                "perfume_count": int(len(subset)),
                "source_group_count": int(subset[SOURCE_GROUP_COLUMN].nunique()),
            }
            for label in OFFICIAL_LABELS:
                row[f"{label}_positive_support"] = int(support[label].sum())
            rows.append(row)
    return pd.DataFrame(rows)


def _fold_metric_values(
    *,
    truth: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, Any]:
    support_pos = int(truth.sum())
    support_neg = int(len(truth) - support_pos)
    roc_defined = support_pos > 0 and support_neg > 0
    average_precision_defined = support_pos > 0
    return {
        "support_pos": support_pos,
        "support_neg": support_neg,
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probability)) if roc_defined else None,
        "average_precision": (
            float(average_precision_score(truth, probability))
            if average_precision_defined
            else None
        ),
        "roc_auc_defined": roc_defined,
        "average_precision_defined": average_precision_defined,
    }


def build_outer_fold_per_label_metrics(
    inputs: ValidatedInputs,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Compute row-level diagnostic metrics for every model/fold/label."""

    rows: list[Dict[str, Any]] = []
    for model_id in MODEL_IDS:
        oof = inputs.oof_by_model[model_id]
        for fold_id in range(1, 6):
            subset = oof.loc[oof["fold_id"] == fold_id]
            n_groups = int(subset[SOURCE_GROUP_COLUMN].nunique())
            for label in OFFICIAL_LABELS:
                values = _fold_metric_values(
                    truth=subset[f"true_{label}"].to_numpy(dtype=int),
                    probability=subset[f"prob_{label}"].to_numpy(dtype=float),
                    prediction=subset[f"pred_{label}"].to_numpy(dtype=int),
                )
                rows.append(
                    {
                        "model_id": model_id,
                        "fold_id": fold_id,
                        "label": label,
                        "test_perfume_count": int(len(subset)),
                        "test_source_group_count": n_groups,
                        **values,
                        "threshold": float(threshold),
                    }
                )
    columns = [
        "model_id",
        "fold_id",
        "label",
        "test_perfume_count",
        "test_source_group_count",
        "support_pos",
        "support_neg",
        "threshold",
        "f1",
        "roc_auc",
        "average_precision",
        "roc_auc_defined",
        "average_precision_defined",
    ]
    return pd.DataFrame(rows)[columns]


def _weighted_label_metrics(
    *,
    truth: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, float | None]:
    positive_weight = float(weights[truth == 1].sum())
    negative_weight = float(weights[truth == 0].sum())
    return {
        "f1": float(f1_score(truth, prediction, sample_weight=weights, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(truth, probability, sample_weight=weights))
            if positive_weight > 0.0 and negative_weight > 0.0
            else None
        ),
        "average_precision": (
            float(average_precision_score(truth, probability, sample_weight=weights))
            if positive_weight > 0.0
            else None
        ),
    }


def build_component_weighted_model_metrics(
    inputs: ValidatedInputs,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Compute pooled metrics giving every source component total weight one."""

    size_by_id = inputs.outer.set_index("perfume_id")["source_group_size"]
    component_weights = 1.0 / inputs.features["perfume_id"].map(size_by_id).to_numpy(dtype=float)
    rows: list[Dict[str, Any]] = []
    for model_id in MODEL_IDS:
        oof = inputs.oof_by_model[model_id]
        f1_values: list[float] = []
        roc_values: list[float] = []
        ap_values: list[float] = []
        for label in OFFICIAL_LABELS:
            values = _weighted_label_metrics(
                truth=oof[f"true_{label}"].to_numpy(dtype=int),
                probability=oof[f"prob_{label}"].to_numpy(dtype=float),
                prediction=oof[f"pred_{label}"].to_numpy(dtype=int),
                weights=component_weights,
            )
            f1_values.append(float(values["f1"]))
            if values["roc_auc"] is not None:
                roc_values.append(float(values["roc_auc"]))
            if values["average_precision"] is not None:
                ap_values.append(float(values["average_precision"]))
        rows.append(
            {
                "model_id": model_id,
                "n_perfumes": int(len(inputs.features)),
                "n_source_components": int(inputs.outer[SOURCE_GROUP_COLUMN].nunique()),
                "component_weight_total": float(component_weights.sum()),
                "threshold": float(threshold),
                "sample_weight_method": COMPONENT_WEIGHT_METHOD,
                "label_aggregation": "unweighted mean across seven official labels",
                "macro_f1": float(np.mean(f1_values)),
                "macro_roc_auc": float(np.mean(roc_values)) if roc_values else None,
                "macro_roc_auc_defined_label_count": len(roc_values),
                "macro_average_precision": float(np.mean(ap_values)) if ap_values else None,
                "macro_average_precision_defined_label_count": len(ap_values),
            }
        )
    return pd.DataFrame(rows)


def _component_confusion_contributions(
    inputs: ValidatedInputs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return per-component fractional TP/FP/FN contributions by model/label."""

    component_ids = sorted(inputs.outer[SOURCE_GROUP_COLUMN].unique().tolist())
    component_index = {group_id: index for index, group_id in enumerate(component_ids)}
    group_sizes = inputs.outer.groupby(SOURCE_GROUP_COLUMN)["perfume_id"].size().to_dict()
    shape = (len(MODEL_IDS), len(OFFICIAL_LABELS), len(component_ids))
    tp = np.zeros(shape, dtype=float)
    fp = np.zeros(shape, dtype=float)
    fn = np.zeros(shape, dtype=float)

    for model_index, model_id in enumerate(MODEL_IDS):
        oof = inputs.oof_by_model[model_id]
        group_indices = oof[SOURCE_GROUP_COLUMN].map(component_index).to_numpy(dtype=int)
        row_weights = oof[SOURCE_GROUP_COLUMN].map(
            {group_id: 1.0 / int(size) for group_id, size in group_sizes.items()}
        ).to_numpy(dtype=float)
        for label_index, label in enumerate(OFFICIAL_LABELS):
            truth = oof[f"true_{label}"].to_numpy(dtype=int)
            prediction = oof[f"pred_{label}"].to_numpy(dtype=int)
            np.add.at(
                tp[model_index, label_index],
                group_indices,
                row_weights * ((truth == 1) & (prediction == 1)),
            )
            np.add.at(
                fp[model_index, label_index],
                group_indices,
                row_weights * ((truth == 0) & (prediction == 1)),
            )
            np.add.at(
                fn[model_index, label_index],
                group_indices,
                row_weights * ((truth == 1) & (prediction == 0)),
            )
    return tp, fp, fn, component_ids


def _macro_f1_from_component_counts(
    component_draw_counts: np.ndarray,
    *,
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
) -> np.ndarray:
    """Return bootstrap macro-F1 values with shape ``(reps, models)``."""

    reps = component_draw_counts.shape[0]
    output = np.zeros((reps, len(MODEL_IDS)), dtype=float)
    for model_index in range(len(MODEL_IDS)):
        per_label: list[np.ndarray] = []
        for label_index in range(len(OFFICIAL_LABELS)):
            total_tp = component_draw_counts @ tp[model_index, label_index]
            total_fp = component_draw_counts @ fp[model_index, label_index]
            total_fn = component_draw_counts @ fn[model_index, label_index]
            denominator = (2.0 * total_tp) + total_fp + total_fn
            f1 = np.divide(
                2.0 * total_tp,
                denominator,
                out=np.zeros_like(total_tp),
                where=denominator > 0.0,
            )
            per_label.append(f1)
        output[:, model_index] = np.mean(np.column_stack(per_label), axis=1)
    return output


def build_paired_component_bootstrap(
    inputs: ValidatedInputs,
    *,
    seed: int,
    reps: int,
    confidence_level: float,
) -> pd.DataFrame:
    """Build deterministic paired component-bootstrap macro-F1 intervals."""

    if reps < 2:
        raise ValueError("bootstrap reps must be at least 2")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    tp, fp, fn, component_ids = _component_confusion_contributions(inputs)
    n_components = len(component_ids)
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, n_components, size=(reps, n_components))
    draw_counts = np.zeros((reps, n_components), dtype=np.int32)
    row_indices = np.repeat(np.arange(reps), n_components)
    np.add.at(draw_counts, (row_indices, sampled_indices.ravel()), 1)
    bootstrap_macro_f1 = _macro_f1_from_component_counts(
        draw_counts,
        tp=tp,
        fp=fp,
        fn=fn,
    )
    point_macro_f1 = _macro_f1_from_component_counts(
        np.ones((1, n_components), dtype=np.int32),
        tp=tp,
        fp=fp,
        fn=fn,
    )[0]

    alpha = 1.0 - confidence_level
    rows: list[Dict[str, Any]] = []
    for left_index, right_index in combinations(range(len(MODEL_IDS)), 2):
        differences = bootstrap_macro_f1[:, left_index] - bootstrap_macro_f1[:, right_index]
        ci_lower, ci_upper = np.quantile(
            differences,
            [alpha / 2.0, 1.0 - (alpha / 2.0)],
            method="linear",
        )
        point_difference = float(point_macro_f1[left_index] - point_macro_f1[right_index])
        rows.append(
            {
                "model_a": MODEL_IDS[left_index],
                "model_b": MODEL_IDS[right_index],
                "metric": "component_equal_macro_f1",
                "difference_definition": "model_a_minus_model_b",
                "point_estimate_difference": point_difference,
                "bootstrap_mean_difference": float(np.mean(differences)),
                "bootstrap_standard_error": float(np.std(differences, ddof=1)),
                "confidence_level": float(confidence_level),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "ci_excludes_zero": bool(ci_lower > 0.0 or ci_upper < 0.0),
                "bootstrap_reps": int(reps),
                "seed": int(seed),
                "n_source_components": int(n_components),
                "resampling_unit": SOURCE_GROUP_COLUMN,
                "method": BOOTSTRAP_METHOD,
            }
        )
    return pd.DataFrame(rows)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="",
        float_format="%.12g",
    )
    return text.encode("utf-8")


def _write_atomically(payloads: Mapping[Path, bytes]) -> None:
    temporary_paths: Dict[Path, Path] = {}
    try:
        for path, payload in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_name(f".{path.name}.tmp")
            temp_path.write_bytes(payload)
            temporary_paths[path] = temp_path
        for path, temp_path in temporary_paths.items():
            temp_path.replace(path)
    finally:
        for temp_path in temporary_paths.values():
            if temp_path.exists():
                temp_path.unlink()


def generate_source_grouped_diagnostics(
    *,
    feature_label_csv: Path,
    outer_manifest_csv: Path,
    inner_manifest_csv: Path,
    benchmark_root: Path,
    out_dir: Path,
    threshold: float = 0.3,
    seed: int = 42,
    bootstrap_reps: int = 2000,
    confidence_level: float = 0.95,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Validate inputs, build all diagnostics, and write them as one artifact set."""

    feature_label_csv = Path(feature_label_csv)
    outer_manifest_csv = Path(outer_manifest_csv)
    inner_manifest_csv = Path(inner_manifest_csv)
    benchmark_root = Path(benchmark_root)
    out_dir = Path(out_dir)
    output_paths = {key: out_dir / filename for key, filename in OUTPUT_FILENAMES.items()}
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite diagnostics artifact: {existing[0]}")

    inputs = load_and_validate_inputs(
        feature_label_csv=feature_label_csv,
        outer_manifest_csv=outer_manifest_csv,
        inner_manifest_csv=inner_manifest_csv,
        benchmark_root=benchmark_root,
        threshold=threshold,
    )
    frames = {
        "inner_fold_balance": build_inner_fold_balance(inputs),
        "outer_fold_per_label_metrics": build_outer_fold_per_label_metrics(
            inputs, threshold=threshold
        ),
        "component_weighted_model_metrics": build_component_weighted_model_metrics(
            inputs, threshold=threshold
        ),
        "paired_component_bootstrap": build_paired_component_bootstrap(
            inputs,
            seed=seed,
            reps=bootstrap_reps,
            confidence_level=confidence_level,
        ),
    }
    csv_payloads = {
        output_paths[key]: _csv_bytes(frame)
        for key, frame in frames.items()
    }
    input_paths: Dict[str, Path] = {
        "feature_label_csv": feature_label_csv,
        "outer_manifest_csv": outer_manifest_csv,
        "inner_manifest_csv": inner_manifest_csv,
    }
    for model_id, path in inputs.oof_paths.items():
        input_paths[f"oof_{model_id}"] = path

    implementation_path = Path(__file__).resolve()
    report: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "model_ids": list(MODEL_IDS),
        "official_labels": list(OFFICIAL_LABELS),
        "n_perfumes": int(len(inputs.features)),
        "n_source_components": int(inputs.outer[SOURCE_GROUP_COLUMN].nunique()),
        "outer_fold_ids": [1, 2, 3, 4, 5],
        "threshold": float(threshold),
        "bootstrap": {
            "seed": int(seed),
            "reps": int(bootstrap_reps),
            "confidence_level": float(confidence_level),
            "method": BOOTSTRAP_METHOD,
            "quantile_method": "linear percentile interval",
            "pairing": "identical component draws reused for every model",
        },
        "component_weighting": {
            "method": COMPONENT_WEIGHT_METHOD,
            "label_aggregation": "unweighted mean across seven official labels",
        },
        "undefined_metric_policy": {
            "roc_auc": "null when support_pos=0 or support_neg=0",
            "average_precision": "null when support_pos=0",
            "csv_representation": "blank with explicit defined flag",
        },
        "provenance": {
            "hash_algorithm": "sha256_binary_file_bytes",
            "inputs": {
                key: {"path": str(path), "sha256": sha256_file(path)}
                for key, path in input_paths.items()
            },
            "implementation": {
                "path": str(implementation_path),
                "sha256": sha256_file(implementation_path),
            },
            "outputs": {
                path.name: {"sha256": _sha256_bytes(payload), "bytes": len(payload)}
                for path, payload in csv_payloads.items()
            },
        },
        "artifact_rows": {key: int(len(frame)) for key, frame in frames.items()},
    }
    report_payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    payloads = dict(csv_payloads)
    payloads[output_paths["report"]] = report_payload
    _write_atomically(payloads)
    return output_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-label-csv", type=Path, required=True)
    parser.add_argument("--outer-manifest-csv", type=Path, required=True)
    parser.add_argument("--inner-manifest-csv", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = generate_source_grouped_diagnostics(
        feature_label_csv=args.feature_label_csv,
        outer_manifest_csv=args.outer_manifest_csv,
        inner_manifest_csv=args.inner_manifest_csv,
        benchmark_root=args.benchmark_root,
        out_dir=args.out_dir,
        threshold=args.threshold,
        seed=args.seed,
        bootstrap_reps=args.bootstrap_reps,
        confidence_level=args.confidence_level,
        overwrite=args.overwrite,
    )
    for key in OUTPUT_FILENAMES:
        print(f"{key}: {paths[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
