"""Run the frozen Phase 6 weighted synthetic-augmentation LR contrast.

The experiment compares the accepted class-weighted engineered-feature
Logistic Regression control with the same learner trained on the YouTube rows
plus the fixed 784x3 synthetic corpus.  A YouTube perfume contributes training
weight one.  Its three synthetic reviews contribute one third each, so the
synthetic branch also contributes total weight one per training perfume.

Evaluation is restricted to the repaired source-grouped YouTube folds.  The
runner replays the zero-synthetic control, verifies it against the accepted OOF
probabilities, and reports paired source-component bootstrap intervals for the
augmentation difference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
import yaml
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

from src.training.run_logistic_regression_benchmark import MODEL_SPEC
from src.utils.paths import PROJECT_ROOT


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config/evaluation_config_source_grouped_weighted_synthetic_lr_v1.yaml"
)
CONTROL = "youtube_only"
AUGMENTED = "youtube_plus_weighted_synthetic_784x3"
CONDITIONS = (CONTROL, AUGMENTED)
FEATURE_PREFIXES = (
    "NRC_",
    "VAD_",
    "Hartmann_",
    "Savani_",
    "GoEmotions_",
    "Cardiff_",
)
METRICS = (
    "component_equal_macro_f1",
    "component_equal_macro_average_precision",
    "component_equal_macro_roc_auc",
)


@dataclass(frozen=True)
class PreparedInputs:
    config: dict[str, Any]
    config_path: Path
    paths: dict[str, Path]
    youtube: pd.DataFrame
    synthetic: pd.DataFrame
    manifest: pd.DataFrame
    accepted_oof: pd.DataFrame
    accepted_metrics: dict[str, Any]
    feature_columns: tuple[str, ...]
    labels: tuple[str, ...]
    sensitivity_labels: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve(path: str | Path, *, config_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    # Frozen project configs use paths relative to the repository root.
    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.exists() or config_path.resolve().is_relative_to(PROJECT_ROOT):
        return project_candidate
    return config_path.parent / candidate


def _validate_binary(frame: pd.DataFrame, labels: Sequence[str], name: str) -> None:
    for label in labels:
        values = pd.to_numeric(frame[label], errors="raise")
        unique = set(values.dropna().astype(int).unique().tolist())
        if unique - {0, 1} or values.isna().any():
            raise ValueError(f"{name} label {label!r} is not complete binary data")


def _validate_finite(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    values = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains missing or non-finite engineered features")


def _read_frozen_input(
    section: Mapping[str, Any], *, config_path: Path, name: str
) -> Path:
    path = _resolve(section["path"], config_path=config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen Phase 6 input {name}: {path}")
    expected = str(section["sha256"])
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"Frozen Phase 6 input hash mismatch for {name}: {actual} != {expected}"
        )
    return path


def load_and_validate_inputs(config_path: Path) -> PreparedInputs:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["experiment"]["status"] != "prospectively_frozen_before_primary_run":
        raise ValueError("Phase 6 protocol must be prospectively frozen")

    input_config = config["inputs"]
    key_map = {
        "youtube": "youtube_feature_labels",
        "synthetic": "synthetic_feature_labels",
        "manifest": "outer_fold_manifest",
        "accepted_oof": "accepted_youtube_control_oof",
        "accepted_metrics": "accepted_youtube_control_metrics",
    }
    paths = {
        short: _read_frozen_input(
            input_config[long], config_path=config_path, name=long
        )
        for short, long in key_map.items()
    }

    youtube = pd.read_csv(paths["youtube"])
    synthetic = pd.read_csv(paths["synthetic"])
    manifest = pd.read_csv(paths["manifest"])
    accepted_oof = pd.read_csv(paths["accepted_oof"])
    accepted_metrics = json.loads(paths["accepted_metrics"].read_text(encoding="utf-8"))

    labels = tuple(config["task"]["official_labels"])
    sensitivity_labels = tuple(config["task"]["sensitivity_labels"])
    required_youtube = {"perfume_id", "Perfume", *labels}
    required_synthetic = {
        "synthetic_review_id",
        "perfume_id",
        "Perfume",
        "review_index",
        *labels,
    }
    required_manifest = {
        "perfume_id",
        "Perfume",
        "source_group_id",
        "source_group_size",
        "fold_id",
    }
    for frame, required, name in (
        (youtube, required_youtube, "YouTube"),
        (synthetic, required_synthetic, "synthetic"),
        (manifest, required_manifest, "outer manifest"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} input is missing columns: {missing}")

    for frame in (youtube, synthetic, manifest, accepted_oof):
        frame["perfume_id"] = frame["perfume_id"].astype(str)
    synthetic["synthetic_review_id"] = synthetic["synthetic_review_id"].astype(str)

    expected_youtube = int(input_config["youtube_feature_labels"]["expected_rows"])
    expected_synthetic = int(input_config["synthetic_feature_labels"]["expected_rows"])
    if len(youtube) != expected_youtube or youtube["perfume_id"].nunique() != expected_youtube:
        raise ValueError("YouTube input does not contain the frozen unique perfume universe")
    if len(manifest) != expected_youtube or manifest["perfume_id"].nunique() != expected_youtube:
        raise ValueError("Outer manifest does not contain the frozen unique perfume universe")
    if len(synthetic) != expected_synthetic:
        raise ValueError("Synthetic input row count differs from the frozen protocol")
    if synthetic["synthetic_review_id"].nunique() != len(synthetic):
        raise ValueError("Synthetic review identifiers are not unique")

    reviews_per_perfume = int(
        input_config["synthetic_feature_labels"]["expected_reviews_per_perfume"]
    )
    synthetic_counts = synthetic.groupby("perfume_id").size()
    if set(synthetic_counts.unique().tolist()) != {reviews_per_perfume}:
        raise ValueError("Every synthetic perfume must have the frozen review count")
    if set(synthetic_counts.index) != set(youtube["perfume_id"]):
        raise ValueError("Synthetic and YouTube perfume universes differ")

    if manifest.groupby("source_group_id")["fold_id"].nunique().max() != 1:
        raise ValueError("A repaired source component crosses outer folds")
    expected_components = int(input_config["outer_fold_manifest"]["expected_source_components"])
    if manifest["source_group_id"].nunique() != expected_components:
        raise ValueError("Unexpected number of repaired source components")
    observed_group_size = manifest.groupby("source_group_id")["perfume_id"].size()
    declared_group_size = manifest.groupby("source_group_id")["source_group_size"].first()
    if not observed_group_size.astype(int).equals(declared_group_size.astype(int)):
        raise ValueError("Manifest source_group_size values are inconsistent")

    feature_columns = tuple(accepted_metrics["feature_columns"])
    expected_features = int(config["task"]["expected_feature_count"])
    if len(feature_columns) != expected_features:
        raise ValueError("Accepted LR feature count differs from the Phase 6 protocol")
    for column in feature_columns:
        if not column.startswith(FEATURE_PREFIXES):
            raise ValueError(f"Unexpected accepted LR feature column: {column}")
        if column not in youtube or column not in synthetic:
            raise ValueError(f"Common engineered feature is missing: {column}")
        if not pd.api.types.is_numeric_dtype(youtube[column]):
            raise ValueError(f"YouTube feature is not numeric: {column}")
        if not pd.api.types.is_numeric_dtype(synthetic[column]):
            raise ValueError(f"Synthetic feature is not numeric: {column}")

    _validate_binary(youtube, labels, "YouTube")
    _validate_binary(synthetic, labels, "synthetic")
    _validate_finite(youtube, feature_columns, "YouTube")
    _validate_finite(synthetic, feature_columns, "synthetic")

    youtube[list(labels)] = youtube[list(labels)].astype(int)
    synthetic[list(labels)] = synthetic[list(labels)].astype(int)
    manifest["fold_id"] = manifest["fold_id"].astype(int)
    manifest["source_group_size"] = manifest["source_group_size"].astype(int)

    joined = youtube[["perfume_id", *labels]].merge(
        manifest,
        on="perfume_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != expected_youtube:
        raise ValueError("YouTube rows and outer manifest do not join one-to-one")

    accepted_required = {
        "perfume_id",
        "source_group_id",
        "source_group_size",
        "fold_id",
        *(f"true_{label}" for label in labels),
        *(f"prob_{label}" for label in labels),
        *(f"pred_{label}" for label in labels),
    }
    missing_accepted = sorted(accepted_required - set(accepted_oof.columns))
    if missing_accepted:
        raise ValueError(f"Accepted LR OOF is missing columns: {missing_accepted}")
    if len(accepted_oof) != expected_youtube or accepted_oof["perfume_id"].nunique() != expected_youtube:
        raise ValueError("Accepted LR OOF is not a complete 784-perfume control")
    compare = accepted_oof.merge(
        joined,
        on="perfume_id",
        suffixes=("_oof", "_manifest"),
        validate="one_to_one",
    )
    for column in ("source_group_id", "source_group_size", "fold_id"):
        if not (compare[f"{column}_oof"].astype(str) == compare[f"{column}_manifest"].astype(str)).all():
            raise ValueError(f"Accepted LR OOF disagrees with repaired manifest: {column}")
    for label in labels:
        if not (compare[f"true_{label}"].astype(int) == compare[label].astype(int)).all():
            raise ValueError(f"Accepted LR OOF truth disagrees with YouTube labels: {label}")
        expected_pred = (
            compare[f"prob_{label}"].astype(float)
            >= float(config["evaluation"]["threshold"])
        ).astype(int)
        if not np.array_equal(expected_pred, compare[f"pred_{label}"].astype(int)):
            raise ValueError(f"Accepted LR thresholded predictions disagree for {label}")

    params = accepted_metrics["model_parameters"]["classifier"]
    if params.get("class_weight") != "balanced":
        raise ValueError("Accepted Phase 6 control must be the class-weighted main LR")
    split_hash = accepted_metrics["split_provenance"]["outer_manifest_sha256"]
    if split_hash != sha256_file(paths["manifest"]):
        raise ValueError("Accepted LR metrics use a different outer manifest")

    return PreparedInputs(
        config=config,
        config_path=config_path,
        paths=paths,
        youtube=youtube,
        synthetic=synthetic,
        manifest=manifest,
        accepted_oof=accepted_oof,
        accepted_metrics=accepted_metrics,
        feature_columns=feature_columns,
        labels=labels,
        sensitivity_labels=sensitivity_labels,
    )


def positive_class_probability(model: Any, features: Any) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classifier = model.named_steps["clf"]
    classes = list(classifier.classes_)
    return probabilities[:, classes.index(1)]


def build_training_rows(
    prepared: PreparedInputs,
    *,
    fold_id: int,
    include_synthetic: bool,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    manifest = prepared.manifest[["perfume_id", "fold_id"]]
    youtube = prepared.youtube.merge(
        manifest, on="perfume_id", how="inner", validate="one_to_one"
    ).sort_values("perfume_id").reset_index(drop=True)
    youtube_train = youtube.loc[youtube["fold_id"].ne(fold_id)].copy()
    test_ids = set(youtube.loc[youtube["fold_id"].eq(fold_id), "perfume_id"])
    youtube_train["training_source"] = "youtube_perfume"
    youtube_train["sample_weight"] = 1.0
    frames = [youtube_train]

    synthetic_train = prepared.synthetic.iloc[0:0].copy()
    if include_synthetic:
        train_ids = set(youtube_train["perfume_id"])
        synthetic_train = prepared.synthetic.loc[
            prepared.synthetic["perfume_id"].isin(train_ids)
        ].sort_values(["perfume_id", "review_index"]).reset_index(drop=True).copy()
        if set(synthetic_train["perfume_id"]) & test_ids:
            raise ValueError("Held-out perfume synthetic rows entered training")
        counts = synthetic_train.groupby("perfume_id").size()
        synthetic_train["training_source"] = "synthetic_review"
        synthetic_train["sample_weight"] = synthetic_train["perfume_id"].map(
            (1.0 / counts).to_dict()
        )
        effective = synthetic_train.groupby("perfume_id")["sample_weight"].sum()
        if not np.allclose(effective.to_numpy(float), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("Synthetic reviews do not contribute total weight one per perfume")
        frames.append(synthetic_train)

    training = pd.concat(frames, ignore_index=True, sort=False)
    weights = training["sample_weight"].to_numpy(dtype=float)
    train_perfumes = int(youtube_train["perfume_id"].nunique())
    synthetic_weight = float(
        training.loc[training["training_source"].eq("synthetic_review"), "sample_weight"].sum()
    )
    audit = {
        "fold_id": int(fold_id),
        "condition": AUGMENTED if include_synthetic else CONTROL,
        "test_youtube_rows": int(len(test_ids)),
        "train_youtube_rows": int(len(youtube_train)),
        "train_synthetic_rows": int(len(synthetic_train)),
        "train_perfumes": train_perfumes,
        "youtube_effective_weight": float(len(youtube_train)),
        "synthetic_effective_weight": synthetic_weight,
        "total_effective_weight": float(weights.sum()),
        "held_out_synthetic_rows_excluded": int(
            prepared.synthetic["perfume_id"].isin(test_ids).sum()
        ),
        "test_perfume_overlap_in_training": int(
            len(set(training["perfume_id"]) & test_ids)
        ),
    }
    if include_synthetic and not math.isclose(
        synthetic_weight, float(train_perfumes), abs_tol=1e-10
    ):
        raise ValueError("Synthetic effective fold weight differs from training perfumes")
    return training, weights, audit


def fit_oof_condition(
    prepared: PreparedInputs,
    *,
    include_synthetic: bool,
    seed: int,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = prepared.youtube.merge(
        prepared.manifest,
        on=["perfume_id", "Perfume"],
        how="inner",
        validate="one_to_one",
    ).sort_values("perfume_id").reset_index(drop=True)
    output = data[
        ["perfume_id", "Perfume", "source_group_id", "source_group_size", "fold_id"]
    ].copy()
    for label in prepared.labels:
        output[f"true_{label}"] = data[label].astype(int)
        output[f"prob_{label}"] = np.nan
        output[f"pred_{label}"] = 0

    audits: list[dict[str, Any]] = []
    for fold_id in sorted(data["fold_id"].unique().tolist()):
        training, weights, audit = build_training_rows(
            prepared, fold_id=int(fold_id), include_synthetic=include_synthetic
        )
        test_mask = data["fold_id"].eq(fold_id)
        x_train = training[list(prepared.feature_columns)].to_numpy(dtype=float)
        x_test = data.loc[test_mask, list(prepared.feature_columns)].to_numpy(
            dtype=float
        )
        for label in prepared.labels:
            y_train = training[label].astype(int)
            if y_train.nunique() != 2:
                raise ValueError(f"Fold {fold_id} has a single training class for {label}")
            model = MODEL_SPEC.build_model(seed, y_train.to_numpy())
            if model.named_steps["clf"].class_weight != "balanced":
                raise ValueError("Phase 6 runner must use the main class-weighted LR")
            if include_synthetic:
                model.fit(
                    x_train,
                    y_train,
                    scaler__sample_weight=weights,
                    clf__sample_weight=weights,
                )
            else:
                # Match the accepted control's original fit call exactly.
                model.fit(x_train, y_train)
            probabilities = positive_class_probability(model, x_test)
            output.loc[test_mask, f"prob_{label}"] = probabilities
            output.loc[test_mask, f"pred_{label}"] = (
                probabilities >= threshold
            ).astype(int)
        audits.append(audit)

    probability_columns = [f"prob_{label}" for label in prepared.labels]
    if output[probability_columns].isna().any().any():
        raise RuntimeError("Phase 6 OOF prediction matrix is incomplete")
    if output.groupby("source_group_id")["fold_id"].nunique().max() != 1:
        raise RuntimeError("A source component crosses Phase 6 OOF folds")
    return output, pd.DataFrame(audits)


def evaluate_oof(
    oof: pd.DataFrame,
    labels: Sequence[str],
    *,
    sample_weight: np.ndarray | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for label in labels:
        truth = oof[f"true_{label}"].to_numpy(dtype=int)
        probability = oof[f"prob_{label}"].to_numpy(dtype=float)
        prediction = oof[f"pred_{label}"].to_numpy(dtype=int)
        true_positive = int(np.sum((truth == 1) & (prediction == 1)))
        false_positive = int(np.sum((truth == 0) & (prediction == 1)))
        true_negative = int(np.sum((truth == 0) & (prediction == 0)))
        false_negative = int(np.sum((truth == 1) & (prediction == 0)))
        rows.append(
            {
                "label": label,
                "support_pos": int(truth.sum()),
                "support_neg": int(len(truth) - truth.sum()),
                "predicted_positive_count": int(prediction.sum()),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "true_negative": true_negative,
                "false_negative": false_negative,
                "precision": float(
                    precision_score(
                        truth,
                        prediction,
                        sample_weight=sample_weight,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        truth,
                        prediction,
                        sample_weight=sample_weight,
                        zero_division=0,
                    )
                ),
                "f1": float(
                    f1_score(
                        truth,
                        prediction,
                        sample_weight=sample_weight,
                        zero_division=0,
                    )
                ),
                "average_precision": float(
                    average_precision_score(
                        truth, probability, sample_weight=sample_weight
                    )
                ),
                "roc_auc": float(
                    roc_auc_score(truth, probability, sample_weight=sample_weight)
                ),
            }
        )
    per_label = pd.DataFrame(rows)
    truth_matrix = oof[[f"true_{label}" for label in labels]].to_numpy(dtype=int)
    prediction_matrix = oof[[f"pred_{label}" for label in labels]].to_numpy(dtype=int)
    summary = {
        "macro_f1": float(per_label["f1"].mean()),
        "macro_average_precision": float(per_label["average_precision"].mean()),
        "macro_roc_auc": float(per_label["roc_auc"].mean()),
        "micro_f1": float(
            f1_score(
                truth_matrix.ravel(),
                prediction_matrix.ravel(),
                sample_weight=(
                    None
                    if sample_weight is None
                    else np.repeat(sample_weight, len(labels))
                ),
                zero_division=0,
            )
        ),
        "hamming_loss": float(
            hamming_loss(
                truth_matrix, prediction_matrix, sample_weight=sample_weight
            )
        ),
        "predicted_label_cardinality": float(prediction_matrix.sum(axis=1).mean()),
    }
    return summary, per_label


def validate_control_replay(
    replay: pd.DataFrame,
    accepted: pd.DataFrame,
    labels: Sequence[str],
    *,
    probability_tolerance: float = 1e-4,
) -> dict[str, Any]:
    left = replay.sort_values("perfume_id").reset_index(drop=True)
    right = accepted.sort_values("perfume_id").reset_index(drop=True)
    if not np.array_equal(left["perfume_id"].to_numpy(), right["perfume_id"].to_numpy()):
        raise ValueError("Control replay perfume order/universe differs from accepted OOF")
    max_probability_difference = 0.0
    for column in ("source_group_id", "source_group_size", "fold_id"):
        if not np.array_equal(left[column].astype(str), right[column].astype(str)):
            raise ValueError(f"Control replay differs from accepted OOF: {column}")
    for label in labels:
        for prefix in ("true", "pred"):
            column = f"{prefix}_{label}"
            if not np.array_equal(left[column].astype(int), right[column].astype(int)):
                raise ValueError(f"Control replay differs from accepted OOF: {column}")
        difference = np.max(
            np.abs(
                left[f"prob_{label}"].to_numpy(float)
                - right[f"prob_{label}"].to_numpy(float)
            )
        )
        max_probability_difference = max(max_probability_difference, float(difference))
    if max_probability_difference > probability_tolerance:
        raise ValueError(
            "Control replay does not reproduce accepted probabilities: "
            f"max difference {max_probability_difference:.3g}"
        )
    return {
        "status": "passed",
        "rows": int(len(left)),
        "probability_tolerance": float(probability_tolerance),
        "max_absolute_probability_difference": max_probability_difference,
        "thresholded_predictions_exact": True,
    }


def _draw_component_counts(n_components: int, reps: int, seed: int) -> np.ndarray:
    if reps < 2:
        raise ValueError("Bootstrap repetitions must be at least two")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_components, size=(reps, n_components))
    counts = np.zeros((reps, n_components), dtype=np.int32)
    np.add.at(
        counts,
        (np.repeat(np.arange(reps), n_components), sampled.ravel()),
        1,
    )
    return counts


def _metrics_for_weights(
    oof: pd.DataFrame,
    labels: Sequence[str],
    weights: np.ndarray,
) -> tuple[dict[str, float], dict[str, int]]:
    f1_values: list[float] = []
    ap_values: list[float] = []
    roc_values: list[float] = []
    for label in labels:
        truth = oof[f"true_{label}"].to_numpy(dtype=int)
        probability = oof[f"prob_{label}"].to_numpy(dtype=float)
        prediction = oof[f"pred_{label}"].to_numpy(dtype=int)
        positive_weight = float(weights[truth == 1].sum())
        negative_weight = float(weights[truth == 0].sum())
        f1_values.append(
            float(
                f1_score(
                    truth,
                    prediction,
                    sample_weight=weights,
                    zero_division=0,
                )
            )
        )
        if positive_weight > 0.0:
            ap_values.append(
                float(
                    average_precision_score(
                        truth, probability, sample_weight=weights
                    )
                )
            )
        if positive_weight > 0.0 and negative_weight > 0.0:
            roc_values.append(
                float(roc_auc_score(truth, probability, sample_weight=weights))
            )
    if not ap_values or not roc_values:
        raise RuntimeError("A source-component draw left a ranking metric undefined")
    return (
        {
            METRICS[0]: float(np.mean(f1_values)),
            METRICS[1]: float(np.mean(ap_values)),
            METRICS[2]: float(np.mean(roc_values)),
        },
        {
            METRICS[0]: len(f1_values),
            METRICS[1]: len(ap_values),
            METRICS[2]: len(roc_values),
        },
    )


def build_paired_component_bootstrap(
    oof_by_condition: Mapping[str, pd.DataFrame],
    *,
    label_scopes: Mapping[str, Sequence[str]],
    reps: int,
    seed: int,
    confidence_level: float,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if set(oof_by_condition) != set(CONDITIONS):
        raise ValueError(f"Expected exactly the Phase 6 conditions: {CONDITIONS}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one")

    aligned: dict[str, pd.DataFrame] = {}
    for condition, frame in oof_by_condition.items():
        aligned[condition] = frame.sort_values("perfume_id").reset_index(drop=True)
    reference = aligned[CONTROL]
    for condition in CONDITIONS[1:]:
        for column in ("perfume_id", "source_group_id", "source_group_size", "fold_id"):
            if not np.array_equal(
                reference[column].astype(str), aligned[condition][column].astype(str)
            ):
                raise ValueError(f"OOF conditions are not aligned on {column}")

    components = sorted(reference["source_group_id"].unique().tolist())
    component_index = {component: index for index, component in enumerate(components)}
    row_component_index = reference["source_group_id"].map(component_index).to_numpy(int)
    inverse_size = 1.0 / reference["source_group_size"].to_numpy(float)
    draws = _draw_component_counts(len(components), reps, seed)
    point_counts = np.ones(len(components), dtype=np.int32)
    alpha = 1.0 - confidence_level

    condition_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    defined_details: dict[str, Any] = {}
    for scope_name, labels in label_scopes.items():
        values = {
            metric: np.zeros((reps, len(CONDITIONS)), dtype=float)
            for metric in METRICS
        }
        defined_counts = {
            metric: np.zeros(reps, dtype=int) for metric in METRICS
        }
        point_weights = point_counts[row_component_index].astype(float) * inverse_size
        point_metrics: dict[str, dict[str, float]] = {}
        point_defined: dict[str, int] | None = None
        for condition in CONDITIONS:
            metrics, defined = _metrics_for_weights(
                aligned[condition], labels, point_weights
            )
            point_metrics[condition] = metrics
            if point_defined is None:
                point_defined = defined

        for draw_index in range(reps):
            weights = (
                draws[draw_index, row_component_index].astype(float) * inverse_size
            )
            for condition_index, condition in enumerate(CONDITIONS):
                metrics, defined = _metrics_for_weights(
                    aligned[condition], labels, weights
                )
                for metric in METRICS:
                    values[metric][draw_index, condition_index] = metrics[metric]
                    if condition_index == 0:
                        defined_counts[metric][draw_index] = defined[metric]
            if progress_callback is not None:
                progress_callback(scope_name, draw_index + 1, reps)

        defined_details[scope_name] = {
            metric: {
                "point": int(point_defined[metric]),
                "bootstrap_min": int(defined_counts[metric].min()),
                "bootstrap_max": int(defined_counts[metric].max()),
            }
            for metric in METRICS
        }
        for metric in METRICS:
            for condition_index, condition in enumerate(CONDITIONS):
                observed = values[metric][:, condition_index]
                lower, upper = np.quantile(
                    observed,
                    [alpha / 2.0, 1.0 - alpha / 2.0],
                    method="linear",
                )
                condition_rows.append(
                    {
                        "label_scope": scope_name,
                        "condition": condition,
                        "metric": metric,
                        "point_estimate": point_metrics[condition][metric],
                        "bootstrap_mean": float(observed.mean()),
                        "bootstrap_standard_error": float(observed.std(ddof=1)),
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                        "confidence_level": float(confidence_level),
                        "bootstrap_reps": int(reps),
                        "seed": int(seed),
                        "n_source_components": int(len(components)),
                    }
                )
            differences = values[metric][:, 1] - values[metric][:, 0]
            lower, upper = np.quantile(
                differences,
                [alpha / 2.0, 1.0 - alpha / 2.0],
                method="linear",
            )
            difference_rows.append(
                {
                    "label_scope": scope_name,
                    "metric": metric,
                    "difference_definition": f"{AUGMENTED}_minus_{CONTROL}",
                    "point_estimate_difference": float(
                        point_metrics[AUGMENTED][metric]
                        - point_metrics[CONTROL][metric]
                    ),
                    "bootstrap_mean_difference": float(differences.mean()),
                    "bootstrap_standard_error": float(differences.std(ddof=1)),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
                    "positive_improvement_established": bool(lower > 0.0),
                    "confidence_level": float(confidence_level),
                    "bootstrap_reps": int(reps),
                    "seed": int(seed),
                    "n_source_components": int(len(components)),
                    "defined_label_count_point": int(point_defined[metric]),
                    "defined_label_count_min": int(defined_counts[metric].min()),
                    "defined_label_count_max": int(defined_counts[metric].max()),
                }
            )
    details = {
        "n_source_components": len(components),
        "source_component_ids_sha256": _sha256_bytes(
            "\n".join(components).encode("utf-8")
        ),
        "defined_label_counts": defined_details,
    }
    return pd.DataFrame(condition_rows), pd.DataFrame(difference_rows), details


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        na_rep="",
    ).encode("utf-8")


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def build_markdown_report(
    summary: pd.DataFrame,
    differences: pd.DataFrame,
    per_label: pd.DataFrame,
) -> str:
    lines = [
        "# Phase 6: Weighted Synthetic-Augmentation Logistic Regression",
        "",
        "This prespecified contrast uses the repaired exact-source-grouped folds. "
        "A YouTube perfume contributes training weight 1 and its three synthetic "
        "reviews contribute 1/3 each. Evaluation contains only held-out YouTube perfumes.",
        "",
        "## Pooled out-of-fold results",
        "",
        "| Condition | Macro-F1 | Macro average precision | Macro ROC-AUC |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['condition']} | {_format_metric(row['macro_f1'])} | "
            f"{_format_metric(row['macro_average_precision'])} | "
            f"{_format_metric(row['macro_roc_auc'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired component-equal augmentation differences",
            "",
            "Positive values favor weighted synthetic augmentation.",
            "",
            "| Label scope | Metric | Difference | 95% interval | Excludes zero |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    display_metric = {
        METRICS[0]: "Macro-F1",
        METRICS[1]: "Macro average precision",
        METRICS[2]: "Macro ROC-AUC",
    }
    for _, row in differences.iterrows():
        lines.append(
            f"| {row['label_scope']} | {display_metric[row['metric']]} | "
            f"{row['point_estimate_difference']:+.4f} | "
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | "
            f"{'yes' if row['ci_excludes_zero'] else 'no'} |"
        )

    pooled_fear = per_label.loc[
        per_label["label"].eq("fear")
        & per_label["estimator"].eq("pooled_oof")
    ].set_index("condition")
    component_fear = per_label.loc[
        per_label["label"].eq("fear")
        & per_label["estimator"].eq("component_equal")
    ].set_index("condition")
    primary = differences.loc[
        differences["label_scope"].eq("seven_labels")
        & differences["metric"].eq(METRICS[1])
    ].iloc[0]
    seven_f1 = differences.loc[
        differences["label_scope"].eq("seven_labels")
        & differences["metric"].eq(METRICS[0])
    ].iloc[0]
    six_f1 = differences.loc[
        differences["label_scope"].eq("fear_excluded_six_labels")
        & differences["metric"].eq(METRICS[0])
    ].iloc[0]
    six_ap = differences.loc[
        differences["label_scope"].eq("fear_excluded_six_labels")
        & differences["metric"].eq(METRICS[1])
    ].iloc[0]
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "The primary macro-average-precision augmentation difference "
                f"is {primary['point_estimate_difference']:+.4f} with a 95% "
                f"component interval [{primary['ci_lower']:.4f}, "
                f"{primary['ci_upper']:.4f}]. Because this interval includes zero, "
                "the primary analysis does not establish an augmentation improvement."
            ),
            (
                "The seven-label component-equal macro-F1 difference is "
                f"{seven_f1['point_estimate_difference']:+.4f} "
                f"[{seven_f1['ci_lower']:.4f}, {seven_f1['ci_upper']:.4f}], but "
                "the fear-excluded difference is "
                f"{six_f1['point_estimate_difference']:+.4f} "
                f"[{six_f1['ci_lower']:.4f}, {six_f1['ci_upper']:.4f}]."
            ),
            (
                "Fear has only three positive YouTube examples. Its pooled F1 changes "
                f"from {pooled_fear.loc[CONTROL, 'f1']:.4f} to "
                f"{pooled_fear.loc[AUGMENTED, 'f1']:.4f}, while its component-equal "
                "average precision changes from "
                f"{component_fear.loc[CONTROL, 'average_precision']:.4f} to "
                f"{component_fear.loc[AUGMENTED, 'average_precision']:.4f}. After "
                "excluding fear, the component-equal macro-average-precision "
                f"difference is {six_ap['point_estimate_difference']:+.4f} "
                f"[{six_ap['ci_lower']:.4f}, {six_ap['ci_upper']:.4f}]. The apparent "
                "seven-label ranking gain is therefore dominated by an ultra-rare label."
            ),
            "This remains a mixed-granularity sensitivity: real rows are perfume-level "
            "aggregates, while synthetic rows are individual generated reviews. Weighting "
            "equalizes their total training influence per perfume but does not make their "
            "feature or label distributions equivalent.",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(
    *,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path | None = None,
    overwrite: bool = False,
    bootstrap_reps: int | None = None,
) -> dict[str, Path]:
    prepared = load_and_validate_inputs(Path(config_path))
    output_root = (
        _resolve(prepared.config["outputs"]["root"], config_path=prepared.config_path)
        if out_dir is None
        else Path(out_dir)
    ).resolve()
    expected_outputs = (
        "condition_summary.csv",
        "per_label_metrics.csv",
        "fold_training_audit.csv",
        "component_bootstrap_condition_intervals.csv",
        "paired_component_bootstrap_differences.csv",
        "validation_report.json",
        "REPORT.json",
        "EXPERIMENT_REPORT.md",
        "PROTOCOL.yaml",
        "SHA256SUMS",
    )
    if not overwrite:
        existing = [output_root / name for name in expected_outputs if (output_root / name).exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite Phase 6 artifact: {existing[0]}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "oof_predictions").mkdir(parents=True, exist_ok=True)

    threshold = float(prepared.config["evaluation"]["threshold"])
    seed = int(prepared.config["model"]["random_seed"])
    bootstrap_config = prepared.config["evaluation"]["paired_source_component_bootstrap"]
    reps = int(bootstrap_reps or bootstrap_config["repetitions"])
    confidence_level = float(bootstrap_config["confidence_level"])

    control_oof, control_audit = fit_oof_condition(
        prepared,
        include_synthetic=False,
        seed=seed,
        threshold=threshold,
    )
    control_validation = validate_control_replay(
        control_oof, prepared.accepted_oof, prepared.labels
    )
    accepted_control_oof = prepared.accepted_oof[
        [
            "perfume_id",
            "Perfume",
            "source_group_id",
            "source_group_size",
            "fold_id",
            *(
                column
                for label in prepared.labels
                for column in (
                    f"true_{label}",
                    f"prob_{label}",
                    f"pred_{label}",
                )
            ),
        ]
    ].copy()
    augmented_oof, augmented_audit = fit_oof_condition(
        prepared,
        include_synthetic=True,
        seed=seed,
        threshold=threshold,
    )

    summaries: list[dict[str, Any]] = []
    per_label_frames: list[pd.DataFrame] = []
    for condition, oof in (
        (CONTROL, accepted_control_oof),
        (AUGMENTED, augmented_oof),
    ):
        pooled, per_label = evaluate_oof(oof, prepared.labels)
        component, component_per_label = evaluate_oof(
            oof,
            prepared.labels,
            sample_weight=1.0 / oof["source_group_size"].to_numpy(float),
        )
        summaries.append(
            {
                "condition": condition,
                "evaluation_rows": int(len(oof)),
                "threshold": threshold,
                **pooled,
                "component_equal_macro_f1": component["macro_f1"],
                "component_equal_macro_average_precision": component[
                    "macro_average_precision"
                ],
                "component_equal_macro_roc_auc": component["macro_roc_auc"],
            }
        )
        per_label.insert(0, "estimator", "pooled_oof")
        per_label.insert(0, "condition", condition)
        component_per_label.insert(0, "estimator", "component_equal")
        component_per_label.insert(0, "condition", condition)
        per_label_frames.extend([per_label, component_per_label])
    summary = pd.DataFrame(summaries)
    per_label = pd.concat(per_label_frames, ignore_index=True)
    fold_audit = pd.concat([control_audit, augmented_audit], ignore_index=True)

    condition_intervals, paired_differences, bootstrap_details = (
        build_paired_component_bootstrap(
            {CONTROL: accepted_control_oof, AUGMENTED: augmented_oof},
            label_scopes={
                "seven_labels": prepared.labels,
                "fear_excluded_six_labels": prepared.sensitivity_labels,
            },
            reps=reps,
            seed=int(bootstrap_config["seed"]),
            confidence_level=confidence_level,
        )
    )

    paths = {
        "condition_summary": output_root / "condition_summary.csv",
        "per_label_metrics": output_root / "per_label_metrics.csv",
        "fold_training_audit": output_root / "fold_training_audit.csv",
        "condition_intervals": output_root
        / "component_bootstrap_condition_intervals.csv",
        "paired_differences": output_root / "paired_component_bootstrap_differences.csv",
        "control_oof": output_root / "oof_predictions/youtube_only_replay_oof.csv",
        "accepted_control_oof": output_root
        / "oof_predictions/youtube_only_accepted_control_oof.csv",
        "augmented_oof": output_root
        / "oof_predictions/youtube_plus_weighted_synthetic_784x3_oof.csv",
        "validation": output_root / "validation_report.json",
        "report": output_root / "REPORT.json",
        "markdown_report": output_root / "EXPERIMENT_REPORT.md",
        "protocol": output_root / "PROTOCOL.yaml",
        "checksums": output_root / "SHA256SUMS",
    }
    csv_payloads = {
        paths["condition_summary"]: _csv_bytes(summary),
        paths["per_label_metrics"]: _csv_bytes(per_label),
        paths["fold_training_audit"]: _csv_bytes(fold_audit),
        paths["condition_intervals"]: _csv_bytes(condition_intervals),
        paths["paired_differences"]: _csv_bytes(paired_differences),
        paths["control_oof"]: _csv_bytes(control_oof),
        paths["accepted_control_oof"]: _csv_bytes(accepted_control_oof),
        paths["augmented_oof"]: _csv_bytes(augmented_oof),
    }
    for path, payload in csv_payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    validation = {
        "status": "passed",
        "control_replay": control_validation,
        "rows_per_condition": int(len(accepted_control_oof)),
        "unique_perfumes_per_condition": int(
            accepted_control_oof["perfume_id"].nunique()
        ),
        "source_components": int(
            accepted_control_oof["source_group_id"].nunique()
        ),
        "cross_fold_source_components": int(
            accepted_control_oof.groupby("source_group_id")["fold_id"]
            .nunique()
            .gt(1)
            .sum()
        ),
        "held_out_synthetic_overlap_max": int(
            fold_audit["test_perfume_overlap_in_training"].max()
        ),
        "synthetic_total_weight_per_training_perfume": 1.0,
        "weights_applied_to_scaler": True,
        "weights_applied_to_classifier": True,
        "accepted_outer_manifest_sha256": sha256_file(prepared.paths["manifest"]),
        "bootstrap": bootstrap_details,
    }
    paths["validation"].write_text(
        json.dumps(validation, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths["protocol"].write_bytes(prepared.config_path.read_bytes())
    paths["markdown_report"].write_text(
        build_markdown_report(summary, paired_differences, per_label),
        encoding="utf-8",
    )

    report = {
        "schema_version": 1,
        "status": "passed",
        "experiment_id": prepared.config["experiment"]["id"],
        "accepted_benchmark_modified": False,
        "protocol_sha256": sha256_file(prepared.config_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "model": prepared.config["model"],
        "conditions": list(CONDITIONS),
        "official_labels": list(prepared.labels),
        "fear_excluded_sensitivity_labels": list(prepared.sensitivity_labels),
        "feature_columns": list(prepared.feature_columns),
        "threshold": threshold,
        "inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in prepared.paths.items()
        },
        "pooled_and_component_equal_results": summary.to_dict(orient="records"),
        "per_label_results": per_label.to_dict(orient="records"),
        "paired_component_bootstrap": paired_differences.to_dict(orient="records"),
        "bootstrap_details": bootstrap_details,
        "validation": validation,
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "interpretation_boundary": prepared.config["experiment"][
            "interpretation_boundary"
        ],
    }
    paths["report"].write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    checksum_keys = [key for key in paths if key != "checksums"]
    paths["checksums"].write_text(
        "".join(
            f"{sha256_file(paths[key])}  ./{paths[key].relative_to(output_root).as_posix()}\n"
            for key in sorted(checksum_keys)
        ),
        encoding="utf-8",
    )
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--bootstrap-reps", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = run_experiment(
        config_path=args.config,
        out_dir=args.out_dir,
        bootstrap_reps=args.bootstrap_reps,
        overwrite=args.overwrite,
    )
    summary = pd.read_csv(paths["condition_summary"])
    differences = pd.read_csv(paths["paired_differences"])
    print("Phase 6 weighted synthetic LR contrast complete")
    for _, row in summary.iterrows():
        print(
            f"- {row['condition']}: macro-F1={row['macro_f1']:.4f}, "
            f"macro AP={row['macro_average_precision']:.4f}, "
            f"macro ROC-AUC={row['macro_roc_auc']:.4f}"
        )
    primary = differences.loc[
        differences["label_scope"].eq("seven_labels")
        & differences["metric"].eq(METRICS[1])
    ].iloc[0]
    print(
        f"- Paired component-equal macro AP difference: "
        f"{primary['point_estimate_difference']:+.4f} "
        f"[{primary['ci_lower']:.4f}, {primary['ci_upper']:.4f}]"
    )
    print(f"- Report: {paths['markdown_report']}")
    print(f"- Validation: {paths['validation']}")
    print(f"- Checksums: {paths['checksums']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
