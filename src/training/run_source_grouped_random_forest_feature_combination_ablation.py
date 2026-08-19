"""Run the formal source-grouped Random Forest feature-combination ablation.

The accepted 58-feature Random Forest model is replayed as a fail-closed gate.  Six
advisor-requested representations are then evaluated with the same five outer
folds, seven labels, class-balance rule, model settings, and threshold.  Each
fold-label fit is checkpointed atomically; rerunning the normal command resumes
automatically.  The first termination signal requests a pause after the active
fit has been saved.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from tqdm.auto import tqdm
import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.run_source_grouped_feature_ablation import (
    build_bootstrap_tables,
    build_feature_families,
)
from src.training.run_source_grouped_weighted_synthetic_lr import (
    _csv_bytes,
    _read_frozen_input,
    _resolve,
    _validate_binary,
    _validate_finite,
    evaluate_oof,
    positive_class_probability,
    sha256_file,
    validate_control_replay,
)
from src.training.run_source_grouped_weighted_synthetic_random_forest import (
    PauseController,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    load_task_checkpoint,
    save_task_checkpoint,
    stable_json_sha256,
    utc_text,
)
from src.training.run_random_forest_benchmark import MODEL_SPEC
from src.training.source_grouped_folds import SOURCE_GROUP_COLUMN
from src.utils.paths import PROJECT_ROOT


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config/evaluation_config_source_grouped_random_forest_feature_combination_ablation_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/source_grouped_random_forest_feature_combination_ablation_v1"
)
BASE_RUNNER = PROJECT_ROOT / "src/training/run_random_forest_benchmark.py"
FEATURE_ABLATION_HELPER = (
    PROJECT_ROOT / "src/training/run_source_grouped_feature_ablation.py"
)
WEIGHTED_LR_HELPER = (
    PROJECT_ROOT / "src/training/run_source_grouped_weighted_synthetic_lr.py"
)
WEIGHTED_RANDOM_FOREST_HELPER = (
    PROJECT_ROOT / "src/training/run_source_grouped_weighted_synthetic_random_forest.py"
)
SOURCE_GROUPED_FOLDS_HELPER = PROJECT_ROOT / "src/training/source_grouped_folds.py"
ACCEPTED_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "outputs/gemini_browser_og784_source_grouped_v1"
)
FEATURE_BRANCH_ROOT = PROJECT_ROOT / "outputs/gemini_browser_og784"
FULL_VARIANT = "full_hybrid"
COMBINATION_VARIANTS = (
    "nrc_only",
    "vad_only",
    "transformer_only",
    "nrc_plus_vad",
    "nrc_plus_transformer",
    "vad_plus_transformer",
)
ALL_VARIANTS = (FULL_VARIANT, *COMBINATION_VARIANTS)
VARIANT_LABELS = {
    FULL_VARIANT: "Full 58-feature control",
    "nrc_only": "NRC only",
    "vad_only": "VAD only",
    "transformer_only": "Transformer only",
    "nrc_plus_vad": "NRC + VAD",
    "nrc_plus_transformer": "NRC + Transformer",
    "vad_plus_transformer": "VAD + Transformer",
}
CHECKPOINT_SCHEMA_VERSION = 1
PROGRESS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FitTask:
    condition: str
    fold_id: int
    label: str


@dataclass
class PreparedInputs:
    config: dict[str, Any]
    config_path: Path
    paths: dict[str, Path]
    data: pd.DataFrame
    manifest: pd.DataFrame
    accepted_oof: pd.DataFrame
    accepted_metrics: dict[str, Any]
    feature_columns: tuple[str, ...]
    families: OrderedDict[str, list[str]]
    variants: OrderedDict[str, list[str]]
    labels: tuple[str, ...]
    sensitivity_labels: tuple[str, ...]
    protected_checksums_sha256: str
    accepted_manifest_validation: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _frozen_paths(config: Mapping[str, Any], config_path: Path) -> dict[str, Path]:
    keys = {
        "base_config": "accepted_benchmark_config",
        "base_runner": "accepted_random_forest_runner",
        "feature_ablation_helper": "feature_ablation_helper",
        "weighted_lr_helper": "weighted_lr_helper",
        "weighted_rf_helper": "weighted_random_forest_checkpoint_helper",
        "source_grouped_folds_helper": "source_grouped_folds_helper",
        "data": "youtube_feature_labels",
        "manifest": "outer_fold_manifest",
        "source_audit": "source_group_audit",
        "accepted_oof": "accepted_random_forest_oof",
        "accepted_metrics": "accepted_random_forest_metrics",
        "accepted_checksums": "accepted_benchmark_checksums",
    }
    return {
        short: _read_frozen_input(
            config["inputs"][long], config_path=config_path, name=long
        )
        for short, long in keys.items()
    }


def _paths_collide(candidate: Path, protected: Path) -> bool:
    candidate = candidate.resolve()
    protected = protected.resolve()
    return (
        candidate == protected
        or protected in candidate.parents
        or candidate in protected.parents
    )


def validate_output_root(output_root: Path, prepared: PreparedInputs) -> Path:
    """Reject roots that could overwrite any accepted input or input branch."""

    output_root = Path(output_root).resolve()
    protected = [
        ACCEPTED_EXPERIMENT_ROOT,
        FEATURE_BRANCH_ROOT,
        prepared.config_path,
        *prepared.paths.values(),
    ]
    for path in protected:
        if _paths_collide(output_root, path):
            raise ValueError(
                "Random Forest ablation output root collides with a protected input: "
                f"{output_root} versus {Path(path).resolve()}"
            )
    return output_root


def _checksum_file_set(output_root: Path) -> set[Path]:
    """Return immutable formal artifacts; checkpoints/progress stay resumable."""

    output_root = output_root.resolve()
    return {
        path.resolve()
        for path in output_root.rglob("*")
        if path.is_file()
        and path.name not in {"SHA256SUMS", "progress.json"}
        and "checkpoints" not in path.relative_to(output_root).parts
    }


def verify_completed_output(output_root: Path, checksum_path: Path) -> dict[str, Any]:
    """Fail closed unless the completed artifact set exactly matches SHA256SUMS."""

    output_root = output_root.resolve()
    checksum_path = checksum_path.resolve()
    if not checksum_path.is_file():
        raise ValueError("Completed output is missing SHA256SUMS")
    declared: dict[Path, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed SHA256SUMS line: {line!r}") from exc
        if len(digest) != 64:
            raise ValueError(f"Malformed SHA-256 digest for {relative}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"Malformed SHA-256 digest for {relative}") from exc
        candidate = (output_root / relative).resolve()
        if output_root not in candidate.parents:
            raise ValueError(f"Unsafe SHA256SUMS path: {relative}")
        if candidate in declared:
            raise ValueError(f"Duplicate SHA256SUMS path: {relative}")
        declared[candidate] = digest
    actual = _checksum_file_set(output_root)
    if set(declared) != actual:
        missing = sorted(str(path.relative_to(output_root)) for path in set(declared) - actual)
        extras = sorted(str(path.relative_to(output_root)) for path in actual - set(declared))
        raise ValueError(
            "Completed output file set differs from SHA256SUMS: "
            f"missing={missing}, extras={extras}"
        )
    for path, digest in declared.items():
        if sha256_file(path) != digest:
            raise ValueError(
                "Completed output checksum mismatch: "
                f"{path.relative_to(output_root)}"
            )
    return {
        "status": "passed",
        "artifact_count": len(declared),
        "checksum_manifest_sha256": sha256_file(checksum_path),
    }


def verify_accepted_benchmark_manifest(checksum_path: Path) -> dict[str, Any]:
    """Exhaustively verify the protected accepted benchmark's owned file set."""

    checksum_path = Path(checksum_path).resolve()
    root = checksum_path.parent
    if not checksum_path.is_file():
        raise ValueError("Accepted benchmark is missing SHA256SUMS")
    declared: dict[Path, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed accepted SHA256SUMS line: {line!r}") from exc
        if len(digest) != 64:
            raise ValueError(f"Malformed accepted SHA-256 digest for {relative}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"Malformed accepted SHA-256 digest for {relative}") from exc
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"Unsafe accepted SHA256SUMS path: {relative}")
        if path in declared:
            raise ValueError(f"Duplicate accepted SHA256SUMS path: {relative}")
        declared[path] = digest
    actual = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != checksum_path
    }
    if set(declared) != actual:
        missing = sorted(str(path.relative_to(root)) for path in set(declared) - actual)
        extras = sorted(str(path.relative_to(root)) for path in actual - set(declared))
        raise ValueError(
            "Accepted benchmark file set differs from SHA256SUMS: "
            f"missing={missing}, extras={extras}"
        )
    for path, digest in declared.items():
        if sha256_file(path) != digest:
            raise ValueError(
                "Accepted benchmark checksum mismatch: "
                f"{path.relative_to(root)}"
            )
    return {
        "status": "passed",
        "root": str(root),
        "artifact_count": len(declared),
        "checksum_manifest_sha256": sha256_file(checksum_path),
    }


def build_feature_variants(
    feature_columns: Sequence[str],
    families: Mapping[str, Sequence[str]],
    config: Mapping[str, Any],
) -> OrderedDict[str, list[str]]:
    """Build the frozen seven representations in canonical column order."""

    canonical = list(feature_columns)

    def keeping(names: Sequence[str]) -> list[str]:
        allowed = {column for name in names for column in families[name]}
        return [column for column in canonical if column in allowed]

    definitions: OrderedDict[str, tuple[str, ...]] = OrderedDict(
        [
            (FULL_VARIANT, ("NRC", "VAD", "Hartmann", "Savani", "GoEmotions", "Cardiff")),
            ("nrc_only", ("NRC",)),
            ("vad_only", ("VAD",)),
            ("transformer_only", ("Hartmann", "Savani", "GoEmotions", "Cardiff")),
            ("nrc_plus_vad", ("NRC", "VAD")),
            ("nrc_plus_transformer", ("NRC", "Hartmann", "Savani", "GoEmotions", "Cardiff")),
            ("vad_plus_transformer", ("VAD", "Hartmann", "Savani", "GoEmotions", "Cardiff")),
        ]
    )
    if tuple(definitions) != ALL_VARIANTS:
        raise RuntimeError("Internal Random Forest variant order changed")
    configured = config["variants"]
    if tuple(configured) != ALL_VARIANTS:
        raise ValueError("Frozen variant order or membership keys changed")

    variants: OrderedDict[str, list[str]] = OrderedDict()
    for variant, family_names in definitions.items():
        if tuple(configured[variant]["families"]) != family_names:
            raise ValueError(f"Frozen family membership changed for {variant}")
        columns = keeping(family_names)
        expected_count = int(configured[variant]["feature_count"])
        if len(columns) != expected_count:
            raise ValueError(
                f"Frozen feature count changed for {variant}: {len(columns)} != {expected_count}"
            )
        variants[variant] = columns
    return variants


def accepted_control_columns(labels: Sequence[str]) -> list[str]:
    return [
        "perfume_id",
        "Perfume",
        SOURCE_GROUP_COLUMN,
        "source_group_size",
        "fold_id",
        *(
            column
            for label in labels
            for column in (f"true_{label}", f"prob_{label}", f"pred_{label}")
        ),
    ]


def _validate_model_contract(
    config: Mapping[str, Any], accepted_metrics: Mapping[str, Any]
) -> None:
    model = config["model"]
    expected_classifier = {
        "type": "RandomForestClassifier",
        "n_estimators": int(model["n_estimators"]),
        "class_weight": str(model["class_weight"]),
        "n_jobs": int(model["n_jobs"]),
    }
    if accepted_metrics["model_parameters"]["classifier"] != expected_classifier:
        raise ValueError("Accepted Random Forest classifier parameters changed")
    if accepted_metrics["model_parameters"].get("imputer_strategy") != "median":
        raise ValueError("Accepted Random Forest imputer is not median")
    if MODEL_SPEC.model_params["classifier"] != expected_classifier:
        raise ValueError("Current accepted Random Forest runner parameters changed")
    if MODEL_SPEC.model_params.get("imputer_strategy") != "median":
        raise ValueError("Current accepted Random Forest runner imputer changed")
    if model.get("training_sample_weight") != "none":
        raise ValueError("Random Forest combination ablation must not use sample weights")


def load_and_validate_inputs(config_path: Path = DEFAULT_CONFIG) -> PreparedInputs:
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    if config["experiment"]["status"] != "frozen_before_first_formal_run":
        raise ValueError("Random Forest feature-combination protocol is not frozen")
    paths = _frozen_paths(config, config_path)
    accepted_manifest_validation = verify_accepted_benchmark_manifest(
        paths["accepted_checksums"]
    )
    data_raw = pd.read_csv(paths["data"], keep_default_na=False)
    manifest = pd.read_csv(paths["manifest"], keep_default_na=False)
    accepted_raw = pd.read_csv(paths["accepted_oof"], keep_default_na=False)
    accepted_metrics = json.loads(paths["accepted_metrics"].read_text(encoding="utf-8"))
    accepted_config = _load_yaml(paths["base_config"])
    source_audit = json.loads(paths["source_audit"].read_text(encoding="utf-8"))
    labels = tuple(config["task"]["official_labels"])
    sensitivity_labels = tuple(config["task"]["sensitivity_labels"])
    accepted_labels = tuple(accepted_config["targets"]["official_emotion_columns"])
    accepted_excluded_mapping = dict(
        accepted_config["targets"]["excluded_emotion_columns"]
    )
    accepted_excluded = tuple(accepted_excluded_mapping)
    if labels != accepted_labels:
        raise ValueError("Official labels differ from the accepted benchmark config")
    if accepted_excluded != (str(config["task"]["excluded_label"]),):
        raise ValueError("Excluded labels differ from the accepted benchmark config")
    if sensitivity_labels != tuple(label for label in labels if label != "fear"):
        raise ValueError("Fear-excluded sensitivity label scope changed")

    required_data = {"perfume_id", "Perfume", *labels}
    required_manifest = {
        "perfume_id",
        "Perfume",
        SOURCE_GROUP_COLUMN,
        "source_group_size",
        "fold_id",
    }
    for frame, required, name in (
        (data_raw, required_data, "feature table"),
        (manifest, required_manifest, "outer manifest"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Frozen {name} lacks columns: {missing}")
        frame["perfume_id"] = frame["perfume_id"].astype(str)
    accepted_raw["perfume_id"] = accepted_raw["perfume_id"].astype(str)

    expected_rows = int(config["inputs"]["youtube_feature_labels"]["expected_rows"])
    if len(data_raw) != expected_rows or data_raw["perfume_id"].nunique() != expected_rows:
        raise ValueError("Feature table is not the frozen 784-perfume universe")
    if len(manifest) != expected_rows or manifest["perfume_id"].nunique() != expected_rows:
        raise ValueError("Manifest is not the frozen 784-perfume universe")
    manifest["fold_id"] = pd.to_numeric(manifest["fold_id"], errors="raise").astype(int)
    manifest["source_group_size"] = pd.to_numeric(
        manifest["source_group_size"], errors="raise"
    ).astype(int)
    if sorted(manifest["fold_id"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise ValueError("Frozen outer folds changed")
    if manifest.groupby(SOURCE_GROUP_COLUMN)["fold_id"].nunique().max() != 1:
        raise ValueError("A source component crosses outer folds")
    if manifest[SOURCE_GROUP_COLUMN].nunique() != int(
        config["inputs"]["outer_fold_manifest"]["expected_source_components"]
    ):
        raise ValueError("Frozen source-component count changed")
    observed_sizes = manifest.groupby(SOURCE_GROUP_COLUMN)["perfume_id"].size()
    declared_sizes = manifest.groupby(SOURCE_GROUP_COLUMN)["source_group_size"].first()
    if not observed_sizes.astype(int).equals(declared_sizes.astype(int)):
        raise ValueError("Manifest source_group_size values are inconsistent")

    if accepted_metrics.get("model_id") != "random_forest":
        raise ValueError("Accepted metrics are not Random Forest")
    if accepted_metrics.get("excluded_target_columns") != accepted_excluded_mapping:
        raise ValueError("Accepted Random Forest excluded-label contract changed")
    if accepted_metrics.get("config_sha256") != sha256_file(paths["base_config"]):
        raise ValueError("Accepted Random Forest config provenance changed")
    if accepted_metrics.get("dataset_sha256") != sha256_file(paths["data"]):
        raise ValueError("Accepted Random Forest dataset provenance changed")
    feature_columns = tuple(accepted_metrics["feature_columns"])
    if len(feature_columns) != int(config["task"]["expected_feature_count"]):
        raise ValueError("Accepted Random Forest feature count changed")
    for column in feature_columns:
        if column not in data_raw:
            raise ValueError(f"Frozen feature is missing: {column}")
        if not pd.api.types.is_numeric_dtype(data_raw[column]):
            raise ValueError(f"Frozen feature is not numeric: {column}")
    _validate_binary(data_raw, labels, "YouTube")
    _validate_finite(data_raw, feature_columns, "YouTube")
    data_raw[list(labels)] = data_raw[list(labels)].astype(int)

    expected_family_counts = {
        key: int(value) for key, value in config["task"]["feature_family_counts"].items()
    }
    families = build_feature_families(feature_columns, enforce_expected_counts=True)
    if {key: len(value) for key, value in families.items()} != expected_family_counts:
        raise ValueError("Frozen feature-family counts changed")
    variants = build_feature_variants(feature_columns, families, config)

    data = data_raw.merge(
        manifest[["perfume_id", "Perfume", SOURCE_GROUP_COLUMN, "source_group_size", "fold_id"]],
        on=["perfume_id", "Perfume"],
        how="inner",
        validate="one_to_one",
    ).sort_values("perfume_id").reset_index(drop=True)
    if len(data) != expected_rows:
        raise ValueError("Feature table and outer manifest do not join one-to-one")

    columns = accepted_control_columns(labels)
    missing_accepted = sorted(set(columns) - set(accepted_raw.columns))
    if missing_accepted:
        raise ValueError(f"Accepted Random Forest OOF lacks columns: {missing_accepted}")
    accepted_oof = accepted_raw[columns].sort_values("perfume_id").reset_index(drop=True)
    if len(accepted_oof) != expected_rows or accepted_oof["perfume_id"].nunique() != expected_rows:
        raise ValueError("Accepted Random Forest OOF is incomplete")
    if not np.array_equal(accepted_oof["perfume_id"].to_numpy(), data["perfume_id"].to_numpy()):
        raise ValueError("Accepted Random Forest OOF perfume order/universe changed")
    for column in (SOURCE_GROUP_COLUMN, "source_group_size", "fold_id"):
        if not np.array_equal(accepted_oof[column].astype(str), data[column].astype(str)):
            raise ValueError(f"Accepted Random Forest OOF disagrees on {column}")
    threshold = float(config["evaluation"]["threshold"])
    for label in labels:
        if not np.array_equal(accepted_oof[f"true_{label}"].astype(int), data[label].astype(int)):
            raise ValueError(f"Accepted Random Forest truth disagrees for {label}")
        probability = accepted_oof[f"prob_{label}"].to_numpy(float)
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError(f"Accepted Random Forest probabilities are invalid for {label}")
        if not np.array_equal(
            accepted_oof[f"pred_{label}"].astype(int), (probability >= threshold).astype(int)
        ):
            raise ValueError(f"Accepted Random Forest predictions use another threshold for {label}")

    if tuple(accepted_metrics["target_columns"]) != labels:
        raise ValueError("Accepted Random Forest label order changed")
    if not math.isclose(float(accepted_metrics["threshold"]), threshold):
        raise ValueError("Accepted Random Forest threshold changed")
    if int(accepted_metrics["model_seed"]) != int(config["model"]["random_seed"]):
        raise ValueError("Accepted Random Forest seed changed")
    split_provenance = accepted_metrics.get("split_provenance", {})
    if split_provenance.get("outer_manifest_sha256") != sha256_file(paths["manifest"]):
        raise ValueError("Accepted Random Forest metrics use another outer manifest")
    if split_provenance.get("source_audit_sha256") != sha256_file(paths["source_audit"]):
        raise ValueError("Accepted Random Forest metrics use another source audit")
    if split_provenance.get("group_column") != SOURCE_GROUP_COLUMN:
        raise ValueError("Accepted Random Forest source-group column changed")
    if split_provenance.get("grouping_version") != "source_identity_v1":
        raise ValueError("Accepted Random Forest grouping version changed")
    if split_provenance.get("type") != "source_component_groupkfold":
        raise ValueError("Accepted Random Forest split type changed")
    if source_audit.get("status") != "passed" or source_audit.get("validation_passed") is not True:
        raise ValueError("Frozen source-group audit is not passed")
    if int(source_audit.get("n_perfumes", -1)) != expected_rows:
        raise ValueError("Frozen source-group audit perfume count changed")
    if int(source_audit.get("n_source_components", -1)) != int(
        config["inputs"]["outer_fold_manifest"]["expected_source_components"]
    ):
        raise ValueError("Frozen source-group audit component count changed")
    if int(source_audit.get("cross_fold_source_group_count", -1)) != 0:
        raise ValueError("Frozen source-group audit reports fold leakage")
    _validate_model_contract(config, accepted_metrics)

    return PreparedInputs(
        config=config,
        config_path=config_path,
        paths=paths,
        data=data,
        manifest=manifest,
        accepted_oof=accepted_oof,
        accepted_metrics=accepted_metrics,
        feature_columns=feature_columns,
        families=families,
        variants=variants,
        labels=labels,
        sensitivity_labels=sensitivity_labels,
        protected_checksums_sha256=sha256_file(paths["accepted_checksums"]),
        accepted_manifest_validation=accepted_manifest_validation,
    )


def build_fit_tasks(prepared: PreparedInputs) -> list[FitTask]:
    folds = sorted(prepared.data["fold_id"].astype(int).unique().tolist())
    tasks = [
        FitTask(variant, fold_id, label)
        for variant in ALL_VARIANTS
        for fold_id in folds
        for label in prepared.labels
    ]
    expected = int(prepared.config["execution"]["fit_task_count"])
    if len(tasks) != expected or len(set(tasks)) != expected:
        raise ValueError(f"Expected {expected} unique fold-label fits, built {len(tasks)}")
    return tasks


def fit_fingerprint(prepared: PreparedInputs) -> str:
    return stable_json_sha256(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "protocol_sha256": sha256_file(prepared.config_path),
            "runner_sha256": sha256_file(Path(__file__)),
            "code_provenance_sha256": {
                "accepted_random_forest_runner": sha256_file(BASE_RUNNER),
                "feature_ablation_helper": sha256_file(FEATURE_ABLATION_HELPER),
                "weighted_lr_helper": sha256_file(WEIGHTED_LR_HELPER),
                "weighted_random_forest_checkpoint_helper": sha256_file(
                    WEIGHTED_RANDOM_FOREST_HELPER
                ),
                "source_grouped_folds_helper": sha256_file(
                    SOURCE_GROUPED_FOLDS_HELPER
                ),
            },
            "inputs": {key: sha256_file(path) for key, path in sorted(prepared.paths.items())},
            "variants": prepared.variants,
            "labels": list(prepared.labels),
            "model": prepared.config["model"],
            "threshold": float(prepared.config["evaluation"]["threshold"]),
            "software": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }
    )


def validate_code_and_protocol_unchanged(
    prepared: PreparedInputs, base_fingerprint: str
) -> None:
    """Fail if this runner, its protocol, helpers, or frozen inputs drift mid-run."""

    observed = fit_fingerprint(prepared)
    if observed != base_fingerprint:
        raise RuntimeError(
            "Runner, protocol, helper code, software, or frozen input drifted "
            "during the Random Forest experiment"
        )


def task_fingerprint(
    base_fingerprint: str, task: FitTask, test_perfume_ids: Sequence[str]
) -> str:
    return stable_json_sha256(
        {
            "fit_fingerprint": base_fingerprint,
            "variant": task.condition,
            "fold_id": task.fold_id,
            "label": task.label,
            "test_perfume_ids": list(test_perfume_ids),
        }
    )


def checkpoint_path(output_root: Path, task: FitTask) -> Path:
    return (
        output_root
        / "checkpoints"
        / task.condition
        / f"fold_{task.fold_id}"
        / f"{task.label}.npz"
    )


def canonical_oof_frames(prepared: PreparedInputs) -> dict[str, pd.DataFrame]:
    base = prepared.data[
        ["perfume_id", "Perfume", SOURCE_GROUP_COLUMN, "source_group_size", "fold_id"]
    ].copy()
    outputs: dict[str, pd.DataFrame] = {}
    for variant in ALL_VARIANTS:
        frame = base.copy()
        for label in prepared.labels:
            frame[f"true_{label}"] = prepared.data[label].astype(int)
            frame[f"prob_{label}"] = np.nan
            frame[f"pred_{label}"] = 0
        outputs[variant] = frame
    return outputs


def _assign_probabilities(
    output: pd.DataFrame,
    *,
    task: FitTask,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    mask = output["fold_id"].astype(int).eq(task.fold_id)
    if int(mask.sum()) != len(probabilities):
        raise ValueError("Fold probability vector length changed")
    output.loc[mask, f"prob_{task.label}"] = probabilities
    output.loc[mask, f"pred_{task.label}"] = (probabilities >= threshold).astype(int)


def progress_payload(
    *,
    status: str,
    fingerprint: str,
    completed: int,
    total: int,
    output_root: Path,
    current_task: FitTask | None,
    durations: Sequence[float],
    last_checkpoint: Path | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    average = float(np.mean(durations)) if durations else None
    remaining = max(0, total - completed)
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "experiment_id": "source_grouped_random_forest_feature_combination_ablation_v1",
        "fit_fingerprint": fingerprint,
        "status": status,
        "completed_fit_tasks": int(completed),
        "total_fit_tasks": int(total),
        "remaining_fit_tasks": int(remaining),
        "fit_percent": float(100.0 * completed / total),
        "mean_completed_fit_seconds": average,
        "estimated_remaining_fit_seconds": None if average is None else average * remaining,
        "current_task": (
            None
            if current_task is None
            else {
                "variant": current_task.condition,
                "fold_id": current_task.fold_id,
                "label": current_task.label,
            }
        ),
        "last_checkpoint": None if last_checkpoint is None else str(last_checkpoint),
        "note": note,
        "resume_command": (
            f"{shlex.quote(str(PROJECT_ROOT / '.venv/bin/python'))} "
            "-m src.training.run_source_grouped_random_forest_feature_combination_ablation "
            f"--out-dir {shlex.quote(str(output_root))}"
        ),
        "updated_utc": utc_text(),
    }


def write_progress(path: Path, **kwargs: Any) -> None:
    atomic_write_json(path, progress_payload(**kwargs))


def validate_full_control(
    prepared: PreparedInputs, replay: pd.DataFrame
) -> dict[str, Any]:
    return validate_control_replay(
        replay,
        prepared.accepted_oof,
        prepared.labels,
        probability_tolerance=float(
            prepared.config["evaluation"]["full_control_probability_tolerance"]
        ),
    )


def run_resumable_fits(
    prepared: PreparedInputs,
    *,
    output_root: Path,
    base_fingerprint: str,
    controller: PauseController,
    stop_after_tasks: int | None,
    show_progress: bool,
) -> dict[str, Any]:
    outputs = canonical_oof_frames(prepared)
    tasks = build_fit_tasks(prepared)
    threshold = float(prepared.config["evaluation"]["threshold"])
    progress_path = output_root / "progress.json"
    metadata: dict[FitTask, tuple[list[str], str, Path]] = {}
    cached: dict[FitTask, tuple[np.ndarray, dict[str, Any]]] = {}
    for task in tasks:
        ids = prepared.data.loc[
            prepared.data["fold_id"].eq(task.fold_id), "perfume_id"
        ].astype(str).tolist()
        fingerprint = task_fingerprint(base_fingerprint, task, ids)
        path = checkpoint_path(output_root, task)
        metadata[task] = (ids, fingerprint, path)
        if path.is_file():
            cached[task] = load_task_checkpoint(
                path,
                expected_fingerprint=fingerprint,
                expected_task=task,
                expected_test_perfume_ids=ids,
            )

    completed = len(cached)
    summaries: dict[FitTask, dict[str, Any]] = {}
    durations = [
        float(summary.get("fit_duration_seconds", 0.0))
        for _, summary in cached.values()
        if float(summary.get("fit_duration_seconds", 0.0)) > 0.0
    ]
    if stop_after_tasks is not None and not 0 <= stop_after_tasks <= len(tasks):
        raise ValueError(f"--stop-after-tasks must be between 0 and {len(tasks)}")
    if stop_after_tasks is not None and completed >= stop_after_tasks:
        controller.request(f"stop-after-tasks target {stop_after_tasks} already reached")
    write_progress(
        progress_path,
        status="paused" if controller.requested else "running",
        fingerprint=base_fingerprint,
        completed=completed,
        total=len(tasks),
        output_root=output_root,
        current_task=None,
        durations=durations,
        note=controller.reason,
    )
    bar = tqdm(
        total=len(tasks),
        initial=completed,
        desc="Random Forest feature ablation",
        unit="fit",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    control_validation: dict[str, Any] | None = None
    last_checkpoint: Path | None = None
    try:
        for task in tasks:
            if controller.requested:
                break
            if task.condition != FULL_VARIANT and control_validation is None:
                probability_columns = [f"prob_{label}" for label in prepared.labels]
                if outputs[FULL_VARIANT][probability_columns].isna().any().any():
                    raise RuntimeError("Full-control OOF is incomplete before ablation fits")
                control_validation = validate_full_control(prepared, outputs[FULL_VARIANT])

            ids, fingerprint, path = metadata[task]
            if task in cached:
                probabilities, summary = cached[task]
                _assign_probabilities(
                    outputs[task.condition],
                    task=task,
                    probabilities=probabilities,
                    threshold=threshold,
                )
                summaries[task] = summary
                continue

            write_progress(
                progress_path,
                status="running",
                fingerprint=base_fingerprint,
                completed=completed,
                total=len(tasks),
                output_root=output_root,
                current_task=task,
                durations=durations,
                last_checkpoint=last_checkpoint,
            )
            train_mask = prepared.data["fold_id"].ne(task.fold_id)
            test_mask = prepared.data["fold_id"].eq(task.fold_id)
            actual_ids = prepared.data.loc[test_mask, "perfume_id"].astype(str).tolist()
            if actual_ids != ids:
                raise RuntimeError("Canonical test-perfume ordering changed")
            columns = prepared.variants[task.condition]
            x_train = prepared.data.loc[train_mask, columns].to_numpy(dtype=float)
            x_test = prepared.data.loc[test_mask, columns].to_numpy(dtype=float)
            y_train = prepared.data.loc[train_mask, task.label].to_numpy(dtype=int)
            if set(np.unique(y_train)) != {0, 1}:
                raise ValueError(f"Fold {task.fold_id}/{task.label} has one training class")
            positive = int(y_train.sum())
            negative = int(len(y_train) - positive)
            model = MODEL_SPEC.build_model(
                int(prepared.config["model"]["random_seed"]), y_train
            )
            classifier = model.named_steps["clf"]
            if classifier.n_estimators != int(prepared.config["model"]["n_estimators"]):
                raise RuntimeError("Random Forest tree count changed")
            if classifier.class_weight != prepared.config["model"]["class_weight"]:
                raise RuntimeError("Random Forest class_weight changed")
            if classifier.random_state != int(prepared.config["model"]["random_seed"]):
                raise RuntimeError("Random Forest seed changed")
            if classifier.n_jobs != int(prepared.config["model"]["n_jobs"]):
                raise RuntimeError("Random Forest n_jobs changed")
            started = time.monotonic()
            model.fit(x_train, y_train)
            probabilities = positive_class_probability(model, x_test).astype(np.float64)
            duration = time.monotonic() - started
            if not np.isfinite(probabilities).all() or np.any(
                (probabilities < 0.0) | (probabilities > 1.0)
            ):
                raise RuntimeError("Random Forest produced invalid probabilities")
            summary = {
                "variant": task.condition,
                "variant_label": VARIANT_LABELS[task.condition],
                "fold_id": task.fold_id,
                "label": task.label,
                "feature_count": len(columns),
                "feature_columns_sha256": hashlib.sha256(
                    "\n".join(columns).encode("utf-8")
                ).hexdigest(),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "train_positive": positive,
                "train_negative": negative,
                "class_weight": classifier.class_weight,
                "training_sample_weight": None,
                "test_perfume_overlap_in_training": int(
                    len(
                        set(prepared.data.loc[train_mask, "perfume_id"])
                        & set(prepared.data.loc[test_mask, "perfume_id"])
                    )
                ),
                "fit_duration_seconds": duration,
                "test_perfume_ids_sha256": hashlib.sha256(
                    "\n".join(ids).encode("utf-8")
                ).hexdigest(),
            }
            save_task_checkpoint(
                path,
                fingerprint=fingerprint,
                task=task,
                test_perfume_ids=ids,
                probabilities=probabilities,
                summary=summary,
            )
            _assign_probabilities(
                outputs[task.condition],
                task=task,
                probabilities=probabilities,
                threshold=threshold,
            )
            summaries[task] = summary
            completed += 1
            durations.append(duration)
            last_checkpoint = path
            bar.update(1)
            bar.set_postfix_str(f"{task.condition}, fold {task.fold_id}, {task.label}")
            if stop_after_tasks is not None and completed >= stop_after_tasks:
                controller.request(f"stop-after-tasks target {stop_after_tasks} reached")
            write_progress(
                progress_path,
                status="pause_requested" if controller.requested else "running",
                fingerprint=base_fingerprint,
                completed=completed,
                total=len(tasks),
                output_root=output_root,
                current_task=task,
                durations=durations,
                last_checkpoint=path,
                note=controller.reason,
            )

        if controller.requested:
            write_progress(
                progress_path,
                status="paused",
                fingerprint=base_fingerprint,
                completed=completed,
                total=len(tasks),
                output_root=output_root,
                current_task=None,
                durations=durations,
                last_checkpoint=last_checkpoint,
                note=(controller.reason or "pause requested") + "; rerun the same command to resume",
            )
            return {
                "status": "paused",
                "completed": completed,
                "total": len(tasks),
                "outputs": outputs,
                "summaries": summaries,
            }
        if completed != len(tasks):
            raise RuntimeError(f"Only {completed}/{len(tasks)} Random Forest fits completed")
        if control_validation is None:
            control_validation = validate_full_control(prepared, outputs[FULL_VARIANT])
        probability_columns = [f"prob_{label}" for label in prepared.labels]
        for variant in ALL_VARIANTS:
            if outputs[variant][probability_columns].isna().any().any():
                raise RuntimeError(f"Incomplete OOF matrix for {variant}")
        return {
            "status": "complete",
            "completed": completed,
            "total": len(tasks),
            "outputs": outputs,
            "summaries": summaries,
            "durations": durations,
            "control_validation": control_validation,
        }
    finally:
        bar.close()


def summarize_variants(
    prepared: PreparedInputs, outputs: Mapping[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    per_label_frames: list[pd.DataFrame] = []
    threshold = float(prepared.config["evaluation"]["threshold"])
    for variant in ALL_VARIANTS:
        oof = outputs[variant]
        pooled, pooled_per_label = evaluate_oof(oof, prepared.labels)
        component, component_per_label = evaluate_oof(
            oof,
            prepared.labels,
            sample_weight=1.0 / oof["source_group_size"].to_numpy(float),
        )
        rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "feature_count": len(prepared.variants[variant]),
                "threshold": threshold,
                "evaluation_rows": len(oof),
                **pooled,
                "component_equal_macro_f1": component["macro_f1"],
                "component_equal_macro_average_precision": component["macro_average_precision"],
                "component_equal_macro_roc_auc": component["macro_roc_auc"],
            }
        )
        for estimator, frame in (
            ("pooled_oof", pooled_per_label),
            ("component_equal", component_per_label),
        ):
            frame.insert(0, "estimator", estimator)
            frame.insert(0, "variant", variant)
            per_label_frames.append(frame)
    return pd.DataFrame(rows), pd.concat(per_label_frames, ignore_index=True)


def build_bootstrap_outputs(
    prepared: PreparedInputs,
    outputs: Mapping[str, pd.DataFrame],
    *,
    reps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    settings = prepared.config["evaluation"]["paired_source_component_bootstrap"]
    interval_frames: list[pd.DataFrame] = []
    difference_frames: list[pd.DataFrame] = []
    details: dict[str, Any] = {}
    for scope, labels in (
        ("seven_labels", prepared.labels),
        ("fear_excluded_six_labels", prepared.sensitivity_labels),
    ):
        intervals, differences, method = build_bootstrap_tables(
            outputs,
            data=prepared.data,
            target_columns=labels,
            reps=reps,
            seed=int(settings["seed"]),
            confidence_level=float(settings["confidence_level"]),
            comparison_variants=ALL_VARIANTS,
        )
        intervals.insert(0, "label_scope", scope)
        differences.insert(0, "label_scope", scope)
        interval_frames.append(intervals)
        difference_frames.append(differences)
        details[scope] = method
    return (
        pd.concat(interval_frames, ignore_index=True),
        pd.concat(difference_frames, ignore_index=True),
        details,
    )


def feature_manifest(prepared: PreparedInputs) -> pd.DataFrame:
    rows = []
    for variant, columns in prepared.variants.items():
        included = [
            family
            for family, family_columns in prepared.families.items()
            if set(family_columns).issubset(columns)
        ]
        rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "feature_count": len(columns),
                "included_families": ";".join(included),
                "feature_columns": ";".join(columns),
                "feature_columns_sha256": hashlib.sha256(
                    "\n".join(columns).encode("utf-8")
                ).hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "variant_summary": output_root / "variant_summary.csv",
        "per_label": output_root / "per_label_metrics.csv",
        "fold_audit": output_root / "fold_training_audit.csv",
        "feature_manifest": output_root / "feature_columns_by_variant.csv",
        "bootstrap_intervals": output_root / "component_bootstrap_variant_intervals.csv",
        "bootstrap_differences": output_root / "paired_component_bootstrap_vs_full.csv",
        "accepted_control": output_root / "oof_predictions/accepted_full_hybrid_oof.csv",
        "replay_control": output_root / "oof_predictions/full_hybrid_replay_oof.csv",
        "validation": output_root / "validation_report.json",
        "report": output_root / "REPORT.json",
        "markdown": output_root / "EXPERIMENT_REPORT.md",
        "protocol": output_root / "PROTOCOL.yaml",
        "checksums": output_root / "SHA256SUMS",
        "progress": output_root / "progress.json",
    }


def build_markdown_report(summary: pd.DataFrame, differences: pd.DataFrame) -> str:
    lines = [
        "# Random Forest Feature-Combination Ablation",
        "",
        "The accepted 58-feature Random Forest control was reproduced before six "
        "advisor-requested feature combinations were evaluated with identical "
        "source-grouped folds, model settings, labels, and threshold.",
        "",
        "| Variant | Features | Macro-F1 | Macro average precision | Macro ROC-AUC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['variant_label']} | {int(row['feature_count'])} | "
            f"{row['macro_f1']:.4f} | {row['macro_average_precision']:.4f} | "
            f"{row['macro_roc_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Paired intervals resample the same 359 source components for the "
            "full control and every comparison. Positive differences favor the "
            "full 58-feature representation. The fear-excluded rows are a "
            "prespecified sensitivity because fear has three positives.",
            "",
            f"Bootstrap comparison rows: {len(differences)}.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_outputs(
    prepared: PreparedInputs,
    *,
    output_root: Path,
    base_fingerprint: str,
    fit_result: Mapping[str, Any],
    bootstrap_reps: int,
) -> dict[str, Any]:
    validate_code_and_protocol_unchanged(prepared, base_fingerprint)
    paths = artifact_paths(output_root)
    replay_outputs = fit_result["outputs"]
    control_validation = validate_full_control(
        prepared, replay_outputs[FULL_VARIANT]
    )
    # The independently recomputed full model is a fail-closed reproducibility
    # gate, not a new statistical comparator.  Once it passes, all point and
    # paired-bootstrap comparisons use the immutable accepted full OOF.
    analysis_outputs = dict(replay_outputs)
    analysis_outputs[FULL_VARIANT] = prepared.accepted_oof.copy()
    summary, per_label = summarize_variants(prepared, analysis_outputs)
    intervals, differences, bootstrap_details = build_bootstrap_outputs(
        prepared, analysis_outputs, reps=bootstrap_reps
    )
    audit = pd.DataFrame(
        [details for _, details in fit_result["summaries"].items()]
    ).sort_values(["variant", "fold_id", "label"]).reset_index(drop=True)
    features = feature_manifest(prepared)

    if len(intervals) != 42 or len(differences) != 36:
        raise RuntimeError("Bootstrap artifact row counts differ from the frozen design")
    if int(audit["test_perfume_overlap_in_training"].max()) != 0:
        raise RuntimeError("A test perfume entered an Random Forest training fold")
    for variant, oof in analysis_outputs.items():
        if oof.groupby(SOURCE_GROUP_COLUMN)["fold_id"].nunique().max() != 1:
            raise RuntimeError(f"Source-component leakage in {variant}")

    for path, frame in (
        (paths["variant_summary"], summary),
        (paths["per_label"], per_label),
        (paths["fold_audit"], audit),
        (paths["feature_manifest"], features),
        (paths["bootstrap_intervals"], intervals),
        (paths["bootstrap_differences"], differences),
    ):
        atomic_write_bytes(path, _csv_bytes(frame))
    for variant, oof in analysis_outputs.items():
        atomic_write_bytes(
            output_root / "oof_predictions" / f"{variant}_oof.csv", _csv_bytes(oof)
        )
    atomic_write_bytes(paths["accepted_control"], _csv_bytes(prepared.accepted_oof))
    atomic_write_bytes(
        paths["replay_control"], _csv_bytes(replay_outputs[FULL_VARIANT])
    )
    atomic_write_bytes(paths["protocol"], prepared.config_path.read_bytes())
    atomic_write_text(paths["markdown"], build_markdown_report(summary, differences))

    accepted_manifest_after = verify_accepted_benchmark_manifest(
        prepared.paths["accepted_checksums"]
    )
    input_hashes_after = {key: sha256_file(path) for key, path in prepared.paths.items()}
    expected_keys = {
        "base_config": "accepted_benchmark_config",
        "base_runner": "accepted_random_forest_runner",
        "feature_ablation_helper": "feature_ablation_helper",
        "weighted_lr_helper": "weighted_lr_helper",
        "weighted_rf_helper": "weighted_random_forest_checkpoint_helper",
        "source_grouped_folds_helper": "source_grouped_folds_helper",
        "data": "youtube_feature_labels",
        "manifest": "outer_fold_manifest",
        "source_audit": "source_group_audit",
        "accepted_oof": "accepted_random_forest_oof",
        "accepted_metrics": "accepted_random_forest_metrics",
        "accepted_checksums": "accepted_benchmark_checksums",
    }
    unchanged = all(
        input_hashes_after[key]
        == prepared.config["inputs"][expected_keys[key]]["sha256"]
        for key in prepared.paths
    )
    validation = {
        "status": "passed",
        "full_control_reproduction_gate": control_validation,
        "statistical_full_comparator": "immutable_accepted_random_forest_oof",
        "statistical_full_comparator_sha256": sha256_file(
            prepared.paths["accepted_oof"]
        ),
        "independent_replay_role": "reproducibility_gate_only",
        "accepted_inputs_unchanged": unchanged,
        "accepted_checksum_manifest_unchanged": (
            sha256_file(prepared.paths["accepted_checksums"])
            == prepared.protected_checksums_sha256
        ),
        "accepted_manifest_exhaustive_before": prepared.accepted_manifest_validation,
        "accepted_manifest_exhaustive_after": accepted_manifest_after,
        "rows_per_variant": 784,
        "variant_count_including_control": 7,
        "advisor_requested_variant_count": 6,
        "fit_tasks_completed": int(fit_result["completed"]),
        "source_components": 359,
        "cross_fold_source_components": 0,
        "test_perfume_overlap_in_training_max": int(
            audit["test_perfume_overlap_in_training"].max()
        ),
        "threshold": float(prepared.config["evaluation"]["threshold"]),
        "bootstrap": bootstrap_details,
    }
    if (
        not unchanged
        or not validation["accepted_checksum_manifest_unchanged"]
        or accepted_manifest_after != prepared.accepted_manifest_validation
    ):
        raise RuntimeError("A frozen accepted input changed during the experiment")
    atomic_write_json(paths["validation"], validation)
    report = {
        "schema_version": 1,
        "status": "passed",
        "experiment_id": prepared.config["experiment"]["id"],
        "fit_fingerprint": base_fingerprint,
        "accepted_benchmark_modified": False,
        "protocol_sha256": sha256_file(prepared.config_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "code_provenance_sha256": {
            "accepted_random_forest_runner": sha256_file(BASE_RUNNER),
            "feature_ablation_helper": sha256_file(FEATURE_ABLATION_HELPER),
            "weighted_lr_helper": sha256_file(WEIGHTED_LR_HELPER),
            "weighted_random_forest_checkpoint_helper": sha256_file(
                WEIGHTED_RANDOM_FOREST_HELPER
            ),
            "source_grouped_folds_helper": sha256_file(
                SOURCE_GROUPED_FOLDS_HELPER
            ),
        },
        "model": prepared.config["model"],
        "variants": features.to_dict(orient="records"),
        "official_labels": list(prepared.labels),
        "fear_excluded_sensitivity_labels": list(prepared.sensitivity_labels),
        "threshold": float(prepared.config["evaluation"]["threshold"]),
        "bootstrap_repetitions": int(bootstrap_reps),
        "inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in prepared.paths.items()
        },
        "pooled_and_component_equal_results": summary.to_dict(orient="records"),
        "paired_component_bootstrap": differences.to_dict(orient="records"),
        "bootstrap_details": bootstrap_details,
        "validation": validation,
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "interpretation_boundary": prepared.config["experiment"]["interpretation_boundary"],
    }
    atomic_write_json(paths["report"], report)
    validate_code_and_protocol_unchanged(prepared, base_fingerprint)
    checksum_files = _checksum_file_set(output_root)
    atomic_write_text(
        paths["checksums"],
        "".join(
            f"{sha256_file(path)}  ./{path.relative_to(output_root).as_posix()}\n"
            for path in sorted(checksum_files)
        ),
    )
    completed_integrity = verify_completed_output(output_root, paths["checksums"])
    write_progress(
        paths["progress"],
        status="complete",
        fingerprint=base_fingerprint,
        completed=fit_result["completed"],
        total=fit_result["total"],
        output_root=output_root,
        current_task=None,
        durations=fit_result.get("durations", []),
        note="all fits, validation, bootstrap, and artifact writes complete",
    )
    return {
        "status": "complete",
        "paths": paths,
        "summary": summary,
        "paired_differences": differences,
        "completed_output_integrity": completed_integrity,
    }


def preflight_report(prepared: PreparedInputs) -> dict[str, Any]:
    tasks = build_fit_tasks(prepared)
    return {
        "status": "passed",
        "perfumes": len(prepared.data),
        "source_components": int(prepared.data[SOURCE_GROUP_COLUMN].nunique()),
        "fold_sizes": {
            str(key): int(value)
            for key, value in prepared.data["fold_id"].value_counts().sort_index().items()
        },
        "labels": list(prepared.labels),
        "feature_family_counts": {
            family: len(columns) for family, columns in prepared.families.items()
        },
        "variant_feature_counts": {
            variant: len(columns) for variant, columns in prepared.variants.items()
        },
        "fit_tasks": len(tasks),
        "input_hashes": {key: sha256_file(path) for key, path in prepared.paths.items()},
    }


def status_report(
    *, config_path: Path = DEFAULT_CONFIG, out_dir: Path | None = None
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    output_root = (
        _resolve(config["outputs"]["root"], config_path=config_path)
        if out_dir is None
        else Path(out_dir).resolve()
    )
    progress = output_root / "progress.json"
    if not progress.is_file():
        return {
            "status": "not_started",
            "completed_fit_tasks": 0,
            "total_fit_tasks": int(config["execution"]["fit_task_count"]),
            "progress_path": str(progress),
        }
    payload = json.loads(progress.read_text(encoding="utf-8"))
    payload["progress_path"] = str(progress)
    return payload


def run_experiment(
    *,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path | None = None,
    bootstrap_reps: int | None = None,
    stop_after_tasks: int | None = None,
    show_progress: bool = True,
    install_signal_handlers: bool = True,
    preflight_only: bool = False,
) -> dict[str, Any]:
    prepared = load_and_validate_inputs(config_path)
    output_root = (
        _resolve(prepared.config["outputs"]["root"], config_path=prepared.config_path)
        if out_dir is None
        else Path(out_dir).resolve()
    )
    output_root = validate_output_root(output_root, prepared)
    configured_output_root = _resolve(
        prepared.config["outputs"]["root"], config_path=prepared.config_path
    )
    configured_reps = int(
        prepared.config["evaluation"]["paired_source_component_bootstrap"]["repetitions"]
    )
    reps = configured_reps if bootstrap_reps is None else int(bootstrap_reps)
    if reps < 2:
        raise ValueError("Bootstrap repetitions must be at least two")
    if output_root == configured_output_root.resolve() and reps != configured_reps:
        raise ValueError(
            "The canonical formal output root requires the frozen "
            f"{configured_reps} bootstrap repetitions"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = preflight_report(prepared)
    atomic_write_json(output_root / "preflight_report.json", preflight)
    if preflight_only:
        return {
            "status": "preflight_complete",
            "output_root": output_root,
            "preflight": output_root / "preflight_report.json",
        }
    base_fingerprint = fit_fingerprint(prepared)
    paths = artifact_paths(output_root)
    if paths["report"].is_file() and paths["progress"].is_file():
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        progress = json.loads(paths["progress"].read_text(encoding="utf-8"))
        if report.get("status") == "passed" and progress.get("status") == "complete":
            if report.get("fit_fingerprint") != base_fingerprint:
                raise ValueError("Completed output belongs to an incompatible run")
            if int(report.get("bootstrap_repetitions", -1)) != reps:
                raise ValueError("Completed output used another bootstrap repetition count")
            completed_integrity = verify_completed_output(
                output_root, paths["checksums"]
            )
            return {
                "status": "complete",
                "paths": paths,
                "summary": pd.read_csv(paths["variant_summary"]),
                "paired_differences": pd.read_csv(paths["bootstrap_differences"]),
                "resumed_complete_output": True,
                "completed_output_integrity": completed_integrity,
            }

    controller = PauseController()
    if install_signal_handlers:
        controller.install()
    try:
        fit_result = run_resumable_fits(
            prepared,
            output_root=output_root,
            base_fingerprint=base_fingerprint,
            controller=controller,
            stop_after_tasks=stop_after_tasks,
            show_progress=show_progress,
        )
        if fit_result["status"] == "paused":
            validate_code_and_protocol_unchanged(prepared, base_fingerprint)
            return {
                "status": "paused",
                "completed": fit_result["completed"],
                "total": fit_result["total"],
                "paths": paths,
            }
        write_progress(
            paths["progress"],
            status="bootstrap",
            fingerprint=base_fingerprint,
            completed=fit_result["completed"],
            total=fit_result["total"],
            output_root=output_root,
            current_task=None,
            durations=fit_result.get("durations", []),
            note="fits complete; deterministic vectorized bootstrap running",
        )
        return finalize_outputs(
            prepared,
            output_root=output_root,
            base_fingerprint=base_fingerprint,
            fit_result=fit_result,
            bootstrap_reps=reps,
        )
    except KeyboardInterrupt:
        existing_completed = 0
        if paths["progress"].is_file():
            try:
                existing_completed = int(
                    json.loads(paths["progress"].read_text(encoding="utf-8")).get(
                        "completed_fit_tasks", 0
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        write_progress(
            paths["progress"],
            status="paused",
            fingerprint=base_fingerprint,
            completed=existing_completed,
            total=int(prepared.config["execution"]["fit_task_count"]),
            output_root=output_root,
            current_task=None,
            durations=[],
            note="immediate interrupt; completed fit checkpoints remain resumable",
        )
        return {"status": "paused", "paths": paths}
    finally:
        if install_signal_handlers:
            controller.restore()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--bootstrap-reps", type=int)
    parser.add_argument("--stop-after-tasks", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.status:
        print(json.dumps(status_report(config_path=args.config, out_dir=args.out_dir), indent=2))
        return 0
    result = run_experiment(
        config_path=args.config,
        out_dir=args.out_dir,
        bootstrap_reps=args.bootstrap_reps,
        stop_after_tasks=args.stop_after_tasks,
        show_progress=not args.no_progress,
        preflight_only=args.preflight_only,
    )
    if result["status"] == "preflight_complete":
        print(f"Random Forest feature-ablation preflight passed: {result['preflight']}")
        return 0
    if result["status"] == "paused":
        print("Random Forest feature ablation paused safely; rerun the same command to resume.")
        if "completed" in result:
            print(f"Saved fit checkpoints: {result['completed']}/{result['total']}")
        return 0
    print("Random Forest feature-combination ablation complete")
    for _, row in result["summary"].iterrows():
        print(
            f"- {row['variant']}: macro-F1={row['macro_f1']:.4f}, "
            f"macro AP={row['macro_average_precision']:.4f}, "
            f"macro ROC-AUC={row['macro_roc_auc']:.4f}"
        )
    print(f"Report: {result['paths']['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
