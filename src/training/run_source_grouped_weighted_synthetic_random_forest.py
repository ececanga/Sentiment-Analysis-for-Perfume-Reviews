"""Run the resumable weighted-synthetic Random Forest comparison.

The accepted YouTube-only Random Forest model is replayed first.  The augmented
condition then adds the fixed three synthetic reviews for every training
perfume, with weight one third per review.  Evaluation remains restricted to
held-out YouTube perfumes in the repaired source-grouped folds.

Every fold-label fit is saved atomically.  Re-running the same command validates
and loads completed checkpoints, so an interrupted run resumes automatically.
SIGINT and SIGTERM request a pause after the current fit has been saved.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import signal
import sys
import time
from types import FrameType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.run_source_grouped_weighted_synthetic_lr import (
    AUGMENTED,
    CONDITIONS,
    CONTROL,
    FEATURE_PREFIXES,
    METRICS,
    PreparedInputs,
    _csv_bytes,
    _read_frozen_input,
    _resolve,
    _validate_binary,
    _validate_finite,
    build_paired_component_bootstrap,
    build_training_rows,
    evaluate_oof,
    positive_class_probability,
    sha256_file,
    validate_control_replay,
)
from src.utils.paths import PROJECT_ROOT


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config/evaluation_config_source_grouped_weighted_synthetic_random_forest_v1.yaml"
)
BASE_RANDOM_FOREST_RUNNER = (
    PROJECT_ROOT / "src/training/run_random_forest_benchmark.py"
)
CHECKPOINT_SCHEMA_VERSION = 1
PROGRESS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FitTask:
    condition: str
    fold_id: int
    label: str


class PauseDuringBootstrap(RuntimeError):
    """Internal control-flow exception raised at a safe bootstrap boundary."""


class PauseController:
    """Convert the first termination signal into a save-then-pause request."""

    def __init__(self) -> None:
        self.requested = False
        self.reason: str | None = None
        self.signal_number: int | None = None
        self._previous: dict[int, Any] = {}

    def request(self, reason: str, signum: int | None = None) -> None:
        self.requested = True
        self.reason = reason
        self.signal_number = signum

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.request(
            f"signal {signal.Signals(signum).name}",
            signum=signum,
        )
        print(
            "\nPause requested. Finishing and saving the current fit first; "
            "press Ctrl-C again only for an immediate stop.",
            flush=True,
        )

    def install(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)

    def restore(self) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()


def utc_text() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def atomic_write_text(path: Path, payload: str) -> None:
    atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def load_and_validate_inputs(config_path: Path = DEFAULT_CONFIG) -> PreparedInputs:
    """Load the frozen Random Forest protocol and fail closed on any drift."""

    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["experiment"]["status"] != "frozen_before_first_formal_run":
        raise ValueError("Random Forest augmentation protocol is not frozen for execution")

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
    accepted_metrics = json.loads(
        paths["accepted_metrics"].read_text(encoding="utf-8")
    )

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
    expected_synthetic = int(
        input_config["synthetic_feature_labels"]["expected_rows"]
    )
    if len(youtube) != expected_youtube or youtube["perfume_id"].nunique() != expected_youtube:
        raise ValueError("YouTube input is not the frozen 784-perfume universe")
    if len(manifest) != expected_youtube or manifest["perfume_id"].nunique() != expected_youtube:
        raise ValueError("Outer manifest is not the frozen 784-perfume universe")
    if len(synthetic) != expected_synthetic:
        raise ValueError("Synthetic row count differs from the frozen protocol")
    if synthetic["synthetic_review_id"].nunique() != len(synthetic):
        raise ValueError("Synthetic review identifiers are not unique")

    reviews_per_perfume = int(
        input_config["synthetic_feature_labels"]["expected_reviews_per_perfume"]
    )
    synthetic_counts = synthetic.groupby("perfume_id").size()
    if set(synthetic_counts.unique()) != {reviews_per_perfume}:
        raise ValueError("Every synthetic perfume must have exactly three reviews")
    if set(synthetic_counts.index) != set(youtube["perfume_id"]):
        raise ValueError("Synthetic and YouTube perfume universes differ")

    manifest["fold_id"] = pd.to_numeric(
        manifest["fold_id"], errors="raise"
    ).astype(int)
    manifest["source_group_size"] = pd.to_numeric(
        manifest["source_group_size"], errors="raise"
    ).astype(int)
    if manifest.groupby("source_group_id")["fold_id"].nunique().max() != 1:
        raise ValueError("A repaired source component crosses outer folds")
    expected_components = int(
        input_config["outer_fold_manifest"]["expected_source_components"]
    )
    if manifest["source_group_id"].nunique() != expected_components:
        raise ValueError("Unexpected number of repaired source components")
    observed_sizes = manifest.groupby("source_group_id")["perfume_id"].size()
    declared_sizes = manifest.groupby("source_group_id")["source_group_size"].first()
    if not observed_sizes.astype(int).equals(declared_sizes.astype(int)):
        raise ValueError("Manifest source_group_size values are inconsistent")

    if accepted_metrics.get("model_id") != "random_forest":
        raise ValueError("Accepted control metrics are not the Random Forest benchmark")
    feature_columns = tuple(accepted_metrics["feature_columns"])
    if len(feature_columns) != int(config["task"]["expected_feature_count"]):
        raise ValueError("Accepted Random Forest feature count differs from the protocol")
    for column in feature_columns:
        if not column.startswith(FEATURE_PREFIXES):
            raise ValueError(f"Unexpected Random Forest feature column: {column}")
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
        "Perfume",
        "source_group_id",
        "source_group_size",
        "fold_id",
        *(f"true_{label}" for label in labels),
        *(f"prob_{label}" for label in labels),
        *(f"pred_{label}" for label in labels),
    }
    missing_accepted = sorted(accepted_required - set(accepted_oof.columns))
    if missing_accepted:
        raise ValueError(f"Accepted Random Forest OOF is missing columns: {missing_accepted}")
    if len(accepted_oof) != expected_youtube or accepted_oof["perfume_id"].nunique() != expected_youtube:
        raise ValueError("Accepted Random Forest OOF is not a complete control")
    compare = accepted_oof.merge(
        joined,
        on="perfume_id",
        suffixes=("_oof", "_manifest"),
        validate="one_to_one",
    )
    for column in ("source_group_id", "source_group_size", "fold_id"):
        if not (
            compare[f"{column}_oof"].astype(str)
            == compare[f"{column}_manifest"].astype(str)
        ).all():
            raise ValueError(f"Accepted Random Forest OOF disagrees on {column}")
    threshold = float(config["evaluation"]["threshold"])
    for label in labels:
        if not (
            compare[f"true_{label}"].astype(int) == compare[label].astype(int)
        ).all():
            raise ValueError(f"Accepted Random Forest truth disagrees for {label}")
        expected_pred = (
            compare[f"prob_{label}"].astype(float) >= threshold
        ).astype(int)
        if not np.array_equal(
            expected_pred,
            compare[f"pred_{label}"].astype(int),
        ):
            raise ValueError(f"Accepted Random Forest predictions use another threshold: {label}")

    if tuple(accepted_metrics["target_columns"]) != labels:
        raise ValueError("Accepted Random Forest label order differs from the protocol")
    if not math.isclose(float(accepted_metrics["threshold"]), threshold):
        raise ValueError("Accepted Random Forest threshold differs from the protocol")
    if int(accepted_metrics["model_seed"]) != int(config["model"]["random_seed"]):
        raise ValueError("Accepted Random Forest seed differs from the protocol")
    expected_classifier = {
        "type": "RandomForestClassifier",
        "n_estimators": int(config["model"]["n_estimators"]),
        "class_weight": str(config["model"]["class_weight"]),
        "n_jobs": int(config["model"]["n_jobs"]),
    }
    if accepted_metrics["model_parameters"]["classifier"] != expected_classifier:
        raise ValueError("Accepted Random Forest parameters differ from the frozen protocol")
    if accepted_metrics["model_parameters"].get("imputer_strategy") != "median":
        raise ValueError("Accepted Random Forest imputer differs from the protocol")
    if accepted_metrics["split_provenance"]["outer_manifest_sha256"] != sha256_file(paths["manifest"]):
        raise ValueError("Accepted Random Forest metrics use a different outer manifest")

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


def effective_class_masses(
    y_train: Sequence[int] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Return positive and negative base sample-weight masses for auditing."""

    y = np.asarray(y_train, dtype=int)
    weights = np.asarray(sample_weight, dtype=float)
    if y.ndim != 1 or weights.shape != y.shape:
        raise ValueError("Training labels and sample weights must be aligned vectors")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("Random Forest sample weights must be finite and positive")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Every Random Forest fold-label fit must contain both classes")
    positive_mass = float(weights[y == 1].sum())
    negative_mass = float(weights[y == 0].sum())
    if positive_mass <= 0.0 or negative_mass <= 0.0:
        raise ValueError("Weighted class masses must both be positive")
    return positive_mass, negative_mass


def build_weighted_random_forest_model(
    model_config: Mapping[str, Any],
) -> Pipeline:
    """Build the accepted Random Forest specification unchanged."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=int(model_config["n_estimators"]),
                    class_weight=str(model_config["class_weight"]),
                    random_state=int(model_config["random_seed"]),
                    n_jobs=int(model_config["n_jobs"]),
                ),
            ),
        ]
    )


def build_fit_tasks(prepared: PreparedInputs) -> list[FitTask]:
    folds = sorted(prepared.manifest["fold_id"].astype(int).unique().tolist())
    tasks = [
        FitTask(condition=condition, fold_id=fold_id, label=label)
        for condition in CONDITIONS
        for fold_id in folds
        for label in prepared.labels
    ]
    expected = int(prepared.config["execution"]["task_count"])
    if len(tasks) != expected:
        raise ValueError(f"Protocol expects {expected} fits, constructed {len(tasks)}")
    return tasks


def fit_fingerprint(prepared: PreparedInputs) -> str:
    return stable_json_sha256(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "protocol_sha256": sha256_file(prepared.config_path),
            "runner_sha256": sha256_file(Path(__file__)),
            "accepted_random_forest_runner_sha256": sha256_file(
                BASE_RANDOM_FOREST_RUNNER
            ),
            "inputs": {
                key: sha256_file(path) for key, path in sorted(prepared.paths.items())
            },
            "feature_columns": list(prepared.feature_columns),
            "labels": list(prepared.labels),
            "model": prepared.config["model"],
            "threshold": float(prepared.config["evaluation"]["threshold"]),
            "software": {
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }
    )


def task_fingerprint(
    base_fingerprint: str,
    task: FitTask,
    test_perfume_ids: Sequence[str],
) -> str:
    return stable_json_sha256(
        {
            "fit_fingerprint": base_fingerprint,
            "condition": task.condition,
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


def save_task_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    task: FitTask,
    test_perfume_ids: Sequence[str],
    probabilities: np.ndarray,
    summary: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(CHECKPOINT_SCHEMA_VERSION),
            fingerprint=np.asarray(fingerprint),
            condition=np.asarray(task.condition),
            fold_id=np.asarray(task.fold_id),
            label=np.asarray(task.label),
            perfume_id=np.asarray(list(test_perfume_ids), dtype=str),
            probabilities=np.asarray(probabilities, dtype=np.float64),
            summary_json=np.asarray(json.dumps(dict(summary), sort_keys=True, allow_nan=False)),
        )
    temporary.replace(path)


def load_task_checkpoint(
    path: Path,
    *,
    expected_fingerprint: str,
    expected_task: FitTask,
    expected_test_perfume_ids: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as artifact:
            if int(artifact["schema_version"].item()) != CHECKPOINT_SCHEMA_VERSION:
                raise ValueError("checkpoint schema version differs")
            if str(artifact["fingerprint"].item()) != expected_fingerprint:
                raise ValueError("checkpoint fingerprint differs")
            if str(artifact["condition"].item()) != expected_task.condition:
                raise ValueError("checkpoint condition differs")
            if int(artifact["fold_id"].item()) != expected_task.fold_id:
                raise ValueError("checkpoint fold differs")
            if str(artifact["label"].item()) != expected_task.label:
                raise ValueError("checkpoint label differs")
            perfume_ids = artifact["perfume_id"].astype(str).tolist()
            probabilities = artifact["probabilities"].astype(np.float64)
            summary = json.loads(str(artifact["summary_json"].item()))
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Corrupt/incomplete checkpoint: {path}") from exc
    if perfume_ids != list(expected_test_perfume_ids):
        raise ValueError(f"Checkpoint test-perfume order differs: {path}")
    if probabilities.shape != (len(perfume_ids),):
        raise ValueError(f"Checkpoint probability vector has the wrong shape: {path}")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError(f"Checkpoint contains invalid probabilities: {path}")
    return probabilities, summary


def artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "condition_summary": output_root / "condition_summary.csv",
        "per_label_metrics": output_root / "per_label_metrics.csv",
        "fold_training_audit": output_root / "fold_training_audit.csv",
        "condition_intervals": output_root / "component_bootstrap_condition_intervals.csv",
        "paired_differences": output_root / "paired_component_bootstrap_differences.csv",
        "control_oof": output_root / "oof_predictions/youtube_only_replay_oof.csv",
        "accepted_control_oof": output_root / "oof_predictions/youtube_only_accepted_control_oof.csv",
        "augmented_oof": output_root / "oof_predictions/youtube_plus_weighted_synthetic_784x3_oof.csv",
        "validation": output_root / "validation_report.json",
        "report": output_root / "REPORT.json",
        "markdown_report": output_root / "EXPERIMENT_REPORT.md",
        "protocol": output_root / "PROTOCOL.yaml",
        "checksums": output_root / "SHA256SUMS",
        "progress": output_root / "progress.json",
    }


def canonical_oof_frame(prepared: PreparedInputs) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    data = prepared.youtube.merge(
        prepared.manifest,
        on=["perfume_id", "Perfume"],
        how="inner",
        validate="one_to_one",
    ).sort_values("perfume_id").reset_index(drop=True)
    outputs: dict[str, pd.DataFrame] = {}
    for condition in CONDITIONS:
        frame = data[
            ["perfume_id", "Perfume", "source_group_id", "source_group_size", "fold_id"]
        ].copy()
        for label in prepared.labels:
            frame[f"true_{label}"] = data[label].astype(int)
            frame[f"prob_{label}"] = np.nan
            frame[f"pred_{label}"] = 0
        outputs[condition] = frame
    return data, outputs


def progress_payload(
    *,
    status: str,
    fit_fingerprint_value: str,
    completed: int,
    total: int,
    current_task: FitTask | None,
    output_root: Path,
    durations: Sequence[float],
    last_checkpoint: Path | None = None,
    note: str | None = None,
    bootstrap_completed: int | None = None,
    bootstrap_total: int | None = None,
) -> dict[str, Any]:
    remaining = max(0, total - completed)
    average = float(np.mean(durations)) if durations else None
    eta = float(average * remaining) if average is not None else None
    command = (
        f"{shlex.quote(str(PROJECT_ROOT / '.venv/bin/python'))} "
        f"{shlex.quote(str(Path(__file__).resolve()))} "
        f"--out-dir {shlex.quote(str(output_root))}"
    )
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "experiment_id": "source_grouped_weighted_synthetic_random_forest_v1",
        "fit_fingerprint": fit_fingerprint_value,
        "status": status,
        "stage": "bootstrap" if bootstrap_total is not None else "model_fits",
        "completed_fit_tasks": int(completed),
        "total_fit_tasks": int(total),
        "remaining_fit_tasks": int(remaining),
        "fit_percent": float(100.0 * completed / total),
        "mean_completed_fit_seconds": average,
        "estimated_remaining_fit_seconds": eta,
        "current_task": (
            None
            if current_task is None
            else {
                "condition": current_task.condition,
                "fold_id": current_task.fold_id,
                "label": current_task.label,
            }
        ),
        "bootstrap_completed_draws": bootstrap_completed,
        "bootstrap_total_draws": bootstrap_total,
        "last_checkpoint": None if last_checkpoint is None else str(last_checkpoint),
        "note": note,
        "resume_command": command,
        "updated_utc": utc_text(),
    }


def write_progress(path: Path, **kwargs: Any) -> None:
    atomic_write_json(path, progress_payload(**kwargs))


def _assign_task_probabilities(
    output: pd.DataFrame,
    *,
    task: FitTask,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    mask = output["fold_id"].astype(int).eq(task.fold_id)
    if int(mask.sum()) != len(probabilities):
        raise ValueError("Checkpoint/test fold length mismatch during OOF assembly")
    output.loc[mask, f"prob_{task.label}"] = probabilities
    output.loc[mask, f"pred_{task.label}"] = (
        probabilities >= threshold
    ).astype(int)


def run_resumable_fits(
    prepared: PreparedInputs,
    *,
    output_root: Path,
    base_fingerprint: str,
    controller: PauseController,
    stop_after_tasks: int | None,
    show_progress: bool,
) -> dict[str, Any]:
    """Replay control and run augmentation, checkpointing every fit."""

    data, outputs = canonical_oof_frame(prepared)
    tasks = build_fit_tasks(prepared)
    threshold = float(prepared.config["evaluation"]["threshold"])
    progress_file = output_root / "progress.json"

    task_metadata: dict[FitTask, tuple[list[str], str, Path]] = {}
    cached: dict[FitTask, tuple[np.ndarray, dict[str, Any]]] = {}
    for task in tasks:
        ids = data.loc[
            data["fold_id"].astype(int).eq(task.fold_id), "perfume_id"
        ].astype(str).tolist()
        fingerprint = task_fingerprint(base_fingerprint, task, ids)
        path = checkpoint_path(output_root, task)
        task_metadata[task] = (ids, fingerprint, path)
        if path.is_file():
            cached[task] = load_task_checkpoint(
                path,
                expected_fingerprint=fingerprint,
                expected_task=task,
                expected_test_perfume_ids=ids,
            )

    summaries: dict[FitTask, dict[str, Any]] = {}
    durations = [
        float(summary.get("fit_duration_seconds", 0.0))
        for _, summary in cached.values()
        if float(summary.get("fit_duration_seconds", 0.0)) > 0.0
    ]
    completed = len(cached)
    if stop_after_tasks is not None and not 0 <= stop_after_tasks <= len(tasks):
        raise ValueError(f"--stop-after-tasks must be between 0 and {len(tasks)}")
    if stop_after_tasks is not None and completed >= stop_after_tasks:
        controller.request(f"stop-after-tasks target {stop_after_tasks} already reached")

    write_progress(
        progress_file,
        status="paused" if controller.requested else "running",
        fit_fingerprint_value=base_fingerprint,
        completed=completed,
        total=len(tasks),
        current_task=None,
        output_root=output_root,
        durations=durations,
        note=controller.reason,
    )
    bar = tqdm(
        total=len(tasks),
        initial=completed,
        desc="Weighted Random Forest",
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

            if task.condition == AUGMENTED and control_validation is None:
                probability_columns = [f"prob_{label}" for label in prepared.labels]
                if outputs[CONTROL][probability_columns].isna().any().any():
                    raise RuntimeError("Control OOF is incomplete before augmentation")
                control_validation = validate_control_replay(
                    outputs[CONTROL],
                    prepared.accepted_oof,
                    prepared.labels,
                )

            ids, fingerprint, path = task_metadata[task]
            if task in cached:
                probabilities, summary = cached[task]
                _assign_task_probabilities(
                    outputs[task.condition],
                    task=task,
                    probabilities=probabilities,
                    threshold=threshold,
                )
                summaries[task] = summary
                continue

            write_progress(
                progress_file,
                status="running",
                fit_fingerprint_value=base_fingerprint,
                completed=completed,
                total=len(tasks),
                current_task=task,
                output_root=output_root,
                durations=durations,
                last_checkpoint=last_checkpoint,
            )
            include_synthetic = task.condition == AUGMENTED
            training, weights, audit = build_training_rows(
                prepared,
                fold_id=task.fold_id,
                include_synthetic=include_synthetic,
            )
            test_mask = data["fold_id"].astype(int).eq(task.fold_id)
            actual_ids = data.loc[test_mask, "perfume_id"].astype(str).tolist()
            if actual_ids != ids:
                raise RuntimeError("Canonical test-perfume ordering changed")
            x_train = training[list(prepared.feature_columns)].to_numpy(dtype=float)
            x_test = data.loc[
                test_mask, list(prepared.feature_columns)
            ].to_numpy(dtype=float)
            y_train = training[task.label].to_numpy(dtype=int)
            positive_mass, negative_mass = effective_class_masses(
                y_train,
                weights,
            )
            model = build_weighted_random_forest_model(prepared.config["model"])
            started = time.monotonic()
            if include_synthetic:
                model.fit(x_train, y_train, clf__sample_weight=weights)
            else:
                # Preserve the accepted control fit call exactly.
                model.fit(x_train, y_train)
            probabilities = positive_class_probability(model, x_test).astype(np.float64)
            fit_duration = time.monotonic() - started
            if not np.isfinite(probabilities).all():
                raise RuntimeError("Random Forest produced non-finite probabilities")
            summary = {
                **audit,
                "label": task.label,
                "raw_positive_count": int(y_train.sum()),
                "raw_negative_count": int(len(y_train) - y_train.sum()),
                "positive_effective_weight": positive_mass,
                "negative_effective_weight": negative_mass,
                "sample_weight_passed_to_classifier": include_synthetic,
                "balanced_subsample_retained": True,
                "class_weight_basis": "per_tree_raw_bootstrap_counts",
                "class_weight_multiplied_by_base_sample_weight": True,
                "bootstrap_sampling_uses_sample_weight": False,
                "fit_duration_seconds": fit_duration,
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
            _assign_task_probabilities(
                outputs[task.condition],
                task=task,
                probabilities=probabilities,
                threshold=threshold,
            )
            summaries[task] = summary
            completed += 1
            durations.append(fit_duration)
            last_checkpoint = path
            bar.update(1)
            bar.set_postfix_str(
                f"{task.condition}, fold {task.fold_id}, {task.label}"
            )
            write_progress(
                progress_file,
                status="pause_requested" if controller.requested else "running",
                fit_fingerprint_value=base_fingerprint,
                completed=completed,
                total=len(tasks),
                current_task=task,
                output_root=output_root,
                durations=durations,
                last_checkpoint=path,
                note=controller.reason,
            )
            if stop_after_tasks is not None and completed >= stop_after_tasks:
                controller.request(f"stop-after-tasks target {stop_after_tasks} reached")

        if controller.requested:
            write_progress(
                progress_file,
                status="paused",
                fit_fingerprint_value=base_fingerprint,
                completed=completed,
                total=len(tasks),
                current_task=None,
                output_root=output_root,
                durations=durations,
                last_checkpoint=last_checkpoint,
                note=(controller.reason or "pause requested")
                + "; rerun the same command to resume",
            )
            return {
                "status": "paused",
                "completed": completed,
                "total": len(tasks),
                "outputs": outputs,
                "summaries": summaries,
                "control_validation": control_validation,
            }

        if completed != len(tasks):
            raise RuntimeError(f"Only {completed}/{len(tasks)} Random Forest fits completed")
        if control_validation is None:
            control_validation = validate_control_replay(
                outputs[CONTROL], prepared.accepted_oof, prepared.labels
            )
        for condition in CONDITIONS:
            probability_columns = [f"prob_{label}" for label in prepared.labels]
            if outputs[condition][probability_columns].isna().any().any():
                raise RuntimeError(f"Incomplete OOF predictions for {condition}")
        return {
            "status": "complete",
            "completed": completed,
            "total": len(tasks),
            "outputs": outputs,
            "summaries": summaries,
            "control_validation": control_validation,
            "durations": durations,
        }
    finally:
        bar.close()


def accepted_control_columns(prepared: PreparedInputs) -> list[str]:
    return [
        "perfume_id",
        "Perfume",
        "source_group_id",
        "source_group_size",
        "fold_id",
        *(
            column
            for label in prepared.labels
            for column in (f"true_{label}", f"prob_{label}", f"pred_{label}")
        ),
    ]


def summarize_conditions(
    prepared: PreparedInputs,
    oof_by_condition: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    per_label_frames: list[pd.DataFrame] = []
    threshold = float(prepared.config["evaluation"]["threshold"])
    for condition in CONDITIONS:
        oof = oof_by_condition[condition]
        pooled, pooled_per_label = evaluate_oof(oof, prepared.labels)
        component, component_per_label = evaluate_oof(
            oof,
            prepared.labels,
            sample_weight=1.0 / oof["source_group_size"].to_numpy(float),
        )
        rows.append(
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
        pooled_per_label.insert(0, "estimator", "pooled_oof")
        pooled_per_label.insert(0, "condition", condition)
        component_per_label.insert(0, "estimator", "component_equal")
        component_per_label.insert(0, "condition", condition)
        per_label_frames.extend([pooled_per_label, component_per_label])
    return pd.DataFrame(rows), pd.concat(per_label_frames, ignore_index=True)


def build_markdown_report(
    summary: pd.DataFrame,
    differences: pd.DataFrame,
) -> str:
    lines = [
        "# Weighted Synthetic-Augmentation Random Forest Comparison",
        "",
        "The accepted Random Forest specification was evaluated with the same repaired "
        "source-grouped folds under two conditions. A YouTube perfume has training "
        "base weight 1; its three synthetic reviews have base weight 1/3 each. "
        "The accepted balanced-subsample rule then applies tree-specific class "
        "weights. Test rows are held-out YouTube perfumes only.",
        "",
        "## Out-of-fold results",
        "",
        "| Condition | Macro-F1 | Macro average precision | Macro ROC-AUC |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['condition']} | {row['macro_f1']:.4f} | "
            f"{row['macro_average_precision']:.4f} | {row['macro_roc_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired source-component differences",
            "",
            "Positive values favor weighted synthetic augmentation.",
            "",
            "| Label scope | Metric | Difference | 95% interval | Improvement established |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    display = {
        METRICS[0]: "Macro-F1",
        METRICS[1]: "Macro average precision",
        METRICS[2]: "Macro ROC-AUC",
    }
    for _, row in differences.iterrows():
        lines.append(
            f"| {row['label_scope']} | {display[row['metric']]} | "
            f"{row['point_estimate_difference']:+.4f} | "
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | "
            f"{'yes' if row['positive_improvement_established'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "All three metrics are reported without selecting one as the sole main "
            "result. The fear-excluded analysis is a sensitivity check because only "
            "three YouTube perfumes have a positive fear label.",
            "",
            "This remains a mixed-granularity sensitivity analysis. The three "
            "generated rows sum to base weight one per perfume before tree-specific "
            "class balancing; this does not make generated reviews equivalent to "
            "natural YouTube observations or guarantee identical final influence.",
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
    controller: PauseController,
    show_progress: bool,
) -> dict[str, Any]:
    paths = artifact_paths(output_root)
    threshold = float(prepared.config["evaluation"]["threshold"])
    accepted = prepared.accepted_oof[accepted_control_columns(prepared)].copy()
    replay = fit_result["outputs"][CONTROL]
    augmented = fit_result["outputs"][AUGMENTED]
    control_validation = validate_control_replay(
        replay, accepted, prepared.labels
    )
    oof_for_analysis = {CONTROL: accepted, AUGMENTED: augmented}
    summary, per_label = summarize_conditions(prepared, oof_for_analysis)
    fold_audit = pd.DataFrame(
        [
            {"condition": task.condition, "fold_id": task.fold_id, "label": task.label, **details}
            for task, details in fit_result["summaries"].items()
        ]
    ).sort_values(["condition", "fold_id", "label"]).reset_index(drop=True)

    bootstrap_config = prepared.config["evaluation"]["paired_source_component_bootstrap"]
    label_scopes = {
        "seven_labels": prepared.labels,
        "fear_excluded_six_labels": prepared.sensitivity_labels,
    }
    total_draws = bootstrap_reps * len(label_scopes)
    draw_count = 0
    bootstrap_bar = tqdm(
        total=total_draws,
        desc="Paired bootstrap",
        unit="draw",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    def on_bootstrap_draw(scope_name: str, completed_in_scope: int, _reps: int) -> None:
        nonlocal draw_count
        draw_count += 1
        bootstrap_bar.update(1)
        if draw_count % 25 == 0 or draw_count == total_draws:
            write_progress(
                paths["progress"],
                status="bootstrap",
                fit_fingerprint_value=base_fingerprint,
                completed=fit_result["completed"],
                total=fit_result["total"],
                current_task=None,
                output_root=output_root,
                durations=fit_result.get("durations", []),
                note=f"bootstrap scope={scope_name} draw={completed_in_scope}/{bootstrap_reps}",
                bootstrap_completed=draw_count,
                bootstrap_total=total_draws,
            )
        if controller.requested:
            raise PauseDuringBootstrap

    write_progress(
        paths["progress"],
        status="bootstrap",
        fit_fingerprint_value=base_fingerprint,
        completed=fit_result["completed"],
        total=fit_result["total"],
        current_task=None,
        output_root=output_root,
        durations=fit_result.get("durations", []),
        note="model fits complete; paired source-component bootstrap running",
        bootstrap_completed=0,
        bootstrap_total=total_draws,
    )
    try:
        condition_intervals, paired_differences, bootstrap_details = (
            build_paired_component_bootstrap(
                oof_for_analysis,
                label_scopes=label_scopes,
                reps=bootstrap_reps,
                seed=int(bootstrap_config["seed"]),
                confidence_level=float(bootstrap_config["confidence_level"]),
                progress_callback=on_bootstrap_draw,
            )
        )
    finally:
        bootstrap_bar.close()

    if controller.requested:
        raise PauseDuringBootstrap

    write_progress(
        paths["progress"],
        status="finalizing",
        fit_fingerprint_value=base_fingerprint,
        completed=fit_result["completed"],
        total=fit_result["total"],
        current_task=None,
        output_root=output_root,
        durations=fit_result.get("durations", []),
        note="bootstrap complete; writing validated artifacts",
        bootstrap_completed=total_draws,
        bootstrap_total=total_draws,
    )

    csv_payloads = {
        paths["condition_summary"]: _csv_bytes(summary),
        paths["per_label_metrics"]: _csv_bytes(per_label),
        paths["fold_training_audit"]: _csv_bytes(fold_audit),
        paths["condition_intervals"]: _csv_bytes(condition_intervals),
        paths["paired_differences"]: _csv_bytes(paired_differences),
        paths["control_oof"]: _csv_bytes(replay),
        paths["accepted_control_oof"]: _csv_bytes(accepted),
        paths["augmented_oof"]: _csv_bytes(augmented),
    }
    for path, payload in csv_payloads.items():
        atomic_write_bytes(path, payload)

    accepted_hashes_after = {
        key: sha256_file(path) for key, path in prepared.paths.items()
    }
    validation = {
        "status": "passed",
        "control_replay": control_validation,
        "rows_per_condition": int(len(accepted)),
        "unique_perfumes_per_condition": int(accepted["perfume_id"].nunique()),
        "source_components": int(accepted["source_group_id"].nunique()),
        "cross_fold_source_components": int(
            accepted.groupby("source_group_id")["fold_id"].nunique().gt(1).sum()
        ),
        "fit_tasks_completed": int(fit_result["completed"]),
        "checkpoint_granularity": "fold_by_label",
        "held_out_synthetic_overlap_max": int(
            fold_audit["test_perfume_overlap_in_training"].max()
        ),
        "synthetic_base_weight_per_training_perfume": 1.0,
        "weights_applied_to_classifier": True,
        "balanced_subsample_retained": True,
        "tree_level_weighting_note": (
            "The three synthetic rows have total base sample weight one per "
            "perfume before each tree's balanced_subsample class multiplier."
        ),
        "weights_applied_to_imputer": False,
        "accepted_inputs_unchanged": all(
            accepted_hashes_after[key] == prepared.config["inputs"][
                {
                    "youtube": "youtube_feature_labels",
                    "synthetic": "synthetic_feature_labels",
                    "manifest": "outer_fold_manifest",
                    "accepted_oof": "accepted_youtube_control_oof",
                    "accepted_metrics": "accepted_youtube_control_metrics",
                }[key]
            ]["sha256"]
            for key in prepared.paths
        ),
        "accepted_outer_manifest_sha256": sha256_file(prepared.paths["manifest"]),
        "bootstrap": bootstrap_details,
    }
    atomic_write_json(paths["validation"], validation)
    atomic_write_bytes(paths["protocol"], prepared.config_path.read_bytes())
    atomic_write_text(
        paths["markdown_report"],
        build_markdown_report(summary, paired_differences),
    )
    report = {
        "schema_version": 1,
        "status": "passed",
        "experiment_id": prepared.config["experiment"]["id"],
        "fit_fingerprint": base_fingerprint,
        "accepted_benchmark_modified": False,
        "protocol_sha256": sha256_file(prepared.config_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "model": prepared.config["model"],
        "conditions": list(CONDITIONS),
        "official_labels": list(prepared.labels),
        "fear_excluded_sensitivity_labels": list(prepared.sensitivity_labels),
        "feature_columns": list(prepared.feature_columns),
        "threshold": threshold,
        "bootstrap_repetitions": int(bootstrap_reps),
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
    atomic_write_json(paths["report"], report)
    checksum_keys = [
        key for key in paths if key not in {"checksums", "progress"}
    ]
    atomic_write_text(
        paths["checksums"],
        "".join(
            f"{sha256_file(paths[key])}  ./{paths[key].relative_to(output_root).as_posix()}\n"
            for key in sorted(checksum_keys)
        ),
    )
    write_progress(
        paths["progress"],
        status="complete",
        fit_fingerprint_value=base_fingerprint,
        completed=fit_result["completed"],
        total=fit_result["total"],
        current_task=None,
        output_root=output_root,
        durations=fit_result.get("durations", []),
        note="all fits, validation, bootstrap, and artifact writes complete",
        bootstrap_completed=total_draws,
        bootstrap_total=total_draws,
    )
    return {
        "status": "complete",
        "paths": paths,
        "summary": summary,
        "paired_differences": paired_differences,
    }


def run_experiment(
    *,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path | None = None,
    bootstrap_reps: int | None = None,
    stop_after_tasks: int | None = None,
    show_progress: bool = True,
    install_signal_handlers: bool = True,
) -> dict[str, Any]:
    prepared = load_and_validate_inputs(Path(config_path))
    output_root = (
        _resolve(prepared.config["outputs"]["root"], config_path=prepared.config_path)
        if out_dir is None
        else Path(out_dir)
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base_fingerprint = fit_fingerprint(prepared)
    paths = artifact_paths(output_root)
    configured_reps = int(
        prepared.config["evaluation"]["paired_source_component_bootstrap"][
            "repetitions"
        ]
    )
    reps = configured_reps if bootstrap_reps is None else int(bootstrap_reps)
    if reps < 2:
        raise ValueError("Bootstrap repetitions must be at least two")

    if paths["report"].is_file() and paths["progress"].is_file():
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        progress = json.loads(paths["progress"].read_text(encoding="utf-8"))
        if report.get("status") == "passed" and progress.get("status") == "complete":
            if report.get("fit_fingerprint") != base_fingerprint:
                raise ValueError("Completed output belongs to a stale/incompatible run")
            if int(report.get("bootstrap_repetitions", -1)) != reps:
                raise ValueError("Completed output used a different bootstrap repetition count")
            return {
                "status": "complete",
                "paths": paths,
                "summary": pd.read_csv(paths["condition_summary"]),
                "paired_differences": pd.read_csv(paths["paired_differences"]),
                "resumed_complete_output": True,
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
            return {
                "status": "paused",
                "completed": fit_result["completed"],
                "total": fit_result["total"],
                "paths": paths,
            }
        try:
            return finalize_outputs(
                prepared,
                output_root=output_root,
                base_fingerprint=base_fingerprint,
                fit_result=fit_result,
                bootstrap_reps=reps,
                controller=controller,
                show_progress=show_progress,
            )
        except PauseDuringBootstrap:
            write_progress(
                paths["progress"],
                status="paused",
                fit_fingerprint_value=base_fingerprint,
                completed=fit_result["completed"],
                total=fit_result["total"],
                current_task=None,
                output_root=output_root,
                durations=fit_result.get("durations", []),
                note=(controller.reason or "pause requested")
                + "; all fits are saved; bootstrap restarts deterministically on resume",
                bootstrap_completed=None,
                bootstrap_total=reps * 2,
            )
            return {
                "status": "paused",
                "completed": fit_result["completed"],
                "total": fit_result["total"],
                "paths": paths,
            }
    except KeyboardInterrupt:
        # A second signal can stop immediately. All previously completed task
        # checkpoints remain valid; at most the active fit is repeated.
        progress = progress_payload(
            status="paused",
            fit_fingerprint_value=base_fingerprint,
            completed=0,
            total=int(prepared.config["execution"]["task_count"]),
            current_task=None,
            output_root=output_root,
            durations=[],
            note="immediate interrupt; rerun the same command to validate checkpoints and resume",
        )
        if paths["progress"].is_file():
            try:
                existing = json.loads(paths["progress"].read_text(encoding="utf-8"))
                progress["completed_fit_tasks"] = int(
                    existing.get("completed_fit_tasks", 0)
                )
                progress["remaining_fit_tasks"] = int(
                    progress["total_fit_tasks"] - progress["completed_fit_tasks"]
                )
                progress["fit_percent"] = float(
                    100.0
                    * progress["completed_fit_tasks"]
                    / progress["total_fit_tasks"]
                )
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        atomic_write_json(paths["progress"], progress)
        return {"status": "paused", "paths": paths}
    except Exception as exc:
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
            status="failed",
            fit_fingerprint_value=base_fingerprint,
            completed=existing_completed,
            total=int(prepared.config["execution"]["task_count"]),
            current_task=None,
            output_root=output_root,
            durations=[],
            note=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        if install_signal_handlers:
            controller.restore()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        help="Testing override; the formal run uses the frozen 2,000 repetitions.",
    )
    parser.add_argument(
        "--stop-after-tasks",
        type=int,
        help="Pause safely after this many of the 70 fit checkpoints exist.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_experiment(
        config_path=args.config,
        out_dir=args.out_dir,
        bootstrap_reps=args.bootstrap_reps,
        stop_after_tasks=args.stop_after_tasks,
        show_progress=not args.no_progress,
    )
    if result["status"] == "paused":
        paths = result["paths"]
        print("Weighted Random Forest experiment paused safely.")
        if "completed" in result:
            print(f"- Saved fit checkpoints: {result['completed']}/{result['total']}")
        print(f"- Progress: {paths['progress']}")
        print("- Run the same command again to resume automatically.")
        return 0

    summary = result["summary"]
    print("Weighted synthetic Random Forest comparison complete")
    for _, row in summary.iterrows():
        print(
            f"- {row['condition']}: macro-F1={row['macro_f1']:.4f}, "
            f"macro AP={row['macro_average_precision']:.4f}, "
            f"macro ROC-AUC={row['macro_roc_auc']:.4f}"
        )
    print(f"- Report: {result['paths']['markdown_report']}")
    print(f"- Progress: {result['paths']['progress']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
