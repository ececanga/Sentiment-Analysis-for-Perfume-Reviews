"""Run the resumable weighted-synthetic BERT comparison.

The accepted YouTube-only BERT result is reused, not retrained.  Five new
augmented folds add exactly three synthetic reviews only for inner-training
perfumes.  A YouTube perfume has total base loss weight one across its chunks;
each synthetic review has base loss weight one third.  Validation and testing
remain YouTube-only.

Each completed epoch and fold is written atomically.  The first SIGINT/SIGTERM
finishes and saves the current epoch, then exits safely; the normal command
automatically resumes from that checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shlex
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
import transformers
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.run_bert_benchmark import (
    BertRunSettings,
    aggregate_chunk_probabilities,
    build_loader_from_arrays,
    choose_best_epoch,
    compute_pos_weight,
    compute_probability_metrics,
    evaluate_model_on_mask,
    load_bert_settings,
    load_deep_text_dataset,
    merge_perfume_probabilities,
    set_global_seed,
    weighted_bce_loss,
)
from src.training.run_bert_source_grouped_experiments import (
    load_or_build_token_cache,
    resolve_immutable_local_checkpoint,
    token_cache_fingerprint,
    validate_synthetic_frame,
)
from src.training.run_source_grouped_weighted_synthetic_lr import (
    AUGMENTED,
    CONDITIONS,
    CONTROL,
    METRICS,
    _csv_bytes,
    _read_frozen_input,
    _resolve,
    build_paired_component_bootstrap,
    evaluate_oof,
    sha256_file,
    validate_control_replay,
)
from src.training.run_source_grouped_weighted_synthetic_random_forest import (
    PauseController,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    stable_json_sha256,
    utc_text,
)
from src.training.source_grouped_folds import (
    SOURCE_GROUP_COLUMN,
    validate_grouped_inner_manifest,
)
from src.utils.paths import PROJECT_ROOT


DEFAULT_CONFIG = (
    PROJECT_ROOT / "config/evaluation_config_source_grouped_weighted_synthetic_bert_v1.yaml"
)
CHECKPOINT_SCHEMA_VERSION = 1
PROGRESS_SCHEMA_VERSION = 1
MODEL_NAME = "BERT"


class SafePause(RuntimeError):
    """Internal signal raised only after an epoch checkpoint is durable."""


class PauseDuringBootstrap(RuntimeError):
    """Internal signal raised between bootstrap draws."""


@dataclass(frozen=True)
class FoldTask:
    condition: str
    fold_id: int


@dataclass
class PreparedInputs:
    config: dict[str, Any]
    config_path: Path
    paths: dict[str, Path]
    perfume_df: pd.DataFrame
    base_chunk_df: pd.DataFrame
    master_chunk_df: pd.DataFrame
    synthetic: pd.DataFrame
    inner_manifest: pd.DataFrame
    accepted_oof: pd.DataFrame
    accepted_metrics: dict[str, Any]
    verified_control_replay_oof: pd.DataFrame
    control_replay_validation: dict[str, Any]
    labels: tuple[str, ...]
    sensitivity_labels: tuple[str, ...]
    settings: BertRunSettings
    checkpoint_provenance: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _frozen_paths(config: Mapping[str, Any], config_path: Path) -> dict[str, Path]:
    keys = {
        "base_config": "accepted_benchmark_config",
        "base_runner": "accepted_bert_runner",
        "bert_augmentation_utility_runner": "bert_augmentation_utility_runner",
        "shared_analysis_runner": "shared_analysis_runner",
        "pause_checkpoint_utility_runner": "pause_checkpoint_utility_runner",
        "source_group_validation_runner": "source_group_validation_runner",
        "deep_text": "youtube_deep_text_jsonl",
        "canonical_labels": "canonical_youtube_labels",
        "synthetic": "synthetic_text_labels",
        "outer_manifest": "outer_fold_manifest",
        "inner_manifest": "inner_validation_manifest",
        "accepted_oof": "accepted_youtube_control_oof",
        "accepted_metrics": "accepted_youtube_control_metrics",
        "verified_control_replay_oof": "verified_youtube_control_replay_oof",
    }
    return {
        short: _read_frozen_input(config["inputs"][long], config_path=config_path, name=long)
        for short, long in keys.items()
    }


def _validate_model_settings(
    config: Mapping[str, Any],
    settings: BertRunSettings,
    base_config: Mapping[str, Any],
) -> None:
    model = config["model"]
    checks = {
        "checkpoint": settings.checkpoint,
        "local_files_only": settings.local_files_only,
        "max_length": settings.max_length,
        "aggregation_strategy": settings.aggregation_strategy,
        "train_batch_size": settings.train_batch_size,
        "eval_batch_size": settings.eval_batch_size,
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "max_epochs": settings.max_epochs,
        "patience": settings.patience,
        "min_delta": settings.min_delta,
        "warmup_ratio": settings.warmup_ratio,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "max_grad_norm": settings.max_grad_norm,
        "selection_metric": settings.selection_metric,
    }
    for key, accepted_value in checks.items():
        configured = model[key]
        if isinstance(accepted_value, float):
            matches = math.isclose(
                float(configured), float(accepted_value), rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = configured == accepted_value
        if not matches:
            raise ValueError(
                f"Frozen BERT setting {key}={configured!r} differs from accepted "
                f"value {accepted_value!r}"
            )
    if model["device"] != "cpu":
        raise ValueError("The frozen corrected BERT run must use CPU")
    if model["training_mode"] != "end_to_end_finetune":
        raise ValueError("The frozen BERT training mode changed")
    accepted_seed = int(base_config["seeds"]["data_seed"])
    if int(base_config["seeds"]["model_seed"]) != accepted_seed:
        raise ValueError("Accepted BERT data/model seeds are inconsistent")
    if int(model["random_seed"]) != accepted_seed:
        raise ValueError("Frozen BERT random seed differs from the accepted run")
    accepted_threshold = float(base_config["evaluation"]["probability_threshold"])
    configured_threshold = float(config["evaluation"]["threshold"])
    if not math.isclose(
        configured_threshold, accepted_threshold, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Frozen BERT threshold differs from the accepted run")
    synthetic_condition = config["conditions"][AUGMENTED]
    reviews_per_perfume = int(
        config["inputs"]["synthetic_text_labels"]["expected_reviews_per_perfume"]
    )
    expected_review_weight = 1.0 / reviews_per_perfume
    if not math.isclose(
        float(synthetic_condition["synthetic_review_base_weight"]),
        expected_review_weight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Frozen synthetic review weight is not one third")
    if not math.isclose(
        float(synthetic_condition["synthetic_total_base_weight_per_perfume"]),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Frozen synthetic total weight per perfume is not one")


def _accepted_control_frame(
    frame: pd.DataFrame, labels: Sequence[str]
) -> pd.DataFrame:
    columns = [
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
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Control OOF is missing columns: {missing}")
    result = frame[columns].copy()
    result["perfume_id"] = result["perfume_id"].astype(str)
    return result.sort_values("perfume_id").reset_index(drop=True)


def build_master_chunk_frame(
    perfume_df: pd.DataFrame,
    base_chunk_df: pd.DataFrame,
    synthetic: pd.DataFrame,
    labels: Sequence[str],
) -> pd.DataFrame:
    """Append one-row synthetic reviews while preserving natural row order."""

    base = base_chunk_df.copy().reset_index(drop=True)
    base["is_synthetic"] = False
    base["training_source"] = "youtube_chunk"
    base["unit_id"] = "og::" + base["perfume_id"].astype(str)
    base["synthetic_review_id"] = ""
    base["review_index"] = 0
    base["base_sample_weight"] = 1.0 / base["chunk_count"].to_numpy(dtype=np.float64)

    lookup = perfume_df.set_index("perfume_id")
    rows: list[dict[str, Any]] = []
    for _, row in synthetic.iterrows():
        perfume_id = str(row["perfume_id"])
        canonical = lookup.loc[perfume_id]
        text = str(row["Final Transcript"]).strip()
        record: dict[str, Any] = {
            "perfume_id": perfume_id,
            "Perfume": str(row["Perfume"]),
            "fold_id": int(canonical["fold_id"]),
            "chunk_count": 1,
            "chunk_index": int(row["review_index"]),
            "chunk_text": f"REVIEW: {text}",
            SOURCE_GROUP_COLUMN: str(canonical[SOURCE_GROUP_COLUMN]),
            "source_group_size": int(canonical["source_group_size"]),
            "video_id": str(row["synthetic_review_id"]),
            "transcript_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "is_synthetic": True,
            "training_source": "synthetic_review",
            "unit_id": f"synthetic::{str(row['synthetic_review_id'])}",
            "synthetic_review_id": str(row["synthetic_review_id"]),
            "review_index": int(row["review_index"]),
            "base_sample_weight": 1.0 / 3.0,
        }
        for label in labels:
            record[label] = int(row[label])
        rows.append(record)

    synthetic_chunks = pd.DataFrame(rows)
    master = pd.concat([base, synthetic_chunks], ignore_index=True, sort=False)
    master["master_row_index"] = np.arange(len(master), dtype=np.int64)
    if master["chunk_text"].astype(str).str.strip().eq("").any():
        raise ValueError("BERT master frame contains empty text")
    youtube_mass = base.groupby("perfume_id")["base_sample_weight"].sum()
    synthetic_mass = synthetic_chunks.groupby("perfume_id")["base_sample_weight"].sum()
    if not np.allclose(youtube_mass.to_numpy(float), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("YouTube chunks do not total base weight one per perfume")
    if not np.allclose(synthetic_mass.to_numpy(float), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("Synthetic reviews do not total base weight one per perfume")
    return master


def load_and_validate_inputs(config_path: Path = DEFAULT_CONFIG) -> PreparedInputs:
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    if config["experiment"]["status"] != "frozen_before_first_formal_run":
        raise ValueError("Corrected BERT augmentation protocol is not frozen")
    paths = _frozen_paths(config, config_path)

    base_config = _load_yaml(paths["base_config"])
    accepted_settings = load_bert_settings(base_config)
    _validate_model_settings(config, accepted_settings, base_config)
    labels = tuple(config["task"]["official_labels"])
    sensitivity_labels = tuple(config["task"]["sensitivity_labels"])
    if labels != tuple(base_config["targets"]["official_emotion_columns"]):
        raise ValueError("Official target order differs from the accepted BERT run")

    resolved_checkpoint, checkpoint_provenance = resolve_immutable_local_checkpoint(
        accepted_settings.checkpoint
    )
    expected_revision = str(config["model"]["checkpoint_revision"])
    if checkpoint_provenance.get("revision") != expected_revision:
        raise ValueError("Cached BERT checkpoint revision differs from the frozen revision")
    expected_files = config["model"]["checkpoint_files"]
    actual_files = checkpoint_provenance.get("files", {})
    for name, expected_hash in expected_files.items():
        actual_hash = actual_files.get(name, {}).get("sha256")
        if actual_hash != expected_hash:
            raise ValueError(f"Frozen BERT checkpoint hash mismatch for {name}")
    settings = replace(accepted_settings, checkpoint=resolved_checkpoint)

    perfume_df, base_chunk_df = load_deep_text_dataset(
        input_jsonl=paths["deep_text"],
        fold_manifest_path=paths["outer_manifest"],
        target_cols=labels,
        excluded_target_cols=base_config["targets"]["excluded_emotion_columns"],
        canonical_labels_path=paths["canonical_labels"],
    )
    expected_perfumes = int(config["inputs"]["youtube_deep_text_jsonl"]["expected_perfumes"])
    expected_chunks = int(config["inputs"]["youtube_deep_text_jsonl"]["expected_chunks"])
    if len(perfume_df) != expected_perfumes or len(base_chunk_df) != expected_chunks:
        raise ValueError("Deep-text row counts differ from the frozen protocol")

    outer = pd.read_csv(paths["outer_manifest"], keep_default_na=False)
    outer["perfume_id"] = outer["perfume_id"].astype(str)
    if outer[SOURCE_GROUP_COLUMN].nunique() != int(
        config["inputs"]["outer_fold_manifest"]["expected_source_components"]
    ):
        raise ValueError("Unexpected source-component count")

    inner = pd.read_csv(paths["inner_manifest"], keep_default_na=False)
    inner["perfume_id"] = inner["perfume_id"].astype(str)
    inner["outer_fold_id"] = pd.to_numeric(inner["outer_fold_id"], errors="raise").astype(int)
    validate_grouped_inner_manifest(
        inner,
        perfume_df,
        unit_id_col="perfume_id",
        group_col=SOURCE_GROUP_COLUMN,
        fold_id_col="fold_id",
    )

    synthetic, _ = validate_synthetic_frame(
        paths["synthetic"],
        expected_reviews_per_perfume=int(
            config["inputs"]["synthetic_text_labels"]["expected_reviews_per_perfume"]
        ),
        perfume_ids=set(perfume_df["perfume_id"].astype(str)),
    )
    if len(synthetic) != int(config["inputs"]["synthetic_text_labels"]["expected_rows"]):
        raise ValueError("Synthetic row count differs from the frozen protocol")

    accepted_raw = pd.read_csv(paths["accepted_oof"], keep_default_na=False)
    replay_raw = pd.read_csv(paths["verified_control_replay_oof"], keep_default_na=False)
    accepted_oof = _accepted_control_frame(accepted_raw, labels)
    replay_oof = _accepted_control_frame(replay_raw, labels)
    replay_validation = validate_control_replay(
        replay_oof,
        accepted_oof,
        labels,
        probability_tolerance=float(config["execution"]["control_probability_tolerance"]),
    )
    if replay_validation["max_absolute_probability_difference"] != 0.0:
        raise ValueError("The archived exact BERT replay is not byte-value exact to accepted OOF")

    accepted_metrics = json.loads(paths["accepted_metrics"].read_text(encoding="utf-8"))
    if accepted_metrics.get("model_id") != "bert":
        raise ValueError("Accepted metrics do not describe BERT")
    if len(accepted_oof) != expected_perfumes:
        raise ValueError("Accepted BERT control OOF has the wrong perfume universe")

    master = build_master_chunk_frame(perfume_df, base_chunk_df, synthetic, labels)
    return PreparedInputs(
        config=config,
        config_path=config_path,
        paths=paths,
        perfume_df=perfume_df,
        base_chunk_df=base_chunk_df,
        master_chunk_df=master,
        synthetic=synthetic,
        inner_manifest=inner,
        accepted_oof=accepted_oof,
        accepted_metrics=accepted_metrics,
        verified_control_replay_oof=replay_oof,
        control_replay_validation=replay_validation,
        labels=labels,
        sensitivity_labels=sensitivity_labels,
        settings=settings,
        checkpoint_provenance=checkpoint_provenance,
    )


def fold_membership(prepared: PreparedInputs, task: FoldTask) -> dict[str, Any]:
    if task.condition not in CONDITIONS:
        raise ValueError(f"Unknown BERT training condition: {task.condition}")
    fold_inner = prepared.inner_manifest.loc[
        prepared.inner_manifest["outer_fold_id"].eq(task.fold_id)
    ]
    train_ids = set(
        fold_inner.loc[fold_inner["inner_role"].eq("train"), "perfume_id"].astype(str)
    )
    val_ids = set(
        fold_inner.loc[fold_inner["inner_role"].eq("val"), "perfume_id"].astype(str)
    )
    test_ids = set(
        prepared.perfume_df.loc[
            prepared.perfume_df["fold_id"].eq(task.fold_id), "perfume_id"
        ].astype(str)
    )
    if not train_ids or not val_ids or not test_ids:
        raise ValueError(f"Fold {task.fold_id} contains an empty split")
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError(f"Perfume leakage in fold {task.fold_id}")

    perfume = prepared.perfume_df.set_index("perfume_id")
    groups = {
        role: set(perfume.loc[list(ids), SOURCE_GROUP_COLUMN].astype(str))
        for role, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))
    }
    if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]:
        raise ValueError(f"Source-component leakage in fold {task.fold_id}")

    master = prepared.master_chunk_df
    original = ~master["is_synthetic"].to_numpy(dtype=bool)
    train_mask = master["perfume_id"].isin(train_ids).to_numpy()
    if task.condition == CONTROL:
        train_mask &= original
    val_mask = original & master["perfume_id"].isin(val_ids).to_numpy()
    test_mask = original & master["perfume_id"].isin(test_ids).to_numpy()
    train_rows = master.loc[train_mask]
    synthetic_train = train_rows.loc[train_rows["is_synthetic"]]
    if set(synthetic_train["perfume_id"]) & (val_ids | test_ids):
        raise ValueError(f"Synthetic validation/test twins entered fold {task.fold_id}")
    if master.loc[val_mask | test_mask, "is_synthetic"].any():
        raise ValueError("Validation or test contains synthetic text")
    expected_synthetic = 3 * len(train_ids) if task.condition == AUGMENTED else 0
    if len(synthetic_train) != expected_synthetic:
        raise ValueError(f"Unexpected synthetic training rows in fold {task.fold_id}")
    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


def weighted_pos_weight(
    prepared: PreparedInputs,
    *,
    membership: Mapping[str, Any],
    condition: str = AUGMENTED,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown BERT training condition: {condition}")
    train_ids = membership["train_ids"]
    youtube = prepared.perfume_df.loc[
        prepared.perfume_df["perfume_id"].isin(train_ids)
    ].sort_values("perfume_id")
    if condition == CONTROL:
        # Match the accepted main benchmark's perfume-level class balance.
        values = compute_pos_weight(youtube, prepared.labels)
        detail = {}
        for index, label in enumerate(prepared.labels):
            positive = float(youtube[label].sum())
            detail[label] = {
                "positive_effective_mass": positive,
                "negative_effective_mass": float(len(youtube) - positive),
                "pos_weight": float(values[index]),
            }
        return values.astype(np.float32), detail
    synthetic = prepared.synthetic.loc[
        prepared.synthetic["perfume_id"].isin(train_ids)
    ].sort_values(["perfume_id", "review_index"])
    weights = np.concatenate(
        [
            np.ones(len(youtube), dtype=np.float64),
            np.full(len(synthetic), 1.0 / 3.0, dtype=np.float64),
        ]
    )
    labels = np.vstack(
        [
            youtube[list(prepared.labels)].to_numpy(dtype=np.float64),
            synthetic[list(prepared.labels)].to_numpy(dtype=np.float64),
        ]
    )
    positive = (labels * weights[:, None]).sum(axis=0)
    negative = ((1.0 - labels) * weights[:, None]).sum(axis=0)
    if np.any(positive <= 0.0):
        raise ValueError("A label has zero weighted positive training mass")
    values = negative / positive
    detail = {
        label: {
            "positive_effective_mass": float(positive[index]),
            "negative_effective_mass": float(negative[index]),
            "pos_weight": float(values[index]),
        }
        for index, label in enumerate(prepared.labels)
    }
    return values.astype(np.float32), detail


def experiment_fingerprint(prepared: PreparedInputs) -> str:
    return stable_json_sha256(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "protocol_sha256": sha256_file(prepared.config_path),
            "runner_sha256": sha256_file(Path(__file__)),
            "inputs": {key: sha256_file(path) for key, path in sorted(prepared.paths.items())},
            "labels": list(prepared.labels),
            "model": prepared.config["model"],
            "conditions": prepared.config["conditions"],
            "threshold": float(prepared.config["evaluation"]["threshold"]),
            "token_cache_fingerprint": token_cache_fingerprint(
                prepared.master_chunk_df, prepared.settings
            ),
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
        }
    )


def fold_fingerprint(base_fingerprint: str, task: FoldTask, test_ids: Sequence[str]) -> str:
    return stable_json_sha256(
        {
            "experiment_fingerprint": base_fingerprint,
            "condition": task.condition,
            "fold_id": task.fold_id,
            "test_perfume_ids": list(test_ids),
        }
    )


def fold_checkpoint_path(root: Path, task: FoldTask) -> Path:
    return root / "checkpoints" / "folds" / task.condition / f"fold_{task.fold_id}.npz"


def epoch_checkpoint_path(root: Path, task: FoldTask) -> Path:
    return root / "checkpoints" / "epochs" / task.condition / f"fold_{task.fold_id}.pt"


def save_fold_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    task: FoldTask,
    test_ids: Sequence[str],
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
            perfume_id=np.asarray(list(test_ids), dtype=str),
            probabilities=np.asarray(probabilities, dtype=np.float64),
            summary_json=np.asarray(json.dumps(dict(summary), sort_keys=True, allow_nan=False)),
        )
    temporary.replace(path)


def load_fold_checkpoint(
    path: Path,
    *,
    expected_fingerprint: str,
    task: FoldTask,
    expected_test_ids: Sequence[str],
    n_labels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as artifact:
            if int(artifact["schema_version"].item()) != CHECKPOINT_SCHEMA_VERSION:
                raise ValueError("fold checkpoint schema differs")
            if str(artifact["fingerprint"].item()) != expected_fingerprint:
                raise ValueError("fold checkpoint fingerprint differs")
            if str(artifact["condition"].item()) != task.condition:
                raise ValueError("fold checkpoint condition differs")
            if int(artifact["fold_id"].item()) != task.fold_id:
                raise ValueError("fold checkpoint ID differs")
            ids = artifact["perfume_id"].astype(str).tolist()
            probabilities = artifact["probabilities"].astype(np.float64)
            summary = json.loads(str(artifact["summary_json"].item()))
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Corrupt fold checkpoint: {path}") from exc
    if ids != list(expected_test_ids):
        raise ValueError(f"Fold checkpoint test order differs: {path}")
    if probabilities.shape != (len(ids), n_labels):
        raise ValueError(f"Fold checkpoint probability shape differs: {path}")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(f"Fold checkpoint probabilities are invalid: {path}")
    return probabilities, summary


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _load_epoch_checkpoint(path: Path, expected_fingerprint: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid epoch checkpoint: {path}")
    if int(payload.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Epoch checkpoint schema differs: {path}")
    if payload.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"Epoch checkpoint fingerprint differs: {path}")
    return payload


def _load_model(
    settings: BertRunSettings, labels: Sequence[str], device: torch.device
) -> torch.nn.Module:
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.checkpoint,
        local_files_only=settings.local_files_only,
        num_labels=len(labels),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
        label2id=label2id,
        id2label=id2label,
    )
    model.to(device)
    if device.type == "cpu":
        model.float()
    return model


def train_fold(
    prepared: PreparedInputs,
    *,
    task: FoldTask,
    membership: Mapping[str, Any],
    tokenized_inputs: Mapping[str, np.ndarray],
    fold_fp: str,
    epoch_path: Path,
    epoch_callback: Callable[[Mapping[str, Any]], None] | None,
    pause_requested: Callable[[], bool],
    model_seed: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train one joint seven-label fold with exact epoch-state resume."""

    settings = prepared.settings
    labels = prepared.labels
    master = prepared.master_chunk_df
    train_mask = membership["train_mask"]
    val_mask = membership["val_mask"]
    test_mask = membership["test_mask"]
    train_ids = membership["train_ids"]
    val_ids = membership["val_ids"]
    test_ids = membership["test_ids"]
    seed = int(prepared.config["model"]["random_seed"] if model_seed is None else model_seed)
    threshold = float(prepared.config["evaluation"]["threshold"])
    device = torch.device("cpu")

    label_matrix = master[list(labels)].to_numpy(dtype=np.float32)
    sample_weights = master["base_sample_weight"].to_numpy(dtype=np.float32)
    pos_weight, pos_weight_detail = weighted_pos_weight(
        prepared, membership=membership, condition=task.condition
    )
    pos_weight_tensor = torch.from_numpy(pos_weight).to(device)

    set_global_seed(seed)
    train_loader = build_loader_from_arrays(
        tokenized_inputs=tokenized_inputs,
        mask=train_mask,
        labels=label_matrix,
        sample_weights=sample_weights,
        batch_size=settings.train_batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = build_loader_from_arrays(
        tokenized_inputs=tokenized_inputs,
        mask=val_mask,
        labels=label_matrix,
        sample_weights=sample_weights,
        batch_size=settings.eval_batch_size,
        shuffle=False,
        seed=seed,
    )
    test_loader = build_loader_from_arrays(
        tokenized_inputs=tokenized_inputs,
        mask=test_mask,
        labels=None,
        sample_weights=None,
        batch_size=settings.eval_batch_size,
        shuffle=False,
        seed=seed,
    )
    model = _load_model(settings, labels, device)
    optimizer = AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    gradient_steps = max(1, settings.gradient_accumulation_steps)
    steps_per_epoch = int(np.ceil(max(1, len(train_loader)) / gradient_steps))
    total_train_steps = max(1, settings.max_epochs * steps_per_epoch)
    warmup_steps = int(round(total_train_steps * settings.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_metrics: dict[str, float] | None = None
    best_val_loss: float | None = None
    stale_epochs = 0
    epoch_history: list[dict[str, Any]] = []
    start_epoch = 1
    training_complete = False
    if epoch_path.is_file():
        saved = _load_epoch_checkpoint(epoch_path, fold_fp)
        if saved.get("condition") != task.condition or int(saved.get("fold_id", -1)) != task.fold_id:
            raise ValueError(f"Epoch checkpoint task differs: {epoch_path}")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        best_state = saved["best_state"]
        best_epoch = int(saved["best_epoch"])
        best_val_metrics = saved["best_val_metrics"]
        best_val_loss = saved["best_val_loss"]
        stale_epochs = int(saved["stale_epochs"])
        epoch_history = list(saved["epoch_history"])
        start_epoch = int(saved["next_epoch"])
        training_complete = bool(saved["training_complete"])
        if train_loader.generator is None:
            raise RuntimeError("Shuffled BERT training loader has no generator")
        train_loader.generator.set_state(saved["loader_generator_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        random.setstate(saved["python_random_state"])
        np.random.set_state(saved["numpy_random_state"])

    val_perfume_df = prepared.perfume_df.loc[
        prepared.perfume_df["perfume_id"].isin(val_ids)
    ].sort_values("perfume_id").reset_index(drop=True)
    val_chunk_df = master.loc[val_mask].copy().reset_index(drop=True)

    if not training_complete:
        for epoch in range(start_epoch, settings.max_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch_losses: list[float] = []
            for step, batch in enumerate(train_loader, start=1):
                input_ids, attention_mask, token_type_ids, batch_labels, batch_weights = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                batch_labels = batch_labels.to(device)
                batch_weights = batch_weights.to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                loss = weighted_bce_loss(
                    outputs.logits,
                    batch_labels,
                    batch_weights,
                    pos_weight=pos_weight_tensor,
                )
                batch_losses.append(float(loss.detach().cpu()))
                (loss / gradient_steps).backward()
                should_step = step % gradient_steps == 0 or step == len(train_loader)
                if should_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            val_chunk_prob, val_loss = evaluate_model_on_mask(
                model, val_loader, device=device, pos_weight_tensor=pos_weight_tensor
            )
            val_aggregated = aggregate_chunk_probabilities(
                val_chunk_df, val_chunk_prob, target_cols=labels
            )
            val_merged = merge_perfume_probabilities(
                val_perfume_df, val_aggregated, target_cols=labels
            )
            val_probability = val_merged[
                [f"prob_{label}" for label in labels]
            ].to_numpy(dtype=np.float32)
            _, val_metrics = compute_probability_metrics(
                y_true=val_perfume_df[list(labels)].to_numpy(dtype=int),
                y_prob=val_probability,
                target_cols=labels,
                threshold=threshold,
            )
            record = {
                "epoch": int(epoch),
                "train_loss": float(np.mean(batch_losses)),
                "val_loss": float(val_loss),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_macro_roc_auc": float(val_metrics["macro_roc_auc"]),
                "val_macro_pr_auc": float(val_metrics["macro_pr_auc"]),
            }
            epoch_history.append(record)
            is_better = choose_best_epoch(
                candidate_metrics=val_metrics,
                candidate_loss=val_loss,
                best_metrics=best_val_metrics,
                best_loss=best_val_loss,
                min_delta=settings.min_delta,
            )
            if is_better:
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_epoch = epoch
                best_val_metrics = dict(val_metrics)
                best_val_loss = float(val_loss)
                stale_epochs = 0
            else:
                stale_epochs += 1
            training_complete = bool(
                stale_epochs >= settings.patience or epoch >= settings.max_epochs
            )
            if train_loader.generator is None:
                raise RuntimeError("Shuffled BERT training loader has no generator")
            _atomic_torch_save(
                epoch_path,
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "fingerprint": fold_fp,
                    "condition": task.condition,
                    "fold_id": task.fold_id,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_state": best_state,
                    "best_epoch": best_epoch,
                    "best_val_metrics": best_val_metrics,
                    "best_val_loss": best_val_loss,
                    "stale_epochs": stale_epochs,
                    "epoch_history": epoch_history,
                    "next_epoch": epoch + 1,
                    "training_complete": training_complete,
                    "loader_generator_state": train_loader.generator.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                },
            )
            print(
                f"[BERT synthetic][fold {task.fold_id}] epoch {epoch}/{settings.max_epochs} "
                f"train_loss={record['train_loss']:.4f} val_loss={record['val_loss']:.4f} "
                f"val_macro_f1={record['val_macro_f1']:.4f}",
                flush=True,
            )
            if epoch_callback is not None:
                epoch_callback(record)
            if pause_requested():
                raise SafePause("pause requested after saved epoch")
            if training_complete:
                break

    if best_state is None or best_val_metrics is None or best_val_loss is None:
        raise RuntimeError(f"No best BERT model selected for {task}")
    model.load_state_dict(best_state)
    model.to(device).float()
    evaluate_model_on_mask(model, val_loader, device=device, pos_weight_tensor=None)
    test_chunk_probability, _ = evaluate_model_on_mask(
        model, test_loader, device=device, pos_weight_tensor=None
    )
    test_chunk_df = master.loc[test_mask].copy().reset_index(drop=True)
    test_perfume_df = prepared.perfume_df.loc[
        prepared.perfume_df["perfume_id"].isin(test_ids)
    ].sort_values("perfume_id").reset_index(drop=True)
    test_aggregated = aggregate_chunk_probabilities(
        test_chunk_df, test_chunk_probability, target_cols=labels
    )
    test_merged = merge_perfume_probabilities(
        test_perfume_df, test_aggregated, target_cols=labels
    )
    probabilities = test_merged[
        [f"prob_{label}" for label in labels]
    ].to_numpy(dtype=np.float64)

    train_rows = master.loc[train_mask]
    synthetic_train = train_rows.loc[train_rows["is_synthetic"]]
    summary = {
        "condition": task.condition,
        "fold_id": task.fold_id,
        "n_train_perfumes": len(train_ids),
        "n_val_perfumes": len(val_ids),
        "n_test_perfumes": len(test_ids),
        "n_train_text_rows": int(train_mask.sum()),
        "n_train_youtube_chunks": int((~train_rows["is_synthetic"]).sum()),
        "n_train_synthetic_reviews": int(train_rows["is_synthetic"].sum()),
        "n_val_youtube_chunks": int(val_mask.sum()),
        "n_test_youtube_chunks": int(test_mask.sum()),
        "youtube_base_weight": float(
            train_rows.loc[~train_rows["is_synthetic"], "base_sample_weight"].sum()
        ),
        "synthetic_base_weight": float(synthetic_train["base_sample_weight"].sum()),
        "synthetic_validation_or_test_rows": 0,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_macro_f1": float(best_val_metrics["macro_f1"]),
        "best_val_macro_roc_auc": float(best_val_metrics["macro_roc_auc"]),
        "best_val_macro_pr_auc": float(best_val_metrics["macro_pr_auc"]),
        "loss_pos_weight": pos_weight_detail,
        "epoch_history": epoch_history,
        "scheduler_total_steps": total_train_steps,
        "scheduler_warmup_steps": warmup_steps,
        "n_trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "test_perfume_ids_sha256": hashlib.sha256(
            "\n".join(test_merged["perfume_id"].astype(str)).encode("utf-8")
        ).hexdigest(),
    }
    return probabilities, summary


def canonical_oof_frames(prepared: PreparedInputs) -> dict[str, pd.DataFrame]:
    control = prepared.accepted_oof.copy()
    augmented = control[
        ["perfume_id", "Perfume", SOURCE_GROUP_COLUMN, "source_group_size", "fold_id"]
    ].copy()
    for label in prepared.labels:
        augmented[f"true_{label}"] = control[f"true_{label}"].astype(int)
        augmented[f"prob_{label}"] = np.nan
        augmented[f"pred_{label}"] = 0
    return {CONTROL: control, AUGMENTED: augmented}


def assign_fold_probabilities(
    frame: pd.DataFrame,
    *,
    fold_id: int,
    probabilities: np.ndarray,
    labels: Sequence[str],
    threshold: float,
) -> None:
    mask = frame["fold_id"].astype(int).eq(fold_id)
    if probabilities.shape != (int(mask.sum()), len(labels)):
        raise ValueError("Fold probability matrix does not match OOF rows")
    for index, label in enumerate(labels):
        frame.loc[mask, f"prob_{label}"] = probabilities[:, index]
        frame.loc[mask, f"pred_{label}"] = (probabilities[:, index] >= threshold).astype(int)


def accepted_control_columns(prepared: PreparedInputs) -> list[str]:
    return list(prepared.accepted_oof.columns)


def _progress_payload(
    *,
    status: str,
    fingerprint: str,
    completed_folds: int,
    total_folds: int,
    output_root: Path,
    current_task: FoldTask | None,
    current_epoch: int | None,
    durations: Sequence[float],
    note: str | None = None,
    last_checkpoint: Path | None = None,
) -> dict[str, Any]:
    average = float(np.mean(durations)) if durations else None
    remaining = total_folds - completed_folds
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "experiment_id": "source_grouped_weighted_synthetic_bert_v1",
        "experiment_fingerprint": fingerprint,
        "status": status,
        "completed_fold_tasks": completed_folds,
        "total_fold_tasks": total_folds,
        "remaining_fold_tasks": remaining,
        "fold_percent": float(100.0 * completed_folds / total_folds),
        "mean_completed_fold_seconds": average,
        "estimated_remaining_fold_seconds": None if average is None else average * remaining,
        "current_task": (
            None
            if current_task is None
            else {"condition": current_task.condition, "fold_id": current_task.fold_id}
        ),
        "current_epoch": current_epoch,
        "last_checkpoint": None if last_checkpoint is None else str(last_checkpoint),
        "note": note,
        "resume_command": (
            f"{shlex.quote(str(PROJECT_ROOT / '.venv/bin/python'))} "
            "-m src.training.run_source_grouped_weighted_synthetic_bert "
            f"--out-dir {shlex.quote(str(output_root))}"
        ),
        "updated_utc": utc_text(),
    }


def _write_progress(path: Path, **kwargs: Any) -> None:
    atomic_write_json(path, _progress_payload(**kwargs))


def run_resumable_folds(
    prepared: PreparedInputs,
    *,
    output_root: Path,
    base_fingerprint: str,
    controller: PauseController,
    show_progress: bool,
    stop_after_new_epochs: int | None,
) -> dict[str, Any]:
    folds = sorted(prepared.perfume_df["fold_id"].astype(int).unique().tolist())
    tasks = [FoldTask(AUGMENTED, fold) for fold in folds]
    expected = int(prepared.config["execution"]["fold_task_count"])
    if len(tasks) != expected:
        raise ValueError(f"Expected {expected} augmented BERT folds, built {len(tasks)}")
    threshold = float(prepared.config["evaluation"]["threshold"])
    outputs = canonical_oof_frames(prepared)
    metadata: dict[FoldTask, tuple[list[str], str, Path, Path]] = {}
    cached: dict[FoldTask, tuple[np.ndarray, dict[str, Any]]] = {}
    for task in tasks:
        ids = outputs[AUGMENTED].loc[
            outputs[AUGMENTED]["fold_id"].eq(task.fold_id), "perfume_id"
        ].astype(str).tolist()
        fingerprint = fold_fingerprint(base_fingerprint, task, ids)
        fold_path = fold_checkpoint_path(output_root, task)
        epoch_path = epoch_checkpoint_path(output_root, task)
        metadata[task] = (ids, fingerprint, fold_path, epoch_path)
        if fold_path.is_file():
            cached[task] = load_fold_checkpoint(
                fold_path,
                expected_fingerprint=fingerprint,
                task=task,
                expected_test_ids=ids,
                n_labels=len(prepared.labels),
            )
            assign_fold_probabilities(
                outputs[AUGMENTED],
                fold_id=task.fold_id,
                probabilities=cached[task][0],
                labels=prepared.labels,
                threshold=threshold,
            )

    completed = len(cached)
    summaries = {task: result[1] for task, result in cached.items()}
    durations = [
        float(summary.get("fold_duration_seconds", 0.0))
        for summary in summaries.values()
        if float(summary.get("fold_duration_seconds", 0.0)) > 0.0
    ]
    progress_path = output_root / "progress.json"
    _write_progress(
        progress_path,
        status="running",
        fingerprint=base_fingerprint,
        completed_folds=completed,
        total_folds=len(tasks),
        output_root=output_root,
        current_task=None,
        current_epoch=None,
        durations=durations,
        note="Accepted control reused; only augmented BERT folds are trained.",
    )
    bar = tqdm(
        total=len(tasks),
        initial=completed,
        desc="Weighted BERT",
        unit="fold",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    new_epochs = 0
    last_checkpoint: Path | None = None
    last_task: FoldTask | None = None
    last_epoch: int | None = None
    tokenized_inputs: Mapping[str, np.ndarray] | None = None
    token_fingerprint: str | None = None
    try:
        # Validate or build the frozen token cache even when all fold checkpoints
        # already exist, so resumed finalization retains non-null token provenance.
        tokenized_inputs, token_fingerprint = load_or_build_token_cache(
            prepared.master_chunk_df,
            prepared.settings,
            cache_path=output_root / "checkpoints" / "tokenized_master_v1.npz",
        )
        for task in tasks:
            if task in cached:
                continue
            if controller.requested:
                break

            ids, fold_fp, fold_path, epoch_path = metadata[task]
            last_task = task
            last_epoch = None
            last_checkpoint = epoch_path if epoch_path.is_file() else None
            membership = fold_membership(prepared, task)
            if ids != sorted(membership["test_ids"]):
                raise RuntimeError("Test perfume ordering changed")
            current_epoch: int | None = None
            _write_progress(
                progress_path,
                status="running",
                fingerprint=base_fingerprint,
                completed_folds=completed,
                total_folds=len(tasks),
                output_root=output_root,
                current_task=task,
                current_epoch=None,
                durations=durations,
                note="Resuming saved epoch state." if epoch_path.is_file() else None,
                last_checkpoint=last_checkpoint,
            )

            def on_epoch(record: Mapping[str, Any]) -> None:
                nonlocal new_epochs, current_epoch, last_checkpoint, last_epoch
                new_epochs += 1
                current_epoch = int(record["epoch"])
                last_epoch = current_epoch
                last_checkpoint = epoch_path
                bar.set_postfix_str(
                    f"fold {task.fold_id}, epoch {current_epoch}/{prepared.settings.max_epochs}"
                )
                _write_progress(
                    progress_path,
                    status="pause_requested" if controller.requested else "running",
                    fingerprint=base_fingerprint,
                    completed_folds=completed,
                    total_folds=len(tasks),
                    output_root=output_root,
                    current_task=task,
                    current_epoch=current_epoch,
                    durations=durations,
                    note=controller.reason,
                    last_checkpoint=epoch_path,
                )
                if stop_after_new_epochs is not None and new_epochs >= stop_after_new_epochs:
                    controller.request(
                        f"stop-after-new-epochs target {stop_after_new_epochs} reached"
                    )

            started = time.monotonic()
            try:
                probabilities, summary = train_fold(
                    prepared,
                    task=task,
                    membership=membership,
                    tokenized_inputs=tokenized_inputs,
                    fold_fp=fold_fp,
                    epoch_path=epoch_path,
                    epoch_callback=on_epoch,
                    pause_requested=lambda: controller.requested,
                )
            except SafePause:
                break
            duration = time.monotonic() - started
            summary = {**summary, "fold_duration_seconds": duration}
            save_fold_checkpoint(
                fold_path,
                fingerprint=fold_fp,
                task=task,
                test_ids=ids,
                probabilities=probabilities,
                summary=summary,
            )
            assign_fold_probabilities(
                outputs[AUGMENTED],
                fold_id=task.fold_id,
                probabilities=probabilities,
                labels=prepared.labels,
                threshold=threshold,
            )
            if epoch_path.is_file():
                epoch_path.unlink()
            summaries[task] = summary
            completed += 1
            durations.append(duration)
            last_checkpoint = fold_path
            last_epoch = None
            bar.update(1)
            _write_progress(
                progress_path,
                status="running",
                fingerprint=base_fingerprint,
                completed_folds=completed,
                total_folds=len(tasks),
                output_root=output_root,
                current_task=task,
                current_epoch=current_epoch,
                durations=durations,
                last_checkpoint=fold_path,
            )

        if controller.requested:
            _write_progress(
                progress_path,
                status="paused",
                fingerprint=base_fingerprint,
                completed_folds=completed,
                total_folds=len(tasks),
                output_root=output_root,
                current_task=last_task,
                current_epoch=last_epoch,
                durations=durations,
                note=(controller.reason or "pause requested") + "; rerun the same command to resume",
                last_checkpoint=last_checkpoint,
            )
            return {
                "status": "paused",
                "completed": completed,
                "total": len(tasks),
                "outputs": outputs,
                "summaries": summaries,
                "token_cache_fingerprint": token_fingerprint,
            }
        if completed != len(tasks):
            raise RuntimeError(f"Only {completed}/{len(tasks)} augmented BERT folds completed")
        probability_columns = [f"prob_{label}" for label in prepared.labels]
        if outputs[AUGMENTED][probability_columns].isna().any().any():
            raise RuntimeError("Incomplete augmented BERT OOF matrix")
        return {
            "status": "complete",
            "completed": completed,
            "total": len(tasks),
            "outputs": outputs,
            "summaries": summaries,
            "durations": durations,
            "token_cache_fingerprint": token_fingerprint,
        }
    finally:
        bar.close()


def summarize_conditions(
    prepared: PreparedInputs, oof_by_condition: Mapping[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    per_label_frames: list[pd.DataFrame] = []
    threshold = float(prepared.config["evaluation"]["threshold"])
    for condition in CONDITIONS:
        oof = oof_by_condition[condition]
        pooled, per_label = evaluate_oof(oof, prepared.labels)
        component, component_per_label = evaluate_oof(
            oof,
            prepared.labels,
            sample_weight=1.0 / oof["source_group_size"].to_numpy(float),
        )
        rows.append(
            {
                "condition": condition,
                "evaluation_rows": len(oof),
                "threshold": threshold,
                **pooled,
                "component_equal_macro_f1": component["macro_f1"],
                "component_equal_macro_average_precision": component["macro_average_precision"],
                "component_equal_macro_roc_auc": component["macro_roc_auc"],
            }
        )
        per_label.insert(0, "estimator", "pooled_oof")
        per_label.insert(0, "condition", condition)
        component_per_label.insert(0, "estimator", "component_equal")
        component_per_label.insert(0, "condition", condition)
        per_label_frames.extend([per_label, component_per_label])
    return pd.DataFrame(rows), pd.concat(per_label_frames, ignore_index=True)


def artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "condition_summary": output_root / "condition_summary.csv",
        "per_label_metrics": output_root / "per_label_metrics.csv",
        "fold_training_audit": output_root / "fold_training_audit.csv",
        "condition_intervals": output_root / "component_bootstrap_condition_intervals.csv",
        "paired_differences": output_root / "paired_component_bootstrap_differences.csv",
        "control_oof": output_root / "oof_predictions/youtube_only_verified_replay_oof.csv",
        "accepted_control_oof": output_root / "oof_predictions/youtube_only_accepted_control_oof.csv",
        "augmented_oof": output_root / "oof_predictions/youtube_plus_weighted_synthetic_784x3_oof.csv",
        "validation": output_root / "validation_report.json",
        "report": output_root / "REPORT.json",
        "markdown_report": output_root / "EXPERIMENT_REPORT.md",
        "protocol": output_root / "PROTOCOL.yaml",
        "checksums": output_root / "SHA256SUMS",
        "progress": output_root / "progress.json",
    }


def build_markdown_report(summary: pd.DataFrame, differences: pd.DataFrame) -> str:
    lines = [
        "# Weighted Synthetic-Augmentation BERT Comparison",
        "",
        "The accepted joint seven-label BERT control is reused without retraining. "
        "The augmented condition adds three synthetic reviews only to each "
        "inner-training perfume at base loss weight 1/3 each. Validation and "
        "testing contain YouTube text only.",
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
    names = {
        METRICS[0]: "Macro-F1",
        METRICS[1]: "Macro average precision",
        METRICS[2]: "Macro ROC-AUC",
    }
    for _, row in differences.iterrows():
        lines.append(
            f"| {row['label_scope']} | {names[row['metric']]} | "
            f"{row['point_estimate_difference']:+.4f} | "
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | "
            f"{'yes' if row['positive_improvement_established'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "All three metrics are reported without selecting a single main result. "
            "The fear-excluded result is a sensitivity check because only three "
            "YouTube perfumes have a positive fear label.",
            "",
            "The 1/chunk-count and 1/3 values are base mini-batch loss weights. "
            "They equalize corpus-level base mass per perfume but do not imply "
            "identical optimizer influence. Synthetic labels are model-assigned "
            "reference labels, not human ground truth.",
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
    show_progress: bool,
    controller: PauseController,
) -> dict[str, Path]:
    paths = artifact_paths(output_root)
    accepted = prepared.accepted_oof.copy()
    replay = prepared.verified_control_replay_oof.copy()
    augmented = fit_result["outputs"][AUGMENTED]
    control_validation = validate_control_replay(
        replay,
        accepted,
        prepared.labels,
        probability_tolerance=float(prepared.config["execution"]["control_probability_tolerance"]),
    )
    analysis_oof = {CONTROL: accepted, AUGMENTED: augmented}
    summary, per_label = summarize_conditions(prepared, analysis_oof)

    audit_rows: list[dict[str, Any]] = []
    for _, detail in fit_result["summaries"].items():
        row = {key: value for key, value in detail.items() if key not in {"epoch_history", "loss_pos_weight"}}
        row["loss_pos_weight_json"] = json.dumps(detail["loss_pos_weight"], sort_keys=True)
        row["epoch_history_json"] = json.dumps(detail["epoch_history"])
        audit_rows.append(row)
    fold_audit = pd.DataFrame(audit_rows).sort_values("fold_id").reset_index(drop=True)

    bootstrap = prepared.config["evaluation"]["paired_source_component_bootstrap"]
    total_draws = bootstrap_reps * 2
    bar = tqdm(
        total=total_draws,
        desc="Paired bootstrap",
        unit="draw",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    def bootstrap_progress(_scope: str, _completed: int, _total: int) -> None:
        bar.update(1)
        if controller.requested:
            raise PauseDuringBootstrap("pause requested between bootstrap draws")

    try:
        intervals, differences, bootstrap_details = build_paired_component_bootstrap(
            analysis_oof,
            label_scopes={
                "seven_labels": prepared.labels,
                "fear_excluded_six_labels": prepared.sensitivity_labels,
            },
            reps=bootstrap_reps,
            seed=int(bootstrap["seed"]),
            confidence_level=float(bootstrap["confidence_level"]),
            progress_callback=bootstrap_progress,
        )
    finally:
        bar.close()

    csv_payloads = {
        paths["condition_summary"]: _csv_bytes(summary),
        paths["per_label_metrics"]: _csv_bytes(per_label),
        paths["fold_training_audit"]: _csv_bytes(fold_audit),
        paths["condition_intervals"]: _csv_bytes(intervals),
        paths["paired_differences"]: _csv_bytes(differences),
        paths["control_oof"]: _csv_bytes(replay),
        paths["accepted_control_oof"]: _csv_bytes(accepted),
        paths["augmented_oof"]: _csv_bytes(augmented),
    }
    for path, payload in csv_payloads.items():
        atomic_write_bytes(path, payload)
    atomic_write_bytes(paths["protocol"], prepared.config_path.read_bytes())
    atomic_write_text(paths["markdown_report"], build_markdown_report(summary, differences))

    validation = {
        "status": "passed",
        "accepted_control_retrained": False,
        "control_replay": control_validation,
        "rows_per_condition": len(accepted),
        "unique_perfumes_per_condition": int(accepted["perfume_id"].nunique()),
        "source_components": int(accepted[SOURCE_GROUP_COLUMN].nunique()),
        "cross_fold_source_components": int(
            accepted.groupby(SOURCE_GROUP_COLUMN)["fold_id"].nunique().gt(1).sum()
        ),
        "synthetic_validation_or_test_rows_max": int(
            fold_audit["synthetic_validation_or_test_rows"].max()
        ),
        "augmented_synthetic_base_weight_per_training_perfume": 1.0,
        "validation_and_test_youtube_only": True,
        "new_joint_multilabel_fold_models": 5,
        "bootstrap": bootstrap_details,
    }
    atomic_write_json(paths["validation"], validation)
    report = {
        "schema_version": 1,
        "status": "passed",
        "experiment_id": prepared.config["experiment"]["id"],
        "experiment_fingerprint": base_fingerprint,
        "protocol_sha256": sha256_file(prepared.config_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "accepted_benchmark_modified": False,
        "accepted_control_retrained": False,
        "model": prepared.config["model"],
        "resolved_checkpoint": prepared.checkpoint_provenance,
        "conditions": list(CONDITIONS),
        "official_labels": list(prepared.labels),
        "fear_excluded_sensitivity_labels": list(prepared.sensitivity_labels),
        "threshold": float(prepared.config["evaluation"]["threshold"]),
        "inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in prepared.paths.items()
        },
        "pooled_and_component_equal_results": summary.to_dict(orient="records"),
        "per_label_results": per_label.to_dict(orient="records"),
        "paired_component_bootstrap": differences.to_dict(orient="records"),
        "bootstrap_details": bootstrap_details,
        "validation": validation,
        "token_cache_fingerprint": fit_result.get("token_cache_fingerprint"),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "interpretation_boundary": prepared.config["experiment"]["interpretation_boundary"],
    }
    atomic_write_json(paths["report"], report)
    checksum_paths = {key: path for key, path in paths.items() if key not in {"checksums", "progress"}}
    atomic_write_text(
        paths["checksums"],
        "".join(
            f"{sha256_file(path)}  ./{path.relative_to(output_root).as_posix()}\n"
            for _, path in sorted(checksum_paths.items())
        ),
    )
    atomic_write_json(
        paths["progress"],
        {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "experiment_id": prepared.config["experiment"]["id"],
            "experiment_fingerprint": base_fingerprint,
            "status": "complete",
            "completed_fold_tasks": 5,
            "total_fold_tasks": 5,
            "bootstrap_completed_draws": total_draws,
            "bootstrap_total_draws": total_draws,
            "updated_utc": utc_text(),
        },
    )
    return paths


def preflight_report(prepared: PreparedInputs) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    folds = sorted(prepared.perfume_df["fold_id"].astype(int).unique())
    for fold in folds:
        task = FoldTask(AUGMENTED, int(fold))
        membership = fold_membership(prepared, task)
        pos_weight, detail = weighted_pos_weight(prepared, membership=membership)
        train = prepared.master_chunk_df.loc[membership["train_mask"]]
        rows.append(
            {
                "condition": AUGMENTED,
                "fold_id": int(fold),
                "train_perfumes": len(membership["train_ids"]),
                "val_perfumes": len(membership["val_ids"]),
                "test_perfumes": len(membership["test_ids"]),
                "train_total_rows": int(len(train)),
                "train_youtube_chunks": int((~train["is_synthetic"]).sum()),
                "train_synthetic_reviews": int(train["is_synthetic"].sum()),
                "youtube_base_weight": float(
                    train.loc[~train["is_synthetic"], "base_sample_weight"].sum()
                ),
                "synthetic_base_weight": float(
                    train.loc[train["is_synthetic"], "base_sample_weight"].sum()
                ),
                "validation_youtube_chunks": int(membership["val_mask"].sum()),
                "test_youtube_chunks": int(membership["test_mask"].sum()),
                "pos_weight": [float(value) for value in pos_weight],
                "pos_weight_detail": detail,
                "pos_weight_finite": bool(np.isfinite(pos_weight).all()),
            }
        )
    expected_total_rows = [3584, 5877, 5854, 5860, 5958]
    expected_synthetic_rows = [1215, 1899, 1899, 1902, 1902]
    if [row["train_total_rows"] for row in rows] != expected_total_rows:
        raise ValueError("Augmented BERT training-row counts changed from the frozen audit")
    if [row["train_synthetic_reviews"] for row in rows] != expected_synthetic_rows:
        raise ValueError("Augmented BERT synthetic-row counts changed from the frozen audit")
    return {
        "status": "passed",
        "input_hashes": {key: sha256_file(path) for key, path in prepared.paths.items()},
        "control_replay_validation": prepared.control_replay_validation,
        "accepted_control_retrained": False,
        "perfumes": len(prepared.perfume_df),
        "youtube_chunks": len(prepared.base_chunk_df),
        "synthetic_reviews": len(prepared.synthetic),
        "source_components": int(prepared.perfume_df[SOURCE_GROUP_COLUMN].nunique()),
        "resolved_checkpoint": prepared.checkpoint_provenance,
        "folds": rows,
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
    progress_path = output_root / "progress.json"
    if not progress_path.is_file():
        return {
            "status": "not_started",
            "completed_fold_tasks": 0,
            "total_fold_tasks": int(config["execution"]["fold_task_count"]),
            "progress_path": str(progress_path),
            "resume_command": (
                f"{shlex.quote(str(PROJECT_ROOT / '.venv/bin/python'))} "
                "-m src.training.run_source_grouped_weighted_synthetic_bert"
            ),
        }
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    payload["progress_path"] = str(progress_path)
    return payload


def run_experiment(
    *,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path | None = None,
    bootstrap_reps: int | None = None,
    show_progress: bool = True,
    stop_after_new_epochs: int | None = None,
    preflight_only: bool = False,
) -> dict[str, Any]:
    if stop_after_new_epochs is not None and stop_after_new_epochs < 1:
        raise ValueError("stop_after_new_epochs must be at least one")
    prepared = load_and_validate_inputs(config_path)
    output_root = (
        _resolve(prepared.config["outputs"]["root"], config_path=prepared.config_path)
        if out_dir is None
        else Path(out_dir).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    base_fingerprint = experiment_fingerprint(prepared)
    preflight = preflight_report(prepared)
    atomic_write_json(output_root / "preflight_report.json", preflight)
    if preflight_only:
        return {
            "status": "preflight_complete",
            "output_root": output_root,
            "preflight": output_root / "preflight_report.json",
        }

    controller = PauseController()
    controller.install()
    try:
        fit_result = run_resumable_folds(
            prepared,
            output_root=output_root,
            base_fingerprint=base_fingerprint,
            controller=controller,
            show_progress=show_progress,
            stop_after_new_epochs=stop_after_new_epochs,
        )
        if fit_result["status"] == "paused":
            return {
                "status": "paused",
                "output_root": output_root,
                "progress": output_root / "progress.json",
            }
        configured_reps = int(
            prepared.config["evaluation"]["paired_source_component_bootstrap"]["repetitions"]
        )
        reps = configured_reps if bootstrap_reps is None else int(bootstrap_reps)
        if reps < 2:
            raise ValueError("Bootstrap repetitions must be at least two")
        try:
            paths = finalize_outputs(
                prepared,
                output_root=output_root,
                base_fingerprint=base_fingerprint,
                fit_result=fit_result,
                bootstrap_reps=reps,
                show_progress=show_progress,
                controller=controller,
            )
        except PauseDuringBootstrap:
            progress = _progress_payload(
                status="paused",
                fingerprint=base_fingerprint,
                completed_folds=5,
                total_folds=5,
                output_root=output_root,
                current_task=None,
                current_epoch=None,
                durations=fit_result.get("durations", []),
                note="paused during bootstrap; rerun the same command to restart bootstrap",
            )
            atomic_write_json(output_root / "progress.json", progress)
            return {
                "status": "paused",
                "output_root": output_root,
                "progress": output_root / "progress.json",
            }
        return {"status": "complete", "output_root": output_root, "paths": paths}
    finally:
        controller.restore()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--bootstrap-reps", type=int)
    parser.add_argument("--stop-after-new-epochs", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.status:
        status = status_report(config_path=args.config, out_dir=args.out_dir)
        print(json.dumps(status, indent=2))
        return 0
    result = run_experiment(
        config_path=args.config,
        out_dir=args.out_dir,
        bootstrap_reps=args.bootstrap_reps,
        show_progress=not args.no_progress,
        stop_after_new_epochs=args.stop_after_new_epochs,
        preflight_only=args.preflight_only,
    )
    if result["status"] == "preflight_complete":
        print(f"BERT augmentation preflight passed: {result['preflight']}")
        return 0
    if result["status"] == "paused":
        print("BERT augmentation paused safely; rerun the same command to resume.")
        print(f"Progress: {result['progress']}")
        return 0

    paths = result["paths"]
    summary = pd.read_csv(paths["condition_summary"])
    differences = pd.read_csv(paths["paired_differences"])
    print("Weighted synthetic BERT comparison complete")
    for _, row in summary.iterrows():
        print(
            f"- {row['condition']}: macro-F1={row['macro_f1']:.4f}, "
            f"macro AP={row['macro_average_precision']:.4f}, "
            f"macro ROC-AUC={row['macro_roc_auc']:.4f}"
        )
    seven = differences.loc[differences["label_scope"].eq("seven_labels")]
    for _, row in seven.iterrows():
        print(
            f"- {row['metric']} difference={row['point_estimate_difference']:+.4f} "
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
        )
    print(f"Report: {paths['markdown_report']}")
    print(f"Validation: {paths['validation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
