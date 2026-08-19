"""Run the resumable weighted-synthetic BiLSTM + Attention comparison.

The accepted YouTube-only text model is replayed first.  The augmented
condition adds the fixed three synthetic reviews only for inner-training
perfumes, at base loss weight one third per review.  Inner validation and outer
evaluation remain YouTube-only.  Epoch and fold checkpoints make the run safe
to pause and resume.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
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
import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.run_bert_benchmark import (
    aggregate_chunk_probabilities,
    choose_best_epoch,
    compute_probability_metrics,
    compute_pos_weight,
    load_deep_text_dataset,
    merge_perfume_probabilities,
    set_global_seed,
    weighted_bce_loss,
)
from src.training.run_bert_source_grouped_experiments import validate_synthetic_frame
from src.training.run_bilstm_attention_benchmark import (
    BiLSTMAttentionClassifier,
    BiLSTMAttentionRunSettings,
    build_fold_vocabulary,
    build_sequence_loader,
    encode_token_sequences,
    evaluate_sequence_model,
    load_bilstm_settings,
    pretokenize_chunk_texts,
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
    PROJECT_ROOT
    / "config/evaluation_config_source_grouped_weighted_synthetic_bilstm_attention_v1.yaml"
)
CHECKPOINT_SCHEMA_VERSION = 1
PROGRESS_SCHEMA_VERSION = 1
MODEL_NAME = "BiLSTM + Attention"


class SafePause(RuntimeError):
    """Internal signal that training stopped after an atomic epoch checkpoint."""


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
    labels: tuple[str, ...]
    sensitivity_labels: tuple[str, ...]
    settings: BiLSTMAttentionRunSettings


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _frozen_paths(config: Mapping[str, Any], config_path: Path) -> dict[str, Path]:
    keys = {
        "base_config": "accepted_benchmark_config",
        "base_runner": "accepted_bilstm_runner",
        "deep_text": "youtube_deep_text_jsonl",
        "canonical_labels": "canonical_youtube_labels",
        "synthetic": "synthetic_text_labels",
        "outer_manifest": "outer_fold_manifest",
        "inner_manifest": "inner_validation_manifest",
        "accepted_oof": "accepted_youtube_control_oof",
        "accepted_metrics": "accepted_youtube_control_metrics",
    }
    return {
        short: _read_frozen_input(
            config["inputs"][long], config_path=config_path, name=long
        )
        for short, long in keys.items()
    }


def _validate_model_settings(
    config: Mapping[str, Any], settings: BiLSTMAttentionRunSettings
) -> None:
    model = config["model"]
    checks = {
        "lowercase_tokens": settings.lowercase_tokens,
        "min_token_freq": settings.min_token_freq,
        "max_vocab_size": settings.max_vocab_size,
        "max_tokens": settings.max_tokens,
        "embedding_dim": settings.embedding_dim,
        "hidden_size": settings.hidden_size,
        "attention_dim": settings.attention_dim,
        "lstm_layers": settings.lstm_layers,
        "dropout": settings.dropout,
        "train_batch_size": settings.train_batch_size,
        "eval_batch_size": settings.eval_batch_size,
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "max_epochs": settings.max_epochs,
        "patience": settings.patience,
        "min_delta": settings.min_delta,
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
                f"Frozen BiLSTM setting {key}={configured!r} differs from "
                f"accepted value {accepted_value!r}"
            )


def build_master_chunk_frame(
    perfume_df: pd.DataFrame,
    base_chunk_df: pd.DataFrame,
    synthetic: pd.DataFrame,
    labels: Sequence[str],
) -> pd.DataFrame:
    """Append single-review synthetic rows without changing natural-row order."""

    base = base_chunk_df.copy().reset_index(drop=True)
    base["is_synthetic"] = False
    base["training_source"] = "youtube_chunk"
    base["synthetic_review_id"] = ""
    base["review_index"] = 0
    base["base_sample_weight"] = (
        1.0 / base["chunk_count"].to_numpy(dtype=np.float64)
    )

    lookup = perfume_df.set_index("perfume_id")
    synthetic_rows: list[dict[str, Any]] = []
    for _, row in synthetic.iterrows():
        perfume_id = str(row["perfume_id"])
        canonical = lookup.loc[perfume_id]
        source_text = str(row["Final Transcript"]).strip()
        record: dict[str, Any] = {
            "perfume_id": perfume_id,
            "Perfume": str(row["Perfume"]),
            "fold_id": int(canonical["fold_id"]),
            "chunk_count": 1,
            "chunk_index": int(row["review_index"]),
            # Natural chunks use the same neutral review-boundary marker.
            "chunk_text": f"REVIEW: {source_text}",
            SOURCE_GROUP_COLUMN: str(canonical[SOURCE_GROUP_COLUMN]),
            "source_group_size": int(canonical["source_group_size"]),
            "video_id": str(row["synthetic_review_id"]),
            "transcript_sha256": hashlib.sha256(
                source_text.encode("utf-8")
            ).hexdigest(),
            "is_synthetic": True,
            "training_source": "synthetic_review",
            "synthetic_review_id": str(row["synthetic_review_id"]),
            "review_index": int(row["review_index"]),
            "base_sample_weight": 1.0 / 3.0,
        }
        for label in labels:
            record[label] = int(row[label])
        synthetic_rows.append(record)

    synthetic_chunk_df = pd.DataFrame(synthetic_rows)
    master = pd.concat([base, synthetic_chunk_df], ignore_index=True, sort=False)
    master["master_row_index"] = np.arange(len(master), dtype=np.int64)
    if master["chunk_text"].astype(str).str.strip().eq("").any():
        raise ValueError("Text master frame contains an empty review")
    if not np.allclose(
        base.groupby("perfume_id")["base_sample_weight"].sum().to_numpy(float),
        1.0,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("Natural chunks do not total base weight one per perfume")
    synthetic_weight = synthetic_chunk_df.groupby("perfume_id")[
        "base_sample_weight"
    ].sum()
    if not np.allclose(
        synthetic_weight.to_numpy(float), 1.0, atol=1e-12, rtol=0.0
    ):
        raise ValueError("Synthetic reviews do not total base weight one per perfume")
    return master


def load_and_validate_inputs(config_path: Path = DEFAULT_CONFIG) -> PreparedInputs:
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    if config["experiment"]["status"] != "frozen_before_first_formal_run":
        raise ValueError("BiLSTM augmentation protocol is not frozen")
    if config["model"]["device"] != "cpu":
        raise ValueError("The accepted control must be replayed on CPU")

    paths = _frozen_paths(config, config_path)
    base_config = _load_yaml(paths["base_config"])
    settings = load_bilstm_settings(base_config)
    _validate_model_settings(config, settings)
    labels = tuple(config["task"]["official_labels"])
    sensitivity_labels = tuple(config["task"]["sensitivity_labels"])
    accepted_targets = tuple(base_config["targets"]["official_emotion_columns"])
    if labels != accepted_targets:
        raise ValueError("Official target order differs from the accepted BiLSTM")

    perfume_df, base_chunk_df = load_deep_text_dataset(
        input_jsonl=paths["deep_text"],
        fold_manifest_path=paths["outer_manifest"],
        target_cols=labels,
        excluded_target_cols=base_config["targets"]["excluded_emotion_columns"],
        canonical_labels_path=paths["canonical_labels"],
    )
    expected_perfumes = int(
        config["inputs"]["youtube_deep_text_jsonl"]["expected_perfumes"]
    )
    expected_chunks = int(
        config["inputs"]["youtube_deep_text_jsonl"]["expected_chunks"]
    )
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
    inner["outer_fold_id"] = pd.to_numeric(
        inner["outer_fold_id"], errors="raise"
    ).astype(int)
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
            config["inputs"]["synthetic_text_labels"][
                "expected_reviews_per_perfume"
            ]
        ),
        perfume_ids=set(perfume_df["perfume_id"].astype(str)),
    )
    if len(synthetic) != int(
        config["inputs"]["synthetic_text_labels"]["expected_rows"]
    ):
        raise ValueError("Synthetic row count differs from the frozen protocol")

    accepted_oof = pd.read_csv(paths["accepted_oof"], keep_default_na=False)
    accepted_oof["perfume_id"] = accepted_oof["perfume_id"].astype(str)
    accepted_metrics = json.loads(paths["accepted_metrics"].read_text(encoding="utf-8"))
    if accepted_metrics.get("model_id") != "bilstm_attention":
        raise ValueError("Accepted metrics do not describe BiLSTM + Attention")
    if len(accepted_oof) != expected_perfumes:
        raise ValueError("Accepted control OOF has the wrong perfume universe")

    master = build_master_chunk_frame(
        perfume_df, base_chunk_df, synthetic, labels
    )
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
        labels=labels,
        sensitivity_labels=sensitivity_labels,
        settings=settings,
    )


def fold_membership(
    prepared: PreparedInputs, task: FoldTask
) -> dict[str, Any]:
    fold_id = task.fold_id
    fold_inner = prepared.inner_manifest.loc[
        prepared.inner_manifest["outer_fold_id"].eq(fold_id)
    ]
    train_ids = set(
        fold_inner.loc[fold_inner["inner_role"].eq("train"), "perfume_id"]
        .astype(str)
        .tolist()
    )
    val_ids = set(
        fold_inner.loc[fold_inner["inner_role"].eq("val"), "perfume_id"]
        .astype(str)
        .tolist()
    )
    test_ids = set(
        prepared.perfume_df.loc[
            prepared.perfume_df["fold_id"].eq(fold_id), "perfume_id"
        ].astype(str)
    )
    if not train_ids or not val_ids or not test_ids:
        raise ValueError(f"Fold {fold_id} contains an empty split")
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError(f"Perfume leakage in fold {fold_id}")

    perfume = prepared.perfume_df.set_index("perfume_id")
    groups = {
        role: set(perfume.loc[list(ids), SOURCE_GROUP_COLUMN].astype(str))
        for role, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))
    }
    if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]:
        raise ValueError(f"Source-component leakage in fold {fold_id}")

    master = prepared.master_chunk_df
    original = ~master["is_synthetic"].to_numpy(dtype=bool)
    train_mask = original & master["perfume_id"].isin(train_ids).to_numpy()
    if task.condition == AUGMENTED:
        train_mask |= (
            master["is_synthetic"].to_numpy(dtype=bool)
            & master["perfume_id"].isin(train_ids).to_numpy()
        )
    val_mask = original & master["perfume_id"].isin(val_ids).to_numpy()
    test_mask = original & master["perfume_id"].isin(test_ids).to_numpy()

    train_rows = master.loc[train_mask]
    synthetic_train = train_rows.loc[train_rows["is_synthetic"]]
    if set(synthetic_train["perfume_id"]) & (val_ids | test_ids):
        raise ValueError(f"Synthetic validation/test twins entered fold {fold_id}")
    if master.loc[val_mask | test_mask, "is_synthetic"].any():
        raise ValueError("Validation or test contains synthetic text")
    expected_synthetic = 3 * len(train_ids) if task.condition == AUGMENTED else 0
    if len(synthetic_train) != expected_synthetic:
        raise ValueError(f"Unexpected synthetic training rows in fold {fold_id}")

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
    condition: str,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    train_ids = membership["train_ids"]
    train_perfumes = prepared.perfume_df.loc[
        prepared.perfume_df["perfume_id"].isin(train_ids)
    ].sort_values("perfume_id")
    if condition == CONTROL:
        values = compute_pos_weight(train_perfumes, prepared.labels)
        detail = {}
        for index, label in enumerate(prepared.labels):
            positive = float(train_perfumes[label].sum())
            negative = float(len(train_perfumes) - positive)
            detail[label] = {
                "positive_effective_mass": positive,
                "negative_effective_mass": negative,
                "pos_weight": float(values[index]),
            }
        return values.astype(np.float32), detail

    synthetic = prepared.synthetic.loc[
        prepared.synthetic["perfume_id"].isin(train_ids)
    ].sort_values(["perfume_id", "review_index"])
    weights = np.concatenate(
        [
            np.ones(len(train_perfumes), dtype=np.float64),
            np.full(len(synthetic), 1.0 / 3.0, dtype=np.float64),
        ]
    )
    labels = np.vstack(
        [
            train_perfumes[list(prepared.labels)].to_numpy(dtype=np.float64),
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
            "inputs": {
                key: sha256_file(path) for key, path in sorted(prepared.paths.items())
            },
            "labels": list(prepared.labels),
            "model": prepared.config["model"],
            "conditions": prepared.config["conditions"],
            "threshold": float(prepared.config["evaluation"]["threshold"]),
            "software": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
            },
        }
    )


def fold_fingerprint(
    base_fingerprint: str,
    task: FoldTask,
    test_ids: Sequence[str],
) -> str:
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
            summary_json=np.asarray(
                json.dumps(dict(summary), sort_keys=True, allow_nan=False)
            ),
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
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
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


def train_fold(
    prepared: PreparedInputs,
    *,
    task: FoldTask,
    membership: Mapping[str, Any],
    fold_fp: str,
    epoch_path: Path,
    epoch_callback: Callable[[Mapping[str, Any]], None] | None,
    pause_requested: Callable[[], bool],
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
    seed = int(prepared.config["model"]["random_seed"])
    threshold = float(prepared.config["evaluation"]["threshold"])
    device = torch.device("cpu")

    tokens = pretokenize_chunk_texts(master, lowercase=settings.lowercase_tokens)
    vocabulary, vocabulary_summary = build_fold_vocabulary(
        tokens,
        train_mask=train_mask,
        min_token_freq=settings.min_token_freq,
        max_vocab_size=settings.max_vocab_size,
    )
    encoded, encoding_summary = encode_token_sequences(
        tokens, vocab=vocabulary, max_tokens=settings.max_tokens
    )
    label_matrix = master[list(labels)].to_numpy(dtype=np.float32)
    sample_weights = master["base_sample_weight"].to_numpy(dtype=np.float32)
    pos_weight, pos_weight_detail = weighted_pos_weight(
        prepared, membership=membership, condition=task.condition
    )
    pos_weight_tensor = torch.from_numpy(pos_weight).to(device)

    set_global_seed(seed)
    train_loader = build_sequence_loader(
        encoded_inputs=encoded,
        mask=train_mask,
        labels=label_matrix,
        sample_weights=sample_weights,
        batch_size=settings.train_batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = build_sequence_loader(
        encoded_inputs=encoded,
        mask=val_mask,
        labels=label_matrix,
        sample_weights=sample_weights,
        batch_size=settings.eval_batch_size,
        shuffle=False,
        seed=seed,
    )
    test_loader = build_sequence_loader(
        encoded_inputs=encoded,
        mask=test_mask,
        labels=None,
        sample_weights=None,
        batch_size=settings.eval_batch_size,
        shuffle=False,
        seed=seed,
    )
    model = BiLSTMAttentionClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=settings.embedding_dim,
        hidden_size=settings.hidden_size,
        attention_dim=settings.attention_dim,
        lstm_layers=settings.lstm_layers,
        dropout=settings.dropout,
        n_labels=len(labels),
    ).to(device)
    model.float()
    # Match the accepted CPU benchmark's deterministic execution path.
    torch.set_num_threads(1)
    optimizer = AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
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
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        best_state = saved["best_state"]
        best_epoch = int(saved["best_epoch"])
        best_val_metrics = saved["best_val_metrics"]
        best_val_loss = saved["best_val_loss"]
        stale_epochs = int(saved["stale_epochs"])
        epoch_history = list(saved["epoch_history"])
        start_epoch = int(saved["next_epoch"])
        training_complete = bool(saved["training_complete"])
        if train_loader.generator is None:
            raise RuntimeError("Shuffled training loader has no generator")
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
            batch_losses: list[float] = []
            for batch in train_loader:
                input_ids, lengths, batch_labels, batch_weights = batch
                input_ids = input_ids.to(device)
                lengths = lengths.to(device)
                batch_labels = batch_labels.to(device)
                batch_weights = batch_weights.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids, lengths)
                loss = weighted_bce_loss(
                    logits,
                    batch_labels,
                    batch_weights,
                    pos_weight=pos_weight_tensor,
                )
                batch_losses.append(float(loss.detach().cpu()))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings.max_grad_norm
                )
                optimizer.step()

            val_chunk_prob, val_loss = evaluate_sequence_model(
                model,
                val_loader,
                device=device,
                pos_weight_tensor=pos_weight_tensor,
            )
            val_aggregated = aggregate_chunk_probabilities(
                val_chunk_df, val_chunk_prob, target_cols=labels
            )
            val_merged = merge_perfume_probabilities(
                val_perfume_df, val_aggregated, target_cols=labels
            )
            val_truth = val_perfume_df[list(labels)].to_numpy(dtype=int)
            val_probability = val_merged[
                [f"prob_{label}" for label in labels]
            ].to_numpy(dtype=np.float32)
            _, val_metrics = compute_probability_metrics(
                y_true=val_truth,
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
                raise RuntimeError("Shuffled training loader has no generator")
            _atomic_torch_save(
                epoch_path,
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "fingerprint": fold_fp,
                    "condition": task.condition,
                    "fold_id": task.fold_id,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
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
                f"[bilstm synthetic][{task.condition}][fold {task.fold_id}] "
                f"epoch {epoch}/{settings.max_epochs} "
                f"train_loss={record['train_loss']:.4f} "
                f"val_loss={record['val_loss']:.4f} "
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
        raise RuntimeError(f"No best model selected for {task}")
    model.load_state_dict(best_state)
    model.to(device).float()
    # Preserve the accepted runner's final validation-then-test evaluation order.
    evaluate_sequence_model(
        model, val_loader, device=device, pos_weight_tensor=None
    )
    test_chunk_prob, _ = evaluate_sequence_model(
        model, test_loader, device=device, pos_weight_tensor=None
    )
    test_chunk_df = master.loc[test_mask].copy().reset_index(drop=True)
    test_perfume_df = prepared.perfume_df.loc[
        prepared.perfume_df["perfume_id"].isin(test_ids)
    ].sort_values("perfume_id").reset_index(drop=True)
    test_aggregated = aggregate_chunk_probabilities(
        test_chunk_df, test_chunk_prob, target_cols=labels
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
        "synthetic_base_weight": float(
            synthetic_train["base_sample_weight"].sum()
        ),
        "synthetic_validation_or_test_rows": 0,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_macro_f1": float(best_val_metrics["macro_f1"]),
        "best_val_macro_roc_auc": float(best_val_metrics["macro_roc_auc"]),
        "best_val_macro_pr_auc": float(best_val_metrics["macro_pr_auc"]),
        "vocab_size": len(vocabulary),
        "vocabulary_summary": vocabulary_summary,
        "encoding_summary": encoding_summary,
        "loss_pos_weight": pos_weight_detail,
        "epoch_history": epoch_history,
        "n_trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "test_perfume_ids_sha256": hashlib.sha256(
            "\n".join(test_merged["perfume_id"].astype(str)).encode("utf-8")
        ).hexdigest(),
    }
    return probabilities, summary


def canonical_oof_frames(
    prepared: PreparedInputs,
) -> dict[str, pd.DataFrame]:
    base = prepared.perfume_df[
        [
            "perfume_id",
            "Perfume",
            SOURCE_GROUP_COLUMN,
            "source_group_size",
            "fold_id",
        ]
    ].sort_values("perfume_id").reset_index(drop=True)
    outputs: dict[str, pd.DataFrame] = {}
    for condition in CONDITIONS:
        frame = base.copy()
        for label in prepared.labels:
            truth = prepared.perfume_df.set_index("perfume_id").loc[
                frame["perfume_id"], label
            ].to_numpy(dtype=int)
            frame[f"true_{label}"] = truth
            frame[f"prob_{label}"] = np.nan
            frame[f"pred_{label}"] = 0
        outputs[condition] = frame
    return outputs


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
        frame.loc[mask, f"pred_{label}"] = (
            probabilities[:, index] >= threshold
        ).astype(int)


def accepted_control_columns(prepared: PreparedInputs) -> list[str]:
    return [
        "perfume_id",
        "Perfume",
        SOURCE_GROUP_COLUMN,
        "source_group_size",
        "fold_id",
        *(
            column
            for label in prepared.labels
            for column in (f"true_{label}", f"prob_{label}", f"pred_{label}")
        ),
    ]


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
        "experiment_id": "source_grouped_weighted_synthetic_bilstm_attention_v1",
        "experiment_fingerprint": fingerprint,
        "status": status,
        "completed_fold_tasks": completed_folds,
        "total_fold_tasks": total_folds,
        "remaining_fold_tasks": remaining,
        "fold_percent": float(100.0 * completed_folds / total_folds),
        "mean_completed_fold_seconds": average,
        "estimated_remaining_fold_seconds": (
            None if average is None else average * remaining
        ),
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
            f"-m src.training.run_source_grouped_weighted_synthetic_bilstm_attention "
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
    tasks = [FoldTask(condition, fold) for condition in CONDITIONS for fold in folds]
    expected = int(prepared.config["execution"]["fold_task_count"])
    if len(tasks) != expected:
        raise ValueError(f"Expected {expected} fold tasks, built {len(tasks)}")
    threshold = float(prepared.config["evaluation"]["threshold"])
    outputs = canonical_oof_frames(prepared)
    metadata: dict[FoldTask, tuple[list[str], str, Path, Path]] = {}
    cached: dict[FoldTask, tuple[np.ndarray, dict[str, Any]]] = {}
    for task in tasks:
        ids = outputs[task.condition].loc[
            outputs[task.condition]["fold_id"].eq(task.fold_id), "perfume_id"
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
                outputs[task.condition],
                fold_id=task.fold_id,
                probabilities=cached[task][0],
                labels=prepared.labels,
                threshold=threshold,
            )

    completed = len(cached)
    summaries: dict[FoldTask, dict[str, Any]] = {
        task: value[1] for task, value in cached.items()
    }
    durations = [
        float(summary.get("fold_duration_seconds", 0.0))
        for summary in summaries.values()
        if float(summary.get("fold_duration_seconds", 0.0)) > 0
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
    )
    bar = tqdm(
        total=len(tasks),
        initial=completed,
        desc="Weighted BiLSTM",
        unit="fold",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    new_epochs = 0
    control_validation: dict[str, Any] | None = None
    last_checkpoint: Path | None = None
    try:
        for task in tasks:
            if task in cached:
                continue
            if controller.requested:
                break
            if task.condition == AUGMENTED and control_validation is None:
                control_validation = validate_control_replay(
                    outputs[CONTROL],
                    prepared.accepted_oof,
                    prepared.labels,
                    probability_tolerance=float(
                        prepared.config["execution"]["control_probability_tolerance"]
                    ),
                )

            ids, fold_fp, fold_path, epoch_path = metadata[task]
            membership = fold_membership(prepared, task)
            actual_ids = sorted(membership["test_ids"])
            if ids != actual_ids:
                raise RuntimeError("Test perfume ordering changed")
            current_epoch: int | None = None

            def on_epoch(record: Mapping[str, Any]) -> None:
                nonlocal new_epochs, current_epoch, last_checkpoint
                new_epochs += 1
                current_epoch = int(record["epoch"])
                last_checkpoint = epoch_path
                bar.set_postfix_str(
                    f"{task.condition}, fold {task.fold_id}, epoch {current_epoch}/{prepared.settings.max_epochs}"
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
                if (
                    stop_after_new_epochs is not None
                    and new_epochs >= stop_after_new_epochs
                ):
                    controller.request(
                        f"stop-after-new-epochs target {stop_after_new_epochs} reached"
                    )

            started = time.monotonic()
            try:
                probabilities, summary = train_fold(
                    prepared,
                    task=task,
                    membership=membership,
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
                outputs[task.condition],
                fold_id=task.fold_id,
                probabilities=probabilities,
                labels=prepared.labels,
                threshold=threshold,
            )
            summaries[task] = summary
            completed += 1
            durations.append(duration)
            last_checkpoint = fold_path
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
                current_task=None,
                current_epoch=None,
                durations=durations,
                note=(controller.reason or "pause requested")
                + "; rerun the same command to resume",
                last_checkpoint=last_checkpoint,
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
            raise RuntimeError(f"Only {completed}/{len(tasks)} folds completed")
        if control_validation is None:
            control_validation = validate_control_replay(
                outputs[CONTROL],
                prepared.accepted_oof,
                prepared.labels,
                probability_tolerance=float(
                    prepared.config["execution"]["control_probability_tolerance"]
                ),
            )
        probability_columns = [f"prob_{label}" for label in prepared.labels]
        for condition in CONDITIONS:
            if outputs[condition][probability_columns].isna().any().any():
                raise RuntimeError(f"Incomplete OOF matrix for {condition}")
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


def summarize_conditions(
    prepared: PreparedInputs,
    oof_by_condition: Mapping[str, pd.DataFrame],
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
    return pd.DataFrame(rows), pd.concat(per_label_frames, ignore_index=True)


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


def build_markdown_report(
    summary: pd.DataFrame, differences: pd.DataFrame
) -> str:
    lines = [
        "# Weighted Synthetic-Augmentation BiLSTM + Attention Comparison",
        "",
        "The accepted joint seven-label BiLSTM + Attention model was replayed "
        "under the frozen source-grouped protocol. The augmented condition adds "
        "three synthetic reviews only to each inner-training perfume at base loss "
        "weight 1/3 each. Validation and testing contain YouTube text only.",
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
) -> dict[str, Path]:
    paths = artifact_paths(output_root)
    accepted = prepared.accepted_oof[accepted_control_columns(prepared)].copy()
    replay = fit_result["outputs"][CONTROL]
    augmented = fit_result["outputs"][AUGMENTED]
    control_validation = validate_control_replay(
        replay,
        accepted,
        prepared.labels,
        probability_tolerance=float(
            prepared.config["execution"]["control_probability_tolerance"]
        ),
    )
    analysis_oof = {CONTROL: accepted, AUGMENTED: augmented}
    summary, per_label = summarize_conditions(prepared, analysis_oof)

    audit_rows: list[dict[str, Any]] = []
    for task, detail in fit_result["summaries"].items():
        row = {
            key: value
            for key, value in detail.items()
            if key not in {
                "epoch_history",
                "loss_pos_weight",
                "vocabulary_summary",
                "encoding_summary",
            }
        }
        row["loss_pos_weight_json"] = json.dumps(
            detail["loss_pos_weight"], sort_keys=True
        )
        row["epoch_history_json"] = json.dumps(detail["epoch_history"])
        audit_rows.append(row)
    fold_audit = pd.DataFrame(audit_rows).sort_values(
        ["condition", "fold_id"]
    ).reset_index(drop=True)

    bootstrap = prepared.config["evaluation"]["paired_source_component_bootstrap"]
    total_draws = bootstrap_reps * 2
    bar = tqdm(
        total=total_draws,
        desc="Paired bootstrap",
        unit="draw",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    completed_draws = 0

    def bootstrap_progress(_scope: str, _completed: int, _total: int) -> None:
        nonlocal completed_draws
        completed_draws += 1
        bar.update(1)

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
    atomic_write_text(
        paths["markdown_report"], build_markdown_report(summary, differences)
    )

    validation = {
        "status": "passed",
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
        "joint_multilabel_fold_models": 10,
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
        "model": prepared.config["model"],
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
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "interpretation_boundary": prepared.config["experiment"][
            "interpretation_boundary"
        ],
    }
    atomic_write_json(paths["report"], report)
    checksum_paths = {
        key: path
        for key, path in paths.items()
        if key not in {"checksums", "progress"}
    }
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
            "completed_fold_tasks": 10,
            "total_fold_tasks": 10,
            "bootstrap_completed_draws": total_draws,
            "bootstrap_total_draws": total_draws,
            "updated_utc": utc_text(),
        },
    )
    return paths


def preflight_report(prepared: PreparedInputs) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    folds = sorted(prepared.perfume_df["fold_id"].astype(int).unique())
    for condition in CONDITIONS:
        for fold in folds:
            task = FoldTask(condition, int(fold))
            membership = fold_membership(prepared, task)
            pos_weight, _ = weighted_pos_weight(
                prepared, membership=membership, condition=condition
            )
            train = prepared.master_chunk_df.loc[membership["train_mask"]]
            rows.append(
                {
                    "condition": condition,
                    "fold_id": int(fold),
                    "train_perfumes": len(membership["train_ids"]),
                    "val_perfumes": len(membership["val_ids"]),
                    "test_perfumes": len(membership["test_ids"]),
                    "train_youtube_chunks": int((~train["is_synthetic"]).sum()),
                    "train_synthetic_reviews": int(train["is_synthetic"].sum()),
                    "validation_youtube_chunks": int(membership["val_mask"].sum()),
                    "test_youtube_chunks": int(membership["test_mask"].sum()),
                    "pos_weight_finite": bool(np.isfinite(pos_weight).all()),
                }
            )
    return {
        "status": "passed",
        "input_hashes": {
            key: sha256_file(path) for key, path in prepared.paths.items()
        },
        "perfumes": len(prepared.perfume_df),
        "youtube_chunks": len(prepared.base_chunk_df),
        "synthetic_reviews": len(prepared.synthetic),
        "source_components": int(
            prepared.perfume_df[SOURCE_GROUP_COLUMN].nunique()
        ),
        "folds": rows,
    }


def run_experiment(
    *,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path | None = None,
    bootstrap_reps: int | None = None,
    show_progress: bool = True,
    stop_after_new_epochs: int | None = None,
    preflight_only: bool = False,
) -> dict[str, Any]:
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
            prepared.config["evaluation"]["paired_source_component_bootstrap"][
                "repetitions"
            ]
        )
        reps = configured_reps if bootstrap_reps is None else int(bootstrap_reps)
        if reps < 2:
            raise ValueError("Bootstrap repetitions must be at least two")
        paths = finalize_outputs(
            prepared,
            output_root=output_root,
            base_fingerprint=base_fingerprint,
            fit_result=fit_result,
            bootstrap_reps=reps,
            show_progress=show_progress,
        )
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
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_experiment(
        config_path=args.config,
        out_dir=args.out_dir,
        bootstrap_reps=args.bootstrap_reps,
        show_progress=not args.no_progress,
        stop_after_new_epochs=args.stop_after_new_epochs,
        preflight_only=args.preflight_only,
    )
    if result["status"] == "preflight_complete":
        print(f"BiLSTM augmentation preflight passed: {result['preflight']}")
        return 0
    if result["status"] == "paused":
        print("BiLSTM augmentation paused safely; rerun the same command to resume.")
        print(f"Progress: {result['progress']}")
        return 0

    paths = result["paths"]
    summary = pd.read_csv(paths["condition_summary"])
    differences = pd.read_csv(paths["paired_differences"])
    print("Weighted synthetic BiLSTM + Attention comparison complete")
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
