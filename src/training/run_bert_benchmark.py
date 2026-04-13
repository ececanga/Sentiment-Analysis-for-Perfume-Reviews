"""Run the official fine-tuned BERT benchmark on the canonical deep-text dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, roc_auc_score
from sklearn.model_selection import KFold
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.model_benchmark_runner import (
    collect_package_versions,
    get_git_commit,
    load_config,
    save_curve_plot,
    save_per_label_plot,
    stringify_path,
    update_comparison_table,
)
from src.utils.paths import PROJECT_ROOT

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "bert"
MODEL_FAMILY = "BERT"
DEFAULT_CHECKPOINT = "monologg/bert-base-cased-goemotions-original"
INNER_MANIFEST_COLUMNS = ["outer_fold_id", "perfume_id", "Perfume", "inner_role"]
TRAINING_MODE = "end_to_end_finetune"
FROZEN_ENCODER_MODE = "frozen_encoder_head"


@dataclass(frozen=True)
class BertRunSettings:
    checkpoint: str
    local_files_only: bool
    max_length: int
    aggregation_strategy: str
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    min_delta: float
    warmup_ratio: float
    gradient_accumulation_steps: int
    max_grad_norm: float
    selection_metric: str


def _metric_or_nan(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def set_global_seed(seed: int) -> Dict[str, bool]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    deterministic_enabled = False
    try:
        torch.use_deterministic_algorithms(True)
        deterministic_enabled = True
    except Exception:
        deterministic_enabled = False

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return {
        "torch_use_deterministic_algorithms": deterministic_enabled,
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
    }


def detect_device(preferred: str = "auto") -> torch.device:
    if preferred not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device option: {preferred}")

    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if preferred == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_bert_settings(config: Mapping[str, Any]) -> BertRunSettings:
    models_cfg = config.get("models", {})
    bert_cfg = models_cfg.get(MODEL_ID, {}) if isinstance(models_cfg, Mapping) else {}
    return BertRunSettings(
        checkpoint=str(bert_cfg.get("checkpoint", DEFAULT_CHECKPOINT)),
        local_files_only=bool(bert_cfg.get("local_files_only", True)),
        max_length=int(bert_cfg.get("max_length", 96)),
        aggregation_strategy=str(bert_cfg.get("aggregation_strategy", "mean_probability")),
        train_batch_size=int(bert_cfg.get("train_batch_size", 8)),
        eval_batch_size=int(bert_cfg.get("eval_batch_size", 16)),
        learning_rate=float(bert_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(bert_cfg.get("weight_decay", 0.01)),
        max_epochs=int(bert_cfg.get("max_epochs", 3)),
        patience=int(bert_cfg.get("patience", 1)),
        min_delta=float(bert_cfg.get("min_delta", 1e-4)),
        warmup_ratio=float(bert_cfg.get("warmup_ratio", 0.1)),
        gradient_accumulation_steps=int(bert_cfg.get("gradient_accumulation_steps", 1)),
        max_grad_norm=float(bert_cfg.get("max_grad_norm", 1.0)),
        selection_metric=str(bert_cfg.get("selection_metric", "macro_f1")),
    )


def load_deep_text_dataset(
    *,
    input_jsonl: Path,
    fold_manifest_path: Path,
    target_cols: Sequence[str],
    excluded_target_cols: Mapping[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Deep-text dataset not found: {input_jsonl}")
    if not fold_manifest_path.exists():
        raise FileNotFoundError(f"Frozen fold manifest not found: {fold_manifest_path}")

    perfume_rows: List[Dict[str, Any]] = []
    chunk_rows: List[Dict[str, Any]] = []
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = json.loads(line)
            perfume_id = str(payload.get("perfume_id", "")).strip()
            perfume_name = str(payload.get("perfume_name", "")).strip()
            if not perfume_id:
                raise ValueError(f"Missing perfume_id at line {line_no} in {input_jsonl}")

            labels = payload.get("labels")
            excluded_labels = payload.get("excluded_labels")
            review_chunks = payload.get("review_chunks")
            if not isinstance(labels, dict):
                raise ValueError(f"Missing labels object for perfume_id={perfume_id}")
            if not isinstance(excluded_labels, dict):
                raise ValueError(f"Missing excluded_labels object for perfume_id={perfume_id}")
            if not isinstance(review_chunks, list) or not review_chunks:
                raise ValueError(f"Missing review_chunks for perfume_id={perfume_id}")

            fold_id = int(payload.get("fold_id", 0))
            perfume_row: Dict[str, Any] = {
                "perfume_id": perfume_id,
                "Perfume": perfume_name,
                "fold_id": fold_id,
                "chunk_count": len(review_chunks),
            }
            for label in target_cols:
                value = pd.to_numeric(labels.get(label), errors="coerce")
                if pd.isna(value) or int(value) not in (0, 1):
                    raise ValueError(f"Invalid label value for {label} perfume_id={perfume_id}")
                perfume_row[label] = int(value)
            for label in excluded_target_cols:
                value = pd.to_numeric(excluded_labels.get(label), errors="coerce")
                if pd.isna(value) or int(value) not in (0, 1):
                    raise ValueError(f"Invalid excluded label value for {label} perfume_id={perfume_id}")
                perfume_row[label] = int(value)
            perfume_rows.append(perfume_row)

            ordered_chunks = sorted(review_chunks, key=lambda item: int(item.get("chunk_index", 0)))
            for chunk in ordered_chunks:
                chunk_text = str(chunk.get("chunk_text", "")).strip()
                if not chunk_text:
                    raise ValueError(f"Empty chunk text for perfume_id={perfume_id}")
                chunk_row = {
                    "perfume_id": perfume_id,
                    "Perfume": perfume_name,
                    "fold_id": fold_id,
                    "chunk_count": len(review_chunks),
                    "chunk_index": int(chunk.get("chunk_index", 0)),
                    "chunk_text": chunk_text,
                }
                for label in target_cols:
                    chunk_row[label] = perfume_row[label]
                chunk_rows.append(chunk_row)

    perfume_df = pd.DataFrame(perfume_rows).sort_values("perfume_id").reset_index(drop=True)
    if perfume_df.empty:
        raise ValueError("Deep-text dataset is empty.")
    if not perfume_df["perfume_id"].is_unique:
        raise ValueError("perfume_id must be unique in the deep-text dataset.")

    chunk_df = pd.DataFrame(chunk_rows).sort_values(["perfume_id", "chunk_index"]).reset_index(drop=True)
    if chunk_df.empty:
        raise ValueError("Deep-text chunk dataset is empty.")

    manifest_df = pd.read_csv(fold_manifest_path).sort_values("perfume_id").reset_index(drop=True)
    if "perfume_id" not in manifest_df.columns or "fold_id" not in manifest_df.columns:
        raise ValueError(f"Fold manifest missing required columns: {fold_manifest_path}")
    manifest_df["perfume_id"] = manifest_df["perfume_id"].astype(str)
    manifest_df["fold_id"] = pd.to_numeric(manifest_df["fold_id"], errors="raise").astype(int)

    if perfume_df["perfume_id"].tolist() != manifest_df["perfume_id"].tolist():
        raise ValueError("Deep-text dataset perfume_id set/order does not match the frozen fold manifest.")
    if perfume_df["fold_id"].tolist() != manifest_df["fold_id"].tolist():
        raise ValueError("Deep-text dataset fold ids do not match the frozen fold manifest.")
    if "Perfume" in manifest_df.columns:
        manifest_names = manifest_df["Perfume"].astype(str).tolist()
        if manifest_names != perfume_df["Perfume"].astype(str).tolist():
            raise ValueError("Deep-text dataset perfume names do not match the frozen fold manifest.")

    return perfume_df, chunk_df


def build_expected_inner_validation_manifest(
    perfume_df: pd.DataFrame,
    *,
    inner_n_splits: int,
    seed: int,
) -> pd.DataFrame:
    if inner_n_splits < 2:
        raise ValueError("Inner validation n_splits must be >= 2.")

    rows: List[Dict[str, Any]] = []
    for outer_fold_id in sorted(perfume_df["fold_id"].unique().tolist()):
        outer_train_df = perfume_df.loc[perfume_df["fold_id"] != outer_fold_id].sort_values("perfume_id").reset_index(drop=True)
        n_splits = min(inner_n_splits, len(outer_train_df))
        if n_splits < 2:
            raise ValueError(f"Outer fold {outer_fold_id} has too few training perfumes for inner validation.")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        _, val_idx = next(splitter.split(outer_train_df))
        val_ids = set(outer_train_df.iloc[val_idx]["perfume_id"].tolist())
        for row in outer_train_df.itertuples(index=False):
            rows.append(
                {
                    "outer_fold_id": int(outer_fold_id),
                    "perfume_id": str(row.perfume_id),
                    "Perfume": str(row.Perfume),
                    "inner_role": "val" if row.perfume_id in val_ids else "train",
                }
            )
    return pd.DataFrame(rows, columns=INNER_MANIFEST_COLUMNS)


def load_or_write_inner_validation_manifest(
    expected_df: pd.DataFrame,
    *,
    output_path: Path,
) -> pd.DataFrame:
    expected_sorted = expected_df.sort_values(["outer_fold_id", "perfume_id"]).reset_index(drop=True)
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        if list(existing_df.columns) != INNER_MANIFEST_COLUMNS:
            raise ValueError(f"Existing inner validation manifest has unexpected columns: {output_path}")
        existing_sorted = existing_df.sort_values(["outer_fold_id", "perfume_id"]).reset_index(drop=True)
        if not expected_sorted.equals(existing_sorted):
            raise ValueError(f"Existing inner validation manifest does not match the frozen rule: {output_path}")
        return existing_sorted

    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_sorted.to_csv(output_path, index=False)
    return expected_sorted


def tokenize_chunk_texts(
    chunk_df: pd.DataFrame,
    *,
    model_name: str,
    max_length: int,
    local_files_only: bool,
) -> Dict[str, np.ndarray]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load BERT tokenizer '{model_name}'. Ensure the model is cached locally or update the config."
        ) from exc

    encoded = tokenizer(
        chunk_df["chunk_text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    tokenized: Dict[str, np.ndarray] = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in encoded:
        tokenized["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
    else:
        tokenized["token_type_ids"] = np.zeros_like(tokenized["input_ids"], dtype=np.int64)
    return tokenized


def compute_pos_weight(train_perfume_df: pd.DataFrame, target_cols: Sequence[str]) -> np.ndarray:
    weights: List[float] = []
    n_rows = float(len(train_perfume_df))
    for label in target_cols:
        pos = float(train_perfume_df[label].sum())
        neg = n_rows - pos
        if pos <= 0.0:
            weights.append(1.0)
        else:
            weights.append(neg / pos)
    return np.asarray(weights, dtype=np.float32)


def build_loader_from_arrays(
    *,
    tokenized_inputs: Mapping[str, np.ndarray],
    mask: np.ndarray,
    labels: np.ndarray | None,
    sample_weights: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    mask_idx = np.where(mask)[0]
    tensors: List[torch.Tensor] = [
        torch.from_numpy(tokenized_inputs["input_ids"][mask_idx]).long(),
        torch.from_numpy(tokenized_inputs["attention_mask"][mask_idx]).long(),
        torch.from_numpy(tokenized_inputs["token_type_ids"][mask_idx]).long(),
    ]
    if labels is not None:
        tensors.append(torch.from_numpy(labels[mask_idx].astype(np.float32)))
    if sample_weights is not None:
        tensors.append(torch.from_numpy(sample_weights[mask_idx].astype(np.float32)))

    dataset = TensorDataset(*tensors)
    actual_batch_size = min(batch_size, len(dataset)) if len(dataset) else 1
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def weighted_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    per_label = nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
        reduction="none",
    )
    per_row = per_label.mean(dim=1)
    return (per_row * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def aggregate_chunk_probabilities(
    chunk_df: pd.DataFrame,
    chunk_probs: np.ndarray,
    *,
    target_cols: Sequence[str],
) -> pd.DataFrame:
    if len(chunk_df) != len(chunk_probs):
        raise ValueError("chunk_df and chunk_probs must have the same number of rows.")

    prob_cols = [f"prob_{label}" for label in target_cols]
    work_df = chunk_df[["perfume_id", "Perfume", "fold_id", "chunk_count"]].copy()
    for idx, label in enumerate(target_cols):
        work_df[f"prob_{label}"] = chunk_probs[:, idx]

    agg_spec: Dict[str, Any] = {
        "Perfume": "first",
        "fold_id": "first",
        "chunk_count": "first",
    }
    for col in prob_cols:
        agg_spec[col] = "mean"

    return work_df.groupby("perfume_id", sort=True, as_index=False).agg(agg_spec)


def compute_probability_metrics(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_cols: Sequence[str],
    threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    y_pred = (y_prob >= threshold).astype(int)
    per_label_rows: List[Dict[str, Any]] = []
    roc_values: List[float] = []
    pr_values: List[float] = []

    for label_idx, label in enumerate(target_cols):
        yt = y_true[:, label_idx]
        yp = y_pred[:, label_idx]
        yp_prob = y_prob[:, label_idx]
        roc_auc = _metric_or_nan(roc_auc_score, yt, yp_prob)
        pr_auc = _metric_or_nan(average_precision_score, yt, yp_prob)
        roc_values.append(roc_auc)
        pr_values.append(pr_auc)
        per_label_rows.append(
            {
                "label": label,
                "support_pos": int(yt.sum()),
                "support_neg": int(len(yt) - yt.sum()),
                "f1": _metric_or_nan(f1_score, yt, yp, zero_division=0),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

    macro_metrics = {
        "macro_f1": float(np.mean([row["f1"] for row in per_label_rows])),
        "macro_roc_auc": float(np.nanmean(roc_values)),
        "macro_pr_auc": float(np.nanmean(pr_values)),
        "micro_f1": _metric_or_nan(f1_score, y_true, y_pred, average="micro", zero_division=0),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": _metric_or_nan(hamming_loss, y_true, y_pred),
        "label_cardinality": float(np.mean(np.sum(y_true, axis=1))),
    }
    return per_label_rows, macro_metrics


def merge_perfume_probabilities(
    perfume_df: pd.DataFrame,
    aggregated_probs: pd.DataFrame,
    *,
    target_cols: Sequence[str],
) -> pd.DataFrame:
    merged = perfume_df[["perfume_id", "Perfume", "fold_id", "chunk_count"]].merge(
        aggregated_probs,
        on="perfume_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_agg"),
    )
    prob_cols = [f"prob_{label}" for label in target_cols]
    if merged[prob_cols].isna().any().any():
        raise ValueError("Missing perfume-level probabilities after chunk aggregation.")
    return merged


def evaluate_model_on_mask(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    pos_weight_tensor: torch.Tensor | None,
) -> Tuple[np.ndarray, float]:
    model.eval()
    prob_batches: List[np.ndarray] = []
    losses: List[float] = []

    with torch.inference_mode():
        for batch in loader:
            input_ids, attention_mask, token_type_ids = [item.to(device) for item in batch[:3]]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            prob_batches.append(probs)

            if len(batch) >= 5 and pos_weight_tensor is not None:
                labels = batch[3].to(device)
                sample_weights = batch[4].to(device)
                loss = weighted_bce_loss(logits, labels, sample_weights, pos_weight=pos_weight_tensor)
                losses.append(float(loss.detach().cpu()))

    out_probs = np.vstack(prob_batches) if prob_batches else np.zeros((0, 0), dtype=np.float32)
    out_loss = float(np.mean(losses)) if losses else float("nan")
    return out_probs, out_loss


def load_previous_bert_summary(metrics_path: Path) -> Dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    model_params = payload.get("model_parameters", {})
    training_mode = str(model_params.get("training_mode", "")).strip()
    if not training_mode:
        if bool(model_params.get("freeze_encoder", False)):
            training_mode = FROZEN_ENCODER_MODE
        else:
            training_mode = "unknown"

    return {
        "training_mode": training_mode,
        "macro_f1": float(payload.get("macro_f1", float("nan"))),
        "macro_roc_auc": float(payload.get("macro_roc_auc", float("nan"))),
        "macro_pr_auc": float(payload.get("macro_pr_auc", float("nan"))),
        "run_id": str(payload.get("run_id", "")),
    }


def build_previous_run_comparison(
    previous_run: Dict[str, Any] | None,
    current_metrics: Mapping[str, float],
) -> Dict[str, Any] | None:
    if previous_run is None:
        return None

    return {
        "previous_training_mode": previous_run["training_mode"],
        "previous_macro_f1": previous_run["macro_f1"],
        "previous_macro_roc_auc": previous_run["macro_roc_auc"],
        "previous_macro_pr_auc": previous_run["macro_pr_auc"],
        "delta_macro_f1": current_metrics["macro_f1"] - previous_run["macro_f1"],
        "delta_macro_roc_auc": current_metrics["macro_roc_auc"] - previous_run["macro_roc_auc"],
        "delta_macro_pr_auc": current_metrics["macro_pr_auc"] - previous_run["macro_pr_auc"],
    }


def build_bert_interpretation_note(
    *,
    settings: BertRunSettings,
    per_label_df: pd.DataFrame,
    global_metrics: Mapping[str, float],
    previous_run_comparison: Mapping[str, Any] | None,
) -> str:
    strongest = per_label_df.sort_values("f1", ascending=False)["label"].head(2).tolist()
    weakest = per_label_df.sort_values("support_pos", ascending=True)["label"].head(4).tolist()
    strongest_text = ", ".join(strongest) if strongest else "the supported labels"
    weakest_text = ", ".join(weakest) if weakest else "the sparse labels"

    sentence_one = (
        f"BERT now fine-tunes the full {settings.checkpoint} model end-to-end on chunk-level perfume reviews, "
        f"then aggregates chunk probabilities back to perfume level by deterministic arithmetic mean, reaching "
        f"macro-F1 {global_metrics['macro_f1']:.4f}, macro ROC-AUC {global_metrics['macro_roc_auc']:.4f}, and "
        f"macro PR-AUC {global_metrics['macro_pr_auc']:.4f}."
    )

    if previous_run_comparison is not None and previous_run_comparison["previous_training_mode"] == FROZEN_ENCODER_MODE:
        sentence_two = (
            f"This replaces the earlier frozen-encoder run "
            f"(macro-F1 {previous_run_comparison['previous_macro_f1']:.4f}, macro ROC-AUC "
            f"{previous_run_comparison['previous_macro_roc_auc']:.4f}, macro PR-AUC "
            f"{previous_run_comparison['previous_macro_pr_auc']:.4f}) with deltas of "
            f"{previous_run_comparison['delta_macro_f1']:+.4f}, "
            f"{previous_run_comparison['delta_macro_roc_auc']:+.4f}, and "
            f"{previous_run_comparison['delta_macro_pr_auc']:+.4f} respectively."
        )
    else:
        sentence_two = (
            f"It is strongest on {strongest_text}, while {weakest_text} remain unstable under extremely sparse "
            f"support and source noise."
        )

    sentence_three = (
        f"It is strongest on {strongest_text}, while {weakest_text} remain unstable under extremely sparse "
        f"support and source noise. The row is benchmark-ready for the official comparison table."
    )
    return " ".join([sentence_one, sentence_two, sentence_three])


def choose_best_epoch(
    *,
    candidate_metrics: Mapping[str, float],
    candidate_loss: float,
    best_metrics: Mapping[str, float] | None,
    best_loss: float | None,
    min_delta: float,
) -> bool:
    candidate_score = float(candidate_metrics["macro_f1"])
    if best_metrics is None:
        return True

    best_score = float(best_metrics["macro_f1"])
    if candidate_score > best_score + min_delta:
        return True
    if candidate_score < best_score - min_delta:
        return False

    candidate_pr = float(candidate_metrics["macro_pr_auc"])
    best_pr = float(best_metrics["macro_pr_auc"])
    if candidate_pr > best_pr + min_delta:
        return True
    if candidate_pr < best_pr - min_delta:
        return False

    if best_loss is None or not np.isfinite(best_loss):
        return True
    if not np.isfinite(candidate_loss):
        return False
    return bool(candidate_loss < best_loss - 1e-8)


def run_finetuned_bert_outer_fold(
    *,
    tokenized_inputs: Mapping[str, np.ndarray],
    chunk_df: pd.DataFrame,
    train_chunk_mask: np.ndarray,
    val_chunk_mask: np.ndarray,
    test_chunk_mask: np.ndarray,
    train_perfume_df: pd.DataFrame,
    val_perfume_df: pd.DataFrame,
    test_perfume_df: pd.DataFrame,
    target_cols: Sequence[str],
    settings: BertRunSettings,
    threshold: float,
    seed: int,
    device: torch.device,
    outer_fold_id: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    set_global_seed(seed)

    label_matrix = chunk_df[target_cols].to_numpy(dtype=np.float32)
    sample_weights = (1.0 / chunk_df["chunk_count"].to_numpy(dtype=np.float32)).astype(np.float32)
    pos_weight = compute_pos_weight(train_perfume_df, target_cols)
    pos_weight_tensor = torch.from_numpy(pos_weight.astype(np.float32)).to(device)

    train_loader = build_loader_from_arrays(
        tokenized_inputs=tokenized_inputs,
        mask=train_chunk_mask,
        labels=label_matrix,
        sample_weights=sample_weights,
        batch_size=settings.train_batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = build_loader_from_arrays(
        tokenized_inputs=tokenized_inputs,
        mask=val_chunk_mask,
        labels=label_matrix,
        sample_weights=sample_weights,
        batch_size=settings.eval_batch_size,
        shuffle=False,
        seed=seed,
    )
    test_loader = build_loader_from_arrays(
        tokenized_inputs=tokenized_inputs,
        mask=test_chunk_mask,
        labels=None,
        sample_weights=None,
        batch_size=settings.eval_batch_size,
        shuffle=False,
        seed=seed,
    )

    label2id = {label: idx for idx, label in enumerate(target_cols)}
    id2label = {idx: label for label, idx in label2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.checkpoint,
        local_files_only=settings.local_files_only,
        num_labels=len(target_cols),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
        label2id=label2id,
        id2label=id2label,
    )
    model.to(device)
    if device.type == "cpu":
        model.float()

    optimizer = AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    grad_steps = max(1, settings.gradient_accumulation_steps)
    steps_per_epoch = int(np.ceil(max(1, len(train_loader)) / grad_steps))
    total_train_steps = max(1, settings.max_epochs * steps_per_epoch)
    warmup_steps = int(round(total_train_steps * settings.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_metrics: Dict[str, float] | None = None
    best_val_loss: float | None = None
    stale_epochs = 0
    epoch_history: List[Dict[str, Any]] = []

    val_chunk_df = chunk_df.loc[val_chunk_mask].copy().reset_index(drop=True)
    test_chunk_df = chunk_df.loc[test_chunk_mask].copy().reset_index(drop=True)

    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_losses: List[float] = []

        for step, batch in enumerate(train_loader, start=1):
            input_ids, attention_mask, token_type_ids, labels, batch_weights = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            batch_weights = batch_weights.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            batch_loss = weighted_bce_loss(outputs.logits, labels, batch_weights, pos_weight=pos_weight_tensor)
            batch_losses.append(float(batch_loss.detach().cpu()))
            (batch_loss / grad_steps).backward()

            should_step = step % grad_steps == 0 or step == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        val_chunk_probs, val_loss = evaluate_model_on_mask(
            model,
            val_loader,
            device=device,
            pos_weight_tensor=pos_weight_tensor,
        )
        val_aggregated = aggregate_chunk_probabilities(
            val_chunk_df,
            val_chunk_probs,
            target_cols=target_cols,
        )
        val_perfume_probs = merge_perfume_probabilities(
            val_perfume_df,
            val_aggregated,
            target_cols=target_cols,
        )
        _, val_metrics = compute_probability_metrics(
            y_true=val_perfume_df[target_cols].to_numpy(dtype=int),
            y_prob=val_perfume_probs[[f"prob_{label}" for label in target_cols]].to_numpy(dtype=np.float32),
            target_cols=target_cols,
            threshold=threshold,
        )

        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(batch_losses)) if batch_losses else float("nan"),
            "val_loss": float(val_loss),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_macro_roc_auc": float(val_metrics["macro_roc_auc"]),
            "val_macro_pr_auc": float(val_metrics["macro_pr_auc"]),
        }
        epoch_history.append(epoch_record)
        print(
            f"[bert][fold {outer_fold_id}] epoch {epoch}/{settings.max_epochs} "
            f"train_loss={epoch_record['train_loss']:.4f} "
            f"val_loss={epoch_record['val_loss']:.4f} "
            f"val_macro_f1={epoch_record['val_macro_f1']:.4f}"
        )

        is_better = choose_best_epoch(
            candidate_metrics=val_metrics,
            candidate_loss=val_loss,
            best_metrics=best_val_metrics,
            best_loss=best_val_loss,
            min_delta=settings.min_delta,
        )
        if is_better:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_val_loss = float(val_loss)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= settings.patience:
                break

    if best_state is None or best_val_metrics is None:
        raise RuntimeError(f"No best model checkpoint selected for outer fold {outer_fold_id}")

    model.load_state_dict(best_state)
    model.to(device)
    if device.type == "cpu":
        model.float()

    test_chunk_probs, _ = evaluate_model_on_mask(
        model,
        test_loader,
        device=device,
        pos_weight_tensor=None,
    )

    fold_summary = {
        "outer_fold_id": int(outer_fold_id),
        "n_train_perfumes": int(len(train_perfume_df)),
        "n_val_perfumes": int(len(val_perfume_df)),
        "n_test_perfumes": int(len(test_perfume_df)),
        "n_train_chunks": int(train_chunk_mask.sum()),
        "n_val_chunks": int(val_chunk_mask.sum()),
        "n_test_chunks": int(test_chunk_mask.sum()),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_macro_f1": float(best_val_metrics["macro_f1"]),
        "best_val_macro_roc_auc": float(best_val_metrics["macro_roc_auc"]),
        "best_val_macro_pr_auc": float(best_val_metrics["macro_pr_auc"]),
        "train_positive_perfumes": {label: int(train_perfume_df[label].sum()) for label in target_cols},
        "val_positive_perfumes": {label: int(val_perfume_df[label].sum()) for label in target_cols},
        "test_positive_perfumes": {label: int(test_perfume_df[label].sum()) for label in target_cols},
        "epoch_history": epoch_history,
    }
    return test_chunk_probs, fold_summary


def run_bert_benchmark(
    *,
    input_jsonl: Path,
    out_root: Path,
    config_path: Path,
    device_preference: str = "auto",
) -> Dict[str, Any]:
    config = load_config(config_path)
    settings = load_bert_settings(config)
    target_cols = list(config["targets"]["official_emotion_columns"])
    excluded_target_cols = dict(config["targets"]["excluded_emotion_columns"])
    seed = int(config["seeds"]["data_seed"])
    threshold = float(config["evaluation"]["probability_threshold"])
    outer_n_splits = int(config["splits"]["outer_cv"]["n_splits"])
    inner_cfg = config["splits"]["inner_validation"]
    inner_n_splits = int(inner_cfg["n_splits"])

    deterministic_flags = set_global_seed(seed)
    device = detect_device(device_preference)

    metrics_dir = out_root / "metrics"
    figures_dir = out_root / "figures"
    predictions_dir = out_root / "predictions"
    logs_dir = out_root / "logs"
    tables_dir = out_root / "tables"
    for path in [metrics_dir, figures_dir, predictions_dir, logs_dir, tables_dir]:
        path.mkdir(parents=True, exist_ok=True)

    predictions_path = predictions_dir / f"{MODEL_ID}_oof.csv"
    per_label_path = metrics_dir / f"{MODEL_ID}_per_label.csv"
    metrics_path = metrics_dir / f"{MODEL_ID}.json"
    per_label_plot_path = figures_dir / f"{MODEL_ID}_per_label_metrics.png"
    curve_plot_path = figures_dir / f"{MODEL_ID}_roc_pr_curves.png"
    log_path = logs_dir / f"{MODEL_ID}.log"
    run_manifest_path = logs_dir / f"{MODEL_ID}_run.json"
    note_path = tables_dir / f"{MODEL_ID}_note.md"
    comparison_table_path = tables_dir / "comparison_table.csv"
    comparison_template_path = PROJECT_ROOT / "docs" / "comparison_table_template.csv"

    previous_run = load_previous_bert_summary(metrics_path)

    fold_manifest_path = tables_dir / "fold_assignments.csv"
    perfume_df, chunk_df = load_deep_text_dataset(
        input_jsonl=input_jsonl,
        fold_manifest_path=fold_manifest_path,
        target_cols=target_cols,
        excluded_target_cols=excluded_target_cols,
    )

    expected_inner_manifest = build_expected_inner_validation_manifest(
        perfume_df,
        inner_n_splits=inner_n_splits,
        seed=seed,
    )
    inner_manifest_path = tables_dir / f"{MODEL_ID}_inner_val_folds.csv"
    inner_manifest_df = load_or_write_inner_validation_manifest(
        expected_inner_manifest,
        output_path=inner_manifest_path,
    )

    tokenized_inputs = tokenize_chunk_texts(
        chunk_df,
        model_name=settings.checkpoint,
        max_length=settings.max_length,
        local_files_only=settings.local_files_only,
    )

    y_true = perfume_df[target_cols].to_numpy(dtype=int)
    oof_prob = np.full((len(perfume_df), len(target_cols)), np.nan, dtype=np.float32)
    fold_summaries: List[Dict[str, Any]] = []

    fold_ids = sorted(perfume_df["fold_id"].unique().tolist())
    if fold_ids != list(range(1, outer_n_splits + 1)):
        raise ValueError(f"Expected frozen outer folds 1..{outer_n_splits}, got {fold_ids}")

    for outer_fold_id in fold_ids:
        print(f"[bert] starting outer fold {outer_fold_id}/{outer_n_splits}")
        test_perfume_df = perfume_df.loc[perfume_df["fold_id"] == outer_fold_id].sort_values("perfume_id").reset_index(drop=True)
        inner_fold_df = inner_manifest_df.loc[inner_manifest_df["outer_fold_id"] == outer_fold_id].copy()
        train_perfume_ids = set(inner_fold_df.loc[inner_fold_df["inner_role"] == "train", "perfume_id"].tolist())
        val_perfume_ids = set(inner_fold_df.loc[inner_fold_df["inner_role"] == "val", "perfume_id"].tolist())
        if not train_perfume_ids or not val_perfume_ids:
            raise ValueError(f"Inner validation split is empty for outer fold {outer_fold_id}")

        train_perfume_df = perfume_df.loc[perfume_df["perfume_id"].isin(train_perfume_ids)].sort_values("perfume_id")
        val_perfume_df = perfume_df.loc[perfume_df["perfume_id"].isin(val_perfume_ids)].sort_values("perfume_id")

        train_chunk_mask = chunk_df["perfume_id"].isin(train_perfume_ids).to_numpy()
        val_chunk_mask = chunk_df["perfume_id"].isin(val_perfume_ids).to_numpy()
        test_chunk_mask = (chunk_df["fold_id"] == outer_fold_id).to_numpy()

        test_chunk_probs, fold_summary = run_finetuned_bert_outer_fold(
            tokenized_inputs=tokenized_inputs,
            chunk_df=chunk_df,
            train_chunk_mask=train_chunk_mask,
            val_chunk_mask=val_chunk_mask,
            test_chunk_mask=test_chunk_mask,
            train_perfume_df=train_perfume_df,
            val_perfume_df=val_perfume_df,
            test_perfume_df=test_perfume_df,
            target_cols=target_cols,
            settings=settings,
            threshold=threshold,
            seed=seed,
            device=device,
            outer_fold_id=outer_fold_id,
        )

        test_chunk_df = chunk_df.loc[test_chunk_mask].copy().reset_index(drop=True)
        aggregated_probs = aggregate_chunk_probabilities(
            test_chunk_df,
            test_chunk_probs,
            target_cols=target_cols,
        )
        merged_test = merge_perfume_probabilities(
            test_perfume_df,
            aggregated_probs,
            target_cols=target_cols,
        )
        prob_cols = [f"prob_{label}" for label in target_cols]
        test_indices = perfume_df.index[perfume_df["fold_id"] == outer_fold_id].to_numpy()
        oof_prob[test_indices, :] = merged_test[prob_cols].to_numpy(dtype=np.float32)
        fold_summaries.append(fold_summary)

    if not np.isfinite(oof_prob).all():
        raise ValueError("OOF probability matrix contains missing values after BERT benchmark run.")

    per_label_rows, global_metrics = compute_probability_metrics(
        y_true=y_true,
        y_prob=oof_prob,
        target_cols=target_cols,
        threshold=threshold,
    )
    oof_pred = (oof_prob >= threshold).astype(int)

    for row in per_label_rows:
        label = row["label"]
        row["outer_fold_status"] = ";".join(
            [
                f"outer{fold_summary['outer_fold_id']}:"
                f"train={fold_summary['train_positive_perfumes'][label]},"
                f"val={fold_summary['val_positive_perfumes'][label]},"
                f"test={fold_summary['test_positive_perfumes'][label]}"
                for fold_summary in fold_summaries
            ]
        )

    per_label_df = pd.DataFrame(per_label_rows)
    previous_run_comparison = build_previous_run_comparison(previous_run, global_metrics)

    oof_df = perfume_df[["perfume_id", "fold_id", "Perfume", "chunk_count"]].copy()
    oof_df.rename(columns={"chunk_count": "review_chunk_count"}, inplace=True)
    for label_idx, label in enumerate(target_cols):
        oof_df[f"true_{label}"] = y_true[:, label_idx]
        oof_df[f"prob_{label}"] = oof_prob[:, label_idx]
        oof_df[f"pred_{label}"] = oof_pred[:, label_idx]

    oof_df.to_csv(predictions_path, index=False)
    per_label_df.to_csv(per_label_path, index=False)
    save_per_label_plot(per_label_df, model_family=MODEL_FAMILY, path=per_label_plot_path)
    save_curve_plot(
        y_true=y_true,
        y_prob=oof_prob,
        labels=target_cols,
        model_family=MODEL_FAMILY,
        path=curve_plot_path,
    )

    note_text = build_bert_interpretation_note(
        settings=settings,
        per_label_df=per_label_df,
        global_metrics=global_metrics,
        previous_run_comparison=previous_run_comparison,
    )
    note_path.write_text(note_text + "\n", encoding="utf-8")

    timestamp_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    dataset_relpath = stringify_path(input_jsonl)
    fold_manifest_relpath = stringify_path(fold_manifest_path)
    inner_manifest_relpath = stringify_path(inner_manifest_path)
    artifact_paths = {
        "metrics_json": stringify_path(metrics_path),
        "per_label_csv": stringify_path(per_label_path),
        "predictions_csv": stringify_path(predictions_path),
        "per_label_plot": stringify_path(per_label_plot_path),
        "roc_pr_plot": stringify_path(curve_plot_path),
        "log": stringify_path(log_path),
        "run_manifest": stringify_path(run_manifest_path),
        "note": stringify_path(note_path),
        "comparison_table": stringify_path(comparison_table_path),
        "fold_manifest": fold_manifest_relpath,
        "inner_validation_manifest": inner_manifest_relpath,
    }
    run_id = f"{MODEL_ID}_seed{seed}_kfold{outer_n_splits}"

    model_parameters = {
        "training_mode": TRAINING_MODE,
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
        "fine_tuning_scope": "full_encoder_and_classifier",
    }

    metrics_payload: Dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_family": MODEL_FAMILY,
        "run_id": run_id,
        "dataset_path": dataset_relpath,
        "n_rows": int(len(perfume_df)),
        "n_chunks": int(len(chunk_df)),
        "feature_columns": ["chunk_text_token_ids"],
        "target_columns": target_cols,
        "excluded_target_columns": excluded_target_cols,
        "split_manifest_path": fold_manifest_relpath,
        "inner_validation_manifest_path": inner_manifest_relpath,
        "data_seed": seed,
        "model_seed": seed,
        "threshold": threshold,
        "model_parameters": model_parameters,
        "macro_f1": global_metrics["macro_f1"],
        "macro_roc_auc": global_metrics["macro_roc_auc"],
        "macro_pr_auc": global_metrics["macro_pr_auc"],
        "secondary_metrics": {
            "micro_f1": global_metrics["micro_f1"],
            "subset_accuracy": global_metrics["subset_accuracy"],
            "hamming_loss": global_metrics["hamming_loss"],
            "label_cardinality": global_metrics["label_cardinality"],
        },
        "chunk_aggregation": {
            "unit": "perfume_id",
            "strategy": settings.aggregation_strategy,
            "probability_reducer": "mean",
            "threshold_after_aggregation": threshold,
        },
        "previous_run_comparison": previous_run_comparison,
        "outer_fold_runs": fold_summaries,
        "per_label": per_label_rows,
        "artifact_paths": artifact_paths,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    run_manifest = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "package_versions": collect_package_versions(("torch", "transformers", "accelerate")),
        "dataset_path": dataset_relpath,
        "split_manifest_path": fold_manifest_relpath,
        "inner_validation_manifest_path": inner_manifest_relpath,
        "data_seed": seed,
        "model_seed": seed,
        "threshold": threshold,
        "command": " ".join(sys.argv),
        "device": str(device),
        "deterministic_flags": deterministic_flags,
        "model_id": MODEL_ID,
        "model_family": MODEL_FAMILY,
        "model_parameters": model_parameters,
        "previous_run_comparison": previous_run_comparison,
        "artifact_paths": artifact_paths,
    }
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    log_lines = [
        f"timestamp_utc={timestamp_utc}",
        f"model_id={MODEL_ID}",
        f"model_family={MODEL_FAMILY}",
        f"dataset={dataset_relpath}",
        f"n_rows={len(perfume_df)}",
        f"n_chunks={len(chunk_df)}",
        f"fold_manifest={fold_manifest_relpath}",
        f"inner_validation_manifest={inner_manifest_relpath}",
        f"device={device}",
        f"macro_f1={global_metrics['macro_f1']:.6f}",
        f"macro_roc_auc={global_metrics['macro_roc_auc']:.6f}",
        f"macro_pr_auc={global_metrics['macro_pr_auc']:.6f}",
        f"micro_f1={global_metrics['micro_f1']:.6f}",
        f"subset_accuracy={global_metrics['subset_accuracy']:.6f}",
        f"hamming_loss={global_metrics['hamming_loss']:.6f}",
        f"model_parameters={json.dumps(model_parameters, sort_keys=True)}",
    ]
    if previous_run_comparison is not None:
        log_lines.append(f"previous_run_comparison={json.dumps(previous_run_comparison, sort_keys=True)}")
    for fold_summary in fold_summaries:
        log_lines.append(
            "outer_fold="
            + str(fold_summary["outer_fold_id"])
            + f" train_perfumes={fold_summary['n_train_perfumes']}"
            + f" val_perfumes={fold_summary['n_val_perfumes']}"
            + f" test_perfumes={fold_summary['n_test_perfumes']}"
            + f" train_chunks={fold_summary['n_train_chunks']}"
            + f" val_chunks={fold_summary['n_val_chunks']}"
            + f" test_chunks={fold_summary['n_test_chunks']}"
            + f" best_epoch={fold_summary['best_epoch']}"
            + f" best_val_loss={fold_summary['best_val_loss']:.6f}"
            + f" best_val_macro_f1={fold_summary['best_val_macro_f1']:.6f}"
        )
        for epoch_record in fold_summary["epoch_history"]:
            log_lines.append(
                "epoch="
                + str(epoch_record["epoch"])
                + f" outer_fold={fold_summary['outer_fold_id']}"
                + f" train_loss={epoch_record['train_loss']:.6f}"
                + f" val_loss={epoch_record['val_loss']:.6f}"
                + f" val_macro_f1={epoch_record['val_macro_f1']:.6f}"
                + f" val_macro_roc_auc={epoch_record['val_macro_roc_auc']:.6f}"
                + f" val_macro_pr_auc={epoch_record['val_macro_pr_auc']:.6f}"
            )
    for row in per_label_rows:
        log_lines.append(
            "label="
            + row["label"]
            + f" support_pos={row['support_pos']}"
            + f" f1={row['f1']:.6f}"
            + f" roc_auc={row['roc_auc']:.6f}"
            + f" pr_auc={row['pr_auc']:.6f}"
            + f" outer_fold_status={row['outer_fold_status']}"
        )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    update_comparison_table(
        output_path=comparison_table_path,
        template_path=comparison_template_path,
        model_id=MODEL_ID,
        model_family=MODEL_FAMILY,
        dataset_relpath=dataset_relpath,
        target_columns=target_cols,
        split_design=f"{outer_n_splits}-fold KFold + deterministic inner holdout",
        seed=seed,
        threshold=threshold,
        metrics=global_metrics,
        note_relpath=stringify_path(note_path),
    )

    return {
        "metrics_path": metrics_path,
        "per_label_path": per_label_path,
        "predictions_path": predictions_path,
        "fold_manifest_path": fold_manifest_path,
        "inner_validation_manifest_path": inner_manifest_path,
        "comparison_table_path": comparison_table_path,
        "note_path": note_path,
        "per_label_plot_path": per_label_plot_path,
        "curve_plot_path": curve_plot_path,
        "run_manifest_path": run_manifest_path,
        "log_path": log_path,
        "macro_metrics": global_metrics,
        "fold_summaries": fold_summaries,
        "previous_run_comparison": previous_run_comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "evaluation_config.yaml",
        help="Frozen benchmark config path.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "model_benchmark" / "datasets" / "perfume_text_dataset_v1.jsonl",
        help="Canonical deep-text JSONL path.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "model_benchmark",
        help="Benchmark output root.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device for fine-tuned BERT training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_bert_benchmark(
        input_jsonl=args.input_jsonl,
        out_root=args.out_root,
        config_path=args.config,
        device_preference=args.device,
    )
    metrics = result["macro_metrics"]
    print("BERT benchmark complete")
    print(f"- Fold manifest: {result['fold_manifest_path']}")
    print(f"- Inner validation manifest: {result['inner_validation_manifest_path']}")
    print(f"- OOF predictions: {result['predictions_path']}")
    print(f"- Metrics JSON: {result['metrics_path']}")
    print(f"- Per-label CSV: {result['per_label_path']}")
    print(f"- Comparison table: {result['comparison_table_path']}")
    print(
        f"- Main metrics: macro_f1={metrics['macro_f1']:.4f}, "
        f"macro_roc_auc={metrics['macro_roc_auc']:.4f}, macro_pr_auc={metrics['macro_pr_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
