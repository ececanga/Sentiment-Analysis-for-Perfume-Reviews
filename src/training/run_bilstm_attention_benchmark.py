"""Run the official BiLSTM + Attention benchmark on the canonical deep-text dataset."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

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
from src.training.threshold_tuning import (
    add_threshold_tuning_columns,
    build_threshold_tuning_report,
    load_threshold_tuning_settings,
    select_fold_thresholds,
    threshold_tuning_log_lines,
)
from src.training.run_bert_benchmark import (
    aggregate_chunk_probabilities,
    build_expected_inner_validation_manifest,
    choose_best_epoch,
    compute_pos_weight,
    compute_probability_metrics,
    detect_device,
    load_deep_text_dataset,
    load_or_write_inner_validation_manifest,
    merge_perfume_probabilities,
    set_global_seed,
    weighted_bce_loss,
)
from src.utils.paths import PROJECT_ROOT

MODEL_ID = "bilstm_attention"
MODEL_FAMILY = "BiLSTM + Attention"
TRAINING_MODE = "from_scratch_sequence_encoder"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


@dataclass(frozen=True)
class BiLSTMAttentionRunSettings:
    aggregation_strategy: str
    lowercase_tokens: bool
    min_token_freq: int
    max_vocab_size: int
    max_tokens: int
    embedding_dim: int
    hidden_size: int
    attention_dim: int
    lstm_layers: int
    dropout: float
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    min_delta: float
    max_grad_norm: float
    selection_metric: str


def load_bilstm_settings(config: Mapping[str, Any]) -> BiLSTMAttentionRunSettings:
    models_cfg = config.get("models", {})
    bilstm_cfg = models_cfg.get(MODEL_ID, {}) if isinstance(models_cfg, Mapping) else {}
    return BiLSTMAttentionRunSettings(
        aggregation_strategy=str(bilstm_cfg.get("aggregation_strategy", "mean_probability")),
        lowercase_tokens=bool(bilstm_cfg.get("lowercase_tokens", True)),
        min_token_freq=int(bilstm_cfg.get("min_token_freq", 2)),
        max_vocab_size=int(bilstm_cfg.get("max_vocab_size", 20000)),
        max_tokens=int(bilstm_cfg.get("max_tokens", 208)),
        embedding_dim=int(bilstm_cfg.get("embedding_dim", 96)),
        hidden_size=int(bilstm_cfg.get("hidden_size", 64)),
        attention_dim=int(bilstm_cfg.get("attention_dim", 64)),
        lstm_layers=int(bilstm_cfg.get("lstm_layers", 1)),
        dropout=float(bilstm_cfg.get("dropout", 0.3)),
        train_batch_size=int(bilstm_cfg.get("train_batch_size", 64)),
        eval_batch_size=int(bilstm_cfg.get("eval_batch_size", 128)),
        learning_rate=float(bilstm_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(bilstm_cfg.get("weight_decay", 1e-4)),
        max_epochs=int(bilstm_cfg.get("max_epochs", 12)),
        patience=int(bilstm_cfg.get("patience", 2)),
        min_delta=float(bilstm_cfg.get("min_delta", 1e-4)),
        max_grad_norm=float(bilstm_cfg.get("max_grad_norm", 1.0)),
        selection_metric=str(bilstm_cfg.get("selection_metric", "macro_f1")),
    )


def tokenize_chunk_text(text: str, *, lowercase: bool) -> List[str]:
    source = text.lower() if lowercase else text
    tokens = TOKEN_PATTERN.findall(source)
    return tokens if tokens else [UNK_TOKEN]


def pretokenize_chunk_texts(chunk_df: pd.DataFrame, *, lowercase: bool) -> List[List[str]]:
    return [tokenize_chunk_text(text, lowercase=lowercase) for text in chunk_df["chunk_text"].tolist()]


def build_fold_vocabulary(
    token_sequences: Sequence[Sequence[str]],
    *,
    train_mask: np.ndarray,
    min_token_freq: int,
    max_vocab_size: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    if max_vocab_size < 2:
        raise ValueError("max_vocab_size must be at least 2.")
    if min_token_freq < 1:
        raise ValueError("min_token_freq must be at least 1.")

    counter: Counter[str] = Counter()
    for idx in np.where(train_mask)[0]:
        counter.update(token_sequences[idx])

    vocab: Dict[str, int] = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }
    for token, count in counter.most_common():
        if token in vocab:
            continue
        if count < min_token_freq:
            break
        if len(vocab) >= max_vocab_size:
            break
        vocab[token] = len(vocab)

    summary = {
        "vocab_size": len(vocab),
        "train_unique_tokens": len(counter),
        "retained_non_special_tokens": max(0, len(vocab) - 2),
        "min_token_freq": min_token_freq,
        "max_vocab_size": max_vocab_size,
    }
    return vocab, summary


def encode_token_sequences(
    token_sequences: Sequence[Sequence[str]],
    *,
    vocab: Mapping[str, int],
    max_tokens: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1.")

    unk_id = int(vocab[UNK_TOKEN])
    n_rows = len(token_sequences)
    input_ids = np.zeros((n_rows, max_tokens), dtype=np.int64)
    lengths = np.zeros(n_rows, dtype=np.int64)
    raw_lengths = np.zeros(n_rows, dtype=np.int64)
    truncated_rows = 0
    truncated_tokens = 0
    unknown_tokens = 0

    for row_idx, tokens in enumerate(token_sequences):
        raw_lengths[row_idx] = len(tokens)
        if len(tokens) > max_tokens:
            truncated_rows += 1
            truncated_tokens += len(tokens) - max_tokens
        effective_tokens = list(tokens[:max_tokens])
        if not effective_tokens:
            effective_tokens = [UNK_TOKEN]

        ids: List[int] = []
        for token in effective_tokens:
            token_id = int(vocab.get(token, unk_id))
            ids.append(token_id)
            if token_id == unk_id and token != UNK_TOKEN:
                unknown_tokens += 1

        seq_len = len(ids)
        input_ids[row_idx, :seq_len] = ids
        lengths[row_idx] = seq_len

    summary = {
        "n_rows": int(n_rows),
        "max_tokens": int(max_tokens),
        "mean_raw_token_length": float(raw_lengths.mean()) if len(raw_lengths) else 0.0,
        "max_raw_token_length": int(raw_lengths.max()) if len(raw_lengths) else 0,
        "truncated_rows": int(truncated_rows),
        "truncated_tokens": int(truncated_tokens),
        "unknown_tokens": int(unknown_tokens),
    }
    return {"input_ids": input_ids, "lengths": lengths}, summary


def build_sequence_loader(
    *,
    encoded_inputs: Mapping[str, np.ndarray],
    mask: np.ndarray,
    labels: np.ndarray | None,
    sample_weights: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    mask_idx = np.where(mask)[0]
    tensors: List[torch.Tensor] = [
        torch.from_numpy(encoded_inputs["input_ids"][mask_idx]).long(),
        torch.from_numpy(encoded_inputs["lengths"][mask_idx]).long(),
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


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        attention_dim: int,
        lstm_layers: int,
        dropout: float,
        n_labels: int,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.attention_proj = nn.Linear(hidden_size * 2, attention_dim)
        self.attention_vector = nn.Linear(attention_dim, 1, bias=False)
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, n_labels)
        self.padding_idx = padding_idx
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight[self.padding_idx].fill_(0.0)
        nn.init.xavier_uniform_(self.attention_proj.weight)
        nn.init.zeros_(self.attention_proj.bias)
        nn.init.xavier_uniform_(self.attention_vector.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.clamp(min=1)
        embedded = self.embedding_dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=input_ids.size(1),
        )

        attn_hidden = torch.tanh(self.attention_proj(outputs))
        attn_scores = self.attention_vector(attn_hidden).squeeze(-1)
        steps = torch.arange(input_ids.size(1), device=lengths.device).unsqueeze(0)
        mask = steps < lengths.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(~mask, torch.finfo(attn_scores.dtype).min)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_weights = attn_weights.masked_fill(~mask, 0.0)
        context = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)
        context = self.output_dropout(context)
        return self.classifier(context)


def evaluate_sequence_model(
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
            input_ids, lengths = [item.to(device) for item in batch[:2]]
            logits = model(input_ids, lengths)
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            prob_batches.append(probs)

            if len(batch) >= 4 and pos_weight_tensor is not None:
                labels = batch[2].to(device)
                sample_weights = batch[3].to(device)
                loss = weighted_bce_loss(logits, labels, sample_weights, pos_weight=pos_weight_tensor)
                losses.append(float(loss.detach().cpu()))

    out_probs = np.vstack(prob_batches) if prob_batches else np.zeros((0, 0), dtype=np.float32)
    out_loss = float(np.mean(losses)) if losses else float("nan")
    return out_probs, out_loss


def build_bilstm_interpretation_note(
    *,
    per_label_df: pd.DataFrame,
    global_metrics: Mapping[str, float],
) -> str:
    strongest = per_label_df.sort_values("f1", ascending=False)["label"].head(2).tolist()
    weakest = per_label_df.sort_values("support_pos", ascending=True)["label"].head(4).tolist()
    strongest_text = ", ".join(strongest) if strongest else "the supported labels"
    weakest_text = ", ".join(weakest) if weakest else "the sparse labels"
    return (
        f"BiLSTM + Attention trains a from-scratch chunk-level sequence encoder with fold-local vocabulary building "
        f"and deterministic mean chunk-to-perfume aggregation, reaching macro-F1 {global_metrics['macro_f1']:.4f}, "
        f"macro ROC-AUC {global_metrics['macro_roc_auc']:.4f}, and macro PR-AUC {global_metrics['macro_pr_auc']:.4f}. "
        f"It is strongest on {strongest_text}, while {weakest_text} remain unstable under sparse support and noisy review anchoring. "
        f"The row is benchmark-ready for the official comparison table."
    )


def run_bilstm_attention_outer_fold(
    *,
    encoded_inputs: Mapping[str, np.ndarray],
    encoding_summary: Mapping[str, float],
    chunk_df: pd.DataFrame,
    train_chunk_mask: np.ndarray,
    val_chunk_mask: np.ndarray,
    test_chunk_mask: np.ndarray,
    train_perfume_df: pd.DataFrame,
    val_perfume_df: pd.DataFrame,
    test_perfume_df: pd.DataFrame,
    target_cols: Sequence[str],
    settings: BiLSTMAttentionRunSettings,
    threshold: float,
    seed: int,
    device: torch.device,
    outer_fold_id: int,
    vocab_size: int,
    vocabulary_summary: Mapping[str, int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    set_global_seed(seed)

    label_matrix = chunk_df[target_cols].to_numpy(dtype=np.float32)
    sample_weights = (1.0 / chunk_df["chunk_count"].to_numpy(dtype=np.float32)).astype(np.float32)
    pos_weight = compute_pos_weight(train_perfume_df, target_cols)
    pos_weight_tensor = torch.from_numpy(pos_weight.astype(np.float32)).to(device)

    train_loader = build_sequence_loader(
        encoded_inputs=encoded_inputs,
        mask=train_chunk_mask,
        labels=label_matrix,
        sample_weights=sample_weights,
        batch_size=settings.train_batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = build_sequence_loader(
        encoded_inputs=encoded_inputs,
        mask=val_chunk_mask,
        labels=label_matrix,
        sample_weights=sample_weights,
        batch_size=settings.eval_batch_size,
        shuffle=False,
        seed=seed,
    )
    test_loader = build_sequence_loader(
        encoded_inputs=encoded_inputs,
        mask=test_chunk_mask,
        labels=None,
        sample_weights=None,
        batch_size=settings.eval_batch_size,
        shuffle=False,
        seed=seed,
    )

    model = BiLSTMAttentionClassifier(
        vocab_size=vocab_size,
        embedding_dim=settings.embedding_dim,
        hidden_size=settings.hidden_size,
        attention_dim=settings.attention_dim,
        lstm_layers=settings.lstm_layers,
        dropout=settings.dropout,
        n_labels=len(target_cols),
    )
    model.to(device)
    if device.type == "cpu":
        model.float()

    optimizer = AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_metrics: Dict[str, float] | None = None
    best_val_loss: float | None = None
    stale_epochs = 0
    epoch_history: List[Dict[str, Any]] = []

    val_chunk_df = chunk_df.loc[val_chunk_mask].copy().reset_index(drop=True)
    n_trainable_params = int(sum(param.numel() for param in model.parameters() if param.requires_grad))

    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        batch_losses: List[float] = []

        for batch in train_loader:
            input_ids, lengths, labels, batch_weights = batch
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            batch_weights = batch_weights.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, lengths)
            batch_loss = weighted_bce_loss(logits, labels, batch_weights, pos_weight=pos_weight_tensor)
            batch_losses.append(float(batch_loss.detach().cpu()))
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
            optimizer.step()

        val_chunk_probs, val_loss = evaluate_sequence_model(
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
            f"[{MODEL_ID}][fold {outer_fold_id}] epoch {epoch}/{settings.max_epochs} "
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

    val_chunk_probs, _ = evaluate_sequence_model(
        model,
        val_loader,
        device=device,
        pos_weight_tensor=None,
    )
    test_chunk_probs, _ = evaluate_sequence_model(
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
        "vocab_size": int(vocab_size),
        "vocabulary_summary": dict(vocabulary_summary),
        "encoding_summary": dict(encoding_summary),
        "n_trainable_parameters": n_trainable_params,
        "epoch_history": epoch_history,
    }
    return test_chunk_probs, val_chunk_probs, fold_summary


def run_bilstm_attention_benchmark(
    *,
    input_jsonl: Path,
    out_root: Path,
    config_path: Path,
    device_preference: str = "auto",
) -> Dict[str, Any]:
    config = load_config(config_path)
    settings = load_bilstm_settings(config)
    target_cols = list(config["targets"]["official_emotion_columns"])
    excluded_target_cols = dict(config["targets"]["excluded_emotion_columns"])
    seed = int(config["seeds"]["data_seed"])
    threshold = float(config["evaluation"]["probability_threshold"])
    threshold_tuning_settings = load_threshold_tuning_settings(config, baseline_threshold=threshold)
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

    token_sequences = pretokenize_chunk_texts(chunk_df, lowercase=settings.lowercase_tokens)
    y_true = perfume_df[target_cols].to_numpy(dtype=int)
    oof_prob = np.full((len(perfume_df), len(target_cols)), np.nan, dtype=np.float32)
    fold_summaries: List[Dict[str, Any]] = []
    fold_threshold_selections: Dict[str, List[Dict[str, Any]]] = {
        strategy: [] for strategy in threshold_tuning_settings.strategies
    }

    fold_ids = sorted(perfume_df["fold_id"].unique().tolist())
    if fold_ids != list(range(1, outer_n_splits + 1)):
        raise ValueError(f"Expected frozen outer folds 1..{outer_n_splits}, got {fold_ids}")

    for outer_fold_id in fold_ids:
        print(f"[{MODEL_ID}] starting outer fold {outer_fold_id}/{outer_n_splits}")
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

        vocab, vocabulary_summary = build_fold_vocabulary(
            token_sequences,
            train_mask=train_chunk_mask,
            min_token_freq=settings.min_token_freq,
            max_vocab_size=settings.max_vocab_size,
        )
        encoded_inputs, encoding_summary = encode_token_sequences(
            token_sequences,
            vocab=vocab,
            max_tokens=settings.max_tokens,
        )

        test_chunk_probs, val_chunk_probs, fold_summary = run_bilstm_attention_outer_fold(
            encoded_inputs=encoded_inputs,
            encoding_summary=encoding_summary,
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
            vocab_size=len(vocab),
            vocabulary_summary=vocabulary_summary,
        )

        prob_cols = [f"prob_{label}" for label in target_cols]
        if threshold_tuning_settings.enabled:
            val_chunk_df = chunk_df.loc[val_chunk_mask].copy().reset_index(drop=True)
            val_aggregated_probs = aggregate_chunk_probabilities(
                val_chunk_df,
                val_chunk_probs,
                target_cols=target_cols,
            )
            merged_val = merge_perfume_probabilities(
                val_perfume_df,
                val_aggregated_probs,
                target_cols=target_cols,
            )
            val_prob = merged_val[prob_cols].to_numpy(dtype=np.float32)
            val_true = val_perfume_df[target_cols].to_numpy(dtype=int)
            for strategy in threshold_tuning_settings.strategies:
                fold_threshold_selections[strategy].append(
                    select_fold_thresholds(
                        y_val=val_true,
                        prob_val=val_prob,
                        target_cols=target_cols,
                        fold_id=int(outer_fold_id),
                        strategy=strategy,
                        settings=threshold_tuning_settings,
                    )
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
        test_indices = perfume_df.index[perfume_df["fold_id"] == outer_fold_id].to_numpy()
        oof_prob[test_indices, :] = merged_test[prob_cols].to_numpy(dtype=np.float32)
        fold_summaries.append(fold_summary)

    if not np.isfinite(oof_prob).all():
        raise ValueError("OOF probability matrix contains missing values after BiLSTM + Attention benchmark run.")

    per_label_rows, global_metrics = compute_probability_metrics(
        y_true=y_true,
        y_prob=oof_prob,
        target_cols=target_cols,
        threshold=threshold,
    )
    oof_pred = (oof_prob >= threshold).astype(int)
    threshold_tuning_report, tuned_predictions_by_strategy = build_threshold_tuning_report(
        settings=threshold_tuning_settings,
        y_true=y_true,
        y_prob=oof_prob,
        fold_ids=perfume_df["fold_id"].to_numpy(dtype=int),
        target_cols=target_cols,
        selections_by_strategy=fold_threshold_selections,
        baseline_per_label_rows=per_label_rows,
        baseline_metrics=global_metrics,
    )

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
    per_label_df = add_threshold_tuning_columns(
        per_label_df,
        threshold_tuning_report=threshold_tuning_report,
    )
    oof_df = perfume_df[["perfume_id", "fold_id", "Perfume", "chunk_count"]].copy()
    oof_df.rename(columns={"chunk_count": "review_chunk_count"}, inplace=True)
    for label_idx, label in enumerate(target_cols):
        oof_df[f"true_{label}"] = y_true[:, label_idx]
        oof_df[f"prob_{label}"] = oof_prob[:, label_idx]
        oof_df[f"pred_{label}"] = oof_pred[:, label_idx]
        for strategy, tuned_pred in tuned_predictions_by_strategy.items():
            oof_df[f"pred_tuned_{strategy}_{label}"] = tuned_pred[:, label_idx]

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

    note_text = build_bilstm_interpretation_note(
        per_label_df=per_label_df,
        global_metrics=global_metrics,
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
        "tokenizer": "regex_word_tokenizer",
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
        "attention_pooling": "learned_additive_attention",
        "chunk_weighting": "inverse_chunk_count_per_perfume",
        "vocabulary_scope_per_outer_fold": "inner_train_chunks_only",
    }

    metrics_payload: Dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_family": MODEL_FAMILY,
        "run_id": run_id,
        "dataset_path": dataset_relpath,
        "n_rows": int(len(perfume_df)),
        "n_chunks": int(len(chunk_df)),
        "feature_columns": ["chunk_text_token_ids", "chunk_text_token_lengths"],
        "target_columns": target_cols,
        "excluded_target_columns": excluded_target_cols,
        "split_manifest_path": fold_manifest_relpath,
        "inner_validation_manifest_path": inner_manifest_relpath,
        "data_seed": seed,
        "model_seed": seed,
        "threshold": threshold,
        "baseline_threshold": threshold,
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
        "outer_fold_runs": fold_summaries,
        "per_label": per_label_rows,
        "threshold_tuning": threshold_tuning_report,
        "artifact_paths": artifact_paths,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    run_manifest = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "package_versions": collect_package_versions(("torch",)),
        "dataset_path": dataset_relpath,
        "split_manifest_path": fold_manifest_relpath,
        "inner_validation_manifest_path": inner_manifest_relpath,
        "data_seed": seed,
        "model_seed": seed,
        "threshold": threshold,
        "baseline_threshold": threshold,
        "command": " ".join(sys.argv),
        "device": str(device),
        "deterministic_flags": deterministic_flags,
        "model_id": MODEL_ID,
        "model_family": MODEL_FAMILY,
        "model_parameters": model_parameters,
        "threshold_tuning": threshold_tuning_report,
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
    log_lines.extend(threshold_tuning_log_lines(threshold_tuning_report))
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
            + f" vocab_size={fold_summary['vocab_size']}"
            + f" best_epoch={fold_summary['best_epoch']}"
            + f" best_val_loss={fold_summary['best_val_loss']:.6f}"
            + f" best_val_macro_f1={fold_summary['best_val_macro_f1']:.6f}"
            + f" truncated_rows={fold_summary['encoding_summary']['truncated_rows']}"
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
        "threshold_tuning": threshold_tuning_report,
        "fold_summaries": fold_summaries,
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
        help="Execution device for BiLSTM + Attention training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_bilstm_attention_benchmark(
        input_jsonl=args.input_jsonl,
        out_root=args.out_root,
        config_path=args.config,
        device_preference=args.device,
    )
    metrics = result["macro_metrics"]
    print("BiLSTM + Attention benchmark complete")
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
    tuning = result.get("threshold_tuning", {})
    if tuning.get("enabled", False):
        for strategy, strategy_report in tuning["strategies"].items():
            print(
                f"- Tuned {strategy} Macro-F1={strategy_report['tuned_macro_f1']:.4f} "
                f"(baseline {tuning['baseline_threshold']:.2f}={tuning['baseline_macro_f1']:.4f})"
            )


if __name__ == "__main__":
    main()
