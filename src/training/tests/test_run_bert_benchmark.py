"""Tests for the official BERT benchmark runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.training.run_bert_benchmark import aggregate_chunk_probabilities, run_bert_benchmark


OFFICIAL_LABELS = ["joy", "sadness", "fear", "trust", "surprise", "anticipation", "disgust"]


def _label_payload(i: int) -> dict[str, int]:
    return {
        "joy": 1 if i % 2 == 0 else 0,
        "sadness": 1 if i % 3 == 0 else 0,
        "fear": 1 if i % 4 == 0 else 0,
        "trust": 1 if i % 5 in (0, 1) else 0,
        "surprise": 1 if i % 6 == 0 else 0,
        "anticipation": 1 if i % 3 in (0, 1) else 0,
        "disgust": 1 if i % 7 == 0 else 0,
    }


def _chunk_text(i: int, chunk_index: int, labels: dict[str, int]) -> str:
    positive_tokens = [label for label, value in labels.items() if value == 1]
    filler = "warm powder iris cedar amber"
    repeated = " ".join(positive_tokens or ["neutral"])
    return f"TITLE: perfume {i} chunk {chunk_index} || REVIEW: {repeated} {filler} {repeated}"


def _write_config(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
benchmark:
  name: thesis_model_benchmark_v1
data:
  canonical_input_csv: outputs/analysis_out/gemini_unified_merged.csv
  sample_granularity: perfume_level
  unit_id_column: perfume_id
  deep_text_dataset_jsonl: outputs/model_benchmark/datasets/perfume_text_dataset_v1.jsonl
targets:
  official_emotion_columns:
    - joy
    - sadness
    - fear
    - trust
    - surprise
    - anticipation
    - disgust
  excluded_emotion_columns:
    anger: zero_positive_support_in_frozen_784_row_dataset
evaluation:
  probability_threshold: 0.3
  threshold_tuning:
    enabled: true
    selection_metric: macro_f1
    strategies:
      - global
      - per_label
    threshold_grid:
      values:
        - 0.3
        - 0.5
        - 0.7
splits:
  outer_cv:
    n_splits: 5
  inner_validation:
    applies_to:
      - bert
    n_splits: 10
seeds:
  data_seed: 42
models:
  bert:
    checkpoint: mock-bert
    local_files_only: true
    max_length: 64
    aggregation_strategy: mean_probability
    train_batch_size: 8
    eval_batch_size: 16
    learning_rate: 2.0e-5
    weight_decay: 0.01
    max_epochs: 3
    patience: 1
    min_delta: 0.0001
    warmup_ratio: 0.1
    gradient_accumulation_steps: 1
    max_grad_norm: 1.0
    selection_metric: macro_f1
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _seed_existing_comparison_table(out_root: Path) -> Path:
    comparison_table_path = out_root / "tables" / "comparison_table.csv"
    comparison_table_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_table_path.write_text(
        "\n".join(
            [
                "model_id,model_family,dataset,granularity,target_columns,split_design,data_seed,model_seed,threshold,macro_f1,macro_roc_auc,macro_pr_auc,reproducible_run,saved_metrics,comparable_plots,interpretation_note,row_complete,note_path",
                "logistic_regression,Logistic Regression,outputs/analysis_out/gemini_unified_merged.csv,perfume_level,joy;sadness;fear;trust;surprise;anticipation;disgust,5-fold KFold,42,42,0.3,0.2712,0.627774,0.301525,True,True,True,True,True,outputs/model_benchmark/tables/logistic_regression_note.md",
                "xgboost,XGBoost,outputs/analysis_out/gemini_unified_merged.csv,perfume_level,joy;sadness;fear;trust;surprise;anticipation;disgust,5-fold KFold,42,42,0.3,,,,False,False,False,False,False,outputs/model_benchmark/tables/xgboost_note.md",
                "random_forest,Random Forest,outputs/analysis_out/gemini_unified_merged.csv,perfume_level,joy;sadness;fear;trust;surprise;anticipation;disgust,5-fold KFold,42,42,0.3,,,,False,False,False,False,False,outputs/model_benchmark/tables/random_forest_note.md",
                "bert,BERT,outputs/analysis_out/gemini_unified_merged.csv,perfume_level,joy;sadness;fear;trust;surprise;anticipation;disgust,5-fold KFold,42,42,0.3,,,,False,False,False,False,False,outputs/model_benchmark/tables/bert_note.md",
                "bilstm_attention,BiLSTM + Attention,outputs/analysis_out/gemini_unified_merged.csv,perfume_level,joy;sadness;fear;trust;surprise;anticipation;disgust,5-fold KFold,42,42,0.3,,,,False,False,False,False,False,outputs/model_benchmark/tables/bilstm_attention_note.md",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return comparison_table_path


def _seed_existing_frozen_bert_metrics(out_root: Path) -> None:
    metrics_path = out_root / "metrics" / "bert.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "model_id": "bert",
                "run_id": "bert_seed42_kfold5",
                "macro_f1": 0.211723,
                "macro_roc_auc": 0.574478,
                "macro_pr_auc": 0.218328,
                "model_parameters": {
                    "freeze_encoder": True,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_synthetic_text_dataset(input_jsonl: Path, fold_manifest_path: Path, n_perfumes: int = 30) -> None:
    records = []
    perfume_ids = [f"perfume-{i:03d}" for i in range(n_perfumes)]
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_ids = np.zeros(n_perfumes, dtype=int)
    ordered_ids = np.array(sorted(perfume_ids))
    for fold_id, (_, test_idx) in enumerate(splitter.split(ordered_ids), start=1):
        fold_ids[test_idx] = fold_id
    fold_lookup = dict(zip(ordered_ids.tolist(), fold_ids.tolist()))

    fold_rows = []
    for perfume_id in ordered_ids:
        i = int(perfume_id.split("-")[-1])
        perfume_name = f"Perfume {i:03d}"
        labels = _label_payload(i)
        chunk_count = 2 + (i % 2)
        review_chunks = []
        for chunk_index in range(chunk_count):
            review_chunks.append(
                {
                    "chunk_index": chunk_index,
                    "video_id": f"video-{i:03d}-{chunk_index}",
                    "url": f"https://example.com/{perfume_id}/{chunk_index}",
                    "match_strategy": "alias_phrase_window",
                    "title_match": True,
                    "description_match": True,
                    "transcript_phrase_match": True,
                    "transcript_anchor_match": True,
                    "any_name_match": True,
                    "chunk_word_count": 20,
                    "chunk_text": _chunk_text(i, chunk_index, labels),
                }
            )
        records.append(
            {
                "perfume_id": perfume_id,
                "perfume_name": perfume_name,
                "perfume_id_hash": f"hash-{i:03d}",
                "fold_id": fold_lookup[perfume_id],
                "source_review_count": chunk_count,
                "deduplicated_review_count": chunk_count,
                "labels": labels,
                "excluded_labels": {"anger": 0},
                "vad": {"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
                "document_text": " <REVIEW_SEP> ".join(chunk["chunk_text"] for chunk in review_chunks),
                "review_chunks": review_chunks,
            }
        )
        fold_rows.append(
            {
                "perfume_id": perfume_id,
                "fold_id": fold_lookup[perfume_id],
                "Perfume": perfume_name,
            }
        )

    input_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with input_jsonl.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")

    fold_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(fold_manifest_path, index=False)


def _mock_tokenize_chunk_texts(chunk_df: pd.DataFrame, **_: object) -> dict[str, np.ndarray]:
    n = len(chunk_df)
    max_len = 6
    return {
        "input_ids": np.zeros((n, max_len), dtype=np.int64),
        "attention_mask": np.ones((n, max_len), dtype=np.int64),
        "token_type_ids": np.zeros((n, max_len), dtype=np.int64),
    }


def _mock_run_finetuned_outer_fold(
    *,
    chunk_df: pd.DataFrame,
    val_chunk_mask: np.ndarray,
    test_chunk_mask: np.ndarray,
    train_perfume_df: pd.DataFrame,
    val_perfume_df: pd.DataFrame,
    test_perfume_df: pd.DataFrame,
    target_cols: list[str],
    outer_fold_id: int,
    **_: object,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    val_chunk_df = chunk_df.loc[val_chunk_mask].copy().reset_index(drop=True)
    test_chunk_df = chunk_df.loc[test_chunk_mask].copy().reset_index(drop=True)
    val_probs = val_chunk_df[target_cols].to_numpy(dtype=np.float32) * 0.8 + 0.1
    test_probs = test_chunk_df[target_cols].to_numpy(dtype=np.float32) * 0.8 + 0.1
    fold_summary = {
        "outer_fold_id": int(outer_fold_id),
        "n_train_perfumes": int(len(train_perfume_df)),
        "n_val_perfumes": int(len(val_perfume_df)),
        "n_test_perfumes": int(len(test_perfume_df)),
        "n_train_chunks": int(chunk_df["perfume_id"].isin(train_perfume_df["perfume_id"]).sum()),
        "n_val_chunks": int(chunk_df["perfume_id"].isin(val_perfume_df["perfume_id"]).sum()),
        "n_test_chunks": int(len(test_chunk_df)),
        "best_epoch": 1,
        "best_val_loss": 0.42,
        "best_val_macro_f1": 0.61,
        "best_val_macro_roc_auc": 0.72,
        "best_val_macro_pr_auc": 0.58,
        "train_positive_perfumes": {label: int(train_perfume_df[label].sum()) for label in target_cols},
        "val_positive_perfumes": {label: int(val_perfume_df[label].sum()) for label in target_cols},
        "test_positive_perfumes": {label: int(test_perfume_df[label].sum()) for label in target_cols},
        "epoch_history": [
            {
                "epoch": 1,
                "train_loss": 0.51,
                "val_loss": 0.42,
                "val_macro_f1": 0.61,
                "val_macro_roc_auc": 0.72,
                "val_macro_pr_auc": 0.58,
            }
        ],
    }
    return test_probs, val_probs, fold_summary


class RunBertBenchmarkTests(unittest.TestCase):
    def test_aggregate_chunk_probabilities_uses_mean(self) -> None:
        chunk_df = pd.DataFrame(
            {
                "perfume_id": ["a", "a", "b"],
                "Perfume": ["A", "A", "B"],
                "fold_id": [1, 1, 2],
                "chunk_count": [2, 2, 1],
            }
        )
        chunk_probs = np.asarray(
            [
                [0.2, 0.8],
                [0.6, 0.4],
                [0.9, 0.1],
            ],
            dtype=np.float32,
        )

        aggregated = aggregate_chunk_probabilities(
            chunk_df,
            chunk_probs,
            target_cols=["joy", "sadness"],
        )

        a_row = aggregated.loc[aggregated["perfume_id"] == "a"].iloc[0]
        self.assertAlmostEqual(float(a_row["prob_joy"]), 0.4, places=6)
        self.assertAlmostEqual(float(a_row["prob_sadness"]), 0.6, places=6)

    def test_runner_writes_official_bert_artifacts_and_replaces_frozen_encoder_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_jsonl = tmp / "outputs" / "model_benchmark" / "datasets" / "perfume_text_dataset_v1.jsonl"
            out_root = tmp / "outputs" / "model_benchmark"
            config_path = tmp / "config" / "evaluation_config.yaml"
            fold_manifest_path = out_root / "tables" / "fold_assignments.csv"

            _write_config(config_path)
            _write_synthetic_text_dataset(input_jsonl, fold_manifest_path, n_perfumes=30)
            _seed_existing_comparison_table(out_root)
            _seed_existing_frozen_bert_metrics(out_root)

            with mock.patch(
                "src.training.run_bert_benchmark.tokenize_chunk_texts",
                side_effect=_mock_tokenize_chunk_texts,
            ), mock.patch(
                "src.training.run_bert_benchmark.run_finetuned_bert_outer_fold",
                side_effect=_mock_run_finetuned_outer_fold,
            ):
                result = run_bert_benchmark(
                    input_jsonl=input_jsonl,
                    out_root=out_root,
                    config_path=config_path,
                    device_preference="cpu",
                )

            self.assertTrue(result["fold_manifest_path"].exists())
            self.assertTrue(result["inner_validation_manifest_path"].exists())
            self.assertTrue(result["predictions_path"].exists())
            self.assertTrue(result["metrics_path"].exists())
            self.assertTrue(result["per_label_path"].exists())
            self.assertTrue(result["per_label_plot_path"].exists())
            self.assertTrue(result["curve_plot_path"].exists())
            self.assertTrue(result["run_manifest_path"].exists())
            self.assertTrue(result["log_path"].exists())
            self.assertTrue(result["note_path"].exists())
            self.assertTrue(result["comparison_table_path"].exists())

            inner_df = pd.read_csv(result["inner_validation_manifest_path"])
            self.assertEqual(sorted(inner_df["outer_fold_id"].unique().tolist()), [1, 2, 3, 4, 5])
            self.assertEqual(set(inner_df["inner_role"].unique().tolist()), {"train", "val"})

            pred_df = pd.read_csv(result["predictions_path"])
            self.assertEqual(len(pred_df), 30)
            self.assertIn("review_chunk_count", pred_df.columns)
            for label in OFFICIAL_LABELS:
                self.assertIn(f"true_{label}", pred_df.columns)
                self.assertIn(f"prob_{label}", pred_df.columns)
                self.assertIn(f"pred_{label}", pred_df.columns)
                self.assertIn(f"pred_tuned_global_{label}", pred_df.columns)
                self.assertIn(f"pred_tuned_per_label_{label}", pred_df.columns)

            payload = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
            self.assertEqual(payload["model_id"], "bert")
            self.assertEqual(payload["target_columns"], OFFICIAL_LABELS)
            self.assertEqual(payload["chunk_aggregation"]["strategy"], "mean_probability")
            self.assertEqual(payload["model_parameters"]["training_mode"], "end_to_end_finetune")
            self.assertEqual(payload["previous_run_comparison"]["previous_training_mode"], "frozen_encoder_head")
            self.assertEqual(len(payload["outer_fold_runs"]), 5)
            self.assertTrue(payload["threshold_tuning"]["enabled"])
            self.assertEqual(payload["threshold_tuning"]["baseline_macro_f1"], payload["macro_f1"])
            self.assertIn("global", payload["threshold_tuning"]["strategies"])
            self.assertIn("per_label", payload["threshold_tuning"]["strategies"])

            note_text = result["note_path"].read_text(encoding="utf-8")
            self.assertIn("frozen-encoder", note_text)
            self.assertIn("fine-tunes the full", note_text)

            comparison_df = pd.read_csv(result["comparison_table_path"])
            bert_row = comparison_df.loc[comparison_df["model_id"] == "bert"].iloc[0]
            xgb_row = comparison_df.loc[comparison_df["model_id"] == "xgboost"].iloc[0]
            bilstm_row = comparison_df.loc[comparison_df["model_id"] == "bilstm_attention"].iloc[0]
            self.assertEqual(str(bert_row["row_complete"]).lower(), "true")
            self.assertEqual(str(bert_row["saved_metrics"]).lower(), "true")
            self.assertEqual(str(xgb_row["row_complete"]).lower(), "false")
            self.assertEqual(str(bilstm_row["row_complete"]).lower(), "false")


if __name__ == "__main__":
    unittest.main()
