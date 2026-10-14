"""Focused tests for the resumable corrected weighted-synthetic BERT runner."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.training.run_bert_benchmark import BertRunSettings
from src.training.run_source_grouped_weighted_synthetic_bert import (
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_CONFIG,
    FoldTask,
    PreparedInputs,
    SafePause,
    _atomic_torch_save,
    _load_epoch_checkpoint,
    canonical_oof_frames,
    fold_fingerprint,
    fold_membership,
    load_and_validate_inputs,
    load_fold_checkpoint,
    main,
    preflight_report,
    run_resumable_folds,
    save_fold_checkpoint,
    train_fold,
    weighted_pos_weight,
)
from src.training.run_source_grouped_weighted_synthetic_lr import (
    AUGMENTED,
    CONTROL,
)
from src.training.run_source_grouped_weighted_synthetic_random_forest import (
    PauseController,
)
from src.training.source_grouped_folds import SOURCE_GROUP_COLUMN


LABELS = (
    "joy",
    "sadness",
    "fear",
    "trust",
    "surprise",
    "anticipation",
    "disgust",
)


class TinySequenceClassifier(nn.Module):
    """Small stochastic stand-in that exercises shuffle and RNG restoration."""

    def __init__(self, n_labels: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 6)
        self.dropout = nn.Dropout(0.35)
        self.classifier = nn.Linear(6, n_labels)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> SimpleNamespace:
        del token_type_ids
        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(embedded.dtype)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return SimpleNamespace(logits=self.classifier(self.dropout(pooled)))


def _tiny_prepared() -> tuple[PreparedInputs, dict[str, object], dict[str, np.ndarray]]:
    train_ids = {"train-a", "train-b"}
    val_ids = {"val-a", "val-b"}
    test_ids = {"test-a", "test-b"}
    ordered_ids = ["train-a", "train-b", "val-a", "val-b", "test-a", "test-b"]

    perfume_rows: list[dict[str, object]] = []
    master_rows: list[dict[str, object]] = []
    for index, perfume_id in enumerate(ordered_ids):
        values = {label: int((label_index + index) % 2 == 0) for label_index, label in enumerate(LABELS)}
        perfume_rows.append(
            {
                "perfume_id": perfume_id,
                "Perfume": perfume_id,
                "fold_id": 1,
                "chunk_count": 1,
                SOURCE_GROUP_COLUMN: f"group-{perfume_id}",
                "source_group_size": 1,
                **values,
            }
        )
        master_rows.append(
            {
                "perfume_id": perfume_id,
                "Perfume": perfume_id,
                "fold_id": 1,
                "chunk_count": 1,
                "chunk_index": 0,
                "chunk_text": f"REVIEW: {perfume_id}",
                SOURCE_GROUP_COLUMN: f"group-{perfume_id}",
                "source_group_size": 1,
                "is_synthetic": False,
                "base_sample_weight": 1.0,
                **values,
            }
        )

    synthetic_rows: list[dict[str, object]] = []
    for perfume_index, perfume_id in enumerate(sorted(train_ids)):
        for review_index in range(1, 4):
            values = {
                label: int((label_index + perfume_index + review_index) % 2 == 0)
                for label_index, label in enumerate(LABELS)
            }
            synthetic_rows.append(
                {
                    "perfume_id": perfume_id,
                    "Perfume": perfume_id,
                    "synthetic_review_id": f"synthetic-{perfume_id}-{review_index}",
                    "review_index": review_index,
                    "Final Transcript": f"synthetic review {review_index} for {perfume_id}",
                    **values,
                }
            )
            master_rows.append(
                {
                    "perfume_id": perfume_id,
                    "Perfume": perfume_id,
                    "fold_id": 1,
                    "chunk_count": 1,
                    "chunk_index": review_index,
                    "chunk_text": f"REVIEW: synthetic {perfume_id} {review_index}",
                    SOURCE_GROUP_COLUMN: f"group-{perfume_id}",
                    "source_group_size": 1,
                    "is_synthetic": True,
                    "base_sample_weight": 1.0 / 3.0,
                    **values,
                }
            )

    perfume_df = pd.DataFrame(perfume_rows)
    master = pd.DataFrame(master_rows)
    synthetic = pd.DataFrame(synthetic_rows)
    train_mask = master["perfume_id"].isin(train_ids).to_numpy()
    val_mask = (
        ~master["is_synthetic"].to_numpy(dtype=bool)
        & master["perfume_id"].isin(val_ids).to_numpy()
    )
    test_mask = (
        ~master["is_synthetic"].to_numpy(dtype=bool)
        & master["perfume_id"].isin(test_ids).to_numpy()
    )
    membership: dict[str, object] = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }
    settings = BertRunSettings(
        checkpoint="tiny-test-model",
        local_files_only=True,
        max_length=5,
        aggregation_strategy="mean_probability",
        train_batch_size=2,
        eval_batch_size=2,
        learning_rate=0.01,
        weight_decay=0.0,
        max_epochs=3,
        patience=99,
        min_delta=0.0001,
        warmup_ratio=0.1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        selection_metric="macro_f1",
    )
    prepared = PreparedInputs(
        config={"model": {"random_seed": 42}, "evaluation": {"threshold": 0.3}},
        config_path=Path("tiny-config.yaml"),
        paths={},
        perfume_df=perfume_df,
        base_chunk_df=master.loc[~master["is_synthetic"]].copy(),
        master_chunk_df=master,
        synthetic=synthetic,
        inner_manifest=pd.DataFrame(),
        accepted_oof=pd.DataFrame(),
        accepted_metrics={},
        verified_control_replay_oof=pd.DataFrame(),
        control_replay_validation={},
        labels=LABELS,
        sensitivity_labels=tuple(label for label in LABELS if label != "fear"),
        settings=settings,
        checkpoint_provenance={},
    )
    row_count = len(master)
    input_ids = (np.arange(row_count * settings.max_length).reshape(row_count, -1) % 31 + 1).astype(np.int64)
    tokenized = {
        "input_ids": input_ids,
        "attention_mask": np.ones_like(input_ids, dtype=np.int64),
        "token_type_ids": np.zeros_like(input_ids, dtype=np.int64),
    }
    return prepared, membership, tokenized


def _assert_nested_equal(test: unittest.TestCase, left: object, right: object) -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        test.assertTrue(torch.is_tensor(left) and torch.is_tensor(right))
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        test.assertIsInstance(left, np.ndarray)
        test.assertIsInstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict) or isinstance(right, dict):
        test.assertIsInstance(left, dict)
        test.assertIsInstance(right, dict)
        test.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        test.assertIsInstance(left, type(right))
        test.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(test, left_item, right_item)
    elif isinstance(left, float) and isinstance(right, float):
        if np.isnan(left) and np.isnan(right):
            return
        test.assertEqual(left, right)
    else:
        test.assertEqual(left, right)


class WeightedSyntheticBertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.training.run_thesis_three_seed import derive_protocol_config
        cls.protocol_directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.protocol_directory.cleanup)
        cls.current_config = derive_protocol_config(
            DEFAULT_CONFIG, Path(cls.protocol_directory.name) / 'current_protocol.yaml')
        cls.prepared = load_and_validate_inputs(cls.current_config)

    def test_preflight_counts_weights_and_split_leakage_contract(self) -> None:
        prepared = self.prepared
        report = preflight_report(prepared)
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["accepted_control_retrained"])
        self.assertEqual(report["perfumes"], 784)
        self.assertEqual(report["youtube_chunks"], 4_879)
        self.assertEqual(report["synthetic_reviews"], 2_352)
        self.assertEqual(report["source_components"], 359)
        self.assertEqual(len(report["folds"]), 5)
        self.assertTrue(all(row["pos_weight_finite"] for row in report["folds"]))
        self.assertEqual(
            prepared.synthetic.groupby("perfume_id").size().unique().tolist(), [3]
        )

        expected = {
            1: (405, 45, 334, 3_584, 2_369, 1_215, 269, 2_241),
            2: (633, 38, 113, 5_877, 3_978, 1_899, 255, 646),
            3: (633, 38, 113, 5_854, 3_955, 1_899, 214, 710),
            4: (634, 38, 112, 5_860, 3_958, 1_902, 251, 670),
            5: (634, 38, 112, 5_958, 4_056, 1_902, 211, 612),
        }
        observed = {
            int(row["fold_id"]): (
                int(row["train_perfumes"]),
                int(row["val_perfumes"]),
                int(row["test_perfumes"]),
                int(row["train_total_rows"]),
                int(row["train_youtube_chunks"]),
                int(row["train_synthetic_reviews"]),
                int(row["validation_youtube_chunks"]),
                int(row["test_youtube_chunks"]),
            )
            for row in report["folds"]
        }
        self.assertEqual(observed, expected)

        master = prepared.master_chunk_df
        youtube_mass = master.loc[~master["is_synthetic"]].groupby("perfume_id")[
            "base_sample_weight"
        ].sum()
        synthetic_rows = master.loc[master["is_synthetic"]]
        synthetic_mass = synthetic_rows.groupby("perfume_id")["base_sample_weight"].sum()
        np.testing.assert_allclose(youtube_mass.to_numpy(float), 1.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(synthetic_mass.to_numpy(float), 1.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            synthetic_rows["base_sample_weight"].to_numpy(float),
            1.0 / 3.0,
            rtol=0,
            atol=0,
        )

        all_test_ids: set[str] = set()
        perfume = prepared.perfume_df.set_index("perfume_id")
        for fold_id in range(1, 6):
            membership = fold_membership(prepared, FoldTask(AUGMENTED, fold_id))
            train_ids = membership["train_ids"]
            val_ids = membership["val_ids"]
            test_ids = membership["test_ids"]
            self.assertFalse(train_ids & val_ids)
            self.assertFalse(train_ids & test_ids)
            self.assertFalse(val_ids & test_ids)
            self.assertFalse(all_test_ids & test_ids)
            all_test_ids.update(test_ids)

            split_groups = [
                set(perfume.loc[list(ids), SOURCE_GROUP_COLUMN].astype(str))
                for ids in (train_ids, val_ids, test_ids)
            ]
            self.assertFalse(split_groups[0] & split_groups[1])
            self.assertFalse(split_groups[0] & split_groups[2])
            self.assertFalse(split_groups[1] & split_groups[2])

            train = master.loc[membership["train_mask"]]
            synthetic_train = train.loc[train["is_synthetic"]]
            self.assertEqual(set(synthetic_train["perfume_id"]), train_ids)
            self.assertEqual(len(synthetic_train), 3 * len(train_ids))
            self.assertFalse(master.loc[membership["val_mask"], "is_synthetic"].any())
            self.assertFalse(master.loc[membership["test_mask"], "is_synthetic"].any())
            self.assertAlmostEqual(
                train.loc[~train["is_synthetic"], "base_sample_weight"].sum(),
                len(train_ids),
            )
            self.assertAlmostEqual(
                synthetic_train["base_sample_weight"].sum(), len(train_ids)
            )

        self.assertEqual(all_test_ids, set(prepared.perfume_df["perfume_id"]))
        control = fold_membership(prepared, FoldTask(CONTROL, 1))
        augmented = fold_membership(prepared, FoldTask(AUGMENTED, 1))
        for role in ("train_ids", "val_ids", "test_ids"):
            self.assertEqual(control[role], augmented[role])
        self.assertFalse(master.loc[control["train_mask"], "is_synthetic"].any())
        with self.assertRaisesRegex(ValueError, "Unknown BERT training condition"):
            fold_membership(prepared, FoldTask("unknown", 1))

    def test_weighted_pos_weight_matches_manual_logical_mass(self) -> None:
        prepared = self.prepared
        membership = fold_membership(prepared, FoldTask(AUGMENTED, 1))
        train_ids = membership["train_ids"]
        youtube = prepared.perfume_df.loc[
            prepared.perfume_df["perfume_id"].isin(train_ids)
        ]
        synthetic = prepared.synthetic.loc[
            prepared.synthetic["perfume_id"].isin(train_ids)
        ]
        label_matrix = np.vstack(
            [
                youtube[list(prepared.labels)].to_numpy(dtype=np.float64),
                synthetic[list(prepared.labels)].to_numpy(dtype=np.float64),
            ]
        )
        logical_weights = np.concatenate(
            [
                np.ones(len(youtube), dtype=np.float64),
                np.full(len(synthetic), 1.0 / 3.0, dtype=np.float64),
            ]
        )
        positive = (label_matrix * logical_weights[:, None]).sum(axis=0)
        negative = ((1.0 - label_matrix) * logical_weights[:, None]).sum(axis=0)
        expected = negative / positive

        actual, detail = weighted_pos_weight(prepared, membership=membership)
        np.testing.assert_array_equal(actual, expected.astype(np.float32))
        for index, label in enumerate(prepared.labels):
            self.assertAlmostEqual(detail[label]["positive_effective_mass"], positive[index])
            self.assertAlmostEqual(detail[label]["negative_effective_mass"], negative[index])
            self.assertAlmostEqual(detail[label]["pos_weight"], expected[index])

    def test_accepted_control_replay_is_exact_and_control_is_reused(self) -> None:
        prepared = self.prepared
        validation = prepared.control_replay_validation
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(validation["rows"], 784)
        self.assertEqual(validation["max_absolute_probability_difference"], 0.0)
        self.assertTrue(validation["thresholded_predictions_exact"])
        pd.testing.assert_frame_equal(
            prepared.verified_control_replay_oof,
            prepared.accepted_oof,
            check_exact=True,
        )

        frames = canonical_oof_frames(prepared)
        pd.testing.assert_frame_equal(frames[CONTROL], prepared.accepted_oof, check_exact=True)
        identity_columns = [
            "perfume_id",
            "Perfume",
            SOURCE_GROUP_COLUMN,
            "source_group_size",
            "fold_id",
            *[f"true_{label}" for label in prepared.labels],
        ]
        pd.testing.assert_frame_equal(
            frames[AUGMENTED][identity_columns],
            frames[CONTROL][identity_columns],
            check_exact=True,
        )
        self.assertTrue(
            frames[AUGMENTED][[f"prob_{label}" for label in prepared.labels]]
            .isna()
            .all()
            .all()
        )

    def test_fold_checkpoint_round_trip_and_validation(self) -> None:
        task = FoldTask(AUGMENTED, 1)
        test_ids = ["perfume-a", "perfume-b"]
        probabilities = np.asarray(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            ],
            dtype=np.float64,
        )
        summary = {"best_epoch": 2, "n_train_synthetic_reviews": 1_215}
        fingerprint = fold_fingerprint("experiment-fingerprint", task, test_ids)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fold_1.npz"
            save_fold_checkpoint(
                path,
                fingerprint=fingerprint,
                task=task,
                test_ids=test_ids,
                probabilities=probabilities,
                summary=summary,
            )
            loaded_probability, loaded_summary = load_fold_checkpoint(
                path,
                expected_fingerprint=fingerprint,
                task=task,
                expected_test_ids=test_ids,
                n_labels=len(LABELS),
            )
            np.testing.assert_array_equal(loaded_probability, probabilities)
            self.assertEqual(loaded_summary, summary)
            self.assertFalse(path.with_suffix(".tmp.npz").exists())

            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                load_fold_checkpoint(
                    path,
                    expected_fingerprint="stale",
                    task=task,
                    expected_test_ids=test_ids,
                    n_labels=len(LABELS),
                )
            with self.assertRaisesRegex(ValueError, "test order differs"):
                load_fold_checkpoint(
                    path,
                    expected_fingerprint=fingerprint,
                    task=task,
                    expected_test_ids=list(reversed(test_ids)),
                    n_labels=len(LABELS),
                )
            with self.assertRaisesRegex(ValueError, "shape differs"):
                load_fold_checkpoint(
                    path,
                    expected_fingerprint=fingerprint,
                    task=task,
                    expected_test_ids=test_ids,
                    n_labels=len(LABELS) - 1,
                )

            invalid = probabilities.copy()
            invalid[0, 0] = np.nan
            save_fold_checkpoint(
                path,
                fingerprint=fingerprint,
                task=task,
                test_ids=test_ids,
                probabilities=invalid,
                summary=summary,
            )
            with self.assertRaisesRegex(ValueError, "probabilities are invalid"):
                load_fold_checkpoint(
                    path,
                    expected_fingerprint=fingerprint,
                    task=task,
                    expected_test_ids=test_ids,
                    n_labels=len(LABELS),
                )

    def test_epoch_checkpoint_round_trip_has_complete_resume_state(self) -> None:
        generator = torch.Generator().manual_seed(42)
        required_state = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": "fold-fingerprint",
            "condition": AUGMENTED,
            "fold_id": 1,
            "model_state": {"weight": torch.tensor([1.0, 2.0])},
            "optimizer_state": {"state": {}, "param_groups": [{"lr": 0.01}]},
            "scheduler_state": {"last_epoch": 1},
            "best_state": {"weight": torch.tensor([0.5, 1.5])},
            "best_epoch": 1,
            "best_val_metrics": {"macro_f1": 0.4, "macro_pr_auc": 0.3},
            "best_val_loss": 0.7,
            "stale_epochs": 0,
            "epoch_history": [{"epoch": 1, "train_loss": 0.8}],
            "next_epoch": 2,
            "training_complete": False,
            "loader_generator_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "python_random_state": __import__("random").getstate(),
            "numpy_random_state": np.random.get_state(),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "epochs" / "fold_1.pt"
            _atomic_torch_save(path, required_state)
            loaded = _load_epoch_checkpoint(path, "fold-fingerprint")
            self.assertEqual(set(loaded), set(required_state))
            _assert_nested_equal(self, loaded, required_state)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())
            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                _load_epoch_checkpoint(path, "stale-fold-fingerprint")

            wrong_schema = {**required_state, "schema_version": CHECKPOINT_SCHEMA_VERSION + 1}
            _atomic_torch_save(path, wrong_schema)
            with self.assertRaisesRegex(ValueError, "schema differs"):
                _load_epoch_checkpoint(path, "fold-fingerprint")

    def test_tiny_pause_resume_matches_uninterrupted_training_exactly(self) -> None:
        prepared, membership, tokenized = _tiny_prepared()
        task = FoldTask(AUGMENTED, 1)
        fingerprint = "tiny-fold-fingerprint"

        def load_tiny_model(
            _settings: BertRunSettings,
            labels: tuple[str, ...],
            device: torch.device,
        ) -> nn.Module:
            return TinySequenceClassifier(len(labels)).to(device)

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "src.training.run_source_grouped_weighted_synthetic_bert._load_model",
            side_effect=load_tiny_model,
        ):
            root = Path(tmpdir)
            uninterrupted_path = root / "uninterrupted.pt"
            resumed_path = root / "resumed.pt"

            uninterrupted_probability, uninterrupted_summary = train_fold(
                prepared,
                task=task,
                membership=membership,
                tokenized_inputs=tokenized,
                fold_fp=fingerprint,
                epoch_path=uninterrupted_path,
                epoch_callback=None,
                pause_requested=lambda: False,
            )

            pause = {"requested": False}

            def request_pause(record: dict[str, object]) -> None:
                if int(record["epoch"]) == 1:
                    pause["requested"] = True

            with self.assertRaisesRegex(SafePause, "saved epoch"):
                train_fold(
                    prepared,
                    task=task,
                    membership=membership,
                    tokenized_inputs=tokenized,
                    fold_fp=fingerprint,
                    epoch_path=resumed_path,
                    epoch_callback=request_pause,
                    pause_requested=lambda: pause["requested"],
                )
            paused = _load_epoch_checkpoint(resumed_path, fingerprint)
            self.assertEqual(paused["next_epoch"], 2)
            self.assertFalse(paused["training_complete"])
            self.assertEqual([row["epoch"] for row in paused["epoch_history"]], [1])
            self.assertIn("scheduler_state", paused)
            self.assertIn("loader_generator_state", paused)

            resumed_probability, resumed_summary = train_fold(
                prepared,
                task=task,
                membership=membership,
                tokenized_inputs=tokenized,
                fold_fp=fingerprint,
                epoch_path=resumed_path,
                epoch_callback=None,
                pause_requested=lambda: False,
            )
            np.testing.assert_array_equal(resumed_probability, uninterrupted_probability)
            self.assertEqual(
                [row["epoch"] for row in resumed_summary["epoch_history"]], [1, 2, 3]
            )
            _assert_nested_equal(
                self,
                resumed_summary["epoch_history"],
                uninterrupted_summary["epoch_history"],
            )
            self.assertEqual(resumed_summary["best_epoch"], uninterrupted_summary["best_epoch"])

            uninterrupted_checkpoint = _load_epoch_checkpoint(
                uninterrupted_path, fingerprint
            )
            resumed_checkpoint = _load_epoch_checkpoint(resumed_path, fingerprint)
            for key in (
                "model_state",
                "optimizer_state",
                "scheduler_state",
                "best_state",
                "loader_generator_state",
                "torch_rng_state",
                "python_random_state",
                "numpy_random_state",
                "epoch_history",
            ):
                _assert_nested_equal(
                    self, resumed_checkpoint[key], uninterrupted_checkpoint[key]
                )

    def test_status_cli_is_read_only_and_reports_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "bert-output"
            output_root.mkdir(parents=True)
            progress = {
                "schema_version": 1,
                "experiment_id": "source_grouped_weighted_synthetic_bert_v1",
                "status": "paused",
                "completed_fold_tasks": 1,
                "total_fold_tasks": 5,
                "remaining_fold_tasks": 4,
                "current_task": {"condition": AUGMENTED, "fold_id": 2},
                "current_epoch": 1,
                "last_checkpoint": str(output_root / "checkpoints/epochs/fold_2.pt"),
                "resume_command": "resume-bert-command",
            }
            progress_path = output_root / "progress.json"
            progress_path.write_text(json.dumps(progress), encoding="utf-8")
            before = progress_path.read_bytes()
            stdout = io.StringIO()
            with mock.patch(
                "src.training.run_source_grouped_weighted_synthetic_bert.run_experiment",
                side_effect=AssertionError("status must not start the experiment"),
            ), redirect_stdout(stdout):
                result = main(
                    [
                        "--config",
                        str(DEFAULT_CONFIG),
                        "--out-dir",
                        str(output_root),
                        "--status",
                    ]
                )
            self.assertEqual(result, 0)
            shown = json.loads(stdout.getvalue())
            self.assertEqual(shown["status"], "paused")
            self.assertEqual(shown["completed_fold_tasks"], 1)
            self.assertEqual(shown["total_fold_tasks"], 5)
            self.assertEqual(shown["current_task"]["fold_id"], 2)
            self.assertEqual(shown["current_epoch"], 1)
            self.assertEqual(shown["resume_command"], "resume-bert-command")
            self.assertEqual(progress_path.read_bytes(), before)

    def test_stop_after_one_epoch_preserves_paused_fold_and_epoch_in_progress(self) -> None:
        prepared = self.prepared
        controller = PauseController()

        def fake_train_fold(*_args: object, **kwargs: object) -> object:
            epoch_path = kwargs["epoch_path"]
            fold_fp = kwargs["fold_fp"]
            self.assertIsInstance(epoch_path, Path)
            self.assertIsInstance(fold_fp, str)
            _atomic_torch_save(
                epoch_path,
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "fingerprint": fold_fp,
                    "condition": AUGMENTED,
                    "fold_id": 1,
                    "next_epoch": 2,
                    "training_complete": False,
                },
            )
            callback = kwargs["epoch_callback"]
            self.assertTrue(callable(callback))
            callback({"epoch": 1})
            pause_requested = kwargs["pause_requested"]
            self.assertTrue(callable(pause_requested))
            self.assertTrue(pause_requested())
            raise SafePause("pause requested after saved epoch")

        fake_tokens = {
            "input_ids": np.zeros((1, 1), dtype=np.int64),
            "attention_mask": np.ones((1, 1), dtype=np.int64),
            "token_type_ids": np.zeros((1, 1), dtype=np.int64),
        }
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "src.training.run_source_grouped_weighted_synthetic_bert.load_or_build_token_cache",
            return_value=(fake_tokens, "tiny-token-fingerprint"),
        ) as token_cache, mock.patch(
            "src.training.run_source_grouped_weighted_synthetic_bert.train_fold",
            side_effect=fake_train_fold,
        ) as fit:
            output_root = Path(tmpdir) / "bert-output"
            result = run_resumable_folds(
                prepared,
                output_root=output_root,
                base_fingerprint="test-experiment-fingerprint",
                controller=controller,
                show_progress=False,
                stop_after_new_epochs=1,
            )
            self.assertEqual(result["status"], "paused")
            self.assertEqual(result["completed"], 0)
            self.assertEqual(result["total"], 5)
            token_cache.assert_called_once()
            fit.assert_called_once()

            progress = json.loads(
                (output_root / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["status"], "paused")
            self.assertEqual(progress["completed_fold_tasks"], 0)
            self.assertEqual(progress["total_fold_tasks"], 5)
            self.assertEqual(progress["remaining_fold_tasks"], 5)
            self.assertEqual(
                progress["current_task"],
                {"condition": AUGMENTED, "fold_id": 1},
            )
            self.assertEqual(progress["current_epoch"], 1)
            self.assertIn("stop-after-new-epochs target 1 reached", progress["note"])
            self.assertEqual(
                progress["last_checkpoint"],
                str(
                    output_root
                    / "checkpoints"
                    / "epochs"
                    / AUGMENTED
                    / "fold_1.pt"
                ),
            )
            self.assertTrue(Path(progress["last_checkpoint"]).is_file())


if __name__ == "__main__":
    unittest.main()
