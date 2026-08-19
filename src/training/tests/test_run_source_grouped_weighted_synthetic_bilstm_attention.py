"""Focused tests for the resumable weighted-synthetic BiLSTM runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.training.run_bert_benchmark import compute_pos_weight
from src.training.run_source_grouped_weighted_synthetic_bilstm_attention import (
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_CONFIG,
    FoldTask,
    _atomic_torch_save,
    _load_epoch_checkpoint,
    fold_fingerprint,
    fold_membership,
    load_and_validate_inputs,
    load_fold_checkpoint,
    preflight_report,
    save_fold_checkpoint,
    weighted_pos_weight,
)
from src.training.run_source_grouped_weighted_synthetic_lr import (
    AUGMENTED,
    CONTROL,
)
from src.training.source_grouped_folds import SOURCE_GROUP_COLUMN


class WeightedSyntheticBiLSTMAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = load_and_validate_inputs(DEFAULT_CONFIG)

    def test_preflight_counts_and_split_leakage_contract(self) -> None:
        prepared = self.prepared
        self.assertEqual(len(prepared.perfume_df), 784)
        self.assertEqual(len(prepared.base_chunk_df), 4_879)
        self.assertEqual(len(prepared.synthetic), 2_352)
        self.assertEqual(prepared.synthetic["perfume_id"].nunique(), 784)
        self.assertEqual(
            set(prepared.synthetic.groupby("perfume_id").size().unique()), {3}
        )
        self.assertEqual(
            prepared.perfume_df[SOURCE_GROUP_COLUMN].nunique(), 359
        )

        report = preflight_report(prepared)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["perfumes"], 784)
        self.assertEqual(report["youtube_chunks"], 4_879)
        self.assertEqual(report["synthetic_reviews"], 2_352)
        self.assertEqual(len(report["folds"]), 10)
        self.assertTrue(all(row["pos_weight_finite"] for row in report["folds"]))

        all_test_ids: set[str] = set()
        for fold_id in range(1, 6):
            control = fold_membership(prepared, FoldTask(CONTROL, fold_id))
            augmented = fold_membership(prepared, FoldTask(AUGMENTED, fold_id))
            self.assertEqual(control["train_ids"], augmented["train_ids"])
            self.assertEqual(control["val_ids"], augmented["val_ids"])
            self.assertEqual(control["test_ids"], augmented["test_ids"])
            self.assertFalse(control["train_ids"] & control["val_ids"])
            self.assertFalse(control["train_ids"] & control["test_ids"])
            self.assertFalse(control["val_ids"] & control["test_ids"])
            self.assertFalse(all_test_ids & control["test_ids"])
            all_test_ids.update(control["test_ids"])

            master = prepared.master_chunk_df
            control_train = master.loc[control["train_mask"]]
            augmented_train = master.loc[augmented["train_mask"]]
            self.assertFalse(control_train["is_synthetic"].any())
            self.assertEqual(
                int(augmented_train["is_synthetic"].sum()),
                3 * len(augmented["train_ids"]),
            )
            self.assertFalse(master.loc[augmented["val_mask"], "is_synthetic"].any())
            self.assertFalse(master.loc[augmented["test_mask"], "is_synthetic"].any())
            self.assertEqual(
                set(augmented_train.loc[augmented_train["is_synthetic"], "perfume_id"]),
                augmented["train_ids"],
            )

            perfume = prepared.perfume_df.set_index("perfume_id")
            split_groups = [
                set(perfume.loc[list(ids), SOURCE_GROUP_COLUMN].astype(str))
                for ids in (
                    augmented["train_ids"],
                    augmented["val_ids"],
                    augmented["test_ids"],
                )
            ]
            self.assertFalse(split_groups[0] & split_groups[1])
            self.assertFalse(split_groups[0] & split_groups[2])
            self.assertFalse(split_groups[1] & split_groups[2])

        self.assertEqual(all_test_ids, set(prepared.perfume_df["perfume_id"]))

    def test_base_weights_and_weighted_pos_weight_use_logical_units(self) -> None:
        prepared = self.prepared
        master = prepared.master_chunk_df
        youtube_totals = master.loc[~master["is_synthetic"]].groupby(
            "perfume_id"
        )["base_sample_weight"].sum()
        synthetic_totals = master.loc[master["is_synthetic"]].groupby(
            "perfume_id"
        )["base_sample_weight"].sum()
        self.assertTrue(np.allclose(youtube_totals.to_numpy(float), 1.0))
        self.assertTrue(np.allclose(synthetic_totals.to_numpy(float), 1.0))

        membership = fold_membership(prepared, FoldTask(AUGMENTED, 1))
        train_ids = membership["train_ids"]
        control_weight, _ = weighted_pos_weight(
            prepared, membership=membership, condition=CONTROL
        )
        train_perfumes = prepared.perfume_df.loc[
            prepared.perfume_df["perfume_id"].isin(train_ids)
        ]
        expected_control = compute_pos_weight(train_perfumes, prepared.labels)
        np.testing.assert_allclose(control_weight, expected_control, rtol=0, atol=0)

        augmented_weight, detail = weighted_pos_weight(
            prepared, membership=membership, condition=AUGMENTED
        )
        synthetic = prepared.synthetic.loc[
            prepared.synthetic["perfume_id"].isin(train_ids)
        ]
        logical_labels = np.vstack(
            [
                train_perfumes[list(prepared.labels)].to_numpy(dtype=float),
                synthetic[list(prepared.labels)].to_numpy(dtype=float),
            ]
        )
        logical_weights = np.concatenate(
            [
                np.ones(len(train_perfumes), dtype=float),
                np.full(len(synthetic), 1.0 / 3.0, dtype=float),
            ]
        )
        positive = (logical_labels * logical_weights[:, None]).sum(axis=0)
        negative = ((1.0 - logical_labels) * logical_weights[:, None]).sum(axis=0)
        expected_augmented = negative / positive
        np.testing.assert_allclose(
            augmented_weight, expected_augmented.astype(np.float32), rtol=0, atol=0
        )
        for index, label in enumerate(prepared.labels):
            self.assertAlmostEqual(
                detail[label]["positive_effective_mass"], positive[index]
            )
            self.assertAlmostEqual(
                detail[label]["negative_effective_mass"], negative[index]
            )
            self.assertAlmostEqual(
                detail[label]["pos_weight"], expected_augmented[index]
            )

        augmented_train = master.loc[membership["train_mask"]]
        youtube_weight = augmented_train.loc[
            ~augmented_train["is_synthetic"], "base_sample_weight"
        ].sum()
        synthetic_weight = augmented_train.loc[
            augmented_train["is_synthetic"], "base_sample_weight"
        ].sum()
        self.assertAlmostEqual(youtube_weight, len(train_ids))
        self.assertAlmostEqual(synthetic_weight, len(train_ids))

    def test_fold_checkpoint_round_trip_and_validation(self) -> None:
        task = FoldTask(CONTROL, 1)
        perfume_ids = ["perfume-a", "perfume-b"]
        probabilities = np.asarray(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            ],
            dtype=np.float64,
        )
        summary = {"best_epoch": 3, "n_train_synthetic_reviews": 0}
        fingerprint = fold_fingerprint("base-fingerprint", task, perfume_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fold_1.npz"
            save_fold_checkpoint(
                path,
                fingerprint=fingerprint,
                task=task,
                test_ids=perfume_ids,
                probabilities=probabilities,
                summary=summary,
            )
            loaded_probabilities, loaded_summary = load_fold_checkpoint(
                path,
                expected_fingerprint=fingerprint,
                task=task,
                expected_test_ids=perfume_ids,
                n_labels=7,
            )
            np.testing.assert_array_equal(loaded_probabilities, probabilities)
            self.assertEqual(loaded_summary, summary)
            self.assertFalse(path.with_suffix(".tmp.npz").exists())

            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                load_fold_checkpoint(
                    path,
                    expected_fingerprint="stale-fingerprint",
                    task=task,
                    expected_test_ids=perfume_ids,
                    n_labels=7,
                )
            with self.assertRaisesRegex(ValueError, "test order differs"):
                load_fold_checkpoint(
                    path,
                    expected_fingerprint=fingerprint,
                    task=task,
                    expected_test_ids=list(reversed(perfume_ids)),
                    n_labels=7,
                )
            with self.assertRaisesRegex(ValueError, "probability shape differs"):
                load_fold_checkpoint(
                    path,
                    expected_fingerprint=fingerprint,
                    task=task,
                    expected_test_ids=perfume_ids,
                    n_labels=6,
                )

    def test_epoch_checkpoint_rejects_stale_fingerprint(self) -> None:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": "current-fold-fingerprint",
            "next_epoch": 4,
            "model_state": {"weight": torch.tensor([1.0, 2.0])},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "epochs" / "fold_1.pt"
            _atomic_torch_save(path, payload)
            loaded = _load_epoch_checkpoint(path, "current-fold-fingerprint")
            self.assertEqual(loaded["next_epoch"], 4)
            torch.testing.assert_close(
                loaded["model_state"]["weight"], payload["model_state"]["weight"]
            )
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                _load_epoch_checkpoint(path, "stale-fold-fingerprint")


if __name__ == "__main__":
    unittest.main()
