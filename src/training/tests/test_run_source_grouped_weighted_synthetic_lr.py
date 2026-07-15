"""Tests for the frozen Phase 6 weighted synthetic LR contrast."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.run_source_grouped_weighted_synthetic_lr import (
    AUGMENTED,
    CONDITIONS,
    CONTROL,
    DEFAULT_CONFIG,
    METRICS,
    build_paired_component_bootstrap,
    build_training_rows,
    fit_oof_condition,
    load_and_validate_inputs,
    run_experiment,
    validate_control_replay,
)


class WeightedSyntheticLogisticRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = load_and_validate_inputs(DEFAULT_CONFIG)
        cls.control_oof, cls.control_audit = fit_oof_condition(
            cls.prepared,
            include_synthetic=False,
            seed=42,
            threshold=0.3,
        )
        cls.augmented_oof, cls.augmented_audit = fit_oof_condition(
            cls.prepared,
            include_synthetic=True,
            seed=42,
            threshold=0.3,
        )

    def test_frozen_input_contract(self) -> None:
        self.assertEqual(len(self.prepared.youtube), 784)
        self.assertEqual(len(self.prepared.synthetic), 2352)
        self.assertEqual(self.prepared.synthetic["perfume_id"].nunique(), 784)
        self.assertEqual(len(self.prepared.feature_columns), 58)
        self.assertEqual(self.prepared.manifest["source_group_id"].nunique(), 359)
        self.assertEqual(
            set(self.prepared.synthetic.groupby("perfume_id").size().unique()), {3}
        )

    def test_training_weights_and_holdout_exclusion(self) -> None:
        training, weights, audit = build_training_rows(
            self.prepared, fold_id=1, include_synthetic=True
        )
        synthetic = training.loc[training["training_source"].eq("synthetic_review")]
        per_perfume_weight = synthetic.groupby("perfume_id")["sample_weight"].sum()
        self.assertTrue(np.allclose(per_perfume_weight.to_numpy(float), 1.0))
        self.assertAlmostEqual(
            audit["synthetic_effective_weight"], audit["train_perfumes"]
        )
        self.assertEqual(audit["test_perfume_overlap_in_training"], 0)
        self.assertEqual(len(training), audit["train_youtube_rows"] + audit["train_synthetic_rows"])
        self.assertAlmostEqual(weights.sum(), audit["total_effective_weight"])

    def test_control_replay_and_augmented_oof_are_valid(self) -> None:
        validation = validate_control_replay(
            self.control_oof,
            self.prepared.accepted_oof,
            self.prepared.labels,
        )
        self.assertEqual(validation["status"], "passed")
        self.assertLessEqual(validation["max_absolute_probability_difference"], 1e-4)
        self.assertEqual(len(self.augmented_oof), 784)
        self.assertFalse(
            self.augmented_oof[
                [f"prob_{label}" for label in self.prepared.labels]
            ].isna().any().any()
        )
        self.assertEqual(
            self.augmented_oof.groupby("source_group_id")["fold_id"].nunique().max(),
            1,
        )

    def test_paired_bootstrap_is_deterministic_and_has_both_scopes(self) -> None:
        accepted = self.prepared.accepted_oof
        arguments = {
            "oof_by_condition": {
                CONTROL: accepted,
                AUGMENTED: self.augmented_oof,
            },
            "label_scopes": {
                "seven_labels": self.prepared.labels,
                "fear_excluded_six_labels": self.prepared.sensitivity_labels,
            },
            "reps": 12,
            "seed": 19,
            "confidence_level": 0.95,
        }
        first_intervals, first_differences, first_details = (
            build_paired_component_bootstrap(**arguments)
        )
        second_intervals, second_differences, second_details = (
            build_paired_component_bootstrap(**arguments)
        )
        pd.testing.assert_frame_equal(first_intervals, second_intervals)
        pd.testing.assert_frame_equal(first_differences, second_differences)
        self.assertEqual(first_details, second_details)
        self.assertEqual(set(first_differences["metric"]), set(METRICS))
        self.assertEqual(
            set(first_differences["label_scope"]),
            {"seven_labels", "fear_excluded_six_labels"},
        )
        self.assertEqual(len(first_intervals), len(CONDITIONS) * len(METRICS) * 2)
        self.assertEqual(len(first_differences), len(METRICS) * 2)

    def test_full_runner_writes_validated_outputs_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "phase6"
            paths = run_experiment(
                config_path=DEFAULT_CONFIG,
                out_dir=out_dir,
                bootstrap_reps=6,
            )
            validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
            self.assertEqual(validation["status"], "passed")
            self.assertEqual(validation["cross_fold_source_components"], 0)
            self.assertEqual(validation["held_out_synthetic_overlap_max"], 0)
            summary = pd.read_csv(paths["condition_summary"])
            self.assertEqual(set(summary["condition"]), set(CONDITIONS))
            self.assertTrue(paths["checksums"].is_file())
            with self.assertRaises(FileExistsError):
                run_experiment(
                    config_path=DEFAULT_CONFIG,
                    out_dir=out_dir,
                    bootstrap_reps=6,
                )


if __name__ == "__main__":
    unittest.main()
