# Date Created: 2026-07-13

"""Tests for deterministic dummy and train-fold prevalence diagnostics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.training.run_dummy_prevalence_baselines import (
    ARTIFACT_FILENAMES,
    BASELINE_ORDER,
    compute_dummy_prevalence_diagnostics,
    run_dummy_prevalence_baselines,
    sha256_file,
)


class DummyPrevalenceBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_csv = self.root / "feature_labels.csv"
        self.fold_csv = self.root / "fold_assignments.csv"
        self._write_inputs()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_inputs(self) -> None:
        pd.DataFrame(
            {
                "perfume_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
                "Perfume": ["One", "Two", "Three", "Four", "Five", "Six"],
                "joy": [1, 0, 1, 0, 1, 0],
                "sadness": [0, 0, 0, 0, 0, 0],
                "fear": [1, 1, 0, 0, 0, 0],
                "trust": [1, 0, 0, 1, 0, 1],
                "surprise": [0, 0, 1, 0, 0, 0],
                "anticipation": [1, 1, 1, 0, 0, 0],
                "disgust": [0, 1, 0, 0, 1, 0],
            }
        ).to_csv(self.input_csv, index=False)
        pd.DataFrame(
            {
                "perfume_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
                "source_group_id": ["g1", "g2", "g3", "g4", "g5", "g6"],
                "fold_id": [1, 1, 2, 2, 3, 3],
            }
        ).to_csv(self.fold_csv, index=False)

    def test_train_fold_prevalence_uses_only_complementary_folds(self) -> None:
        result = compute_dummy_prevalence_diagnostics(
            feature_label_csv=self.input_csv,
            fold_manifest_csv=self.fold_csv,
        )
        rows = pd.DataFrame(result["oof_predictions"])
        fear = rows.loc[
            (rows["baseline"] == "train_fold_prevalence") & (rows["label"] == "fear")
        ]
        self.assertEqual(set(fear.loc[fear["fold_id"] == 1, "y_probability"]), {0.0})
        self.assertEqual(set(fear.loc[fear["fold_id"] == 2, "y_probability"]), {0.5})
        self.assertEqual(set(fear.loc[fear["fold_id"] == 3, "y_probability"]), {0.5})
        self.assertEqual(set(fear.loc[fear["fold_id"] == 1, "y_pred"]), {0})
        self.assertEqual(set(fear.loc[fear["fold_id"] == 2, "y_pred"]), {1})

    def test_pooled_metrics_and_undefined_values_are_explicit(self) -> None:
        result = compute_dummy_prevalence_diagnostics(
            feature_label_csv=self.input_csv,
            fold_manifest_csv=self.fold_csv,
        )
        per_label = {
            (row["baseline"], row["label"]): row for row in result["per_label"]
        }
        joy_positive = per_label[("all_positive", "joy")]
        self.assertAlmostEqual(joy_positive["f1"], 2.0 / 3.0)
        self.assertAlmostEqual(joy_positive["roc_auc"], 0.5)
        self.assertAlmostEqual(joy_positive["average_precision"], 0.5)
        sadness = per_label[("train_fold_prevalence", "sadness")]
        self.assertEqual(sadness["f1"], 0.0)
        self.assertIsNone(sadness["roc_auc"])
        self.assertIsNone(sadness["average_precision"])

        summaries = {row["baseline"]: row for row in result["summary"]}
        self.assertEqual(list(summaries), list(BASELINE_ORDER))
        self.assertEqual(summaries["all_negative"]["roc_auc_defined_label_count"], 6)
        self.assertEqual(
            summaries["all_negative"]["average_precision_defined_label_count"], 6
        )

    def test_artifact_shapes_and_metric_terminology(self) -> None:
        out_dir = self.root / "out"
        result = run_dummy_prevalence_baselines(
            feature_label_csv=self.input_csv,
            fold_manifest_csv=self.fold_csv,
            out_dir=out_dir,
        )
        for path in result["paths"].values():
            self.assertTrue(path.is_file())
        self.assertEqual(len(pd.read_csv(result["paths"]["summary_csv"])), 3)
        self.assertEqual(len(pd.read_csv(result["paths"]["per_label_csv"])), 21)
        self.assertEqual(len(pd.read_csv(result["paths"]["fold_diagnostics_csv"])), 63)
        self.assertEqual(len(pd.read_csv(result["paths"]["oof_predictions_csv"])), 126)

        report_text = result["paths"]["report_json"].read_text(encoding="utf-8")
        self.assertNotIn("NaN", report_text)
        self.assertNotIn("pr_auc", report_text.lower())
        report = json.loads(report_text)
        sadness = next(
            row
            for row in report["per_label"]
            if row["baseline"] == "all_negative" and row["label"] == "sadness"
        )
        self.assertIsNone(sadness["roc_auc"])
        self.assertIsNone(sadness["average_precision"])
        summary_columns = pd.read_csv(result["paths"]["summary_csv"]).columns
        self.assertIn("macro_average_precision", summary_columns)
        self.assertNotIn("macro_pr_auc", summary_columns)

    def test_outputs_are_byte_deterministic(self) -> None:
        first = run_dummy_prevalence_baselines(
            feature_label_csv=self.input_csv,
            fold_manifest_csv=self.fold_csv,
            out_dir=self.root / "first",
        )
        second = run_dummy_prevalence_baselines(
            feature_label_csv=self.input_csv,
            fold_manifest_csv=self.fold_csv,
            out_dir=self.root / "second",
        )
        for key in ARTIFACT_FILENAMES:
            self.assertEqual(
                sha256_file(first["paths"][key]),
                sha256_file(second["paths"][key]),
                key,
            )

    def test_refuses_overwrite_without_explicit_flag(self) -> None:
        out_dir = self.root / "out"
        run_dummy_prevalence_baselines(
            feature_label_csv=self.input_csv,
            fold_manifest_csv=self.fold_csv,
            out_dir=out_dir,
        )
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            run_dummy_prevalence_baselines(
                feature_label_csv=self.input_csv,
                fold_manifest_csv=self.fold_csv,
                out_dir=out_dir,
            )
        rerun = run_dummy_prevalence_baselines(
            feature_label_csv=self.input_csv,
            fold_manifest_csv=self.fold_csv,
            out_dir=out_dir,
            overwrite=True,
        )
        self.assertTrue(rerun["paths"]["report_json"].is_file())

    def test_rejects_mismatched_universe_and_source_group_leakage(self) -> None:
        folds = pd.read_csv(self.fold_csv)
        folds.loc[0, "perfume_id"] = "unknown"
        folds.to_csv(self.fold_csv, index=False)
        with self.assertRaisesRegex(ValueError, "perfume universes differ"):
            compute_dummy_prevalence_diagnostics(
                feature_label_csv=self.input_csv,
                fold_manifest_csv=self.fold_csv,
            )

        self._write_inputs()
        folds = pd.read_csv(self.fold_csv)
        folds.loc[2, "source_group_id"] = "g1"
        folds.to_csv(self.fold_csv, index=False)
        with self.assertRaisesRegex(ValueError, "source group crosses outer folds"):
            compute_dummy_prevalence_diagnostics(
                feature_label_csv=self.input_csv,
                fold_manifest_csv=self.fold_csv,
            )

    def test_threshold_is_frozen_at_point_three(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen at threshold 0.3"):
            compute_dummy_prevalence_diagnostics(
                feature_label_csv=self.input_csv,
                fold_manifest_csv=self.fold_csv,
                threshold=0.5,
            )


if __name__ == "__main__":
    unittest.main()
