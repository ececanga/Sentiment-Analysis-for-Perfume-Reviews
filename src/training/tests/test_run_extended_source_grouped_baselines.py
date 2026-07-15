# Date Created: 2026-07-15

"""Tests for the complete deterministic Phase 3 baseline set."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.training.run_extended_source_grouped_baselines import (
    ARTIFACT_FILENAMES,
    BASELINE_ORDER,
    compute_extended_baselines,
    run_extended_baselines,
    sha256_file,
)


class ExtendedSourceGroupedBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.features = self.root / "features.csv"
        self.folds = self.root / "folds.csv"
        pd.DataFrame(
            {
                "perfume_id": [f"p{i}" for i in range(1, 9)],
                "joy": [1, 0, 1, 0, 1, 0, 1, 0],
                "sadness": [0, 1, 0, 0, 0, 0, 0, 0],
                "fear": [0, 0, 0, 1, 0, 0, 0, 0],
                "trust": [1, 1, 0, 0, 1, 1, 0, 0],
                "surprise": [0, 0, 1, 0, 0, 0, 1, 0],
                "anticipation": [1, 0, 1, 0, 1, 0, 1, 0],
                "disgust": [0, 0, 0, 0, 1, 0, 0, 0],
            }
        ).to_csv(self.features, index=False)
        pd.DataFrame(
            {
                "perfume_id": [f"p{i}" for i in range(1, 9)],
                "fold_id": [1, 1, 2, 2, 3, 3, 4, 4],
                "source_group_id": [f"g{i}" for i in range(1, 9)],
            }
        ).to_csv(self.folds, index=False)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_complete_baseline_set_and_common_label_rule(self) -> None:
        result = compute_extended_baselines(
            feature_label_csv=self.features,
            fold_manifest_csv=self.folds,
        )
        summary = pd.DataFrame(result["summary"])
        self.assertEqual(summary["baseline"].tolist(), list(BASELINE_ORDER))
        rows = pd.DataFrame(result["oof_predictions"])
        common = rows.loc[rows["baseline"] == "always_common_labels"]
        predicted_by_label = common.groupby("label")["y_pred"].unique().to_dict()
        for label in ("joy", "trust", "anticipation"):
            self.assertEqual(predicted_by_label[label].tolist(), [1])
        for label in ("sadness", "fear", "surprise", "disgust"):
            self.assertEqual(predicted_by_label[label].tolist(), [0])

    def test_random_outputs_are_seeded_and_train_prevalence_is_fold_safe(self) -> None:
        first = compute_extended_baselines(
            feature_label_csv=self.features,
            fold_manifest_csv=self.folds,
            seed=42,
        )
        second = compute_extended_baselines(
            feature_label_csv=self.features,
            fold_manifest_csv=self.folds,
            seed=42,
        )
        third = compute_extended_baselines(
            feature_label_csv=self.features,
            fold_manifest_csv=self.folds,
            seed=43,
        )
        self.assertEqual(first["oof_predictions"], second["oof_predictions"])
        self.assertNotEqual(first["oof_predictions"], third["oof_predictions"])
        rows = pd.DataFrame(first["fold_diagnostics"])
        prevalence = rows.loc[
            (rows["baseline"] == "train_fold_prevalence")
            & (rows["fold_id"] == 1)
            & (rows["label"] == "sadness")
        ].iloc[0]
        self.assertEqual(int(prevalence["train_positive_support"]), 0)
        self.assertEqual(float(prevalence["score_mean"]), 0.0)

    def test_artifacts_are_deterministic_and_self_verifying(self) -> None:
        first = run_extended_baselines(
            feature_label_csv=self.features,
            fold_manifest_csv=self.folds,
            out_dir=self.root / "first",
        )
        second = run_extended_baselines(
            feature_label_csv=self.features,
            fold_manifest_csv=self.folds,
            out_dir=self.root / "second",
        )
        for key in ARTIFACT_FILENAMES:
            self.assertEqual(
                sha256_file(first["paths"][key]), sha256_file(second["paths"][key]), key
            )
        report = json.loads(first["paths"]["report"].read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["accepted_benchmark_modified"])
        self.assertEqual(len(pd.read_csv(first["paths"]["summary"])), 6)
        self.assertEqual(len(pd.read_csv(first["paths"]["per_label"])), 42)


if __name__ == "__main__":
    unittest.main()
