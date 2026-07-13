# Date Created: 2026-07-13
# Date Last Modified: 2026-07-13

"""Focused tests for source-grouped diagnostic artifact generation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.generate_source_grouped_diagnostics import (
    MODEL_IDS,
    OFFICIAL_LABELS,
    OUTPUT_FILENAMES,
    generate_source_grouped_diagnostics,
    sha256_file,
)


class SourceGroupedDiagnosticsTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> dict[str, Path]:
        feature_path = root / "features.csv"
        outer_path = root / "fold_assignments.csv"
        inner_path = root / "inner_validation_folds.csv"
        benchmark_root = root / "model_benchmark"
        predictions_dir = benchmark_root / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        group_members = {
            "g00": ["p000", "p001", "p002"],
            "g01": ["p003", "p004"],
            "g02": ["p005"],
            "g03": ["p006", "p007"],
            "g04": ["p008"],
            "g05": ["p009", "p010"],
            "g06": ["p011"],
            "g07": ["p012"],
            "g08": ["p013"],
            "g09": ["p014"],
        }
        group_fold = {
            "g00": 1,
            "g01": 2,
            "g02": 2,
            "g03": 3,
            "g04": 3,
            "g05": 4,
            "g06": 4,
            "g07": 5,
            "g08": 5,
            "g09": 5,
        }
        group_by_id = {
            perfume_id: group_id
            for group_id, members in group_members.items()
            for perfume_id in members
        }
        perfume_ids = sorted(group_by_id)

        feature_rows = []
        for index, perfume_id in enumerate(perfume_ids):
            feature_rows.append(
                {
                    "perfume_id": perfume_id,
                    "joy": int(index % 2 == 0),
                    "sadness": int(index % 3 == 0),
                    "fear": int(index == 0),
                    "trust": int(index % 3 != 0),
                    "surprise": int(index in {2, 7}),
                    "anticipation": int(index % 2 == 0),
                    "disgust": int(index in {4, 12}),
                }
            )
        features = pd.DataFrame(feature_rows)
        features.to_csv(feature_path, index=False)

        outer_rows = []
        for perfume_id in perfume_ids:
            group_id = group_by_id[perfume_id]
            outer_rows.append(
                {
                    "perfume_id": perfume_id,
                    "Perfume": f"Perfume {perfume_id}",
                    "source_group_id": group_id,
                    "source_group_size": len(group_members[group_id]),
                    "fold_id": group_fold[group_id],
                }
            )
        outer = pd.DataFrame(outer_rows)
        outer.to_csv(outer_path, index=False)

        inner_rows = []
        for outer_fold_id in range(1, 6):
            eligible_groups = sorted(
                group_id for group_id, fold_id in group_fold.items() if fold_id != outer_fold_id
            )
            validation_group = eligible_groups[0]
            for perfume_id in perfume_ids:
                group_id = group_by_id[perfume_id]
                if group_fold[group_id] == outer_fold_id:
                    continue
                inner_rows.append(
                    {
                        "outer_fold_id": outer_fold_id,
                        "perfume_id": perfume_id,
                        "Perfume": f"Perfume {perfume_id}",
                        "source_group_id": group_id,
                        "inner_role": "val" if group_id == validation_group else "train",
                    }
                )
        pd.DataFrame(inner_rows).to_csv(inner_path, index=False)

        for model_index, model_id in enumerate(MODEL_IDS):
            oof = outer[
                ["perfume_id", "fold_id", "Perfume", "source_group_id", "source_group_size"]
            ].copy()
            for label_index, label in enumerate(OFFICIAL_LABELS):
                truth = features[label].to_numpy(dtype=int)
                flip = (
                    (np.arange(len(features)) + (2 * model_index) + label_index)
                    % (4 + model_index)
                    == 0
                )
                prediction = np.where(flip, 1 - truth, truth).astype(int)
                probability = np.where(prediction == 1, 0.8, 0.1).astype(float)
                oof[f"true_{label}"] = truth
                oof[f"prob_{label}"] = probability
                oof[f"pred_{label}"] = prediction
            oof.to_csv(predictions_dir / f"{model_id}_oof.csv", index=False)

        return {
            "feature_label_csv": feature_path,
            "outer_manifest_csv": outer_path,
            "inner_manifest_csv": inner_path,
            "benchmark_root": benchmark_root,
        }

    def _generate(self, inputs: dict[str, Path], out_dir: Path, *, seed: int = 17) -> dict[str, Path]:
        return generate_source_grouped_diagnostics(
            **inputs,
            out_dir=out_dir,
            threshold=0.3,
            seed=seed,
            bootstrap_reps=128,
            confidence_level=0.95,
        )

    def test_outputs_use_explicit_undefined_metrics_and_strict_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs = self._write_fixture(root)
            paths = self._generate(inputs, root / "diagnostics")

            inner = pd.read_csv(paths["inner_fold_balance"])
            outer_metrics = pd.read_csv(paths["outer_fold_per_label_metrics"])
            component_metrics = pd.read_csv(paths["component_weighted_model_metrics"])
            bootstrap = pd.read_csv(paths["paired_component_bootstrap"])

            self.assertEqual(len(inner), 10)
            self.assertEqual(len(outer_metrics), 5 * 5 * len(OFFICIAL_LABELS))
            self.assertEqual(len(component_metrics), len(MODEL_IDS))
            self.assertEqual(len(bootstrap), 10)
            self.assertEqual(set(bootstrap["metric"]), {"component_equal_macro_f1"})
            self.assertTrue(
                component_metrics["sample_weight_method"]
                .str.contains("each_source_group_total_weight=1", regex=False)
                .all()
            )
            self.assertTrue(
                np.allclose(
                    component_metrics["component_weight_total"].to_numpy(float),
                    component_metrics["n_source_components"].to_numpy(float),
                )
            )

            undefined = outer_metrics.loc[
                (outer_metrics["model_id"] == "logistic_regression")
                & (outer_metrics["fold_id"] == 2)
                & (outer_metrics["label"] == "fear")
            ].iloc[0]
            self.assertEqual(int(undefined["support_pos"]), 0)
            self.assertFalse(bool(undefined["roc_auc_defined"]))
            self.assertFalse(bool(undefined["average_precision_defined"]))
            self.assertTrue(pd.isna(undefined["roc_auc"]))
            self.assertTrue(pd.isna(undefined["average_precision"]))

            defined = outer_metrics.loc[
                (outer_metrics["model_id"] == "logistic_regression")
                & (outer_metrics["fold_id"] == 1)
                & (outer_metrics["label"] == "fear")
            ].iloc[0]
            self.assertTrue(bool(defined["roc_auc_defined"]))
            self.assertTrue(bool(defined["average_precision_defined"]))
            self.assertTrue(np.isfinite(float(defined["roc_auc"])))
            self.assertTrue(np.isfinite(float(defined["average_precision"])))

            report_text = paths["report"].read_text(encoding="utf-8")
            self.assertNotIn("NaN", report_text)
            self.assertNotIn("Infinity", report_text)
            report = json.loads(report_text)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["artifact_rows"]["paired_component_bootstrap"], 10)
            for key in (
                "inner_fold_balance",
                "outer_fold_per_label_metrics",
                "component_weighted_model_metrics",
                "paired_component_bootstrap",
            ):
                path = paths[key]
                self.assertEqual(
                    report["provenance"]["outputs"][path.name]["sha256"],
                    sha256_file(path),
                )

            with self.assertRaises(FileExistsError):
                self._generate(inputs, root / "diagnostics")

    def test_bootstrap_and_all_artifacts_are_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs = self._write_fixture(root)
            first = self._generate(inputs, root / "diagnostics_one", seed=99)
            second = self._generate(inputs, root / "diagnostics_two", seed=99)

            for key in OUTPUT_FILENAMES:
                self.assertEqual(first[key].read_bytes(), second[key].read_bytes(), key)

            third = self._generate(inputs, root / "diagnostics_three", seed=100)
            self.assertNotEqual(
                first["paired_component_bootstrap"].read_bytes(),
                third["paired_component_bootstrap"].read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
