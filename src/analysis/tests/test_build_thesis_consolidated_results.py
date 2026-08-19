"""Regression tests for the deterministic thesis-results consolidation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.build_thesis_consolidated_results import (
    EXPECTED_BOOTSTRAP_REPS,
    EXPECTED_ROWS,
    EXPECTED_SOURCE_COMPONENTS,
    EXPECTED_THRESHOLD,
    METRICS,
    MODEL_SPECS,
    PROJECT_ROOT,
    VARIANT_LOOKUP,
    build_package,
    parse_checksum_manifest,
    sha256_file,
)


class ThesisConsolidatedResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.output_root = Path(cls._temporary.name) / "thesis_consolidated_results_v1"
        cls.result = build_package(output_root=cls.output_root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.output_root / name)

    def test_required_table_cardinalities_and_keys(self) -> None:
        expected = {
            "youtube_five_model_pooled.csv": 5,
            "youtube_five_model_component_equal.csv": 15,
            "synthetic_five_model_augmented_pooled.csv": 5,
            "synthetic_five_model_pooled.csv": 15,
            "synthetic_five_model_component_intervals.csv": 60,
            "synthetic_five_model_component_differences.csv": 30,
            "feature_combination_three_model_pooled.csv": 21,
            "feature_combination_three_model_component_intervals.csv": 126,
            "feature_combination_three_model_component_differences.csv": 108,
        }
        self.assertEqual(self.result["table_row_counts"], expected)
        self.assertEqual(len(self.csv("youtube_five_model_pooled.csv")), 5)
        self.assertFalse(
            self.csv("synthetic_five_model_pooled.csv")
            .duplicated(["model_id", "metric"])
            .any()
        )
        self.assertFalse(
            self.csv("feature_combination_three_model_pooled.csv")
            .duplicated(["model_id", "variant"])
            .any()
        )

    def test_exact_accepted_controls_and_frozen_contract(self) -> None:
        youtube = self.csv("youtube_five_model_pooled.csv").set_index("model_id")
        self.assertEqual(set(youtube.index), {spec[1] for spec in MODEL_SPECS})
        self.assertEqual(set(youtube["evaluation_rows"].astype(int)), {EXPECTED_ROWS})
        np.testing.assert_allclose(
            youtube["threshold"].to_numpy(dtype=float), EXPECTED_THRESHOLD, rtol=0, atol=1e-15
        )
        for _, model_id, _, _ in MODEL_SPECS:
            accepted_path = (
                PROJECT_ROOT
                / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/metrics"
                / f"{model_id}.json"
            )
            accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
            observed = youtube.loc[model_id]
            self.assertAlmostEqual(
                float(observed["macro_f1"]), float(accepted["macro_f1"]), places=15
            )
            self.assertAlmostEqual(
                float(observed["macro_average_precision"]),
                float(accepted["macro_pr_auc"]),
                places=15,
            )
            self.assertAlmostEqual(
                float(observed["macro_roc_auc"]),
                float(accepted["macro_roc_auc"]),
                places=15,
            )

        validation = json.loads((self.output_root / "VALIDATION.json").read_text())
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(validation["fixed_contract"]["evaluation_rows"], EXPECTED_ROWS)
        self.assertEqual(
            validation["fixed_contract"]["source_components"], EXPECTED_SOURCE_COMPONENTS
        )
        self.assertEqual(
            validation["fixed_contract"]["bootstrap_reps"], EXPECTED_BOOTSTRAP_REPS
        )
        self.assertEqual(validation["fixed_contract"]["threshold"], EXPECTED_THRESHOLD)

    def test_source_provenance_and_cross_model_feature_identity(self) -> None:
        manifest = self.csv("SOURCE_MANIFEST.csv")
        self.assertEqual(len(manifest), 59)
        self.assertTrue(manifest["verified_against_owning_manifest"].all())
        self.assertEqual(manifest["source_group"].nunique(), 9)
        self.assertTrue(manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all())
        self.assertTrue(manifest["owning_manifest_sha256"].str.fullmatch(r"[0-9a-f]{64}").all())

        ablation = self.csv("feature_combination_three_model_pooled.csv")
        self.assertEqual(set(ablation["variant"]), set(VARIANT_LOOKUP))
        self.assertTrue(
            (
                ablation.groupby("variant")["source_feature_columns_sha256"].nunique()
                == 1
            ).all()
        )
        lr = ablation.loc[ablation["model_id"].eq("logistic_regression")].set_index(
            "variant"
        )
        self.assertEqual(lr.loc["nrc_plus_vad", "source_variant"], "lexicon_only")
        self.assertEqual(lr.loc["nrc_plus_transformer", "source_variant"], "no_vad")
        self.assertEqual(lr.loc["vad_plus_transformer", "source_variant"], "no_nrc")

    def test_paired_effect_signs_and_established_direction_rules(self) -> None:
        synthetic_intervals = self.csv("synthetic_five_model_component_intervals.csv")
        synthetic_effects = self.csv("synthetic_five_model_component_differences.csv")
        points = synthetic_intervals.set_index(
            ["model_id", "label_scope", "condition", "metric"]
        )["point_estimate"]
        for row in synthetic_effects.itertuples(index=False):
            expected = points.loc[
                (
                    row.model_id,
                    row.label_scope,
                    "youtube_plus_weighted_synthetic_784x3",
                    row.metric,
                )
            ] - points.loc[(row.model_id, row.label_scope, "youtube_only", row.metric)]
            self.assertAlmostEqual(row.point_estimate_difference, expected, places=9)
            direction = (
                "augmentation_better"
                if row.ci_lower > 0
                else "augmentation_worse"
                if row.ci_upper < 0
                else "not_established"
            )
            self.assertEqual(row.established_direction, direction)

        ablation_intervals = self.csv(
            "feature_combination_three_model_component_intervals.csv"
        )
        ablation_effects = self.csv(
            "feature_combination_three_model_component_differences.csv"
        )
        ablation_points = ablation_intervals.set_index(
            ["model_id", "label_scope", "variant", "metric"]
        )["point_estimate"]
        for row in ablation_effects.itertuples(index=False):
            expected = ablation_points.loc[
                (row.model_id, row.label_scope, "full_hybrid", row.metric)
            ] - ablation_points.loc[
                (row.model_id, row.label_scope, row.comparison_variant, row.metric)
            ]
            self.assertAlmostEqual(row.point_estimate_difference, expected, places=9)
            direction = (
                "full_better"
                if row.ci_lower > 0
                else "comparison_better"
                if row.ci_upper < 0
                else "not_established"
            )
            self.assertEqual(row.established_direction, direction)
            self.assertEqual(row.interval_method, "percentile_linear")
            self.assertEqual(row.interval_scope, "conditional_on_fitted_outer_fold_models")
            self.assertEqual(row.multiplicity_adjustment, "none")

    def test_output_manifest_is_exact_and_rerun_is_deterministic(self) -> None:
        first_manifest = (self.output_root / "SHA256SUMS").read_bytes()
        first_entries = parse_checksum_manifest(self.output_root / "SHA256SUMS")
        actual = {
            path.name
            for path in self.output_root.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        }
        self.assertEqual(set(first_entries), actual)
        for name, expected_hash in first_entries.items():
            self.assertEqual(sha256_file(self.output_root / name), expected_hash)

        second = build_package(output_root=self.output_root)
        self.assertEqual(second["status"], "completed")
        self.assertEqual((self.output_root / "SHA256SUMS").read_bytes(), first_manifest)


if __name__ == "__main__":
    unittest.main()
