# Date Created: 2026-04-13 23:02:44+03:00
# Date Last Modified: 2026-07-13

"""Tests for the official Logistic Regression benchmark runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.training.run_logistic_regression_benchmark import run_logistic_regression_benchmark
from src.training.source_grouped_folds import (
    build_grouped_inner_validation_manifest,
    sha256_file,
)


OFFICIAL_LABELS = ["joy", "sadness", "fear", "trust", "surprise", "anticipation", "disgust"]


def _make_dataset(n: int = 40) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "perfume_id": f"perfume-{i:03d}",
                "Perfume": f"Perfume {i:03d}",
                "NRC_joy": float((i % 5) + 1),
                "NRC_sadness": float((i * 2) % 7),
                "NRC_fear": float((i * 3) % 11),
                "VAD_valence": 0.1 + (i % 10) * 0.05,
                "Hartmann_joy": 0.2 + (i % 4) * 0.1,
                "Savani_joy": 0.9 if i % 2 == 0 else 0.2,
                "GoEmotions_joy": 0.8 if i % 3 == 0 else 0.1,
                "Cardiff_joy": 0.7 if i % 4 == 0 else 0.1,
                "joy": 1 if i % 2 == 0 else 0,
                "sadness": 1 if i % 5 in (0, 1) else 0,
                "fear": 1 if i % 7 in (0, 1) else 0,
                "trust": 1 if i % 3 == 0 else 0,
                "surprise": 1 if i % 6 in (0, 2) else 0,
                "anticipation": 1 if i % 4 in (0, 1) else 0,
                "disgust": 1 if i % 8 in (0, 3) else 0,
                "anger": 0,
            }
        )
    return pd.DataFrame(rows)


class RunLogisticRegressionBenchmarkTests(unittest.TestCase):
    def test_runner_writes_official_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_csv = tmp / "benchmark.csv"
            out_root = tmp / "outputs" / "model_benchmark"
            config_path = tmp / "config" / "evaluation_config.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)

            config_path.write_text(
                """
benchmark:
  name: thesis_model_benchmark_v1
data:
  canonical_input_csv: outputs/analysis_out/gemini_unified_merged.csv
  sample_granularity: perfume_level
  unit_id_column: perfume_id
  feature_prefixes:
    - NRC_
    - VAD_
    - Hartmann_
    - Savani_
    - GoEmotions_
    - Cardiff_
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
seeds:
  data_seed: 42
artifacts:
  root_dir: {out_root}
""".strip()
                + "\n",
                encoding="utf-8",
            )

            _make_dataset().to_csv(input_csv, index=False)

            result = run_logistic_regression_benchmark(
                input_csv=input_csv,
                out_root=out_root,
                config_path=config_path,
            )

            self.assertTrue(result["fold_manifest_path"].exists())
            self.assertTrue(result["predictions_path"].exists())
            self.assertTrue(result["metrics_path"].exists())
            self.assertTrue(result["per_label_path"].exists())
            self.assertTrue(result["inner_validation_manifest_path"].exists())
            self.assertTrue(result["per_label_plot_path"].exists())
            self.assertTrue(result["curve_plot_path"].exists())
            self.assertTrue(result["note_path"].exists())
            self.assertTrue(result["comparison_table_path"].exists())

            fold_df = pd.read_csv(result["fold_manifest_path"])
            self.assertEqual(sorted(fold_df["fold_id"].unique().tolist()), [1, 2, 3, 4, 5])

            pred_df = pd.read_csv(result["predictions_path"])
            self.assertEqual(len(pred_df), 40)
            for label in OFFICIAL_LABELS:
                self.assertIn(f"true_{label}", pred_df.columns)
                self.assertIn(f"prob_{label}", pred_df.columns)
                self.assertIn(f"pred_{label}", pred_df.columns)
                self.assertIn(f"pred_tuned_global_{label}", pred_df.columns)
                self.assertIn(f"pred_tuned_per_label_{label}", pred_df.columns)

            per_label_df = pd.read_csv(result["per_label_path"])
            self.assertEqual(per_label_df["label"].tolist(), OFFICIAL_LABELS)
            self.assertIn("f1_baseline", per_label_df.columns)
            self.assertIn("f1_tuned_global", per_label_df.columns)
            self.assertIn("f1_tuned_per_label", per_label_df.columns)

            payload = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
            self.assertEqual(payload["model_id"], "logistic_regression")
            self.assertEqual(payload["target_columns"], OFFICIAL_LABELS)
            self.assertEqual(payload["threshold"], 0.3)
            self.assertTrue(payload["threshold_tuning"]["enabled"])
            self.assertEqual(payload["threshold_tuning"]["baseline_macro_f1"], payload["macro_f1"])
            self.assertIn("global", payload["threshold_tuning"]["strategies"])
            self.assertIn("per_label", payload["threshold_tuning"]["strategies"])
            self.assertEqual(len(payload["threshold_tuning"]["strategies"]["global"]["selected_thresholds_by_fold"]), 5)

            comparison_df = pd.read_csv(result["comparison_table_path"])
            lr_row = comparison_df.loc[comparison_df["model_id"] == "logistic_regression"].iloc[0]
            self.assertEqual(str(lr_row["row_complete"]).lower(), "true")

    def test_source_grouped_runner_aligns_shuffled_manifest_by_perfume_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_csv = tmp / "benchmark.csv"
            out_root = tmp / "model_benchmark"
            fold_path = tmp / "fold_assignments.csv"
            inner_path = tmp / "inner_validation_folds.csv"
            audit_path = tmp / "source_group_audit.json"
            config_path = tmp / "evaluation_config_source_grouped.yaml"

            dataset = _make_dataset(40).sample(frac=1.0, random_state=17).reset_index(drop=True)
            dataset.to_csv(input_csv, index=False)
            ordered_ids = sorted(dataset["perfume_id"].tolist())
            fold_rows = []
            for index, perfume_id in enumerate(ordered_ids):
                group_number = index // 8
                fold_rows.append(
                    {
                        "perfume_id": perfume_id,
                        "Perfume": f"Perfume {perfume_id.rsplit('-', 1)[-1]}",
                        "source_group_id": f"group-{group_number}",
                        "source_group_size": 8,
                        "fold_id": group_number + 1,
                    }
                )
            fold_df = pd.DataFrame(fold_rows)
            fold_df.sample(frac=1.0, random_state=99).to_csv(fold_path, index=False)
            build_grouped_inner_validation_manifest(
                fold_df,
                inner_n_splits=2,
                seed=42,
            ).to_csv(inner_path, index=False)
            audit_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "validation_passed": True,
                        "cross_fold_video_id_count": 0,
                        "cross_fold_transcript_hash_count": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path.write_text(
                f"""
benchmark:
  name: source_grouped_test
provenance:
  frozen_input_sha256:
    feature_label_csv: {sha256_file(input_csv)}
data:
  unit_id_column: perfume_id
  feature_prefixes: [NRC_, VAD_, Hartmann_, Savani_, GoEmotions_, Cardiff_]
targets:
  official_emotion_columns: [joy, sadness, fear, trust, surprise, anticipation, disgust]
  excluded_emotion_columns:
    anger: excluded
evaluation:
  probability_threshold: 0.3
  threshold_tuning:
    enabled: true
    strategies: [global, per_label]
    threshold_grid:
      values: [0.3, 0.5]
splits:
  outer_cv:
    type: source_component_groupkfold
    n_splits: 5
    manifest_path: {fold_path}
    audit_report_path: {audit_path}
    group_column: source_group_id
    grouping_version: source_identity_v1
    split_design_label: 5-fold source-component grouped CV
  inner_validation:
    type: deterministic_source_group_holdout
    n_splits: 2
    manifest_path: {inner_path}
    require_prebuilt_manifest: true
seeds:
  data_seed: 42
artifacts:
  root_dir: {out_root}
""".strip()
                + "\n",
                encoding="utf-8",
            )

            result = run_logistic_regression_benchmark(
                input_csv=input_csv,
                out_root=out_root,
                config_path=config_path,
            )
            predictions = pd.read_csv(result["predictions_path"])
            expected_folds = fold_df.set_index("perfume_id")["fold_id"]
            observed_folds = predictions.set_index("perfume_id")["fold_id"]
            pd.testing.assert_series_equal(
                observed_folds.sort_index(),
                expected_folds.sort_index(),
                check_names=False,
            )
            self.assertEqual(
                int(predictions.groupby("source_group_id")["fold_id"].nunique().max()),
                1,
            )

            with self.assertRaisesRegex(ValueError, "must exactly match"):
                run_logistic_regression_benchmark(
                    input_csv=input_csv,
                    out_root=tmp / "wrong_benchmark_root",
                    config_path=config_path,
                )
            metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
            self.assertEqual(metrics["split_design"], "5-fold source-component grouped CV")
            self.assertEqual(
                metrics["split_provenance"]["outer_manifest_sha256"],
                sha256_file(fold_path),
            )
            self.assertEqual(
                metrics["split_provenance"]["inner_manifest_sha256"],
                sha256_file(inner_path),
            )
            self.assertEqual(
                metrics["split_provenance"]["source_audit_sha256"],
                sha256_file(audit_path),
            )
            self.assertEqual(metrics["dataset_sha256"], sha256_file(input_csv))
            self.assertEqual(metrics["config_sha256"], sha256_file(config_path))
            self.assertEqual(
                metrics["frozen_input_sha256"],
                {"feature_label_csv": sha256_file(input_csv)},
            )


if __name__ == "__main__":
    unittest.main()
