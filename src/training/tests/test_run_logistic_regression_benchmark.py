"""Tests for the official Logistic Regression benchmark runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.training.run_logistic_regression_benchmark import run_logistic_regression_benchmark


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
  probability_threshold: 0.5
splits:
  outer_cv:
    n_splits: 5
seeds:
  data_seed: 42
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

            per_label_df = pd.read_csv(result["per_label_path"])
            self.assertEqual(per_label_df["label"].tolist(), OFFICIAL_LABELS)

            payload = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
            self.assertEqual(payload["model_id"], "logistic_regression")
            self.assertEqual(payload["target_columns"], OFFICIAL_LABELS)

            comparison_df = pd.read_csv(result["comparison_table_path"])
            lr_row = comparison_df.loc[comparison_df["model_id"] == "logistic_regression"].iloc[0]
            self.assertEqual(str(lr_row["row_complete"]).lower(), "true")


if __name__ == "__main__":
    unittest.main()
