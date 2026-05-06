"""Tests for the official classical benchmark runners."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.model_benchmark_runner import BenchmarkModelSpec
from src.training.run_random_forest_benchmark import run_random_forest_benchmark
from src.training.run_xgboost_benchmark import run_xgboost_benchmark

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
splits:
  outer_cv:
    n_splits: 5
seeds:
  data_seed: 42
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


def _build_dummy_xgb_model(seed: int, _y_train) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )


class RunClassicalBenchmarkTests(unittest.TestCase):
    def test_random_forest_runner_preserves_existing_lr_row_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_csv = tmp / "benchmark.csv"
            out_root = tmp / "outputs" / "model_benchmark"
            config_path = tmp / "config" / "evaluation_config.yaml"

            _write_config(config_path)
            _make_dataset().to_csv(input_csv, index=False)
            _seed_existing_comparison_table(out_root)

            result = run_random_forest_benchmark(
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
            self.assertTrue(result["run_manifest_path"].exists())
            self.assertTrue(result["log_path"].exists())
            self.assertTrue(result["note_path"].exists())
            self.assertTrue(result["comparison_table_path"].exists())
            self.assertTrue(result["classical_summary_table_path"].exists())
            self.assertTrue(result["classical_summary_figure_path"].exists())

            comparison_df = pd.read_csv(result["comparison_table_path"])
            lr_row = comparison_df.loc[comparison_df["model_id"] == "logistic_regression"].iloc[0]
            rf_row = comparison_df.loc[comparison_df["model_id"] == "random_forest"].iloc[0]
            self.assertAlmostEqual(float(lr_row["macro_f1"]), 0.2712, places=6)
            self.assertEqual(str(lr_row["row_complete"]).lower(), "true")
            self.assertEqual(str(rf_row["row_complete"]).lower(), "true")

            summary_df = pd.read_csv(result["classical_summary_table_path"])
            self.assertEqual(
                summary_df["model_id"].tolist(),
                ["logistic_regression", "xgboost", "random_forest"],
            )
            self.assertEqual(
                str(summary_df.loc[summary_df["model_id"] == "random_forest", "row_complete"].iloc[0]).lower(),
                "true",
            )

    def test_xgboost_runner_writes_artifacts_with_patched_model_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_csv = tmp / "benchmark.csv"
            out_root = tmp / "outputs" / "model_benchmark"
            config_path = tmp / "config" / "evaluation_config.yaml"

            _write_config(config_path)
            _make_dataset().to_csv(input_csv, index=False)

            patched_spec = BenchmarkModelSpec(
                model_id="xgboost",
                model_family="XGBoost",
                build_model=_build_dummy_xgb_model,
                model_params={"classifier": {"type": "PatchedLogisticRegression"}},
                package_names=(),
            )

            with mock.patch("src.training.run_xgboost_benchmark.MODEL_SPEC", patched_spec):
                result = run_xgboost_benchmark(
                    input_csv=input_csv,
                    out_root=out_root,
                    config_path=config_path,
                )

            pred_df = pd.read_csv(result["predictions_path"])
            self.assertEqual(len(pred_df), 40)
            for label in OFFICIAL_LABELS:
                self.assertIn(f"true_{label}", pred_df.columns)
                self.assertIn(f"prob_{label}", pred_df.columns)
                self.assertIn(f"pred_{label}", pred_df.columns)

            payload = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
            self.assertEqual(payload["model_id"], "xgboost")
            self.assertEqual(payload["target_columns"], OFFICIAL_LABELS)

            comparison_df = pd.read_csv(result["comparison_table_path"])
            xgb_row = comparison_df.loc[comparison_df["model_id"] == "xgboost"].iloc[0]
            self.assertEqual(str(xgb_row["row_complete"]).lower(), "true")


if __name__ == "__main__":
    unittest.main()
