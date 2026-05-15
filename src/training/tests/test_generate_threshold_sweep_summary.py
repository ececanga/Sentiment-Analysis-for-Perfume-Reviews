# Date Created: 2026-05-06 15:46:46+03:00
# Date Last Modified: 2026-05-15

"""Tests for fixed-threshold sweep summary generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.training.generate_threshold_sweep_summary import generate_threshold_sweep_summary


def _seed_threshold_root(root: Path, threshold: float, rows: list[tuple[str, str, float, float, float]]) -> None:
    tables_dir = root / "tables"
    metrics_dir = root / "metrics"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    header = (
        "model_id,model_family,dataset,granularity,target_columns,split_design,data_seed,model_seed,threshold,"
        "macro_f1,macro_roc_auc,macro_pr_auc,reproducible_run,saved_metrics,comparable_plots,interpretation_note,"
        "row_complete,note_path"
    )
    lines = [header]
    for model_id, model_family, macro_f1, roc_auc, pr_auc in rows:
        lines.append(
            f"{model_id},{model_family},synthetic.csv,perfume_level,joy;sadness,5-fold KFold,42,42,{threshold},"
            f"{macro_f1},{roc_auc},{pr_auc},True,True,True,True,True,{root / 'tables' / f'{model_id}_note.md'}"
        )
        (metrics_dir / f"{model_id}.json").write_text("{}", encoding="utf-8")
    (tables_dir / "comparison_table.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


class GenerateThresholdSweepSummaryTests(unittest.TestCase):
    def test_summary_builds_tables_and_figures_from_synthetic_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary_root = tmp / "outputs" / "model_benchmark_threshold_sweep"
            t02 = tmp / "outputs" / "model_benchmark_t02"
            t03 = tmp / "outputs" / "model_benchmark_t03"

            _seed_threshold_root(
                t02,
                0.2,
                [
                    ("logistic_regression", "Logistic Regression", 0.210, 0.610, 0.250),
                    ("random_forest", "Random Forest", 0.180, 0.640, 0.240),
                    ("xgboost", "XGBoost", 0.220, 0.670, 0.260),
                ],
            )
            _seed_threshold_root(
                t03,
                0.3,
                [
                    ("logistic_regression", "Logistic Regression", 0.240, 0.620, 0.255),
                    ("random_forest", "Random Forest", 0.190, 0.650, 0.245),
                    ("xgboost", "XGBoost", 0.215, 0.675, 0.258),
                ],
            )

            outputs = generate_threshold_sweep_summary(
                summary_root=summary_root,
                threshold_roots=[t02, t03],
                models=["logistic_regression", "random_forest", "xgboost"],
            )

            for path in outputs.values():
                self.assertTrue(Path(path).exists(), path)

            wide_df = pd.read_csv(outputs["wide_table_path"])
            self.assertEqual(
                wide_df["model_id"].tolist(),
                ["logistic_regression", "random_forest", "xgboost"],
            )
            self.assertIn("macro_f1_t02", wide_df.columns)
            self.assertIn("macro_f1_t03", wide_df.columns)
            self.assertAlmostEqual(
                float(wide_df.loc[wide_df["model_id"] == "logistic_regression", "best_threshold"].iloc[0]),
                0.3,
                places=6,
            )
            self.assertAlmostEqual(
                float(wide_df.loc[wide_df["model_id"] == "xgboost", "best_threshold"].iloc[0]),
                0.2,
                places=6,
            )

            best_df = pd.read_csv(outputs["best_threshold_path"])
            self.assertEqual(best_df["model_id"].tolist(), ["logistic_regression", "random_forest", "xgboost"])

    def test_tie_break_uses_threshold_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary_root = tmp / "outputs" / "model_benchmark_threshold_sweep"
            t02 = tmp / "outputs" / "model_benchmark_t02"
            t05 = tmp / "outputs" / "model_benchmark_t05"

            _seed_threshold_root(
                t02,
                0.2,
                [("logistic_regression", "Logistic Regression", 0.220, 0.610, 0.250)],
            )
            _seed_threshold_root(
                t05,
                0.5,
                [("logistic_regression", "Logistic Regression", 0.220, 0.615, 0.251)],
            )

            outputs = generate_threshold_sweep_summary(
                summary_root=summary_root,
                threshold_roots=[t02, t05],
                models=["logistic_regression"],
            )

            wide_df = pd.read_csv(outputs["wide_table_path"])
            self.assertAlmostEqual(float(wide_df["best_threshold"].iloc[0]), 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
