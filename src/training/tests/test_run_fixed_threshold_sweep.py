"""Tests for the fixed-threshold sweep runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml

from src.training.run_fixed_threshold_sweep import (
    normalize_models,
    run_fixed_threshold_sweep,
    threshold_config_path,
    threshold_output_root,
    write_threshold_config,
)


def _write_base_config(path: Path, *, local_files_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {"name": "threshold_sweep_test"},
                "data": {"canonical_input_csv": "outputs/analysis_out/gemini_unified_merged.csv"},
                "targets": {"official_emotion_columns": ["joy"], "excluded_emotion_columns": {}},
                "evaluation": {
                    "probability_threshold": 0.3,
                    "threshold_tuning": {
                        "enabled": True,
                        "selection_metric": "macro_f1",
                        "strategies": ["global", "per_label"],
                        "threshold_grid": {"values": [0.1, 0.3, 0.5]},
                    },
                },
                "splits": {"outer_cv": {"n_splits": 5}, "inner_validation": {"n_splits": 10}},
                "seeds": {"data_seed": 42},
                "models": {"bert": {"checkpoint": "mock-bert", "local_files_only": local_files_only}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_placeholder_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"perfume_id": "p1", "joy": 1, "NRC_joy": 0.1}]).to_csv(path, index=False)


def _write_placeholder_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")


def _seed_complete_model_run(out_root: Path, model_id: str) -> None:
    (out_root / "metrics").mkdir(parents=True, exist_ok=True)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    (out_root / "tables").mkdir(parents=True, exist_ok=True)
    (out_root / "metrics" / f"{model_id}.json").write_text("{}", encoding="utf-8")
    (out_root / "metrics" / f"{model_id}_per_label.csv").write_text("label,f1\njoy,0.1\n", encoding="utf-8")
    (out_root / "predictions" / f"{model_id}_oof.csv").write_text("perfume_id\np1\n", encoding="utf-8")
    (out_root / "tables" / "comparison_table.csv").write_text(
        "\n".join(
            [
                "model_id,threshold,macro_f1,macro_roc_auc,macro_pr_auc,row_complete",
                f"{model_id},0.2,0.2,0.6,0.3,True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _stub_result(out_root: Path, model_id: str) -> dict[str, Path]:
    metrics_dir = out_root / "metrics"
    tables_dir = out_root / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{model_id}.json"
    metrics_path.write_text("{}", encoding="utf-8")
    comparison_table_path = tables_dir / "comparison_table.csv"
    comparison_table_path.write_text(
        "\n".join(
            [
                "model_id,threshold,macro_f1,macro_roc_auc,macro_pr_auc,row_complete",
                f"{model_id},0.2,0.2,0.6,0.3,True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (metrics_dir / f"{model_id}_per_label.csv").write_text("label,f1\njoy,0.1\n", encoding="utf-8")
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    (out_root / "predictions" / f"{model_id}_oof.csv").write_text("perfume_id\np1\n", encoding="utf-8")
    return {"metrics_path": metrics_path, "comparison_table_path": comparison_table_path}


class RunFixedThresholdSweepTests(unittest.TestCase):
    def test_threshold_root_and_config_paths_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_root = Path(tmpdir) / "outputs" / "model_benchmark_threshold_sweep"
            self.assertEqual(
                threshold_output_root(summary_root, 0.2),
                Path(tmpdir) / "outputs" / "model_benchmark_t02",
            )
            self.assertEqual(
                threshold_config_path(summary_root, 0.7),
                summary_root / "configs" / "evaluation_config_t07.yaml",
            )

    def test_write_threshold_config_changes_only_baseline_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_config = tmp / "config" / "evaluation_config.yaml"
            out_config = tmp / "generated" / "evaluation_config_t05.yaml"
            _write_base_config(base_config)

            write_threshold_config(base_config_path=base_config, destination_path=out_config, threshold=0.5)

            original = yaml.safe_load(base_config.read_text(encoding="utf-8"))
            generated = yaml.safe_load(out_config.read_text(encoding="utf-8"))
            original["evaluation"]["probability_threshold"] = 0.5
            self.assertEqual(generated, original)

    def test_normalize_models_preserves_canonical_order(self) -> None:
        self.assertEqual(
            normalize_models(["xgboost", "logistic_regression", "xgboost"]),
            ["logistic_regression", "xgboost"],
        )

    def test_existing_nonempty_threshold_root_fails_without_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary_root = tmp / "outputs" / "model_benchmark_threshold_sweep"
            t02 = tmp / "outputs" / "model_benchmark_t02"
            t02.mkdir(parents=True, exist_ok=True)
            (t02 / "sentinel.txt").write_text("occupied", encoding="utf-8")

            base_config = tmp / "config" / "evaluation_config.yaml"
            input_csv = tmp / "data" / "benchmark.csv"
            _write_base_config(base_config)
            _write_placeholder_csv(input_csv)

            with self.assertRaisesRegex(ValueError, "non-empty"):
                run_fixed_threshold_sweep(
                    base_config_path=base_config,
                    input_csv=input_csv,
                    input_jsonl=tmp / "missing.jsonl",
                    summary_root=summary_root,
                    thresholds=[0.2],
                    models=["logistic_regression"],
                )

    def test_deep_input_required_only_when_deep_models_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary_root = tmp / "outputs" / "model_benchmark_threshold_sweep"
            base_config = tmp / "config" / "evaluation_config.yaml"
            input_csv = tmp / "data" / "benchmark.csv"
            _write_base_config(base_config)
            _write_placeholder_csv(input_csv)

            stub_paths = {"metrics_path": tmp / "metrics.json", "comparison_table_path": tmp / "comparison.csv"}
            with mock.patch("src.training.run_fixed_threshold_sweep.generate_threshold_sweep_summary", return_value={"wide_table_path": "wide.csv", "line_plot_path": "line.png"}), mock.patch(
                "src.training.run_fixed_threshold_sweep.generate_threshold_root_figures",
                return_value={},
            ), mock.patch(
                "src.training.run_fixed_threshold_sweep.MODEL_RUNNERS",
                {"logistic_regression": mock.Mock(return_value=stub_paths)},
            ):
                run_fixed_threshold_sweep(
                    base_config_path=base_config,
                    input_csv=input_csv,
                    input_jsonl=tmp / "missing.jsonl",
                    summary_root=summary_root,
                    thresholds=[0.2],
                    models=["logistic_regression"],
                )

            with self.assertRaises(FileNotFoundError):
                run_fixed_threshold_sweep(
                    base_config_path=base_config,
                    input_csv=input_csv,
                    input_jsonl=tmp / "missing.jsonl",
                    summary_root=summary_root,
                    thresholds=[0.2],
                    models=["bert"],
                )

    def test_resume_skips_complete_models_and_writes_command_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary_root = tmp / "outputs" / "model_benchmark_threshold_sweep"
            base_config = tmp / "config" / "evaluation_config.yaml"
            input_csv = tmp / "data" / "benchmark.csv"
            input_jsonl = tmp / "data" / "dataset.jsonl"
            _write_base_config(base_config)
            _write_placeholder_csv(input_csv)
            _write_placeholder_jsonl(input_jsonl)

            complete_root = tmp / "outputs" / "model_benchmark_t02"
            _seed_complete_model_run(complete_root, "logistic_regression")
            self.assertTrue((complete_root / "metrics" / "logistic_regression.json").exists())

            rf_runner = mock.Mock(return_value=_stub_result(complete_root, "random_forest"))
            lr_runner = mock.Mock(return_value=_stub_result(complete_root, "logistic_regression"))
            summary_stub = {
                "wide_table_path": str(summary_root / "tables" / "fixed_threshold_macro_f1_wide.csv"),
                "line_plot_path": str(summary_root / "figures" / "macro_f1_vs_threshold_by_model.png"),
            }

            with mock.patch.dict(
                "src.training.run_fixed_threshold_sweep.MODEL_RUNNERS",
                {
                    "logistic_regression": lr_runner,
                    "random_forest": rf_runner,
                },
                clear=True,
            ), mock.patch(
                "src.training.run_fixed_threshold_sweep.generate_threshold_sweep_summary",
                return_value=summary_stub,
            ), mock.patch(
                "src.training.run_fixed_threshold_sweep.maybe_assert_local_bert_checkpoint",
                return_value=None,
            ):
                manifest = run_fixed_threshold_sweep(
                    base_config_path=base_config,
                    input_csv=input_csv,
                    input_jsonl=input_jsonl,
                    summary_root=summary_root,
                    thresholds=[0.2],
                    models=["logistic_regression", "random_forest"],
                    resume=True,
                )

            self.assertEqual(rf_runner.call_count, 1)
            self.assertEqual(lr_runner.call_count, 0)
            threshold_record = manifest["threshold_runs"][0]
            self.assertEqual(
                [entry["status"] for entry in threshold_record["models"]],
                ["skipped_resume", "completed"],
            )

            command_text = Path(manifest["command_manifest_path"]).read_text(encoding="utf-8")
            self.assertIn("run_logistic_regression_benchmark", command_text)
            self.assertIn("run_random_forest_benchmark", command_text)
            self.assertNotIn("run_bert_benchmark", command_text)
            self.assertIn("--models logistic_regression random_forest", command_text)


if __name__ == "__main__":
    unittest.main()
