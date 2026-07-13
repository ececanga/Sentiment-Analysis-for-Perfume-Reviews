# Date Created: 2026-07-13

"""Focused tests for the source-grouped Gemini-browser benchmark runner."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.pipeline.run_gemini_browser_source_grouped_benchmark import (
    SourceGroupedBenchmarkRunner,
    build_command_manifest,
    build_workflow_plan,
    canonical_stage_names,
    select_stages,
    sha256_file,
    recompute_oof_metrics,
    validate_model_outputs,
    validate_deep_dataset,
    validate_output_root,
    validate_source_group_audit,
)


EXPECTED_STAGE_ORDER = [
    "prepare",
    "source_audit_gate",
    "dummy_prevalence_baselines",
    "logistic_regression_smoke",
    "random_forest",
    "xgboost",
    "deep_dataset",
    "deep_dataset_audit_gate",
    "bert",
    "bilstm_attention",
    "source_grouped_diagnostics",
    "final_thesis_export",
    "final_audit_gate",
]


class SourceGroupedBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name) / "repo"
        self.project_root.mkdir(parents=True)
        (self.project_root / "outputs").mkdir()
        self.output_root = self.project_root / "outputs/gemini_browser_og784_source_grouped_v1"
        self.raw_csv = self.project_root / "data/raw.csv"
        self.labels_csv = self.project_root / "labels/labels.csv"
        self.feature_csv = self.project_root / "inputs/features.csv"
        self.config_path = self.project_root / "config/source_grouped.yaml"
        for path in (self.raw_csv, self.labels_csv, self.feature_csv, self.config_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_csv.write_text("Perfume,Video ID,Final Transcript\n", encoding="utf-8")
        feature_rows = []
        labels = ("joy", "sadness", "fear", "trust", "surprise", "anticipation", "disgust")
        for index, perfume_id in enumerate(("p1", "p2", "p3", "p4")):
            row = {"perfume_id": perfume_id, "NRC_joy": float(index + 1)}
            for label_index, label in enumerate(labels):
                row[label] = int((index + label_index) % 2 == 0)
            feature_rows.append(row)
        pd.DataFrame(feature_rows).to_csv(self.feature_csv, index=False)
        self._write_labels()
        self.config_path.write_text(
            "provenance:\n"
            "  frozen_input_sha256:\n"
            f"    raw_csv: {sha256_file(self.raw_csv)}\n"
            f"    labels_csv: {sha256_file(self.labels_csv)}\n"
            f"    feature_label_csv: {sha256_file(self.feature_csv)}\n"
            "benchmark: {}\n"
            "evaluation:\n"
            "  probability_threshold: 0.3\n"
            "splits:\n"
            "  outer_cv:\n"
            "    type: source_component_groupkfold\n"
            "    split_design_label: 2-fold source-component grouped CV\n"
            "artifacts:\n"
            "  root_dir: outputs/gemini_browser_og784_source_grouped_v1/model_benchmark\n",
            encoding="utf-8",
        )
        self.plan = self._build_plan()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _build_plan(self):
        return build_workflow_plan(
            project_root=self.project_root,
            output_root=self.output_root,
            config_path=self.config_path,
            raw_csv=self.raw_csv,
            labels_csv=self.labels_csv,
            feature_label_csv=self.feature_csv,
            python_executable=".venv/bin/python",
            n_splits=2,
            inner_n_splits=2,
            seed=42,
            device="cpu",
        )

    def _write_labels(self) -> None:
        frame = pd.read_csv(self.feature_csv)[
            [
                "perfume_id",
                "joy",
                "sadness",
                "fear",
                "trust",
                "surprise",
                "anticipation",
                "disgust",
            ]
        ].copy()
        frame["perfume_name"] = ["One", "Two", "Three", "Four"]
        frame["anger"] = 0
        frame["valence"] = 5.0
        frame["arousal"] = 4.0
        frame["dominance"] = 5.0
        frame.to_csv(self.labels_csv, index=False)

    def _write_source_audit_fixture(self, *, cross_fold_video_count: int = 0) -> None:
        paths = self.plan.paths
        paths.source_audit_dir.mkdir(parents=True, exist_ok=True)
        paths.tables_dir.mkdir(parents=True, exist_ok=True)
        outer = pd.DataFrame(
            {
                "perfume_id": ["p1", "p2", "p3", "p4"],
                "fold_id": [1, 1, 2, 2],
                "source_group_id": ["g1", "g2", "g3", "g4"],
                "source_group_size": [1, 1, 1, 1],
            }
        )
        outer.to_csv(paths.fold_assignments_csv, index=False)
        pd.DataFrame(
            {
                "perfume_id": ["p1", "p2", "p3", "p4"],
                "fold_id": [1, 1, 2, 2],
                "source_group_id": ["g1", "g2", "g3", "g4"],
                "video_id": ["v1", "v2", "v3", "v4"],
                "transcript_sha256": ["h1", "h2", "h3", "h4"],
            }
        ).to_csv(paths.source_memberships_csv, index=False)
        pd.DataFrame(
            {
                "source_group_id": ["g1", "g2", "g3", "g4"],
                "fold_id": [1, 1, 2, 2],
                "perfume_count": [1, 1, 1, 1],
            }
        ).to_csv(paths.source_components_csv, index=False)
        pd.DataFrame(
            [
                {"outer_fold_id": 1, "perfume_id": "p3", "inner_role": "train"},
                {"outer_fold_id": 1, "perfume_id": "p4", "inner_role": "val"},
                {"outer_fold_id": 2, "perfume_id": "p1", "inner_role": "train"},
                {"outer_fold_id": 2, "perfume_id": "p2", "inner_role": "val"},
            ]
        ).to_csv(paths.inner_validation_folds_csv, index=False)
        balance_rows = []
        label_frame = pd.read_csv(self.labels_csv)
        for fold_id, test_ids in ((1, {"p1", "p2"}), (2, {"p3", "p4"})):
            test = label_frame.loc[label_frame["perfume_id"].isin(test_ids)]
            train = label_frame.loc[~label_frame["perfume_id"].isin(test_ids)]
            row = {
                "fold_id": fold_id,
                "test_perfume_count": 2,
                "test_source_group_count": 2,
                "train_perfume_count": 2,
                "train_source_group_count": 2,
            }
            for label in (
                "joy",
                "sadness",
                "fear",
                "trust",
                "surprise",
                "anticipation",
                "disgust",
            ):
                row[f"test_{label}_positive"] = int(test[label].sum())
                row[f"train_{label}_positive"] = int(train[label].sum())
            balance_rows.append(row)
        pd.DataFrame(balance_rows).to_csv(paths.fold_balance_csv, index=False)
        paths.source_group_audit_json.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "validation_passed": True,
                    "grouping_version": "source_identity_v1",
                    "n_splits": 2,
                    "largest_component": {"source_group_size": 1},
                    "cross_fold_video_id_count": cross_fold_video_count,
                    "cross_fold_transcript_hash_count": 0,
                    "manifest_sha256": {
                        "outer": sha256_file(paths.fold_assignments_csv),
                        "inner": sha256_file(paths.inner_validation_folds_csv),
                        "fold_balance": sha256_file(paths.fold_balance_csv),
                    },
                    "input_sha256": {
                        "raw_csv": sha256_file(self.raw_csv),
                        "labels_csv": sha256_file(self.labels_csv),
                        "feature_label_csv": sha256_file(self.feature_csv),
                    },
                    "frozen_input_validation": {
                        "enforced": True,
                        "config_sha256": sha256_file(self.config_path),
                    },
                }
            ),
            encoding="utf-8",
        )

    def _write_deep_dataset_fixture(self) -> dict[str, object]:
        paths = self.plan.paths
        paths.datasets_dir.mkdir(parents=True, exist_ok=True)
        labels = pd.read_csv(self.labels_csv).set_index("perfume_id")
        outer = pd.read_csv(paths.fold_assignments_csv).set_index("perfume_id")
        payloads = []
        chunk_rows = []
        for perfume_id in ("p1", "p2", "p3", "p4"):
            row = labels.loc[perfume_id]
            chunk_text = f"REVIEW: transcript for {perfume_id}"
            payloads.append(
                {
                    "perfume_id": perfume_id,
                    "perfume_name": str(row["perfume_name"]),
                    "fold_id": int(outer.loc[perfume_id, "fold_id"]),
                    "source_group_id": str(outer.loc[perfume_id, "source_group_id"]),
                    "source_group_size": 1,
                    "labels": {
                        label: int(row[label])
                        for label in (
                            "joy",
                            "sadness",
                            "fear",
                            "trust",
                            "surprise",
                            "anticipation",
                            "disgust",
                        )
                    },
                    "excluded_labels": {"anger": int(row["anger"])},
                    "vad": {
                        dimension: float(row[dimension])
                        for dimension in ("valence", "arousal", "dominance")
                    },
                    "review_chunks": [
                        {
                            "chunk_index": 0,
                            "chunk_text": chunk_text,
                            "video_id": f"video_{perfume_id}",
                            "transcript_sha256": f"hash_{perfume_id}",
                        }
                    ],
                }
            )
            chunk_rows.append(
                {
                    "perfume_id": perfume_id,
                    "fold_id": int(outer.loc[perfume_id, "fold_id"]),
                    "source_group_id": str(outer.loc[perfume_id, "source_group_id"]),
                    "chunk_index": 0,
                    "chunk_text": chunk_text,
                    "video_id": f"video_{perfume_id}",
                    "transcript_sha256": f"hash_{perfume_id}",
                    **{
                        label: int(row[label])
                        for label in (
                            "joy",
                            "sadness",
                            "fear",
                            "trust",
                            "surprise",
                            "anticipation",
                            "disgust",
                            "anger",
                        )
                    },
                    **{
                        dimension: float(row[dimension])
                        for dimension in ("valence", "arousal", "dominance")
                    },
                }
            )
        paths.deep_jsonl.write_text(
            "".join(json.dumps(payload) + "\n" for payload in payloads),
            encoding="utf-8",
        )
        pd.DataFrame(chunk_rows).to_csv(paths.deep_chunks_csv, index=False)
        paths.deep_summary_json.write_text(
            json.dumps(
                {
                    "source_grouped": True,
                    "cross_fold_video_id_count": 0,
                    "cross_fold_transcript_hash_count": 0,
                    "n_perfumes": 4,
                    "fold_manifest_sha256": sha256_file(paths.fold_assignments_csv),
                    "raw_csv_sha256": sha256_file(self.raw_csv),
                    "labels_csv_sha256": sha256_file(self.labels_csv),
                    "model_input_fields": ["Final Transcript"],
                    "model_input_policy": "transcript_excerpt_only; metadata excluded",
                }
            ),
            encoding="utf-8",
        )
        return {"payloads": payloads, "chunks": chunk_rows}

    def test_stage_order_is_lr_first_then_classical_then_deep(self) -> None:
        self.assertEqual(canonical_stage_names(self.plan), EXPECTED_STAGE_ORDER)
        selected = select_stages(
            plan=self.plan,
            from_stage="logistic_regression_smoke",
            to_stage="deep_dataset_audit_gate",
        )
        self.assertEqual(
            [stage.name for stage in selected],
            [
                "logistic_regression_smoke",
                "random_forest",
                "xgboost",
                "deep_dataset",
                "deep_dataset_audit_gate",
            ],
        )

    def test_preparation_command_matches_source_group_contract(self) -> None:
        prepare = self.plan.stages[0]
        command = list(prepare.commands[0])
        self.assertEqual(command[:3], [".venv/bin/python", "-m", "src.training.source_grouped_folds"])
        expected_pairs = {
            "--config": "config/source_grouped.yaml",
            "--raw-csv": "data/raw.csv",
            "--labels-csv": "labels/labels.csv",
            "--feature-label-csv": "inputs/features.csv",
            "--out-root": "outputs/gemini_browser_og784_source_grouped_v1",
            "--n-splits": "2",
            "--inner-n-splits": "2",
            "--seed": "42",
        }
        for flag, value in expected_pairs.items():
            position = command.index(flag)
            self.assertEqual(command[position + 1], value)

    def test_dry_run_persists_command_manifest_without_execution(self) -> None:
        calls: list[str] = []

        def unexpected_executor(
            stage: str,
            command: Sequence[str],
            cwd: Path,
            log_path: Path,
        ) -> None:
            calls.append(stage)

        selected = select_stages(plan=self.plan)
        runner = SourceGroupedBenchmarkRunner(
            plan=self.plan,
            selected_stages=selected,
            dry_run=True,
            resume=False,
            command_executor=unexpected_executor,
        )
        result = runner.run()
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(calls, [])
        manifest = json.loads(self.plan.paths.command_manifest_json.read_text(encoding="utf-8"))
        self.assertEqual(manifest["stage_order"], EXPECTED_STAGE_ORDER)
        self.assertEqual(manifest["selected_stages"], EXPECTED_STAGE_ORDER)
        self.assertEqual(
            [row["stage"] for row in manifest["resolved_commands"]],
            [
                "prepare",
                "dummy_prevalence_baselines",
                "logistic_regression_smoke",
                "random_forest",
                "xgboost",
                "deep_dataset",
                "bert",
                "bilstm_attention",
                "source_grouped_diagnostics",
                "final_thesis_export",
            ],
        )
        self.assertTrue(manifest["safety"]["source_audit_cannot_be_bypassed"])
        self.assertEqual(
            manifest["code_provenance"]["config_sha256"],
            sha256_file(self.config_path),
        )

    def test_command_manifest_marks_internal_hard_gates(self) -> None:
        manifest = build_command_manifest(
            plan=self.plan,
            selected_stages=self.plan.stages,
            dry_run=True,
            resume=False,
        )
        kinds = {row["name"]: row["kind"] for row in manifest["stages"]}
        self.assertEqual(kinds["source_audit_gate"], "hard_gate")
        self.assertEqual(kinds["deep_dataset_audit_gate"], "hard_gate")
        self.assertEqual(kinds["final_audit_gate"], "hard_gate")

    def test_protected_legacy_output_root_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "protected legacy"):
            validate_output_root(
                output_root=self.project_root / "outputs/gemini_browser_og784/model_benchmark/new",
                project_root=self.project_root,
            )
        validate_output_root(output_root=self.output_root, project_root=self.project_root)

    def test_source_gate_accepts_exact_source_safe_manifests(self) -> None:
        self._write_source_audit_fixture()
        result = validate_source_group_audit(
            paths=self.plan.paths,
            labels_csv=self.labels_csv,
            n_splits=2,
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["n_perfumes"], 4)
        self.assertEqual(result["n_source_components"], 4)
        self.assertEqual(result["cross_fold_video_id_count"], 0)

    def test_source_gate_rejects_nonzero_report_overlap(self) -> None:
        self._write_source_audit_fixture(cross_fold_video_count=1)
        with self.assertRaisesRegex(ValueError, "cross_fold_video_id_count=0"):
            validate_source_group_audit(
                paths=self.plan.paths,
                labels_csv=self.labels_csv,
                n_splits=2,
            )

    def test_source_gate_directly_rejects_component_split(self) -> None:
        self._write_source_audit_fixture()
        outer = pd.read_csv(self.plan.paths.fold_assignments_csv)
        outer.loc[outer["perfume_id"] == "p3", "source_group_id"] = "g1"
        outer.to_csv(self.plan.paths.fold_assignments_csv, index=False)
        report = json.loads(self.plan.paths.source_group_audit_json.read_text(encoding="utf-8"))
        report["manifest_sha256"]["outer"] = sha256_file(self.plan.paths.fold_assignments_csv)
        self.plan.paths.source_group_audit_json.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "component crosses outer folds"):
            validate_source_group_audit(
                paths=self.plan.paths,
                labels_csv=self.labels_csv,
                n_splits=2,
            )

    def test_deep_gate_checks_frozen_labels_vad_and_transcript_only_text(self) -> None:
        self._write_source_audit_fixture()
        source_gate = validate_source_group_audit(
            paths=self.plan.paths,
            labels_csv=self.labels_csv,
            n_splits=2,
        )
        fixture = self._write_deep_dataset_fixture()
        result = validate_deep_dataset(plan=self.plan, source_gate=source_gate)
        self.assertEqual(result["n_perfumes"], 4)

        payloads = fixture["payloads"]
        self.assertIsInstance(payloads, list)
        payloads[0]["labels"]["joy"] = 1 - payloads[0]["labels"]["joy"]
        self.plan.paths.deep_jsonl.write_text(
            "".join(json.dumps(payload) + "\n" for payload in payloads),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "label joy disagrees"):
            validate_deep_dataset(plan=self.plan, source_gate=source_gate)

        self._write_deep_dataset_fixture()
        chunks = pd.read_csv(self.plan.paths.deep_chunks_csv)
        chunks.loc[0, "chunk_text"] = "TITLE: leaked metadata || REVIEW: text"
        chunks.to_csv(self.plan.paths.deep_chunks_csv, index=False)
        with self.assertRaisesRegex(ValueError, "forbidden title/description"):
            validate_deep_dataset(plan=self.plan, source_gate=source_gate)

    def test_model_gate_validates_oof_truth_groups_probabilities_and_hashes(self) -> None:
        self._write_source_audit_fixture()
        source_gate = validate_source_group_audit(
            paths=self.plan.paths,
            labels_csv=self.labels_csv,
            n_splits=2,
        )
        paths = self.plan.paths
        for directory in (
            paths.metrics_dir,
            paths.predictions_dir,
            paths.logs_dir,
            paths.benchmark_root / "figures",
            paths.tables_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        outer = pd.read_csv(paths.fold_assignments_csv).set_index("perfume_id")
        features = pd.read_csv(self.feature_csv)
        predictions = features[["perfume_id"]].copy()
        predictions["fold_id"] = predictions["perfume_id"].map(outer["fold_id"])
        predictions["source_group_id"] = predictions["perfume_id"].map(
            outer["source_group_id"]
        )
        labels = ("joy", "sadness", "fear", "trust", "surprise", "anticipation", "disgust")
        for label in labels:
            predictions[f"true_{label}"] = features[label]
            predictions[f"prob_{label}"] = features[label].map({0: 0.2, 1: 0.8})
            predictions[f"pred_{label}"] = features[label]
        predictions.to_csv(paths.model_predictions("logistic_regression"), index=False)
        recomputed, per_label = recompute_oof_metrics(predictions)
        per_label.to_csv(paths.model_per_label("logistic_regression"), index=False)
        paths.model_per_label_figure("logistic_regression").write_bytes(b"figure")
        paths.model_curve_figure("logistic_regression").write_bytes(b"figure")
        paths.model_log("logistic_regression").write_text("run complete\n", encoding="utf-8")
        paths.model_note("logistic_regression").write_text("note\n", encoding="utf-8")
        provenance = {
            "type": "source_component_groupkfold",
            "split_design": "2-fold source-component grouped CV",
            "grouping_version": "source_identity_v1",
            "group_column": "source_group_id",
            "outer_manifest_sha256": source_gate["outer_manifest_sha256"],
            "inner_manifest_sha256": source_gate["inner_manifest_sha256"],
            "source_audit_sha256": source_gate["source_audit_sha256"],
        }
        payload = {
            "model_id": "logistic_regression",
            "dataset_sha256": sha256_file(self.feature_csv),
            "config_sha256": sha256_file(self.config_path),
            "frozen_input_sha256": {"feature_label_csv": sha256_file(self.feature_csv)},
            "split_provenance": provenance,
            "split_design": "2-fold source-component grouped CV",
            "threshold": 0.3,
            "macro_f1": recomputed["macro_f1"],
            "macro_roc_auc": recomputed["macro_roc_auc"],
            "macro_pr_auc": recomputed["macro_pr_auc"],
            "secondary_metrics": {
                key: recomputed[key]
                for key in ("micro_f1", "subset_accuracy", "hamming_loss", "label_cardinality")
            },
            "per_label": per_label.to_dict(orient="records"),
        }
        paths.model_metrics("logistic_regression").write_text(json.dumps(payload), encoding="utf-8")
        paths.model_run_manifest("logistic_regression").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        result = validate_model_outputs(
            plan=self.plan,
            model_id="logistic_regression",
            source_gate=source_gate,
        )
        self.assertEqual(result["n_oof_rows"], 4)

        threshold_corrupt = predictions.copy()
        threshold_corrupt.loc[0, "pred_joy"] = 1 - int(threshold_corrupt.loc[0, "pred_joy"])
        threshold_corrupt.to_csv(paths.model_predictions("logistic_regression"), index=False)
        with self.assertRaisesRegex(ValueError, "disagrees with probability"):
            validate_model_outputs(
                plan=self.plan,
                model_id="logistic_regression",
                source_gate=source_gate,
            )
        predictions.to_csv(paths.model_predictions("logistic_regression"), index=False)

        metrics_payload = json.loads(
            paths.model_metrics("logistic_regression").read_text(encoding="utf-8")
        )
        metrics_payload["macro_f1"] += 0.01
        paths.model_metrics("logistic_regression").write_text(
            json.dumps(metrics_payload), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "metrics JSON macro_f1 mismatch"):
            validate_model_outputs(
                plan=self.plan,
                model_id="logistic_regression",
                source_gate=source_gate,
            )
        paths.model_metrics("logistic_regression").write_text(
            json.dumps(payload), encoding="utf-8"
        )

        paths.model_note("logistic_regression").unlink()
        with self.assertRaises(FileNotFoundError):
            validate_model_outputs(
                plan=self.plan,
                model_id="logistic_regression",
                source_gate=source_gate,
            )
        paths.model_note("logistic_regression").write_text("note\n", encoding="utf-8")

        predictions.loc[0, "prob_joy"] = 1.2
        predictions.to_csv(paths.model_predictions("logistic_regression"), index=False)
        with self.assertRaisesRegex(ValueError, "probability is invalid for joy"):
            validate_model_outputs(
                plan=self.plan,
                model_id="logistic_regression",
                source_gate=source_gate,
            )

    def test_lr_failure_stops_before_rf_and_xgboost(self) -> None:
        calls: list[str] = []

        def fake_executor(
            stage: str,
            command: Sequence[str],
            cwd: Path,
            log_path: Path,
        ) -> None:
            calls.append(stage)
            if stage == "prepare":
                self._write_source_audit_fixture()
                return
            if stage == "logistic_regression_smoke":
                raise subprocess.CalledProcessError(7, list(command))
            self.fail(f"Unexpected downstream command after LR failure: {stage}")

        runner = SourceGroupedBenchmarkRunner(
            plan=self.plan,
            selected_stages=select_stages(
                plan=self.plan,
                only=["prepare", "logistic_regression_smoke"],
            ),
            dry_run=False,
            resume=False,
            command_executor=fake_executor,
        )
        with self.assertRaises(subprocess.CalledProcessError):
            runner.run()
        self.assertEqual(calls, ["prepare", "logistic_regression_smoke"])
        status = json.loads(self.plan.paths.run_status_json.read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["current_stage"], "logistic_regression_smoke")


if __name__ == "__main__":
    unittest.main()
