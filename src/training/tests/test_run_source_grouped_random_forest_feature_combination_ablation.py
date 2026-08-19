"""Focused tests for the formal resumable Random Forest combination ablation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.training.run_source_grouped_random_forest_feature_combination_ablation import (
    ACCEPTED_EXPERIMENT_ROOT,
    ALL_VARIANTS,
    COMBINATION_VARIANTS,
    DEFAULT_CONFIG,
    FitTask,
    FULL_VARIANT,
    build_bootstrap_outputs,
    build_fit_tasks,
    checkpoint_path,
    fit_fingerprint,
    load_and_validate_inputs,
    load_task_checkpoint,
    run_experiment,
    save_task_checkpoint,
    task_fingerprint,
    validate_output_root,
    verify_completed_output,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class SourceGroupedRandomForestCombinationAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = load_and_validate_inputs(DEFAULT_CONFIG)

    def test_frozen_schema_and_task_order(self) -> None:
        prepared = self.prepared
        self.assertEqual(len(prepared.data), 784)
        self.assertEqual(prepared.data["source_group_id"].nunique(), 359)
        self.assertEqual(
            prepared.data["fold_id"].value_counts().sort_index().to_dict(),
            {1: 334, 2: 113, 3: 113, 4: 112, 5: 112},
        )
        self.assertEqual(tuple(prepared.variants), ALL_VARIANTS)
        self.assertEqual(
            {key: len(value) for key, value in prepared.variants.items()},
            {
                "full_hybrid": 58,
                "nrc_only": 10,
                "vad_only": 3,
                "transformer_only": 45,
                "nrc_plus_vad": 13,
                "nrc_plus_transformer": 55,
                "vad_plus_transformer": 48,
            },
        )
        tasks = build_fit_tasks(prepared)
        self.assertEqual(len(tasks), 245)
        self.assertEqual(len(set(tasks)), 245)
        self.assertTrue(all(task.condition == FULL_VARIANT for task in tasks[:35]))
        self.assertEqual(tasks[35].condition, COMBINATION_VARIANTS[0])

    def test_model_and_accepted_provenance_contract(self) -> None:
        prepared = self.prepared
        self.assertEqual(prepared.accepted_metrics["model_id"], "random_forest")
        self.assertEqual(tuple(prepared.accepted_metrics["target_columns"]), prepared.labels)
        self.assertEqual(
            prepared.accepted_metrics["excluded_target_columns"],
            {"anger": "excluded_from_the_prespecified_seven_label_benchmark"},
        )
        self.assertEqual(float(prepared.accepted_metrics["threshold"]), 0.3)
        self.assertEqual(int(prepared.accepted_metrics["model_seed"]), 42)
        self.assertEqual(
            prepared.accepted_metrics["dataset_sha256"],
            _sha256(prepared.paths["data"]),
        )
        self.assertEqual(
            prepared.accepted_metrics["config_sha256"],
            _sha256(prepared.paths["base_config"]),
        )
        self.assertEqual(
            prepared.accepted_metrics["split_provenance"]["source_audit_sha256"],
            _sha256(prepared.paths["source_audit"]),
        )
        self.assertEqual(len(prepared.paths), 12)
        self.assertEqual(prepared.accepted_manifest_validation["status"], "passed")
        self.assertEqual(prepared.accepted_manifest_validation["artifact_count"], 95)
        self.assertEqual(
            prepared.accepted_metrics["model_parameters"],
            {
                "imputer_strategy": "median",
                "classifier": {
                    "type": "RandomForestClassifier",
                    "n_estimators": 500,
                    "class_weight": "balanced_subsample",
                    "n_jobs": 1,
                },
            },
        )

    def test_output_root_collision_guard(self) -> None:
        with self.assertRaisesRegex(ValueError, "collides with a protected input"):
            validate_output_root(ACCEPTED_EXPERIMENT_ROOT, self.prepared)
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = Path(tmpdir) / "safe-output"
            self.assertEqual(validate_output_root(candidate, self.prepared), candidate.resolve())

    def test_canonical_output_rejects_reduced_bootstrap(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "canonical formal output root requires the frozen 2000"
        ):
            run_experiment(
                config_path=DEFAULT_CONFIG,
                bootstrap_reps=2,
                show_progress=False,
                install_signal_handlers=False,
            )

    def test_checkpoint_round_trip_and_stale_rejection(self) -> None:
        task = FitTask("nrc_only", 2, "joy")
        ids = ["perfume-a", "perfume-b"]
        probabilities = np.array([0.2, 0.8], dtype=np.float64)
        base = fit_fingerprint(self.prepared)
        fingerprint = task_fingerprint(base, task, ids)
        summary = {"variant": task.condition, "fit_duration_seconds": 1.25}
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = checkpoint_path(root, task)
            save_task_checkpoint(
                path,
                fingerprint=fingerprint,
                task=task,
                test_perfume_ids=ids,
                probabilities=probabilities,
                summary=summary,
            )
            observed, observed_summary = load_task_checkpoint(
                path,
                expected_fingerprint=fingerprint,
                expected_task=task,
                expected_test_perfume_ids=ids,
            )
            np.testing.assert_array_equal(observed, probabilities)
            self.assertEqual(observed_summary, summary)
            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                load_task_checkpoint(
                    path,
                    expected_fingerprint="stale",
                    expected_task=task,
                    expected_test_perfume_ids=ids,
                )

    def test_completed_output_integrity_requires_exact_file_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "result.csv"
            artifact.write_text("a,b\n1,2\n", encoding="utf-8")
            (root / "progress.json").write_text("{}\n", encoding="utf-8")
            checkpoint = root / "checkpoints" / "x.npz"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            checksum = root / "SHA256SUMS"
            checksum.write_text(f"{_sha256(artifact)}  ./result.csv\n", encoding="utf-8")
            verified = verify_completed_output(root, checksum)
            self.assertEqual(verified["artifact_count"], 1)

            extra = root / "unexpected.txt"
            extra.write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file set differs"):
                verify_completed_output(root, checksum)
            extra.unlink()
            artifact.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_completed_output(root, checksum)

    def test_paired_bootstrap_contract_uses_all_six_comparisons(self) -> None:
        # Identical copies make every full-minus-comparison difference exactly
        # zero while exercising both prespecified label scopes.
        outputs = {
            variant: self.prepared.accepted_oof.copy() for variant in ALL_VARIANTS
        }
        intervals, differences, details = build_bootstrap_outputs(
            self.prepared, outputs, reps=2
        )
        self.assertEqual(len(intervals), 42)
        self.assertEqual(len(differences), 36)
        self.assertEqual(
            set(differences["label_scope"]),
            {"seven_labels", "fear_excluded_six_labels"},
        )
        self.assertEqual(set(differences["comparison_variant"]), set(COMBINATION_VARIANTS))
        self.assertTrue(np.allclose(differences["point_estimate_difference"], 0.0))
        self.assertEqual(set(details), {"seven_labels", "fear_excluded_six_labels"})

    def test_real_pause_then_resume_preserves_first_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "random_forest-combinations"
            first = run_experiment(
                config_path=DEFAULT_CONFIG,
                out_dir=out_dir,
                bootstrap_reps=2,
                stop_after_tasks=1,
                show_progress=False,
                install_signal_handlers=False,
            )
            self.assertEqual(first["status"], "paused")
            self.assertEqual(first["completed"], 1)
            checkpoints = sorted((out_dir / "checkpoints").rglob("*.npz"))
            self.assertEqual(len(checkpoints), 1)
            first_path = checkpoints[0]
            first_hash = _sha256(first_path)
            progress = json.loads((out_dir / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["completed_fit_tasks"], 1)
            self.assertEqual(progress["total_fit_tasks"], 245)

            second = run_experiment(
                config_path=DEFAULT_CONFIG,
                out_dir=out_dir,
                bootstrap_reps=2,
                stop_after_tasks=2,
                show_progress=False,
                install_signal_handlers=False,
            )
            self.assertEqual(second["status"], "paused")
            self.assertEqual(second["completed"], 2)
            self.assertEqual(len(list((out_dir / "checkpoints").rglob("*.npz"))), 2)
            self.assertEqual(_sha256(first_path), first_hash)
            self.assertFalse((out_dir / "REPORT.json").exists())


if __name__ == "__main__":
    unittest.main()
