"""Focused tests for the resumable weighted-synthetic XGBoost runner."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.training.run_source_grouped_weighted_synthetic_lr import CONTROL
from src.training.run_source_grouped_weighted_synthetic_xgboost import (
    DEFAULT_CONFIG,
    FitTask,
    build_fit_tasks,
    load_and_validate_inputs,
    load_task_checkpoint,
    run_experiment,
    save_task_checkpoint,
    task_fingerprint,
    weighted_scale_pos_weight,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WeightedSyntheticXGBoostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = load_and_validate_inputs(DEFAULT_CONFIG)

    def test_frozen_input_and_task_contract(self) -> None:
        self.assertEqual(len(self.prepared.youtube), 784)
        self.assertEqual(self.prepared.youtube["perfume_id"].nunique(), 784)
        self.assertEqual(len(self.prepared.synthetic), 2352)
        self.assertEqual(self.prepared.synthetic["perfume_id"].nunique(), 784)
        self.assertEqual(
            set(self.prepared.synthetic.groupby("perfume_id").size().unique()),
            {3},
        )
        self.assertEqual(len(self.prepared.feature_columns), 58)
        self.assertEqual(len(self.prepared.labels), 7)
        self.assertEqual(
            sorted(self.prepared.manifest["fold_id"].astype(int).unique().tolist()),
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(self.prepared.manifest["source_group_id"].nunique(), 359)

        tasks = build_fit_tasks(self.prepared)
        self.assertEqual(len(tasks), 70)
        self.assertEqual(tasks[0], FitTask(CONTROL, 1, self.prepared.labels[0]))
        self.assertEqual(
            len({(task.condition, task.fold_id, task.label) for task in tasks}),
            70,
        )

    def test_weighted_scale_pos_weight_uses_effective_mass(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1], dtype=int)
        weights = np.asarray([1.0, 1.0, 1.0 / 3.0, 1.0, 1.0 / 3.0])

        positive, negative, ratio = weighted_scale_pos_weight(labels, weights)

        self.assertAlmostEqual(positive, 4.0 / 3.0)
        self.assertAlmostEqual(negative, 7.0 / 3.0)
        self.assertAlmostEqual(ratio, 7.0 / 4.0)
        self.assertNotAlmostEqual(ratio, 3.0 / 2.0)  # Raw row-count ratio.

    def test_checkpoint_round_trip_and_stale_rejection(self) -> None:
        task = FitTask(CONTROL, 1, "joy")
        perfume_ids = ["perfume-a", "perfume-b"]
        probabilities = np.asarray([0.125, 0.875], dtype=np.float64)
        summary = {
            "label": "joy",
            "fit_duration_seconds": 0.25,
            "scale_pos_weight": 1.5,
        }
        fingerprint = task_fingerprint("base-fingerprint", task, perfume_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoints" / "joy.npz"
            save_task_checkpoint(
                path,
                fingerprint=fingerprint,
                task=task,
                test_perfume_ids=perfume_ids,
                probabilities=probabilities,
                summary=summary,
            )

            loaded_probabilities, loaded_summary = load_task_checkpoint(
                path,
                expected_fingerprint=fingerprint,
                expected_task=task,
                expected_test_perfume_ids=perfume_ids,
            )
            np.testing.assert_array_equal(loaded_probabilities, probabilities)
            self.assertEqual(loaded_summary, summary)
            self.assertFalse(path.with_suffix(".tmp.npz").exists())

            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                load_task_checkpoint(
                    path,
                    expected_fingerprint="stale-fingerprint",
                    expected_task=task,
                    expected_test_perfume_ids=perfume_ids,
                )

    def test_real_pause_then_resume_adds_one_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "weighted_xgboost"

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
            checkpoints_after_first = sorted(
                (out_dir / "checkpoints").rglob("*.npz")
            )
            self.assertEqual(len(checkpoints_after_first), 1)
            first_checkpoint = checkpoints_after_first[0]
            first_checkpoint_hash = _sha256(first_checkpoint)

            first_progress = json.loads(
                (out_dir / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first_progress["status"], "paused")
            self.assertEqual(first_progress["completed_fit_tasks"], 1)
            self.assertEqual(first_progress["total_fit_tasks"], 70)

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
            checkpoints_after_second = sorted(
                (out_dir / "checkpoints").rglob("*.npz")
            )
            self.assertEqual(len(checkpoints_after_second), 2)
            self.assertEqual(_sha256(first_checkpoint), first_checkpoint_hash)

            second_progress = json.loads(
                (out_dir / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(second_progress["status"], "paused")
            self.assertEqual(second_progress["completed_fit_tasks"], 2)
            self.assertEqual(second_progress["remaining_fit_tasks"], 68)
            self.assertAlmostEqual(second_progress["fit_percent"], 200.0 / 70.0)
            self.assertFalse((out_dir / "REPORT.json").exists())


if __name__ == "__main__":
    unittest.main()
