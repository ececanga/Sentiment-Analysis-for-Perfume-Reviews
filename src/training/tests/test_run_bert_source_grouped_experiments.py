"""Focused tests for the resumable source-grouped BERT experiment runner."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.training.run_bert_source_grouped_experiments import (
    TARGETS,
    build_master_frames,
    expand_units_to_chunk_indices,
    experiment_plan,
    logical_pos_weight,
)


class BertSourceGroupedExperimentTests(unittest.TestCase):
    def test_plan_contains_same_device_control_review_sweep_and_balancing(self) -> None:
        plan = experiment_plan()
        self.assertEqual(len(plan), 14)
        self.assertEqual(sum(item.stage == "review_count" for item in plan), 8)
        self.assertEqual(sum(item.stage == "balancing" for item in plan), 6)
        self.assertEqual(plan[0].synthetic_review_count, 0)
        self.assertTrue(plan[0].positive_weighting)
        self.assertTrue(all(not item.positive_weighting for item in plan if item.stage == "balancing"))
        self.assertNotIn("smote", {item.strategy for item in plan})

    def test_master_frame_preserves_logical_units_and_review_labels(self) -> None:
        perfume = pd.DataFrame(
            [
                {
                    "perfume_id": "p1",
                    "Perfume": "Perfume One",
                    "fold_id": 1,
                    "source_group_id": "g1",
                    "source_group_size": 1,
                    **{label: int(label in {"joy", "trust"}) for label in TARGETS},
                }
            ]
        )
        chunks = pd.DataFrame(
            [
                {
                    "perfume_id": "p1",
                    "Perfume": "Perfume One",
                    "fold_id": 1,
                    "chunk_count": 2,
                    "chunk_index": idx,
                    "chunk_text": f"REVIEW: original {idx}",
                    "source_group_id": "g1",
                    "source_group_size": 1,
                    "video_id": f"v{idx}",
                    "transcript_sha256": f"h{idx}",
                    **{label: int(label in {"joy", "trust"}) for label in TARGETS},
                }
                for idx in range(2)
            ]
        )
        synthetic = pd.DataFrame(
            [
                {
                    "synthetic_review_id": "s1",
                    "perfume_id": "p1",
                    "Perfume": "Perfume One",
                    "review_index": 1,
                    "Final Transcript": "synthetic review",
                    **{label: int(label == "sadness") for label in TARGETS},
                }
            ]
        )
        master, units, mapping = build_master_frames(perfume, chunks, synthetic)
        self.assertEqual(len(master), 3)
        self.assertEqual(len(units), 2)
        self.assertEqual(mapping["og::p1"].tolist(), [0, 1])
        self.assertEqual(mapping["synthetic::s1"].tolist(), [2])
        self.assertEqual(master["sample_weight"].tolist(), [0.5, 0.5, 1.0])
        self.assertEqual(int(master.loc[2, "sadness"]), 1)
        self.assertEqual(int(master.loc[2, "joy"]), 0)

    def test_duplicate_logical_units_expand_to_duplicate_chunk_indices(self) -> None:
        selected = pd.DataFrame({"unit_id": ["a", "b", "a"]})
        mapping = {"a": np.asarray([0, 1]), "b": np.asarray([2])}
        expanded = expand_units_to_chunk_indices(selected, mapping)
        self.assertEqual(expanded.tolist(), [0, 1, 2, 0, 1])

    def test_logical_pos_weight_uses_logical_rows_not_chunk_rows(self) -> None:
        units = pd.DataFrame(
            {
                label: [1, 0, 0, 0] if label == "joy" else [0, 0, 0, 0]
                for label in TARGETS
            }
        )
        weights = logical_pos_weight(units)
        self.assertAlmostEqual(float(weights[0]), 3.0)
        self.assertTrue(np.array_equal(weights[1:], np.ones(len(TARGETS) - 1, dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
