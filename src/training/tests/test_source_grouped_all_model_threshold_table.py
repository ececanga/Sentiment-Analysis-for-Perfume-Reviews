"""Tests for the fair all-model fixed-threshold table."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.build_source_grouped_all_model_threshold_table import (
    calculate_threshold_rows,
    build_best_table,
)


class AllModelThresholdTableTests(unittest.TestCase):
    def test_thresholds_reuse_probabilities_and_best_table(self) -> None:
        labels = ["joy", "sadness", "fear", "trust", "surprise", "anticipation", "disgust"]
        frame = pd.DataFrame(
            {
                **{f"true_{label}": [1, 1, 0, 0] for label in labels},
                **{f"prob_{label}": [0.8, 0.4, 0.25, 0.1] for label in labels},
            }
        )
        rows, per_label, ranking = calculate_threshold_rows(
            frame,
            model_id="test",
            model_name="Test",
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(per_label), 28)
        self.assertTrue(ranking["ranking_metrics_threshold_independent"])
        self.assertEqual(rows[1]["threshold"], 0.3)
        self.assertEqual(rows[1]["threshold_role"], "official")
        best = build_best_table(pd.DataFrame(rows))
        self.assertEqual(len(best), 1)
        self.assertEqual(best.iloc[0]["selection_note"], "diagnostic_only; official reporting remains threshold 0.3")


if __name__ == "__main__":
    unittest.main()
