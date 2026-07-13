"""Tests for OOF confusion and predicted-positive diagnostics."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.build_source_grouped_prediction_diagnostics import calculate_model_diagnostics


class SourceGroupedPredictionDiagnosticsTests(unittest.TestCase):
    def test_confusion_and_predicted_positive_counts(self) -> None:
        frame = pd.DataFrame(
            {
                **{f"true_{label}": [1, 1, 0, 0] for label in [
                    "joy", "sadness", "fear", "trust", "surprise", "anticipation", "disgust"
                ]},
                **{f"pred_{label}": [1, 0, 1, 0] for label in [
                    "joy", "sadness", "fear", "trust", "surprise", "anticipation", "disgust"
                ]},
            }
        )
        rows, summary = calculate_model_diagnostics(frame, model_id="test", model_name="Test")
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[0]["true_positive"], 1)
        self.assertEqual(rows[0]["false_positive"], 1)
        self.assertEqual(rows[0]["true_negative"], 1)
        self.assertEqual(rows[0]["false_negative"], 1)
        self.assertEqual(rows[0]["predicted_positive_count"], 2)
        self.assertEqual(summary["total_predicted_positive"], 14)
        self.assertAlmostEqual(summary["micro_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
