"""Tests for the transcript-only TF-IDF + LR baseline."""

from __future__ import annotations

import unittest

import pandas as pd

from src.training.run_tfidf_logistic_regression_baseline import run_tfidf_baseline
from src.training.source_grouped_folds import OFFICIAL_LABELS


class TfidfLogisticRegressionBaselineTests(unittest.TestCase):
    def test_runner_produces_complete_binary_oof_predictions(self) -> None:
        rows = []
        for index in range(30):
            row = {
                "perfume_id": f"p{index}",
                "fold_id": 1 if index < 15 else 2,
                "source_group_id": f"g{index}",
                "source_group_size": 1,
                "document_text": (
                    "bright joyful citrus trusted fragrance"
                    if index % 2 == 0
                    else "dark unusual woody fragrance"
                ),
            }
            for label_index, label in enumerate(OFFICIAL_LABELS):
                row[label] = int((index + label_index) % 3 == 0)
            rows.append(row)
        result = run_tfidf_baseline(pd.DataFrame(rows))
        predictions = result["predictions"]
        self.assertEqual(len(predictions), 30)
        self.assertEqual(len(result["per_label"]), 7)
        self.assertEqual(len(result["fold_diagnostics"]), 2)
        for label in OFFICIAL_LABELS:
            self.assertTrue(predictions[f"prob_{label}"].between(0, 1).all())
            self.assertTrue(predictions[f"pred_{label}"].isin([0, 1]).all())


if __name__ == "__main__":
    unittest.main()
